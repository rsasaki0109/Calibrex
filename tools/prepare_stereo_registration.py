"""Prepare independent frame-paired camera/LiDAR geometry without pose labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tools.koide_shared_bundle_pandey import read_koide_ply


def backproject(depth: np.ndarray, intrinsics: list[float]) -> np.ndarray:
    fx, fy, cx, cy = intrinsics
    v, u = np.indices(depth.shape)
    valid = np.isfinite(depth) & (depth > 0.1) & (depth <= 80)
    z = depth[valid]
    return np.column_stack(((u[valid] - cx) * z / fx, (v[valid] - cy) * z / fy, z))


def voxel_first(points: np.ndarray, size: float = 0.2) -> np.ndarray:
    valid = np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) <= 80)
    points = points[valid]
    _, indices = np.unique(np.floor(points / size).astype(np.int64), axis=0, return_index=True)
    return points[np.sort(indices)]


def run(plan_path: Path, workspace: Path, output: Path) -> None:
    if output.exists() or workspace.exists():
        raise ValueError("refuse overwrite")
    plan = json.loads(plan_path.read_text())
    for group in ["source_sha256", "input_sha256"]:
        for p, sha in plan[group].items():
            if hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha:
                raise ValueError("changed frozen input " + p)
    workspace.mkdir(parents=True)
    frames = []
    for case in plan["cases"]:
        for frame in case["frames"]:
            target = voxel_first(
                backproject(np.load(frame["depth"], allow_pickle=False), case["intrinsics"])
            )
            source = voxel_first(read_koide_ply(Path(frame["cloud"]))[:, :3])
            path = workspace / (case["case_id"] + "-" + frame["name"] + ".npz")
            np.savez_compressed(path, source_lidar=source, target_camera=target)
            frames.append(
                {
                    "drive": case["case_id"],
                    "frame": frame["name"],
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "source_points": len(source),
                    "target_points": len(target),
                    "input_cloud": frame["cloud"],
                    "input_depth": frame["depth"],
                }
            )
    output.write_text(
        json.dumps(
            {
                "scope": "18 independent stereo/LiDAR geometry pairs; no fits",
                "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                "frames": frames,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["plan", "workspace", "output"]:
        p.add_argument(name, type=Path)
    a = p.parse_args()
    run(a.plan, a.workspace, a.output)
