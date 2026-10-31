"""Disjoint-ray, reference-free metric checks for frozen synthetic pose candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
from tools.joint_multistart_control import matrix
from tools.known_geometry_joint import CAMERA, CONDITIONS, Report, observations, raycast, reference
from tools.run_camera_lidar_development_matrix import pose_metrics

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_six_dof_solver import apply_local_se3_delta


def holdout_points() -> list[np.ndarray]:
    """New angular rays; all LiDAR-visible hits retained, including camera occlusion."""
    v, u = np.meshgrid(
        np.linspace(6.3, 185.3, 48) + 0.29, np.linspace(8.7, 310.7, 80) + 0.37, indexing="ij"
    )
    rays = np.stack(
        [(u - CAMERA.cx) / CAMERA.fx, (v - CAMERA.cy) / CAMERA.fy, np.ones_like(u)], -1
    ).reshape(-1, 3)
    ref = reference()
    rotation = Rotation.from_quat(ref.rotation_quat_xyzw).as_matrix()
    t = np.array(ref.translation_m)
    clouds = []
    for origin in [np.zeros(3), np.array([1.0, 0.2, 0.4]), np.array([-0.8, -0.3, 0.7])]:
        hit = raycast(origin + t, rays)
        clouds.append((hit[:, None] * rays) @ rotation)
    return clouds


def validate_pose(
    depth: np.ndarray, points: np.ndarray, pose: np.ndarray, radial: bool = False
) -> dict[str, Any]:
    """Judge measured optical-z consistency only; never access scene/reference."""
    camera = points @ pose[:3, :3].T + pose[:3, 3]
    positive = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0.1)
    ids = np.flatnonzero(positive)
    u = np.rint(CAMERA.fx * camera[ids, 0] / camera[ids, 2] + CAMERA.cx).astype(int)
    v = np.rint(CAMERA.fy * camera[ids, 1] / camera[ids, 2] + CAMERA.cy).astype(int)
    inside = (u >= 0) & (u < CAMERA.width) & (v >= 0) & (v < CAMERA.height)
    ids, u, v = ids[inside], u[inside], v[inside]
    measured = depth[v, u].copy()
    if radial:
        measured /= np.sqrt(
            1 + ((u - CAMERA.cx) / CAMERA.fx) ** 2 + ((v - CAMERA.cy) / CAMERA.fy) ** 2
        )
    valid = np.isfinite(measured) & (measured > 0)
    residual = camera[ids[valid], 2] - measured[valid]
    absolute = abs(residual)
    median = float(np.median(absolute)) if len(residual) else None
    p90 = float(np.percentile(absolute, 90)) if len(residual) else None
    supported = len(residual) / len(points) if len(points) else 0.0
    passed = len(residual) >= 100 and supported >= 0.75 and median <= 0.05 and p90 <= 0.2
    return {
        "accepted": bool(passed),
        "total": len(points),
        "supported": len(residual),
        "nonpositive_or_nonfinite": int((~positive).sum()),
        "outside_image": int((~inside).sum()),
        "invalid_depth": int((~valid).sum()),
        "support_fraction": supported,
        "median_absolute_z_m": median,
        "p90_absolute_z_m": p90,
        "behind_surface_gt_20cm": int((residual > 0.2).sum()),
        "in_front_gt_20cm": int((residual < -0.2).sum()),
    }


def run(plan: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("refuse overwrite")
    spec = json.loads(plan.read_text())
    for p, sha in spec["source_sha256"].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha:
            raise ValueError("changed source")
    candidates = json.loads(Path(spec["candidates"]).read_text())["cases"]
    report = Report(
        scope="disjoint synthetic LiDAR metric validation; no reference in validator",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
    )
    clouds = holdout_points()
    for condition in CONDITIONS:
        depths = [o.depth_map for o in observations(condition)]
        poses = [
            ("reference_control", matrix(reference()), True),
            (
                "wrong_control",
                matrix(apply_local_se3_delta(reference(), (2.0, -2.0, 1.5), (0.4, -0.3, 0.3))),
                True,
            ),
        ]
        for c in candidates:
            if c["condition"] != condition:
                continue
            original = c["baseline"]["result"]["transform_camera_lidar"]
            pose = SE3(tuple(original["translation_m"]), tuple(original["rotation_quat_xyzw"]))
            poses.append(
                (
                    f"baseline_{c['sign']}",
                    matrix(pose),
                    c["baseline"]["result"]["status"] == "converged",
                )
            )
            poses.append(
                (f"continuation_{c['sign']}", np.array(c["continuation"]["output_matrix"]), True)
            )
        for name, pose, converged in poses:
            checks = [
                validate_pose(d, p, pose, condition == "exact_range_visible")
                for d, p in zip(depths, clouds, strict=True)
            ]
            # Truth only labels accuracy AFTER candidate validation; never affects its decision.
            report.cases.append(
                {
                    "condition": condition,
                    "candidate": name,
                    "validation_accepted": all(c["accepted"] for c in checks),
                    "frames": checks,
                    "pose_metrics": pose_metrics(pose, matrix(reference()), converged),
                }
            )
    output.write_text(report.model_dump_json(indent=2) + "\n")
    output.with_suffix(".schema.json").write_text(
        json.dumps(Report.model_json_schema(), indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.plan, args.output)
