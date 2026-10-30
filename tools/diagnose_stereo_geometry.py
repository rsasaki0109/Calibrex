"""Offline reference-pose overlap and surface consistency of prepared geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from tools.known_geometry_joint import Report

from calibrex.solvers.chen_medioni_point_to_plane_icp_solver import (
    ChenMedioniPointToPlaneOptions,
    _estimate_target_normals,
)


def geometry_stats(source: np.ndarray, target: np.ndarray, fov: np.ndarray) -> dict:
    distance, index = cKDTree(target).query(source)
    reverse_distance, reverse_index = cKDTree(source).query(target)
    mutual = reverse_index[index] == np.arange(len(source))
    source_normal, source_valid, _ = _estimate_target_normals(
        source, ChenMedioniPointToPlaneOptions()
    )
    target_normal, target_valid, _ = _estimate_target_normals(
        target, ChenMedioniPointToPlaneOptions()
    )
    supported = fov & (distance <= 1) & source_valid & target_valid[index]
    cosine = np.abs(np.sum(source_normal[supported] * target_normal[index[supported]], axis=1))
    angle = np.degrees(np.arccos(np.clip(cosine, 0, 1)))
    rows = []
    for label, mask in [("all", np.ones(len(source), dtype=bool)), ("camera_fov", fov)]:
        d = distance[mask]
        rows.append(
            {
                "population": label,
                "count": int(mask.sum()),
                "distance_quantiles_m": np.quantile(d, [0.1, 0.5, 0.9, 0.99]).tolist()
                if len(d)
                else None,
                "within_fraction": {
                    str(t): float(np.mean(d <= t)) if len(d) else None for t in [0.2, 0.5, 1.0]
                },
                "mutual_within_1m_count": int((mask & mutual & (distance <= 1)).sum()),
            }
        )
    return {
        "source_populations": rows,
        "reverse_target_distance_quantiles_m": np.quantile(
            reverse_distance, [0.1, 0.5, 0.9, 0.99]
        ).tolist(),
        "source_valid_normal_fraction": float(source_valid.mean()),
        "target_valid_normal_fraction": float(target_valid.mean()),
        "normal_pair_count": int(supported.sum()),
        "normal_angle_quantiles_deg": np.quantile(angle, [0.1, 0.5, 0.9]).tolist()
        if len(angle)
        else None,
        "normal_within_15deg_fraction": float(np.mean(angle <= 15)) if len(angle) else None,
    }


def run(plan_path: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("refuse overwrite")
    plan = json.loads(plan_path.read_text())
    for group in ["source_sha256", "input_sha256"]:
        for p, sha in plan[group].items():
            if hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha:
                raise ValueError("changed " + p)
    report = Report(
        scope="18 offline reference-pose stereo geometry diagnostics; no fits",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    )
    for frame in plan["frames"]:
        a = np.load(frame["path"], allow_pickle=False)
        transform = np.array(plan["reference"][frame["drive"]])
        source = a["source_lidar"] @ transform[:3, :3].T + transform[:3, 3]
        fx, fy, cx, cy = plan["cameras"][frame["drive"]]["intrinsics"]
        h, w = plan["cameras"][frame["drive"]]["shape"]
        z = source[:, 2]
        u = fx * source[:, 0] / np.maximum(z, 1e-9) + cx
        v = fy * source[:, 1] / np.maximum(z, 1e-9) + cy
        fov = (z > 0.1) & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
        report.cases.append(
            {
                "drive": frame["drive"],
                "frame": frame["frame"],
                "source_count": len(source),
                "target_count": len(a["target_camera"]),
                "diagnostic": geometry_stats(source, a["target_camera"], fov),
            }
        )
        output.write_text(report.model_dump_json(indent=2) + "\n")
    output.with_suffix(".schema.json").write_text(
        json.dumps(Report.model_json_schema(), indent=2) + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("plan", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    run(a.plan, a.output)
