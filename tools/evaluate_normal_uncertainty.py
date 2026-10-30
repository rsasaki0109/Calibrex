"""Measurement-only local disparity plane reliability against analytic surfaces."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from tools.evaluate_direct_disparity import noisy_depth
from tools.known_geometry_joint import CAMERA, Report
from tools.normal_uncertainty import normal_cone
from tools.occlusion_geometry import SCENES, observations
from tools.prepare_stereo_registration import backproject


def surface_gate(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    disparity = np.divide(
        120.0, depth, out=np.full_like(depth, np.nan), where=np.isfinite(depth) & (depth > 0)
    )
    patches = sliding_window_view(disparity, (5, 5))
    dy, dx = np.mgrid[-2:3, -2:3]
    center = np.mean(patches, axis=(-2, -1))
    a = np.sum(patches * dx, axis=(-2, -1)) / np.sum(dx * dx)
    b = np.sum(patches * dy, axis=(-2, -1)) / np.sum(dy * dy)
    predicted = center[..., None, None] + a[..., None, None] * dx + b[..., None, None] * dy
    rms = np.sqrt(np.mean((patches - predicted) ** 2, axis=(-2, -1)))
    v, u = np.indices(center.shape)
    u = u + 2
    v = v + 2
    normals = np.stack(
        [a * CAMERA.fx, b * CAMERA.fy, center - a * (u - CAMERA.cx) - b * (v - CAMERA.cy)], axis=-1
    )
    length = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.maximum(length, 1e-15)
    sigma = 0.5 * (120 / np.maximum(center, 1e-12)) ** 2 / 120
    valid = np.isfinite(patches).all(axis=(-2, -1)) & (center > 0) & (length[..., 0] > 0)
    accepted = valid & (rms <= 1.0) & (sigma <= 0.2)
    return accepted, normals, valid


def run(plan_path: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("refuse overwrite")
    plan = json.loads(plan_path.read_text())
    for p, sha in plan["source_sha256"].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha:
            raise ValueError("changed " + p)
    report = Report(
        scope="27 angular-uncertainty controls; no calibration fits",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    )
    for scene, spheres in SCENES.items():
        clean = observations(scene, "exact_z_all")
        for i, o in enumerate(clean):
            origin = np.array([(0, 0, 0), (1, 0.2, 0.4), (-0.8, -0.3, 0.7)][i])
            xyz = backproject(o.depth_map, [CAMERA.fx, CAMERA.fy, CAMERA.cx, CAMERA.cy]).reshape(
                192, 320, 3
            )
            world = xyz + origin
            truth = np.broadcast_to(np.array([0.12, -0.08, 1.0]), world.shape).copy()
            for center, radius in spheres:
                delta = world - np.array(center)
                hit = np.abs(np.linalg.norm(delta, axis=-1) - radius) < 1e-6
                truth[hit] = delta[hit] / radius
            truth = truth / np.linalg.norm(truth, axis=-1, keepdims=True)
            for seed in [0, 730, 731]:
                depth = (
                    o.depth_map.copy()
                    if seed == 0
                    else noisy_depth(o.depth_map, seed + 100 * i, 0.5)
                )
                keep, normals, valid = surface_gate(depth)
                radius = normal_cone(depth)
                selected = keep & (radius <= 15)
                angle = np.degrees(
                    np.arccos(np.clip(np.abs(np.sum(normals * truth[2:-2, 2:-2], axis=-1)), 0, 1))
                )
                position = np.linalg.norm(
                    xyz[2:-2, 2:-2] * (depth[2:-2, 2:-2] / o.depth_map[2:-2, 2:-2] - 1)[..., None],
                    axis=-1,
                )
                rows = []
                for name, mask in [
                    ("all_valid", valid),
                    ("previous_gate", keep),
                    ("angular_gate", selected),
                ]:
                    correct = (position <= 0.2) & (angle <= 15)
                    cells = set(zip(*np.nonzero(mask), strict=False))
                    occupied = len({(v // 48, u // 80) for v, u in cells})
                    n = normals[mask]
                    singular = np.linalg.svd(n, compute_uv=False) if len(n) >= 3 else np.zeros(3)
                    rows.append(
                        {
                            "population": name,
                            "count": int(mask.sum()),
                            "cone_covers_true_normal_fraction": float(
                                (angle[mask] <= radius[mask]).mean()
                            )
                            if mask.any()
                            else None,
                            "cone_radius_quantiles_deg": np.quantile(
                                radius[mask], [0.1, 0.5, 0.9]
                            ).tolist()
                            if mask.any()
                            else None,
                            "fraction_of_domain": float(mask.mean()),
                            "joint_surface_correct_fraction": float(correct[mask].mean())
                            if mask.any()
                            else None,
                            "position_correct_fraction": float((position[mask] <= 0.2).mean())
                            if mask.any()
                            else None,
                            "normal_correct_fraction": float((angle[mask] <= 15).mean())
                            if mask.any()
                            else None,
                            "occupied_4x4_cells": occupied,
                            "normal_singular_values": singular.tolist(),
                        }
                    )
                report.cases.append(
                    {
                        "scene": scene,
                        "frame": i,
                        "noise_seed": seed,
                        "depth_sha256": hashlib.sha256(depth.tobytes()).hexdigest(),
                        "populations": rows,
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
