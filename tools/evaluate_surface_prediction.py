"""Measurement-only local disparity plane reliability against analytic surfaces."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from tools.coherent_surface import estimate
from tools.evaluate_direct_disparity import noisy_depth
from tools.known_geometry_joint import CAMERA, Report
from tools.occlusion_geometry import SCENES, observations
from tools.prepare_stereo_registration import backproject
from tools.surface_prediction_gate import gated_estimate, plane_validation


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
        scope="54 baseline/predictive surface controls,27 shared measurements",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    )
    output.with_suffix(".validation.json").write_text(
        json.dumps(plane_validation(), indent=2) + "\n"
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
                estimates = [
                    ("baseline21", estimate(depth, 21)),
                    ("predictive21", gated_estimate(depth)),
                ]
                for label, e in estimates:
                    mask = e["accepted"]
                    angle = np.degrees(
                        np.arccos(
                            np.clip(
                                np.abs(np.sum(e["normal"] * truth[10:-10, 10:-10], axis=-1)), 0, 1
                            )
                        )
                    )
                    rays = xyz[10:-10, 10:-10] / o.depth_map[10:-10, 10:-10, None]
                    position = np.linalg.norm(
                        rays * (e["depth"] - o.depth_map[10:-10, 10:-10])[..., None], axis=-1
                    )
                    correct = (angle <= 15) & (position <= 0.2)
                    vv, uu = np.nonzero(mask)
                    cells = len(
                        set(
                            zip(
                                (vv * 4 // mask.shape[0]).tolist(),
                                (uu * 4 // mask.shape[1]).tolist(),
                                strict=False,
                            )
                        )
                    )
                    singular = (
                        np.linalg.svd(e["normal"][mask], compute_uv=False)
                        if mask.sum() >= 3
                        else np.zeros(3)
                    )
                    report.cases.append(
                        {
                            "scene": scene,
                            "frame": i,
                            "noise_seed": seed,
                            "window": label,
                            "depth_sha256": hashlib.sha256(depth.tobytes()).hexdigest(),
                            "selected_count": int(mask.sum()),
                            "domain_count": int(mask.size),
                            "retention": float(mask.mean()),
                            "occupied_cells": cells,
                            "normal_singular_values": singular.tolist(),
                            "joint_precision": float(correct[mask].mean()) if mask.any() else None,
                            "angular_coverage": float((angle[mask] <= e["radius"][mask]).mean())
                            if mask.any()
                            else None,
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
