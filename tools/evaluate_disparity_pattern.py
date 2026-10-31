"""Paired fixed-metric versus uncertainty-normalized synthetic stereo control."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from tools.evaluate_pose_bias import planes
from tools.joint_disparity_pattern import solve as uncertain_solve
from tools.joint_multistart_control import matrix
from tools.known_geometry_joint import Report, reference
from tools.occlusion_geometry import SCENES, observations
from tools.run_camera_lidar_development_matrix import pose_metrics

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_six_dof_solver import apply_local_se3_delta


def noisy_depth(depth: np.ndarray, seed: int, sigma_px: float) -> np.ndarray:
    """Add disparity noise before inversion; invalid nonpositive disparities stay missing."""
    disparity = 120.0 / depth + np.random.default_rng(seed).normal(0, sigma_px, depth.shape)
    return np.divide(120.0, disparity, out=np.full(depth.shape, np.nan), where=disparity > 0)


def run(plan_path: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("refuse overwrite")
    plan = json.loads(plan_path.read_text())
    for p, sha in plan["source_sha256"].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha:
            raise ValueError("changed frozen source: " + p)
    report = Report(
        scope="171 coordinate-search disparity fits versus archived TRF",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    )
    setups = [(s, c) for s in SCENES for c in ["exact_z_visible", "exact_z_all"]]
    setups.append(("plane", "parallel_planes"))
    for scene, condition in setups:
        clean = planes() if scene == "plane" else observations(scene, condition)
        noise_specs = [(0.0, 0)] if scene == "plane" else [(0.0, 0), (0.5, 730), (0.5, 731)]
        for sigma_px, seed in noise_specs:
            depths = [
                o.depth_map.copy()
                if sigma_px == 0
                else noisy_depth(o.depth_map, seed + 100 * i, sigma_px)
                for i, o in enumerate(clean)
            ]
            for bias in [0.0, 0.25, -0.25]:
                batch = [replace(o, depth_map=d + bias) for o, d in zip(clean, depths, strict=True)]
                hashes = [hashlib.sha256(o.depth_map.tobytes()).hexdigest() for o in batch]
                for sign in [1, -1, 0]:
                    initial = apply_local_se3_delta(
                        reference(),
                        tuple(sign * np.array([1.0, -0.5, 0.75])),
                        tuple(sign * np.array([0.15, -0.1, 0.1])),
                    )
                    for method, solve in [("pattern", uncertain_solve)]:
                        began = time.perf_counter()
                        result = solve(batch, initial)
                        seconds = time.perf_counter() - began
                        p = result["transform"]
                        pose = SE3(tuple(p["translation_m"]), tuple(p["rotation_quat_xyzw"]))
                        metrics = pose_metrics(
                            matrix(pose), matrix(reference()), result["converged"]
                        )
                        bias_error = abs(result["bias_m"] - bias)
                        report.cases.append(
                            {
                                "scene": scene,
                                "condition": condition,
                                "sigma_px": sigma_px,
                                "noise_seed": seed,
                                "injected_bias_m": bias,
                                "sign": sign,
                                "method": method,
                                "depth_array_sha256": hashes,
                                "result": result,
                                "seconds_including_diagnostics": seconds,
                                "pose_metrics": metrics,
                                "bias_error_m": bias_error,
                                "joint_recovered": bool(
                                    metrics["recovered"]
                                    and result["accepted"]
                                    and bias_error < 0.05
                                ),
                            }
                        )
                        output.write_text(report.model_dump_json(indent=2) + "\n")
                        print(
                            scene,
                            condition,
                            sigma_px,
                            seed,
                            bias,
                            sign,
                            method,
                            report.cases[-1]["joint_recovered"],
                            flush=True,
                        )
    output.with_suffix(".schema.json").write_text(
        json.dumps(Report.model_json_schema(), indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.plan, args.output)
