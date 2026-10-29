"""Joint bias fitting against frozen six-parameter controls, including planar degeneracy."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from tools.analytic_scene_depth import AnalyticObservation
from tools.joint_block_bias import solve
from tools.joint_multistart_control import matrix
from tools.known_geometry_joint import Report, reference
from tools.occlusion_geometry import SCENES, observations
from tools.run_camera_lidar_development_matrix import pose_metrics

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_six_dof_solver import apply_local_se3_delta


def run(plan: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("refuse overwrite")
    spec = json.loads(plan.read_text())
    for p, sha in spec["source_sha256"].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha:
            raise ValueError("changed frozen source")
    baseline = json.loads(Path(spec["baseline"]).read_text())["cases"]
    report = Report(
        scope="separated bias/pose blocks versus joint TRF; same analytic sampler and objective",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
    )
    setups = [(s, c) for s in SCENES for c in ["exact_z_visible", "exact_z_all"]]
    for scene, condition in setups:
        clean = observations(scene, condition)
        for bias in [0.0, 0.25, -0.25]:
            origins = [(0.0, 0.0, 0.0), (1.0, 0.2, 0.4), (-0.8, -0.3, 0.7)]
            batch = [
                AnalyticObservation(
                    o.frame_id,
                    o.depth_map + bias,
                    o.lidar_points,
                    o.camera,
                    o.depth_scale_convention,
                    scene=scene,
                    camera_origin=origin,
                    measurement_offset_m=bias,
                )
                for o, origin in zip(clean, origins, strict=True)
            ]
            for sign in [1, -1, 0]:
                initial = apply_local_se3_delta(
                    reference(),
                    tuple(sign * np.array([1.0, -0.5, 0.75])),
                    tuple(sign * np.array([0.15, -0.1, 0.1])),
                )
                began = time.perf_counter()
                result = solve(batch, initial)
                seconds = time.perf_counter() - began
                p = result["transform"]
                pose = SE3(tuple(p["translation_m"]), tuple(p["rotation_quat_xyzw"]))
                metrics = pose_metrics(matrix(pose), matrix(reference()), result["converged"])
                bias_error = abs(result["bias_m"] - bias)
                previous = next(
                    (
                        c
                        for c in baseline
                        if all(
                            c[k] == v
                            for k, v in [
                                ("scene", scene),
                                ("condition", condition),
                                ("injected_bias_m", bias),
                                ("sign", sign),
                            ]
                        )
                    ),
                    None,
                )
                report.cases.append(
                    {
                        "scene": scene,
                        "condition": condition,
                        "injected_bias_m": bias,
                        "sign": sign,
                        "baseline": {
                            k: previous[k]
                            for k in ["result", "pose_metrics", "bias_error_m", "joint_recovered"]
                        },
                        "result": result,
                        "seconds_including_diagnostics": seconds,
                        "pose_metrics": metrics,
                        "bias_error_m": bias_error,
                        "joint_recovered": bool(
                            metrics["recovered"] and result["accepted"] and bias_error < 0.05
                        ),
                    }
                )
                output.write_text(report.model_dump_json(indent=2) + "\n")
                print(
                    scene,
                    condition,
                    bias,
                    sign,
                    metrics,
                    "estimated bias",
                    result["bias_m"],
                    "identifiable",
                    result["locally_identifiable"],
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
