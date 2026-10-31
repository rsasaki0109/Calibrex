"""Joint bias fitting against frozen six-parameter controls, including planar degeneracy."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from tools.joint_edge_scalar_rescue import solve
from tools.joint_multistart_control import matrix
from tools.known_geometry_joint import CAMERA, Report, reference
from tools.occlusion_geometry import SCENES, observations
from tools.run_camera_lidar_development_matrix import pose_metrics

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_depth_to_depth_solver import DepthToDepthObservation
from calibrex.solvers.borer_six_dof_solver import apply_local_se3_delta


def planes() -> list[DepthToDepthObservation]:
    v, u = np.meshgrid(np.linspace(8.3, 182.3, 48), np.linspace(9.7, 308.7, 80), indexing="ij")
    rays = np.stack(
        [(u - CAMERA.cx) / CAMERA.fx, (v - CAMERA.cy) / CAMERA.fy, np.ones_like(u)], -1
    ).reshape(-1, 3)
    rotation = Rotation.from_quat(reference().rotation_quat_xyzw).as_matrix()
    return [
        DepthToDepthObservation(
            f"plane-{i}",
            np.full((192, 320), z),
            (rays * z - np.array(reference().translation_m)) @ rotation,
            CAMERA,
            "metric_z",
        )
        for i, z in enumerate([10.0, 12.0, 14.0])
    ]


def run(plan: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("refuse overwrite")
    spec = json.loads(plan.read_text())
    for p, sha in spec["source_sha256"].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha:
            raise ValueError("changed frozen source")
    baseline = json.loads(Path(spec["baseline"]).read_text())["cases"]
    report = Report(
        scope="scalar rescue with image-only sampling; 54 sphere and nine plane fits",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
    )
    setups = [(s, c) for s in SCENES for c in ["exact_z_visible", "exact_z_all"]] + [
        ("plane", "parallel_planes")
    ]
    for scene, condition in setups:
        clean = planes() if scene == "plane" else observations(scene, condition)
        for bias in [0.0, 0.25, -0.25]:
            batch = [replace(o, depth_map=o.depth_map + bias) for o in clean]
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
