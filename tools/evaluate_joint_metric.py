"""Frozen metric-depth fitting versus MI on independent analytic scene controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from tools.joint_geometric_validation import holdout_points, validate_pose
from tools.joint_metric_pose import solve
from tools.joint_multistart_control import matrix
from tools.known_geometry_joint import CONDITIONS, Report, observations, reference
from tools.run_camera_lidar_development_matrix import pose_metrics

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_six_dof_solver import BorerSixDofSolver, apply_local_se3_delta


def run(plan: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("refuse overwrite")
    for p, sha in json.loads(plan.read_text())["source_sha256"].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha:
            raise ValueError("changed frozen source")
    report = Report(
        scope="metric optical depth versus MI; synthetic diagnostic only",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
    )
    clouds = holdout_points()
    for condition in CONDITIONS:
        batch = observations(condition)
        for sign in [1, -1, 0]:
            initial = apply_local_se3_delta(
                reference(),
                tuple(sign * np.array([1.0, -0.5, 0.75])),
                tuple(sign * np.array([0.15, -0.1, 0.1])),
            )
            began = time.perf_counter()
            baseline = BorerSixDofSolver().solve(batch, initial)
            baseline_seconds = time.perf_counter() - began
            began = time.perf_counter()
            fit = solve(batch, initial)
            seconds = time.perf_counter() - began
            pose = SE3(
                tuple(fit["transform"]["translation_m"]),
                tuple(fit["transform"]["rotation_quat_xyzw"]),
            )
            checks = [
                validate_pose(o.depth_map, p, matrix(pose), condition == "exact_range_visible")
                for o, p in zip(batch, clouds, strict=True)
            ]
            metrics = pose_metrics(matrix(pose), matrix(reference()), fit["converged"])
            report.cases.append(
                {
                    "condition": condition,
                    "sign": sign,
                    "baseline": {
                        "result": baseline.as_dict(),
                        "seconds": baseline_seconds,
                        "metrics": pose_metrics(
                            matrix(baseline.transform_camera_lidar),
                            matrix(reference()),
                            baseline.status == "converged",
                        ),
                    },
                    "metric": {
                        "result": fit,
                        "seconds": seconds,
                        "metrics": metrics,
                        "holdout": checks,
                    },
                }
            )
            output.write_text(report.model_dump_json(indent=2) + "\n")
            print(condition, sign, metrics, flush=True)
    output.with_suffix(".schema.json").write_text(
        json.dumps(Report.model_json_schema(), indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.plan, args.output)
