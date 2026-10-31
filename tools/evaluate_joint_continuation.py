"""Baseline-preserving continuation on frozen synthetic joint D2D cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from tools.joint_multistart_control import STARTS
from tools.joint_search_continuation import advance
from tools.known_geometry_joint import CONDITIONS, Report, observations, reference
from tools.run_camera_lidar_development_matrix import pose_metrics

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_depth_to_depth_solver import (
    resolve_depth_to_depth_options,
)
from calibrex.solvers.borer_six_dof_solver import BorerSixDofSolver, apply_local_se3_delta


def matrix(pose: SE3) -> np.ndarray:
    value = np.eye(4)
    value[:3, :3] = Rotation.from_quat(pose.rotation_quat_xyzw).as_matrix()
    value[:3, 3] = pose.translation_m
    return value


def run(plan: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("refuse overwrite")
    for p, sha in json.loads(plan.read_text())["source_sha256"].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha:
            raise ValueError("changed frozen source")
    report = Report(
        scope="baseline-preserving continuation, total maximum 800 objective calls",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
    )
    for condition in CONDITIONS:
        batch = observations(condition)
        options = resolve_depth_to_depth_options(batch)
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
            runs = []
            remaining = 800 - baseline.evaluation_count
            chosen = None
            if remaining >= 160:
                for start in STARTS[1:]:
                    state = {"start": start.tolist(), "delta": start.tolist()}
                    advance(batch, initial, options, state, 40)
                    runs.append(state)
                remaining -= sum(r["evaluations"] for r in runs)
                chosen = max(range(4), key=lambda i: (runs[i]["mi"], -i))
                advance(batch, initial, options, runs[chosen], remaining)
            eligible = [
                i
                for i, r in enumerate(runs)
                if r["converged"] and r["mi"] > baseline.final_evaluation.mutual_information + 1e-9
            ]
            selected_index = max(eligible, key=lambda i: (runs[i]["mi"], -i)) if eligible else None
            pose = baseline.transform_camera_lidar
            converged = baseline.status == "converged"
            if selected_index is not None:
                selected = runs[selected_index]
                pose = apply_local_se3_delta(initial, selected["delta"][:3], selected["delta"][3:])
                converged = True
            seconds = time.perf_counter() - began
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
                    "continuation": {
                        "runs": runs,
                        "refined_start": chosen,
                        "selected": selected_index,
                        "seconds": seconds,
                        "evaluations": baseline.evaluation_count
                        + sum(r["evaluations"] for r in runs),
                        "output_matrix": matrix(pose).tolist(),
                        "metrics": pose_metrics(matrix(pose), matrix(reference()), converged),
                    },
                }
            )
            output.write_text(report.model_dump_json(indent=2) + "\n")
            print(condition, sign, report.cases[-1]["continuation"]["metrics"], flush=True)
    output.with_suffix(".schema.json").write_text(
        json.dumps(Report.model_json_schema(), indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.plan, args.output)
