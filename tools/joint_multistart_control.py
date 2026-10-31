"""Budgeted fixed-start search on the frozen synthetic joint D2D objective."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
from tools.known_geometry_joint import CONDITIONS, Report, observations, reference
from tools.run_camera_lidar_development_matrix import pose_metrics

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_depth_to_depth_solver import (
    DepthToDepthObservation,
    DepthToDepthOptions,
    evaluate_depth_to_depth_mi,
    resolve_depth_to_depth_options,
)
from calibrex.solvers.borer_six_dof_solver import BorerSixDofSolver, apply_local_se3_delta

STARTS = np.array(
    [
        [0, 0, 0, 0, 0, 0],
        [0.6, 0.6, 0.6, 0, 0, 0],
        [0.6, -0.6, -0.6, 0, 0, 0],
        [-0.6, 0.6, -0.6, 0, 0, 0],
        [-0.6, -0.6, 0.6, 0, 0, 0],
    ]
)


def search(
    batch: list[DepthToDepthObservation],
    initial: SE3,
    start: np.ndarray,
    options: DepthToDepthOptions,
    budget: int = 160,
) -> dict[str, Any]:
    """Pattern search in one common bounded coordinate system for all starts."""
    best = start.copy()
    step = np.array([0.25] * 3 + [0.1] * 3)
    minimum = np.array([0.01] * 3 + [0.005] * 3)
    bound = np.array([2.0] * 3 + [1.0] * 3)
    trace: list[dict[str, Any]] = []

    def evaluate(delta: np.ndarray) -> float:
        value = evaluate_depth_to_depth_mi(
            batch, apply_local_se3_delta(initial, delta[:3], delta[3:]), options
        )
        trace.append(
            {
                "delta": delta.tolist(),
                "mi": value.mutual_information,
                "frames": value.evaluated_frame_count,
            }
        )
        return value.mutual_information

    score = evaluate(best)
    while np.any(step >= minimum) and len(trace) < budget:
        candidates = []
        for axis in range(6):
            if step[axis] < minimum[axis]:
                continue
            for direction in [-1.0, 1.0]:
                delta = best.copy()
                delta[axis] = np.clip(
                    delta[axis] + direction * step[axis], -bound[axis], bound[axis]
                )
                if np.array_equal(delta, best):
                    continue
                candidates.append((evaluate(delta), tuple(-delta), delta))
                if len(trace) >= budget:
                    break
            if len(trace) >= budget:
                break
        if not candidates:
            break
        value, _, delta = max(candidates, key=lambda c: (c[0], c[1]))
        if value > score + 1e-9:
            best, score = delta, value
        else:
            step *= 0.5
    return {
        "start": start.tolist(),
        "delta": best.tolist(),
        "mi": score,
        "converged": bool(np.all(step < minimum)),
        "evaluations": len(trace),
        "trace": trace,
    }


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
        scope="fixed five-start versus single-start, equal maximum 800 objective calls",
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
            runs = [search(batch, initial, start, options) for start in STARTS]
            seconds = time.perf_counter() - began
            chosen = max(range(5), key=lambda i: (runs[i]["mi"], -i))
            selected = runs[chosen]
            pose = apply_local_se3_delta(initial, selected["delta"][:3], selected["delta"][3:])
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
                    "multistart": {
                        "runs": runs,
                        "selected": chosen,
                        "seconds": seconds,
                        "evaluations": sum(r["evaluations"] for r in runs),
                        "output_matrix": matrix(pose).tolist(),
                        "metrics": pose_metrics(
                            matrix(pose), matrix(reference()), selected["converged"]
                        ),
                    },
                }
            )
            output.write_text(report.model_dump_json(indent=2) + "\n")
            print(condition, sign, report.cases[-1]["multistart"]["metrics"], flush=True)
    output.with_suffix(".schema.json").write_text(
        json.dumps(Report.model_json_schema(), indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.plan, args.output)
