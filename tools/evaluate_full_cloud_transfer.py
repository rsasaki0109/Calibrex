"""Frozen spent-KITTI transfer of metric bias fitting and image-only rescue."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from tools.diagnose_analytic_stopping import soft_cost
from tools.diagnose_real_metric_transfer import summarize
from tools.joint_edge_pose_bias import residuals as edge_residuals
from tools.joint_edge_scalar_rescue import solve as rescue_solve
from tools.joint_multistart_control import matrix
from tools.joint_pose_bias import residuals as bilinear_residuals
from tools.joint_pose_bias import solve as bilinear_solve
from tools.known_geometry_joint import Report
from tools.koide_shared_bundle_pandey import read_koide_ply
from tools.run_camera_lidar_development_matrix import pose_metrics

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_depth_to_depth_solver import (
    DepthToDepthCameraModel,
    DepthToDepthObservation,
)
from calibrex.solvers.borer_six_dof_solver import BorerSixDofSolver


def solve_mi(batch: list[DepthToDepthObservation], initial: SE3) -> dict:
    fit = BorerSixDofSolver().solve(batch, initial)
    return {
        "transform": fit.transform_camera_lidar.as_dict(),
        "converged": fit.status == "converged",
        "accepted": fit.status == "converged",
        "evaluation_count": fit.evaluation_count,
        "mi_result": fit.as_dict(),
    }


def as_pose(transform: list[list[float]]) -> SE3:
    m = np.asarray(transform)
    return SE3(tuple(m[:3, 3]), tuple(Rotation.from_matrix(m[:3, :3]).as_quat()))


def run(plan_path: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("refuse overwrite")
    plan = json.loads(plan_path.read_text())
    for group in ["source_sha256", "input_sha256"]:
        for path, sha in plan[group].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
                raise ValueError("changed frozen file: " + path)
    report = Report(
        scope="full-cloud spent KITTI SGBM; 6 drives x 3 starts x MI/bilinear/edge-rescue",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    )
    for case in plan["cases"]:
        batch = []
        counts = []
        for frame in case["frames"]:
            depth = np.load(frame["depth"], allow_pickle=False)
            full = read_koide_ply(Path(frame["cloud"]))[:, :3]
            points = full
            model = DepthToDepthCameraModel(
                width=depth.shape[1],
                height=depth.shape[0],
                **dict(zip(("fx", "fy", "cx", "cy"), case["intrinsics"], strict=True)),
            )
            batch.append(DepthToDepthObservation(frame["name"], depth, points, model))
            counts.append({"frame": frame["name"], "full": len(full), "selected": len(points)})
        reference = np.array(case["reference"])
        initial = as_pose(case["initial"])
        for method, solve in [
            ("bilinear", bilinear_solve),
            ("mi", solve_mi),
            ("edge_rescue", rescue_solve),
        ]:
            began = time.perf_counter()
            result = solve(batch, initial)
            seconds = time.perf_counter() - began
            p = result["transform"]
            pose = SE3(tuple(p["translation_m"]), tuple(p["rotation_quat_xyzw"]))
            metrics = pose_metrics(matrix(pose), reference, result["converged"])
            diagnostic = None
            if method != "mi":
                residuals = bilinear_residuals if method == "bilinear" else edge_residuals
                bias = result["bias_m"]
                states = {
                    "reference_zero": residuals(batch, as_pose(case["reference"]), 0.0),
                    "reference_fitted_bias": residuals(batch, as_pose(case["reference"]), bias),
                    "fit_zero": residuals(batch, pose, 0.0),
                    "fit_fitted_bias": residuals(batch, pose, bias),
                }
                common = np.logical_and.reduce([v[2] for v in states.values()])
                diagnostic = {"common_supported": int(common.sum()), "states": {}}
                for name, (raw, saturated, valid) in states.items():
                    stats = summarize(raw, saturated, valid)
                    stats["common_support_cost"] = soft_cost(saturated[common])
                    diagnostic["states"][name] = stats
            report.cases.append(
                {
                    "case_id": case["case_id"],
                    "start": case["start"],
                    "method": method,
                    "point_counts": counts,
                    "diagnostic": diagnostic,
                    "result": result,
                    "pose_metrics": metrics,
                    "accepted_pose_recovered": bool(metrics["recovered"] and result["accepted"]),
                    "seconds_including_diagnostics": seconds,
                }
            )
            output.write_text(report.model_dump_json(indent=2) + "\n")
            print(case["case_id"], method, metrics, "accepted", result["accepted"], flush=True)
    output.with_suffix(".schema.json").write_text(
        json.dumps(Report.model_json_schema(), indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.plan, args.output)
