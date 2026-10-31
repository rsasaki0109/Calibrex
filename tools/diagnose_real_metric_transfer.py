"""Matched-input MI control and offline metric reference-versus-fit diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from tools.diagnose_analytic_stopping import soft_cost
from tools.evaluate_real_metric_transfer import as_pose, select_points
from tools.joint_edge_pose_bias import residuals as edge_residuals
from tools.joint_multistart_control import matrix
from tools.joint_pose_bias import residuals as bilinear_residuals
from tools.known_geometry_joint import Report
from tools.koide_shared_bundle_pandey import read_koide_ply
from tools.run_camera_lidar_development_matrix import pose_metrics

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_depth_to_depth_solver import (
    DepthToDepthCameraModel,
    DepthToDepthObservation,
)
from calibrex.solvers.borer_six_dof_solver import BorerSixDofSolver


def summarize(raw: np.ndarray, saturated: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    """Separate supported signed depth errors from fixed unsupported penalties."""
    values = raw[valid]
    return {
        "total": len(raw),
        "supported": int(valid.sum()),
        "cost": soft_cost(saturated),
        "supported_cost": soft_cost(saturated[valid]),
        "signed_quantiles_m": np.quantile(values, [0, 0.1, 0.5, 0.9, 1]).tolist()
        if len(values)
        else None,
        "negative_fraction": float(np.mean(values < 0)) if len(values) else None,
        "absolute_over_1m_fraction": float(np.mean(np.abs(values) > 1)) if len(values) else None,
    }


def run(plan_path: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("refuse overwrite")
    plan = json.loads(plan_path.read_text())
    for group in ["source_sha256", "input_sha256"]:
        for p, sha in plan[group].items():
            if hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha:
                raise ValueError("changed frozen file: " + p)
    transfer = json.loads(Path(plan["transfer_plan"]).read_text())
    archive = json.loads(Path(plan["transfer_results"]).read_text())["cases"]
    report = Report(
        scope="18 same-input MI fits and offline real metric cost/support diagnosis",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    )
    for case in transfer["cases"]:
        batch = []
        for frame in case["frames"]:
            depth = np.load(frame["depth"], allow_pickle=False)
            points = select_points(read_koide_ply(Path(frame["cloud"]))[:, :3])
            camera = DepthToDepthCameraModel(
                width=depth.shape[1],
                height=depth.shape[0],
                **dict(zip(("fx", "fy", "cx", "cy"), case["intrinsics"], strict=True)),
            )
            batch.append(DepthToDepthObservation(frame["name"], depth, points, camera))
        reference = as_pose(case["reference"])
        began = time.perf_counter()
        mi = BorerSixDofSolver().solve(batch, as_pose(case["initial"]))
        seconds = time.perf_counter() - began
        metrics = pose_metrics(
            matrix(mi.transform_camera_lidar), np.array(case["reference"]), mi.status == "converged"
        )
        diagnostics = []
        for method in ["bilinear", "edge", "edge_rescue"]:
            old = next(
                x for x in archive if x["case_id"] == case["case_id"] and x["method"] == method
            )
            p = old["result"]["transform"]
            fitted = SE3(tuple(p["translation_m"]), tuple(p["rotation_quat_xyzw"]))
            bias = old["result"]["bias_m"]
            residuals = bilinear_residuals if method == "bilinear" else edge_residuals
            states = {
                "reference_zero": residuals(batch, reference, 0.0),
                "reference_fitted_bias": residuals(batch, reference, bias),
                "fit_zero": residuals(batch, fitted, 0.0),
                "fit_fitted_bias": residuals(batch, fitted, bias),
            }
            common = np.logical_and.reduce([v[2] for v in states.values()])
            summaries = {}
            for name, (raw, saturated, valid) in states.items():
                summaries[name] = summarize(raw, saturated, valid)
                summaries[name]["common_support_cost"] = soft_cost(saturated[common])
            diagnostics.append(
                {
                    "method": method,
                    "archived_pose_metrics": old["pose_metrics"],
                    "archived_accepted": old["result"]["accepted"],
                    "fitted_bias_m": bias,
                    "archived_cost": old["result"]["cost"],
                    "common_supported": int(common.sum()),
                    "states": summaries,
                }
            )
        report.cases.append(
            {
                "case_id": case["case_id"],
                "start": case["start"],
                "mi_result": mi.as_dict(),
                "mi_pose_metrics": metrics,
                "mi_seconds": seconds,
                "metric_diagnostics": diagnostics,
            }
        )
        output.write_text(report.model_dump_json(indent=2) + "\n")
        print(case["case_id"], metrics, flush=True)
    output.with_suffix(".schema.json").write_text(
        json.dumps(Report.model_json_schema(), indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.plan, args.output)
