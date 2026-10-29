"""Frozen bias response and local pose/bias ambiguity diagnostic; no fitted bias."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from tools.joint_metric_pose import depth_residual
from tools.joint_multistart_control import matrix
from tools.joint_occlusion_pose import solve
from tools.known_geometry_joint import Report, reference
from tools.occlusion_geometry import SCENES, observations
from tools.run_camera_lidar_development_matrix import pose_metrics

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_depth_to_depth_solver import DepthToDepthObservation
from calibrex.solvers.borer_six_dof_solver import apply_local_se3_delta


def local_ambiguity(batch: list[DepthToDepthObservation], pose: SE3) -> dict[str, Any]:
    """Raw supported-residual Jacobian; bias correction adds b to measured z residual."""
    columns = []
    valid = np.concatenate([depth_residual(o, pose)[1] for o in batch])
    steps = np.array([0.001] * 3 + [0.001] * 3)
    scales = np.array([1.0] * 3 + [0.1] * 3)
    for axis in range(6):
        values = []
        for sign in [1, -1]:
            delta = np.zeros(6)
            delta[axis] = sign * steps[axis]
            shifted = apply_local_se3_delta(pose, delta[:3], delta[3:])
            pairs = [depth_residual(o, shifted) for o in batch]
            valid &= np.concatenate([p[1] for p in pairs])
            values.append(np.concatenate([p[0] for p in pairs]))
        columns.append((values[0] - values[1]) / (2 * steps[axis]) * scales[axis])
    j = np.column_stack(columns)[valid]
    if len(j) < 100:
        return {"supported_common": len(j), "available": False}
    bias = np.full(len(j), 0.1)
    coefficients = np.linalg.lstsq(j, bias, rcond=1e-10)[0]
    remainder = bias - j @ coefficients
    s = np.linalg.svd(np.column_stack([j, bias]), compute_uv=False)
    ratio = float(np.linalg.norm(remainder) / np.linalg.norm(bias))
    return {
        "supported_common": len(j),
        "available": True,
        "bias_unexplained_fraction": ratio,
        "near_confounded": ratio < 0.05,
        "pose_delta_mimicking_plus_10cm_bias": (coefficients * scales).tolist(),
        "scaled_singular_values": s.tolist(),
        "smallest_to_largest_singular_ratio": float(s[-1] / s[0]) if s[0] else 0.0,
        "residual_passes": 13,
    }


def run(plan: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("refuse overwrite")
    for p, sha in json.loads(plan.read_text())["source_sha256"].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha:
            raise ValueError("changed source")
    report = Report(
        scope="fixed visibility solver under depth bias; local ambiguity only, no bias fitting",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
    )
    for scene in SCENES:
        for condition in ["exact_z_visible", "exact_z_all"]:
            clean = observations(scene, condition)
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
                    ambiguity = local_ambiguity(batch, pose)
                    report.cases.append(
                        {
                            "scene": scene,
                            "condition": condition,
                            "bias_m": bias,
                            "sign": sign,
                            "result": result,
                            "seconds": seconds,
                            "metrics": metrics,
                            "translation_error_xyz_m": (
                                matrix(pose)[:3, 3] - matrix(reference())[:3, 3]
                            ).tolist(),
                            "local_ambiguity": ambiguity,
                        }
                    )
                    output.write_text(report.model_dump_json(indent=2) + "\n")
                    print(scene, condition, bias, sign, metrics, flush=True)
    output.with_suffix(".schema.json").write_text(
        json.dumps(Report.model_json_schema(), indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.plan, args.output)
