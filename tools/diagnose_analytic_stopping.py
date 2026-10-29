"""Fixed objective/finite-difference probes of archived analytic truth-start fits."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from tools.analytic_scene_depth import AnalyticObservation
from tools.joint_analytic_pose_bias import residuals
from tools.known_geometry_joint import Report, reference
from tools.occlusion_geometry import observations

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_six_dof_solver import apply_local_se3_delta


def soft_cost(s: np.ndarray) -> float:
    return float(np.sum(s * s / (np.sqrt(1 + (s / 0.1) ** 2) + 1)))


def bias_gradient(raw: np.ndarray, saturated: np.ndarray, valid: np.ndarray) -> float:
    derivative = np.ones_like(raw)
    positive = raw > 0
    derivative[positive] = (1 + (raw[positive] / 0.2) ** 2) ** (-1.5)
    return float(np.sum((saturated / np.sqrt(1 + (saturated / 0.1) ** 2)) * derivative * valid))


def probe(batch: list[AnalyticObservation], pose: SE3, bias: float) -> dict[str, Any]:
    raw, s, valid = residuals(batch, pose, bias)
    cost = soft_cost(s)
    records = []
    for h in [1e-8, 1e-6, 1e-4]:
        forward = []
        central = []
        jumps = []
        support = []
        for axis in range(7):
            samples = []
            for sign in [1, -1]:
                delta = np.zeros(7)
                delta[axis] = sign * h
                shifted = apply_local_se3_delta(pose, delta[:3], delta[3:6])
                rr, ss, vv = residuals(batch, shifted, bias + delta[6])
                samples.append(ss)
                jumps.append(int((np.abs(rr - raw) > 1.0).sum()))
                support.append(int((vv != valid).sum()))
            forward.append((samples[0] - s) / h)
            central.append((samples[0] - samples[1]) / (2 * h))
        jf = np.column_stack(forward)
        jc = np.column_stack(central)
        weights = s / np.sqrt(1 + (s / 0.1) ** 2)
        records.append(
            {
                "step": h,
                "forward_gradient": (jf.T @ weights).tolist(),
                "central_gradient": (jc.T @ weights).tolist(),
                "forward_singular_values": np.linalg.svd(jf, compute_uv=False).tolist(),
                "central_singular_values": np.linalg.svd(jc, compute_uv=False).tolist(),
                "forward_central_relative_difference": float(
                    np.linalg.norm(jf - jc) / max(np.linalg.norm(jc), 1e-30)
                ),
                "raw_jump_over_1m_by_axis_plus_minus": jumps,
                "support_changes_by_axis_plus_minus": support,
            }
        )
    grid = np.linspace(-0.5, 0.5, 101)
    costs = [soft_cost(residuals(batch, pose, float(b))[1]) for b in grid]
    best = int(np.argmin(costs))
    return {
        "bias_m": bias,
        "cost": cost,
        "exact_bias_gradient": bias_gradient(raw, s, valid),
        "finite_difference": records,
        "bias_grid_m": grid.tolist(),
        "bias_grid_cost": costs,
        "grid_best_bias_m": float(grid[best]),
        "grid_best_cost": costs[best],
        "nearby_bias_cost": {
            str(d): soft_cost(residuals(batch, pose, bias + d)[1]) for d in [-0.01, 0.01]
        },
    }


def run(plan: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("refuse overwrite")
    spec = json.loads(plan.read_text())
    for p, sha in spec["source_sha256"].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha:
            raise ValueError("changed source")
    candidates = json.loads(Path(spec["candidates"]).read_text())["cases"]
    report = Report(
        scope="fixed diagnostic probes only; no optimizer changes or recovery claims",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
    )
    for c in candidates:
        if c["sign"] != 0:
            continue
        clean = observations(c["scene"], c["condition"])
        origins = [(0.0, 0.0, 0.0), (1.0, 0.2, 0.4), (-0.8, -0.3, 0.7)]
        batch = [
            AnalyticObservation(
                o.frame_id,
                o.depth_map + c["injected_bias_m"],
                o.lidar_points,
                o.camera,
                scene=c["scene"],
                camera_origin=origin,
                measurement_offset_m=c["injected_bias_m"],
            )
            for o, origin in zip(clean, origins, strict=True)
        ]
        p = c["result"]["transform"]
        pose = SE3(tuple(p["translation_m"]), tuple(p["rotation_quat_xyzw"]))
        report.cases.append(
            {
                "scene": c["scene"],
                "condition": c["condition"],
                "injected_bias_m": c["injected_bias_m"],
                "archived_result": c["result"],
                "archived_joint_recovered": c["joint_recovered"],
                "initial_probe": probe(batch, reference(), 0.0),
                "final_probe": probe(batch, pose, c["result"]["bias_m"]),
            }
        )
        output.write_text(report.model_dump_json(indent=2) + "\n")
        print(c["scene"], c["condition"], c["injected_bias_m"], flush=True)
    output.with_suffix(".schema.json").write_text(
        json.dumps(Report.model_json_schema(), indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.plan, args.output)
