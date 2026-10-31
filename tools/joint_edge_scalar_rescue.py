"""Retain joint pose fit, then conditionally correct bias at fixed pose."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from scipy.optimize import minimize_scalar
from tools.diagnose_analytic_stopping import soft_cost
from tools.joint_edge_pose_bias import residuals, sensitivity
from tools.joint_edge_pose_bias import solve as joint_solve

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_depth_to_depth_solver import DepthToDepthObservation


def solve(batch: list[DepthToDepthObservation], initial: SE3) -> dict[str, Any]:
    baseline = joint_solve(batch, initial)
    result = deepcopy(baseline)
    result["baseline_fit"] = baseline
    audit: dict[str, Any] = {"triggered": False, "adopted": False, "probes": []}
    result["rescue"] = audit
    calls = baseline["actual_residual_calls"]
    if not baseline["accepted"] or 800 - calls < 43:
        audit["reason"] = "baseline not accepted or insufficient budget"
        return result
    p = baseline["transform"]
    pose = SE3(tuple(p["translation_m"]), tuple(p["rotation_quat_xyzw"]))

    def objective(bias: float) -> float:
        nonlocal calls
        if calls >= 800:
            raise RuntimeError("hard residual budget exceeded")
        calls += 1
        return soft_cost(residuals(batch, pose, float(bias))[1])

    bias = baseline["bias_m"]
    before = objective(bias)
    threshold = 1e-8 * max(1.0, before)
    for step in [-0.01, 0.01]:
        candidate = min(0.5, max(-0.5, bias + step))
        audit["probes"].append({"bias_m": candidate, "cost": objective(candidate)})
    audit["cost_before"] = before
    if min(p["cost"] for p in audit["probes"]) < before - threshold:
        audit["triggered"] = True
        fit = minimize_scalar(
            objective,
            bounds=(-0.5, 0.5),
            method="bounded",
            options={"maxiter": 40, "xatol": 1e-6},
        )
        audit.update(
            scalar_success=bool(fit.success),
            candidate_bias_m=float(fit.x),
            candidate_cost=float(fit.fun),
            scalar_calls=int(fit.nfev),
        )
        if fit.success and fit.fun < before - threshold:
            diagnostic = sensitivity(batch, pose, float(fit.x))
            if not diagnostic["robust"]["near_degenerate"]:
                audit["adopted"] = True
                result.update(bias_m=float(fit.x), cost=float(fit.fun), sensitivity=diagnostic)
                result["delta"][6] = float(fit.x)
                # Status/nfev remain the joint stage's record, not a new joint optimum.
                audit["convergence_scope"] = "baseline joint termination plus scalar termination"
    result["actual_residual_calls"] = calls
    return result
