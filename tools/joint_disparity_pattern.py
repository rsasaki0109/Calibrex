"""Bounded derivative-free coordinate polling of the fixed disparity objective."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from tools.diagnose_analytic_stopping import soft_cost
from tools.joint_disparity_bias import residuals, sensitivity

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_depth_to_depth_solver import DepthToDepthObservation
from calibrex.solvers.borer_six_dof_solver import apply_local_se3_delta


def search(objective: Callable[[np.ndarray], float], budget: int = 800) -> dict[str, Any]:
    delta = np.zeros(7)
    bounds = np.array([2.0] * 3 + [1.0] * 3 + [0.5])
    step = np.array([0.25] * 3 + [0.1] * 3 + [0.05])
    minimum = np.array([0.01] * 3 + [0.005] * 4)
    calls = 1
    cost = objective(delta)
    trace = [cost]
    success = False
    while calls < budget:
        best, best_cost = delta.copy(), cost
        complete = True
        for axis in range(7):
            for direction in [-1, 1]:
                candidate = delta.copy()
                candidate[axis] = np.clip(
                    delta[axis] + direction * step[axis], -bounds[axis], bounds[axis]
                )
                if np.array_equal(candidate, delta):
                    continue
                if calls >= budget:
                    complete = False
                    break
                value = objective(candidate)
                calls += 1
                if value < best_cost - 1e-9:
                    best, best_cost = candidate, value
            if not complete:
                break
        improved = best_cost < cost - 1e-9
        delta, cost = best, best_cost
        trace.append(cost)
        if not complete:
            break
        if not improved:
            if np.all(step <= minimum):
                success = True
                break
            step = np.maximum(minimum, step / 2)
    return {
        "delta": delta,
        "cost": cost,
        "calls": calls,
        "success": success,
        "trace": trace,
        "step": step,
    }


def solve(batch: list[DepthToDepthObservation], initial: SE3) -> dict[str, Any]:
    def objective(delta: np.ndarray) -> float:
        pose = apply_local_se3_delta(initial, delta[:3], delta[3:6])
        return soft_cost(residuals(batch, pose, float(delta[6]))[1])

    before = [int(residuals([o], initial, 0)[2].sum()) for o in batch]
    fit = search(objective)
    delta = fit["delta"]
    pose = apply_local_se3_delta(initial, delta[:3], delta[3:6])
    after = [int(residuals([o], pose, float(delta[6]))[2].sum()) for o in batch]
    retained = all(b >= 0.9 * a and b >= 64 for a, b in zip(before, after, strict=True))
    diagnostic = sensitivity(batch, pose, float(delta[6]))
    identifiable = not diagnostic["robust"]["near_degenerate"]
    success = fit["success"]
    return {
        "transform": pose.as_dict(),
        "delta": delta.tolist(),
        "bias_m": float(delta[6]),
        "cost": fit["cost"],
        "actual_residual_calls": fit["calls"],
        "optimizer_success": success,
        "converged": bool(success and retained),
        "accepted": bool(success and retained and identifiable),
        "locally_identifiable": identifiable,
        "initial_support": before,
        "final_support": after,
        "sensitivity": diagnostic,
        "cost_trace": fit["trace"],
        "final_steps": fit["step"].tolist(),
        "stopping_reason": "complete minimum-step poll without improvement"
        if success
        else "budget",
    }
