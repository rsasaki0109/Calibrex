"""Budgeted alternating scalar-bias and pose updates for the analytic control."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from tools.diagnose_analytic_stopping import soft_cost
from tools.joint_analytic_pose_bias import residuals, sensitivity

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_six_dof_solver import apply_local_se3_delta


def solve(batch: list, initial: SE3) -> dict[str, Any]:
    calls = 0
    current = np.zeros(7)
    trace = []

    def evaluate(delta: np.ndarray) -> np.ndarray:
        nonlocal calls
        if calls >= 800:
            raise RuntimeError("hard budget exceeded")
        calls += 1
        pose = apply_local_se3_delta(initial, delta[:3], delta[3:6])
        return residuals(batch, pose, float(delta[6]))[1]

    value = soft_cost(evaluate(current))
    before = [int(residuals([o], initial, 0.0)[2].sum()) for o in batch]
    converged = False
    for cycle in range(8):
        if 800 - calls < 48:
            break
        previous = current.copy()
        previous_cost = value
        fixed = current.copy()

        def scalar(b: float, fixed: np.ndarray = fixed) -> float:
            candidate = fixed.copy()
            candidate[6] = b
            return soft_cost(evaluate(candidate))

        scalar_fit = minimize_scalar(
            scalar, bounds=(-0.5, 0.5), method="bounded", options={"maxiter": 40, "xatol": 1e-6}
        )
        scalar_improved = bool(scalar_fit.fun < value)
        if scalar_improved:
            current[6] = scalar_fit.x
            value = float(scalar_fit.fun)
        after_bias = value
        remaining = 800 - calls
        pose_cap = min(20, remaining // 7)
        if pose_cap < 2:
            break
        fixed_bias = float(current[6])

        def pose_residual(delta: np.ndarray, fixed_bias: float = fixed_bias) -> np.ndarray:
            return evaluate(np.r_[delta, fixed_bias])

        fit = least_squares(
            pose_residual,
            current[:6],
            bounds=([-2.0] * 3 + [-1.0] * 3, [2.0] * 3 + [1.0] * 3),
            method="trf",
            jac="2-point",
            x_scale=[1.0] * 3 + [0.1] * 3,
            loss="soft_l1",
            f_scale=0.1,
            max_nfev=pose_cap,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
        pose_improved = bool(fit.cost < value)
        if pose_improved:
            current[:6] = fit.x
            value = float(fit.cost)
        stable_cost = (previous_cost - value) <= 1e-8 * max(1.0, previous_cost)
        stable_bias = abs(current[6] - previous[6]) <= 1e-6
        converged = bool(scalar_fit.success and fit.success and stable_cost and stable_bias)
        trace.append(
            {
                "cycle": cycle,
                "cost_before": previous_cost,
                "cost_after_bias": after_bias,
                "cost_after_pose": value,
                "bias_update_accepted": scalar_improved,
                "pose_update_accepted": pose_improved,
                "bias_stage_success": bool(scalar_fit.success),
                "pose_stage_success": bool(fit.success),
                "pose_stage_status": int(fit.status),
                "stable_cost": stable_cost,
                "stable_bias": stable_bias,
                "delta": current.tolist(),
                "cumulative_calls": calls,
            }
        )
        if converged:
            break
    pose = apply_local_se3_delta(initial, current[:3], current[3:6])
    after = [int(residuals([o], pose, float(current[6]))[2].sum()) for o in batch]
    retained = all(b >= 0.9 * a and b >= 64 for a, b in zip(before, after, strict=True))
    diagnostic = sensitivity(batch, pose, float(current[6]))
    identifiable = not diagnostic["robust"]["near_degenerate"]
    return {
        "transform": pose.as_dict(),
        "bias_m": float(current[6]),
        "delta": current.tolist(),
        "converged": bool(converged and retained),
        "optimizer_success": converged,
        "accepted": bool(converged and retained and identifiable),
        "locally_identifiable": identifiable,
        "actual_residual_calls": calls,
        "cost": value,
        "initial_support": before,
        "final_support": after,
        "cycles": trace,
        "sensitivity": diagnostic,
        "stopping_reason": "stable complete cycle" if converged else "budget or cycle cap",
    }
