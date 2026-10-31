"""One-sided saturating residual for potentially camera-occluded LiDAR returns."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import least_squares
from tools.joint_metric_pose import depth_residual

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_depth_to_depth_solver import DepthToDepthObservation
from calibrex.solvers.borer_six_dof_solver import apply_local_se3_delta


def occlusion_residual(residual: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Saturate only supported behind-surface residuals; preserve missing penalties."""
    result = residual.copy()
    behind = valid & (residual > 0)
    result[behind] = residual[behind] / np.sqrt(1 + (residual[behind] / 0.2) ** 2)
    return result


def solve(batch: list[DepthToDepthObservation], initial: SE3) -> dict[str, Any]:
    """Fit all six axes with fixed bounds, soft-L1 scale and explicit call accounting."""
    calls = 0

    def residual(delta: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls > 800:
            raise RuntimeError("hard residual budget exceeded")
        pose = apply_local_se3_delta(initial, delta[:3], delta[3:])
        return np.concatenate([occlusion_residual(*depth_residual(o, pose)) for o in batch])

    initial_support = [int(depth_residual(o, initial)[1].sum()) for o in batch]
    fit = least_squares(
        residual,
        np.zeros(6),
        bounds=([-2.0] * 3 + [-1.0] * 3, [2.0] * 3 + [1.0] * 3),
        method="trf",
        jac="2-point",
        x_scale=[1.0] * 3 + [0.1] * 3,
        loss="soft_l1",
        f_scale=0.1,
        max_nfev=100,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
    )
    pose = apply_local_se3_delta(initial, fit.x[:3], fit.x[3:])
    final_support = [int(depth_residual(o, pose)[1].sum()) for o in batch]
    retained = all(
        b >= 0.9 * a and b >= 64 for a, b in zip(initial_support, final_support, strict=True)
    )
    converged = bool(fit.success and retained)
    return {
        "transform": pose.as_dict(),
        "delta": fit.x.tolist(),
        "converged": converged,
        "optimizer_success": bool(fit.success),
        "status": int(fit.status),
        "message": str(fit.message),
        "actual_residual_calls": calls,
        "nfev": fit.nfev,
        "njev": fit.njev,
        "initial_support": initial_support,
        "final_support": final_support,
        "support_retained": retained,
        "cost": float(fit.cost),
        "optimality": float(fit.optimality),
    }
