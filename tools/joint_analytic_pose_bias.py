"""Oracle-sampler control: same seven-parameter solver and diagnostics, new sampler only."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import least_squares
from tools.analytic_scene_depth import depth_residual
from tools.joint_occlusion_pose import occlusion_residual

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_depth_to_depth_solver import DepthToDepthObservation
from calibrex.solvers.borer_six_dof_solver import apply_local_se3_delta


def residuals(
    batch: list[DepthToDepthObservation], pose: SE3, bias: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pairs = [depth_residual(o, pose) for o in batch]
    valid = np.concatenate([v for _, v in pairs])
    raw = np.concatenate([r for r, _ in pairs])
    raw = raw + np.where(valid, bias, 0.0)
    saturated = occlusion_residual(raw, valid)
    return raw, saturated, valid


def sensitivity(batch: list[DepthToDepthObservation], pose: SE3, bias: float) -> dict[str, Any]:
    """Local raw versus exact soft-L1 equivalent-residual sensitivity; not covariance."""
    valid = residuals(batch, pose, bias)[2]
    raw_columns, weighted_columns = [], []
    scales = np.array([1.0] * 3 + [0.1] * 4)
    for axis in range(7):
        values = []
        for sign in [1, -1]:
            delta = np.zeros(7)
            delta[axis] = sign * 0.001
            shifted = apply_local_se3_delta(pose, delta[:3], delta[3:6])
            raw, saturated, supported = residuals(batch, shifted, bias + delta[6])
            valid &= supported
            equivalent = saturated * np.sqrt(2 / (np.sqrt(1 + (saturated / 0.1) ** 2) + 1))
            values.append((raw, equivalent))
        raw_columns.append((values[0][0] - values[1][0]) / 0.002 * scales[axis])
        weighted_columns.append((values[0][1] - values[1][1]) / 0.002 * scales[axis])
    result: dict[str, Any] = {"common_supported": int(valid.sum()), "diagnostic_passes": 15}
    for name, columns in [("raw", raw_columns), ("robust", weighted_columns)]:
        j = np.column_stack(columns)[valid]
        if len(j) < 100:
            result[name] = {"available": False, "near_degenerate": True}
            continue
        singular = np.linalg.svd(j, compute_uv=False)
        ratio = float(singular[-1] / singular[0]) if singular[0] > 0 else 0.0
        b = j[:, 6]
        projection = j[:, :6] @ np.linalg.lstsq(j[:, :6], b, rcond=1e-10)[0]
        unexplained = (
            float(np.linalg.norm(b - projection) / np.linalg.norm(b))
            if np.linalg.norm(b) > 0
            else 0.0
        )
        result[name] = {
            "available": True,
            "scaled_singular_values": singular.tolist(),
            "singular_ratio": ratio,
            "bias_unexplained_fraction": unexplained,
            "near_degenerate": ratio < 1e-6 or unexplained < 0.05,
        }
    return result


def solve(batch: list[DepthToDepthObservation], initial: SE3) -> dict[str, Any]:
    calls = 0

    def objective(delta: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls > 800:
            raise RuntimeError("hard residual budget exceeded")
        pose = apply_local_se3_delta(initial, delta[:3], delta[3:6])
        return residuals(batch, pose, float(delta[6]))[1]

    before = [int(depth_residual(o, initial)[1].sum()) for o in batch]
    fit = least_squares(
        objective,
        np.zeros(7),
        bounds=([-2.0] * 3 + [-1.0] * 3 + [-0.5], [2.0] * 3 + [1.0] * 3 + [0.5]),
        method="trf",
        jac="2-point",
        x_scale=[1.0] * 3 + [0.1] * 4,
        loss="soft_l1",
        f_scale=0.1,
        max_nfev=100,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
    )
    pose = apply_local_se3_delta(initial, fit.x[:3], fit.x[3:6])
    after = [int(depth_residual(o, pose)[1].sum()) for o in batch]
    retained = all(b >= 0.9 * a and b >= 64 for a, b in zip(before, after, strict=True))
    diagnostic = sensitivity(batch, pose, float(fit.x[6]))
    identifiable = not diagnostic["robust"]["near_degenerate"]
    return {
        "transform": pose.as_dict(),
        "bias_m": float(fit.x[6]),
        "delta": fit.x.tolist(),
        "converged": bool(fit.success and retained),
        "optimizer_success": bool(fit.success),
        "locally_identifiable": identifiable,
        "accepted": bool(fit.success and retained and identifiable),
        "status": int(fit.status),
        "message": str(fit.message),
        "actual_residual_calls": calls,
        "nfev": fit.nfev,
        "njev": fit.njev,
        "initial_support": before,
        "final_support": after,
        "cost": float(fit.cost),
        "sensitivity": diagnostic,
    }
