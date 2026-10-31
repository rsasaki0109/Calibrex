"""Camera-depth bias versus extrinsic translation on the frozen one-sided metric fit."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import least_squares
from tools.joint_metric_pose import depth_residual
from tools.joint_occlusion_pose import occlusion_residual
from tools.occlusion_geometry import observations

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_depth_to_depth_solver import DepthToDepthObservation
from calibrex.solvers.borer_six_dof_solver import apply_local_se3_delta

F_SCALE = 0.1  # soft-L1 scale of the frozen one-sided fit
# Central-difference steps: rotation deg (x, y, z), translation m (x, y, z), bias m.
STEPS = (1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4, 1e-4)
# Locally distinguishable when >= 4 % of the bias signature (1 cm RMS per 0.25 m
# bias on unit-slope rows) cannot be reproduced by any pose change.
DISTINGUISHABLE_FRACTION = 0.04


def shifted(batch: list[DepthToDepthObservation], offset_m: float) -> list[DepthToDepthObservation]:
    """Add one constant to every camera depth sample; LiDAR points are unchanged."""
    return [
        DepthToDepthObservation(
            o.frame_id,
            o.depth_map + offset_m,
            o.lidar_points,
            o.camera,
            depth_scale_convention=o.depth_scale_convention,
        )
        for o in batch
    ]


def biased_observations(scene: str, condition: str, bias_m: float) -> list[DepthToDepthObservation]:
    """Camera depth reads true optical z plus bias_m; LiDAR geometry stays exact."""
    return shifted(observations(scene, condition), bias_m)


def bias_residual(batch: list[DepthToDepthObservation], pose: SE3, bias_m: float) -> np.ndarray:
    """One-sided residual after subtracting an estimated camera-depth bias.

    Equivalent to fitting depth_map - bias_m: bilinear weights sum to one, so the
    shift moves every supported residual by +bias_m. Missing penalties stay 1 m.
    """
    parts = []
    for o in batch:
        residual, valid = depth_residual(o, pose)
        parts.append(occlusion_residual(np.where(valid, residual + bias_m, residual), valid))
    return np.concatenate(parts)


def solve(
    batch: list[DepthToDepthObservation], initial: SE3, initial_bias_m: float = 0.0
) -> dict[str, Any]:
    """Frozen one-sided fit plus one bias variable; same loss, caps and support rule."""
    calls = 0

    def residual(delta: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls > 800:
            raise RuntimeError("hard residual budget exceeded")
        pose = apply_local_se3_delta(initial, delta[:3], delta[3:6])
        return bias_residual(batch, pose, initial_bias_m + delta[6])

    initial_support = [int(depth_residual(o, initial)[1].sum()) for o in batch]
    fit = least_squares(
        residual,
        np.zeros(7),
        bounds=([-2.0] * 3 + [-1.0] * 3 + [-1.0], [2.0] * 3 + [1.0] * 3 + [1.0]),
        method="trf",
        jac="2-point",
        x_scale=[1.0] * 3 + [0.1] * 3 + [0.1],
        loss="soft_l1",
        f_scale=F_SCALE,
        max_nfev=100,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
    )
    pose = apply_local_se3_delta(initial, fit.x[:3], fit.x[3:6])
    final_support = [int(depth_residual(o, pose)[1].sum()) for o in batch]
    retained = all(
        b >= 0.9 * a and b >= 64 for a, b in zip(initial_support, final_support, strict=True)
    )
    converged = bool(fit.success and retained)
    return {
        "transform": pose.as_dict(),
        "delta": fit.x.tolist(),
        "initial_bias_m": initial_bias_m,
        "bias_m": float(initial_bias_m + fit.x[6]),
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


def identifiability(
    batch: list[DepthToDepthObservation], pose: SE3, bias_m: float
) -> dict[str, Any]:
    """Local bias/pose sensitivity of the one-sided objective at (pose, bias_m).

    Columns are central differences in solver units (deg, m, m). Rows are points
    supported at the point and under every probe, weighted by sqrt(rho') of the
    soft-L1 loss as the solver sees them. The alias is the pose change that best
    reproduces a unit bias change; a pose-only fit facing bias b moves by about
    b * alias. No reference pose is needed: callers choose where to evaluate.
    """
    evaluations = 0

    def evaluate(delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nonlocal evaluations
        evaluations += 1
        probe = apply_local_se3_delta(pose, delta[:3], delta[3:6])
        parts, valid = [], []
        for o in batch:
            residual, supported = depth_residual(o, probe)
            shifted_residual = np.where(supported, residual + bias_m + delta[6], residual)
            parts.append(occlusion_residual(shifted_residual, supported))
            valid.append(supported)
        return np.concatenate(parts), np.concatenate(valid)

    base, base_valid = evaluate(np.zeros(7))
    stable = base_valid.copy()
    columns = []
    for k, h in enumerate(STEPS):
        step = np.zeros(7)
        step[k] = h
        plus, plus_valid = evaluate(step)
        minus, minus_valid = evaluate(-step)
        stable &= (plus_valid == base_valid) & (minus_valid == base_valid)
        columns.append((plus - minus) / (2 * h))
    weight = (1 + (base / F_SCALE) ** 2) ** -0.25
    jacobian = np.stack(columns, axis=1)[stable] * weight[stable, None]
    pose_columns, bias_column = jacobian[:, :6], jacobian[:, 6]
    # Directions below 1e-9 of the largest singular value are numerical null space.
    alias = np.linalg.lstsq(pose_columns, bias_column, rcond=1e-9)[0]
    remaining = bias_column - pose_columns @ alias
    fraction = float(np.linalg.norm(remaining) / np.linalg.norm(bias_column))
    normal = jacobian.T @ jacobian
    bias_sd = correlation = None  # unbounded when the normal matrix is singular
    if np.linalg.cond(normal) < 1e14:
        covariance = np.linalg.inv(normal)
        bias_sd = float(np.sqrt(covariance[6, 6]))
        correlation = float(covariance[6, 5] / np.sqrt(covariance[6, 6] * covariance[5, 5]))
    norms = np.linalg.norm(jacobian, axis=0)
    condition = None  # unbounded inside the same 1e-9 numerical null space
    if np.all(norms > 0):
        singular = np.linalg.svd(jacobian / norms, compute_uv=False)
        if singular[-1] > 1e-9 * singular[0]:
            condition = float(singular[0] / singular[-1])
    return {
        "rows": int(stable.sum()),
        "supported_rows": int(base_valid.sum()),
        "probe_unstable_rows": int((base_valid & ~stable).sum()),
        "residual_evaluations": evaluations,
        "unexplained_bias_fraction": fraction,
        "bias_r2_on_pose": float(1 - fraction**2),
        "distinguishable": bool(fraction >= DISTINGUISHABLE_FRACTION),
        "alias_rotation_deg_per_m": alias[:3].tolist(),
        "alias_translation_m_per_m": alias[3:].tolist(),
        "bias_sd_joint_per_unit_noise_m": bias_sd,
        "bias_sd_pose_known_per_unit_noise_m": float(1 / np.sqrt(normal[6, 6])),
        "bias_tz_correlation": correlation,
        "unit_column_condition": condition,
    }
