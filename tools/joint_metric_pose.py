"""Experimental bounded metric optical-depth fitting on fixed point populations."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_depth_to_depth_solver import DepthToDepthObservation
from calibrex.solvers.borer_six_dof_solver import apply_local_se3_delta


def depth_residual(
    observation: DepthToDepthObservation, pose: SE3
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear metric-z residuals; every unsupported point keeps a 1 m penalty."""
    camera = observation.camera
    rotation = Rotation.from_quat(pose.rotation_quat_xyzw).as_matrix()
    xyz = observation.lidar_points @ rotation.T + pose.translation_m
    u = camera.fx * xyz[:, 0] / np.maximum(xyz[:, 2], 1e-9) + camera.cx
    v = camera.fy * xyz[:, 1] / np.maximum(xyz[:, 2], 1e-9) + camera.cy
    valid = (
        (xyz[:, 2] > 0.1) & (u >= 0) & (u < camera.width - 1) & (v >= 0) & (v < camera.height - 1)
    )
    x = np.floor(np.clip(u, 0, camera.width - 2)).astype(int)
    y = np.floor(np.clip(v, 0, camera.height - 2)).astype(int)
    depth = observation.depth_map
    if observation.depth_scale_convention == "metric_range":
        vv, uu = np.indices(depth.shape)
        depth = depth / np.sqrt(
            1 + ((uu - camera.cx) / camera.fx) ** 2 + ((vv - camera.cy) / camera.fy) ** 2
        )
    samples = np.stack([depth[y, x], depth[y, x + 1], depth[y + 1, x], depth[y + 1, x + 1]], axis=1)
    valid &= np.isfinite(samples).all(axis=1) & (samples > 0).all(axis=1)
    a, b = u - x, v - y
    weights = np.stack([(1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b], axis=1)
    measured = np.sum(np.where(np.isfinite(samples), samples, 0) * weights, axis=1)
    residual = np.where(valid, xyz[:, 2] - measured, 1.0)
    return residual, valid


def solve(batch: list[DepthToDepthObservation], initial: SE3) -> dict[str, Any]:
    """Fit all six axes with fixed bounds, soft-L1 scale and explicit call accounting."""
    calls = 0

    def residual(delta: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls > 800:
            raise RuntimeError("hard residual budget exceeded")
        pose = apply_local_se3_delta(initial, delta[:3], delta[3:])
        return np.concatenate([depth_residual(o, pose)[0] for o in batch])

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
