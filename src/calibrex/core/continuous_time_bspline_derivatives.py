"""Analytic right-tangent control derivatives of the cumulative SO(3) spline.

Differentiate the same rotation/rate/acceleration recursion as the trajectory
evaluator. Columns are the three right perturbations of each of four local
controls. This avoids reevaluating a complete trajectory for 24 perturbations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from calibrex.core.continuous_time_bspline import (
    FloatArray,
    cumulative_basis,
    skew_batch,
    so3_exp_batch,
    so3_log_batch,
)


@dataclass(frozen=True)
class _RotationDerivatives:
    # Each derivative is (timestamps, output axis, 12 local control axes).
    rotation_tangent: FloatArray
    angular_velocity: FloatArray
    angular_acceleration: FloatArray


def _right_jacobian(vectors: FloatArray, *, inverse: bool = False) -> FloatArray:
    """SO(3) right Jacobian (or inverse), with stable small-angle coefficients."""

    squared = np.sum(vectors * vectors, axis=1)
    small = squared < 1.0e-8
    angle = np.sqrt(np.where(small, 1.0, squared))
    k = skew_batch(vectors)
    kk = k @ k
    identity = np.broadcast_to(np.eye(3), k.shape)
    if inverse:
        coefficient = np.where(
            small,
            1.0 / 12.0 + squared / 720.0 + squared * squared / 30240.0,
            (1.0 - 0.5 * angle / np.tan(0.5 * angle)) / (angle * angle),
        )
        return np.asarray(identity + 0.5 * k + coefficient[:, None, None] * kk)
    a = np.where(
        small,
        0.5 - squared / 24.0 + squared * squared / 720.0,
        (1.0 - np.cos(angle)) / (angle * angle),
    )
    b = np.where(
        small,
        1.0 / 6.0 - squared / 120.0 + squared * squared / 5040.0,
        (angle - np.sin(angle)) / (angle * angle * angle),
    )
    return np.asarray(identity - a[:, None, None] * k + b[:, None, None] * kk)


def _rotation_control_derivatives(
    rotations: FloatArray,
    fraction: FloatArray,
    spacing: float,
    *,
    order: Literal[0, 1, 2] = 2,
    relatives: FloatArray | None = None,
    deltas: FloatArray | None = None,
    steps: FloatArray | None = None,
    log_jacobians: FloatArray | None = None,
) -> _RotationDerivatives:
    """Differentiate R, body omega and body alpha for four local controls.

    For D = R_prev.T R_next and d = Log(D), the log derivative is
    Jr(d)^-1 (epsilon_next - D.T epsilon_prev). For A = Exp(l d),
    its right tangent is Jr(l d) l dd. Propagate this through
    omega_new = A.T omega + l' d and
    alpha_new = A.T alpha + l'' d + (A.T omega) cross (l' d).
    """

    basis, d_basis, dd_basis = cumulative_basis(fraction, spacing)
    count = fraction.size
    tangent = np.zeros((count, 3, 12))
    tangent[:, :, :3] = np.eye(3)
    omega = np.zeros((count, 3))
    alpha = np.zeros((count, 3))
    d_omega = np.zeros((count, 3, 12))
    d_alpha = np.zeros((count, 3, 12))
    if order == 0:
        # Only adjacent control slots influence each log increment. Accumulate
        # its two 3x3 blocks through the suffix instead of propagating dense 3x12
        # arrays containing mostly zeros at every step.
        tangent.fill(0)
        suffix = np.broadcast_to(np.eye(3), (count, 3, 3))
        for slot in range(3, 0, -1):
            relative = (np.einsum("tji,tjk->tik", rotations[:, slot - 1], rotations[:, slot])
                        if relatives is None else relatives[:, slot - 1])
            delta = so3_log_batch(relative) if deltas is None else deltas[:, slot - 1]
            log_jacobian = (_right_jacobian(delta, inverse=True)
                            if log_jacobians is None else log_jacobians[:, slot - 1])
            scaled = basis[:, slot, None] * delta
            step = so3_exp_batch(scaled) if steps is None else steps[:, slot - 1]
            block = (suffix @ _right_jacobian(scaled) @ log_jacobian) * basis[:, slot, None, None]
            tangent[:, :, 3 * slot : 3 * (slot + 1)] += block
            tangent[:, :, 3 * (slot - 1) : 3 * slot] -= block @ np.transpose(relative, (0, 2, 1))
            suffix = suffix @ np.transpose(step, (0, 2, 1))
        tangent[:, :, :3] += suffix
        return _RotationDerivatives(tangent, d_omega, d_alpha)
    for slot in range(1, 4):
        relative = (np.einsum("tji,tjk->tik", rotations[:, slot - 1], rotations[:, slot])
                    if relatives is None else relatives[:, slot - 1])
        delta = so3_log_batch(relative) if deltas is None else deltas[:, slot - 1]
        log_jacobian = (_right_jacobian(delta, inverse=True)
                        if log_jacobians is None else log_jacobians[:, slot - 1])
        d_delta = np.zeros((count, 3, 12))
        d_delta[:, :, 3 * (slot - 1) : 3 * slot] = -log_jacobian @ np.transpose(relative, (0, 2, 1))
        d_delta[:, :, 3 * slot : 3 * (slot + 1)] = log_jacobian
        scaled = basis[:, slot, None] * delta
        step = so3_exp_batch(scaled) if steps is None else steps[:, slot - 1]
        step_t = np.transpose(step, (0, 2, 1))
        d_step = (_right_jacobian(scaled) @ d_delta) * basis[:, slot, None, None]
        tangent = step_t @ tangent + d_step
        rotated_omega = np.einsum("tij,tj->ti", step_t, omega)
        rate = d_basis[:, slot, None] * delta
        d_rotated_omega = step_t @ d_omega + skew_batch(rotated_omega) @ d_step
        if order == 2:
            rotated_alpha = np.einsum("tij,tj->ti", step_t, alpha)
            d_alpha = (
                step_t @ d_alpha
                + skew_batch(rotated_alpha) @ d_step
                + dd_basis[:, slot, None, None] * d_delta
                - skew_batch(rate) @ d_rotated_omega
                + skew_batch(rotated_omega) @ (d_basis[:, slot, None, None] * d_delta)
            )
            alpha = rotated_alpha + dd_basis[:, slot, None] * delta + np.cross(rotated_omega, rate)
        d_omega = d_rotated_omega + d_basis[:, slot, None, None] * d_delta
        omega = rotated_omega + rate
    return _RotationDerivatives(tangent, d_omega, d_alpha)
