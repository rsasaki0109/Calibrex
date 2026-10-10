"""Uniform cumulative cubic B-spline trajectory on SO(3) x R^3.

The rotation and the position are separate cumulative cubic B-splines that
share uniform knots (the "split" representation):

``R(t) = R_i prod_{j=1..3} Exp(l_j(u) d_j)`` with ``d_j = Log(R_{i+j-1}^T R_{i+j})``
and ``p(t) = p_i + sum_{j=1..3} l_j(u) (p_{i+j} - p_{i+j-1})``,

where ``i`` is the segment containing ``t``, ``u`` its fraction, and ``l_j``
the cumulative cubic basis.  Unlike the screw-linear knot path, the curve is
C2: body angular velocity, angular acceleration and world linear acceleration
are continuous and analytic, which is what gyro and accelerometer factors need.
Angular derivatives use the recursion of Sommer et al. (2020),
``w_j = A_j^T w_{j-1} + l_j' d_j`` and
``a_j = A_j^T a_{j-1} + l_j'' d_j + (A_j^T w_{j-1}) x (l_j' d_j)``
with ``A_j = Exp(l_j d_j)``.

Control point ``k`` sits near the curve at ``start + (k - 1) * spacing``; the
valid time domain is ``[start, start + (N - 3) * spacing]`` for ``N`` control
points.  Evaluation is vectorized over many timestamps at once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3
from calibrex.core.se3_manifold import _matrix_pose, _pose_from_matrix

FloatArray: TypeAlias = NDArray[np.float64]

# Cumulative uniform cubic basis: l(u) = C @ [1, u, u^2, u^3].
_CUMULATIVE_BASIS: FloatArray = (
    np.array(
        [
            [6.0, 0.0, 0.0, 0.0],
            [5.0, 3.0, -3.0, 1.0],
            [1.0, 3.0, 3.0, -2.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    / 6.0
)


@dataclass(frozen=True)
class BSplineStates:
    """Trajectory states at ``T`` timestamps (leading axis ``T``)."""

    rotation_world_body: FloatArray
    position_world_m: FloatArray
    angular_velocity_body_rad_s: FloatArray
    angular_acceleration_body_rad_s2: FloatArray
    linear_velocity_world_m_s: FloatArray
    linear_acceleration_world_m_s2: FloatArray


@dataclass(frozen=True)
class CubicBSplineTrajectory:
    """Control points of a uniform cumulative cubic B-spline on SO(3) x R^3."""

    start_time_sec: float
    knot_spacing_sec: float
    rotations: FloatArray
    positions: FloatArray

    def __post_init__(self) -> None:
        rotations = np.asarray(self.rotations, dtype=float)
        positions = np.asarray(self.positions, dtype=float)
        if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
            raise ValueError("B-spline rotations must have shape (N, 3, 3)")
        if positions.shape != (rotations.shape[0], 3):
            raise ValueError("B-spline positions must have shape (N, 3)")
        if rotations.shape[0] < 4:
            raise ValueError("a cubic B-spline needs at least four control points")
        if not math.isfinite(self.start_time_sec):
            raise ValueError("B-spline start time must be finite")
        if not (math.isfinite(self.knot_spacing_sec) and self.knot_spacing_sec > 0.0):
            raise ValueError("B-spline knot spacing must be positive")
        if not (np.all(np.isfinite(rotations)) and np.all(np.isfinite(positions))):
            raise ValueError("B-spline control points must be finite")
        object.__setattr__(self, "rotations", rotations)
        object.__setattr__(self, "positions", positions)

    @property
    def control_point_count(self) -> int:
        """Return the number of control points."""

        return int(self.rotations.shape[0])

    @property
    def end_time_sec(self) -> float:
        """Return the last time inside the valid domain."""

        return self.start_time_sec + (self.control_point_count - 3) * self.knot_spacing_sec

    def segments(self, times_sec: FloatArray) -> tuple[NDArray[np.int64], FloatArray]:
        """Return the segment index and fraction ``u`` of each timestamp."""

        times = np.asarray(times_sec, dtype=float).reshape(-1)
        if times.size and (
            float(np.min(times)) < self.start_time_sec - 1.0e-9
            or float(np.max(times)) > self.end_time_sec + 1.0e-9
        ):
            raise ValueError("timestamp is outside the B-spline time domain")
        scaled = (times - self.start_time_sec) / self.knot_spacing_sec
        index = np.clip(np.floor(scaled).astype(np.int64), 0, self.control_point_count - 4)
        fraction = np.clip(scaled - index, 0.0, 1.0)
        return index, fraction

    def evaluate(self, times_sec: FloatArray) -> BSplineStates:
        """Evaluate pose and derivatives at many timestamps at once."""

        index, fraction = self.segments(times_sec)
        slots = index[:, None] + np.arange(4)[None, :]
        return evaluate_local_spline(
            self.rotations[slots],
            self.positions[slots],
            fraction,
            self.knot_spacing_sec,
        )

    def pose_at(self, time_sec: float) -> SE3:
        """Return the world-from-body pose at one timestamp."""

        states = self.evaluate(np.asarray([time_sec], dtype=float))
        return _pose_from_matrix(states.rotation_world_body[0], states.position_world_m[0])

    @classmethod
    def from_poses(
        cls,
        start_time_sec: float,
        knot_spacing_sec: float,
        poses: list[SE3] | tuple[SE3, ...],
    ) -> CubicBSplineTrajectory:
        """Seed control point ``k`` with the pose at ``start + (k - 1) * spacing``.

        ``poses`` must hold ``N`` poses sampled at those times (the first one a
        spacing before the domain start).  The result passes near the samples;
        a fit refines it.
        """

        rotations = []
        positions = []
        for pose in poses:
            rotation, translation = _matrix_pose(pose)
            rotations.append(rotation)
            positions.append(translation)
        return cls(
            start_time_sec=start_time_sec,
            knot_spacing_sec=knot_spacing_sec,
            rotations=np.asarray(rotations, dtype=float),
            positions=np.asarray(positions, dtype=float),
        )


def evaluate_local_spline(
    rotations: FloatArray,
    positions: FloatArray,
    fraction: FloatArray,
    knot_spacing_sec: float,
) -> BSplineStates:
    """Evaluate ``T`` segments from their four local control points each.

    ``rotations`` has shape ``(T, 4, 3, 3)``, ``positions`` ``(T, 4, 3)`` and
    ``fraction`` ``(T,)``.  Exposed so that a fitter can perturb one local slot
    for every timestamp at once.
    """

    u = np.asarray(fraction, dtype=float).reshape(-1)
    count = u.size
    powers = np.stack([np.ones(count), u, u * u, u * u * u], axis=1)
    d_powers = np.stack([np.zeros(count), np.ones(count), 2.0 * u, 3.0 * u * u], axis=1)
    dd_powers = np.stack([np.zeros(count), np.zeros(count), 2.0 * np.ones(count), 6.0 * u], axis=1)
    basis = powers @ _CUMULATIVE_BASIS.T
    d_basis = d_powers @ _CUMULATIVE_BASIS.T / knot_spacing_sec
    dd_basis = dd_powers @ _CUMULATIVE_BASIS.T / (knot_spacing_sec * knot_spacing_sec)

    rotation = np.array(rotations[:, 0], dtype=float, copy=True)
    omega = np.zeros((count, 3), dtype=float)
    alpha = np.zeros((count, 3), dtype=float)
    position = np.array(positions[:, 0], dtype=float, copy=True)
    velocity = np.zeros((count, 3), dtype=float)
    acceleration = np.zeros((count, 3), dtype=float)
    for j in range(1, 4):
        relative = np.einsum("tji,tjk->tik", rotations[:, j - 1], rotations[:, j])
        delta = so3_log_batch(relative)
        step = so3_exp_batch(basis[:, j, None] * delta)
        step_t = np.transpose(step, (0, 2, 1))
        rate = d_basis[:, j, None] * delta
        rotated_omega = np.einsum("tij,tj->ti", step_t, omega)
        alpha = (
            np.einsum("tij,tj->ti", step_t, alpha)
            + dd_basis[:, j, None] * delta
            + np.cross(rotated_omega, rate)
        )
        omega = rotated_omega + rate
        rotation = np.einsum("tij,tjk->tik", rotation, step)
        difference = positions[:, j] - positions[:, j - 1]
        position = position + basis[:, j, None] * difference
        velocity = velocity + d_basis[:, j, None] * difference
        acceleration = acceleration + dd_basis[:, j, None] * difference
    return BSplineStates(
        rotation_world_body=rotation,
        position_world_m=position,
        angular_velocity_body_rad_s=omega,
        angular_acceleration_body_rad_s2=alpha,
        linear_velocity_world_m_s=velocity,
        linear_acceleration_world_m_s2=acceleration,
    )


def skew_batch(vectors: FloatArray) -> FloatArray:
    """Return skew-symmetric matrices for an ``(T, 3)`` array."""

    v = np.asarray(vectors, dtype=float)
    zero = np.zeros(v.shape[0], dtype=float)
    return np.stack(
        [
            np.stack([zero, -v[:, 2], v[:, 1]], axis=1),
            np.stack([v[:, 2], zero, -v[:, 0]], axis=1),
            np.stack([-v[:, 1], v[:, 0], zero], axis=1),
        ],
        axis=1,
    )


def so3_exp_batch(rotation_vectors: FloatArray) -> FloatArray:
    """Rodrigues' formula for an ``(T, 3)`` array of rotation vectors."""

    vectors = np.asarray(rotation_vectors, dtype=float).reshape(-1, 3)
    angle = np.linalg.norm(vectors, axis=1)
    small = angle < 1.0e-8
    safe = np.where(small, 1.0, angle)
    # Series expansions keep the small-angle coefficients accurate.
    a = np.where(small, 1.0 - angle * angle / 6.0, np.sin(safe) / safe)
    b = np.where(small, 0.5 - angle * angle / 24.0, (1.0 - np.cos(safe)) / (safe * safe))
    k = skew_batch(vectors)
    identity = np.broadcast_to(np.eye(3), k.shape)
    return np.asarray(
        identity + a[:, None, None] * k + b[:, None, None] * np.einsum("tij,tjk->tik", k, k),
        dtype=float,
    )


def so3_log_batch(rotations: FloatArray) -> FloatArray:
    """Rotation vectors of an ``(T, 3, 3)`` array of rotations (angle < pi)."""

    matrices = np.asarray(rotations, dtype=float).reshape(-1, 3, 3)
    trace = np.trace(matrices, axis1=1, axis2=2)
    cosine = np.clip(0.5 * (trace - 1.0), -1.0, 1.0)
    angle = np.arccos(cosine)
    vee = 0.5 * np.stack(
        [
            matrices[:, 2, 1] - matrices[:, 1, 2],
            matrices[:, 0, 2] - matrices[:, 2, 0],
            matrices[:, 1, 0] - matrices[:, 0, 1],
        ],
        axis=1,
    )
    small = angle < 1.0e-8
    sine = np.sin(np.where(small, 1.0, angle))
    scale = np.where(small, 1.0 + angle * angle / 6.0, angle / sine)
    if np.any(angle > math.pi - 1.0e-6):
        raise ValueError("relative control-point rotation is too close to pi")
    return np.asarray(scale[:, None] * vee, dtype=float)
