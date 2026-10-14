"""Optional fused CPU plane residual/Jacobian evaluator for the joint spline fit.

Small SO(3) temporaries are reused within a serial, nogil kernel. Independent
row slices run in Python threads; Numba's automatic parallel pass is disabled.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from typing import TYPE_CHECKING, TypeAlias, cast

import numpy as np
from numba import njit
from numpy.typing import NDArray

from calibrex.core.continuous_time_bspline import (
    CubicBSplineTrajectory,
    cumulative_basis,
    so3_log_batch,
)
from calibrex.core.continuous_time_bspline_derivatives import _right_jacobian

if TYPE_CHECKING:
    from calibrex.core.continuous_time_bspline_fit import BSplinePlaneReturns

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


@njit(inline="always", fastmath=False)  # type: ignore[untyped-decorator]
def _multiply(a: FloatArray, b: FloatArray, out: FloatArray) -> None:
    for i in range(3):
        for j in range(3):
            out[i, j] = (a[i, 0] * b[0, j] + a[i, 1] * b[1, j]) + a[i, 2] * b[2, j]


@njit(inline="always", fastmath=False)  # type: ignore[untyped-decorator]
def _exponential(
    v: FloatArray, step: FloatArray, jr: FloatArray, k: FloatArray, kk: FloatArray
) -> None:
    squared = (v[0] * v[0] + v[1] * v[1]) + v[2] * v[2]
    angle = math.sqrt(squared)
    if angle < 1e-8:
        a = 1.0 - squared / 6.0
        b = 0.5 - squared / 24.0
    else:
        a = math.sin(angle) / angle
        b = (1.0 - math.cos(angle)) / squared
    if squared < 1e-8:
        ja = 0.5 - squared / 24.0 + squared * squared / 720.0
        jb = 1.0 / 6.0 - squared / 120.0 + squared * squared / 5040.0
    else:
        ja = (1.0 - math.cos(angle)) / squared
        jb = (angle - math.sin(angle)) / (squared * angle)
    k[0, 0] = 0
    k[0, 1] = -v[2]
    k[0, 2] = v[1]
    k[1, 0] = v[2]
    k[1, 1] = 0
    k[1, 2] = -v[0]
    k[2, 0] = -v[1]
    k[2, 1] = v[0]
    k[2, 2] = 0
    _multiply(k, k, kk)
    for i in range(3):
        for j in range(3):
            identity = 1.0 if i == j else 0.0
            step[i, j] = (identity + a * k[i, j]) + b * kk[i, j]
            jr[i, j] = (identity - ja * k[i, j]) + jb * kk[i, j]


@njit(cache=True, nogil=True, fastmath=False)  # type: ignore[untyped-decorator]
def _plane_kernel(
    rotations: FloatArray,
    positions: FloatArray,
    segment: IntArray,
    basis: FloatArray,
    relative: FloatArray,
    delta: FloatArray,
    log_jacobian: FloatArray,
    points: FloatArray,
    anchors: FloatArray,
    normals: FloatArray,
    sigma: float,
    derivatives: bool,
) -> tuple[FloatArray, FloatArray]:
    n = len(segment)
    residual = np.empty(n)
    jacobian = np.empty((n, 24)) if derivatives else np.empty((0, 24))
    rotation = np.empty((3, 3))
    position = np.empty(3)
    steps = np.empty((3, 3, 3))
    jrs = np.empty((3, 3, 3))
    v = np.empty(3)
    k = np.empty((3, 3))
    kk = np.empty((3, 3))
    tangent = np.empty((3, 12))
    suffix = np.empty((3, 3))
    scratch = np.empty((3, 3))
    scratch2 = np.empty((3, 3))
    block = np.empty((3, 3))
    previous = np.empty((3, 3))
    world_derivative = np.empty((3, 3))
    for row in range(n):
        s = segment[row]
        rotation[:, :] = rotations[s]
        position[:] = positions[s]
        for slot in range(1, 4):
            for a in range(3):
                v[a] = basis[row, slot] * delta[s + slot - 1, a]
            _exponential(v, steps[slot - 1], jrs[slot - 1], k, kk)
            _multiply(rotation, steps[slot - 1], scratch)
            rotation[:, :] = scratch
            for a in range(3):
                position[a] += basis[row, slot] * (
                    positions[s + slot, a] - positions[s + slot - 1, a]
                )
        r = 0.0
        for a in range(3):
            world = (rotation[a, 0] * points[row, 0] + rotation[a, 1] * points[row, 1]) + rotation[
                a, 2
            ] * points[row, 2]
            r += normals[row, a] * ((world + position[a]) - anchors[row, a])
        residual[row] = r / sigma
        if not derivatives:
            continue
        tangent[:, :] = 0.0
        for a in range(3):
            for b in range(3):
                suffix[a, b] = 1.0 if a == b else 0.0
        for slot in range(3, 0, -1):
            edge = s + slot - 1
            _multiply(suffix, jrs[slot - 1], scratch)
            _multiply(scratch, log_jacobian[edge], scratch2)
            for a in range(3):
                for b in range(3):
                    block[a, b] = scratch2[a, b] * basis[row, slot]
            _multiply(block, relative[edge].T, previous)
            for a in range(3):
                for b in range(3):
                    tangent[a, 3 * slot + b] += block[a, b]
                    tangent[a, 3 * (slot - 1) + b] -= previous[a, b]
            _multiply(suffix, steps[slot - 1].T, scratch)
            suffix[:, :] = scratch
        tangent[:, :3] += suffix
        v[:] = points[row]
        k[0, 0] = 0
        k[0, 1] = -v[2]
        k[0, 2] = v[1]
        k[1, 0] = v[2]
        k[1, 1] = 0
        k[1, 2] = -v[0]
        k[2, 0] = -v[1]
        k[2, 1] = v[0]
        k[2, 2] = 0
        _multiply(rotation, k, scratch)
        world_derivative[:, :] = -scratch
        for slot in range(4):
            weight = basis[row, slot] - (basis[row, slot + 1] if slot < 3 else 0.0)
            for axis in range(3):
                jacobian[row, 6 * slot + axis] = (normals[row, axis] / sigma) * weight
                derivative = 0.0
                for a in range(3):
                    value = (
                        world_derivative[a, 0] * tangent[0, 3 * slot + axis]
                        + world_derivative[a, 1] * tangent[1, 3 * slot + axis]
                    ) + world_derivative[a, 2] * tangent[2, 3 * slot + axis]
                    derivative += normals[row, a] * value
                jacobian[row, 6 * slot + 3 + axis] = derivative / sigma
    return residual, jacobian


class _NumbaPlaneEvaluator:
    """Own one fit's timestamp schedule and optional reusable worker pool."""

    def __init__(
        self,
        trajectory: CubicBSplineTrajectory,
        planes: BSplinePlaneReturns,
        workers: int,
    ) -> None:
        self.segment, fraction = trajectory.segments(planes.times_sec)
        self._basis = np.ascontiguousarray(
            cumulative_basis(fraction, trajectory.knot_spacing_sec)[0]
        )
        self._edges = np.unique(self.segment[:, None] + np.arange(3)[None, :])
        self._planes = planes
        self._domain = (
            trajectory.start_time_sec,
            trajectory.knot_spacing_sec,
            trajectory.control_point_count,
        )
        self._workers = min(workers, max(1, (len(self.segment) + 4095) // 4096))
        bounds = np.linspace(0, len(self.segment), self._workers + 1, dtype=np.int64)
        self._bounds = list(zip(bounds[:-1].tolist(), bounds[1:].tolist(), strict=True))
        self._pool = ThreadPoolExecutor(max_workers=self._workers) if self._workers > 1 else None
        self.identity = (
            f"calibrex.numba_bspline_planes/v0.1;numba={version('numba')};"
            f"workers={self._workers};fastmath=false;cache=true;parallel_pass=false;"
            "imu=numpy;normal=scipy;covariance=numpy"
        )

    def close(self) -> None:
        """Release only this evaluator's worker pool, including on failed fits."""
        if self._pool is not None:
            self._pool.shutdown(wait=True)

    def evaluate(
        self,
        trajectory: CubicBSplineTrajectory,
        *,
        derivatives: bool,
    ) -> tuple[FloatArray, FloatArray]:
        domain = (
            trajectory.start_time_sec,
            trajectory.knot_spacing_sec,
            trajectory.control_point_count,
        )
        if domain != self._domain:
            raise ValueError("compiled plane schedule requires the initial spline time domain")
        relative = np.zeros((trajectory.control_point_count - 1, 3, 3))
        # Only referenced edges are evaluated, preserving the native near-pi guard.
        relative[self._edges] = np.einsum(
            "tji,tjk->tik",
            trajectory.rotations[self._edges],
            trajectory.rotations[self._edges + 1],
        )
        delta = np.zeros((len(relative), 3))
        delta[self._edges] = so3_log_batch(relative[self._edges])
        log_jacobian = np.zeros_like(relative)
        if derivatives:
            log_jacobian[self._edges] = _right_jacobian(delta[self._edges], inverse=True)

        def chunk(bounds: tuple[int, int]) -> tuple[FloatArray, FloatArray]:
            lo, hi = bounds
            return cast(
                tuple[FloatArray, FloatArray],
                _plane_kernel(
                    trajectory.rotations,
                    trajectory.positions,
                    self.segment[lo:hi],
                    self._basis[lo:hi],
                    relative,
                    delta,
                    log_jacobian,
                    self._planes.points_body_m[lo:hi],
                    self._planes.plane_points_world_m[lo:hi],
                    self._planes.plane_normals_world[lo:hi],
                    self._planes.sigma_m,
                    derivatives,
                ),
            )

        if self._pool is None:
            return chunk(self._bounds[0])
        # Compile in the caller before dispatching the first concurrent call.
        chunk((0, 0))
        parts = list(self._pool.map(chunk, self._bounds))
        return np.concatenate([part[0] for part in parts]), np.concatenate(
            [part[1] for part in parts]
        )
