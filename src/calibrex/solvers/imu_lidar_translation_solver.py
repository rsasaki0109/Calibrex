"""IMU-LiDAR translation (IMU lever arm in the LiDAR frame) from the accelerometer.

With the rotation ``R`` of ``T_lidar_imu`` and the clock offset
(``t_imu = t_lidar + dt``) already calibrated, the IMU origin follows the
LiDAR odometry as ``p_imu = p_lidar + R_odom_lidar t`` where ``t`` is the
translation of ``T_lidar_imu``.  Over a short segment starting at scan ``0``,
double integration of the accelerometer gives, for every later scan ``k``::

    (R_k - R_0) t - v_0 tau_k - g tau_k^2 / 2 + M B_k b_a = M P_k - (p_k - p_0)

where ``M`` is the IMU orientation at the segment start, ``P_k`` and ``B_k``
are the double integrals of the gyro-rotated specific force and of the
rotation itself (for the accelerometer bias ``b_a``), ``v_0`` is the segment
start velocity, and ``g`` is gravity in the odometry frame.  The system is
linear.  ``t`` is shared by every window; the accelerometer bias is a
nuisance of one odometry window, and ``v_0`` and ``g`` of one segment.  The
nuisances are projected out window by window, so only the 3x3 normal
equations of ``t`` are accumulated.

Gravity is fitted per segment by default because the LiDAR odometry's tilt
drifts within a window: with one gravity vector per 10-second window, the
lever arm moved by more than 10 mm with the segment duration on every
MID360 development recording, and per-segment gravity removed most of that.

``M`` is the chordal mean of the IMU orientation at the segment start implied
by every scan of the segment (LiDAR orientation, extrinsic rotation, and gyro
increments), because the double-integrated gravity is metres long and a
single scan's orientation noise would dominate the lever arm.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from calibrex.solvers.gnss_lever_arm_solver import OdometryWindow

FloatArray: TypeAlias = NDArray[np.float64]
TRANSLATION_AXES: tuple[str, str, str] = ("x", "y", "z")


@dataclass(frozen=True)
class TranslationOptions:
    """Segmenting, robust loss, and observability policy."""

    segment_duration_s: float = 2.0
    min_segment_scans: int = 10
    gravity_per_segment: bool = True
    huber_k: float = 1.5
    irls_iterations: int = 5
    observable_translation_std_m: float = 0.01


class ImuPreintegrator:
    """Cumulative integrals of an IMU stream for vectorized double integration.

    ``Q`` is the orientation of the gyro (bias removed) relative to the first
    sample.  ``V``/``P`` are the single and double integrals of ``Q f`` and
    ``VQ``/``PQ`` those of ``Q`` itself, all by the trapezoid rule.
    """

    def __init__(
        self,
        times_s: FloatArray,
        gyro_rps: FloatArray,
        accel_mps2: FloatArray,
        gyro_bias_rps: FloatArray,
    ) -> None:
        if len(times_s) < 2 or np.any(np.diff(times_s) <= 0.0):
            raise ValueError("IMU samples need strictly increasing timestamps")
        self.times_s = np.asarray(times_s, dtype=np.float64)
        steps = np.diff(self.times_s)
        rates = np.asarray(gyro_rps, dtype=np.float64) - np.asarray(gyro_bias_rps)
        step_rotations = Rotation.from_rotvec(
            0.5 * (rates[1:] + rates[:-1]) * steps[:, None]
        ).as_matrix()
        orientations = np.empty((len(self.times_s), 3, 3))
        orientations[0] = np.eye(3)
        for index, step_rotation in enumerate(step_rotations):
            orientations[index + 1] = orientations[index] @ step_rotation
        force = np.einsum("kij,kj->ki", orientations, np.asarray(accel_mps2, dtype=np.float64))
        self._orientations = orientations
        self._velocity = _cumulative(force, steps)
        self._position = _cumulative(self._velocity, steps)
        self._velocity_bias = _cumulative(orientations, steps)
        self._position_bias = _cumulative(self._velocity_bias, steps)

    def covers(self, start_s: float, end_s: float) -> bool:
        """Return whether ``[start_s, end_s]`` lies inside the stream."""

        return bool(self.times_s[0] <= start_s and end_s <= self.times_s[-1])

    def at(
        self, times_s: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        """Return ``Q``, ``V``, ``P``, ``VQ``, ``PQ`` interpolated at ``times_s``."""

        orientation = _interpolate(self.times_s, self._orientations, times_s)
        left, _, right = np.linalg.svd(orientation)
        return (
            left @ right,
            _interpolate(self.times_s, self._velocity, times_s),
            _interpolate(self.times_s, self._position, times_s),
            _interpolate(self.times_s, self._velocity_bias, times_s),
            _interpolate(self.times_s, self._position_bias, times_s),
        )


@dataclass(frozen=True)
class WindowSystem:
    """Linear rows of one odometry window: ``A t + L n = b`` for nuisances ``n``."""

    window_index: int
    design: FloatArray  # (rows, 3) columns of t
    nuisance: FloatArray  # (rows, columns): window gravity (unused when per segment), accel
    # bias, then per segment its velocity and (when per segment) its gravity
    target: FloatArray  # (rows,)
    segments: int

    @property
    def scan_rows(self) -> int:
        """Number of scans that contribute three rows each."""

        return len(self.target) // 3


@dataclass(frozen=True)
class TranslationResult:
    """Weighted least-squares lever arm with its analytic covariance."""

    status: Literal["converged", "insufficient_segments"]
    translation_m: FloatArray | None
    covariance_m2: FloatArray | None
    sigma_m: float | None
    windows: int
    scan_rows: int


def window_systems(
    windows: Sequence[OdometryWindow],
    imu: ImuPreintegrator,
    rotation: FloatArray,
    time_offset_s: float,
    options: TranslationOptions | None = None,
) -> list[WindowSystem]:
    """Build the linear rows of every odometry window with IMU coverage."""

    opts = options or TranslationOptions()
    systems = []
    for index, window in enumerate(windows):
        system = _window_system(index, window, imu, rotation, time_offset_s, opts)
        if system is not None:
            systems.append(system)
    return systems


def solve_translation(
    systems: Sequence[WindowSystem], options: TranslationOptions | None = None
) -> TranslationResult:
    """Huber IRLS over the lever arm with window nuisances projected out."""

    opts = options or TranslationOptions()
    rows = sum(system.scan_rows for system in systems)
    if len(systems) < 3 or rows < 30:
        return TranslationResult("insufficient_segments", None, None, None, len(systems), rows)
    weights = [np.ones(len(system.target)) for system in systems]
    translation = np.zeros(3)
    normal = np.eye(3)
    sigma = 1.0
    for _ in range(max(opts.irls_iterations, 1)):
        normal = np.zeros((3, 3))
        rhs = np.zeros(3)
        for system, weight in zip(systems, weights, strict=True):
            design, target = _projected(system, weight)
            normal += design.T @ design
            rhs += design.T @ target
        translation = np.linalg.solve(normal, rhs)
        residuals = [window_residuals(system, translation) for system in systems]
        stacked = np.concatenate(residuals)
        sigma = max(float(1.4826 * np.median(np.abs(stacked))), 1e-9)
        weights = [
            np.minimum(1.0, opts.huber_k * sigma / np.maximum(np.abs(residual), 1e-12))
            for residual in residuals
        ]
    covariance = np.linalg.inv(normal) * sigma**2
    return TranslationResult("converged", translation, covariance, sigma, len(systems), rows)


def window_residuals(
    system: WindowSystem, translation: FloatArray, weight: FloatArray | None = None
) -> FloatArray:
    """Residuals of one window after refitting its nuisances for ``translation``."""

    target = system.target - system.design @ translation
    root = np.ones(len(target)) if weight is None else np.sqrt(weight)
    nuisance, *_ = np.linalg.lstsq(system.nuisance * root[:, None], target * root, rcond=None)
    return np.asarray(target - system.nuisance @ nuisance, dtype=np.float64)


def gravity_norm(system: WindowSystem, translation: FloatArray) -> float:
    """Return the norm of the window's fitted gravity vector."""

    target = system.target - system.design @ translation
    nuisance, *_ = np.linalg.lstsq(system.nuisance, target, rcond=None)
    if system.nuisance[:, :3].any():
        return float(np.linalg.norm(nuisance[:3]))
    # Per-segment gravity: the median norm over the segments.
    gravities = nuisance[6:].reshape(-1, 6)[:, 3:]
    return float(np.median(np.linalg.norm(gravities, axis=1)))


def _window_system(
    index: int,
    window: OdometryWindow,
    imu: ImuPreintegrator,
    rotation: FloatArray,
    time_offset_s: float,
    opts: TranslationOptions,
) -> WindowSystem | None:
    times = np.asarray(window.times_s, dtype=np.float64)
    imu_times = times + time_offset_s
    inside = (imu_times >= imu.times_s[0]) & (imu_times <= imu.times_s[-1])
    segment = np.floor((times - times[0]) / opts.segment_duration_s).astype(np.int64)
    segment[~inside] = -1
    labels = [
        label
        for label in np.unique(segment[segment >= 0])
        if np.count_nonzero(segment == label) >= opts.min_segment_scans
    ]
    if not labels:
        return None
    orientation, velocity, position, velocity_bias, position_bias = imu.at(
        np.clip(imu_times, imu.times_s[0], imu.times_s[-1])
    )
    poses = window.poses
    per_segment = 6 if opts.gravity_per_segment else 3
    columns = 6 + per_segment * len(labels)
    designs, nuisances, targets = [], [], []
    for slot, label in enumerate(labels):
        members = np.flatnonzero(segment == label)
        first = members[0]
        # Chordal mean of the IMU orientation at the segment start implied by
        # every scan: R_odom_lidar(k) R Q(k)^T Q(first).
        implied = poses[members, :3, :3] @ rotation @ np.transpose(
            orientation[members], (0, 2, 1)
        ) @ orientation[first]
        left, _, right = np.linalg.svd(implied.sum(axis=0))
        start = left @ right @ orientation[first].T
        later = members[1:]
        elapsed = (times[later] - times[first])[:, None, None]
        imu_elapsed = (imu_times[later] - imu_times[first])[:, None]
        force = position[later] - position[first] - velocity[first] * imu_elapsed
        bias = (
            position_bias[later]
            - position_bias[first]
            - velocity_bias[first] * imu_elapsed[:, :, None]
        )
        count = len(later)
        design = poses[later, :3, :3] - poses[first, :3, :3]
        nuisance = np.zeros((count, 3, columns))
        base = 6 + per_segment * slot
        gravity = slice(base + 3, base + 6) if opts.gravity_per_segment else slice(0, 3)
        nuisance[:, :, gravity] = -0.5 * elapsed**2 * np.eye(3)
        nuisance[:, :, 3:6] = start @ bias
        nuisance[:, :, base : base + 3] = -elapsed * np.eye(3)
        target = force @ start.T - (poses[later, :3, 3] - poses[first, :3, 3])
        designs.append(design.reshape(-1, 3))
        nuisances.append(nuisance.reshape(-1, columns))
        targets.append(target.reshape(-1))
    return WindowSystem(
        window_index=index,
        design=np.vstack(designs),
        nuisance=np.vstack(nuisances),
        target=np.concatenate(targets),
        segments=len(labels),
    )


def _projected(system: WindowSystem, weight: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Weighted rows with the window nuisances projected out (rank-safe)."""

    root = np.sqrt(weight)[:, None]
    nuisance = system.nuisance * root
    basis, singular, _ = np.linalg.svd(nuisance, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * 1e-10)) if singular.size else 0
    basis = basis[:, :rank]
    design = system.design * root
    target = system.target * root[:, 0]
    return (
        design - basis @ (basis.T @ design),
        target - basis @ (basis.T @ target),
    )


def _cumulative(values: FloatArray, steps: FloatArray) -> FloatArray:
    shape = (1, *values.shape[1:])
    widths = steps.reshape((-1,) + (1,) * (values.ndim - 1))
    increments = 0.5 * (values[1:] + values[:-1]) * widths
    return np.concatenate([np.zeros(shape), np.cumsum(increments, axis=0)])


def _interpolate(times: FloatArray, values: FloatArray, at: FloatArray) -> FloatArray:
    flat = values.reshape(len(times), -1)
    result = np.column_stack(
        [np.interp(at, times, flat[:, column]) for column in range(flat.shape[1])]
    )
    return result.reshape((len(at), *values.shape[1:]))


def translation_std_m(result: TranslationResult) -> FloatArray:
    """Analytic std of the lever arm per axis (inf when unsolved)."""

    if result.covariance_m2 is None:
        return np.full(3, math.inf)
    return np.sqrt(np.clip(np.diag(result.covariance_m2), 0.0, None))
