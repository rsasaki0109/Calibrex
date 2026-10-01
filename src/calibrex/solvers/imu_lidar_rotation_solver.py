"""IMU-LiDAR rotation, gyro bias, and clock offset from angular rates.

Between consecutive LiDAR odometry poses ``i`` and ``i + 1`` the mean angular
rate in the LiDAR frame is ``log(R_i^T R_{i+1}) / (t_{i+1} - t_i)``.  The IMU
measures the same rotation through its gyro, so::

    omega_lidar = R_lidar_imu (mean_gyro[t_i + dt, t_{i+1} + dt] - b)

for the rotation ``R_lidar_imu`` of ``T_lidar_imu``, a constant gyro bias
``b``, and the clock offset ``dt`` (``t_imu = t_lidar + dt``).  Interval means
of the gyro come from a cumulative trapezoid integral, so every evaluation is
vectorized.  The initial rotation is the weighted Procrustes alignment of the
two rate sets, which handles arbitrary mounting orientations.

Rotation about an axis is observable only when the motion rotates about other
axes as well: a vehicle that only yaws leaves the rotation about the vertical
unobservable.  Each DoF is classified from its data-only standard deviation.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from calibrex.solvers.gnss_lever_arm_solver import OdometryWindow, weighted_procrustes

FloatArray: TypeAlias = NDArray[np.float64]
RotationDofName = Literal[
    "roll", "pitch", "yaw", "time_offset", "gyro_bias_x", "gyro_bias_y", "gyro_bias_z"
]
ROTATION_DOFS: tuple[RotationDofName, ...] = (
    "roll",
    "pitch",
    "yaw",
    "time_offset",
    "gyro_bias_x",
    "gyro_bias_y",
    "gyro_bias_z",
)


@dataclass(frozen=True)
class GyroSeries:
    """Gyro samples with a cumulative integral for fast interval means."""

    times_s: FloatArray
    gyro_rps: FloatArray

    def __post_init__(self) -> None:
        if len(self.times_s) < 2 or np.any(np.diff(self.times_s) <= 0.0):
            raise ValueError("gyro samples need strictly increasing timestamps")
        steps = np.diff(self.times_s)
        midpoint_rates = 0.5 * (self.gyro_rps[1:] + self.gyro_rps[:-1])
        increments = midpoint_rates * steps[:, None]
        object.__setattr__(
            self, "_integral", np.vstack([np.zeros((1, 3)), np.cumsum(increments, axis=0)])
        )
        # Cumulative orientation Q_k of the raw gyro (piecewise-constant midpoint
        # rates) and S_k = sum_j Q_{j+1} dt_j for first-order bias Jacobians.
        step_rotations = Rotation.from_rotvec(increments).as_matrix()
        orientations = np.empty((len(self.times_s), 3, 3))
        orientations[0] = np.eye(3)
        for index, step_rotation in enumerate(step_rotations):
            orientations[index + 1] = orientations[index] @ step_rotation
        weighted = orientations[1:] * steps[:, None, None]
        object.__setattr__(self, "_orientations", orientations)
        object.__setattr__(self, "_rates", midpoint_rates)
        object.__setattr__(
            self,
            "_bias_sum",
            np.concatenate([np.zeros((1, 3, 3)), np.cumsum(weighted, axis=0)]),
        )

    def orientation_and_bias_sum(self, times_s: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return ``Q(t)`` and ``S(t)`` at arbitrary times inside the series."""

        orientations: FloatArray = self._orientations  # type: ignore[attr-defined]
        rates: FloatArray = self._rates  # type: ignore[attr-defined]
        bias_sum: FloatArray = self._bias_sum  # type: ignore[attr-defined]
        index = np.clip(np.searchsorted(self.times_s, times_s, side="right") - 1, 0, len(rates) - 1)
        elapsed = times_s - self.times_s[index]
        partial = Rotation.from_rotvec(rates[index] * elapsed[:, None]).as_matrix()
        fraction = (elapsed / (self.times_s[index + 1] - self.times_s[index]))[:, None, None]
        return (
            orientations[index] @ partial,
            bias_sum[index] + fraction * (bias_sum[index + 1] - bias_sum[index]),
        )

    def increments(self, start_s: FloatArray, end_s: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return bias-free rotation increments and their bias Jacobians.

        ``delta(b) = delta(0) Exp(J b)`` to first order in the bias ``b``.
        """

        start_orientation, start_sum = self.orientation_and_bias_sum(start_s)
        end_orientation, end_sum = self.orientation_and_bias_sum(end_s)
        end_transpose = np.transpose(end_orientation, (0, 2, 1))
        delta = np.transpose(start_orientation, (0, 2, 1)) @ end_orientation
        return delta, -end_transpose @ (end_sum - start_sum)

    def increments_from(
        self, start_s: FloatArray, end_s: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Like :meth:`increments` for one shared start time (a length-1 ``start_s``)."""

        start_orientation, start_sum = self.orientation_and_bias_sum(start_s)
        end_orientation, end_sum = self.orientation_and_bias_sum(end_s)
        end_transpose = np.transpose(end_orientation, (0, 2, 1))
        delta = np.transpose(start_orientation, (0, 2, 1)) @ end_orientation
        return delta, -end_transpose @ (end_sum - start_sum)

    def integral_at(self, times_s: FloatArray) -> FloatArray:
        """Return the trapezoid integral of the gyro from the first sample."""

        integral: FloatArray = self._integral  # type: ignore[attr-defined]
        return np.column_stack(
            [np.interp(times_s, self.times_s, integral[:, axis]) for axis in range(3)]
        )

    def covers(self, start_s: float, end_s: float, max_gap_s: float = 0.05) -> bool:
        """Return whether ``[start_s, end_s]`` has gyro data without gaps."""

        if start_s < self.times_s[0] or end_s > self.times_s[-1]:
            return False
        first = max(int(np.searchsorted(self.times_s, start_s, side="right")) - 1, 0)
        last = int(np.searchsorted(self.times_s, end_s, side="left"))
        window = self.times_s[first : last + 1]
        return bool(window.size < 2 or np.max(np.diff(window)) <= max_gap_s)


@dataclass(frozen=True)
class RotationOptions:
    """Noise model, bounds, and observability policy."""

    model: Literal["preintegrated", "mean_rate"] = "preintegrated"
    pair_steps: tuple[int, ...] = (1,)
    estimate_time_offset: bool = True
    time_offset_bound_s: float = 0.1
    rate_sigma_floor_rps: float = 0.01
    rate_sigma_fraction: float = 0.02
    huber_threshold: float = 1.5
    irls_iterations: int = 4
    observable_rotation_std_deg: float = 0.1
    observable_time_offset_std_s: float = 0.002
    observable_gyro_bias_std_rps: float = 0.002
    max_step_s: float = 0.25
    anchor_sigma: float = 1.0e3


@dataclass(frozen=True)
class RotationDof:
    """One estimated quantity with its data-only standard deviation."""

    name: RotationDofName
    value: float
    std: float
    status: Literal["estimated", "unobservable"]


@dataclass(frozen=True)
class RateIntervals:
    """LiDAR rate intervals with their mean LiDAR angular velocity."""

    window_ids: NDArray[np.int64]
    start_s: FloatArray
    end_s: FloatArray
    lidar_rate_rps: FloatArray
    lidar_delta: FloatArray

    def subset(self, keep: NDArray[np.bool_]) -> RateIntervals:
        """Return the intervals selected by ``keep``."""

        return RateIntervals(
            self.window_ids[keep],
            self.start_s[keep],
            self.end_s[keep],
            self.lidar_rate_rps[keep],
            self.lidar_delta[keep],
        )

    def shifted(self, offset_s: float) -> RateIntervals:
        """Return the intervals with ``offset_s`` subtracted from every time."""

        return RateIntervals(
            self.window_ids,
            self.start_s - offset_s,
            self.end_s - offset_s,
            self.lidar_rate_rps,
            self.lidar_delta,
        )

    def __len__(self) -> int:
        return len(self.start_s)


@dataclass(frozen=True)
class RotationResult:
    """``R_lidar_imu``, gyro bias, clock offset, and per-DoF observability."""

    status: Literal["converged", "insufficient_intervals"]
    rotation: FloatArray | None
    gyro_bias_rps: FloatArray
    time_offset_s: float
    dofs: tuple[RotationDof, ...]
    variance_factor: float | None
    interval_count: int

    def dof(self, name: RotationDofName) -> RotationDof:
        """Return one DoF."""

        return next(item for item in self.dofs if item.name == name)


def rate_intervals(
    windows: Sequence[OdometryWindow],
    gyro: GyroSeries,
    opts: RotationOptions,
) -> RateIntervals:
    """LiDAR rotation increments over ``pair_steps`` scans with gap-free gyro coverage."""

    margin = opts.time_offset_bound_s if opts.estimate_time_offset else 0.0
    ids: list[int] = []
    starts: list[float] = []
    ends: list[float] = []
    deltas: list[FloatArray] = []
    for index, window in enumerate(windows):
        for step in opts.pair_steps:
            for first in range(len(window.times_s) - step):
                last = first + step
                start, end = float(window.times_s[first]), float(window.times_s[last])
                if not 0.0 < end - start <= opts.max_step_s * step:
                    continue
                if not gyro.covers(start - margin, end + margin):
                    continue
                ids.append(index)
                starts.append(start)
                ends.append(end)
                deltas.append(window.poses[first, :3, :3].T @ window.poses[last, :3, :3])
    delta_array = np.array(deltas, dtype=np.float64).reshape(-1, 3, 3)
    start_array = np.array(starts, dtype=np.float64)
    end_array = np.array(ends, dtype=np.float64)
    rates = (
        Rotation.from_matrix(delta_array).as_rotvec() / (end_array - start_array)[:, None]
        if len(delta_array)
        else np.empty((0, 3))
    )
    return RateIntervals(np.array(ids, dtype=np.int64), start_array, end_array, rates, delta_array)


def rate_residuals(
    gyro: GyroSeries,
    intervals: RateIntervals,
    rotation: FloatArray,
    bias: FloatArray,
    time_offset_s: float,
    opts: RotationOptions,
    *,
    normalize: bool = True,
) -> FloatArray:
    """Return ``omega_lidar - R (mean_gyro - b)`` per interval (normalized by default)."""

    duration = (intervals.end_s - intervals.start_s)[:, None]
    mean_gyro = (
        gyro.integral_at(intervals.end_s + time_offset_s)
        - gyro.integral_at(intervals.start_s + time_offset_s)
    ) / duration
    predicted = (mean_gyro - bias) @ rotation.T
    error = intervals.lidar_rate_rps - predicted
    if not normalize:
        return error.reshape(-1)
    sigma = opts.rate_sigma_floor_rps + opts.rate_sigma_fraction * np.linalg.norm(
        intervals.lidar_rate_rps, axis=1
    )
    return (error / sigma[:, None]).reshape(-1)


def increment_residuals(
    gyro: GyroSeries,
    intervals: RateIntervals,
    rotation: FloatArray,
    bias: FloatArray,
    time_offset_s: float,
    opts: RotationOptions,
    *,
    normalize: bool = True,
) -> FloatArray:
    """Return ``Log((R dR_imu(b) R^T)^T dR_lidar)`` per interval.

    Normalized by the noise model by default; otherwise divided by the
    interval duration so the values are rad/s, comparable with mean rates.
    """

    delta, jacobian = gyro.increments(
        intervals.start_s + time_offset_s, intervals.end_s + time_offset_s
    )
    corrected = delta @ Rotation.from_rotvec(np.einsum("nij,j->ni", jacobian, bias)).as_matrix()
    predicted = rotation @ corrected @ rotation.T
    error = Rotation.from_matrix(
        np.transpose(predicted, (0, 2, 1)) @ intervals.lidar_delta
    ).as_rotvec()
    duration = (intervals.end_s - intervals.start_s)[:, None]
    if not normalize:
        return (error / duration).reshape(-1)
    angle = np.linalg.norm(intervals.lidar_rate_rps, axis=1) * duration[:, 0]
    sigma = (opts.rate_sigma_floor_rps * duration[:, 0]) + opts.rate_sigma_fraction * angle
    return (error / sigma[:, None]).reshape(-1)


def model_residuals(
    gyro: GyroSeries,
    intervals: RateIntervals,
    rotation: FloatArray,
    bias: FloatArray,
    time_offset_s: float,
    opts: RotationOptions,
    *,
    normalize: bool = True,
) -> FloatArray:
    """Dispatch to the configured measurement model."""

    function = increment_residuals if opts.model == "preintegrated" else rate_residuals
    return function(gyro, intervals, rotation, bias, time_offset_s, opts, normalize=normalize)


class ImuLidarRotationSolver:
    """Robust ``R_lidar_imu``, gyro bias, and clock-offset estimation."""

    def solve(
        self,
        gyro: GyroSeries,
        intervals: RateIntervals,
        options: RotationOptions | None = None,
    ) -> RotationResult:
        opts = options or RotationOptions()
        if len(intervals) < 20:
            return RotationResult("insufficient_intervals", None, np.zeros(3), 0.0, (), None, 0)
        epoch = float(gyro.times_s[0])
        gyro = GyroSeries(gyro.times_s - epoch, gyro.gyro_rps)
        intervals = intervals.shifted(epoch)
        base = _initial_rotation(gyro, intervals)
        params, weights, variance = _irls(gyro, intervals, base, opts)
        dofs = _observability(gyro, intervals, base, params, weights, variance, opts)
        return RotationResult(
            status="converged",
            rotation=_rotation(base, params),
            gyro_bias_rps=params[4:7].copy(),
            time_offset_s=float(params[3]) if opts.estimate_time_offset else 0.0,
            dofs=dofs,
            variance_factor=variance,
            interval_count=len(intervals),
        )


def _initial_rotation(gyro: GyroSeries, intervals: RateIntervals) -> FloatArray:
    duration = (intervals.end_s - intervals.start_s)[:, None]
    mean_gyro = (gyro.integral_at(intervals.end_s) - gyro.integral_at(intervals.start_s)) / duration
    return weighted_procrustes(mean_gyro, intervals.lidar_rate_rps, np.ones(len(intervals)))


def _rotation(base: FloatArray, params: FloatArray) -> FloatArray:
    return np.asarray(Rotation.from_rotvec(params[:3]).as_matrix() @ base, dtype=np.float64)


def _residuals(
    gyro: GyroSeries,
    intervals: RateIntervals,
    base: FloatArray,
    params: FloatArray,
    opts: RotationOptions,
) -> FloatArray:
    offset = float(params[3]) if opts.estimate_time_offset else 0.0
    return model_residuals(gyro, intervals, _rotation(base, params), params[4:7], offset, opts)


def _irls(
    gyro: GyroSeries,
    intervals: RateIntervals,
    base: FloatArray,
    opts: RotationOptions,
) -> tuple[FloatArray, FloatArray, float]:
    params = np.zeros(7)
    bound = opts.time_offset_bound_s if opts.estimate_time_offset else 1e-12
    lower = np.array([-np.inf] * 3 + [-bound] + [-np.inf] * 3)
    upper = np.array([np.inf] * 3 + [bound] + [np.inf] * 3)
    weights = np.ones(3 * len(intervals))
    variance = 1.0
    for _ in range(max(opts.irls_iterations, 1)):
        root = np.sqrt(weights)

        def residuals(
            values: FloatArray, root: FloatArray = root, variance: float = variance
        ) -> FloatArray:
            data = root * _residuals(gyro, intervals, base, values, opts) / math.sqrt(variance)
            return np.concatenate([data, values / opts.anchor_sigma])

        params = least_squares(residuals, params, bounds=(lower, upper), x_scale="jac").x
        raw = _residuals(gyro, intervals, base, params, opts)
        spread = max(1.4826 * float(np.median(np.abs(raw - np.median(raw)))), 1e-6)
        variance = spread**2
        scaled = np.abs(raw) / spread
        weights = np.where(
            scaled <= opts.huber_threshold,
            1.0,
            opts.huber_threshold / np.maximum(scaled, 1e-12),
        )
    return params, weights, variance


def _observability(
    gyro: GyroSeries,
    intervals: RateIntervals,
    base: FloatArray,
    params: FloatArray,
    weights: FloatArray,
    variance: float,
    opts: RotationOptions,
) -> tuple[RotationDof, ...]:
    active = [0, 1, 2, 3, 4, 5, 6] if opts.estimate_time_offset else [0, 1, 2, 4, 5, 6]
    step = 1e-5
    columns = []
    for index in active:
        delta = np.zeros(7)
        delta[index] = step
        forward = _residuals(gyro, intervals, base, params + delta, opts)
        backward = _residuals(gyro, intervals, base, params - delta, opts)
        columns.append((forward - backward) / (2.0 * step))
    jacobian = np.column_stack(columns) * (np.sqrt(weights)[:, None] / math.sqrt(variance))
    information = jacobian.T @ jacobian + np.eye(len(active)) / opts.anchor_sigma**2
    std = np.sqrt(np.diag(np.linalg.inv(information)))
    euler = Rotation.from_matrix(_rotation(base, params)).as_euler("xyz")
    values = [*euler, float(params[3]), *params[4:7]]
    thresholds = [
        *(math.radians(opts.observable_rotation_std_deg),) * 3,
        opts.observable_time_offset_std_s,
        *(opts.observable_gyro_bias_std_rps,) * 3,
    ]
    dofs = []
    for position, index in enumerate(active):
        dofs.append(
            RotationDof(
                name=ROTATION_DOFS[index],
                value=float(values[index]),
                std=float(std[position]),
                status="estimated" if std[position] <= thresholds[index] else "unobservable",
            )
        )
    return tuple(dofs)


def gyro_rotation_model(
    gyro: GyroSeries,
    rotation: FloatArray,
    bias: FloatArray,
    time_offset_s: float,
) -> Callable[[float, FloatArray], FloatArray]:
    """Return in-sweep LiDAR-frame rotations predicted by the gyro and an extrinsic.

    Used to deskew LiDAR sweeps.  Because the extrinsic under test enters the
    odometry this way, callers must iterate and measure how much of a
    deliberate extrinsic error survives (see the evaluation's feedback check).
    """

    def model(scan_time_s: float, offsets_s: FloatArray) -> FloatArray:
        times = np.asarray(offsets_s, dtype=np.float64)
        # Every point's increment starts at the sweep time, so that orientation is
        # computed once and broadcast instead of once per point.
        start = np.full(1, scan_time_s + time_offset_s)
        delta, jacobian = gyro.increments_from(start, start + times)
        corrected = delta @ Rotation.from_rotvec(np.einsum("nij,j->ni", jacobian, bias)).as_matrix()
        return np.asarray(rotation @ corrected @ rotation.T, dtype=np.float64)

    return model
