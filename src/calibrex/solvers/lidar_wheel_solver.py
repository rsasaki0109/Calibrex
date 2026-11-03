"""LiDAR-to-vehicle rotation, wheel-speed scale, and clock offset against wheel odometry.

Wheel odometry reports the vehicle's forward speed ``v_w(t)`` and yaw rate
``w_w(t)`` in the vehicle frame (x forward, z up).  Seen by the LiDAR over a
short interval, the same motion must satisfy, in the vehicle frame:

* ``v_x = s v_w(t + dt)``: forward speed, with a wheel-speed scale ``s``
  (tyre radius) and a clock offset ``dt`` (``t_wheel = t_lidar + dt``);
* ``v_y = w_z l``: lateral speed from the lever ``l`` ahead of the axle;
* ``v_z = 0`` and zero-mean roll and pitch rates; and
* ``w_z = w_w(t + dt)``: yaw rate.

Unlike the motion-only vehicle frame (``vehicle_frame_solver``), the forward
speed and yaw rate tie the LiDAR to the wheel sensors in scale and time.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from calibrex.solvers.vehicle_frame_solver import VehicleMotion, initial_vehicle_rotation

FloatArray: TypeAlias = NDArray[np.float64]
WHEEL_PARAMETERS: tuple[str, ...] = ("roll", "pitch", "yaw", "lever", "speed_scale", "time_offset")
_ResidualEvaluator: TypeAlias = Callable[[FloatArray], FloatArray]
_PREPARATION_MAX_BYTES = 64 * 1024 * 1024
_PREPARATION_BYTES_PER_MOTION = 1024


@dataclass(frozen=True)
class WheelOdometry:
    """Time-sorted wheel odometry: forward speed (m/s) and yaw rate (rad/s)."""

    times_s: FloatArray
    speed_mps: FloatArray
    yaw_rate_rps: FloatArray

    def at(self, times_s: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Linearly interpolated speed and yaw rate."""

        return (
            np.interp(times_s, self.times_s, self.speed_mps),
            np.interp(times_s, self.times_s, self.yaw_rate_rps),
        )


CHECK_MIN_SPEED_MPS = 0.5
"""The speed gate of ``calibrex check``: ten times ``velocity_sigma_mps``, so the forward
direction of a kept motion is within about 6 deg. A wheeled robot such as NCLT's Segway
drives at 1-1.5 m/s and would keep almost no motion at 2 m/s."""


@dataclass(frozen=True)
class LidarWheelOptions:
    """Noise model and bounds."""

    min_speed_mps: float = 2.0
    """Motions slower than this are left out. 2 m/s suits cars (and is what the frozen KITTI
    benchmarks ran with); ``calibrex check`` uses ``CHECK_MIN_SPEED_MPS`` for robots."""
    velocity_sigma_mps: float = 0.05
    velocity_sigma_fraction: float = 0.01
    rate_sigma_rps: float = 0.01
    time_offset_bound_s: float = 0.3
    huber_threshold: float = 1.5


@dataclass(frozen=True)
class LidarWheelResult:
    """Estimate and analytic covariance over ``WHEEL_PARAMETERS``."""

    status: Literal["converged", "insufficient_motions"]
    rotation: FloatArray | None
    lever_m: float | None
    speed_scale: float | None
    time_offset_s: float | None
    covariance: FloatArray | None
    motions: int


def _moving(motions: Sequence[VehicleMotion], opts: LidarWheelOptions) -> list[VehicleMotion]:
    return [
        item for item in motions if float(np.linalg.norm(item.velocity_mps)) >= opts.min_speed_mps
    ]


def lidar_wheel_residuals(
    motions: Sequence[VehicleMotion],
    wheel: WheelOdometry,
    params: FloatArray,
    base: FloatArray,
    options: LidarWheelOptions | None = None,
) -> FloatArray:
    """Normalized residuals for ``params`` (rotation perturbation, lever, scale, offset)."""

    opts = options or LidarWheelOptions()
    moving = _moving(motions, opts)
    if not moving:
        return np.zeros(0)
    rotation = Rotation.from_rotvec(params[:3]).as_matrix() @ base
    velocity = np.array([item.velocity_mps for item in moving]) @ rotation.T
    rates = np.array([item.angular_rate_rps for item in moving]) @ rotation.T
    middle = np.array([0.5 * (item.start_s + item.end_s) for item in moving])
    speed, yaw_rate = wheel.at(middle + params[5])
    sigma = opts.velocity_sigma_mps + opts.velocity_sigma_fraction * np.abs(speed)
    return _residual_rows(velocity, rates, speed, yaw_rate, sigma, params, opts)


def _residual_rows(
    velocity: FloatArray,
    rates: FloatArray,
    speed: FloatArray,
    yaw_rate: FloatArray,
    sigma: FloatArray,
    params: FloatArray,
    opts: LidarWheelOptions,
) -> FloatArray:
    return np.concatenate(
        [
            (velocity[:, 0] - params[4] * speed) / sigma,
            (velocity[:, 1] - rates[:, 2] * params[3]) / sigma,
            velocity[:, 2] / sigma,
            (rates[:, 2] - yaw_rate) / opts.rate_sigma_rps,
            rates[:, :2].ravel() / opts.rate_sigma_rps,
        ]
    )


def _make_residual_evaluator(
    moving: Sequence[VehicleMotion], wheel: WheelOdometry, base: FloatArray, opts: LidarWheelOptions
) -> _ResidualEvaluator:
    # moving is the solve's already gated, fixed residual set. Bound prepared
    # arrays, eight interpolation states and working memory conservatively.
    if not moving or len(moving) * _PREPARATION_BYTES_PER_MOTION > _PREPARATION_MAX_BYTES:
        return lambda params: lidar_wheel_residuals(moving, wheel, params, base, opts)
    velocity = np.array([item.velocity_mps for item in moving])
    rates = np.array([item.angular_rate_rps for item in moving])
    middle = np.array([0.5 * (item.start_s + item.end_s) for item in moving])

    def interpolate(offset: float) -> tuple[FloatArray, FloatArray, FloatArray]:
        speed, yaw_rate = wheel.at(middle + offset)
        sigma = opts.velocity_sigma_mps + opts.velocity_sigma_fraction * np.abs(speed)
        return speed, yaw_rate, sigma

    @lru_cache(maxsize=8)
    def clock_state(bits: bytes) -> tuple[FloatArray, FloatArray, FloatArray]:
        values = interpolate(struct.unpack("d", bits)[0])
        for value in values:
            value.setflags(write=False)
        return values

    def evaluate(params: FloatArray) -> FloatArray:
        # Preserve every finite-difference step and signed zero. Derived wheel
        # implementations retain their per-request interpolation callback.
        rotation = Rotation.from_rotvec(params[:3]).as_matrix() @ base
        rotated_velocity = velocity @ rotation.T
        rotated_rates = rates @ rotation.T
        values = (
            clock_state(struct.pack("d", float(params[5])))
            if type(wheel) is WheelOdometry
            else interpolate(float(params[5]))
        )
        return _residual_rows(rotated_velocity, rotated_rates, *values, params, opts)

    return evaluate


def solve_lidar_wheel(
    motions: Sequence[VehicleMotion],
    wheel: WheelOdometry,
    options: LidarWheelOptions | None = None,
    *,
    initial: FloatArray | None = None,
) -> LidarWheelResult:
    """Robust least squares over the wheel-odometry residuals."""

    opts = options or LidarWheelOptions()
    moving = _moving(motions, opts)
    if len(moving) < 10:
        return LidarWheelResult("insufficient_motions", None, None, None, None, None, len(moving))
    base = initial if initial is not None else initial_vehicle_rotation(moving)
    if initial is None:
        base = _orient_up(moving, wheel, base)
    bound = opts.time_offset_bound_s
    lower = np.array([-np.inf] * 4 + [0.5, -bound])
    upper = np.array([np.inf] * 4 + [1.5, bound])
    start_offset = _yaw_rate_offset_scan(moving, wheel, base, opts)
    evaluate = _make_residual_evaluator(moving, wheel, base, opts)
    fit = least_squares(
        evaluate,
        np.array([0.0, 0.0, 0.0, 0.0, 1.0, start_offset]),
        bounds=(lower, upper),
        loss="huber",
        f_scale=opts.huber_threshold,
        # Explicit scales: the clock offset moves the cost far less per unit than the
        # rotation, and with Jacobian scaling the solver stopped ~20 ms short in tests.
        x_scale=np.array([1e-3, 1e-3, 1e-3, 0.1, 1e-3, 1e-3]),
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=200,
    )
    raw = evaluate(fit.x)
    spread = max(1.4826 * float(np.median(np.abs(raw))), 1e-6)
    covariance = np.linalg.pinv(fit.jac.T @ fit.jac) * spread**2
    rotation = Rotation.from_rotvec(fit.x[:3]).as_matrix() @ base
    return LidarWheelResult(
        "converged",
        rotation,
        float(fit.x[3]),
        float(fit.x[4]),
        float(fit.x[5]),
        covariance,
        len(moving),
    )


MIN_TURN_RATE_RPS = 0.05
"""Wheel yaw rates at least this large (about 3 deg/s) decide which way the vehicle's up is."""


def _orient_up(
    moving: Sequence[VehicleMotion], wheel: WheelOdometry, base: FloatArray
) -> FloatArray:
    """Turn ``base`` so the LiDAR's yaw rate about its up axis agrees with the wheels' in sign.

    The LiDAR's motion gives the turning axis but not which way along it is up. The wheel
    yaw rate does: a left turn is positive about up. (Without turns, the up axis is taken
    to have a positive component along the sensor's z, which assumes an upright sensor: an
    upside-down LiDAR, as on NCLT's Segway, would come out turned by 180 deg.)
    """

    middle = np.array([0.5 * (item.start_s + item.end_s) for item in moving])
    _speed, yaw_rate = wheel.at(middle)
    lidar_yaw = np.array([item.angular_rate_rps for item in moving]) @ base[2]
    turning = np.abs(yaw_rate) >= MIN_TURN_RATE_RPS
    flip = np.diag([1.0, -1.0, -1.0])
    if np.count_nonzero(turning) >= 10:
        agreement = float(np.sum(np.sign(lidar_yaw[turning]) * np.sign(yaw_rate[turning])))
        return flip @ base if agreement < 0.0 else base
    return flip @ base if base[2, 2] < 0.0 else base


def _yaw_rate_offset_scan(
    moving: Sequence[VehicleMotion],
    wheel: WheelOdometry,
    base: FloatArray,
    opts: LidarWheelOptions,
    step_s: float = 0.002,
) -> float:
    """Coarse clock-offset start from the yaw rate alone.

    Wheel odometry is interpolated linearly between samples, so the cost has
    kinks at every sample spacing and a gradient method started at zero can
    stop tens of milliseconds short.  The yaw-rate residual does not depend on
    the speed scale or the lever, so it is scanned over the whole bound first.
    """

    rates = np.array([item.angular_rate_rps for item in moving]) @ base.T
    middle = np.array([0.5 * (item.start_s + item.end_s) for item in moving])
    candidates = np.arange(-opts.time_offset_bound_s, opts.time_offset_bound_s + step_s, step_s)
    costs = [
        float(np.sum((rates[:, 2] - wheel.at(middle + offset)[1]) ** 2)) for offset in candidates
    ]
    return float(candidates[int(np.argmin(costs))])


def wheel_std(result: LidarWheelResult) -> FloatArray:
    """Analytic std over ``WHEEL_PARAMETERS`` (rotations in rad); inf when unsolved."""

    if result.covariance is None:
        return np.full(6, math.inf)
    return np.sqrt(np.clip(np.diag(result.covariance), 0.0, None))


def parse_wheel_csv(text: str) -> WheelOdometry:
    """Parse ``t, speed, yaw_rate`` rows (commas or whitespace; a header line is allowed)."""

    rows = []
    for raw in text.splitlines():
        fields = raw.split("#", 1)[0].replace(",", " ").split()
        if not fields:
            continue
        try:
            rows.append([float(value) for value in fields[:3]])
        except ValueError:
            if rows:
                raise
    array = np.asarray(rows, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) < 2:
        raise ValueError("wheel odometry needs at least two rows of t, speed, yaw_rate")
    array = array[np.argsort(array[:, 0], kind="stable")]
    return WheelOdometry(array[:, 0], array[:, 1], array[:, 2])
