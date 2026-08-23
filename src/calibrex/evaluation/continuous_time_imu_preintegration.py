"""Synthetic recovery for continuous-time IMU gyro pre-integration factors."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from calibrex.core.continuous_time_imu_preintegration import (
    ContinuousTimeImuPreintegrationArtifact,
    ContinuousTimeImuPreintegrationProvenance,
)
from calibrex.core.continuous_time_sparse import (
    ContinuousTimeTrajectoryFitOptions,
    ContinuousTimeTrajectoryFitProblem,
    TrajectoryImuGyroSample,
    TrajectoryImuPreintegrationMeasurement,
    TrajectoryPoseMeasurement,
    fit_continuous_trajectory,
    imu_preintegration_rmse,
    interpolate_pose_at,
)
from calibrex.core.geometry import SE3, _tuple3
from calibrex.core.provenance import git_commit
from calibrex.core.se3_manifold import se3_exp, se3_log

_SYNTHETIC_SEED = 20260820
_TRUE_GYRO_BIAS_RAD_S = (0.012, -0.018, 0.009)
_KNOWN_BAD_GYRO_BIAS_RAD_S = (0.0, 0.0, 0.08)
_GYRO_RATE_HZ = 50.0


def run_synthetic_imu_preintegration_recovery(
    *,
    seed: int = _SYNTHETIC_SEED,
    command: list[str] | None = None,
    recovery_id: str = "ct-imu-preintegration-synthetic",
) -> ContinuousTimeImuPreintegrationArtifact:
    """Recover knot rotations and gyro bias with holdout and a signed control."""

    rng = np.random.default_rng(seed)
    timestamps = tuple(float(index) for index in range(6))
    truth = _truth_knots(timestamps)
    windows = [
        (float(start), float(start) + 0.5)
        for start in (0.1, 0.7, 1.3, 1.9, 2.5, 3.1, 3.7, 4.3)
    ]
    split_start = 0.35 * timestamps[-1]
    split_end = 0.65 * timestamps[-1]
    train_windows = [
        window
        for window in windows
        if window[1] <= split_start or window[0] >= split_end
    ]
    holdout_windows = [
        window
        for window in windows
        if window[0] >= split_start and window[1] <= split_end
    ]
    train = _measurements(truth, timestamps, train_windows, rng, prefix="train")
    holdout = _measurements(truth, timestamps, holdout_windows, rng, prefix="holdout")
    initial = tuple(
        se3_exp(knot, np.concatenate([np.zeros(3), rng.normal(0.0, 0.03, 3)]))
        for knot in truth
    )
    initial_bias = _tuple3(
        float(value + rng.normal(0.0, 0.01)) for value in _TRUE_GYRO_BIAS_RAD_S
    )
    result = fit_continuous_trajectory(
        ContinuousTimeTrajectoryFitProblem(
            knot_timestamps=timestamps,
            initial_knot_poses=initial,
            pose_measurements=(
                TrajectoryPoseMeasurement(
                    measurement_id="gauge-start",
                    timestamp_sec=timestamps[0],
                    pose_world_body=truth[0],
                    weight=25.0,
                ),
            ),
            imu_preintegration_measurements=train,
            initial_gyro_bias_rad_s=initial_bias,
            estimate_gyro_bias=True,
        ),
        ContinuousTimeTrajectoryFitOptions(max_iterations=80),
    )
    train_rmse = result.final_imu_rotation_rmse_rad
    if train_rmse is None:
        raise ValueError("IMU recovery produced no train RMSE")
    fitted_bias = result.gyro_bias_rad_s
    if fitted_bias is None:
        raise ValueError("IMU recovery produced no gyro bias")
    holdout_rmse = imu_preintegration_rmse(
        holdout, timestamps, result.knot_poses, fitted_bias
    )
    if holdout_rmse is None:
        raise ValueError("IMU recovery produced no holdout RMSE")
    known_bad_bias = tuple(
        float(left + right)
        for left, right in zip(fitted_bias, _KNOWN_BAD_GYRO_BIAS_RAD_S, strict=True)
    )
    known_bad_rmse = imu_preintegration_rmse(
        holdout, timestamps, result.knot_poses, known_bad_bias
    )
    if known_bad_rmse is None:
        raise ValueError("IMU recovery produced no known-bad RMSE")
    rotation_errors = [
        float(np.linalg.norm(se3_log(estimate, knot)[3:]))
        for estimate, knot in zip(result.knot_poses, truth, strict=True)
    ]
    max_rotation_error = max(rotation_errors)
    bias_error = float(
        np.linalg.norm(
            np.asarray(fitted_bias, dtype=float)
            - np.asarray(_TRUE_GYRO_BIAS_RAD_S, dtype=float)
        )
    )
    known_bad_delta = known_bad_rmse - holdout_rmse
    policy_status, policy_reason = _policy(
        max_rotation_error=max_rotation_error,
        bias_error=bias_error,
        holdout_rmse=holdout_rmse,
        known_bad_delta=known_bad_delta,
    )
    return ContinuousTimeImuPreintegrationArtifact(
        recovery_id=recovery_id,
        interpolation="screw_linear",
        seed=seed,
        knot_count=len(truth),
        train_measurement_count=len(train),
        holdout_measurement_count=len(holdout),
        fit_status=result.status,
        max_knot_rotation_error_rad=max_rotation_error,
        gyro_bias_error_rad_s=bias_error,
        train_imu_rotation_rmse_rad=train_rmse,
        holdout_imu_rotation_rmse_rad=holdout_rmse,
        known_bad_holdout_rmse_rad=known_bad_rmse,
        known_bad_rmse_delta_rad=known_bad_delta,
        known_bad_gyro_bias_rad_s=_KNOWN_BAD_GYRO_BIAS_RAD_S,
        policy_status=policy_status,
        policy_reason=policy_reason,
        provenance=ContinuousTimeImuPreintegrationProvenance(
            generator=__name__,
            generator_version="0.1",
            git_commit=git_commit(),
            command=command or [],
            seed=seed,
        ),
    )


def _truth_knots(timestamps: tuple[float, ...]) -> tuple[SE3, ...]:
    knots = []
    for timestamp in timestamps:
        translation = np.array([0.2 * timestamp, 0.0, 0.0])
        rotation = np.array(
            [0.08 * math.sin(timestamp), 0.05 * timestamp, 0.25 * timestamp]
        )
        knots.append(se3_exp(SE3.identity(), np.concatenate([translation, rotation])))
    return tuple(knots)


def _measurements(
    knots: tuple[SE3, ...],
    timestamps: tuple[float, ...],
    windows: list[tuple[float, float]],
    rng: np.random.Generator,
    *,
    prefix: str,
) -> tuple[TrajectoryImuPreintegrationMeasurement, ...]:
    dt = 1.0 / _GYRO_RATE_HZ
    measurements: list[TrajectoryImuPreintegrationMeasurement] = []
    bias = np.asarray(_TRUE_GYRO_BIAS_RAD_S, dtype=float)
    for index, (start, end) in enumerate(windows):
        sample_times = np.arange(start, end + 0.5 * dt, dt)
        samples: list[TrajectoryImuGyroSample] = []
        for time in sample_times:
            omega = _body_rate(knots, timestamps, float(time))
            noisy = omega + bias + rng.normal(0.0, 0.002, 3)
            samples.append(
                TrajectoryImuGyroSample(
                    timestamp_sec=float(time),
                    omega_body_rad_s=_tuple3(noisy),
                )
            )
        measurements.append(
            TrajectoryImuPreintegrationMeasurement(
                measurement_id=f"{prefix}-{index:04d}",
                gyro_samples=tuple(samples),
            )
        )
    return tuple(measurements)


def _body_rate(
    knots: tuple[SE3, ...],
    timestamps: tuple[float, ...],
    timestamp_sec: float,
) -> NDArray[np.float64]:
    epsilon = 1.0e-3
    t0 = min(max(timestamp_sec, timestamps[0]), timestamps[-1] - epsilon)
    t1 = t0 + epsilon
    pose0 = interpolate_pose_at(knots, timestamps, t0)
    pose1 = interpolate_pose_at(knots, timestamps, t1)
    return se3_log(pose0, pose1)[3:] / epsilon


def _policy(
    *,
    max_rotation_error: float,
    bias_error: float,
    holdout_rmse: float,
    known_bad_delta: float,
) -> tuple[Literal["pass", "fail"], str]:
    if max_rotation_error >= 0.08:
        return (
            "fail",
            f"knot rotation error {max_rotation_error:.4f} rad exceeds the 0.08 budget",
        )
    if bias_error >= 0.03:
        return (
            "fail",
            f"gyro bias error {bias_error:.4f} rad/s exceeds the 0.03 budget",
        )
    if holdout_rmse >= 0.03:
        return (
            "fail",
            f"holdout IMU rotation RMSE {holdout_rmse:.4f} rad is too large",
        )
    if known_bad_delta <= 0.02:
        return (
            "fail",
            (
                f"known-bad +{_KNOWN_BAD_GYRO_BIAS_RAD_S[2]:.2f} rad/s z bias only "
                f"increased holdout RMSE by {known_bad_delta:.4f} rad"
            ),
        )
    return (
        "pass",
        (
            "synthetic IMU recovery meets rotation and bias budgets, holdout RMSE "
            "stays tight, and the signed gyro-bias control increases holdout error"
        ),
    )
