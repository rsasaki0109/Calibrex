"""Synthetic recovery for a shared continuous-time IMU clock offset."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from calibrex.core.continuous_time_imu_clock_offset import (
    ContinuousTimeImuClockOffsetArtifact,
    ContinuousTimeImuClockOffsetProvenance,
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
from calibrex.core.geometry import SE3
from calibrex.core.provenance import git_commit
from calibrex.core.se3_manifold import se3_exp, se3_log

_SYNTHETIC_SEED = 20260820
_TRUE_CLOCK_OFFSET_SEC = 0.015
_KNOWN_BAD_CLOCK_OFFSET_SEC = 0.02
_GYRO_RATE_HZ = 50.0


def run_synthetic_imu_clock_offset_recovery(
    *,
    seed: int = _SYNTHETIC_SEED,
    command: list[str] | None = None,
    recovery_id: str = "ct-imu-clock-offset-synthetic",
) -> ContinuousTimeImuClockOffsetArtifact:
    """Recover a shared IMU clock offset with holdout and a signed control."""

    rng = np.random.default_rng(seed)
    timestamps = tuple(0.5 * index for index in range(11))
    truth = _truth_knots(timestamps)
    windows = [
        (float(start), float(start) + 0.4)
        for start in (0.3, 0.8, 1.3, 1.8, 2.3, 2.8, 3.3, 3.8, 4.3)
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
    holdout = _measurements(
        truth, timestamps, holdout_windows, rng, prefix="holdout"
    )
    result = fit_continuous_trajectory(
        ContinuousTimeTrajectoryFitProblem(
            knot_timestamps=timestamps,
            initial_knot_poses=truth,
            pose_measurements=_pose_gauges(truth, timestamps),
            imu_preintegration_measurements=train,
            initial_imu_clock_offset_sec=0.0,
            estimate_imu_clock_offset=True,
        ),
        ContinuousTimeTrajectoryFitOptions(max_iterations=40),
    )
    train_rmse = result.final_imu_rotation_rmse_rad
    if train_rmse is None:
        raise ValueError("clock-offset recovery produced no train RMSE")
    fitted_clock = result.imu_clock_offset_sec
    if fitted_clock is None:
        raise ValueError("clock-offset recovery produced no clock offset")
    holdout_rmse = imu_preintegration_rmse(
        holdout, timestamps, result.knot_poses, (0.0, 0.0, 0.0), fitted_clock
    )
    if holdout_rmse is None:
        raise ValueError("clock-offset recovery produced no holdout RMSE")
    known_bad_clock = fitted_clock + _KNOWN_BAD_CLOCK_OFFSET_SEC
    known_bad_rmse = imu_preintegration_rmse(
        holdout, timestamps, result.knot_poses, (0.0, 0.0, 0.0), known_bad_clock
    )
    if known_bad_rmse is None:
        raise ValueError("clock-offset recovery produced no known-bad RMSE")
    rotation_errors = [
        float(np.linalg.norm(se3_log(estimate, knot)[3:]))
        for estimate, knot in zip(result.knot_poses, truth, strict=True)
    ]
    max_rotation_error = max(rotation_errors)
    clock_error = abs(fitted_clock - _TRUE_CLOCK_OFFSET_SEC)
    known_bad_delta = known_bad_rmse - holdout_rmse
    policy_status, policy_reason = _policy(
        max_rotation_error=max_rotation_error,
        clock_error=clock_error,
        holdout_rmse=holdout_rmse,
        known_bad_delta=known_bad_delta,
    )
    return ContinuousTimeImuClockOffsetArtifact(
        recovery_id=recovery_id,
        interpolation="screw_linear",
        seed=seed,
        knot_count=len(truth),
        train_measurement_count=len(train),
        holdout_measurement_count=len(holdout),
        fit_status=result.status,
        max_knot_rotation_error_rad=max_rotation_error,
        imu_clock_offset_error_sec=clock_error,
        train_imu_rotation_rmse_rad=train_rmse,
        holdout_imu_rotation_rmse_rad=holdout_rmse,
        known_bad_holdout_rmse_rad=known_bad_rmse,
        known_bad_rmse_delta_rad=known_bad_delta,
        known_bad_imu_clock_offset_sec=_KNOWN_BAD_CLOCK_OFFSET_SEC,
        policy_status=policy_status,
        policy_reason=policy_reason,
        provenance=ContinuousTimeImuClockOffsetProvenance(
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
        translation = np.array([0.15 * timestamp, 0.04 * timestamp, 0.0])
        rotation = np.array(
            [
                0.9 * math.sin(2.4 * timestamp),
                0.8 * math.sin(3.1 * timestamp + 0.5),
                0.35 * timestamp * timestamp + 0.7 * math.sin(1.8 * timestamp),
            ]
        )
        knots.append(se3_exp(SE3.identity(), np.concatenate([translation, rotation])))
    return tuple(knots)


def _pose_gauges(
    knots: tuple[SE3, ...],
    timestamps: tuple[float, ...],
) -> tuple[TrajectoryPoseMeasurement, ...]:
    return tuple(
        TrajectoryPoseMeasurement(
            measurement_id=f"gauge-{index:02d}",
            timestamp_sec=timestamp,
            pose_world_body=pose,
            weight=100.0,
        )
        for index, (timestamp, pose) in enumerate(zip(timestamps, knots, strict=True))
    )


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
    for index, (start, end) in enumerate(windows):
        sample_times = np.arange(start, end + 0.5 * dt, dt)
        samples: list[TrajectoryImuGyroSample] = []
        for body_time in sample_times:
            omega = _body_rate(knots, timestamps, float(body_time))
            noisy = omega + rng.normal(0.0, 0.002, 3)
            samples.append(
                TrajectoryImuGyroSample(
                    timestamp_sec=float(body_time) + _TRUE_CLOCK_OFFSET_SEC,
                    omega_body_rad_s=tuple(float(value) for value in noisy),
                )
            )
        measurements.append(
            TrajectoryImuPreintegrationMeasurement(
                measurement_id=f"{prefix}-{index:04d}",
                gyro_samples=tuple(samples),
                weight=25.0,
            )
        )
    return tuple(measurements)


def _body_rate(
    knots: tuple[SE3, ...],
    timestamps: tuple[float, ...],
    timestamp_sec: float,
) -> np.ndarray:
    epsilon = 1.0e-3
    t0 = min(max(timestamp_sec, timestamps[0]), timestamps[-1] - epsilon)
    t1 = t0 + epsilon
    pose0 = interpolate_pose_at(knots, timestamps, t0)
    pose1 = interpolate_pose_at(knots, timestamps, t1)
    return se3_log(pose0, pose1)[3:] / epsilon


def _policy(
    *,
    max_rotation_error: float,
    clock_error: float,
    holdout_rmse: float,
    known_bad_delta: float,
) -> tuple[Literal["pass", "fail"], str]:
    if max_rotation_error >= 0.02:
        return (
            "fail",
            (
                f"knot rotation error {max_rotation_error:.4f} rad exceeds "
                "the 0.02 budget"
            ),
        )
    if clock_error >= 0.006:
        return (
            "fail",
            f"IMU clock-offset error {clock_error:.4f} s exceeds the 0.006 budget",
        )
    if holdout_rmse >= 0.04:
        return (
            "fail",
            f"holdout IMU rotation RMSE {holdout_rmse:.4f} rad is too large",
        )
    if known_bad_delta <= 0.008:
        return (
            "fail",
            (
                f"known-bad +{_KNOWN_BAD_CLOCK_OFFSET_SEC:.3f} s clock offset "
                f"only increased holdout RMSE by {known_bad_delta:.4f} rad"
            ),
        )
    return (
        "pass",
        (
            "synthetic IMU clock-offset recovery meets rotation and delay "
            "budgets, holdout RMSE stays tight, and the signed clock control "
            "increases holdout error"
        ),
    )
