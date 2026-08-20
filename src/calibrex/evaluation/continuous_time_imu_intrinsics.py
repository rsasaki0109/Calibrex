"""Synthetic recovery for diagonal IMU scale intrinsics."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from calibrex.core.continuous_time_imu_intrinsics import (
    ContinuousTimeImuIntrinsicsArtifact,
    ContinuousTimeImuIntrinsicsProvenance,
)
from calibrex.core.continuous_time_sparse import (
    ContinuousTimeTrajectoryFitOptions,
    ContinuousTimeTrajectoryFitProblem,
    TrajectoryImuGyroSample,
    TrajectoryImuLeverArmMeasurement,
    TrajectoryImuPreintegrationMeasurement,
    TrajectoryPoseMeasurement,
    fit_continuous_trajectory,
    imu_lever_arm_rmse,
    imu_preintegration_rmse,
    interpolate_pose_at,
    predicted_imu_specific_force,
)
from calibrex.core.geometry import SE3
from calibrex.core.provenance import git_commit
from calibrex.core.se3_manifold import se3_exp, se3_log

_SYNTHETIC_SEED = 20260820
_TRUE_LEVER_ARM_BODY_M = (0.12, -0.08, 0.22)
_TRUE_GYRO_SCALE = (1.05, 0.98, 1.03)
_TRUE_ACCEL_SCALE = (1.02, 0.97, 1.04)
_TRUE_GRAVITY_WORLD_M_S2 = (0.0, 0.0, -9.81)
_KNOWN_BAD_GYRO_SCALE_DELTA = (0.0, 0.0, 0.05)
_GYRO_RATE_HZ = 50.0
_ACCEL_NOISE_M_S2 = 0.012


def run_synthetic_imu_intrinsics_recovery(
    *,
    seed: int = _SYNTHETIC_SEED,
    command: list[str] | None = None,
    recovery_id: str = "ct-imu-intrinsics-synthetic",
) -> ContinuousTimeImuIntrinsicsArtifact:
    """Recover diagonal gyro and accelerometer scales with holdout and control."""

    rng = np.random.default_rng(seed)
    timestamps = tuple(0.5 * index for index in range(11))
    truth = _truth_knots(timestamps)
    gyro_windows = [
        (float(start), float(start) + 0.4)
        for start in (0.3, 0.8, 1.3, 1.8, 2.3, 2.8, 3.3, 3.8, 4.3)
    ]
    accel_times = [
        float(time)
        for time in np.arange(0.25, timestamps[-1] - 0.25 + 1.0e-9, 0.18)
    ]
    split_start = 0.35 * timestamps[-1]
    split_end = 0.65 * timestamps[-1]
    train_gyro_windows = [
        window
        for window in gyro_windows
        if window[1] <= split_start or window[0] >= split_end
    ]
    holdout_gyro_windows = [
        window
        for window in gyro_windows
        if window[0] >= split_start and window[1] <= split_end
    ]
    train_accel_times = [
        time
        for time in accel_times
        if time <= split_start or time >= split_end
    ]
    holdout_accel_times = [
        time for time in accel_times if split_start <= time <= split_end
    ]
    train_gyro = _gyro_measurements(
        truth, timestamps, train_gyro_windows, rng, prefix="train"
    )
    holdout_gyro = _gyro_measurements(
        truth, timestamps, holdout_gyro_windows, rng, prefix="holdout"
    )
    train_accel = _accel_measurements(
        truth, timestamps, train_accel_times, rng, prefix="train"
    )
    holdout_accel = _accel_measurements(
        truth, timestamps, holdout_accel_times, rng, prefix="holdout"
    )
    result = fit_continuous_trajectory(
        ContinuousTimeTrajectoryFitProblem(
            knot_timestamps=timestamps,
            initial_knot_poses=truth,
            pose_measurements=_pose_gauges(truth, timestamps),
            imu_preintegration_measurements=train_gyro,
            imu_lever_arm_measurements=train_accel,
            initial_lever_arm_body_m=_TRUE_LEVER_ARM_BODY_M,
            initial_gravity_world_m_s2=_TRUE_GRAVITY_WORLD_M_S2,
            estimate_gyro_scale=True,
            estimate_accel_scale=True,
        ),
        ContinuousTimeTrajectoryFitOptions(max_iterations=40),
    )
    train_imu_rmse = result.final_imu_rotation_rmse_rad
    train_accel_rmse = result.final_lever_arm_rmse_m_s2
    fitted_gyro_scale = result.gyro_scale
    fitted_accel_scale = result.accel_scale
    if (
        train_imu_rmse is None
        or train_accel_rmse is None
        or fitted_gyro_scale is None
        or fitted_accel_scale is None
    ):
        raise ValueError("intrinsics recovery produced incomplete fit outputs")
    holdout_imu_rmse = imu_preintegration_rmse(
        holdout_gyro, timestamps, result.knot_poses, (0.0, 0.0, 0.0), 0.0, fitted_gyro_scale
    )
    holdout_accel_rmse = imu_lever_arm_rmse(
        holdout_accel,
        timestamps,
        result.knot_poses,
        _TRUE_LEVER_ARM_BODY_M,
        accel_scale=fitted_accel_scale,
        gravity_world_m_s2=_TRUE_GRAVITY_WORLD_M_S2,
    )
    if holdout_imu_rmse is None or holdout_accel_rmse is None:
        raise ValueError("intrinsics recovery produced no holdout RMSE")
    known_bad_scale = tuple(
        float(left + right)
        for left, right in zip(fitted_gyro_scale, _KNOWN_BAD_GYRO_SCALE_DELTA, strict=True)
    )
    known_bad_imu_rmse = imu_preintegration_rmse(
        holdout_gyro, timestamps, result.knot_poses, (0.0, 0.0, 0.0), 0.0, known_bad_scale
    )
    if known_bad_imu_rmse is None:
        raise ValueError("intrinsics recovery produced no known-bad RMSE")
    gyro_error = float(
        np.linalg.norm(
            np.asarray(fitted_gyro_scale, dtype=float)
            - np.asarray(_TRUE_GYRO_SCALE, dtype=float)
        )
    )
    accel_error = float(
        np.linalg.norm(
            np.asarray(fitted_accel_scale, dtype=float)
            - np.asarray(_TRUE_ACCEL_SCALE, dtype=float)
        )
    )
    known_bad_delta = known_bad_imu_rmse - holdout_imu_rmse
    policy_status, policy_reason = _policy(
        gyro_error=gyro_error,
        accel_error=accel_error,
        holdout_imu_rmse=holdout_imu_rmse,
        holdout_accel_rmse=holdout_accel_rmse,
        known_bad_delta=known_bad_delta,
    )
    return ContinuousTimeImuIntrinsicsArtifact(
        recovery_id=recovery_id,
        interpolation="screw_linear",
        seed=seed,
        knot_count=len(truth),
        train_gyro_measurement_count=len(train_gyro),
        train_accel_measurement_count=len(train_accel),
        holdout_gyro_measurement_count=len(holdout_gyro),
        holdout_accel_measurement_count=len(holdout_accel),
        fit_status=result.status,
        gyro_scale_error=gyro_error,
        accel_scale_error=accel_error,
        train_imu_rotation_rmse_rad=train_imu_rmse,
        holdout_imu_rotation_rmse_rad=holdout_imu_rmse,
        train_lever_arm_rmse_m_s2=train_accel_rmse,
        holdout_lever_arm_rmse_m_s2=holdout_accel_rmse,
        known_bad_holdout_imu_rmse_rad=known_bad_imu_rmse,
        known_bad_imu_rmse_delta_rad=known_bad_delta,
        known_bad_gyro_scale_delta=_KNOWN_BAD_GYRO_SCALE_DELTA,
        policy_status=policy_status,
        policy_reason=policy_reason,
        provenance=ContinuousTimeImuIntrinsicsProvenance(
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


def _gyro_measurements(
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
            scaled = omega / np.asarray(_TRUE_GYRO_SCALE, dtype=float)
            noisy = scaled + rng.normal(0.0, 0.002, 3)
            samples.append(
                TrajectoryImuGyroSample(
                    timestamp_sec=float(body_time),
                    omega_body_rad_s=tuple(float(value) for value in noisy),
                )
            )
        measurements.append(
            TrajectoryImuPreintegrationMeasurement(
                measurement_id=f"{prefix}-gyro-{index:04d}",
                gyro_samples=tuple(samples),
                weight=25.0,
            )
        )
    return tuple(measurements)


def _accel_measurements(
    knots: tuple[SE3, ...],
    timestamps: tuple[float, ...],
    sample_times: list[float],
    rng: np.random.Generator,
    *,
    prefix: str,
) -> tuple[TrajectoryImuLeverArmMeasurement, ...]:
    measurements: list[TrajectoryImuLeverArmMeasurement] = []
    scale = np.asarray(_TRUE_ACCEL_SCALE, dtype=float)
    for index, time in enumerate(sample_times):
        kinematic = predicted_imu_specific_force(
            knots,
            timestamps,
            time,
            _TRUE_LEVER_ARM_BODY_M,
            gravity_world_m_s2=_TRUE_GRAVITY_WORLD_M_S2,
            accel_scale=(1.0, 1.0, 1.0),
        )
        predicted = scale * kinematic
        noisy = predicted + rng.normal(0.0, _ACCEL_NOISE_M_S2, 3)
        measurements.append(
            TrajectoryImuLeverArmMeasurement(
                measurement_id=f"{prefix}-accel-{index:04d}",
                timestamp_sec=time,
                accel_body_m_s2=tuple(float(value) for value in noisy),
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
    gyro_error: float,
    accel_error: float,
    holdout_imu_rmse: float,
    holdout_accel_rmse: float,
    known_bad_delta: float,
) -> tuple[Literal["pass", "fail"], str]:
    if gyro_error >= 0.04:
        return (
            "fail",
            f"gyro-scale error {gyro_error:.4f} exceeds the 0.04 budget",
        )
    if accel_error >= 0.04:
        return (
            "fail",
            f"accelerometer-scale error {accel_error:.4f} exceeds the 0.04 budget",
        )
    if holdout_imu_rmse >= 0.04:
        return (
            "fail",
            f"holdout IMU rotation RMSE {holdout_imu_rmse:.4f} rad is too large",
        )
    if holdout_accel_rmse >= 0.08:
        return (
            "fail",
            f"holdout specific-force RMSE {holdout_accel_rmse:.4f} m/s^2 is too large",
        )
    if known_bad_delta <= 0.008:
        return (
            "fail",
            (
                f"known-bad +{_KNOWN_BAD_GYRO_SCALE_DELTA[2]:.2f} gyro-scale "
                f"only increased holdout RMSE by {known_bad_delta:.4f} rad"
            ),
        )
    return (
        "pass",
        (
            "synthetic IMU intrinsics recovery meets scale, rotation, and "
            "specific-force budgets, and the signed gyro-scale control "
            "increases holdout error"
        ),
    )
