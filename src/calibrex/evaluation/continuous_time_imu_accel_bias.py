"""Synthetic recovery for continuous-time accelerometer bias and gravity."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from calibrex.core.continuous_time_imu_accel_bias import (
    ContinuousTimeImuAccelBiasArtifact,
    ContinuousTimeImuAccelBiasProvenance,
)
from calibrex.core.continuous_time_sparse import (
    ContinuousTimeTrajectoryFitOptions,
    ContinuousTimeTrajectoryFitProblem,
    TrajectoryImuLeverArmMeasurement,
    TrajectoryPoseMeasurement,
    fit_continuous_trajectory,
    imu_lever_arm_rmse,
    predicted_imu_specific_force,
)
from calibrex.core.geometry import SE3, _tuple3
from calibrex.core.provenance import git_commit
from calibrex.core.se3_manifold import se3_exp, se3_log

_SYNTHETIC_SEED = 20260820
_TRUE_LEVER_ARM_BODY_M = (0.12, -0.08, 0.22)
_TRUE_ACCEL_BIAS_BODY_M_S2 = (0.05, -0.04, 0.03)
_TRUE_GRAVITY_WORLD_M_S2 = (0.0, 0.0, -9.81)
_KNOWN_BAD_ACCEL_BIAS_BODY_M_S2 = (0.0, 0.0, 0.3)
_ACCEL_NOISE_M_S2 = 0.015


def run_synthetic_imu_accel_bias_recovery(
    *,
    seed: int = _SYNTHETIC_SEED,
    command: list[str] | None = None,
    recovery_id: str = "ct-imu-accel-bias-synthetic",
) -> ContinuousTimeImuAccelBiasArtifact:
    """Recover accelerometer bias and gravity with holdout and a signed control."""

    rng = np.random.default_rng(seed)
    timestamps = tuple(float(index) for index in range(7))
    truth = _truth_knots(timestamps)
    sample_times = [
        float(time)
        for time in np.arange(0.2, timestamps[-1] - 0.2 + 1.0e-9, 0.15)
    ]
    split_start = 0.35 * timestamps[-1]
    split_end = 0.65 * timestamps[-1]
    train_times = [
        time
        for time in sample_times
        if time <= split_start or time >= split_end
    ]
    holdout_times = [
        time for time in sample_times if split_start <= time <= split_end
    ]
    train = _measurements(truth, timestamps, train_times, rng, prefix="train")
    holdout = _measurements(
        truth, timestamps, holdout_times, rng, prefix="holdout"
    )
    result = fit_continuous_trajectory(
        ContinuousTimeTrajectoryFitProblem(
            knot_timestamps=timestamps,
            initial_knot_poses=truth,
            pose_measurements=_pose_gauges(truth, timestamps),
            imu_lever_arm_measurements=train,
            initial_lever_arm_body_m=_TRUE_LEVER_ARM_BODY_M,
            initial_accel_bias_body_m_s2=(0.0, 0.0, 0.0),
            initial_gravity_world_m_s2=_TRUE_GRAVITY_WORLD_M_S2,
            estimate_accel_bias=True,
            estimate_gravity=True,
        ),
        ContinuousTimeTrajectoryFitOptions(max_iterations=40),
    )
    train_rmse = result.final_lever_arm_rmse_m_s2
    if train_rmse is None:
        raise ValueError("accel-bias recovery produced no train RMSE")
    fitted_bias = result.accel_bias_body_m_s2
    fitted_gravity = result.gravity_world_m_s2
    fitted_lever = result.lever_arm_body_m
    if fitted_bias is None or fitted_gravity is None or fitted_lever is None:
        raise ValueError("accel-bias recovery produced no bias or gravity")
    holdout_rmse = imu_lever_arm_rmse(
        holdout,
        timestamps,
        result.knot_poses,
        fitted_lever,
        accel_bias_body_m_s2=fitted_bias,
        gravity_world_m_s2=fitted_gravity,
    )
    if holdout_rmse is None:
        raise ValueError("accel-bias recovery produced no holdout RMSE")
    known_bad_bias = tuple(
        float(left + right)
        for left, right in zip(
            fitted_bias, _KNOWN_BAD_ACCEL_BIAS_BODY_M_S2, strict=True
        )
    )
    known_bad_rmse = imu_lever_arm_rmse(
        holdout,
        timestamps,
        result.knot_poses,
        fitted_lever,
        accel_bias_body_m_s2=known_bad_bias,
        gravity_world_m_s2=fitted_gravity,
    )
    if known_bad_rmse is None:
        raise ValueError("accel-bias recovery produced no known-bad RMSE")
    translation_errors = [
        float(np.linalg.norm(se3_log(estimate, knot)[:3]))
        for estimate, knot in zip(result.knot_poses, truth, strict=True)
    ]
    max_translation_error = max(translation_errors)
    bias_error = float(
        np.linalg.norm(
            np.asarray(fitted_bias, dtype=float)
            - np.asarray(_TRUE_ACCEL_BIAS_BODY_M_S2, dtype=float)
        )
    )
    gravity_error = float(
        np.linalg.norm(
            np.asarray(fitted_gravity, dtype=float)
            - np.asarray(_TRUE_GRAVITY_WORLD_M_S2, dtype=float)
        )
    )
    known_bad_delta = known_bad_rmse - holdout_rmse
    policy_status, policy_reason = _policy(
        max_translation_error=max_translation_error,
        bias_error=bias_error,
        gravity_error=gravity_error,
        holdout_rmse=holdout_rmse,
        known_bad_delta=known_bad_delta,
    )
    return ContinuousTimeImuAccelBiasArtifact(
        recovery_id=recovery_id,
        interpolation="screw_linear",
        seed=seed,
        knot_count=len(truth),
        train_measurement_count=len(train),
        holdout_measurement_count=len(holdout),
        fit_status=result.status,
        max_knot_translation_error_m=max_translation_error,
        accel_bias_error_m_s2=bias_error,
        gravity_error_m_s2=gravity_error,
        train_lever_arm_rmse_m_s2=train_rmse,
        holdout_lever_arm_rmse_m_s2=holdout_rmse,
        known_bad_holdout_rmse_m_s2=known_bad_rmse,
        known_bad_rmse_delta_m_s2=known_bad_delta,
        known_bad_accel_bias_body_m_s2=_KNOWN_BAD_ACCEL_BIAS_BODY_M_S2,
        policy_status=policy_status,
        policy_reason=policy_reason,
        provenance=ContinuousTimeImuAccelBiasProvenance(
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
        translation = np.array(
            [
                0.18 * timestamp * timestamp,
                0.08 * timestamp,
                0.05 * math.sin(timestamp),
            ]
        )
        rotation = np.array(
            [
                0.9 * math.sin(1.7 * timestamp),
                0.7 * math.sin(2.1 * timestamp + 0.4),
                0.55 * timestamp,
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
    sample_times: list[float],
    rng: np.random.Generator,
    *,
    prefix: str,
) -> tuple[TrajectoryImuLeverArmMeasurement, ...]:
    measurements: list[TrajectoryImuLeverArmMeasurement] = []
    for index, time in enumerate(sample_times):
        predicted = predicted_imu_specific_force(
            knots,
            timestamps,
            time,
            _TRUE_LEVER_ARM_BODY_M,
            accel_bias_body_m_s2=_TRUE_ACCEL_BIAS_BODY_M_S2,
            gravity_world_m_s2=_TRUE_GRAVITY_WORLD_M_S2,
        )
        noisy = predicted + rng.normal(0.0, _ACCEL_NOISE_M_S2, 3)
        measurements.append(
            TrajectoryImuLeverArmMeasurement(
                measurement_id=f"{prefix}-{index:04d}",
                timestamp_sec=time,
                accel_body_m_s2=_tuple3(noisy),
            )
        )
    return tuple(measurements)


def _policy(
    *,
    max_translation_error: float,
    bias_error: float,
    gravity_error: float,
    holdout_rmse: float,
    known_bad_delta: float,
) -> tuple[Literal["pass", "fail"], str]:
    if max_translation_error >= 0.02:
        return (
            "fail",
            (
                f"knot translation error {max_translation_error:.4f} m exceeds "
                "the 0.02 budget"
            ),
        )
    if bias_error >= 0.06:
        return (
            "fail",
            f"accelerometer-bias error {bias_error:.4f} m/s^2 exceeds the 0.06 budget",
        )
    if gravity_error >= 0.12:
        return (
            "fail",
            f"gravity error {gravity_error:.4f} m/s^2 exceeds the 0.12 budget",
        )
    if holdout_rmse >= 0.08:
        return (
            "fail",
            f"holdout specific-force RMSE {holdout_rmse:.4f} m/s^2 is too large",
        )
    if known_bad_delta <= 0.08:
        return (
            "fail",
            (
                f"known-bad +{_KNOWN_BAD_ACCEL_BIAS_BODY_M_S2[2]:.2f} m/s^2 z "
                "accel bias only increased holdout RMSE by "
                f"{known_bad_delta:.4f} m/s^2"
            ),
        )
    return (
        "pass",
        (
            "synthetic IMU accelerometer-bias recovery meets translation, bias, "
            "and gravity budgets, holdout RMSE stays tight, and the signed "
            "accel-bias control increases holdout error"
        ),
    )
