"""Synthetic recovery for continuous-time IMU lever-arm factors."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from calibrex.core.continuous_time_imu_lever_arm import (
    ContinuousTimeImuLeverArmArtifact,
    ContinuousTimeImuLeverArmProvenance,
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
_KNOWN_BAD_LEVER_ARM_BODY_M = (0.0, 0.0, 0.15)
_ACCEL_NOISE_M_S2 = 0.02


def run_synthetic_imu_lever_arm_recovery(
    *,
    seed: int = _SYNTHETIC_SEED,
    command: list[str] | None = None,
    recovery_id: str = "ct-imu-lever-arm-synthetic",
) -> ContinuousTimeImuLeverArmArtifact:
    """Recover the IMU lever arm with holdout and a signed control."""

    rng = np.random.default_rng(seed)
    timestamps = tuple(float(index) for index in range(6))
    truth = _truth_knots(timestamps)
    sample_times = [
        float(time)
        for time in np.arange(0.15, timestamps[-1] - 0.15 + 1.0e-9, 0.2)
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
    initial_lever = _tuple3(
        float(value + rng.normal(0.0, 0.04)) for value in _TRUE_LEVER_ARM_BODY_M
    )
    result = fit_continuous_trajectory(
        ContinuousTimeTrajectoryFitProblem(
            knot_timestamps=timestamps,
            initial_knot_poses=truth,
            pose_measurements=_pose_gauges(truth, timestamps),
            imu_lever_arm_measurements=train,
            initial_lever_arm_body_m=initial_lever,
            estimate_lever_arm=True,
        ),
        ContinuousTimeTrajectoryFitOptions(max_iterations=40),
    )
    train_rmse = result.final_lever_arm_rmse_m_s2
    if train_rmse is None:
        raise ValueError("lever-arm recovery produced no train RMSE")
    fitted_lever = result.lever_arm_body_m
    if fitted_lever is None:
        raise ValueError("lever-arm recovery produced no lever arm")
    holdout_rmse = imu_lever_arm_rmse(
        holdout, timestamps, result.knot_poses, fitted_lever
    )
    if holdout_rmse is None:
        raise ValueError("lever-arm recovery produced no holdout RMSE")
    known_bad_lever = tuple(
        float(left + right)
        for left, right in zip(
            fitted_lever, _KNOWN_BAD_LEVER_ARM_BODY_M, strict=True
        )
    )
    known_bad_rmse = imu_lever_arm_rmse(
        holdout, timestamps, result.knot_poses, known_bad_lever
    )
    if known_bad_rmse is None:
        raise ValueError("lever-arm recovery produced no known-bad RMSE")
    translation_errors = [
        float(np.linalg.norm(se3_log(estimate, knot)[:3]))
        for estimate, knot in zip(result.knot_poses, truth, strict=True)
    ]
    max_translation_error = max(translation_errors)
    lever_error = float(
        np.linalg.norm(
            np.asarray(fitted_lever, dtype=float)
            - np.asarray(_TRUE_LEVER_ARM_BODY_M, dtype=float)
        )
    )
    known_bad_delta = known_bad_rmse - holdout_rmse
    policy_status, policy_reason = _policy(
        max_translation_error=max_translation_error,
        lever_error=lever_error,
        holdout_rmse=holdout_rmse,
        known_bad_delta=known_bad_delta,
    )
    return ContinuousTimeImuLeverArmArtifact(
        recovery_id=recovery_id,
        interpolation="screw_linear",
        seed=seed,
        knot_count=len(truth),
        train_measurement_count=len(train),
        holdout_measurement_count=len(holdout),
        fit_status=result.status,
        max_knot_translation_error_m=max_translation_error,
        lever_arm_error_m=lever_error,
        train_lever_arm_rmse_m_s2=train_rmse,
        holdout_lever_arm_rmse_m_s2=holdout_rmse,
        known_bad_holdout_rmse_m_s2=known_bad_rmse,
        known_bad_rmse_delta_m_s2=known_bad_delta,
        known_bad_lever_arm_body_m=_KNOWN_BAD_LEVER_ARM_BODY_M,
        policy_status=policy_status,
        policy_reason=policy_reason,
        provenance=ContinuousTimeImuLeverArmProvenance(
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
                0.55 * math.sin(1.4 * timestamp),
                0.35 * timestamp,
                0.7 * timestamp,
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
            knots, timestamps, time, _TRUE_LEVER_ARM_BODY_M
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
    lever_error: float,
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
    if lever_error >= 0.03:
        return (
            "fail",
            f"lever-arm error {lever_error:.4f} m exceeds the 0.03 budget",
        )
    if holdout_rmse >= 0.08:
        return (
            "fail",
            f"holdout lever-arm RMSE {holdout_rmse:.4f} m/s^2 is too large",
        )
    if known_bad_delta <= 0.04:
        return (
            "fail",
            (
                f"known-bad +{_KNOWN_BAD_LEVER_ARM_BODY_M[2]:.2f} m z lever arm "
                f"only increased holdout RMSE by {known_bad_delta:.4f} m/s^2"
            ),
        )
    return (
        "pass",
        (
            "synthetic IMU lever-arm recovery meets translation and lever "
            "budgets, holdout RMSE stays tight, and the signed lever-arm "
            "control increases holdout error"
        ),
    )
