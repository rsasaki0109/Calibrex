"""Tests for continuous-time IMU accelerometer bias and gravity."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from calibrex.cli.main import main
from calibrex.core.continuous_time_sparse import (
    ContinuousTimeTrajectoryFitOptions,
    ContinuousTimeTrajectoryFitProblem,
    TrajectoryImuLeverArmMeasurement,
    TrajectoryPoseMeasurement,
    _lever_arm_residual,
    fit_continuous_trajectory,
    predicted_imu_specific_force,
)
from calibrex.core.geometry import SE3
from calibrex.core.se3_manifold import se3_exp
from calibrex.core.validation import validate_file
from calibrex.evaluation.continuous_time_imu_accel_bias import (
    run_synthetic_imu_accel_bias_recovery,
)


def _truth_knots(count: int = 5) -> tuple[tuple[float, ...], tuple[SE3, ...]]:
    timestamps = tuple(float(index) for index in range(count))
    knots = tuple(
        se3_exp(
            SE3.identity(),
            np.concatenate(
                [
                    np.array([0.15 * index * index, 0.05 * index, 0.0]),
                    np.array(
                        [
                            0.8 * np.sin(1.6 * index),
                            0.6 * np.sin(2.0 * index + 0.3),
                            0.4 * index,
                        ]
                    ),
                ]
            ),
        )
        for index in timestamps
    )
    return timestamps, knots


def test_accel_bias_and_gravity_jacobians_match_finite_differences() -> None:
    timestamps, knots = _truth_knots()
    lever = np.array([0.1, -0.05, 0.2], dtype=float)
    bias = np.array([0.04, -0.03, 0.02], dtype=float)
    gravity = np.array([0.1, -0.2, -9.7], dtype=float)
    measurement = TrajectoryImuLeverArmMeasurement(
        measurement_id="accel0",
        timestamp_sec=1.6,
        accel_body_m_s2=(0.3, -0.1, 9.5),
    )
    residual, _lever, _clock, jacobian_bias, jacobian_gravity, _scale = _lever_arm_residual(
        list(knots),
        timestamps,
        measurement,
        lever,
        0.0,
        bias,
        gravity,
    )
    epsilon = 1.0e-6
    numeric_bias = np.zeros((3, 3))
    numeric_gravity = np.zeros((3, 3))
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = epsilon
        perturbed_bias, *_ = _lever_arm_residual(
            list(knots), timestamps, measurement, lever, 0.0, bias + step, gravity
        )
        numeric_bias[:, axis] = (perturbed_bias - residual) / epsilon
        perturbed_gravity, *_ = _lever_arm_residual(
            list(knots), timestamps, measurement, lever, 0.0, bias, gravity + step
        )
        numeric_gravity[:, axis] = (perturbed_gravity - residual) / epsilon
    assert np.allclose(jacobian_bias, numeric_bias, atol=1.0e-6, rtol=1.0e-5)
    assert np.allclose(jacobian_gravity, numeric_gravity, atol=1.0e-6, rtol=1.0e-5)


def test_accel_bias_and_gravity_recover_from_specific_force() -> None:
    timestamps, truth = _truth_knots()
    true_lever = (0.12, -0.08, 0.22)
    true_bias = (0.05, -0.04, 0.03)
    true_gravity = (0.0, 0.0, -9.81)
    sample_times = [0.4, 0.9, 1.4, 1.9, 2.4, 2.9, 3.4]
    measurements = tuple(
        TrajectoryImuLeverArmMeasurement(
            measurement_id=f"accel{index}",
            timestamp_sec=time,
            accel_body_m_s2=tuple(
                float(value)
                for value in predicted_imu_specific_force(
                    truth,
                    timestamps,
                    time,
                    true_lever,
                    accel_bias_body_m_s2=true_bias,
                    gravity_world_m_s2=true_gravity,
                )
            ),
        )
        for index, time in enumerate(sample_times)
    )
    result = fit_continuous_trajectory(
        ContinuousTimeTrajectoryFitProblem(
            knot_timestamps=timestamps,
            initial_knot_poses=truth,
            pose_measurements=tuple(
                TrajectoryPoseMeasurement(
                    measurement_id=f"gauge{index}",
                    timestamp_sec=timestamp,
                    pose_world_body=pose,
                    weight=50.0,
                )
                for index, (timestamp, pose) in enumerate(
                    zip(timestamps, truth, strict=True)
                )
            ),
            imu_lever_arm_measurements=measurements,
            initial_lever_arm_body_m=true_lever,
            initial_accel_bias_body_m_s2=(0.0, 0.0, 0.0),
            initial_gravity_world_m_s2=true_gravity,
            estimate_accel_bias=True,
            estimate_gravity=True,
        ),
        ContinuousTimeTrajectoryFitOptions(max_iterations=25),
    )
    assert result.status in {"converged", "max_iterations"}
    assert result.accel_bias_body_m_s2 is not None
    assert result.gravity_world_m_s2 is not None
    assert (
        np.linalg.norm(
            np.asarray(result.accel_bias_body_m_s2, dtype=float)
            - np.asarray(true_bias, dtype=float)
        )
        < 0.03
    )
    assert (
        np.linalg.norm(
            np.asarray(result.gravity_world_m_s2, dtype=float)
            - np.asarray(true_gravity, dtype=float)
        )
        < 0.08
    )


def test_synthetic_accel_bias_recovery_holdout_and_known_bad_control(
    tmp_path: Path,
) -> None:
    artifact = run_synthetic_imu_accel_bias_recovery(seed=20260820)
    output = tmp_path / "recovery.yaml"
    artifact.save(output)
    assert validate_file(output, kind="continuous-time-imu-accel-bias").valid
    assert artifact.policy_status == "pass"
    assert artifact.holdout_measurement_count >= 1
    assert artifact.known_bad_rmse_delta_m_s2 > 0.08
    assert artifact.provenance.seed == 20260820


def test_cli_recover_imu_accel_bias(tmp_path: Path) -> None:
    output = tmp_path / "recovery.yaml"
    exit_code = main(
        [
            "trajectory",
            "recover-imu-accel-bias",
            "--output",
            str(output),
            "--seed",
            "20260820",
            "--json",
        ]
    )
    assert exit_code == 0
    assert validate_file(output).valid
