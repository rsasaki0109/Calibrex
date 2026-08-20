"""Tests for continuous-time IMU lever-arm factors."""

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
from calibrex.evaluation.continuous_time_imu_lever_arm import (
    run_synthetic_imu_lever_arm_recovery,
)


def _truth_knots(count: int = 4) -> tuple[tuple[float, ...], tuple[SE3, ...]]:
    timestamps = tuple(float(index) for index in range(count))
    knots = tuple(
        se3_exp(
            SE3.identity(),
            np.concatenate(
                [
                    np.array([0.15 * index * index, 0.05 * index, 0.0]),
                    np.array([0.2 * np.sin(1.3 * index), 0.25 * index, 0.45 * index]),
                ]
            ),
        )
        for index in timestamps
    )
    return timestamps, knots


def test_lever_arm_jacobian_matches_finite_differences() -> None:
    timestamps, knots = _truth_knots()
    lever = np.array([0.1, -0.05, 0.2], dtype=float)
    measurement = TrajectoryImuLeverArmMeasurement(
        measurement_id="lever0",
        timestamp_sec=1.4,
        accel_body_m_s2=(0.3, -0.1, 0.05),
    )
    residual, jacobian = _lever_arm_residual(
        list(knots), timestamps, measurement, lever
    )[:2]
    epsilon = 1.0e-6
    numeric = np.zeros((3, 3))
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = epsilon
        perturbed_residual, _ = _lever_arm_residual(
            list(knots), timestamps, measurement, lever + step
        )[:2]
        numeric[:, axis] = (perturbed_residual - residual) / epsilon
    assert np.allclose(jacobian, numeric, atol=1.0e-6, rtol=1.0e-5)


def test_lever_arm_recovers_displaced_imu_origin() -> None:
    timestamps, truth = _truth_knots()
    true_lever = (0.12, -0.08, 0.22)
    sample_times = [0.4, 0.9, 1.4, 1.9, 2.4, 2.8]
    measurements = tuple(
        TrajectoryImuLeverArmMeasurement(
            measurement_id=f"lever{index}",
            timestamp_sec=time,
            accel_body_m_s2=tuple(
                float(value)
                for value in predicted_imu_specific_force(
                    truth, timestamps, time, true_lever
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
            initial_lever_arm_body_m=(0.0, 0.0, 0.0),
            estimate_lever_arm=True,
        ),
        ContinuousTimeTrajectoryFitOptions(max_iterations=20),
    )
    assert result.status in {"converged", "max_iterations"}
    assert result.lever_arm_body_m is not None
    assert (
        np.linalg.norm(
            np.asarray(result.lever_arm_body_m, dtype=float)
            - np.asarray(true_lever, dtype=float)
        )
        < 0.02
    )


def test_synthetic_lever_arm_recovery_holdout_and_known_bad_control(
    tmp_path: Path,
) -> None:
    artifact = run_synthetic_imu_lever_arm_recovery(seed=20260820)
    output = tmp_path / "recovery.yaml"
    artifact.save(output)
    assert validate_file(output, kind="continuous-time-imu-lever-arm").valid
    assert artifact.policy_status == "pass"
    assert artifact.holdout_measurement_count >= 1
    assert artifact.known_bad_rmse_delta_m_s2 > 0.04
    assert artifact.provenance.seed == 20260820


def test_cli_recover_imu_lever_arm(tmp_path: Path) -> None:
    output = tmp_path / "recovery.yaml"
    exit_code = main(
        [
            "trajectory",
            "recover-imu-lever-arm",
            "--output",
            str(output),
            "--seed",
            "20260820",
            "--json",
        ]
    )
    assert exit_code == 0
    assert validate_file(output).valid
