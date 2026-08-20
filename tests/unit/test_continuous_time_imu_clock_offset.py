"""Tests for continuous-time IMU clock-offset factors."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from calibrex.cli.main import main
from calibrex.core.continuous_time_sparse import (
    ContinuousTimeTrajectoryFitOptions,
    ContinuousTimeTrajectoryFitProblem,
    TrajectoryImuGyroSample,
    TrajectoryImuPreintegrationMeasurement,
    TrajectoryPoseMeasurement,
    _imu_preintegration_residual,
    fit_continuous_trajectory,
    interpolate_pose_at,
)
from calibrex.core.geometry import SE3
from calibrex.core.se3_manifold import se3_exp, se3_log
from calibrex.core.validation import validate_file
from calibrex.evaluation.continuous_time_imu_clock_offset import (
    run_synthetic_imu_clock_offset_recovery,
)


def _truth_knots(count: int = 7) -> tuple[tuple[float, ...], tuple[SE3, ...]]:
    timestamps = tuple(0.5 * index for index in range(count))
    knots = tuple(
        se3_exp(
            SE3.identity(),
            np.concatenate(
                [
                    np.array([0.1 * timestamp, 0.0, 0.0]),
                    np.array(
                        [
                            0.9 * np.sin(2.4 * timestamp),
                            0.8 * np.sin(3.1 * timestamp + 0.5),
                            0.35 * timestamp * timestamp
                            + 0.7 * np.sin(1.8 * timestamp),
                        ]
                    ),
                ]
            ),
        )
        for timestamp in timestamps
    )
    return timestamps, knots


def test_imu_clock_offset_jacobian_matches_finite_differences() -> None:
    timestamps, knots = _truth_knots()
    bias = np.zeros(3, dtype=float)
    clock = 0.0
    samples = []
    dt = 0.02
    true_clock = 0.012
    for time in np.arange(0.4, 1.6 + 0.5 * dt, dt):
        t0 = min(max(float(time), timestamps[0]), timestamps[-1] - 1.0e-3)
        omega = (
            se3_log(
                interpolate_pose_at(knots, timestamps, t0),
                interpolate_pose_at(knots, timestamps, t0 + 1.0e-3),
            )[3:]
            / 1.0e-3
        )
        samples.append(
            TrajectoryImuGyroSample(
                timestamp_sec=float(time) + true_clock,
                omega_body_rad_s=tuple(float(value) for value in omega),
            )
        )
    measurement = TrajectoryImuPreintegrationMeasurement(
        measurement_id="imu0",
        gyro_samples=tuple(samples),
        weight=25.0,
    )
    residual, _knots, _bias, jacobian_clock, _scale = _imu_preintegration_residual(
        list(knots), timestamps, measurement, bias, clock
    )
    epsilon = 1.0e-6
    perturbed, _, _, _, _ = _imu_preintegration_residual(
        list(knots), timestamps, measurement, bias, clock + epsilon
    )
    numeric = ((perturbed - residual) / epsilon).reshape(3, 1)
    assert float(np.linalg.norm(jacobian_clock)) > 0.05
    assert np.allclose(jacobian_clock, numeric, atol=1.0e-1, rtol=8.0e-2)


def test_imu_clock_offset_recovers_delayed_gyro_stamps() -> None:
    timestamps, truth = _truth_knots()
    true_clock = 0.015
    windows = [(0.3, 0.9), (1.1, 1.7), (2.0, 2.6)]
    measurements = []
    dt = 0.02
    for index, (start, end) in enumerate(windows):
        samples = []
        for time in np.arange(start, end + 0.5 * dt, dt):
            t0 = min(max(float(time), timestamps[0]), timestamps[-1] - 1.0e-3)
            omega = (
                se3_log(
                    interpolate_pose_at(truth, timestamps, t0),
                    interpolate_pose_at(truth, timestamps, t0 + 1.0e-3),
                )[3:]
                / 1.0e-3
            )
            samples.append(
                TrajectoryImuGyroSample(
                    timestamp_sec=float(time) + true_clock,
                    omega_body_rad_s=tuple(float(value) for value in omega),
                )
            )
        measurements.append(
            TrajectoryImuPreintegrationMeasurement(
                measurement_id=f"imu{index}",
                gyro_samples=tuple(samples),
                weight=25.0,
            )
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
            imu_preintegration_measurements=tuple(measurements),
            initial_imu_clock_offset_sec=0.0,
            estimate_imu_clock_offset=True,
        ),
        ContinuousTimeTrajectoryFitOptions(max_iterations=30),
    )
    assert result.status in {"converged", "max_iterations"}
    assert result.imu_clock_offset_sec is not None
    assert abs(result.imu_clock_offset_sec - true_clock) < 0.005


def test_synthetic_clock_offset_recovery_holdout_and_known_bad_control(
    tmp_path: Path,
) -> None:
    artifact = run_synthetic_imu_clock_offset_recovery(seed=20260820)
    output = tmp_path / "recovery.yaml"
    artifact.save(output)
    assert validate_file(output, kind="continuous-time-imu-clock-offset").valid
    assert artifact.policy_status == "pass"
    assert artifact.holdout_measurement_count >= 1
    assert artifact.known_bad_rmse_delta_rad > 0.01
    assert artifact.provenance.seed == 20260820


def test_cli_recover_imu_clock_offset(tmp_path: Path) -> None:
    output = tmp_path / "recovery.yaml"
    exit_code = main(
        [
            "trajectory",
            "recover-imu-clock-offset",
            "--output",
            str(output),
            "--seed",
            "20260820",
            "--json",
        ]
    )
    assert exit_code == 0
    assert validate_file(output).valid
