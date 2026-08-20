"""Tests for continuous-time IMU diagonal scale intrinsics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from calibrex.cli.main import main
from calibrex.core.continuous_time_sparse import (
    ContinuousTimeTrajectoryFitOptions,
    ContinuousTimeTrajectoryFitProblem,
    TrajectoryImuGyroSample,
    TrajectoryImuLeverArmMeasurement,
    TrajectoryImuPreintegrationMeasurement,
    TrajectoryPoseMeasurement,
    _imu_preintegration_residual,
    _lever_arm_residual,
    fit_continuous_trajectory,
    predicted_imu_specific_force,
)
from calibrex.core.geometry import SE3
from calibrex.core.se3_manifold import se3_exp, se3_log
from calibrex.core.validation import validate_file
from calibrex.evaluation.continuous_time_imu_intrinsics import (
    run_synthetic_imu_intrinsics_recovery,
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


def _gyro_window(
    knots: tuple[SE3, ...],
    timestamps: tuple[float, ...],
    start: float,
    end: float,
    gyro_scale: tuple[float, float, float],
) -> TrajectoryImuPreintegrationMeasurement:
    dt = 0.02
    samples: list[TrajectoryImuGyroSample] = []
    scale = np.asarray(gyro_scale, dtype=float)
    for time in np.arange(start, end + 0.5 * dt, dt):
        t0 = min(max(float(time), timestamps[0]), timestamps[-1] - 1.0e-3)
        omega = (
            se3_log(
                knots[min(int(t0), len(knots) - 1)],
                knots[min(int(t0) + 1, len(knots) - 1)],
            )[3:]
            / 1.0
        )
        measured = omega / scale
        samples.append(
            TrajectoryImuGyroSample(
                timestamp_sec=float(time),
                omega_body_rad_s=tuple(float(value) for value in measured),
            )
        )
    return TrajectoryImuPreintegrationMeasurement(
        measurement_id="gyro0",
        gyro_samples=tuple(samples),
        weight=25.0,
    )


def test_gyro_scale_jacobian_matches_finite_differences() -> None:
    timestamps, knots = _truth_knots()
    bias = np.zeros(3, dtype=float)
    scale = np.array([1.05, 0.98, 1.03], dtype=float)
    measurement = _gyro_window(knots, timestamps, 0.4, 1.4, tuple(scale))
    residual, _knots, _bias, _clock, jacobian_scale = _imu_preintegration_residual(
        list(knots), timestamps, measurement, bias, 0.0, scale
    )
    epsilon = 1.0e-6
    numeric = np.zeros((3, 3))
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = epsilon
        perturbed, _, _, _, _ = _imu_preintegration_residual(
            list(knots), timestamps, measurement, bias, 0.0, scale + step
        )
        numeric[:, axis] = (perturbed - residual) / epsilon
    assert np.allclose(jacobian_scale, numeric, atol=1.0e-1, rtol=8.0e-2)


def test_accel_scale_jacobian_matches_finite_differences() -> None:
    timestamps, knots = _truth_knots()
    lever = np.array([0.1, -0.05, 0.2], dtype=float)
    scale = np.array([1.02, 0.97, 1.04], dtype=float)
    measurement = TrajectoryImuLeverArmMeasurement(
        measurement_id="accel0",
        timestamp_sec=1.6,
        accel_body_m_s2=(0.3, -0.1, 9.5),
    )
    residual, _lever, _clock, _bias, _gravity, jacobian_scale = _lever_arm_residual(
        list(knots),
        timestamps,
        measurement,
        lever,
        0.0,
        np.zeros(3),
        np.array([0.0, 0.0, -9.81]),
        scale,
    )
    epsilon = 1.0e-6
    numeric = np.zeros((3, 3))
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = epsilon
        perturbed, *_ = _lever_arm_residual(
            list(knots),
            timestamps,
            measurement,
            lever,
            0.0,
            np.zeros(3),
            np.array([0.0, 0.0, -9.81]),
            scale + step,
        )
        numeric[:, axis] = (perturbed - residual) / epsilon
    assert np.allclose(jacobian_scale, numeric, atol=1.0e-6, rtol=1.0e-5)


def test_gyro_and_accel_scale_recover_from_measurements() -> None:
    timestamps, truth = _truth_knots(count=6)
    true_gyro_scale = (1.05, 0.98, 1.03)
    true_accel_scale = (1.02, 0.97, 1.04)
    true_lever = (0.12, -0.08, 0.22)
    true_gravity = (0.0, 0.0, -9.81)
    gyro = tuple(
        _gyro_window(truth, timestamps, start, start + 0.4, true_gyro_scale)
        for start in (0.3, 0.9, 1.5, 2.1)
    )
    accel = tuple(
        TrajectoryImuLeverArmMeasurement(
            measurement_id=f"accel{index}",
            timestamp_sec=time,
            accel_body_m_s2=tuple(
                float(value)
                for value in np.asarray(true_accel_scale, dtype=float)
                * predicted_imu_specific_force(
                    truth,
                    timestamps,
                    time,
                    true_lever,
                    gravity_world_m_s2=true_gravity,
                )
            ),
        )
        for index, time in enumerate((0.4, 0.9, 1.4, 1.9, 2.4, 2.9))
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
            imu_preintegration_measurements=gyro,
            imu_lever_arm_measurements=accel,
            initial_lever_arm_body_m=true_lever,
            initial_gravity_world_m_s2=true_gravity,
            estimate_gyro_scale=True,
            estimate_accel_scale=True,
        ),
        ContinuousTimeTrajectoryFitOptions(max_iterations=30),
    )
    assert result.status in {"converged", "max_iterations"}
    assert result.gyro_scale is not None
    assert result.accel_scale is not None
    assert (
        np.linalg.norm(
            np.asarray(result.gyro_scale, dtype=float)
            - np.asarray(true_gyro_scale, dtype=float)
        )
        < 0.05
    )
    assert (
        np.linalg.norm(
            np.asarray(result.accel_scale, dtype=float)
            - np.asarray(true_accel_scale, dtype=float)
        )
        < 0.05
    )


def test_synthetic_imu_intrinsics_recovery_holdout_and_known_bad_control(
    tmp_path: Path,
) -> None:
    artifact = run_synthetic_imu_intrinsics_recovery(seed=20260820)
    output = tmp_path / "recovery.yaml"
    artifact.save(output)
    assert validate_file(output, kind="continuous-time-imu-intrinsics").valid
    assert artifact.policy_status == "pass"
    assert artifact.holdout_gyro_measurement_count >= 1
    assert artifact.holdout_accel_measurement_count >= 1
    assert artifact.known_bad_imu_rmse_delta_rad > 0.008
    assert artifact.provenance.seed == 20260820


def test_cli_recover_imu_intrinsics(tmp_path: Path) -> None:
    output = tmp_path / "recovery.yaml"
    exit_code = main(
        [
            "trajectory",
            "recover-imu-intrinsics",
            "--output",
            str(output),
            "--seed",
            "20260820",
            "--json",
        ]
    )
    assert exit_code == 0
    assert validate_file(output).valid
