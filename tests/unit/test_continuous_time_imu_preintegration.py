"""Tests for continuous-time IMU gyro pre-integration factors."""

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
from calibrex.evaluation.continuous_time_imu_preintegration import (
    run_synthetic_imu_preintegration_recovery,
)

rng = np.random.default_rng(21)


def _samples_from_rate(
    start: float,
    end: float,
    omega: tuple[float, float, float],
    *,
    bias: tuple[float, float, float] = (0.0, 0.0, 0.0),
    dt: float = 0.02,
) -> tuple[TrajectoryImuGyroSample, ...]:
    times = np.arange(start, end + 0.5 * dt, dt)
    measured = np.asarray(omega, dtype=float) + np.asarray(bias, dtype=float)
    return tuple(
        TrajectoryImuGyroSample(
            timestamp_sec=float(time),
            omega_body_rad_s=(
                float(measured[0]),
                float(measured[1]),
                float(measured[2]),
            ),
        )
        for time in times
    )


def test_imu_preintegration_jacobian_matches_finite_differences() -> None:
    timestamps = (0.0, 1.0, 2.0)
    knots = [
        se3_exp(SE3.identity(), np.array([0.1, 0.0, 0.0, 0.05, -0.02, 0.1])),
        se3_exp(SE3.identity(), np.array([0.3, 0.05, 0.0, 0.04, 0.01, 0.35])),
        se3_exp(SE3.identity(), np.array([0.5, 0.1, 0.0, 0.02, 0.03, 0.7])),
    ]
    bias = np.array([0.01, -0.02, 0.015], dtype=float)
    measurement = TrajectoryImuPreintegrationMeasurement(
        measurement_id="imu0",
        gyro_samples=_samples_from_rate(0.4, 1.4, (0.0, 0.05, 0.3), bias=(0.01, -0.02, 0.015)),
    )
    residual, knot_blocks, jacobian_bias, _clock, _scale = _imu_preintegration_residual(
        knots, timestamps, measurement, bias
    )
    jacobian_by_knot = dict(knot_blocks)
    epsilon = 1.0e-6
    for knot_index, analytic in jacobian_by_knot.items():
        numeric = np.zeros((3, 6))
        for axis in range(6):
            step = np.zeros(6)
            step[axis] = epsilon
            perturbed = list(knots)
            perturbed[knot_index] = se3_exp(knots[knot_index], step)
            perturbed_residual, _, _, _, _ = _imu_preintegration_residual(
                perturbed, timestamps, measurement, bias
            )
            numeric[:, axis] = (perturbed_residual - residual) / epsilon
        assert np.allclose(analytic, numeric, atol=2.0e-2, rtol=5.0e-2)
    numeric_bias = np.zeros((3, 3))
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = epsilon
        perturbed_residual, _, _, _, _ = _imu_preintegration_residual(
            knots, timestamps, measurement, bias + step
        )
        numeric_bias[:, axis] = (perturbed_residual - residual) / epsilon
    assert np.allclose(jacobian_bias, numeric_bias, atol=2.0e-2, rtol=5.0e-2)


def test_imu_preintegration_recovers_rotation_and_gyro_bias() -> None:
    timestamps = tuple(float(index) for index in range(4))
    truth = tuple(
        se3_exp(
            SE3.identity(),
            np.concatenate(
                [
                    np.array([0.2 * index, 0.0, 0.0]),
                    np.array([0.0, 0.04 * index, 0.2 * index]),
                ]
            ),
        )
        for index in range(4)
    )
    true_bias = (0.01, -0.015, 0.02)
    windows = [(0.1, 0.9), (1.1, 1.9), (2.1, 2.9)]
    measurements = []
    for index, (start, end) in enumerate(windows):
        dt = 0.02
        samples = []
        for time in np.arange(start, end + 0.5 * dt, dt):
            t0 = min(max(float(time), timestamps[0]), timestamps[-1] - 1.0e-3)
            omega = se3_log(
                interpolate_pose_at(truth, timestamps, t0),
                interpolate_pose_at(truth, timestamps, t0 + 1.0e-3),
            )[3:] / 1.0e-3
            measured = omega + np.asarray(true_bias, dtype=float)
            samples.append(
                TrajectoryImuGyroSample(
                    timestamp_sec=float(time),
                    omega_body_rad_s=(
                        float(measured[0]),
                        float(measured[1]),
                        float(measured[2]),
                    ),
                )
            )
        measurements.append(
            TrajectoryImuPreintegrationMeasurement(
                measurement_id=f"imu{index}",
                gyro_samples=tuple(samples),
            )
        )
    initial = tuple(
        se3_exp(knot, np.concatenate([np.zeros(3), rng.normal(0.0, 0.03, 3)]))
        for knot in truth
    )
    result = fit_continuous_trajectory(
        ContinuousTimeTrajectoryFitProblem(
            knot_timestamps=timestamps,
            initial_knot_poses=initial,
            pose_measurements=(
                TrajectoryPoseMeasurement(
                    measurement_id="gauge",
                    timestamp_sec=0.0,
                    pose_world_body=truth[0],
                    weight=25.0,
                ),
            ),
            imu_preintegration_measurements=tuple(measurements),
            initial_gyro_bias_rad_s=(0.0, 0.0, 0.0),
            estimate_gyro_bias=True,
        ),
        ContinuousTimeTrajectoryFitOptions(max_iterations=80),
    )
    assert result.status in {"converged", "max_iterations"}
    assert result.gyro_bias_rad_s is not None
    rotation_errors = [
        float(np.linalg.norm(se3_log(estimate, knot)[3:]))
        for estimate, knot in zip(result.knot_poses, truth, strict=True)
    ]
    assert max(rotation_errors) < 0.05
    assert (
        np.linalg.norm(
            np.asarray(result.gyro_bias_rad_s, dtype=float)
            - np.asarray(true_bias, dtype=float)
        )
        < 0.02
    )


def test_synthetic_imu_recovery_holdout_and_known_bad_control(tmp_path: Path) -> None:
    artifact = run_synthetic_imu_preintegration_recovery(seed=20260820)
    output = tmp_path / "recovery.yaml"
    artifact.save(output)
    assert validate_file(output, kind="continuous-time-imu-preintegration").valid
    assert artifact.policy_status == "pass"
    assert artifact.holdout_measurement_count >= 1
    assert artifact.known_bad_rmse_delta_rad > 0.02
    assert artifact.provenance.seed == 20260820


def test_cli_recover_imu_preintegration(tmp_path: Path) -> None:
    output = tmp_path / "recovery.yaml"
    exit_code = main(
        [
            "trajectory",
            "recover-imu-preintegration",
            "--output",
            str(output),
            "--seed",
            "20260820",
            "--json",
        ]
    )
    assert exit_code == 0
    assert validate_file(output).valid
