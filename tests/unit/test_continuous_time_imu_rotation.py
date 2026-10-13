"""Tests for the continuous-time IMU mounting rotation ``R_body_imu``."""

from __future__ import annotations

import numpy as np

from calibrex.core.continuous_time_sparse import (
    ContinuousTimeTrajectoryFitOptions,
    ContinuousTimeTrajectoryFitProblem,
    TrajectoryImuGyroSample,
    TrajectoryImuLeverArmMeasurement,
    TrajectoryImuPreintegrationMeasurement,
    TrajectoryPoseMeasurement,
    _imu_preintegration_residual,
    _imu_preintegration_rotation_jacobian,
    _lever_arm_residual,
    _lever_arm_rotation_jacobian,
    fit_continuous_trajectory,
    imu_preintegration_rmse,
    interpolate_pose_at,
    predicted_imu_specific_force,
)
from calibrex.core.geometry import SE3
from calibrex.core.se3_manifold import (
    rotation_matrix_to_rotation_vector,
    rotation_vector_to_matrix,
    se3_exp,
    se3_log,
)

TRUE_ROTATION = np.array([0.03, -0.05, 0.04], dtype=float)


def _truth_knots(count: int = 6) -> tuple[tuple[float, ...], tuple[SE3, ...]]:
    timestamps = tuple(float(index) for index in range(count))
    knots = tuple(
        se3_exp(
            SE3.identity(),
            np.concatenate(
                [
                    np.array([0.15 * index * index, 0.05 * index, 0.02 * index]),
                    np.array(
                        [
                            0.5 * np.sin(1.6 * index),
                            0.4 * np.sin(2.0 * index + 0.3),
                            0.4 * index,
                        ]
                    ),
                ]
            ),
        )
        for index in timestamps
    )
    return timestamps, knots


def _body_rate(
    knots: tuple[SE3, ...], timestamps: tuple[float, ...], time: float
) -> np.ndarray:
    t0 = min(max(time, timestamps[0]), timestamps[-1] - 1.0e-4)
    return np.asarray(
        se3_log(
            interpolate_pose_at(knots, timestamps, t0),
            interpolate_pose_at(knots, timestamps, t0 + 1.0e-4),
        )[3:]
        / 1.0e-4,
        dtype=float,
    )


def _gyro_window(
    knots: tuple[SE3, ...],
    timestamps: tuple[float, ...],
    start: float,
    end: float,
    *,
    rotation_vector: np.ndarray,
    bias: np.ndarray,
    measurement_id: str,
    dt: float = 0.005,
) -> TrajectoryImuPreintegrationMeasurement:
    imu_rotation = rotation_vector_to_matrix(rotation_vector)
    samples = []
    for time in np.arange(start, end + 0.5 * dt, dt):
        omega_imu = imu_rotation.T @ _body_rate(knots, timestamps, float(time)) + bias
        samples.append(
            TrajectoryImuGyroSample(
                timestamp_sec=float(time),
                omega_body_rad_s=(
                    float(omega_imu[0]),
                    float(omega_imu[1]),
                    float(omega_imu[2]),
                ),
            )
        )
    return TrajectoryImuPreintegrationMeasurement(
        measurement_id=measurement_id, gyro_samples=tuple(samples)
    )


def _numeric_rotation_jacobian(residual_at: object, rotation_vector: np.ndarray) -> np.ndarray:
    assert callable(residual_at)
    epsilon = 1.0e-6
    base = rotation_vector_to_matrix(rotation_vector)
    reference = residual_at(base)
    numeric = np.zeros((3, 3))
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = epsilon
        perturbed = residual_at(base @ rotation_vector_to_matrix(step))
        numeric[:, axis] = (perturbed - reference) / epsilon
    return numeric


def test_preintegration_rotation_jacobian_matches_finite_differences() -> None:
    timestamps, knots = _truth_knots()
    measurement = _gyro_window(
        knots,
        timestamps,
        1.1,
        1.9,
        rotation_vector=np.array([0.2, -0.1, 0.3]),
        bias=np.zeros(3),
        measurement_id="imu0",
        dt=0.02,
    )
    bias = np.array([0.01, -0.02, 0.015])
    scale = np.ones(3)
    rotation_vector = np.array([0.05, 0.02, -0.04])

    def residual_at(rotation: np.ndarray) -> np.ndarray:
        return _imu_preintegration_residual(
            list(knots), timestamps, measurement, bias, 0.0, scale, imu_rotation=rotation
        )[0]

    analytic = _imu_preintegration_rotation_jacobian(
        list(knots),
        timestamps,
        measurement,
        bias,
        0.0,
        scale,
        rotation_vector_to_matrix(rotation_vector),
    )
    numeric = _numeric_rotation_jacobian(residual_at, rotation_vector)
    assert np.allclose(analytic, numeric, atol=1.0e-5, rtol=1.0e-4)


def test_lever_arm_rotation_jacobian_matches_finite_differences() -> None:
    timestamps, knots = _truth_knots()
    measurement = TrajectoryImuLeverArmMeasurement(
        measurement_id="accel0", timestamp_sec=2.3, accel_body_m_s2=(0.3, -0.1, 9.5)
    )
    lever = np.array([0.1, -0.05, 0.2])
    gravity = np.array([0.1, -0.2, -9.7])
    scale = np.array([1.01, 0.99, 1.02])
    rotation_vector = np.array([0.05, 0.02, -0.04])

    def residual_at(rotation: np.ndarray) -> np.ndarray:
        return _lever_arm_residual(
            list(knots),
            timestamps,
            measurement,
            lever,
            0.0,
            np.zeros(3),
            gravity,
            scale,
            imu_rotation=rotation,
        )[0]

    analytic = _lever_arm_rotation_jacobian(
        list(knots),
        timestamps,
        measurement,
        lever,
        0.0,
        gravity,
        scale,
        rotation_vector_to_matrix(rotation_vector),
    )
    numeric = _numeric_rotation_jacobian(residual_at, rotation_vector)
    assert np.allclose(analytic, numeric, atol=1.0e-5, rtol=1.0e-4)


def test_identity_rotation_matches_body_frame_imu() -> None:
    timestamps, knots = _truth_knots()
    gyro = _gyro_window(
        knots,
        timestamps,
        0.5,
        1.5,
        rotation_vector=np.zeros(3),
        bias=np.zeros(3),
        measurement_id="imu0",
        dt=0.02,
    )
    accel = TrajectoryImuLeverArmMeasurement(
        measurement_id="accel0", timestamp_sec=1.2, accel_body_m_s2=(0.3, -0.1, 9.5)
    )
    bias = np.array([0.01, -0.02, 0.015])
    body = _imu_preintegration_residual(list(knots), timestamps, gyro, bias)[0]
    identity = _imu_preintegration_residual(
        list(knots), timestamps, gyro, bias, imu_rotation=np.eye(3)
    )[0]
    assert np.allclose(body, identity, atol=1.0e-15)
    lever = np.array([0.1, -0.05, 0.2])
    gravity = np.array([0.0, 0.0, -9.81])
    body_accel = _lever_arm_residual(
        list(knots), timestamps, accel, lever, 0.0, np.zeros(3), gravity
    )[0]
    identity_accel = _lever_arm_residual(
        list(knots),
        timestamps,
        accel,
        lever,
        0.0,
        np.zeros(3),
        gravity,
        imu_rotation=np.eye(3),
    )[0]
    assert np.allclose(body_accel, identity_accel, atol=1.0e-12)


def test_joint_fit_recovers_imu_rotation_with_biases_lever_arm_and_gravity() -> None:
    timestamps, truth = _truth_knots()
    true_gyro_bias = np.array([0.01, -0.015, 0.02])
    true_accel_bias = (0.05, -0.04, 0.03)
    true_lever = (0.12, -0.08, 0.22)
    true_gravity = (0.0, 0.0, -9.81)
    windows = [(start + 0.05, start + 0.95) for start in timestamps[:-1]]
    gyro = tuple(
        _gyro_window(
            truth,
            timestamps,
            start,
            end,
            rotation_vector=TRUE_ROTATION,
            bias=true_gyro_bias,
            measurement_id=f"imu{index}",
        )
        for index, (start, end) in enumerate(windows)
    )
    accel_times = np.arange(0.3, timestamps[-1] - 0.25, 0.25)
    accel = tuple(
        TrajectoryImuLeverArmMeasurement(
            measurement_id=f"accel{index}",
            timestamp_sec=float(time),
            accel_body_m_s2=tuple(
                float(value)
                for value in predicted_imu_specific_force(
                    truth,
                    timestamps,
                    float(time),
                    true_lever,
                    accel_bias_body_m_s2=true_accel_bias,
                    gravity_world_m_s2=true_gravity,
                    imu_rotation_vector_rad=TRUE_ROTATION,
                )
            ),
        )
        for index, time in enumerate(accel_times)
    )
    gauges = tuple(
        TrajectoryPoseMeasurement(
            measurement_id=f"gauge{index}",
            timestamp_sec=timestamp,
            pose_world_body=pose,
            weight=50.0,
        )
        for index, (timestamp, pose) in enumerate(zip(timestamps, truth, strict=True))
    )
    problem = ContinuousTimeTrajectoryFitProblem(
        knot_timestamps=timestamps,
        initial_knot_poses=truth,
        pose_measurements=gauges,
        imu_preintegration_measurements=gyro,
        imu_lever_arm_measurements=accel,
        initial_gravity_world_m_s2=(0.0, 0.0, -9.7),
        estimate_imu_rotation=True,
        estimate_gyro_bias=True,
        estimate_lever_arm=True,
        estimate_accel_bias=True,
        estimate_gravity=True,
    )
    result = fit_continuous_trajectory(
        problem, ContinuousTimeTrajectoryFitOptions(max_iterations=60)
    )

    assert result.imu_rotation_vector_rad is not None
    error = rotation_matrix_to_rotation_vector(
        rotation_vector_to_matrix(TRUE_ROTATION).T
        @ rotation_vector_to_matrix(np.asarray(result.imu_rotation_vector_rad))
    )
    assert np.degrees(np.linalg.norm(error)) < 0.05
    assert result.gyro_bias_rad_s is not None
    assert np.linalg.norm(np.asarray(result.gyro_bias_rad_s) - true_gyro_bias) < 2.0e-3
    assert result.accel_bias_body_m_s2 is not None
    assert np.linalg.norm(np.asarray(result.accel_bias_body_m_s2) - true_accel_bias) < 0.02
    assert result.lever_arm_body_m is not None
    assert np.linalg.norm(np.asarray(result.lever_arm_body_m) - true_lever) < 0.02
    assert result.gravity_world_m_s2 is not None
    assert np.linalg.norm(np.asarray(result.gravity_world_m_s2) - true_gravity) < 0.02

    # The IMU-frame gyro is inconsistent with a body-frame model: ignoring the
    # mounting rotation leaves a clearly larger pre-integration residual.
    with_rotation = imu_preintegration_rmse(
        gyro,
        timestamps,
        truth,
        true_gyro_bias,
        imu_rotation_vector_rad=TRUE_ROTATION,
    )
    without_rotation = imu_preintegration_rmse(gyro, timestamps, truth, true_gyro_bias)
    assert with_rotation is not None and without_rotation is not None
    assert without_rotation > 10.0 * with_rotation


def test_imu_rotation_estimation_requires_imu_measurements() -> None:
    timestamps, truth = _truth_knots(3)
    try:
        ContinuousTimeTrajectoryFitProblem(
            knot_timestamps=timestamps,
            initial_knot_poses=truth,
            pose_measurements=(
                TrajectoryPoseMeasurement(
                    measurement_id="gauge",
                    timestamp_sec=0.0,
                    pose_world_body=truth[0],
                ),
            ),
            estimate_imu_rotation=True,
        )
    except ValueError as error:
        assert "IMU rotation" in str(error)
    else:
        raise AssertionError("expected a ValueError")
