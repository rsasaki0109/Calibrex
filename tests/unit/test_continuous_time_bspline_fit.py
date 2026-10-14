"""Tests for the joint LiDAR-IMU fit on a cubic B-spline trajectory."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from calibrex.core.continuous_time_bspline import CubicBSplineTrajectory
from calibrex.core.continuous_time_bspline_fit import (
    BSplineImuSamples,
    BSplineLidarImuFitResult,
    BSplineLidarImuProblem,
    BSplinePlaneReturns,
    LidarImuCalibration,
    ResidualFamily,
    fit_bspline_lidar_imu,
    lidar_imu_residuals,
)
from calibrex.core.se3_manifold import (
    rotation_matrix_to_rotation_vector,
    rotation_vector_to_matrix,
)

TRUTH = LidarImuCalibration(
    imu_rotation_vector_rad=np.array([0.04, -0.03, 0.05]),
    gyro_bias_rad_s=np.array([0.008, -0.006, 0.01]),
    lever_arm_body_m=np.array([0.10, -0.06, 0.15]),
    imu_clock_offset_sec=0.004,
    accel_bias_m_s2=np.array([0.06, -0.05, 0.04]),
    gravity_world_m_s2=np.array([0.0, 0.0, -9.81]),
)


def _truth_spline() -> CubicBSplineTrajectory:
    rng = np.random.default_rng(11)
    count = 23
    rotations = [np.eye(3)]
    for _ in range(count - 1):
        rotations.append(rotations[-1] @ rotation_vector_to_matrix(rng.normal(0.0, 0.12, 3)))
    positions = np.cumsum(rng.normal(0.0, 0.05, (count, 3)), axis=0)
    return CubicBSplineTrajectory(
        start_time_sec=0.0,
        knot_spacing_sec=0.1,
        rotations=np.asarray(rotations),
        positions=positions,
    )


def _measurements(
    spline: CubicBSplineTrajectory, rng: np.random.Generator
) -> tuple[BSplinePlaneReturns, BSplineImuSamples, BSplineImuSamples]:
    normals = np.array(
        [[0, 0, 1], [1, 0, 0], [0, 1, 0], [0.7, 0.7, 0], [-0.6, 0.6, 0.5], [0, -0.8, 0.6]],
        dtype=float,
    )
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    offsets = np.array([1.5, 4.0, 5.0, 3.0, 2.5, 3.5])
    times = np.sort(rng.uniform(0.0, spline.end_time_sec, 3000))
    choice = rng.integers(len(normals), size=times.size)
    states = spline.evaluate(times)
    # A point on plane n.x = -d near the trajectory, expressed in the body frame.
    tangent = rng.normal(0.0, 2.0, (times.size, 3))
    tangent -= np.sum(tangent * normals[choice], axis=1)[:, None] * normals[choice]
    anchors = -offsets[choice, None] * normals[choice]
    world = anchors + tangent
    body = np.einsum("tji,tj->ti", states.rotation_world_body, world - states.position_world_m)
    planes = BSplinePlaneReturns(times, body, anchors, normals[choice], sigma_m=0.01)
    body_times = np.arange(0.05, spline.end_time_sec - 0.05, 0.01)
    imu = spline.evaluate(body_times)
    rotation = rotation_vector_to_matrix(TRUTH.imu_rotation_vector_rad)
    omega = imu.angular_velocity_body_rad_s
    lever = np.broadcast_to(TRUTH.lever_arm_body_m, omega.shape)
    kinematic = (
        np.einsum(
            "tji,tj->ti",
            imu.rotation_world_body,
            imu.linear_acceleration_world_m_s2 - TRUTH.gravity_world_m_s2,
        )
        + np.cross(imu.angular_acceleration_body_rad_s2, lever)
        + np.cross(omega, np.cross(omega, lever))
    )
    stamps = body_times + TRUTH.imu_clock_offset_sec
    gyro = BSplineImuSamples(stamps, omega @ rotation + TRUTH.gyro_bias_rad_s, 0.003)
    accel = BSplineImuSamples(stamps, kinematic @ rotation + TRUTH.accel_bias_m_s2, 0.03)
    return planes, gyro, accel


def test_residuals_vanish_on_noise_free_spline_data() -> None:
    spline = _truth_spline()
    planes, gyro, accel = _measurements(spline, np.random.default_rng(5))
    residuals = lidar_imu_residuals(spline, TRUTH, planes, gyro, accel)
    for family, values in residuals.items():
        assert np.max(np.abs(values)) < 1.0e-8, family


@pytest.mark.parametrize("families", [("plane",), ("gyro",), ("accel",),
                                      ("accel", "gyro"), ()])
def test_selected_residuals_keep_exact_values_and_order(
    families: tuple[ResidualFamily, ...],
) -> None:
    spline = _truth_spline()
    planes, gyro, accel = _measurements(spline, np.random.default_rng(18))
    changed = replace(TRUTH, gyro_bias_rad_s=TRUTH.gyro_bias_rad_s + .07,
                      accel_bias_m_s2=TRUTH.accel_bias_m_s2 - .15,
                      imu_clock_offset_sec=TRUTH.imu_clock_offset_sec + .001)
    expected = lidar_imu_residuals(spline, changed, planes, gyro, accel)
    actual = lidar_imu_residuals(spline, changed, planes, gyro, accel, families=families)
    assert tuple(actual) == families
    for name in families:
        np.testing.assert_array_equal(actual[name].view(np.uint64),
                                      expected[name].view(np.uint64))


def test_selected_imu_residuals_do_not_evaluate_unrelated_plane_times() -> None:
    spline = _truth_spline()
    planes, gyro, accel = _measurements(spline, np.random.default_rng(19))
    outside = replace(planes, times_sec=planes.times_sec + spline.end_time_sec + 1)
    expected = lidar_imu_residuals(spline, TRUTH, planes, gyro, accel)
    actual = lidar_imu_residuals(spline, TRUTH, outside, gyro, accel,
                               families=("gyro", "accel"))
    for name in ("gyro", "accel"):
        np.testing.assert_array_equal(actual[name], expected[name])
    assert lidar_imu_residuals(spline, TRUTH, outside, gyro, accel, families=()) == {}
    with pytest.raises(ValueError, match="outside"):
        lidar_imu_residuals(spline, TRUTH, outside, gyro, accel)
    with pytest.raises(ValueError, match="residual families"):
        lidar_imu_residuals(spline, TRUTH, planes, gyro, accel,
                           families=("unknown",))  # type: ignore[arg-type]


def test_fit_recovers_calibration_and_reports_marginal_std() -> None:
    spline = _truth_spline()
    rng = np.random.default_rng(6)
    planes, gyro, accel = _measurements(spline, rng)
    initial = CubicBSplineTrajectory(
        start_time_sec=spline.start_time_sec,
        knot_spacing_sec=spline.knot_spacing_sec,
        rotations=np.asarray(
            [
                rotation @ rotation_vector_to_matrix(rng.normal(0.0, 0.01, 3))
                for rotation in spline.rotations
            ]
        ),
        positions=spline.positions + rng.normal(0.0, 0.01, spline.positions.shape),
    )
    zero = np.zeros(3)
    result = fit_bspline_lidar_imu(
        BSplineLidarImuProblem(
            initial_trajectory=initial,
            planes=planes,
            gyro=gyro,
            accel=accel,
            initial_calibration=LidarImuCalibration(
                imu_rotation_vector_rad=zero,
                gyro_bias_rad_s=zero,
                lever_arm_body_m=zero,
                imu_clock_offset_sec=0.0,
                accel_bias_m_s2=zero,
                gravity_world_m_s2=np.array([0.0, 0.0, -9.7]),
            ),
        )
    )

    assert result.status == "converged"
    assert result.final_cost < 1.0e-6 * result.initial_cost
    calibration = result.calibration
    rotation_error = rotation_matrix_to_rotation_vector(
        rotation_vector_to_matrix(TRUTH.imu_rotation_vector_rad).T
        @ rotation_vector_to_matrix(calibration.imu_rotation_vector_rad)
    )
    assert np.linalg.norm(rotation_error) < 1.0e-6
    assert calibration.imu_clock_offset_sec == pytest.approx(TRUTH.imu_clock_offset_sec, abs=1.0e-7)
    for name in ("gyro_bias_rad_s", "lever_arm_body_m", "accel_bias_m_s2", "gravity_world_m_s2"):
        assert np.allclose(getattr(calibration, name), getattr(TRUTH, name), atol=1.0e-5), name
    assert set(result.calibration_parameter_names) >= {"imu_rotation[0]", "imu_clock_offset[0]"}
    assert np.all(result.block_std("lever_arm") > 0.0)
    assert ("accel_bias", "gravity") in result.block_correlations()


def test_measurement_containers_validate_shapes() -> None:
    with pytest.raises(ValueError, match="shape"):
        BSplineImuSamples(np.zeros(3), np.zeros((2, 3)), 0.1)
    with pytest.raises(ValueError, match="sigma"):
        BSplineImuSamples(np.zeros(2), np.zeros((2, 3)), 0.0)


def test_linearized_jacobian_matches_finite_differences_of_every_parameter() -> None:
    from calibrex.core.continuous_time_bspline_fit import (
        CALIBRATION_BLOCKS,
        _families,
        _linearize,
    )

    spline = _truth_spline()
    rng = np.random.default_rng(9)
    planes, gyro, accel = _measurements(spline, rng)
    calibration = replace(TRUTH, lever_arm_body_m=np.array([0.2, 0.1, -0.1]))
    families = _families(planes, gyro, accel)
    jacobian, base = _linearize(spline, calibration, families, CALIBRATION_BLOCKS)
    dense = jacobian.toarray()

    def stacked(trajectory: CubicBSplineTrajectory, values: LidarImuCalibration) -> np.ndarray:
        residuals = lidar_imu_residuals(trajectory, values, planes, gyro, accel)
        return np.concatenate([residuals[name].reshape(-1) for name in ("plane", "gyro", "accel")])

    assert np.allclose(stacked(spline, calibration), base)
    epsilon = 1.0e-6
    for control in (3, 10):
        for axis in range(3):
            step = np.zeros_like(spline.positions)
            step[control, axis] = epsilon
            numeric = (
                stacked(replace(spline, positions=spline.positions + step), calibration)
                - stacked(replace(spline, positions=spline.positions - step), calibration)
            ) / (2.0 * epsilon)
            assert np.allclose(dense[:, 6 * control + axis], numeric, atol=1.0e-4, rtol=1.0e-5)
            plus, minus = spline.rotations.copy(), spline.rotations.copy()
            tangent = np.zeros(3)
            tangent[axis] = epsilon
            plus[control] = plus[control] @ rotation_vector_to_matrix(tangent)
            minus[control] = minus[control] @ rotation_vector_to_matrix(-tangent)
            numeric = (
                stacked(replace(spline, rotations=plus), calibration)
                - stacked(replace(spline, rotations=minus), calibration)
            ) / (2 * epsilon)
            assert np.allclose(dense[:, 6 * control + 3 + axis], numeric, atol=1e-4, rtol=1e-5)
    column = 6 * spline.control_point_count
    for block in CALIBRATION_BLOCKS:
        width = 1 if block == "imu_clock_offset" else 3
        for axis in range(width):
            step = np.zeros(width)
            step[axis] = epsilon
            numeric = (
                stacked(spline, calibration.stepped(block, step))
                - stacked(spline, calibration.stepped(block, -step))
            ) / (2.0 * epsilon)
            assert np.allclose(dense[:, column], numeric, atol=1.0e-4, rtol=1.0e-5), block
            column += 1


def _level_spline() -> CubicBSplineTrajectory:
    """A level platform that only turns about the vertical (gravity is not tilted)."""

    rng = np.random.default_rng(12)
    count = 23
    yaw = np.cumsum(rng.normal(0.0, 0.15, count))
    rotations = np.asarray(
        [rotation_vector_to_matrix(np.array([0.0, 0.0, angle])) for angle in yaw]
    )
    positions = np.cumsum(rng.normal(0.0, 0.05, (count, 3)), axis=0)
    positions[:, 2] = 0.0
    return CubicBSplineTrajectory(0.0, 0.1, rotations, positions)


def test_gravity_magnitude_prior_separates_gravity_from_accel_bias_on_a_level_platform() -> None:
    spline = _level_spline()
    planes, gyro, accel = _measurements(spline, np.random.default_rng(13))
    noise = np.random.default_rng(14)
    accel = BSplineImuSamples(
        accel.times_sec, accel.values + noise.normal(0.0, accel.sigma, accel.values.shape), 0.03
    )
    start = replace(
        TRUTH, accel_bias_m_s2=np.zeros(3), gravity_world_m_s2=np.array([0.0, 0.0, -9.7])
    )

    def fit(magnitude: float | None) -> BSplineLidarImuFitResult:
        return fit_bspline_lidar_imu(
            BSplineLidarImuProblem(
                spline,
                planes,
                gyro,
                accel,
                start,
                estimate=("accel_bias", "gravity"),
                gravity_magnitude_m_s2=magnitude,
                gravity_magnitude_sigma_m_s2=0.01,
            )
        )

    free = fit(None)
    constrained = fit(9.81)
    # Without the prior only g_z + b_a,z is observed.
    assert free.block_correlations()[("accel_bias", "gravity")] > 0.99
    assert free.block_std("gravity")[2] > 1.0
    assert constrained.block_correlations()[("accel_bias", "gravity")] < 0.99
    assert constrained.block_std("gravity")[2] < 0.02
    assert constrained.block_std("accel_bias")[2] < 0.02
    magnitude = np.linalg.norm(constrained.calibration.gravity_world_m_s2)
    assert magnitude == pytest.approx(9.81, abs=0.02)
    # The prior fixes the vertical split; the horizontal one rests on the yaw.
    assert constrained.calibration.accel_bias_m_s2[2] == pytest.approx(
        TRUTH.accel_bias_m_s2[2], abs=0.02
    )
    assert constrained.calibration.gravity_world_m_s2[2] == pytest.approx(
        TRUTH.gravity_world_m_s2[2], abs=0.02
    )


def test_gravity_magnitude_prior_rejects_non_positive_values() -> None:
    spline = _level_spline()
    planes, gyro, accel = _measurements(spline, np.random.default_rng(13))
    with pytest.raises(ValueError, match="gravity magnitude"):
        BSplineLidarImuProblem(spline, planes, gyro, accel, TRUTH, gravity_magnitude_m_s2=-1.0)
