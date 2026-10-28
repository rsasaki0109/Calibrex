"""Compare value-only and shared IMU evaluation with the previous expressions."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from numpy.typing import NDArray

from calibrex.core import continuous_time_sparse as sparse
from calibrex.core import se3_manifold as manifold
from calibrex.core.geometry import SE3


def _bits(actual: NDArray[np.float64], expected: NDArray[np.float64]) -> None:
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


def _knots() -> list[SE3]:
    return [manifold.se3_exp(SE3.identity(), np.array([
        0.15 * t * t, 0.03 * t, 0.1 * np.cos(t),
        0.2 * np.sin(0.7 * t), -0.15 * np.cos(0.5 * t), 0.3 * t,
    ])) for t in range(4)]


_TIMES = (0.0, 1.0, 2.0, 3.0)
_BIAS = np.array([0.01, -0.02, 0.03])
_SCALE = np.array([1.05, 0.98, 1.02])


def _mounting(kind: str) -> NDArray[np.float64] | None:
    if kind == "unmounted":
        return None
    if kind == "identity":
        return np.eye(3)
    return manifold.rotation_vector_to_matrix(np.array([0.1, -0.2, 0.05]))


def _measurement(window: tuple[float, float], magnitude: float):
    times = window[0] + (window[1] - window[0]) * np.array([
        0.0, 0.03, 0.12, 0.25, 0.47, 0.7, 0.8, 0.91, 1.0,
    ])
    return sparse.TrajectoryImuPreintegrationMeasurement("gyro", tuple(
        sparse.TrajectoryImuGyroSample(float(t), tuple(_BIAS + magnitude * np.array([
            0.2 + t, -0.1 * t, 0.3,
        ]))) for t in times
    ))


def _previous_gyro(
    knots, timestamps, measurement, gyro_bias, clock_offset_sec=0.0, gyro_scale=None, *,
    imu_rotation=None, interpolation=None, rotation_jacobian=False, clock_side=None,
):
    """Pre-change factor and separate mounting helper, with all three integrals."""
    scale = np.ones(3) if gyro_scale is None else np.asarray(gyro_scale).reshape(3)
    start_time = measurement.start_time_sec - clock_offset_sec
    end_time = measurement.end_time_sec - clock_offset_sec
    start = sparse._interpolated_pose_at(knots, timestamps, start_time,
                                         interpolation=interpolation)
    end = sparse._interpolated_pose_at(knots, timestamps, end_time,
                                       interpolation=interpolation)
    rs, re = sparse._rotation_matrix(start), sparse._rotation_matrix(end)
    delta, jb, js = sparse._integrate_gyro(measurement.gyro_samples, gyro_bias, scale)
    if imu_rotation is not None:
        delta = imu_rotation @ delta @ imu_rotation.T
        jb, js = imu_rotation @ jb, imu_rotation @ js
    residual = manifold.rotation_matrix_to_rotation_vector(delta.T @ rs.T @ re)
    log = manifold.so3_left_jacobian_inverse(-residual)
    start_block, end_block = -log @ delta.T, manifold.so3_left_jacobian_inverse(residual)
    jb, js = -log @ jb, -log @ js
    side = clock_side if clock_side is not None else "left"
    ws = sparse._angular_rate_at(
        knots, timestamps, start_time, side=side, interpolation=interpolation,
    )
    we = sparse._angular_rate_at(
        knots, timestamps, end_time, side=side, interpolation=interpolation,
    )
    clock = (start_block @ (-ws) + end_block @ (-we)).reshape(3, 1)
    combined = {}
    for index, block in (*sparse._interpolate_rotation_blocks(
        knots, timestamps, start_time, start_block, interpolation=interpolation,
    ), *sparse._interpolate_rotation_blocks(
        knots, timestamps, end_time, end_block, interpolation=interpolation,
    )):
        combined[index] = combined[index] + block if index in combined else block
    mounting_block = None
    if rotation_jacobian:
        mounting = np.eye(3) if imu_rotation is None else imu_rotation
        separate = _previous_gyro(
            knots, timestamps, measurement, gyro_bias, clock_offset_sec, scale,
            imu_rotation=mounting, interpolation=interpolation,
        )
        delta_imu = sparse._integrate_gyro(measurement.gyro_samples, gyro_bias, scale)[0]
        mounting_block = (
            -manifold.so3_left_jacobian_inverse(-separate.residual)
            @ mounting @ (delta_imu.T - np.eye(3))
        )
    return sparse._ImuPreintegrationLinearization(
        residual, sorted(combined.items()), jb, clock, js, mounting_block,
    )


def _previous_force(knots, measurement, lever, clock, bias, gravity, scale, mounting):
    """Pre-change force expression, including calibration derivatives."""
    omega, alpha, origin, rotation = sparse._body_motion(
        knots, _TIMES, measurement.timestamp_sec - clock,
    )
    kinematic = (origin + np.cross(alpha, lever) + np.cross(omega, np.cross(omega, lever))
                 - rotation.T @ gravity)
    jl = manifold.skew(alpha) + manifold.skew(omega) @ manifold.skew(omega)
    jg = -np.diag(scale) @ rotation.T
    if mounting is not None:
        kinematic = mounting.T @ kinematic
        jl = mounting.T @ jl
        jg = -np.diag(scale) @ mounting.T @ rotation.T
    jl = scale[:, None] * jl
    predicted = scale * kinematic + bias
    residual = predicted - np.asarray(measurement.accel_body_m_s2)
    return residual, jl, np.eye(3), jg, np.diag(kinematic)


@pytest.mark.parametrize("kind", ["unmounted", "identity", "mounted"])
@pytest.mark.parametrize("window", [(0.015, 0.7), (0.999, 2.001), (2.2, 3.015)])
@pytest.mark.parametrize("magnitude", [0.0, 1e-12, 1.0])
def test_one_integral_keeps_every_gyro_block_and_residual_bit(kind, window, magnitude):
    knots, mounting = _knots(), _mounting(kind)
    measurement = _measurement(window, magnitude)
    args = (knots, _TIMES, measurement, _BIAS, 0.015, _SCALE)
    expected = _previous_gyro(*args, imu_rotation=mounting, rotation_jacobian=True)
    actual = sparse._linearize_imu_preintegration(
        *args, imu_rotation=mounting, rotation_jacobian=True,
        interpolation=sparse._ScrewInterpolation(knots),
    )
    for name in ("residual", "jacobian_bias", "jacobian_clock", "jacobian_scale",
                 "jacobian_rotation"):
        _bits(getattr(actual, name), getattr(expected, name))
    for (ai, a), (bi, b) in zip(actual.knot_blocks, expected.knot_blocks, strict=True):
        assert ai == bi
        _bits(a, b)
    value = sparse._evaluate_imu_preintegration(*args, imu_rotation=mounting)
    _bits(value.residual, expected.residual)


@pytest.mark.parametrize("kind", ["unmounted", "identity", "mounted"])
@pytest.mark.parametrize("time", [-0.01, 0.0, 0.001, 0.9999, 1.0, 1.0001, 2.999, 3.01])
def test_force_value_and_sampled_clock_derivative_keep_all_bits(kind, time):
    knots, mounting = _knots(), _mounting(kind)
    measurement = sparse.TrajectoryImuLeverArmMeasurement("accel", time, (0.1, 0.2, 9.7))
    lever, gravity = np.array([0.1, -0.03, 0.02]), np.array([0.1, 0.0, -9.8])
    args = (knots, measurement, lever, 0.015, _BIAS, gravity, _SCALE, mounting)
    reference = _previous_force(*args)
    shifted = _previous_force(knots, measurement, lever, 0.015 + 1e-6,
                              _BIAS, gravity, _SCALE, mounting)[0]
    expected = (reference[0], reference[1], ((shifted - reference[0]) / 1e-6).reshape(3, 1),
                *reference[2:])
    actual = sparse._lever_arm_residual(
        knots, _TIMES, measurement, lever, 0.015, _BIAS, gravity, _SCALE,
        imu_rotation=mounting, interpolation=sparse._ScrewInterpolation(knots),
    )
    for a, b in zip(actual, expected, strict=True):
        _bits(a, b)


def test_linearization_integrates_once_and_retains_no_cross_call_cache(monkeypatch):
    knots, measurement = _knots(), _measurement((0.2, 0.8), 1.0)
    original, calls = sparse._integrate_gyro, []

    def integrate(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(sparse, "_integrate_gyro", integrate)
    first = sparse._linearize_imu_preintegration(
        knots, _TIMES, measurement, _BIAS, rotation_jacobian=True,
    )
    assert len(calls) == 1
    first.jacobian_bias[:] = 17.0
    second = sparse._linearize_imu_preintegration(
        knots, _TIMES, measurement, _BIAS + 0.01, rotation_jacobian=True,
    )
    assert len(calls) == 2
    assert not np.any(second.jacobian_bias == 17.0)
    assert not np.array_equal(first.residual, second.residual)


@pytest.mark.parametrize("kind", ["unmounted", "identity", "mounted"])
def test_objective_and_rmse_skip_imu_derivatives_and_clock_perturbations(kind, monkeypatch):
    knots, mounting = _knots(), _mounting(kind)
    gyro = (_measurement((0.2, 0.8), 1.0), _measurement((1.1, 2.3), 1e-12))
    lever = tuple(sparse.TrajectoryImuLeverArmMeasurement(str(t), t, (0.1, 0.2, 9.7))
                  for t in (0.4, 1.0, 2.9))
    arm, gravity = np.array([0.1, -0.03, 0.02]), np.array([0.1, 0.0, -9.8])
    rg = [_previous_gyro(knots, _TIMES, row, _BIAS, 0.015, _SCALE,
                         imu_rotation=mounting).residual for row in gyro]
    ra = [_previous_force(knots, row, arm, 0.015, _BIAS, gravity, _SCALE, mounting)[0]
          for row in lever]
    sg, sa = (sum(float(np.sum(r * r)) for r in rows) for rows in (rg, ra))
    problem = sparse.ContinuousTimeTrajectoryFitProblem(
        _TIMES, tuple(knots), imu_preintegration_measurements=gyro,
        imu_lever_arm_measurements=lever,
    )

    def unused(*args, **kwargs):
        raise AssertionError("value-only scoring requested an IMU derivative")

    for name in ("_integrate_gyro", "_linearize_imu_preintegration", "_angular_rate_at",
                 "_interpolation_jacobians", "_lever_arm_residual", "skew"):
        monkeypatch.setattr(sparse, name, unused)
    motion = sparse._body_motion

    def only_nominal(poses, timestamps, timestamp, **kwargs):
        assert timestamp in [row.timestamp_sec - 0.015 for row in lever]
        return motion(poses, timestamps, timestamp, **kwargs)

    monkeypatch.setattr(sparse, "_body_motion", only_nominal)
    objective = sparse._objective(problem, [], [], [], list(gyro), list(lever), knots,
        _BIAS, arm, 0.015, _BIAS, gravity, _SCALE, _SCALE, imu_rotation=mounting)
    # Retain the original sequential accumulation over both sensor families.
    expected = 0.0
    for r in (*rg, *ra):
        expected += float(np.sum(r * r))
    _bits(np.array([objective]), np.array([expected]))
    vector = None if kind == "unmounted" else (
        np.zeros(3) if kind == "identity" else np.array([0.1, -0.2, 0.05]))
    actual_g = sparse.imu_preintegration_rmse(
        gyro, _TIMES, knots, _BIAS, 0.015, _SCALE, imu_rotation_vector_rad=vector,
    )
    actual_a = sparse.imu_lever_arm_rmse(
        lever, _TIMES, knots, arm, 0.015, _BIAS, gravity, _SCALE,
        imu_rotation_vector_rad=vector,
    )
    _bits(np.array([actual_g, actual_a]), np.array([
        math.sqrt(sg / (3 * len(gyro))), math.sqrt(sa / (3 * len(lever))),
    ]))


@pytest.mark.parametrize("mounting", [(0.0, 0.0, 0.0), (0.1, -0.2, 0.05)])
@pytest.mark.parametrize("damping", [1e-3, 1e3])
def test_complete_fit_and_all_sparse_models_match_legacy_gyro(mounting, damping, monkeypatch):
    knots = _knots()
    problem = sparse.ContinuousTimeTrajectoryFitProblem(
        _TIMES, tuple(knots),
        pose_measurements=tuple(sparse.TrajectoryPoseMeasurement(
            str(i), time, SE3.identity(), weight=4.0,
        ) for i, time in enumerate(_TIMES)),
        imu_preintegration_measurements=tuple(_measurement(window, 1.0)
            for window in ((0.2, 0.8), (1.1, 1.8), (2.1, 2.8))),
        imu_lever_arm_measurements=tuple(sparse.TrajectoryImuLeverArmMeasurement(
            str(t), t, (0.1, 0.2, 9.7), weight=0.3,
        ) for t in (0.4, 1.1, 2.6)),
        initial_gyro_scale=tuple(_SCALE), initial_imu_rotation_vector_rad=mounting,
        estimate_gyro_bias=True, estimate_gyro_scale=True, estimate_imu_clock_offset=True,
        estimate_imu_rotation=True, estimate_lever_arm=True,
        estimate_accel_bias=True, estimate_accel_scale=True,
    )
    models = []
    assemble = sparse._assemble_sparse_jacobian

    def capture(*args, **kwargs):
        result = assemble(*args, **kwargs)
        models.append((result.data.copy(), result.indices.copy(), result.indptr.copy()))
        return result

    monkeypatch.setattr(sparse, "_assemble_sparse_jacobian", capture)
    options = sparse.ContinuousTimeTrajectoryFitOptions(max_iterations=4, initial_damping=damping)
    actual = sparse.fit_continuous_trajectory(problem, options)
    reference_models = list(models)
    models.clear()
    monkeypatch.setattr(sparse, "_linearize_imu_preintegration", _previous_gyro)
    expected = sparse.fit_continuous_trajectory(problem, options)
    assert dataclasses.asdict(actual) == dataclasses.asdict(expected)
    assert len(models) == len(reference_models)
    for a, b in zip(models, reference_models, strict=True):
        for array, reference in zip(a, b, strict=True):
            np.testing.assert_array_equal(array.view(np.uint8), reference.view(np.uint8))
