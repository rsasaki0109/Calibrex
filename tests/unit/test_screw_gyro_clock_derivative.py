"""Clock columns at smooth times and declared one-sided screw boundaries."""

from __future__ import annotations

from bisect import bisect_left, bisect_right

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibrex.core import continuous_time_sparse as sparse
from calibrex.core.geometry import SE3
from calibrex.core.se3_manifold import rotation_vector_to_matrix

_TIMES = (0.0, 0.5, 1.25, 2.0, 3.0)
_ANGLES = (0.0, 0.15, 0.6, 0.8, 1.25)
_BIAS = np.array([0.01, -0.02, 0.03])
_SCALE = np.array([1.05, 0.98, 1.03])


def _knots(times=_TIMES, angles=_ANGLES, *, multi_axis=False):
    vectors = [np.array([0.2*np.sin(t), -0.1*np.cos(0.7*t), angle])
               if multi_axis else np.array([0.0, 0.0, angle])
               for t, angle in zip(times, angles, strict=True)]
    return [SE3((0.1*t*t, -0.03*t, 0.02), tuple(Rotation.from_rotvec(v).as_quat()))
            for t, v in zip(times, vectors, strict=True)]


def _known_rate(times, angles, time, *, right=False):
    search = bisect_right if right else bisect_left
    index = max(0, min(search(times, time)-1, len(times)-2))
    return (angles[index+1] - angles[index]) / (times[index+1] - times[index])


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("time", [
    0.0, 0.1, 0.4998, 0.5, 0.5002, 1.2498, 1.25, 1.2502,
    1.9998, 2.0, 2.0002, 2.9998, 3.0,
    np.nextafter(0.5, -np.inf), np.nextafter(0.5, np.inf),
])
def test_angular_rate_is_exact_segment_slope_without_a_time_stencil(time, cached):
    knots = _knots()
    workspace = sparse._ScrewInterpolation(knots) if cached else None
    actual = sparse._angular_rate_at(knots, _TIMES, time, interpolation=workspace)
    expected = np.array([0.0, 0.0, _known_rate(_TIMES, _ANGLES, time)])
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("time", [0.0, 0.0001, 0.0001999, 0.0002, 0.0003, 0.00045, 0.0008])
def test_angular_rate_supports_domains_shorter_than_the_old_sample_stencil(time):
    times, angles = (0.0, 0.0002, 0.00045, 0.0008), (0.0, 0.03, -0.04, 0.1)
    actual = sparse._angular_rate_at(_knots(times, angles), times, time)
    expected = np.array([0.0, 0.0, _known_rate(times, angles, time)])
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)


_WINDOWS = ((0.1, 0.4998), (0.5, 1.25), (0.4998, 1.2498), (0.5002, 1.2502),
            (0.0, 1.25), (0.5, 3.0), (0.1, 2.8))


def _measurement(window, clock, *, multi_axis):
    return sparse.TrajectoryImuPreintegrationMeasurement("clock", tuple(
        sparse.TrajectoryImuGyroSample(float(t + clock),
            (0.2 + 0.05*t, -0.1 + 0.02*t*t, 0.35) if multi_axis else (0.0, 0.0, 0.13))
        for t in np.linspace(*window, 7)
    ), weight=2.5)


@pytest.mark.parametrize("clock", [0.0, 0.017])
@pytest.mark.parametrize("window", _WINDOWS)
def test_clock_column_matches_closed_form_piecewise_z_rotation(window, clock):
    knots, measurement = _knots(), _measurement(window, clock, multi_axis=False)
    factor = sparse._linearize_imu_preintegration(knots, _TIMES, measurement, np.zeros(3), clock)
    start, end = measurement.start_time_sec-clock, measurement.end_time_sec-clock
    # At the first domain endpoint only a negative clock update is feasible.
    right = start == _TIMES[0]
    expected = np.array([0.0, 0.0,
        _known_rate(_TIMES, _ANGLES, start, right=right)
        - _known_rate(_TIMES, _ANGLES, end, right=right),
    ]).reshape(3, 1)
    np.testing.assert_allclose(factor.jacobian_clock, expected, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("mounting", [None, (0.0, 0.0, 0.0), (0.1, -0.2, 0.05)])
@pytest.mark.parametrize("window", _WINDOWS)
def test_multiaxis_clock_column_matches_independent_one_sided_residual_differences(
    window, mounting,
):
    clock = 0.017
    knots, measurement = _knots(multi_axis=True), _measurement(window, clock, multi_axis=True)
    rotation = None if mounting is None else rotation_vector_to_matrix(np.asarray(mounting))
    actual = sparse._linearize_imu_preintegration(
        knots, _TIMES, measurement, _BIAS, clock, _SCALE, imu_rotation=rotation,
    )
    assert np.linalg.norm(actual.residual) > 0.01

    def value(offset):
        return sparse._evaluate_imu_preintegration(
            knots, _TIMES, measurement, _BIAS, offset, _SCALE, imu_rotation=rotation,
        ).residual

    sign = -1.0 if measurement.start_time_sec-clock == _TIMES[0] else 1.0
    h = 1e-5
    coarse = (value(clock+sign*h)-value(clock))/(sign*h)
    fine = (value(clock+sign*h/2)-value(clock))/(sign*h/2)
    numeric = (2.0*fine-coarse).reshape(3, 1)
    np.testing.assert_allclose(actual.jacobian_clock, numeric, rtol=1e-6, atol=1e-7)


def test_cached_angular_rate_needs_neither_sampled_poses_nor_interval_jacobians(monkeypatch):
    knots = _knots()
    workspace = sparse._ScrewInterpolation(knots)

    def unused(*args, **kwargs):
        raise AssertionError("exact interval rate must not sample poses or prepare Jacobians")

    monkeypatch.setattr(workspace, "pose", unused)
    monkeypatch.setattr(workspace, "jacobians", unused)
    actual = sparse._angular_rate_at(knots, _TIMES, 0.4998, interpolation=workspace)
    np.testing.assert_allclose(actual, (0.0, 0.0, 0.3), atol=1e-12, rtol=1e-12)
    assert workspace._segments and not workspace._poses and not workspace._jacobians
    assert all(segment._jacobian_factors is None for segment in workspace._segments.values())


@pytest.mark.parametrize("time", [-1e-6, 3.000001])
def test_exact_angular_rate_rejects_times_outside_the_pose_domain(time):
    with pytest.raises(ValueError, match="outside the knot domain"):
        sparse._angular_rate_at(_knots(), _TIMES, time)


@pytest.mark.parametrize("window", (*_WINDOWS[:3], _WINDOWS[4]))
def test_fitter_clock_normal_gradient_uses_the_declared_one_sided_column(window, monkeypatch):
    clock = 0.017
    measurement = _measurement(window, clock, multi_axis=False)
    problem = sparse.ContinuousTimeTrajectoryFitProblem(
        _TIMES, tuple(_knots()), imu_preintegration_measurements=(measurement,),
        initial_imu_clock_offset_sec=clock, estimate_imu_clock_offset=True,
    )
    start, end = measurement.start_time_sec-clock, measurement.end_time_sec-clock

    def angle(time):
        index = max(0, min(bisect_left(_TIMES, time)-1, len(_TIMES)-2))
        return _ANGLES[index] + (time-_TIMES[index])*_known_rate(_TIMES, _ANGLES, time)

    expected_residual = (
        angle(end)-angle(start)-0.13*(measurement.end_time_sec-measurement.start_time_sec)
    )
    right = start == _TIMES[0]
    expected_column = (_known_rate(_TIMES, _ANGLES, start, right=right)
                       - _known_rate(_TIMES, _ANGLES, end, right=right))
    expected_gradient = measurement.weight**2 * expected_residual * expected_column

    class Captured(Exception):
        pass

    def capture(gradient, diagonal, residual_norm):
        assert gradient.shape == (31,)
        assert gradient[-1] == pytest.approx(expected_gradient, rel=1e-11, abs=1e-12)
        raise Captured

    monkeypatch.setattr(sparse, "normalized_gradient_norm", capture)
    with pytest.raises(Captured):
        sparse.fit_continuous_trajectory(problem)
