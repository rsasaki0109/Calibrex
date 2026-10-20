"""Independent SO(3) derivative checks away from a zero gyro residual."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse import csc_matrix

from calibrex.core import continuous_time_sparse as sparse
from calibrex.core.geometry import SE3
from calibrex.core.se3_manifold import (
    rotation_matrix_to_rotation_vector,
    rotation_vector_to_matrix,
    se3_exp,
)

_WINDOWS = ((0.2, 0.8), (0.4, 1.4), (1.0, 2.0), (1.2, 2.2), (2.1, 2.8))
_BIAS = np.array([0.01, -0.02, 0.015])
_SCALE = np.array([1.05, 0.98, 1.03])
_MOUNTING = np.array([0.1, -0.2, 0.05])
_CLOCK = 0.017
_STEP = 1e-6
_CASES = [
    (mounted, window, parameter)
    for mounted in (False, True)
    for window in _WINDOWS
    for parameter in ("knots", "bias", "scale", "clock", "mounting")
    if (parameter != "mounting" or mounted)
    # Time is nonsmooth at a knot. Pose/bias/scale derivatives there are smooth;
    # central time differences must not be treated as their clock oracle.
    and (parameter != "clock" or window != (1.0, 2.0))
]


def _fixture() -> tuple[tuple[float, ...], list[SE3]]:
    timestamps = tuple(float(i) for i in range(4))
    knots = [se3_exp(SE3.identity(), np.array([
        0.15 * t * t, 0.03 * t, 0.1 * np.cos(t),
        0.2 * np.sin(0.7 * t), -0.15 * np.cos(0.5 * t), 0.3 * t,
    ])) for t in timestamps]
    return timestamps, knots


def _measurement(window: tuple[float, float]) -> sparse.TrajectoryImuPreintegrationMeasurement:
    return sparse.TrajectoryImuPreintegrationMeasurement("gyro", tuple(
        sparse.TrajectoryImuGyroSample(float(t + _CLOCK), (
            0.2 + 0.15 * float(np.sin(1.3 * t)),
            -0.1 + 0.3 * float(np.cos(t)), 0.35 + 0.1 * float(t),
        )) for t in np.linspace(*window, 11)
    ))


@pytest.mark.parametrize("mounted,window,parameter", _CASES)
def test_gyro_state_blocks_match_central_finite_differences(
    mounted: bool, window: tuple[float, float], parameter: str,
) -> None:
    timestamps, knots = _fixture()
    measurement = _measurement(window)
    rotation = rotation_vector_to_matrix(_MOUNTING) if mounted else None

    def value(
        poses: list[SE3], bias: NDArray[np.float64], scale: NDArray[np.float64],
        clock: float, mounting: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        return sparse._imu_preintegration_residual(
            poses, timestamps, measurement, bias, clock, scale, imu_rotation=mounting,
        )[0]

    residual, blocks, jb, jc, js = sparse._imu_preintegration_residual(
        knots, timestamps, measurement, _BIAS, _CLOCK, _SCALE, imu_rotation=rotation,
    )
    # Nonzero error exercises the left/right log differential rather than its
    # common identity limit, which could hide either sign convention mistake.
    assert np.linalg.norm(residual) > 0.01
    if parameter == "knots":
        analytic = np.zeros((3, 6 * len(knots)))
        for index, block in blocks:
            analytic[:, 6 * index:6 * index + 6] = block
    elif parameter == "bias":
        analytic = jb
    elif parameter == "scale":
        analytic = js
    elif parameter == "clock":
        analytic = jc
    else:
        assert rotation is not None
        analytic = sparse._imu_preintegration_rotation_jacobian(
            knots, timestamps, measurement, _BIAS, _CLOCK, _SCALE, rotation,
        )
    numeric = np.zeros_like(analytic)
    for axis in range(analytic.shape[1]):
        values = []
        for sign in (1.0, -1.0):
            poses, bias, scale, clock, mounting = list(knots), _BIAS, _SCALE, _CLOCK, rotation
            if parameter == "knots":
                delta = np.zeros(6)
                delta[axis % 6] = sign * _STEP
                poses[axis // 6] = se3_exp(poses[axis // 6], delta)
            elif parameter == "clock":
                clock += sign * _STEP
            else:
                delta = np.zeros(3)
                delta[axis] = sign * _STEP
                if parameter == "bias":
                    bias = _BIAS + delta
                elif parameter == "scale":
                    scale = _SCALE + delta
                else:
                    assert rotation is not None
                    mounting = rotation @ rotation_vector_to_matrix(delta)
            values.append(value(poses, bias, scale, clock, mounting))
        numeric[:, axis] = (values[0] - values[1]) / (2 * _STEP)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("parameter", ["bias", "scale"])
@pytest.mark.parametrize("unit_scale", [False, True])
@pytest.mark.parametrize("motion", ["zero", "small", "multi_axis"])
def test_integral_returns_right_exp_derivatives_for_nonuniform_sample_times(
    parameter: str, unit_scale: bool, motion: str,
) -> None:
    # Direct integration checks isolate Exp from the endpoint residual's Log.
    times = (0.0, 0.03, 0.2, 0.21, 0.7, 1.0)
    samples = []
    for t in times:
        if motion == "zero":
            rate = _BIAS
        elif motion == "small":
            rate = _BIAS + np.array([1e-9, -2e-9, 3e-9])
        else:
            rate = np.array([0.8 * np.sin(2 * t), -0.6 * np.cos(t), 0.4 + t])
        samples.append(sparse.TrajectoryImuGyroSample(t, tuple(float(v) for v in rate)))
    rows = tuple(samples)
    scale = np.ones(3) if unit_scale else _SCALE
    base, jb, js = sparse._integrate_gyro(rows, _BIAS, scale)
    analytic = jb if parameter == "bias" else js
    numeric = np.zeros((3, 3))
    for axis in range(3):
        delta = np.zeros(3)
        delta[axis] = _STEP
        high, low = (
            sparse._integrate_gyro(
                rows, _BIAS + sign * delta if parameter == "bias" else _BIAS,
                scale + sign * delta if parameter == "scale" else scale,
            )[0] for sign in (1.0, -1.0)
        )
        numeric[:, axis] = (
            rotation_matrix_to_rotation_vector(base.T @ high)
            - rotation_matrix_to_rotation_vector(base.T @ low)
        ) / (2 * _STEP)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-7, atol=1e-8)


def test_fitter_gyro_columns_match_weighted_objective_directional_derivative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps, knots = _fixture()
    measurement = _measurement((0.4, 1.4))
    measurement = sparse.TrajectoryImuPreintegrationMeasurement(
        measurement.measurement_id, measurement.gyro_samples, weight=2.5,
    )
    problem = sparse.ContinuousTimeTrajectoryFitProblem(
        timestamps, tuple(knots), imu_preintegration_measurements=(measurement,),
        initial_gyro_bias_rad_s=tuple(_BIAS), initial_gyro_scale=tuple(_SCALE),
        initial_imu_clock_offset_sec=_CLOCK, initial_imu_rotation_vector_rad=tuple(_MOUNTING),
        estimate_gyro_bias=True, estimate_gyro_scale=True, estimate_imu_clock_offset=True,
        estimate_imu_rotation=True,
    )
    captured: list[NDArray[np.float64]] = []
    original = sparse._assemble_sparse_jacobian

    class Captured(Exception):
        pass

    def capture(
        blocks: list[tuple[int, int, NDArray[np.float64]]], *,
        residual_count: int, parameter_count: int,
    ) -> csc_matrix:
        captured.append(original(blocks, residual_count=residual_count,
                                 parameter_count=parameter_count).toarray())
        raise Captured

    monkeypatch.setattr(sparse, "_assemble_sparse_jacobian", capture)
    with pytest.raises(Captured):
        sparse.fit_continuous_trajectory(problem)
    matrix = captured[0]
    # Fitter column order: all knots, bias, clock, gyro scale, mounting rotation.
    direction = np.sin(np.arange(matrix.shape[1]) + 1.0)
    rotation = rotation_vector_to_matrix(_MOUNTING)

    def score(sign: float) -> float:
        d = sign * _STEP * direction
        poses = [se3_exp(pose, d[6 * i:6 * i + 6]) for i, pose in enumerate(knots)]
        return sparse._objective(problem, [], [], [], [measurement], [], poses,
            _BIAS + d[24:27], np.zeros(3), _CLOCK + float(d[27]), np.zeros(3),
            np.zeros(3), _SCALE + d[28:31], np.ones(3),
            imu_rotation=rotation @ rotation_vector_to_matrix(d[31:34]))

    residual = sparse._imu_preintegration_residual(knots, timestamps, measurement,
        _BIAS, _CLOCK, _SCALE, imu_rotation=rotation)[0]
    numeric = (score(1.0) - score(-1.0)) / (2 * _STEP)
    model = 2 * measurement.weight**2 * float(residual @ matrix @ direction)
    assert model == pytest.approx(numeric, rel=1e-7, abs=1e-8)


def test_nonunit_gyro_scale_recovers_constant_bias_with_anchored_poses() -> None:
    truth = np.array([0.3, -0.2, 0.4])
    measurement = sparse.TrajectoryImuPreintegrationMeasurement("static", tuple(
        sparse.TrajectoryImuGyroSample(time, tuple(truth)) for time in (0.2, 0.8)
    ))
    problem = sparse.ContinuousTimeTrajectoryFitProblem(
        (0.0, 1.0), (SE3.identity(), SE3.identity()),
        pose_measurements=tuple(sparse.TrajectoryPoseMeasurement(
            str(i), float(i), SE3.identity(), weight=100.0,
        ) for i in range(2)),
        imu_preintegration_measurements=(measurement,), estimate_gyro_bias=True,
        initial_gyro_scale=(1.05, 0.98, 1.3),
    )
    result = sparse.fit_continuous_trajectory(
        problem, sparse.ContinuousTimeTrajectoryFitOptions(max_iterations=5),
    )
    # Static anchored poses imply corrected angular rate zero. The bias must
    # equal the constant raw reading, independent of the nonunit scale.
    assert result.gyro_bias_rad_s is not None
    assert np.linalg.norm(np.asarray(result.gyro_bias_rad_s) - truth) < 1e-6
    assert result.final_objective < 1e-10
