"""Independent derivative checks for the discrete screw accelerometer factor."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse import csc_matrix

from calibrex.core import continuous_time_sparse as sparse
from calibrex.core.geometry import SE3
from calibrex.core.se3_manifold import rotation_vector_to_matrix, se3_exp

_TIMES = (0.0, 0.0005, 0.3, 0.9995, 1.0, 1.0005, 1.4, 2.1, 2.9999, 3.0)
_LEVER = np.array([0.12, -0.08, 0.22])
_GRAVITY = np.array([0.2, -0.3, -9.80665])
_SCALE = np.array([1.02, 0.98, 1.05])
_BIAS = np.array([0.03, -0.02, 0.015])
_MOUNTING = np.array([0.1, -0.2, 0.05])
_CLOCK = 0.017


def _fixture(count: int = 4) -> tuple[tuple[float, ...], list[SE3]]:
    timestamps = tuple(float(i) for i in range(count))
    knots = [
        se3_exp(
            SE3.identity(),
            np.array([
                0.15 * t * t, 0.03 * t, 0.1 * np.cos(t),
                0.2 * np.sin(0.7 * t), -0.15 * np.cos(0.5 * t), 0.3 * t,
            ]),
        )
        for t in timestamps
    ]
    return timestamps, knots


def _knot_finite_difference(
    knots: list[SE3], timestamps: tuple[float, ...], time: float,
    mounting: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    # Use the public prediction, perturbing every knot, including inactive ones.
    # Its second time difference has cancellation noise at very small FD steps.
    step = 1e-4
    numeric = np.zeros((3, 6 * len(knots)))
    for index in range(len(knots)):
        for axis in range(6):
            delta = np.zeros(6)
            delta[axis] = step
            plus, minus = list(knots), list(knots)
            plus[index] = se3_exp(knots[index], delta)
            minus[index] = se3_exp(knots[index], -delta)
            high, low = (
                sparse.predicted_imu_specific_force(
                    poses, timestamps, time, _LEVER,
                    gravity_world_m_s2=_GRAVITY, accel_scale=_SCALE,
                    imu_rotation_vector_rad=mounting,
                )
                for poses in (plus, minus)
            )
            numeric[:, 6 * index + axis] = (high - low) / (2 * step)
    return numeric


@pytest.mark.parametrize("body_time", _TIMES)
@pytest.mark.parametrize("mounted", [False, True])
def test_accelerometer_knot_blocks_match_all_knot_finite_differences(
    body_time: float, mounted: bool,
) -> None:
    timestamps, knots = _fixture()
    mounting = _MOUNTING if mounted else None
    rotation = rotation_vector_to_matrix(mounting) if mounting is not None else None
    measurement = sparse.TrajectoryImuLeverArmMeasurement(
        "accel", body_time + _CLOCK, (0.3, -0.1, 9.5),
    )
    blocks = sparse._lever_arm_knot_jacobians(
        knots, timestamps, measurement, _LEVER, _CLOCK, _GRAVITY, _SCALE,
        imu_rotation=rotation,
    )
    analytic = np.zeros((3, 6 * len(knots)))
    for index, jacobian in blocks:
        analytic[:, 6 * index:6 * index + 6] = jacobian
    assert len({index for index, _ in blocks}) == len(blocks)
    if body_time in (0.9995, 1.0, 1.0005):
        assert {index for index, _ in blocks} == {0, 1, 2}
    numeric = _knot_finite_difference(knots, timestamps, body_time, mounting)
    np.testing.assert_allclose(analytic, numeric, rtol=2e-5, atol=2e-5)


def test_stencil_collects_disjoint_supports_on_dense_nonuniform_knots() -> None:
    _, knots = _fixture(6)
    timestamps = (0.0, 0.1, 0.1004, 0.1008, 0.1012, 1.0)
    time = 0.1006
    measurement = sparse.TrajectoryImuLeverArmMeasurement("dense", time, (0.0,) * 3)
    blocks = sparse._lever_arm_knot_jacobians(
        knots, timestamps, measurement, _LEVER, 0.0, _GRAVITY, _SCALE,
        imu_rotation=rotation_vector_to_matrix(_MOUNTING),
    )
    # Three samples can touch six knots if intervals are shorter than the stencil.
    assert [index for index, _ in blocks] == list(range(6))
    analytic = np.hstack([block for _, block in blocks])
    numeric = _knot_finite_difference(knots, timestamps, time, _MOUNTING)
    np.testing.assert_allclose(analytic, numeric, rtol=2e-5, atol=2e-5)


def test_acceleration_updates_trajectory_with_all_calibration_states_fixed() -> None:
    timestamps = (0.0, 1.0, 2.0)
    initial = (SE3.identity(),) * 3
    measurement = sparse.TrajectoryImuLeverArmMeasurement("accel", 1.0, (-2.0, 0.0, 0.0))
    result = sparse.fit_continuous_trajectory(
        sparse.ContinuousTimeTrajectoryFitProblem(
            timestamps, initial, imu_lever_arm_measurements=(measurement,),
            pose_measurements=tuple(
                sparse.TrajectoryPoseMeasurement(str(i), t, pose)
                for i, (t, pose) in enumerate(zip(timestamps, initial, strict=True))
            ),
        ),
        sparse.ContinuousTimeTrajectoryFitOptions(max_iterations=20),
    )
    predicted = sparse.predicted_imu_specific_force(
        result.knot_poses, timestamps, 1.0, (0.0, 0.0, 0.0),
    )
    # Without trajectory columns the fitter declares the zero initial gradient
    # converged, leaving a 2 m/s^2 residual despite adjustable knot positions.
    assert np.linalg.norm(predicted - np.asarray(measurement.accel_body_m_s2)) < 1e-4
    assert result.final_objective < 1e-5


@pytest.mark.parametrize("body_time", _TIMES)
@pytest.mark.parametrize("mounted", [False, True])
def test_lever_derivative_includes_nonunit_scale_after_mounting_rotation(
    body_time: float, mounted: bool,
) -> None:
    timestamps, knots = _fixture()
    rotation = rotation_vector_to_matrix(_MOUNTING) if mounted else None
    measurement = sparse.TrajectoryImuLeverArmMeasurement(
        "accel", body_time + _CLOCK, (0.3, -0.1, 9.5),
    )
    _, jacobian, *_ = sparse._lever_arm_residual(
        knots, timestamps, measurement, _LEVER, _CLOCK, _BIAS, _GRAVITY, _SCALE,
        imu_rotation=rotation,
    )
    numeric = np.zeros((3, 3))
    step = 1e-5
    for axis in range(3):
        delta = np.zeros(3)
        delta[axis] = step
        high, low = (
            sparse._lever_arm_kinematics_residual(
                knots, timestamps, measurement, lever, _CLOCK, _BIAS, _GRAVITY, _SCALE,
                imu_rotation=rotation,
            )[0]
            for lever in (_LEVER + delta, _LEVER - delta)
        )
        numeric[:, axis] = (high - low) / (2 * step)
    np.testing.assert_allclose(jacobian, numeric, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("body_time", [0.3, 1.0, 2.9999])
def test_fitter_assembles_knot_blocks_and_matches_weighted_objective_gradient(
    body_time: float, monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps, knots = _fixture()
    measurement = sparse.TrajectoryImuLeverArmMeasurement(
        "accel", body_time + _CLOCK, (0.3, -0.1, 9.5), weight=2.5,
    )
    problem = sparse.ContinuousTimeTrajectoryFitProblem(
        timestamps, tuple(knots), imu_lever_arm_measurements=(measurement,),
        initial_lever_arm_body_m=tuple(_LEVER), initial_imu_clock_offset_sec=_CLOCK,
        initial_accel_bias_body_m_s2=tuple(_BIAS),
        initial_gravity_world_m_s2=tuple(_GRAVITY), initial_accel_scale=tuple(_SCALE),
        initial_imu_rotation_vector_rad=tuple(_MOUNTING),
    )
    matrices: list[NDArray[np.float64]] = []
    original = sparse._assemble_sparse_jacobian

    class Captured(Exception):
        pass

    def capture(
        blocks: list[tuple[int, int, NDArray[np.float64]]], *,
        residual_count: int, parameter_count: int,
    ) -> csc_matrix:
        matrix = original(blocks, residual_count=residual_count, parameter_count=parameter_count)
        matrices.append(matrix.toarray())
        # Stop before solving an intentionally underconstrained acceleration-only
        # problem. Inspect the actual fitter assembly rather than just a helper.
        raise Captured

    monkeypatch.setattr(sparse, "_assemble_sparse_jacobian", capture)
    with pytest.raises(Captured):
        sparse.fit_continuous_trajectory(problem)
    jacobian = matrices[0]
    numeric = _knot_finite_difference(knots, timestamps, body_time, _MOUNTING)
    np.testing.assert_allclose(jacobian, numeric, rtol=2e-5, atol=2e-5)
    rotation = rotation_vector_to_matrix(_MOUNTING)

    def score(poses: list[SE3]) -> float:
        return sparse._objective(
            problem, [], [], [], [], [measurement], poses,
            np.zeros(3), _LEVER, _CLOCK, _BIAS, _GRAVITY, np.ones(3), _SCALE,
            imu_rotation=rotation,
        )

    residual = sparse.predicted_imu_specific_force(
        knots, timestamps, body_time, _LEVER, accel_bias_body_m_s2=_BIAS,
        gravity_world_m_s2=_GRAVITY, accel_scale=_SCALE,
        imu_rotation_vector_rad=_MOUNTING,
    ) - np.asarray(measurement.accel_body_m_s2)
    # A deterministic mixed translation/rotation direction, with all knots active.
    direction = np.sin(np.arange(24) + 1.0).reshape(4, 6)
    step = 1e-4
    high = [se3_exp(pose, step * d) for pose, d in zip(knots, direction, strict=True)]
    low = [se3_exp(pose, -step * d) for pose, d in zip(knots, direction, strict=True)]
    numerical_gradient = (score(high) - score(low)) / (2 * step)
    model_gradient = 2 * measurement.weight**2 * float(residual @ jacobian @ direction.ravel())
    assert model_gradient == pytest.approx(numerical_gradient, rel=2e-5, abs=2e-3)
