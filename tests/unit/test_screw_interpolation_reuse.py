"""Exact fixed-state reuse, boundaries, and fresh optimizer state checks."""

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


def _previous_jacobians(
    left: SE3, right: SE3, alpha: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Independent pre-reuse expression, including both inverse evaluations."""
    u = manifold.se3_log(left, right)
    rotation, translation = manifold.se3_exp_local(u[:3], u[3:])
    scaled = alpha * u
    scaled_rotation, scaled_translation = manifold.se3_exp_local(scaled[:3], scaled[3:])
    scaled_jacobian = manifold.se3_left_jacobian(scaled[:3], scaled[3:])
    jacobian = manifold.se3_left_jacobian(u[:3], u[3:])
    adjoint = manifold.se3_adjoint(rotation, translation)
    scaled_adjoint = manifold.se3_adjoint(scaled_rotation, scaled_translation)
    right_jacobian = alpha * (scaled_jacobian @ np.linalg.inv(jacobian))
    left_jacobian = np.linalg.inv(scaled_adjoint) - alpha * (
        scaled_jacobian @ np.linalg.inv(jacobian) @ np.linalg.inv(adjoint)
    )
    return left_jacobian, right_jacobian


@pytest.mark.parametrize("angle", [0.0, 1e-13, 1e-9, 0.5, math.pi - 1e-8])
@pytest.mark.parametrize("alpha", [-0.1, 0.0, 0.25, 1.0, 1.1])
def test_prepared_segment_preserves_previous_pose_and_jacobian_bits(
    angle: float, alpha: float,
) -> None:
    left = manifold.se3_exp(SE3.identity(), np.array([0.2, -0.1, 0.3, 0.1, -0.2, 0.05]))
    delta = np.array([0.4, -0.2, 0.3, angle, -angle * 0.2, angle * 0.1])
    # Keep the tested relative rotation below pi, including its coupled axes.
    delta[3:] /= np.linalg.norm([1.0, 0.2, 0.1])
    right = manifold.se3_exp(left, delta)
    segment = manifold._ScrewSegment(left, right)
    expected = _previous_jacobians(left, right, alpha)
    pairs = (segment.jacobians(alpha), manifold.interpolate_screw_jacobians(left, right, alpha))
    for pair in pairs:
        for actual, reference in zip(pair, expected, strict=True):
            _bits(actual, reference)
    assert segment.pose(alpha) == manifold.se3_exp(left, alpha * manifold.se3_log(left, right))


def _knots() -> list[SE3]:
    return [manifold.se3_exp(SE3.identity(), np.array([
        0.15 * t * t, 0.03 * t, 0.1 * np.cos(t),
        0.2 * np.sin(0.7 * t), -0.15 * np.cos(0.5 * t), 0.3 * t,
    ])) for t in range(4)]


@pytest.mark.parametrize("time", [-0.01, 0.0, 0.001, 0.9999, 1.0, 1.0001, 2.999, 3.01])
@pytest.mark.parametrize("mounted", [False, True])
def test_shared_accelerometer_stencils_and_clock_samples_keep_all_bits(
    time: float, mounted: bool,
) -> None:
    knots, timestamps = _knots(), (0.0, 1.0, 2.0, 3.0)
    workspace = sparse._ScrewInterpolation(knots)
    rotation = manifold.rotation_vector_to_matrix(np.array([0.1, -0.2, 0.05])) if mounted else None
    measurement = sparse.TrajectoryImuLeverArmMeasurement("accel", time, (0.1, 0.2, 9.7))
    lever, gravity = np.array([0.1, -0.03, 0.02]), np.array([0.1, 0.0, -9.8])
    bias, scale = np.array([0.01, -0.02, 0.03]), np.array([1.05, 0.98, 1.02])
    for _ in range(2):
        expected = sparse._lever_arm_residual(
            knots, timestamps, measurement, lever, 0.0, bias, gravity, scale,
            imu_rotation=rotation,
        )
        actual = sparse._lever_arm_residual(
            knots, timestamps, measurement, lever, 0.0, bias, gravity, scale,
            imu_rotation=rotation, interpolation=workspace,
        )
        for a, b in zip(actual, expected, strict=True):
            _bits(a, b)
        expected_blocks = sparse._lever_arm_knot_jacobians(
            knots, timestamps, measurement, lever, 0.0, gravity, scale, imu_rotation=rotation,
        )
        actual_blocks = sparse._lever_arm_knot_jacobians(
            knots, timestamps, measurement, lever, 0.0, gravity, scale,
            imu_rotation=rotation, interpolation=workspace,
        )
        for (ai, a), (bi, b) in zip(actual_blocks, expected_blocks, strict=True):
            assert ai == bi
            _bits(a, b)


def test_pose_only_workspace_is_lazy_and_snapshots_immutable_knots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    knots = _knots()
    original = list(knots)
    workspace = sparse._ScrewInterpolation(knots)
    expected = sparse._interpolate(original[0], original[1], 0.25)
    inverse = np.linalg.inv

    def unused(*args: object, **kwargs: object) -> None:
        raise AssertionError("pose-only evaluation must not prepare a Jacobian")

    def pose_inverse(array: NDArray[np.float64]) -> NDArray[np.float64]:
        assert array.shape == (3, 3)  # Log already requires the existing SO(3) inverse.
        return inverse(array)

    monkeypatch.setattr(manifold, "se3_left_jacobian", unused)
    monkeypatch.setattr(np.linalg, "inv", pose_inverse)
    pose = workspace.pose(0, 0.25)
    knots[0] = SE3.identity()
    assert workspace.pose(0, 0.25) == expected
    assert workspace.pose(0, 0.25) is pose


def test_shared_arrays_reject_mutation_and_new_state_uses_fresh_values() -> None:
    knots = _knots()
    old = sparse._ScrewInterpolation(knots)
    left, right = old.jacobians(0, 0.5)
    for array in (left, right, *sparse._body_motion(
        knots, (0.0, 1.0, 2.0, 3.0), 0.5, interpolation=old,
    )):
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = 3.0
    knots[1] = manifold.se3_exp(knots[1], np.array([0.02, 0.0, 0.0, 0.1, 0.0, 0.0]))
    fresh = sparse._ScrewInterpolation(knots)
    assert fresh.pose(0, 0.5) == sparse._interpolate(knots[0], knots[1], 0.5)
    assert fresh.pose(0, 0.5) != old.pose(0, 0.5)
    assert not np.array_equal(fresh.jacobians(0, 0.5)[0], left)


def test_reuse_keys_keep_signed_zero_distinct() -> None:
    knots = _knots()
    workspace = sparse._ScrewInterpolation(knots)
    for alpha in (0.0, -0.0, 0.0, -0.0):
        expected = _previous_jacobians(knots[0], knots[1], alpha)
        for actual, reference in zip(workspace.jacobians(0, alpha), expected, strict=True):
            _bits(actual, reference)
        assert workspace.pose(0, alpha) == sparse._interpolate(knots[0], knots[1], alpha)


@pytest.mark.parametrize("angle", [-math.pi, -1e-8, 0.0, 1e-13, 0.7, math.pi])
def test_fixed_matrix_assembly_matches_general_blocks_in_every_bit(
    angle: float, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(20261020)
    cases = [(rng.normal(size=3), np.array([angle, -0.0, 0.0])) for _ in range(12)]
    actual = [(manifold.se3_left_jacobian(v, w), manifold.se3_adjoint(
        manifold.rotation_vector_to_matrix(w), v,
    )) for v, w in cases]

    def general(a: NDArray[np.float64], b: NDArray[np.float64], c: NDArray[np.float64]):
        return np.block([[a, b], [np.zeros((3, 3), dtype=float), c]])

    monkeypatch.setattr(manifold, "_se3_blocks", general)
    for (v, w), (jacobian, adjoint) in zip(cases, actual, strict=True):
        _bits(jacobian, manifold.se3_left_jacobian(v, w))
        _bits(adjoint, manifold.se3_adjoint(manifold.rotation_vector_to_matrix(w), v))


@pytest.mark.parametrize("kind", ["point", "plane", "pose", "public_plane"])
def test_geometric_rmse_preserves_values_without_discarded_derivatives(
    kind: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    knots, timestamps, time, interval = _knots(), (0.0, 1.0, 2.0, 3.0), 0.4, 0
    point = sparse.TrajectoryPointMeasurement("point", time, (0.2, 0.3, 0.4), (0.1, 0.0, 0.3))
    plane = sparse.TrajectoryPointToPlaneMeasurement(
        "plane", time, point.point_body_m, point.target_world_m, (0.1, 0.2, 1.0),
    )
    pose = sparse.TrajectoryPoseMeasurement("pose", time, knots[2])
    problem = sparse.ContinuousTimeTrajectoryFitProblem(
        timestamps, tuple(knots), point_measurements=(point,),
        point_to_plane_measurements=(plane,), pose_measurements=(pose,),
    )
    if kind == "point":
        residuals = sparse._point_residual(
            knots, timestamps, interval, time, point.point_body_m, point.target_world_m,
        )[0]
    elif kind in ("plane", "public_plane"):
        residuals = sparse._point_to_plane_residual(knots, timestamps, interval, plane)[0]
    else:
        residuals = np.concatenate([sparse._anchor_residual(
            knots, timestamps, interval, time, anchor, pose.pose_world_body,
        )[0] for anchor in sparse._ANCHOR_POINTS])
    # Preserve the original per-anchor reduction order, too.
    squared = (sum(float(np.sum(r * r)) for r in residuals.reshape(-1, 3))
               if kind == "pose" else float(np.sum(residuals * residuals)))
    expected = math.sqrt(squared / len(residuals))

    def unused(*args: object, **kwargs: object) -> None:
        raise AssertionError("RMSE must not construct a derivative")

    monkeypatch.setattr(manifold, "_se3_jacobian_from_exp", unused)
    monkeypatch.setattr(sparse, "point_transform_jacobian", unused)
    if kind == "point":
        actual = sparse._point_rmse(problem, [(interval, point.point_body_m)], knots)
    elif kind == "pose":
        actual = sparse._pose_rmse(problem, [(interval, pose.pose_world_body)], knots)
    elif kind == "public_plane":
        actual = sparse.point_to_plane_rmse((plane,), timestamps, tuple(knots))
    else:
        actual = sparse._point_to_plane_rmse([(interval, plane)], timestamps, knots)
    assert actual is not None
    _bits(np.array([actual]), np.array([expected]))


@pytest.mark.parametrize("window", [(0.2, 0.8), (1.0, 2.0), (2.1, 2.8)])
@pytest.mark.parametrize("mounted", [False, True])
def test_shared_gyro_samples_preserve_boundary_and_nonunit_scale_bits(
    window: tuple[float, float], mounted: bool,
) -> None:
    knots, timestamps = _knots(), (0.0, 1.0, 2.0, 3.0)
    workspace = sparse._ScrewInterpolation(knots)
    measurement = sparse.TrajectoryImuPreintegrationMeasurement("gyro", tuple(
        sparse.TrajectoryImuGyroSample(float(t), (0.2, -0.1, 0.3))
        for t in np.linspace(*window, 11)
    ))
    rotation = manifold.rotation_vector_to_matrix(np.array([0.1, -0.2, 0.05])) if mounted else None
    bias, scale = np.array([0.01, -0.02, 0.03]), np.array([1.05, 0.98, 1.02])
    expected = sparse._imu_preintegration_residual(
        knots, timestamps, measurement, bias, 0.0, scale, imu_rotation=rotation,
    )
    for _ in range(2):
        actual = sparse._imu_preintegration_residual(
            knots, timestamps, measurement, bias, 0.0, scale,
            imu_rotation=rotation, interpolation=workspace,
        )
        for index in (0, 2, 3, 4):
            _bits(actual[index], expected[index])
        for (ai, a), (bi, b) in zip(actual[1], expected[1], strict=True):
            assert ai == bi
            _bits(a, b)


@pytest.mark.parametrize("damping", [1e-3, 1e3])
def test_complete_fit_matches_without_workspace_across_state_updates(
    damping: float, monkeypatch: pytest.MonkeyPatch,
) -> None:
    knots = _knots()
    timestamps = (0.0, 1.0, 2.0, 3.0)
    truth = tuple(SE3.identity() for _ in timestamps)
    problem = sparse.ContinuousTimeTrajectoryFitProblem(
        timestamps, tuple(knots),
        point_measurements=tuple(sparse.TrajectoryPointMeasurement(
            f"point-{i}", time, (0.2, 0.3, -0.1), (0.2, 0.3, -0.1), weight=2.0,
        ) for i, time in enumerate([0.0, 0.5, 1.0, 1.5, 3.0])),
        pose_measurements=tuple(sparse.TrajectoryPoseMeasurement(
            f"pose-{i}", time, pose, weight=3.0,
        ) for i, (time, pose) in enumerate(zip(timestamps, truth, strict=True))),
    )
    options = sparse.ContinuousTimeTrajectoryFitOptions(max_iterations=8, initial_damping=damping)
    actual = sparse.fit_continuous_trajectory(problem, options)
    assert actual.iterations > 1
    monkeypatch.setattr(sparse, "_ScrewInterpolation", lambda knots: None)
    expected = sparse.fit_continuous_trajectory(problem, options)
    assert dataclasses.asdict(actual) == dataclasses.asdict(expected)
