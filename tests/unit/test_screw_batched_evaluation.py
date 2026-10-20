"""Bit equality of complete batched geometry and fresh fixed-state preparation."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from calibrex.core import _se3_batch as batch
from calibrex.core import continuous_time_sparse as sparse
from calibrex.core import se3_manifold as manifold
from calibrex.core.geometry import SE3


def _bits(actual, expected):
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint64),
                                  np.asarray(expected).view(np.uint64))


def _knots():
    return [manifold.se3_exp(SE3.identity(), np.array([
        0.15*t*t, 0.03*t, 0.1*np.cos(t),
        0.2*np.sin(0.7*t), -0.15*np.cos(0.5*t), 0.3*t,
    ])) for t in range(4)]


@pytest.mark.parametrize("size", [0, 1, 7, 257, 513])
@pytest.mark.parametrize("angle", [0.0, 1e-13, 1e-9, 0.5, math.pi - 1e-8])
def test_batched_segment_keeps_every_pose_and_jacobian_bit(size, angle):
    rng = np.random.default_rng(20261020)
    left = manifold.se3_exp(SE3.identity(), rng.normal(size=6)*0.1)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    right = manifold.se3_exp(left, np.r_[rng.normal(size=3), axis*angle])
    alphas = np.resize(np.array([-0.1, -0.0, 0.0, 0.25, 1.0, 1.1]), size)
    actual = manifold._ScrewSegment(left, right)
    reference = manifold._ScrewSegment(left, right)
    poses, blocks = batch._sample_segment(actual, alphas, jacobians=True)
    assert blocks is not None
    assert len(poses) == size
    for index, alpha in enumerate(alphas):
        assert poses[index] == reference.pose(float(alpha))
        expected = reference.jacobians(float(alpha))
        _bits(blocks[0][index], expected[0])
        _bits(blocks[1][index], expected[1])
    if not size:
        assert actual._jacobian_factors is None


@pytest.mark.parametrize("size", [0, 1, 7, 513])
@pytest.mark.parametrize("strided", [False, True])
def test_batched_points_and_point_jacobians_preserve_scalar_bits(size, strided):
    rng = np.random.default_rng(20261020)
    poses = [manifold.se3_exp(SE3.identity(), rng.normal(size=6)) for _ in range(size)]
    points = rng.normal(size=(size, 3))
    if size:
        points[0] = (0.0, -0.0, 0.0)
    if strided:
        wide = np.zeros((size, 6))
        wide[:, ::2] = points
        points = wide[:, ::2]
    expected = np.asarray([pose.transform_point(point)
                           for pose, point in zip(poses, points, strict=True)]).reshape(-1, 3)
    _bits(batch._transform_points(poses, points), expected)
    expected_j = np.asarray([manifold.point_transform_jacobian(pose, point)
                             for pose, point in zip(poses, points, strict=True)]).reshape(-1, 3, 6)
    _bits(batch._point_jacobians(poses, points), expected_j)


def _planes(size):
    times = np.linspace(0.0, 3.0, size)
    rng = np.random.default_rng(20261020)
    return [(
        sparse._interval_index((0.0, 1.0, 2.0, 3.0), float(t)),
        sparse.TrajectoryPointToPlaneMeasurement(
            str(i), float(t), tuple(rng.normal(size=3)), tuple(rng.normal(size=3)),
            tuple(rng.normal(size=3)*3.0), weight=(0.3, 1.0, 2.5)[i % 3],
        ),
    ) for i, t in enumerate(times)]


@pytest.mark.parametrize("size", [0, 7, 8, 193, 513])
def test_all_plane_values_and_sparse_blocks_keep_bits_across_batch_boundaries(size):
    knots, timestamps = _knots(), (0.0, 1.0, 2.0, 3.0)
    rows = _planes(size)
    planes = sparse._PlaneBatch(timestamps, rows)
    workspace = sparse._ScrewInterpolation(knots)
    planes.prepare(workspace, jacobians=True)
    residuals, left, right = planes.linearize(knots, workspace)
    for index, (interval, measurement) in enumerate(rows):
        expected = sparse._point_to_plane_residual(knots, timestamps, interval, measurement)
        _bits(np.array([residuals[index]]), expected[0])
        _bits(left[index], expected[1])
        _bits(right[index], expected[2])
    if size:
        with pytest.raises(ValueError, match="read-only"):
            planes.normals.flat[0] = 3.0


def test_value_preparation_is_lazy_and_shared_arrays_are_read_only(monkeypatch):
    knots = _knots()
    workspace = sparse._ScrewInterpolation(knots)

    def unused(*args, **kwargs):
        raise AssertionError("pose-only batches must not prepare interval Jacobians")

    monkeypatch.setattr(manifold._ScrewSegment, "_factors", unused)
    requests = [(0, float(alpha), False) for alpha in np.linspace(0, 1, 513)]
    workspace.prepare(requests)
    assert not workspace._jacobians
    assert all(segment._jacobian_factors is None for segment in workspace._segments.values())
    for alpha in np.linspace(0, 1, 513):
        expected = sparse._interpolate(knots[0], knots[1], float(alpha))
        assert workspace.pose(0, float(alpha)) == expected


def test_preparation_retains_signed_zero_and_refreshes_after_knot_updates():
    knots = _knots()
    workspace = sparse._ScrewInterpolation(knots)
    queries = [(0, float(a), True) for a in np.r_[[-0.0, 0.0], np.linspace(.1, .9, 15)]]
    workspace.prepare(queries)
    assert len(workspace._poses) == len(queries)
    cached = workspace.pose(0, .5)
    for left, right in workspace._jacobians.values():
        for array in (left, right):
            with pytest.raises(ValueError, match="read-only"):
                array.flat[0] = 1.0
    knots[1] = manifold.se3_exp(knots[1], np.array([.1, .2, .3, -.1, .2, -.1]))
    fresh = sparse._ScrewInterpolation(knots)
    fresh.prepare(queries)
    assert workspace.pose(0, .5) == cached
    assert fresh.pose(0, .5) != cached


@pytest.mark.parametrize("damping", [1e-3, 1e3])
def test_complete_fit_and_sparse_matrices_equal_scalar_evaluation(damping, monkeypatch):
    knots, timestamps, rows = _knots(), (0.0, 1.0, 2.0, 3.0), _planes(193)
    problem = sparse.ContinuousTimeTrajectoryFitProblem(
        timestamps, tuple(knots), point_to_plane_measurements=tuple(m for _i, m in rows),
        pose_measurements=tuple(sparse.TrajectoryPoseMeasurement(
            str(i), time, SE3.identity(), weight=3.0,
        ) for i, time in enumerate(timestamps)),
        imu_preintegration_measurements=(sparse.TrajectoryImuPreintegrationMeasurement(
            "gyro", tuple(sparse.TrajectoryImuGyroSample(t, (.2, -.1, .3))
                          for t in np.linspace(.2, 2.8, 31)),
        ),),
        imu_lever_arm_measurements=tuple(sparse.TrajectoryImuLeverArmMeasurement(
            str(t), t, (.1, .2, 9.7), weight=.3,
        ) for t in (.4, 1.0, 2.6)),
        estimate_gyro_bias=True, estimate_imu_rotation=True,
        initial_imu_rotation_vector_rad=(.1, -.2, .05),
    )
    settings = sparse.ContinuousTimeTrajectoryFitOptions(max_iterations=4, initial_damping=damping)
    matrices = []
    original = sparse._assemble_sparse_jacobian

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        matrices.append((result.data.copy(), result.indices.copy(), result.indptr.copy()))
        return result

    monkeypatch.setattr(sparse, "_assemble_sparse_jacobian", capture)
    actual = sparse.fit_continuous_trajectory(problem, settings)
    actual_matrices = list(matrices)
    matrices.clear()
    monkeypatch.setattr(sparse._ScrewInterpolation, "prepare", lambda *args, **kwargs: None)
    monkeypatch.setattr(sparse, "_plane_batch", lambda *args: None)
    expected = sparse.fit_continuous_trajectory(problem, settings)
    assert dataclasses.asdict(actual) == dataclasses.asdict(expected)
    assert len(matrices) == len(actual_matrices)
    for actual_model, expected_model in zip(actual_matrices, matrices, strict=True):
        for a, b in zip(actual_model, expected_model, strict=True):
            np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))
