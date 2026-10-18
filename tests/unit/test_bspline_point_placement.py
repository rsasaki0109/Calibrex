"""Pose-only placement preserves the public evaluator without caching mutable knots."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibrex.core.continuous_time_bspline import CubicBSplineTrajectory


@pytest.mark.parametrize("angle", [0.0, 1e-10, 0.2, np.pi - 1e-7])
@pytest.mark.parametrize("count", [0, 1, 257])
def test_point_placement_matches_full_states(angle: float, count: int) -> None:
    rng = np.random.default_rng(52)
    vectors = rng.normal(size=(12, 3))
    vectors *= angle / np.maximum(np.linalg.norm(vectors, axis=1), 1e-30)[:, None]
    trajectory = CubicBSplineTrajectory(
        7.0,
        0.1,
        Rotation.from_rotvec(vectors).as_matrix(),
        rng.normal(size=(12, 3)),
    )
    times = rng.uniform(7.0, trajectory.end_time_sec, count)
    if count:
        times[: min(2, count)] = [7.0, trajectory.end_time_sec][: min(2, count)]
    # Noncontiguous, repeated and out-of-order inputs are valid.
    backing = rng.normal(size=(count, 6))
    points = backing[:, ::2]
    states = trajectory.evaluate(times)
    expected = np.einsum("tij,tj->ti", states.rotation_world_body, points)
    expected += states.position_world_m
    np.testing.assert_array_equal(trajectory.transform_points(times, points), expected)
    np.testing.assert_array_equal(points, backing[:, ::2])
    trajectory.positions[2] += 0.7
    trajectory.rotations[1] = Rotation.from_rotvec([0.03, -0.05, 0.1]).as_matrix()
    changed = trajectory.evaluate(times)
    expected = np.einsum("tij,tj->ti", changed.rotation_world_body, points)
    expected += changed.position_world_m
    np.testing.assert_array_equal(trajectory.transform_points(times, points), expected)


@pytest.mark.parametrize(
    ("times", "points", "message"),
    [
        ([0.1], [[1.0, 2.0]], "points"),
        ([0.1], [[1.0, np.nan, 3.0]], "points"),
        ([np.inf], [[1.0, 2.0, 3.0]], "timestamps"),
        ([np.nan], [[1.0, 2.0, 3.0]], "timestamps"),
        ([-0.1], [[1.0, 2.0, 3.0]], "domain"),
        ([0.4], [[1.0, 2.0, 3.0]], "domain"),
    ],
)
def test_point_placement_validates_domain_and_dimensions(
    times: list[float],
    points: list[list[float]],
    message: str,
) -> None:
    trajectory = CubicBSplineTrajectory(0.0, 0.1, np.tile(np.eye(3), (6, 1, 1)), np.zeros((6, 3)))
    with pytest.raises(ValueError, match=message):
        trajectory.transform_points(np.array(times), np.array(points))
