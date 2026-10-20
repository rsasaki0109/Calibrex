"""Exact arithmetic and score-only regression checks for sparse screw fitting."""

from __future__ import annotations

import math

import numpy as np
import pytest

from calibrex.core import continuous_time_sparse as sparse
from calibrex.core import se3_manifold as manifold
from calibrex.core.geometry import SE3


@pytest.mark.parametrize(
    "angle",
    [
        -math.pi,
        -1.0,
        -1.0e-8,
        -1.0e-12,
        -1.0e-13,
        -0.0,
        0.0,
        1.0e-300,
        1.0e-13,
        1.0e-12,
        1.0e-8,
        1.0,
        math.pi - 1.0e-9,
        math.pi,
    ],
)
def test_se3_jacobian_bits_match_general_cross(
    angle: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(20261020)
    cases = []
    for _ in range(12):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        cases.append((rng.normal(size=3), axis * angle))
    cases.append((np.array([-0.0, 0.0, -0.0]), np.array([angle, -0.0, 0.0])))
    actual = [manifold.se3_left_jacobian(v, w) for v, w in cases]
    # Restore NumPy's general cross at the private boundary: every other
    # operation is the pre-change formula, including its small-angle branch.
    monkeypatch.setattr(manifold, "_cross3", np.cross)
    for (v, w), result in zip(cases, actual, strict=True):
        expected = manifold.se3_left_jacobian(v, w)
        np.testing.assert_array_equal(result.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize("alpha", [0.0, 1.0e-13, 0.4, 1.0 - 1.0e-13, 1.0])
def test_objective_scores_weighted_mixed_factors_without_jacobians(
    alpha: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = manifold.se3_exp(SE3.identity(), np.array([0.2, -0.1, 0.3, 0.1, -0.2, 0.05]))
    right = manifold.se3_exp(left, np.array([0.4, 0.2, -0.3, -0.03, 0.08, 0.1]))
    knots = [left, right]
    timestamps = (2.0, 4.0)
    timestamp = 2.0 + 2.0 * alpha
    point = sparse.TrajectoryPointMeasurement(
        "point",
        timestamp,
        (0.1, -0.7, 0.0),
        (0.3, -0.0, 1.2),
        weight=2.3,
    )
    plane = sparse.TrajectoryPointToPlaneMeasurement(
        "plane",
        timestamp,
        (1.0, -2.0, 0.2),
        (0.3, 0.4, -0.5),
        (0.1, -0.3, 1.7),
        weight=0.7,
    )
    pose = sparse.TrajectoryPoseMeasurement("pose", timestamp, right, weight=3.1)
    problem = sparse.ContinuousTimeTrajectoryFitProblem(
        timestamps,
        tuple(knots),
        point_measurements=(point,),
        point_to_plane_measurements=(plane,),
        pose_measurements=(pose,),
    )
    point_residual = sparse._point_residual(
        knots,
        timestamps,
        0,
        timestamp,
        point.point_body_m,
        point.target_world_m,
    )[0]
    plane_residual = sparse._point_to_plane_residual(knots, timestamps, 0, plane)[0]
    anchors = [
        sparse._anchor_residual(knots, timestamps, 0, timestamp, anchor, right)[0]
        for anchor in sparse._ANCHOR_POINTS
    ]
    expected = 0.0
    expected += point.weight**2 * float(np.sum(point_residual * point_residual))
    expected += plane.weight**2 * float(plane_residual[0] * plane_residual[0])
    for residual in anchors:
        expected += pose.weight**2 * float(np.sum(residual * residual))

    def unused(*args: object, **kwargs: object) -> None:
        raise AssertionError("scoring must not compute geometric Jacobians")

    for name in (
        "point_transform_jacobian",
        "interpolate_screw_jacobians",
        "_point_residual",
        "_point_to_plane_residual",
        "_anchor_residual",
    ):
        monkeypatch.setattr(sparse, name, unused)
    zeros, ones = np.zeros(3), np.ones(3)
    actual = sparse._objective(
        problem,
        [(0, point.point_body_m)],
        [(0, plane)],
        [(0, right)],
        [],
        [],
        knots,
        zeros,
        zeros,
        0.0,
        zeros,
        zeros,
        ones,
        ones,
    )
    assert np.float64(actual).view(np.uint64) == np.float64(expected).view(np.uint64)


@pytest.mark.parametrize("kind", ["point", "plane", "pose"])
def test_fast_scoring_keeps_outside_domain_rejection(kind: str) -> None:
    values: dict = {}
    if kind == "point":
        values["point_measurements"] = (
            sparse.TrajectoryPointMeasurement("outside", 2.1, (0.0,) * 3, (0.0,) * 3),
        )
    elif kind == "plane":
        values["point_to_plane_measurements"] = (
            sparse.TrajectoryPointToPlaneMeasurement(
                "outside",
                2.1,
                (0.0,) * 3,
                (0.0,) * 3,
                (0.0, 0.0, 1.0),
            ),
        )
    else:
        values["pose_measurements"] = (
            sparse.TrajectoryPoseMeasurement("outside", 2.1, SE3.identity()),
        )
    problem = sparse.ContinuousTimeTrajectoryFitProblem(
        (0.0, 1.0),
        (SE3.identity(), SE3.identity()),
        **values,
    )
    with pytest.raises(ValueError, match="outside the knot domain"):
        sparse.fit_continuous_trajectory(problem)
