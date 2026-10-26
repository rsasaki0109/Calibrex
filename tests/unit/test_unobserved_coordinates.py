"""Exactly missing information must not prevent observed coordinates from updating."""
from __future__ import annotations

import warnings

import numpy as np
from scipy.sparse.linalg import MatrixRankWarning

from calibrex.core.continuous_time_sparse import (
    ContinuousTimeTrajectoryFitProblem,
    TrajectoryPoseMeasurement,
    fit_continuous_trajectory,
)
from calibrex.core.geometry import SE3


def test_unobserved_knot_is_unchanged_while_observed_knot_recovers():
    initial = SE3((.3, -.2, .1), (0., 0., 0., 1.))
    unsupported = SE3((5., 6., 7.), (0., 0., 0., 1.))
    target = SE3.identity()
    problem = ContinuousTimeTrajectoryFitProblem(
        knot_timestamps=(0., 1.), initial_knot_poses=(initial, unsupported),
        pose_measurements=(TrajectoryPoseMeasurement("observed", 0., target),),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        result = fit_continuous_trajectory(problem)
    assert result.status == "converged"
    assert result.diagnostics.stationary
    assert result.frozen_parameter_indices == tuple(range(6, 12))
    np.testing.assert_allclose(result.knot_poses[0].translation_m, target.translation_m, atol=1e-7)
    assert result.knot_poses[1] == unsupported
    assert result.final_objective < 1e-14


def test_zero_residual_still_reports_missing_information():
    pose = SE3.identity()
    result = fit_continuous_trajectory(ContinuousTimeTrajectoryFitProblem(
        knot_timestamps=(0., 1.), initial_knot_poses=(pose, pose),
        pose_measurements=(TrajectoryPoseMeasurement("observed", 0., pose),),
    ))
    assert result.status == "converged"
    assert result.frozen_parameter_indices == tuple(range(6, 12))


def test_weak_nonzero_information_is_not_classified_as_missing():
    pose = SE3.identity()
    result = fit_continuous_trajectory(ContinuousTimeTrajectoryFitProblem(
        knot_timestamps=(0., 1.), initial_knot_poses=(pose, pose),
        pose_measurements=(TrajectoryPoseMeasurement("weak", 0., pose, weight=1e-14),),
    ))
    assert result.frozen_parameter_indices == tuple(range(6, 12))


def test_sliding_freeze_history_uses_global_parameter_indices():
    from calibrex.core.continuous_time_sliding_window_fit import fit_sliding_window_trajectory

    initial = SE3((.2, 0., 0.), (0., 0., 0., 1.))
    target = SE3.identity()
    problem = ContinuousTimeTrajectoryFitProblem(
        knot_timestamps=(0., 1., 2., 3.), initial_knot_poses=(initial,) * 4,
        pose_measurements=tuple(TrajectoryPoseMeasurement(str(t), t, target) for t in (0., 1., 3.)),
    )
    result = fit_sliding_window_trajectory(problem, window_knot_counts=(3, 2), overlap_knot_count=1)
    assert result.sliding_result.frozen_parameter_indices == tuple(range(12, 18))
    assert result.window_results[1].frozen_parameter_indices == tuple(range(6))
    assert result.sliding_result.knot_poses[2] == initial
    assert result.window_observability[1].frozen_parameters == [
        f"knot[2].{axis}" for axis in ("tx", "ty", "tz", "rx", "ry", "rz")
    ]


def test_screw_endpoint_jacobians_have_exact_support_in_scalar_and_batch_paths():
    from calibrex.core._se3_batch import _sample_segment
    from calibrex.core.se3_manifold import _ScrewSegment, se3_exp

    segment = _ScrewSegment(SE3.identity(), se3_exp(
        SE3.identity(), np.array([.3, -.2, .5, .2, .4, -.1]),
    ))
    _, pair = _sample_segment(segment, np.array([0., 1., .5]), jacobians=True)
    assert pair is not None
    for index, alpha in enumerate((0., 1.)):
        left, right = segment.jacobians(alpha)
        expected_left = np.eye(6) if alpha == 0. else np.zeros((6, 6))
        expected_right = np.eye(6) if alpha == 1. else np.zeros((6, 6))
        np.testing.assert_array_equal(left, expected_left)
        np.testing.assert_array_equal(right, expected_right)
        np.testing.assert_array_equal(pair[0][index], expected_left)
        np.testing.assert_array_equal(pair[1][index], expected_right)
    assert np.linalg.norm(pair[0][2]) > 0
    assert np.linalg.norm(pair[1][2]) > 0
