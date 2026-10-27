"""Reject invalid window layouts before the expensive batch reference fit."""
import numpy as np
import pytest

import calibrex.core.continuous_time_sliding_window_fit as window
from calibrex.core.continuous_time_sparse import (
    ContinuousTimeTrajectoryFitProblem,
    TrajectoryPoseMeasurement,
)
from calibrex.core.geometry import SE3


def problem():
    return ContinuousTimeTrajectoryFitProblem(
        knot_timestamps=(0., 1., 2.), initial_knot_poses=(SE3.identity(),) * 3,
        pose_measurements=tuple(TrajectoryPoseMeasurement(str(t), t, SE3.identity())
                                for t in (0., 1., 2.)),
    )


@pytest.mark.parametrize('counts,overlap', [
    ((2., 2.), 1), ((2, 2), 1.), ((2, 2), True), ((True, 3), 1),
    ((2, 2), np.bool_(True)), ((np.bool_(True), 3), 1),
    ((2, float('nan')), 1), ((2, float('inf')), 1),
    ((2, 2), float('nan')), ((2, 2), float('inf')),
    ((2, 2), 0), ((2, 2), -1), ((0, 4), 1), ((-1, 5), 1),
    ((2, 2), 2), ((2,), 1), ((), 1), ((3, 3), 1),
    ((np.uint64(2**64-1), np.uint64(2**64-1)), 1),
    ((10**500, 10**500), 1),
])
def test_invalid_layout_is_rejected_before_any_fit(monkeypatch, counts, overlap):
    def unexpected_fit(*args, **kwargs):
        pytest.fail('invalid layout reached the batch fitter')

    monkeypatch.setattr(window, 'fit_continuous_trajectory', unexpected_fit)
    with pytest.raises(ValueError):
        window.fit_sliding_window_trajectory(problem(), window_knot_counts=counts,
                                             overlap_knot_count=overlap)


@pytest.mark.parametrize('square_root', [False, True])
def test_numpy_integer_layout_has_same_results_as_python_integers(square_root):
    expected = window.fit_sliding_window_trajectory(
        problem(), window_knot_counts=(2, 2), overlap_knot_count=1,
        square_root_marginalization=square_root,
    )
    actual = window.fit_sliding_window_trajectory(
        problem(), window_knot_counts=(np.int64(2), np.uint64(2)),
        overlap_knot_count=np.int64(1), square_root_marginalization=square_root,
    )
    assert actual.sliding_result.status == expected.sliding_result.status == 'converged'
    assert actual.sliding_result.knot_poses == expected.sliding_result.knot_poses
    assert actual.window_count == expected.window_count == 2
    assert actual.sliding_result.final_objective == expected.sliding_result.final_objective
