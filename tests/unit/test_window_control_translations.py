"""Fault-injection translations must not broadcast or fail after fitting."""
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


@pytest.mark.parametrize('field', ['corrupt_marginalization_translation_m',
                                   'bad_overlap_translation_m'])
@pytest.mark.parametrize('value', [.1, (.1,), (.1, .2), (1., 2., 3., 4.),
                                  [[1., 2., 3.]], [1., [2., 3.]], (float('nan'), 0., 0.),
                                  (0., float('inf'), 0.), (0., 0., -float('inf')),
                                  (1j, 0., 0.), ('invalid', 0., 0.), (10**500, 0., 0.)])
def test_invalid_controls_fail_before_any_fit(monkeypatch, field, value):
    def unexpected(*args, **kwargs):
        pytest.fail('malformed control reached the fitter')

    monkeypatch.setattr(window, 'fit_continuous_trajectory', unexpected)
    with pytest.raises(ValueError, match=field):
        window.fit_sliding_window_trajectory(problem(), window_knot_counts=(2, 2),
                                             overlap_knot_count=1, **{field: value})


@pytest.mark.parametrize('square_root', [False, True])
@pytest.mark.parametrize('field', ['corrupt_marginalization_translation_m',
                                   'bad_overlap_translation_m'])
def test_valid_controls_reach_intended_knots_in_both_prior_modes(monkeypatch, square_root, field):
    captured = []
    fit = window.fit_continuous_trajectory

    def capture(problem, options):
        captured.append(problem)
        return fit(problem, options)

    monkeypatch.setattr(window, 'fit_continuous_trajectory', capture)
    translation = np.array([.1, -.2, .3])
    window.fit_sliding_window_trajectory(
        problem(), window_knot_counts=(2, 2), overlap_knot_count=1,
        square_root_marginalization=square_root, **{field: translation},
    )
    assert len(captured) == 3  # Batch reference, first window, second window.
    second = captured[-1]
    if field == 'bad_overlap_translation_m':
        np.testing.assert_array_equal(second.initial_knot_poses[0].translation_m, translation)
        assert second.initial_knot_poses[1] == SE3.identity()
        assert second.joint_knot_marginalization_priors == ()
    else:
        prior = second.joint_knot_marginalization_priors[0]
        np.testing.assert_array_equal(prior.anchor_poses[0].translation_m, translation)
        assert second.initial_knot_poses == (SE3.identity(),) * 2
    np.testing.assert_array_equal(translation, [.1, -.2, .3])
