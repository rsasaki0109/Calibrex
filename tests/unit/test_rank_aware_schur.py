"""Generalized Schur complements must agree with residual-space elimination."""
from __future__ import annotations

import numpy as np
import pytest

from calibrex.core.continuous_time_marginalization import schur_complement_normal_with_evidence


def test_coupled_nullspace_matches_residual_projection_and_records_rank():
    rng = np.random.default_rng(164)
    a = rng.normal(size=(15, 2))
    eliminated = np.column_stack((a, a[:, 0] + 2 * a[:, 1]))
    retained = rng.normal(size=(15, 3))
    j = np.column_stack((eliminated, retained))
    residual = rng.normal(size=15)
    h, g, evidence = schur_complement_normal_with_evidence(j.T @ j, j.T @ residual, 3)
    projector = np.eye(15) - eliminated @ np.linalg.pinv(eliminated)
    np.testing.assert_allclose(h, retained.T @ projector @ retained, atol=1e-12)
    np.testing.assert_allclose(g, retained.T @ projector @ residual, atol=1e-12)
    assert evidence.numerical_rank == 2
    assert evidence.coupled_nullity == 1
    assert evidence.zero_information_dimension == 0
    assert evidence.max_relative_range_error <= evidence.relative_range_tolerance
    for x in rng.normal(size=(4, 3)):
        shifted = residual + retained @ x
        solved = np.linalg.lstsq(eliminated, -shifted, rcond=None)[0]
        cost = np.linalg.norm(shifted + eliminated @ solved)**2
        base = np.linalg.norm(projector @ residual)**2
        assert cost - base == pytest.approx(x @ h @ x + 2 * g @ x, abs=1e-12)


@pytest.mark.parametrize("bad_gradient", [False, True])
def test_nullspace_coupling_and_unbounded_linear_term_are_rejected(bad_gradient):
    h = np.array([[1., 1., .5], [1., 1., .5], [.5, .5, 2.]])
    g = np.zeros(3)
    if bad_gradient:
        g[:2] = [1., -1.]
    else:
        h[0, 2] = h[2, 0] = .7
    with pytest.raises(ValueError, match="outside the numerical range"):
        schur_complement_normal_with_evidence(h, g, 2)


def test_independent_coordinate_unit_changes_preserve_retained_information():
    h = np.array([[1., 1., .5], [1., 1., .5], [.5, .5, 2.]])
    g = np.array([.2, .2, .3])
    units = np.array([1e-12, 1e12, 1.])
    reference = schur_complement_normal_with_evidence(h, g, 2)
    actual = schur_complement_normal_with_evidence(h * units[:, None] * units[None, :],
                                                  g * units, 2)
    np.testing.assert_allclose(actual[0], reference[0], atol=1e-14)
    np.testing.assert_allclose(actual[1], reference[1], atol=1e-14)
    assert actual[2].numerical_rank == reference[2].numerical_rank == 1


def test_weak_coupled_positive_direction_with_material_rhs_is_not_silently_discarded():
    h = np.eye(3)
    h[:2, :2] = [[1., 1. - 1e-14], [1. - 1e-14, 1.]]
    with pytest.raises(ValueError, match="outside the numerical range"):
        schur_complement_normal_with_evidence(h, np.array([1., -1., 0.]), 2)


@pytest.mark.parametrize("eliminated_count", [1, 2])
def test_subnormal_information_keeps_representable_schur_result(eliminated_count):
    # The eliminated solution exceeds float64, but its contribution to the
    # retained quadratic does not. Two identical columns exercise coupled nulls.
    h = np.full((eliminated_count + 1, eliminated_count + 1), 1e-320)
    h[-1, :-1] = h[:-1, -1] = 5e-161
    h[-1, -1] = 1.
    g = np.ones(eliminated_count + 1)
    g[-1] = 0.
    actual_h, actual_g, evidence = schur_complement_normal_with_evidence(
        h, g, eliminated_count,
    )
    # Change units so that eliminated coordinates have unit information. This
    # independent equivalent system has no overflowing solve intermediates.
    units = np.ones(eliminated_count + 1)
    units[:-1] = 1e160
    reference_h, reference_g, _ = schur_complement_normal_with_evidence(
        h * units[:, None] * units[None, :], g * units, eliminated_count,
    )
    np.testing.assert_allclose(actual_h, reference_h, rtol=1e-14)
    np.testing.assert_allclose(actual_g, reference_g, rtol=1e-14)
    # Analytic one-dimensional quotient, ordered to avoid intermediate overflow.
    contribution = (h[0, -1] / np.sqrt(h[0, 0])) ** 2
    np.testing.assert_allclose(actual_h, [[1. - contribution]], rtol=1e-14)
    np.testing.assert_allclose(actual_g, [-contribution / h[0, -1]], rtol=1e-14)
    assert evidence.numerical_rank == 1
    assert evidence.coupled_nullity == eliminated_count - 1


def test_nonrepresentable_retained_gradient_is_rejected():
    # A valid SPD quadratic with finite inputs can still overflow its Schur RHS.
    h = np.array([[1., 1e154], [1e154, 1.5e308]])
    with pytest.raises(ValueError, match="marginalized normal equations must be finite"):
        schur_complement_normal_with_evidence(h, np.array([1e200, 0.]), 1)


def test_sliding_eliminates_coupled_unobservable_old_knots_without_a_pose_prior():
    from calibrex.core.continuous_time_sliding_window_fit import fit_sliding_window_trajectory
    from calibrex.core.continuous_time_sparse import (
        ContinuousTimeTrajectoryFitProblem,
        TrajectoryPoseMeasurement,
    )
    from calibrex.core.geometry import SE3

    target = SE3.identity()
    initial = SE3((.2, -.1, .3), (0., 0., 0., 1.))
    # The midpoint observes a combination of knots 0 and 1, not each separately.
    problem = ContinuousTimeTrajectoryFitProblem(
        knot_timestamps=(0., 1., 2., 3.), initial_knot_poses=(initial,) * 4,
        pose_measurements=tuple(TrajectoryPoseMeasurement(str(t), t, target)
                                for t in (.5, 2., 3.)),
    )
    result = fit_sliding_window_trajectory(problem, window_knot_counts=(3, 2), overlap_knot_count=1)
    evidence = result.window_observability[0].elimination
    assert evidence is not None
    assert evidence.eliminated_dimension == 12
    assert evidence.numerical_rank == 6
    assert evidence.coupled_nullity == 6
    assert evidence.zero_information_dimension == 0
    assert result.sliding_result.final_objective < 1e-12
    for knot in result.sliding_result.knot_poses[2:]:
        np.testing.assert_allclose(knot.translation_m, target.translation_m, atol=1e-7)
