"""Eliminating absent information must not manufacture a constraint."""
from __future__ import annotations

import numpy as np
import pytest

from calibrex.core.continuous_time_marginalization import schur_complement_normal
from calibrex.core.continuous_time_sliding_window_fit import fit_sliding_window_trajectory
from calibrex.core.continuous_time_sparse import (
    ContinuousTimeTrajectoryFitProblem,
    TrajectoryPoseMeasurement,
)
from calibrex.core.geometry import SE3


def test_schur_with_absent_coordinate_matches_independent_residual_profile():
    rng = np.random.default_rng(62)
    jacobian = rng.normal(size=(20, 5))
    jacobian[:, 1] = 0.
    residual = rng.normal(size=20)
    h = jacobian.T @ jacobian
    g = jacobian.T @ residual
    reduced_h, reduced_g = schur_complement_normal(h, g, 3)
    # Profile directly in residual space, independently of normal-equation solve.
    eliminated, retained = jacobian[:, :3], jacobian[:, 3:]
    projection = np.eye(20) - eliminated @ np.linalg.pinv(eliminated)
    np.testing.assert_allclose(reduced_h, retained.T @ projection @ retained, atol=1e-12)
    np.testing.assert_allclose(reduced_g, retained.T @ projection @ residual, atol=1e-12)
    for x in (np.array([.1, -.2]), np.array([-.3, .4])):
        shifted = residual + retained @ x
        e = np.linalg.lstsq(eliminated, -shifted, rcond=None)[0]
        profiled = np.linalg.norm(shifted + eliminated @ e)**2
        base = np.linalg.norm(projection @ residual)**2
        assert profiled - base == pytest.approx(x @ reduced_h @ x + 2 * reduced_g @ x)


def test_entirely_absent_prefix_leaves_retained_information_unchanged():
    h = np.diag([0., 0., 3., 4.])
    g = np.array([0., 0., .2, -.3])
    reduced_h, reduced_g = schur_complement_normal(h, g, 2)
    np.testing.assert_array_equal(reduced_h, h[2:, 2:])
    np.testing.assert_array_equal(reduced_g, g[2:])


@pytest.mark.parametrize("bad_coupling,bad_gradient", [(True, False), (False, True)])
def test_missing_information_with_coupling_or_gradient_is_rejected(bad_coupling, bad_gradient):
    h = np.diag([0., 1., 2.])
    g = np.zeros(3)
    if bad_coupling:
        h[0, 2] = h[2, 0] = 1e-20
    if bad_gradient:
        g[0] = 1e-20
    with pytest.raises(ValueError, match="zero coupling and gradient"):
        schur_complement_normal(h, g, 2)


def test_tiny_positive_information_is_eliminated_without_a_cutoff():
    h = np.array([[1e-28, 0., 1e-14], [0., 0., 0.], [1e-14, 0., 2.]])
    g = np.array([1e-14, 0., 3.])
    reduced_h, reduced_g = schur_complement_normal(h, g, 2)
    np.testing.assert_allclose(reduced_h, [[1.]])
    np.testing.assert_allclose(reduced_g, [2.])


def test_coupled_singular_active_block_adds_no_artificial_information():
    h = np.ones((3, 3))
    retained, gradient = schur_complement_normal(h, np.zeros(3), 2)
    np.testing.assert_allclose(retained, np.zeros((1, 1)), atol=1e-14)
    np.testing.assert_array_equal(gradient, np.zeros(1))


def test_sliding_can_remove_an_unobserved_old_knot():
    initial = SE3((.2, -.1, .3), (0., 0., 0., 1.))
    target = SE3.identity()
    problem = ContinuousTimeTrajectoryFitProblem(
        knot_timestamps=(0., 1., 2., 3.), initial_knot_poses=(initial,) * 4,
        pose_measurements=tuple(TrajectoryPoseMeasurement(str(t), t, target) for t in (1., 2., 3.)),
    )
    result = fit_sliding_window_trajectory(problem, window_knot_counts=(3, 2), overlap_knot_count=1)
    assert result.sliding_result.status == "converged"
    assert result.sliding_result.knot_poses[0] == initial
    assert result.window_observability[0].with_marginal_prior.numerical_nullity == 6
    for knot in result.sliding_result.knot_poses[1:]:
        np.testing.assert_allclose(knot.translation_m, target.translation_m, atol=1e-7)
    assert result.window_observability[1].with_marginal_prior.numerical_nullity == 0
