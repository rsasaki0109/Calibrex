"""The retained prior preserves the local quadratic, including its linear term."""
from __future__ import annotations

import numpy as np
import pytest

from calibrex.core.continuous_time_marginalization import (
    KnotMarginalizationPrior,
    information_from_normal,
    marginalization_prior_blocks,
    marginalization_prior_residual,
    schur_complement_normal,
)
from calibrex.core.continuous_time_sparse import (
    ContinuousTimeTrajectoryFitProblem,
    TrajectoryPoseMeasurement,
    fit_continuous_trajectory,
)
from calibrex.core.geometry import SE3
from calibrex.core.se3_manifold import se3_exp

ANCHOR = SE3((0., 0., 0.), (0., 0., 0., 1.))


def test_retained_prior_matches_profiled_quadratic_and_dense_optimum():
    rng = np.random.default_rng(20261026)
    a = rng.normal(size=(24, 12))
    h = a.T @ a + np.eye(12)
    g = rng.normal(size=12)
    retained, gradient = schur_complement_normal(h, g, 6)
    prior = information_from_normal(retained, gradient, anchor_pose=ANCHOR, knot_index=0)
    residual, jacobian = marginalization_prior_blocks(prior, ANCHOR)
    np.testing.assert_allclose(jacobian.T @ jacobian, retained, atol=1e-8)
    np.testing.assert_allclose(jacobian.T @ residual, gradient, atol=1e-8)
    np.testing.assert_allclose(-np.linalg.solve(retained, gradient),
                               -np.linalg.solve(h, g)[6:], atol=1e-12)

    def profiled(x):
        eliminated = -np.linalg.solve(h[:6, :6], g[:6] + h[:6, 6:] @ x)
        full = np.concatenate((eliminated, x))
        return full @ h @ full + 2 * g @ full

    for _ in range(6):
        x = rng.normal(scale=.03, size=6)
        moved = marginalization_prior_residual(prior, se3_exp(ANCHOR, x))
        np.testing.assert_allclose(moved @ moved - residual @ residual,
                                   profiled(x) - profiled(np.zeros(6)), atol=1e-10)


def test_fit_uses_linear_prior_term_in_candidate_cost():
    gradient = np.array([-.15, .1, -.05, 0., 0., 0.])
    priors = tuple(KnotMarginalizationPrior(i, ANCHOR, np.eye(6), gradient)
                   for i in range(2))
    result = fit_continuous_trajectory(ContinuousTimeTrajectoryFitProblem(
        knot_timestamps=(0., 1.), initial_knot_poses=(ANCHOR, ANCHOR),
        knot_marginalization_priors=priors,
        pose_measurements=(TrajectoryPoseMeasurement("anchor-target", 0.,
                                                    se3_exp(ANCHOR, -gradient)),),
    ))
    assert result.status == "converged"
    for knot in result.knot_poses:
        np.testing.assert_allclose(knot.translation_m, -gradient[:3], atol=1e-7)


def test_weak_coordinate_information_preserves_gradient():
    h = np.diag([1e-28, 1e28, 1., 2., 3., 0.])
    g = np.array([1e-14, 1e14, .1, .2, .3, 0.])
    prior = KnotMarginalizationPrior(0, ANCHOR, h, g)
    residual, jacobian = marginalization_prior_blocks(prior, ANCHOR)
    np.testing.assert_allclose(jacobian.T @ residual, g, rtol=1e-12, atol=0.)


@pytest.mark.parametrize("gradient", [np.ones(6), np.full(6, np.nan), np.zeros(5)])
def test_invalid_gradient_cannot_be_silently_dropped(gradient):
    prior = KnotMarginalizationPrior(0, ANCHOR, np.zeros((6, 6)), gradient)
    with pytest.raises(ValueError, match="gradient"):
        marginalization_prior_residual(prior, ANCHOR)


def test_coupled_nullspace_gradient_is_rejected():
    prior = KnotMarginalizationPrior(0, ANCHOR, np.ones((6, 6)), np.arange(6.))
    with pytest.raises(ValueError, match="range"):
        marginalization_prior_blocks(prior, ANCHOR)


def test_shifted_prior_jacobian_matches_quadratic_away_from_anchor():
    rng = np.random.default_rng(71)
    a = rng.normal(size=(6, 6))
    h = a.T @ a + np.eye(6)
    g = rng.normal(size=6)
    prior = KnotMarginalizationPrior(0, ANCHOR, h, g)
    knot = se3_exp(ANCHOR, np.array([.2, -.1, .3, .1, .2, -.15]))
    residual, jacobian = marginalization_prior_blocks(prior, knot)
    direction = rng.normal(size=6)
    epsilon = 1e-6
    plus = marginalization_prior_residual(prior, se3_exp(knot, epsilon * direction))
    minus = marginalization_prior_residual(prior, se3_exp(knot, -epsilon * direction))
    numerical = (plus @ plus - minus @ minus) / (2 * epsilon)
    assert numerical == pytest.approx(2 * residual @ jacobian @ direction, rel=1e-6)
