"""Retained cross-knot information agrees with the complete profiled quadratic."""
from __future__ import annotations

import numpy as np
import pytest

from calibrex.core.continuous_time_marginalization import (
    JointKnotMarginalizationPrior,
    joint_marginalization_prior_blocks,
    joint_marginalization_prior_residual,
    schur_complement_normal,
)
from calibrex.core.continuous_time_sparse import (
    ContinuousTimeTrajectoryFitProblem,
    TrajectoryPoseMeasurement,
    build_dense_knot_normal_equations,
    fit_continuous_trajectory,
)
from calibrex.core.geometry import SE3
from calibrex.core.se3_manifold import se3_exp

ANCHOR = SE3.identity()


def test_joint_prior_preserves_profiled_cost_gradient_and_cross_curvature():
    rng = np.random.default_rng(41)
    a = rng.normal(size=(30, 18))
    h = a.T @ a + np.eye(18)
    g = rng.normal(size=18)
    retained, gradient = schur_complement_normal(h, g, 6)
    prior = JointKnotMarginalizationPrior((1, 0), (ANCHOR, ANCHOR), retained, gradient)
    residual, blocks = joint_marginalization_prior_blocks(prior, [ANCHOR, ANCHOR])
    jacobian = np.column_stack([block for _, block in blocks])
    assert [index for index, _ in blocks] == [1, 0]
    np.testing.assert_allclose(jacobian.T @ jacobian, retained, atol=1e-8)
    np.testing.assert_allclose(jacobian.T @ residual, gradient, atol=1e-8)
    assert np.linalg.norm(retained[:6, 6:]) > 1.
    np.testing.assert_allclose(-np.linalg.solve(retained, gradient),
                               -np.linalg.solve(h, g)[6:], atol=1e-12)

    def profiled(x):
        removed = -np.linalg.solve(h[:6, :6], g[:6] + h[:6, 6:] @ x)
        full = np.concatenate((removed, x))
        return full @ h @ full + 2 * g @ full

    for _ in range(4):
        delta = rng.normal(scale=.03, size=12)
        knots = [se3_exp(ANCHOR, delta[6:]), se3_exp(ANCHOR, delta[:6])]
        moved = joint_marginalization_prior_residual(prior, knots)
        assert moved @ moved - residual @ residual == pytest.approx(
            profiled(delta) - profiled(np.zeros(12)), abs=1e-10,
        )


def test_joint_prior_dense_assembly_and_fit_keep_coupled_optimum():
    h = np.block([[np.eye(6), .4 * np.eye(6)], [.4 * np.eye(6), 2 * np.eye(6)]])
    g = np.array([.2, -.1, .1, 0., 0., 0., -.1, .15, -.2, 0., 0., 0.])
    optimum = -np.linalg.solve(h, g)
    prior = JointKnotMarginalizationPrior((1, 0), (ANCHOR, ANCHOR), h, g)
    # The pose observation agrees with the joint optimum and fixes no new bias.
    problem = ContinuousTimeTrajectoryFitProblem(
        knot_timestamps=(0., 1.), initial_knot_poses=(ANCHOR, ANCHOR),
        pose_measurements=(TrajectoryPoseMeasurement(
            "target", 0., se3_exp(ANCHOR, optimum[6:]),
        ),),
    )
    from dataclasses import replace
    joint_problem = replace(problem, joint_knot_marginalization_priors=(prior,))
    h0, g0 = build_dense_knot_normal_equations(problem, [ANCHOR, ANCHOR])
    h1, g1 = build_dense_knot_normal_equations(joint_problem, [ANCHOR, ANCHOR])
    order = np.r_[6:12, 0:6]
    np.testing.assert_allclose(h1 - h0, h[np.ix_(order, order)], atol=1e-8)
    np.testing.assert_allclose(g1 - g0, g[order], atol=1e-8)
    result = fit_continuous_trajectory(joint_problem)
    assert result.status == "converged"
    for index, expected in ((1, optimum[:3]), (0, optimum[6:9])):
        np.testing.assert_allclose(result.knot_poses[index].translation_m, expected, atol=1e-7)


@pytest.mark.parametrize("indices", [(), (0, 0), (-1, 1)])
def test_invalid_joint_indices_are_rejected(indices):
    with pytest.raises(ValueError):
        JointKnotMarginalizationPrior(indices, (ANCHOR,) * len(indices),
                                      np.eye(6 * len(indices)), np.zeros(6 * len(indices)))


def test_joint_prior_rejects_index_outside_window():
    prior = JointKnotMarginalizationPrior((0, 2), (ANCHOR, ANCHOR), np.eye(12), np.zeros(12))
    with pytest.raises(ValueError, match="out of range"):
        joint_marginalization_prior_blocks(prior, [ANCHOR, ANCHOR])


def test_joint_jacobian_away_from_anchors_matches_cost_difference():
    rng = np.random.default_rng(52)
    a = rng.normal(size=(12, 12))
    prior = JointKnotMarginalizationPrior((1, 0), (ANCHOR, ANCHOR),
                                          a.T @ a + np.eye(12), rng.normal(size=12))
    knots = [se3_exp(ANCHOR, rng.normal(scale=.1, size=6)) for _ in range(2)]
    residual, blocks = joint_marginalization_prior_blocks(prior, knots)
    directions = rng.normal(size=(2, 6))
    analytic = sum(2 * residual @ block @ directions[index] for index, block in blocks)
    epsilon = 1e-6
    plus = joint_marginalization_prior_residual(prior, [
        se3_exp(knot, epsilon * directions[index]) for index, knot in enumerate(knots)
    ])
    minus = joint_marginalization_prior_residual(prior, [
        se3_exp(knot, -epsilon * directions[index]) for index, knot in enumerate(knots)
    ])
    assert (plus @ plus - minus @ minus) / (2 * epsilon) == pytest.approx(analytic, rel=1e-6)


@pytest.mark.parametrize("information,gradient", [
    (np.eye(6), np.zeros(12)),
    (np.eye(12), np.zeros(6)),
    (-np.eye(12), np.zeros(12)),
    (np.ones((12, 12)), np.arange(12.)),
])
def test_invalid_joint_normal_information_is_rejected(information, gradient):
    with pytest.raises(ValueError):
        JointKnotMarginalizationPrior((0, 1), (ANCHOR, ANCHOR), information, gradient)
