"""Knot-only assembly must match optimizer weights and reject malformed states."""
from dataclasses import replace

import numpy as np
import pytest

import calibrex.core.continuous_time_sparse as sparse
from calibrex.core.continuous_time_marginalization import (
    JointKnotMarginalizationPrior,
    KnotMarginalizationPrior,
    SquareRootKnotMarginalizationPrior,
)
from calibrex.core.geometry import SE3
from calibrex.core.se3_manifold import se3_exp


def mixed_problem():
    rng = np.random.default_rng(102)
    identity = SE3.identity()
    knots = tuple(se3_exp(identity, rng.normal(scale=.1, size=6)) for _ in range(3))
    a = rng.normal(size=(12, 12))
    return sparse.ContinuousTimeTrajectoryFitProblem(
        knot_timestamps=(0., 1., 2.), initial_knot_poses=knots,
        point_measurements=tuple(sparse.TrajectoryPointMeasurement(
            str(i), t, (.2, -.1, .4), (.3, .1, .6), w,
        ) for i, (t, w) in enumerate(((0., .2), (.3, 3.), (2., 1.5)))),
        point_to_plane_measurements=(sparse.TrajectoryPointToPlaneMeasurement(
            'plane', 1.3, (.5, .2, .1), (.2, .3, .4), (0., 0., 1.), 2.5,
        ),),
        pose_measurements=(sparse.TrajectoryPoseMeasurement('pose', .7, identity, .4),),
        knot_marginalization_priors=(KnotMarginalizationPrior(
            1, identity, np.diag(np.arange(1., 7.)), rng.normal(size=6),
        ),),
        joint_knot_marginalization_priors=(
            JointKnotMarginalizationPrior((2, 0), (identity, identity),
                                          a.T @ a + np.eye(12), rng.normal(size=12)),
            SquareRootKnotMarginalizationPrior((1,), (identity,),
                                               rng.normal(size=(8, 6)), rng.normal(size=8)),
        ),
    )


def test_mixed_weighted_factors_match_optimizer_and_manifold_derivatives(monkeypatch):
    problem = mixed_problem()
    knots = list(problem.initial_knot_poses)
    j, r = sparse.build_dense_knot_residual_equations(problem, knots)
    costs, normals = [], []
    objective = sparse._objective
    stationarity = sparse.normalized_gradient_norm

    def capture_cost(*args, **kwargs):
        value = objective(*args, **kwargs)
        costs.append(value)
        return value

    def capture_normal(gradient, diagonal, residual_norm):
        normals.append((gradient.copy(), diagonal.copy(), residual_norm))
        return stationarity(gradient, diagonal, residual_norm)

    monkeypatch.setattr(sparse, '_objective', capture_cost)
    monkeypatch.setattr(sparse, 'normalized_gradient_norm', capture_normal)
    result = sparse.fit_continuous_trajectory(
        problem, sparse.ContinuousTimeTrajectoryFitOptions(max_iterations=1),
    )
    assert costs[0] == pytest.approx(r @ r, rel=1e-13)
    np.testing.assert_allclose(normals[0][0], j.T @ r, atol=1e-12)
    np.testing.assert_allclose(normals[0][1], np.sum(j*j, axis=0), atol=1e-12)
    assert normals[0][2]**2 == pytest.approx(r @ r, rel=1e-13)
    _, final_r = sparse.build_dense_knot_residual_equations(problem, list(result.knot_poses))
    assert result.final_objective == pytest.approx(final_r @ final_r, rel=1e-12)
    epsilon = 1e-6
    for index in range(18):
        direction = np.eye(6)[index % 6] * epsilon
        plus, minus = knots.copy(), knots.copy()
        plus[index // 6] = se3_exp(knots[index // 6], direction)
        minus[index // 6] = se3_exp(knots[index // 6], -direction)
        _, rp = sparse.build_dense_knot_residual_equations(problem, plus)
        _, rm = sparse.build_dense_knot_residual_equations(problem, minus)
        np.testing.assert_allclose(j[:, index], (rp-rm)/(2*epsilon), atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize('timestamp', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_interior_timestamps_are_rejected(timestamp):
    with pytest.raises(ValueError, match='knot timestamps must be'):
        replace(mixed_problem(), knot_timestamps=(0., timestamp, 2.))


@pytest.mark.parametrize('builder', [sparse.build_dense_knot_residual_equations,
                                     sparse.build_dense_knot_normal_equations])
@pytest.mark.parametrize('count', [0, 2, 4])
def test_incomplete_or_extra_knot_states_are_rejected(builder, count):
    with pytest.raises(ValueError, match='poses must match'):
        builder(mixed_problem(), [SE3.identity()] * count)


@pytest.mark.parametrize('index', [-1, 3])
@pytest.mark.parametrize('builder', [sparse.build_dense_knot_residual_equations,
                                     sparse.build_dense_knot_normal_equations])
def test_single_knot_prior_indices_match_optimizer_validation(builder, index):
    problem = mixed_problem()
    problem = replace(problem, knot_marginalization_priors=(
        KnotMarginalizationPrior(index, SE3.identity(), np.eye(6)),
    ))
    with pytest.raises(ValueError, match='prior knot index is out of range'):
        builder(problem, list(problem.initial_knot_poses))
    with pytest.raises(ValueError, match='prior knot index is out of range'):
        sparse.fit_continuous_trajectory(problem)
