"""Residual factors must survive window boundaries without a normal round trip."""
import json

import numpy as np
import pytest

from calibrex.core.continuous_time_marginalization import (
    SquareRootKnotMarginalizationPrior,
    joint_marginalization_prior_blocks,
    joint_marginalization_prior_residual,
)
from calibrex.core.continuous_time_sliding_window_fit import fit_sliding_window_trajectory
from calibrex.core.continuous_time_sparse import (
    ContinuousTimeTrajectoryFitProblem,
    TrajectoryPointToPlaneMeasurement,
    build_dense_knot_normal_equations,
    build_dense_knot_residual_equations,
)
from calibrex.core.geometry import SE3
from calibrex.core.se3_manifold import se3_exp


def test_rectangular_prior_keeps_weak_factor_constant_and_tangent_derivative():
    j = np.zeros((3, 6))
    j[0, 0], j[1, 1] = 1e-9, 2.
    prior = SquareRootKnotMarginalizationPrior((0,), (SE3.identity(),), j,
                                             np.array([1., 2., 3.]))
    knot = se3_exp(SE3.identity(), np.array([.2, .3, 0., 0., 0., .1]))
    r, blocks = joint_marginalization_prior_blocks(prior, [knot])
    assert r.shape == (3,)
    assert r[-1] == 3.
    epsilon = 1e-7
    for axis in range(6):
        step = np.eye(6)[axis] * epsilon
        changed = joint_marginalization_prior_residual(prior, [se3_exp(knot, step)])
        np.testing.assert_allclose(blocks[0][1][:, axis], (changed-r)/epsilon,
                                   atol=3e-9)
    problem = ContinuousTimeTrajectoryFitProblem(
        knot_timestamps=(0., 1.), initial_knot_poses=(knot, SE3.identity()),
        joint_knot_marginalization_priors=(prior,),
    )
    jacobian, residual = build_dense_knot_residual_equations(
        problem, list(problem.initial_knot_poses),
    )
    h, g = build_dense_knot_normal_equations(problem, list(problem.initial_knot_poses))
    np.testing.assert_allclose(jacobian.T @ jacobian, h)
    np.testing.assert_allclose(jacobian.T @ residual, g)
    np.testing.assert_array_equal(residual, r)


def test_weak_information_crosses_window_without_normal_round_trip(monkeypatch):
    import calibrex.core.continuous_time_sliding_window_fit as module

    captured = []
    original = module.fit_continuous_trajectory

    def capture(problem, options):
        captured.extend(problem.joint_knot_marginalization_priors)
        return original(problem, options)

    monkeypatch.setattr(module, 'fit_continuous_trajectory', capture)
    measurements = tuple(TrajectoryPointToPlaneMeasurement(
        str(t), t, (0., 0., 0.), (target, 0., 0.), (1., 0., 0.), weight,
    ) for t, target, weight in ((.5, 0., 1.), (1., 1., 1e-9), (2., 0., 1.)))
    problem = ContinuousTimeTrajectoryFitProblem(
        knot_timestamps=(0., 1., 2.), initial_knot_poses=(SE3.identity(),) * 3,
        point_to_plane_measurements=measurements,
    )
    result = fit_sliding_window_trajectory(problem, window_knot_counts=(2, 2),
                                          overlap_knot_count=1, square_root_marginalization=True)
    assert len(captured) == 1
    prior = captured[0]
    assert isinstance(prior, SquareRootKnotMarginalizationPrior)
    assert (prior.jacobian.T @ prior.jacobian)[0, 0] == pytest.approx(1e-18, rel=1e-12, abs=0.)
    assert (prior.jacobian.T @ prior.residual)[0] == pytest.approx(-1e-18, rel=1e-12, abs=0.)
    assert result.window_observability[0].square_root_elimination.numerical_rank == 1
    assert result.window_observability[0].elimination is None
    assert result.window_observability[-1].square_root_elimination is None
    with pytest.raises(ValueError, match='outside the information range'):
        fit_sliding_window_trajectory(problem, window_knot_counts=(2, 2), overlap_knot_count=1)


def test_cli_records_square_root_mode_rank_and_unchanged_failed_policy(tmp_path):
    from calibrex.cli.main import main
    from calibrex.core.validation import validate_file

    path = tmp_path / 'square-root.json'
    code = main(['trajectory', 'recover-sliding-window', '--output', str(path),
                 '--square-root-marginalization', '--json'])
    assert code == 2
    assert validate_file(path).valid
    result = json.loads(path.read_text())
    assert result['square_root_marginalization'] is True
    assert '--square-root-marginalization' in result['provenance']['command']
    assert result['provenance']['generator_version'] == '0.11'
    assert result['window_observability'][0]['square_root_elimination']
    assert result['window_observability'][0].get('elimination') is None
    assert result['window_fit_statuses'] == ['converged', 'converged']


@pytest.mark.parametrize('j,r', [(np.zeros((2, 5)), np.zeros(2)),
                                 (np.zeros((2, 6)), np.zeros(3)),
                                 (np.full((1, 6), np.nan), np.zeros(1)),
                                 (np.zeros((1, 6), dtype=complex), np.zeros(1))])
def test_invalid_square_root_prior_rejected(j, r):
    with pytest.raises(ValueError):
        SquareRootKnotMarginalizationPrior((0,), (SE3.identity(),), j, r)


def test_multiple_windows_retain_joint_factors_and_match_normal_reference():
    from calibrex.core.continuous_time_sparse import TrajectoryPoseMeasurement

    problem = ContinuousTimeTrajectoryFitProblem(
        knot_timestamps=(0., 1., 2., 3., 4.),
        initial_knot_poses=(SE3((.1, -.2, .3), (0., 0., 0., 1.)),) * 5,
        pose_measurements=tuple(TrajectoryPoseMeasurement(str(t), t, SE3.identity())
                                for t in (0., .5, 1., 1.5, 2., 2.5, 3., 3.5, 4.)),
    )
    reference = fit_sliding_window_trajectory(problem, window_knot_counts=(3, 3, 3),
                                              overlap_knot_count=2)
    actual = fit_sliding_window_trajectory(problem, window_knot_counts=(3, 3, 3),
                                           overlap_knot_count=2, square_root_marginalization=True)
    assert all(e.square_root_elimination is not None for e in actual.window_observability[:-1])
    assert actual.window_observability[-1].square_root_elimination is None
    assert actual.sliding_result.status == reference.sliding_result.status == 'converged'
    for left, right in zip(actual.sliding_result.knot_poses,
                           reference.sliding_result.knot_poses, strict=True):
        np.testing.assert_allclose(left.translation_m, right.translation_m, atol=1e-8)
    assert actual.sliding_result.final_objective < 1e-12


def test_zero_row_prior_can_be_evaluated_and_assembled():
    prior = SquareRootKnotMarginalizationPrior((0,), (SE3.identity(),),
                                             np.zeros((0, 6)), np.zeros(0))
    problem = ContinuousTimeTrajectoryFitProblem(
        knot_timestamps=(0., 1.), initial_knot_poses=(SE3.identity(),) * 2,
        joint_knot_marginalization_priors=(prior,),
    )
    j, r = build_dense_knot_residual_equations(problem, list(problem.initial_knot_poses))
    assert j.shape == (0, 12)
    assert r.shape == (0,)


def test_compressed_window_priors_match_uncompressed_development_control(monkeypatch):
    import calibrex.core.continuous_time_sliding_window_fit as module
    from calibrex.evaluation.continuous_time_sliding_window import (
        run_synthetic_sliding_window_recovery,
    )

    compressed = run_synthetic_sliding_window_recovery(square_root_marginalization=True)
    monkeypatch.setattr(module, 'compress_residual_factor', lambda j, r: (j.copy(), r.copy()))
    uncompressed = run_synthetic_sliding_window_recovery(square_root_marginalization=True)
    assert compressed.policy_status == uncompressed.policy_status == 'fail'
    assert compressed.window_fit_statuses == uncompressed.window_fit_statuses
    assert compressed.sliding_holdout_rmse_m == pytest.approx(
        uncompressed.sliding_holdout_rmse_m, abs=1e-10,
    )
    for evidence in compressed.window_observability[:-1]:
        record = evidence.square_root_elimination
        assert record is not None
        assert record.retained_rows_after_compression <= 7  # One retained 6-DoF knot.
        assert record.retained_rows_before_compression > record.retained_rows_after_compression
