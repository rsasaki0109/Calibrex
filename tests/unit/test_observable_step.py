"""Projected updates respect local null modes without inventing information."""
from __future__ import annotations

import numpy as np
import pytest

from calibrex.core.observable_step import observable_subspace_step


def test_damping_cannot_move_a_coupled_null_direction():
    h = np.ones((2, 2))
    gradient = np.array([-2., -2.])
    damped = h + np.diag([.1, .4])
    step, evidence = observable_subspace_step(h, gradient, damped)
    assert step[0] == pytest.approx(step[1], abs=1e-14)
    assert evidence.numerical_rank == 1
    assert evidence.zero_information_count == 0
    assert step @ h @ step + 2 * gradient @ step < 0
    assert abs(np.diff(np.linalg.solve(damped, -gradient))[0]) > .1
    np.testing.assert_allclose(step, np.full(2, 2 / 2.25))


def test_projected_updates_are_covariant_under_independent_unit_changes():
    h = np.ones((2, 2))
    gradient = np.array([-2., -2.])
    damped = h + np.diag([.1, .4])
    units = np.array([1e-9, 1e9])
    reference, _ = observable_subspace_step(h, gradient, damped)
    actual, evidence = observable_subspace_step(h * units[:, None] * units[None, :],
        gradient * units, damped * units[:, None] * units[None, :])
    np.testing.assert_allclose(actual * units, reference, rtol=1e-13)
    assert evidence.numerical_rank == 1


def test_full_rank_step_matches_direct_damped_solve():
    h = np.array([[2., .3], [.3, 1.]])
    gradient = np.array([.1, -.2])
    damped = h + np.eye(2) * .04
    actual, evidence = observable_subspace_step(h, gradient, damped)
    np.testing.assert_array_equal(actual, np.linalg.solve(damped, -gradient))
    assert evidence.numerical_rank == 2


def test_missing_columns_receive_zero_step_and_inconsistent_gradient_is_rejected():
    h = np.diag([1., 0.])
    step, evidence = observable_subspace_step(h, np.array([-1., 0.]), h + np.eye(2))
    np.testing.assert_array_equal(step, [.5, 0.])
    assert evidence.zero_information_count == 1
    with pytest.raises(ValueError, match="zero coupling and gradient"):
        observable_subspace_step(h, np.ones(2), h + np.eye(2))
    with pytest.raises(ValueError, match="outside the numerical information range"):
        observable_subspace_step(np.ones((2, 2)), np.array([1., -1.]), np.eye(2))


def test_opt_in_midpoint_fit_and_window_evidence():
    from calibrex.core.continuous_time_sliding_window_fit import fit_sliding_window_trajectory
    from calibrex.core.continuous_time_sparse import (
        ContinuousTimeTrajectoryFitOptions,
        ContinuousTimeTrajectoryFitProblem,
        TrajectoryPoseMeasurement,
        fit_continuous_trajectory,
    )
    from calibrex.core.geometry import SE3

    target = SE3.identity()
    initial = SE3((.2, -.1, .3), (0., 0., 0., 1.))
    problem = ContinuousTimeTrajectoryFitProblem(
        knot_timestamps=(0., 1., 2., 3.), initial_knot_poses=(initial,) * 4,
        pose_measurements=tuple(TrajectoryPoseMeasurement(str(t), t, target)
                                for t in (.5, 2., 3.)),
    )
    options = ContinuousTimeTrajectoryFitOptions(project_observable_subspace=True)
    result = fit_sliding_window_trajectory(problem, window_knot_counts=(3, 2),
                                         overlap_knot_count=1, options=options)
    assert result.sliding_result.status == "converged"
    assert result.sliding_result.final_objective < 1e-12
    evidence = result.window_observability[0].projected_steps
    assert evidence and all(step.numerical_rank == 12 for step in evidence)
    assert all(step.parameter_count == 18 for step in evidence)
    assert result.sliding_result.projected_step_diagnostics
    assert not fit_continuous_trajectory(problem).projected_step_diagnostics


def test_selected_mode_and_attempted_steps_survive_schema_serialization(tmp_path):
    from calibrex.core.validation import validate_file
    from calibrex.evaluation.continuous_time_sliding_window import (
        run_synthetic_sliding_window_recovery,
    )

    report = run_synthetic_sliding_window_recovery(project_observable_subspace=True)
    output = tmp_path / "projected.json"
    report.save(output)
    assert validate_file(output).valid
    assert report.project_observable_subspace
    assert report.provenance.generator_version == "0.11"
    assert all(window.projected_steps for window in report.window_observability)
    assert report.window_fit_statuses == ["converged", "converged"]
    assert report.policy_status == "fail"  # The independent control gate is unchanged.


def test_nonpositive_damped_model_in_observed_space_is_rejected():
    with pytest.raises(np.linalg.LinAlgError):
        observable_subspace_step(np.ones((2, 2)), np.ones(2), -np.eye(2))


def test_cli_records_projection_mode_and_keeps_failed_policy_exit(tmp_path):
    import json

    from calibrex.cli.main import main
    from calibrex.core.validation import validate_file

    path = tmp_path / "projected-cli.json"
    code = main(["trajectory", "recover-sliding-window", "--output", str(path),
                 "--project-observable-subspace", "--json"])
    assert code == 2
    assert validate_file(path).valid
    result = json.loads(path.read_text())
    assert result["project_observable_subspace"] is True
    assert "--project-observable-subspace" in result["provenance"]["command"]
    assert result["window_observability"][0]["projected_steps"]
