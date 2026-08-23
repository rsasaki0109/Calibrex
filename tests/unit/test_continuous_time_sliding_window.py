"""Tests for continuous-time sliding-window marginalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from calibrex.cli.main import main
from calibrex.core.continuous_time_marginalization import (
    marginalize_knot_prefix,
    schur_complement_normal,
)
from calibrex.core.continuous_time_sliding_window_fit import (
    fit_sliding_window_trajectory,
)
from calibrex.core.continuous_time_sparse import (
    ContinuousTimeTrajectoryFitOptions,
    ContinuousTimeTrajectoryFitProblem,
    build_dense_knot_normal_equations,
)
from calibrex.core.se3_manifold import se3_exp
from calibrex.core.validation import validate_file
from calibrex.evaluation.continuous_time_sliding_window import (
    run_synthetic_sliding_window_recovery,
)


def test_schur_complement_matches_dense_inverse() -> None:
    rng = np.random.default_rng(11)
    dimension = 12
    matrix = rng.normal(size=(dimension, dimension))
    hessian = matrix.T @ matrix + np.eye(dimension)
    gradient = rng.normal(size=dimension)
    eliminated = 6
    marginalized, marginal_gradient = schur_complement_normal(
        hessian, gradient, eliminated
    )
    hee = hessian[:eliminated, :eliminated]
    her = hessian[:eliminated, eliminated:]
    hre = hessian[eliminated:, :eliminated]
    hrr = hessian[eliminated:, eliminated:]
    expected_hessian = hrr - hre @ np.linalg.inv(hee) @ her
    expected_gradient = gradient[eliminated:] - hre @ np.linalg.inv(hee) @ gradient[:eliminated]
    assert np.allclose(marginalized, expected_hessian, atol=1.0e-8)
    assert np.allclose(marginal_gradient, expected_gradient, atol=1.0e-8)


def test_sliding_window_matches_batch_overlap() -> None:
    from calibrex.evaluation.continuous_time_lidar_point_to_plane import (
        _measurements,
        _truth_knots,
    )

    timestamps = tuple(float(index) for index in range(6))
    truth = _truth_knots(timestamps)
    rng = np.random.default_rng(7)
    samples = [(float(rng.uniform(0.05, 4.95)), int(index % 3)) for index in range(90)]
    measurements = _measurements(truth, timestamps, samples, rng, prefix="train")
    initial = tuple(
        se3_exp(knot, np.array([0.02, -0.01, 0.01, 0.0, 0.0, 0.01])) for knot in truth
    )
    problem = ContinuousTimeTrajectoryFitProblem(
        knot_timestamps=timestamps,
        initial_knot_poses=initial,
        point_to_plane_measurements=measurements,
    )
    result = fit_sliding_window_trajectory(
        problem,
        window_knot_counts=(4, 3),
        overlap_knot_count=1,
        options=ContinuousTimeTrajectoryFitOptions(max_iterations=80),
    )
    assert result.batch_result.status == "converged"
    assert result.max_overlap_translation_error_m < 0.05
    hessian, gradient = build_dense_knot_normal_equations(
        problem, list(result.batch_result.knot_poses)
    )
    marginalized, _ = marginalize_knot_prefix(hessian, gradient, eliminated_knot_count=1)
    assert marginalized.shape == (30, 30)


def test_synthetic_sliding_window_recovery_holdout_and_known_bad_control(
    tmp_path: Path,
) -> None:
    artifact = run_synthetic_sliding_window_recovery(seed=20260820)
    output = tmp_path / "recovery.yaml"
    artifact.save(output)
    assert validate_file(output, kind="continuous-time-sliding-window").valid
    assert artifact.policy_status == "pass"
    assert artifact.window_count == 2
    assert artifact.known_bad_rmse_delta_m > 0.01
    assert artifact.provenance.seed == 20260820


def test_cli_recover_sliding_window(tmp_path: Path) -> None:
    output = tmp_path / "recovery.yaml"
    exit_code = main(
        [
            "trajectory",
            "recover-sliding-window",
            "--output",
            str(output),
            "--seed",
            "20260820",
            "--json",
        ]
    )
    assert exit_code == 0
    assert validate_file(output).valid
