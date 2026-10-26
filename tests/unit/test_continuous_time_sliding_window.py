"""Tests for continuous-time sliding-window marginalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

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


@pytest.mark.parametrize("overlap", [1, 2])
def test_sliding_window_matches_batch_overlap(overlap: int) -> None:
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
        window_knot_counts=(4, 2 + overlap),
        overlap_knot_count=overlap,
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
    # Preserve the earlier singular window and the failed no-prior control;
    # convergence of the last window alone cannot make the recovery pass.
    assert artifact.policy_status == "fail"
    assert "sliding fit status" in artifact.policy_reason
    assert artifact.window_fit_statuses == ["singular_system", "converged"]
    assert artifact.sliding_fit_status == "singular_system"
    assert not artifact.sliding_fit_diagnostics.stationary
    first, second = artifact.window_observability
    assert first.with_marginal_prior.numerical_rank == 18
    assert first.with_marginal_prior.parameter_count == 24
    assert first.with_marginal_prior.zero_information_parameters == [
        f"knot[3].{axis}" for axis in ("tx", "ty", "tz", "rx", "ry", "rz")
    ]
    assert second.with_marginal_prior.numerical_rank == 18
    assert second.with_marginal_prior.numerical_nullity == 0
    assert artifact.window_count == 2
    assert artifact.known_bad_rmse_delta_m <= 0.01
    assert artifact.sliding_holdout_rmse_m < 0.08
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
    assert exit_code == 2
    assert validate_file(output).valid


@pytest.mark.parametrize("apply_prior", [True, False])
def test_overlapping_observations_have_one_owner_when_summarized(monkeypatch, apply_prior):
    from dataclasses import replace

    from calibrex.core import continuous_time_sliding_window_fit as module
    from calibrex.evaluation.continuous_time_lidar_point_to_plane import (
        _measurements,
        _truth_knots,
    )

    timestamps = tuple(float(i) for i in range(8))
    truth = _truth_knots(timestamps)
    rng = np.random.default_rng(43)
    samples = [(float(t), int(i % 3)) for i, t in enumerate(np.linspace(0., 7., 281))]
    measurements = _measurements(truth, timestamps, samples, rng, prefix="accounting")
    # Distinct rows may share identifiers: accounting must retain both occurrences.
    measurements = tuple(replace(m, measurement_id="shared-id") for m in measurements)
    problem = ContinuousTimeTrajectoryFitProblem(
        knot_timestamps=timestamps, initial_knot_poses=truth,
        point_to_plane_measurements=measurements,
    )
    captured = []
    original = module.fit_continuous_trajectory

    def fit(window, options):
        captured.append(window)
        return original(window, options)

    monkeypatch.setattr(module, "fit_continuous_trajectory", fit)
    module.fit_sliding_window_trajectory(
        problem, window_knot_counts=(4, 4, 4), overlap_knot_count=2,
        apply_marginalization=apply_prior,
        # Stop before updates to compare all quadratics at identical anchors.
        options=ContinuousTimeTrajectoryFitOptions(max_iterations=1,
                                                   convergence_gradient_norm=1e10),
    )
    windows = captured[1:]  # The first call is the separate batch reference.
    assert all(not window.pose_measurements for window in windows)
    occurrences = [id(m) for window in windows for m in window.point_to_plane_measurements]
    if not apply_prior:
        assert len(occurrences) > len(measurements)
        assert all(not window.joint_knot_marginalization_priors for window in windows)
        return
    assert sorted(occurrences) == sorted(id(m) for m in measurements)
    assert len(windows[1].joint_knot_marginalization_priors) == 1
    assert min(m.timestamp_sec for m in windows[1].point_to_plane_measurements) > 3.
    assert min(m.timestamp_sec for m in windows[2].point_to_plane_measurements) > 5.
    # With fixed linearization anchors, repeated transfer must equal eliminating
    # the same prefix from all original rows, without invented pose anchors.
    reference = replace(problem, pose_measurements=windows[0].pose_measurements)
    full_h, full_g = build_dense_knot_normal_equations(reference, list(truth))
    expected_h, expected_g = marginalize_knot_prefix(full_h, full_g, eliminated_knot_count=4)
    actual_h, actual_g = build_dense_knot_normal_equations(windows[-1], list(truth[4:]))
    np.testing.assert_allclose(actual_h, expected_h, rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(actual_g, expected_g, rtol=1e-6, atol=1e-7)


def test_window_selector_keeps_new_row_just_after_summarized_endpoint():
    from calibrex.core.continuous_time_sliding_window_fit import (
        _window_point_to_plane_measurements,
    )
    from calibrex.core.continuous_time_sparse import TrajectoryPointToPlaneMeasurement

    times = (3., float(np.nextafter(3., np.inf)), 4.)
    rows = tuple(TrajectoryPointToPlaneMeasurement(
        "same-id", t, (0., 0., 0.), (0., 0., 0.), (0., 0., 1.),
    ) for t in times)
    selected = _window_point_to_plane_measurements(
        rows, (0., 1., 2., 3., 4.), 2., 4., summarized_through=3.,
    )
    assert selected == rows[1:]


def test_nonadvancing_windows_fail_before_fitting(monkeypatch):
    from calibrex.core import continuous_time_sliding_window_fit as module
    from calibrex.core.continuous_time_sparse import TrajectoryPoseMeasurement
    from calibrex.core.geometry import SE3

    problem = ContinuousTimeTrajectoryFitProblem(
        knot_timestamps=(0., 1., 2.), initial_knot_poses=(SE3.identity(),) * 3,
        pose_measurements=(TrajectoryPoseMeasurement("anchor", 0., SE3.identity()),),
    )
    def forbidden(*args):
        pytest.fail("invalid window layout must fail before any fit")
    monkeypatch.setattr(module, "fit_continuous_trajectory", forbidden)
    with pytest.raises(ValueError, match="extend beyond"):
        module.fit_sliding_window_trajectory(problem, window_knot_counts=(2, 3),
                                             overlap_knot_count=2)


def test_prior_only_problem_remains_valid_after_raw_rows_are_summarized():
    from calibrex.core.continuous_time_marginalization import JointKnotMarginalizationPrior
    from calibrex.core.continuous_time_sparse import fit_continuous_trajectory
    from calibrex.core.geometry import SE3

    anchor = SE3.identity()
    gradient = np.array([.1, -.2, .3, 0., 0., 0.] * 2)
    prior = JointKnotMarginalizationPrior((0, 1), (anchor, anchor), np.eye(12), gradient)
    problem = ContinuousTimeTrajectoryFitProblem(
        knot_timestamps=(0., 1.), initial_knot_poses=(anchor, anchor),
        joint_knot_marginalization_priors=(prior,),
    )
    result = fit_continuous_trajectory(problem)
    assert result.status == "converged"
    for knot in result.knot_poses:
        np.testing.assert_allclose(knot.translation_m, -gradient[:3], atol=1e-7)


def test_explicit_pose_observations_are_transferred_once_without_initial_pose_bias(monkeypatch):
    from calibrex.core import continuous_time_sliding_window_fit as module
    from calibrex.core.continuous_time_sparse import TrajectoryPoseMeasurement
    from calibrex.core.geometry import SE3

    timestamps = tuple(float(i) for i in range(8))
    target = SE3((.7, -.4, .2), (0., 0., 0., 1.))
    rows = tuple(TrajectoryPoseMeasurement("shared-id", t, target) for t in timestamps)
    problem = ContinuousTimeTrajectoryFitProblem(
        knot_timestamps=timestamps, initial_knot_poses=(SE3.identity(),) * 8,
        pose_measurements=rows,
    )
    captured = []
    original = module.fit_continuous_trajectory

    def fit(window, options):
        captured.append(window)
        return original(window, options)

    monkeypatch.setattr(module, "fit_continuous_trajectory", fit)
    result = module.fit_sliding_window_trajectory(
        problem, window_knot_counts=(4, 4, 4), overlap_knot_count=2,
    )
    occurrences = [id(p) for window in captured[1:] for p in window.pose_measurements]
    assert sorted(occurrences) == sorted(id(p) for p in rows)
    assert result.sliding_result.status == "converged"
    for knot in result.sliding_result.knot_poses:
        np.testing.assert_allclose(knot.translation_m, target.translation_m, atol=1e-7)
    assert result.max_overlap_translation_error_m < 1e-7


def test_retained_gauge_audit_keeps_failed_control_and_source_bindings():
    import hashlib
    import json

    root = Path(__file__).resolve().parents[2] / "docs/assets/sliding_gauge_audit"
    manifest = json.loads((root / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    for version, expected in (("baseline", "pass"), ("current", "fail")):
        path = root / version / "result.json"
        assert validate_file(path).valid
        result = json.loads(path.read_text())
        assert result["seed"] == 20260820
        assert result["policy_status"] == expected
    sources = json.loads((root / "current/source_sha256.json").read_text())
    for path in (root / "measured_source").rglob("*.txt"):
        name = str(path.relative_to(root / "measured_source"))[:-4]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sources[name]


@pytest.mark.parametrize("early_status,expected", [
    ("singular_system", "singular_system"), ("max_iterations", "max_iterations"),
    ("stalled", "stalled"), ("converged", "stalled"),
])
def test_sliding_evidence_uses_returned_knots_and_retains_earlier_failures(
    monkeypatch, early_status, expected,
):
    from dataclasses import replace

    from calibrex.core import continuous_time_sliding_window_fit as module
    from calibrex.core.continuous_time_sparse import TrajectoryPoseMeasurement
    from calibrex.core.geometry import SE3

    anchor = SE3.identity()
    problem = ContinuousTimeTrajectoryFitProblem(
        knot_timestamps=tuple(float(i) for i in range(6)), initial_knot_poses=(anchor,) * 6,
        pose_measurements=tuple(TrajectoryPoseMeasurement("pose", float(i), anchor)
                                for i in range(6)),
    )
    original = module.fit_continuous_trajectory
    calls = 0

    def fit(window, options):
        nonlocal calls
        calls += 1
        result = original(window, options)
        if calls == 1:
            return replace(result, final_objective=12345., gradient_norm=12345.,
                           max_step_translation_m=12345., nonzero_jacobian_blocks=12345)
        if calls == 2:
            return replace(result, status=early_status, max_step_translation_m=.7)
        shifted = tuple(se3_exp(knot, np.array([.2, 0., 0., 0., 0., 0.]))
                        for knot in result.knot_poses)
        return replace(result, knot_poses=shifted, status="converged",
                       max_step_translation_m=.4)

    monkeypatch.setattr(module, "fit_continuous_trajectory", fit)
    result = module.fit_sliding_window_trajectory(
        problem, window_knot_counts=(4, 4), overlap_knot_count=2,
    )
    sliding = result.sliding_result
    assert sliding.status == expected
    assert [w.status for w in result.window_results] == [early_status, "converged"]
    assert not sliding.diagnostics.stationary
    # Four translated knots, each with four three-coordinate pose anchors.
    assert sliding.final_objective == pytest.approx(4 * 4 * .2**2)
    assert sliding.final_pose_rmse == pytest.approx(np.sqrt(4 * 4 * .2**2 / (6 * 12)))
    assert sliding.final_point_to_plane_rmse is None
    _, gradient = build_dense_knot_normal_equations(problem, list(sliding.knot_poses))
    assert sliding.gradient_norm == pytest.approx(np.linalg.norm(gradient, ord=np.inf))
    assert sliding.max_step_translation_m == .7
    assert sliding.nonzero_jacobian_blocks == 6 * 4 * 2


def test_sliding_rejects_unrouted_factor_families_before_batch_fit(monkeypatch):
    from calibrex.core import continuous_time_sliding_window_fit as module
    from calibrex.core.continuous_time_sparse import TrajectoryPointMeasurement
    from calibrex.core.geometry import SE3

    problem = ContinuousTimeTrajectoryFitProblem(
        knot_timestamps=(0., 1., 2.), initial_knot_poses=(SE3.identity(),) * 3,
        point_measurements=(TrajectoryPointMeasurement(
            "point", .5, (0., 0., 0.), (0., 0., 0.),
        ),),
    )
    def forbidden(*args):
        pytest.fail("unsupported observations must not be silently discarded")
    monkeypatch.setattr(module, "fit_continuous_trajectory", forbidden)
    with pytest.raises(ValueError, match="only plane and pose"):
        module.fit_sliding_window_trajectory(problem, window_knot_counts=(2, 2),
                                             overlap_knot_count=1)
