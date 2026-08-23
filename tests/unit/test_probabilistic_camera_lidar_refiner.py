from __future__ import annotations

import math
from pathlib import Path

import pytest

from calibrex.cli.main import main
from calibrex.core.camera_lidar_artifacts import (
    CalibrationCandidateOutcome,
    CalibrationCandidateTrace,
    CameraLidarArtifactProvenance,
    CameraLidarCalibrationProblem,
    CameraLidarObservationBinding,
)
from calibrex.core.camera_lidar_confidence_calibration import (
    load_camera_lidar_confidence_calibration,
)
from calibrex.core.camera_lidar_correspondence_quality import (
    load_camera_lidar_correspondence_quality,
)
from calibrex.core.geometry import SE3
from calibrex.core.probabilistic_correspondence import (
    CorrespondenceProviderIdentity,
    ProbabilisticCorrespondenceArtifact,
    ProbabilisticCorrespondenceFrame,
    ProbabilisticCorrespondenceProvenance,
    ProbabilisticImageCorrespondence,
    load_probabilistic_refinement_result,
)
from calibrex.core.provenance import sha256_path
from calibrex.core.result import TransformResult
from calibrex.core.validation import validate_file
from calibrex.data.depth import DepthCameraIntrinsics, DepthFileReference
from calibrex.evaluation.borer_rotation_benchmark import (
    build_borer_six_dof_protocol,
)
from calibrex.evaluation.borer_six_dof_benchmark import (
    BORER_SIX_DOF_BENCHMARK_VERSION,
)
from calibrex.evaluation.camera_lidar_confidence_calibration import (
    calibrate_camera_lidar_confidence,
)
from calibrex.evaluation.camera_lidar_correspondence_quality import (
    analyze_camera_lidar_correspondence_quality,
)
from calibrex.evaluation.holdout import split_indices
from calibrex.evaluation.probabilistic_camera_lidar_falsification import (
    _candidate_holdout_pass,
    run_probabilistic_camera_lidar_falsification,
)
from calibrex.evaluation.probabilistic_camera_lidar_run import (
    run_probabilistic_camera_lidar_refinement,
)
from calibrex.evaluation.probabilistic_refinement_ablation import (
    run_probabilistic_refinement_ablation,
)
from calibrex.solvers import ProbabilisticCameraLidarRefiner as PublicRefiner
from calibrex.solvers.probabilistic_camera_lidar_refiner import (
    ProbabilisticCameraLidarRefinementOptions,
    ProbabilisticCameraLidarRefiner,
)


def test_multiframe_probabilistic_refiner_improves_disjoint_holdout() -> None:
    assert PublicRefiner is ProbabilisticCameraLidarRefiner
    truth = SE3(
        (0.10, -0.05, 0.02),
        (0.0, 0.0, math.sin(math.radians(1.0) / 2.0), math.cos(math.radians(1.0) / 2.0)),
    )
    frames = _frames(truth)
    initial = SE3((0.05, -0.05, 0.02), (0.0, 0.0, 0.0, 1.0))
    options = ProbabilisticCameraLidarRefinementOptions(
        split_seed=7,
        initial_rotation_step_deg=1.0,
        initial_translation_step_m=0.05,
        minimum_rotation_step_deg=0.05,
        minimum_translation_step_m=0.005,
        max_evaluations=300,
    )

    result = ProbabilisticCameraLidarRefiner().solve(
        frames, initial, options
    )

    assert result.status == "converged"
    assert result.transform_camera_lidar.translation_m[0] == pytest.approx(
        0.10, abs=0.006
    )
    assert set(result.train_frame_ids).isdisjoint(result.holdout_frame_ids)
    assert (
        result.final_holdout_evaluation.weighted_reprojection_rmse_px
        < result.initial_holdout_evaluation.weighted_reprojection_rmse_px
    )
    assert result.acceptance.accepted is True
    assert result.acceptance.holdout_used_for_selection is False
    assert result.candidate_transform_camera_lidar == result.transform_camera_lidar
    assert result.as_dict()["method"] == "probabilistic_multiframe_refinement/v0.3"


def test_at_bound_candidate_is_retained_but_initializer_is_selected() -> None:
    truth = SE3((0.50, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    initial = SE3.identity()
    result = ProbabilisticCameraLidarRefiner().solve(
        _frames(truth),
        initial,
        ProbabilisticCameraLidarRefinementOptions(
            translation_bound_m=0.05,
            initial_translation_step_m=0.05,
            minimum_translation_step_m=0.005,
            max_evaluations=300,
        ),
    )

    assert result.status == "at_bound"
    assert result.acceptance.accepted is False
    assert result.acceptance.selected_source == "initializer_rollback"
    assert "solver_not_converged" in result.acceptance.reasons
    assert "candidate_near_correction_bound" in result.acceptance.reasons
    assert result.candidate_transform_camera_lidar != initial
    assert result.transform_camera_lidar == initial
    assert result.final_train_evaluation == result.initial_train_evaluation
    assert result.final_holdout_evaluation == result.initial_holdout_evaluation
    assert result.candidate_train_evaluation.objective < (
        result.initial_train_evaluation.objective
    )


def test_acceptance_and_candidate_selection_do_not_use_holdout_residuals() -> None:
    truth = SE3((0.10, -0.05, 0.02), (0.0, 0.0, 0.0, 1.0))
    initial = SE3((0.05, -0.05, 0.02), (0.0, 0.0, 0.0, 1.0))
    frames = sorted(_frames(truth), key=lambda item: item.frame_id)
    options = ProbabilisticCameraLidarRefinementOptions(
        split_seed=7,
        initial_translation_step_m=0.05,
        minimum_translation_step_m=0.005,
        max_evaluations=300,
    )
    _train_indices, holdout_indices = split_indices(
        len(frames), options.holdout_ratio, options.split_seed
    )
    holdout_set = set(holdout_indices)
    corrupted = [
        frame.model_copy(
            update={
                "correspondences": [
                    item.model_copy(
                        update={
                            "image_mean_px": [
                                item.image_mean_px[0] + 100.0,
                                item.image_mean_px[1] - 75.0,
                            ]
                        }
                    )
                    for item in frame.correspondences
                ]
            }
        )
        if index in holdout_set
        else frame
        for index, frame in enumerate(frames)
    ]

    baseline = ProbabilisticCameraLidarRefiner().solve(frames, initial, options)
    changed_holdout = ProbabilisticCameraLidarRefiner().solve(
        corrupted, initial, options
    )

    assert baseline.candidate_transform_camera_lidar == (
        changed_holdout.candidate_transform_camera_lidar
    )
    assert baseline.acceptance == changed_holdout.acceptance
    assert baseline.transform_camera_lidar == changed_holdout.transform_camera_lidar
    assert baseline.candidate_holdout_evaluation != (
        changed_holdout.candidate_holdout_evaluation
    )


def test_probabilistic_refiner_exposes_prespecified_uncertainty_ablation() -> None:
    truth = SE3.identity()
    frames = _frames(truth)

    full = ProbabilisticCameraLidarRefiner().solve(
        frames,
        truth,
        ProbabilisticCameraLidarRefinementOptions(
            use_covariance=True,
            use_outlier_probability=True,
            use_reliability=True,
        ),
    )
    ablated = ProbabilisticCameraLidarRefiner().solve(
        frames,
        truth,
        ProbabilisticCameraLidarRefinementOptions(
            use_covariance=False,
            use_outlier_probability=False,
            use_reliability=False,
        ),
    )

    assert full.options.use_covariance is True
    assert ablated.options.use_covariance is False
    assert full.final_train_evaluation.valid_correspondence_count > 0


def test_probabilistic_multiframe_cli_pins_both_inputs(
    tmp_path: Path,
) -> None:
    truth = SE3((0.10, -0.05, 0.02), (0.0, 0.0, 0.0, 1.0))
    frames = _frames(truth)
    correspondence = _artifact(frames)
    problem = _problem(frames, reference=truth)
    correspondence_path = tmp_path / "correspondence.yaml"
    problem_path = tmp_path / "problem.yaml"
    result_path = tmp_path / "result.yaml"
    correspondence.save(correspondence_path)
    problem.save(problem_path)

    exit_code = main(
        [
            "camera-lidar",
            "refine-probabilistic-multiframe",
            str(correspondence_path),
            str(problem_path),
            "--output",
            str(result_path),
            "--json",
        ]
    )

    assert exit_code == 0
    assert validate_file(
        result_path, "probabilistic-refinement-result"
    ).valid
    result = load_probabilistic_refinement_result(result_path)
    assert result.initialization_source == "problem_initial_transform"
    assert result.initialization_trace_id is None
    assert result.final_translation_error_m is not None
    assert result.final_translation_error_m < 0.01
    assert result.final_rotation_error_deg is not None
    assert result.final_rotation_error_deg < 0.1


def test_probabilistic_multiframe_accepts_digest_matched_d2d_trace(
    tmp_path: Path,
) -> None:
    truth = SE3((0.10, -0.05, 0.02), (0.0, 0.0, 0.0, 1.0))
    frames = _frames(truth)
    problem = _problem(frames, reference=truth)
    correspondence_path = tmp_path / "correspondence.yaml"
    problem_path = tmp_path / "problem.yaml"
    trace_path = tmp_path / "d2d-trace.yaml"
    result_path = tmp_path / "result.yaml"
    _artifact(frames).save(correspondence_path)
    problem.save(problem_path)
    problem_digest = sha256_path(problem_path)
    assert problem_digest is not None
    truth_transform = TransformResult(
        parent="camera",
        child="lidar",
        translation_m=list(truth.translation_m),
        rotation_quat_xyzw=list(truth.rotation_quat_xyzw),
    )
    trace = CalibrationCandidateTrace(
        trace_id="d2d-trace",
        trial_id="fixture",
        problem_sha256=problem_digest,
        protocol_sha256="f" * 64,
        solver="native_borer_d2d_rotation",
        solver_version="0.1",
        status="converged",
        initial_transform_camera_lidar=(
            problem.initial_transform_camera_lidar
        ),
        output_transform_camera_lidar=truth_transform,
        evaluations=[],
        stopping_reason="fixture",
        runtime_seconds=0.1,
        outcome=CalibrationCandidateOutcome(
            rotation_error_deg=0.0,
            translation_error_m=0.0,
            hit=True,
            hit_reason="fixture",
        ),
        provenance=CameraLidarArtifactProvenance(
            generator="pytest",
            generator_version="1",
            source_sha256=problem_digest,
        ),
    )
    trace.save(trace_path)

    exit_code = main(
        [
            "camera-lidar",
            "refine-probabilistic-multiframe",
            str(correspondence_path),
            str(problem_path),
            "--initial-trace",
            str(trace_path),
            "--output",
            str(result_path),
            "--json",
        ]
    )

    assert exit_code == 0
    result = load_probabilistic_refinement_result(result_path)
    assert result.initialization_source == "d2d_candidate_trace"
    assert result.provenance.generator_version == "0.3"
    assert result.initialization_trace_id == "d2d-trace"
    assert result.initialization_trace_status == "converged"
    assert result.initialization_trace_hit is True
    assert result.initial_transform_camera_lidar == truth_transform
    assert result.initial_rotation_error_deg == pytest.approx(0.0)
    assert result.initial_translation_error_m == pytest.approx(0.0)
    assert result.provenance.initialization_trace_sha256 == sha256_path(
        trace_path
    )
    mismatched_path = tmp_path / "mismatched-trace.yaml"
    trace.model_copy(update={"problem_sha256": "0" * 64}).save(
        mismatched_path
    )
    with pytest.raises(ValueError, match="digest does not match"):
        run_probabilistic_camera_lidar_refinement(
            correspondence_path,
            problem_path,
            initialization_trace_path=mismatched_path,
        )


def test_correspondence_quality_report_is_result_bound_and_schema_valid(
    tmp_path: Path,
) -> None:
    truth = SE3((0.10, -0.05, 0.02), (0.0, 0.0, 0.0, 1.0))
    frames = _frames(truth)
    correspondence_path = tmp_path / "correspondence.yaml"
    problem_path = tmp_path / "problem.yaml"
    result_path = tmp_path / "result.yaml"
    report_path = tmp_path / "quality.yaml"
    cli_report_path = tmp_path / "candidate-quality.yaml"
    _artifact(frames).save(correspondence_path)
    _problem(frames, reference=truth).save(problem_path)
    result = run_probabilistic_camera_lidar_refinement(
        correspondence_path,
        problem_path,
    )
    result.save(result_path)

    report = analyze_camera_lidar_correspondence_quality(
        correspondence_path,
        result_path,
        pose_role="initializer",
        command=["pytest", "quality"],
    )
    report.save(report_path)

    assert report.holdout_used_for_selection is False
    assert report.refinement_result_sha256 == sha256_path(result_path)
    assert report.correspondence_artifact_sha256 == sha256_path(correspondence_path)
    assert report.summary.frame_count == len(frames)
    assert report.summary.total_correspondence_count == 96
    assert report.summary.accepted_correspondence_count == 96
    assert report.summary.aggregate_observability.rank == 6
    assert all(
        item.gates.total_count
        == item.gates.below_confidence_count
        + item.gates.behind_camera_count
        + item.gates.invalid_projection_count
        + item.gates.outside_image_count
        + item.gates.accepted_count
        for item in report.frames
    )
    assert validate_file(
        report_path, "camera-lidar-correspondence-quality"
    ).valid
    assert (
        load_camera_lidar_correspondence_quality(report_path).report_id
        == report.report_id
    )

    assert (
        main(
            [
                "camera-lidar",
                "diagnose-probabilistic-correspondence",
                str(correspondence_path),
                str(result_path),
                "--pose-role",
                "candidate",
                "--output",
                str(cli_report_path),
                "--json",
            ]
        )
        == 0
    )
    assert validate_file(
        cli_report_path, "camera-lidar-correspondence-quality"
    ).valid


def test_correspondence_quality_explains_confidence_support_failure(
    tmp_path: Path,
) -> None:
    truth = SE3.identity()
    frames = [
        frame.model_copy(
            update={
                "correspondences": [
                    item.model_copy(update={"reliability": 0.1})
                    for item in frame.correspondences
                ]
            }
        )
        for frame in _frames(truth)
    ]
    correspondence_path = tmp_path / "low-confidence-correspondence.yaml"
    problem_path = tmp_path / "problem.yaml"
    result_path = tmp_path / "result.yaml"
    _artifact(frames).save(correspondence_path)
    _problem(frames, reference=truth).save(problem_path)
    result = run_probabilistic_camera_lidar_refinement(
        correspondence_path,
        problem_path,
    )
    result.save(result_path)

    report = analyze_camera_lidar_correspondence_quality(
        correspondence_path,
        result_path,
    )

    assert result.status == "insufficient_correspondences"
    assert result.acceptance.selected_source == "initializer_rollback"
    assert report.summary.grade == "fail"
    assert report.summary.accepted_correspondence_count == 0
    assert report.summary.rejection_counts["below_confidence"] == 96
    assert report.summary.aggregate_observability.rank == 0
    assert "insufficient_train_correspondence_support" in (
        report.summary.gate_reasons
    )


def test_development_confidence_calibration_locks_highest_passing_threshold(
    tmp_path: Path,
) -> None:
    truth = SE3.identity()
    frames = _frames(truth)
    correspondence_path = tmp_path / "development-correspondence.yaml"
    problem_path = tmp_path / "development-problem.yaml"
    output = tmp_path / "confidence-calibration.yaml"
    cli_output = tmp_path / "confidence-calibration-cli.yaml"
    locked_refinement_output = tmp_path / "locked-refinement.yaml"
    _artifact(frames).model_copy(
        update={"split_id": "development", "sequence_id": "development-sequence"}
    ).save(correspondence_path)
    problem = _problem(frames, reference=truth)
    problem.model_copy(
        update={
            "sequence_id": "development-sequence",
            "observations": [
                item.model_copy(update={"split_id": "development"})
                for item in problem.observations
            ],
        }
    ).save(problem_path)

    calibration = calibrate_camera_lidar_confidence(
        correspondence_path,
        problem_path,
        thresholds=(0.5, 0.98, 0.99),
        calibration_split_seeds=(0, 1),
        evaluation_split_seeds=(0, 1),
        evaluation_dataset_ids_excluded=("evaluation-a", "evaluation-b"),
    )
    calibration.save(output)

    assert calibration.selected_minimum_confidence == pytest.approx(0.98)
    assert calibration.locked_refinement_options.minimum_confidence == pytest.approx(
        0.98
    )
    assert [item.gate_pass for item in calibration.candidates] == [True, True, False]
    assert calibration.runtime_holdout_used_for_pose_selection is False
    assert calibration.release_sota_claim_allowed is False
    assert validate_file(output, "camera-lidar-confidence-calibration").valid
    assert (
        load_camera_lidar_confidence_calibration(output).calibration_id
        == calibration.calibration_id
    )

    assert (
        main(
            [
                "camera-lidar",
                "calibrate-probabilistic-confidence",
                str(correspondence_path),
                str(problem_path),
                "--thresholds",
                "0.5,0.98,0.99",
                "--calibration-split-seeds",
                "0,1",
                "--evaluation-split-seeds",
                "0,1",
                "--excluded-evaluation-dataset-ids",
                "evaluation-a,evaluation-b",
                "--output",
                str(cli_output),
                "--json",
            ]
        )
        == 0
    )
    assert validate_file(cli_output, "camera-lidar-confidence-calibration").valid

    locked_result = run_probabilistic_camera_lidar_refinement(
        correspondence_path,
        problem_path,
        confidence_calibration_path=output,
    )
    assert locked_result.options["minimum_confidence"] == pytest.approx(0.98)
    assert locked_result.provenance.confidence_calibration_id == (
        calibration.calibration_id
    )
    assert locked_result.provenance.confidence_calibration_sha256 == sha256_path(
        output
    )
    with pytest.raises(ValueError, match="differ from confidence lock"):
        run_probabilistic_camera_lidar_refinement(
            correspondence_path,
            problem_path,
            confidence_calibration_path=output,
            options=ProbabilisticCameraLidarRefinementOptions(
                minimum_confidence=0.5
            ),
        )

    assert (
        main(
            [
                "camera-lidar",
                "refine-probabilistic-multiframe",
                str(correspondence_path),
                str(problem_path),
                "--confidence-calibration",
                str(cli_output),
                "--split-seed",
                "0",
                "--output",
                str(locked_refinement_output),
                "--json",
            ]
        )
        == 0
    )
    cli_result = load_probabilistic_refinement_result(locked_refinement_output)
    assert cli_result.provenance.confidence_calibration_sha256 == sha256_path(
        cli_output
    )


def test_development_confidence_calibration_retains_geometric_rejection(
    tmp_path: Path,
) -> None:
    truth = SE3.identity()
    frames = [
        frame.model_copy(
            update={
                "correspondences": [
                    item.model_copy(
                        update={
                            "image_mean_px": [
                                item.image_mean_px[0] + 100.0,
                                item.image_mean_px[1] + 100.0,
                            ]
                        }
                    )
                    for item in frame.correspondences
                ]
            }
        )
        for frame in _frames(truth)
    ]
    correspondence_path = tmp_path / "bad-development-correspondence.yaml"
    problem_path = tmp_path / "bad-development-problem.yaml"
    output = tmp_path / "rejected-confidence-calibration.yaml"
    _artifact(frames).model_copy(
        update={"split_id": "development", "sequence_id": "development-sequence"}
    ).save(correspondence_path)
    problem = _problem(frames, reference=truth)
    problem.model_copy(
        update={
            "sequence_id": "development-sequence",
            "observations": [
                item.model_copy(update={"split_id": "development"})
                for item in problem.observations
            ],
        }
    ).save(problem_path)

    calibration = calibrate_camera_lidar_confidence(
        correspondence_path,
        problem_path,
        thresholds=(0.5, 0.98, 0.99),
        calibration_split_seeds=(0, 1),
        evaluation_split_seeds=(0, 1),
        evaluation_dataset_ids_excluded=("evaluation-a",),
    )
    calibration.save(output)

    assert calibration.schema_version.endswith("/v0.2")
    assert calibration.status == "rejected"
    assert calibration.selected_minimum_confidence is None
    assert calibration.locked_refinement_options is None
    assert not any(item.gate_pass for item in calibration.candidates)
    assert "geometric_inlier_rate_failed" in calibration.candidates[0].gate_reasons
    assert validate_file(output, "camera-lidar-confidence-calibration").valid
    with pytest.raises(ValueError, match="rejected on development geometry"):
        run_probabilistic_camera_lidar_refinement(
            correspondence_path,
            problem_path,
            confidence_calibration_path=output,
        )


def test_probabilistic_uncertainty_ablation_is_paired_and_aggregated(
    tmp_path: Path,
) -> None:
    truth = SE3((0.10, -0.05, 0.02), (0.0, 0.0, 0.0, 1.0))
    frames = _frames(truth)
    correspondence_path = tmp_path / "correspondence.yaml"
    problem_path = tmp_path / "problem.yaml"
    _artifact(frames).save(correspondence_path)
    _problem(frames, reference=truth).save(problem_path)

    definition, benchmark = run_probabilistic_refinement_ablation(
        correspondence_path,
        problem_path,
        result_directory=tmp_path / "ablation-results",
        command="pytest probabilistic ablation",
        split_seeds=(3, 7),
        base_options=ProbabilisticCameraLidarRefinementOptions(
            initial_rotation_step_deg=0.5,
            initial_translation_step_m=0.05,
            minimum_rotation_step_deg=0.1,
            minimum_translation_step_m=0.01,
            max_evaluations=180,
        ),
        bootstrap_samples=100,
    )

    assert len(definition.methods) == 4
    assert {item.tool_version for item in definition.methods} == {"0.3"}
    assert len(definition.trials) == 8
    assert all(item.status == "success" for item in definition.trials)
    assert {
        item.name for item in definition.metrics
    } == {
        "holdout_reprojection_rmse_px",
        "rotation_error_deg",
        "translation_error_m",
    }
    assert len(benchmark.paired_comparisons) == 9
    assert benchmark.reference_method_id == "full_uncertainty"


def test_integrated_probabilistic_falsification_writes_verified_bundle(
    tmp_path: Path,
) -> None:
    truth = SE3((0.10, -0.05, 0.02), (0.0, 0.0, 0.0, 1.0))
    frames = _frames(truth)
    correspondence_path = tmp_path / "correspondence.yaml"
    problem_path = tmp_path / "problem.yaml"
    trace_path = tmp_path / "d2d-trace.yaml"
    protocol_path = tmp_path / "six-dof-protocol.yaml"
    _artifact(frames).save(correspondence_path)
    problem = _problem(frames, reference=truth)
    problem.save(problem_path)
    protocol = build_borer_six_dof_protocol(
        problem_path,
        perturbation_count=2,
        rotation_bound_deg=1.0,
        translation_bound_m=0.2,
        max_evaluations=40,
    )
    protocol.save(protocol_path)
    problem_digest = sha256_path(problem_path)
    protocol_digest = sha256_path(protocol_path)
    assert problem_digest is not None
    assert protocol_digest is not None
    d2d_initial = SE3(
        (0.05, -0.05, 0.02), (0.0, 0.0, 0.0, 1.0)
    )
    d2d_transform = TransformResult(
        parent="camera",
        child="lidar",
        translation_m=list(d2d_initial.translation_m),
        rotation_quat_xyzw=list(d2d_initial.rotation_quat_xyzw),
    )
    trace = CalibrationCandidateTrace(
        trace_id="integrated-d2d-trace",
        trial_id=protocol.perturbations[0].trial_id,
        problem_sha256=problem_digest,
        protocol_sha256=protocol_digest,
        solver="native_borer_d2d_six_dof",
        solver_version=BORER_SIX_DOF_BENCHMARK_VERSION,
        status="converged",
        initial_transform_camera_lidar=problem.initial_transform_camera_lidar,
        output_transform_camera_lidar=d2d_transform,
        evaluations=[],
        stopping_reason="fixture",
        runtime_seconds=0.1,
        outcome=CalibrationCandidateOutcome(
            rotation_error_deg=0.0,
            translation_error_m=0.05,
            hit=True,
            hit_reason="fixture",
        ),
        provenance=CameraLidarArtifactProvenance(
            generator="pytest",
            generator_version="1",
            source_sha256=problem_digest,
        ),
    )
    trace.save(trace_path)
    trace_directory = tmp_path / "d2d-traces"
    trace_directory.mkdir()
    trace.save(trace_directory / f"{trace.trial_id}.trace.yaml")
    second_trace = trace.model_copy(
        update={
            "trace_id": "integrated-d2d-trace-second",
            "trial_id": protocol.perturbations[1].trial_id,
            "outcome": CalibrationCandidateOutcome(
                rotation_error_deg=1.0,
                translation_error_m=0.5,
                hit=False,
                hit_reason="fixture known miss",
            ),
        }
    )
    second_trace.save(trace_directory / f"{second_trace.trial_id}.trace.yaml")

    mixed_trace_directory = tmp_path / "mixed-d2d-traces"
    mixed_trace_directory.mkdir()
    trace.save(mixed_trace_directory / f"{trace.trial_id}.trace.yaml")
    second_trace.model_copy(
        update={"solver": "different_native_six_dof_backend"}
    ).save(mixed_trace_directory / f"{second_trace.trial_id}.trace.yaml")
    with pytest.raises(ValueError, match="mixed D2D solver identities"):
        run_probabilistic_camera_lidar_falsification(
            correspondence_path,
            problem_path,
            initialization_trace_directory=mixed_trace_directory,
            benchmark_protocol_path=protocol_path,
            output_directory=tmp_path / "falsification-mixed-solvers",
            command="pytest rejects mixed D2D solver identities",
            split_seeds=(3,),
            bootstrap_samples=10,
        )

    artifacts = run_probabilistic_camera_lidar_falsification(
        correspondence_path,
        problem_path,
        initialization_trace_path=trace_path,
        benchmark_protocol_path=protocol_path,
        output_directory=tmp_path / "falsification",
        command="pytest integrated falsification",
        split_seeds=(3, 7),
        base_options=ProbabilisticCameraLidarRefinementOptions(
            initial_rotation_step_deg=0.5,
            initial_translation_step_m=0.05,
            minimum_rotation_step_deg=0.1,
            minimum_translation_step_m=0.01,
            max_evaluations=180,
        ),
        known_bad_min_holdout_rmse_delta_px=0.1,
        bootstrap_samples=100,
    )

    assert artifacts.assessment.status == "pass"
    assert artifacts.verification.valid is True
    assert artifacts.definition.benchmark_id.endswith(
        "-camera-lidar-falsification-v02"
    )
    assert (
        artifacts.definition.protocol.protocol_id
        == "camera_lidar_targetless_6dof_refinement/v0.2"
    )
    assert (
        artifacts.definition.provenance.generator_version
        == "calibrex.probabilistic_camera_lidar_falsification/v0.2"
    )
    probabilistic_method = next(
        item
        for item in artifacts.definition.methods
        if item.method_id == "d2d_initialized_probabilistic"
    )
    d2d_method = next(
        item
        for item in artifacts.definition.methods
        if item.method_id == "d2d_candidate_initial"
    )
    assert d2d_method.tool_version == BORER_SIX_DOF_BENCHMARK_VERSION
    assert probabilistic_method.tool_version == "0.3"
    assert all(
        load_probabilistic_refinement_result(path).provenance.generator_version
        == "0.3"
        for path in artifacts.result_paths
    )
    assert artifacts.bundle.artifact_count == 9
    assert len(artifacts.evidence.cases) == 24
    assert all(case.status == "pass" for case in artifacts.evidence.cases)
    assert all(
        validate_file(path, "probabilistic-refinement-result").valid
        for path in artifacts.result_paths
    )
    assert validate_file(
        protocol_path, "camera-lidar-benchmark-protocol"
    ).valid
    assert validate_file(
        artifacts.benchmark_path, "benchmark"
    ).valid
    assert validate_file(
        artifacts.definition_path, "benchmark-definition"
    ).valid
    assert validate_file(
        artifacts.evidence_path, "report-evidence"
    ).valid
    assert validate_file(
        artifacts.failure_analysis_path, "camera-lidar-failure-analysis"
    ).valid
    assert artifacts.failure_analysis.summary.case_count == len(artifacts.result_paths)
    assert validate_file(artifacts.assessment_path, "assessment").valid
    assert validate_file(artifacts.bundle_path, "evidence-bundle").valid
    assert validate_file(
        artifacts.verification_path, "evidence-bundle-verification"
    ).valid

    all_artifacts = run_probabilistic_camera_lidar_falsification(
        correspondence_path,
        problem_path,
        initialization_trace_directory=trace_directory,
        benchmark_protocol_path=protocol_path,
        output_directory=tmp_path / "falsification-all-trials",
        command="pytest integrated falsification all trials",
        split_seeds=(3,),
        base_options=ProbabilisticCameraLidarRefinementOptions(
            initial_rotation_step_deg=0.5,
            initial_translation_step_m=0.05,
            minimum_rotation_step_deg=0.1,
            minimum_translation_step_m=0.01,
            max_evaluations=180,
        ),
        known_bad_min_holdout_rmse_delta_px=0.1,
        bootstrap_samples=10,
    )
    assert len(all_artifacts.initialization_trace_paths) == 2
    assert len(all_artifacts.result_paths) == 2
    assert any(
        "d2d_initial_rotation_failure" in case.failure_categories
        for case in all_artifacts.failure_analysis.cases
    )
    assert all_artifacts.evidence.protocols[0].parameters[
        "d2d_full_protocol_coverage"
    ] is True
    assert len(all_artifacts.evidence.cases) == 24
    assert all(case.status == "pass" for case in all_artifacts.evidence.cases)
    assert all_artifacts.verification.valid is True

    cli_output = tmp_path / "cli-falsification"
    assert (
        main(
            [
                "camera-lidar",
                "benchmark-probabilistic-falsification",
                str(correspondence_path),
                str(problem_path),
                "--initial-trace",
                str(trace_path),
                "--protocol",
                str(protocol_path),
                "--output-dir",
                str(cli_output),
                "--split-seeds",
                "3,7",
                "--known-bad-min-holdout-rmse-delta-px",
                "0.1",
                "--bootstrap-samples",
                "100",
                "--json",
            ]
        )
        == 0
    )
    assert (cli_output / "verification.json").exists()
    assert "--bootstrap-samples 100" in (
        cli_output / "benchmark.definition.yaml"
    ).read_text(encoding="utf-8")

    rejected = run_probabilistic_camera_lidar_falsification(
        correspondence_path,
        problem_path,
        initialization_trace_path=trace_path,
        benchmark_protocol_path=protocol_path,
        output_directory=tmp_path / "falsification-rejected",
        command="pytest integrated falsification known-bad rejection",
        split_seeds=(3,),
        base_options=ProbabilisticCameraLidarRefinementOptions(
            initial_rotation_step_deg=0.5,
            initial_translation_step_m=0.05,
            minimum_rotation_step_deg=0.1,
            minimum_translation_step_m=0.01,
            max_evaluations=180,
        ),
        known_bad_min_holdout_rmse_delta_px=1.0e9,
        bootstrap_samples=10,
    )
    assert rejected.assessment.status == "fail"
    assert rejected.verification.valid is True
    assert all(case.status == "fail" for case in rejected.evidence.cases)


def test_falsification_does_not_count_initializer_rollback_as_refinement_pass() -> None:
    assert not _candidate_holdout_pass(
        "converged",
        False,
        1.0,
        1.0,
        8,
        8,
        None,
    )
    assert _candidate_holdout_pass(
        "converged",
        True,
        1.0,
        1.0,
        8,
        8,
        None,
    )


def _frames(transform: SE3) -> list[ProbabilisticCorrespondenceFrame]:
    camera = DepthCameraIntrinsics(
        width=1280,
        height=720,
        fx=600.0,
        fy=600.0,
        cx=640.0,
        cy=360.0,
    )
    base_points = (
        (-2.0, -1.0, 8.0),
        (-1.0, 1.0, 7.0),
        (0.5, -1.5, 9.0),
        (2.0, 1.0, 10.0),
        (-2.5, 0.5, 12.0),
        (1.5, -0.5, 6.0),
        (0.2, 1.8, 11.0),
        (2.8, -1.2, 13.0),
    )
    frames = []
    for frame_index in range(12):
        points = [
            (
                point[0] + 0.02 * frame_index,
                point[1] - 0.01 * frame_index,
                point[2],
            )
            for point in base_points
        ]
        correspondences = []
        for index, point in enumerate(points):
            camera_point = transform.transform_point(point)
            correspondences.append(
                ProbabilisticImageCorrespondence(
                    correspondence_id=str(index),
                    point_lidar_m=list(point),
                    image_mean_px=[
                        camera.fx * camera_point[0] / camera_point[2] + camera.cx,
                        camera.fy * camera_point[1] / camera_point[2] + camera.cy,
                    ],
                    image_covariance_px2=[1.0, 0.0, 0.0, 1.0],
                    outlier_probability=0.01,
                    reliability=0.99,
                )
            )
        frames.append(
            ProbabilisticCorrespondenceFrame(
                frame_id=f"frame-{frame_index:03d}",
                capture_time_ns=frame_index,
                camera_frame="camera",
                lidar_frame="lidar",
                intrinsics=camera,
                correspondences=correspondences,
            )
        )
    return frames


def _artifact(
    frames: list[ProbabilisticCorrespondenceFrame],
) -> ProbabilisticCorrespondenceArtifact:
    return ProbabilisticCorrespondenceArtifact(
        artifact_id="synthetic-multiframe",
        dataset_id="synthetic",
        split_id="test",
        dataset_license_spdx="CC0-1.0",
        provider=CorrespondenceProviderIdentity(
            provider="fixture",
            model="exact",
            version="1",
            source_repository="https://example.test/provider",
            source_commit="a" * 40,
            license_spdx="Apache-2.0",
            checkpoint_redistribution="not applicable",
        ),
        frames=frames,
        provenance=ProbabilisticCorrespondenceProvenance(
            generator="pytest",
            generator_version="1",
            input_sha256={"fixture": "b" * 64},
        ),
    )


def _problem(
    frames: list[ProbabilisticCorrespondenceFrame],
    *,
    reference: SE3 | None = None,
) -> CameraLidarCalibrationProblem:
    initial = SE3((0.05, -0.05, 0.02), (0.0, 0.0, 0.0, 1.0))
    transform = TransformResult(
        parent="camera",
        child="lidar",
        translation_m=list(initial.translation_m),
        rotation_quat_xyzw=list(initial.rotation_quat_xyzw),
    )
    reference_transform = (
        TransformResult(
            parent="camera",
            child="lidar",
            translation_m=list(reference.translation_m),
            rotation_quat_xyzw=list(reference.rotation_quat_xyzw),
        )
        if reference is not None
        else transform
    )
    return CameraLidarCalibrationProblem(
        problem_id="synthetic-d2d-initializer",
        dataset_id="synthetic",
        dataset_family="synthetic",
        sequence_id="test",
        depth_provider_path="/tmp/depth-provider.yaml",
        depth_provider_sha256="c" * 64,
        observations=[
            CameraLidarObservationBinding(
                frame_id=frame.frame_id,
                split_id="test",
                depth_observation_frame_id=frame.frame_id,
                lidar=DepthFileReference(
                    path=f"/tmp/{frame.frame_id}.npy",
                    sha256="d" * 64,
                    size_bytes=1,
                    encoding="npy_float64_xyz",
                ),
                lidar_capture_time_ns=frame.capture_time_ns,
            )
            for frame in frames
        ],
        reference_transform_camera_lidar=reference_transform,
        initial_transform_camera_lidar=transform,
        time_convention="synchronized synthetic observations",
        rotation_bound_deg=2.0,
        translation_bound_m=0.25,
        provenance=CameraLidarArtifactProvenance(
            generator="pytest",
            generator_version="1",
            source_sha256="e" * 64,
        ),
    )
