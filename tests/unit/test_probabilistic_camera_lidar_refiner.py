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
    assert result.as_dict()["method"] == (
        "probabilistic_multiframe_refinement/v0.2"
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
