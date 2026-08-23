from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from tools import analyze_i2pnet_pose_benchmark as analyzer

from calibrex.core.camera_lidar_artifacts import CameraLidarHitDefinition
from calibrex.core.camera_lidar_pose_initializer_failure_analysis import (
    CameraLidarCorrectionVector,
    CameraLidarPoseInitializerFailureAnalysis,
    CameraLidarPoseInitializerFailureCase,
    CameraLidarPoseInitializerFailureFinding,
    CameraLidarPoseInitializerFailureProvenance,
    CameraLidarPoseInitializerFailureSummary,
    CameraLidarPoseInitializerFrameResponse,
    load_camera_lidar_pose_initializer_failure_analysis,
)
from calibrex.core.validation import validate_file

SHA = "a" * 64


def _artifact() -> CameraLidarPoseInitializerFailureAnalysis:
    correction = CameraLidarCorrectionVector(
        rotation_vector_deg=[-5.0, 0.0, 0.0],
        translation_m=[-0.25, 0.0, 0.0],
    )
    case = CameraLidarPoseInitializerFailureCase(
        trial_id="rx",
        initial_path="initial.yaml",
        initial_sha256=SHA,
        manifest_path="manifest.yaml",
        manifest_sha256=SHA,
        pose_path="pose.yaml",
        pose_sha256=SHA,
        initial_rotation_error_deg=5.0,
        initial_translation_error_m=0.25,
        required_correction=correction,
        aggregate_predicted_correction=correction,
        rotation_parallel_gain=1.0,
        rotation_orthogonal_ratio=0.0,
        rotation_alignment_cosine=1.0,
        translation_parallel_gain=1.0,
        translation_orthogonal_ratio=0.0,
        translation_alignment_cosine=1.0,
        output_rotation_error_deg=0.1,
        output_translation_error_m=0.1,
        hit=True,
        frames=[
            CameraLidarPoseInitializerFrameResponse(
                frame_id="frame-1",
                export_path="frame-1.npz",
                export_sha256=SHA,
                predicted_correction=correction,
                output_rotation_error_deg=0.1,
                output_translation_error_m=0.1,
            )
        ],
    )
    return CameraLidarPoseInitializerFailureAnalysis(
        analysis_id="analysis",
        protocol_id="protocol",
        protocol_sha256=SHA,
        problem_id="problem",
        problem_sha256=SHA,
        benchmark_definition_id="definition",
        benchmark_definition_sha256=SHA,
        dataset_id="dataset",
        partition="development",
        provider_id="provider",
        hit=CameraLidarHitDefinition(
            rotation_error_max_deg=0.5,
            translation_error_max_m=0.2,
        ),
        summary=CameraLidarPoseInitializerFailureSummary(
            trial_count=1,
            frame_response_count=1,
            hit_count=1,
            hit_rate=1.0,
            mean_initial_rotation_error_deg=5.0,
            mean_output_rotation_error_deg=0.1,
            mean_initial_translation_error_m=0.25,
            mean_output_translation_error_m=0.1,
            single_axis_rotation_parallel_gain={"x": 1.0},
            single_axis_rotation_orthogonal_ratio={"x": 0.0},
            single_axis_translation_parallel_gain={"x": 1.0},
            single_axis_translation_orthogonal_ratio={"x": 0.0},
        ),
        cases=[case],
        findings=[
            CameraLidarPoseInitializerFailureFinding(
                finding_id="fixture",
                severity="info",
                confidence="high",
                title="fixture",
                evidence="fixture evidence",
                supporting_trial_ids=["rx"],
                caveat="fixture caveat",
            )
        ],
        conclusion="fixture conclusion",
        provenance=CameraLidarPoseInitializerFailureProvenance(
            generator="pytest",
            generator_version="v0.1",
            source_paths={"protocol": "protocol.yaml"},
            source_sha256={"protocol": SHA},
        ),
    )


def test_failure_analysis_round_trip_and_auto_validation(tmp_path: Path) -> None:
    path = tmp_path / "failure-analysis.yaml"
    _artifact().save(path)

    loaded = load_camera_lidar_pose_initializer_failure_analysis(path)
    report = validate_file(path)

    assert loaded.summary.hit_rate == 1.0
    assert report.kind == "camera-lidar-pose-initializer-failure-analysis"


def test_failure_analysis_rejects_incomplete_provenance() -> None:
    payload = _artifact().model_dump(mode="json")
    payload["provenance"]["source_sha256"]["extra"] = SHA

    with pytest.raises(ValueError, match="equal keys"):
        CameraLidarPoseInitializerFailureAnalysis.model_validate(payload)


def test_failure_analysis_rejects_inconsistent_hit_summary() -> None:
    payload = _artifact().model_dump(mode="json")
    payload["summary"]["hit_rate"] = 0.5

    with pytest.raises(ValueError, match="hit_rate"):
        CameraLidarPoseInitializerFailureAnalysis.model_validate(payload)


def test_response_separates_parallel_gain_and_off_axis_ratio() -> None:
    gain, orthogonal, cosine = analyzer._response(
        np.asarray([2.0, 0.0, 0.0]),
        np.asarray([1.0, 2.0, 0.0]),
    )

    assert gain == pytest.approx(0.5)
    assert orthogonal == pytest.approx(1.0)
    assert cosine == pytest.approx(1.0 / math.sqrt(5.0))
