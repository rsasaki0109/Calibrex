import pytest
from pydantic import ValidationError

from calibrex.core.camera_lidar_artifacts import (
    CalibrationCandidateEvaluation,
    CalibrationCandidateOutcome,
    CalibrationCandidateTrace,
    CameraLidarArtifactProvenance,
    CameraLidarBenchmarkProtocol,
    CameraLidarHitDefinition,
    CameraLidarIsolationDeclaration,
    CameraLidarPerturbation,
)
from calibrex.core.result import RunInfo, TransformResult


def _transform() -> TransformResult:
    return TransformResult(
        parent="camera0",
        child="lidar0",
        translation_m=[0.0, 0.0, 0.0],
        rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
    )


def _provenance() -> CameraLidarArtifactProvenance:
    return CameraLidarArtifactProvenance(
        generator="fixture",
        generator_version="v0.1",
        source_sha256="1" * 64,
    )


def test_rotation_protocol_rejects_translation_perturbation() -> None:
    payload = {
        "protocol_id": "fixture",
        "primary_source": "https://example.test/paper",
        "dataset_id": "fixture",
        "problem_sha256": "1" * 64,
        "degrees_of_freedom": "rotation_only",
        "frame_sampling": "uniform_over_sequence",
        "frame_ids": ["000000"],
        "perturbation_method": "fibonacci_sphere",
        "perturbation_count": 1,
        "rotation_magnitude_deg": 10.0,
        "translation_magnitude_m": 0.0,
        "perturbations": [
            {
                "trial_id": "trial-0",
                "rotation_deg_xyz": [1.0, 0.0, 0.0],
                "translation_m_xyz": [0.1, 0.0, 0.0],
            }
        ],
        "hit": {
            "rotation_error_max_deg": 0.5,
            "translation_error_max_m": 0.2,
        },
        "objective": "per_frame_mean_mutual_information",
        "histogram_bins": 32,
        "min_visible_points": 64,
        "visibility": "z_buffer_nearest_range",
        "optimizer": "fixture",
        "optimizer_options": {},
        "isolation": {
            "provider_selection_split": "train",
            "threshold_selection_split": "validation",
            "evaluation_split": "test",
            "evidence": "fixture",
        },
        "metric_definitions": {"hit": "fixture"},
        "provenance": _provenance().model_dump(mode="json"),
    }

    with pytest.raises(ValidationError, match="cannot perturb translation"):
        CameraLidarBenchmarkProtocol.model_validate(payload)


def test_candidate_trace_rejects_noncontiguous_evaluations() -> None:
    trace = CalibrationCandidateTrace(
        trace_id="fixture",
        trial_id="trial-0",
        problem_sha256="1" * 64,
        protocol_sha256="2" * 64,
        solver="fixture",
        solver_version="v0.1",
        status="converged",
        initial_transform_camera_lidar=_transform(),
        output_transform_camera_lidar=_transform(),
        evaluations=[
            CalibrationCandidateEvaluation(
                evaluation=1,
                parameters={},
                objective=0.0,
                mutual_information=0.0,
                normalized_mutual_information=0.0,
                evaluated_frame_count=1,
                accepted=True,
            )
        ],
        stopping_reason="fixture",
        runtime_seconds=0.1,
        outcome=CalibrationCandidateOutcome(
            rotation_error_deg=0.0,
            translation_error_m=0.0,
            hit=True,
            hit_reason="fixture",
        ),
        provenance=_provenance(),
    )
    payload = trace.model_dump(mode="json")
    payload["evaluations"][0]["evaluation"] = 2

    with pytest.raises(ValidationError, match="contiguous"):
        CalibrationCandidateTrace.model_validate(payload)


def test_artifact_module_does_not_depend_on_run_info() -> None:
    assert RunInfo.model_fields["provenance"] is not None
    assert CameraLidarPerturbation.model_fields["trial_id"] is not None
    assert CameraLidarHitDefinition.model_fields["comparison"] is not None
    assert CameraLidarIsolationDeclaration.model_fields[
        "test_data_used_for_selection"
    ].default is False
