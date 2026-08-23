"""Schema-valid failure analysis for Camera--LiDAR pose initializers."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from calibrex.core.camera_lidar_artifacts import CameraLidarHitDefinition
from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.result import StrictModel

CAMERA_LIDAR_POSE_INITIALIZER_FAILURE_ANALYSIS_SCHEMA_VERSION: Literal[
    "slac.camera_lidar_pose_initializer_failure_analysis/v0.1"
] = "slac.camera_lidar_pose_initializer_failure_analysis/v0.1"


class CameraLidarCorrectionVector(StrictModel):
    """One left-camera-frame SE(3) correction in axis-angle coordinates."""

    rotation_vector_deg: list[float] = Field(min_length=3, max_length=3)
    translation_m: list[float] = Field(min_length=3, max_length=3)

    @field_validator("rotation_vector_deg", "translation_m")
    @classmethod
    def validate_finite_vector(cls, value: list[float]) -> list[float]:
        """Reject non-finite correction components."""

        if not all(math.isfinite(item) for item in value):
            raise ValueError("correction vectors must be finite")
        return value


class CameraLidarPoseInitializerFrameResponse(StrictModel):
    """Provider correction and resulting error for one frozen frame."""

    frame_id: str = Field(min_length=1)
    export_path: str = Field(min_length=1)
    export_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    predicted_correction: CameraLidarCorrectionVector
    output_rotation_error_deg: float = Field(ge=0.0)
    output_translation_error_m: float = Field(ge=0.0)


class CameraLidarPoseInitializerFailureCase(StrictModel):
    """Required and observed correction response for one perturbation."""

    trial_id: str = Field(min_length=1)
    initial_path: str = Field(min_length=1)
    initial_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_path: str = Field(min_length=1)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pose_path: str = Field(min_length=1)
    pose_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    initial_rotation_error_deg: float = Field(ge=0.0)
    initial_translation_error_m: float = Field(ge=0.0)
    required_correction: CameraLidarCorrectionVector
    aggregate_predicted_correction: CameraLidarCorrectionVector
    rotation_parallel_gain: float | None = None
    rotation_orthogonal_ratio: float | None = Field(default=None, ge=0.0)
    rotation_alignment_cosine: float | None = Field(default=None, ge=-1.0, le=1.0)
    translation_parallel_gain: float | None = None
    translation_orthogonal_ratio: float | None = Field(default=None, ge=0.0)
    translation_alignment_cosine: float | None = Field(default=None, ge=-1.0, le=1.0)
    output_rotation_error_deg: float = Field(ge=0.0)
    output_translation_error_m: float = Field(ge=0.0)
    hit: bool
    frames: list[CameraLidarPoseInitializerFrameResponse] = Field(min_length=1)

    @field_validator("rotation_parallel_gain", "translation_parallel_gain")
    @classmethod
    def validate_finite_gain(cls, value: float | None) -> float | None:
        """Reject non-finite signed response gains."""

        if value is not None and not math.isfinite(value):
            raise ValueError("pose-initializer response gains must be finite")
        return value

    @model_validator(mode="after")
    def check_frame_ids(self) -> CameraLidarPoseInitializerFailureCase:
        """Require one response per unique frame."""

        frame_ids = [item.frame_id for item in self.frames]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("pose-initializer frame responses must be unique")
        return self


class CameraLidarPoseInitializerFailureFinding(StrictModel):
    """One evidence-backed diagnostic association, not a causal claim."""

    finding_id: str = Field(min_length=1)
    severity: Literal["info", "warning", "candidate_cause"]
    confidence: Literal["low", "medium", "high"]
    title: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    supporting_trial_ids: list[str] = Field(default_factory=list)
    caveat: str = Field(min_length=1)


class CameraLidarPoseInitializerFailureSummary(StrictModel):
    """Aggregate recovery and single-axis response statistics."""

    trial_count: int = Field(gt=0)
    frame_response_count: int = Field(gt=0)
    hit_count: int = Field(ge=0)
    hit_rate: float = Field(ge=0.0, le=1.0)
    mean_initial_rotation_error_deg: float = Field(ge=0.0)
    mean_output_rotation_error_deg: float = Field(ge=0.0)
    mean_initial_translation_error_m: float = Field(ge=0.0)
    mean_output_translation_error_m: float = Field(ge=0.0)
    single_axis_rotation_parallel_gain: dict[str, float] = Field(default_factory=dict)
    single_axis_rotation_orthogonal_ratio: dict[str, float] = Field(default_factory=dict)
    single_axis_translation_parallel_gain: dict[str, float] = Field(default_factory=dict)
    single_axis_translation_orthogonal_ratio: dict[str, float] = Field(default_factory=dict)

    @field_validator(
        "single_axis_rotation_parallel_gain",
        "single_axis_rotation_orthogonal_ratio",
        "single_axis_translation_parallel_gain",
        "single_axis_translation_orthogonal_ratio",
    )
    @classmethod
    def validate_axis_maps(cls, value: dict[str, float]) -> dict[str, float]:
        """Require finite values for declared Cartesian axes."""

        if not set(value).issubset({"x", "y", "z"}):
            raise ValueError("single-axis response maps may contain only x, y, and z")
        if not all(math.isfinite(item) for item in value.values()):
            raise ValueError("single-axis response values must be finite")
        return value

    @model_validator(mode="after")
    def check_axis_map_pairs(self) -> CameraLidarPoseInitializerFailureSummary:
        """Align parallel and orthogonal statistics for each response type."""

        if self.hit_count > self.trial_count:
            raise ValueError("failure-analysis hit_count cannot exceed trial_count")
        if not math.isclose(
            self.hit_rate,
            self.hit_count / self.trial_count,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("failure-analysis hit_rate does not match hit_count")
        if set(self.single_axis_rotation_parallel_gain) != set(
            self.single_axis_rotation_orthogonal_ratio
        ):
            raise ValueError("single-axis rotation response maps must have equal keys")
        if set(self.single_axis_translation_parallel_gain) != set(
            self.single_axis_translation_orthogonal_ratio
        ):
            raise ValueError("single-axis translation response maps must have equal keys")
        return self


class CameraLidarPoseInitializerFailureProvenance(StrictModel):
    """Complete source lineage for a pose-initializer failure analysis."""

    generator: str = Field(min_length=1)
    generator_version: str = Field(min_length=1)
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    source_paths: dict[str, str] = Field(min_length=1)
    source_sha256: dict[str, str] = Field(min_length=1)
    notes: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @model_validator(mode="after")
    def check_sources(self) -> CameraLidarPoseInitializerFailureProvenance:
        """Require every named source path to have one valid SHA-256 digest."""

        if set(self.source_paths) != set(self.source_sha256):
            raise ValueError("failure-analysis source paths and digests must have equal keys")
        invalid = [
            name
            for name, digest in self.source_sha256.items()
            if len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ]
        if invalid:
            raise ValueError("invalid source SHA-256 values: " + ", ".join(invalid))
        return self


class CameraLidarPoseInitializerFailureAnalysis(StrictModel):
    """Digest-bound response analysis for one frozen pose-initializer matrix."""

    schema_version: Literal[
        "slac.camera_lidar_pose_initializer_failure_analysis/v0.1"
    ] = CAMERA_LIDAR_POSE_INITIALIZER_FAILURE_ANALYSIS_SCHEMA_VERSION
    analysis_id: str = Field(min_length=1)
    protocol_id: str = Field(min_length=1)
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    problem_id: str = Field(min_length=1)
    problem_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark_definition_id: str = Field(min_length=1)
    benchmark_definition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_id: str = Field(min_length=1)
    partition: Literal["development", "evaluation"]
    provider_id: str = Field(min_length=1)
    correction_convention: Literal["left_camera_frame_se3"] = "left_camera_frame_se3"
    hit: CameraLidarHitDefinition
    summary: CameraLidarPoseInitializerFailureSummary
    cases: list[CameraLidarPoseInitializerFailureCase] = Field(min_length=1)
    findings: list[CameraLidarPoseInitializerFailureFinding] = Field(default_factory=list)
    conclusion: str = Field(min_length=1)
    provenance: CameraLidarPoseInitializerFailureProvenance

    @model_validator(mode="after")
    def check_matrix(self) -> CameraLidarPoseInitializerFailureAnalysis:
        """Require complete unique trial and finding identities."""

        trial_ids = [item.trial_id for item in self.cases]
        if len(trial_ids) != len(set(trial_ids)):
            raise ValueError("pose-initializer failure-analysis trials must be unique")
        if self.summary.trial_count != len(self.cases):
            raise ValueError("failure-analysis summary trial_count does not match cases")
        if self.summary.frame_response_count != sum(len(item.frames) for item in self.cases):
            raise ValueError("failure-analysis frame_response_count does not match cases")
        if self.summary.hit_count != sum(item.hit for item in self.cases):
            raise ValueError("failure-analysis summary hit_count does not match cases")
        finding_ids = [item.finding_id for item in self.findings]
        if len(finding_ids) != len(set(finding_ids)):
            raise ValueError("pose-initializer finding IDs must be unique")
        known_trials = set(trial_ids)
        unknown_trials = sorted(
            {
                trial_id
                for finding in self.findings
                for trial_id in finding.supporting_trial_ids
                if trial_id not in known_trials
            }
        )
        if unknown_trials:
            raise ValueError(
                "pose-initializer findings reference unknown trials: "
                + ", ".join(unknown_trials)
            )
        return self

    def save(self, path: str | Path) -> None:
        """Save this analysis as schema-valid YAML or JSON."""

        write_mapping(Path(path), self.model_dump(mode="json", exclude_none=True))


def load_camera_lidar_pose_initializer_failure_analysis(
    path: str | Path,
) -> CameraLidarPoseInitializerFailureAnalysis:
    """Load and validate a pose-initializer failure analysis."""

    return CameraLidarPoseInitializerFailureAnalysis.model_validate(read_mapping(Path(path)))


def camera_lidar_pose_initializer_failure_analysis_json_schema() -> dict[str, Any]:
    """Return the standalone JSON Schema for pose-initializer failure analysis."""

    return CameraLidarPoseInitializerFailureAnalysis.model_json_schema()
