"""Schema-valid failure analysis for Camera--LiDAR refinement runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.result import StrictModel

CAMERA_LIDAR_FAILURE_ANALYSIS_SCHEMA_VERSION: Literal[
    "slac.camera_lidar_failure_analysis/v0.1"
] = "slac.camera_lidar_failure_analysis/v0.1"

FailureAnalysisSeverity = Literal["info", "warning", "candidate_cause"]
FailureAnalysisConfidence = Literal["low", "medium", "high"]


class CameraLidarFailureCase(StrictModel):
    """One D2D initialization and probabilistic refinement outcome."""

    case_id: str
    trial_id: str
    seed: int
    result_path: str
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    initialization_trace_path: str
    initialization_trace_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    d2d_status: str
    refinement_status: str
    candidate_accepted: bool | None = None
    selected_source: Literal[
        "refined_candidate", "initializer_rollback"
    ] | None = None
    acceptance_reasons: list[str] = Field(default_factory=list)
    d2d_initial_hit: bool
    d2d_initial_rotation_error_deg: float = Field(ge=0.0)
    d2d_initial_translation_error_m: float = Field(ge=0.0)
    initial_rotation_error_deg: float | None = Field(default=None, ge=0.0)
    candidate_rotation_error_deg: float | None = Field(default=None, ge=0.0)
    final_rotation_error_deg: float | None = Field(default=None, ge=0.0)
    initial_translation_error_m: float | None = Field(default=None, ge=0.0)
    candidate_translation_error_m: float | None = Field(default=None, ge=0.0)
    final_translation_error_m: float | None = Field(default=None, ge=0.0)
    initial_holdout_rmse_px: float | None = Field(default=None, ge=0.0)
    candidate_holdout_rmse_px: float | None = Field(default=None, ge=0.0)
    final_holdout_rmse_px: float | None = Field(default=None, ge=0.0)
    candidate_rotation_failure: bool | None = None
    candidate_translation_failure: bool | None = None
    candidate_holdout_degradation: bool | None = None
    rotation_failure: bool
    translation_failure: bool
    holdout_degradation: bool
    failure_categories: list[str] = Field(default_factory=list)
    diagnosis: str


class CameraLidarFailureFinding(StrictModel):
    """An observed pattern with an explicit non-causal caveat."""

    id: str
    severity: FailureAnalysisSeverity
    confidence: FailureAnalysisConfidence
    title: str
    evidence: str
    supporting_case_ids: list[str] = Field(default_factory=list)
    caveat: str


class CameraLidarFailureSummary(StrictModel):
    """Aggregate pose and holdout failure statistics."""

    case_count: int = Field(ge=0)
    scored_case_count: int = Field(ge=0)
    d2d_initial_hit_count: int = Field(ge=0)
    d2d_initial_hit_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    refinement_converged_count: int = Field(ge=0)
    refinement_converged_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    candidate_accepted_count: int | None = Field(default=None, ge=0)
    candidate_accepted_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    initializer_rollback_count: int | None = Field(default=None, ge=0)
    initializer_rollback_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    candidate_rotation_failure_count: int | None = Field(default=None, ge=0)
    candidate_rotation_failure_rate: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    candidate_translation_failure_count: int | None = Field(default=None, ge=0)
    candidate_translation_failure_rate: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    candidate_holdout_degradation_count: int | None = Field(default=None, ge=0)
    candidate_holdout_degradation_rate: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    rotation_failure_count: int = Field(ge=0)
    rotation_failure_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    translation_failure_count: int = Field(ge=0)
    translation_failure_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    holdout_degradation_count: int = Field(ge=0)
    holdout_degradation_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_final_rotation_error_deg: float | None = Field(default=None, ge=0.0)
    p90_final_rotation_error_deg: float | None = Field(default=None, ge=0.0)
    mean_final_translation_error_m: float | None = Field(default=None, ge=0.0)
    p90_final_translation_error_m: float | None = Field(default=None, ge=0.0)
    diagnosis_counts: dict[str, int] = Field(default_factory=dict)


class CameraLidarFailureAnalysisProvenance(StrictModel):
    """Digest-bound inputs used to derive the failure analysis."""

    source_paths: list[str] = Field(min_length=1)
    source_sha256: dict[str, str] = Field(min_length=1)
    generator: str = "calibrex.evaluation.camera_lidar_failure_analysis"
    generator_version: str = "v0.1"
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @field_validator("source_sha256")
    @classmethod
    def validate_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        invalid = [
            key
            for key, digest in value.items()
            if len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ]
        if invalid:
            raise ValueError(f"source_sha256 contains invalid SHA-256 values: {invalid}")
        return value


class CameraLidarFailureAnalysisArtifact(StrictModel):
    """Observed translation/rotation failure profile for one frozen run."""

    schema_version: Literal[
        "slac.camera_lidar_failure_analysis/v0.1"
    ] = CAMERA_LIDAR_FAILURE_ANALYSIS_SCHEMA_VERSION
    artifact_id: str
    dataset_id: str
    protocol_id: str
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_path: str
    rotation_threshold_deg: float = Field(gt=0.0)
    translation_threshold_m: float = Field(gt=0.0)
    analysis_scope: str
    summary: CameraLidarFailureSummary
    cases: list[CameraLidarFailureCase] = Field(min_length=1)
    findings: list[CameraLidarFailureFinding] = Field(default_factory=list)
    conclusion: str
    provenance: CameraLidarFailureAnalysisProvenance

    @model_validator(mode="after")
    def check_cases(self) -> CameraLidarFailureAnalysisArtifact:
        """Require one unique case record per scored refinement result."""

        case_ids = [item.case_id for item in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Camera-LiDAR failure-analysis case IDs must be unique")
        if self.summary.case_count != len(self.cases):
            raise ValueError("failure-analysis summary case_count does not match cases")
        return self

    def save(self, path: str | Path) -> None:
        """Save this analysis as schema-valid YAML or JSON."""

        write_mapping(Path(path), self.model_dump(mode="json", exclude_none=True))


def load_camera_lidar_failure_analysis(
    path: str | Path,
) -> CameraLidarFailureAnalysisArtifact:
    """Load and validate a Camera-LiDAR failure-analysis artifact."""

    return CameraLidarFailureAnalysisArtifact.model_validate(read_mapping(Path(path)))


def camera_lidar_failure_analysis_json_schema() -> dict[str, Any]:
    """Return the standalone JSON Schema for failure-analysis artifacts."""

    return CameraLidarFailureAnalysisArtifact.model_json_schema()
