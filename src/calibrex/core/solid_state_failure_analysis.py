"""Schema-valid diagnostics for solid-state LiDAR benchmark counterexamples."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import Field, field_validator

from calibrex.core.continuous_time_lidar_artifacts import (
    ContinuousTimeLidarPairOptionsArtifact,
)
from calibrex.core.result import StrictModel

SOLID_STATE_FAILURE_ANALYSIS_SCHEMA_VERSION: Literal["slac.solid_state_failure_analysis/v0.1"] = (
    "slac.solid_state_failure_analysis/v0.1"
)

FailureAnalysisSeverity = Literal["info", "warning", "candidate_cause"]
FailureAnalysisConfidence = Literal["low", "medium", "high"]

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class SolidStateFailureAnalysisVariant(StrictModel):
    """Diagnostics copied from one continuous-time solver artifact."""

    id: str
    voxel_strategy: Literal["uniform", "adaptive"]
    outlier_policy: Literal["none", "mad"]
    result_path: str
    result_sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    artifact_schema_version: str | None = None
    optimization_status: str | None = None
    quality_grade: str | None = None
    initial_time_offset_sec: float | None = None
    estimated_time_offset_sec: float | None = None
    initial_train_rmse_m: float | None = Field(default=None, ge=0.0)
    final_train_rmse_m: float | None = Field(default=None, ge=0.0)
    final_holdout_rmse_m: float | None = Field(default=None, ge=0.0)
    train_correspondence_count: int = Field(default=0, ge=0)
    holdout_correspondence_count: int = Field(default=0, ge=0)
    observability_rank: int | None = Field(default=None, ge=0, le=6)
    condition_number: float | None = Field(default=None, ge=0.0)
    weak_directions: list[str] = Field(default_factory=list)
    residual_count: int = Field(default=0, ge=0)
    outlier_rejected_count: int = Field(default=0, ge=0)
    outlier_rejection_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    train_holdout_gap_m: float | None = None
    holdout_train_ratio: float | None = Field(default=None, ge=0.0)
    train_rmse_change_m: float | None = None
    profile_iteration_count: int = Field(default=0, ge=0)
    accepted_iteration_count: int = Field(default=0, ge=0)
    accepted_iteration_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    best_profile_holdout_rmse_m: float | None = Field(default=None, ge=0.0)
    profile_holdout_span_m: float | None = Field(default=None, ge=0.0)
    profile_offset_span_sec: float | None = Field(default=None, ge=0.0)
    options: ContinuousTimeLidarPairOptionsArtifact | None = None
    artifact_source_sha256: dict[str, str] = Field(default_factory=dict)

    @field_validator("artifact_source_sha256")
    @classmethod
    def validate_artifact_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        invalid = [key for key, digest in value.items() if _SHA256_RE.fullmatch(digest) is None]
        if invalid:
            raise ValueError(f"artifact_source_sha256 contains invalid SHA-256 values: {invalid}")
        return value


class SolidStateFailureAnalysisReplicate(StrictModel):
    """Paired adaptive/uniform diagnostics for one split and sampling seed."""

    id: str
    split_id: str
    seed: int
    status: str
    winner: Literal["adaptive", "uniform", "tie", "inconclusive"]
    holdout_improvement_percent: float | None = None
    adaptive: SolidStateFailureAnalysisVariant | None = None
    uniform: SolidStateFailureAnalysisVariant | None = None
    adaptive_holdout_minus_uniform_m: float | None = None
    adaptive_train_minus_uniform_m: float | None = None
    adaptive_generalization_gap_minus_uniform_m: float | None = None
    adaptive_offset_minus_uniform_sec: float | None = None
    adaptive_correspondence_delta: int | None = None
    adaptive_outlier_rate_minus_uniform: float | None = None
    adaptive_condition_number_minus_uniform: float | None = None
    diagnosis: str
    signals: list[str] = Field(default_factory=list)


class SolidStateFailureAnalysisFinding(StrictModel):
    """An evidence-backed observation or explicitly labelled causal hypothesis."""

    id: str
    severity: FailureAnalysisSeverity
    confidence: FailureAnalysisConfidence
    title: str
    evidence: str
    supporting_replicates: list[str] = Field(default_factory=list)
    caveat: str


class SolidStateFailureAnalysisSummary(StrictModel):
    """Aggregate statistics for the selected dataset."""

    replicate_count: int = Field(ge=0)
    scored_replicate_count: int = Field(ge=0)
    adaptive_wins: int = Field(ge=0)
    uniform_wins: int = Field(ge=0)
    ties: int = Field(ge=0)
    adaptive_win_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_holdout_improvement_percent: float | None = None
    mean_adaptive_train_minus_uniform_m: float | None = None
    mean_adaptive_generalization_gap_minus_uniform_m: float | None = None
    mean_adaptive_outlier_rejection_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    adaptive_rank_six_count: int = Field(ge=0)
    uniform_rank_six_count: int = Field(ge=0)
    adaptive_converged_count: int = Field(ge=0)
    uniform_converged_count: int = Field(ge=0)
    diagnosis_counts: dict[str, int] = Field(default_factory=dict)


class SolidStateFailureAnalysisProvenance(StrictModel):
    """Digest-bound inputs used to derive the analysis."""

    source_paths: list[str] = Field(min_length=1)
    source_sha256: dict[str, str] = Field(min_length=1)
    tool_name: str = "calibrex"
    tool_version: str | None = None
    git_commit: str | None = None
    notes: list[str] = Field(default_factory=list)

    @field_validator("source_sha256")
    @classmethod
    def validate_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        invalid = [key for key, digest in value.items() if _SHA256_RE.fullmatch(digest) is None]
        if invalid:
            raise ValueError(f"source_sha256 contains invalid SHA-256 values: {invalid}")
        return value


class SolidStateFailureAnalysisManifest(StrictModel):
    """Reproducible failure analysis for one benchmark dataset."""

    schema_version: Literal["slac.solid_state_failure_analysis/v0.1"] = (
        SOLID_STATE_FAILURE_ANALYSIS_SCHEMA_VERSION
    )
    tool: str
    tool_version: str
    benchmark_manifest_path: str
    benchmark_manifest_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    dataset_id: str
    dataset_name: str
    analysis_scope: str
    summary: SolidStateFailureAnalysisSummary
    replicates: list[SolidStateFailureAnalysisReplicate] = Field(default_factory=list)
    findings: list[SolidStateFailureAnalysisFinding] = Field(default_factory=list)
    conclusion: str
    provenance: SolidStateFailureAnalysisProvenance


def solid_state_failure_analysis_json_schema() -> dict[str, Any]:
    """Return the standalone JSON schema for failure-analysis artifacts."""

    return SolidStateFailureAnalysisManifest.model_json_schema()
