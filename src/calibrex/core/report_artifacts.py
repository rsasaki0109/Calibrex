"""Typed schemas for machine-readable report sidecars."""

from __future__ import annotations

from typing import Any, Final, Literal

from pydantic import BaseModel, Field

from calibrex.core.result import (
    ArtifactSet,
    DegeneracyResult,
    Grade,
    MetricResult,
    ObservabilityResult,
    QualitySummary,
    StrictModel,
)

REPORT_SUMMARY_SCHEMA_VERSION: Literal["slac.report.summary/v0.1"] = (
    "slac.report.summary/v0.1"
)
REPORT_METRICS_SCHEMA_VERSION: Literal["slac.report.metrics/v0.1"] = (
    "slac.report.metrics/v0.1"
)
REPORT_OBSERVABILITY_SCHEMA_VERSION: Literal["slac.report.observability/v0.1"] = (
    "slac.report.observability/v0.1"
)
REPORT_DEGENERACY_SCHEMA_VERSION: Literal["slac.report.degeneracy/v0.1"] = (
    "slac.report.degeneracy/v0.1"
)
REPORT_EVIDENCE_SCHEMA_VERSION: Literal["slac.report.evidence/v0.1"] = (
    "slac.report.evidence/v0.1"
)


class ReportRunInfo(StrictModel):
    """Minimal run metadata repeated in report sidecars."""

    id: str
    status: Literal["success", "warning", "failed", "dry_run"]
    domain: str
    slac_version: str
    git_commit: str | None = None
    created_at: str
    dataset_type: str | None = None
    dataset_path: str | None = None


class GradeCounts(StrictModel):
    """PASS/WARN/FAIL counts for report summaries."""

    pass_: int = Field(default=0, ge=0, alias="pass")
    warn: int = Field(default=0, ge=0)
    fail: int = Field(default=0, ge=0)


class EvidenceSummaryItem(StrictModel):
    """Machine-readable evidence summary row for report consumers."""

    family: str
    check: str
    status: Grade
    evidence: str
    interpretation: str
    metric_ids: list[str] = Field(default_factory=list)


class EvidenceCaseItem(StrictModel):
    """Machine-readable evidence case for one negative-control probe."""

    family: str
    case_id: str
    check: str
    status: Grade
    dof: str | None = None
    amount: float | None = None
    unit: str | None = None
    convention: str | None = None
    metric_values: dict[str, float | None] = Field(default_factory=dict)
    delta_values: dict[str, float | None] = Field(default_factory=dict)


class EvidenceInputFileItem(StrictModel):
    """Raw input file materialized into an evidence artifact."""

    path: str
    role: str | None = None
    sha256: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    source_url: str | None = None


class EvidenceMaterializationInfo(StrictModel):
    """How an evidence artifact was materialized for this report."""

    metrics_origin: Literal["recomputed", "cached", "unknown"] = "unknown"
    data_verified: bool | None = None
    computed_at: str | None = None
    report_generated_at: str | None = None


class SourceEvidenceReference(StrictModel):
    """Reference from a derived sidecar back to its evidence artifact."""

    path: str | None = None
    sha256: str | None = None
    schema_version: str | None = None
    run_id: str | None = None


class EvidenceProtocolItem(StrictModel):
    """Declared protocol metadata for one evidence family."""

    family: str
    protocol_id: str
    status: str | None = None
    split_policy: str | None = None
    independent_holdout: bool | None = None
    candidate_transform: str | None = None
    transform_convention: str | None = None
    known_bad_perturbation: str | None = None
    known_bad_case_count: int | None = Field(default=None, ge=0)
    metric_ids: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)


class ReportSummaryArtifact(StrictModel):
    """Schema for `summary.json` report sidecars."""

    schema_version: Literal["slac.report.summary/v0.1"] = REPORT_SUMMARY_SCHEMA_VERSION
    run: ReportRunInfo
    source_evidence: SourceEvidenceReference | None = None
    materialization: EvidenceMaterializationInfo = Field(
        default_factory=EvidenceMaterializationInfo
    )
    quality: QualitySummary
    metric_counts: GradeCounts
    transform_counts: GradeCounts
    candidate_extrinsic_count: int = Field(ge=0)
    reference_extrinsic_count: int = Field(ge=0)
    matched_candidate_reference_count: int = Field(ge=0)
    weak_direction_count: int = Field(ge=0)
    artifact_paths: ArtifactSet = Field(default_factory=ArtifactSet)
    evidence_summaries: list[EvidenceSummaryItem] = Field(default_factory=list)


class ReportMetricsArtifact(StrictModel):
    """Schema for `metrics.json` report sidecars."""

    schema_version: Literal["slac.report.metrics/v0.1"] = REPORT_METRICS_SCHEMA_VERSION
    run: ReportRunInfo
    source_evidence: SourceEvidenceReference | None = None
    metrics: dict[str, MetricResult] = Field(default_factory=dict)
    metric_families: dict[str, MetricFamilySummary] = Field(default_factory=dict)


class MetricFamilySummary(StrictModel):
    """Metric counts and warning/failure rollup for one metric family."""

    family: str
    metric_count: int = Field(ge=0)
    grade_counts: GradeCounts
    worst_grade: Grade = "pass"
    holdout_metric_count: int = Field(default=0, ge=0)
    warn_or_fail_metrics: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)


class ReportObservabilityArtifact(StrictModel):
    """Schema for `observability.json` report sidecars."""

    schema_version: Literal["slac.report.observability/v0.1"] = (
        REPORT_OBSERVABILITY_SCHEMA_VERSION
    )
    run: ReportRunInfo
    source_evidence: SourceEvidenceReference | None = None
    observability: ObservabilityResult
    weak_directions: list[str] = Field(default_factory=list)
    metrics: dict[str, MetricResult] = Field(default_factory=dict)


class ReportDegeneracyArtifact(StrictModel):
    """Schema for `degeneracy.json` report sidecars."""

    schema_version: Literal["slac.report.degeneracy/v0.1"] = REPORT_DEGENERACY_SCHEMA_VERSION
    run: ReportRunInfo
    source_evidence: SourceEvidenceReference | None = None
    degeneracy: DegeneracyResult
    quality_warnings: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    metrics: dict[str, MetricResult] = Field(default_factory=dict)


class ReportEvidenceArtifact(StrictModel):
    """Schema for `evidence.json` report sidecars."""

    schema_version: Literal["slac.report.evidence/v0.1"] = REPORT_EVIDENCE_SCHEMA_VERSION
    run: ReportRunInfo
    materialization: EvidenceMaterializationInfo = Field(
        default_factory=EvidenceMaterializationInfo
    )
    protocols: list[EvidenceProtocolItem] = Field(default_factory=list)
    input_files: list[EvidenceInputFileItem] = Field(default_factory=list)
    summaries: list[EvidenceSummaryItem] = Field(default_factory=list)
    cases: list[EvidenceCaseItem] = Field(default_factory=list)


_REPORT_ARTIFACT_MODELS: Final[dict[str, type[BaseModel]]] = {
    "report-summary": ReportSummaryArtifact,
    "report-metrics": ReportMetricsArtifact,
    "report-observability": ReportObservabilityArtifact,
    "report-degeneracy": ReportDegeneracyArtifact,
    "report-evidence": ReportEvidenceArtifact,
}


def report_artifact_schema_kinds() -> tuple[str, ...]:
    """Return supported report sidecar schema names for the CLI."""

    return tuple(_REPORT_ARTIFACT_MODELS)


def is_report_artifact_schema_kind(kind: str) -> bool:
    """Return whether `kind` names a report sidecar schema."""

    return kind in _REPORT_ARTIFACT_MODELS


def report_artifact_json_schema(kind: str) -> dict[str, Any]:
    """Return the JSON schema for one machine-readable report sidecar."""

    return _report_artifact_model(kind).model_json_schema()


def validate_report_sidecar_payload(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a report sidecar payload before writing it."""

    model = _report_artifact_model(kind).model_validate(payload)
    return model.model_dump(mode="json", by_alias=True, exclude_none=True)


def _report_artifact_model(kind: str) -> type[BaseModel]:
    try:
        return _REPORT_ARTIFACT_MODELS[kind]
    except KeyError as exc:
        supported = ", ".join(report_artifact_schema_kinds())
        msg = f"unknown report artifact schema kind: {kind}; expected one of {supported}"
        raise ValueError(msg) from exc
