"""Validation helpers for schema-versioned slac artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel

from slac.core.assessment import ASSESSMENT_SCHEMA_VERSION, AssessmentArtifact
from slac.core.config import CONFIG_SCHEMA_VERSION, CalibrationConfig
from slac.core.evidence_bundle import (
    EVIDENCE_BUNDLE_SCHEMA_VERSION,
    EVIDENCE_BUNDLE_VERIFICATION_SCHEMA_VERSION,
    EvidenceBundleManifest,
    EvidenceBundleVerification,
)
from slac.core.evidence_contract import (
    POLICY_SCHEMA_VERSION,
    PROTOCOL_SCHEMA_VERSION,
    PolicyArtifact,
    ProtocolArtifact,
)
from slac.core.exceptions import SlacError
from slac.core.io import read_mapping
from slac.core.online_timeline import (
    ONLINE_TIMELINE_SCHEMA_VERSION,
    OnlineCalibrationTimelineArtifact,
)
from slac.core.report_artifacts import (
    REPORT_DEGENERACY_SCHEMA_VERSION,
    REPORT_EVIDENCE_SCHEMA_VERSION,
    REPORT_METRICS_SCHEMA_VERSION,
    REPORT_OBSERVABILITY_SCHEMA_VERSION,
    REPORT_SUMMARY_SCHEMA_VERSION,
    ReportDegeneracyArtifact,
    ReportEvidenceArtifact,
    ReportMetricsArtifact,
    ReportObservabilityArtifact,
    ReportSummaryArtifact,
)
from slac.core.result import RESULT_SCHEMA_VERSION, CalibrationResult, StrictModel
from slac.core.transform_artifacts import (
    TRANSFORMS_SCHEMA_VERSION,
    TransformArtifact,
)
from slac.data.manifest import DATASET_MANIFEST_SCHEMA_VERSION, DatasetManifest
from slac.evaluation.compare import COMPARISON_SCHEMA_VERSION, ResultComparison
from slac.evaluation.report_compare import (
    REPORT_COMPARISON_SCHEMA_VERSION,
    ReportComparison,
)

ValidationKind = Literal[
    "auto",
    "config",
    "result",
    "comparison",
    "report-comparison",
    "assessment",
    "policy",
    "protocol",
    "transforms",
    "dataset-manifest",
    "report-summary",
    "report-metrics",
    "report-observability",
    "report-degeneracy",
    "report-evidence",
    "evidence-bundle",
    "evidence-bundle-verification",
    "online-timeline",
]

_MODEL_BY_KIND: Final[dict[str, type[BaseModel]]] = {
    "config": CalibrationConfig,
    "result": CalibrationResult,
    "comparison": ResultComparison,
    "report-comparison": ReportComparison,
    "assessment": AssessmentArtifact,
    "policy": PolicyArtifact,
    "protocol": ProtocolArtifact,
    "transforms": TransformArtifact,
    "dataset-manifest": DatasetManifest,
    "report-summary": ReportSummaryArtifact,
    "report-metrics": ReportMetricsArtifact,
    "report-observability": ReportObservabilityArtifact,
    "report-degeneracy": ReportDegeneracyArtifact,
    "report-evidence": ReportEvidenceArtifact,
    "evidence-bundle": EvidenceBundleManifest,
    "evidence-bundle-verification": EvidenceBundleVerification,
    "online-timeline": OnlineCalibrationTimelineArtifact,
}

_KIND_BY_SCHEMA_VERSION: Final[dict[str, str]] = {
    CONFIG_SCHEMA_VERSION: "config",
    RESULT_SCHEMA_VERSION: "result",
    COMPARISON_SCHEMA_VERSION: "comparison",
    REPORT_COMPARISON_SCHEMA_VERSION: "report-comparison",
    ASSESSMENT_SCHEMA_VERSION: "assessment",
    POLICY_SCHEMA_VERSION: "policy",
    PROTOCOL_SCHEMA_VERSION: "protocol",
    TRANSFORMS_SCHEMA_VERSION: "transforms",
    DATASET_MANIFEST_SCHEMA_VERSION: "dataset-manifest",
    REPORT_SUMMARY_SCHEMA_VERSION: "report-summary",
    REPORT_METRICS_SCHEMA_VERSION: "report-metrics",
    REPORT_OBSERVABILITY_SCHEMA_VERSION: "report-observability",
    REPORT_DEGENERACY_SCHEMA_VERSION: "report-degeneracy",
    REPORT_EVIDENCE_SCHEMA_VERSION: "report-evidence",
    EVIDENCE_BUNDLE_SCHEMA_VERSION: "evidence-bundle",
    EVIDENCE_BUNDLE_VERIFICATION_SCHEMA_VERSION: "evidence-bundle-verification",
    ONLINE_TIMELINE_SCHEMA_VERSION: "online-timeline",
}


class ValidationReport(StrictModel):
    """Machine-readable validation result."""

    path: str
    kind: str
    schema_version: str
    valid: bool = True


def validation_kinds() -> tuple[str, ...]:
    """Return supported explicit validation kinds."""

    return tuple(_MODEL_BY_KIND)


def validation_kind_choices() -> tuple[str, ...]:
    """Return all CLI validation kind choices, including auto-detection."""

    return ("auto", *validation_kinds())


def validate_file(path: str | Path, kind: ValidationKind = "auto") -> ValidationReport:
    """Validate a schema-versioned slac artifact."""

    artifact_path = Path(path)
    payload = read_mapping(artifact_path)
    detected_kind = _detect_kind(payload) if kind == "auto" else kind
    model = _model_for_kind(detected_kind)
    validated = model.model_validate(payload)
    schema_version = _schema_version(validated)
    return ValidationReport(
        path=str(artifact_path),
        kind=detected_kind,
        schema_version=schema_version,
    )


def _detect_kind(payload: dict[str, object]) -> str:
    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, str):
        msg = "cannot auto-detect artifact kind without a string schema_version"
        raise SlacError(msg)
    try:
        return _KIND_BY_SCHEMA_VERSION[schema_version]
    except KeyError as exc:
        supported = ", ".join(sorted(_KIND_BY_SCHEMA_VERSION))
        msg = f"unsupported schema_version {schema_version!r}; expected one of {supported}"
        raise SlacError(msg) from exc


def _model_for_kind(kind: str) -> type[BaseModel]:
    try:
        return _MODEL_BY_KIND[kind]
    except KeyError as exc:
        supported = ", ".join(validation_kind_choices())
        msg = f"unsupported validation kind {kind!r}; expected one of {supported}"
        raise SlacError(msg) from exc


def _schema_version(model: BaseModel) -> str:
    value = getattr(model, "schema_version", None)
    if not isinstance(value, str):
        msg = "validated artifact did not expose a string schema_version"
        raise SlacError(msg)
    return value
