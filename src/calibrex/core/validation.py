"""Validation helpers for schema-versioned Calibrex artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel

from calibrex.core.config import CONFIG_SCHEMA_VERSION, CalibrationConfig
from calibrex.core.exceptions import CalibrexError
from calibrex.core.io import read_mapping
from calibrex.core.report_artifacts import (
    REPORT_DEGENERACY_SCHEMA_VERSION,
    REPORT_METRICS_SCHEMA_VERSION,
    REPORT_OBSERVABILITY_SCHEMA_VERSION,
    REPORT_SUMMARY_SCHEMA_VERSION,
    ReportDegeneracyArtifact,
    ReportMetricsArtifact,
    ReportObservabilityArtifact,
    ReportSummaryArtifact,
)
from calibrex.core.result import RESULT_SCHEMA_VERSION, CalibrationResult, StrictModel
from calibrex.data.manifest import DATASET_MANIFEST_SCHEMA_VERSION, DatasetManifest

ValidationKind = Literal[
    "auto",
    "config",
    "result",
    "dataset-manifest",
    "report-summary",
    "report-metrics",
    "report-observability",
    "report-degeneracy",
]

_MODEL_BY_KIND: Final[dict[str, type[BaseModel]]] = {
    "config": CalibrationConfig,
    "result": CalibrationResult,
    "dataset-manifest": DatasetManifest,
    "report-summary": ReportSummaryArtifact,
    "report-metrics": ReportMetricsArtifact,
    "report-observability": ReportObservabilityArtifact,
    "report-degeneracy": ReportDegeneracyArtifact,
}

_KIND_BY_SCHEMA_VERSION: Final[dict[str, str]] = {
    CONFIG_SCHEMA_VERSION: "config",
    RESULT_SCHEMA_VERSION: "result",
    DATASET_MANIFEST_SCHEMA_VERSION: "dataset-manifest",
    REPORT_SUMMARY_SCHEMA_VERSION: "report-summary",
    REPORT_METRICS_SCHEMA_VERSION: "report-metrics",
    REPORT_OBSERVABILITY_SCHEMA_VERSION: "report-observability",
    REPORT_DEGENERACY_SCHEMA_VERSION: "report-degeneracy",
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
    """Validate a schema-versioned Calibrex artifact."""

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
        raise CalibrexError(msg)
    try:
        return _KIND_BY_SCHEMA_VERSION[schema_version]
    except KeyError as exc:
        supported = ", ".join(sorted(_KIND_BY_SCHEMA_VERSION))
        msg = f"unsupported schema_version {schema_version!r}; expected one of {supported}"
        raise CalibrexError(msg) from exc


def _model_for_kind(kind: str) -> type[BaseModel]:
    try:
        return _MODEL_BY_KIND[kind]
    except KeyError as exc:
        supported = ", ".join(validation_kind_choices())
        msg = f"unsupported validation kind {kind!r}; expected one of {supported}"
        raise CalibrexError(msg) from exc


def _schema_version(model: BaseModel) -> str:
    value = getattr(model, "schema_version", None)
    if not isinstance(value, str):
        msg = "validated artifact did not expose a string schema_version"
        raise CalibrexError(msg)
    return value
