"""Backward-compatible re-exports for doctor diagnostics."""

from __future__ import annotations

from pathlib import Path

from calibrex.core.config import DatasetType
from calibrex.core.environment_readiness import (
    ENVIRONMENT_READINESS_SCHEMA_VERSION,
    EnvironmentReadinessArtifact,
    ReadinessStatus,
    WorkflowStatus,
    WorkflowSuggestion,
    build_environment_readiness_artifact,
    environment_readiness_json_schema,
    infer_dataset_type,
)

DOCTOR_SCHEMA_VERSION = ENVIRONMENT_READINESS_SCHEMA_VERSION
DoctorStatus = ReadinessStatus
DoctorArtifact = EnvironmentReadinessArtifact
DoctorWorkflow = WorkflowSuggestion


def doctor_json_schema() -> dict[str, object]:
    """Return the JSON schema for doctor artifacts."""

    return environment_readiness_json_schema()


def build_doctor_artifact(
    *,
    calibrex_version: str,
    command: list[str],
    path: Path | None = None,
    dataset_type: DatasetType | None = None,
    sample_limit: int | None = None,
) -> EnvironmentReadinessArtifact:
    """Diagnose the runtime and, optionally, one dataset path."""

    return build_environment_readiness_artifact(
        calibrex_version=calibrex_version,
        command=command,
        path=path,
        dataset_type=dataset_type,
        sample_limit=sample_limit,
    )
