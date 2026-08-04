"""Schema-versioned evidence for calibration consistency across capture windows."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import Field, field_validator

from calibrex.core.result import Grade, StrictModel

DYNAMIC_WINDOW_CONSISTENCY_SCHEMA_VERSION: Literal[
    "slac.dynamic_window_consistency/v0.1"
] = "slac.dynamic_window_consistency/v0.1"

DynamicWindowProtocolStatus = Literal["compatible", "warning", "not_comparable"]

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class DynamicWindowConsistencyThresholds(StrictModel):
    """Declared transform-delta limits for accepting one window family."""

    max_translation_delta_m: float = Field(ge=0.0)
    max_rotation_delta_deg: float = Field(ge=0.0)


class DynamicWindowCaptureWindow(StrictModel):
    """One replay window and its selected observation counts, when available."""

    window_index: int | None = Field(default=None, ge=0)
    start_timestamp_ns: int | None = None
    end_timestamp_ns: int | None = None
    selected_source_message_count: int | None = Field(default=None, ge=0)
    selected_source_point_count: int | None = Field(default=None, ge=0)
    selected_target_message_count: int | None = Field(default=None, ge=0)
    selected_target_point_count: int | None = Field(default=None, ge=0)


class DynamicWindowConsistencyInput(StrictModel):
    """One schema-validated calibration result used by the consistency gate."""

    label: str
    result_path: str
    result_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    run_id: str
    quality_grade: Grade
    transform_present: bool
    capture_window_start_timestamp_ns: int | None = None
    capture_window_end_timestamp_ns: int | None = None
    capture_windows: list[DynamicWindowCaptureWindow] = Field(default_factory=list)


class DynamicWindowConsistencyPair(StrictModel):
    """Pairwise transform delta and gate verdict."""

    left_label: str
    right_label: str
    translation_delta_m: float | None = Field(default=None, ge=0.0)
    rotation_delta_deg: float | None = Field(default=None, ge=0.0)
    grade: Grade
    reason: str


class DynamicWindowConsistencyProvenance(StrictModel):
    """Input digests and generation details for a consistency artifact."""

    source_paths: list[str] = Field(min_length=2)
    source_sha256: dict[str, str] = Field(min_length=2)
    tool_name: str = "calibrex"
    tool_version: str | None = None
    notes: list[str] = Field(default_factory=list)

    @field_validator("source_sha256")
    @classmethod
    def validate_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        invalid = [key for key, digest in value.items() if not _SHA256_RE.fullmatch(digest)]
        if invalid:
            raise ValueError(f"source_sha256 contains invalid SHA-256 values: {invalid}")
        return value


class DynamicWindowConsistencyArtifact(StrictModel):
    """Machine-readable gate for reusing one transform across capture windows."""

    schema_version: Literal[
        "slac.dynamic_window_consistency/v0.1"
    ] = DYNAMIC_WINDOW_CONSISTENCY_SCHEMA_VERSION
    transform_id: str
    comparison_mode: Literal["all_pairs"] = "all_pairs"
    reference_label: str | None = None
    thresholds: DynamicWindowConsistencyThresholds
    inputs: list[DynamicWindowConsistencyInput] = Field(min_length=2)
    pairs: list[DynamicWindowConsistencyPair] = Field(min_length=1)
    protocol_compatibility_status: DynamicWindowProtocolStatus
    grade: Grade
    blocking_failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    provenance: DynamicWindowConsistencyProvenance


def dynamic_window_consistency_json_schema() -> dict[str, Any]:
    """Return the standalone JSON schema for dynamic-window consistency."""

    return DynamicWindowConsistencyArtifact.model_json_schema()
