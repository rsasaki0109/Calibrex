"""Schema-versioned, ground-truth-free drift evidence by capture window."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import Field, field_validator

from calibrex.core.result import Grade, StrictModel
from calibrex.core.trajectory import TrajectoryGateStatus

TRAJECTORY_WINDOW_DRIFT_SCHEMA_VERSION: Literal[
    "slac.trajectory_window_drift/v0.1"
] = "slac.trajectory_window_drift/v0.1"

TrajectoryWindowDriftInterpretation = Literal[
    "local_inconsistency_suspected",
    "long_span_inconsistency_suspected",
    "no_inconsistency_detected",
    "inconclusive",
]

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class TrajectoryWindowDriftThresholds(StrictModel):
    """Thresholds used by the per-window cross-segment gate."""

    max_cross_segment_rmse_m: float = Field(ge=0.0)
    min_correspondences: int = Field(ge=1)


class TrajectoryWindowDriftParameters(StrictModel):
    """Replay and correspondence settings needed to reproduce the diagnostic."""

    voxel_size_m: float = Field(gt=0.0)
    correspondence_gate_m: float = Field(gt=0.0)
    max_scans_per_half: int = Field(ge=1)
    max_points_per_half: int = Field(ge=1)
    source_point_time_field: str | None = None
    use_point_time_offsets: bool = True
    odometry_burst_policy: Literal["preserve", "keep_first", "keep_last"] = "preserve"
    odometry_burst_min_interval_s: float = Field(gt=0.0)
    capture_window_record_prefilter_margin_s: float = Field(ge=0.0)
    capture_window_sampling_policy: str


class TrajectoryWindowOdometryStats(StrictModel):
    """Odometry coverage and motion summary for one declared capture window."""

    pose_count: int = Field(ge=0)
    first_timestamp_ns: int | None = Field(default=None, ge=0)
    last_timestamp_ns: int | None = Field(default=None, ge=0)
    time_span_s: float = Field(ge=0.0)
    path_length_m: float = Field(ge=0.0)
    endpoint_displacement_m: float = Field(ge=0.0)
    endpoint_rotation_deg: float = Field(ge=0.0)


class TrajectoryWindowCrossSegmentEvidence(StrictModel):
    """Cross-segment map-consistency evidence for one span."""

    rmse_m: float | None = Field(default=None, ge=0.0)
    first_half_scan_count: int = Field(ge=0)
    second_half_scan_count: int = Field(ge=0)
    first_half_point_count: int = Field(ge=0)
    second_half_point_count: int = Field(ge=0)
    first_half_start_timestamp_ns: int | None = Field(default=None, ge=0)
    first_half_end_timestamp_ns: int | None = Field(default=None, ge=0)
    second_half_start_timestamp_ns: int | None = Field(default=None, ge=0)
    second_half_end_timestamp_ns: int | None = Field(default=None, ge=0)
    sampling_policy: str | None = None
    correspondence_count: int = Field(ge=0)
    gate_status: TrajectoryGateStatus
    gate_reason: str


class TrajectoryWindowDriftWindow(StrictModel):
    """Evidence collected for one configured capture window."""

    window_index: int = Field(ge=0)
    start_timestamp_ns: int | None = Field(default=None, ge=0)
    end_timestamp_ns: int | None = Field(default=None, ge=0)
    odometry: TrajectoryWindowOdometryStats
    cross_segment: TrajectoryWindowCrossSegmentEvidence


class TrajectoryWindowDriftProvenance(StrictModel):
    """Input digests and generation details for a drift artifact."""

    source_paths: list[str] = Field(min_length=2)
    source_sha256: dict[str, str] = Field(min_length=2)
    tool_name: str = "calibrex"
    tool_version: str | None = None
    git_commit: str | None = None
    notes: list[str] = Field(default_factory=list)

    @field_validator("source_sha256")
    @classmethod
    def validate_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        invalid = [key for key, digest in value.items() if not _SHA256_RE.fullmatch(digest)]
        if invalid:
            raise ValueError(f"source_sha256 contains invalid SHA-256 values: {invalid}")
        return value


class TrajectoryWindowDriftArtifact(StrictModel):
    """Machine-readable diagnostic separating local and long-span drift signals."""

    schema_version: Literal[
        "slac.trajectory_window_drift/v0.1"
    ] = TRAJECTORY_WINDOW_DRIFT_SCHEMA_VERSION
    config_path: str
    dataset_path: str
    source_sensor: str
    target_sensor: str
    odometry_topic: str
    parameters: TrajectoryWindowDriftParameters
    thresholds: TrajectoryWindowDriftThresholds
    windows: list[TrajectoryWindowDriftWindow] = Field(min_length=1)
    full_span_cross_segment: TrajectoryWindowCrossSegmentEvidence
    interpretation: TrajectoryWindowDriftInterpretation
    grade: Grade
    interpretation_reason: str
    provenance: TrajectoryWindowDriftProvenance


def interpret_trajectory_window_drift(
    window_statuses: Sequence[TrajectoryGateStatus],
    full_span_status: TrajectoryGateStatus,
) -> tuple[TrajectoryWindowDriftInterpretation, Grade, str]:
    """Classify local versus long-span inconsistency without ground truth.

    A local failure means at least one declared window is internally
    inconsistent. A full-span-only failure is weaker evidence of accumulated
    map/odometry drift over the longer interval. The labels are hypotheses,
    not absolute attribution of the error source.
    """

    local_failures = [
        str(index) for index, status in enumerate(window_statuses) if status == "fail"
    ]
    if local_failures:
        return (
            "local_inconsistency_suspected",
            "fail",
            "window-local cross-segment gate failed in windows "
            + ", ".join(local_failures)
            + "; the evidence is not explained by long-span accumulation alone",
        )
    if full_span_status == "fail":
        return (
            "long_span_inconsistency_suspected",
            "fail",
            "full-span cross-segment gate failed while every window-local gate passed; "
            "long-span map or odometry drift is suspected",
        )
    if full_span_status == "inconclusive" or any(
        status == "inconclusive" for status in window_statuses
    ):
        return (
            "inconclusive",
            "warn",
            "one or more cross-segment gates were inconclusive because the available "
            "scan or correspondence evidence was insufficient",
        )
    return (
        "no_inconsistency_detected",
        "pass",
        "full-span and window-local cross-segment gates passed",
    )


def trajectory_window_drift_json_schema() -> dict[str, Any]:
    """Return the standalone JSON schema for trajectory-window drift."""

    return TrajectoryWindowDriftArtifact.model_json_schema()
