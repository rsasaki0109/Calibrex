"""Machine-readable trajectory artifact for motion-compensated online calibration.

When ``dataset.odometry_topic`` is configured on a rosbag2 replay, the online
pipeline interpolates ``T_world_base(t)`` poses to motion-compensate LiDAR
frames. This module defines the schema-validated artifact that records the
odometry trajectory, interpolation health, and ground-truth-free quality
metrics emitted alongside ``timeline.json``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from slac.core.report_artifacts import ReportRunInfo
from slac.core.result import Grade, MetricResult, StrictModel

TRAJECTORY_SCHEMA_VERSION: Literal["slac.trajectory/v0.1"] = "slac.trajectory/v0.1"

TrajectoryGateStatus = Literal["pass", "fail", "inconclusive"]


class TrajectoryOdometrySource(StrictModel):
    """Provenance for the odometry topic that supplied the pose track."""

    topic: str
    message_count: int = Field(ge=0)
    first_timestamp_ns: int = Field(ge=0)
    last_timestamp_ns: int = Field(ge=0)
    time_span_s: float = Field(ge=0.0)


class TrajectoryFrameSemantics(StrictModel):
    """Frame conventions for recorded ``T_world_base`` poses."""

    world_frame_id: str
    child_frame_id: str
    note: str = (
        "poses are T_world_base following the online motion-compensation "
        "formulation (world frame from odometry header, child/base from "
        "child_frame_id)"
    )


class TrajectorySubsampling(StrictModel):
    """How the recorded pose list was reduced from the full odometry track."""

    max_pose_samples: int = Field(ge=1)
    method: Literal["even_time_spacing"] = "even_time_spacing"


class TrajectoryPoseSample(StrictModel):
    """One subsampled ``T_world_base`` pose."""

    timestamp_ns: int = Field(ge=0)
    translation: list[float] = Field(min_length=3, max_length=3)
    rotation_quat_xyzw: list[float] = Field(min_length=4, max_length=4)


class TrajectoryInterpolationHealth(StrictModel):
    """Interpolation statistics accumulated on the replayed odometry track."""

    interpolation_count: int = Field(ge=0)
    clamp_count: int = Field(ge=0)
    max_extrapolation_s: float = Field(ge=0.0)


class TrajectoryQualityMetric(StrictModel):
    """One graded trajectory-quality metric with its recorded gate threshold."""

    value: float | None = None
    unit: str | None = None
    grade: Grade = "warn"
    reason: str | None = None
    threshold: float | None = None


class TrajectoryQualityBlock(StrictModel):
    """Ground-truth-free trajectory quality metrics and gate statuses."""

    kinematic_linear_speed_p95_mps: TrajectoryQualityMetric
    kinematic_linear_speed_max_mps: TrajectoryQualityMetric
    kinematic_angular_speed_p95_dps: TrajectoryQualityMetric
    kinematic_angular_speed_max_dps: TrajectoryQualityMetric
    kinematic_gate_status: TrajectoryGateStatus
    kinematic_gate_reason: str
    interpolation_clamp_fraction: TrajectoryQualityMetric
    interpolation_max_extrapolation_s: TrajectoryQualityMetric
    interpolation_gate_status: TrajectoryGateStatus
    interpolation_gate_reason: str
    cross_segment_rmse_m: TrajectoryQualityMetric
    cross_segment_first_half_scan_count: int = Field(ge=0)
    cross_segment_second_half_scan_count: int = Field(ge=0)
    cross_segment_first_half_point_count: int = Field(ge=0)
    cross_segment_second_half_point_count: int = Field(ge=0)
    cross_segment_first_half_start_timestamp_ns: int | None = Field(default=None, ge=0)
    cross_segment_first_half_end_timestamp_ns: int | None = Field(default=None, ge=0)
    cross_segment_second_half_start_timestamp_ns: int | None = Field(default=None, ge=0)
    cross_segment_second_half_end_timestamp_ns: int | None = Field(default=None, ge=0)
    cross_segment_sampling_policy: str | None = None
    cross_segment_correspondence_count: int = Field(ge=0)
    cross_segment_gate_status: TrajectoryGateStatus
    cross_segment_gate_reason: str
    verdict: TrajectoryGateStatus
    verdict_reason: str


class TrajectoryArtifact(StrictModel):
    """Machine-readable odometry trajectory for one motion-compensated online run."""

    schema_version: Literal["slac.trajectory/v0.1"] = TRAJECTORY_SCHEMA_VERSION
    run: ReportRunInfo
    odometry_source: TrajectoryOdometrySource
    frame_semantics: TrajectoryFrameSemantics
    pose_count_total: int = Field(ge=0)
    pose_count_recorded: int = Field(ge=0)
    subsampling: TrajectorySubsampling
    poses: list[TrajectoryPoseSample] = Field(default_factory=list)
    interpolation_health: TrajectoryInterpolationHealth
    quality: TrajectoryQualityBlock


def trajectory_quality_metric_from_result(
    metric: MetricResult,
    *,
    threshold: float | None = None,
) -> TrajectoryQualityMetric:
    """Convert a :class:`MetricResult` into a trajectory-artifact quality row."""

    return TrajectoryQualityMetric(
        value=metric.value,
        unit=metric.unit,
        grade=metric.grade,
        reason=metric.reason,
        threshold=threshold,
    )


def trajectory_json_schema() -> dict[str, Any]:
    """Return the JSON schema for trajectory artifacts."""

    return TrajectoryArtifact.model_json_schema()
