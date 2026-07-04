"""Machine-readable timeline artifact for online/streaming calibration sessions.

An online calibration session (see ``slac.pipelines.online``) replays LiDAR
frames in batches and, per batch, updates the extrinsic estimate, evaluates a
holdout gate, and emits a snapshot. This module defines the schema-validated
artifact that records the full per-batch timeline, mirroring how
``evidence.json``/``assessment.json`` are defined in ``core/evidence*.py``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from slac.core.report_artifacts import ReportRunInfo
from slac.core.result import (
    MetricResult,
    ObservabilityResult,
    StrictModel,
    TransformResult,
)

ONLINE_TIMELINE_SCHEMA_VERSION: Literal["slac.online_timeline/v0.3"] = (
    "slac.online_timeline/v0.3"
)

# Mirrors the pass/fail/inconclusive semantics of `AssessmentStatus` in
# `core/assessment.py`, with an online-specific split between evidence failure
# and observability-only inconclusive batches when multi-batch accumulation is
# enabled (see ``slac.pipelines.online.OnlineCalibrationSession``):
#
# - ``pass``: accumulated observability and holdout evidence support adopting
#   this batch's tentative estimate; the rolling residual window is updated.
# - ``fail``: known-bad residual evidence (absolute holdout RMSE or rolling
#   regression spike), or rank deficiency when ``accumulation_batches == 1``;
#   the batch is fully excluded from the retention buffer.
# - ``inconclusive``: either too little train/holdout evidence to score, or —
#   when accumulating — observability alone is insufficient while holdout
#   evidence on the observable directions is acceptable; the running estimate
#   and rolling window stay unchanged, but observability-only inconclusive
#   batches may still enter the retention buffer
#   (``retained_for_accumulation``).
#
# ``estimate_accepted`` is true exactly when the gate status is ``pass``.
OnlineGateStatus = Literal["pass", "fail", "inconclusive"]


class OnlineBatchSnapshot(StrictModel):
    """Per-batch evidence snapshot for one online calibration session."""

    batch_index: int = Field(ge=0)
    frame_count: int = Field(ge=0)
    point_count: int = Field(ge=0)
    train_point_count: int = Field(ge=0)
    holdout_point_count: int = Field(ge=0)
    correspondence_count: int = Field(ge=0)
    holdout_correspondence_count: int = Field(ge=0)
    estimate: TransformResult
    estimate_accepted: bool
    batch_holdout_rmse_m: float | None = None
    rolling_rmse_m: float | None = None
    rolling_window_residual_count: int = Field(default=0, ge=0)
    observability: ObservabilityResult = Field(default_factory=ObservabilityResult)
    batch_observability: ObservabilityResult | None = None
    gate_status: OnlineGateStatus
    gate_reason: str
    retained_for_accumulation: bool = False
    metrics: dict[str, MetricResult] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)


class OnlineCalibrationTimelineArtifact(StrictModel):
    """Machine-readable per-batch timeline for one online calibration run."""

    schema_version: Literal["slac.online_timeline/v0.3"] = (
        ONLINE_TIMELINE_SCHEMA_VERSION
    )
    run: ReportRunInfo
    variable: str
    source_sensor: str
    target_sensor: str
    batch_size: int = Field(ge=1)
    rolling_window: int = Field(ge=1)
    holdout_ratio: float = Field(ge=0.0, le=0.9)
    seed: int
    accumulation_batches: int | None = Field(default=None, ge=1)
    max_accumulated_train_points: int | None = Field(default=None, ge=1)
    batches: list[OnlineBatchSnapshot] = Field(default_factory=list)
    final_gate_status: OnlineGateStatus
    accepted_batch_count: int = Field(default=0, ge=0)
    rejected_batch_count: int = Field(default=0, ge=0)
    inconclusive_batch_count: int = Field(default=0, ge=0)
    motion_compensated: bool | None = None


def online_timeline_json_schema() -> dict[str, Any]:
    """Return the JSON schema for online calibration timeline artifacts."""

    return OnlineCalibrationTimelineArtifact.model_json_schema()
