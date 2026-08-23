"""Train-only diagnostics for continuous-time LiDAR-pair refinement."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median
from typing import Any, Literal

from pydantic import Field, field_validator

from calibrex.core.result import StrictModel

CONTINUOUS_TIME_LIDAR_TRAIN_DIAGNOSTICS_SCHEMA_VERSION: Literal[
    "slac.continuous_time_lidar_train_diagnostics/v0.1"
] = "slac.continuous_time_lidar_train_diagnostics/v0.1"

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_RANGE_BIN_EDGES_M = (0.0, 5.0, 10.0, 20.0, 40.0, 80.0)


class ContinuousTimeLidarTrainResidualSummary(StrictModel):
    """Distribution summary of residuals from the final train factor."""

    count: int = Field(ge=0)
    rmse_m: float | None = Field(default=None, ge=0.0)
    mean_m: float | None = None
    mean_abs_m: float | None = Field(default=None, ge=0.0)
    median_abs_m: float | None = Field(default=None, ge=0.0)
    p95_abs_m: float | None = Field(default=None, ge=0.0)
    p99_abs_m: float | None = Field(default=None, ge=0.0)
    min_m: float | None = None
    max_m: float | None = None


class ContinuousTimeLidarTrainRangeBin(StrictModel):
    """Residual summary for one deterministic LiDAR-range interval."""

    lower_bound_m: float = Field(ge=0.0)
    upper_bound_m: float | None = Field(default=None, gt=0.0)
    correspondence_count: int = Field(ge=0)
    residuals: ContinuousTimeLidarTrainResidualSummary


class ContinuousTimeLidarTrainProfilePoint(StrictModel):
    """One clock-profile step with train-only candidate evidence."""

    iteration: int = Field(ge=1)
    candidate_offsets_sec: list[float] = Field(min_length=1)
    candidate_train_rmse_m: list[float | None] = Field(min_length=1)
    selected_offset_sec: float
    train_rmse_m: float | None = Field(default=None, ge=0.0)
    train_correspondence_count: int = Field(ge=0)
    train_outlier_rejected_count: int = Field(ge=0)
    accepted: bool


class ContinuousTimeLidarTrainDiagnosticsProvenance(StrictModel):
    """Input digests and generation metadata for train diagnostics."""

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


class ContinuousTimeLidarTrainDiagnostics(StrictModel):
    """Detailed evidence that is safe to use for train-only solver decisions."""

    schema_version: Literal[
        "slac.continuous_time_lidar_train_diagnostics/v0.1"
    ] = CONTINUOUS_TIME_LIDAR_TRAIN_DIAGNOSTICS_SCHEMA_VERSION
    selection_metric: Literal["train_rmse_m"] = "train_rmse_m"
    holdout_used_for_selection: Literal[False] = False
    sampled_source_record_count: int = Field(ge=0)
    train_target_point_count: int = Field(ge=0)
    train_candidate_correspondence_count: int = Field(ge=0)
    train_correspondence_count: int = Field(ge=0)
    train_outlier_rejected_count: int = Field(ge=0)
    train_outlier_rejection_rate: float = Field(ge=0.0, le=1.0)
    selected_time_offset_sec: float
    final_residuals: ContinuousTimeLidarTrainResidualSummary
    range_bins: list[ContinuousTimeLidarTrainRangeBin] = Field(min_length=1)
    profile: list[ContinuousTimeLidarTrainProfilePoint] = Field(default_factory=list)
    provenance: ContinuousTimeLidarTrainDiagnosticsProvenance


@dataclass(frozen=True)
class TrainResidualSummaryData:
    """Internal, provenance-free residual summary produced by the solver."""

    count: int
    rmse_m: float | None
    mean_m: float | None
    mean_abs_m: float | None
    median_abs_m: float | None
    p95_abs_m: float | None
    p99_abs_m: float | None
    min_m: float | None
    max_m: float | None


@dataclass(frozen=True)
class TrainRangeBinData:
    """Internal, provenance-free range-bin summary produced by the solver."""

    lower_bound_m: float
    upper_bound_m: float | None
    correspondence_count: int
    residuals: TrainResidualSummaryData


@dataclass(frozen=True)
class TrainProfilePointData:
    """Internal train-only clock-profile trace point."""

    iteration: int
    candidate_offsets_sec: tuple[float, ...]
    candidate_train_rmse_m: tuple[float | None, ...]
    selected_offset_sec: float
    train_rmse_m: float | None
    train_correspondence_count: int
    train_outlier_rejected_count: int
    accepted: bool


@dataclass(frozen=True)
class ContinuousTimeLidarTrainDiagnosticsData:
    """Internal train-only diagnostics payload before provenance is attached."""

    sampled_source_record_count: int
    train_target_point_count: int
    train_candidate_correspondence_count: int
    train_correspondence_count: int
    train_outlier_rejected_count: int
    selected_time_offset_sec: float
    final_residuals: TrainResidualSummaryData
    range_bins: tuple[TrainRangeBinData, ...]
    profile: tuple[TrainProfilePointData, ...]


def summarize_train_residuals(residuals: Sequence[float]) -> TrainResidualSummaryData:
    """Summarize signed residuals with deterministic absolute percentiles."""

    values = [float(value) for value in residuals]
    if not values:
        return TrainResidualSummaryData(
            count=0,
            rmse_m=None,
            mean_m=None,
            mean_abs_m=None,
            median_abs_m=None,
            p95_abs_m=None,
            p99_abs_m=None,
            min_m=None,
            max_m=None,
        )
    absolute = sorted(abs(value) for value in values)
    return TrainResidualSummaryData(
        count=len(values),
        rmse_m=math.sqrt(sum(value * value for value in values) / len(values)),
        mean_m=sum(values) / len(values),
        mean_abs_m=sum(absolute) / len(absolute),
        median_abs_m=float(median(absolute)),
        p95_abs_m=_nearest_rank(absolute, 0.95),
        p99_abs_m=_nearest_rank(absolute, 0.99),
        min_m=min(values),
        max_m=max(values),
    )


def build_train_range_bins(
    ranges_m: Sequence[float],
    residuals: Sequence[float],
) -> tuple[TrainRangeBinData, ...]:
    """Build fixed, boundary-stable residual summaries by point range."""

    if len(ranges_m) != len(residuals):
        raise ValueError("ranges and residuals must have equal lengths")
    buckets: list[list[float]] = [[] for _ in range(len(_RANGE_BIN_EDGES_M))]
    for range_m, residual in zip(ranges_m, residuals, strict=True):
        value = float(range_m)
        if value < 0.0:
            raise ValueError("point ranges must be non-negative")
        bucket_index = len(_RANGE_BIN_EDGES_M) - 1
        for index, edge_upper_bound in enumerate(_RANGE_BIN_EDGES_M[1:]):
            if value < edge_upper_bound:
                bucket_index = index
                break
        buckets[bucket_index].append(float(residual))
    bins: list[TrainRangeBinData] = []
    for index, values in enumerate(buckets):
        lower_bound = _RANGE_BIN_EDGES_M[index]
        upper_bound: float | None = (
            _RANGE_BIN_EDGES_M[index + 1]
            if index + 1 < len(_RANGE_BIN_EDGES_M)
            else None
        )
        bins.append(
            TrainRangeBinData(
                lower_bound_m=lower_bound,
                upper_bound_m=upper_bound,
                correspondence_count=len(values),
                residuals=summarize_train_residuals(values),
            )
        )
    return tuple(bins)


def train_diagnostics_artifact_from_data(
    data: ContinuousTimeLidarTrainDiagnosticsData,
    *,
    provenance: ContinuousTimeLidarTrainDiagnosticsProvenance,
) -> ContinuousTimeLidarTrainDiagnostics:
    """Attach provenance and convert solver diagnostics to the public artifact."""

    total_before_filter = data.train_candidate_correspondence_count
    rejection_rate = (
        data.train_outlier_rejected_count / total_before_filter
        if total_before_filter
        else 0.0
    )
    return ContinuousTimeLidarTrainDiagnostics(
        sampled_source_record_count=data.sampled_source_record_count,
        train_target_point_count=data.train_target_point_count,
        train_candidate_correspondence_count=data.train_candidate_correspondence_count,
        train_correspondence_count=data.train_correspondence_count,
        train_outlier_rejected_count=data.train_outlier_rejected_count,
        train_outlier_rejection_rate=rejection_rate,
        selected_time_offset_sec=data.selected_time_offset_sec,
        final_residuals=_summary_model(data.final_residuals),
        range_bins=[
            ContinuousTimeLidarTrainRangeBin(
                lower_bound_m=range_bin.lower_bound_m,
                upper_bound_m=range_bin.upper_bound_m,
                correspondence_count=range_bin.correspondence_count,
                residuals=_summary_model(range_bin.residuals),
            )
            for range_bin in data.range_bins
        ],
        profile=[
            ContinuousTimeLidarTrainProfilePoint(
                iteration=point.iteration,
                candidate_offsets_sec=list(point.candidate_offsets_sec),
                candidate_train_rmse_m=list(point.candidate_train_rmse_m),
                selected_offset_sec=point.selected_offset_sec,
                train_rmse_m=point.train_rmse_m,
                train_correspondence_count=point.train_correspondence_count,
                train_outlier_rejected_count=point.train_outlier_rejected_count,
                accepted=point.accepted,
            )
            for point in data.profile
        ],
        provenance=provenance,
    )


def continuous_time_lidar_train_diagnostics_json_schema() -> dict[str, Any]:
    """Return the standalone JSON schema for train-only diagnostics."""

    return ContinuousTimeLidarTrainDiagnostics.model_json_schema()


def _summary_model(data: TrainResidualSummaryData) -> ContinuousTimeLidarTrainResidualSummary:
    return ContinuousTimeLidarTrainResidualSummary(
        count=data.count,
        rmse_m=data.rmse_m,
        mean_m=data.mean_m,
        mean_abs_m=data.mean_abs_m,
        median_abs_m=data.median_abs_m,
        p95_abs_m=data.p95_abs_m,
        p99_abs_m=data.p99_abs_m,
        min_m=data.min_m,
        max_m=data.max_m,
    )


def _nearest_rank(sorted_values: Sequence[float], quantile: float) -> float:
    rank = max(1, math.ceil(float(quantile) * len(sorted_values)))
    return float(sorted_values[rank - 1])
