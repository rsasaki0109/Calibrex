"""Timestamp alignment metric extraction helpers."""

from __future__ import annotations

from collections.abc import Mapping

from calibrex.core.result import MetricResult
from calibrex.data.inspect import DatasetInspection


def timing_metrics_from_inspection(inspection: DatasetInspection) -> dict[str, MetricResult]:
    """Build timestamp alignment metrics from dataset inspection diagnostics."""

    alignment = inspection.diagnostics.get("timestamp_alignment")
    if not isinstance(alignment, Mapping):
        return {}

    metrics: dict[str, MetricResult] = {}
    _add_metric(
        metrics,
        "camera_lidar_timestamp_alignment_ms",
        alignment,
        "camera_lidar_max_abs_dt_ms",
    )
    _add_metric(
        metrics,
        "lidar_oxts_timestamp_alignment_ms",
        alignment,
        "lidar_oxts_max_abs_dt_ms",
    )
    return metrics


def _add_metric(
    metrics: dict[str, MetricResult],
    metric_name: str,
    source: Mapping[object, object],
    source_key: str,
) -> None:
    value = _float_or_none(source.get(source_key))
    if value is None:
        return
    metrics[metric_name] = MetricResult(
        value=value,
        unit="ms",
        reason=f"{source_key} from KITTI timestamp alignment diagnostics",
    )


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None
