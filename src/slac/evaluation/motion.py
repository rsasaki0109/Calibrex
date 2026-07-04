"""Vehicle-motion metric extraction helpers."""

from __future__ import annotations

from collections.abc import Mapping

from slac.core.result import MetricResult
from slac.data.inspect import DatasetInspection


def motion_metrics_from_inspection(inspection: DatasetInspection) -> dict[str, MetricResult]:
    """Build motion excitation metrics from dataset inspection diagnostics."""

    oxts = inspection.diagnostics.get("oxts_motion")
    if not isinstance(oxts, Mapping):
        return {}

    metrics: dict[str, MetricResult] = {}
    _add_metric(metrics, "vehicle_motion_duration_s", oxts, "duration_sec", "s")
    _add_metric(metrics, "vehicle_mean_speed_mps", oxts, "mean_speed_mps", "m/s")
    _add_metric(metrics, "vehicle_speed_range_mps", oxts, "speed_range_mps", "m/s")
    _add_metric(metrics, "vehicle_yaw_excitation_deg", oxts, "yaw_excitation_deg", "deg")
    _add_metric(metrics, "vehicle_pitch_excitation_deg", oxts, "pitch_excitation_deg", "deg")
    _add_metric(metrics, "vehicle_roll_excitation_deg", oxts, "roll_excitation_deg", "deg")
    _add_metric(
        metrics,
        "vehicle_mean_acceleration_norm_mps2",
        oxts,
        "mean_acceleration_norm_mps2",
        "m/s^2",
    )
    return metrics


def _add_metric(
    metrics: dict[str, MetricResult],
    metric_name: str,
    source: Mapping[object, object],
    source_key: str,
    unit: str,
) -> None:
    value = _float_or_none(source.get(source_key))
    if value is None:
        return
    metrics[metric_name] = MetricResult(
        value=value,
        unit=unit,
        reason=f"{source_key} from KITTI OXTS motion diagnostics",
    )


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None
