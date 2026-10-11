"""Solid-state LiDAR acquisition and calibration evidence metrics."""

from __future__ import annotations

import math
import struct
from collections.abc import Iterable, Mapping
from itertools import pairwise

from calibrex.core.config import CalibrationConfig
from calibrex.core.result import CalibrationResult, Grade, MetricResult
from calibrex.core.solid_state import SolidStateEvaluationConfig
from calibrex.data.inspect import DatasetInspection
from calibrex.data.livox import (
    find_livox_pcd_files,
    read_livox_binary_pcd,
    summarize_livox_pcd,
)
from calibrex.data.rosbag1 import summarize_rosbag1
from calibrex.data.rosbag2 import summarize_rosbag2


def solid_state_metrics_from_inspection(
    config: CalibrationConfig,
    inspection: DatasetInspection | None = None,
    result: CalibrationResult | None = None,
) -> dict[str, MetricResult]:
    """Build explicit solid-state acquisition/evidence metrics.

    The metrics intentionally distinguish metadata availability from a claim
    that a calibration is accurate.  Missing point times or temperatures are
    warnings by default and become failures only when the config declares them
    mandatory.  Livox PCD geometry is scored for distance and angular coverage;
    the existing LiDAR pair evaluator supplies independent holdout and
    known-bad perturbation evidence when the data support it.
    """

    profiles = {
        sensor_name: sensor
        for sensor_name, sensor in config.sensors.items()
        if sensor.solid_state is not None
    }
    evaluation = config.evaluation.solid_state
    if not profiles or not evaluation.enabled:
        return {}

    profile_values = {
        sensor_name: sensor.solid_state
        for sensor_name, sensor in profiles.items()
        if sensor.solid_state is not None
    }
    profile_count = len(profile_values)
    metrics: dict[str, MetricResult] = {
        "solid_state_profile_count": MetricResult(
            value=float(profile_count),
            unit="sensors",
            grade="pass",
            reason="schema-validated solid-state LiDAR profiles are present",
        ),
        "solid_state_non_repetitive_profile_fraction": _fraction_metric(
            _count(
                profile.scan_pattern == "non_repetitive"
                for profile in profile_values.values()
            ),
            profile_count,
            label="non-repetitive scan-pattern declarations",
        ),
        "solid_state_point_time_profile_fraction": _fraction_metric(
            _count(
                (
                    _point_time_declared(profile)
                    if (observed := _inspection_point_time_available(
                        config, inspection, sensor_name
                    )) is None
                    else observed
                )
                or _capture_point_time_available(result, sensor_name)
                for sensor_name, profile in profile_values.items()
            ),
            profile_count,
            label="profiles with explicit per-point capture-time semantics",
            required=evaluation.require_point_time,
        ),
        "solid_state_integration_window_profile_fraction": _fraction_metric(
            _count(profile.integration_window_s is not None for profile in profile_values.values()),
            profile_count,
            label="profiles with a declared integration window",
        ),
        "solid_state_intrinsic_calibration_profile_fraction": _fraction_metric(
            _count(
                profile.intrinsic_calibration.source != "unknown"
                for profile in profile_values.values()
            ),
            profile_count,
            label="profiles with a declared internal-calibration source",
        ),
        "solid_state_temperature_profile_fraction": _fraction_metric(
            _count(
                _temperature_is_observed(profile.temperature)
                or _capture_temperature_is_observed(result, sensor_name)
                for sensor_name, profile in profile_values.items()
            ),
            profile_count,
            label="profiles with an observed temperature condition",
            required=evaluation.require_temperature,
        ),
    }
    temperature_values = _temperature_values(profile_values, result)
    temperature_span = (
        max(temperature_values) - min(temperature_values)
        if temperature_values
        else None
    )
    metrics["solid_state_temperature_span_c"] = _threshold_metric(
        temperature_span,
        evaluation.min_temperature_span_c,
        unit="degC",
        label="observed solid-state LiDAR temperature span",
    )

    livox_diagnostics = _livox_diagnostics(config, inspection)
    if livox_diagnostics is not None:
        if result is not None:
            result.run.provenance["solid_state_geometry_diagnostics"] = {
                "source": livox_diagnostics.get("source", "dataset_adapter_sample"),
                "dataset_type": config.dataset.type,
                "sample_limit": config.dataset.sample_limit or 4,
                "selection_policy": (
                    "first sample_limit decoded messages per supported topic"
                ),
                "diagnostics": dict(livox_diagnostics),
            }
        metrics.update(_livox_geometry_metrics(livox_diagnostics, evaluation))
        counts = _livox_distance_bin_counts(
            config, evaluation, diagnostics=livox_diagnostics
        )
        if counts is not None:
            supported = sum(
                count >= evaluation.min_points_per_distance_bin for count in counts
            )
            total_bins = len(counts)
            coverage = supported / total_bins if total_bins else None
            metrics["solid_state_distance_bin_coverage_fraction"] = MetricResult(
                value=coverage,
                grade="pass" if coverage == 1.0 else "warn",
                reason=(
                    "distance-bin support counts (edges declared in evaluation config): "
                    f"{counts}; minimum per bin={evaluation.min_points_per_distance_bin}"
                ),
            )
            metrics["solid_state_distance_bin_supported_count"] = MetricResult(
                value=float(supported),
                unit="bins",
                grade="pass" if supported else "warn",
                reason="distance bins with enough sampled points for stratified evaluation",
            )
    else:
        metrics.update(_unavailable_geometry_metrics("Livox geometry diagnostics are unavailable"))

    metrics.update(_holdout_metrics(config, result))
    return metrics


def _fraction_metric(
    count: int,
    total: int,
    *,
    label: str,
    required: bool = False,
) -> MetricResult:
    fraction = count / total if total else None
    grade: Grade
    if fraction is None:
        grade = "fail" if required else "warn"
    elif fraction >= 1.0:
        grade = "pass"
    else:
        grade = "fail" if required else "warn"
    return MetricResult(
        value=fraction,
        grade=grade,
        reason=f"{count}/{total} {label}; required={required}",
    )


def _livox_diagnostics(
    config: CalibrationConfig,
    inspection: DatasetInspection | None,
) -> Mapping[str, object] | None:
    if inspection is not None:
        diagnostics = inspection.diagnostics.get("livox_pcd")
        if isinstance(diagnostics, Mapping):
            return diagnostics
        diagnostics = inspection.diagnostics.get("rosbag1")
        if isinstance(diagnostics, Mapping):
            rosbag1 = _rosbag1_solid_state_diagnostics(config, diagnostics)
            if rosbag1 is not None:
                return rosbag1
        diagnostics = inspection.diagnostics.get("rosbag2")
        if isinstance(diagnostics, Mapping):
            rosbag2 = _rosbag2_solid_state_diagnostics(config, diagnostics)
            if rosbag2 is not None:
                return rosbag2
    if config.dataset.type != "livox_pcd":
        if config.dataset.type == "rosbag1":
            diagnostics = summarize_rosbag1(
                config.dataset.path,
                sample_limit=config.dataset.sample_limit or 4,
            ).as_dict()
            return _rosbag1_solid_state_diagnostics(config, diagnostics)
        if config.dataset.type == "rosbag2":
            diagnostics = summarize_rosbag2(
                config.dataset.path,
                sample_limit=config.dataset.sample_limit or 4,
            ).as_dict()
            return _rosbag2_solid_state_diagnostics(config, diagnostics)
        return None
    stats = summarize_livox_pcd(
        config.dataset.path,
        sample_limit=config.dataset.sample_limit or 4,
    )
    return stats.as_dict()


def _rosbag1_solid_state_diagnostics(
    config: CalibrationConfig,
    diagnostics: Mapping[str, object],
) -> Mapping[str, object] | None:
    """Aggregate sampled ROS1 diagnostics for configured solid-state topics."""

    solid_topics = {
        sensor.topic
        for sensor in config.sensors.values()
        if sensor.solid_state is not None and sensor.topic
    }
    streams = diagnostics.get("streams")
    if not isinstance(streams, Iterable):
        return None
    selected: list[Mapping[str, object]] = []
    for stream in streams:
        if not isinstance(stream, Mapping):
            continue
        topic = stream.get("topic")
        if topic in solid_topics:
            selected.append(stream)
    if not selected:
        return None

    range_min_values = [
        value
        for stream in selected
        if (value := _float_or_none(stream.get("range_min_m"))) is not None
    ]
    range_max_values = [
        value
        for stream in selected
        if (value := _float_or_none(stream.get("range_max_m"))) is not None
    ]
    azimuth_values = [
        value
        for stream in selected
        if (value := _float_or_none(stream.get("fov_azimuth_deg"))) is not None
    ]
    elevation_values = [
        value
        for stream in selected
        if (value := _float_or_none(stream.get("fov_elevation_deg"))) is not None
    ]
    point_time_streams = [
        stream for stream in selected if stream.get("point_time_available") is True
    ]
    point_time_min_values = [
        value
        for stream in point_time_streams
        if (value := _float_or_none(stream.get("point_time_min_s"))) is not None
    ]
    point_time_max_values = [
        value
        for stream in point_time_streams
        if (value := _float_or_none(stream.get("point_time_max_s"))) is not None
    ]
    output: dict[str, object] = {
        "source": "rosbag1_sampled_solid_state_topics",
        "stream_count": len(selected),
        "point_time_available_stream_count": len(point_time_streams),
        "point_time_available_fraction": (
            len(point_time_streams) / len(selected) if selected else None
        ),
        "range_min_m": min(range_min_values) if range_min_values else None,
        "range_max_m": max(range_max_values) if range_max_values else None,
        "fov_azimuth_deg": max(azimuth_values) if azimuth_values else None,
        "fov_elevation_deg": max(elevation_values) if elevation_values else None,
        "point_time_min_s": min(point_time_min_values)
        if point_time_min_values
        else None,
        "point_time_max_s": max(point_time_max_values)
        if point_time_max_values
        else None,
        "streams": selected,
    }
    return output


def _rosbag2_solid_state_diagnostics(
    config: CalibrationConfig,
    diagnostics: Mapping[str, object],
) -> Mapping[str, object] | None:
    """Aggregate sampled ROS2 diagnostics for configured solid-state topics."""

    solid_topics = {
        sensor.topic
        for sensor in config.sensors.values()
        if sensor.solid_state is not None and sensor.topic
    }
    streams = diagnostics.get("streams")
    if not isinstance(streams, Iterable):
        return None
    selected = [
        stream
        for stream in streams
        if isinstance(stream, Mapping) and stream.get("topic") in solid_topics
    ]
    if not selected:
        return None

    def values_for(key: str) -> list[float]:
        return [
            value
            for stream in selected
            if (value := _float_or_none(stream.get(key))) is not None
        ]

    point_time_streams = [
        stream for stream in selected if stream.get("point_time_available") is True
    ]
    point_time_min_values = [
        value
        for stream in point_time_streams
        if (
            value := _float_or_none(
                stream.get("sampled_point_time_min_s")
                if stream.get("sampled_point_time_min_s") is not None
                else stream.get("point_time_min_s")
            )
        )
        is not None
    ]
    point_time_max_values = [
        value
        for stream in point_time_streams
        if (
            value := _float_or_none(
                stream.get("sampled_point_time_max_s")
                if stream.get("sampled_point_time_max_s") is not None
                else stream.get("point_time_max_s")
            )
        )
        is not None
    ]
    return {
        "source": "rosbag2_sampled_solid_state_topics",
        "stream_count": len(selected),
        "point_time_available_stream_count": len(point_time_streams),
        "point_time_available_fraction": (
            len(point_time_streams) / len(selected) if selected else None
        ),
        "range_min_m": min(values_for("range_min_m"), default=None),
        "range_max_m": max(values_for("range_max_m"), default=None),
        "fov_azimuth_deg": max(values_for("fov_azimuth_deg"), default=None),
        "fov_elevation_deg": max(values_for("fov_elevation_deg"), default=None),
        "point_time_min_s": min(point_time_min_values, default=None),
        "point_time_max_s": max(point_time_max_values, default=None),
        "streams": selected,
    }


def _livox_geometry_metrics(
    diagnostics: Mapping[str, object],
    evaluation: SolidStateEvaluationConfig,
) -> dict[str, MetricResult]:
    range_min = _float_or_none(diagnostics.get("range_min_m"))
    range_max = _float_or_none(diagnostics.get("range_max_m"))
    azimuth = _float_or_none(diagnostics.get("fov_azimuth_deg"))
    elevation = _float_or_none(diagnostics.get("fov_elevation_deg"))
    range_span = (
        range_max - range_min
        if range_min is not None and range_max is not None
        else None
    )
    return {
        "solid_state_distance_min_m": _observed_metric(
            range_min,
            unit="m",
            label="minimum sampled solid-state LiDAR range",
        ),
        "solid_state_distance_max_m": _observed_metric(
            range_max,
            unit="m",
            label="maximum sampled solid-state LiDAR range",
        ),
        "solid_state_distance_span_m": _threshold_metric(
            range_span,
            evaluation.min_distance_span_m,
            unit="m",
            label="sampled solid-state LiDAR distance span",
        ),
        "solid_state_fov_azimuth_deg": _threshold_metric(
            azimuth,
            evaluation.min_fov_azimuth_deg,
            unit="deg",
            label="sampled solid-state LiDAR azimuth coverage",
        ),
        "solid_state_fov_elevation_deg": _threshold_metric(
            elevation,
            evaluation.min_fov_elevation_deg,
            unit="deg",
            label="sampled solid-state LiDAR elevation coverage",
        ),
    }


def _unavailable_geometry_metrics(reason: str) -> dict[str, MetricResult]:
    return {
        name: MetricResult(value=None, grade="warn", reason=reason)
        for name in (
            "solid_state_distance_min_m",
            "solid_state_distance_max_m",
            "solid_state_distance_span_m",
            "solid_state_fov_azimuth_deg",
            "solid_state_fov_elevation_deg",
            "solid_state_distance_bin_coverage_fraction",
            "solid_state_distance_bin_supported_count",
        )
    }


def _observed_metric(
    value: float | None,
    *,
    unit: str,
    label: str,
) -> MetricResult:
    return MetricResult(
        value=value,
        unit=unit,
        grade="pass" if value is not None else "warn",
        reason=f"{label}; unavailable" if value is None else f"{label}={value:g} {unit}",
    )


def _threshold_metric(
    value: float | None,
    minimum: float,
    *,
    unit: str,
    label: str,
) -> MetricResult:
    grade: Grade = "pass" if value is not None and value >= minimum else "warn"
    return MetricResult(
        value=value,
        unit=unit,
        grade=grade,
        reason=(
            f"{label}={value:g} {unit}; target >= {minimum:g} {unit}"
            if value is not None
            else f"{label} is unavailable; target >= {minimum:g} {unit}"
        ),
    )


def _holdout_metrics(
    config: CalibrationConfig,
    result: CalibrationResult | None,
) -> dict[str, MetricResult]:
    evaluation = config.evaluation.solid_state
    if result is None:
        return {
            "solid_state_holdout_window_count": MetricResult(
                value=None,
                unit="windows",
                grade="fail" if evaluation.require_independent_holdout else "warn",
                reason="result-level holdout window count is not available",
            ),
            "solid_state_holdout_independent": MetricResult(
                value=None,
                grade="fail" if evaluation.require_independent_holdout else "warn",
                reason="result-level holdout evidence is not available",
            ),
            "solid_state_known_bad_detectable_fraction": MetricResult(
                value=None,
                grade="warn",
                reason="result-level known-bad perturbation evidence is not available",
            ),
        }

    evidence = result.run.provenance.get("solid_state_evaluation")
    evidence_map = evidence if isinstance(evidence, Mapping) else {}
    independent = evidence_map.get("independent_holdout")
    train_window_count = _int_or_none(evidence_map.get("train_window_count"))
    holdout_window_count = _int_or_none(evidence_map.get("holdout_window_count"))
    window_count = (
        train_window_count + holdout_window_count
        if train_window_count is not None and holdout_window_count is not None
        else None
    )
    window_grade: Grade = (
        "pass"
        if window_count is not None and window_count >= evaluation.min_holdout_windows
        else "warn"
    )
    if isinstance(independent, bool):
        sufficient_windows = (
            window_count is not None and window_count >= evaluation.min_holdout_windows
        )
        holdout_metric = MetricResult(
            value=1.0 if independent else 0.0,
            grade=(
                "pass"
                if independent and sufficient_windows
                else ("fail" if evaluation.require_independent_holdout else "warn")
            ),
            reason=(
                "solid-state holdout is independent of the source map and has "
                "enough declared windows"
                if independent and sufficient_windows
                else (
                    "solid-state holdout is independent but has too few declared "
                    "windows"
                    if independent
                    else (
                        "solid-state holdout is a single-pair/source-map query and is "
                        "not an independent temporal window"
                    )
                )
            ),
        )
    else:
        holdout_metric = MetricResult(
            value=None,
            grade="fail" if evaluation.require_independent_holdout else "warn",
            reason="solid-state holdout independence was not declared",
        )

    known_bad = result.metrics.get("lidar_pair_known_bad_detectable_fraction")
    known_bad_value = known_bad.value if known_bad is not None else None
    known_bad_grade: Grade = (
        "pass"
        if known_bad_value is not None
        and known_bad_value >= evaluation.min_known_bad_detectable_fraction
        else "warn"
    )
    return {
        "solid_state_holdout_window_count": MetricResult(
            value=float(window_count) if window_count is not None else None,
            unit="windows",
            grade=window_grade,
            reason=(
                f"declared train+holdout windows={window_count}; target >= "
                f"{evaluation.min_holdout_windows}"
                if window_count is not None
                else "declared train+holdout window count is unavailable"
            ),
        ),
        "solid_state_holdout_independent": holdout_metric,
        "solid_state_known_bad_detectable_fraction": MetricResult(
            value=known_bad_value,
            grade=known_bad_grade,
            reason=(
                "known-bad 6DoF perturbation detection fraction; target >= "
                f"{evaluation.min_known_bad_detectable_fraction:g}"
                if known_bad_value is not None
                else "known-bad perturbation detection fraction is unavailable"
            ),
        ),
    }


def _livox_distance_bin_counts(
    config: CalibrationConfig,
    evaluation: SolidStateEvaluationConfig,
    *,
    diagnostics: Mapping[str, object] | None = None,
) -> list[int] | None:
    if config.dataset.type != "livox_pcd":
        if config.dataset.type == "rosbag1":
            rosbag_diagnostics = diagnostics
            if rosbag_diagnostics is None:
                rosbag_diagnostics = summarize_rosbag1(
                    config.dataset.path,
                    sample_limit=config.dataset.sample_limit or 4,
                ).as_dict()
            solid_state = (
                rosbag_diagnostics
                if rosbag_diagnostics.get("source")
                == "rosbag1_sampled_solid_state_topics"
                else _rosbag1_solid_state_diagnostics(config, rosbag_diagnostics)
            )
            if solid_state is None:
                return None
            streams = solid_state.get("streams")
            if not isinstance(streams, Iterable):
                return None
            expected_edges = list(evaluation.distance_bins_m)
            counts = [0] * (len(expected_edges) - 1)
            matched = False
            for stream in streams:
                if not isinstance(stream, Mapping):
                    continue
                edges = stream.get("distance_bin_edges_m")
                values = stream.get("distance_bin_counts")
                if edges != [0.0, 10.0, 20.0, 40.0, 80.0] or not isinstance(
                    values, list
                ):
                    continue
                if expected_edges != [0.0, 10.0, 20.0, 40.0, 80.0]:
                    return None
                for index, value in enumerate(values):
                    if index < len(counts) and isinstance(value, int):
                        counts[index] += value
                matched = True
            return counts if matched else None
        if config.dataset.type == "rosbag2":
            rosbag2_diagnostics = diagnostics
            if rosbag2_diagnostics is None:
                rosbag2_diagnostics = summarize_rosbag2(
                    config.dataset.path,
                    sample_limit=config.dataset.sample_limit or 4,
                ).as_dict()
            solid_state = (
                rosbag2_diagnostics
                if rosbag2_diagnostics.get("source")
                == "rosbag2_sampled_solid_state_topics"
                else _rosbag2_solid_state_diagnostics(config, rosbag2_diagnostics)
            )
            if solid_state is None:
                return None
            streams = solid_state.get("streams")
            if not isinstance(streams, Iterable):
                return None
            expected_edges = list(evaluation.distance_bins_m)
            counts = [0] * (len(expected_edges) - 1)
            matched = False
            for stream in streams:
                if not isinstance(stream, Mapping):
                    continue
                edges = stream.get("distance_bin_edges_m")
                values = stream.get("distance_bin_counts")
                if edges != [0.0, 10.0, 20.0, 40.0, 80.0] or not isinstance(
                    values, list
                ):
                    continue
                if expected_edges != [0.0, 10.0, 20.0, 40.0, 80.0]:
                    return None
                for index, value in enumerate(values):
                    if index < len(counts) and isinstance(value, int):
                        counts[index] += value
                matched = True
            return counts if matched else None
        return None
    files = find_livox_pcd_files(config.dataset.path)
    if not files:
        return None
    counts = [0] * (len(evaluation.distance_bins_m) - 1)
    for file_path in files[: config.dataset.sample_limit or 4]:
        try:
            points = read_livox_binary_pcd(file_path)
        except (OSError, ValueError, struct.error):
            continue
        for x, y, z, _intensity in points:
            distance = math.sqrt(x * x + y * y + z * z)
            for index, (lower, upper) in enumerate(
                pairwise(evaluation.distance_bins_m)
            ):
                is_last = index == len(counts) - 1
                if lower <= distance < upper or (is_last and distance == upper):
                    counts[index] += 1
                    break
    return counts


def _temperature_is_observed(temperature: object) -> bool:
    if temperature is None:
        return False
    return any(
        getattr(temperature, field_name, None) is not None
        for field_name in ("sensor_c", "ambient_c", "min_c", "max_c")
    )


def _capture_point_time_available(
    result: CalibrationResult | None,
    sensor_name: str,
) -> bool:
    if result is None or result.solid_state is None:
        return False
    window = result.solid_state.capture_windows.get(sensor_name)
    return bool(window is not None and window.point_time_offsets_available)


def _point_time_declared(profile: object) -> bool:
    return bool(
        getattr(profile, "point_time_available", False)
        and getattr(profile, "point_time_reference", "unknown") != "unknown"
        and getattr(profile, "point_time_unit", "unknown") != "unknown"
    )


def _inspection_point_time_available(
    config: CalibrationConfig,
    inspection: DatasetInspection | None,
    sensor_name: str,
) -> bool | None:
    """Return observed ROS1 point-time availability, when inspection can say."""

    if inspection is None or config.dataset.type != "rosbag1":
        return None
    if not inspection.exists:
        return False
    sensor = config.sensors.get(sensor_name)
    if sensor is None or not sensor.topic:
        return None
    diagnostics = inspection.diagnostics.get("rosbag1")
    if not isinstance(diagnostics, Mapping):
        return None
    streams = diagnostics.get("streams")
    if not isinstance(streams, Iterable):
        return None
    for stream in streams:
        if not isinstance(stream, Mapping) or stream.get("topic") != sensor.topic:
            continue
        return stream.get("point_time_available") is True
    return None


def _capture_temperature_is_observed(
    result: CalibrationResult | None,
    sensor_name: str,
) -> bool:
    if result is None or result.solid_state is None:
        return False
    window = result.solid_state.capture_windows.get(sensor_name)
    return bool(window is not None and _temperature_is_observed(window.temperature))


def _temperature_values(
    profiles: Mapping[str, object],
    result: CalibrationResult | None,
) -> list[float]:
    values: list[float] = []
    for sensor_name, profile in profiles.items():
        temperature = getattr(profile, "temperature", None)
        if temperature is not None:
            for field_name in ("sensor_c", "ambient_c", "min_c", "max_c"):
                value = _float_or_none(getattr(temperature, field_name, None))
                if value is not None:
                    values.append(value)
        if result is None or result.solid_state is None:
            continue
        window = result.solid_state.capture_windows.get(sensor_name)
        if window is None or window.temperature is None:
            continue
        for field_name in ("sensor_c", "ambient_c", "min_c", "max_c"):
            value = _float_or_none(getattr(window.temperature, field_name, None))
            if value is not None:
                values.append(value)
    return values


def _count(values: Iterable[bool]) -> int:
    return sum(1 for value in values if value)


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None
