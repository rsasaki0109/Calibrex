"""Actionable data-collection recommendations from quality diagnostics."""

from __future__ import annotations

from slac.core.result import CalibrationResult, DegeneracyResult, MetricResult


def build_recommendations(result: CalibrationResult) -> list[str]:
    """Return concise collection or validation actions for the current result."""

    return _build_recommendations(
        metrics=result.metrics,
        degeneracy=result.degeneracy,
        observability_grade=result.observability.grade,
        has_estimating_time_offsets=_has_estimating_time_offsets(result),
        domain=result.run.domain,
    )


def build_inspection_recommendations(
    metrics: dict[str, MetricResult],
    degeneracy: DegeneracyResult,
    *,
    domain: str,
) -> list[str]:
    """Return data-collection actions for dataset inspection output."""

    return _build_recommendations(
        metrics=metrics,
        degeneracy=degeneracy,
        observability_grade="pass",
        has_estimating_time_offsets=False,
        domain=domain,
    )


def _build_recommendations(
    *,
    metrics: dict[str, MetricResult],
    degeneracy: DegeneracyResult,
    observability_grade: str,
    has_estimating_time_offsets: bool,
    domain: str,
) -> list[str]:
    recommendations: list[str] = []
    degeneracy_reason = degeneracy.reason or ""

    if _metric_needs_attention(metrics.get("lidar_frame_coverage")) or (
        "no LiDAR frames" in degeneracy_reason
    ):
        recommendations.append(
            "collect a longer fixed-LiDAR sequence before calibration"
        )
    if _metric_needs_attention(metrics.get("lidar_point_coverage")) or (
        "no sampled LiDAR points" in degeneracy_reason
    ):
        recommendations.append(
            "verify the LiDAR topic, packet decoding, and data extraction before solving"
        )
    if _metric_needs_attention(metrics.get("lidar_spatial_coverage_m")) or (
        "limited vertical LiDAR structure" in degeneracy_reason
    ):
        recommendations.append(
            "include vertical structure such as facades, poles, ramps, or road grade changes"
        )
    if (
        _metric_needs_attention(metrics.get("lidar_local_planarity"))
        or "local planar LiDAR neighborhoods" in degeneracy_reason
        or "weak local LiDAR planarity" in degeneracy_reason
    ):
        recommendations.append(
            "drive past static planar surfaces such as walls, building fronts, and curbs"
        )
    if _metric_needs_attention(metrics.get("lidar_map_sharpness")) or (
        "low LiDAR map sharpness" in degeneracy_reason
    ):
        recommendations.append(
            "avoid feature-poor straight-road-only logs; add static edges and varied geometry"
        )
    if _metric_needs_attention(metrics.get("lidar_point_to_plane_rmse_m")) or (
        "holdout point-to-plane" in degeneracy_reason
    ):
        recommendations.append(
            "repeat passes over mostly static structure and keep dynamic traffic out of holdout"
        )
    if (
        _metric_needs_attention(metrics.get("lidar_world_map_weak_dof_count"))
        or "weak LiDAR world-map DoF" in degeneracy_reason
    ):
        recommendations.append(
            "collect a fixed-LiDAR sequence with richer turns, grade changes, "
            "and static 3D structure"
        )
    if _metric_needs_attention(metrics.get("vehicle_motion_duration_s")) or (
        "short vehicle motion duration" in degeneracy_reason
    ):
        recommendations.append(
            "record a longer driving segment with enough motion before fixed-LiDAR calibration"
        )
    if (
        _metric_needs_attention(metrics.get("vehicle_mean_speed_mps"))
        or _metric_needs_attention(metrics.get("vehicle_speed_range_mps"))
        or "low vehicle speed excitation" in degeneracy_reason
        or "low vehicle acceleration excitation" in degeneracy_reason
    ):
        recommendations.append(
            "include acceleration, braking, and speed changes in the driving log"
        )
    if _metric_needs_attention(metrics.get("vehicle_yaw_excitation_deg")) or (
        "limited yaw excitation" in degeneracy_reason
    ):
        recommendations.append(
            "include turns, lane changes, or figure-eight style motion for yaw observability"
        )
    if (
        _metric_needs_attention(metrics.get("camera_lidar_timestamp_alignment_ms"))
        or _metric_needs_attention(metrics.get("lidar_oxts_timestamp_alignment_ms"))
        or "timestamp mismatch" in degeneracy_reason
        or "timestamp pairs" in degeneracy_reason
    ):
        recommendations.append(
            "verify sensor timestamp synchronization or enable time-offset estimation"
        )
    if (
        _metric_needs_attention(metrics.get("lidar_camera_overlay_readiness"))
        or _metric_needs_attention(metrics.get("lidar_camera_overlay_score"))
        or _metric_needs_attention(metrics.get("lidar_camera_projected_points"))
        or _metric_needs_attention(metrics.get("lidar_camera_projection_ratio"))
        or _metric_needs_attention(metrics.get("lidar_camera_projection_horizontal_coverage"))
        or _metric_needs_attention(metrics.get("lidar_camera_projection_vertical_coverage"))
        or _metric_needs_attention(metrics.get("lidar_camera_edge_alignment_score"))
        or _metric_needs_attention(metrics.get("lidar_camera_depth_discontinuity_points"))
        or _metric_needs_attention(metrics.get("lidar_camera_depth_edge_alignment_score"))
    ):
        recommendations.append(
            "verify camera-LiDAR calibration files and collect logs with clear shared field of view"
        )
    if _metric_needs_attention(metrics.get("lidar_camera_mutual_information_score")):
        recommendations.append(
            "preserve image exposure and LiDAR intensity fields for mutual-information checks"
        )
    if degeneracy.grade != "pass" and not recommendations:
        recommendations.append(
            "collect motion with richer yaw, pitch, roll, and translation excitation"
        )
    if observability_grade != "pass":
        recommendations.append(
            "add cross-modal overlap and holdout sequences before trusting deployment"
        )
    if has_estimating_time_offsets:
        recommendations.append(
            "capture acceleration, braking, and turns to make time offsets observable"
        )
    if domain == "autonomous_driving":
        recommendations.append(
            "validate camera-lidar-radar timing on driving logs with dynamic objects held out"
        )
    if not recommendations:
        recommendations.append(
            "archive config, result, report, and dataset manifest for reproducibility"
        )
    return _dedupe(recommendations)


def _metric_needs_attention(metric: MetricResult | None) -> bool:
    return metric is not None and metric.grade != "pass"


def _has_estimating_time_offsets(result: CalibrationResult) -> bool:
    return any(offset.quality.grade != "pass" for offset in result.time_offsets.values())


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output
