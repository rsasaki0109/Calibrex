"""Degeneracy checks for calibration diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from calibrex.core.result import DegeneracyResult, Grade
from calibrex.data.inspect import DatasetInspection
from calibrex.evaluation.metrics import worst_grade


def degeneracy_from_inspection(inspection: DatasetInspection) -> DegeneracyResult:
    """Build a provisional degeneracy result from dataset inspection diagnostics."""

    velodyne = inspection.diagnostics.get("velodyne_points")
    oxts = inspection.diagnostics.get("oxts_motion")
    timing = inspection.diagnostics.get("timestamp_alignment")
    if (
        not isinstance(velodyne, Mapping)
        and not isinstance(oxts, Mapping)
        and not isinstance(timing, Mapping)
    ):
        return DegeneracyResult(
            grade="warn",
            reason=(
                "native SLAC observability and degeneracy analysis are not implemented "
                "in this alpha"
            ),
        )

    issues: list[tuple[Grade, str]] = []
    if isinstance(velodyne, Mapping):
        _append_lidar_issues(issues, velodyne)
    if isinstance(oxts, Mapping):
        _append_motion_issues(issues, oxts)
    if isinstance(timing, Mapping):
        _append_timing_issues(issues, timing)

    if not issues:
        return DegeneracyResult(grade="pass", reason=None)
    grade = cast(Grade, worst_grade([grade for grade, _ in issues]))
    return DegeneracyResult(
        grade=grade,
        reason="; ".join(reason for _, reason in issues),
    )


def _append_lidar_issues(
    issues: list[tuple[Grade, str]],
    velodyne: Mapping[object, object],
) -> None:
    frame_count = _float_or_none(velodyne.get("frame_count"))
    sampled_point_count = _float_or_none(velodyne.get("sampled_point_count"))
    planarity_voxel_count = _float_or_none(velodyne.get("planarity_voxel_count"))
    local_planarity = _float_or_none(velodyne.get("local_planarity_mean"))
    map_sharpness = _float_or_none(velodyne.get("map_sharpness_score"))
    point_to_plane_train = _float_or_none(velodyne.get("point_to_plane_rmse_train_m"))
    point_to_plane_holdout = _float_or_none(velodyne.get("point_to_plane_rmse_holdout_m"))
    bounds_min = _vector3_or_none(velodyne.get("bounds_min_m"))
    bounds_max = _vector3_or_none(velodyne.get("bounds_max_m"))
    if frame_count is not None and frame_count < 1:
        issues.append(("fail", "no LiDAR frames are available"))
    if sampled_point_count is not None and sampled_point_count < 1:
        issues.append(("fail", "no sampled LiDAR points are available"))
    if planarity_voxel_count is not None:
        if planarity_voxel_count < 1:
            issues.append(("fail", "no local planar LiDAR neighborhoods were found"))
        elif planarity_voxel_count < 3:
            issues.append(("warn", "too few local planar LiDAR neighborhoods"))
    if bounds_min is not None and bounds_max is not None:
        z_extent = bounds_max[2] - bounds_min[2]
        if z_extent < 1.0:
            issues.append(("warn", f"limited vertical LiDAR structure: z_extent={z_extent:g} m"))
    if local_planarity is not None and local_planarity < 0.10:
        issues.append(("warn", f"weak local LiDAR planarity: {local_planarity:g}"))
    if map_sharpness is not None and map_sharpness < 0.05:
        issues.append(("warn", f"low LiDAR map sharpness proxy: {map_sharpness:g}"))
    if point_to_plane_train is not None and point_to_plane_holdout is not None:
        issues.extend(_holdout_degeneracy(point_to_plane_train, point_to_plane_holdout))


def _append_motion_issues(issues: list[tuple[Grade, str]], oxts: Mapping[object, object]) -> None:
    packet_count = _float_or_none(oxts.get("packet_count"))
    duration_sec = _float_or_none(oxts.get("duration_sec"))
    mean_speed = _float_or_none(oxts.get("mean_speed_mps"))
    speed_range = _float_or_none(oxts.get("speed_range_mps"))
    yaw_excitation = _float_or_none(oxts.get("yaw_excitation_deg"))
    if packet_count is not None and packet_count < 2:
        issues.append(("warn", "too few OXTS motion packets for excitation checks"))
    if duration_sec is not None and duration_sec < 10.0:
        issues.append(("warn", f"short vehicle motion duration: {duration_sec:g} s"))
    if mean_speed is not None and mean_speed < 1.0:
        issues.append(("warn", f"low vehicle speed excitation: mean_speed={mean_speed:g} m/s"))
    if speed_range is not None and speed_range < 1.0:
        issues.append(
            ("warn", f"low vehicle acceleration excitation: speed_range={speed_range:g} m/s")
        )
    if yaw_excitation is not None and yaw_excitation < 10.0:
        issues.append(("warn", f"limited yaw excitation: yaw_range={yaw_excitation:g} deg"))


def _append_timing_issues(issues: list[tuple[Grade, str]], timing: Mapping[object, object]) -> None:
    camera_lidar = _float_or_none(timing.get("camera_lidar_max_abs_dt_ms"))
    lidar_oxts = _float_or_none(timing.get("lidar_oxts_max_abs_dt_ms"))
    camera_pairs = _float_or_none(timing.get("camera_lidar_pair_count"))
    oxts_pairs = _float_or_none(timing.get("lidar_oxts_pair_count"))
    if camera_pairs is not None and camera_pairs < 1:
        issues.append(("warn", "no camera-LiDAR timestamp pairs were available"))
    if oxts_pairs is not None and oxts_pairs < 1:
        issues.append(("warn", "no LiDAR-OXTS timestamp pairs were available"))
    if camera_lidar is not None and camera_lidar > 50.0:
        issues.append(("warn", f"large camera-LiDAR timestamp mismatch: {camera_lidar:g} ms"))
    if lidar_oxts is not None and lidar_oxts > 50.0:
        issues.append(("warn", f"large LiDAR-OXTS timestamp mismatch: {lidar_oxts:g} ms"))


def _holdout_degeneracy(train: float, holdout: float) -> list[tuple[Grade, str]]:
    if holdout <= 0.05:
        return []
    if train <= 1.0e-9:
        return [("warn", f"LiDAR holdout point-to-plane RMSE is high: {holdout:g} m")]
    ratio = holdout / train
    if ratio > 3.0:
        return [
            (
                "warn",
                f"LiDAR holdout point-to-plane RMSE is much worse than train: ratio={ratio:g}",
            )
        ]
    return []


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _vector3_or_none(value: object) -> tuple[float, float, float] | None:
    if not isinstance(value, list | tuple) or len(value) != 3:
        return None
    vector: list[float] = []
    for item in value:
        numeric = _float_or_none(item)
        if numeric is None:
            return None
        vector.append(numeric)
    return (vector[0], vector[1], vector[2])
