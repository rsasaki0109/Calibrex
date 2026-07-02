"""Radar extrinsic velocity-consistency evaluation."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from calibrex.core.config import CalibrationConfig
from calibrex.core.geometry import (
    SE3,
    Vector3,
    quaternion_conjugate_xyzw,
    rotate_vector_xyzw,
)
from calibrex.core.result import CalibrationResult, MetricResult
from calibrex.data.base import TimestampedRecord
from calibrex.data.inspect import DatasetInspection
from calibrex.data.nuscenes import (
    NuScenesDataset,
    NuScenesEgoPose,
    find_nuscenes_version_path,
    read_nuscenes_ego_poses,
)
from calibrex.data.nuscenes_radar import NuScenesRadarPCD, read_nuscenes_radar_pcd

_MIN_EGO_SPEED_MPS = 0.5
_MIN_INLIER_COUNT = 5
_MIN_RANGE_M = 1.0
_STATIC_DYN_PROP = {1, 3, 5, 7}
_STATIC_VCOMP_THRESHOLD_MPS = 0.5
_MAX_FRAMES_PER_SENSOR = 30
_YAW_PERTURBATION_DEG = 5.0
_PERTURBATION_DETECT_MARGIN_MPS = 0.05


@dataclass(frozen=True)
class RadarVelocityConsistencyStats:
    """Aggregate radar velocity-consistency diagnostics."""

    median_abs_residual_mps: float | None
    inlier_rmse_mps: float | None
    return_count: int
    inlier_count: int
    frame_count: int
    status: str
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "median_abs_residual_mps": self.median_abs_residual_mps,
            "inlier_rmse_mps": self.inlier_rmse_mps,
            "return_count": self.return_count,
            "inlier_count": self.inlier_count,
            "frame_count": self.frame_count,
            "status": self.status,
            "reason": self.reason,
        }


def radar_velocity_metrics_from_result(
    config: CalibrationConfig,
    result: CalibrationResult,
    inspection: DatasetInspection,
) -> dict[str, MetricResult]:
    """Score radar radial velocity consistency for nuScenes radar extrinsics."""

    _ = inspection

    if config.dataset.type != "nuscenes":
        return {}
    if "radar_lidar_velocity_consistency" not in config.evaluation.metrics:
        return {}
    radar_sensors = _configured_radar_sensors(config)
    if not radar_sensors:
        return {}

    stats, provenance = summarize_nuscenes_radar_velocity_consistency(
        dataset_path=config.dataset.path,
        radar_sensors=radar_sensors,
        extrinsics_lookup=lambda sensor_name: _resolve_radar_extrinsic(result, sensor_name),
    )
    if provenance:
        result.run.provenance["radar_velocity_consistency"] = provenance

    if stats.status == "scored" and stats.median_abs_residual_mps is not None:
        return {
            "radar_lidar_velocity_consistency": MetricResult(
                value=stats.median_abs_residual_mps,
                unit="m/s",
                reason=(
                    f"median absolute static-target radial velocity residual over "
                    f"{stats.inlier_count} inliers in {stats.frame_count} radar frame(s)"
                ),
            )
        }
    return {
        "radar_lidar_velocity_consistency": MetricResult(
            value=None,
            unit="m/s",
            grade="warn",
            reason=stats.reason or "radar velocity consistency could not be scored",
        )
    }


def summarize_nuscenes_radar_velocity_consistency(
    *,
    dataset_path: str | Path,
    radar_sensors: Mapping[str, str],
    extrinsics_lookup: Callable[[str], SE3 | None],
    max_frames_per_sensor: int = _MAX_FRAMES_PER_SENSOR,
) -> tuple[RadarVelocityConsistencyStats, dict[str, object]]:
    """Compute radar velocity consistency and optional yaw perturbation probe."""

    version_path = find_nuscenes_version_path(dataset_path)
    if version_path is None:
        return _unavailable_stats("nuScenes metadata tables were not found"), {}

    dataset = NuScenesDataset(dataset_path)
    ego_poses = read_nuscenes_ego_poses(dataset_path)
    if not ego_poses:
        return _unavailable_stats("nuScenes ego_pose table is empty"), {}

    residuals: list[float] = []
    frame_count = 0
    return_count = 0
    inlier_count = 0
    scored_sensor = False
    baseline_value: float | None = None

    for sensor_name, channel in sorted(radar_sensors.items()):
        extrinsic = extrinsics_lookup(sensor_name)
        if extrinsic is None:
            continue
        records = sorted(dataset.records(channel), key=lambda record: record.timestamp_ns)
        if len(records) < 2:
            continue
        scored_sensor = True
        sampled = _sample_records(records, max_frames_per_sensor)
        for index in range(1, len(sampled)):
            current = sampled[index]
            previous = sampled[index - 1]
            pose_current = ego_poses.get(str(current.metadata.get("ego_pose_token")))
            pose_previous = ego_poses.get(str(previous.metadata.get("ego_pose_token")))
            if pose_current is None or pose_previous is None:
                continue
            ego_velocity = _ego_velocity_mps(pose_previous, pose_current)
            if _vector_norm(ego_velocity) < _MIN_EGO_SPEED_MPS:
                continue
            payload_path = current.payload_path
            if payload_path is None or not Path(payload_path).exists():
                continue
            try:
                pcd = read_nuscenes_radar_pcd(payload_path)
            except Exception:
                continue
            frame_stats = score_radar_velocity_frame(
                pcd=pcd,
                ego_velocity_mps=ego_velocity,
                t_ego_radar=extrinsic,
            )
            if frame_stats is None:
                continue
            frame_count += 1
            return_count += frame_stats.return_count
            inlier_count += frame_stats.inlier_count
            residuals.extend(frame_stats.abs_residuals_mps)

    if not scored_sensor:
        return _unavailable_stats("no radar extrinsics or sample_data were available"), {}

    if inlier_count < _MIN_INLIER_COUNT or not residuals:
        return (
            RadarVelocityConsistencyStats(
                median_abs_residual_mps=None,
                inlier_rmse_mps=None,
                return_count=return_count,
                inlier_count=inlier_count,
                frame_count=frame_count,
                status="inconclusive",
                reason=(
                    "insufficient static radar returns or ego motion for velocity consistency"
                ),
            ),
            {},
        )

    median_abs = _median(residuals)
    inlier_rmse = math.sqrt(sum(value * value for value in residuals) / len(residuals))
    baseline_value = median_abs
    provenance: dict[str, object] = {
        "status": "scored",
        "frame_count": frame_count,
        "return_count": return_count,
        "inlier_count": inlier_count,
        "median_abs_residual_mps": median_abs,
        "inlier_rmse_mps": inlier_rmse,
    }

    perturbation = _yaw_perturbation_probe(
        dataset=dataset,
        radar_sensors=radar_sensors,
        ego_poses=ego_poses,
        extrinsics_lookup=extrinsics_lookup,
        max_frames_per_sensor=max_frames_per_sensor,
        baseline_median_abs_mps=baseline_value,
    )
    if perturbation is not None:
        provenance["perturbation_probe"] = perturbation

    return (
        RadarVelocityConsistencyStats(
            median_abs_residual_mps=median_abs,
            inlier_rmse_mps=inlier_rmse,
            return_count=return_count,
            inlier_count=inlier_count,
            frame_count=frame_count,
            status="scored",
        ),
        provenance,
    )


@dataclass(frozen=True)
class _FrameResidualStats:
    return_count: int
    inlier_count: int
    abs_residuals_mps: list[float]


def score_radar_velocity_frame(
    *,
    pcd: NuScenesRadarPCD,
    ego_velocity_mps: Vector3,
    t_ego_radar: SE3,
) -> _FrameResidualStats | None:
    """Score one radar frame against ego velocity and a candidate extrinsic."""

    if pcd.point_count == 0:
        return None
    x = pcd.array("x")
    y = pcd.array("y")
    z = pcd.array("z")
    vx = pcd.array("vx")
    vy = pcd.array("vy")
    dyn_prop = pcd.fields.get("dyn_prop")
    vx_comp = pcd.fields.get("vx_comp")
    vy_comp = pcd.fields.get("vy_comp")

    inv_rotation = quaternion_conjugate_xyzw(t_ego_radar.rotation_quat_xyzw)
    v_radar = rotate_vector_xyzw(inv_rotation, ego_velocity_mps)

    abs_residuals: list[float] = []
    return_count = int(pcd.point_count)
    for index in range(return_count):
        px = float(x[index])
        py = float(y[index])
        pz = float(z[index])
        range_m = math.sqrt(px * px + py * py + pz * pz)
        if range_m < _MIN_RANGE_M:
            continue
        if not _is_static_return(
            dyn_prop=dyn_prop,
            index=index,
            vx_comp=vx_comp,
            vy_comp=vy_comp,
        ):
            continue
        measured = _radial_velocity(float(vx[index]), float(vy[index]), px, py, pz)
        predicted = -_dot(v_radar, _unit_vector(px, py, pz))
        abs_residuals.append(abs(measured - predicted))

    if not abs_residuals:
        return None
    return _FrameResidualStats(
        return_count=return_count,
        inlier_count=len(abs_residuals),
        abs_residuals_mps=abs_residuals,
    )


def _yaw_perturbation_probe(
    *,
    dataset: NuScenesDataset,
    radar_sensors: Mapping[str, str],
    ego_poses: Mapping[str, NuScenesEgoPose],
    extrinsics_lookup: Callable[[str], SE3 | None],
    max_frames_per_sensor: int,
    baseline_median_abs_mps: float,
) -> dict[str, object] | None:
    yaw_rad = math.radians(_YAW_PERTURBATION_DEG)
    half = yaw_rad / 2.0
    perturbation = SE3((0.0, 0.0, 0.0), (0.0, 0.0, math.sin(half), math.cos(half)))
    perturbed_residuals: list[float] = []

    for sensor_name, channel in sorted(radar_sensors.items()):
        baseline_extrinsic = extrinsics_lookup(sensor_name)
        if baseline_extrinsic is None:
            continue
        perturbed_extrinsic = perturbation.compose(baseline_extrinsic)
        records = sorted(dataset.records(channel), key=lambda record: record.timestamp_ns)
        sampled = _sample_records(records, max_frames_per_sensor)
        for index in range(1, len(sampled)):
            current = sampled[index]
            previous = sampled[index - 1]
            pose_current = ego_poses.get(str(current.metadata.get("ego_pose_token")))
            pose_previous = ego_poses.get(str(previous.metadata.get("ego_pose_token")))
            if pose_current is None or pose_previous is None:
                continue
            ego_velocity = _ego_velocity_mps(pose_previous, pose_current)
            if _vector_norm(ego_velocity) < _MIN_EGO_SPEED_MPS:
                continue
            payload_path = current.payload_path
            if payload_path is None or not Path(payload_path).exists():
                continue
            try:
                pcd = read_nuscenes_radar_pcd(payload_path)
            except Exception:
                continue
            frame_stats = score_radar_velocity_frame(
                pcd=pcd,
                ego_velocity_mps=ego_velocity,
                t_ego_radar=perturbed_extrinsic,
            )
            if frame_stats is None:
                continue
            perturbed_residuals.extend(frame_stats.abs_residuals_mps)

    if len(perturbed_residuals) < _MIN_INLIER_COUNT:
        return None
    perturbed_median = _median(perturbed_residuals)
    detectable = perturbed_median > baseline_median_abs_mps + _PERTURBATION_DETECT_MARGIN_MPS
    return {
        "dof": "yaw_deg",
        "amount_deg": _YAW_PERTURBATION_DEG,
        "baseline_median_abs_residual_mps": baseline_median_abs_mps,
        "perturbed_median_abs_residual_mps": perturbed_median,
        "detectable": detectable,
    }


def _ego_velocity_mps(previous: NuScenesEgoPose, current: NuScenesEgoPose) -> Vector3:
    dt_s = (current.timestamp_ns - previous.timestamp_ns) / 1_000_000_000.0
    if dt_s <= 0.0:
        return (0.0, 0.0, 0.0)
    v_global = (
        (current.translation_m[0] - previous.translation_m[0]) / dt_s,
        (current.translation_m[1] - previous.translation_m[1]) / dt_s,
        (current.translation_m[2] - previous.translation_m[2]) / dt_s,
    )
    inv_rotation = quaternion_conjugate_xyzw(current.rotation_quat_xyzw)
    return rotate_vector_xyzw(inv_rotation, v_global)


def _resolve_radar_extrinsic(result: CalibrationResult, sensor_name: str) -> SE3 | None:
    transform_name = f"T_ego_{sensor_name}"
    for mapping in (
        result.transforms,
        result.candidate_extrinsics,
        result.reference_extrinsics,
    ):
        payload = mapping.get(transform_name)
        if payload is None:
            continue
        if hasattr(payload, "translation_m"):
            return SE3.from_lists(list(payload.translation_m), list(payload.rotation_quat_xyzw))
        if isinstance(payload, Mapping):
            translation = payload.get("translation_m")
            rotation = payload.get("rotation_quat_xyzw")
            if isinstance(translation, list) and isinstance(rotation, list):
                return SE3.from_lists(translation, rotation)
    return None


def _configured_radar_sensors(config: CalibrationConfig) -> dict[str, str]:
    sensors: dict[str, str] = {}
    for name, sensor in config.sensors.items():
        if sensor.type != "radar":
            continue
        topic = sensor.topic or name.upper()
        sensors[name] = topic
    return sensors


def _sample_records(
    records: Sequence[TimestampedRecord],
    max_count: int,
) -> list[TimestampedRecord]:
    if len(records) <= max_count:
        return list(records)
    if max_count < 2:
        return list(records[:max_count])
    step = max(1, (len(records) - 1) // (max_count - 1))
    sampled = [records[index] for index in range(0, len(records), step)]
    if sampled[-1] is not records[-1]:
        sampled.append(records[-1])
    return sampled[:max_count]


def _is_static_return(
    *,
    dyn_prop: object | None,
    index: int,
    vx_comp: object | None,
    vy_comp: object | None,
) -> bool:
    if dyn_prop is not None:
        try:
            value = int(dyn_prop[index])  # type: ignore[index]
        except (TypeError, ValueError, IndexError):
            return False
        return value in _STATIC_DYN_PROP
    if vx_comp is not None and vy_comp is not None:
        speed = math.hypot(float(vx_comp[index]), float(vy_comp[index]))  # type: ignore[index]
        return speed <= _STATIC_VCOMP_THRESHOLD_MPS
    return False


def _radial_velocity(vx: float, vy: float, x: float, y: float, z: float) -> float:
    ux, uy, _uz = _unit_vector(x, y, z)
    return vx * ux + vy * uy


def _unit_vector(x: float, y: float, z: float) -> Vector3:
    norm = math.sqrt(x * x + y * y + z * z)
    if norm < 1e-9:
        return (0.0, 0.0, 0.0)
    return (x / norm, y / norm, z / norm)


def _dot(left: Vector3, right: Vector3) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _vector_norm(value: Vector3) -> float:
    return math.sqrt(_dot(value, value))


def _median(values: Iterable[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _unavailable_stats(reason: str) -> RadarVelocityConsistencyStats:
    return RadarVelocityConsistencyStats(
        median_abs_residual_mps=None,
        inlier_rmse_mps=None,
        return_count=0,
        inlier_count=0,
        frame_count=0,
        status="unavailable",
        reason=reason,
    )
