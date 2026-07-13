"""KITTI raw dataset reader for fixed-mounted LiDAR calibration examples."""

from __future__ import annotations

import struct
import zlib
from bisect import bisect_left
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import acos, atan2, cos, floor, pi, radians, sin, sqrt
from pathlib import Path

from calibrex.core.evidence import (
    CorrespondenceArtifact,
    DatasetSliceArtifact,
    MapArtifact,
    stable_artifact_id,
    validate_no_frame_overlap,
)
from calibrex.core.geometry import SE3, Vector3, quaternion_xyzw_from_rotation_matrix
from calibrex.data.base import StreamSummary, TimestampedRecord
from calibrex.data.manifest import DatasetManifest, find_manifest, load_manifest


@dataclass(frozen=True)
class KITTITimestamp:
    """One KITTI timestamp line."""

    index: int
    timestamp_ns: int
    raw: str


@dataclass(frozen=True)
class KITTICameraLidarPair:
    """A concrete KITTI image/Velodyne frame pair for overlay diagnostics."""

    camera_index: int
    lidar_index: int
    camera_timestamp_ns: int | None
    lidar_timestamp_ns: int | None
    delta_ms: float | None
    camera_path: str
    lidar_path: str
    lidar_point_count: int

    def as_dict(self) -> dict[str, object]:
        """Return a JSON/YAML-friendly representation."""

        return {
            "camera_index": self.camera_index,
            "lidar_index": self.lidar_index,
            "camera_timestamp_ns": self.camera_timestamp_ns,
            "lidar_timestamp_ns": self.lidar_timestamp_ns,
            "delta_ms": self.delta_ms,
            "camera_path": self.camera_path,
            "lidar_path": self.lidar_path,
            "lidar_point_count": self.lidar_point_count,
        }


@dataclass(frozen=True)
class KITTICameraProjectionModel:
    """Projection model from KITTI raw camera calibration files."""

    camera: str
    projection_matrix: tuple[float, ...]
    rectification_matrix: tuple[float, ...]
    image_size_px: tuple[int, int] | None = None


@dataclass(frozen=True)
class KITTILidarCameraProjectedPoint:
    """One Velodyne point projected into a KITTI camera image."""

    u_px: float
    v_px: float
    depth_m: float
    intensity: float

    def as_dict(self) -> dict[str, float]:
        """Return a JSON/YAML-friendly representation."""

        return {
            "u_px": self.u_px,
            "v_px": self.v_px,
            "depth_m": self.depth_m,
            "intensity": self.intensity,
        }


@dataclass(frozen=True)
class KITTILidarCameraProjection:
    """Projected LiDAR sample for one KITTI camera/Velodyne pair."""

    status: str
    camera_path: str
    lidar_path: str
    sampled_point_count: int
    projected_point_count: int
    image_size_px: tuple[int, int] | None
    points: tuple[KITTILidarCameraProjectedPoint, ...] = ()
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON/YAML-friendly representation."""

        return {
            "status": self.status,
            "camera_path": self.camera_path,
            "lidar_path": self.lidar_path,
            "sampled_point_count": self.sampled_point_count,
            "projected_point_count": self.projected_point_count,
            "image_size_px": self.image_size_px,
            "points": [point.as_dict() for point in self.points],
            "reason": self.reason,
        }


@dataclass(frozen=True)
class KITTILidarCameraEdgeAlignment:
    """Image-edge agreement proxy for projected KITTI LiDAR points."""

    status: str
    scored_point_count: int
    edge_point_count: int
    edge_fraction: float | None
    mean_gradient: float | None
    gradient_threshold: float
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON/YAML-friendly representation."""

        return {
            "status": self.status,
            "scored_point_count": self.scored_point_count,
            "edge_point_count": self.edge_point_count,
            "edge_fraction": self.edge_fraction,
            "mean_gradient": self.mean_gradient,
            "gradient_threshold": self.gradient_threshold,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class KITTILidarCameraDepthEdgeAlignment:
    """Depth-discontinuity to image-edge agreement for projected KITTI LiDAR."""

    status: str
    discontinuity_point_count: int
    edge_point_count: int
    edge_fraction: float | None
    mean_gradient: float | None
    depth_jump_m: float
    neighbor_radius_px: float
    gradient_threshold: float
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON/YAML-friendly representation."""

        return {
            "status": self.status,
            "discontinuity_point_count": self.discontinuity_point_count,
            "edge_point_count": self.edge_point_count,
            "edge_fraction": self.edge_fraction,
            "mean_gradient": self.mean_gradient,
            "depth_jump_m": self.depth_jump_m,
            "neighbor_radius_px": self.neighbor_radius_px,
            "gradient_threshold": self.gradient_threshold,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class KITTIOXTSPacket:
    """Subset of one KITTI OXTS packet useful for calibration diagnostics."""

    lat_deg: float
    lon_deg: float
    alt_m: float
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    vn_mps: float
    ve_mps: float
    vf_mps: float
    ax_mps2: float
    ay_mps2: float
    az_mps2: float

    @property
    def horizontal_speed_mps(self) -> float:
        """Return horizontal speed from north/east velocity components."""

        return sqrt((self.vn_mps * self.vn_mps) + (self.ve_mps * self.ve_mps))

    @property
    def acceleration_norm_mps2(self) -> float:
        """Return acceleration norm from raw OXTS acceleration fields."""

        return sqrt(
            (self.ax_mps2 * self.ax_mps2)
            + (self.ay_mps2 * self.ay_mps2)
            + (self.az_mps2 * self.az_mps2)
        )


@dataclass(frozen=True)
class OXTSMotionStats:
    """Motion excitation diagnostics from KITTI OXTS files."""

    packet_count: int
    timestamp_count: int
    duration_sec: float | None = None
    mean_speed_mps: float | None = None
    max_speed_mps: float | None = None
    speed_range_mps: float | None = None
    yaw_excitation_deg: float | None = None
    pitch_excitation_deg: float | None = None
    roll_excitation_deg: float | None = None
    mean_acceleration_norm_mps2: float | None = None
    malformed_files: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        """Return a JSON/YAML-friendly representation."""

        return {
            "packet_count": self.packet_count,
            "timestamp_count": self.timestamp_count,
            "duration_sec": self.duration_sec,
            "mean_speed_mps": self.mean_speed_mps,
            "max_speed_mps": self.max_speed_mps,
            "speed_range_mps": self.speed_range_mps,
            "yaw_excitation_deg": self.yaw_excitation_deg,
            "pitch_excitation_deg": self.pitch_excitation_deg,
            "roll_excitation_deg": self.roll_excitation_deg,
            "mean_acceleration_norm_mps2": self.mean_acceleration_norm_mps2,
            "malformed_files": list(self.malformed_files),
        }


@dataclass(frozen=True)
class KITTITimestampAlignmentStats:
    """Cross-stream KITTI timestamp alignment diagnostics."""

    camera_lidar_pair_count: int = 0
    camera_lidar_mean_abs_dt_ms: float | None = None
    camera_lidar_max_abs_dt_ms: float | None = None
    lidar_oxts_pair_count: int = 0
    lidar_oxts_mean_abs_dt_ms: float | None = None
    lidar_oxts_max_abs_dt_ms: float | None = None
    camera_timestamp_count: int = 0
    lidar_timestamp_count: int = 0
    oxts_timestamp_count: int = 0

    def as_dict(self) -> dict[str, object]:
        """Return a JSON/YAML-friendly representation."""

        return {
            "camera_lidar_pair_count": self.camera_lidar_pair_count,
            "camera_lidar_mean_abs_dt_ms": self.camera_lidar_mean_abs_dt_ms,
            "camera_lidar_max_abs_dt_ms": self.camera_lidar_max_abs_dt_ms,
            "lidar_oxts_pair_count": self.lidar_oxts_pair_count,
            "lidar_oxts_mean_abs_dt_ms": self.lidar_oxts_mean_abs_dt_ms,
            "lidar_oxts_max_abs_dt_ms": self.lidar_oxts_max_abs_dt_ms,
            "camera_timestamp_count": self.camera_timestamp_count,
            "lidar_timestamp_count": self.lidar_timestamp_count,
            "oxts_timestamp_count": self.oxts_timestamp_count,
        }


VelodynePoint = tuple[float, float, float, float]


@dataclass(frozen=True)
class VelodynePointCloudStats:
    """Lightweight diagnostics for KITTI Velodyne point cloud files."""

    frame_count: int
    sampled_frame_count: int
    sampled_point_count: int
    min_points_per_frame: int | None = None
    max_points_per_frame: int | None = None
    bounds_min_m: tuple[float, float, float] | None = None
    bounds_max_m: tuple[float, float, float] | None = None
    intensity_min: float | None = None
    intensity_max: float | None = None
    intensity_mean: float | None = None
    voxel_size_m: float | None = None
    planarity_min_points_per_voxel: int | None = None
    planarity_voxel_count: int = 0
    local_planarity_mean: float | None = None
    roughness_mean_m: float | None = None
    map_sharpness_score: float | None = None
    point_to_plane_rmse_train_m: float | None = None
    point_to_plane_rmse_holdout_m: float | None = None
    malformed_files: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        """Return a JSON/YAML-friendly representation."""

        return {
            "frame_count": self.frame_count,
            "sampled_frame_count": self.sampled_frame_count,
            "sampled_point_count": self.sampled_point_count,
            "min_points_per_frame": self.min_points_per_frame,
            "max_points_per_frame": self.max_points_per_frame,
            "bounds_min_m": self.bounds_min_m,
            "bounds_max_m": self.bounds_max_m,
            "intensity_min": self.intensity_min,
            "intensity_max": self.intensity_max,
            "intensity_mean": self.intensity_mean,
            "voxel_size_m": self.voxel_size_m,
            "planarity_min_points_per_voxel": self.planarity_min_points_per_voxel,
            "planarity_voxel_count": self.planarity_voxel_count,
            "local_planarity_mean": self.local_planarity_mean,
            "roughness_mean_m": self.roughness_mean_m,
            "map_sharpness_score": self.map_sharpness_score,
            "point_to_plane_rmse_train_m": self.point_to_plane_rmse_train_m,
            "point_to_plane_rmse_holdout_m": self.point_to_plane_rmse_holdout_m,
            "malformed_files": list(self.malformed_files),
        }


@dataclass(frozen=True)
class KITTILidarWorldMapConsistencyStats:
    """LiDAR point-to-plane consistency using KITTI OXTS ego poses."""

    status: str
    frame_count: int
    train_frame_count: int
    holdout_frame_count: int
    sampled_point_count: int
    train_voxel_count: int
    train_residual_count: int
    holdout_residual_count: int
    point_to_plane_rmse_train_m: float | None = None
    point_to_plane_rmse_holdout_m: float | None = None
    point_to_plane_median_holdout_m: float | None = None
    point_to_plane_p95_holdout_m: float | None = None
    reason: str | None = None
    split_policy: str = "temporal_tail_holdout"
    train_frame_ids: tuple[str, ...] = ()
    holdout_frame_ids: tuple[str, ...] = ()
    dataset_slices: tuple[dict[str, object], ...] = field(default_factory=tuple)
    map_artifact: dict[str, object] = field(default_factory=dict)
    correspondence_artifact: dict[str, object] = field(default_factory=dict)
    leakage_validation: dict[str, object] = field(default_factory=dict)
    stability: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON/YAML-friendly representation."""

        return {
            "status": self.status,
            "frame_count": self.frame_count,
            "train_frame_count": self.train_frame_count,
            "holdout_frame_count": self.holdout_frame_count,
            "sampled_point_count": self.sampled_point_count,
            "train_voxel_count": self.train_voxel_count,
            "train_residual_count": self.train_residual_count,
            "holdout_residual_count": self.holdout_residual_count,
            "point_to_plane_rmse_train_m": self.point_to_plane_rmse_train_m,
            "point_to_plane_rmse_holdout_m": self.point_to_plane_rmse_holdout_m,
            "point_to_plane_median_holdout_m": self.point_to_plane_median_holdout_m,
            "point_to_plane_p95_holdout_m": self.point_to_plane_p95_holdout_m,
            "reason": self.reason,
            "split_policy": self.split_policy,
            "train_frame_ids": list(self.train_frame_ids),
            "holdout_frame_ids": list(self.holdout_frame_ids),
            "dataset_slices": [dict(item) for item in self.dataset_slices],
            "map_artifact": dict(self.map_artifact),
            "correspondence_artifact": dict(self.correspondence_artifact),
            "leakage_validation": dict(self.leakage_validation),
            "stability": dict(self.stability),
        }


@dataclass(frozen=True)
class _WorldMapLineage:
    split_policy: str
    train_frame_ids: tuple[str, ...]
    holdout_frame_ids: tuple[str, ...]
    dataset_slices: tuple[dict[str, object], ...]
    map_artifact: dict[str, object]
    correspondence_artifact: dict[str, object]
    leakage_validation: dict[str, object]


@dataclass(frozen=True)
class _WorldMapScore:
    train_points: list[Vector3]
    holdout_points: list[Vector3]
    train_voxel_count: int
    train_residuals: list[float]
    holdout_residuals: list[float]
    malformed_files: tuple[str, ...]

    @property
    def sampled_point_count(self) -> int:
        return len(self.train_points) + len(self.holdout_points)

    @property
    def scored(self) -> bool:
        return bool(self.train_voxel_count and self.train_residuals and self.holdout_residuals)


@dataclass
class _VoxelAccumulator:
    count: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_z: float = 0.0
    sum_xx: float = 0.0
    sum_xy: float = 0.0
    sum_xz: float = 0.0
    sum_yy: float = 0.0
    sum_yz: float = 0.0
    sum_zz: float = 0.0

    def add(self, x: float, y: float, z: float) -> None:
        self.count += 1
        self.sum_x += x
        self.sum_y += y
        self.sum_z += z
        self.sum_xx += x * x
        self.sum_xy += x * y
        self.sum_xz += x * z
        self.sum_yy += y * y
        self.sum_yz += y * z
        self.sum_zz += z * z

    def covariance(self) -> tuple[tuple[float, float, float], ...]:
        inv_count = 1.0 / self.count
        mean_x = self.sum_x * inv_count
        mean_y = self.sum_y * inv_count
        mean_z = self.sum_z * inv_count
        return (
            (
                self.sum_xx * inv_count - mean_x * mean_x,
                self.sum_xy * inv_count - mean_x * mean_y,
                self.sum_xz * inv_count - mean_x * mean_z,
            ),
            (
                self.sum_xy * inv_count - mean_x * mean_y,
                self.sum_yy * inv_count - mean_y * mean_y,
                self.sum_yz * inv_count - mean_y * mean_z,
            ),
            (
                self.sum_xz * inv_count - mean_x * mean_z,
                self.sum_yz * inv_count - mean_y * mean_z,
                self.sum_zz * inv_count - mean_z * mean_z,
            ),
        )

    def mean(self) -> Vector3:
        inv_count = 1.0 / self.count
        return (self.sum_x * inv_count, self.sum_y * inv_count, self.sum_z * inv_count)


@dataclass(frozen=True)
class _VoxelPlane:
    center_m: Vector3
    normal: Vector3


@dataclass(frozen=True)
class _VoxelPlanaritySample:
    planarity: float
    roughness_m: float
    sharpness: float


@dataclass(frozen=True)
class _VoxelPlanaritySummary:
    count: int
    planarity_mean: float | None
    roughness_mean_m: float | None
    sharpness_mean: float | None
    point_to_plane_rmse_train_m: float | None
    point_to_plane_rmse_holdout_m: float | None


@dataclass(frozen=True)
class LuminanceImage:
    """Decoded 8-bit PNG luminance image without optional image dependencies."""

    width: int
    height: int
    rows: tuple[tuple[float, ...], ...]


class KITTIRawDataset:
    """Reader for KITTI raw extracted sequence directories."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.manifest = _load_optional_manifest(self.path)

    def streams(self) -> list[StreamSummary]:
        """Return KITTI stream summaries."""

        counts = {
            "camera_left_color": _count_files(self.path / "image_02" / "data", "*.png"),
            "camera_right_color": _count_files(self.path / "image_03" / "data", "*.png"),
            "velodyne_points": _count_files(self.path / "velodyne_points" / "data", "*.bin"),
            "oxts": _count_files(self.path / "oxts" / "data", "*.txt"),
            "calibration": len(list(self.path.parent.glob("calib_*.txt"))),
        }
        if not any(counts.values()) and self.manifest is not None:
            return _manifest_streams(self.manifest)
        return [
            StreamSummary(
                "camera_left_color",
                "image",
                counts["camera_left_color"],
                sensor="camera0",
            ),
            StreamSummary(
                "camera_right_color",
                "image",
                counts["camera_right_color"],
                sensor="camera1",
            ),
            StreamSummary(
                "velodyne_points",
                "pointcloud",
                counts["velodyne_points"],
                sensor="lidar0",
            ),
            StreamSummary("oxts", "imu", counts["oxts"], sensor="imu0"),
            StreamSummary("calibration", "metadata", counts["calibration"]),
        ]

    def records(self, stream: str) -> Iterable[TimestampedRecord]:
        """Yield normalized records for common KITTI streams."""

        if stream == "camera_left_color":
            return _file_records(
                stream,
                self.path / "image_02" / "data",
                "*.png",
                read_timestamps(self.path / "image_02" / "timestamps.txt"),
            )
        if stream == "camera_right_color":
            return _file_records(
                stream,
                self.path / "image_03" / "data",
                "*.png",
                read_timestamps(self.path / "image_03" / "timestamps.txt"),
            )
        if stream == "velodyne_points":
            return _file_records(
                stream,
                self.path / "velodyne_points" / "data",
                "*.bin",
                read_timestamps(self.path / "velodyne_points" / "timestamps.txt"),
            )
        if stream == "oxts":
            return _file_records(
                stream,
                self.path / "oxts" / "data",
                "*.txt",
                read_timestamps(self.path / "oxts" / "timestamps.txt"),
            )
        if stream == "calibration":
            return _calibration_records(self.path.parent)
        return []


def read_timestamps(path: Path) -> list[KITTITimestamp]:
    """Read KITTI timestamp files into integer nanoseconds."""

    if not path.exists():
        return []
    timestamps: list[KITTITimestamp] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        timestamps.append(
            KITTITimestamp(
                index=index,
                timestamp_ns=_parse_kitti_time(stripped),
                raw=stripped,
            )
        )
    return timestamps


def read_calibration_file(path: Path) -> dict[str, list[float]]:
    """Read a KITTI calibration text file."""

    if not path.exists():
        return {}
    values: dict[str, list[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or ":" not in stripped:
            continue
        key, raw_values = stripped.split(":", 1)
        fields = raw_values.strip().split()
        try:
            values[key] = [float(value) for value in fields]
        except ValueError:
            continue
    return values


def read_oxts_packet(path: str | Path) -> KITTIOXTSPacket:
    """Read one KITTI OXTS packet file."""

    values = _read_oxts_values(Path(path))
    return _packet_from_oxts_values(values)


def summarize_oxts_motion(root: str | Path) -> OXTSMotionStats:
    """Summarize KITTI OXTS motion excitation for calibration diagnostics."""

    root_path = Path(root)
    directory = root_path / "oxts" / "data"
    files = sorted(directory.glob("*.txt")) if directory.exists() else []
    timestamps = read_timestamps(root_path / "oxts" / "timestamps.txt")
    packets: list[KITTIOXTSPacket] = []
    malformed_files: list[str] = []

    for file_path in files:
        try:
            packets.append(read_oxts_packet(file_path))
        except ValueError as exc:
            malformed_files.append(f"{file_path}: {exc}")

    if not packets:
        return OXTSMotionStats(
            packet_count=0,
            timestamp_count=len(timestamps),
            malformed_files=tuple(malformed_files),
        )

    speeds = [packet.horizontal_speed_mps for packet in packets]
    accelerations = [packet.acceleration_norm_mps2 for packet in packets]
    yaw_values = _unwrap_angles([packet.yaw_rad for packet in packets])
    pitch_values = [packet.pitch_rad for packet in packets]
    roll_values = [packet.roll_rad for packet in packets]
    return OXTSMotionStats(
        packet_count=len(packets),
        timestamp_count=len(timestamps),
        duration_sec=_duration_sec(timestamps),
        mean_speed_mps=_mean(speeds),
        max_speed_mps=max(speeds),
        speed_range_mps=max(speeds) - min(speeds),
        yaw_excitation_deg=_range_deg(yaw_values),
        pitch_excitation_deg=_range_deg(pitch_values),
        roll_excitation_deg=_range_deg(roll_values),
        mean_acceleration_norm_mps2=_mean(accelerations),
        malformed_files=tuple(malformed_files),
    )


def summarize_timestamp_alignment(root: str | Path) -> KITTITimestampAlignmentStats:
    """Summarize nearest-neighbor timestamp alignment for KITTI streams."""

    root_path = Path(root)
    camera_timestamps = read_timestamps(root_path / "image_02" / "timestamps.txt")
    lidar_timestamps = read_timestamps(root_path / "velodyne_points" / "timestamps.txt")
    oxts_timestamps = read_timestamps(root_path / "oxts" / "timestamps.txt")
    camera_lidar_deltas = _nearest_abs_deltas_ms(
        _timestamp_values(lidar_timestamps),
        _timestamp_values(camera_timestamps),
    )
    lidar_oxts_deltas = _nearest_abs_deltas_ms(
        _timestamp_values(lidar_timestamps),
        _timestamp_values(oxts_timestamps),
    )
    return KITTITimestampAlignmentStats(
        camera_lidar_pair_count=len(camera_lidar_deltas),
        camera_lidar_mean_abs_dt_ms=(
            _mean(camera_lidar_deltas) if camera_lidar_deltas else None
        ),
        camera_lidar_max_abs_dt_ms=(
            max(camera_lidar_deltas) if camera_lidar_deltas else None
        ),
        lidar_oxts_pair_count=len(lidar_oxts_deltas),
        lidar_oxts_mean_abs_dt_ms=(_mean(lidar_oxts_deltas) if lidar_oxts_deltas else None),
        lidar_oxts_max_abs_dt_ms=(max(lidar_oxts_deltas) if lidar_oxts_deltas else None),
        camera_timestamp_count=len(camera_timestamps),
        lidar_timestamp_count=len(lidar_timestamps),
        oxts_timestamp_count=len(oxts_timestamps),
    )


def find_camera_lidar_pairs(
    root: str | Path,
    *,
    camera: str = "image_02",
    max_pairs: int = 5,
) -> list[KITTICameraLidarPair]:
    """Find concrete camera image and Velodyne frame pairs in a KITTI raw sequence."""

    if max_pairs <= 0:
        return []

    root_path = Path(root)
    camera_files = _camera_files(root_path, camera)
    lidar_files = _lidar_files(root_path)
    if not camera_files or not lidar_files:
        return []

    camera_timestamps = read_timestamps(root_path / camera / "timestamps.txt")
    lidar_timestamps = read_timestamps(root_path / "velodyne_points" / "timestamps.txt")
    if camera_timestamps and lidar_timestamps:
        return _timestamped_camera_lidar_pairs(
            camera_files,
            lidar_files,
            camera_timestamps,
            lidar_timestamps,
            max_pairs,
        )

    return _same_index_camera_lidar_pairs(
        camera_files,
        lidar_files,
        camera_timestamps,
        lidar_timestamps,
        max_pairs,
    )


def read_velodyne_bin(path: str | Path) -> list[VelodynePoint]:
    """Read a KITTI Velodyne `.bin` file as `(x, y, z, intensity)` tuples."""

    return list(_iter_velodyne_bin(Path(path)))


def read_camera_projection_model(
    root: str | Path,
    *,
    camera: str = "image_02",
) -> KITTICameraProjectionModel | None:
    """Read KITTI rectified camera projection for a camera stream."""

    root_path = Path(root)
    calibration = read_calibration_file(_find_calibration_file(root_path, "calib_cam_to_cam.txt"))
    suffix = _camera_suffix(camera)
    projection = _first_calibration_value(
        calibration,
        (f"P_rect_{suffix}", f"P_rect_{int(suffix)}", f"P{int(suffix)}"),
    )
    if projection is None or len(projection) != 12:
        return None
    rectification = _first_calibration_value(calibration, ("R_rect_00", "R0_rect"))
    if rectification is None or len(rectification) != 9:
        rectification_tuple = _identity_matrix_3x3()
    else:
        rectification_tuple = tuple(rectification)
    image_size = _read_rectified_image_size(calibration, suffix)
    return KITTICameraProjectionModel(
        camera=camera,
        projection_matrix=tuple(projection),
        rectification_matrix=rectification_tuple,
        image_size_px=image_size,
    )


def project_velodyne_to_camera(
    root: str | Path,
    *,
    camera_path: str | Path,
    lidar_path: str | Path,
    camera: str = "image_02",
    max_points: int = 800,
    t_camera_lidar_override: SE3 | None = None,
) -> KITTILidarCameraProjection:
    """Project a deterministic sample of one KITTI Velodyne frame into a camera image."""

    camera_file = Path(camera_path)
    lidar_file = Path(lidar_path)
    model = read_camera_projection_model(root, camera=camera)
    if model is None:
        return _projection_failure(camera_file, lidar_file, "KITTI camera projection is missing")
    t_camera_lidar = t_camera_lidar_override or read_velodyne_to_camera_transform(root)
    if t_camera_lidar is None:
        return _projection_failure(
            camera_file,
            lidar_file,
            "KITTI Velodyne-camera transform is missing",
        )
    if max_points <= 0:
        return _projection_failure(camera_file, lidar_file, "max_points must be positive")

    try:
        lidar_points = read_velodyne_bin(lidar_file)
    except ValueError as exc:
        return _projection_failure(camera_file, lidar_file, str(exc))

    image_size = _png_size(camera_file) or model.image_size_px
    sampled_points = _sample_velodyne_points(lidar_points, max_points)
    projected_points: list[KITTILidarCameraProjectedPoint] = []
    for x, y, z, intensity in sampled_points:
        point_camera = t_camera_lidar.transform_point((x, y, z))
        point_rectified = _mat3_transform(model.rectification_matrix, point_camera)
        projected = _project_rectified_point(model.projection_matrix, point_rectified)
        if projected is None:
            continue
        u_px, v_px, depth_m = projected
        if image_size is not None and not _inside_image(u_px, v_px, image_size):
            continue
        projected_points.append(
            KITTILidarCameraProjectedPoint(
                u_px=u_px,
                v_px=v_px,
                depth_m=depth_m,
                intensity=float(intensity),
            )
        )

    status = "projected" if projected_points else "empty"
    reason = None if projected_points else "no sampled LiDAR points projected into the camera frame"
    return KITTILidarCameraProjection(
        status=status,
        camera_path=str(camera_file),
        lidar_path=str(lidar_file),
        sampled_point_count=len(sampled_points),
        projected_point_count=len(projected_points),
        image_size_px=image_size,
        points=tuple(projected_points),
        reason=reason,
    )


def score_lidar_camera_edge_alignment(
    projection: KITTILidarCameraProjection,
    *,
    gradient_threshold: float = 30.0,
    window_px: int = 1,
) -> KITTILidarCameraEdgeAlignment:
    """Score how often projected LiDAR points land near image intensity edges."""

    if projection.status != "projected" or not projection.points:
        return _edge_alignment_failure(
            gradient_threshold,
            projection.reason or "no projected LiDAR points are available",
        )

    image = read_png_luminance(Path(projection.camera_path))
    if image is None:
        return _edge_alignment_failure(
            gradient_threshold,
            "camera image cannot be decoded as supported PNG",
        )

    gradients: list[float] = []
    edge_count = 0
    radius = max(0, window_px)
    for point in projection.points:
        x = round(point.u_px)
        y = round(point.v_px)
        gradient = _local_gradient_max(image, x, y, radius)
        if gradient is None:
            continue
        gradients.append(gradient)
        if gradient >= gradient_threshold:
            edge_count += 1

    scored_count = len(gradients)
    if scored_count == 0:
        return _edge_alignment_failure(
            gradient_threshold,
            "projected points are too close to image borders for edge scoring",
        )

    return KITTILidarCameraEdgeAlignment(
        status="scored",
        scored_point_count=scored_count,
        edge_point_count=edge_count,
        edge_fraction=edge_count / scored_count,
        mean_gradient=_mean(gradients),
        gradient_threshold=gradient_threshold,
    )


def score_lidar_camera_depth_edge_alignment(
    projection: KITTILidarCameraProjection,
    *,
    depth_jump_m: float = 1.0,
    neighbor_radius_px: float = 6.0,
    gradient_threshold: float = 30.0,
    window_px: int = 1,
) -> KITTILidarCameraDepthEdgeAlignment:
    """Score projected LiDAR depth jumps against image intensity edges."""

    if projection.status != "projected" or not projection.points:
        return _depth_edge_alignment_failure(
            depth_jump_m,
            neighbor_radius_px,
            gradient_threshold,
            projection.reason or "no projected LiDAR points are available",
        )

    image = read_png_luminance(Path(projection.camera_path))
    if image is None:
        return _depth_edge_alignment_failure(
            depth_jump_m,
            neighbor_radius_px,
            gradient_threshold,
            "camera image cannot be decoded as supported PNG",
        )

    discontinuity_points = _depth_discontinuity_points(
        projection.points,
        depth_jump_m=depth_jump_m,
        neighbor_radius_px=neighbor_radius_px,
    )
    if not discontinuity_points:
        return _depth_edge_alignment_failure(
            depth_jump_m,
            neighbor_radius_px,
            gradient_threshold,
            "no projected LiDAR depth discontinuities were found",
        )

    gradients: list[float] = []
    edge_count = 0
    radius = max(0, window_px)
    for point in discontinuity_points:
        gradient = _local_gradient_max(image, round(point.u_px), round(point.v_px), radius)
        if gradient is None:
            continue
        gradients.append(gradient)
        if gradient >= gradient_threshold:
            edge_count += 1

    scored_count = len(gradients)
    if scored_count == 0:
        return _depth_edge_alignment_failure(
            depth_jump_m,
            neighbor_radius_px,
            gradient_threshold,
            "depth discontinuity points are too close to image borders for edge scoring",
        )

    return KITTILidarCameraDepthEdgeAlignment(
        status="scored",
        discontinuity_point_count=scored_count,
        edge_point_count=edge_count,
        edge_fraction=edge_count / scored_count,
        mean_gradient=_mean(gradients),
        depth_jump_m=depth_jump_m,
        neighbor_radius_px=neighbor_radius_px,
        gradient_threshold=gradient_threshold,
    )


def summarize_velodyne_points(
    root: str | Path,
    *,
    sample_limit: int = 3,
    voxel_size_m: float = 4.0,
    planarity_min_points_per_voxel: int = 6,
    point_transform: SE3 | None = None,
) -> VelodynePointCloudStats:
    """Summarize KITTI Velodyne point cloud coverage without loading every frame."""

    root_path = Path(root)
    directory = root_path / "velodyne_points" / "data"
    files = sorted(directory.glob("*.bin")) if directory.exists() else []
    sampled_files = files[: max(sample_limit, 0)]
    point_counts: list[int] = []
    malformed_files: list[str] = []

    min_x = min_y = min_z = float("inf")
    max_x = max_y = max_z = float("-inf")
    intensity_min = float("inf")
    intensity_max = float("-inf")
    intensity_sum = 0.0
    sampled_point_count = 0
    voxels: dict[tuple[int, int, int], _VoxelAccumulator] = {}

    for file_path in sampled_files:
        point_count = 0
        try:
            for x, y, z, intensity in _iter_velodyne_bin(file_path):
                if point_transform is not None:
                    x, y, z = point_transform.transform_point((x, y, z))
                point_count += 1
                sampled_point_count += 1
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                min_z = min(min_z, z)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
                max_z = max(max_z, z)
                intensity_min = min(intensity_min, intensity)
                intensity_max = max(intensity_max, intensity)
                intensity_sum += intensity
                voxel = voxels.setdefault(
                    _voxel_key(x, y, z, voxel_size_m),
                    _VoxelAccumulator(),
                )
                voxel.add(x, y, z)
        except ValueError as exc:
            malformed_files.append(f"{file_path}: {exc}")
            continue
        point_counts.append(point_count)

    planarity_summary = _summarize_voxel_planarity(voxels, planarity_min_points_per_voxel)
    bounds_min = (min_x, min_y, min_z) if sampled_point_count else None
    bounds_max = (max_x, max_y, max_z) if sampled_point_count else None
    return VelodynePointCloudStats(
        frame_count=len(files),
        sampled_frame_count=len(point_counts),
        sampled_point_count=sampled_point_count,
        min_points_per_frame=min(point_counts) if point_counts else None,
        max_points_per_frame=max(point_counts) if point_counts else None,
        bounds_min_m=bounds_min,
        bounds_max_m=bounds_max,
        intensity_min=intensity_min if sampled_point_count else None,
        intensity_max=intensity_max if sampled_point_count else None,
        intensity_mean=(intensity_sum / sampled_point_count) if sampled_point_count else None,
        voxel_size_m=voxel_size_m,
        planarity_min_points_per_voxel=planarity_min_points_per_voxel,
        planarity_voxel_count=planarity_summary.count,
        local_planarity_mean=planarity_summary.planarity_mean,
        roughness_mean_m=planarity_summary.roughness_mean_m,
        map_sharpness_score=planarity_summary.sharpness_mean,
        point_to_plane_rmse_train_m=planarity_summary.point_to_plane_rmse_train_m,
        point_to_plane_rmse_holdout_m=planarity_summary.point_to_plane_rmse_holdout_m,
        malformed_files=tuple(malformed_files),
    )


def summarize_lidar_world_map_consistency(
    root: str | Path,
    *,
    sample_limit: int = 3,
    max_points_per_frame: int = 800,
    voxel_size_m: float = 4.0,
    min_points_per_voxel: int = 6,
    t_ego_lidar: SE3 | None = None,
) -> KITTILidarWorldMapConsistencyStats:
    """Evaluate LiDAR map consistency by transforming KITTI frames with OXTS poses.

    This is an alpha fixed-vehicle LiDAR diagnostic: OXTS poses are treated as
    fixed ego poses, train LiDAR frames build a local voxel-plane map, and
    holdout LiDAR frames are scored by point-to-plane residuals.
    """

    root_path = Path(root)
    paired_frames = _paired_lidar_oxts_frames(root_path, sample_limit=sample_limit)
    if len(paired_frames) < 2:
        return KITTILidarWorldMapConsistencyStats(
            status="unavailable",
            frame_count=len(paired_frames),
            train_frame_count=0,
            holdout_frame_count=0,
            sampled_point_count=0,
            train_voxel_count=0,
            train_residual_count=0,
            holdout_residual_count=0,
            reason="at least two LiDAR/OXTS frame pairs are required",
        )

    holdout_count = max(1, round(len(paired_frames) * 0.2))
    train_frames = paired_frames[:-holdout_count] or paired_frames[:1]
    holdout_frames = paired_frames[-holdout_count:]
    lineage = _lidar_world_map_lineage(train_frames, holdout_frames)
    t_sensor = t_ego_lidar or SE3.identity()
    score = _score_lidar_world_map_frames(
        train_frames=train_frames,
        holdout_frames=holdout_frames,
        t_sensor=t_sensor,
        max_points_per_frame=max_points_per_frame,
        voxel_size_m=voxel_size_m,
        min_points_per_voxel=min_points_per_voxel,
    )
    stability = _lidar_world_map_window_stability(
        paired_frames,
        t_sensor=t_sensor,
        max_points_per_frame=max_points_per_frame,
        voxel_size_m=voxel_size_m,
        min_points_per_voxel=min_points_per_voxel,
    )

    if not score.scored:
        reason = "not enough shared train/holdout local planes"
        if score.malformed_files:
            reason = f"{reason}; malformed files: {len(score.malformed_files)}"
        return KITTILidarWorldMapConsistencyStats(
            status="unavailable",
            frame_count=len(paired_frames),
            train_frame_count=len(train_frames),
            holdout_frame_count=len(holdout_frames),
            sampled_point_count=score.sampled_point_count,
            train_voxel_count=score.train_voxel_count,
            train_residual_count=len(score.train_residuals),
            holdout_residual_count=len(score.holdout_residuals),
            reason=reason,
            split_policy=lineage.split_policy,
            train_frame_ids=lineage.train_frame_ids,
            holdout_frame_ids=lineage.holdout_frame_ids,
            dataset_slices=lineage.dataset_slices,
            map_artifact=lineage.map_artifact,
            correspondence_artifact=lineage.correspondence_artifact,
            leakage_validation=lineage.leakage_validation,
            stability=stability,
        )

    return KITTILidarWorldMapConsistencyStats(
        status="scored",
        frame_count=len(paired_frames),
        train_frame_count=len(train_frames),
        holdout_frame_count=len(holdout_frames),
        sampled_point_count=score.sampled_point_count,
        train_voxel_count=score.train_voxel_count,
        train_residual_count=len(score.train_residuals),
        holdout_residual_count=len(score.holdout_residuals),
        point_to_plane_rmse_train_m=_rmse(score.train_residuals),
        point_to_plane_rmse_holdout_m=_rmse(score.holdout_residuals),
        point_to_plane_median_holdout_m=_percentile(score.holdout_residuals, 0.5),
        point_to_plane_p95_holdout_m=_percentile(score.holdout_residuals, 0.95),
        split_policy=lineage.split_policy,
        train_frame_ids=lineage.train_frame_ids,
        holdout_frame_ids=lineage.holdout_frame_ids,
        dataset_slices=lineage.dataset_slices,
        map_artifact=lineage.map_artifact,
        correspondence_artifact=lineage.correspondence_artifact,
        leakage_validation=lineage.leakage_validation,
        stability=stability,
    )


def _lidar_world_map_lineage(
    train_frames: list[tuple[Path, SE3]],
    holdout_frames: list[tuple[Path, SE3]],
) -> _WorldMapLineage:
    train_frame_ids = tuple(file_path.stem for file_path, _pose in train_frames)
    holdout_frame_ids = tuple(file_path.stem for file_path, _pose in holdout_frames)
    selection_policy = "temporal_tail_holdout"
    train_slice = DatasetSliceArtifact(
        artifact_id=stable_artifact_id("slice_train", ("kitti_world_map", *train_frame_ids)),
        role="train",
        frame_ids=train_frame_ids,
        selection_policy=selection_policy,
    )
    holdout_slice = DatasetSliceArtifact(
        artifact_id=stable_artifact_id(
            "slice_holdout",
            ("kitti_world_map", *holdout_frame_ids),
        ),
        role="holdout",
        frame_ids=holdout_frame_ids,
        selection_policy=selection_policy,
    )
    map_artifact = MapArtifact(
        artifact_id=stable_artifact_id("map_voxel_plane", train_frame_ids),
        map_type="voxel_plane_map",
        source_slice_id=train_slice.artifact_id,
        frame_ids=train_frame_ids,
        dependency_ids=(train_slice.artifact_id,),
    )
    correspondence_artifact = CorrespondenceArtifact(
        artifact_id=stable_artifact_id(
            "corr_point_to_plane",
            (*train_frame_ids, "query", *holdout_frame_ids),
        ),
        correspondence_type="point_to_voxel_plane",
        source_map_id=map_artifact.artifact_id,
        query_slice_id=holdout_slice.artifact_id,
        query_frame_ids=holdout_frame_ids,
        dependency_ids=(map_artifact.artifact_id, holdout_slice.artifact_id),
    )
    leakage_validation = validate_no_frame_overlap(
        map_frame_ids=train_frame_ids,
        query_frame_ids=holdout_frame_ids,
        checked_dependency_ids=(
            train_slice.artifact_id,
            holdout_slice.artifact_id,
            map_artifact.artifact_id,
            correspondence_artifact.artifact_id,
        ),
    )
    return _WorldMapLineage(
        split_policy=selection_policy,
        train_frame_ids=train_frame_ids,
        holdout_frame_ids=holdout_frame_ids,
        dataset_slices=(train_slice.as_dict(), holdout_slice.as_dict()),
        map_artifact=map_artifact.as_dict(),
        correspondence_artifact=correspondence_artifact.as_dict(),
        leakage_validation=leakage_validation.as_dict(),
    )


def _score_lidar_world_map_frames(
    *,
    train_frames: list[tuple[Path, SE3]],
    holdout_frames: list[tuple[Path, SE3]],
    t_sensor: SE3,
    max_points_per_frame: int,
    voxel_size_m: float,
    min_points_per_voxel: int,
) -> _WorldMapScore:
    train_voxels: dict[tuple[int, int, int], _VoxelAccumulator] = {}
    train_points: list[Vector3] = []
    holdout_points: list[Vector3] = []
    malformed_files: list[str] = []

    for file_path, t_world_ego in train_frames:
        try:
            lidar_points = read_velodyne_bin(file_path)
        except ValueError as exc:
            malformed_files.append(f"{file_path}: {exc}")
            continue
        t_world_lidar = t_world_ego.compose(t_sensor)
        for x, y, z, _intensity in _sample_velodyne_points(lidar_points, max_points_per_frame):
            point_world = t_world_lidar.transform_point((x, y, z))
            train_points.append(point_world)
            voxel = train_voxels.setdefault(
                _voxel_key(point_world[0], point_world[1], point_world[2], voxel_size_m),
                _VoxelAccumulator(),
            )
            voxel.add(point_world[0], point_world[1], point_world[2])

    for file_path, t_world_ego in holdout_frames:
        try:
            lidar_points = read_velodyne_bin(file_path)
        except ValueError as exc:
            malformed_files.append(f"{file_path}: {exc}")
            continue
        t_world_lidar = t_world_ego.compose(t_sensor)
        for x, y, z, _intensity in _sample_velodyne_points(lidar_points, max_points_per_frame):
            holdout_points.append(t_world_lidar.transform_point((x, y, z)))

    planes = _voxel_plane_map(train_voxels, min_points_per_voxel)
    return _WorldMapScore(
        train_points=train_points,
        holdout_points=holdout_points,
        train_voxel_count=len(planes),
        train_residuals=_point_to_plane_residuals(
            train_points,
            planes,
            voxel_size_m=voxel_size_m,
        ),
        holdout_residuals=_point_to_plane_residuals(
            holdout_points,
            planes,
            voxel_size_m=voxel_size_m,
        ),
        malformed_files=tuple(malformed_files),
    )


def _lidar_world_map_window_stability(
    paired_frames: list[tuple[Path, SE3]],
    *,
    t_sensor: SE3,
    max_points_per_frame: int,
    voxel_size_m: float,
    min_points_per_voxel: int,
) -> dict[str, object]:
    if len(paired_frames) < 2:
        return {
            "status": "unavailable",
            "window_count": 0,
            "scored_window_count": 0,
            "reason": "at least two frame blocks are required",
            "windows": [],
        }

    windows: list[dict[str, object]] = []
    holdout_rmse_values: list[float] = []
    for holdout_index in range(1, len(paired_frames)):
        train_frames = paired_frames[:holdout_index]
        holdout_frames = [paired_frames[holdout_index]]
        score = _score_lidar_world_map_frames(
            train_frames=train_frames,
            holdout_frames=holdout_frames,
            t_sensor=t_sensor,
            max_points_per_frame=max_points_per_frame,
            voxel_size_m=voxel_size_m,
            min_points_per_voxel=min_points_per_voxel,
        )
        holdout_rmse = _rmse(score.holdout_residuals) if score.scored else None
        if holdout_rmse is not None:
            holdout_rmse_values.append(holdout_rmse)
        windows.append(
            {
                "window_id": f"temporal_prefix_holdout_{holdout_index:04d}",
                "train_frame_ids": [file_path.stem for file_path, _pose in train_frames],
                "holdout_frame_ids": [holdout_frames[0][0].stem],
                "train_voxel_count": score.train_voxel_count,
                "holdout_residual_count": len(score.holdout_residuals),
                "holdout_rmse_m": holdout_rmse,
                "status": "scored" if score.scored else "unavailable",
            }
        )

    if not holdout_rmse_values:
        return {
            "status": "unavailable",
            "window_count": len(windows),
            "scored_window_count": 0,
            "reason": "no temporal holdout windows had shared local planes",
            "windows": windows,
        }

    return {
        "status": "scored" if len(holdout_rmse_values) >= 2 else "limited",
        "window_count": len(windows),
        "scored_window_count": len(holdout_rmse_values),
        "holdout_rmse_mean_m": _mean(holdout_rmse_values),
        "holdout_rmse_min_m": min(holdout_rmse_values),
        "holdout_rmse_max_m": max(holdout_rmse_values),
        "holdout_rmse_spread_m": max(holdout_rmse_values) - min(holdout_rmse_values),
        "reason": (
            "temporal block stability from prefix-train/single-frame-holdout windows"
        ),
        "windows": windows,
    }


def read_velodyne_to_camera_transform(root: str | Path) -> SE3 | None:
    """Read KITTI `calib_velo_to_cam.txt` as `T_camera_lidar`."""

    calibration = read_calibration_file(
        _find_calibration_file(Path(root), "calib_velo_to_cam.txt")
    )
    rotation = calibration.get("R")
    translation = calibration.get("T")
    if rotation is None or translation is None:
        return None
    return SE3(
        translation_m=(translation[0], translation[1], translation[2]),
        rotation_quat_xyzw=quaternion_xyzw_from_rotation_matrix(rotation),
    )


def read_kitti_initial_transforms(root: str | Path) -> dict[str, SE3]:
    """Read available KITTI calibration transforms in Calibrex convention."""

    root_path = Path(root)
    transforms: dict[str, SE3] = {}
    t_camera_lidar = read_velodyne_to_camera_transform(root_path)
    if t_camera_lidar is not None:
        transforms["T_camera0_lidar0"] = t_camera_lidar
        transforms["T_lidar0_camera0"] = t_camera_lidar.inverse()
    return transforms


def _projection_failure(
    camera_path: Path,
    lidar_path: Path,
    reason: str,
) -> KITTILidarCameraProjection:
    return KITTILidarCameraProjection(
        status="unavailable",
        camera_path=str(camera_path),
        lidar_path=str(lidar_path),
        sampled_point_count=0,
        projected_point_count=0,
        image_size_px=_png_size(camera_path),
        reason=reason,
    )


def _edge_alignment_failure(
    gradient_threshold: float,
    reason: str,
) -> KITTILidarCameraEdgeAlignment:
    return KITTILidarCameraEdgeAlignment(
        status="unavailable",
        scored_point_count=0,
        edge_point_count=0,
        edge_fraction=None,
        mean_gradient=None,
        gradient_threshold=gradient_threshold,
        reason=reason,
    )


def _depth_edge_alignment_failure(
    depth_jump_m: float,
    neighbor_radius_px: float,
    gradient_threshold: float,
    reason: str,
) -> KITTILidarCameraDepthEdgeAlignment:
    return KITTILidarCameraDepthEdgeAlignment(
        status="unavailable",
        discontinuity_point_count=0,
        edge_point_count=0,
        edge_fraction=None,
        mean_gradient=None,
        depth_jump_m=depth_jump_m,
        neighbor_radius_px=neighbor_radius_px,
        gradient_threshold=gradient_threshold,
        reason=reason,
    )


def _camera_suffix(camera: str) -> str:
    if "_" in camera:
        raw_suffix = camera.rsplit("_", 1)[-1]
        if raw_suffix.isdigit():
            return raw_suffix.zfill(2)
    digits = "".join(character for character in camera if character.isdigit())
    if digits:
        return digits[-2:].zfill(2)
    return "02"


def _first_calibration_value(
    calibration: dict[str, list[float]],
    keys: tuple[str, ...],
) -> list[float] | None:
    for key in keys:
        value = calibration.get(key)
        if value is not None:
            return value
    return None


def _read_rectified_image_size(
    calibration: dict[str, list[float]],
    suffix: str,
) -> tuple[int, int] | None:
    values = _first_calibration_value(
        calibration,
        (f"S_rect_{suffix}", f"S_rect_{int(suffix)}", f"S{int(suffix)}"),
    )
    if values is None or len(values) < 2:
        return None
    width = round(values[0])
    height = round(values[1])
    if width <= 0 or height <= 0:
        return None
    return (width, height)


def _png_size(path: Path) -> tuple[int, int] | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            header = handle.read(24)
    except OSError:
        return None
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    if width <= 0 or height <= 0:
        return None
    return (width, height)


def read_png_luminance(path: str | Path) -> LuminanceImage | None:
    """Decode a non-interlaced 8-bit PNG into luminance rows.

    Grayscale, grayscale-alpha, RGB, and RGBA images are supported using only
    the Python standard library.  ``None`` denotes an unreadable or unsupported
    PNG encoding.
    """

    image_path = Path(path)
    try:
        data = image_path.read_bytes()
    except OSError:
        return None
    decoded = _decode_png_luminance(data)
    return decoded


def _decode_png_luminance(data: bytes) -> LuminanceImage | None:
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    offset = 8
    width = height = bit_depth = color_type = interlace = None
    idat_chunks: list[bytes] = []
    while offset + 8 <= len(data):
        length = int.from_bytes(data[offset : offset + 4], "big")
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data_start = offset + 8
        chunk_data_end = chunk_data_start + length
        if chunk_data_end + 4 > len(data):
            return None
        chunk_data = data[chunk_data_start:chunk_data_end]
        offset = chunk_data_end + 4
        if chunk_type == b"IHDR":
            if len(chunk_data) != 13:
                return None
            width = int.from_bytes(chunk_data[0:4], "big")
            height = int.from_bytes(chunk_data[4:8], "big")
            bit_depth = chunk_data[8]
            color_type = chunk_data[9]
            interlace = chunk_data[12]
        elif chunk_type == b"IDAT":
            idat_chunks.append(chunk_data)
        elif chunk_type == b"IEND":
            break
    if (
        width is None
        or height is None
        or bit_depth != 8
        or color_type not in {0, 2, 4, 6}
        or interlace != 0
        or width <= 0
        or height <= 0
        or not idat_chunks
    ):
        return None
    channels = _png_channels(color_type)
    try:
        raw = zlib.decompress(b"".join(idat_chunks))
    except zlib.error:
        return None
    return _unfilter_png_luminance(raw, width, height, channels)


def _unfilter_png_luminance(
    raw: bytes,
    width: int,
    height: int,
    channels: int,
) -> LuminanceImage | None:
    row_length = width * channels
    expected_length = height * (row_length + 1)
    if len(raw) < expected_length:
        return None
    previous = bytearray(row_length)
    rows: list[tuple[float, ...]] = []
    cursor = 0
    for _row_index in range(height):
        filter_type = raw[cursor]
        cursor += 1
        scanline = bytearray(raw[cursor : cursor + row_length])
        cursor += row_length
        if not _apply_png_filter(scanline, previous, channels, filter_type):
            return None
        rows.append(_scanline_luminance(scanline, width, channels))
        previous = scanline
    return LuminanceImage(width=width, height=height, rows=tuple(rows))


def _apply_png_filter(
    scanline: bytearray,
    previous: bytearray,
    bytes_per_pixel: int,
    filter_type: int,
) -> bool:
    for index, value in enumerate(scanline):
        left = scanline[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
        up = previous[index] if previous else 0
        up_left = previous[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
        if filter_type == 0:
            predictor = 0
        elif filter_type == 1:
            predictor = left
        elif filter_type == 2:
            predictor = up
        elif filter_type == 3:
            predictor = (left + up) // 2
        elif filter_type == 4:
            predictor = _paeth_predictor(left, up, up_left)
        else:
            return False
        scanline[index] = (value + predictor) & 0xFF
    return True


def _scanline_luminance(
    scanline: bytearray,
    width: int,
    channels: int,
) -> tuple[float, ...]:
    values: list[float] = []
    for x in range(width):
        offset = x * channels
        if channels == 1 or channels == 2:
            values.append(float(scanline[offset]))
        else:
            red = scanline[offset]
            green = scanline[offset + 1]
            blue = scanline[offset + 2]
            values.append((0.299 * red) + (0.587 * green) + (0.114 * blue))
    return tuple(values)


def _png_channels(color_type: int) -> int:
    if color_type == 0:
        return 1
    if color_type == 2:
        return 3
    if color_type == 4:
        return 2
    return 4


def _paeth_predictor(left: int, up: int, up_left: int) -> int:
    estimate = left + up - up_left
    distance_left = abs(estimate - left)
    distance_up = abs(estimate - up)
    distance_up_left = abs(estimate - up_left)
    if distance_left <= distance_up and distance_left <= distance_up_left:
        return left
    if distance_up <= distance_up_left:
        return up
    return up_left


def _local_gradient_max(
    image: LuminanceImage,
    x: int,
    y: int,
    radius: int,
) -> float | None:
    x_min = max(1, x - radius)
    x_max = min(image.width - 2, x + radius)
    y_min = max(1, y - radius)
    y_max = min(image.height - 2, y + radius)
    if x_min > x_max or y_min > y_max:
        return None
    best = 0.0
    for sample_y in range(y_min, y_max + 1):
        row = image.rows[sample_y]
        previous_row = image.rows[sample_y - 1]
        next_row = image.rows[sample_y + 1]
        for sample_x in range(x_min, x_max + 1):
            dx = row[sample_x + 1] - row[sample_x - 1]
            dy = next_row[sample_x] - previous_row[sample_x]
            best = max(best, sqrt((dx * dx) + (dy * dy)))
    return best


def _depth_discontinuity_points(
    points: tuple[KITTILidarCameraProjectedPoint, ...],
    *,
    depth_jump_m: float,
    neighbor_radius_px: float,
) -> tuple[KITTILidarCameraProjectedPoint, ...]:
    radius_squared = neighbor_radius_px * neighbor_radius_px
    discontinuities: list[KITTILidarCameraProjectedPoint] = []
    for index, point in enumerate(points):
        if _has_depth_jump_neighbor(
            point,
            points,
            skip_index=index,
            radius_squared=radius_squared,
            depth_jump_m=depth_jump_m,
        ):
            discontinuities.append(point)
    return tuple(discontinuities)


def _has_depth_jump_neighbor(
    point: KITTILidarCameraProjectedPoint,
    points: tuple[KITTILidarCameraProjectedPoint, ...],
    *,
    skip_index: int,
    radius_squared: float,
    depth_jump_m: float,
) -> bool:
    for candidate_index, candidate in enumerate(points):
        if candidate_index == skip_index:
            continue
        dx = candidate.u_px - point.u_px
        dy = candidate.v_px - point.v_px
        if (dx * dx) + (dy * dy) > radius_squared:
            continue
        if abs(candidate.depth_m - point.depth_m) >= depth_jump_m:
            return True
    return False


def _sample_velodyne_points(
    points: list[VelodynePoint],
    max_points: int,
) -> list[VelodynePoint]:
    if len(points) <= max_points:
        return points
    stride = max(1, len(points) // max_points)
    return points[::stride][:max_points]


def _mat3_transform(matrix: tuple[float, ...], point: tuple[float, float, float]) -> Vector3:
    x, y, z = point
    return (
        (matrix[0] * x) + (matrix[1] * y) + (matrix[2] * z),
        (matrix[3] * x) + (matrix[4] * y) + (matrix[5] * z),
        (matrix[6] * x) + (matrix[7] * y) + (matrix[8] * z),
    )


def _project_rectified_point(
    projection: tuple[float, ...],
    point: tuple[float, float, float],
) -> tuple[float, float, float] | None:
    x, y, z = point
    if z <= 1.0e-9:
        return None
    u_num = (projection[0] * x) + (projection[1] * y) + (projection[2] * z) + projection[3]
    v_num = (projection[4] * x) + (projection[5] * y) + (projection[6] * z) + projection[7]
    w = (projection[8] * x) + (projection[9] * y) + (projection[10] * z) + projection[11]
    if w <= 1.0e-9:
        return None
    return (u_num / w, v_num / w, z)


def _inside_image(u_px: float, v_px: float, image_size: tuple[int, int]) -> bool:
    width, height = image_size
    return 0.0 <= u_px < float(width) and 0.0 <= v_px < float(height)


def _identity_matrix_3x3() -> tuple[float, ...]:
    return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def _find_calibration_file(root: Path, filename: str) -> Path:
    """Find KITTI calibration files from either date or extracted sequence roots."""

    direct = root / filename
    if direct.exists():
        return direct
    parent = root.parent / filename
    if parent.exists():
        return parent
    return direct


def _parse_kitti_time(value: str) -> int:
    date_part, time_part = value.split(" ", 1)
    if "." in time_part:
        clock_part, fractional_part = time_part.split(".", 1)
    else:
        clock_part, fractional_part = time_part, ""
    dt = datetime.fromisoformat(f"{date_part}T{clock_part}").replace(tzinfo=timezone.utc)
    fractional_ns = int((fractional_part + "000000000")[:9])
    return int(dt.timestamp()) * 1_000_000_000 + fractional_ns


def _read_oxts_values(path: Path) -> list[float]:
    raw = path.read_text(encoding="utf-8").strip().split()
    if len(raw) < 15:
        raise ValueError("OXTS packet must contain at least 15 numeric fields")
    try:
        return [float(value) for value in raw]
    except ValueError as exc:
        raise ValueError("OXTS packet contains non-numeric fields") from exc


def _packet_from_oxts_values(values: list[float]) -> KITTIOXTSPacket:
    return KITTIOXTSPacket(
        lat_deg=values[0],
        lon_deg=values[1],
        alt_m=values[2],
        roll_rad=values[3],
        pitch_rad=values[4],
        yaw_rad=values[5],
        vn_mps=values[6],
        ve_mps=values[7],
        vf_mps=values[8],
        ax_mps2=values[11],
        ay_mps2=values[12],
        az_mps2=values[13],
    )


def _duration_sec(timestamps: list[KITTITimestamp]) -> float | None:
    if len(timestamps) < 2:
        return None
    return (timestamps[-1].timestamp_ns - timestamps[0].timestamp_ns) / 1_000_000_000.0


def _timestamp_values(timestamps: list[KITTITimestamp]) -> list[int]:
    return [timestamp.timestamp_ns for timestamp in timestamps]


def _camera_files(root: Path, camera: str) -> list[Path]:
    directory = root / camera / "data"
    return sorted(directory.glob("*.png")) if directory.exists() else []


def _lidar_files(root: Path) -> list[Path]:
    directory = root / "velodyne_points" / "data"
    return sorted(directory.glob("*.bin")) if directory.exists() else []


def _timestamped_camera_lidar_pairs(
    camera_files: list[Path],
    lidar_files: list[Path],
    camera_timestamps: list[KITTITimestamp],
    lidar_timestamps: list[KITTITimestamp],
    max_pairs: int,
) -> list[KITTICameraLidarPair]:
    lidar_values = _timestamp_values(lidar_timestamps)
    pairs: list[KITTICameraLidarPair] = []
    for camera_timestamp in camera_timestamps:
        if len(pairs) >= max_pairs:
            break
        if camera_timestamp.index >= len(camera_files):
            continue
        lidar_timestamp_index = _nearest_timestamp_index(
            lidar_values,
            camera_timestamp.timestamp_ns,
        )
        if lidar_timestamp_index is None:
            continue
        lidar_timestamp = lidar_timestamps[lidar_timestamp_index]
        if lidar_timestamp.index >= len(lidar_files):
            continue
        pairs.append(
            _camera_lidar_pair(
                camera_index=camera_timestamp.index,
                lidar_index=lidar_timestamp.index,
                camera_timestamp_ns=camera_timestamp.timestamp_ns,
                lidar_timestamp_ns=lidar_timestamp.timestamp_ns,
                camera_path=camera_files[camera_timestamp.index],
                lidar_path=lidar_files[lidar_timestamp.index],
            )
        )
    return pairs


def _same_index_camera_lidar_pairs(
    camera_files: list[Path],
    lidar_files: list[Path],
    camera_timestamps: list[KITTITimestamp],
    lidar_timestamps: list[KITTITimestamp],
    max_pairs: int,
) -> list[KITTICameraLidarPair]:
    camera_timestamp_by_index = _timestamp_by_index(camera_timestamps)
    lidar_timestamp_by_index = _timestamp_by_index(lidar_timestamps)
    pair_count = min(len(camera_files), len(lidar_files), max_pairs)
    return [
        _camera_lidar_pair(
            camera_index=index,
            lidar_index=index,
            camera_timestamp_ns=camera_timestamp_by_index.get(index),
            lidar_timestamp_ns=lidar_timestamp_by_index.get(index),
            camera_path=camera_files[index],
            lidar_path=lidar_files[index],
        )
        for index in range(pair_count)
    ]


def _camera_lidar_pair(
    *,
    camera_index: int,
    lidar_index: int,
    camera_timestamp_ns: int | None,
    lidar_timestamp_ns: int | None,
    camera_path: Path,
    lidar_path: Path,
) -> KITTICameraLidarPair:
    delta_ms = None
    if camera_timestamp_ns is not None and lidar_timestamp_ns is not None:
        delta_ms = (lidar_timestamp_ns - camera_timestamp_ns) / 1_000_000.0
    return KITTICameraLidarPair(
        camera_index=camera_index,
        lidar_index=lidar_index,
        camera_timestamp_ns=camera_timestamp_ns,
        lidar_timestamp_ns=lidar_timestamp_ns,
        delta_ms=delta_ms,
        camera_path=str(camera_path),
        lidar_path=str(lidar_path),
        lidar_point_count=_velodyne_point_count(lidar_path),
    )


def _timestamp_by_index(timestamps: list[KITTITimestamp]) -> dict[int, int]:
    return {timestamp.index: timestamp.timestamp_ns for timestamp in timestamps}


def _nearest_timestamp_index(
    sorted_timestamps_ns: list[int],
    reference_timestamp_ns: int,
) -> int | None:
    if not sorted_timestamps_ns:
        return None
    insertion_index = bisect_left(sorted_timestamps_ns, reference_timestamp_ns)
    if insertion_index == 0:
        return 0
    if insertion_index == len(sorted_timestamps_ns):
        return len(sorted_timestamps_ns) - 1
    before = insertion_index - 1
    after = insertion_index
    if abs(sorted_timestamps_ns[after] - reference_timestamp_ns) < abs(
        sorted_timestamps_ns[before] - reference_timestamp_ns
    ):
        return after
    return before


def _velodyne_point_count(path: Path) -> int:
    size = path.stat().st_size
    if size % 16 != 0:
        return 0
    return size // 16


def _nearest_abs_deltas_ms(
    reference_timestamps_ns: list[int],
    candidate_timestamps_ns: list[int],
) -> list[float]:
    if not reference_timestamps_ns or not candidate_timestamps_ns:
        return []
    candidates = sorted(candidate_timestamps_ns)
    deltas: list[float] = []
    cursor = 0
    for reference in sorted(reference_timestamps_ns):
        while cursor + 1 < len(candidates) and abs(candidates[cursor + 1] - reference) <= abs(
            candidates[cursor] - reference
        ):
            cursor += 1
        deltas.append(abs(candidates[cursor] - reference) / 1_000_000.0)
    return deltas


def _unwrap_angles(values: list[float]) -> list[float]:
    if not values:
        return []
    unwrapped = [values[0]]
    for value in values[1:]:
        previous = unwrapped[-1]
        delta = atan2(sin(value - previous), cos(value - previous))
        unwrapped.append(previous + delta)
    return unwrapped


def _range_deg(values: list[float]) -> float:
    if not values:
        return 0.0
    return (max(values) - min(values)) * 180.0 / pi


def _paired_lidar_oxts_frames(
    root: Path,
    *,
    sample_limit: int,
) -> list[tuple[Path, SE3]]:
    lidar_files = _lidar_files(root)[: max(sample_limit, 0)]
    oxts_files = sorted((root / "oxts" / "data").glob("*.txt"))
    if not lidar_files or not oxts_files:
        return []

    oxts_timestamps = read_timestamps(root / "oxts" / "timestamps.txt")
    lidar_timestamps = read_timestamps(root / "velodyne_points" / "timestamps.txt")
    pose_records = _oxts_pose_records(oxts_files, oxts_timestamps)
    if not pose_records:
        return []

    pairs: list[tuple[Path, SE3]] = []
    for index, lidar_file in enumerate(lidar_files):
        timestamp_ns = (
            lidar_timestamps[index].timestamp_ns if index < len(lidar_timestamps) else None
        )
        pose = (
            _nearest_oxts_pose(timestamp_ns, pose_records)
            if timestamp_ns is not None
            else pose_records[min(index, len(pose_records) - 1)][1]
        )
        pairs.append((lidar_file, pose))
    return pairs


def _oxts_pose_records(
    oxts_files: list[Path],
    timestamps: list[KITTITimestamp],
) -> list[tuple[int | None, SE3]]:
    packets: list[tuple[int | None, KITTIOXTSPacket]] = []
    for index, file_path in enumerate(oxts_files):
        try:
            timestamp_ns = timestamps[index].timestamp_ns if index < len(timestamps) else None
            packets.append((timestamp_ns, read_oxts_packet(file_path)))
        except ValueError:
            continue
    if not packets:
        return []

    origin = packets[0][1]
    return [
        (timestamp_ns, _oxts_packet_to_world_ego(packet, origin))
        for timestamp_ns, packet in packets
    ]


def _oxts_packet_to_world_ego(packet: KITTIOXTSPacket, origin: KITTIOXTSPacket) -> SE3:
    latitude_scale = pi * 6378137.0 / 180.0
    longitude_scale = latitude_scale * cos(radians(origin.lat_deg))
    translation = (
        (packet.lon_deg - origin.lon_deg) * longitude_scale,
        (packet.lat_deg - origin.lat_deg) * latitude_scale,
        packet.alt_m - origin.alt_m,
    )
    return SE3(
        translation_m=translation,
        rotation_quat_xyzw=_quaternion_xyzw_from_rpy(
            packet.roll_rad,
            packet.pitch_rad,
            packet.yaw_rad,
        ),
    )


def _nearest_oxts_pose(
    timestamp_ns: int,
    pose_records: list[tuple[int | None, SE3]],
) -> SE3:
    timestamped = [
        (record_timestamp, pose)
        for record_timestamp, pose in pose_records
        if record_timestamp is not None
    ]
    if not timestamped:
        return pose_records[0][1]
    return min(
        timestamped,
        key=lambda record: abs(record[0] - timestamp_ns),
    )[1]


def _quaternion_xyzw_from_rpy(
    roll_rad: float,
    pitch_rad: float,
    yaw_rad: float,
) -> tuple[float, float, float, float]:
    half_roll = roll_rad / 2.0
    half_pitch = pitch_rad / 2.0
    half_yaw = yaw_rad / 2.0
    cr = cos(half_roll)
    sr = sin(half_roll)
    cp = cos(half_pitch)
    sp = sin(half_pitch)
    cy = cos(half_yaw)
    sy = sin(half_yaw)
    return (
        (sr * cp * cy) - (cr * sp * sy),
        (cr * sp * cy) + (sr * cp * sy),
        (cr * cp * sy) - (sr * sp * cy),
        (cr * cp * cy) + (sr * sp * sy),
    )


def _iter_velodyne_bin(path: Path) -> Iterable[VelodynePoint]:
    data = path.read_bytes()
    if len(data) % 16 != 0:
        raise ValueError("Velodyne file length must be a multiple of 16 bytes")
    for x, y, z, intensity in struct.iter_unpack("<ffff", data):
        yield (float(x), float(y), float(z), float(intensity))


def _voxel_key(x: float, y: float, z: float, voxel_size_m: float) -> tuple[int, int, int]:
    return (
        floor(x / voxel_size_m),
        floor(y / voxel_size_m),
        floor(z / voxel_size_m),
    )


def _voxel_plane_map(
    voxels: dict[tuple[int, int, int], _VoxelAccumulator],
    min_points_per_voxel: int,
) -> dict[tuple[int, int, int], _VoxelPlane]:
    planes: dict[tuple[int, int, int], _VoxelPlane] = {}
    for key, voxel in sorted(voxels.items()):
        if voxel.count < min_points_per_voxel:
            continue
        covariance = voxel.covariance()
        _, _, lambda_3 = _symmetric_eigenvalues_3x3(covariance)
        normal = _smallest_eigenvector_3x3(covariance, lambda_3)
        if normal is None:
            continue
        planes[key] = _VoxelPlane(center_m=voxel.mean(), normal=normal)
    return planes


def _point_to_plane_residuals(
    points: list[Vector3],
    planes: dict[tuple[int, int, int], _VoxelPlane],
    *,
    voxel_size_m: float,
) -> list[float]:
    residuals: list[float] = []
    for point in points:
        plane = planes.get(_voxel_key(point[0], point[1], point[2], voxel_size_m))
        if plane is None:
            continue
        dx = point[0] - plane.center_m[0]
        dy = point[1] - plane.center_m[1]
        dz = point[2] - plane.center_m[2]
        residual = (
            (plane.normal[0] * dx)
            + (plane.normal[1] * dy)
            + (plane.normal[2] * dz)
        )
        residuals.append(abs(residual))
    return residuals


def _smallest_eigenvector_3x3(
    matrix: tuple[tuple[float, float, float], ...],
    eigenvalue: float,
) -> Vector3 | None:
    rows = [
        (
            matrix[0][0] - eigenvalue,
            matrix[0][1],
            matrix[0][2],
        ),
        (
            matrix[1][0],
            matrix[1][1] - eigenvalue,
            matrix[1][2],
        ),
        (
            matrix[2][0],
            matrix[2][1],
            matrix[2][2] - eigenvalue,
        ),
    ]
    candidates = [
        _cross3(rows[0], rows[1]),
        _cross3(rows[0], rows[2]),
        _cross3(rows[1], rows[2]),
    ]
    best = max(candidates, key=_norm3)
    norm = _norm3(best)
    if norm <= 1.0e-12:
        return None
    return (best[0] / norm, best[1] / norm, best[2] / norm)


def _cross3(left: Vector3, right: Vector3) -> Vector3:
    return (
        (left[1] * right[2]) - (left[2] * right[1]),
        (left[2] * right[0]) - (left[0] * right[2]),
        (left[0] * right[1]) - (left[1] * right[0]),
    )


def _norm3(value: Vector3) -> float:
    return sqrt((value[0] * value[0]) + (value[1] * value[1]) + (value[2] * value[2]))


def _rmse(values: list[float]) -> float | None:
    if not values:
        return None
    return sqrt(sum(value * value for value in values) / len(values))


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, max(0, round((len(sorted_values) - 1) * fraction)))
    return sorted_values[index]


def _summarize_voxel_planarity(
    voxels: dict[tuple[int, int, int], _VoxelAccumulator],
    min_points_per_voxel: int,
) -> _VoxelPlanaritySummary:
    samples: list[_VoxelPlanaritySample] = []
    for _, voxel in sorted(voxels.items()):
        if voxel.count < min_points_per_voxel:
            continue
        lambda_1, lambda_2, lambda_3 = _symmetric_eigenvalues_3x3(voxel.covariance())
        if lambda_1 <= 1.0e-12:
            continue
        planarity = max(0.0, min(1.0, (lambda_2 - lambda_3) / lambda_1))
        roughness = sqrt(max(lambda_3, 0.0))
        sharpness = planarity / (1.0 + roughness)
        samples.append(
            _VoxelPlanaritySample(
                planarity=planarity,
                roughness_m=roughness,
                sharpness=sharpness,
            )
        )

    count = len(samples)
    if count == 0:
        return _VoxelPlanaritySummary(0, None, None, None, None, None)

    holdout_count = max(1, round(count * 0.2))
    holdout_samples = samples[-holdout_count:]
    train_samples = samples[:-holdout_count] or samples
    return _VoxelPlanaritySummary(
        count=count,
        planarity_mean=_mean(sample.planarity for sample in samples),
        roughness_mean_m=_mean(sample.roughness_m for sample in samples),
        sharpness_mean=_mean(sample.sharpness for sample in samples),
        point_to_plane_rmse_train_m=_mean(sample.roughness_m for sample in train_samples),
        point_to_plane_rmse_holdout_m=_mean(sample.roughness_m for sample in holdout_samples),
    )


def _mean(values: Iterable[float]) -> float:
    total = 0.0
    count = 0
    for value in values:
        total += value
        count += 1
    return total / count


def _symmetric_eigenvalues_3x3(
    matrix: tuple[tuple[float, float, float], ...],
) -> tuple[float, float, float]:
    a11, a12, a13 = matrix[0]
    _, a22, a23 = matrix[1]
    _, _, a33 = matrix[2]
    p1 = a12 * a12 + a13 * a13 + a23 * a23
    if p1 <= 1.0e-18:
        return _sorted_nonnegative(a11, a22, a33)

    q = (a11 + a22 + a33) / 3.0
    b11 = a11 - q
    b22 = a22 - q
    b33 = a33 - q
    p2 = b11 * b11 + b22 * b22 + b33 * b33 + 2.0 * p1
    p = sqrt(p2 / 6.0)
    if p <= 1.0e-18:
        return _sorted_nonnegative(q, q, q)

    c11 = b11 / p
    c12 = a12 / p
    c13 = a13 / p
    c22 = b22 / p
    c23 = a23 / p
    c33 = b33 / p
    determinant = (
        c11 * c22 * c33
        + 2.0 * c12 * c13 * c23
        - c13 * c13 * c22
        - c12 * c12 * c33
        - c23 * c23 * c11
    )
    r = max(-1.0, min(1.0, determinant / 2.0))
    phi = acos(r) / 3.0
    eigen_1 = q + 2.0 * p * cos(phi)
    eigen_3 = q + 2.0 * p * cos(phi + (2.0 * pi / 3.0))
    eigen_2 = 3.0 * q - eigen_1 - eigen_3
    return _sorted_nonnegative(eigen_1, eigen_2, eigen_3)


def _sorted_nonnegative(a: float, b: float, c: float) -> tuple[float, float, float]:
    values = sorted((max(a, 0.0), max(b, 0.0), max(c, 0.0)), reverse=True)
    return (values[0], values[1], values[2])


def _file_records(
    stream: str,
    directory: Path,
    pattern: str,
    timestamps: list[KITTITimestamp],
) -> list[TimestampedRecord]:
    files = sorted(directory.glob(pattern))
    records: list[TimestampedRecord] = []
    for index, file_path in enumerate(files):
        timestamp = timestamps[index].timestamp_ns if index < len(timestamps) else index
        records.append(
            TimestampedRecord(
                stream=stream,
                timestamp_ns=timestamp,
                payload_path=str(file_path),
                metadata={"index": index},
            )
        )
    return records


def _calibration_records(root: Path) -> list[TimestampedRecord]:
    records: list[TimestampedRecord] = []
    for index, file_path in enumerate(sorted(root.glob("calib_*.txt"))):
        records.append(
            TimestampedRecord(
                stream="calibration",
                timestamp_ns=index,
                payload_path=str(file_path),
                metadata={"values": read_calibration_file(file_path)},
            )
        )
    return records


def _count_files(directory: Path, pattern: str) -> int:
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


def _load_optional_manifest(path: Path) -> DatasetManifest | None:
    if find_manifest(path) is None:
        return None
    return load_manifest(path)


def _manifest_streams(manifest: DatasetManifest) -> list[StreamSummary]:
    return [
        StreamSummary(
            name=name,
            kind=stream.kind,
            message_count=stream.count,
            topic=stream.topic,
            sensor=stream.sensor,
        )
        for name, stream in sorted(manifest.streams.items())
    ]
