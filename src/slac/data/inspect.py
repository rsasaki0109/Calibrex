"""Lightweight dataset inspection for dry runs and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from slac.core.config import DatasetConfig
from slac.core.exceptions import DatasetError
from slac.data.a2d2 import A2D2LidarDataset, summarize_a2d2_lidar_npz
from slac.data.base import StreamSummary
from slac.data.filesystem import FilesystemDataset
from slac.data.kitti import (
    KITTIRawDataset,
    find_camera_lidar_pairs,
    summarize_lidar_world_map_consistency,
    summarize_oxts_motion,
    summarize_timestamp_alignment,
    summarize_velodyne_points,
)
from slac.data.livox import LivoxPCDDataset, summarize_livox_pcd
from slac.data.manifest import find_manifest
from slac.data.mcap import inspect_mcap
from slac.data.nuscenes import NuScenesDataset, summarize_nuscenes_metadata
from slac.data.rosbag1 import summarize_rosbag1
from slac.data.rosbag2 import summarize_rosbag2
from slac.data.tum_rgbd import TUMRGBDDataset

# Default diagnostic sample/frame counts, preserved from the previously hardcoded
# call sites. `DatasetConfig.sample_limit` overrides these on a per-dataset basis.
DEFAULT_A2D2_SAMPLE_LIMIT = 3
DEFAULT_LIVOX_SAMPLE_LIMIT = 4
DEFAULT_KITTI_SAMPLE_LIMIT = 3


@dataclass(frozen=True)
class DatasetInspection:
    """Summary returned by `slac inspect`."""

    dataset_type: str
    path: str
    exists: bool
    manifest: str | None = None
    streams: list[StreamSummary] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        """Return JSON/YAML-friendly representation."""

        return {
            "dataset_type": self.dataset_type,
            "path": self.path,
            "exists": self.exists,
            "manifest": self.manifest,
            "streams": [stream.__dict__ for stream in self.streams],
            "warnings": list(self.warnings),
            "diagnostics": dict(self.diagnostics),
        }


def inspect_dataset(dataset: DatasetConfig) -> DatasetInspection:
    """Inspect a dataset path without binding core code to ROS message types."""

    path = Path(dataset.path)
    if dataset.type == "a2d2_lidar":
        sample_limit = dataset.sample_limit or DEFAULT_A2D2_SAMPLE_LIMIT
        return _inspect_a2d2_lidar(path, dataset.type, sample_limit=sample_limit)
    if dataset.type == "filesystem":
        return _inspect_filesystem(path, dataset.type)
    if dataset.type == "kitti_raw":
        sample_limit = dataset.sample_limit or DEFAULT_KITTI_SAMPLE_LIMIT
        return _inspect_kitti_raw(path, dataset.type, sample_limit=sample_limit)
    if dataset.type == "livox_pcd":
        sample_limit = dataset.sample_limit or DEFAULT_LIVOX_SAMPLE_LIMIT
        return _inspect_livox_pcd(path, dataset.type, sample_limit=sample_limit)
    if dataset.type == "tum_rgbd":
        return _inspect_tum_rgbd(path, dataset.type)
    if dataset.type == "mcap":
        streams, warnings = inspect_mcap(path)
        return DatasetInspection(
            dataset_type=dataset.type,
            path=str(path),
            exists=path.exists(),
            streams=streams,
            warnings=warnings,
        )
    if dataset.type == "nuscenes":
        return _inspect_nuscenes(path, dataset.type)
    if dataset.type == "rosbag1":
        return _inspect_rosbag1(path, dataset.type)
    if dataset.type == "rosbag2":
        return _inspect_rosbag2(path, dataset.type)
    msg = f"unsupported dataset type: {dataset.type}"
    raise DatasetError(msg)


def _inspect_a2d2_lidar(
    path: Path,
    dataset_type: str,
    *,
    sample_limit: int = DEFAULT_A2D2_SAMPLE_LIMIT,
) -> DatasetInspection:
    if not path.exists():
        return DatasetInspection(
            dataset_type=dataset_type,
            path=str(path),
            exists=False,
            warnings=["dataset path does not exist"],
        )
    manifest = find_manifest(path)
    reader = A2D2LidarDataset(path)
    stats = summarize_a2d2_lidar_npz(path, sample_limit=sample_limit)
    warnings: list[str] = []
    if stats.status != "scored":
        warnings.append(stats.reason or "A2D2 LiDAR NPZ samples could not be parsed")
    if stats.sample_count == 0:
        warnings.append("A2D2 LiDAR NPZ files are missing")
    if len(stats.physical_lidar_ids) < 2:
        warnings.append("A2D2 sample contains fewer than two physical LiDAR ids")
    if stats.malformed_files:
        warnings.append("some A2D2 LiDAR NPZ files are malformed")
    return DatasetInspection(
        dataset_type=dataset_type,
        path=str(path),
        exists=True,
        manifest=str(manifest) if manifest is not None else None,
        streams=reader.streams(),
        warnings=warnings,
        diagnostics={"a2d2_lidar": stats.as_dict()},
    )


def _inspect_filesystem(path: Path, dataset_type: str) -> DatasetInspection:
    if not path.exists():
        return DatasetInspection(
            dataset_type=dataset_type,
            path=str(path),
            exists=False,
            warnings=["dataset path does not exist"],
        )
    manifest = find_manifest(path)
    reader = FilesystemDataset(path)
    return DatasetInspection(
        dataset_type=dataset_type,
        path=str(path),
        exists=True,
        manifest=str(manifest) if manifest is not None else None,
        streams=reader.streams(),
    )


def _inspect_livox_pcd(
    path: Path,
    dataset_type: str,
    *,
    sample_limit: int = DEFAULT_LIVOX_SAMPLE_LIMIT,
) -> DatasetInspection:
    if not path.exists():
        return DatasetInspection(
            dataset_type=dataset_type,
            path=str(path),
            exists=False,
            warnings=["dataset path does not exist"],
        )
    manifest = find_manifest(path)
    reader = LivoxPCDDataset(path)
    stats = summarize_livox_pcd(path, sample_limit=sample_limit)
    warnings: list[str] = []
    if stats.status != "scored":
        warnings.append(stats.reason or "Livox PCD files could not be parsed")
    if stats.sample_count == 0:
        warnings.append("Livox PCD files are missing")
    if stats.sample_count < 2:
        warnings.append("at least two Livox PCD frames are recommended for LiDAR-to-LiDAR evidence")
    if stats.pair_shared_voxel_count == 0:
        warnings.append("base/target Livox PCD frames have no coarse voxel overlap")
    if stats.malformed_files:
        warnings.append("some Livox PCD files are malformed")
    return DatasetInspection(
        dataset_type=dataset_type,
        path=str(path),
        exists=True,
        manifest=str(manifest) if manifest is not None else None,
        streams=reader.streams(),
        warnings=warnings,
        diagnostics={"livox_pcd": stats.as_dict()},
    )


def _inspect_tum_rgbd(path: Path, dataset_type: str) -> DatasetInspection:
    if not path.exists():
        return DatasetInspection(
            dataset_type=dataset_type,
            path=str(path),
            exists=False,
            warnings=["dataset path does not exist"],
        )
    manifest = find_manifest(path)
    reader = TUMRGBDDataset(path)
    warnings: list[str] = []
    stream_counts = {stream.name: stream.message_count for stream in reader.streams()}
    if stream_counts.get("rgbd_associations") == 0 and stream_counts.get("rgb", 0):
        warnings.append("associations.txt is missing or empty; generate it before RGB-D processing")
    return DatasetInspection(
        dataset_type=dataset_type,
        path=str(path),
        exists=True,
        manifest=str(manifest) if manifest is not None else None,
        streams=reader.streams(),
        warnings=warnings,
    )


def _inspect_kitti_raw(
    path: Path,
    dataset_type: str,
    *,
    sample_limit: int = DEFAULT_KITTI_SAMPLE_LIMIT,
) -> DatasetInspection:
    if not path.exists():
        return DatasetInspection(
            dataset_type=dataset_type,
            path=str(path),
            exists=False,
            warnings=["dataset path does not exist"],
        )
    manifest = find_manifest(path)
    reader = KITTIRawDataset(path)
    warnings: list[str] = []
    counts = {stream.name: stream.message_count or 0 for stream in reader.streams()}
    velodyne_stats = summarize_velodyne_points(path, sample_limit=sample_limit)
    lidar_world_map_stats = summarize_lidar_world_map_consistency(path, sample_limit=sample_limit)
    oxts_stats = summarize_oxts_motion(path)
    timestamp_stats = summarize_timestamp_alignment(path)
    camera_lidar_pairs = find_camera_lidar_pairs(path, max_pairs=3)
    if counts.get("velodyne_points", 0) == 0:
        warnings.append("velodyne point cloud files are missing")
    elif velodyne_stats.sampled_point_count == 0:
        warnings.append("velodyne point cloud samples contain no points")
    elif velodyne_stats.planarity_voxel_count == 0:
        warnings.append("velodyne samples do not contain enough local neighborhoods")
    if velodyne_stats.malformed_files:
        warnings.append("some velodyne point cloud files are malformed")
    if counts.get("oxts", 0) == 0:
        warnings.append("OXTS motion files are missing")
    if oxts_stats.malformed_files:
        warnings.append("some OXTS motion files are malformed")
    if counts.get("calibration", 0) == 0:
        warnings.append("KITTI calibration files are missing")
    if timestamp_stats.camera_timestamp_count == 0:
        warnings.append("KITTI camera timestamps are missing")
    if timestamp_stats.lidar_timestamp_count == 0:
        warnings.append("KITTI LiDAR timestamps are missing")
    if timestamp_stats.oxts_timestamp_count == 0:
        warnings.append("KITTI OXTS timestamps are missing")
    if (
        counts.get("camera_left_color", 0) > 0
        and counts.get("velodyne_points", 0) > 0
        and not camera_lidar_pairs
    ):
        warnings.append("KITTI camera-LiDAR frame pairs could not be formed")
    return DatasetInspection(
        dataset_type=dataset_type,
        path=str(path),
        exists=True,
        manifest=str(manifest) if manifest is not None else None,
        streams=reader.streams(),
        warnings=warnings,
        diagnostics={
            "velodyne_points": velodyne_stats.as_dict(),
            "lidar_world_map_consistency": lidar_world_map_stats.as_dict(),
            "oxts_motion": oxts_stats.as_dict(),
            "timestamp_alignment": timestamp_stats.as_dict(),
            "camera_lidar_pairs": [pair.as_dict() for pair in camera_lidar_pairs],
        },
    )


def _inspect_rosbag1(path: Path, dataset_type: str) -> DatasetInspection:
    if not path.exists():
        return DatasetInspection(
            dataset_type=dataset_type,
            path=str(path),
            exists=False,
            warnings=["dataset path does not exist"],
        )
    manifest = find_manifest(path)
    stats = summarize_rosbag1(path, sample_limit=4)
    warnings: list[str] = []
    if stats.status == "malformed":
        warnings.append(stats.reason or "ROS bag could not be parsed")
    if stats.pointcloud_topic_count == 0:
        warnings.append("no sensor_msgs/PointCloud2 topics found in ROS bag")
    elif stats.pointcloud_topic_count < 2:
        warnings.append(
            "fewer than two PointCloud2 topics; a LiDAR-to-LiDAR pair needs two clouds"
        )
    for stream in stats.streams:
        if stream.sampled_point_count == 0:
            warnings.append(f"PointCloud2 topic {stream.topic} decoded no points")
    streams = [
        StreamSummary(
            name=stream.topic,
            kind="pointcloud",
            message_count=stream.message_count,
            topic=stream.topic,
            sensor=stream.message_type,
        )
        for stream in stats.streams
    ]
    return DatasetInspection(
        dataset_type=dataset_type,
        path=str(path),
        exists=True,
        manifest=str(manifest) if manifest is not None else None,
        streams=streams,
        warnings=warnings,
        diagnostics={"rosbag1": stats.as_dict()},
    )


def _inspect_rosbag2(path: Path, dataset_type: str) -> DatasetInspection:
    if not path.exists():
        return DatasetInspection(
            dataset_type=dataset_type,
            path=str(path),
            exists=False,
            warnings=["dataset path does not exist"],
        )
    manifest = find_manifest(path)
    stats = summarize_rosbag2(path, sample_limit=4)
    warnings: list[str] = []
    if stats.status == "malformed":
        warnings.append(stats.reason or "rosbag2 bag could not be parsed")
    if stats.topic_count == 0:
        warnings.append("no supported PointCloud2 or Odometry topics found in rosbag2 bag")
    for stream in stats.streams:
        if stream.message_type == "sensor_msgs/msg/PointCloud2" and stream.sampled_point_count == 0:
            warnings.append(f"PointCloud2 topic {stream.topic} decoded no points")
    streams = [
        StreamSummary(
            name=stream.topic,
            kind="pointcloud"
            if stream.message_type == "sensor_msgs/msg/PointCloud2"
            else "odometry",
            message_count=stream.message_count,
            topic=stream.topic,
            sensor=stream.message_type,
        )
        for stream in stats.streams
    ]
    return DatasetInspection(
        dataset_type=dataset_type,
        path=str(path),
        exists=True,
        manifest=str(manifest) if manifest is not None else None,
        streams=streams,
        warnings=warnings,
        diagnostics={"rosbag2": stats.as_dict()},
    )


def _inspect_nuscenes(path: Path, dataset_type: str) -> DatasetInspection:
    if not path.exists():
        return DatasetInspection(
            dataset_type=dataset_type,
            path=str(path),
            exists=False,
            warnings=["dataset path does not exist"],
        )
    manifest = find_manifest(path)
    reader = NuScenesDataset(path)
    metadata = summarize_nuscenes_metadata(path)
    warnings: list[str] = []
    if metadata.status != "scored":
        warnings.append(metadata.reason or "nuScenes metadata tables could not be loaded")
    if metadata.sample_data_count == 0:
        warnings.append("nuScenes sample_data table is empty or missing")
    if metadata.modality_counts.get("lidar", 0) == 0:
        warnings.append("nuScenes LiDAR sample_data entries are missing")
    if metadata.modality_counts.get("camera", 0) == 0:
        warnings.append("nuScenes camera sample_data entries are missing")
    if metadata.modality_counts.get("radar", 0) == 0:
        warnings.append("nuScenes radar sample_data entries are missing")
    if metadata.missing_file_count:
        warnings.append(f"nuScenes sample files missing locally: {metadata.missing_file_count}")
    return DatasetInspection(
        dataset_type=dataset_type,
        path=str(path),
        exists=True,
        manifest=str(manifest) if manifest is not None else None,
        streams=reader.streams(),
        warnings=warnings,
        diagnostics={"nuscenes": metadata.as_dict()},
    )
