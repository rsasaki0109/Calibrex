"""Dataset abstractions, imported on first use."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from calibrex.data.a2d2 import (
        A2D2LidarDataset,
        read_a2d2_lidar_boundary_points,
        read_a2d2_lidar_points_reflectivity,
    )
    from calibrex.data.base import DatasetReader, StreamSummary, TimestampedRecord
    from calibrex.data.ethz_hand_eye import (
        ETHZ_HAND_EYE_COMMIT,
        ETHZ_ROBOT_ARM_REAL_ARCHIVE,
        ETHZ_ROBOT_ARM_REAL_SHA256,
        ETHZ_ROBOT_ARM_REAL_URL,
        AlignedHandEyePosePair,
        HandEyeMotionDataset,
        RelativeMotionPair,
        TimestampedPose,
        read_ethz_robot_arm_hand_eye_motions,
    )
    from calibrex.data.inspect import DatasetInspection, inspect_dataset
    from calibrex.data.kitti import (
        KITTIRawDataset,
        LuminanceImage,
        read_png_luminance,
        read_png_luminance_array,
    )
    from calibrex.data.kitti_benchmark import (
        KITTI_RAW_0005_BENCHMARK_FRAME_IDS,
        KITTIBenchmarkInputManifest,
        build_kitti_raw_0005_benchmark_input,
        load_kitti_benchmark_input,
        verify_kitti_benchmark_input,
    )
    from calibrex.data.livox import LivoxPCDDataset
    from calibrex.data.manifest import DatasetManifest, StreamManifest, load_manifest
    from calibrex.data.nuscenes import NuScenesDataset
    from calibrex.data.public_datasets import PublicDatasetCatalog, PublicDatasetEntry
    from calibrex.data.tum_rgbd import (
        TUMDepthImage,
        TUMDepthIntrinsics,
        TUMDepthPoseEntry,
        TUMRGBDDataset,
        associate_depth_groundtruth,
        read_tum_depth_png,
        sample_tum_depth_points,
    )

_EXPORT_MODULES = {
    "A2D2LidarDataset": "calibrex.data.a2d2",
    "read_a2d2_lidar_boundary_points": "calibrex.data.a2d2",
    "read_a2d2_lidar_points_reflectivity": "calibrex.data.a2d2",
    "DatasetReader": "calibrex.data.base",
    "StreamSummary": "calibrex.data.base",
    "TimestampedRecord": "calibrex.data.base",
    "ETHZ_HAND_EYE_COMMIT": "calibrex.data.ethz_hand_eye",
    "ETHZ_ROBOT_ARM_REAL_ARCHIVE": "calibrex.data.ethz_hand_eye",
    "ETHZ_ROBOT_ARM_REAL_SHA256": "calibrex.data.ethz_hand_eye",
    "ETHZ_ROBOT_ARM_REAL_URL": "calibrex.data.ethz_hand_eye",
    "AlignedHandEyePosePair": "calibrex.data.ethz_hand_eye",
    "HandEyeMotionDataset": "calibrex.data.ethz_hand_eye",
    "RelativeMotionPair": "calibrex.data.ethz_hand_eye",
    "TimestampedPose": "calibrex.data.ethz_hand_eye",
    "read_ethz_robot_arm_hand_eye_motions": "calibrex.data.ethz_hand_eye",
    "DatasetInspection": "calibrex.data.inspect",
    "inspect_dataset": "calibrex.data.inspect",
    "KITTIRawDataset": "calibrex.data.kitti",
    "LuminanceImage": "calibrex.data.kitti",
    "read_png_luminance": "calibrex.data.kitti",
    "read_png_luminance_array": "calibrex.data.kitti",
    "KITTI_RAW_0005_BENCHMARK_FRAME_IDS": "calibrex.data.kitti_benchmark",
    "KITTIBenchmarkInputManifest": "calibrex.data.kitti_benchmark",
    "build_kitti_raw_0005_benchmark_input": "calibrex.data.kitti_benchmark",
    "load_kitti_benchmark_input": "calibrex.data.kitti_benchmark",
    "verify_kitti_benchmark_input": "calibrex.data.kitti_benchmark",
    "LivoxPCDDataset": "calibrex.data.livox",
    "DatasetManifest": "calibrex.data.manifest",
    "StreamManifest": "calibrex.data.manifest",
    "load_manifest": "calibrex.data.manifest",
    "NuScenesDataset": "calibrex.data.nuscenes",
    "PublicDatasetCatalog": "calibrex.data.public_datasets",
    "PublicDatasetEntry": "calibrex.data.public_datasets",
    "TUMDepthImage": "calibrex.data.tum_rgbd",
    "TUMDepthIntrinsics": "calibrex.data.tum_rgbd",
    "TUMDepthPoseEntry": "calibrex.data.tum_rgbd",
    "TUMRGBDDataset": "calibrex.data.tum_rgbd",
    "associate_depth_groundtruth": "calibrex.data.tum_rgbd",
    "read_tum_depth_png": "calibrex.data.tum_rgbd",
    "sample_tum_depth_points": "calibrex.data.tum_rgbd",
}


def __getattr__(name: str) -> Any:
    module = _EXPORT_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "ETHZ_HAND_EYE_COMMIT",
    "ETHZ_ROBOT_ARM_REAL_ARCHIVE",
    "ETHZ_ROBOT_ARM_REAL_SHA256",
    "ETHZ_ROBOT_ARM_REAL_URL",
    "KITTI_RAW_0005_BENCHMARK_FRAME_IDS",
    "A2D2LidarDataset",
    "AlignedHandEyePosePair",
    "DatasetInspection",
    "DatasetManifest",
    "DatasetReader",
    "HandEyeMotionDataset",
    "KITTIBenchmarkInputManifest",
    "KITTIRawDataset",
    "LivoxPCDDataset",
    "LuminanceImage",
    "NuScenesDataset",
    "PublicDatasetCatalog",
    "PublicDatasetEntry",
    "RelativeMotionPair",
    "StreamManifest",
    "StreamSummary",
    "TUMDepthImage",
    "TUMDepthIntrinsics",
    "TUMDepthPoseEntry",
    "TUMRGBDDataset",
    "TimestampedPose",
    "TimestampedRecord",
    "associate_depth_groundtruth",
    "build_kitti_raw_0005_benchmark_input",
    "inspect_dataset",
    "load_kitti_benchmark_input",
    "load_manifest",
    "read_a2d2_lidar_boundary_points",
    "read_a2d2_lidar_points_reflectivity",
    "read_ethz_robot_arm_hand_eye_motions",
    "read_png_luminance",
    "read_png_luminance_array",
    "read_tum_depth_png",
    "sample_tum_depth_points",
    "verify_kitti_benchmark_input",
]
