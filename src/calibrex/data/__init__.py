"""Dataset abstractions."""

from calibrex.data.a2d2 import A2D2LidarDataset, read_a2d2_lidar_points_reflectivity
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
from calibrex.data.kitti import KITTIRawDataset, LuminanceImage, read_png_luminance
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

__all__ = [
    "ETHZ_HAND_EYE_COMMIT",
    "ETHZ_ROBOT_ARM_REAL_ARCHIVE",
    "ETHZ_ROBOT_ARM_REAL_SHA256",
    "ETHZ_ROBOT_ARM_REAL_URL",
    "A2D2LidarDataset",
    "AlignedHandEyePosePair",
    "DatasetInspection",
    "DatasetManifest",
    "DatasetReader",
    "HandEyeMotionDataset",
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
    "inspect_dataset",
    "load_manifest",
    "read_a2d2_lidar_points_reflectivity",
    "read_ethz_robot_arm_hand_eye_motions",
    "read_png_luminance",
    "read_tum_depth_png",
    "sample_tum_depth_points",
]
