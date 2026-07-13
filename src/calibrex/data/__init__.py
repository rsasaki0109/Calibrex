"""Dataset abstractions."""

from calibrex.data.a2d2 import A2D2LidarDataset
from calibrex.data.base import DatasetReader, StreamSummary, TimestampedRecord
from calibrex.data.ethz_hand_eye import (
    ETHZ_HAND_EYE_COMMIT,
    ETHZ_ROBOT_ARM_REAL_ARCHIVE,
    ETHZ_ROBOT_ARM_REAL_SHA256,
    ETHZ_ROBOT_ARM_REAL_URL,
    HandEyeMotionDataset,
    RelativeMotionPair,
    TimestampedPose,
    read_ethz_robot_arm_hand_eye_motions,
)
from calibrex.data.inspect import DatasetInspection, inspect_dataset
from calibrex.data.kitti import KITTIRawDataset
from calibrex.data.livox import LivoxPCDDataset
from calibrex.data.manifest import DatasetManifest, StreamManifest, load_manifest
from calibrex.data.nuscenes import NuScenesDataset
from calibrex.data.public_datasets import PublicDatasetCatalog, PublicDatasetEntry
from calibrex.data.tum_rgbd import TUMRGBDDataset

__all__ = [
    "ETHZ_HAND_EYE_COMMIT",
    "ETHZ_ROBOT_ARM_REAL_ARCHIVE",
    "ETHZ_ROBOT_ARM_REAL_SHA256",
    "ETHZ_ROBOT_ARM_REAL_URL",
    "A2D2LidarDataset",
    "DatasetInspection",
    "DatasetManifest",
    "DatasetReader",
    "HandEyeMotionDataset",
    "KITTIRawDataset",
    "LivoxPCDDataset",
    "NuScenesDataset",
    "PublicDatasetCatalog",
    "PublicDatasetEntry",
    "RelativeMotionPair",
    "StreamManifest",
    "StreamSummary",
    "TUMRGBDDataset",
    "TimestampedPose",
    "TimestampedRecord",
    "inspect_dataset",
    "load_manifest",
    "read_ethz_robot_arm_hand_eye_motions",
]
