"""Targetless camera-IMU rotation, clock offset, and gyro bias from ROS 2 bags.

Camera rotations come from tracked image features (see
:mod:`calibrex.evaluation.visual_rotation`), so no calibration target is
needed.  They are aligned with the gyro by the same held-out evidence as the
IMU-LiDAR rotation: every third window held out, a window jackknife, and 1 deg
and 10 ms known-bad controls that held-out windows must detect.

The result is a ``slac.imu_lidar_rotation/v0.1`` artifact with
``sensor_modality: camera``; its "lidar" frame is the camera frame, so the
rotation is that of ``T_cam_imu`` and the clock offset follows
``t_imu = t_cam + dt``, the conventions of Kalibr's ``T_cam_imu`` and
``timeshift_cam_imu``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex import __version__
from calibrex.core.imu_lidar_rotation import (
    ImuLidarRotationArtifact,
    ImuLidarRotationProvenance,
    ImuLidarWindowSummary,
)
from calibrex.core.provenance import git_commit
from calibrex.data.livox_ros2 import ImuSamples, bag_input_digest
from calibrex.data.ros2_camera_imu import iter_gray_images, load_ros2_imu
from calibrex.evaluation.imu_lidar_rotation import (
    ImuLidarRunOptions,
    evaluate_imu_lidar_rotation,
    rotation_artifact_from_evaluation,
)
from calibrex.evaluation.visual_rotation import (
    CameraModel,
    CameraRotationTrack,
    VisualRotationOptions,
    track_camera_rotations,
)
from calibrex.solvers.gnss_lever_arm_solver import OdometryWindow
from calibrex.solvers.imu_lidar_rotation_solver import GyroSeries

FloatArray: TypeAlias = NDArray[np.float64]
CAMERA_LIMITATIONS: tuple[str, ...] = (
    "Camera rotations come from tracked features: essential-matrix fits, or "
    "pure-rotation fits for small-baseline pairs, whose parallax bias grows with "
    "near scenes.",
    "The artifact calls the camera frame 'lidar': the rotation is that of T_cam_imu.",
    "The reference, when given, is a target-based calibration (for example Kalibr); "
    "it is a comparison, not ground truth.",
)


@dataclass(frozen=True)
class CameraImuRunOptions:
    """Frame stride and the tracking and evaluation settings."""

    frame_stride: int = 4
    max_seconds: float | None = None
    visual: VisualRotationOptions = field(default_factory=VisualRotationOptions)
    evaluation: ImuLidarRunOptions = field(default_factory=ImuLidarRunOptions)


def camera_rotation_windows(
    bags: Sequence[str | Path],
    image_topic: str,
    camera: CameraModel,
    options: CameraImuRunOptions,
) -> tuple[list[OdometryWindow], list[CameraRotationTrack]]:
    """Track every bag separately and pool their windows."""

    windows: list[OdometryWindow] = []
    tracks = []
    for bag in bags:
        track = track_camera_rotations(
            iter_gray_images(
                bag, image_topic, stride=options.frame_stride, max_seconds=options.max_seconds
            ),
            camera,
            options.visual,
        )
        tracks.append(track)
        for window in track.windows(options.visual):
            windows.append(
                OdometryWindow(
                    f"{Path(bag).name}/{window.window_id}",
                    len(tracks) * 100_000 + window.block,
                    window.times_s,
                    window.poses,
                )
            )
    return windows, tracks


def run_ros2_camera_imu_rotation(
    bags: Sequence[str | Path],
    *,
    image_topic: str,
    imu_topic: str,
    camera: CameraModel,
    dataset_family: str,
    dataset_license: str,
    acceleration_unit: Literal["mps2", "g"] = "mps2",
    reference_rotation: FloatArray | None = None,
    reference: str | None = None,
    options: CameraImuRunOptions | None = None,
    command: list[str] | None = None,
) -> ImuLidarRotationArtifact:
    """Calibrate one camera against the IMU over one or more bags of one rig."""

    opts = options or CameraImuRunOptions()
    if not bags:
        raise ValueError("at least one bag is required")
    samples = [load_ros2_imu(bag, imu_topic, acceleration_unit=acceleration_unit) for bag in bags]
    times = np.concatenate([sample.times_s for sample in samples])
    order = np.argsort(times, kind="stable")
    keep = np.r_[True, np.diff(times[order]) > 0.0]
    imu = ImuSamples(
        times_s=times[order][keep],
        gyro_rps=np.concatenate([sample.gyro_rps for sample in samples])[order][keep],
        accel_mps2=np.concatenate([sample.accel_mps2 for sample in samples])[order][keep],
    )
    windows, tracks = camera_rotation_windows(bags, image_topic, camera, opts)
    if len(windows) < 3:
        raise ValueError(f"camera tracking produced {len(windows)} windows; at least 3 are needed")
    evaluation = evaluate_imu_lidar_rotation(
        GyroSeries(imu.times_s, imu.gyro_rps),
        windows,
        opts.evaluation,
        reference_rotation=reference_rotation,
    )
    digest, scope = bag_input_digest(bags)
    frames = sum(len(track.times_s) for track in tracks)
    return rotation_artifact_from_evaluation(
        evaluation,
        opts.evaluation,
        windows=ImuLidarWindowSummary(
            scans_read=frames,
            odometry_segments=sum(track.stats.segments for track in tracks),
            unreliable_registrations=sum(track.stats.failed for track in tracks),
            windows=len(windows),
            rate_intervals=evaluation.interval_count,
            imu_samples=len(imu.times_s),
        ),
        provenance=ImuLidarRotationProvenance(
            generator=__name__,
            generator_version=__version__,
            git_commit=git_commit(),
            command=command or [],
            dataset_family=dataset_family,
            sequence_ids=[Path(bag).name for bag in bags],
            stream_profile=f"ros2-image:{image_topic}+imu:{imu_topic}",
            input_sha256=hashlib.sha256(digest.encode("ascii")).hexdigest(),
            input_digest_scope=scope,
            dataset_license=dataset_license,
        ),
        reference=reference,
        extra_options={
            "frame_stride": opts.frame_stride,
            "visual_rotation": [
                {
                    "bag": Path(bag).name,
                    "pairs": track.stats.pairs,
                    "essential": track.stats.essential,
                    "pure_rotation": track.stats.pure_rotation,
                    "failed": track.stats.failed,
                }
                for bag, track in zip(bags, tracks, strict=True)
            ],
            "essential_inlier_fraction": opts.visual.essential_inlier_fraction,
            "max_features": opts.visual.max_features,
        },
        limitations=list(CAMERA_LIMITATIONS),
        sensor_modality="camera",
    )
