"""KITTI raw INS (OXTS) trajectories and Velodyne scans for INS--LiDAR calibration.

Frames are always associated by file index and timestamp, never by position in
a directory listing: some KITTI "sync" drives omit Velodyne files (for example
2011_09_26_drive_0009 lacks frames 177-180) while their timestamp files keep a
blank line per missing frame, so positional pairing silently shifts every later
frame by several hundred milliseconds.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3
from calibrex.data.kitti import (
    kitti_oxts_pose_world_imu,
    read_calibration_file,
    read_oxts_packet,
    read_timestamps,
)

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class KittiInsTrajectory:
    """OXTS poses ``T_world_imu`` at their own timestamps (seconds)."""

    times_s: FloatArray
    poses: FloatArray  # (N, 4, 4)
    frame_indices: tuple[int, ...]
    quaternions_xyzw: FloatArray | None = None
    """The orientation of each pose as the quaternion it was built from, (N, 4)."""


@dataclass(frozen=True)
class KittiVelodyneFrame:
    """One Velodyne sweep and its forward-facing (mid-sweep) timestamp."""

    frame_index: int
    time_s: float
    path: Path


@dataclass(frozen=True)
class KittiInsLidarDrive:
    """Everything a KITTI INS--LiDAR calibration run reads from one drive."""

    drive_dir: Path
    trajectory: KittiInsTrajectory
    velodyne_frames: tuple[KittiVelodyneFrame, ...]
    vendor_t_imu_lidar: FloatArray
    missing_velodyne_indices: tuple[int, ...]
    input_sha256: str


def load_kitti_ins_lidar_drive(
    drive_dir: str | Path, *, calibration_dir: str | Path | None = None
) -> KittiInsLidarDrive:
    """Load OXTS poses, Velodyne frame timestamps, and the vendor extrinsic.

    ``calib_imu_to_velo.txt`` is read from ``calibration_dir``, by default the
    drive's parent (the KITTI date directory).
    """

    root = Path(drive_dir)
    oxts_times = {
        item.index: item.timestamp_ns * 1.0e-9
        for item in read_timestamps(root / "oxts" / "timestamps.txt")
    }
    oxts_files = {int(path.stem): path for path in (root / "oxts" / "data").glob("*.txt")}
    oxts_indices = sorted(set(oxts_times) & set(oxts_files))
    if len(oxts_indices) < 2:
        raise ValueError(f"{root} has fewer than two timestamped OXTS packets")
    packets = [read_oxts_packet(oxts_files[index]) for index in oxts_indices]
    transforms = [kitti_oxts_pose_world_imu(p, packets[0]) for p in packets]
    poses = np.stack([se3_matrix(transform) for transform in transforms])
    quaternions = np.array([transform.rotation_quat_xyzw for transform in transforms])
    times = np.array([oxts_times[index] for index in oxts_indices])
    order = np.argsort(times)
    trajectory = KittiInsTrajectory(
        times_s=times[order],
        poses=poses[order],
        frame_indices=tuple(oxts_indices[i] for i in order),
        quaternions_xyzw=quaternions[order],
    )

    velodyne_times = {
        item.index: item.timestamp_ns * 1.0e-9
        for item in read_timestamps(root / "velodyne_points" / "timestamps.txt")
    }
    velodyne_files = {
        int(path.stem): path for path in (root / "velodyne_points" / "data").glob("*.bin")
    }
    frames = tuple(
        KittiVelodyneFrame(
            frame_index=index, time_s=velodyne_times[index], path=velodyne_files[index]
        )
        for index in sorted(set(velodyne_times) & set(velodyne_files))
    )
    expected = range(min(velodyne_files, default=0), max(velodyne_files, default=-1) + 1)
    missing = tuple(index for index in expected if index not in velodyne_files)

    calibration_path = (
        Path(calibration_dir) if calibration_dir is not None else root.parent
    ) / "calib_imu_to_velo.txt"
    if not calibration_path.is_file():
        raise ValueError(f"{calibration_path} not found (calib_imu_to_velo.txt)")
    calibration = read_calibration_file(calibration_path)
    t_velo_imu = np.eye(4)
    t_velo_imu[:3, :3] = np.array(calibration["R"], dtype=np.float64).reshape(3, 3)
    t_velo_imu[:3, 3] = np.array(calibration["T"], dtype=np.float64)
    return KittiInsLidarDrive(
        drive_dir=root,
        trajectory=trajectory,
        velodyne_frames=frames,
        vendor_t_imu_lidar=np.linalg.inv(t_velo_imu),
        missing_velodyne_indices=missing,
        input_sha256=_input_digest(root, calibration_path),
    )


def read_velodyne_xyz(path: Path) -> FloatArray:
    """Read a KITTI ``.bin`` sweep as ``(N, 3)`` float64 coordinates."""

    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64)


def se3_matrix(transform: SE3) -> FloatArray:
    """Return the 4x4 matrix of an :class:`SE3`."""

    from scipy.spatial.transform import Rotation

    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(transform.rotation_quat_xyzw).as_matrix()
    matrix[:3, 3] = transform.translation_m
    return matrix


def _input_digest(root: Path, calibration_path: Path) -> str:
    """Digest every file the run reads: OXTS, Velodyne, timestamps, calibration."""

    digest = hashlib.sha256()
    files = [
        calibration_path,
        root / "oxts" / "timestamps.txt",
        root / "velodyne_points" / "timestamps.txt",
        *sorted((root / "oxts" / "data").glob("*.txt")),
        *sorted((root / "velodyne_points" / "data").glob("*.bin")),
    ]
    for path in files:
        try:
            label = path.relative_to(root.parent).as_posix()
        except ValueError:  # a calibration directory outside the date directory
            label = path.name
        digest.update(label.encode("utf-8"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()
