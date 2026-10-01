"""Convert a KITTI raw drive into a rosbag2 bag that ``calibrex check`` can audit.

The bag carries the sensors of one ``*_sync`` drive and a ``/tf_static`` built
from ``calib_imu_to_velo.txt``, so the vendor calibration becomes the *candidate*
the check judges. Nothing is estimated here; OXTS parsing and the INS pose
convention are the ones of :mod:`calibrex.data.kitti` and
:mod:`calibrex.data.kitti_ins_lidar`.

Contents and frames
-------------------
``/velodyne_points`` (``sensor_msgs/PointCloud2``, ``velo_link``)
    x, y, z, intensity as float32. KITTI raw Velodyne sweeps carry **no per-point
    time**, and none is invented: the cloud has no time field, so scan deskewing
    (and ``imu-lidar``) is unsupported on it, exactly as the KITTI text
    pipelines register sweeps as rigid snapshots. The header stamp is KITTI's
    ``velodyne_points/timestamps.txt`` (the forward-facing, mid-sweep time).
``/oxts/imu`` (``sensor_msgs/Imu``, ``imu_link``)
    OXTS body-frame angular rate ``wx, wy, wz`` and acceleration ``ax, ay, az``.
    No orientation (``orientation_covariance[0] = -1``).
``/oxts/fix`` (``sensor_msgs/NavSatFix``, ``imu_link``)
    OXTS latitude, longitude, altitude. KITTI places the GNSS antenna position
    solution at the OXTS (IMU) unit, so the fix is expressed in ``imu_link``;
    the antenna lever arm is not published.
``/oxts/odometry`` (``nav_msgs/Odometry``, ``odom`` -> ``imu_link``)
    The INS trajectory: pose ``T_odom_imu`` as in ``kitti_oxts_pose_world_imu``
    (Mercator projection of the first packet's latitude, origin at the first
    packet, orientation from roll/pitch/yaw), and in the twist field the
    **body-frame** velocity and angular rate of the OXTS unit (``vf, vl, vu`` and
    ``wf, wl, wu`` are level-frame quantities, level = ``Ry(pitch) Rx(roll)``
    body; they are rotated back with each packet's roll and pitch).
``/oxts/twist`` (``geometry_msgs/TwistStamped``, ``base_link``)
    The same body-frame velocity and angular rate as a wheel-odometry **proxy**
    (KITTI has no wheel odometry). Because ``base_link`` is the OXTS frame the
    components are expressed in ``base_link``. Declare it with
    ``--topic-kind /oxts/twist=wheel``: by name it reads as an INS topic.
``/tf_static`` (``tf2_msgs/TFMessage``, transient-local)
    ``base_link -> imu_link`` (identity) and ``imu_link -> velo_link``
    (``T_imu_velo``, the inverse of ``calib_imu_to_velo.txt``).

``base_link`` is **defined as the OXTS/IMU frame** (KITTI's own convention: the
calibration files give sensors relative to the OXTS unit, and no separate
vehicle frame is published). The OXTS unit is not aligned with the direction of
motion to better than about 1 deg, so the vehicle pairs of ``calibrex check``
report that offset against this ``base_link``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from calibrex import __version__
from calibrex.core.exceptions import DatasetError
from calibrex.core.provenance import git_commit
from calibrex.data.kitti import read_timestamps
from calibrex.data.kitti_ins_lidar import KittiInsLidarDrive, load_kitti_ins_lidar_drive
from calibrex.data.ros_cdr_writer import (
    POINT_FIELD_FLOAT32,
    Transform,
    encode_imu,
    encode_navsatfix,
    encode_odometry,
    encode_pointcloud2,
    encode_tf_message,
    encode_twist_stamped,
)
from calibrex.data.rosbag2_writer import Rosbag2Writer
from calibrex.evaluation.vehicle_frame import oxts_body_frame_motion

FloatArray: TypeAlias = NDArray[np.float64]

CONVERSION_FILENAME = "calibrex_conversion.json"
CONVERSION_SCHEMA = "calibrex.rosbag_conversion/v0.1"
LIDAR_TOPIC = "/velodyne_points"
IMU_TOPIC = "/oxts/imu"
FIX_TOPIC = "/oxts/fix"
ODOMETRY_TOPIC = "/oxts/odometry"
TWIST_TOPIC = "/oxts/twist"
TF_STATIC_TOPIC = "/tf_static"
LIDAR_FRAME = "velo_link"
IMU_FRAME = "imu_link"
BASE_FRAME = "base_link"
ODOM_FRAME = "odom"
KITTI_LICENSE = "CC BY-NC-SA 3.0"

CLOUD_FIELDS = (
    ("x", 0, POINT_FIELD_FLOAT32, 1),
    ("y", 4, POINT_FIELD_FLOAT32, 1),
    ("z", 8, POINT_FIELD_FLOAT32, 1),
    ("intensity", 12, POINT_FIELD_FLOAT32, 1),
)


@dataclass(frozen=True)
class KittiConversionResult:
    """What :func:`convert_kitti_raw_to_rosbag2` wrote."""

    bag: Path
    provenance_path: Path
    message_counts: dict[str, int]
    provenance: dict[str, Any]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quat(matrix: FloatArray) -> tuple[float, float, float, float]:
    x, y, z, w = (float(value) for value in Rotation.from_matrix(matrix).as_quat())
    return x, y, z, w


def _check_drive(root: Path) -> None:
    if not (root / "oxts" / "data").is_dir() or not (root / "velodyne_points" / "data").is_dir():
        raise DatasetError(f"{root} is not a KITTI raw drive (needs oxts/ and velodyne_points/)")


def convert_kitti_raw_to_rosbag2(
    drive_dirs: str | Path | Sequence[str | Path],
    output: str | Path,
    *,
    calibration_dir: str | Path | None = None,
    command: list[str] | None = None,
    overwrite: bool = False,
) -> KittiConversionResult:
    """Write KITTI raw ``*_sync`` drive(s) as one rosbag2 directory ``output``.

    Several drives are written into one bag, in time order, only when they share
    one calibration (the same rig on the same day). The minutes between drives
    are gaps in every stream; each drive keeps its own local OXTS origin, as the
    KITTI runners treat pooled drives.
    """

    roots = (
        [Path(drive_dirs)] if isinstance(drive_dirs, (str, Path)) else [Path(p) for p in drive_dirs]
    )
    if not roots:
        raise DatasetError("at least one KITTI drive is required")
    calibration_root = Path(calibration_dir) if calibration_dir is not None else roots[0].parent
    calibration_path = calibration_root / "calib_imu_to_velo.txt"
    loaded = []
    for root in roots:
        _check_drive(root)
        try:
            loaded.append(
                (root, load_kitti_ins_lidar_drive(root, calibration_dir=calibration_root))
            )
        except ValueError as exc:
            raise DatasetError(str(exc)) from exc
    loaded.sort(key=lambda item: float(item[1].trajectory.times_s[0]))
    vendor = loaded[0][1].vendor_t_imu_lidar  # one calibration file: identical for every drive
    digests = [drive.input_sha256 for _root, drive in loaded]
    combined_digest = hashlib.sha256("".join(digests).encode("ascii")).hexdigest()

    stamps: list[tuple[dict[int, int], dict[int, int]]] = []
    for root, _drive in loaded:
        stamps.append(
            (
                {
                    item.index: item.timestamp_ns
                    for item in read_timestamps(root / "oxts" / "timestamps.txt")
                },
                {
                    item.index: item.timestamp_ns
                    for item in read_timestamps(root / "velodyne_points" / "timestamps.txt")
                },
            )
        )
    first_ns = min(
        oxts_ns[drive.trajectory.frame_indices[0]]
        for (_root, drive), (oxts_ns, _velodyne_ns) in zip(loaded, stamps, strict=True)
    )
    lever = vendor[:3, 3]
    tf_static: list[Transform] = [
        (BASE_FRAME, IMU_FRAME, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        (
            IMU_FRAME,
            LIDAR_FRAME,
            (float(lever[0]), float(lever[1]), float(lever[2])),
            _quat(vendor[:3, :3]),
        ),
    ]
    provenance: dict[str, Any] = {
        "schema_version": CONVERSION_SCHEMA,
        "generator": "calibrex convert kitti-raw",
        "generator_version": __version__,
        "git_commit": git_commit(),
        "command": list(command or []),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset_family": "kitti_raw",
            "drives": [
                {
                    "drive": root.name,
                    "input_sha256": drive.input_sha256,
                    "missing_velodyne_indices": list(drive.missing_velodyne_indices),
                }
                for root, drive in loaded
            ],
            "input_sha256": combined_digest,
            "input_digest_scope": "per drive: calib_imu_to_velo.txt, OXTS and Velodyne "
            "timestamps, every OXTS packet and every Velodyne sweep "
            "(calibrex.data.kitti_ins_lidar); the combined digest hashes the per-drive digests "
            "in time order",
            "calib_imu_to_velo_sha256": _file_sha256(calibration_path),
            "license": KITTI_LICENSE,
        },
        "frames": {
            "base_link": "defined as the OXTS/IMU frame (KITTI convention; no separate vehicle "
            "frame is published). base_link -> imu_link is the identity.",
            "imu_link": "OXTS unit; x forward, y left, z up (KITTI IMU/GPS frame)",
            "velo_link": "Velodyne HDL-64E; tf imu_link -> velo_link is the inverse of "
            "calib_imu_to_velo.txt",
            "odom": "ENU-like local frame of the OXTS trajectory: Mercator projection of the "
            "first packet's latitude, origin at the first packet of each drive "
            "(kitti_oxts_pose_world_imu)",
        },
        "topics": {
            LIDAR_TOPIC: "PointCloud2 velo_link: x y z intensity (float32); no per-point time "
            "field because KITTI raw sweeps have none (nothing was synthesized)",
            IMU_TOPIC: "Imu imu_link: OXTS body-frame wx wy wz and ax ay az; no orientation",
            FIX_TOPIC: "NavSatFix imu_link: OXTS lat/lon/alt; the GNSS position solution is "
            "taken at the OXTS unit (antenna lever arm not published)",
            ODOMETRY_TOPIC: "Odometry odom->imu_link: INS pose; twist = body-frame velocity and "
            "angular rate (OXTS vf,vl,vu / wf,wl,wu rotated from the level frame by roll, pitch)",
            TWIST_TOPIC: "TwistStamped base_link: the same body-frame velocity and rate as a "
            "wheel-odometry proxy (KITTI has no wheel odometry); declare with "
            "--topic-kind /oxts/twist=wheel",
            TF_STATIC_TOPIC: "TFMessage transient-local: base_link->imu_link (identity), "
            "imu_link->velo_link (T_imu_velo)",
        },
    }
    counts = dict.fromkeys((LIDAR_TOPIC, IMU_TOPIC, FIX_TOPIC, ODOMETRY_TOPIC, TWIST_TOPIC), 0)
    with Rosbag2Writer(
        output,
        overwrite=overwrite,
        custom_data={
            "generator": f"calibrex convert kitti-raw {__version__}",
            "source_drives": ",".join(root.name for root, _drive in loaded),
            "source_input_sha256": combined_digest,
            "base_link": "OXTS/IMU frame (KITTI convention)",
        },
    ) as writer:
        writer.add_topic(TF_STATIC_TOPIC, "tf2_msgs/msg/TFMessage", latched=True)
        writer.write(
            TF_STATIC_TOPIC,
            first_ns,
            encode_tf_message(tf_static, secs=first_ns // 10**9, nsecs=first_ns % 10**9),
        )
        for topic, message_type in (
            (LIDAR_TOPIC, "sensor_msgs/msg/PointCloud2"),
            (IMU_TOPIC, "sensor_msgs/msg/Imu"),
            (FIX_TOPIC, "sensor_msgs/msg/NavSatFix"),
            (ODOMETRY_TOPIC, "nav_msgs/msg/Odometry"),
            (TWIST_TOPIC, "geometry_msgs/msg/TwistStamped"),
        ):
            writer.add_topic(topic, message_type)
        for (root, drive), (oxts_ns, velodyne_ns) in zip(loaded, stamps, strict=True):
            _write_drive(writer, root, drive, oxts_ns, velodyne_ns, counts)
    counts[TF_STATIC_TOPIC] = 1
    provenance["message_counts"] = counts
    sidecar = Path(output) / CONVERSION_FILENAME
    sidecar.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return KittiConversionResult(Path(output), sidecar, counts, provenance)


def _write_drive(
    writer: Rosbag2Writer,
    root: Path,
    drive: KittiInsLidarDrive,
    oxts_ns: dict[int, int],
    velodyne_ns: dict[int, int],
    counts: dict[str, int],
) -> None:
    trajectory = drive.trajectory
    oxts_files = {int(path.stem): path for path in (root / "oxts" / "data").glob("*.txt")}
    assert trajectory.quaternions_xyzw is not None
    for position, index in enumerate(trajectory.frame_indices):
        stamp = oxts_ns[index]
        values = np.loadtxt(oxts_files[index])
        pose = trajectory.poses[position]
        qx, qy, qz, qw = (float(v) for v in trajectory.quaternions_xyzw[position])
        orientation = (qx, qy, qz, qw)  # the quaternion the pose matrix was built from
        velocity, rate = oxts_body_frame_motion(values)
        writer.write(
            IMU_TOPIC,
            stamp,
            encode_imu(
                frame_id=IMU_FRAME,
                timestamp_ns=stamp,
                angular_velocity=(float(values[17]), float(values[18]), float(values[19])),
                linear_acceleration=(float(values[11]), float(values[12]), float(values[13])),
            ),
        )
        accuracy = float(values[23]) if len(values) > 23 else 0.0
        variance = accuracy * accuracy
        writer.write(
            FIX_TOPIC,
            stamp,
            encode_navsatfix(
                frame_id=IMU_FRAME,
                secs=stamp // 10**9,
                nsecs=stamp % 10**9,
                status=0,
                service=1,
                latitude=float(values[0]),
                longitude=float(values[1]),
                altitude=float(values[2]),
                covariance=(variance, 0.0, 0.0, 0.0, variance, 0.0, 0.0, 0.0, variance),
                covariance_type=2 if variance > 0.0 else 0,
            ),
        )
        writer.write(
            ODOMETRY_TOPIC,
            stamp,
            encode_odometry(
                frame_id=ODOM_FRAME,
                child_frame_id=IMU_FRAME,
                timestamp_ns=stamp,
                position=(float(pose[0, 3]), float(pose[1, 3]), float(pose[2, 3])),
                orientation_xyzw=orientation,
                linear_velocity=(float(velocity[0]), float(velocity[1]), float(velocity[2])),
                angular_velocity=(float(rate[0]), float(rate[1]), float(rate[2])),
            ),
        )
        writer.write(
            TWIST_TOPIC,
            stamp,
            encode_twist_stamped(
                frame_id=BASE_FRAME,
                timestamp_ns=stamp,
                linear=(float(velocity[0]), float(velocity[1]), float(velocity[2])),
                angular=(float(rate[0]), float(rate[1]), float(rate[2])),
            ),
        )
        for topic in (IMU_TOPIC, FIX_TOPIC, ODOMETRY_TOPIC, TWIST_TOPIC):
            counts[topic] += 1
    for frame in drive.velodyne_frames:
        stamp = velodyne_ns[frame.frame_index]
        points = np.fromfile(frame.path, dtype="<f4").reshape(-1, 4)
        writer.write(
            LIDAR_TOPIC,
            stamp,
            encode_pointcloud2(
                frame_id=LIDAR_FRAME,
                timestamp_ns=stamp,
                fields=CLOUD_FIELDS,
                point_step=16,
                data=np.ascontiguousarray(points).tobytes(),
            ),
        )
        counts[LIDAR_TOPIC] += 1
