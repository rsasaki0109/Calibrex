"""RTK-SLAM dataset (University of Stuttgart) GNSS tracks and Livox scans.

The dataset pairs a Livox MID360 with an RTK GNSS receiver on a hand-held
rig.  ``rtk.txt`` lists ``timestamp lat lon height status blt_std`` at 10 Hz;
``calib.yaml`` gives ``T_lidar_imu`` and the CAD offset of the GNSS antenna
phase centre from the IMU origin.

``T_lidar_imu`` in that file maps LiDAR-frame points into the IMU frame: its
translation is the negative of the IMU position in the LiDAR frame that the
MID360 manual lists (11.0, 23.29, -44.12 mm).  The reference lever arm (the
antenna phase centre in the LiDAR frame) follows from that reading and is an
evaluation-only CAD value, not metrology.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
import yaml
from numpy.typing import NDArray

from calibrex.data import ros_cdr
from calibrex.data.rosbag2 import iter_messages

FloatArray: TypeAlias = NDArray[np.float64]

_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)
RTK_FIXED_STATUS = 4
_BAG_DIGEST_PREFIX_BYTES = 64 << 20


@dataclass(frozen=True)
class GnssTrack:
    """RTK antenna positions in a local ENU frame (metres, seconds)."""

    times_s: FloatArray
    enu_m: FloatArray
    sigma_m: FloatArray
    origin_lat_lon_height: tuple[float, float, float]
    rejected_epochs: int


def geodetic_to_ecef(lat_deg: FloatArray, lon_deg: FloatArray, height_m: FloatArray) -> FloatArray:
    """Convert WGS84 geodetic coordinates to ECEF."""

    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    normal = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * np.sin(lat) ** 2)
    return np.column_stack(
        [
            (normal + height_m) * np.cos(lat) * np.cos(lon),
            (normal + height_m) * np.cos(lat) * np.sin(lon),
            (normal * (1.0 - _WGS84_E2) + height_m) * np.sin(lat),
        ]
    )


def geodetic_to_enu(
    lat_deg: FloatArray,
    lon_deg: FloatArray,
    height_m: FloatArray,
    origin: tuple[float, float, float],
) -> FloatArray:
    """Convert WGS84 geodetic coordinates to east-north-up about ``origin``."""

    ecef = geodetic_to_ecef(lat_deg, lon_deg, height_m)
    origin_ecef = geodetic_to_ecef(
        np.array([origin[0]]), np.array([origin[1]]), np.array([origin[2]])
    )[0]
    lat0 = math.radians(origin[0])
    lon0 = math.radians(origin[1])
    rotation = np.array(
        [
            [-math.sin(lon0), math.cos(lon0), 0.0],
            [-math.sin(lat0) * math.cos(lon0), -math.sin(lat0) * math.sin(lon0), math.cos(lat0)],
            [math.cos(lat0) * math.cos(lon0), math.cos(lat0) * math.sin(lon0), math.sin(lat0)],
        ]
    )
    return (ecef - origin_ecef) @ rotation.T


def load_rtk_track(
    path: str | Path,
    *,
    fixed_only: bool = True,
    sigma_floor_m: float = 0.01,
    origin: tuple[float, float, float] | None = None,
) -> GnssTrack:
    """Read ``rtk.txt`` and keep RTK-fixed epochs by default.

    ``origin`` sets the ENU origin; it defaults to the first kept epoch.  Pass
    one origin to pool several sequences in a common frame.
    """

    table = np.loadtxt(Path(path), comments="#", ndmin=2)
    if table.shape[1] < 6:
        raise ValueError(f"{path} needs columns: timestamp lat lon height status blt_std")
    keep = table[:, 4] == RTK_FIXED_STATUS if fixed_only else np.ones(len(table), dtype=bool)
    kept = table[keep]
    if len(kept) < 2:
        raise ValueError(f"{path} has fewer than two usable GNSS epochs")
    order = np.argsort(kept[:, 0])
    kept = kept[order]
    if origin is None:
        origin = (float(kept[0, 1]), float(kept[0, 2]), float(kept[0, 3]))
    return GnssTrack(
        times_s=kept[:, 0],
        enu_m=geodetic_to_enu(kept[:, 1], kept[:, 2], kept[:, 3], origin),
        sigma_m=np.maximum(kept[:, 5], sigma_floor_m),
        origin_lat_lon_height=origin,
        rejected_epochs=int(np.count_nonzero(~keep)),
    )


def rtk_slam_reference_lever_arm(calib_path: str | Path) -> FloatArray:
    """Return the CAD antenna phase centre in the LiDAR frame (metres)."""

    calibration = yaml.safe_load(Path(calib_path).read_text(encoding="utf-8"))
    lidar_to_imu = np.array(calibration["lidar0"]["T_lidar_imu"], dtype=np.float64)
    antenna_in_imu = np.array(
        calibration["reference_offsets"]["gnss_antenna_phase_center"], dtype=np.float64
    )
    imu_to_lidar = np.linalg.inv(lidar_to_imu)
    return imu_to_lidar[:3, :3] @ antenna_in_imu + imu_to_lidar[:3, 3]


def livox_scan_count(bag_dir: str | Path, topic: str = "/livox/points") -> int:
    """Return the number of point-cloud messages declared in ``metadata.yaml``."""

    metadata = yaml.safe_load((Path(bag_dir) / "metadata.yaml").read_text(encoding="utf-8"))
    info = metadata["rosbag2_bagfile_information"]
    for entry in info["topics_with_message_count"]:
        if entry["topic_metadata"]["name"] == topic:
            return int(entry["message_count"])
    raise ValueError(f"{bag_dir} has no topic {topic}")


def iter_livox_scans(
    bag_dir: str | Path, topic: str = "/livox/points"
) -> Iterator[tuple[float, FloatArray]]:
    """Yield ``(header time in seconds, (N, 3) points)`` in bag order."""

    for _, timestamp_ns, payload in iter_messages(bag_dir, topics={topic}):
        cloud = ros_cdr.decode_ros2_pointcloud2(topic, timestamp_ns, payload)
        yield cloud.timestamp_ns * 1.0e-9, np.asarray(cloud.xyz, dtype=np.float64)


def rtk_slam_input_digest(
    bag_dir: str | Path, rtk_path: str | Path, calib_path: str | Path
) -> tuple[str, str]:
    """Digest the inputs and name the scope of what the digest covers.

    The GNSS track, calibration, and bag metadata are hashed completely.  The
    tens-of-gigabyte bag database is represented by its size and its first
    64 MiB, and the returned scope string says so.
    """

    digest = hashlib.sha256()
    for path in (Path(rtk_path), Path(calib_path), Path(bag_dir) / "metadata.yaml"):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    databases = sorted(Path(bag_dir).glob("*.db3")) + sorted(Path(bag_dir).glob("*.mcap"))
    for database in databases:
        digest.update(database.name.encode("utf-8"))
        digest.update(str(database.stat().st_size).encode("ascii"))
        with database.open("rb") as stream:
            digest.update(stream.read(_BAG_DIGEST_PREFIX_BYTES))
    scope = "rtk.txt, calib.yaml, and metadata.yaml in full; bag databases by size and first 64 MiB"
    return digest.hexdigest(), scope
