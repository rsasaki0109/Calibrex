"""Livox LiDAR scans and built-in IMU samples from ROS 2 bags, without ROS.

Livox drivers and recordings disagree on conventions, so a
:class:`LivoxStreamProfile` states them explicitly per dataset:

* the per-point time field and whether it holds an offset after the header
  stamp (seconds), an absolute time in nanoseconds, or an absolute time in
  seconds (for example a Hesai PandarXT ``timestamp`` float64); and
* whether the IMU reports acceleration in m/s^2 or in g.

The MID360 manual places the built-in IMU at (11.0, 23.29, -44.12) mm in the
LiDAR frame with axes aligned to it; :data:`MID360_T_LIDAR_IMU` encodes that
design value as a reference, not a measurement.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.progress import emit_stage, emit_tick
from calibrex.data import ros_cdr
from calibrex.data.rosbag2 import iter_messages

FloatArray: TypeAlias = NDArray[np.float64]
STANDARD_GRAVITY_MPS2 = 9.80665

MID360_T_LIDAR_IMU: FloatArray = np.array(
    [
        [1.0, 0.0, 0.0, 0.011],
        [0.0, 1.0, 0.0, 0.02329],
        [0.0, 0.0, 1.0, -0.04412],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


@dataclass(frozen=True)
class LivoxStreamProfile:
    """Topic names and unit conventions of one Livox recording."""

    name: str
    point_topic: str
    imu_topic: str
    point_time_field: str | None
    point_time_encoding: Literal["offset_s", "absolute_ns", "absolute_s"]
    acceleration_unit: Literal["mps2", "g"]


LIVOX_PROFILES: dict[str, LivoxStreamProfile] = {
    "rtk-slam": LivoxStreamProfile(
        name="rtk-slam",
        point_topic="/livox/points",
        imu_topic="/livox/imu",
        point_time_field="offset_time",
        point_time_encoding="offset_s",
        acceleration_unit="mps2",
    ),
    "livox-ros-driver2": LivoxStreamProfile(
        name="livox-ros-driver2",
        point_topic="/livox/lidar",
        imu_topic="/livox/imu",
        point_time_field="timestamp",
        point_time_encoding="absolute_ns",
        acceleration_unit="g",
    ),
}


@dataclass(frozen=True)
class ImuSamples:
    """Time-sorted IMU samples: seconds, rad/s, m/s^2."""

    times_s: FloatArray
    gyro_rps: FloatArray
    accel_mps2: FloatArray


def resolve_profile(profile: str | LivoxStreamProfile) -> LivoxStreamProfile:
    """Return a named built-in profile, or pass a custom profile through."""

    if isinstance(profile, LivoxStreamProfile):
        return profile
    return LIVOX_PROFILES[profile]


def iter_livox_points(
    bag_dir: str | Path,
    profile: LivoxStreamProfile,
    *,
    max_seconds: float | None = None,
) -> Iterator[tuple[float, FloatArray, FloatArray | None]]:
    """Yield ``(header time s, (N, 3) points, per-point offsets s or None)``.

    ``max_seconds`` stops after the scans within that many seconds of the
    first scan's header time.
    """

    first_s: float | None = None
    messages = iter_messages(bag_dir, topics={profile.point_topic})
    for scans_read, (_, timestamp_ns, payload) in enumerate(messages, start=1):
        cloud = ros_cdr.decode_ros2_pointcloud2(
            profile.point_topic,
            timestamp_ns,
            payload,
            point_time_field=profile.point_time_field,
        )
        offsets = cloud.point_time_offsets_s
        if offsets is not None:
            raw = np.asarray(offsets, dtype=np.float64)
            if profile.point_time_encoding == "absolute_ns":
                offsets = (raw - float(cloud.timestamp_ns)) * 1.0e-9
            elif profile.point_time_encoding == "absolute_s":
                offsets = raw - cloud.timestamp_ns * 1.0e-9
            else:
                offsets = raw
        header_s = cloud.timestamp_ns * 1.0e-9
        first_s = header_s if first_s is None else first_s
        if max_seconds is not None and header_s - first_s > max_seconds:
            break
        emit_tick(scans_read)
        yield header_s, np.asarray(cloud.xyz, dtype=np.float64), offsets


LivoxScan: TypeAlias = tuple[float, FloatArray, "FloatArray | None"]


class ScanStore:
    """Decode a bag's scans and IMU once and replay them from memory.

    The IMU-LiDAR estimators read the same scans several times (an odometry pass
    per gyro-deskew iteration, the feedback check, the lever-arm pass). Passing
    one store to all of them removes the repeated bag read and CDR decode; the
    replayed values are bit-identical to a fresh read. Scans are held as decoded
    (float32 when that is lossless) up to ``budget_bytes``; beyond that, or when a
    consumer stops early, the stream is simply read from the bag again.
    """

    def __init__(self, budget_bytes: int = 2 << 30) -> None:
        self.budget_bytes = budget_bytes
        self.bytes_held = 0
        self.hits = 0
        self.reads = 0
        self._scans: dict[tuple[object, ...], list[LivoxScan]] = {}
        self._oversize: set[tuple[object, ...]] = set()
        self._imu: dict[tuple[object, ...], ImuSamples] = {}

    def scans(
        self,
        bag_dir: str | Path,
        profile: LivoxStreamProfile,
        *,
        max_seconds: float | None = None,
    ) -> Iterator[LivoxScan]:
        """Yield what :func:`iter_livox_points` yields, replayed from memory when held."""

        key: tuple[object, ...] = (
            str(bag_dir),
            profile.point_topic,
            profile.point_time_field,
            profile.point_time_encoding,
            max_seconds,
        )
        held = self._scans.get(key)
        if held is not None:
            self.hits += 1
            emit_stage("replaying decoded scans from memory")
            for replayed, (header_s, xyz, offsets) in enumerate(held, start=1):
                emit_tick(replayed, len(held))
                yield header_s, np.asarray(xyz, dtype=np.float64), offsets
            return
        self.reads += 1
        collected: list[LivoxScan] | None = None if key in self._oversize else []
        size = 0
        complete = False
        try:
            for header_s, xyz, offsets in iter_livox_points(
                bag_dir, profile, max_seconds=max_seconds
            ):
                if collected is not None:
                    packed = _pack_points(xyz)
                    size += packed.nbytes + (0 if offsets is None else offsets.nbytes)
                    if self.bytes_held + size > self.budget_bytes:
                        collected = None
                        self._oversize.add(key)
                    else:
                        collected.append((header_s, packed, offsets))
                yield header_s, xyz, offsets
            complete = True
        finally:
            if complete and collected is not None:
                self._scans[key] = collected
                self.bytes_held += size

    def imu(self, bag_dir: str | Path, profile: LivoxStreamProfile) -> ImuSamples:
        """Return the IMU samples of a bag, reading them once."""

        key: tuple[object, ...] = (str(bag_dir), profile.imu_topic, profile.acceleration_unit)
        samples = self._imu.get(key)
        if samples is None:
            samples = load_livox_imu(bag_dir, profile)
            self._imu[key] = samples
        return samples


def _pack_points(xyz: FloatArray) -> FloatArray:
    """Store points as float32 when that reproduces them exactly (it does for decoded clouds)."""

    narrow = xyz.astype(np.float32)
    if np.array_equal(narrow.astype(np.float64), xyz):
        return narrow
    return xyz


def load_livox_imu(bag_dir: str | Path, profile: LivoxStreamProfile) -> ImuSamples:
    """Read every IMU sample of a bag into SI units."""

    times: list[float] = []
    gyro: list[tuple[float, float, float]] = []
    accel: list[tuple[float, float, float]] = []
    for _, timestamp_ns, payload in iter_messages(bag_dir, topics={profile.imu_topic}):
        message = ros_cdr.decode_ros2_imu(profile.imu_topic, timestamp_ns, payload)
        times.append(message.timestamp_ns * 1.0e-9)
        gyro.append(message.angular_velocity)
        accel.append(message.linear_acceleration)
    if len(times) < 2:
        raise ValueError(f"{bag_dir} has fewer than two IMU samples on {profile.imu_topic}")
    order = np.argsort(times)
    scale = STANDARD_GRAVITY_MPS2 if profile.acceleration_unit == "g" else 1.0
    return ImuSamples(
        times_s=np.asarray(times, dtype=np.float64)[order],
        gyro_rps=np.asarray(gyro, dtype=np.float64)[order],
        accel_mps2=np.asarray(accel, dtype=np.float64)[order] * scale,
    )


def bag_input_digest(bag_dirs: Sequence[str | Path]) -> tuple[str, str]:
    """Digest ROS 2 bags by full metadata plus each database's size and first 64 MiB."""

    digest = hashlib.sha256()
    for bag_dir in bag_dirs:
        root = Path(bag_dir)
        digest.update(root.name.encode("utf-8"))
        digest.update((root / "metadata.yaml").read_bytes())
        for database in sorted(root.glob("*.db3")) + sorted(root.glob("*.mcap")):
            digest.update(database.name.encode("utf-8"))
            digest.update(str(database.stat().st_size).encode("ascii"))
            with database.open("rb") as stream:
                digest.update(stream.read(64 << 20))
    return digest.hexdigest(), "bag metadata.yaml in full; bag databases by size and first 64 MiB"
