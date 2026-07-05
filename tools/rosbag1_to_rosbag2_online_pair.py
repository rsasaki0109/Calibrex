#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.24",
#   "rosbags>=0.10",
#   "kiss-icp>=1.3",
# ]
# ///
"""Convert a ROS 1 bag into a rosbag2 online-calibration pair.

This standalone data-prep tool is **not** part of the slac package. It reads a
ROS 1 ``.bag`` with ``rosbags``, writes a rosbag2 directory containing:

* ``sensor_msgs/msg/PointCloud2`` topics passed through unchanged in content
  (ROS 1 payloads are deserialized and re-serialized as ROS 2 CDR).
* A synthesized ``nav_msgs/msg/Odometry`` topic, either:

  * ``--odom-source pose-topic`` (default): built 1:1 from each
    ``geometry_msgs/PoseStamped`` on ``--pose-topic``. The pose trajectory is
    real upstream data (for TIERS Indoor02, VRPN MOCAP). **Only the Odometry
    message envelope is synthesized here**; slac never authors ``/odom``
    during calibration.
  * ``--odom-source kiss-icp``: estimated from the source LiDAR itself via
    KISS-ICP on ``--kiss-icp-topic``. Poses are ``T_world_sensor`` in the
    source sensor frame (rig-frame odometry; no external body alignment
    needed). ``kiss-icp`` is imported lazily inside that branch; PEP 723 lists
    it so ``uv run`` installs it automatically.
  * ``--odom-source kiss-icp-two-pass``: pass 1 runs KISS-ICP on raw scans;
    each scan is rigidified to its message stamp with the pass-1 track and
    per-point ``--kiss-icp-time-field`` offsets; pass 2 runs a fresh KISS-ICP
    instance on the deskewed scans. Mutually exclusive with
    ``--kiss-icp-native-deskew``.
  * ``--kiss-icp-native-deskew``: single-pass KISS-ICP with upstream-native
    deskew (``cfg.data.deskew = True`` and normalized per-point timestamps).
    Only valid with ``--odom-source kiss-icp``.

``--restamp-topic`` (repeatable) rewrites each listed topic's message
``header.stamp`` by a constant per-topic offset: the median of
``bag_receive_time - header_stamp`` over that topic's messages (two-pass:
header-only scan, then conversion). This moves a sensor's since-boot or
misaligned clock into the recording clock domain while preserving the
sensor's own relative timing (receive jitter is not injected). The offset
includes mean transport/assembly latency, so restamped absolute stamps carry
a bias of that order; residual timing error is roughly latency x platform
speed. Only list topics that need correction (for example Ouster OS1); do
not restamp topics already on the recording epoch clock unless you intend to.

``--duplicate-topic SRC:DST`` (repeatable) mirrors every PointCloud2 message
written for ``SRC`` onto a second rosbag2 connection ``DST`` (same type, same
bag timestamps). Use this to build identity self-consistency control bags:
duplicate a LiDAR topic, then calibrate ``SRC`` (root) vs ``DST`` with an
identity initial transform — ground truth is exactly identity.

Install with ``uv run`` (PEP 723 deps) or ``pip install kiss-icp`` for the
kiss-icp odometry mode.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import numpy as np
from lidar_scan_deskew import (
    PoseTrackSample,
    normalize_scan_timestamps,
    rigidify_scan_to_timestamp,
)
from rosbags.highlevel import AnyReader
from rosbags.rosbag2 import (
    CompressionFormat,
    CompressionMode,
    StoragePlugin,
    Writer,
)
from rosbags.typesys import Stores, get_typestore

POINTCLOUD2_MSGTYPE = "sensor_msgs/msg/PointCloud2"
ODOMETRY_MSGTYPE = "nav_msgs/msg/Odometry"
POSE_STAMPED_MSGTYPE = "geometry_msgs/msg/PoseStamped"
IMU_MSGTYPE = "sensor_msgs/msg/Imu"

POINTFIELD_NUMPY = {
    1: "i1",
    2: "u1",
    3: "i2",
    4: "u2",
    5: "i4",
    6: "u4",
    7: "f4",
    8: "f8",
}


@dataclass(frozen=True)
class TopicStats:
    """Per-topic message count and timestamp span."""

    message_count: int
    first_timestamp_ns: int | None
    last_timestamp_ns: int | None


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="Source ROS 1 .bag path")
    parser.add_argument("--dst", type=Path, required=True, help="Destination rosbag2 directory")
    parser.add_argument(
        "--topic",
        action="append",
        default=[],
        dest="topics",
        help="PointCloud2 topic to pass through (repeatable)",
    )
    parser.add_argument(
        "--odom-source",
        choices=("pose-topic", "kiss-icp", "kiss-icp-two-pass"),
        default="pose-topic",
        help="Odometry synthesis mode (default: pose-topic)",
    )
    parser.add_argument(
        "--pose-topic",
        help="PoseStamped topic converted 1:1 into --odom-topic (required for pose-topic mode)",
    )
    parser.add_argument(
        "--kiss-icp-topic",
        help="PointCloud2 topic fed to KISS-ICP (required for kiss-icp mode)",
    )
    parser.add_argument(
        "--kiss-icp-max-range",
        type=float,
        default=30.0,
        help="KISS-ICP cfg.data.max_range in meters (default: 30.0)",
    )
    parser.add_argument(
        "--kiss-icp-time-field",
        default="time",
        help="PointCloud2 per-point time field for two-pass deskew (default: time)",
    )
    parser.add_argument(
        "--kiss-icp-native-deskew",
        action="store_true",
        help=(
            "Single-pass KISS-ICP with cfg.data.deskew=True and normalized "
            "per-point timestamps (mutually exclusive with kiss-icp-two-pass)"
        ),
    )
    parser.add_argument(
        "--odom-topic",
        default="/odom",
        help="Synthesized nav_msgs/msg/Odometry topic (default: /odom)",
    )
    parser.add_argument(
        "--child-frame-id",
        default="base_link",
        help="child_frame_id for synthesized Odometry messages",
    )
    parser.add_argument(
        "--storage",
        choices=("sqlite3", "mcap"),
        default="sqlite3",
        help="rosbag2 storage backend (default: sqlite3)",
    )
    parser.add_argument(
        "--compress",
        choices=("none", "zstd"),
        default="none",
        help="rosbag2 compression (default: none)",
    )
    parser.add_argument(
        "--restamp-topic",
        action="append",
        default=[],
        dest="restamp_topics",
        help=(
            "Rewrite header.stamp on this topic by median(bag_receive_time - "
            "header_stamp); repeatable"
        ),
    )
    parser.add_argument(
        "--duplicate-topic",
        action="append",
        default=[],
        dest="duplicate_topics",
        metavar="SRC:DST",
        help=(
            "Mirror PointCloud2 messages from SRC onto DST (same stamps); "
            "repeatable; for identity self-consistency control bags"
        ),
    )
    parser.add_argument(
        "--imu-topic",
        action="append",
        default=[],
        dest="imu_topics",
        help="sensor_msgs/Imu topic to pass through (repeatable)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.odom_source == "pose-topic" and not args.pose_topic:
        parser.error("--pose-topic is required when --odom-source pose-topic")
    kiss_modes = {"kiss-icp", "kiss-icp-two-pass"}
    if args.odom_source in kiss_modes and not args.kiss_icp_topic:
        parser.error(f"--kiss-icp-topic is required when --odom-source {args.odom_source}")
    if args.odom_source == "kiss-icp-two-pass" and args.kiss_icp_native_deskew:
        parser.error("--kiss-icp-native-deskew is mutually exclusive with kiss-icp-two-pass")
    if args.kiss_icp_native_deskew and args.odom_source != "kiss-icp":
        parser.error("--kiss-icp-native-deskew requires --odom-source kiss-icp")
    return args


def _parse_duplicate_topics(
    entries: list[str],
    *,
    pointcloud_topics: set[str],
    reserved_topics: set[str],
) -> dict[str, str]:
    """Parse ``SRC:DST`` pairs and validate source/destination topics."""

    duplicate_by_source: dict[str, str] = {}
    destinations: set[str] = set()
    for entry in entries:
        if ":" not in entry:
            msg = f"--duplicate-topic must be SRC:DST, got {entry!r}"
            raise SystemExit(msg)
        src, dst = entry.split(":", 1)
        if not src or not dst:
            msg = f"--duplicate-topic must be SRC:DST, got {entry!r}"
            raise SystemExit(msg)
        if src not in pointcloud_topics:
            msg = (
                f"--duplicate-topic source {src!r} must be listed in --topic "
                f"(got {sorted(pointcloud_topics)})"
            )
            raise SystemExit(msg)
        if dst in reserved_topics or dst in destinations:
            msg = f"--duplicate-topic destination {dst!r} collides with an existing topic"
            raise SystemExit(msg)
        if src in duplicate_by_source:
            msg = f"duplicate --duplicate-topic source {src!r}"
            raise SystemExit(msg)
        duplicate_by_source[src] = dst
        destinations.add(dst)
    return duplicate_by_source


def _storage_plugin(name: str) -> StoragePlugin:
    if name == "mcap":
        return StoragePlugin.MCAP
    return StoragePlugin.SQLITE3


def _normalize_quaternion_xyzw(
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x, y, z, w = quaternion
    norm = sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        msg = "quaternion norm must be non-zero"
        raise ValueError(msg)
    return (x / norm, y / norm, z / norm, w / norm)


def _quaternion_xyzw_from_rotation_matrix(matrix: np.ndarray) -> tuple[float, float, float, float]:
    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (m21 - m12) / scale
        y = (m02 - m20) / scale
        z = (m10 - m01) / scale
    elif m00 > m11 and m00 > m22:
        scale = sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / scale
        x = 0.25 * scale
        y = (m01 + m10) / scale
        z = (m02 + m20) / scale
    elif m11 > m22:
        scale = sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / scale
        x = (m01 + m10) / scale
        y = 0.25 * scale
        z = (m12 + m21) / scale
    else:
        scale = sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / scale
        x = (m02 + m20) / scale
        y = (m12 + m21) / scale
        z = 0.25 * scale
    return _normalize_quaternion_xyzw((x, y, z, w))


def _pointcloud2_xyz(pc_msg: object) -> np.ndarray:
    """Decode xyz float64 (N, 3) from a deserialized PointCloud2 message."""

    fields = [
        (field.name, int(field.offset), int(field.datatype), int(field.count))
        for field in pc_msg.fields
    ]
    by_name = {name: (offset, datatype, count) for name, offset, datatype, count in fields}
    required = ("x", "y", "z")
    if not all(name in by_name for name in required):
        msg = "PointCloud2 message does not carry x/y/z fields"
        raise ValueError(msg)

    byteorder = ">" if bool(pc_msg.is_bigendian) else "<"
    point_step = int(pc_msg.point_step)
    height = int(pc_msg.height)
    width = int(pc_msg.width)
    payload = bytes(pc_msg.data)
    point_count = height * width if height * width else (len(payload) // point_step)

    names: list[str] = []
    formats: list[str] = []
    offsets: list[int] = []
    for name in required:
        offset, datatype, _count = by_name[name]
        base = POINTFIELD_NUMPY.get(datatype)
        if base is None:
            msg = f"unsupported PointField datatype {datatype} for {name}"
            raise ValueError(msg)
        names.append(name)
        formats.append(byteorder + base)
        offsets.append(offset)

    dtype = np.dtype(
        {"names": names, "formats": formats, "offsets": offsets, "itemsize": point_step}
    )
    structured = np.frombuffer(payload, dtype=dtype, count=point_count)
    xyz = np.column_stack(
        (
            structured["x"].astype(np.float64, copy=False),
            structured["y"].astype(np.float64, copy=False),
            structured["z"].astype(np.float64, copy=False),
        )
    )
    return np.ascontiguousarray(xyz, dtype=np.float64)


def _pointcloud2_field(pc_msg: object, field_name: str) -> np.ndarray:
    """Decode one named PointCloud2 field as float64 (N,)."""

    fields = [
        (field.name, int(field.offset), int(field.datatype), int(field.count))
        for field in pc_msg.fields
    ]
    by_name = {name: (offset, datatype, count) for name, offset, datatype, count in fields}
    if field_name not in by_name:
        available = sorted(by_name)
        msg = f"PointCloud2 missing field {field_name!r}; available: {available}"
        raise ValueError(msg)

    byteorder = ">" if bool(pc_msg.is_bigendian) else "<"
    point_step = int(pc_msg.point_step)
    height = int(pc_msg.height)
    width = int(pc_msg.width)
    payload = bytes(pc_msg.data)
    point_count = height * width if height * width else (len(payload) // point_step)

    offset, datatype, _count = by_name[field_name]
    base = POINTFIELD_NUMPY.get(datatype)
    if base is None:
        msg = f"unsupported PointField datatype {datatype} for {field_name}"
        raise ValueError(msg)
    dtype = np.dtype(
        {
            "names": [field_name],
            "formats": [byteorder + base],
            "offsets": [offset],
            "itemsize": point_step,
        }
    )
    structured = np.frombuffer(payload, dtype=dtype, count=point_count)
    return structured[field_name].astype(np.float64, copy=False).reshape(-1)


def _pose_to_odometry(
    pose_msg: object,
    *,
    child_frame_id: str,
    typestore: object,
) -> object:
    PoseWithCovariance = typestore.types["geometry_msgs/msg/PoseWithCovariance"]
    Twist = typestore.types["geometry_msgs/msg/Twist"]
    TwistWithCovariance = typestore.types["geometry_msgs/msg/TwistWithCovariance"]
    Vector3 = typestore.types["geometry_msgs/msg/Vector3"]
    Odometry = typestore.types[ODOMETRY_MSGTYPE]

    covariance = np.zeros(36, dtype=np.float64)
    pose_with_covariance = PoseWithCovariance(pose=pose_msg.pose, covariance=covariance)
    zero_twist = Twist(
        linear=Vector3(x=0.0, y=0.0, z=0.0),
        angular=Vector3(x=0.0, y=0.0, z=0.0),
    )
    twist_with_covariance = TwistWithCovariance(twist=zero_twist, covariance=covariance.copy())
    return Odometry(
        header=pose_msg.header,
        child_frame_id=child_frame_id,
        pose=pose_with_covariance,
        twist=twist_with_covariance,
    )


def _kiss_icp_pose_to_odometry(
    pc_msg: object,
    *,
    pose_matrix: np.ndarray,
    child_frame_id: str,
    frame_id: str,
    typestore: object,
) -> object:
    Pose = typestore.types["geometry_msgs/msg/Pose"]
    Point = typestore.types["geometry_msgs/msg/Point"]
    Quaternion = typestore.types["geometry_msgs/msg/Quaternion"]
    PoseWithCovariance = typestore.types["geometry_msgs/msg/PoseWithCovariance"]
    Twist = typestore.types["geometry_msgs/msg/Twist"]
    TwistWithCovariance = typestore.types["geometry_msgs/msg/TwistWithCovariance"]
    Vector3 = typestore.types["geometry_msgs/msg/Vector3"]
    Odometry = typestore.types[ODOMETRY_MSGTYPE]

    x, y, z, w = _quaternion_xyzw_from_rotation_matrix(pose_matrix[:3, :3])
    pose = Pose(
        position=Point(
            x=float(pose_matrix[0, 3]),
            y=float(pose_matrix[1, 3]),
            z=float(pose_matrix[2, 3]),
        ),
        orientation=Quaternion(x=x, y=y, z=z, w=w),
    )
    covariance = np.zeros(36, dtype=np.float64)
    pose_with_covariance = PoseWithCovariance(pose=pose, covariance=covariance)
    zero_twist = Twist(
        linear=Vector3(x=0.0, y=0.0, z=0.0),
        angular=Vector3(x=0.0, y=0.0, z=0.0),
    )
    twist_with_covariance = TwistWithCovariance(twist=zero_twist, covariance=covariance.copy())
    Header = type(pc_msg.header)
    header = Header(
        seq=pc_msg.header.seq,
        stamp=pc_msg.header.stamp,
        frame_id=frame_id,
    )
    return Odometry(
        header=header,
        child_frame_id=child_frame_id,
        pose=pose_with_covariance,
        twist=twist_with_covariance,
    )


def _create_kiss_icp(max_range: float, *, native_deskew: bool) -> tuple[object, str]:
    import kiss_icp
    from kiss_icp.config import load_config
    from kiss_icp.kiss_icp import KissICP

    cfg = load_config(None)
    cfg.data.max_range = float(max_range)
    cfg.data.deskew = bool(native_deskew)
    if cfg.mapping.voxel_size is None:
        cfg.mapping.voxel_size = cfg.data.max_range / 100.0
    return KissICP(cfg), kiss_icp.__version__


def _collect_pass1_track(
    src: Path,
    *,
    kiss_icp_topic: str,
    max_range: float,
    restamp_offsets_ns: dict[str, int],
    ros1_typestore: object,
) -> tuple[tuple[PoseTrackSample, ...], str]:
    kiss_icp, version = _create_kiss_icp(max_range, native_deskew=False)
    track: list[PoseTrackSample] = []
    with AnyReader([src], default_typestore=ros1_typestore) as reader:
        connections = [
            connection for connection in reader.connections if connection.topic == kiss_icp_topic
        ]
        if not connections:
            msg = f"kiss-icp topic missing from source bag: {kiss_icp_topic}"
            raise SystemExit(msg)
        for _connection, _timestamp, rawdata in reader.messages(connections=connections):
            pc_msg = reader.deserialize(rawdata, _connection.msgtype)
            if kiss_icp_topic in restamp_offsets_ns:
                _apply_header_stamp_offset(pc_msg, restamp_offsets_ns[kiss_icp_topic])
            points = _pointcloud2_xyz(pc_msg)
            kiss_icp.register_frame(points, np.array([]))
            track.append(
                PoseTrackSample(
                    timestamp_ns=_header_stamp_ns(pc_msg),
                    pose=np.array(kiss_icp.last_pose, dtype=np.float64, copy=True),
                )
            )
    return tuple(track), version


def _header_stamp_ns(msg: object) -> int:
    stamp = msg.header.stamp
    nsec = int(stamp.nanosec) if hasattr(stamp, "nanosec") else int(stamp.nsecs)
    return int(stamp.sec) * 1_000_000_000 + nsec


def _apply_header_stamp_offset(msg: object, offset_ns: int) -> None:
    stamp_ns = _header_stamp_ns(msg) + offset_ns
    msg.header.stamp.sec = stamp_ns // 1_000_000_000
    remainder = stamp_ns % 1_000_000_000
    if hasattr(msg.header.stamp, "nanosec"):
        msg.header.stamp.nanosec = remainder
    else:
        msg.header.stamp.nsecs = remainder


def _compute_restamp_offsets(
    src: Path,
    restamp_topics: set[str],
    *,
    ros1_typestore: object,
) -> dict[str, int]:
    offsets_by_topic: dict[str, list[int]] = defaultdict(list)
    with AnyReader([src], default_typestore=ros1_typestore) as reader:
        available = {connection.topic for connection in reader.connections}
        missing = sorted(restamp_topics - available)
        if missing:
            msg = f"restamp topics missing from source bag: {', '.join(missing)}"
            raise SystemExit(msg)
        connections = [
            connection
            for connection in reader.connections
            if connection.topic in restamp_topics
        ]
        for connection, timestamp_ns, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            offsets_by_topic[connection.topic].append(timestamp_ns - _header_stamp_ns(msg))
    return {
        topic: int(np.median(values))
        for topic, values in sorted(offsets_by_topic.items())
        if values
    }


def _format_timestamp_ns(timestamp_ns: int | None) -> str:
    if timestamp_ns is None:
        return "n/a"
    return f"{timestamp_ns} ns"


def _print_summary(
    *,
    src: Path,
    dst: Path,
    odom_topic: str,
    odom_source: str,
    pose_topic: str | None,
    kiss_icp_topic: str | None,
    kiss_icp_version: str | None,
    kiss_icp_pass1_version: str | None,
    kiss_icp_pass2_version: str | None,
    kiss_icp_max_range: float | None,
    kiss_icp_time_field: str | None,
    kiss_icp_native_deskew: bool,
    kiss_icp_passes: int | None,
    stats: dict[str, TopicStats],
    restamp_offsets_ns: dict[str, int] | None = None,
) -> None:
    print(f"source: {src}")
    print(f"destination: {dst}")
    print(f"odom_source: {odom_source}")
    if pose_topic is not None:
        print(f"pose_topic: {pose_topic}")
    if kiss_icp_topic is not None:
        print(f"kiss_icp_topic: {kiss_icp_topic}")
        if kiss_icp_passes is not None:
            print(f"kiss_icp_passes: {kiss_icp_passes}")
        if kiss_icp_time_field is not None:
            print(f"kiss_icp_time_field: {kiss_icp_time_field}")
        print(f"kiss_icp_native_deskew: {kiss_icp_native_deskew}")
        if kiss_icp_pass1_version is not None:
            print(f"kiss_icp_pass1_version: {kiss_icp_pass1_version}")
        if kiss_icp_pass2_version is not None:
            print(f"kiss_icp_pass2_version: {kiss_icp_pass2_version}")
        if kiss_icp_version is not None:
            print(f"kiss_icp_version: {kiss_icp_version}")
        print(f"kiss_icp_max_range: {kiss_icp_max_range}")
    print(f"synthesized_odometry_topic: {odom_topic} (authored by this tool)")
    if restamp_offsets_ns:
        print("restamp_offsets_ns (median bag_receive_time - header_stamp):")
        for topic, offset_ns in sorted(restamp_offsets_ns.items()):
            print(f"  {topic}: {offset_ns} ns ({offset_ns / 1e9:.6f} s)")
    print("topics:")
    for topic in sorted(stats):
        entry = stats[topic]
        span = "n/a"
        if entry.first_timestamp_ns is not None and entry.last_timestamp_ns is not None:
            span = f"{entry.last_timestamp_ns - entry.first_timestamp_ns} ns"
        print(
            f"  {topic}: count={entry.message_count} "
            f"first={_format_timestamp_ns(entry.first_timestamp_ns)} "
            f"last={_format_timestamp_ns(entry.last_timestamp_ns)} span={span}"
        )


@dataclass(frozen=True)
class KissIcpProvenance:
    """KISS-ICP metadata emitted by ``convert_bag``."""

    version: str | None = None
    pass1_version: str | None = None
    pass2_version: str | None = None
    passes: int | None = None


def convert_bag(
    args: argparse.Namespace,
) -> tuple[dict[str, TopicStats], KissIcpProvenance, dict[str, int]]:
    if not args.src.is_file():
        msg = f"source bag not found: {args.src}"
        raise SystemExit(msg)
    if not args.topics:
        msg = "at least one --topic PointCloud2 topic is required"
        raise SystemExit(msg)

    ros1_typestore = get_typestore(Stores.ROS1_NOETIC)
    ros2_typestore = get_typestore(Stores.ROS2_HUMBLE)

    restamp_topics = set(args.restamp_topics)
    restamp_offsets_ns = (
        _compute_restamp_offsets(args.src, restamp_topics, ros1_typestore=ros1_typestore)
        if restamp_topics
        else {}
    )

    selected_topics = set(args.topics) | set(args.imu_topics)
    if args.odom_source == "pose-topic":
        selected_topics.add(args.pose_topic)
    else:
        selected_topics.add(args.kiss_icp_topic)

    duplicate_topics = _parse_duplicate_topics(
        args.duplicate_topics,
        pointcloud_topics=set(args.topics),
        reserved_topics=selected_topics | {args.odom_topic},
    )

    kiss_modes = {"kiss-icp", "kiss-icp-two-pass"}
    kiss_icp = None
    kiss_provenance = KissIcpProvenance()
    pass1_track: tuple[PoseTrackSample, ...] = ()
    if args.odom_source == "kiss-icp":
        kiss_icp, version = _create_kiss_icp(
            args.kiss_icp_max_range,
            native_deskew=args.kiss_icp_native_deskew,
        )
        kiss_provenance = KissIcpProvenance(version=version, passes=1)
    elif args.odom_source == "kiss-icp-two-pass":
        pass1_track, pass1_version = _collect_pass1_track(
            args.src,
            kiss_icp_topic=args.kiss_icp_topic,
            max_range=args.kiss_icp_max_range,
            restamp_offsets_ns=restamp_offsets_ns,
            ros1_typestore=ros1_typestore,
        )
        kiss_icp, pass2_version = _create_kiss_icp(args.kiss_icp_max_range, native_deskew=False)
        kiss_provenance = KissIcpProvenance(
            pass1_version=pass1_version,
            pass2_version=pass2_version,
            passes=2,
        )

    stats: dict[str, TopicStats] = defaultdict(
        lambda: TopicStats(message_count=0, first_timestamp_ns=None, last_timestamp_ns=None)
    )
    pc_connections_out: dict[str, object] = {}
    imu_connections_out: dict[str, object] = {}
    duplicate_connections_out: dict[str, object] = {}
    odom_connection = None

    writer = Writer(args.dst, version=9, storage_plugin=_storage_plugin(args.storage))
    if args.compress == "zstd":
        mode = CompressionMode.MESSAGE if args.storage == "mcap" else CompressionMode.FILE
        writer.set_compression(mode, CompressionFormat.ZSTD)
    writer.open()

    try:
        with AnyReader([args.src], default_typestore=ros1_typestore) as reader:
            available = {connection.topic for connection in reader.connections}
            missing = sorted(selected_topics - available)
            if missing:
                msg = f"topics missing from source bag: {', '.join(missing)}"
                raise SystemExit(msg)

            connections = [
                connection
                for connection in reader.connections
                if connection.topic in selected_topics
            ]
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                topic = connection.topic
                if args.odom_source == "pose-topic" and topic == args.pose_topic:
                    pose_msg = reader.deserialize(rawdata, connection.msgtype)
                    if pose_msg.__msgtype__ != POSE_STAMPED_MSGTYPE:
                        msg = (
                            f"{args.pose_topic} has type {connection.msgtype}; "
                            f"expected {POSE_STAMPED_MSGTYPE}"
                        )
                        raise SystemExit(msg)
                    odom_msg = _pose_to_odometry(
                        pose_msg,
                        child_frame_id=args.child_frame_id,
                        typestore=ros2_typestore,
                    )
                    if odom_connection is None:
                        odom_connection = writer.add_connection(
                            args.odom_topic,
                            ODOMETRY_MSGTYPE,
                            typestore=ros2_typestore,
                        )
                    writer.write(
                        odom_connection,
                        timestamp,
                        ros2_typestore.serialize_cdr(odom_msg, ODOMETRY_MSGTYPE),
                    )
                    _record_stats(stats, args.odom_topic, timestamp)
                    continue

                if topic in args.imu_topics:
                    if connection.msgtype not in {IMU_MSGTYPE, "sensor_msgs/Imu"}:
                        msg = f"{topic} has type {connection.msgtype}; expected Imu"
                        raise SystemExit(msg)
                    if topic not in imu_connections_out:
                        imu_connections_out[topic] = writer.add_connection(
                            topic,
                            IMU_MSGTYPE,
                            typestore=ros2_typestore,
                        )
                    imu_msg = reader.deserialize(rawdata, connection.msgtype)
                    if topic in restamp_offsets_ns:
                        _apply_header_stamp_offset(imu_msg, restamp_offsets_ns[topic])
                    writer.write(
                        imu_connections_out[topic],
                        timestamp,
                        ros2_typestore.serialize_cdr(imu_msg, IMU_MSGTYPE),
                    )
                    _record_stats(stats, topic, timestamp)
                    continue

                if topic not in pc_connections_out:
                    if connection.msgtype not in {POINTCLOUD2_MSGTYPE, "sensor_msgs/PointCloud2"}:
                        msg = f"{topic} has type {connection.msgtype}; expected PointCloud2"
                        raise SystemExit(msg)
                    pc_connections_out[topic] = writer.add_connection(
                        topic,
                        POINTCLOUD2_MSGTYPE,
                        typestore=ros2_typestore,
                    )
                pc_msg = reader.deserialize(rawdata, connection.msgtype)
                if topic in restamp_offsets_ns:
                    _apply_header_stamp_offset(pc_msg, restamp_offsets_ns[topic])

                if args.odom_source in kiss_modes and topic == args.kiss_icp_topic:
                    points = _pointcloud2_xyz(pc_msg)
                    if args.odom_source == "kiss-icp-two-pass":
                        time_offsets = _pointcloud2_field(pc_msg, args.kiss_icp_time_field)
                        points = rigidify_scan_to_timestamp(
                            points,
                            time_offsets,
                            _header_stamp_ns(pc_msg),
                            pass1_track,
                        )
                        kiss_icp.register_frame(points, np.array([]))
                    elif args.kiss_icp_native_deskew:
                        time_offsets = _pointcloud2_field(pc_msg, args.kiss_icp_time_field)
                        timestamps = normalize_scan_timestamps(time_offsets)
                        kiss_icp.register_frame(points, timestamps)
                    else:
                        kiss_icp.register_frame(points, np.array([]))
                    odom_msg = _kiss_icp_pose_to_odometry(
                        pc_msg,
                        pose_matrix=kiss_icp.last_pose,
                        child_frame_id=args.child_frame_id,
                        frame_id="kiss_icp_odom",
                        typestore=ros2_typestore,
                    )
                    if odom_connection is None:
                        odom_connection = writer.add_connection(
                            args.odom_topic,
                            ODOMETRY_MSGTYPE,
                            typestore=ros2_typestore,
                        )
                    writer.write(
                        odom_connection,
                        timestamp,
                        ros2_typestore.serialize_cdr(odom_msg, ODOMETRY_MSGTYPE),
                    )
                    _record_stats(stats, args.odom_topic, timestamp)

                serialized_pc = ros2_typestore.serialize_cdr(pc_msg, POINTCLOUD2_MSGTYPE)
                writer.write(pc_connections_out[topic], timestamp, serialized_pc)
                _record_stats(stats, topic, timestamp)
                if topic in duplicate_topics:
                    duplicate_topic = duplicate_topics[topic]
                    if duplicate_topic not in duplicate_connections_out:
                        duplicate_connections_out[duplicate_topic] = writer.add_connection(
                            duplicate_topic,
                            POINTCLOUD2_MSGTYPE,
                            typestore=ros2_typestore,
                        )
                    writer.write(
                        duplicate_connections_out[duplicate_topic],
                        timestamp,
                        serialized_pc,
                    )
                    _record_stats(stats, duplicate_topic, timestamp)
    finally:
        writer.close()

    return dict(stats), kiss_provenance, restamp_offsets_ns


def _record_stats(stats: dict[str, TopicStats], topic: str, timestamp_ns: int) -> None:
    current = stats[topic]
    first = current.first_timestamp_ns
    last = current.last_timestamp_ns
    if first is None or timestamp_ns < first:
        first = timestamp_ns
    if last is None or timestamp_ns > last:
        last = timestamp_ns
    stats[topic] = TopicStats(
        message_count=current.message_count + 1,
        first_timestamp_ns=first,
        last_timestamp_ns=last,
    )


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    stats, kiss_provenance, restamp_offsets_ns = convert_bag(args)
    kiss_modes = {"kiss-icp", "kiss-icp-two-pass"}
    _print_summary(
        src=args.src,
        dst=args.dst,
        odom_topic=args.odom_topic,
        odom_source=args.odom_source,
        pose_topic=args.pose_topic if args.odom_source == "pose-topic" else None,
        kiss_icp_topic=args.kiss_icp_topic if args.odom_source in kiss_modes else None,
        kiss_icp_version=kiss_provenance.version,
        kiss_icp_pass1_version=kiss_provenance.pass1_version,
        kiss_icp_pass2_version=kiss_provenance.pass2_version,
        kiss_icp_max_range=args.kiss_icp_max_range if args.odom_source in kiss_modes else None,
        kiss_icp_time_field=args.kiss_icp_time_field if args.odom_source in kiss_modes else None,
        kiss_icp_native_deskew=args.kiss_icp_native_deskew,
        kiss_icp_passes=kiss_provenance.passes,
        stats=stats,
        restamp_offsets_ns=restamp_offsets_ns or None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
