#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.24",
#   "rosbags>=0.10",
# ]
# ///
"""Convert a ROS 1 bag into a rosbag2 online-calibration pair.

This standalone data-prep tool is **not** part of the Calibrex package. It reads a
ROS 1 ``.bag`` with ``rosbags``, writes a rosbag2 directory containing:

* ``sensor_msgs/msg/PointCloud2`` topics passed through unchanged in content
  (ROS 1 payloads are deserialized and re-serialized as ROS 2 CDR).
* A synthesized ``nav_msgs/msg/Odometry`` topic built 1:1 from each
  ``geometry_msgs/PoseStamped`` on ``--pose-topic``: header stamp and
  ``frame_id`` are copied, ``child_frame_id`` comes from ``--child-frame-id``,
  ``pose.pose`` is copied verbatim, pose/twist covariances are zero, and twist
  is zero.

The pose trajectory itself is real upstream data (for TIERS Indoor02, VRPN
MOCAP). **Only the Odometry message envelope is synthesized here**; Calibrex
never authors ``/odom`` during calibration.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
        "--pose-topic",
        required=True,
        help="PoseStamped topic converted 1:1 into --odom-topic",
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
    return parser.parse_args(list(argv) if argv is not None else None)


def _storage_plugin(name: str) -> StoragePlugin:
    if name == "mcap":
        return StoragePlugin.MCAP
    return StoragePlugin.SQLITE3


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


def _format_timestamp_ns(timestamp_ns: int | None) -> str:
    if timestamp_ns is None:
        return "n/a"
    return f"{timestamp_ns} ns"


def _print_summary(
  *,
  src: Path,
  dst: Path,
  odom_topic: str,
  pose_topic: str,
  stats: dict[str, TopicStats],
) -> None:
    print(f"source: {src}")
    print(f"destination: {dst}")
    print(f"pose_topic: {pose_topic}")
    print(f"synthesized_odometry_topic: {odom_topic} (authored by this tool)")
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


def convert_bag(args: argparse.Namespace) -> dict[str, TopicStats]:
    if not args.src.is_file():
        msg = f"source bag not found: {args.src}"
        raise SystemExit(msg)
    if not args.topics:
        msg = "at least one --topic PointCloud2 topic is required"
        raise SystemExit(msg)

    ros1_typestore = get_typestore(Stores.ROS1_NOETIC)
    ros2_typestore = get_typestore(Stores.ROS2_HUMBLE)

    selected_topics = set(args.topics) | {args.pose_topic}
    stats: dict[str, TopicStats] = defaultdict(
        lambda: TopicStats(message_count=0, first_timestamp_ns=None, last_timestamp_ns=None)
    )
    pc_connections_out: dict[str, object] = {}
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
                if topic == args.pose_topic:
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
                writer.write(
                    pc_connections_out[topic],
                    timestamp,
                    ros2_typestore.serialize_cdr(pc_msg, POINTCLOUD2_MSGTYPE),
                )
                _record_stats(stats, topic, timestamp)
    finally:
        writer.close()

    return dict(stats)


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
    stats = convert_bag(args)
    _print_summary(
        src=args.src,
        dst=args.dst,
        odom_topic=args.odom_topic,
        pose_topic=args.pose_topic,
        stats=stats,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
