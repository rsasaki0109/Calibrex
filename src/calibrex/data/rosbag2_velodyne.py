"""Convert the raw Velodyne packet topics of a rosbag2 into ``PointCloud2`` topics.

``calibrex check`` reads ``sensor_msgs/msg/PointCloud2``. Vehicle recordings
(Autoware sample data, for example) often hold the sensor's raw
``velodyne_msgs/msg/VelodyneScan`` packets stamped in the sensor's own frame,
next to a concatenated cloud that was already transformed into ``base_link``.
:func:`convert_velodyne_bag` writes a derived bag that keeps every other topic
byte for byte and adds one ``PointCloud2`` per packet topic
(``.../velodyne_packets`` becomes ``.../velodyne_points``) with a per-point
``time`` field, decoded by :mod:`calibrex.data.velodyne_packets`.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from calibrex.core.exceptions import DatasetError
from calibrex.data.ros_cdr_writer import encode_pointcloud2
from calibrex.data.rosbag2 import IMAGE_TYPE, iter_messages, list_rosbag2_connections
from calibrex.data.rosbag2_writer import Rosbag2Writer
from calibrex.data.velodyne_packets import VELODYNE_SCAN_TYPE, decode_velodyne_scan

_POINT_FIELDS = (
    ("x", 0, 7, 1),
    ("y", 4, 7, 1),
    ("z", 8, 7, 1),
    ("intensity", 12, 7, 1),
    ("time", 16, 7, 1),
)
POINTCLOUD2_TYPE = "sensor_msgs/msg/PointCloud2"


@dataclass(frozen=True)
class VelodyneConversionSummary:
    """What :func:`convert_velodyne_bag` wrote."""

    source: str
    destination: str
    cloud_topics: dict[str, str]
    clouds_written: int
    messages_copied: int
    messages_dropped: int


def points_topic_name(packet_topic: str) -> str:
    """Name of the cloud topic written for a packet topic."""

    if packet_topic.endswith("_packets"):
        return packet_topic[: -len("_packets")] + "_points"
    return packet_topic + "_points"


def encode_scan_as_pointcloud2(topic: str, timestamp_ns: int, payload: bytes) -> bytes:
    """CDR ``PointCloud2`` (x y z intensity time, float32) of one ``VelodyneScan`` payload."""

    cloud = decode_velodyne_scan(topic, timestamp_ns, payload)
    count = cloud.point_count
    packed = np.zeros((count, 5), dtype="<f4")
    packed[:, :3] = cloud.xyz
    if cloud.intensity is not None:
        packed[:, 3] = cloud.intensity
    if cloud.point_time_offsets_s is not None:
        packed[:, 4] = cloud.point_time_offsets_s
    return encode_pointcloud2(
        frame_id=cloud.frame_id,
        timestamp_ns=cloud.timestamp_ns,
        fields=_POINT_FIELDS,
        point_step=20,
        data=packed.tobytes(),
    )


def convert_velodyne_bag(
    source: str | Path,
    destination: str | Path,
    *,
    drop_topics: Collection[str] = (),
    keep_images: bool = False,
    max_duration_s: float | None = None,
    overwrite: bool = False,
) -> VelodyneConversionSummary:
    """Copy ``source`` to ``destination`` adding a cloud topic per ``VelodyneScan`` topic.

    The packet topics themselves and (unless ``keep_images``) the image topics
    are not copied, nor are ``drop_topics``; with ``max_duration_s`` only the
    first seconds are converted.
    """

    connections = [connection for connection, _ in list_rosbag2_connections(source)]
    packet_topics = [c.topic for c in connections if c.message_type == VELODYNE_SCAN_TYPE]
    if not packet_topics:
        raise DatasetError(f"{source} has no {VELODYNE_SCAN_TYPE} topic")
    dropped_topics = set(drop_topics)
    cloud_topics = {
        topic: points_topic_name(topic) for topic in packet_topics if topic not in dropped_topics
    }
    skipped = {
        c.topic
        for c in connections
        if c.topic in dropped_topics
        or (c.message_type == VELODYNE_SCAN_TYPE
        and c.topic not in cloud_topics)
        or (c.message_type == IMAGE_TYPE and not keep_images)
    }
    written = copied = dropped = 0
    first_ns: int | None = None
    with Rosbag2Writer(destination, overwrite=overwrite) as writer:
        for connection in connections:
            if connection.topic in cloud_topics:
                writer.add_topic(cloud_topics[connection.topic], POINTCLOUD2_TYPE)
            elif connection.topic not in skipped:
                latched = "durability: 1" in (connection.offered_qos_profiles or "")
                writer.add_topic(connection.topic, connection.message_type, latched=latched)
        for connection, timestamp_ns, payload in iter_messages(source):
            if first_ns is None:
                first_ns = timestamp_ns
            if max_duration_s is not None and timestamp_ns - first_ns > max_duration_s * 1e9:
                break
            if connection.topic in cloud_topics:
                writer.write(
                    cloud_topics[connection.topic],
                    timestamp_ns,
                    encode_scan_as_pointcloud2(connection.topic, timestamp_ns, payload),
                )
                written += 1
            elif connection.topic in skipped:
                dropped += 1
            else:
                writer.write(connection.topic, timestamp_ns, payload)
                copied += 1
    return VelodyneConversionSummary(
        source=str(source),
        destination=str(destination),
        cloud_topics=cloud_topics,
        clouds_written=written,
        messages_copied=copied,
        messages_dropped=dropped,
    )
