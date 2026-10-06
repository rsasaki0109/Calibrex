"""Classify bag topics into sensor roles and map them to tree frames."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from calibrex.check.frame_tree import FrameHints, StaticFrameTree, normalize_frame_id
from calibrex.core.calibration_check import CheckTopicRecord, OdometryKind, SensorRole
from calibrex.core.exceptions import DatasetError
from calibrex.data.ros_cdr import (
    AUTOWARE_VELOCITY_REPORT_TYPES,
    decode_ros2_header_frame_id,
    decode_ros2_odometry,
    ros2_pointcloud2_payload_problem,
)
from calibrex.data.rosbag2 import (
    IMAGE_TYPE,
    IMU_TYPE,
    LIDAR_MESSAGE_TYPES,
    NAVSATFIX_TYPE,
    ODOMETRY_TYPE,
    POINTCLOUD2_TYPE,
    TF_MESSAGE_TYPE,
    Rosbag2Connection,
    iter_topic_messages,
)
from calibrex.data.velodyne_packets import VELODYNE_SCAN_TYPE

VELODYNE_PACKETS_REASON = (
    "raw Velodyne packets (velodyne_msgs/msg/VelodyneScan), not read by the check; "
    "convert them to per-sensor point clouds with tools/velodyne_scan_to_pointcloud2.py "
    "and check the converted bag"
)
COMPRESSED_IMAGE_TYPE = "sensor_msgs/msg/CompressedImage"
TWIST_TYPES = frozenset(
    {
        "geometry_msgs/msg/TwistStamped",
        "geometry_msgs/msg/TwistWithCovarianceStamped",
        *AUTOWARE_VELOCITY_REPORT_TYPES,
    }
)
_WHEEL_TOKENS = frozenset({"wheel", "wheels", "encoder", "encoders", "wheelodom"})
_INS_TOKENS = frozenset({"ins", "gnss", "gps", "navsat", "oxts", "novatel", "applanix"})


def classify_message_type(message_type: str, topic: str) -> SensorRole | None:
    """Return the sensor role of a topic from its message type, or ``None``."""

    if message_type in LIDAR_MESSAGE_TYPES or message_type == VELODYNE_SCAN_TYPE:
        return "lidar"
    if message_type == IMU_TYPE:
        return "imu"
    if message_type in {IMAGE_TYPE, COMPRESSED_IMAGE_TYPE}:
        return "camera"
    if message_type == NAVSATFIX_TYPE:
        return "gnss"
    if message_type == ODOMETRY_TYPE:
        return "odometry"
    if message_type in TWIST_TYPES:
        return "twist"
    if message_type == TF_MESSAGE_TYPE and topic.rstrip("/").endswith("tf_static"):
        return "tf_static"
    return None


def classify_odometry_kind(topic: str) -> OdometryKind:
    """Guess by name whether an odometry or twist topic is wheel odometry or an INS/GNSS."""

    tokens = {token for token in re.split(r"[^a-z0-9]+", topic.lower()) if token}
    if tokens & _WHEEL_TOKENS:
        return "wheel"
    if tokens & _INS_TOKENS:
        return "ins"
    return "unknown"


def classify_topics(
    connections: Sequence[tuple[Rosbag2Connection, int | None]],
    kind_overrides: Mapping[str, OdometryKind] | None = None,
) -> list[CheckTopicRecord]:
    """Build one record per topic that has a sensor role, sorted by topic.

    Odometry and twist topics get a kind (``wheel``, ``ins`` or ``unknown``) from
    their name; ``kind_overrides`` (``--topic-kind``) sets it explicitly.
    """

    overrides = kind_overrides or {}
    records: list[CheckTopicRecord] = []
    for connection, count in connections:
        role = classify_message_type(connection.message_type, connection.topic)
        if role is None:
            continue
        kind: OdometryKind | None = None
        notes: list[str] = []
        if role in {"odometry", "twist"}:
            if connection.topic in overrides:
                kind = overrides[connection.topic]
                notes.append(f"kind '{kind}' set by --topic-kind")
            else:
                kind = classify_odometry_kind(connection.topic)
                if kind == "unknown" and connection.message_type in AUTOWARE_VELOCITY_REPORT_TYPES:
                    kind = "wheel"
                    notes.append("Autoware VelocityReport is the vehicle's own speed report")
        records.append(
            CheckTopicRecord(
                topic=connection.topic,
                message_type=connection.message_type,
                message_count=count,
                role=role,
                odometry_kind=kind,
                notes=notes,
            )
        )
    return sorted(records, key=lambda record: record.topic)


def parse_topic_kinds(items: Sequence[str]) -> dict[str, OdometryKind]:
    """Parse repeated ``TOPIC=KIND`` overrides (``KIND`` is ``wheel`` or ``ins``)."""

    kinds: dict[str, OdometryKind] = {}
    for item in items:
        topic, separator, kind = item.partition("=")
        topic, kind = topic.strip(), kind.strip().lower()
        if not separator or not topic or kind not in {"wheel", "ins"}:
            msg = f"--topic-kind expects TOPIC=wheel or TOPIC=ins, got {item!r}"
            raise DatasetError(msg)
        kinds[topic] = "wheel" if kind == "wheel" else "ins"
    return kinds


def read_header_frames(
    bag: str | Path,
    records: Sequence[CheckTopicRecord],
) -> dict[str, str | None]:
    """Read the frame id of the first message of each sensor topic.

    Odometry topics report ``child_frame_id`` (the body the odometry tracks);
    every other role reports ``header.frame_id``. A topic with no messages or an
    unreadable first message maps to ``None``.
    """

    frames: dict[str, str | None] = {}
    for record in records:
        if record.role in (None, "tf_static"):
            continue
        frame: str | None = None
        try:
            for _conn, timestamp_ns, data in iter_topic_messages(bag, record.topic, limit=1):
                if record.role == "odometry":
                    frame = decode_ros2_odometry(record.topic, timestamp_ns, data).child_frame_id
                else:
                    frame = decode_ros2_header_frame_id(data)
        except DatasetError:
            frame = None
        frames[record.topic] = normalize_frame_id(frame) if frame else None
    return frames


def flag_unreadable_pointclouds(
    bag: str | Path,
    records: Sequence[CheckTopicRecord],
) -> list[CheckTopicRecord]:
    """Mark PointCloud2 topics whose first message is not a raw point array.

    A transport-compressed cloud (for example Draco) published next to its
    decompressed twin would otherwise be picked as the sensor stream. Such a
    topic gets ``ignored_reason`` and fills no sensor slot.
    """

    flagged: list[CheckTopicRecord] = []
    for record in records:
        if record.role == "lidar" and record.message_type == VELODYNE_SCAN_TYPE:
            flagged.append(
                record.model_copy(
                    update={
                        "ignored_reason": VELODYNE_PACKETS_REASON,
                        "notes": [*record.notes, VELODYNE_PACKETS_REASON],
                    }
                )
            )
            continue
        if record.role != "lidar" or record.message_type != POINTCLOUD2_TYPE:
            flagged.append(record)
            continue
        problem: str | None = None
        try:
            for _conn, _timestamp_ns, data in iter_topic_messages(bag, record.topic, limit=1):
                problem = ros2_pointcloud2_payload_problem(data)
        except DatasetError as error:
            problem = f"the first message could not be read ({error})"
        if problem is None:
            flagged.append(record)
        else:
            flagged.append(
                record.model_copy(
                    update={"ignored_reason": problem, "notes": [*record.notes, problem]}
                )
            )
    return flagged


def map_topics_to_frames(
    records: Sequence[CheckTopicRecord],
    header_frames: Mapping[str, str | None],
    tree: StaticFrameTree,
    hints: FrameHints,
    overrides: Mapping[str, str] | None = None,
) -> list[CheckTopicRecord]:
    """Assign each sensor topic a tree frame.

    Order: explicit override, header frame id found in the tree, topic hint
    from a calibration file, then the role default of a calibration file. A
    role default applies only when the role's remaining topics belong to one
    sensor (a single topic, or topics sharing one header frame id).
    """

    overrides = overrides or {}
    mapped: dict[str, CheckTopicRecord] = {}
    for record in records:
        header = header_frames.get(record.topic)
        updated = record.model_copy(update={"header_frame_id": header})
        if record.role in (None, "tf_static"):
            mapped[record.topic] = updated
            continue
        if record.topic in overrides:
            updated = updated.model_copy(
                update={
                    "mapped_frame": normalize_frame_id(overrides[record.topic]),
                    "frame_source": "override",
                }
            )
        elif header is not None and header in tree:
            updated = updated.model_copy(update={"mapped_frame": header, "frame_source": "header"})
        elif record.topic in hints.topic_frames:
            updated = updated.model_copy(
                update={
                    "mapped_frame": hints.topic_frames[record.topic],
                    "frame_source": "source_topic_hint",
                }
            )
        mapped[record.topic] = updated

    for role, default_frame in hints.role_frames.items():
        if default_frame not in tree:
            continue
        pending = [
            record
            for record in mapped.values()
            if record.role == role and record.mapped_frame is None
        ]
        if not pending:
            continue
        distinct = {record.header_frame_id or record.topic for record in pending}
        if len(pending) == 1 or len(distinct) == 1:
            for record in pending:
                mapped[record.topic] = record.model_copy(
                    update={"mapped_frame": default_frame, "frame_source": "role_default"}
                )
    return [mapped[record.topic] for record in records]
