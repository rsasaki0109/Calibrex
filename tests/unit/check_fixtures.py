"""Shared fixtures for ``calibrex check`` tests: hand-encoded CDR and synthetic bags."""

from __future__ import annotations

from pathlib import Path

from tests.unit.test_rosbag2 import _encode_odometry, _write_sqlite_bag

from calibrex.data.ros_cdr_writer import (
    Transform,
    encode_header_only,
    encode_navsatfix,
    encode_tf_message,
)

__all__ = ["Transform", "encode_navsatfix", "encode_tf_message"]

TF_TYPE = "tf2_msgs/msg/TFMessage"
IMU_TYPE = "sensor_msgs/msg/Imu"
CLOUD_TYPE = "sensor_msgs/msg/PointCloud2"
IMAGE_TYPE = "sensor_msgs/msg/Image"
NAVSAT_TYPE = "sensor_msgs/msg/NavSatFix"
ODOM_TYPE = "nav_msgs/msg/Odometry"
TWIST_TYPE = "geometry_msgs/msg/TwistStamped"

def header_payload(frame_id: str) -> bytes:
    """A stamped-message prefix; enough for header frame id reads."""

    return encode_header_only(frame_id)


def write_check_bag(
    path: Path,
    *,
    tf_messages: list[list[Transform]] | None = None,
    include_tf_topic: bool = True,
    sensor_frames: dict[str, tuple[str, str]] | None = None,
) -> Path:
    """Write a db3 bag with sensors; ``sensor_frames`` maps topic -> (type, frame_id)."""

    sensors = sensor_frames or {
        "/imu": (IMU_TYPE, "imu_link"),
        "/lidar_front/points": (CLOUD_TYPE, "lidar_front"),
        "/lidar_rear/points": (CLOUD_TYPE, "lidar_rear"),
        "/camera/image_raw": (IMAGE_TYPE, "camera_optical"),
        "/gnss/fix": (NAVSAT_TYPE, "gnss_link"),
    }
    topics: list[tuple[str, str]] = []
    messages: list[tuple[str, int, bytes]] = []
    if include_tf_topic:
        topics.append(("/tf_static", TF_TYPE))
        for index, transforms in enumerate(tf_messages or []):
            messages.append(("/tf_static", 1_000 + index, encode_tf_message(transforms)))
    for index, (topic, (message_type, frame_id)) in enumerate(sensors.items()):
        topics.append((topic, message_type))
        if message_type == ODOM_TYPE:
            payload = _encode_odometry(
                frame_id="odom",
                child_frame_id=frame_id,
                secs=1,
                nsecs=0,
                position=(0.0, 0.0, 0.0),
                orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
        else:
            payload = header_payload(frame_id)
        messages.append((topic, 2_000 + index, payload))
    topics.append(("/rosout", "rcl_interfaces/msg/Log"))
    messages.append(("/rosout", 3_000, header_payload("x")))
    _write_sqlite_bag(path, topics=topics, messages=messages)
    return path


IDENTITY = (0.0, 0.0, 0.0, 1.0)


def default_tf() -> list[Transform]:
    """A tree: base_link -> {imu_link, lidar_front, lidar_rear, camera_optical, gnss_link}."""

    return [
        ("base_link", "imu_link", (0.0, 0.0, 0.1), IDENTITY),
        ("base_link", "lidar_front", (1.0, 0.0, 0.5), IDENTITY),
        ("base_link", "lidar_rear", (-1.0, 0.0, 0.5), (0.0, 0.0, 1.0, 0.0)),
        ("base_link", "camera_optical", (0.5, 0.0, 0.8), IDENTITY),
        ("base_link", "gnss_link", (0.0, 0.0, 1.2), IDENTITY),
    ]
