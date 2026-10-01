"""CDR decoders for ``tf2_msgs/msg/TFMessage`` and ``sensor_msgs/msg/NavSatFix``."""

from __future__ import annotations

import pytest
from tests.unit.check_fixtures import encode_navsatfix, encode_tf_message, header_payload

from calibrex.core.exceptions import DatasetError
from calibrex.data.ros_cdr import (
    decode_ros2_header_frame_id,
    decode_ros2_navsatfix,
    decode_ros2_tf_message,
)
from calibrex.data.rosbag2 import NAVSATFIX_TYPE, TF_MESSAGE_TYPE, decode_rosbag2_message

TRANSFORMS = [
    ("base_link", "imu_link", (0.1, -0.2, 0.3), (0.0, 0.0, 0.0, 1.0)),
    ("imu_link", "lidar", (1.0, 2.0, 3.0), (0.0, 0.0, 0.7071067811865476, 0.7071067811865476)),
]


@pytest.mark.parametrize("little_endian", [True, False])
def test_tf_message_roundtrip(little_endian: bool) -> None:
    data = encode_tf_message(TRANSFORMS, secs=12, nsecs=34, little_endian=little_endian)

    message = decode_ros2_tf_message("/tf_static", 99, data)

    assert message.topic == "/tf_static"
    assert message.timestamp_ns == 99
    assert len(message.transforms) == 2
    first, second = message.transforms
    assert (first.frame_id, first.child_frame_id) == ("base_link", "imu_link")
    assert first.translation_m == (0.1, -0.2, 0.3)
    assert first.timestamp_ns == 12_000_000_034
    assert second.rotation_xyzw == (0.0, 0.0, 0.7071067811865476, 0.7071067811865476)


def test_empty_tf_message_and_dispatch() -> None:
    data = encode_tf_message([])

    message = decode_rosbag2_message("/tf", TF_MESSAGE_TYPE, 1, data)

    assert message.transforms == ()


def test_tf_message_rejects_truncated_and_absurd_counts() -> None:
    data = encode_tf_message(TRANSFORMS)

    with pytest.raises(DatasetError):
        decode_ros2_tf_message("/tf", 0, data[:-5])
    bad = bytearray(data)
    bad[4:8] = (0xFFFFFFFF).to_bytes(4, "little")
    with pytest.raises(DatasetError, match="exceeds"):
        decode_ros2_tf_message("/tf", 0, bytes(bad))


@pytest.mark.parametrize("little_endian", [True, False])
def test_navsatfix_roundtrip(little_endian: bool) -> None:
    data = encode_navsatfix(
        status=-1,
        service=3,
        latitude=-33.5,
        longitude=151.25,
        altitude=-4.5,
        covariance_type=3,
        little_endian=little_endian,
    )

    message = decode_ros2_navsatfix("/gnss/fix", 7, data)

    assert message.frame_id == "gps"
    assert message.timestamp_ns == 9_000_000_003
    assert message.status == -1
    assert message.service == 3
    assert (message.latitude_deg, message.longitude_deg, message.altitude_m) == (
        -33.5,
        151.25,
        -4.5,
    )
    assert message.position_covariance == (1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0)
    assert message.position_covariance_type == 3


def test_navsatfix_dispatch_and_truncation() -> None:
    data = encode_navsatfix()

    assert decode_rosbag2_message("/f", NAVSATFIX_TYPE, 1, data).altitude_m == 42.0
    with pytest.raises(DatasetError):
        decode_ros2_navsatfix("/f", 0, data[:-3])


def test_header_frame_id_reads_any_stamped_message() -> None:
    assert decode_ros2_header_frame_id(header_payload("lidar_link")) == "lidar_link"
    assert decode_ros2_header_frame_id(encode_navsatfix(frame_id="gps")) == "gps"
