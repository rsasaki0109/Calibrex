"""Autoware ``VelocityReport`` decoding and its role as a wheel-speed topic."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
from tests.unit.test_rosbag2 import _write_sqlite_bag

from calibrex.check.roles import classify_message_type, classify_topics
from calibrex.check.vehicle_inputs import read_twist_track
from calibrex.core.exceptions import DatasetError
from calibrex.data.ros_cdr import AUTOWARE_VELOCITY_REPORT_TYPES, decode_ros2_velocity_report
from calibrex.data.rosbag2 import Rosbag2Connection

REPORT_TYPE = "autoware_auto_vehicle_msgs/msg/VelocityReport"


def _report(stamp_ns: int, lon: float, lat: float, rate: float, frame: str = "base_link") -> bytes:
    name = frame.encode() + b"\x00"
    out = bytearray(b"\x00\x01\x00\x00")
    out += struct.pack("<iI", stamp_ns // 10**9, stamp_ns % 10**9)
    out += struct.pack("<I", len(name)) + name
    out += bytes((-(len(out) - 4)) % 4)
    out += struct.pack("<3f", lon, lat, rate)
    return bytes(out)


def test_velocity_report_decodes_to_a_body_twist() -> None:
    twist = decode_ros2_velocity_report("/v", 0, _report(2_500_000_000, 3.5, -0.25, 0.125))

    assert twist.timestamp_ns == 2_500_000_000
    assert twist.frame_id == "base_link"
    assert twist.linear_velocity == pytest.approx((3.5, -0.25, 0.0))
    assert twist.angular_velocity == pytest.approx((0.0, 0.0, 0.125))


def test_velocity_report_rejects_trailing_garbage() -> None:
    with pytest.raises(DatasetError, match="trailing"):
        decode_ros2_velocity_report("/v", 0, _report(1, 1.0, 0.0, 0.0) + b"\x01\x02\x03\x04")


def test_velocity_report_topic_is_a_wheel_twist() -> None:
    assert classify_message_type(REPORT_TYPE, "/vehicle/status/velocity_status") == "twist"
    connection = Rosbag2Connection(1, "/vehicle/status/velocity_status", REPORT_TYPE)

    (record,) = classify_topics([(connection, 10)])

    assert record.role == "twist"
    assert record.odometry_kind == "wheel"
    assert "autoware_vehicle_msgs/msg/VelocityReport" in AUTOWARE_VELOCITY_REPORT_TYPES


def test_velocity_report_track_is_readable(tmp_path: Path) -> None:
    bag = tmp_path / "bag.db3"
    _write_sqlite_bag(
        bag,
        topics=[("/v", REPORT_TYPE)],
        messages=[
            ("/v", 1_000_000_000, _report(1_000_000_000, 2.0, 0.0, 0.1)),
            ("/v", 1_100_000_000, _report(1_100_000_000, 2.5, 0.0, 0.2)),
        ],
    )

    track = read_twist_track(bag, "/v", REPORT_TYPE)

    assert track.frame == "base_link"
    assert track.linear_mps[:, 0].tolist() == pytest.approx([2.0, 2.5])
    assert track.angular_rps[:, 2].tolist() == pytest.approx([0.1, 0.2])
