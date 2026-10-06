"""Velodyne packet decoding and the packets-to-PointCloud2 bag conversion."""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np
import pytest
from tests.unit.test_rosbag2 import _write_sqlite_bag

from calibrex.check.roles import (
    VELODYNE_PACKETS_REASON,
    classify_message_type,
    flag_unreadable_pointclouds,
)
from calibrex.core.calibration_check import CheckTopicRecord
from calibrex.core.exceptions import DatasetError
from calibrex.data.ros_cdr import decode_ros2_pointcloud2
from calibrex.data.rosbag2 import iter_messages
from calibrex.data.rosbag2_velodyne import convert_velodyne_bag, points_topic_name
from calibrex.data.velodyne_packets import (
    PACKET_SIZE,
    VELODYNE_SCAN_TYPE,
    decode_velodyne_scan,
    decode_velodyne_scan_packets,
    velodyne_model_of_packet,
)


def _vlp32c_packet(range_m: float, azimuths_deg: list[float]) -> bytes:
    body = bytearray()
    for azimuth in azimuths_deg:
        body += b"\xff\xee" + struct.pack("<H", round(azimuth * 100))
        body += struct.pack("<HB", round(range_m / 0.004), 7) * 32
    return bytes(body) + bytes(4) + bytes([0x37, 0x28])


def _vlp16_packet(range_m: float, azimuth_deg: float) -> bytes:
    body = bytearray()
    for block in range(12):
        body += b"\xff\xee" + struct.pack("<H", round((azimuth_deg + 0.2 * block) * 100))
        body += struct.pack("<HB", round(range_m / 0.002), 9) * 32
    return bytes(body) + bytes(4) + bytes([0x37, 0x22])


def _scan_payload(frame_id: str, packets: list[bytes], stamp_ns: int = 5_000_000_000) -> bytes:
    """CDR ``VelodyneScan``: header, then packets of (stamp, 1206 bytes) 4-byte aligned."""

    out = bytearray(b"\x00\x01\x00\x00")

    def align4() -> None:
        out.extend(bytes((-(len(out) - 4)) % 4))

    out += struct.pack("<iI", stamp_ns // 10**9, stamp_ns % 10**9)
    name = frame_id.encode() + b"\x00"
    out += struct.pack("<I", len(name)) + name
    align4()
    out += struct.pack("<I", len(packets))
    for index, packet in enumerate(packets):
        align4()
        out += struct.pack("<iI", stamp_ns // 10**9, stamp_ns % 10**9 + index * 1_000_000)
        out += packet
    return bytes(out)


def test_vlp32c_packet_geometry() -> None:
    packet = _vlp32c_packet(10.0, [90.0] * 12)
    assert len(packet) == PACKET_SIZE
    assert velodyne_model_of_packet(packet) == "vlp32c"
    payload = _scan_payload("velodyne_left", [packet])

    cloud = decode_velodyne_scan("/l", 0, payload, x_forward=False)

    assert cloud.frame_id == "velodyne_left"
    assert cloud.point_count == 12 * 32
    # Azimuth 90 deg is +x (clockwise from +y); channel 5 has elevation 0 and offset -1.4 deg.
    index = 5
    expected_azimuth = math.radians(90.0 - 1.4)
    assert cloud.xyz[index, 0] == pytest.approx(10.0 * math.sin(expected_azimuth), abs=0.01)
    assert cloud.xyz[index, 1] == pytest.approx(10.0 * math.cos(expected_azimuth), abs=0.01)
    assert cloud.xyz[index, 2] == pytest.approx(0.0, abs=1e-6)
    assert np.linalg.norm(cloud.xyz, axis=1) == pytest.approx(10.0, abs=0.01)
    offsets = cloud.point_time_offsets_s
    assert offsets is not None
    assert offsets[0] == 0.0
    assert offsets[32] == pytest.approx(55.296e-6)  # next block
    assert cloud.intensity is not None and cloud.intensity[0] == 7.0


def test_vlp16_packet_ranges_and_sequence_time() -> None:
    packet = _vlp16_packet(6.0, 0.0)
    assert velodyne_model_of_packet(packet) == "vlp16"
    cloud = decode_velodyne_scan("/f", 0, _scan_payload("velodyne_front", [packet]))

    assert cloud.point_count == 12 * 2 * 16
    assert np.linalg.norm(cloud.xyz, axis=1) == pytest.approx(6.0, abs=0.01)
    offsets = cloud.point_time_offsets_s
    assert offsets is not None
    assert offsets[16] == pytest.approx(55.296e-6)  # second firing sequence
    assert offsets[1] == pytest.approx(2.304e-6)
    # Lowest beam (-15 deg) sits below the sensor.
    assert cloud.xyz[0, 2] == pytest.approx(6.0 * math.sin(math.radians(-15.0)), abs=0.01)


def test_packet_stamps_extend_per_point_time_and_empty_ranges_are_dropped() -> None:
    first = _vlp32c_packet(0.0, [0.0] * 12)  # all returns invalid
    second = _vlp32c_packet(8.0, [1.0] * 12)
    payload = _scan_payload("v", [first, second])

    scan = decode_velodyne_scan_packets(payload)
    cloud = decode_velodyne_scan("/v", 0, payload)

    assert len(scan.packets) == 2
    assert scan.packet_stamps_ns[1] - scan.packet_stamps_ns[0] == 1_000_000
    assert cloud.point_count == 12 * 32
    assert cloud.point_time_offsets_s is not None
    assert cloud.point_time_offsets_s.min() == pytest.approx(1.0e-3)


def test_unknown_product_and_bad_flag_are_rejected() -> None:
    packet = bytearray(_vlp32c_packet(5.0, [0.0] * 12))
    packet[-1] = 0x99
    with pytest.raises(DatasetError, match="product id"):
        velodyne_model_of_packet(bytes(packet))
    bad = bytearray(_vlp32c_packet(5.0, [0.0] * 12))
    bad[0] = 0
    with pytest.raises(DatasetError, match="block flag"):
        decode_velodyne_scan("/v", 0, _scan_payload("v", [bytes(bad)]))
    with pytest.raises(DatasetError, match="truncated"):
        decode_velodyne_scan("/v", 0, _scan_payload("v", [_vlp32c_packet(5.0, [0.0] * 12)])[:-10])


def test_velodyne_topics_are_lidar_but_ignored_with_a_conversion_hint(tmp_path: Path) -> None:
    assert classify_message_type(VELODYNE_SCAN_TYPE, "/front/velodyne_packets") == "lidar"
    record = CheckTopicRecord(
        topic="/front/velodyne_packets", message_type=VELODYNE_SCAN_TYPE, role="lidar"
    )

    (flagged,) = flag_unreadable_pointclouds(tmp_path / "unused", [record])

    assert flagged.ignored_reason == VELODYNE_PACKETS_REASON
    assert "velodyne_scan_to_pointcloud2" in VELODYNE_PACKETS_REASON


def test_convert_bag_writes_clouds_in_the_sensor_frame(tmp_path: Path) -> None:
    payload = _scan_payload("velodyne_front", [_vlp16_packet(5.0, 10.0)])
    source = tmp_path / "src.db3"
    _write_sqlite_bag(
        source,
        topics=[
            ("/lidar/front/velodyne_packets", VELODYNE_SCAN_TYPE),
            ("/other", "std_msgs/msg/String"),
        ],
        messages=[
            ("/lidar/front/velodyne_packets", 1_000, payload),
            ("/other", 2_000, b"\x00\x01\x00\x00abcd"),
        ],
    )

    summary = convert_velodyne_bag(source, tmp_path / "out")

    assert summary.cloud_topics == {"/lidar/front/velodyne_packets": "/lidar/front/velodyne_points"}
    assert points_topic_name("/a/velodyne_packets") == "/a/velodyne_points"
    assert summary.clouds_written == 1 and summary.messages_copied == 1
    rows = list(iter_messages(tmp_path / "out"))
    topics = {connection.topic: data for connection, _ts, data in rows}
    assert set(topics) == {"/lidar/front/velodyne_points", "/other"}
    cloud = decode_ros2_pointcloud2(
        "/p", 0, topics["/lidar/front/velodyne_points"], point_time_field="time"
    )
    assert cloud.frame_id == "velodyne_front"
    assert cloud.point_count == 12 * 32
    assert cloud.point_time_offsets_s is not None


def test_convert_requires_a_packet_topic(tmp_path: Path) -> None:
    source = tmp_path / "src.db3"
    _write_sqlite_bag(source, topics=[("/x", "std_msgs/msg/String")], messages=[])
    with pytest.raises(DatasetError, match="VelodyneScan"):
        convert_velodyne_bag(source, tmp_path / "out")
