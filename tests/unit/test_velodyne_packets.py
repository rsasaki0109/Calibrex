"""Velodyne packet decoding and the packets-to-PointCloud2 bag conversion."""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np
import pytest
import yaml
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
    decode_velodyne_packet,
    decode_velodyne_scan,
    decode_velodyne_scan_packets,
    load_velodyne_calibration,
    velodyne_model_of_packet,
    velodyne_packet_format,
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


def _vlp16_dual_packet(first_m: float, second_m: float, azimuth_deg: float) -> bytes:
    """Six firings; each pair of blocks shares an azimuth. Only channel 0 differs."""

    body = bytearray()
    for block in range(12):
        body += b"\xff\xee" + struct.pack("<H", round((azimuth_deg + 0.4 * (block // 2)) * 100))
        for _sequence in range(2):
            for channel in range(16):
                second = block % 2 == 1 and channel == 0
                body += struct.pack(
                    "<HB", round((second_m if second else first_m) / 0.002), 20 if second else 9
                )
    return bytes(body) + bytes(4) + bytes([0x39, 0x22])


def _hdl64e_packet(ranges_m: list[float], azimuth_deg: float, *, dual: bool = False) -> bytes:
    """Alternating upper/lower blocks; ``ranges_m`` gives the range per block."""

    body = bytearray()
    for block in range(12):
        flag = b"\xff\xee" if block % 2 == 0 else b"\xff\xdd"
        firing = block // 4 if dual else block // 2
        body += flag + struct.pack("<H", round((azimuth_deg + 0.09 * firing) * 100))
        body += struct.pack("<HB", round(ranges_m[block] / 0.002), 30 + block % 2) * 32
    return bytes(body) + b"\x10\x27\x00\x00" + b"H\x01"


def _hdl64e_calibration_yaml(path: Path, *, vert_offset_m: float = 0.0) -> Path:
    lasers = [
        {
            "laser_id": laser,
            "rot_correction": 0.0,
            "vert_correction": math.radians(-24.0 + 0.4 * laser),
            "dist_correction": 0.0,
            "vert_offset_correction": vert_offset_m,
            "horiz_offset_correction": 0.0,
        }
        for laser in range(64)
    ]
    path.write_text(yaml.safe_dump({"lasers": lasers, "distance_resolution": 0.002}))
    return path


@pytest.mark.parametrize(
    ("product", "mode", "name"),
    [
        (0x21, 0x37, "HDL-32E strongest"),
        (0x23, 0x38, "VLP-32A/B last"),
        (0x24, 0x37, "VLP-16 Hi-Res strongest"),
        (0xA1, 0x39, "VLS-128 dual"),
        (0x28, 0x39, "VLP-32C dual"),
    ],
)
def test_unverified_formats_are_refused_by_name(product: int, mode: int, name: str) -> None:
    packet = bytearray(_vlp32c_packet(5.0, [0.0] * 12))
    packet[-2:] = bytes([mode, product])

    packet_format = velodyne_packet_format(bytes(packet))

    assert not packet_format.supported
    assert name in packet_format.description
    with pytest.raises(DatasetError, match=name.split()[0]):
        decode_velodyne_packet(bytes(packet))


def test_vlp16_dual_return_keeps_distinct_second_returns_only() -> None:
    packet = _vlp16_dual_packet(6.0, 9.0, 10.0)
    assert velodyne_packet_format(packet).return_mode == "dual"

    points = decode_velodyne_packet(packet)

    # 6 firings x 2 sequences x 16 channels first returns, plus the one differing
    # channel of each sequence as a second return; identical second returns are dropped.
    assert points.xyz.shape[0] == 6 * 2 * 16 + 6 * 2
    second = points.second_return
    assert np.all(points.range_m[second] == pytest.approx(9.0, abs=0.002))
    assert np.all(points.laser[second] == 0)
    # A second return fires with its first return: same time and direction.
    first = (~second) & (points.laser == 0)
    assert np.array_equal(np.sort(points.time_s[second]), np.sort(points.time_s[first]))
    unit = points.xyz / np.linalg.norm(points.xyz, axis=1, keepdims=True)
    assert np.allclose(np.sort(unit[second], axis=0), np.sort(unit[first], axis=0))
    # Firing k starts at k x 110.592 us, not at the block index.
    assert points.time_s.max() == pytest.approx((5 * 110.592 + 55.296 + 15 * 2.304) * 1e-6)


def test_hdl64e_needs_its_calibration(tmp_path: Path) -> None:
    packet = _hdl64e_packet([10.0] * 12, 45.0)
    packet_format = velodyne_packet_format(packet)
    assert (packet_format.sensor, packet_format.return_mode) == ("HDL-64E", "single")
    with pytest.raises(DatasetError, match="calibration"):
        decode_velodyne_packet(packet)

    calibration = load_velodyne_calibration(_hdl64e_calibration_yaml(tmp_path / "c.yaml"))
    points = decode_velodyne_packet(packet, calibration)

    assert points.xyz.shape[0] == 12 * 32
    assert np.linalg.norm(points.xyz, axis=1) == pytest.approx(10.0, abs=1e-9)
    elevation = np.degrees(np.arcsin(points.xyz[:, 2] / 10.0))
    assert elevation == pytest.approx(-24.0 + 0.4 * points.laser, abs=1e-9)
    assert set(points.laser[:32]) == set(range(32))  # upper block
    assert set(points.laser[32:64]) == set(range(32, 64))  # lower block
    # Upper and lower blocks fire together; firings are 48 us apart.
    assert points.time_s[0] == 0.0 and points.time_s[32] == 0.0
    assert points.time_s[64] == pytest.approx(48e-6)


def test_hdl64e_vertical_offset_and_dual_return(tmp_path: Path) -> None:
    calibration = load_velodyne_calibration(
        _hdl64e_calibration_yaml(tmp_path / "c.yaml", vert_offset_m=0.2)
    )
    single = decode_velodyne_packet(_hdl64e_packet([10.0] * 12, 0.0), calibration)
    vertical = np.radians(-24.0 + 0.4 * single.laser)
    assert single.xyz[:, 2] == pytest.approx(10.0 * np.sin(vertical) + 0.2, abs=1e-9)

    ranges = [10.0, 10.0, 12.0, 10.0] * 3  # upper second return differs, lower repeats
    dual = decode_velodyne_packet(_hdl64e_packet(ranges, 0.0, dual=True), calibration)
    assert velodyne_packet_format(_hdl64e_packet(ranges, 0.0, dual=True)).return_mode == "dual"
    assert dual.xyz.shape[0] == 3 * 64 + 3 * 32
    assert np.all(dual.laser[dual.second_return] < 32)
    assert dual.time_s.max() == pytest.approx(2 * 57.6e-6 + (8 * 7.2 - 3.5) * 1e-6)


def test_db_xml_and_ros_yaml_calibrations_agree(tmp_path: Path) -> None:
    items = "".join(
        f"<item><px><id_>{laser}.0</id_><rotCorrection_>{0.5 * (laser % 3)}</rotCorrection_>"
        f"<vertCorrection_>{-20 + 0.5 * laser}</vertCorrection_>"
        f"<distCorrection_>{120 + laser}</distCorrection_>"
        f"<vertOffsetCorrection_>21.5</vertOffsetCorrection_>"
        f"<horizOffsetCorrection_>{2.6 if laser % 2 else -2.6}</horizOffsetCorrection_></px></item>"
        for laser in range(64)
    )
    db = tmp_path / "db.xml"
    db.write_text(
        f"<boost_serialization><DB><distLSB_>0.2</distLSB_><points_>{items}</points_></DB></boost_serialization>"
    )
    ros = tmp_path / "db.yaml"
    ros.write_text(
        yaml.safe_dump(
            {
                "lasers": [
                    {
                        "laser_id": laser,
                        "rot_correction": math.radians(0.5 * (laser % 3)),
                        "vert_correction": math.radians(-20 + 0.5 * laser),
                        "dist_correction": (120 + laser) / 100,
                        "vert_offset_correction": 0.215,
                        "horiz_offset_correction": 0.026 if laser % 2 else -0.026,
                    }
                    for laser in range(64)
                ]
            }
        )
    )
    packet = _hdl64e_packet([float(5 + block) for block in range(12)], 120.0)

    from_xml = decode_velodyne_packet(packet, load_velodyne_calibration(db))
    from_yaml = decode_velodyne_packet(packet, load_velodyne_calibration(ros))

    assert from_xml.xyz == pytest.approx(from_yaml.xyz, abs=1e-9)
    with pytest.raises(DatasetError, match="laser ids"):
        ros.write_text(
            yaml.safe_dump({"lasers": [{"laser_id": 3, "rot_correction": 0, "vert_correction": 0}]})
        )
        load_velodyne_calibration(ros)


def test_planner_names_the_packet_format(tmp_path: Path) -> None:
    unsupported = bytearray(_vlp32c_packet(5.0, [0.0] * 12))
    unsupported[-2:] = bytes([0x37, 0x21])
    source = tmp_path / "src.db3"
    _write_sqlite_bag(
        source,
        topics=[
            ("/a/velodyne_packets", VELODYNE_SCAN_TYPE),
            ("/b/velodyne_packets", VELODYNE_SCAN_TYPE),
        ],
        messages=[
            ("/a/velodyne_packets", 1_000, _scan_payload("a", [bytes(unsupported)])),
            ("/b/velodyne_packets", 1_000, _scan_payload("b", [_hdl64e_packet([5.0] * 12, 0.0)])),
        ],
    )
    records = [
        CheckTopicRecord(topic=topic, message_type=VELODYNE_SCAN_TYPE, role="lidar")
        for topic in ("/a/velodyne_packets", "/b/velodyne_packets")
    ]

    hdl32, hdl64 = flag_unreadable_pointclouds(source, records)

    assert hdl32.ignored_reason is not None and "HDL-32E" in hdl32.ignored_reason
    assert "does not decode" in hdl32.ignored_reason
    assert hdl64.ignored_reason is not None and "--calibration" in hdl64.ignored_reason


def test_convert_bag_applies_an_hdl64e_calibration(tmp_path: Path) -> None:
    source = tmp_path / "src.db3"
    _write_sqlite_bag(
        source,
        topics=[("/top/velodyne_packets", VELODYNE_SCAN_TYPE)],
        messages=[
            (
                "/top/velodyne_packets",
                1_000,
                _scan_payload("velodyne", [_hdl64e_packet([7.0] * 12, 0.0)]),
            )
        ],
    )
    with pytest.raises(DatasetError, match="calibration"):
        convert_velodyne_bag(source, tmp_path / "missing")
    calibration = load_velodyne_calibration(_hdl64e_calibration_yaml(tmp_path / "c.yaml"))

    summary = convert_velodyne_bag(source, tmp_path / "out", calibration=calibration)

    assert summary.calibration_sha256 == calibration.sha256
    ((_connection, _ts, data),) = list(iter_messages(tmp_path / "out"))
    cloud = decode_ros2_pointcloud2("/p", 0, data, point_time_field="time")
    assert cloud.point_count == 12 * 32
    assert np.linalg.norm(cloud.xyz, axis=1) == pytest.approx(7.0, abs=1e-5)


def _vls128_packet(
    range_m: float, sequence_azimuths_deg: list[float], *, mode: int = 0x37
) -> bytes:
    """Three firing sequences of four banks (0xEE, 0xDD, 0xCC, 0xBB), one azimuth each."""

    body = bytearray()
    for azimuth in sequence_azimuths_deg:
        for bank in (0xEE, 0xDD, 0xCC, 0xBB):
            body += bytes([0xFF, bank]) + struct.pack("<H", round(azimuth * 100) % 36000)
            body += struct.pack("<HB", round(range_m / 0.004), 12) * 32
    return bytes(body) + bytes(4) + bytes([mode, 0xA1])


def _vls128_calibration_yaml(path: Path, lasers: int = 128) -> Path:
    rows = [
        {
            "laser_id": laser,
            "rot_correction": 0.0,
            "vert_correction": math.radians(-25.0 + 0.25 * laser),
            "dist_correction": 0.0,
        }
        for laser in range(lasers)
    ]
    path.write_text(yaml.safe_dump({"lasers": rows, "distance_resolution": 0.004}))
    return path


def test_vls128_needs_its_128_laser_calibration(tmp_path: Path) -> None:
    packet = _vls128_packet(10.0, [0.0, 0.2, 0.4])
    packet_format = velodyne_packet_format(packet)
    assert (packet_format.sensor, packet_format.return_mode) == ("VLS-128", "strongest")
    assert packet_format.supported
    with pytest.raises(DatasetError, match="VLS-128 packets need the sensor's calibration"):
        decode_velodyne_packet(packet)
    hdl64 = load_velodyne_calibration(_hdl64e_calibration_yaml(tmp_path / "64.yaml"))
    with pytest.raises(DatasetError, match="128-laser calibration, got 64"):
        decode_velodyne_packet(packet, hdl64)


def test_vls128_banks_ranges_and_firing_times(tmp_path: Path) -> None:
    calibration = load_velodyne_calibration(_vls128_calibration_yaml(tmp_path / "c.yaml"))

    points = decode_velodyne_packet(_vls128_packet(10.0, [0.0, 0.2, 0.4]), calibration)

    assert points.xyz.shape[0] == 12 * 32
    assert np.linalg.norm(points.xyz, axis=1) == pytest.approx(10.0, abs=1e-9)
    elevation = np.degrees(np.arcsin(points.xyz[:, 2] / 10.0))
    assert elevation == pytest.approx(-25.0 + 0.25 * points.laser, abs=1e-9)
    # Each sequence carries all 128 lasers, bank by bank.
    for sequence in range(3):
        lasers = points.laser[sequence * 128 : (sequence + 1) * 128]
        assert lasers.tolist() == list(range(128))
    # Eight lasers per 2.89 us cycle, lasers 64-127 two cycles later, 7 us before
    # the stamp; sequences are 20 cycles apart.
    time_us = points.time_s * 1e6
    assert time_us[0] == pytest.approx(-7.0)
    assert time_us[8] == pytest.approx(-7.0 + 2.89)
    assert time_us[63] == pytest.approx(-7.0 + 7 * 2.89)
    assert time_us[64] == pytest.approx(-7.0 + 10 * 2.89)
    assert time_us[128] == pytest.approx(-7.0 + 57.8)


def test_vls128_azimuth_advances_within_a_sequence(tmp_path: Path) -> None:
    calibration = load_velodyne_calibration(_vls128_calibration_yaml(tmp_path / "c.yaml"))

    def azimuth_deg(points: object, index: int) -> float:
        x, y = points.xyz[index, 0], points.xyz[index, 1]  # type: ignore[attr-defined]
        return float(np.degrees(np.arctan2(x, y)))

    points = decode_velodyne_packet(_vls128_packet(10.0, [0.0, 0.2, 0.4]), calibration)
    # Laser 64 fires 10 of the sequence's 20 cycles in: half the 0.2 deg step.
    assert azimuth_deg(points, 64) == pytest.approx(0.1, abs=1e-9)
    assert azimuth_deg(points, 8) == pytest.approx(0.01, abs=1e-9)  # 1 cycle, rounded
    # The last sequence has no next one: it reuses the step before it.
    assert azimuth_deg(points, 256 + 64) == pytest.approx(0.5, abs=1e-9)

    # A step over 1 deg is a field-of-view gap: the previous step is reused.
    gap = decode_velodyne_packet(_vls128_packet(10.0, [0.0, 0.2, 5.0]), calibration)
    assert azimuth_deg(gap, 128 + 64) == pytest.approx(0.3, abs=1e-9)


def test_db_xml_with_a_repeated_declaration_and_lidarview_spelling(tmp_path: Path) -> None:
    rows = "".join(
        f"<px><id_>{laser}</id_><rotCorrection_>0.5</rotCorrection_>"
        f"<vertCorrection_>{-15 + 2 * laser}</vertCorrection_><distCorrection_>0</distCorrection_>"
        "<vertOffsetCorrection_>0</vertOffsetCorrection_>"
        "<horizOffsetCorection_>2.5</horizOffsetCorection_></px>"
        for laser in range(16)
    )
    path = tmp_path / "Alpha_Prime_like.xml"
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<!DOCTYPE boost_serialization><?xml version="1.0" ?>\n'
        f"<boost_serialization><DB><distLSB_>0.4</distLSB_><points_><item>{rows}</item>"
        "</points_></DB></boost_serialization>"
    )

    calibration = load_velodyne_calibration(path)

    assert calibration.distance_resolution_m == pytest.approx(0.004)
    table = calibration.laser_table()
    assert table["horiz_offset_m"] == pytest.approx(0.025)
    assert table["vert_correction_deg"][3] == pytest.approx(-9.0)


def test_planner_asks_for_a_vls128_calibration(tmp_path: Path) -> None:
    source = tmp_path / "src.db3"
    _write_sqlite_bag(
        source,
        topics=[("/vls/velodyne_packets", VELODYNE_SCAN_TYPE)],
        messages=[
            (
                "/vls/velodyne_packets",
                1_000,
                _scan_payload("vls", [_vls128_packet(5.0, [0.0, 0.2, 0.4])]),
            )
        ],
    )
    record = CheckTopicRecord(
        topic="/vls/velodyne_packets", message_type=VELODYNE_SCAN_TYPE, role="lidar"
    )

    (flagged,) = flag_unreadable_pointclouds(source, [record])

    assert flagged.ignored_reason is not None
    assert "VLS-128: pass the sensor's calibration with --calibration" in flagged.ignored_reason
