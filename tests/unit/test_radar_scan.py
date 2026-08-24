"""Contract tests for the standard radar_msgs RadarScan adapters."""

from __future__ import annotations

import json
import math
import sqlite3
import struct
from pathlib import Path

import jsonschema
import pytest

from calibrex.cli.main import main
from calibrex.core.capture_manifest import (
    SensorIdentity,
    capture_manifest_json_schema,
    inspect_capture,
    load_capture_manifest,
)
from calibrex.core.exceptions import DatasetError
from calibrex.data.ros_cdr import decode_ros2_radar_scan
from calibrex.data.ros_messages import (
    MAX_RADAR_RETURNS,
    RADAR_MSGS_LICENSE,
    RADAR_MSGS_ROS1_SPEC_COMMIT,
    RADAR_MSGS_ROS2_SPEC_COMMIT,
)
from calibrex.data.rosbag1 import decode_radar_scan
from calibrex.data.rosbag2 import decode_rosbag2_message


def _cdr_scan(
    frame: str = "radar_front",
    returns: tuple[tuple[float, ...], ...] = (),
) -> bytes:
    payload = bytearray(b"\x00\x01\x00\x00")
    payload += struct.pack("<iI", 4, 5)
    encoded_frame = frame.encode("utf-8") + b"\x00"
    payload += struct.pack("<I", len(encoded_frame)) + encoded_frame
    payload += b"\x00" * ((-(len(payload) - 4)) % 4)
    payload += struct.pack("<I", len(returns))
    for values in returns:
        payload += struct.pack("<5f", *values)
    return bytes(payload)


def _ros1_scan(
    frame: str = "radar_front",
    returns: tuple[tuple[float, ...], ...] = (),
) -> bytes:
    payload = bytearray(struct.pack("<III", 1, 4, 5))
    encoded_frame = frame.encode("utf-8") + b"\x00"
    payload += struct.pack("<I", len(encoded_frame)) + encoded_frame
    payload += struct.pack("<I", len(returns))
    for values in returns:
        payload += struct.pack("<5f", *values)
    return bytes(payload)


def _returns() -> tuple[tuple[float, ...], ...]:
    return ((10.0, 0.2, 0.1, -3.0, 8.0), (20.0, -0.2, -0.1, 2.0, 9.0))


def test_ros1_ros2_decoders_normalize_to_equal_returns() -> None:
    ros1 = decode_radar_scan("/radar", 1, _ros1_scan(returns=_returns()))
    ros2 = decode_ros2_radar_scan("/radar", 1, _cdr_scan(returns=_returns()))
    assert ros1.returns == ros2.returns
    assert ros1.return_count == 2
    assert ros1.diversity_status == "strong"
    assert ros1.range_min_m == pytest.approx(10.0)
    assert ros1.doppler_span_mps == pytest.approx(5.0)
    assert ros1.to_xyz()[0][0] == pytest.approx(10.0 * math.cos(0.1) * math.cos(0.2))
    assert ros1.source_spec_provenance["license"] == RADAR_MSGS_LICENSE
    assert ros1.source_spec_provenance["commit"] == RADAR_MSGS_ROS1_SPEC_COMMIT
    assert ros2.source_spec_provenance["commit"] == RADAR_MSGS_ROS2_SPEC_COMMIT
    assert decode_rosbag2_message(
        "/radar", "radar_msgs/msg/RadarScan", 1, _cdr_scan(returns=_returns())
    ).returns == ros1.returns


@pytest.mark.parametrize(
    "decoder,payload",
    [
        (decode_ros2_radar_scan, _cdr_scan(returns=((1.0, 0.0, 0.0, 0.0, 0.0),))[:-1]),
        (decode_radar_scan, _ros1_scan(returns=((1.0, 0.0, 0.0, 0.0, 0.0),))[:-1]),
    ],
)
def test_truncated_payloads_fail_closed(decoder: object, payload: bytes) -> None:
    with pytest.raises(DatasetError):
        decoder("/radar", 1, payload)  # type: ignore[operator]


@pytest.mark.parametrize(
    "decoder,payload",
    [(decode_ros2_radar_scan, _cdr_scan()), (decode_radar_scan, _ros1_scan())],
)
def test_empty_scan_is_valid_but_weak(decoder: object, payload: bytes) -> None:
    message = decoder("/radar", 1, payload)  # type: ignore[operator]
    assert message.return_count == 0
    assert message.diversity_status == "empty"
    assert message.diversity_summary()["status"] == "empty"


@pytest.mark.parametrize(
    "values",
    [
        (math.nan, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0),
        (1.0, 4.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 2.0, 0.0, 0.0),
    ],
)
def test_invalid_returns_are_rejected(values: tuple[float, ...]) -> None:
    with pytest.raises(DatasetError):
        decode_ros2_radar_scan("/radar", 1, _cdr_scan(returns=(values,)))
    with pytest.raises(DatasetError):
        decode_radar_scan("/radar", 1, _ros1_scan(returns=(values,)))


def test_oversized_sequence_is_rejected_before_iteration() -> None:
    cdr = bytearray(_cdr_scan())
    cdr[-4:] = struct.pack("<I", MAX_RADAR_RETURNS + 1)
    with pytest.raises(DatasetError, match="exceeds safe bound"):
        decode_ros2_radar_scan("/radar", 1, bytes(cdr))
    ros1 = bytearray(_ros1_scan())
    ros1[-4:] = struct.pack("<I", MAX_RADAR_RETURNS + 1)
    with pytest.raises(DatasetError, match="exceeds safe bound"):
        decode_radar_scan("/radar", 1, bytes(ros1))


def _write_radar_bag(path: Path, frames: tuple[str, ...]) -> None:
    database = sqlite3.connect(path)
    try:
        database.executescript(
            """
            CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT, type TEXT,
                                serialization_format TEXT, offered_qos_profiles TEXT);
            CREATE TABLE messages(id INTEGER PRIMARY KEY, topic_id INTEGER,
                                  timestamp INTEGER, data BLOB);
            """
        )
        database.execute(
            "INSERT INTO topics VALUES (1, '/radar', 'radar_msgs/msg/RadarScan', 'cdr', '')"
        )
        for index, frame in enumerate(frames, start=1):
            database.execute(
                "INSERT INTO messages VALUES (?, 1, ?, ?)",
                (index, index, _cdr_scan(frame, _returns())),
            )
        database.commit()
    finally:
        database.close()


def _radar_sensor() -> SensorIdentity:
    return SensorIdentity(
        sensor_id="radar-front",
        type="radar",
        serial="RADAR-001",
        model="fixture-radar",
        firmware="1.0",
        mount_id="mount-front",
        frame_id="radar_front",
    )


def test_capture_records_radar_evidence_and_blocks_frame_conflict(tmp_path: Path) -> None:
    bag = tmp_path / "radar.db3"
    _write_radar_bag(bag, ("radar_front", "radar_rear"))
    manifest = inspect_capture(
        bag,
        source_format="rosbag2",
        capture_id="capture-1",
        session_id="session-1",
        vehicle_id="vehicle-1",
        sensor_kit_id="kit-1",
        sensors=[_radar_sensor()],
    )
    stream = manifest.streams[0]
    assert stream.decode_status == "supported"
    assert stream.radar_return_count == 4
    assert stream.radar_diversity_status == "strong"
    assert stream.frame_conflict is True
    assert (
        next(check for check in manifest.checks if check.check_id == "streams.frame").status
        == "blocked"
    )
    assert manifest.status == "blocked"


def test_capture_cli_emits_schema_valid_radar_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bag = tmp_path / "radar.db3"
    _write_radar_bag(bag, ("radar_front",))
    output = tmp_path / "manifest.json"
    assert main(
        [
            "capture",
            "inspect",
            str(bag),
            "--type",
            "rosbag2",
            "--capture-id",
            "capture-1",
            "--session-id",
            "session-1",
            "--vehicle-id",
            "vehicle-1",
            "--sensor-kit-id",
            "kit-1",
            "--sensor",
            "radar-front:radar:RADAR-001:fixture-radar:1.0:mount-front:radar_front",
            "--output",
            str(output),
            "--json",
        ]
    ) == 0
    capsys.readouterr()
    manifest = load_capture_manifest(output)
    jsonschema.validate(
        manifest.model_dump(mode="json", exclude_none=False), capture_manifest_json_schema()
    )
    assert json.loads(output.read_text(encoding="utf-8"))["streams"][0]["kind"] == "radar"
