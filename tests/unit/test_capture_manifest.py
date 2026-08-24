"""Capture/session manifest contract tests."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import jsonschema
import pytest
from pydantic import ValidationError

from calibrex.cli.main import main
from calibrex.core.capture_manifest import (
    CaptureManifest,
    SensorIdentity,
    SourceInventory,
    StreamInventory,
    capture_manifest_json_schema,
    inspect_capture,
    load_capture_manifest,
    verify_capture_manifest_inputs,
)


def _complete_sensor() -> SensorIdentity:
    return SensorIdentity(
        sensor_id="lidar-front",
        type="lidar",
        serial="SN-001",
        model="fixture-lidar",
        firmware="1.0.0",
        mount_id="mount-front",
        frame_id="lidar_front",
    )


def _write_rosbag2_fixture(path: Path) -> Path:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE topics(
              id INTEGER PRIMARY KEY,
              name TEXT NOT NULL,
              type TEXT NOT NULL,
              serialization_format TEXT NOT NULL,
              offered_qos_profiles TEXT NOT NULL
            );
            CREATE TABLE messages(
              id INTEGER PRIMARY KEY,
              topic_id INTEGER NOT NULL,
              timestamp INTEGER NOT NULL,
              data BLOB NOT NULL
            );
            CREATE INDEX timestamp_idx ON messages (timestamp ASC);
            """
        )
        topics = [
            (1, "/lidar", "sensor_msgs/msg/PointCloud2"),
            (2, "/camera/image", "sensor_msgs/msg/Image"),
            (3, "/radar", "radar_msgs/msg/RadarScan"),
        ]
        connection.executemany(
            "INSERT INTO topics (id, name, type, serialization_format, offered_qos_profiles) "
            "VALUES (?, ?, ?, ?, ?)",
            [(index, name, message_type, "cdr", "") for index, name, message_type in topics],
        )
        connection.executemany(
            "INSERT INTO messages (id, topic_id, timestamp, data) VALUES (?, ?, ?, ?)",
            [
                (index, topic_id, timestamp, b"not-decoded")
                for index, (topic_id, timestamp) in enumerate(
                    [(1, 10), (2, 20), (3, 30)], start=1
                )
            ],
        )
        connection.commit()
    finally:
        connection.close()
    return path


def test_rosbag2_inventory_keeps_unsupported_streams_and_unknown_integrity(
    tmp_path: Path,
) -> None:
    bag = _write_rosbag2_fixture(tmp_path / "capture.db3")
    artifact = inspect_capture(
        bag,
        source_format="rosbag2",
        capture_id="capture-1",
        session_id="session-1",
        vehicle_id="vehicle-1",
        sensor_kit_id="kit-1",
        sensors=[_complete_sensor()],
    )

    by_topic = {stream.topic: stream for stream in artifact.streams}
    assert set(by_topic) == {"/camera/image", "/lidar", "/radar"}
    # The built-in Image adapter is available; this intentionally malformed
    # payload is therefore ``unknown`` rather than ``unsupported``.
    assert by_topic["/camera/image"].decode_status == "unknown"
    # The standard RadarScan type is recognized, but this intentionally
    # malformed sample cannot be claimed supported until a valid decode.
    assert by_topic["/radar"].decode_status == "unknown"
    assert artifact.source is not None
    assert artifact.source.index_status == "unknown"
    assert artifact.source.crc_status == "unknown"
    assert artifact.status == "blocked"
    assert all(
        check.status != "pass" or check.check_id != "streams.capability"
        for check in artifact.checks
    )


def test_plain_file_inventory_does_not_claim_ready_for_image(tmp_path: Path) -> None:
    (tmp_path / "frame.png").write_bytes(b"not-a-decoded-image")
    artifact = inspect_capture(
        tmp_path,
        source_format="files",
        capture_id="capture-1",
        session_id="session-1",
        vehicle_id="vehicle-1",
        sensor_kit_id="kit-1",
        sensors=[_complete_sensor()],
    )
    assert artifact.sources[0].format == "files"
    assert artifact.streams[0].decode_status == "unsupported"
    assert artifact.status == "blocked"


def test_missing_identity_is_blocked(tmp_path: Path) -> None:
    source = tmp_path / "capture.bin"
    source.write_bytes(b"capture")
    artifact = inspect_capture(source, source_format="files")
    identity_check = next(check for check in artifact.checks if check.check_id == "identity.ids")
    assert identity_check.status == "blocked"
    assert artifact.status == "blocked"


def test_manifest_digest_is_deterministic_and_tamper_evident(tmp_path: Path) -> None:
    source = tmp_path / "capture.bin"
    source.write_bytes(b"capture")
    kwargs = {
        "source_format": "files",
        "capture_id": "capture-1",
        "session_id": "session-1",
        "vehicle_id": "vehicle-1",
        "sensor_kit_id": "kit-1",
        "sensors": [_complete_sensor()],
    }
    first = inspect_capture(source, **kwargs)
    second = inspect_capture(source, **kwargs)
    assert first.model_dump(mode="json", exclude_none=False) == second.model_dump(
        mode="json", exclude_none=False
    )
    output = tmp_path / "manifest.json"
    first.save(output)
    loaded = load_capture_manifest(output)
    assert loaded.artifact_sha256 == first.artifact_sha256
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["summary"] = "tampered"
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="self-digest mismatch"):
        load_capture_manifest(output)


def test_capture_manifest_schema_and_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "capture.bin"
    source.write_bytes(b"capture")
    output = tmp_path / "manifest.yaml"
    assert (
        main(
            [
                "capture",
                "inspect",
                str(source),
                "--type",
                "files",
                "--capture-id",
                "capture-1",
                "--session-id",
                "session-1",
                "--vehicle-id",
                "vehicle-1",
                "--sensor-kit-id",
                "kit-1",
                "--sensor",
                "lidar-front:lidar:SN-001:fixture-lidar:1.0.0:mount-front:lidar_front",
                "--output",
                str(output),
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    payload = json.loads(output.read_text(encoding="utf-8")) if output.suffix == ".json" else None
    assert output.is_file()
    artifact = load_capture_manifest(output)
    jsonschema.validate(
        artifact.model_dump(mode="json", exclude_none=False),
        capture_manifest_json_schema(),
    )
    assert payload is None


def test_input_verification_detects_source_and_config_changes(tmp_path: Path) -> None:
    source = tmp_path / "capture.bin"
    config = tmp_path / "config.yaml"
    source.write_bytes(b"capture-v1")
    config.write_text("profile: commercial\n", encoding="utf-8")
    manifest = inspect_capture(
        source,
        source_format="files",
        capture_id="capture-1",
        session_id="session-1",
        vehicle_id="vehicle-1",
        sensor_kit_id="kit-1",
        config_path=config,
    )
    output = tmp_path / "manifest.json"
    manifest.save(output)
    assert verify_capture_manifest_inputs(output).valid

    source.write_bytes(b"capture-v2")
    source_report = verify_capture_manifest_inputs(output)
    assert not source_report.valid
    assert source_report.source_inputs[0].status == "mismatch"

    source.write_bytes(b"capture-v1")
    config.write_text("profile: debug\n", encoding="utf-8")
    config_report = verify_capture_manifest_inputs(output)
    assert not config_report.valid
    assert config_report.config_input is not None
    assert config_report.config_input.status == "mismatch"


def test_input_verification_classifies_missing_path_and_self_tamper(tmp_path: Path) -> None:
    source = tmp_path / "capture.bin"
    source.write_bytes(b"capture")
    output = tmp_path / "manifest.json"
    inspect_capture(
        source,
        source_format="files",
        capture_id="capture-1",
        session_id="session-1",
        vehicle_id="vehicle-1",
        sensor_kit_id="kit-1",
    ).save(output)
    source.unlink()
    missing = verify_capture_manifest_inputs(output)
    assert not missing.valid
    assert missing.source_inputs[0].status == "missing"

    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["summary"] = "tampered"
    output.write_text(json.dumps(payload), encoding="utf-8")
    tampered = verify_capture_manifest_inputs(output)
    assert not tampered.valid
    assert tampered.self_digest_status == "mismatch"


def test_manifest_rejects_duplicate_ids_dangling_refs_and_bad_time_fields() -> None:
    source = SourceInventory(source_id="source-1", path="capture.bin", exists=True)
    sensor = _complete_sensor()
    stream = StreamInventory(stream_id="stream-1", sensor_id=sensor.sensor_id)
    with pytest.raises(ValidationError, match="duplicate IDs"):
        CaptureManifest(
            sources=[source, source],
            sensors=[sensor],
            streams=[stream],
            artifact_sha256="a" * 64,
        )
    with pytest.raises(ValidationError, match="undeclared sensor IDs"):
        CaptureManifest(
            sources=[source],
            sensors=[sensor],
            streams=[StreamInventory(stream_id="stream-1", sensor_id="missing")],
            artifact_sha256="a" * 64,
        )
    with pytest.raises(ValidationError, match="first_timestamp_ns"):
        StreamInventory(stream_id="bad", first_timestamp_ns=2, last_timestamp_ns=1)


def test_declared_profile_makes_optional_debug_stream_non_gating(tmp_path: Path) -> None:
    source = tmp_path / "capture.bin"
    source.write_bytes(b"capture")
    manifest = inspect_capture(
        source,
        source_format="files",
        capture_id="capture-1",
        session_id="session-1",
        vehicle_id="vehicle-1",
        sensor_kit_id="kit-1",
        readiness_profile="declared",
        required_streams=[source.as_posix()],
    )
    assert manifest.status == "blocked"
    assert any(check.check_id == "streams.policy" for check in manifest.checks)


def test_alias_collision_is_rejected() -> None:
    with pytest.raises(ValidationError, match="ambiguous field aliases"):
        SourceInventory.model_validate(
            {
                "source_id": "source-1",
                "path": "capture.bin",
                "format": "files",
                "source_format": "files",
            }
        )
