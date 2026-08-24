"""MCAP framing and integrity evidence fixtures."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

from calibrex.core.capture_manifest import SensorIdentity, inspect_capture
from calibrex.data.mcap import MCAP_MAGIC, inspect_mcap_integrity


def _string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<I", len(encoded)) + encoded


def _record(opcode: int, content: bytes) -> bytes:
    return bytes([opcode]) + struct.pack("<Q", len(content)) + content


def _schema(schema_id: int = 1, name: str = "example/Message") -> bytes:
    return _record(
        0x03,
        struct.pack("<H", schema_id)
        + _string(name)
        + _string("ros2msg")
        + struct.pack("<I", 0),
    )


def _channel(channel_id: int = 1, schema_id: int = 1) -> bytes:
    return _record(
        0x04,
        struct.pack("<HH", channel_id, schema_id)
        + _string("/example")
        + _string("cdr")
        + struct.pack("<I", 0),
    )


def _message(channel_id: int = 1) -> bytes:
    return _record(0x05, struct.pack("<HIQQ", channel_id, 0, 10, 10) + b"payload")


def _write_mcap(
    path: Path,
    *,
    data_crc: bool = True,
    summary: bool = True,
    summary_crc: bool = True,
    body: bytes | None = None,
) -> Path:
    header = _record(0x01, _string("ros2") + _string("calibrex-test"))
    body = body if body is not None else _schema() + _channel() + _message()
    data_prefix = MCAP_MAGIC + header + body
    declared_data_crc = zlib.crc32(data_prefix) & 0xFFFFFFFF if data_crc else 0
    data_end = _record(0x0F, struct.pack("<I", declared_data_crc))

    summary_bytes = b""
    summary_start = 0
    summary_offset_start = 0
    if summary:
        summary_start = len(data_prefix) + len(data_end)
        summary_schema = _schema()
        summary_channel = _channel()
        summary_offset_start = summary_start + len(summary_schema) + len(summary_channel)
        summary_offset = _record(
            0x0E,
            bytes([0x03]) + struct.pack("<QQ", summary_start, len(summary_schema)),
        )
        summary_bytes = summary_schema + summary_channel + summary_offset

    footer_offset = len(data_prefix) + len(data_end) + len(summary_bytes)
    footer_prefix = _record(0x02, struct.pack("<QQI", summary_start, summary_offset_start, 0))
    if summary_crc and summary:
        summary_crc_value = zlib.crc32(
            summary_bytes + footer_prefix[: 1 + 8 + 8 + 8]
        ) & 0xFFFFFFFF
    else:
        summary_crc_value = 0
    footer = _record(
        0x02,
        struct.pack("<QQI", summary_start, summary_offset_start, summary_crc_value),
    )
    assert footer_offset == len(data_prefix) + len(data_end) + len(summary_bytes)
    path.write_bytes(data_prefix + data_end + summary_bytes + footer + MCAP_MAGIC)
    return path


def test_valid_unchunked_footer_data_crc_and_index_evidence(tmp_path: Path) -> None:
    evidence = inspect_mcap_integrity(_write_mcap(tmp_path / "valid.mcap"))

    assert evidence.status == "pass"
    assert evidence.magic_status == "known"
    assert evidence.framing_status == "known"
    assert evidence.links_status == "known"
    assert evidence.data_crc_status == "known"
    assert evidence.summary_crc_status == "known"
    assert evidence.index_status == "known"
    assert evidence.summary_present is True
    assert evidence.summary_offset_present is True


def test_no_summary_fastwrite_is_warn_and_keeps_index_unknown(tmp_path: Path) -> None:
    evidence = inspect_mcap_integrity(
        _write_mcap(tmp_path / "fastwrite.mcap", summary=False, summary_crc=False)
    )

    assert evidence.status == "warn"
    assert evidence.summary_status == "unknown"
    assert evidence.index_status == "unknown"
    assert any("Summary section is absent" in warning for warning in evidence.warnings)


def test_bad_data_crc_blocks(tmp_path: Path) -> None:
    path = _write_mcap(tmp_path / "bad-crc.mcap")
    payload = bytearray(path.read_bytes())
    # The DataEnd CRC is immediately after its 9-byte record header.  Locate
    # it from the first known body records rather than relying on a fixture
    # magic constant.
    data_end_offset = len(MCAP_MAGIC) + len(
        _record(0x01, _string("ros2") + _string("calibrex-test"))
    )
    data_end_offset += len(_schema()) + len(_channel()) + len(_message())
    payload[data_end_offset + 9] ^= 0xFF
    path.write_bytes(payload)

    evidence = inspect_mcap_integrity(path)
    assert evidence.status == "blocked"
    assert evidence.data_crc_status == "blocked"
    assert any("data_section_crc mismatch" in error for error in evidence.errors)


def test_truncated_footer_or_magic_blocks(tmp_path: Path) -> None:
    path = _write_mcap(tmp_path / "truncated.mcap")
    path.write_bytes(path.read_bytes()[:-3])

    evidence = inspect_mcap_integrity(path)
    assert evidence.status == "blocked"
    assert any(
        "magic" in error.lower() or "truncated" in error.lower()
        for error in evidence.errors
    )


def test_dangling_channel_schema_link_blocks(tmp_path: Path) -> None:
    path = _write_mcap(
        tmp_path / "dangling.mcap",
        summary=False,
        body=_channel(schema_id=99) + _message(),
    )

    evidence = inspect_mcap_integrity(path)
    assert evidence.status == "blocked"
    assert evidence.links_status == "blocked"
    assert any("missing Schema 99" in error for error in evidence.errors)


def test_unsupported_compressed_chunk_blocks(tmp_path: Path) -> None:
    inner_records = _schema() + _channel() + _message()
    chunk_content = (
        struct.pack("<QQQI", 10, 10, len(inner_records), 0)
        + _string("brotli")
        + struct.pack("<Q", len(inner_records))
        + inner_records
    )
    evidence = inspect_mcap_integrity(
        _write_mcap(
            tmp_path / "unsupported-compression.mcap",
            summary=False,
            body=_record(0x06, chunk_content),
        )
    )

    assert evidence.status == "blocked"
    assert evidence.compression_status == "blocked"
    assert evidence.unsupported_compressions == ["brotli"]


def test_capture_manifest_carries_mcap_integrity_evidence(tmp_path: Path) -> None:
    manifest = inspect_capture(
        _write_mcap(tmp_path / "manifest.mcap", summary=False, summary_crc=False),
        source_format="mcap",
        capture_id="capture-1",
        session_id="session-1",
        vehicle_id="vehicle-1",
        sensor_kit_id="kit-1",
        sensors=[
            SensorIdentity(
                sensor_id="lidar-front",
                type="lidar",
                serial="SN-1",
                model="fixture",
                firmware="1",
                mount_id="mount",
                frame_id="lidar",
            )
        ],
    )

    assert manifest.source is not None
    assert manifest.source.mcap_integrity is not None
    assert manifest.source.mcap_integrity.status == "warn"
    assert any(check.check_id == "source.mcap_integrity" for check in manifest.checks)
