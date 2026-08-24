"""Pure-Python MCAP adapter and bounded integrity inspection.

The core package does not depend on the optional ``mcap`` package.  Stream
inspection remains intentionally small, while :func:`inspect_mcap_integrity`
checks the framing and integrity fields that are available directly in an
MCAP file.  It does not decompress chunks: a supported-but-unavailable codec
therefore leaves the relevant CRC/link evidence ``unknown``; an unrecognised
codec is explicitly reported as unsupported and blocks intake.
"""

from __future__ import annotations

import importlib.util
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from calibrex.core.mcap_integrity import McapFieldStatus, McapIntegrityEvidence
from calibrex.data.base import StreamSummary

MCAP_MAGIC = b"\x89MCAP0\r\n"

OP_HEADER = 0x01
OP_FOOTER = 0x02
OP_SCHEMA = 0x03
OP_CHANNEL = 0x04
OP_MESSAGE = 0x05
OP_CHUNK = 0x06
OP_MESSAGE_INDEX = 0x07
OP_CHUNK_INDEX = 0x08
OP_ATTACHMENT = 0x09
OP_ATTACHMENT_INDEX = 0x0A
OP_STATISTICS = 0x0B
OP_METADATA = 0x0C
OP_METADATA_INDEX = 0x0D
OP_SUMMARY_OFFSET = 0x0E
OP_DATA_END = 0x0F

_KNOWN_OPCODES = frozenset(
    {
        OP_HEADER,
        OP_FOOTER,
        OP_SCHEMA,
        OP_CHANNEL,
        OP_MESSAGE,
        OP_CHUNK,
        OP_MESSAGE_INDEX,
        OP_CHUNK_INDEX,
        OP_ATTACHMENT,
        OP_ATTACHMENT_INDEX,
        OP_STATISTICS,
        OP_METADATA,
        OP_METADATA_INDEX,
        OP_SUMMARY_OFFSET,
        OP_DATA_END,
    }
)
_DATA_OPCODES = frozenset(
    {
        OP_SCHEMA,
        OP_CHANNEL,
        OP_MESSAGE,
        OP_CHUNK,
        OP_MESSAGE_INDEX,
        OP_ATTACHMENT,
        OP_METADATA,
        OP_DATA_END,
    }
)
_SUMMARY_OPCODES = frozenset(
    {
        OP_SCHEMA,
        OP_CHANNEL,
        OP_CHUNK_INDEX,
        OP_ATTACHMENT_INDEX,
        OP_METADATA_INDEX,
        OP_STATISTICS,
    }
)
_SUPPORTED_CHUNK_COMPRESSIONS = frozenset({"", "none", "lz4", "zstd"})
_DIAGNOSTIC_LIMIT = 32
# Large records are still framed and CRC-scanned without being retained in
# memory.  Parsing links inside a record this large is reported unknown.
_MAX_RETAINED_RECORD_BYTES = 16 * 1024 * 1024
_STREAM_BUFFER_BYTES = 1024 * 1024


def inspect_mcap(path: str | Path) -> tuple[list[StreamSummary], list[str]]:
    """Inspect MCAP streams when the optional dependency is available."""

    mcap_path = Path(path)
    warnings: list[str] = []
    if not mcap_path.exists():
        warnings.append("MCAP path does not exist")
        return [], warnings

    if importlib.util.find_spec("mcap") is None:
        warnings.append("MCAP inspection requires optional dependency calibrex[mcap].")
        return [], warnings

    warnings.append("MCAP channel decoding is not enabled in this alpha adapter.")
    return [], warnings


class _McapParseError(ValueError):
    """A bounded record-content parse failure."""


@dataclass(frozen=True)
class _Record:
    offset: int
    opcode: int
    length: int
    content: bytes | None
    crc_after: int | None


@dataclass
class _ScanState:
    file_size: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    unsupported_compressions: list[str] = field(default_factory=list)
    schema_ids: set[int] = field(default_factory=set)
    channel_schema_ids: dict[int, int] = field(default_factory=dict)
    message_channel_ids: set[int] = field(default_factory=set)
    summary_record_offsets: list[int] = field(default_factory=list)
    summary_record_meta: list[tuple[int, int, int]] = field(default_factory=list)
    summary_offset_record_offsets: list[int] = field(default_factory=list)
    summary_opcodes: list[int] = field(default_factory=list)
    summary_offset_opcodes: list[int] = field(default_factory=list)
    summary_offset_groups: list[tuple[int, int, int]] = field(default_factory=list)
    records_checked: int = 0
    data_records_checked: int = 0
    summary_records_checked: int = 0
    schema_count: int = 0
    channel_count: int = 0
    message_count: int = 0
    chunk_count: int = 0
    index_count: int = 0
    chunk_count_with_crc: int = 0
    chunk_count_without_crc: int = 0
    chunk_crc_mismatch: bool = False
    chunk_compressed_unknown: bool = False
    chunk_link_unknown: bool = False
    data_end_offset: int | None = None
    data_end_record_end: int | None = None
    data_crc_expected: int | None = None
    data_crc_actual: int | None = None
    footer_offset: int | None = None
    footer_record_end: int | None = None
    summary_start: int = 0
    summary_offset_start: int = 0
    summary_crc_expected: int | None = None
    summary_crc_actual: int | None = None
    summary_present: bool = False
    summary_offset_present: bool = False
    leading_magic_valid: bool = False
    trailing_magic_valid: bool = False
    header_valid: bool = False
    data_end_valid: bool = False
    footer_valid: bool = False
    framing_valid: bool = False

    def error(self, message: str) -> None:
        _bounded_append(self.errors, message)

    def warning(self, message: str) -> None:
        _bounded_append(self.warnings, message)

    def unsupported(self, compression: str) -> None:
        if compression not in self.unsupported_compressions:
            _bounded_append(self.unsupported_compressions, compression)


def _bounded_append(items: list[str], value: str) -> None:
    if len(items) < _DIAGNOSTIC_LIMIT:
        items.append(value[:512])


def inspect_mcap_integrity(path: str | Path) -> McapIntegrityEvidence:
    """Validate MCAP framing, links, optional indexes, and available CRCs.

    The scanner is streaming and retains at most a bounded prefix of one
    record.  CRCs over large records and over the data section are still
    computed while bytes are read.  Chunk CRCs that cover *uncompressed*
    bytes are checked for uncompressed chunks; compressed chunks remain
    ``unknown`` unless their compression is not recognised at all, in which
    case the evidence is blocking.
    """

    mcap_path = Path(path)
    if not mcap_path.exists():
        return _blocked_evidence(None, "MCAP path does not exist")
    if not mcap_path.is_file():
        return _blocked_evidence(None, "MCAP path is not a regular file")
    try:
        file_size = mcap_path.stat().st_size
    except OSError as exc:
        return _blocked_evidence(None, f"MCAP stat failed: {type(exc).__name__}: {exc}")

    state = _ScanState(file_size=file_size)
    try:
        with mcap_path.open("rb") as source:
            _scan_mcap_file(source, state)
    except OSError as exc:
        state.error(f"MCAP read failed: {type(exc).__name__}: {exc}")
    except _McapParseError as exc:
        state.error(str(exc))

    return _evidence_from_state(state, mcap_path)


def _blocked_evidence(file_size: int | None, error: str) -> McapIntegrityEvidence:
    return McapIntegrityEvidence(
        status="blocked",
        file_size_bytes=file_size,
        magic_status="blocked",
        framing_status="blocked",
        errors=[error],
    )


def _scan_mcap_file(source: BinaryIO, state: _ScanState) -> None:
    leading_magic = source.read(len(MCAP_MAGIC))
    if leading_magic != MCAP_MAGIC:
        state.error("invalid leading MCAP magic")
        return
    state.leading_magic_valid = True
    data_crc = zlib.crc32(leading_magic) & 0xFFFFFFFF

    header = _read_record(source, state.file_size, data_crc)
    if header is None:
        state.error("missing MCAP Header record")
        return
    state.records_checked += 1
    if header.opcode != OP_HEADER:
        state.error(f"first MCAP record is opcode 0x{header.opcode:02x}, expected Header")
        return
    if header.content is None:
        state.error("MCAP Header record is too large to parse")
        return
    try:
        _parse_header(header.content)
        state.header_valid = True
    except _McapParseError as exc:
        state.error(f"invalid MCAP Header: {exc}")
        return
    if header.crc_after is None:
        state.error("MCAP Header CRC scan was unavailable")
        return
    data_crc = header.crc_after

    # Data section.  The DataEnd record itself is excluded from the CRC input.
    while True:
        record = _read_record(source, state.file_size, data_crc)
        if record is None:
            state.error("truncated MCAP data section before DataEnd")
            return
        state.records_checked += 1
        if record.opcode == OP_DATA_END:
            state.data_records_checked += 1
            state.data_end_offset = record.offset
            state.data_end_record_end = record.offset + 9 + record.length
            if record.content is None:
                state.error("MCAP DataEnd record is too large to parse")
            elif len(record.content) != 4:
                state.error("MCAP DataEnd record must contain exactly 4 bytes")
            else:
                state.data_crc_expected = struct.unpack_from("<I", record.content)[0]
                state.data_end_valid = True
                state.data_crc_actual = data_crc & 0xFFFFFFFF
                if state.data_crc_expected != 0:
                    if state.data_crc_actual != state.data_crc_expected:
                        state.error(
                            "MCAP data_section_crc mismatch: "
                            f"declared=0x{state.data_crc_expected:08x}, "
                            f"actual=0x{state.data_crc_actual:08x}"
                        )
                else:
                    state.warning("MCAP DataEnd data_section_crc is unavailable (zero)")
            break
        state.data_records_checked += 1
        if record.opcode not in _DATA_OPCODES:
            state.error(
                f"MCAP opcode 0x{record.opcode:02x} is not allowed in the data section"
            )
        _scan_record(record, state, section="data")
        if record.crc_after is None:
            state.error("MCAP data-section CRC scan was unavailable")
            return
        data_crc = record.crc_after

    # Summary and Summary Offset sections are optional, but when present the
    # Footer offsets must agree with the actual record boundaries.
    while True:
        record = _read_record(source, state.file_size, None)
        if record is None:
            state.error("truncated MCAP file before Footer")
            return
        state.records_checked += 1
        if record.opcode == OP_FOOTER:
            state.footer_offset = record.offset
            state.footer_record_end = record.offset + 9 + record.length
            if record.content is None:
                state.error("MCAP Footer record is too large to parse")
            elif len(record.content) != 20:
                state.error("MCAP Footer record must contain exactly 20 bytes")
            else:
                state.summary_start, state.summary_offset_start, summary_crc = struct.unpack(
                    "<QQI", record.content
                )
                state.summary_crc_expected = summary_crc
                state.footer_valid = True
            break
        state.summary_records_checked += 1
        state.summary_record_offsets.append(record.offset)
        state.summary_record_meta.append((record.offset, record.opcode, record.length))
        state.summary_opcodes.append(record.opcode)
        if record.opcode not in _SUMMARY_OPCODES and record.opcode != OP_SUMMARY_OFFSET:
            state.error(
                f"MCAP opcode 0x{record.opcode:02x} is not allowed in the summary section"
            )
        if record.opcode == OP_SUMMARY_OFFSET:
            state.summary_offset_record_offsets.append(record.offset)
            state.summary_offset_opcodes.append(record.opcode)
        _scan_record(record, state, section="summary")

    trailing_magic = source.read(len(MCAP_MAGIC))
    if trailing_magic == MCAP_MAGIC:
        state.trailing_magic_valid = True
    else:
        state.error("invalid or missing trailing MCAP magic")
    if source.read(1):
        state.error("bytes follow the trailing MCAP magic")

    _validate_sections_and_links(state)
    _verify_summary_crc(source=source, state=state)


def _read_record(source: BinaryIO, file_size: int, crc_seed: int | None) -> _Record | None:
    offset = source.tell()
    header = source.read(9)
    if len(header) == 0:
        return None
    if len(header) != 9:
        raise _McapParseError(f"truncated MCAP record header at offset {offset}")
    opcode = header[0]
    (length,) = struct.unpack_from("<Q", header, 1)
    content_start = source.tell()
    if length > file_size - content_start:
        raise _McapParseError(
            f"truncated MCAP record content at offset {offset}: "
            f"declared={length}, available={max(0, file_size - content_start)}"
        )

    crc_after = zlib.crc32(header, crc_seed) & 0xFFFFFFFF if crc_seed is not None else None
    retained = bytearray()
    retain = length <= _MAX_RETAINED_RECORD_BYTES
    remaining = length
    while remaining:
        chunk = source.read(min(remaining, _STREAM_BUFFER_BYTES))
        if not chunk:
            raise _McapParseError(f"truncated MCAP record content at offset {offset}")
        if retain:
            retained.extend(chunk)
        if crc_after is not None:
            crc_after = zlib.crc32(chunk, crc_after) & 0xFFFFFFFF
        remaining -= len(chunk)
    return _Record(
        offset=offset,
        opcode=opcode,
        length=length,
        content=bytes(retained) if retain else None,
        crc_after=crc_after,
    )


class _Cursor:
    """Bounds-checked cursor for MCAP fixed and length-prefixed fields."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.offset = 0

    @property
    def remaining(self) -> int:
        return len(self.data) - self.offset

    def take(self, size: int, field_name: str) -> bytes:
        if size < 0 or self.offset + size > len(self.data):
            raise _McapParseError(f"truncated {field_name}")
        value = self.data[self.offset : self.offset + size]
        self.offset += size
        return value

    def u16(self, field_name: str) -> int:
        return int(struct.unpack("<H", self.take(2, field_name))[0])

    def u32(self, field_name: str) -> int:
        return int(struct.unpack("<I", self.take(4, field_name))[0])

    def u64(self, field_name: str) -> int:
        return int(struct.unpack("<Q", self.take(8, field_name))[0])

    def string(self, field_name: str) -> str:
        length = self.u32(f"{field_name} length")
        try:
            return self.take(length, field_name).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _McapParseError(f"{field_name} is not valid UTF-8") from exc

    def byte_array(self, field_name: str) -> bytes:
        length = self.u32(f"{field_name} length")
        return self.take(length, field_name)

    def string_map(self, field_name: str) -> None:
        length = self.u32(f"{field_name} byte length")
        end = self.offset + length
        if end > len(self.data):
            raise _McapParseError(f"truncated {field_name}")
        while self.offset < end:
            self.string(f"{field_name} key")
            self.string(f"{field_name} value")
        if self.offset != end:
            raise _McapParseError(f"malformed {field_name}")


def _require_end(cursor: _Cursor, field_name: str) -> None:
    if cursor.remaining:
        raise _McapParseError(f"unexpected trailing bytes in {field_name}")


def _parse_header(content: bytes) -> None:
    cursor = _Cursor(content)
    cursor.string("Header profile")
    cursor.string("Header library")
    _require_end(cursor, "Header")


def _parse_schema(content: bytes) -> int:
    cursor = _Cursor(content)
    schema_id = cursor.u16("Schema id")
    cursor.string("Schema name")
    cursor.string("Schema encoding")
    cursor.byte_array("Schema data")
    _require_end(cursor, "Schema")
    return schema_id


def _parse_channel(content: bytes) -> tuple[int, int]:
    cursor = _Cursor(content)
    channel_id = cursor.u16("Channel id")
    schema_id = cursor.u16("Channel schema_id")
    cursor.string("Channel topic")
    cursor.string("Channel message_encoding")
    cursor.string_map("Channel metadata")
    _require_end(cursor, "Channel")
    return channel_id, schema_id


def _parse_message_channel(content: bytes) -> int:
    cursor = _Cursor(content)
    channel_id = cursor.u16("Message channel_id")
    cursor.u32("Message sequence")
    cursor.u64("Message log_time")
    cursor.u64("Message publish_time")
    # The remaining bytes are the opaque serialized message payload.
    return channel_id


def _parse_chunk_header(content: bytes) -> tuple[str, int, int, bytes] | None:
    """Return compression, declared size, declared CRC, and raw records."""

    if len(content) > _MAX_RETAINED_RECORD_BYTES:
        return None
    cursor = _Cursor(content)
    cursor.u64("Chunk message_start_time")
    cursor.u64("Chunk message_end_time")
    uncompressed_size = cursor.u64("Chunk uncompressed_size")
    uncompressed_crc = cursor.u32("Chunk uncompressed_crc")
    compression = cursor.string("Chunk compression")
    records_length = cursor.u64("Chunk records length")
    records = cursor.take(records_length, "Chunk records")
    _require_end(cursor, "Chunk")
    if compression in {"", "none"} and len(records) != uncompressed_size:
        raise _McapParseError(
            "Chunk uncompressed_size does not match records length for an uncompressed chunk"
        )
    return compression, uncompressed_size, uncompressed_crc, records


def _scan_record(record: _Record, state: _ScanState, *, section: str) -> None:
    if record.opcode not in _KNOWN_OPCODES:
        state.error(f"unknown MCAP opcode 0x{record.opcode:02x}")
        return
    if record.content is None:
        if record.opcode == OP_CHUNK:
            state.warning("MCAP Chunk record exceeds bounded inspection budget")
            state.chunk_link_unknown = True
        elif record.opcode in {OP_SCHEMA, OP_CHANNEL, OP_MESSAGE}:
            state.warning(
                f"MCAP opcode 0x{record.opcode:02x} exceeds bounded inspection budget"
            )
        return
    try:
        if record.opcode == OP_SCHEMA:
            schema_id = _parse_schema(record.content)
            state.schema_ids.add(schema_id)
            state.schema_count += 1
        elif record.opcode == OP_CHANNEL:
            channel_id, schema_id = _parse_channel(record.content)
            state.channel_schema_ids[channel_id] = schema_id
            state.channel_count += 1
        elif record.opcode == OP_MESSAGE:
            state.message_channel_ids.add(_parse_message_channel(record.content))
            state.message_count += 1
        elif record.opcode == OP_CHUNK:
            state.chunk_count += 1
            _scan_chunk(record.content, state)
        elif record.opcode in {
            OP_MESSAGE_INDEX,
            OP_CHUNK_INDEX,
            OP_ATTACHMENT_INDEX,
            OP_METADATA_INDEX,
            OP_SUMMARY_OFFSET,
        }:
            state.index_count += 1
            if record.opcode == OP_SUMMARY_OFFSET:
                group_opcode, group_start, group_length = _parse_summary_offset(record.content)
                state.summary_offset_groups.append((group_opcode, group_start, group_length))
    except _McapParseError as exc:
        state.error(f"invalid MCAP {section} record opcode 0x{record.opcode:02x}: {exc}")


def _scan_chunk(content: bytes, state: _ScanState) -> None:
    parsed = _parse_chunk_header(content)
    if parsed is None:
        state.chunk_link_unknown = True
        state.chunk_compressed_unknown = True
        return
    compression, _uncompressed_size, uncompressed_crc, records = parsed
    if compression not in _SUPPORTED_CHUNK_COMPRESSIONS:
        state.unsupported(compression or "<unknown>")
        return
    if compression in {"", "none"}:
        if uncompressed_crc == 0:
            state.chunk_count_without_crc += 1
            state.warning("MCAP uncompressed Chunk uncompressed_crc is unavailable (zero)")
        else:
            actual = zlib.crc32(records) & 0xFFFFFFFF
            if actual != uncompressed_crc:
                state.chunk_crc_mismatch = True
                state.error(
                    "MCAP Chunk uncompressed_crc mismatch: "
                    f"declared=0x{uncompressed_crc:08x}, actual=0x{actual:08x}"
                )
            else:
                state.chunk_count_with_crc += 1
        _scan_inner_records(records, state)
    else:
        state.chunk_compressed_unknown = True
        state.chunk_link_unknown = True
        state.warning(
            f"MCAP {compression!r} Chunk links/uncompressed CRC remain unknown "
            "without decompression"
        )


def _scan_inner_records(buffer: bytes, state: _ScanState) -> None:
    offset = 0
    while offset < len(buffer):
        if offset + 9 > len(buffer):
            state.error("truncated MCAP record header inside Chunk")
            return
        opcode = buffer[offset]
        (length,) = struct.unpack_from("<Q", buffer, offset + 1)
        content_start = offset + 9
        content_end = content_start + length
        if content_end > len(buffer):
            state.error("truncated MCAP record content inside Chunk")
            return
        content = buffer[content_start:content_end]
        offset = content_end
        state.records_checked += 1
        if opcode not in {OP_SCHEMA, OP_CHANNEL, OP_MESSAGE}:
            state.error(f"MCAP opcode 0x{opcode:02x} is not allowed inside a Chunk")
            continue
        _scan_record(
            _Record(offset=0, opcode=opcode, length=length, content=content, crc_after=None),
            state,
            section="chunk",
        )


def _parse_summary_offset(content: bytes) -> tuple[int, int, int]:
    cursor = _Cursor(content)
    group_opcode = cursor.take(1, "Summary Offset group_opcode")[0]
    if group_opcode not in _SUMMARY_OPCODES:
        raise _McapParseError(
            f"Summary Offset group_opcode 0x{group_opcode:02x} is not a summary opcode"
        )
    group_start = cursor.u64("Summary Offset group_start")
    group_length = cursor.u64("Summary Offset group_length")
    _require_end(cursor, "Summary Offset")
    return group_opcode, group_start, group_length


def _validate_sections_and_links(state: _ScanState) -> None:
    if state.footer_offset is None or state.footer_record_end is None:
        return
    summary_offsets = state.summary_record_offsets
    offset_offsets = state.summary_offset_record_offsets
    if state.summary_start == 0:
        if summary_offsets:
            state.error("MCAP Footer declares no Summary section but summary records are present")
        state.summary_present = False
    else:
        state.summary_present = True
        if not summary_offsets or summary_offsets[0] != state.summary_start:
            state.error("MCAP Footer summary_start does not point to the first Summary record")
        if (
            state.data_end_record_end is not None
            and state.summary_start < state.data_end_record_end
        ):
            state.error("MCAP Footer summary_start is not after DataEnd")
        if state.summary_start >= state.footer_offset:
            state.error("MCAP Footer summary_start is outside the file")
    if state.summary_offset_start == 0:
        state.summary_offset_present = False
        if offset_offsets:
            state.error(
                "MCAP Footer declares no Summary Offset section but Summary Offset "
                "records are present"
            )
    else:
        state.summary_offset_present = True
        if not state.summary_present:
            state.error("MCAP Footer declares Summary Offset without a Summary section")
        if not offset_offsets or offset_offsets[0] != state.summary_offset_start:
            state.error(
                "MCAP Footer summary_offset_start does not point to the first Summary Offset"
            )
        if state.summary_offset_start >= state.footer_offset:
            state.error("MCAP Footer summary_offset_start is outside the file")

    _require_grouped_opcodes(state.summary_opcodes, state, "Summary")
    _require_grouped_opcodes(state.summary_offset_opcodes, state, "Summary Offset")
    _validate_summary_offset_groups(state)

    for channel_id, schema_id in state.channel_schema_ids.items():
        if schema_id != 0 and schema_id not in state.schema_ids:
            state.error(f"MCAP Channel {channel_id} references missing Schema {schema_id}")
    missing_channels = sorted(state.message_channel_ids - set(state.channel_schema_ids))
    for channel_id in missing_channels:
        state.error(f"MCAP Message references missing Channel {channel_id}")

    state.framing_valid = (
        state.leading_magic_valid
        and state.trailing_magic_valid
        and state.header_valid
        and state.data_end_valid
        and state.footer_valid
    )


def _require_grouped_opcodes(opcodes: list[int], state: _ScanState, section: str) -> None:
    if not opcodes:
        return
    seen: set[int] = set()
    previous = opcodes[0]
    seen.add(previous)
    for opcode in opcodes[1:]:
        if opcode != previous and opcode in seen:
            state.error(f"MCAP {section} records are not grouped by opcode")
        seen.add(opcode)
        previous = opcode


def _validate_summary_offset_groups(state: _ScanState) -> None:
    """Validate Summary Offset targets against the scanned Summary groups."""

    if state.footer_offset is None:
        return
    summary_end = state.summary_offset_start or state.footer_offset
    record_by_offset = {
        offset: (opcode, length) for offset, opcode, length in state.summary_record_meta
    }
    for group_opcode, group_start, group_length in state.summary_offset_groups:
        if group_start not in record_by_offset:
            state.error(
                f"MCAP Summary Offset group_start {group_start} is not a Summary record"
            )
            continue
        if group_start < state.summary_start or group_start + group_length > summary_end:
            state.error("MCAP Summary Offset group range is outside the Summary section")
            continue
        if group_length <= 0:
            state.error("MCAP Summary Offset group_length must be positive")
            continue
        cursor = group_start
        group_end = group_start + group_length
        grouped_records = 0
        while cursor < group_end:
            record = record_by_offset.get(cursor)
            if record is None:
                state.error("MCAP Summary Offset group range is not record-aligned")
                break
            opcode, length = record
            if opcode != group_opcode:
                state.error("MCAP Summary Offset group contains a different opcode")
                break
            cursor += 9 + length
            grouped_records += 1
        if cursor != group_end:
            state.error("MCAP Summary Offset group_length does not match records")
        elif grouped_records == 0:
            state.error("MCAP Summary Offset group is empty")


def _verify_summary_crc(*, source: BinaryIO, state: _ScanState) -> None:
    """Verify Footer.summary_crc over Summary + Summary Offset + Footer prefix."""

    if state.footer_offset is None or state.footer_record_end is None:
        return
    expected = state.summary_crc_expected
    if expected is None:
        return
    if expected == 0:
        state.warning("MCAP Footer summary_crc is unavailable (zero)")
        return
    start = state.summary_start if state.summary_start != 0 else state.footer_offset
    end = state.footer_offset + 9 + 16  # through summary_offset_start, before CRC field
    if start < 0 or end > state.file_size or start >= end:
        state.error("MCAP Footer summary CRC range is invalid")
        return
    try:
        source.seek(start)
        remaining = end - start
        actual = 0
        while remaining:
            chunk = source.read(min(remaining, _STREAM_BUFFER_BYTES))
            if not chunk:
                state.error("truncated MCAP summary CRC input")
                return
            actual = zlib.crc32(chunk, actual) & 0xFFFFFFFF
            remaining -= len(chunk)
    except (OSError, ValueError) as exc:
        state.error(f"MCAP summary CRC read failed: {type(exc).__name__}: {exc}")
        return
    state.summary_crc_actual = actual
    if actual != expected:
        state.error(
            "MCAP summary_crc mismatch: "
            f"declared=0x{expected:08x}, actual=0x{actual:08x}"
        )


def _status_from_error(
    *, blocked: bool, known: bool, not_applicable: bool = False
) -> McapFieldStatus:
    if blocked:
        return "blocked"
    if known:
        return "known"
    if not_applicable:
        return "not_applicable"
    return "unknown"


def _is_framing_error(error: str) -> bool:
    """Classify errors that invalidate the byte-level MCAP envelope."""

    token = error.lower()
    if "crc" in token or "references missing" in token:
        return False
    return any(
        marker in token
        for marker in (
            "magic",
            "truncated",
            "first mcap record",
            "header record",
            "footer record",
            "dataend record",
            "not allowed in the data section",
            "not allowed in the summary section",
            "not allowed inside a chunk",
            "unknown mcap opcode",
            "unexpected trailing bytes",
            "record content",
            "record header",
        )
    )


def _is_summary_structure_error(error: str) -> bool:
    """Classify summary pointer/group errors, excluding summary CRC mismatch."""

    token = error.lower()
    if "summary_crc" in token:
        return False
    return any(
        marker in token
        for marker in (
            "summary_start",
            "summary_offset_start",
            "summary section",
            "summary offset",
            "summary records are not grouped",
            "invalid mcap summary",
        )
    )


def _evidence_from_state(state: _ScanState, _path: Path) -> McapIntegrityEvidence:
    link_blocked = any("references missing" in error for error in state.errors)
    summary_structure_blocked = any(
        _is_summary_structure_error(error) for error in state.errors
    )
    framing_blocked = any(_is_framing_error(error) for error in state.errors)
    unsupported = bool(state.unsupported_compressions)
    data_crc_mismatch = any("data_section_crc mismatch" in error for error in state.errors)
    summary_crc_mismatch = any("summary_crc mismatch" in error for error in state.errors)
    crc_blocked = data_crc_mismatch or summary_crc_mismatch or state.chunk_crc_mismatch

    data_crc_known = state.data_crc_expected not in {None, 0} and not data_crc_mismatch
    summary_crc_known = state.summary_crc_expected not in {None, 0} and not summary_crc_mismatch
    chunk_crc_known = (
        state.chunk_count > 0
        and state.chunk_count_without_crc == 0
        and not state.chunk_compressed_unknown
        and not state.chunk_crc_mismatch
        and state.chunk_count_with_crc == state.chunk_count
    )
    warnings = list(state.warnings)
    if not state.summary_present:
        _bounded_append(warnings, "MCAP Summary section is absent (fast-write/no-summary file)")
    if not state.summary_offset_present:
        _bounded_append(warnings, "MCAP Summary Offset/index section is absent")
    if state.chunk_compressed_unknown:
        _bounded_append(warnings, "MCAP compressed Chunk integrity/link evidence is unknown")
    if state.chunk_link_unknown:
        _bounded_append(warnings, "MCAP Chunk link evidence is unknown")

    if state.errors or unsupported:
        status = "blocked"
    elif warnings or not state.framing_valid:
        status = "warn"
    else:
        status = "pass"
    if state.summary_present and state.summary_crc_expected == 0:
        _bounded_append(warnings, "MCAP Summary is present but Footer.summary_crc is unavailable")

    return McapIntegrityEvidence(
        status=status,  # type: ignore[arg-type]
        magic_status=_status_from_error(
            blocked=not state.leading_magic_valid or not state.trailing_magic_valid,
            known=state.leading_magic_valid and state.trailing_magic_valid,
        ),
        framing_status=_status_from_error(
            blocked=not state.framing_valid or framing_blocked,
            known=state.framing_valid and not framing_blocked,
        ),
        header_status=_status_from_error(blocked=not state.header_valid, known=state.header_valid),
        data_end_status=_status_from_error(
            blocked=not state.data_end_valid, known=state.data_end_valid
        ),
        footer_status=_status_from_error(blocked=not state.footer_valid, known=state.footer_valid),
        links_status=_status_from_error(
            blocked=link_blocked,
            known=bool(state.channel_schema_ids or state.message_channel_ids)
            and not link_blocked
            and not state.chunk_link_unknown,
        ),
        summary_status=_status_from_error(
            blocked=summary_structure_blocked,
            known=state.summary_present and not summary_structure_blocked,
        ),
        summary_offset_status=_status_from_error(
            blocked=summary_structure_blocked,
            known=state.summary_offset_present and not summary_structure_blocked,
        ),
        index_status=_status_from_error(
            blocked=summary_structure_blocked,
            known=state.index_count > 0 and not summary_structure_blocked,
        ),
        crc_status=_status_from_error(
            blocked=crc_blocked,
            known=data_crc_known or summary_crc_known or chunk_crc_known,
        ),
        data_crc_status=_status_from_error(
            blocked=data_crc_mismatch,
            known=data_crc_known,
        ),
        summary_crc_status=_status_from_error(
            blocked=summary_crc_mismatch,
            known=summary_crc_known,
        ),
        chunk_crc_status=_status_from_error(
            blocked=state.chunk_crc_mismatch,
            known=chunk_crc_known,
            not_applicable=state.chunk_count == 0,
        ),
        compression_status=_status_from_error(
            blocked=unsupported,
            known=state.chunk_count == 0
            or (not state.chunk_compressed_unknown and not unsupported),
        ),
        summary_present=state.summary_present,
        summary_offset_present=state.summary_offset_present,
        index_present=state.index_count > 0,
        data_crc_encoded=state.data_crc_expected not in {None, 0},
        summary_crc_encoded=state.summary_crc_expected not in {None, 0},
        file_size_bytes=state.file_size,
        records_checked=state.records_checked,
        data_records_checked=state.data_records_checked,
        summary_records_checked=state.summary_records_checked,
        schema_count=state.schema_count,
        channel_count=state.channel_count,
        message_count=state.message_count,
        chunk_count=state.chunk_count,
        index_count=state.index_count,
        unsupported_compressions=state.unsupported_compressions,
        errors=state.errors,
        warnings=warnings,
    )


__all__ = [
    "MCAP_MAGIC",
    "OP_ATTACHMENT",
    "OP_ATTACHMENT_INDEX",
    "OP_CHANNEL",
    "OP_CHUNK",
    "OP_CHUNK_INDEX",
    "OP_DATA_END",
    "OP_FOOTER",
    "OP_HEADER",
    "OP_MESSAGE",
    "OP_MESSAGE_INDEX",
    "OP_METADATA",
    "OP_METADATA_INDEX",
    "OP_SCHEMA",
    "OP_STATISTICS",
    "OP_SUMMARY_OFFSET",
    "inspect_mcap",
    "inspect_mcap_integrity",
]
