"""Pure-Python ROS 2 bag (rosbag2) reader.

The core package stays ROS-independent: this module parses rosbag2 container
formats directly with the standard library and (optionally) numpy. It never
imports ``rclpy``, ``rosbag2_py``, or other ROS packages.

Supported storage backends:

* **sqlite3** (``.db3``) — default through ROS 2 Humble; messages are streamed
  with ``ORDER BY timestamp`` so multi-gigabyte bags are never fully loaded.
* **mcap** (``.mcap``) — default from Iron; a minimal MCAP reader implements
  Header, Schema, Channel, Message, Chunk, and DataEnd records. Chunk
  compression ``none`` works out of the box; ``lz4`` and ``zstd`` require the
  optional ``calibrex[rosbag2-compression]`` extra.

When ``metadata.yaml`` declares ``compression_mode: message``, each stored
message payload is decompressed individually (``zstd`` or ``lz4``) before CDR
decode. ``compression_mode: file`` is rejected — decompress the bag first.
Bare storage files without ``metadata.yaml`` are read as stored.

CDR decoding supports ``sensor_msgs/msg/PointCloud2`` and ``nav_msgs/msg/Odometry``.
"""

from __future__ import annotations

import sqlite3
import struct
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from calibrex.core.exceptions import DatasetError
from calibrex.data.base import StreamSummary, TimestampedRecord
from calibrex.data.ros_cdr import decode_ros2_odometry, decode_ros2_pointcloud2
from calibrex.data.ros_messages import (
    OdometryMessage,
    PointCloud2Message,
    require_numpy,
)

MCAP_MAGIC = b"\x89MCAP0\r\n"

OP_HEADER = 0x01
OP_FOOTER = 0x02
OP_SCHEMA = 0x03
OP_CHANNEL = 0x04
OP_MESSAGE = 0x05
OP_CHUNK = 0x06
OP_DATA_END = 0x0F

POINTCLOUD2_TYPE = "sensor_msgs/msg/PointCloud2"
ODOMETRY_TYPE = "nav_msgs/msg/Odometry"
DECODED_MESSAGE_TYPES = frozenset({POINTCLOUD2_TYPE, ODOMETRY_TYPE})
_INSPECT_MESSAGE_TYPES = frozenset({POINTCLOUD2_TYPE, ODOMETRY_TYPE})


@dataclass(frozen=True)
class Rosbag2Connection:
    """A rosbag2 topic binding (sqlite topic row or MCAP channel)."""

    topic_id: int
    topic: str
    message_type: str
    serialization_format: str = "cdr"


Rosbag2DecodedMessage = PointCloud2Message | OdometryMessage


@dataclass(frozen=True)
class Rosbag2TopicStreamStats:
    """Per-topic statistics sampled from a rosbag2 bag."""

    topic: str
    message_type: str
    message_count: int
    sampled_message_count: int
    sampled_point_count: int | None = None
    has_intensity: bool | None = None
    bounds_min_m: tuple[float, float, float] | None = None
    bounds_max_m: tuple[float, float, float] | None = None
    first_timestamp_ns: int | None = None
    last_timestamp_ns: int | None = None
    sample_pose_position_m: tuple[float, float, float] | None = None
    sample_pose_orientation_xyzw: tuple[float, float, float, float] | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON/YAML-friendly representation."""

        return {
            "topic": self.topic,
            "message_type": self.message_type,
            "message_count": self.message_count,
            "sampled_message_count": self.sampled_message_count,
            "sampled_point_count": self.sampled_point_count,
            "has_intensity": self.has_intensity,
            "bounds_min_m": list(self.bounds_min_m) if self.bounds_min_m else None,
            "bounds_max_m": list(self.bounds_max_m) if self.bounds_max_m else None,
            "first_timestamp_ns": self.first_timestamp_ns,
            "last_timestamp_ns": self.last_timestamp_ns,
            "sample_pose_position_m": (
                list(self.sample_pose_position_m) if self.sample_pose_position_m else None
            ),
            "sample_pose_orientation_xyzw": (
                list(self.sample_pose_orientation_xyzw)
                if self.sample_pose_orientation_xyzw
                else None
            ),
        }


@dataclass(frozen=True)
class Rosbag2DatasetStats:
    """Bag-level statistics returned to ``calibrex inspect``."""

    status: str
    storage_identifier: str | None
    connection_count: int
    topic_count: int
    streams: tuple[Rosbag2TopicStreamStats, ...] = ()
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON/YAML-friendly representation."""

        return {
            "status": self.status,
            "storage_identifier": self.storage_identifier,
            "connection_count": self.connection_count,
            "topic_count": self.topic_count,
            "streams": [stream.as_dict() for stream in self.streams],
            "reason": self.reason,
        }


class Rosbag2Reader:
    """Reader for rosbag2 bags exposing the dataset adapter surface."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def streams(self) -> list[StreamSummary]:
        """Return one stream per decoded topic in the bag."""

        counts, connections = _topic_counts(self.path)
        summaries: list[StreamSummary] = []
        for topic in sorted(counts):
            connection = connections[topic]
            kind = "pointcloud" if connection.message_type == POINTCLOUD2_TYPE else "odometry"
            summaries.append(
                StreamSummary(
                    name=topic,
                    kind=kind,
                    message_count=counts[topic],
                    topic=topic,
                    sensor=connection.message_type,
                )
            )
        return summaries

    def records(self, stream: str) -> Iterable[TimestampedRecord]:
        """Yield one normalized record per message on ``stream`` (a topic)."""

        for connection, timestamp_ns, _data in iter_messages(self.path, topics={stream}):
            yield TimestampedRecord(
                stream=stream,
                timestamp_ns=timestamp_ns,
                payload_path=None,
                metadata={
                    "format": "rosbag2",
                    "message_type": connection.message_type,
                    "topic": connection.topic,
                },
            )

    def read_pointclouds(self, topic: str) -> Iterator[PointCloud2Message]:
        """Yield decoded ``PointCloud2`` messages for ``topic``."""

        yield from read_pointcloud2_messages(self.path, topic=topic)


def iter_messages(
    path: str | Path,
    *,
    topics: set[str] | None = None,
) -> Iterator[tuple[Rosbag2Connection, int, bytes]]:
    """Yield ``(connection, timestamp_ns, cdr_payload)`` in timestamp order."""

    storage_path, storage_id, decompress_message = _resolve_storage(path)
    if storage_id == "sqlite3":
        yield from _iter_sqlite_messages(
            storage_path,
            topics=topics,
            decompress_message=decompress_message,
        )
        return
    if storage_id == "mcap":
        yield from _iter_mcap_messages(
            storage_path,
            topics=topics,
            decompress_message=decompress_message,
        )
        return
    msg = f"unsupported rosbag2 storage identifier: {storage_id!r}"
    raise DatasetError(msg)


def _resolve_storage(
    path: str | Path,
) -> tuple[Path, str, Callable[[bytes], bytes] | None]:
    """Resolve storage and optional per-message decompression."""

    storage_path, storage_id = resolve_storage(path)
    decompress_message = _message_decompressor_from_path(path)
    return storage_path, storage_id, decompress_message


def resolve_storage(path: str | Path) -> tuple[Path, str]:
    """Resolve a bag directory (with optional metadata.yaml) or bare storage file."""

    bag_path = Path(path)
    if not bag_path.exists():
        msg = f"rosbag2 path does not exist: {bag_path}"
        raise DatasetError(msg)

    if bag_path.is_file():
        suffix = bag_path.suffix.lower()
        if suffix == ".db3":
            return bag_path, "sqlite3"
        if suffix == ".mcap":
            return bag_path, "mcap"
        msg = f"unsupported rosbag2 storage file extension: {suffix!r}"
        raise DatasetError(msg)

    metadata_path = bag_path / "metadata.yaml"
    storage_id: str | None = None
    relative_paths: list[str] = []
    if metadata_path.is_file():
        metadata = _load_metadata(metadata_path)
        storage_id = metadata.get("storage_identifier")
        relative_paths = metadata.get("relative_file_paths", [])

    if storage_id == "mcap" or (storage_id is None and list(bag_path.glob("*.mcap"))):
        storage_file = _pick_storage_file(bag_path, relative_paths, ".mcap")
        return storage_file, "mcap"

    storage_file = _pick_storage_file(bag_path, relative_paths, ".db3")
    return storage_file, storage_id or "sqlite3"


def _pick_storage_file(
    bag_dir: Path,
    relative_paths: list[str],
    suffix: str,
) -> Path:
    if relative_paths:
        candidate = bag_dir / relative_paths[0]
        if candidate.is_file():
            return candidate
    matches = sorted(bag_dir.glob(f"*{suffix}"))
    if not matches:
        msg = f"no {suffix} storage file found under rosbag2 directory: {bag_dir}"
        raise DatasetError(msg)
    return matches[0]


def _load_metadata(metadata_path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = f"malformed rosbag2 metadata.yaml: {metadata_path}"
        raise DatasetError(msg)
    info = raw.get("rosbag2_bagfile_information")
    if not isinstance(info, dict):
        msg = f"rosbag2 metadata.yaml missing rosbag2_bagfile_information: {metadata_path}"
        raise DatasetError(msg)
    storage_id = info.get("storage_identifier")
    relative_paths = info.get("relative_file_paths", [])
    if relative_paths is not None and not isinstance(relative_paths, list):
        msg = "rosbag2 metadata relative_file_paths must be a list"
        raise DatasetError(msg)
    return {
        "storage_identifier": str(storage_id) if storage_id is not None else None,
        "relative_file_paths": [str(item) for item in relative_paths or []],
        "topics_with_message_count": info.get("topics_with_message_count", []),
        "compression_format": _normalize_metadata_token(info.get("compression_format")),
        "compression_mode": _normalize_metadata_token(info.get("compression_mode")),
    }


def _normalize_metadata_token(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text.lower()


def _message_decompressor_from_path(path: str | Path) -> Callable[[bytes], bytes] | None:
    bag_path = Path(path)
    if not bag_path.is_dir():
        return None
    metadata_path = bag_path / "metadata.yaml"
    if not metadata_path.is_file():
        return None
    metadata = _load_metadata(metadata_path)
    return _message_decompressor(
        compression_format=metadata["compression_format"],
        compression_mode=metadata["compression_mode"],
    )


def _message_decompressor(
    *,
    compression_format: str,
    compression_mode: str,
) -> Callable[[bytes], bytes] | None:
    if not compression_format or not compression_mode:
        return None
    if compression_mode == "file":
        msg = (
            "rosbag2 file-mode compression is not supported; "
            "decompress the bag before reading"
        )
        raise DatasetError(msg)
    if compression_mode != "message":
        msg = f"unsupported rosbag2 compression_mode: {compression_mode!r}"
        raise DatasetError(msg)
    if compression_format == "zstd":
        return _zstd_decompress_message
    if compression_format == "lz4":
        return _lz4_decompress_message
    msg = f"unsupported rosbag2 message compression format: {compression_format!r}"
    raise DatasetError(msg)


def _maybe_decompress_message(
    data: bytes,
    decompress_message: Callable[[bytes], bytes] | None,
) -> bytes:
    if decompress_message is None:
        return data
    return decompress_message(data)


def _iter_sqlite_messages(
    db_path: Path,
    *,
    topics: set[str] | None,
    decompress_message: Callable[[bytes], bytes] | None = None,
) -> Iterator[tuple[Rosbag2Connection, int, bytes]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        topic_rows = conn.execute(
            "SELECT id, name, type, serialization_format FROM topics"
        ).fetchall()
        connections = {
            int(topic_id): Rosbag2Connection(
                topic_id=int(topic_id),
                topic=str(name),
                message_type=str(message_type),
                serialization_format=str(serialization_format),
            )
            for topic_id, name, message_type, serialization_format in topic_rows
        }
        allowed_topic_ids: set[int] | None = None
        if topics is not None:
            allowed_topic_ids = {
                connection.topic_id
                for connection in connections.values()
                if connection.topic in topics
            }
        query = (
            "SELECT topic_id, timestamp, data FROM messages "
            "ORDER BY timestamp ASC, id ASC"
        )
        for topic_id, timestamp, data in conn.execute(query):
            resolved = connections.get(int(topic_id))
            if resolved is None:
                continue
            if allowed_topic_ids is not None and int(topic_id) not in allowed_topic_ids:
                continue
            payload = _maybe_decompress_message(bytes(data), decompress_message)
            yield resolved, int(timestamp), payload
    finally:
        conn.close()


def _iter_mcap_messages(
    mcap_path: Path,
    *,
    topics: set[str] | None,
    decompress_message: Callable[[bytes], bytes] | None = None,
) -> Iterator[tuple[Rosbag2Connection, int, bytes]]:
    channels: dict[int, Rosbag2Connection] = {}
    schemas: dict[int, str] = {}
    with mcap_path.open("rb") as source:
        magic = source.read(len(MCAP_MAGIC))
        if magic != MCAP_MAGIC:
            msg = "not an MCAP file (bad magic header)"
            raise DatasetError(msg)
        for opcode, content in _iter_mcap_records(source):
            if opcode == OP_SCHEMA:
                schema_id, schema_name = _parse_mcap_schema(content)
                schemas[schema_id] = schema_name
            elif opcode == OP_CHANNEL:
                channel = _parse_mcap_channel(content, schemas)
                channels[channel.topic_id] = channel
            elif opcode == OP_CHUNK:
                _compression, records = _parse_mcap_chunk(content)
                for inner_opcode, inner_content in _iter_mcap_record_bytes(records):
                    if inner_opcode == OP_SCHEMA:
                        schema_id, schema_name = _parse_mcap_schema(inner_content)
                        schemas[schema_id] = schema_name
                    elif inner_opcode == OP_CHANNEL:
                        channel = _parse_mcap_channel(inner_content, schemas)
                        channels[channel.topic_id] = channel
                    elif inner_opcode == OP_MESSAGE:
                        yield from _yield_mcap_message(
                            inner_content,
                            channels=channels,
                            topics=topics,
                            decompress_message=decompress_message,
                        )
            elif opcode == OP_MESSAGE:
                yield from _yield_mcap_message(
                    content,
                    channels=channels,
                    topics=topics,
                    decompress_message=decompress_message,
                )
            elif opcode in {OP_FOOTER, OP_DATA_END}:
                break


def _yield_mcap_message(
    content: bytes,
    *,
    channels: dict[int, Rosbag2Connection],
    topics: set[str] | None,
    decompress_message: Callable[[bytes], bytes] | None = None,
) -> Iterator[tuple[Rosbag2Connection, int, bytes]]:
    channel_id, log_time, payload = _parse_mcap_message(content)
    connection = channels.get(channel_id)
    if connection is None:
        return
    if topics is not None and connection.topic not in topics:
        return
    payload = _maybe_decompress_message(payload, decompress_message)
    yield connection, log_time, payload


def _iter_mcap_records(source: Any) -> Iterator[tuple[int, bytes]]:
    while True:
        header = source.read(9)
        if len(header) == 0:
            break
        if len(header) != 9:
            msg = "truncated MCAP record header"
            raise DatasetError(msg)
        opcode = header[0]
        (length,) = struct.unpack("<Q", header[1:9])
        content = source.read(length)
        if len(content) != length:
            msg = "truncated MCAP record content"
            raise DatasetError(msg)
        yield opcode, content
        if opcode == OP_FOOTER:
            break


def _iter_mcap_record_bytes(buffer: bytes) -> Iterator[tuple[int, bytes]]:
    offset = 0
    total = len(buffer)
    while offset < total:
        if offset + 9 > total:
            msg = "truncated MCAP chunk records"
            raise DatasetError(msg)
        opcode = buffer[offset]
        (length,) = struct.unpack_from("<Q", buffer, offset + 1)
        offset += 9
        if offset + length > total:
            msg = "truncated MCAP chunk record content"
            raise DatasetError(msg)
        content = buffer[offset : offset + length]
        offset += length
        yield opcode, content


def _parse_mcap_schema(content: bytes) -> tuple[int, str]:
    offset = 0
    schema_id, offset = _read_mcap_uint16(content, offset)
    name, _offset = _read_mcap_string(content, offset)
    return schema_id, name


def _parse_mcap_channel(
    content: bytes,
    schemas: dict[int, str],
) -> Rosbag2Connection:
    offset = 0
    channel_id, offset = _read_mcap_uint16(content, offset)
    schema_id, offset = _read_mcap_uint16(content, offset)
    topic, offset = _read_mcap_string(content, offset)
    _encoding, offset = _read_mcap_string(content, offset)
    offset = _skip_mcap_string_map(content, offset)
    return Rosbag2Connection(
        topic_id=channel_id,
        topic=topic,
        message_type=schemas.get(schema_id, ""),
        serialization_format="cdr",
    )


def _parse_mcap_message(content: bytes) -> tuple[int, int, bytes]:
    offset = 0
    channel_id, offset = _read_mcap_uint16(content, offset)
    _sequence, offset = _read_mcap_uint32(content, offset)
    log_time, offset = _read_mcap_uint64(content, offset)
    _publish_time, offset = _read_mcap_uint64(content, offset)
    payload = content[offset:]
    return channel_id, log_time, payload


def _parse_mcap_chunk(content: bytes) -> tuple[str, bytes]:
    offset = 0
    _start, offset = _read_mcap_uint64(content, offset)
    _end, offset = _read_mcap_uint64(content, offset)
    uncompressed_size, offset = _read_mcap_uint64(content, offset)
    _crc, offset = _read_mcap_uint32(content, offset)
    compression, offset = _read_mcap_string(content, offset)
    records, _offset = _read_mcap_length_prefixed_bytes(content, offset)
    if compression:
        records = _decompress_mcap_chunk(compression, records, int(uncompressed_size))
    return compression, records


def _decompress_mcap_chunk(compression: str, data: bytes, uncompressed_size: int) -> bytes:
    if compression == "" or compression == "none":
        return data
    if compression == "lz4":
        return _lz4_decompress(data, uncompressed_size)
    if compression == "zstd":
        return _zstd_decompress(data, uncompressed_size)
    msg = f"unsupported MCAP chunk compression: {compression!r}"
    raise DatasetError(msg)


def _lz4_decompress(data: bytes, size: int) -> bytes:
    try:
        import lz4.block
    except ImportError as exc:  # pragma: no cover - exercised via error path test
        msg = (
            "lz4-compressed MCAP chunks require the optional dependency "
            "calibrex[rosbag2-compression]"
        )
        raise DatasetError(msg) from exc
    decompressed: bytes = lz4.block.decompress(data, uncompressed_size=size)
    if size and len(decompressed) != size:
        msg = "lz4 chunk decompressed to an unexpected size"
        raise DatasetError(msg)
    return decompressed


def _zstd_decompress(data: bytes, size: int) -> bytes:
    try:
        import zstandard
    except ImportError as exc:  # pragma: no cover - exercised via error path test
        msg = (
            "zstd-compressed MCAP chunks require the optional dependency "
            "calibrex[rosbag2-compression]"
        )
        raise DatasetError(msg) from exc
    decompressed: bytes = zstandard.ZstdDecompressor().decompress(data)
    if size and len(decompressed) != size:
        msg = "zstd chunk decompressed to an unexpected size"
        raise DatasetError(msg)
    return decompressed


def _zstd_decompress_message(data: bytes) -> bytes:
    try:
        import zstandard
    except ImportError as exc:  # pragma: no cover - exercised via error path test
        msg = (
            "zstd-compressed rosbag2 messages require the optional dependency "
            "calibrex[rosbag2-compression]"
        )
        raise DatasetError(msg) from exc
    decompressor = zstandard.ZstdDecompressor()
    try:
        decompressed: bytes = decompressor.decompress(data)
        return decompressed
    except zstandard.ZstdError:
        streamed: bytes = decompressor.decompressobj().decompress(data)
        return streamed


def _lz4_decompress_message(data: bytes) -> bytes:
    try:
        import lz4.frame
    except ImportError as exc:  # pragma: no cover - exercised via error path test
        msg = (
            "lz4-compressed rosbag2 messages require the optional dependency "
            "calibrex[rosbag2-compression]"
        )
        raise DatasetError(msg) from exc
    decompressed: bytes = lz4.frame.decompress(data)
    return decompressed


def _read_mcap_uint16(buffer: bytes, offset: int) -> tuple[int, int]:
    (value,) = struct.unpack_from("<H", buffer, offset)
    return int(value), offset + 2


def _read_mcap_uint32(buffer: bytes, offset: int) -> tuple[int, int]:
    (value,) = struct.unpack_from("<I", buffer, offset)
    return int(value), offset + 4


def _read_mcap_uint64(buffer: bytes, offset: int) -> tuple[int, int]:
    (value,) = struct.unpack_from("<Q", buffer, offset)
    return int(value), offset + 8


def _read_mcap_string(buffer: bytes, offset: int) -> tuple[str, int]:
    (length,) = struct.unpack_from("<I", buffer, offset)
    offset += 4
    text = buffer[offset : offset + length].decode("utf-8", errors="replace")
    return text, offset + length


def _read_mcap_length_prefixed_bytes(buffer: bytes, offset: int) -> tuple[bytes, int]:
    (length,) = struct.unpack_from("<Q", buffer, offset)
    offset += 8
    payload = buffer[offset : offset + length]
    return payload, offset + length


def _skip_mcap_string_map(buffer: bytes, offset: int) -> int:
    (length,) = struct.unpack_from("<I", buffer, offset)
    return offset + 4 + int(length)


def read_pointcloud2_messages(
    path: str | Path,
    *,
    topic: str | None = None,
) -> Iterator[PointCloud2Message]:
    """Yield decoded ``PointCloud2`` messages, optionally filtered by ``topic``."""

    topics = {topic} if topic is not None else None
    for connection, timestamp_ns, data in iter_messages(path, topics=topics):
        if connection.message_type != POINTCLOUD2_TYPE:
            continue
        yield decode_pointcloud2(connection.topic, timestamp_ns, data)


def decode_rosbag2_message(
    topic: str,
    message_type: str,
    timestamp_ns: int,
    data: bytes,
) -> Rosbag2DecodedMessage:
    """Decode a supported rosbag2 CDR payload."""

    if message_type == POINTCLOUD2_TYPE:
        return decode_pointcloud2(topic, timestamp_ns, data)
    if message_type == ODOMETRY_TYPE:
        return decode_odometry(topic, timestamp_ns, data)
    msg = f"unsupported rosbag2 message type: {message_type!r}"
    raise DatasetError(msg)


def decode_pointcloud2(
    topic: str,
    timestamp_ns: int,
    data: bytes,
    *,
    point_time_field: str | None = None,
) -> PointCloud2Message:
    """Decode a CDR ``sensor_msgs/msg/PointCloud2`` payload."""

    return decode_ros2_pointcloud2(
        topic, timestamp_ns, data, point_time_field=point_time_field
    )


def decode_odometry(topic: str, timestamp_ns: int, data: bytes) -> OdometryMessage:
    """Decode a CDR ``nav_msgs/msg/Odometry`` payload."""

    return decode_ros2_odometry(topic, timestamp_ns, data)


def summarize_rosbag2(
    path: str | Path,
    *,
    sample_limit: int = 4,
) -> Rosbag2DatasetStats:
    """Summarize decoded topics in a rosbag2 bag for ``calibrex inspect``."""

    bag_path = Path(path)
    if not bag_path.exists():
        return Rosbag2DatasetStats(
            status="missing",
            storage_identifier=None,
            connection_count=0,
            topic_count=0,
            reason="rosbag2 path does not exist",
        )

    storage_id: str | None = None
    try:
        _storage_path, storage_id = resolve_storage(bag_path)
    except DatasetError:
        storage_id = None

    counts: dict[str, int] = {}
    connections: dict[str, Rosbag2Connection] = {}
    sampled_counts: dict[str, int] = {}
    sampled_points: dict[str, int] = {}
    has_intensity: dict[str, bool] = {}
    bounds_min: dict[str, tuple[float, float, float]] = {}
    bounds_max: dict[str, tuple[float, float, float]] = {}
    first_ts: dict[str, int] = {}
    last_ts: dict[str, int] = {}
    sample_pose_position: dict[str, tuple[float, float, float]] = {}
    sample_pose_orientation: dict[str, tuple[float, float, float, float]] = {}

    try:
        for connection, timestamp_ns, data in iter_messages(bag_path):
            if connection.message_type not in _INSPECT_MESSAGE_TYPES:
                continue
            topic = connection.topic
            counts[topic] = counts.get(topic, 0) + 1
            connections.setdefault(topic, connection)
            first_ts.setdefault(topic, timestamp_ns)
            last_ts[topic] = timestamp_ns
            if sampled_counts.get(topic, 0) >= sample_limit:
                continue
            message = decode_rosbag2_message(
                topic,
                connection.message_type,
                timestamp_ns,
                data,
            )
            sampled_counts[topic] = sampled_counts.get(topic, 0) + 1
            if isinstance(message, PointCloud2Message):
                sampled_points[topic] = sampled_points.get(topic, 0) + message.point_count
                has_intensity[topic] = has_intensity.get(topic, False) or (
                    message.intensity is not None
                )
                _accumulate_pointcloud_bounds(message, topic, bounds_min, bounds_max)
            elif isinstance(message, OdometryMessage):
                sample_pose_position.setdefault(topic, message.position)
                sample_pose_orientation.setdefault(topic, message.orientation_xyzw)
    except DatasetError as exc:
        return Rosbag2DatasetStats(
            status="malformed",
            storage_identifier=storage_id,
            connection_count=len(connections),
            topic_count=len(counts),
            reason=str(exc),
        )

    streams = tuple(
        Rosbag2TopicStreamStats(
            topic=topic,
            message_type=connections[topic].message_type,
            message_count=counts[topic],
            sampled_message_count=sampled_counts.get(topic, 0),
            sampled_point_count=sampled_points.get(topic),
            has_intensity=has_intensity.get(topic),
            bounds_min_m=bounds_min.get(topic),
            bounds_max_m=bounds_max.get(topic),
            first_timestamp_ns=first_ts.get(topic),
            last_timestamp_ns=last_ts.get(topic),
            sample_pose_position_m=sample_pose_position.get(topic),
            sample_pose_orientation_xyzw=sample_pose_orientation.get(topic),
        )
        for topic in sorted(counts)
    )
    status = "scored" if streams else "empty"
    return Rosbag2DatasetStats(
        status=status,
        storage_identifier=storage_id,
        connection_count=len(connections),
        topic_count=len(counts),
        streams=streams,
        reason=None if streams else "no supported decoded topics found",
    )


def _topic_counts(
    path: str | Path,
) -> tuple[dict[str, int], dict[str, Rosbag2Connection]]:
    counts: dict[str, int] = {}
    connections: dict[str, Rosbag2Connection] = {}
    for connection, _timestamp_ns, _data in iter_messages(path):
        if connection.message_type not in _INSPECT_MESSAGE_TYPES:
            continue
        counts[connection.topic] = counts.get(connection.topic, 0) + 1
        connections.setdefault(connection.topic, connection)
    return counts, connections


def _accumulate_pointcloud_bounds(
    message: PointCloud2Message,
    topic: str,
    bounds_min: dict[str, tuple[float, float, float]],
    bounds_max: dict[str, tuple[float, float, float]],
) -> None:
    if message.point_count == 0:
        return
    np_mod = require_numpy(extra_name="rosbag2")
    finite = np_mod.isfinite(message.xyz).all(axis=1)
    finite_xyz = message.xyz[finite]
    if finite_xyz.shape[0] == 0:
        return
    mins = finite_xyz.min(axis=0)
    maxs = finite_xyz.max(axis=0)
    frame_min = (float(mins[0]), float(mins[1]), float(mins[2]))
    frame_max = (float(maxs[0]), float(maxs[1]), float(maxs[2]))
    existing_min = bounds_min.get(topic)
    existing_max = bounds_max.get(topic)
    bounds_min[topic] = frame_min if existing_min is None else _elementwise_min(
        existing_min, frame_min
    )
    bounds_max[topic] = frame_max if existing_max is None else _elementwise_max(
        existing_max, frame_max
    )


def _elementwise_min(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (min(left[0], right[0]), min(left[1], right[1]), min(left[2], right[2]))


def _elementwise_max(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (max(left[0], right[0]), max(left[1], right[1]), max(left[2], right[2]))
