"""Pure-Python ROS 1 bag (v2.0) reader.

The core package stays ROS-independent: this module parses the ROS bag *file
format* directly with the standard library and (optionally) numpy. It never
imports ``rospy``, ``rosbag``, or any other ROS / GPL package.

Scope
-----
* Parse the bag magic, the bag-header record, connection records, and chunk
  records. ``none`` and ``bz2`` chunk compression are supported out of the box
  (``bz2`` via the standard library). ``lz4`` requires the optional
  ``calibrex[rosbag1-lz4]`` extra and raises a clear error when missing.
* Deserialize ``sensor_msgs/PointCloud2`` and ``livox_ros_driver/CustomMsg``
  payloads into numpy arrays (``x``, ``y``, ``z`` and intensity or reflectivity
  when present) with nanosecond timestamps.
* Deserialize standard ``sensor_msgs/Image`` and ``sensor_msgs/CameraInfo``
  payloads into ROS-independent typed records; image bytes are optional.
* Deserialize the pinned standard ``radar_msgs/RadarScan`` payload into a
  bounded ROS-independent typed record.

The reader follows the same ``streams()`` / ``records()`` surface used by the
other dataset adapters in :mod:`calibrex.data`.
"""

from __future__ import annotations

import bz2
import math
import struct
from collections.abc import Iterable, Iterator
from contextlib import suppress
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import BinaryIO

from calibrex.core.exceptions import DatasetError
from calibrex.data.base import StreamSummary, TimestampedRecord
from calibrex.data.ros_messages import (
    MAX_CAMERA_INFO_DISTORTION_COEFFICIENTS,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_ENCODING_LENGTH,
    MAX_IMAGE_HEIGHT,
    MAX_IMAGE_WIDTH,
    MAX_RADAR_PAYLOAD_BYTES,
    MAX_RADAR_RETURNS,
    MAX_ROS_STRING_BYTES,
    CameraInfoMessage,
    ImageMessage,
    LivoxCustomMessage,
    PointCloud2Message,
    PointField,
    RadarReturn,
    RadarScanMessage,
    RegionOfInterest,
    decode_point_time_offsets,
    decode_pointcloud_payload,
    filter_nonfinite_pointcloud_rows,
    require_numpy,
    validate_camera_info_metadata,
    validate_image_metadata,
)

BAG_MAGIC = b"#ROSBAG V2.0\n"

# Record op codes (rosbag v2.0 format).
OP_MSG_DATA = 0x02
OP_BAG_HEADER = 0x03
OP_INDEX_DATA = 0x04
OP_CHUNK = 0x05
OP_CHUNK_INFO = 0x06
OP_CONNECTION = 0x07

POINTCLOUD2_TYPE = "sensor_msgs/PointCloud2"
IMAGE_TYPE = "sensor_msgs/Image"
CAMERA_INFO_TYPE = "sensor_msgs/CameraInfo"
RADAR_SCAN_TYPE = "radar_msgs/RadarScan"
LIVOX_CUSTOMMSG_TYPE = "livox_ros_driver/CustomMsg"
POSE_STAMPED_TYPE = "geometry_msgs/PoseStamped"
LIDAR_MESSAGE_TYPES = frozenset({POINTCLOUD2_TYPE, LIVOX_CUSTOMMSG_TYPE})
RADAR_MESSAGE_TYPES = frozenset({RADAR_SCAN_TYPE})
_LIDAR_MESSAGE_TYPES = LIDAR_MESSAGE_TYPES
DECODED_MESSAGE_TYPES = frozenset(
    {
        POINTCLOUD2_TYPE,
        IMAGE_TYPE,
        CAMERA_INFO_TYPE,
        LIVOX_CUSTOMMSG_TYPE,
        RADAR_SCAN_TYPE,
    }
)
_LIVOX_CUSTOM_POINT_STEP = 19
_DEFAULT_DISTANCE_BIN_EDGES_M = (0.0, 10.0, 20.0, 40.0, 80.0)
_BAG_HEADER_PADDING_SCAN_LIMIT = 16 * 1024 * 1024
_RECORD_OPS = frozenset(
    {
        OP_MSG_DATA,
        OP_BAG_HEADER,
        OP_INDEX_DATA,
        OP_CHUNK,
        OP_CHUNK_INFO,
        OP_CONNECTION,
    }
)

@dataclass(frozen=True)
class Rosbag1Connection:
    """A ROS bag connection (a topic bound to a message type)."""

    conn_id: int
    topic: str
    message_type: str
    md5sum: str | None = None


@dataclass(frozen=True)
class Ros1PoseStampedMessage:
    """A decoded ROS 1 ``geometry_msgs/PoseStamped`` payload."""

    topic: str
    timestamp_ns: int
    frame_id: str
    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]


BagLidarMessage = PointCloud2Message | LivoxCustomMessage


@dataclass(frozen=True)
class Rosbag1PointCloudStreamStats:
    """Per-topic point-cloud statistics sampled from a bag."""

    topic: str
    message_type: str
    message_count: int
    sampled_message_count: int
    sampled_point_count: int
    has_intensity: bool
    bounds_min_m: tuple[float, float, float] | None = None
    bounds_max_m: tuple[float, float, float] | None = None
    first_timestamp_ns: int | None = None
    last_timestamp_ns: int | None = None
    range_min_m: float | None = None
    range_max_m: float | None = None
    fov_azimuth_deg: float | None = None
    fov_elevation_deg: float | None = None
    point_time_available: bool = False
    point_time_field: str | None = None
    point_time_min_s: float | None = None
    point_time_max_s: float | None = None
    distance_bin_edges_m: tuple[float, ...] = _DEFAULT_DISTANCE_BIN_EDGES_M
    distance_bin_counts: tuple[int, ...] | None = None

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
            "range_min_m": self.range_min_m,
            "range_max_m": self.range_max_m,
            "fov_azimuth_deg": self.fov_azimuth_deg,
            "fov_elevation_deg": self.fov_elevation_deg,
            "point_time_available": self.point_time_available,
            "point_time_field": self.point_time_field,
            "point_time_min_s": self.point_time_min_s,
            "point_time_max_s": self.point_time_max_s,
            "distance_bin_edges_m": list(self.distance_bin_edges_m),
            "distance_bin_counts": (
                list(self.distance_bin_counts)
                if self.distance_bin_counts is not None
                else None
            ),
        }


@dataclass(frozen=True)
class Rosbag1RadarStreamStats:
    """Bounded RadarScan evidence sampled from a ROS 1 bag topic."""

    topic: str
    message_type: str
    message_count: int
    sampled_message_count: int
    sampled_return_count: int
    sample_frame_id: str | None = None
    range_min_m: float | None = None
    range_max_m: float | None = None
    azimuth_span_rad: float | None = None
    elevation_span_rad: float | None = None
    doppler_min_mps: float | None = None
    doppler_max_mps: float | None = None
    doppler_span_mps: float | None = None
    duplicate_return_count: int = 0
    diversity_status: str = "unknown"

    def as_dict(self) -> dict[str, object]:
        """Return JSON/YAML-friendly RadarScan evidence."""

        return {
            "topic": self.topic,
            "message_type": self.message_type,
            "message_count": self.message_count,
            "sampled_message_count": self.sampled_message_count,
            "sampled_return_count": self.sampled_return_count,
            "sample_frame_id": self.sample_frame_id,
            "range_min_m": self.range_min_m,
            "range_max_m": self.range_max_m,
            "azimuth_span_rad": self.azimuth_span_rad,
            "elevation_span_rad": self.elevation_span_rad,
            "doppler_min_mps": self.doppler_min_mps,
            "doppler_max_mps": self.doppler_max_mps,
            "doppler_span_mps": self.doppler_span_mps,
            "duplicate_return_count": self.duplicate_return_count,
            "diversity_status": self.diversity_status,
        }


@dataclass(frozen=True)
class Rosbag1DatasetStats:
    """Bag-level statistics returned to ``calibrex inspect``."""

    status: str
    connection_count: int
    pointcloud_topic_count: int
    streams: tuple[Rosbag1PointCloudStreamStats, ...] = ()
    radar_streams: tuple[Rosbag1RadarStreamStats, ...] = ()
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON/YAML-friendly representation."""

        return {
            "status": self.status,
            "connection_count": self.connection_count,
            "pointcloud_topic_count": self.pointcloud_topic_count,
            "streams": [stream.as_dict() for stream in self.streams],
            "radar_streams": [stream.as_dict() for stream in self.radar_streams],
            "reason": self.reason,
        }


class Rosbag1Reader:
    """Reader for ROS 1 bag (v2.0) files exposing the dataset adapter surface."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def streams(self) -> list[StreamSummary]:
        """Return one stream per supported LiDAR or Radar topic in the bag."""

        counts, connections = _supported_sensor_topic_counts(self.path)
        summaries: list[StreamSummary] = []
        for topic in sorted(counts):
            connection = connections[topic]
            kind = "radar" if connection.message_type in RADAR_MESSAGE_TYPES else "pointcloud"
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
                    "format": "rosbag1",
                    "message_type": connection.message_type,
                    "topic": connection.topic,
                },
            )

    def read_pointclouds(self, topic: str) -> Iterator[PointCloud2Message]:
        """Yield decoded ``PointCloud2`` messages for ``topic``."""

        yield from read_pointcloud2_messages(self.path, topic=topic)

    def read_images(self, topic: str | None = None) -> Iterator[ImageMessage]:
        """Yield validated ``sensor_msgs/Image`` messages without pixel copies."""

        yield from read_image_messages(self.path, topic=topic)

    def read_camera_info(self, topic: str | None = None) -> Iterator[CameraInfoMessage]:
        """Yield validated ``sensor_msgs/CameraInfo`` messages."""

        yield from read_camera_info_messages(self.path, topic=topic)

    def read_radar_scans(self, topic: str | None = None) -> Iterator[RadarScanMessage]:
        """Yield normalized ROS 1 ``radar_msgs/RadarScan`` messages."""

        yield from read_radar_scan_messages(self.path, topic=topic)

    def read_radar_messages(self, topic: str | None = None) -> Iterator[RadarScanMessage]:
        """Compatibility alias for :meth:`read_radar_scans`."""

        yield from self.read_radar_scans(topic=topic)


def _read_exact(source: BinaryIO, size: int) -> bytes:
    data = source.read(size)
    if len(data) != size:
        msg = "unexpected end of ROS bag file"
        raise DatasetError(msg)
    return data


def _read_header_fields(header: bytes) -> dict[str, bytes]:
    fields: dict[str, bytes] = {}
    offset = 0
    total = len(header)
    while offset < total:
        if offset + 4 > total:
            msg = "truncated ROS bag record header"
            raise DatasetError(msg)
        (field_len,) = struct.unpack_from("<I", header, offset)
        offset += 4
        if offset + field_len > total:
            msg = "truncated ROS bag record header field"
            raise DatasetError(msg)
        chunk = header[offset : offset + field_len]
        offset += field_len
        separator = chunk.find(b"=")
        if separator < 0:
            msg = "malformed ROS bag header field (missing '=')"
            raise DatasetError(msg)
        name = chunk[:separator].decode("ascii", errors="strict")
        fields[name] = chunk[separator + 1 :]
    return fields


def _decompress_chunk(compression: str, size: int, data: bytes) -> bytes:
    if compression == "none":
        return data
    if compression == "bz2":
        return bz2.decompress(data)
    if compression == "lz4":
        return _lz4_decompress(data, size)
    msg = f"unsupported ROS bag chunk compression: {compression!r}"
    raise DatasetError(msg)


def _lz4_decompress(data: bytes, size: int) -> bytes:
    try:
        import lz4.frame
    except ImportError as exc:  # pragma: no cover - exercised via error path test
        msg = (
            "lz4-compressed ROS bags require the optional dependency "
            "calibrex[rosbag1-lz4]"
        )
        raise DatasetError(msg) from exc
    decompressed: bytes = lz4.frame.decompress(data)
    if size and len(decompressed) != size:  # pragma: no cover - defensive
        msg = "lz4 chunk decompressed to an unexpected size"
        raise DatasetError(msg)
    return decompressed


def _iter_inner_records(buffer: bytes) -> Iterator[tuple[dict[str, bytes], bytes]]:
    offset = 0
    total = len(buffer)
    while offset < total:
        (header_len,) = struct.unpack_from("<I", buffer, offset)
        offset += 4
        header = buffer[offset : offset + header_len]
        offset += header_len
        (data_len,) = struct.unpack_from("<I", buffer, offset)
        offset += 4
        data = buffer[offset : offset + data_len]
        offset += data_len
        yield _read_header_fields(header), data


def _open_bag(path: Path) -> BinaryIO:
    if not path.exists():
        msg = f"ROS bag does not exist: {path}"
        raise DatasetError(msg)
    source = path.open("rb")
    magic = source.read(len(BAG_MAGIC))
    if magic != BAG_MAGIC:
        source.close()
        msg = "not a ROS 1 bag v2.0 file (bad magic header)"
        raise DatasetError(msg)
    return source


def iter_messages(
    path: str | Path,
    *,
    topics: set[str] | None = None,
) -> Iterator[tuple[Rosbag1Connection, int, bytes]]:
    """Yield ``(connection, timestamp_ns, message_data)`` for message records.

    The bag is scanned sequentially. Connection records (both at the top level
    and inside chunks) populate a connection table; message records are yielded
    with their resolved connection. When ``topics`` is given, only messages on
    those topics are yielded.
    """

    bag_path = Path(path)
    connections: dict[int, Rosbag1Connection] = {}
    index_pos: int | None = None
    remaining_chunk_count: int | None = None
    source = _open_bag(bag_path)
    try:
        while True:
            if index_pos is not None and source.tell() >= index_pos:
                break
            length_prefix = source.read(4)
            if len(length_prefix) == 0:
                break
            if len(length_prefix) != 4:
                msg = "truncated ROS bag record header length"
                raise DatasetError(msg)
            (header_len,) = struct.unpack("<I", length_prefix)
            header = _read_header_fields(_read_exact(source, header_len))
            (data_len,) = struct.unpack("<I", _read_exact(source, 4))
            data = _read_exact(source, data_len)
            op = _record_op(header)
            if op == OP_CONNECTION:
                connection = _parse_connection(header, data)
                connections[connection.conn_id] = connection
            elif op == OP_CHUNK:
                yield from _iter_chunk_messages(header, data, connections, topics)
                if remaining_chunk_count is not None:
                    remaining_chunk_count -= 1
                if (
                    (remaining_chunk_count == 0 or not _skip_record_padding(source))
                    and index_pos is not None
                ):
                    source.seek(index_pos)
            elif op == OP_BAG_HEADER:
                parsed_index_pos = _uint64(header.get("index_pos"))
                index_pos = parsed_index_pos if parsed_index_pos and parsed_index_pos > 0 else None
                parsed_chunk_count = _uint32(header.get("chunk_count"))
                remaining_chunk_count = (
                    parsed_chunk_count if parsed_chunk_count and parsed_chunk_count > 0 else None
                )
                _skip_record_padding(source)
            # Other record types (index data and chunk info) are skipped.
    finally:
        source.close()


def _record_op(header: dict[str, bytes]) -> int:
    op_bytes = header.get("op")
    if op_bytes is None or len(op_bytes) < 1:
        msg = "ROS bag record header missing 'op' field"
        raise DatasetError(msg)
    return op_bytes[0]


def _skip_record_padding(source: BinaryIO) -> bool:
    """Skip optional padding between top-level ROS bag records.

    ``rosbag`` may reserve padded areas after the initial ``BAG_HEADER`` and
    between chunks.  The next actual record is not necessarily aligned to a
    simple power-of-two boundary (the TIERS Indoor02 bag has 67--702 byte gaps
    between chunks), so alignment arithmetic alone is insufficient.  Scan only
    the bounded padding area for the next header that contains a valid record
    opcode, then rewind to that header.  Normal bags without padding take the
    first-candidate path and are unchanged.

    Return ``True`` when a next record header was found.  A bounded final gap
    can return ``False`` so the caller can use the bag index position.
    """

    start = source.tell()
    for _ in range(_BAG_HEADER_PADDING_SCAN_LIMIT):
        candidate = source.tell()
        prefix = source.read(4)
        if len(prefix) != 4:
            source.seek(start)
            return False
        (header_length,) = struct.unpack("<I", prefix)
        if 0 < header_length <= 1024 * 1024:
            header_bytes = source.read(header_length)
            if len(header_bytes) == header_length:
                with suppress(DatasetError):
                    fields = _read_header_fields(header_bytes)
                    op_bytes = fields.get("op")
                    if op_bytes and op_bytes[0] in _RECORD_OPS:
                        source.seek(candidate)
                        return True
        source.seek(candidate + 1)
    source.seek(start)
    return False


def _iter_chunk_messages(
    chunk_header: dict[str, bytes],
    chunk_data: bytes,
    connections: dict[int, Rosbag1Connection],
    topics: set[str] | None,
) -> Iterator[tuple[Rosbag1Connection, int, bytes]]:
    compression = chunk_header.get("compression", b"none").decode("ascii")
    size = _uint32(chunk_header.get("size")) or 0
    buffer = _decompress_chunk(compression, size, chunk_data)
    pending: list[tuple[int, int, bytes]] = []
    for header, data in _iter_inner_records(buffer):
        op = _record_op(header)
        if op == OP_CONNECTION:
            connection = _parse_connection(header, data)
            connections[connection.conn_id] = connection
        elif op == OP_MSG_DATA:
            conn_id = _uint32(header.get("conn"))
            timestamp_ns = _time_field_to_ns(header.get("time"))
            if conn_id is None or timestamp_ns is None:
                msg = "ROS bag message record missing 'conn' or 'time'"
                raise DatasetError(msg)
            pending.append((conn_id, timestamp_ns, data))
    for conn_id, timestamp_ns, data in pending:
        resolved = connections.get(conn_id)
        if resolved is None:
            continue
        if topics is not None and resolved.topic not in topics:
            continue
        yield resolved, timestamp_ns, data


def _parse_connection(header: dict[str, bytes], data: bytes) -> Rosbag1Connection:
    conn_id = _uint32(header.get("conn"))
    topic = header.get("topic", b"").decode("utf-8")
    if conn_id is None:
        msg = "ROS bag connection record missing 'conn'"
        raise DatasetError(msg)
    data_header = _read_header_fields(data)
    message_type = data_header.get("type", b"").decode("utf-8")
    if not topic:
        topic = data_header.get("topic", b"").decode("utf-8")
    md5 = data_header.get("md5sum")
    return Rosbag1Connection(
        conn_id=conn_id,
        topic=topic,
        message_type=message_type,
        md5sum=md5.decode("ascii") if md5 is not None else None,
    )


def _uint32(value: bytes | None) -> int | None:
    if value is None or len(value) < 4:
        return None
    return int(struct.unpack("<I", value[:4])[0])


def _uint64(value: bytes | None) -> int | None:
    """Decode a little-endian unsigned 64-bit field when present."""

    if value is None or len(value) < 8:
        return None
    return int(struct.unpack("<Q", value[:8])[0])


def _time_field_to_ns(value: bytes | None) -> int | None:
    if value is None or len(value) < 8:
        return None
    secs, nsecs = struct.unpack("<II", value[:8])
    return int(secs) * 1_000_000_000 + int(nsecs)


def _pointcloud_topic_counts(
    path: str | Path,
) -> tuple[dict[str, int], dict[str, Rosbag1Connection]]:
    return _lidar_topic_counts(path)


def _lidar_topic_counts(
    path: str | Path,
) -> tuple[dict[str, int], dict[str, Rosbag1Connection]]:
    counts: dict[str, int] = {}
    connections: dict[str, Rosbag1Connection] = {}
    for connection, _timestamp_ns, _data in iter_messages(path):
        if connection.message_type not in _LIDAR_MESSAGE_TYPES:
            continue
        counts[connection.topic] = counts.get(connection.topic, 0) + 1
        connections.setdefault(connection.topic, connection)
    return counts, connections


def _supported_sensor_topic_counts(
    path: str | Path,
) -> tuple[dict[str, int], dict[str, Rosbag1Connection]]:
    """Count exact built-in LiDAR and Radar topics for ``Reader.streams``."""

    counts: dict[str, int] = {}
    connections: dict[str, Rosbag1Connection] = {}
    for connection, _timestamp_ns, _data in iter_messages(path):
        if connection.message_type not in {*LIDAR_MESSAGE_TYPES, *RADAR_MESSAGE_TYPES}:
            continue
        counts[connection.topic] = counts.get(connection.topic, 0) + 1
        connections.setdefault(connection.topic, connection)
    return counts, connections


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


def read_image_messages(
    path: str | Path,
    *,
    topic: str | None = None,
    include_data: bool = False,
) -> Iterator[ImageMessage]:
    """Yield validated ROS 1 Image messages.

    ``include_data`` defaults to ``False`` because capture inventory needs
    payload validation and frame evidence, not a second in-memory image copy.
    """

    topics = {topic} if topic is not None else None
    for connection, timestamp_ns, data in iter_messages(path, topics=topics):
        if connection.message_type != IMAGE_TYPE:
            continue
        yield decode_image(
            connection.topic,
            timestamp_ns,
            data,
            include_data=include_data,
        )


def read_camera_info_messages(
    path: str | Path,
    *,
    topic: str | None = None,
) -> Iterator[CameraInfoMessage]:
    """Yield validated ROS 1 CameraInfo messages, optionally by topic."""

    topics = {topic} if topic is not None else None
    for connection, timestamp_ns, data in iter_messages(path, topics=topics):
        if connection.message_type != CAMERA_INFO_TYPE:
            continue
        yield decode_camera_info(connection.topic, timestamp_ns, data)


def read_radar_scan_messages(
    path: str | Path,
    *,
    topic: str | None = None,
) -> Iterator[RadarScanMessage]:
    """Yield decoded ROS 1 ``radar_msgs/RadarScan`` messages."""

    topics = {topic} if topic is not None else None
    for connection, timestamp_ns, data in iter_messages(path, topics=topics):
        if connection.message_type != RADAR_SCAN_TYPE:
            continue
        yield decode_radar_scan(connection.topic, timestamp_ns, data)


def read_radar_messages(
    path: str | Path,
    *,
    topic: str | None = None,
) -> Iterator[RadarScanMessage]:
    """Compatibility alias for :func:`read_radar_scan_messages`."""

    yield from read_radar_scan_messages(path, topic=topic)


def decode_rosbag1_message(
    topic: str,
    message_type: str,
    timestamp_ns: int,
    data: bytes,
    *,
    include_image_data: bool = False,
) -> (
    PointCloud2Message
    | LivoxCustomMessage
    | ImageMessage
    | CameraInfoMessage
    | RadarScanMessage
):
    """Decode one supported ROS 1 standard/vendor payload.

    This is the typed adapter entry point used by capture inventory.  It never
    imports ROS packages and keeps Image pixel bytes optional.
    """

    if message_type == POINTCLOUD2_TYPE:
        return decode_pointcloud2(topic, timestamp_ns, data)
    if message_type == LIVOX_CUSTOMMSG_TYPE:
        return decode_livox_custommsg(topic, timestamp_ns, data)
    if message_type == IMAGE_TYPE:
        return decode_image(
            topic,
            timestamp_ns,
            data,
            include_data=include_image_data,
        )
    if message_type == CAMERA_INFO_TYPE:
        return decode_camera_info(topic, timestamp_ns, data)
    if message_type == RADAR_SCAN_TYPE:
        return decode_radar_scan(topic, timestamp_ns, data)
    msg = f"unsupported ROS 1 message type: {message_type!r}"
    raise DatasetError(msg)


def decode_radar_scan(
    topic: str,
    timestamp_ns: int,
    data: bytes,
    *,
    max_returns: int = MAX_RADAR_RETURNS,
    max_payload_bytes: int = MAX_RADAR_PAYLOAD_BYTES,
) -> RadarScanMessage:
    """Deserialize a ROS 1 ``radar_msgs/RadarScan`` payload.

    ROS 1 serialization is little-endian and does not apply CDR padding.  The
    pinned message definition is ``Header header`` followed by an unbounded
    ``RadarReturn[]`` sequence of five ``float32`` fields.
    """

    if len(data) > max_payload_bytes:
        raise DatasetError(
            f"radar_msgs/RadarScan payload length {len(data)} exceeds safe bound "
            f"{max_payload_bytes}"
        )
    offset = 0
    _sequence, offset = _read_u32_bounded(data, offset, "RadarScan header sequence")
    stamp_secs, offset = _read_u32_bounded(data, offset, "RadarScan header seconds")
    stamp_nsecs, offset = _read_u32_bounded(data, offset, "RadarScan header nanoseconds")
    if stamp_nsecs >= 1_000_000_000:
        raise DatasetError("radar_msgs/RadarScan header nanoseconds are outside [0, 1e9)")
    frame_length, frame_offset = _read_u32_bounded(
        data, offset, "RadarScan frame_id length"
    )
    if frame_length == 0 or frame_length > MAX_ROS_STRING_BYTES:
        raise DatasetError("ROS 1 RadarScan frame_id length is outside safe bounds")
    frame_end = _checked_end(data, frame_offset, frame_length, "RadarScan frame_id")
    if data[frame_end - 1] != 0:
        raise DatasetError("ROS 1 RadarScan frame_id is missing its NUL terminator")
    try:
        frame_id = data[frame_offset : frame_end - 1].decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise DatasetError("ROS 1 RadarScan frame_id is not valid UTF-8") from exc
    offset = frame_end
    count, offset = _read_u32_bounded(data, offset, "RadarScan return count")
    if count > max_returns:
        raise DatasetError(
            f"radar_msgs/RadarScan return count {count} exceeds safe bound {max_returns}"
        )
    required_bytes = int(count) * 5 * 4
    if len(data) - offset < required_bytes:
        raise DatasetError(
            "truncated radar_msgs/RadarScan return sequence: "
            f"declared {count} returns need {required_bytes} bytes, "
            f"only {max(0, len(data) - offset)} remain"
        )
    returns: list[RadarReturn] = []
    for _ in range(int(count)):
        values = struct.unpack_from("<5f", data, offset)
        offset += 5 * 4
        returns.append(
            RadarReturn(
                range=float(values[0]),
                azimuth=float(values[1]),
                elevation=float(values[2]),
                doppler_velocity=float(values[3]),
                amplitude=float(values[4]),
            )
        )
    if offset != len(data):
        raise DatasetError(
            f"radar_msgs/RadarScan payload has {len(data) - offset} trailing byte(s)"
        )
    header_stamp_ns = int(stamp_secs) * 1_000_000_000 + int(stamp_nsecs)
    return RadarScanMessage(
        topic=topic,
        timestamp_ns=header_stamp_ns if header_stamp_ns else timestamp_ns,
        frame_id=frame_id,
        returns=tuple(returns),
        source_spec=RADAR_SCAN_TYPE,
        header_stamp_ns=header_stamp_ns,
    )


def decode_ros1_radar_scan(
    topic: str,
    timestamp_ns: int,
    data: bytes,
) -> RadarScanMessage:
    """Named ROS 1 RadarScan decoder alias for adapter callers."""

    return decode_radar_scan(topic, timestamp_ns, data)


def decode_ros1_radar(
    topic: str,
    timestamp_ns: int,
    data: bytes,
) -> RadarScanMessage:
    """Compatibility alias for :func:`decode_radar_scan`."""

    return decode_radar_scan(topic, timestamp_ns, data)


def decode_image(
    topic: str,
    timestamp_ns: int,
    data: bytes,
    *,
    include_data: bool = True,
    max_width: int = MAX_IMAGE_WIDTH,
    max_height: int = MAX_IMAGE_HEIGHT,
    max_data_bytes: int = MAX_IMAGE_BYTES,
) -> ImageMessage:
    """Deserialize and validate a ROS 1 ``sensor_msgs/Image`` payload.

    Every length is checked before slicing.  ``include_data=False`` still
    validates the full serialized array length but avoids retaining image
    bytes in the typed record.
    """

    offset = 0
    _sequence, offset = _read_u32_bounded(data, offset, "Image header sequence")
    stamp_secs, offset = _read_u32_bounded(data, offset, "Image header seconds")
    stamp_nsecs, offset = _read_u32_bounded(data, offset, "Image header nanoseconds")
    if stamp_nsecs >= 1_000_000_000:
        raise DatasetError("sensor_msgs/Image header nanoseconds are outside [0, 1e9)")
    frame_id, offset = _read_string_bounded(
        data,
        offset,
        max_length=MAX_ROS_STRING_BYTES,
        field_name="Image frame_id",
    )
    height, offset = _read_u32_bounded(data, offset, "Image height")
    width, offset = _read_u32_bounded(data, offset, "Image width")
    encoding, offset = _read_string_bounded(
        data,
        offset,
        max_length=MAX_IMAGE_ENCODING_LENGTH,
        field_name="Image encoding",
    )
    is_bigendian, offset = _read_u8_bounded(data, offset, "Image is_bigendian")
    step, offset = _read_u32_bounded(data, offset, "Image step")
    data_length, offset = _read_u32_bounded(data, offset, "Image data length")
    validate_image_metadata(
        height=height,
        width=width,
        encoding=encoding,
        is_bigendian=is_bigendian,
        step=step,
        data_length=data_length,
        max_width=max_width,
        max_height=max_height,
        max_data_bytes=max_data_bytes,
    )
    end = _checked_end(data, offset, data_length, "Image data")
    if end != len(data):
        raise DatasetError(
            f"sensor_msgs/Image payload has {len(data) - end} trailing byte(s)"
        )
    image_data = bytes(data[offset:end]) if include_data else None
    header_stamp_ns = int(stamp_secs) * 1_000_000_000 + int(stamp_nsecs)
    return ImageMessage(
        topic=topic,
        timestamp_ns=header_stamp_ns if header_stamp_ns else timestamp_ns,
        frame_id=frame_id,
        height=int(height),
        width=int(width),
        encoding=encoding,
        is_bigendian=bool(is_bigendian),
        step=int(step),
        data_length=int(data_length),
        data=image_data,
        header_stamp_ns=header_stamp_ns,
    )


def decode_camera_info(
    topic: str,
    timestamp_ns: int,
    data: bytes,
    *,
    max_width: int = MAX_IMAGE_WIDTH,
    max_height: int = MAX_IMAGE_HEIGHT,
    max_distortion_coefficients: int = MAX_CAMERA_INFO_DISTORTION_COEFFICIENTS,
) -> CameraInfoMessage:
    """Deserialize and validate a ROS 1 ``sensor_msgs/CameraInfo`` payload."""

    offset = 0
    _sequence, offset = _read_u32_bounded(data, offset, "CameraInfo header sequence")
    stamp_secs, offset = _read_u32_bounded(data, offset, "CameraInfo header seconds")
    stamp_nsecs, offset = _read_u32_bounded(data, offset, "CameraInfo header nanoseconds")
    if stamp_nsecs >= 1_000_000_000:
        raise DatasetError("sensor_msgs/CameraInfo header nanoseconds are outside [0, 1e9)")
    frame_id, offset = _read_string_bounded(
        data,
        offset,
        max_length=MAX_ROS_STRING_BYTES,
        field_name="CameraInfo frame_id",
    )
    height, offset = _read_u32_bounded(data, offset, "CameraInfo height")
    width, offset = _read_u32_bounded(data, offset, "CameraInfo width")
    distortion_model, offset = _read_string_bounded(
        data,
        offset,
        max_length=MAX_ROS_STRING_BYTES,
        field_name="CameraInfo distortion_model",
    )
    distortion_count, offset = _read_u32_bounded(data, offset, "CameraInfo D length")
    if distortion_count > max_distortion_coefficients:
        raise DatasetError("sensor_msgs/CameraInfo D sequence is too long")
    d, offset = _read_f64_array_bounded(data, offset, int(distortion_count), "CameraInfo D")
    k, offset = _read_f64_array_bounded(data, offset, 9, "CameraInfo K")
    r, offset = _read_f64_array_bounded(data, offset, 9, "CameraInfo R")
    p, offset = _read_f64_array_bounded(data, offset, 12, "CameraInfo P")
    binning_x, offset = _read_u32_bounded(data, offset, "CameraInfo binning_x")
    binning_y, offset = _read_u32_bounded(data, offset, "CameraInfo binning_y")
    roi_x, offset = _read_u32_bounded(data, offset, "CameraInfo ROI x_offset")
    roi_y, offset = _read_u32_bounded(data, offset, "CameraInfo ROI y_offset")
    roi_height, offset = _read_u32_bounded(data, offset, "CameraInfo ROI height")
    roi_width, offset = _read_u32_bounded(data, offset, "CameraInfo ROI width")
    do_rectify, offset = _read_u8_bounded(data, offset, "CameraInfo ROI do_rectify")
    if do_rectify not in (0, 1):
        raise DatasetError("sensor_msgs/CameraInfo ROI do_rectify must be 0 or 1")
    roi = RegionOfInterest(
        x_offset=int(roi_x),
        y_offset=int(roi_y),
        height=int(roi_height),
        width=int(roi_width),
        do_rectify=bool(do_rectify),
    )
    validate_camera_info_metadata(
        height=int(height),
        width=int(width),
        distortion_model=distortion_model,
        d=d,
        k=k,
        r=r,
        p=p,
        binning_x=int(binning_x),
        binning_y=int(binning_y),
        roi=roi,
        max_width=max_width,
        max_height=max_height,
        max_distortion_coefficients=max_distortion_coefficients,
    )
    if offset != len(data):
        raise DatasetError(
            f"sensor_msgs/CameraInfo payload has {len(data) - offset} trailing byte(s)"
        )
    header_stamp_ns = int(stamp_secs) * 1_000_000_000 + int(stamp_nsecs)
    return CameraInfoMessage(
        topic=topic,
        timestamp_ns=header_stamp_ns if header_stamp_ns else timestamp_ns,
        frame_id=frame_id,
        height=int(height),
        width=int(width),
        distortion_model=distortion_model,
        d=d,
        k=k,
        r=r,
        p=p,
        binning_x=int(binning_x),
        binning_y=int(binning_y),
        roi=roi,
        header_stamp_ns=header_stamp_ns,
    )


def decode_ros1_image(
    topic: str,
    timestamp_ns: int,
    data: bytes,
    *,
    include_data: bool = True,
) -> ImageMessage:
    """Named ROS 1 adapter alias for callers that prefer explicit serializer names."""

    return decode_image(
        topic,
        timestamp_ns,
        data,
        include_data=include_data,
    )


def decode_ros1_camera_info(
    topic: str,
    timestamp_ns: int,
    data: bytes,
) -> CameraInfoMessage:
    """Named ROS 1 CameraInfo adapter alias."""

    return decode_camera_info(topic, timestamp_ns, data)


def decode_bag_lidar_message(
    topic: str,
    message_type: str,
    timestamp_ns: int,
    data: bytes,
) -> BagLidarMessage:
    """Decode a supported LiDAR payload from a ROS 1 bag message record."""

    if message_type == POINTCLOUD2_TYPE:
        return decode_pointcloud2(topic, timestamp_ns, data)
    if message_type == LIVOX_CUSTOMMSG_TYPE:
        return decode_livox_custommsg(topic, timestamp_ns, data)
    msg = f"unsupported ROS bag LiDAR message type: {message_type!r}"
    raise DatasetError(msg)


def decode_livox_custommsg(topic: str, timestamp_ns: int, data: bytes) -> LivoxCustomMessage:
    """Decode a serialized ``livox_ros_driver/CustomMsg`` into numpy arrays."""

    numpy_module = require_numpy(extra_name="rosbag1")
    offset = 0

    _seq, offset = _read_uint32(data, offset)
    stamp_secs, offset = _read_uint32(data, offset)
    stamp_nsecs, offset = _read_uint32(data, offset)
    frame_id, offset = _read_string(data, offset)
    if offset + 8 > len(data):
        msg = "truncated livox_ros_driver/CustomMsg (timebase)"
        raise DatasetError(msg)
    (timebase_ns,) = struct.unpack_from("<Q", data, offset)
    offset += 8
    point_num, offset = _read_uint32(data, offset)
    if offset + 4 > len(data):
        msg = "truncated livox_ros_driver/CustomMsg (header)"
        raise DatasetError(msg)
    lidar_id = int(data[offset])
    offset += 1
    offset += 3  # uint8[3] reserved
    array_len, offset = _read_uint32(data, offset)
    count = min(int(point_num), int(array_len))
    available = max(0, (len(data) - offset) // _LIVOX_CUSTOM_POINT_STEP)
    count = min(count, available)
    if count <= 0:
        empty = numpy_module.empty((0, 3), dtype=numpy_module.float64)
        header_stamp_ns = int(stamp_secs) * 1_000_000_000 + int(stamp_nsecs)
        return LivoxCustomMessage(
            topic=topic,
            timestamp_ns=header_stamp_ns if header_stamp_ns else timestamp_ns,
            frame_id=frame_id,
            timebase_ns=int(timebase_ns),
            point_num=int(point_num),
            lidar_id=lidar_id,
            xyz=empty,
            intensity=None,
            offset_time_ns=None,
            line=None,
        )

    point_dtype = numpy_module.dtype(
        [
            ("offset_time", "<u4"),
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("reflectivity", "u1"),
            ("tag", "u1"),
            ("line", "u1"),
        ]
    )
    structured = numpy_module.frombuffer(
        data,
        dtype=point_dtype,
        count=count,
        offset=offset,
    )
    xyz = numpy_module.stack(
        [
            structured["x"].astype(numpy_module.float64),
            structured["y"].astype(numpy_module.float64),
            structured["z"].astype(numpy_module.float64),
        ],
        axis=1,
    )
    intensity = structured["reflectivity"].astype(numpy_module.float64)
    offset_time_ns = structured["offset_time"].astype(numpy_module.int64)
    line = structured["line"].astype(numpy_module.int64)
    header_stamp_ns = int(stamp_secs) * 1_000_000_000 + int(stamp_nsecs)
    return LivoxCustomMessage(
        topic=topic,
        timestamp_ns=header_stamp_ns if header_stamp_ns else timestamp_ns,
        frame_id=frame_id,
        timebase_ns=int(timebase_ns),
        point_num=int(point_num),
        lidar_id=lidar_id,
        xyz=xyz,
        intensity=intensity,
        offset_time_ns=offset_time_ns,
        line=line,
    )


def decode_pose_stamped(
    topic: str,
    timestamp_ns: int,
    data: bytes,
) -> Ros1PoseStampedMessage:
    """Decode a ROS 1 ``geometry_msgs/PoseStamped`` payload."""

    offset = 0
    _sequence, offset = _read_uint32(data, offset)
    stamp_secs, offset = _read_uint32(data, offset)
    stamp_nsecs, offset = _read_uint32(data, offset)
    frame_id, offset = _read_string(data, offset)
    if offset + 56 > len(data):
        msg = "truncated geometry_msgs/PoseStamped payload"
        raise DatasetError(msg)
    values = struct.unpack_from("<7d", data, offset)
    header_stamp_ns = int(stamp_secs) * 1_000_000_000 + int(stamp_nsecs)
    return Ros1PoseStampedMessage(
        topic=topic,
        timestamp_ns=header_stamp_ns if header_stamp_ns else timestamp_ns,
        frame_id=frame_id,
        position=(float(values[0]), float(values[1]), float(values[2])),
        orientation_xyzw=(
            float(values[3]),
            float(values[4]),
            float(values[5]),
            float(values[6]),
        ),
    )


def decode_pointcloud2(
    topic: str,
    timestamp_ns: int,
    data: bytes,
    *,
    point_time_field: str | None = None,
) -> PointCloud2Message:
    """Decode a serialized ``sensor_msgs/PointCloud2`` message into numpy arrays."""

    numpy_module = require_numpy(extra_name="rosbag1")
    offset = 0

    _seq, offset = _read_uint32(data, offset)
    stamp_secs, offset = _read_uint32(data, offset)
    stamp_nsecs, offset = _read_uint32(data, offset)
    frame_id, offset = _read_string(data, offset)
    height, offset = _read_uint32(data, offset)
    width, offset = _read_uint32(data, offset)

    field_count, offset = _read_uint32(data, offset)
    fields: list[PointField] = []
    for _ in range(field_count):
        name, offset = _read_string(data, offset)
        field_offset, offset = _read_uint32(data, offset)
        datatype = data[offset]
        offset += 1
        count, offset = _read_uint32(data, offset)
        fields.append(PointField(name=name, offset=field_offset, datatype=datatype, count=count))

    is_bigendian = data[offset]
    offset += 1
    point_step, offset = _read_uint32(data, offset)
    _row_step, offset = _read_uint32(data, offset)
    payload_len, offset = _read_uint32(data, offset)
    payload = data[offset : offset + payload_len]
    offset += payload_len
    # Trailing is_dense byte is intentionally not required for decoding.

    point_count = height * width if height * width else (payload_len // point_step)
    xyz, intensity = decode_pointcloud_payload(
        numpy_module,
        payload=payload,
        fields=fields,
        point_step=point_step,
        point_count=point_count,
        is_bigendian=bool(is_bigendian),
    )
    point_time_offsets_s = None
    if point_time_field is not None:
        point_time_offsets_s = decode_point_time_offsets(
            numpy_module,
            payload=payload,
            fields=fields,
            point_step=point_step,
            point_count=point_count,
            is_bigendian=bool(is_bigendian),
            field_name=point_time_field,
            topic=topic,
        )
    raw_point_count = int(xyz.shape[0])
    xyz, intensity, point_time_offsets_s, nonfinite_xyz_count = (
        filter_nonfinite_pointcloud_rows(
            numpy_module, xyz, intensity, point_time_offsets_s
        )
    )

    header_stamp_ns = int(stamp_secs) * 1_000_000_000 + int(stamp_nsecs)
    return PointCloud2Message(
        topic=topic,
        timestamp_ns=header_stamp_ns if header_stamp_ns else timestamp_ns,
        frame_id=frame_id,
        width=width,
        height=height,
        point_step=point_step,
        fields=tuple(fields),
        xyz=xyz,
        intensity=intensity,
        point_time_offsets_s=point_time_offsets_s,
        raw_point_count=raw_point_count,
        nonfinite_xyz_count=nonfinite_xyz_count,
    )


def summarize_rosbag1(
    path: str | Path,
    *,
    sample_limit: int = 4,
) -> Rosbag1DatasetStats:
    """Summarize supported LiDAR topics in a ROS 1 bag for ``calibrex inspect``.

    Every message is counted, but at most ``sample_limit`` messages per topic are
    decoded to numpy to gather point counts and spatial bounds.
    """

    bag_path = Path(path)
    if not bag_path.exists():
        return Rosbag1DatasetStats(
            status="missing",
            connection_count=0,
            pointcloud_topic_count=0,
            reason="ROS bag path does not exist",
        )

    counts: dict[str, int] = {}
    connections: dict[str, Rosbag1Connection] = {}
    sampled_counts: dict[str, int] = {}
    sampled_points: dict[str, int] = {}
    has_intensity: dict[str, bool] = {}
    bounds_min: dict[str, tuple[float, float, float]] = {}
    bounds_max: dict[str, tuple[float, float, float]] = {}
    range_min: dict[str, float] = {}
    range_max: dict[str, float] = {}
    azimuth_samples: dict[str, list[float]] = {}
    elevation_min: dict[str, float] = {}
    elevation_max: dict[str, float] = {}
    point_time_available: dict[str, bool] = {}
    point_time_field: dict[str, str] = {}
    point_time_min: dict[str, float] = {}
    point_time_max: dict[str, float] = {}
    distance_bin_counts: dict[str, list[int]] = {}
    first_ts: dict[str, int] = {}
    last_ts: dict[str, int] = {}
    radar_counts: dict[str, int] = {}
    radar_connections: dict[str, Rosbag1Connection] = {}
    radar_sampled_counts: dict[str, int] = {}
    radar_return_counts: dict[str, int] = {}
    radar_frames: dict[str, str] = {}
    radar_range_min: dict[str, float] = {}
    radar_range_max: dict[str, float] = {}
    radar_azimuth_span: dict[str, float] = {}
    radar_elevation_span: dict[str, float] = {}
    radar_doppler_min: dict[str, float] = {}
    radar_doppler_max: dict[str, float] = {}
    radar_doppler_span: dict[str, float] = {}
    radar_duplicates: dict[str, int] = {}
    radar_diversity: dict[str, str] = {}

    try:
        for connection, timestamp_ns, data in iter_messages(bag_path):
            topic = connection.topic
            if connection.message_type in RADAR_MESSAGE_TYPES:
                radar_counts[topic] = radar_counts.get(topic, 0) + 1
                radar_connections.setdefault(topic, connection)
                if radar_sampled_counts.get(topic, 0) >= sample_limit:
                    continue
                radar_message = decode_radar_scan(topic, timestamp_ns, data)
                radar_sampled_counts[topic] = radar_sampled_counts.get(topic, 0) + 1
                if radar_message.frame_id:
                    radar_frames.setdefault(topic, radar_message.frame_id)
                _accumulate_radar_evidence_ros1(
                    radar_message,
                    topic,
                    radar_return_counts,
                    radar_range_min,
                    radar_range_max,
                    radar_azimuth_span,
                    radar_elevation_span,
                    radar_doppler_min,
                    radar_doppler_max,
                    radar_doppler_span,
                    radar_duplicates,
                    radar_diversity,
                )
                continue
            if connection.message_type not in _LIDAR_MESSAGE_TYPES:
                continue
            counts[topic] = counts.get(topic, 0) + 1
            connections.setdefault(topic, connection)
            first_ts.setdefault(topic, timestamp_ns)
            last_ts[topic] = timestamp_ns
            if sampled_counts.get(topic, 0) >= sample_limit:
                continue
            message = decode_bag_lidar_message(
                topic,
                connection.message_type,
                timestamp_ns,
                data,
            )
            if isinstance(message, PointCloud2Message):
                time_field = _pointcloud2_time_field(message)
                if time_field is not None:
                    with suppress(DatasetError):
                        message = decode_pointcloud2(
                            topic,
                            timestamp_ns,
                            data,
                            point_time_field=time_field,
                        )
                        # A field named ``time`` is common, but its datatype is
                        # not guaranteed to be an offset.  Keep geometry
                        # diagnostics useful and leave time availability false
                        # when it cannot be decoded unambiguously.
            sampled_counts[topic] = sampled_counts.get(topic, 0) + 1
            sampled_points[topic] = sampled_points.get(topic, 0) + message.point_count
            has_intensity[topic] = has_intensity.get(topic, False) or (
                message.intensity is not None
            )
            _accumulate_bounds(message, topic, bounds_min, bounds_max)
            _accumulate_geometry(
                message,
                topic,
                range_min,
                range_max,
                azimuth_samples,
                elevation_min,
                elevation_max,
                distance_bin_counts,
            )
            _accumulate_point_time(
                message,
                topic,
                point_time_available,
                point_time_field,
                point_time_min,
                point_time_max,
            )
    except DatasetError as exc:
        return Rosbag1DatasetStats(
            status="malformed",
            connection_count=len({*connections, *radar_connections}),
            pointcloud_topic_count=len(counts),
            reason=str(exc),
        )

    streams = tuple(
        Rosbag1PointCloudStreamStats(
            topic=topic,
            message_type=connections[topic].message_type,
            message_count=counts[topic],
            sampled_message_count=sampled_counts.get(topic, 0),
            sampled_point_count=sampled_points.get(topic, 0),
            has_intensity=has_intensity.get(topic, False),
            bounds_min_m=bounds_min.get(topic),
            bounds_max_m=bounds_max.get(topic),
            first_timestamp_ns=first_ts.get(topic),
            last_timestamp_ns=last_ts.get(topic),
            range_min_m=range_min.get(topic),
            range_max_m=range_max.get(topic),
            fov_azimuth_deg=_azimuth_fov_deg(azimuth_samples.get(topic, [])),
            fov_elevation_deg=_fov_span_deg(
                elevation_min.get(topic), elevation_max.get(topic)
            ),
            point_time_available=point_time_available.get(topic, False),
            point_time_field=point_time_field.get(topic),
            point_time_min_s=point_time_min.get(topic),
            point_time_max_s=point_time_max.get(topic),
            distance_bin_counts=(
                tuple(distance_bin_counts[topic])
                if topic in distance_bin_counts
                else None
            ),
        )
        for topic in sorted(counts)
    )
    radar_streams = tuple(
        Rosbag1RadarStreamStats(
            topic=topic,
            message_type=radar_connections[topic].message_type,
            message_count=radar_counts[topic],
            sampled_message_count=radar_sampled_counts.get(topic, 0),
            sampled_return_count=radar_return_counts.get(topic, 0),
            sample_frame_id=radar_frames.get(topic),
            range_min_m=radar_range_min.get(topic),
            range_max_m=radar_range_max.get(topic),
            azimuth_span_rad=radar_azimuth_span.get(topic),
            elevation_span_rad=radar_elevation_span.get(topic),
            doppler_min_mps=radar_doppler_min.get(topic),
            doppler_max_mps=radar_doppler_max.get(topic),
            doppler_span_mps=radar_doppler_span.get(topic),
            duplicate_return_count=radar_duplicates.get(topic, 0),
            diversity_status=radar_diversity.get(topic, "unknown"),
        )
        for topic in sorted(radar_counts)
    )
    status = "scored" if streams or radar_streams else "empty"
    return Rosbag1DatasetStats(
        status=status,
        connection_count=len({*connections, *radar_connections}),
        pointcloud_topic_count=len(counts),
        streams=streams,
        radar_streams=radar_streams,
        reason=None if streams or radar_streams else "no supported LiDAR or Radar topics found",
    )


def _accumulate_radar_evidence_ros1(
    message: RadarScanMessage,
    topic: str,
    return_counts: dict[str, int],
    range_min: dict[str, float],
    range_max: dict[str, float],
    azimuth_span: dict[str, float],
    elevation_span: dict[str, float],
    doppler_min: dict[str, float],
    doppler_max: dict[str, float],
    doppler_span: dict[str, float],
    duplicate_counts: dict[str, int],
    diversity_status: dict[str, str],
) -> None:
    """Merge one bounded RadarScan into ROS 1 inspect evidence."""

    return_counts[topic] = return_counts.get(topic, 0) + message.return_count
    if message.range_min_m is not None:
        range_min[topic] = min(range_min.get(topic, message.range_min_m), message.range_min_m)
    if message.range_max_m is not None:
        range_max[topic] = max(range_max.get(topic, message.range_max_m), message.range_max_m)
    if message.azimuth_span_rad is not None:
        azimuth_span[topic] = max(
            azimuth_span.get(topic, message.azimuth_span_rad), message.azimuth_span_rad
        )
    if message.elevation_span_rad is not None:
        elevation_span[topic] = max(
            elevation_span.get(topic, message.elevation_span_rad), message.elevation_span_rad
        )
    if message.doppler_min_mps is not None:
        doppler_min[topic] = min(
            doppler_min.get(topic, message.doppler_min_mps), message.doppler_min_mps
        )
    if message.doppler_max_mps is not None:
        doppler_max[topic] = max(
            doppler_max.get(topic, message.doppler_max_mps), message.doppler_max_mps
        )
    if message.doppler_span_mps is not None:
        doppler_span[topic] = max(
            doppler_span.get(topic, message.doppler_span_mps), message.doppler_span_mps
        )
    duplicate_counts[topic] = duplicate_counts.get(topic, 0) + message.duplicate_return_count
    previous = diversity_status.get(topic)
    if previous == "strong" or message.diversity_status == "strong":
        diversity_status[topic] = "strong"
    elif previous == "weak" or message.diversity_status == "weak":
        diversity_status[topic] = "weak"
    else:
        diversity_status[topic] = message.diversity_status


def _accumulate_bounds(
    message: BagLidarMessage,
    topic: str,
    bounds_min: dict[str, tuple[float, float, float]],
    bounds_max: dict[str, tuple[float, float, float]],
) -> None:
    if message.point_count == 0:
        return
    np_mod = require_numpy(extra_name="rosbag1")
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


def _pointcloud2_time_field(message: PointCloud2Message) -> str | None:
    """Return the conventional PointCloud2 point-time field, if present."""

    names = {field.name for field in message.fields}
    for candidate in ("offset_time", "time", "t"):
        if candidate in names:
            return candidate
    return None


def _accumulate_geometry(
    message: BagLidarMessage,
    topic: str,
    range_min: dict[str, float],
    range_max: dict[str, float],
    azimuth_samples: dict[str, list[float]],
    elevation_min: dict[str, float],
    elevation_max: dict[str, float],
    distance_bin_counts: dict[str, list[int]],
) -> None:
    """Accumulate finite range and angular-coverage diagnostics."""

    if message.point_count == 0:
        return
    np_mod = require_numpy(extra_name="rosbag1")
    finite = np_mod.isfinite(message.xyz).all(axis=1)
    points = message.xyz[finite]
    if points.shape[0] == 0:
        return
    horizontal = np_mod.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    ranges = np_mod.sqrt(horizontal**2 + points[:, 2] ** 2)
    current_min = float(ranges.min())
    current_max = float(ranges.max())
    range_min[topic] = min(current_min, range_min.get(topic, current_min))
    range_max[topic] = max(current_max, range_max.get(topic, current_max))
    bins = distance_bin_counts.setdefault(
        topic, [0] * (len(_DEFAULT_DISTANCE_BIN_EDGES_M) - 1)
    )
    for lower_index, lower in enumerate(_DEFAULT_DISTANCE_BIN_EDGES_M[:-1]):
        upper = _DEFAULT_DISTANCE_BIN_EDGES_M[lower_index + 1]
        in_bin = (ranges >= lower) & (
            (ranges < upper)
            | (
                (lower_index == len(bins) - 1)
                & (ranges == upper)
            )
        )
        bins[lower_index] += int(in_bin.sum())

    # A bounded angular sample keeps inspect output small for dense scans while
    # retaining enough support for a coverage diagnostic.
    step = max(1, math.ceil(points.shape[0] / 4096))
    azimuths = np_mod.degrees(np_mod.arctan2(points[::step, 1], points[::step, 0]))
    samples = azimuth_samples.setdefault(topic, [])
    samples.extend(float(value) % 360.0 for value in azimuths)
    if len(samples) > 8192:
        del samples[8192:]
    elevation = np_mod.degrees(np_mod.arctan2(points[:, 2], horizontal))
    elevation_min[topic] = min(
        float(elevation.min()), elevation_min.get(topic, float(elevation.min()))
    )
    elevation_max[topic] = max(
        float(elevation.max()), elevation_max.get(topic, float(elevation.max()))
    )


def _accumulate_point_time(
    message: BagLidarMessage,
    topic: str,
    available: dict[str, bool],
    fields: dict[str, str],
    minimum: dict[str, float],
    maximum: dict[str, float],
) -> None:
    """Accumulate decoded point-time offset bounds in seconds."""

    field_name: str | None
    if isinstance(message, LivoxCustomMessage):
        offsets = message.offset_time_ns
        field_name = "offset_time"
        scale = 1.0e-9
    elif isinstance(message, PointCloud2Message):
        offsets = message.point_time_offsets_s
        field_name = _pointcloud2_time_field(message)
        scale = 1.0
    else:  # pragma: no cover - BagLidarMessage is exhaustive
        return
    if offsets is None or int(offsets.shape[0]) == 0:
        return
    values = offsets.astype(float) * scale
    available[topic] = True
    if field_name is not None:
        fields[topic] = field_name
    current_min = float(values.min())
    current_max = float(values.max())
    minimum[topic] = min(current_min, minimum.get(topic, current_min))
    maximum[topic] = max(current_max, maximum.get(topic, current_max))


def _azimuth_fov_deg(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    ordered = sorted(values)
    largest_gap = max(
        [right - left for left, right in pairwise(ordered)]
        + [ordered[0] + 360.0 - ordered[-1]]
    )
    return max(0.0, min(360.0, 360.0 - largest_gap))


def _fov_span_deg(lower: float | None, upper: float | None) -> float | None:
    if lower is None or upper is None:
        return None
    return max(0.0, upper - lower)


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


def _read_uint32(data: bytes, offset: int) -> tuple[int, int]:
    (value,) = struct.unpack_from("<I", data, offset)
    return int(value), offset + 4


def _read_string(data: bytes, offset: int) -> tuple[str, int]:
    length, offset = _read_uint32(data, offset)
    text = data[offset : offset + length].decode("utf-8", errors="replace")
    return text, offset + length


def _read_u32_bounded(data: bytes, offset: int, field_name: str) -> tuple[int, int]:
    """Read one ROS 1 uint32 while preserving a field-specific error."""

    if offset < 0 or offset + 4 > len(data):
        raise DatasetError(f"truncated ROS 1 {field_name}")
    return int(struct.unpack_from("<I", data, offset)[0]), offset + 4


def _read_u8_bounded(data: bytes, offset: int, field_name: str) -> tuple[int, int]:
    """Read one ROS 1 uint8 with an explicit bounds check."""

    if offset < 0 or offset >= len(data):
        raise DatasetError(f"truncated ROS 1 {field_name}")
    return int(data[offset]), offset + 1


def _checked_end(data: bytes, offset: int, length: int, field_name: str) -> int:
    """Return a bounded end offset without allowing integer wrap or truncation."""

    if length < 0 or offset < 0 or length > len(data) - offset:
        raise DatasetError(f"truncated ROS 1 {field_name}")
    return offset + length


def _read_string_bounded(
    data: bytes,
    offset: int,
    *,
    max_length: int,
    field_name: str,
) -> tuple[str, int]:
    """Read a length-prefixed UTF-8 string without unbounded slicing."""

    length, offset = _read_u32_bounded(data, offset, f"{field_name} length")
    if length > max_length:
        raise DatasetError(f"ROS 1 {field_name} length {length} exceeds bound {max_length}")
    end = _checked_end(data, offset, length, field_name)
    try:
        text = data[offset:end].decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise DatasetError(f"ROS 1 {field_name} is not valid UTF-8") from exc
    return text, end


def _read_f64_array_bounded(
    data: bytes,
    offset: int,
    count: int,
    field_name: str,
) -> tuple[tuple[float, ...], int]:
    """Read a fixed/declared ROS 1 float64 array with bounds protection."""

    if count < 0 or count > (len(data) - offset) // 8:
        raise DatasetError(f"truncated ROS 1 {field_name}")
    end = offset + count * 8
    values = tuple(float(value) for value in struct.unpack_from(f"<{count}d", data, offset))
    return values, end


