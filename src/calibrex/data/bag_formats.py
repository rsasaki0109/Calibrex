"""Bag-format detection and the ROS 1 backend of the common bag reader.

``calibrex check`` / ``estimate`` / ``drift`` read bags through the functions in
:mod:`calibrex.data.rosbag2` (``list_rosbag2_connections``, ``iter_messages``,
``iter_topic_messages``). Those functions dispatch on the storage identifier that
:func:`detect_bag_format` assigns:

``sqlite3``
    rosbag2 ``.db3`` (directory with ``metadata.yaml`` or bare file).
``mcap``
    ROS 2 MCAP with CDR payloads; chunk compression ``none``, ``zstd`` and ``lz4``
    (the last two need ``calibrex[rosbag2-compression]``).
``rosbag1``
    ROS 1 ``.bag`` (v2.0); chunk compression ``none``, ``bz2`` and ``lz4``
    (``lz4`` needs ``calibrex[rosbag1-lz4]``). Payloads are transcoded from the
    ROS 1 wire format to CDR (:mod:`calibrex.data.ros1_cdr`) and the message type
    is reported in ROS 2 spelling (``sensor_msgs/msg/Imu``), so every decoder
    downstream sees the same bytes it would read from a rosbag2.

The module never imports ROS.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from calibrex.core.exceptions import DatasetError
from calibrex.data import rosbag1
from calibrex.data.ros1_cdr import Ros1ToCdrTranscoder, ros2_message_type

if TYPE_CHECKING:
    from calibrex.data.rosbag2 import Rosbag2Connection

ROSBAG1_STORAGE_ID = "rosbag1"
MCAP_MAGIC = b"\x89MCAP0\r\n"
SQLITE_MAGIC = b"SQLite format 3\x00"


def sniff_bag_file(path: Path) -> str | None:
    """Return the storage identifier of a bare bag file from its magic bytes."""

    try:
        with path.open("rb") as stream:
            head = stream.read(16)
    except OSError:
        return None
    if head.startswith(rosbag1.BAG_MAGIC):
        return ROSBAG1_STORAGE_ID
    if head.startswith(MCAP_MAGIC):
        return "mcap"
    if head.startswith(SQLITE_MAGIC):
        return "sqlite3"
    return None


def detect_bag_format(path: str | Path) -> str:
    """Return ``sqlite3``, ``mcap`` or ``rosbag1`` for a bag file or rosbag2 directory."""

    from calibrex.data.rosbag2 import resolve_storage

    return resolve_storage(path)[1]


def is_ros1_bag(path: str | Path) -> bool:
    """Return whether ``path`` is a ROS 1 bag file (by magic bytes)."""

    candidate = Path(path)
    return candidate.is_file() and sniff_bag_file(candidate) == ROSBAG1_STORAGE_ID


class _Index:
    """Connections, per-connection counts and chunk positions from a bag's index."""

    def __init__(self) -> None:
        self.connections: dict[int, rosbag1.Rosbag1Connection] = {}
        self.counts: dict[int, int] = {}
        # (chunk_pos, start_time_ns, connection ids present in the chunk)
        self.chunks: list[tuple[int, int, frozenset[int]]] = []


def _read_index(path: Path) -> _Index | None:
    """Read the bag index section; ``None`` for an unindexed (not closed) bag."""

    index = _Index()
    with path.open("rb") as source:
        if source.read(len(rosbag1.BAG_MAGIC)) != rosbag1.BAG_MAGIC:
            msg = "not a ROS 1 bag v2.0 file (bad magic header)"
            raise DatasetError(msg)
        header_len = struct.unpack("<I", rosbag1._read_exact(source, 4))[0]
        header = rosbag1._read_header_fields(rosbag1._read_exact(source, header_len))
        if rosbag1._record_op(header) != rosbag1.OP_BAG_HEADER:
            msg = "ROS bag does not start with a bag header record"
            raise DatasetError(msg)
        index_pos = rosbag1._uint64(header.get("index_pos")) or 0
        conn_count = rosbag1._uint32(header.get("conn_count")) or 0
        if index_pos <= 0 or conn_count <= 0:
            return None
        source.seek(index_pos)
        while True:
            prefix = source.read(4)
            if len(prefix) < 4:
                break
            fields = rosbag1._read_header_fields(
                rosbag1._read_exact(source, struct.unpack("<I", prefix)[0])
            )
            data_len = struct.unpack("<I", rosbag1._read_exact(source, 4))[0]
            data = rosbag1._read_exact(source, data_len)
            op = rosbag1._record_op(fields)
            if op == rosbag1.OP_CONNECTION:
                connection = rosbag1._parse_connection(fields, data)
                index.connections[connection.conn_id] = connection
            elif op == rosbag1.OP_CHUNK_INFO:
                chunk_pos = rosbag1._uint64(fields.get("chunk_pos")) or 0
                start_ns = rosbag1._time_field_to_ns(fields.get("start_time")) or 0
                present: set[int] = set()
                for offset in range(0, len(data) - 7, 8):
                    conn_id, count = struct.unpack_from("<II", data, offset)
                    present.add(conn_id)
                    index.counts[conn_id] = index.counts.get(conn_id, 0) + count
                index.chunks.append((chunk_pos, start_ns, frozenset(present)))
    if not index.connections or not index.chunks:
        return None
    return index


def list_ros1_connections(path: str | Path) -> list[tuple[Rosbag2Connection, int | None]]:
    """List ``(Rosbag2Connection, message_count)`` per topic of a ROS 1 bag."""

    bag = Path(path)
    index = _read_index(bag)
    per_topic: dict[str, tuple[Rosbag2Connection, int | None]] = {}
    if index is not None:
        for conn_id, connection in index.connections.items():
            count = index.counts.get(conn_id, 0)
            known = per_topic.get(connection.topic)
            total = count + (known[1] or 0 if known else 0)
            per_topic[connection.topic] = (_as_rosbag2(connection), total)
    else:  # unindexed bag: one pass over the records
        scanned: dict[str, int] = {}
        seen: dict[str, Rosbag2Connection] = {}
        for connection, _timestamp_ns, _data in rosbag1.iter_messages(bag):
            seen.setdefault(connection.topic, _as_rosbag2(connection))
            scanned[connection.topic] = scanned.get(connection.topic, 0) + 1
        per_topic = {topic: (seen[topic], scanned[topic]) for topic in seen}
    return [per_topic[topic] for topic in sorted(per_topic)]


def _as_rosbag2(connection: rosbag1.Rosbag1Connection) -> Rosbag2Connection:
    from calibrex.data.rosbag2 import Rosbag2Connection

    return Rosbag2Connection(
        topic_id=connection.conn_id,
        topic=connection.topic,
        message_type=ros2_message_type(connection.message_type),
        serialization_format="cdr",
    )


def iter_ros1_messages(
    path: str | Path,
    *,
    topics: set[str] | None = None,
) -> Iterator[tuple[Rosbag2Connection, int, bytes]]:
    """Yield ``(Rosbag2Connection, timestamp_ns, cdr_payload)`` from a ROS 1 bag.

    Indexed bags are read chunk by chunk and chunks without a requested topic are
    skipped without being decompressed; messages come out in timestamp order within
    each chunk and chunks in start-time order. Unindexed bags fall back to a
    sequential scan. A connection whose message definition cannot be transcoded is
    skipped when ``topics`` is ``None`` and is an error when it was asked for.
    """

    bag = Path(path)
    index = _read_index(bag)
    transcoders: dict[int, Ros1ToCdrTranscoder | None] = {}
    converted: dict[int, Rosbag2Connection] = {}

    def transcode(connection: rosbag1.Rosbag1Connection, data: bytes) -> bytes | None:
        if connection.conn_id not in transcoders:
            transcoder: Ros1ToCdrTranscoder | None = None
            error: DatasetError | None = None
            if connection.message_definition:
                try:
                    transcoder = Ros1ToCdrTranscoder(
                        connection.message_type, connection.message_definition
                    )
                except DatasetError as exc:
                    error = exc
            if transcoder is None and topics is not None:
                msg = (
                    f"cannot read ROS 1 topic {connection.topic!r} "
                    f"({connection.message_type}): "
                    f"{error or 'the connection has no message definition'}"
                )
                raise DatasetError(msg)
            transcoders[connection.conn_id] = transcoder
            converted[connection.conn_id] = _as_rosbag2(connection)
        transcoder = transcoders[connection.conn_id]
        return None if transcoder is None else transcoder.transcode(data)

    if index is None:
        for connection, timestamp_ns, data in rosbag1.iter_messages(bag, topics=topics):
            payload = transcode(connection, data)
            if payload is not None:
                yield converted[connection.conn_id], timestamp_ns, payload
        return

    wanted = {
        conn_id
        for conn_id, connection in index.connections.items()
        if topics is None or connection.topic in topics
    }
    with bag.open("rb") as source:
        for chunk_pos, _start_ns, present in sorted(index.chunks, key=lambda item: item[1]):
            if not present & wanted:
                continue
            source.seek(chunk_pos)
            header_len = struct.unpack("<I", rosbag1._read_exact(source, 4))[0]
            header = rosbag1._read_header_fields(rosbag1._read_exact(source, header_len))
            data_len = struct.unpack("<I", rosbag1._read_exact(source, 4))[0]
            chunk_data = rosbag1._read_exact(source, data_len)
            compression = header.get("compression", b"none").decode("ascii")
            size = rosbag1._uint32(header.get("size")) or 0
            buffer = rosbag1._decompress_chunk(compression, size, chunk_data)
            messages: list[tuple[int, int, bytes]] = []
            for fields, data in rosbag1._iter_inner_records(buffer):
                if rosbag1._record_op(fields) != rosbag1.OP_MSG_DATA:
                    continue
                record_conn = rosbag1._uint32(fields.get("conn"))
                record_time = rosbag1._time_field_to_ns(fields.get("time"))
                if record_conn is None or record_time is None:
                    msg = "ROS bag message record missing 'conn' or 'time'"
                    raise DatasetError(msg)
                if record_conn in wanted:
                    messages.append((record_time, record_conn, data))
            messages.sort(key=lambda item: item[0])
            for timestamp_ns, conn_id, data in messages:
                connection = index.connections[conn_id]
                payload = transcode(connection, data)
                if payload is not None:
                    yield converted[conn_id], timestamp_ns, payload
