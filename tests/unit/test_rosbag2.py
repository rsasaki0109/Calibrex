"""Tests for the pure-Python rosbag2 reader.

Fixtures are constructed byte-for-byte (sqlite3 schema, hand-assembled MCAP, and
hand-written CDR encoders) so the tests exercise the real parsers without ROS.
"""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import pytest

from calibrex.core.config import DatasetConfig
from calibrex.core.exceptions import DatasetError
from calibrex.data.inspect import inspect_dataset
from calibrex.data.ros_cdr import CdrReader, decode_ros2_odometry, decode_ros2_pointcloud2
from calibrex.data.rosbag2 import (
    MCAP_MAGIC,
    ODOMETRY_TYPE,
    POINTCLOUD2_TYPE,
    Rosbag2Reader,
    _lz4_decompress,
    _zstd_decompress,
    decode_odometry,
    decode_pointcloud2,
    iter_messages,
    read_pointcloud2_messages,
    resolve_storage,
    summarize_rosbag2,
)

Point = tuple[float, float, float, float]


class CdrWriter:
    """Minimal ROS 2 CDR (XCDR1) encoder for test fixtures."""

    def __init__(self, *, little_endian: bool = True) -> None:
        encapsulation = 1 if little_endian else 0
        self._buf = bytearray([encapsulation, 0, 0, 0])
        self._little = little_endian
        self._endian = "<" if little_endian else ">"

    def align(self, alignment: int) -> None:
        if alignment <= 1:
            return
        relative_offset = len(self._buf) - 4
        padding = (-relative_offset) % alignment
        self._buf.extend(b"\x00" * padding)

    def write_int32(self, value: int) -> None:
        self.align(4)
        self._buf.extend(struct.pack(f"{self._endian}i", value))

    def write_uint32(self, value: int) -> None:
        self.align(4)
        self._buf.extend(struct.pack(f"{self._endian}I", value))

    def write_uint8(self, value: int) -> None:
        self.align(1)
        self._buf.append(value & 0xFF)

    def write_float64(self, value: float) -> None:
        self.align(8)
        self._buf.extend(struct.pack(f"{self._endian}d", value))

    def write_bool(self, value: bool) -> None:
        self.write_uint8(1 if value else 0)

    def write_string(self, value: str) -> None:
        encoded = value.encode("utf-8") + b"\x00"
        self.write_uint32(len(encoded))
        self._buf.extend(encoded)

    def write_byte_sequence(self, value: bytes) -> None:
        self.write_uint32(len(value))
        self._buf.extend(value)

    def write_float64_array(self, values: tuple[float, ...]) -> None:
        for item in values:
            self.write_float64(item)

    def finish(self) -> bytes:
        return bytes(self._buf)


def _encode_pointcloud2(
    points: list[Point],
    *,
    frame_id: str,
    secs: int,
    nsecs: int,
    with_intensity: bool,
    little_endian: bool = True,
    alignment_sensitive: bool = False,
) -> bytes:
    writer = CdrWriter(little_endian=little_endian)
    writer.write_int32(secs)
    writer.write_uint32(nsecs)
    writer.write_string(frame_id)
    height = 1
    width = len(points)
    writer.write_uint32(height)
    writer.write_uint32(width)

    if alignment_sensitive:
        field_defs = [
            ("label", 0, 2),
            ("x", 4, 7),
            ("y", 8, 7),
            ("z", 12, 7),
        ]
        point_step = 16
    else:
        field_defs = [("x", 0, 7), ("y", 4, 7), ("z", 8, 7)]
        if with_intensity:
            field_defs.append(("intensity", 12, 7))
        point_step = 16 if with_intensity else 12

    writer.write_uint32(len(field_defs))
    for name, offset, datatype in field_defs:
        writer.write_string(name)
        writer.write_uint32(offset)
        writer.write_uint8(datatype)
        writer.align(4)
        writer.write_uint32(1)

    writer.write_bool(False)
    writer.align(4)
    writer.write_uint32(point_step)
    writer.write_uint32(point_step * width)

    payload = bytearray()
    for x, y, z, intensity in points:
        if alignment_sensitive:
            payload.extend(struct.pack("<Bxxx", 7))
            payload.extend(struct.pack("<fff", x, y, z))
        elif with_intensity:
            payload.extend(struct.pack("<ffff", x, y, z, intensity))
        else:
            payload.extend(struct.pack("<fff", x, y, z))
    writer.write_byte_sequence(bytes(payload))
    writer.write_bool(True)
    return writer.finish()


def _encode_odometry(
    *,
    frame_id: str,
    child_frame_id: str,
    secs: int,
    nsecs: int,
    position: tuple[float, float, float],
    orientation_xyzw: tuple[float, float, float, float],
    linear_velocity: tuple[float, float, float] = (0.1, 0.2, 0.3),
    angular_velocity: tuple[float, float, float] = (0.01, 0.02, 0.03),
    pose_covariance_diag: tuple[float, float, float] = (0.5, 0.6, 0.7),
) -> bytes:
    writer = CdrWriter()
    writer.write_int32(secs)
    writer.write_uint32(nsecs)
    writer.write_string(frame_id)
    writer.write_string(child_frame_id)
    for value in position:
        writer.write_float64(value)
    for value in orientation_xyzw:
        writer.write_float64(value)
    pose_cov = [0.0] * 36
    pose_cov[0] = pose_covariance_diag[0]
    pose_cov[7] = pose_covariance_diag[1]
    pose_cov[14] = pose_covariance_diag[2]
    writer.write_float64_array(tuple(pose_cov))
    for value in linear_velocity:
        writer.write_float64(value)
    for value in angular_velocity:
        writer.write_float64(value)
    writer.write_float64_array((0.0,) * 36)
    return writer.finish()


def _mcap_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<I", len(encoded)) + encoded


def _mcap_record(opcode: int, content: bytes) -> bytes:
    return bytes([opcode]) + struct.pack("<Q", len(content)) + content


def _mcap_schema(schema_id: int, name: str) -> bytes:
    content = struct.pack("<H", schema_id)
    content += _mcap_string(name)
    content += _mcap_string("ros2msg")
    content += struct.pack("<I", 0)
    return _mcap_record(0x03, content)


def _mcap_channel(channel_id: int, schema_id: int, topic: str) -> bytes:
    content = struct.pack("<HH", channel_id, schema_id)
    content += _mcap_string(topic)
    content += _mcap_string("cdr")
    content += struct.pack("<I", 0)
    return _mcap_record(0x04, content)


def _mcap_message(channel_id: int, log_time: int, payload: bytes) -> bytes:
    content = struct.pack("<HIQQ", channel_id, 0, log_time, log_time)
    content += payload
    return _mcap_record(0x05, content)


def _mcap_chunk(
    records: bytes,
    *,
    compression: str = "",
    start_time: int = 0,
    end_time: int = 0,
) -> bytes:
    compressed = records
    if compression == "lz4":
        lz4 = pytest.importorskip("lz4.block")
        compressed = lz4.compress(records, store_size=False)
    elif compression == "zstd":
        zstandard = pytest.importorskip("zstandard")
        compressed = zstandard.ZstdCompressor().compress(records)

    content = struct.pack("<QQQ", start_time, end_time, len(records))
    content += struct.pack("<I", 0)
    content += _mcap_string(compression)
    content += struct.pack("<Q", len(compressed)) + compressed
    return _mcap_record(0x06, content)


def _write_mcap_bag(
    path: Path,
    *,
    messages: list[tuple[int, int, str, str, int, bytes]],
    chunked: bool = False,
    compression: str = "",
) -> None:
    """Write a minimal MCAP bag.

    Each message tuple is
    ``(channel_id, schema_id, topic, message_type, log_time_ns, payload)``.
    """

    header = _mcap_record(0x01, _mcap_string("rosbag2") + _mcap_string("calibrex-test"))
    schemas: dict[int, str] = {}
    channels: dict[int, tuple[int, str]] = {}
    for channel_id, schema_id, topic, message_type, _log_time, _payload in messages:
        schemas[schema_id] = message_type
        channels[channel_id] = (schema_id, topic)

    preamble = b""
    for schema_id in sorted(schemas):
        preamble += _mcap_schema(schema_id, schemas[schema_id])
    for channel_id in sorted(channels):
        schema_id, topic = channels[channel_id]
        preamble += _mcap_channel(channel_id, schema_id, topic)

    message_records = b""
    for channel_id, _schema_id, _topic, _message_type, log_time, payload in messages:
        message_records += _mcap_message(channel_id, log_time, payload)

    if chunked:
        start_time = min(log_time for *_rest, log_time, _payload in messages)
        end_time = max(log_time for *_rest, log_time, _payload in messages)
        chunk_body = preamble + message_records
        body = _mcap_chunk(
            chunk_body,
            compression=compression,
            start_time=start_time,
            end_time=end_time,
        )
    else:
        body = preamble + message_records

    data_end = _mcap_record(0x0F, struct.pack("<I", 0))
    footer = _mcap_record(0x02, struct.pack("<QQI", 0, 0, 0))
    path.write_bytes(MCAP_MAGIC + header + body + data_end + footer + MCAP_MAGIC)


def _write_sqlite_bag(
    path: Path,
    *,
    topics: list[tuple[str, str]],
    messages: list[tuple[str, int, bytes]],
) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
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
        topic_ids: dict[str, int] = {}
        for index, (topic, message_type) in enumerate(topics, start=1):
            conn.execute(
                "INSERT INTO topics (id, name, type, serialization_format, offered_qos_profiles) "
                "VALUES (?, ?, ?, ?, ?)",
                (index, topic, message_type, "cdr", ""),
            )
            topic_ids[topic] = index
        for message_index, (topic, timestamp_ns, payload) in enumerate(messages, start=1):
            conn.execute(
                "INSERT INTO messages (id, topic_id, timestamp, data) VALUES (?, ?, ?, ?)",
                (message_index, topic_ids[topic], timestamp_ns, payload),
            )
        conn.commit()
    finally:
        conn.close()


def _sample_messages() -> tuple[list[tuple[str, str]], list[tuple[str, int, bytes]]]:
    cloud_a = _encode_pointcloud2(
        [(1.0, 2.0, 3.0, 10.0), (4.0, 5.0, 6.0, 20.0)],
        frame_id="lidar_frame",
        secs=100,
        nsecs=500,
        with_intensity=True,
    )
    cloud_b = _encode_pointcloud2(
        [(-1.0, -2.0, -3.0, 5.0)],
        frame_id="avia_frame",
        secs=100,
        nsecs=750,
        with_intensity=True,
    )
    odom = _encode_odometry(
        frame_id="odom",
        child_frame_id="base_link",
        secs=100,
        nsecs=600,
        position=(1.5, 2.5, 3.5),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    topics = [
        ("/livox/lidar", POINTCLOUD2_TYPE),
        ("/avia/livox/lidar", POINTCLOUD2_TYPE),
        ("/odom", ODOMETRY_TYPE),
    ]
    messages = [
        ("/livox/lidar", 100_000_000_500, cloud_a),
        ("/odom", 100_000_000_600, odom),
        ("/avia/livox/lidar", 100_000_000_750, cloud_b),
    ]
    return topics, messages


def _write_sqlite_fixture(path: Path) -> Path:
    topics, messages = _sample_messages()
    _write_sqlite_bag(path, topics=topics, messages=messages)
    return path


def _write_mcap_fixture(path: Path, *, chunked: bool = False, compression: str = "") -> Path:
    topics, messages = _sample_messages()
    mcap_messages: list[tuple[int, int, str, str, int, bytes]] = []
    schema_ids = {message_type: index + 1 for index, (_topic, message_type) in enumerate(topics)}
    channel_ids = {topic: index + 1 for index, (topic, _message_type) in enumerate(topics)}
    for topic, timestamp_ns, payload in messages:
        message_type = next(item[1] for item in topics if item[0] == topic)
        mcap_messages.append(
            (
                channel_ids[topic],
                schema_ids[message_type],
                topic,
                message_type,
                timestamp_ns,
                payload,
            )
        )
    _write_mcap_bag(path, messages=mcap_messages, chunked=chunked, compression=compression)
    return path


def _write_metadata_directory(base: Path, storage_name: str, storage_id: str) -> Path:
    bag_dir = base / "bag_dir"
    bag_dir.mkdir()
    if storage_id == "sqlite3":
        _write_sqlite_fixture(bag_dir / storage_name)
    else:
        _write_mcap_fixture(bag_dir / storage_name)
    metadata = f"""rosbag2_bagfile_information:
  version: 5
  storage_identifier: {storage_id}
  relative_file_paths:
    - {storage_name}
  topics_with_message_count:
    - topic_metadata:
        name: /livox/lidar
        type: sensor_msgs/msg/PointCloud2
        serialization_format: cdr
      message_count: 1
"""
    (bag_dir / "metadata.yaml").write_text(metadata, encoding="utf-8")
    return bag_dir


def test_rosbag2_sqlite_roundtrip_pointcloud_and_odometry(tmp_path: Path) -> None:
    bag = _write_sqlite_fixture(tmp_path / "sample.db3")
    decoded_clouds = list(read_pointcloud2_messages(bag))
    assert len(decoded_clouds) == 2
    assert decoded_clouds[0].point_count == 2
    assert decoded_clouds[0].xyz[0].tolist() == [1.0, 2.0, 3.0]
    assert decoded_clouds[1].xyz[0].tolist() == [-1.0, -2.0, -3.0]

    odom_messages = [
        decode_odometry(connection.topic, timestamp_ns, data)
        for connection, timestamp_ns, data in iter_messages(bag)
        if connection.message_type == ODOMETRY_TYPE
    ]
    assert len(odom_messages) == 1
    assert odom_messages[0].position == (1.5, 2.5, 3.5)
    assert odom_messages[0].orientation_xyzw == (0.0, 0.0, 0.0, 1.0)
    assert odom_messages[0].pose_covariance[0] == pytest.approx(0.5)


def test_rosbag2_mcap_unchunked_roundtrip(tmp_path: Path) -> None:
    bag = _write_mcap_fixture(tmp_path / "sample.mcap")
    messages = list(iter_messages(bag))
    assert len(messages) == 3
    assert [item[1] for item in messages] == sorted(item[1] for item in messages)


def test_rosbag2_mcap_chunked_lz4_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("lz4")
    bag = _write_mcap_fixture(tmp_path / "chunked.mcap", chunked=True, compression="lz4")
    clouds = list(read_pointcloud2_messages(bag))
    assert len(clouds) == 2
    assert clouds[0].xyz[1].tolist() == [4.0, 5.0, 6.0]


def test_rosbag2_mcap_chunked_zstd_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("zstandard")
    bag = _write_mcap_fixture(tmp_path / "chunked_zstd.mcap", chunked=True, compression="zstd")
    stats = summarize_rosbag2(bag)
    assert stats.status == "scored"
    assert stats.topic_count == 3


def test_rosbag2_topic_filtering(tmp_path: Path) -> None:
    bag = _write_sqlite_fixture(tmp_path / "sample.db3")
    filtered = list(iter_messages(bag, topics={"/avia/livox/lidar"}))
    assert len(filtered) == 1
    assert filtered[0][0].topic == "/avia/livox/lidar"


def test_rosbag2_time_ordering_across_channels(tmp_path: Path) -> None:
    bag = _write_sqlite_fixture(tmp_path / "sample.db3")
    ordered_topics = [connection.topic for connection, _ts, _data in iter_messages(bag)]
    assert ordered_topics == ["/livox/lidar", "/odom", "/avia/livox/lidar"]


def test_rosbag2_cdr_endianness_flag() -> None:
    payload = _encode_pointcloud2(
        [(9.0, 8.0, 7.0, 1.0)],
        frame_id="be_frame",
        secs=1,
        nsecs=2,
        with_intensity=True,
        little_endian=False,
    )
    assert payload[0] == 0
    decoded = decode_pointcloud2("/cloud", 0, payload)
    assert decoded.xyz[0].tolist() == [9.0, 8.0, 7.0]


def test_rosbag2_alignment_sensitive_pointcloud_layout() -> None:
    payload = _encode_pointcloud2(
        [(1.25, 2.25, 3.25, 0.0)],
        frame_id="aligned",
        secs=3,
        nsecs=4,
        with_intensity=False,
        alignment_sensitive=True,
    )
    decoded = decode_ros2_pointcloud2("/cloud", 0, payload)
    assert decoded.xyz[0].tolist() == [1.25, 2.25, 3.25]


def test_rosbag2_missing_lz4_codec_error() -> None:
    if importlib_available("lz4"):
        pytest.skip("lz4 is installed in this environment")
    with pytest.raises(DatasetError, match=r"calibrex\[rosbag2-compression\]"):
        _lz4_decompress(b"\x00", 1)


def test_rosbag2_missing_zstd_codec_error() -> None:
    if importlib_available("zstandard"):
        pytest.skip("zstandard is installed in this environment")
    with pytest.raises(DatasetError, match=r"calibrex\[rosbag2-compression\]"):
        _zstd_decompress(b"\x00", 1)


def importlib_available(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def test_rosbag2_bare_db3_entry_point(tmp_path: Path) -> None:
    bag = _write_sqlite_fixture(tmp_path / "bare.db3")
    storage_path, storage_id = resolve_storage(bag)
    assert storage_id == "sqlite3"
    assert storage_path == bag


def test_rosbag2_bare_mcap_entry_point(tmp_path: Path) -> None:
    bag = _write_mcap_fixture(tmp_path / "bare.mcap")
    storage_path, storage_id = resolve_storage(bag)
    assert storage_id == "mcap"
    assert storage_path == bag


def test_rosbag2_directory_with_metadata_sqlite(tmp_path: Path) -> None:
    bag_dir = _write_metadata_directory(tmp_path, "bag_0.db3", "sqlite3")
    storage_path, storage_id = resolve_storage(bag_dir)
    assert storage_id == "sqlite3"
    assert storage_path.name == "bag_0.db3"
    assert len(list(iter_messages(bag_dir))) == 3


def test_rosbag2_directory_with_metadata_mcap(tmp_path: Path) -> None:
    bag_dir = _write_metadata_directory(tmp_path, "bag_0.mcap", "mcap")
    storage_path, storage_id = resolve_storage(bag_dir)
    assert storage_id == "mcap"
    assert storage_path.name == "bag_0.mcap"


def test_rosbag2_reader_streams(tmp_path: Path) -> None:
    bag = _write_sqlite_fixture(tmp_path / "sample.db3")
    reader = Rosbag2Reader(bag)
    streams = {stream.name: stream for stream in reader.streams()}
    assert streams["/livox/lidar"].message_count == 1
    assert streams["/odom"].kind == "odometry"


def test_rosbag2_inspect_integration(tmp_path: Path) -> None:
    bag = _write_sqlite_fixture(tmp_path / "sample.db3")
    inspection = inspect_dataset(DatasetConfig(type="rosbag2", path=str(bag)))
    assert inspection.dataset_type == "rosbag2"
    diagnostics = inspection.diagnostics["rosbag2"]
    assert diagnostics["status"] == "scored"
    assert diagnostics["storage_identifier"] == "sqlite3"
    odom_stream = next(item for item in diagnostics["streams"] if item["topic"] == "/odom")
    assert odom_stream["sample_pose_position_m"] == [1.5, 2.5, 3.5]


def test_rosbag2_inspect_missing_path(tmp_path: Path) -> None:
    inspection = inspect_dataset(
        DatasetConfig(type="rosbag2", path=str(tmp_path / "missing.db3"))
    )
    assert inspection.exists is False


def test_rosbag2_rejects_bad_mcap_magic(tmp_path: Path) -> None:
    bad = tmp_path / "bad.mcap"
    bad.write_bytes(b"not-mcap")
    with pytest.raises(DatasetError, match="MCAP"):
        list(iter_messages(bad))


def test_cdr_reader_roundtrip_primitives() -> None:
    writer = CdrWriter()
    writer.write_int32(-7)
    writer.write_uint32(9)
    writer.write_string("frame")
    payload = writer.finish()
    reader = CdrReader(payload)
    assert reader.little_endian is True
    assert reader.read_int32() == -7
    assert reader.read_uint32() == 9
    assert reader.read_string() == "frame"


def test_rosbag2_odometry_golden_cdr_vector() -> None:
    """Decode a hand-assembled XCDR1 Odometry payload (no CdrWriter).

    Offset arithmetic is relative to byte 4 (first byte after encapsulation):
      rel  0- 3: stamp.sec=1, stamp.nanosec=2
      rel  8-11: frame_id length=5
      rel 12-16: "odom\\0"
      rel 17-19: pad 3 bytes (17 mod 4 = 1) before child_frame_id length
      rel 20-23: child_frame_id length=5
      rel 24-28: "base\\0"
      rel 29-31: pad 3 bytes (29 mod 8 = 5) before first float64
      rel 32-39: pose.position.x = 1.0  -> absolute byte 36
    """

    payload = bytes.fromhex(
        "01000000"  # encapsulation (LE CDR)
        "01000000"  # rel 0: stamp.sec = 1
        "02000000"  # rel 4: stamp.nanosec = 2
        "05000000"  # rel 8: frame_id length = 5
        "6f646f6d00"  # rel 12: "odom\\0"
        "000000"  # rel 17: pad to 4-byte boundary
        "05000000"  # rel 20: child_frame_id length = 5
        "6261736500"  # rel 24: "base\\0"
        "000000"  # rel 29: pad to 8-byte boundary
        "000000000000f03f"  # rel 32: pose.position.x = 1.0
        "0000000000000040"  # rel 40: pose.position.y = 2.0
        "0000000000000840"  # rel 48: pose.position.z = 3.0
        "0000000000001040"  # rel 56: pose.orientation.x = 4.0
        "0000000000001440"  # rel 64: pose.orientation.y = 5.0
        "0000000000001840"  # rel 72: pose.orientation.z = 6.0
        "0000000000001c40"  # rel 80: pose.orientation.w = 7.0
        + "00" * (8 * 78)  # rel 88: pose/twist covariances and twist velocities = 0.0
    )
    first_float64_abs = 36
    assert struct.unpack_from("<d", payload, first_float64_abs)[0] == pytest.approx(1.0)

    decoded = decode_ros2_odometry("/odom", 99, payload)
    assert decoded.timestamp_ns == 1_000_000_002
    assert decoded.frame_id == "odom"
    assert decoded.child_frame_id == "base"
    assert decoded.position == pytest.approx((1.0, 2.0, 3.0))
    assert decoded.orientation_xyzw == pytest.approx((4.0, 5.0, 6.0, 7.0))

    reader = CdrReader(payload)
    reader.read_int32()
    reader.read_uint32()
    reader.read_string()
    reader.read_string()
    reader.align(8)
    assert reader.offset == first_float64_abs
    assert reader.read_float64() == pytest.approx(1.0)


def test_rosbag2_odometry_covariance_decode() -> None:
    payload = _encode_odometry(
        frame_id="odom",
        child_frame_id="base",
        secs=1,
        nsecs=0,
        position=(0.0, 0.0, 0.0),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        pose_covariance_diag=(1.1, 2.2, 3.3),
    )
    decoded = decode_ros2_odometry("/odom", 1, payload)
    assert decoded.pose_covariance[0] == pytest.approx(1.1)
    assert decoded.pose_covariance[7] == pytest.approx(2.2)
    assert decoded.pose_covariance[14] == pytest.approx(3.3)
