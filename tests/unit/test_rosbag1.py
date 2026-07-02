"""Tests for the pure-Python ROS 1 bag (v2.0) reader.

The bag fixtures are constructed byte-for-byte here so the tests exercise the
real file-format parser (no ROS / rosbag dependency).
"""

from __future__ import annotations

import bz2
import importlib.util
import struct
from pathlib import Path

import pytest

from calibrex.core.config import DatasetConfig
from calibrex.core.exceptions import DatasetError
from calibrex.data.inspect import inspect_dataset
from calibrex.data.rosbag1 import (
    BAG_MAGIC,
    Rosbag1Reader,
    _decompress_chunk,
    decode_pointcloud2,
    read_pointcloud2_messages,
    summarize_rosbag1,
)

Point = tuple[float, float, float, float]


def _header(fields: dict[str, bytes]) -> bytes:
    out = b""
    for name, value in fields.items():
        payload = name.encode("ascii") + b"=" + value
        out += struct.pack("<I", len(payload)) + payload
    return out


def _record(header_fields: dict[str, bytes], data: bytes) -> bytes:
    header = _header(header_fields)
    return struct.pack("<I", len(header)) + header + struct.pack("<I", len(data)) + data


def _connection_record(conn_id: int, topic: str, message_type: str) -> bytes:
    data = _header(
        {
            "topic": topic.encode("utf-8"),
            "type": message_type.encode("utf-8"),
            "md5sum": b"1158d486dd51d683ce2f1be655c3c181",
            "message_definition": b"# sensor_msgs/PointCloud2",
        }
    )
    return _record(
        {
            "op": b"\x07",
            "conn": struct.pack("<I", conn_id),
            "topic": topic.encode("utf-8"),
        },
        data,
    )


def _serialize_pointcloud2(
    points: list[Point],
    *,
    frame_id: str,
    secs: int,
    nsecs: int,
    with_intensity: bool,
) -> bytes:
    buf = b""
    buf += struct.pack("<I", 0)  # header.seq
    buf += struct.pack("<I", secs)  # header.stamp.secs
    buf += struct.pack("<I", nsecs)  # header.stamp.nsecs
    frame = frame_id.encode("utf-8")
    buf += struct.pack("<I", len(frame)) + frame
    height = 1
    width = len(points)
    buf += struct.pack("<I", height)
    buf += struct.pack("<I", width)
    field_defs = [("x", 0), ("y", 4), ("z", 8)]
    if with_intensity:
        field_defs.append(("intensity", 12))
    buf += struct.pack("<I", len(field_defs))
    for name, offset in field_defs:
        encoded = name.encode("ascii")
        buf += struct.pack("<I", len(encoded)) + encoded
        buf += struct.pack("<I", offset)
        buf += struct.pack("<B", 7)  # FLOAT32
        buf += struct.pack("<I", 1)  # count
    buf += struct.pack("<B", 0)  # is_bigendian
    point_step = 16 if with_intensity else 12
    buf += struct.pack("<I", point_step)
    buf += struct.pack("<I", point_step * width)  # row_step
    payload = b""
    for x, y, z, intensity in points:
        if with_intensity:
            payload += struct.pack("<ffff", x, y, z, intensity)
        else:
            payload += struct.pack("<fff", x, y, z)
    buf += struct.pack("<I", len(payload)) + payload
    buf += struct.pack("<B", 1)  # is_dense
    return buf


def _message_record(conn_id: int, secs: int, nsecs: int, data: bytes) -> bytes:
    return _record(
        {
            "op": b"\x02",
            "conn": struct.pack("<I", conn_id),
            "time": struct.pack("<II", secs, nsecs),
        },
        data,
    )


def _write_bag(
    path: Path,
    topics: list[tuple[int, str, list[tuple[Point, int, int]]]],
    *,
    compression: str = "none",
    with_intensity: bool = True,
) -> None:
    """Write a minimal but valid rosbag1 v2.0 file.

    ``topics`` is a list of ``(conn_id, topic, messages)`` where each message is
    ``(points, secs, nsecs)``.
    """

    inner = b""
    for conn_id, topic, _messages in topics:
        inner += _connection_record(conn_id, topic, "sensor_msgs/PointCloud2")
    for conn_id, topic, messages in topics:
        for points, secs, nsecs in messages:
            serialized = _serialize_pointcloud2(
                list(points),
                frame_id=f"{topic.strip('/').replace('/', '_')}_frame",
                secs=secs,
                nsecs=nsecs,
                with_intensity=with_intensity,
            )
            inner += _message_record(conn_id, secs, nsecs, serialized)

    if compression == "none":
        chunk_data = inner
    elif compression == "bz2":
        chunk_data = bz2.compress(inner)
    else:  # pragma: no cover - not used by tests
        raise ValueError(compression)

    bag_header = _record(
        {
            "op": b"\x03",
            "index_pos": struct.pack("<Q", 0),
            "conn_count": struct.pack("<I", len(topics)),
            "chunk_count": struct.pack("<I", 1),
        },
        b"\x00" * 32,
    )
    chunk = _record(
        {
            "op": b"\x05",
            "compression": compression.encode("ascii"),
            "size": struct.pack("<I", len(inner)),
        },
        chunk_data,
    )
    path.write_bytes(BAG_MAGIC + bag_header + chunk)


def _sample_bag(path: Path, *, compression: str = "none") -> Path:
    horizon = [
        ([(1.0, 2.0, 3.0, 10.0), (4.0, 5.0, 6.0, 20.0)], 100, 500),
        ([(1.1, 2.1, 3.1, 11.0)], 101, 0),
    ]
    avia = [
        ([(-1.0, -2.0, -3.0, 5.0), (0.5, 0.5, 0.5, 7.0)], 100, 750),
    ]
    _write_bag(
        path,
        [
            (0, "/livox/lidar", horizon),
            (1, "/avia/livox/lidar", avia),
        ],
        compression=compression,
    )
    return path


def test_rosbag1_lists_pointcloud_topics(tmp_path: Path) -> None:
    bag = _sample_bag(tmp_path / "sample.bag")
    reader = Rosbag1Reader(bag)
    streams = {stream.name: stream for stream in reader.streams()}
    assert set(streams) == {"/livox/lidar", "/avia/livox/lidar"}
    assert streams["/livox/lidar"].message_count == 2
    assert streams["/avia/livox/lidar"].message_count == 1
    assert streams["/livox/lidar"].kind == "pointcloud"
    assert streams["/livox/lidar"].sensor == "sensor_msgs/PointCloud2"


def test_rosbag1_decodes_pointcloud2_to_numpy(tmp_path: Path) -> None:
    bag = _sample_bag(tmp_path / "sample.bag")
    messages = list(read_pointcloud2_messages(bag, topic="/livox/lidar"))
    assert len(messages) == 2
    first = messages[0]
    assert first.point_count == 2
    assert first.xyz.shape == (2, 3)
    assert first.xyz[0].tolist() == [1.0, 2.0, 3.0]
    assert first.xyz[1].tolist() == [4.0, 5.0, 6.0]
    assert first.intensity is not None
    assert first.intensity.tolist() == [10.0, 20.0]
    assert first.frame_id == "livox_lidar_frame"


def test_rosbag1_records_normalize_ros_time(tmp_path: Path) -> None:
    bag = _sample_bag(tmp_path / "sample.bag")
    records = list(Rosbag1Reader(bag).records("/livox/lidar"))
    assert [record.timestamp_ns for record in records] == [
        100 * 1_000_000_000 + 500,
        101 * 1_000_000_000,
    ]
    assert records[0].metadata["message_type"] == "sensor_msgs/PointCloud2"
    # The decoded message carries the same normalized header stamp.
    message = next(iter(read_pointcloud2_messages(bag, topic="/livox/lidar")))
    assert message.timestamp_ns == 100 * 1_000_000_000 + 500


def test_rosbag1_bz2_chunk_roundtrip(tmp_path: Path) -> None:
    bag = _sample_bag(tmp_path / "sample_bz2.bag", compression="bz2")
    stats = summarize_rosbag1(bag)
    assert stats.status == "scored"
    assert stats.pointcloud_topic_count == 2
    messages = list(read_pointcloud2_messages(bag, topic="/avia/livox/lidar"))
    assert len(messages) == 1
    assert messages[0].xyz.tolist() == [[-1.0, -2.0, -3.0], [0.5, 0.5, 0.5]]


def test_rosbag1_summary_bounds_and_intensity(tmp_path: Path) -> None:
    bag = _sample_bag(tmp_path / "sample.bag")
    stats = summarize_rosbag1(bag)
    by_topic = {stream.topic: stream for stream in stats.streams}
    horizon = by_topic["/livox/lidar"]
    assert horizon.message_count == 2
    assert horizon.sampled_point_count == 3
    assert horizon.has_intensity is True
    assert horizon.bounds_min_m == (1.0, 2.0, 3.0)
    assert horizon.bounds_max_m == pytest.approx((4.0, 5.0, 6.0))
    assert horizon.first_timestamp_ns == 100 * 1_000_000_000 + 500
    assert horizon.last_timestamp_ns == 101 * 1_000_000_000


def test_rosbag1_inspect_integration(tmp_path: Path) -> None:
    bag = _sample_bag(tmp_path / "sample.bag")
    inspection = inspect_dataset(DatasetConfig(type="rosbag1", path=str(bag)))
    assert inspection.dataset_type == "rosbag1"
    assert inspection.exists
    assert {stream.name for stream in inspection.streams} == {
        "/livox/lidar",
        "/avia/livox/lidar",
    }
    diagnostics = inspection.diagnostics["rosbag1"]
    assert isinstance(diagnostics, dict)
    assert diagnostics["pointcloud_topic_count"] == 2
    assert diagnostics["status"] == "scored"
    assert inspection.warnings == []


def test_rosbag1_inspection_feeds_lidar_coverage_metrics(tmp_path: Path) -> None:
    from calibrex.evaluation.lidar import lidar_metrics_from_inspection

    bag = _sample_bag(tmp_path / "sample.bag")
    inspection = inspect_dataset(DatasetConfig(type="rosbag1", path=str(bag)))
    metrics = lidar_metrics_from_inspection(inspection)
    assert metrics["lidar_pointcloud_topic_count"].value == 2.0
    assert metrics["lidar_pointcloud_topic_count"].grade == "pass"
    assert metrics["lidar_frame_coverage"].value == 3.0
    assert metrics["lidar_point_coverage"].value == 5.0
    assert metrics["lidar_spatial_coverage_m"].value is not None
    assert metrics["lidar_spatial_coverage_m"].value > 0.0


def test_rosbag1_inspect_missing_path(tmp_path: Path) -> None:
    inspection = inspect_dataset(DatasetConfig(type="rosbag1", path=str(tmp_path / "nope.bag")))
    assert not inspection.exists
    assert inspection.warnings == ["dataset path does not exist"]


def test_rosbag1_rejects_bad_magic(tmp_path: Path) -> None:
    bad = tmp_path / "bad.bag"
    bad.write_bytes(b"NOT A ROSBAG\n" + b"\x00" * 16)
    with pytest.raises(DatasetError, match="bad magic"):
        list(read_pointcloud2_messages(bad))


def test_rosbag1_decode_requires_xyz() -> None:
    # A PointCloud2 payload advertising only an intensity field must be rejected.
    buf = b""
    buf += struct.pack("<I", 0)  # seq
    buf += struct.pack("<I", 0)  # secs
    buf += struct.pack("<I", 0)  # nsecs
    buf += struct.pack("<I", 0)  # frame_id length
    buf += struct.pack("<I", 1)  # height
    buf += struct.pack("<I", 1)  # width
    buf += struct.pack("<I", 1)  # one field
    name = b"intensity"
    buf += struct.pack("<I", len(name)) + name
    buf += struct.pack("<I", 0)  # offset
    buf += struct.pack("<B", 7)  # FLOAT32
    buf += struct.pack("<I", 1)  # count
    buf += struct.pack("<B", 0)  # is_bigendian
    buf += struct.pack("<I", 4)  # point_step
    buf += struct.pack("<I", 4)  # row_step
    buf += struct.pack("<I", 4) + struct.pack("<f", 1.0)  # data
    buf += struct.pack("<B", 1)  # is_dense
    with pytest.raises(DatasetError, match="x/y/z"):
        decode_pointcloud2("/topic", 0, buf)


def test_rosbag1_unsupported_compression_errors() -> None:
    with pytest.raises(DatasetError, match="unsupported ROS bag chunk compression"):
        _decompress_chunk("zstd", 0, b"payload")


def test_rosbag1_lz4_compression_path() -> None:
    raw = b"calibrex rosbag lz4 payload"
    if importlib.util.find_spec("lz4") is None:
        with pytest.raises(DatasetError, match=r"calibrex\[rosbag1-lz4\]"):
            _decompress_chunk("lz4", len(raw), b"unused")
    else:  # pragma: no cover - depends on optional extra being installed
        import lz4.frame

        assert _decompress_chunk("lz4", len(raw), lz4.frame.compress(raw)) == raw
