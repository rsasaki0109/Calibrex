"""Unit tests for per-point PointCloud2 time field decoding."""

from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import pytest

from calibrex.core.exceptions import DatasetError
from calibrex.data.ros_cdr import decode_ros2_pointcloud2

_FIXTURES = importlib.util.spec_from_file_location(
    "rosbag2_test_fixtures",
    Path(__file__).with_name("test_rosbag2.py"),
)
assert _FIXTURES is not None and _FIXTURES.loader is not None
_rosbag2 = importlib.util.module_from_spec(_FIXTURES)
_FIXTURES.loader.exec_module(_rosbag2)
CdrWriter = _rosbag2.CdrWriter
_encode_pointcloud2 = _rosbag2._encode_pointcloud2


def _encode_pointcloud2_with_time_field(
    points: list[tuple[float, float, float, float]],
    time_values: list[float],
    *,
    time_field: str = "time",
    time_datatype: int = 7,
    time_offset: int = 16,
    point_step: int = 20,
) -> bytes:
    writer = CdrWriter(little_endian=True)
    writer.write_int32(1)
    writer.write_uint32(0)
    writer.write_string("frame")
    writer.write_uint32(1)
    writer.write_uint32(len(points))

    field_defs = [
        ("x", 0, 7),
        ("y", 4, 7),
        ("z", 8, 7),
        ("intensity", 12, 7),
        (time_field, time_offset, time_datatype),
    ]
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
    writer.write_uint32(point_step * len(points))

    payload = bytearray()
    for (x, y, z, intensity), time_value in zip(points, time_values, strict=True):
        payload.extend(struct.pack("<ffff", x, y, z, intensity))
        if time_datatype == 7:
            payload.extend(struct.pack("<f", time_value))
        elif time_datatype == 6:
            payload.extend(struct.pack("<I", int(time_value)))
        else:
            raise AssertionError(f"unsupported datatype {time_datatype}")
    writer.write_byte_sequence(bytes(payload))
    writer.write_bool(True)
    return writer.finish()


def test_decode_point_time_float32_seconds() -> None:
    pytest.importorskip("numpy")
    points = [(1.0, 2.0, 3.0, 0.5), (4.0, 5.0, 6.0, 0.25)]
    payload = _encode_pointcloud2_with_time_field(
        points,
        [-0.05, 0.05],
        time_field="time",
        time_datatype=7,
    )
    decoded = decode_ros2_pointcloud2("/cloud", 0, payload, point_time_field="time")
    assert decoded.point_time_offsets_s is not None
    assert decoded.point_time_offsets_s.tolist() == pytest.approx([-0.05, 0.05])


def test_decode_point_time_uint32_nanoseconds() -> None:
    pytest.importorskip("numpy")
    points = [(1.0, 2.0, 3.0, 0.5)]
    payload = _encode_pointcloud2_with_time_field(
        points,
        [50_000_000],
        time_field="t",
        time_datatype=6,
    )
    decoded = decode_ros2_pointcloud2("/cloud", 0, payload, point_time_field="t")
    assert decoded.point_time_offsets_s is not None
    assert decoded.point_time_offsets_s.tolist() == pytest.approx([0.05])


def test_decode_point_time_stays_aligned_after_nonfinite_xyz_filter() -> None:
    pytest.importorskip("numpy")
    payload = _encode_pointcloud2_with_time_field(
        [(float("nan"), 2.0, 3.0, 0.5), (4.0, 5.0, 6.0, 0.25)],
        [0.01, 0.02],
        time_field="time",
        time_datatype=7,
    )

    decoded = decode_ros2_pointcloud2("/cloud", 0, payload, point_time_field="time")

    assert decoded.point_count == 1
    assert decoded.point_time_offsets_s is not None
    assert decoded.point_time_offsets_s.tolist() == pytest.approx([0.02])


def test_decode_point_time_missing_field_raises() -> None:
    pytest.importorskip("numpy")
    payload = _encode_pointcloud2(
        [(1.0, 2.0, 3.0, 0.5)],
        frame_id="frame",
        secs=1,
        nsecs=0,
        with_intensity=True,
    )
    with pytest.raises(DatasetError, match="no per-point time field 'time'"):
        decode_ros2_pointcloud2("/velodyne_points", 0, payload, point_time_field="time")
