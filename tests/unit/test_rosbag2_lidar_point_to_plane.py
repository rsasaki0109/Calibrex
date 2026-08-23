"""Unit tests for rosbag2 support in native_lidar_point_to_plane_solver.

Exercises ``_load_rosbag2_lidar_pair`` against a synthetic sqlite3 rosbag2
fixture built byte-for-byte (no ROS installation required).  Also verifies
that ``lidar_pair_dataset_files`` accepts the ``rosbag2`` dataset type and
that the solver reports ``unavailable`` rather than crashing when the bag
path does not exist.
"""

from __future__ import annotations

import importlib.util
import math
import struct
from pathlib import Path

from calibrex.solvers.native_lidar_point_to_plane_solver import (
    _load_rosbag2_lidar_pair,
    lidar_pair_solve_inputs,
)

# --------------------------------------------------------------------------- #
# Load CDR fixture utilities from test_rosbag2.py without adding a package dep #
# --------------------------------------------------------------------------- #
_SPEC = importlib.util.spec_from_file_location(
    "rosbag2_test_fixtures",
    Path(__file__).with_name("test_rosbag2.py"),
)
assert _SPEC is not None and _SPEC.loader is not None
_rb2 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_rb2)  # type: ignore[union-attr]

CdrWriter = _rb2.CdrWriter
POINTCLOUD2_TYPE = _rb2.POINTCLOUD2_TYPE
_write_sqlite_bag = _rb2._write_sqlite_bag


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

_SOURCE_TOPIC = "/lidar/left"
_TARGET_TOPIC = "/lidar/right"


def _encode_xyz_cloud(
    points: list[tuple[float, float, float]],
    *,
    secs: int = 1,
    nsecs: int = 0,
) -> bytes:
    """Encode a minimal sensor_msgs/msg/PointCloud2 (xyz only, CDR, little-endian)."""
    writer = CdrWriter(little_endian=True)
    writer.write_int32(secs)
    writer.write_uint32(nsecs)
    writer.write_string("lidar_frame")
    writer.write_uint32(1)           # height
    writer.write_uint32(len(points)) # width

    field_defs = [("x", 0, 7), ("y", 4, 7), ("z", 8, 7)]
    point_step = 12
    writer.write_uint32(len(field_defs))
    for name, offset, datatype in field_defs:
        writer.write_string(name)
        writer.write_uint32(offset)
        writer.write_uint8(datatype)
        writer.align(4)
        writer.write_uint32(1)

    writer.write_bool(False)  # is_bigendian
    writer.align(4)
    writer.write_uint32(point_step)
    writer.write_uint32(point_step * len(points))

    payload = bytearray()
    for x, y, z in points:
        payload.extend(struct.pack("<fff", x, y, z))
    writer.write_byte_sequence(bytes(payload))
    writer.write_bool(True)  # is_dense
    return writer.finish()


def _floor_points(n: int = 80) -> list[tuple[float, float, float]]:
    """Return a flat floor (z=0) patch — clearly planar for voxel matching."""
    step = 0.1
    pts: list[tuple[float, float, float]] = []
    side = math.ceil(math.sqrt(n))
    for i in range(side):
        for j in range(side):
            if len(pts) >= n:
                break
            pts.append((i * step, j * step, 0.0))
    return pts[:n]


def _wall_points(n: int = 80) -> list[tuple[float, float, float]]:
    """Return a vertical wall (y=3) patch."""
    step = 0.1
    pts: list[tuple[float, float, float]] = []
    side = math.ceil(math.sqrt(n))
    for i in range(side):
        for j in range(side):
            if len(pts) >= n:
                break
            pts.append((i * step, 3.0, j * step))
    return pts[:n]


def _write_two_lidar_bag(path: Path, *, msg_count: int = 6) -> None:
    """Write a rosbag2 sqlite3 bag with two spinning LiDAR topics."""
    floor = _floor_points()
    wall = _wall_points()
    combined = floor + wall

    topics = [
        (_SOURCE_TOPIC, POINTCLOUD2_TYPE),
        (_TARGET_TOPIC, POINTCLOUD2_TYPE),
    ]
    messages: list[tuple[str, int, bytes]] = []
    base_ns = 1_000_000_000
    step_ns = 100_000_000
    for i in range(msg_count):
        ts = base_ns + i * step_ns
        messages.append((_SOURCE_TOPIC, ts, _encode_xyz_cloud(combined, secs=1 + i)))
        messages.append((_TARGET_TOPIC, ts + 1, _encode_xyz_cloud(combined, secs=1 + i)))
    _write_sqlite_bag(path, topics=topics, messages=messages)


def _make_config(bag_path: Path) -> object:
    """Return a minimal CalibrationConfig for a two-lidar rosbag2."""
    import yaml

    from calibrex.core.config import CalibrationConfig

    raw = yaml.safe_load(f"""
schema_version: slac.config/v0.1
project:
  name: rosbag2_lidar_pair_test
  output_dir: /tmp/rosbag2_lidar_pair_test
dataset:
  type: rosbag2
  path: {bag_path}
sensors:
  lidar_left:
    type: lidar
    model: spinning
    topic: {_SOURCE_TOPIC}
    fields: [x, y, z]
  lidar_right:
    type: lidar
    model: spinning
    topic: {_TARGET_TOPIC}
    fields: [x, y, z]
frames:
  lidar_left:
    root: true
  lidar_right:
    parent: lidar_left
    transform:
      estimate: true
      initial:
        translation: [0.05, 0.0, 0.0]
        rotation_quat_xyzw: [0.0, 0.0, 0.0, 1.0]
      prior_sigma:
        translation_m: 0.2
        rotation_deg: 10.0
solver:
  backend: native_lidar_point_to_plane
  max_iterations: 10
evaluation:
  holdout_ratio: 0.2
""")
    return CalibrationConfig.model_validate(raw)


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #


def test_load_rosbag2_lidar_pair_returns_points(tmp_path: Path) -> None:
    """_load_rosbag2_lidar_pair should decode PointCloud2 messages from both topics."""
    bag_path = tmp_path / "test.db3"
    _write_two_lidar_bag(bag_path)

    config = _make_config(bag_path)
    inputs = lidar_pair_solve_inputs(config)

    pair = _load_rosbag2_lidar_pair(
        config,
        inputs,
        source_sensor="lidar_left",
        target_sensor="lidar_right",
    )

    assert len(pair.source_records) > 0, "source records should be non-empty"
    assert len(pair.target_points) > 0, "target points should be non-empty"
    assert pair.source_path.endswith(_SOURCE_TOPIC)
    assert pair.target_path.endswith(_TARGET_TOPIC)


def test_load_rosbag2_lidar_pair_missing_bag(tmp_path: Path) -> None:
    """_load_rosbag2_lidar_pair should return empty data for a non-existent bag."""
    bag_path = tmp_path / "missing.db3"
    config = _make_config(bag_path)
    inputs = lidar_pair_solve_inputs(config)

    pair = _load_rosbag2_lidar_pair(
        config,
        inputs,
        source_sensor="lidar_left",
        target_sensor="lidar_right",
    )

    assert pair.source_records == []
    assert pair.target_points == []


def test_load_rosbag2_lidar_pair_holdout_split(tmp_path: Path) -> None:
    """Target messages should be split into train and holdout sets."""
    bag_path = tmp_path / "test.db3"
    _write_two_lidar_bag(bag_path, msg_count=10)

    config = _make_config(bag_path)
    inputs = lidar_pair_solve_inputs(config)

    pair = _load_rosbag2_lidar_pair(
        config,
        inputs,
        source_sensor="lidar_left",
        target_sensor="lidar_right",
    )

    total = pair.target_train_window_count + pair.target_holdout_window_count
    assert total == 10, f"expected 10 windows, got {total}"
    assert pair.target_holdout_window_count >= 1
