"""Synthetic rosbag2 online calibration tests with per-point deskew."""

from __future__ import annotations

import importlib.util
import math
import struct
from pathlib import Path

import pytest

from calibrex.core.geometry import SE3, interpolate_se3
from calibrex.pipelines.online import OnlineCalibrationRunOptions, run_online_calibration

_FIXTURES = importlib.util.spec_from_file_location(
    "rosbag2_test_fixtures",
    Path(__file__).with_name("test_rosbag2.py"),
)
assert _FIXTURES is not None and _FIXTURES.loader is not None
_rosbag2 = importlib.util.module_from_spec(_FIXTURES)
_FIXTURES.loader.exec_module(_rosbag2)

POINTCLOUD2_TYPE = _rosbag2.POINTCLOUD2_TYPE
ODOMETRY_TYPE = _rosbag2.ODOMETRY_TYPE
CdrWriter = _rosbag2.CdrWriter
_encode_odometry = _rosbag2._encode_odometry
_write_sqlite_bag = _rosbag2._write_sqlite_bag

_BASE_NS = 1_000_000_000
_STEP_NS = 100_000_000
_SWEEP_S = 0.08

_TRUE_T_BASE_AVIA = SE3((0.12, -0.05, 0.03), (0.0, 0.0, 0.0871557, 0.9961947))
_WRONG_T_BASE_AVIA = SE3((0.20, 0.08, 0.06), (0.0, 0.0, 0.0, 1.0))


def _corner_world_points(
    *, step: float = 0.2, span: float = 1.0
) -> list[tuple[float, float, float]]:
    coords = [round(index * step, 6) for index in range(int(span / step) + 1)]
    points: list[tuple[float, float, float]] = []
    for a in coords:
        for b in coords:
            points.append((a, b, 0.0))
            points.append((3.0, a, b))
            points.append((a, -3.0, b))
    return points


def _yaw_quaternion(yaw_rad: float) -> tuple[float, float, float, float]:
    half = yaw_rad * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _t_world_base(index: int) -> SE3:
    yaw = 0.12 * index
    return SE3((0.5 * index, 0.0, 0.0), _yaw_quaternion(yaw))


def _world_base_at_capture(message_index: int, offset_s: float) -> SE3:
    t_start = _t_world_base(message_index)
    t_end = _t_world_base(message_index + 1)
    alpha = min(1.0, offset_s / (_STEP_NS / 1_000_000_000))
    return interpolate_se3(t_start, t_end, alpha)


def _world_to_sensor(
    point_world: tuple[float, float, float],
    t_world_sensor: SE3,
) -> tuple[float, float, float, float]:
    point_sensor = t_world_sensor.inverse().transform_point(point_world)
    return (point_sensor[0], point_sensor[1], point_sensor[2], 0.0)


def _encode_pointcloud2_with_time(
    points: list[tuple[float, float, float, float]],
    time_values: list[float],
    *,
    secs: int,
    nsecs: int,
) -> bytes:
    writer = CdrWriter(little_endian=True)
    writer.write_int32(secs)
    writer.write_uint32(nsecs)
    writer.write_string("frame")
    writer.write_uint32(1)
    writer.write_uint32(len(points))

    field_defs = [("x", 0, 7), ("y", 4, 7), ("z", 8, 7), ("intensity", 12, 7), ("time", 16, 7)]
    point_step = 20
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
        payload.extend(struct.pack("<f", time_value))
    writer.write_byte_sequence(bytes(payload))
    writer.write_bool(True)
    return writer.finish()


def _build_deskew_sweep_bag(path: Path, *, message_count: int = 10) -> Path:
    world_points = _corner_world_points()
    topics = [
        ("/livox/lidar", POINTCLOUD2_TYPE),
        ("/avia/livox/lidar", POINTCLOUD2_TYPE),
        ("/odom", ODOMETRY_TYPE),
    ]
    messages: list[tuple[str, int, bytes]] = []
    for index in range(message_count + 1):
        timestamp_ns = _BASE_NS + index * _STEP_NS
        t_world_base = _t_world_base(index)
        messages.append(
            (
                "/odom",
                timestamp_ns,
                _encode_odometry(
                    frame_id="odom",
                    child_frame_id="base_link",
                    secs=timestamp_ns // 1_000_000_000,
                    nsecs=timestamp_ns % 1_000_000_000,
                    position=t_world_base.translation_m,
                    orientation_xyzw=t_world_base.rotation_quat_xyzw,
                ),
            )
        )

    for index in range(message_count):
        timestamp_ns = _BASE_NS + index * _STEP_NS
        point_count = len(world_points)
        time_values = [
            (point_index / max(point_count - 1, 1)) * _SWEEP_S
            for point_index in range(point_count)
        ]
        horizon_points = [
            _world_to_sensor(point, _world_base_at_capture(index, offset_s))
            for point, offset_s in zip(world_points, time_values, strict=True)
        ]
        avia_points = [
            _world_to_sensor(
                point,
                _world_base_at_capture(index, offset_s).compose(_TRUE_T_BASE_AVIA),
            )
            for point, offset_s in zip(world_points, time_values, strict=True)
        ]

        source_stamp_ns = timestamp_ns + 10
        target_stamp_ns = timestamp_ns + 20
        messages.append(
            (
                "/livox/lidar",
                source_stamp_ns,
                _encode_pointcloud2_with_time(
                    horizon_points,
                    time_values,
                    secs=source_stamp_ns // 1_000_000_000,
                    nsecs=source_stamp_ns % 1_000_000_000,
                ),
            )
        )
        messages.append(
            (
                "/avia/livox/lidar",
                target_stamp_ns,
                _encode_pointcloud2_with_time(
                    avia_points,
                    time_values,
                    secs=target_stamp_ns // 1_000_000_000,
                    nsecs=target_stamp_ns % 1_000_000_000,
                ),
            )
        )
    _write_sqlite_bag(path, topics=topics, messages=messages)
    return path


def _write_config(
    config_path: Path,
    *,
    bag_path: Path,
    output_dir: Path,
    deskew: bool,
) -> None:
    deskew_lines = ""
    if deskew:
        deskew_lines = """
    point_time_field: time"""
    config_path.write_text(
        f"""
schema_version: slac.config/v0.1
project:
  name: rosbag2_deskew_fixture
  output_dir: {output_dir}
dataset:
  type: rosbag2
  path: {bag_path}
  odometry_topic: /odom
sensors:
  lidar_map:
    type: lidar
    topic: /livox/lidar{deskew_lines}
  lidar_stream:
    type: lidar
    topic: /avia/livox/lidar{deskew_lines}
frames:
  base_link:
    root: true
  lidar_map:
    parent: base_link
    transform:
      estimate: false
      initial:
        translation: [0.0, 0.0, 0.0]
        rotation_quat_xyzw: [0.0, 0.0, 0.0, 1.0]
  lidar_stream:
    parent: base_link
    transform:
      estimate: true
      initial:
        translation: {_list(_WRONG_T_BASE_AVIA.translation_m)}
        rotation_quat_xyzw: {_list(_WRONG_T_BASE_AVIA.rotation_quat_xyzw)}
pipeline:
  type: multi_sensor_slac
  factors:
    lidar_rig_point_to_plane:
      enabled: true
      options:
        voxel_size_m: 0.3
        correspondence_gate_m: 0.8
        max_source_points: 1500
        max_source_messages: 8
        max_target_points: 150
        max_target_messages: 8
        max_replay_duration_s: 5.0
solver:
  max_iterations: 30
""",
        encoding="utf-8",
    )


def _list(values: tuple[float, float, float] | tuple[float, float, float, float]) -> str:
    return "[" + ", ".join(f"{value:.8f}" for value in values) + "]"


def _ground_truth_source_target() -> SE3:
    return _TRUE_T_BASE_AVIA


def _rotation_error_deg(estimate: SE3, truth: SE3) -> float:
    delta = truth.inverse().compose(estimate)
    _x, _y, _z, w = delta.rotation_quat_xyzw
    return math.degrees(2.0 * math.acos(min(1.0, abs(w))))


def _translation_error_m(estimate: SE3, truth: SE3) -> float:
    delta = truth.inverse().compose(estimate)
    return math.dist((0.0, 0.0, 0.0), delta.translation_m)


def _run_session(tmp_path: Path, *, deskew: bool) -> tuple[SE3, dict[str, object]]:
    bag = _build_deskew_sweep_bag(tmp_path / "deskew_sweep.db3")
    output_dir = tmp_path / ("outputs_deskew" if deskew else "outputs_no_deskew")
    config_path = tmp_path / ("config_deskew.yaml" if deskew else "config_no_deskew.yaml")
    _write_config(config_path, bag_path=bag, output_dir=output_dir, deskew=deskew)
    result = run_online_calibration(
        config_path,
        OnlineCalibrationRunOptions(batch_size=200, rolling_window=400, holdout_ratio=0.2),
    )
    assert result is not None
    return result.transforms["T_base_link_lidar_stream"].as_se3(), result.run.provenance


def test_rosbag2_online_deskew_recovers_ground_truth(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    estimate, provenance = _run_session(tmp_path, deskew=True)
    truth = _ground_truth_source_target()

    trans_error = _translation_error_m(estimate, truth)
    rot_error = _rotation_error_deg(estimate, truth)
    assert provenance["deskew_applied"] is True
    assert provenance["deskew_time_field"] == {
        "lidar_map": "time",
        "lidar_stream": "time",
    }
    assert provenance["deskew_span_s"] == pytest.approx(_SWEEP_S, rel=0.01)
    assert trans_error < 0.02
    assert rot_error < 1.0


def test_rosbag2_online_motion_without_deskew_is_worse_on_sweeping_rig(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    estimate, provenance = _run_session(tmp_path, deskew=False)
    truth = _ground_truth_source_target()

    trans_error = _translation_error_m(estimate, truth)
    rot_error = _rotation_error_deg(estimate, truth)
    assert provenance["deskew_applied"] is False
    assert trans_error > 0.04 or rot_error > 0.15
