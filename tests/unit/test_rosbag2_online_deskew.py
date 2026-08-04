"""Synthetic rosbag2 online calibration tests with per-point deskew."""

from __future__ import annotations

import importlib.util
import math
import struct
from pathlib import Path

import pytest

from calibrex.core.geometry import SE3, interpolate_se3
from calibrex.pipelines.online import (
    OnlineCalibrationRunOptions,
    _select_record_timestamps_per_capture_window,
    run_online_calibration,
)

_FIXTURES = importlib.util.spec_from_file_location(
    "rosbag2_test_fixtures",
    Path(__file__).with_name("test_rosbag2.py"),
)
assert _FIXTURES is not None and _FIXTURES.loader is not None
_rosbag2 = importlib.util.module_from_spec(_FIXTURES)
_FIXTURES.loader.exec_module(_rosbag2)

POINTCLOUD2_TYPE = _rosbag2.POINTCLOUD2_TYPE
LIVOX_CUSTOMMSG_TYPE = _rosbag2.LIVOX_CUSTOMMSG_TYPE
ODOMETRY_TYPE = _rosbag2.ODOMETRY_TYPE
CdrWriter = _rosbag2.CdrWriter
_encode_odometry = _rosbag2._encode_odometry
_encode_livox_custommsg = _rosbag2._encode_livox_custommsg
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


def _build_deskew_sweep_bag(
    path: Path,
    *,
    message_count: int = 10,
    source_custommsg: bool = False,
) -> Path:
    world_points = _corner_world_points()
    topics = [
        (
            "/livox/lidar",
            LIVOX_CUSTOMMSG_TYPE if source_custommsg else POINTCLOUD2_TYPE,
        ),
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
        if source_custommsg:
            source_payload = _encode_livox_custommsg(
                [
                    (
                        round(offset_s * 1_000_000_000),
                        point[0],
                        point[1],
                        point[2],
                        0,
                        0,
                        point_index % 4,
                    )
                    for point_index, (point, offset_s) in enumerate(
                        zip(horizon_points, time_values, strict=True)
                    )
                ],
                frame_id="frame",
                secs=source_stamp_ns // 1_000_000_000,
                nsecs=source_stamp_ns % 1_000_000_000,
                timebase_ns=source_stamp_ns,
            )
        else:
            source_payload = _encode_pointcloud2_with_time(
                horizon_points,
                time_values,
                secs=source_stamp_ns // 1_000_000_000,
                nsecs=source_stamp_ns % 1_000_000_000,
            )
        messages.append(("/livox/lidar", source_stamp_ns, source_payload))
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
    source_custommsg: bool = False,
    capture_window_start_ns: int | None = None,
    capture_window_end_ns: int | None = None,
    capture_window_prefilter_margin_s: float | None = None,
    capture_windows: list[tuple[int, int]] | None = None,
) -> None:
    source_deskew_lines = ""
    target_deskew_lines = ""
    if deskew:
        source_time_field = "offset_time" if source_custommsg else "time"
        source_deskew_lines = f"""
    point_time_field: {source_time_field}"""
        target_deskew_lines = """
    point_time_field: time"""
    capture_window_lines = ""
    if capture_window_start_ns is not None:
        capture_window_lines += (
            f"\n        capture_window_start_timestamp_ns: {capture_window_start_ns}"
        )
    if capture_window_end_ns is not None:
        capture_window_lines += (
            f"\n        capture_window_end_timestamp_ns: {capture_window_end_ns}"
        )
    if capture_window_prefilter_margin_s is not None:
        capture_window_lines += (
            "\n        capture_window_record_prefilter_margin_s: "
            f"{capture_window_prefilter_margin_s}"
        )
    if capture_windows is not None:
        capture_window_lines += "\n        capture_windows:"
        for start_ns, end_ns in capture_windows:
            capture_window_lines += (
                f"\n          - start_timestamp_ns: {start_ns}"
                f"\n            end_timestamp_ns: {end_ns}"
            )
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
    topic: /livox/lidar{source_deskew_lines}
  lidar_stream:
    type: lidar
    topic: /avia/livox/lidar{target_deskew_lines}
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
        max_replay_duration_s: 5.0{capture_window_lines}
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


def _run_session(
    tmp_path: Path,
    *,
    deskew: bool,
    source_custommsg: bool = False,
    capture_window_start_ns: int | None = None,
    capture_window_end_ns: int | None = None,
    capture_window_prefilter_margin_s: float | None = None,
    capture_windows: list[tuple[int, int]] | None = None,
) -> tuple[SE3, dict[str, object]]:
    bag = _build_deskew_sweep_bag(
        tmp_path / "deskew_sweep.db3",
        source_custommsg=source_custommsg,
    )
    output_dir = tmp_path / ("outputs_deskew" if deskew else "outputs_no_deskew")
    config_path = tmp_path / ("config_deskew.yaml" if deskew else "config_no_deskew.yaml")
    _write_config(
        config_path,
        bag_path=bag,
        output_dir=output_dir,
        deskew=deskew,
        source_custommsg=source_custommsg,
        capture_window_start_ns=capture_window_start_ns,
        capture_window_end_ns=capture_window_end_ns,
        capture_window_prefilter_margin_s=capture_window_prefilter_margin_s,
        capture_windows=capture_windows,
    )
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


def test_rosbag2_online_livox_custommsg_deskew_recovers_ground_truth(
    tmp_path: Path,
) -> None:
    pytest.importorskip("numpy")
    estimate, provenance = _run_session(
        tmp_path,
        deskew=True,
        source_custommsg=True,
    )
    truth = _ground_truth_source_target()

    assert provenance["rosbag2_source_message_type"] == LIVOX_CUSTOMMSG_TYPE
    assert provenance["point_time_observed_by_sensor"]["lidar_map"] is True
    assert provenance["point_time_reference_by_sensor"]["lidar_map"] == (
        "timebase_native_ros_epoch"
    )
    assert provenance["deskew_applied"] is True
    # CustomMsg carries integer-nanosecond offsets; the fixture's conversion
    # therefore has a small quantization difference from PointCloud2 float time.
    assert _translation_error_m(estimate, truth) < 0.03
    assert _rotation_error_deg(estimate, truth) < 1.0
    assert provenance["rosbag2_temporal_holdout_independent"] is True
    trajectory = provenance["trajectory_evidence"]
    assert isinstance(trajectory, dict)
    assert trajectory["cross_segment_first_half_scan_count"] > 0
    assert trajectory["cross_segment_second_half_scan_count"] > 0


def test_rosbag2_online_capture_window_filters_capture_clock(
    tmp_path: Path,
) -> None:
    pytest.importorskip("numpy")
    start_ns = _BASE_NS + 2 * _STEP_NS
    end_ns = _BASE_NS + 7 * _STEP_NS
    _estimate, provenance = _run_session(
        tmp_path,
        deskew=True,
        capture_window_start_ns=start_ns,
        capture_window_end_ns=end_ns,
        capture_window_prefilter_margin_s=0.5,
    )

    assert provenance["rosbag2_capture_window_start_timestamp_ns"] == start_ns
    assert provenance["rosbag2_capture_window_end_timestamp_ns"] == end_ns
    assert provenance["rosbag2_capture_window_record_prefilter_margin_s"] == 0.5
    assert provenance["rosbag2_first_source_timestamp_ns"] >= start_ns
    assert provenance["rosbag2_first_target_timestamp_ns"] >= start_ns
    assert provenance["rosbag2_replay_end_timestamp_ns"] <= end_ns
    assert provenance["rosbag2_source_message_count"] > 0
    assert provenance["rosbag2_target_message_count"] > 0
    trajectory = provenance["trajectory_evidence"]
    assert isinstance(trajectory, dict)
    assert trajectory["cross_segment_first_half_start_timestamp_ns"] >= start_ns
    assert trajectory["cross_segment_second_half_end_timestamp_ns"] <= end_ns


def test_rosbag2_online_multi_capture_windows_replay_union_with_provenance(
    tmp_path: Path,
) -> None:
    pytest.importorskip("numpy")
    windows = [
        (_BASE_NS + _STEP_NS, _BASE_NS + 3 * _STEP_NS + 50),
        (6 * _STEP_NS + _BASE_NS, 8 * _STEP_NS + _BASE_NS + 50),
    ]
    _estimate, provenance = _run_session(
        tmp_path,
        deskew=True,
        capture_windows=windows,
    )

    assert provenance["rosbag2_capture_window_count"] == 2
    capture_windows = provenance["rosbag2_capture_windows"]
    assert isinstance(capture_windows, list)
    assert [
        (item["start_timestamp_ns"], item["end_timestamp_ns"])
        for item in capture_windows
    ] == windows
    assert [item["selected_target_message_count"] for item in capture_windows] == [3, 3]
    assert all(item["selected_source_message_count"] > 0 for item in capture_windows)
    assert provenance["rosbag2_target_message_count"] == 6
    trajectory = provenance["trajectory_evidence"]
    assert isinstance(trajectory, dict)
    assert trajectory["cross_segment_first_half_scan_count"] > 0
    assert trajectory["cross_segment_second_half_scan_count"] > 0


def test_capture_window_even_sampling_uses_capture_not_record_clock() -> None:
    selected = _select_record_timestamps_per_capture_window(
        [(100, 1_000), (200, 1_100), (300, 1_200), (400, 1_300)],
        ((1_050, 1_250),),
        max_samples_per_window=2,
    )

    assert selected == {200, 300}
