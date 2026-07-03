"""Synthetic rosbag2 online calibration tests with motion compensation."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

from calibrex.core.geometry import SE3
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
_encode_odometry = _rosbag2._encode_odometry
_encode_pointcloud2 = _rosbag2._encode_pointcloud2
_write_sqlite_bag = _rosbag2._write_sqlite_bag

_BASE_NS = 1_000_000_000
_STEP_NS = 100_000_000

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


def _world_to_sensor(
    point_world: tuple[float, float, float],
    t_world_sensor: SE3,
) -> tuple[float, float, float, float]:
    point_sensor = t_world_sensor.inverse().transform_point(point_world)
    return (point_sensor[0], point_sensor[1], point_sensor[2], 0.0)


def _build_moving_rig_bag(path: Path, *, message_count: int = 10) -> Path:
    world_points = _corner_world_points()
    topics = [
        ("/livox/lidar", POINTCLOUD2_TYPE),
        ("/avia/livox/lidar", POINTCLOUD2_TYPE),
        ("/odom", ODOMETRY_TYPE),
    ]
    messages: list[tuple[str, int, bytes]] = []
    for index in range(message_count):
        timestamp_ns = _BASE_NS + index * _STEP_NS
        t_world_base = _t_world_base(index)
        t_world_horizon = t_world_base
        t_world_avia = t_world_base.compose(_TRUE_T_BASE_AVIA)

        horizon_points = [
            _world_to_sensor(point, t_world_horizon) for point in world_points
        ]
        avia_points = [_world_to_sensor(point, t_world_avia) for point in world_points]

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
        messages.append(
            (
                "/livox/lidar",
                timestamp_ns + 10,
                _encode_pointcloud2(
                    horizon_points,
                    frame_id="livox_horizon",
                    secs=timestamp_ns // 1_000_000_000,
                    nsecs=(timestamp_ns + 10) % 1_000_000_000,
                    with_intensity=True,
                ),
            )
        )
        messages.append(
            (
                "/avia/livox/lidar",
                timestamp_ns + 20,
                _encode_pointcloud2(
                    avia_points,
                    frame_id="livox_avia",
                    secs=timestamp_ns // 1_000_000_000,
                    nsecs=(timestamp_ns + 20) % 1_000_000_000,
                    with_intensity=True,
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
    odometry_topic: str | None,
) -> None:
    odometry_line = f"  odometry_topic: {odometry_topic}" if odometry_topic else ""
    config_path.write_text(
        f"""
schema_version: calibrex.config/v0.1
project:
  name: rosbag2_motion_fixture
  output_dir: {output_dir}
dataset:
  type: rosbag2
  path: {bag_path}
{odometry_line}
sensors:
  lidar_map:
    type: lidar
    topic: /livox/lidar
  lidar_stream:
    type: lidar
    topic: /avia/livox/lidar
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


def _run_session(
    tmp_path: Path,
    *,
    odometry_topic: str | None,
) -> tuple[SE3, dict[str, object]]:
    bag = _build_moving_rig_bag(tmp_path / "moving_rig.db3")
    output_dir = tmp_path / "outputs"
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        bag_path=bag,
        output_dir=output_dir,
        odometry_topic=odometry_topic,
    )
    result = run_online_calibration(
        config_path,
        OnlineCalibrationRunOptions(batch_size=200, rolling_window=400, holdout_ratio=0.2),
    )
    assert result is not None
    return result.transforms["T_base_link_lidar_stream"].as_se3(), result.run.provenance


def test_rosbag2_online_motion_compensation_recovers_ground_truth(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    estimate, provenance = _run_session(tmp_path, odometry_topic="/odom")
    truth = _ground_truth_source_target()

    trans_error = _translation_error_m(estimate, truth)
    rot_error = _rotation_error_deg(estimate, truth)
    assert provenance["motion_compensated"] is True
    assert provenance["odometry_topic"] == "/odom"
    assert trans_error < 0.01
    assert rot_error < 0.5
    assert provenance["online_final_gate_status"] == "pass"


def test_rosbag2_online_without_odometry_is_worse_on_moving_rig(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    estimate, provenance = _run_session(tmp_path, odometry_topic=None)
    truth = _ground_truth_source_target()

    trans_error = _translation_error_m(estimate, truth)
    rot_error = _rotation_error_deg(estimate, truth)
    assert provenance["motion_compensated"] is False
    worse = (
        trans_error > 0.05
        or rot_error > 2.0
        or provenance["online_final_gate_status"] != "pass"
    )
    assert worse


def test_rosbag2_online_replay_provenance_includes_odometry_fields(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    _estimate, provenance = _run_session(tmp_path, odometry_topic="/odom")
    assert provenance["odometry_message_count"] >= 1
    assert provenance["odometry_interpolation_method"] == "linear_translation_slerp_rotation"
    assert "odometry_interpolation_clamp_count" in provenance
    timeline_path = Path(str(provenance["online_timeline_path"]))
    assert timeline_path.exists()
