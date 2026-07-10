"""Trajectory evidence tests for motion-compensated online calibration."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import jsonschema
import pytest

from calibrex.core.geometry import SE3
from calibrex.core.trajectory import trajectory_json_schema
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
    return SE3((0.2 * index, 0.0, 0.0), _yaw_quaternion(yaw))


def _world_to_sensor(
    point_world: tuple[float, float, float],
    t_world_sensor: SE3,
) -> tuple[float, float, float, float]:
    point_sensor = t_world_sensor.inverse().transform_point(point_world)
    return (point_sensor[0], point_sensor[1], point_sensor[2], 0.0)


def _build_moving_rig_bag(
    path: Path,
    *,
    message_count: int = 10,
    teleport_index: int | None = None,
    second_half_offset_m: float = 0.0,
) -> Path:
    world_points = _corner_world_points()
    topics = [
        ("/livox/lidar", POINTCLOUD2_TYPE),
        ("/avia/livox/lidar", POINTCLOUD2_TYPE),
        ("/odom", ODOMETRY_TYPE),
    ]
    messages: list[tuple[str, int, bytes]] = []
    midpoint = message_count // 2
    for index in range(message_count):
        timestamp_ns = _BASE_NS + index * _STEP_NS
        t_world_base_true = _t_world_base(index)
        t_world_base_odom = t_world_base_true
        if teleport_index is not None and index == teleport_index:
            t_world_base_odom = SE3((20.0, 0.0, 0.0), t_world_base_odom.rotation_quat_xyzw)
        if second_half_offset_m != 0.0 and index > midpoint:
            translation = t_world_base_odom.translation_m
            t_world_base_odom = SE3(
                (
                    translation[0],
                    translation[1],
                    translation[2] + second_half_offset_m,
                ),
                t_world_base_odom.rotation_quat_xyzw,
            )
        t_world_horizon = t_world_base_true
        t_world_avia = t_world_base_true.compose(_TRUE_T_BASE_AVIA)

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
                    position=t_world_base_odom.translation_m,
                    orientation_xyzw=t_world_base_odom.rotation_quat_xyzw,
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
    trajectory_options: str = "",
    max_source_messages: int = 8,
    max_source_points: int = 1500,
    max_target_messages: int = 8,
) -> None:
    odometry_line = f"  odometry_topic: {odometry_topic}" if odometry_topic else ""
    wrong_translation = _WRONG_T_BASE_AVIA.translation_m
    wrong_rotation = _WRONG_T_BASE_AVIA.rotation_quat_xyzw
    initial_translation = (
        f"[{wrong_translation[0]:.8f}, {wrong_translation[1]:.8f}, "
        f"{wrong_translation[2]:.8f}]"
    )
    initial_rotation = (
        f"[{wrong_rotation[0]:.8f}, {wrong_rotation[1]:.8f}, "
        f"{wrong_rotation[2]:.8f}, {wrong_rotation[3]:.8f}]"
    )
    config_path.write_text(
        f"""
schema_version: slac.config/v0.1
project:
  name: rosbag2_trajectory_fixture
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
        translation: {initial_translation}
        rotation_quat_xyzw: {initial_rotation}
pipeline:
  type: multi_sensor_slac
  factors:
    lidar_rig_point_to_plane:
      enabled: true
      options:
        voxel_size_m: 0.3
        correspondence_gate_m: 0.8
        max_source_points: {max_source_points}
        max_source_messages: {max_source_messages}
        max_target_points: 150
        max_target_messages: {max_target_messages}
        max_replay_duration_s: 5.0
{trajectory_options}
solver:
  max_iterations: 30
""",
        encoding="utf-8",
    )


def _run_trajectory_session(
    tmp_path: Path,
    *,
    bag_path: Path,
    odometry_topic: str | None = "/odom",
    trajectory_options: str = "",
    max_source_messages: int = 8,
    max_source_points: int = 1500,
    max_target_messages: int = 8,
) -> object:
    output_dir = tmp_path / "outputs"
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        bag_path=bag_path,
        output_dir=output_dir,
        odometry_topic=odometry_topic,
        trajectory_options=trajectory_options,
        max_source_messages=max_source_messages,
        max_source_points=max_source_points,
        max_target_messages=max_target_messages,
    )
    result = run_online_calibration(
        config_path,
        OnlineCalibrationRunOptions(batch_size=200, rolling_window=400, holdout_ratio=0.2),
    )
    assert result is not None
    return result


def test_rosbag2_online_motion_emits_schema_valid_trajectory(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    bag = _build_moving_rig_bag(tmp_path / "smooth.db3", message_count=12)
    result = _run_trajectory_session(tmp_path, bag_path=bag)
    provenance = result.run.provenance

    trajectory_path = Path(str(provenance["trajectory_path"]))
    assert trajectory_path.exists()
    payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
    jsonschema.validate(payload, trajectory_json_schema())
    assert payload["schema_version"] == "slac.trajectory/v0.1"
    assert payload["pose_count_total"] >= payload["pose_count_recorded"]
    assert payload["pose_count_recorded"] <= payload["subsampling"]["max_pose_samples"]
    assert payload["interpolation_health"]["interpolation_count"] >= 1
    assert (
        payload["interpolation_health"]["clamp_count"]
        == provenance["odometry_interpolation_clamp_count"]
    )
    assert "trajectory_evidence" in provenance
    assert result.metrics["trajectory_gate_verdict"].grade == "pass"


def test_rosbag2_online_trajectory_kinematic_teleport_fails(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    bag = _build_moving_rig_bag(
        tmp_path / "teleport.db3",
        message_count=12,
        teleport_index=6,
    )
    result = _run_trajectory_session(tmp_path, bag_path=bag)
    evidence = result.run.provenance["trajectory_evidence"]

    assert evidence["kinematic_gate_status"] == "fail"
    assert evidence["verdict"] == "fail"
    assert result.metrics["trajectory_gate_verdict"].grade == "fail"


def test_rosbag2_online_trajectory_cross_segment_offset_fails(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    bag = _build_moving_rig_bag(
        tmp_path / "drift.db3",
        message_count=16,
        second_half_offset_m=0.5,
    )
    relaxed_kinematic = """
        trajectory_gate_max_speed_mps: 20.0
        trajectory_gate_max_angular_speed_dps: 500.0
"""
    result = _run_trajectory_session(
        tmp_path,
        bag_path=bag,
        trajectory_options=relaxed_kinematic,
        max_source_messages=16,
        max_source_points=12000,
        max_target_messages=16,
    )
    evidence = result.run.provenance["trajectory_evidence"]

    assert evidence["cross_segment_gate_status"] == "fail"
    assert evidence["cross_segment_rmse_m"] is not None
    assert float(evidence["cross_segment_rmse_m"]) > 0.30
    assert result.metrics["trajectory_gate_verdict"].grade == "fail"


def test_rosbag2_online_trajectory_smooth_passes(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    bag = _build_moving_rig_bag(tmp_path / "pass.db3", message_count=12)
    result = _run_trajectory_session(tmp_path, bag_path=bag)
    evidence = result.run.provenance["trajectory_evidence"]

    assert evidence["kinematic_gate_status"] == "pass"
    assert evidence["interpolation_gate_status"] == "pass"
    assert evidence["cross_segment_gate_status"] == "pass"
    assert evidence["verdict"] == "pass"
    assert result.metrics["trajectory_gate_verdict"].grade == "pass"


def test_rosbag2_online_trajectory_cross_segment_samples_beyond_replay_budget(
    tmp_path: Path,
) -> None:
    """Cross-segment streams the full track span, not the calibration replay budget."""
    pytest.importorskip("numpy")
    bag = _build_moving_rig_bag(tmp_path / "long_span.db3", message_count=40)
    result = _run_trajectory_session(
        tmp_path,
        bag_path=bag,
        max_source_messages=6,
        max_source_points=500,
        max_target_messages=6,
    )
    evidence = result.run.provenance["trajectory_evidence"]

    first_end_ns = evidence["cross_segment_first_half_end_timestamp_ns"]
    second_start_ns = evidence["cross_segment_second_half_start_timestamp_ns"]
    assert first_end_ns is not None
    assert second_start_ns is not None
    assert second_start_ns > first_end_ns
    assert evidence["cross_segment_sampling_policy"] == "evenly_spaced_time"
    assert evidence["cross_segment_first_half_scan_count"] >= 2
    assert evidence["cross_segment_second_half_scan_count"] >= 2
    replay_source_end_ns = _BASE_NS + 5 * _STEP_NS
    assert second_start_ns > replay_source_end_ns
    assert evidence["cross_segment_second_half_end_timestamp_ns"] >= _BASE_NS + 35 * _STEP_NS


def test_rosbag2_online_trajectory_skipped_without_odometry(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    bag = _build_moving_rig_bag(tmp_path / "no_odom.db3")
    result = _run_trajectory_session(tmp_path, bag_path=bag, odometry_topic=None)
    provenance = result.run.provenance

    assert provenance.get("trajectory_evidence_skipped_no_odometry") is True
    assert "trajectory_evidence" not in provenance
    assert "trajectory_path" not in provenance
    assert "trajectory_gate_verdict" not in result.metrics
