"""Temporal (time-offset) evidence tests for motion-compensated online calibration."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

from slac.pipelines.online import OnlineCalibrationRunOptions, run_online_calibration

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
_OFFSET_TOLERANCE_S = 0.005


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


def _t_world_base(index: int) -> object:
    from slac.core.geometry import SE3

    yaw = 0.12 * index
    return SE3((0.5 * index, 0.0, 0.0), _yaw_quaternion(yaw))


def _world_to_sensor(point_world: tuple[float, float, float], t_world_sensor: object) -> tuple:
    point_sensor = t_world_sensor.inverse().transform_point(point_world)  # type: ignore[attr-defined]
    return (point_sensor[0], point_sensor[1], point_sensor[2], 0.0)


def _build_moving_rig_bag(
    path: Path,
    *,
    message_count: int = 10,
    target_stamp_offset_s: float = 0.0,
    constant_odometry: bool = False,
) -> Path:
    from slac.core.geometry import SE3

    world_points = _corner_world_points()
    true_t_base_avia = SE3((0.12, -0.05, 0.03), (0.0, 0.0, 0.0871557, 0.9961947))
    topics = [
        ("/livox/lidar", POINTCLOUD2_TYPE),
        ("/avia/livox/lidar", POINTCLOUD2_TYPE),
        ("/odom", ODOMETRY_TYPE),
    ]
    messages: list[tuple[str, int, bytes]] = []
    for index in range(message_count):
        timestamp_ns = _BASE_NS + index * _STEP_NS
        t_world_base = _t_world_base(0 if constant_odometry else index)
        t_world_horizon = t_world_base
        t_world_avia = t_world_base.compose(true_t_base_avia)

        horizon_points = [_world_to_sensor(point, t_world_horizon) for point in world_points]
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
        source_stamp_ns = timestamp_ns + 10
        messages.append(
            (
                "/livox/lidar",
                timestamp_ns + 10,
                _encode_pointcloud2(
                    horizon_points,
                    frame_id="livox_horizon",
                    secs=source_stamp_ns // 1_000_000_000,
                    nsecs=source_stamp_ns % 1_000_000_000,
                    with_intensity=True,
                ),
            )
        )
        target_stamp_ns = timestamp_ns + 20 + round(target_stamp_offset_s * 1_000_000_000)
        messages.append(
            (
                "/avia/livox/lidar",
                timestamp_ns + 20,
                _encode_pointcloud2(
                    avia_points,
                    frame_id="livox_avia",
                    secs=target_stamp_ns // 1_000_000_000,
                    nsecs=target_stamp_ns % 1_000_000_000,
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
    temporal_options: str = "",
    freeze_estimate: bool = False,
) -> None:
    from slac.core.geometry import SE3

    true_t_base_avia = SE3((0.12, -0.05, 0.03), (0.0, 0.0, 0.0871557, 0.9961947))
    if freeze_estimate:
        initial_translation = true_t_base_avia.translation_m
        initial_rotation = true_t_base_avia.rotation_quat_xyzw
        gate_line = "        online_gate_max_holdout_rmse_m: 0.01"
    else:
        initial_translation = (0.20, 0.08, 0.06)
        initial_rotation = (0.0, 0.0, 0.0, 1.0)
        gate_line = ""
    odometry_line = f"  odometry_topic: {odometry_topic}" if odometry_topic else ""
    config_path.write_text(
        f"""
schema_version: slac.config/v0.1
project:
  name: rosbag2_temporal_fixture
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
        translation: [
          {initial_translation[0]}, {initial_translation[1]}, {initial_translation[2]}
        ]
        rotation_quat_xyzw: [
          {initial_rotation[0]}, {initial_rotation[1]},
          {initial_rotation[2]}, {initial_rotation[3]}
        ]
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
{gate_line}
{temporal_options}
solver:
  max_iterations: 30
""",
        encoding="utf-8",
    )


_TEMPORAL_OPTIONS = """
        time_offset_probe_s: [0.05, -0.05]
        estimate_time_offset: true
        time_offset_search_bound_s: 0.10
        online_gate_max_abs_time_offset_s: 0.05
"""

_TEMPORAL_OPTIONS_TIGHT_GATE = """
        time_offset_probe_s: [0.05, -0.05]
        estimate_time_offset: true
        time_offset_search_bound_s: 0.10
        online_gate_max_abs_time_offset_s: 0.04
"""


def _run_temporal_session(
    tmp_path: Path,
    *,
    bag_path: Path,
    odometry_topic: str | None = "/odom",
    temporal_options: str = _TEMPORAL_OPTIONS,
    freeze_estimate: bool = False,
) -> object:
    output_dir = tmp_path / "outputs"
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        bag_path=bag_path,
        output_dir=output_dir,
        odometry_topic=odometry_topic,
        temporal_options=temporal_options,
        freeze_estimate=freeze_estimate,
    )
    result = run_online_calibration(
        config_path,
        OnlineCalibrationRunOptions(batch_size=200, rolling_window=400, holdout_ratio=0.2),
    )
    assert result is not None
    return result


def test_rosbag2_online_temporal_recovers_injected_offset_and_fails_gate(
    tmp_path: Path,
) -> None:
    pytest.importorskip("numpy")
    bag = _build_moving_rig_bag(tmp_path / "offset.db3", target_stamp_offset_s=0.05)
    result = _run_temporal_session(
        tmp_path,
        bag_path=bag,
        temporal_options=_TEMPORAL_OPTIONS_TIGHT_GATE,
        freeze_estimate=True,
    )
    provenance = result.run.provenance
    temporal = provenance["temporal_evidence"]
    estimate = temporal["estimate"]
    estimated_offset_s = float(estimate["estimated_offset_s"])

    assert abs(estimated_offset_s - (-0.05)) <= _OFFSET_TOLERANCE_S
    probes_by_offset = {float(row["offset_s"]): row for row in temporal["probes"]}
    assert probes_by_offset[0.05]["detected"] is True
    assert temporal["estimate_gate_status"] == "fail"
    assert temporal["verdict"] == "fail"
    assert result.metrics["online_temporal_gate_verdict"].grade == "fail"


def test_rosbag2_online_temporal_zero_offset_passes(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    bag = _build_moving_rig_bag(tmp_path / "zero.db3", target_stamp_offset_s=0.0)
    result = _run_temporal_session(tmp_path, bag_path=bag, freeze_estimate=True)
    temporal = result.run.provenance["temporal_evidence"]
    estimated_offset_s = float(temporal["estimate"]["estimated_offset_s"])

    assert abs(estimated_offset_s) <= _OFFSET_TOLERANCE_S
    assert temporal["probe_gate_status"] == "pass"
    assert all(probe["detected"] for probe in temporal["probes"])
    assert temporal["estimate_gate_status"] == "pass"
    assert temporal["verdict"] == "pass"
    assert result.metrics["online_temporal_gate_verdict"].grade == "pass"


def test_rosbag2_online_temporal_skipped_without_odometry(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    bag = _build_moving_rig_bag(tmp_path / "no_odom.db3")
    result = _run_temporal_session(tmp_path, bag_path=bag, odometry_topic=None)
    provenance = result.run.provenance

    assert provenance.get("temporal_evidence_skipped_no_odometry") is True
    assert "temporal_evidence" not in provenance
    assert "online_temporal_gate_verdict" not in result.metrics


def test_rosbag2_online_temporal_flat_curve_inconclusive(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    bag = _build_moving_rig_bag(
        tmp_path / "static.db3",
        message_count=8,
        constant_odometry=True,
    )
    flat_options = """
        estimate_time_offset: true
        time_offset_search_bound_s: 0.10
        online_gate_max_abs_time_offset_s: 0.05
"""
    result = _run_temporal_session(tmp_path, bag_path=bag, temporal_options=flat_options)
    temporal = result.run.provenance["temporal_evidence"]

    assert temporal["estimate_gate_status"] == "inconclusive"
    assert temporal["verdict"] == "inconclusive"
    assert "flat" in temporal["estimate_gate_reason"].lower()
    assert float(temporal["estimate"]["curve_flatness"]) < 0.10
    assert result.metrics["online_temporal_gate_verdict"].grade == "warn"
