"""Temporal (time-offset) evidence tests for motion-compensated online calibration."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

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
_OFFSET_TOLERANCE_S = 0.005
_INJECTION_TOLERANCE_S = 0.010


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
    from calibrex.core.geometry import SE3

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
    source_stamp_offset_s: float = 0.0,
    constant_odometry: bool = False,
    odometry_padding: bool = False,
) -> Path:
    from calibrex.core.geometry import SE3

    world_points = _corner_world_points()
    true_t_base_avia = SE3((0.12, -0.05, 0.03), (0.0, 0.0, 0.0871557, 0.9961947))
    topics = [
        ("/livox/lidar", POINTCLOUD2_TYPE),
        ("/avia/livox/lidar", POINTCLOUD2_TYPE),
        ("/odom", ODOMETRY_TYPE),
    ]
    messages: list[tuple[str, int, bytes]] = []
    if odometry_padding:
        # One extra odometry pose on each side, so stamps shifted by up to one
        # step still interpolate inside the track.
        for index in (-1, message_count):
            timestamp_ns = _BASE_NS + index * _STEP_NS
            t_world_base = _t_world_base(0 if constant_odometry else index)
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
        source_stamp_ns = timestamp_ns + 10 + round(source_stamp_offset_s * 1_000_000_000)
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
    truth_initial: bool = False,
    source_sensor_lines: str = "",
    target_sensor_lines: str = "",
) -> None:
    from calibrex.core.geometry import SE3

    true_t_base_avia = SE3((0.12, -0.05, 0.03), (0.0, 0.0, 0.0871557, 0.9961947))
    if freeze_estimate or truth_initial:
        initial_translation = true_t_base_avia.translation_m
        initial_rotation = true_t_base_avia.rotation_quat_xyzw
    else:
        initial_translation = (0.20, 0.08, 0.06)
        initial_rotation = (0.0, 0.0, 0.0, 1.0)
    gate_line = "        online_gate_max_holdout_rmse_m: 0.01" if freeze_estimate else ""
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
{source_sensor_lines}
  lidar_stream:
    type: lidar
    topic: /avia/livox/lidar
{target_sensor_lines}
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
    truth_initial: bool = False,
    source_sensor_lines: str = "",
    target_sensor_lines: str = "",
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
        truth_initial=truth_initial,
        source_sensor_lines=source_sensor_lines,
        target_sensor_lines=target_sensor_lines,
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
    assert temporal["separability"]["verdict"] == "degenerate"
    assert result.metrics["online_temporal_separability"].grade == "warn"


def test_rosbag2_online_temporal_anchored_recovers_bias_with_free_solver(
    tmp_path: Path,
) -> None:
    pytest.importorskip("numpy")
    bag = _build_moving_rig_bag(tmp_path / "anchored_offset.db3", target_stamp_offset_s=0.05)
    anchored_options = _TEMPORAL_OPTIONS + """
        time_offset_anchor: initial
"""
    result = _run_temporal_session(
        tmp_path,
        bag_path=bag,
        temporal_options=anchored_options,
        freeze_estimate=False,
        truth_initial=True,
    )
    temporal = result.run.provenance["temporal_evidence"]
    anchored = temporal["anchored"]
    adapted = temporal["adapted"]

    anchored_offset_s = float(anchored["estimate"]["estimated_offset_s"])
    assert abs(anchored_offset_s - (-0.05)) <= _OFFSET_TOLERANCE_S
    probes_by_offset = {float(row["offset_s"]): row for row in anchored["probes"]}
    assert probes_by_offset[0.05]["detected"] is True
    assert temporal["separability"]["verdict"] == "separable"
    assert result.metrics["online_temporal_separability"].grade == "pass"
    assert abs(float(adapted["estimate"]["estimated_offset_s"])) < 0.01

    from calibrex.evaluation.evidence_summary import evidence_summaries_from_result

    temporal_rows = [
        item for item in evidence_summaries_from_result(result) if item.family == "temporal"
    ]
    assert len(temporal_rows) == 1
    assert temporal_rows[0].check == "Extrinsic/Temporal Separability"


def test_rosbag2_online_temporal_zero_offset_anchored_consistent(
    tmp_path: Path,
) -> None:
    pytest.importorskip("numpy")
    bag = _build_moving_rig_bag(tmp_path / "zero_anchored.db3", target_stamp_offset_s=0.0)
    anchored_options = _TEMPORAL_OPTIONS + """
        time_offset_anchor: initial
"""
    result = _run_temporal_session(
        tmp_path,
        bag_path=bag,
        temporal_options=anchored_options,
        freeze_estimate=True,
    )
    temporal = result.run.provenance["temporal_evidence"]
    anchored_offset_s = float(temporal["anchored"]["estimate"]["estimated_offset_s"])

    assert abs(anchored_offset_s) <= _OFFSET_TOLERANCE_S
    assert temporal["anchored"]["verdict"] == "pass"
    assert temporal["separability"]["verdict"] == "consistent"
    assert result.metrics["online_temporal_separability"].grade == "pass"


def test_rosbag2_online_temporal_injection_provenance_and_odometry_guard(
    tmp_path: Path,
) -> None:
    pytest.importorskip("numpy")
    from calibrex.core.exceptions import ConfigError

    bag = _build_moving_rig_bag(tmp_path / "inject.db3", target_stamp_offset_s=0.0)
    inject_options = _TEMPORAL_OPTIONS + """
        inject_time_offset_s: 0.05
        time_offset_anchor: initial
"""
    with pytest.raises(ConfigError, match="inject_time_offset_s requires"):
        sub = tmp_path / "no_odom"
        sub.mkdir()
        _run_temporal_session(
            sub,
            bag_path=bag,
            odometry_topic=None,
            temporal_options=inject_options,
        )

    sub_odom = tmp_path / "with_odom"
    sub_odom.mkdir()
    result = _run_temporal_session(
        sub_odom,
        bag_path=bag,
        temporal_options=inject_options,
        freeze_estimate=True,
    )
    temporal = result.run.provenance["temporal_evidence"]
    assert temporal["time_offset_injected_s"] == 0.05
    assert "VALIDATION-ONLY" in temporal["time_offset_injection_warning"]
    anchored_offset_s = float(temporal["anchored"]["estimate"]["estimated_offset_s"])
    assert abs(anchored_offset_s - (-0.05)) <= _INJECTION_TOLERANCE_S
    assert temporal["anchored"]["estimate_gate_status"] == "pass"


# Pillar 3 (ADR 0008): a shared-clock rig whose two LiDARs stamp a sweep at
# different points. Both capture at the odometry instants; the source driver
# stamps at the sweep start (30 ms early) and the target driver at the sweep end
# (30 ms late), so the stamps alone carry a 60 ms convention gap. (The search
# stays within +-0.10 s: further out, correspondences beyond the gate leave only
# the floor plane and the RMSE curve degenerates on this noise-free fixture.)
_SWEEP_PERIOD_S = 0.06
_ANCHORED_SEARCH = """
        time_offset_probe_s: [0.05, -0.05]
        estimate_time_offset: true
        time_offset_search_bound_s: 0.10
        online_gate_max_abs_time_offset_s: 0.05
        time_offset_anchor: initial
"""
_DECLARED_SOURCE = f"""    capture_time_reference: scan_start
    scan_period_s: {_SWEEP_PERIOD_S}"""
_DECLARED_TARGET = f"""    capture_time_reference: scan_end
    scan_period_s: {_SWEEP_PERIOD_S}"""


def _shared_clock_bag(path: Path) -> Path:
    return _build_moving_rig_bag(
        path,
        source_stamp_offset_s=-0.5 * _SWEEP_PERIOD_S,
        target_stamp_offset_s=0.5 * _SWEEP_PERIOD_S,
        odometry_padding=True,
    )


def _anchored_offset_s(result: object) -> float:
    temporal = result.run.provenance["temporal_evidence"]  # type: ignore[attr-defined]
    return float(temporal["anchored"]["estimate"]["estimated_offset_s"])


def test_declared_capture_time_reference_closes_the_shared_clock_baseline(
    tmp_path: Path,
) -> None:
    pytest.importorskip("numpy")
    bag = _shared_clock_bag(tmp_path / "shared_clock.db3")

    undeclared_dir = tmp_path / "undeclared"
    undeclared_dir.mkdir()
    undeclared = _run_temporal_session(
        undeclared_dir, bag_path=bag, temporal_options=_ANCHORED_SEARCH, freeze_estimate=True
    )
    declared_dir = tmp_path / "declared"
    declared_dir.mkdir()
    declared = _run_temporal_session(
        declared_dir,
        bag_path=bag,
        temporal_options=_ANCHORED_SEARCH,
        freeze_estimate=True,
        source_sensor_lines=_DECLARED_SOURCE,
        target_sensor_lines=_DECLARED_TARGET,
    )
    injected_dir = tmp_path / "injected"
    injected_dir.mkdir()
    injected = _run_temporal_session(
        injected_dir,
        bag_path=bag,
        temporal_options=_ANCHORED_SEARCH + "        inject_time_offset_s: 0.05\n",
        freeze_estimate=True,
        source_sensor_lines=_DECLARED_SOURCE,
        target_sensor_lines=_DECLARED_TARGET,
    )

    # Read at face value, the stamps carry the whole 60 ms convention gap.
    assert _anchored_offset_s(undeclared) == pytest.approx(-_SWEEP_PERIOD_S, abs=0.005)
    # Under the declared convention the anchored baseline is within +-10 ms ...
    assert abs(_anchored_offset_s(declared)) <= 0.010
    # ... and an injected +50 ms moves it by exactly -50 ms.
    differential = _anchored_offset_s(injected) - _anchored_offset_s(declared)
    assert differential == pytest.approx(-0.05, abs=0.002)

    provenance = declared.run.provenance  # type: ignore[attr-defined]
    references = provenance["capture_time_reference"]
    assert references["lidar_map"]["applied_shift_s"] == pytest.approx(0.03)
    assert references["lidar_map"]["basis"] == "declared_scan_period"
    assert references["lidar_stream"]["applied_shift_s"] == pytest.approx(-0.03)
    assert "capture_time_reference" not in undeclared.run.provenance  # type: ignore[attr-defined]


def test_capture_time_reference_validation() -> None:
    from pydantic import ValidationError

    from calibrex.core.config import SensorConfig

    with pytest.raises(ValidationError, match="requires scan_period_s"):
        SensorConfig(type="lidar", topic="/points", capture_time_reference="scan_end")
    with pytest.raises(ValidationError, match="only valid for lidar"):
        SensorConfig(
            type="camera",
            topic="/image",
            capture_time_reference="scan_midpoint",
        )
    sensor = SensorConfig(type="lidar", topic="/points", capture_time_reference="scan_midpoint")
    assert sensor.scan_period_s is None


def test_capture_time_reference_is_refused_where_replay_would_ignore_it(tmp_path: Path) -> None:
    from calibrex.core.exceptions import ConfigError

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
schema_version: slac.config/v0.1
project:
  name: capture_reference_guard
  output_dir: {tmp_path / "out"}
dataset:
  type: livox_pcd
  path: {tmp_path / "missing"}
sensors:
  lidar_map:
    type: lidar
    capture_time_reference: scan_midpoint
  lidar_stream:
    type: lidar
frames:
  base_link:
    root: true
  lidar_map:
    parent: base_link
  lidar_stream:
    parent: base_link
    transform:
      estimate: true
pipeline:
  type: multi_sensor_slac
  factors:
    lidar_rig_point_to_plane:
      enabled: true
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="capture_time_reference applies to rosbag1/rosbag2"):
        run_online_calibration(config_path, OnlineCalibrationRunOptions())
