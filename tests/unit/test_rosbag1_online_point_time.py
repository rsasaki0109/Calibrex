"""ROS1 Livox CustomMsg point-time and motion-compensation tests."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from calibrex.core.config import load_config
from calibrex.core.geometry import SE3
from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.result import load_result
from calibrex.data.inspect import inspect_dataset
from calibrex.data.rosbag1 import BAG_MAGIC
from calibrex.evaluation.solid_state import solid_state_metrics_from_inspection
from calibrex.pipelines.calibrate import CalibrationRunOptions, run_calibration
from calibrex.pipelines.online import (
    _load_rosbag1_online_pair,
    _solve_inputs,
)
from calibrex.solvers.native_lidar_point_to_plane_solver import (
    lidar_pair_solve_inputs,
    load_lidar_pair,
)

Point = tuple[float, float, float, float]


def _header(fields: dict[str, bytes]) -> bytes:
    return b"".join(
        struct.pack("<I", len(name.encode("ascii") + b"=" + value))
        + name.encode("ascii")
        + b"="
        + value
        for name, value in fields.items()
    )


def _record(fields: dict[str, bytes], data: bytes) -> bytes:
    header = _header(fields)
    return struct.pack("<I", len(header)) + header + struct.pack("<I", len(data)) + data


def _connection(conn_id: int, topic: str, message_type: str) -> bytes:
    data = _header(
        {
            "topic": topic.encode("utf-8"),
            "type": message_type.encode("ascii"),
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


def _message(conn_id: int, timestamp_ns: int, data: bytes) -> bytes:
    seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
    return _record(
        {
            "op": b"\x02",
            "conn": struct.pack("<I", conn_id),
            "time": struct.pack("<II", seconds, nanoseconds),
        },
        data,
    )


def _livox_custommsg(
    points: list[Point],
    *,
    timestamp_ns: int,
    offsets_ns: list[int],
    frame_id: str,
) -> bytes:
    seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
    frame = frame_id.encode("utf-8")
    payload = b"".join(
        struct.pack("<I", offset_ns)
        + struct.pack("<fff", x, y, z)
        + struct.pack("<BBB", int(intensity), 0, index % 4)
        for index, ((x, y, z, intensity), offset_ns) in enumerate(
            zip(points, offsets_ns, strict=True)
        )
    )
    return b"".join(
        (
            struct.pack("<I", 0),
            struct.pack("<II", seconds, nanoseconds),
            struct.pack("<I", len(frame)),
            frame,
            struct.pack("<Q", timestamp_ns),
            struct.pack("<I", len(points)),
            struct.pack("<B", 1),
            b"\x00\x00\x00",
            struct.pack("<I", len(points)),
            payload,
        )
    )


def _pose_stamped(timestamp_ns: int, x: float) -> bytes:
    seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
    frame = b"world"
    return b"".join(
        (
            struct.pack("<I", 0),
            struct.pack("<II", seconds, nanoseconds),
            struct.pack("<I", len(frame)),
            frame,
            struct.pack("<7d", x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        )
    )


def _write_bag(path: Path) -> Path:
    custom_type = "livox_ros_driver/CustomMsg"
    pose_type = "geometry_msgs/PoseStamped"
    topics = (
        (0, "/livox/lidar", custom_type),
        (1, "/avia/livox/lidar", custom_type),
        (2, "/pose", pose_type),
    )
    inner = b"".join(_connection(*topic) for topic in topics)
    point = [(4.0, 0.0, 0.0, 10.0), (4.0, 1.0, 0.0, 20.0)]
    for index, timestamp_ns in enumerate((1_000_000_000, 2_000_000_000)):
        inner += _message(
            0,
            timestamp_ns,
            _livox_custommsg(
                point,
                timestamp_ns=timestamp_ns,
                offsets_ns=[0, 10_000_000],
                frame_id="horizon_frame",
            ),
        )
        inner += _message(
            1,
            timestamp_ns + 100_000_000,
            _livox_custommsg(
                point,
                timestamp_ns=timestamp_ns + 100_000_000,
                offsets_ns=[0, 10_000_000],
                frame_id="avia_frame",
            ),
        )
        del index
    for timestamp_ns, x in (
        (900_000_000, 0.0),
        (1_000_000_000, 1.0),
        (1_100_000_000, 1.1),
        (2_000_000_000, 2.0),
        (2_100_000_000, 2.1),
        (2_200_000_000, 2.2),
    ):
        inner += _message(2, timestamp_ns, _pose_stamped(timestamp_ns, x))
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
            "compression": b"none",
            "size": struct.pack("<I", len(inner)),
        },
        inner,
    )
    path.write_bytes(BAG_MAGIC + bag_header + chunk)
    return path


def _write_dense_livox_bag(path: Path) -> Path:
    custom_type = "livox_ros_driver/CustomMsg"
    topics = ((0, "/livox/lidar", custom_type), (1, "/avia/livox/lidar", custom_type))
    inner = b"".join(_connection(*topic) for topic in topics)
    points = [
        (0.10, 0.10, 0.00, 1.0),
        (0.20, 0.10, 0.00, 1.0),
        (0.10, 0.20, 0.00, 1.0),
        (5.10, 0.10, 0.10, 1.0),
        (5.10, 0.20, 0.10, 1.0),
        (5.10, 0.10, 0.20, 1.0),
        (0.10, 5.10, 0.10, 1.0),
        (0.20, 5.10, 0.10, 1.0),
        (0.10, 5.10, 0.20, 1.0),
    ]
    offsets = [0] * len(points)
    for index, timestamp_ns in enumerate((1_000_000_000, 2_000_000_000, 3_000_000_000)):
        inner += _message(
            0,
            timestamp_ns,
            _livox_custommsg(
                points,
                timestamp_ns=timestamp_ns,
                offsets_ns=offsets,
                frame_id="horizon_frame",
            ),
        )
        inner += _message(
            1,
            timestamp_ns + 100_000_000,
            _livox_custommsg(
                points,
                timestamp_ns=timestamp_ns + 100_000_000,
                offsets_ns=offsets,
                frame_id="avia_frame",
            ),
        )
        del index
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
            "compression": b"none",
            "size": struct.pack("<I", len(inner)),
        },
        inner,
    )
    path.write_bytes(BAG_MAGIC + bag_header + chunk)
    return path


def _write_config(
    path: Path,
    bag_path: Path,
    *,
    use_time: bool,
    source_messages: int = 2,
    target_messages: int = 2,
) -> None:
    time_lines = """
    fields: [x, y, z, intensity, offset_time, line]
    point_time_field: offset_time
    solid_state:
      architecture: solid_state
      scan_pattern: non_repetitive
      point_time_reference: timebase
      point_time_unit: nanoseconds
      point_time_available: true
      point_time_field: offset_time
      intrinsic_calibration:
        source: manufacturer
        model: manufacturer_native""" if use_time else """
    fields: [x, y, z, intensity]"""
    path.write_text(
        f"""
schema_version: slac.config/v0.1
project:
  name: rosbag1_livox_time_fixture
  output_dir: {path.parent / 'outputs'}
dataset:
  type: rosbag1
  path: {bag_path}
  time_base: ros_time
  odometry_topic: /pose
sensors:
  livox_horizon:
    type: lidar
    topic: /livox/lidar
{time_lines}
  livox_avia:
    type: lidar
    topic: /avia/livox/lidar
{time_lines}
frames:
  livox_horizon:
    root: true
  livox_avia:
    parent: livox_horizon
    transform:
      estimate: true
pipeline:
  type: multi_lidar_evidence
  factors:
    lidar_rig_point_to_plane:
      enabled: true
      options:
        max_source_points: 20
        max_target_points: 20
        max_source_messages: {source_messages}
        max_target_messages: {target_messages}
        max_replay_duration_s: 10.0
        inject_time_offset_s: -0.005
solver:
  max_iterations: 5
""",
        encoding="utf-8",
    )


def test_rosbag1_livox_timebase_deskew_and_signed_injection(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    bag = _write_bag(tmp_path / "livox_time.bag")
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, bag, use_time=True)
    config = load_config(config_path)
    result = _load_rosbag1_online_pair(
        config,
        _solve_inputs(config),
        source_sensor="livox_horizon",
        target_sensor="livox_avia",
        t_base_source=SE3.identity(),
    )
    source_records, target_stream, provenance, motion_compensated, track, _info = result

    assert source_records
    assert target_stream
    assert motion_compensated is True
    assert track is not None
    assert provenance["deskew_applied"] is True
    assert provenance["deskew_time_field"] == {
        "livox_horizon": "offset_time",
        "livox_avia": "offset_time",
    }
    assert provenance["point_time_observed_by_sensor"] == {
        "livox_horizon": True,
        "livox_avia": True,
    }
    assert provenance["point_time_reference_by_sensor"] == {
        "livox_horizon": "timebase_mapped_to_ros_bag_timestamp",
        "livox_avia": "timebase_mapped_to_ros_bag_timestamp",
    }
    assert all(
        mapping["status"] == "stable"
        for mapping in provenance["point_time_clock_mapping_by_sensor"].values()
    )
    assert (
        provenance["point_time_clock_mapping_by_sensor"]["livox_avia"][
            "mapped_reference"
        ]
        == "ROS bag record timestamp"
    )
    assert provenance["point_time_range_by_sensor"]["livox_avia"] == {
        "min_s": 0.0,
        "max_s": pytest.approx(0.01),
    }
    assert provenance["rosbag1_capture_clock"].startswith("bag_record_timestamp_ns")
    assert provenance["rosbag1_target_message_count"] == 2
    assert provenance["time_offset_injected_s"] == pytest.approx(-0.005)
    assert target_stream[0][4] == 1_095_000_000


def test_rosbag1_livox_without_time_keeps_message_stamp_path(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    bag = _write_bag(tmp_path / "livox_no_time.bag")
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, bag, use_time=False)
    config = load_config(config_path)
    _source, target_stream, provenance, motion_compensated, _track, _info = (
        _load_rosbag1_online_pair(
            config,
            _solve_inputs(config),
            source_sensor="livox_horizon",
            target_sensor="livox_avia",
            t_base_source=SE3.identity(),
        )
    )
    assert motion_compensated is True
    assert provenance["deskew_applied"] is False
    assert provenance["point_time_observed_by_sensor"] == {
        "livox_horizon": False,
        "livox_avia": False,
    }
    assert target_stream[0][4] == 1_095_000_000


def test_rosbag1_offline_pair_reserves_temporal_holdout(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    bag = _write_bag(tmp_path / "livox_offline.bag")
    config_path = tmp_path / "offline_config.yaml"
    _write_config(config_path, bag, use_time=True)
    config = load_config(config_path)
    pair = load_lidar_pair(
        config,
        [],
        lidar_pair_solve_inputs(config),
        source_sensor="livox_horizon",
        target_sensor="livox_avia",
    )

    assert pair.source_records
    assert pair.target_points
    assert pair.target_holdout_points
    assert pair.source_window_count == 2
    assert pair.target_train_window_count == 1
    assert pair.target_holdout_window_count == 1
    assert pair.temporal_holdout_independent is True
    assert pair.holdout_start_timestamp_ns == 2_100_000_000
    assert pair.point_time_observed_by_sensor == {
        "livox_horizon": True,
        "livox_avia": True,
    }
    assert pair.point_time_reference_by_sensor == {
        "livox_horizon": "timebase_mapped_to_ros_bag_timestamp",
        "livox_avia": "timebase_mapped_to_ros_bag_timestamp",
    }
    assert pair.point_time_clock_mapping_by_sensor is not None
    assert pair.point_time_clock_mapping_by_sensor["livox_horizon"]["status"] == (
        "stable"
    )
    assert pair.point_time_deskew_applied is False


def test_rosbag1_native_offline_result_contains_holdout_provenance(
    tmp_path: Path,
) -> None:
    pytest.importorskip("numpy")
    bag = _write_dense_livox_bag(tmp_path / "livox_native.bag")
    config_path = tmp_path / "native_config.yaml"
    _write_config(
        config_path,
        bag,
        use_time=True,
        source_messages=3,
        target_messages=3,
    )
    config = read_mapping(config_path)
    config["project"]["output_dir"] = str(tmp_path / "native_output")
    config["solver"] = {
        "backend": "native_lidar_point_to_plane",
        "max_iterations": 40,
        "robust_loss": "huber",
        "convergence_tolerance": 1.0e-6,
    }
    config["evaluation"] = {
        "holdout_ratio": 0.2,
        "solid_state": {
            "enabled": True,
            "require_point_time": True,
            "require_independent_holdout": True,
            "min_holdout_windows": 2,
        },
    }
    write_mapping(config_path, config)

    result = run_calibration(
        config_path,
        CalibrationRunOptions(output_dir=tmp_path / "native_output"),
    )
    assert result is not None
    assert result.run.provenance["solver_adapter"] == "native_lidar_point_to_plane"
    assert result.run.provenance["solver_adapter_status"] in {
        "converged",
        "max_iterations",
    }
    evaluation = result.run.provenance["solid_state_evaluation"]
    assert evaluation["independent_holdout"] is True
    assert evaluation["temporal_holdout_independent"] is True
    assert evaluation["train_window_count"] == 2
    assert evaluation["holdout_window_count"] == 1
    assert result.run.provenance["point_time_deskew_applied"] is False
    assert result.run.provenance["point_time_observed_by_sensor"] == {
        "livox_horizon": True,
        "livox_avia": True,
    }
    assert result.metrics["lidar_pair_holdout_point_to_plane_rmse_m"].value is not None
    assert result.metrics["lidar_pair_known_bad_case_count"].value == 6.0
    saved = load_result(tmp_path / "native_output" / "result.yaml")
    assert saved.run.provenance["solid_state_evaluation"]["independent_holdout"] is True


def test_rosbag1_solid_state_inspection_reports_livox_geometry_and_time(
    tmp_path: Path,
) -> None:
    pytest.importorskip("numpy")
    bag = _write_bag(tmp_path / "livox_inspection.bag")
    config_path = tmp_path / "inspection_config.yaml"
    _write_config(config_path, bag, use_time=True)
    config = load_config(config_path)
    inspection = inspect_dataset(config.dataset)
    metrics = solid_state_metrics_from_inspection(config, inspection)

    assert metrics["solid_state_point_time_profile_fraction"].grade == "pass"
    assert metrics["solid_state_distance_min_m"].value == pytest.approx(4.0)
    assert metrics["solid_state_distance_max_m"].value > 4.0
    assert metrics["solid_state_fov_azimuth_deg"].value is not None
    assert metrics["solid_state_fov_elevation_deg"].value == pytest.approx(0.0)
    diagnostics = inspection.diagnostics["rosbag1"]
    assert isinstance(diagnostics, dict)
    streams = diagnostics["streams"]
    assert isinstance(streams, list)
    livox = next(item for item in streams if item["topic"] == "/livox/lidar")
    assert livox["point_time_available"] is True
    assert livox["point_time_field"] == "offset_time"
    assert livox["point_time_max_s"] == pytest.approx(0.01)
