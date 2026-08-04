from __future__ import annotations

import struct
from pathlib import Path

import jsonschema
import pytest

from calibrex.core.config import (
    CalibrationConfig,
    DatasetConfig,
    EvaluationConfig,
    FrameConfig,
    ProjectConfig,
    SensorConfig,
)
from calibrex.core.geometry import SE3
from calibrex.core.io import write_mapping
from calibrex.core.result import CalibrationResult, FrameGraphSnapshot, RunInfo
from calibrex.core.solid_state import (
    SolidStateCaptureWindow,
    SolidStateEvaluationConfig,
    SolidStateIntrinsicCalibration,
    SolidStateLidarProfile,
    SolidStateTemperatureCondition,
    build_solid_state_context,
    solid_state_context_json_schema,
)
from calibrex.core.validation import validate_file
from calibrex.data.inspect import DatasetInspection, inspect_dataset
from calibrex.evaluation.solid_state import solid_state_metrics_from_inspection
from calibrex.pipelines.calibrate import CalibrationRunOptions, run_calibration
from calibrex.pipelines.online import _attach_online_capture_window


def _profile() -> SolidStateLidarProfile:
    return SolidStateLidarProfile(
        architecture="solid_state",
        scan_pattern="non_repetitive",
        integration_window_s=0.005,
        point_time_reference="message_stamp",
        point_time_unit="seconds",
        point_time_available=True,
        intrinsic_calibration=SolidStateIntrinsicCalibration(
            source="manufacturer",
            model="manufacturer_native",
            parameters={"range_scale": 1.0},
        ),
        temperature=SolidStateTemperatureCondition(
            sensor_c=32.5,
            source="sensor_telemetry",
        ),
    )


def test_solid_state_profile_carries_non_repetitive_acquisition_semantics() -> None:
    profile = _profile()
    assert profile.scan_pattern == "non_repetitive"
    assert profile.point_time_available is True
    assert profile.integration_window_s == 0.005
    assert profile.temperature is not None
    assert profile.temperature.sensor_c == 32.5


def test_point_time_available_requires_explicit_reference_and_unit() -> None:
    with pytest.raises(ValueError, match="point_time_available"):
        SolidStateLidarProfile(point_time_available=True)


def test_solid_state_metadata_is_lidar_only() -> None:
    with pytest.raises(ValueError, match="only valid for lidar"):
        SensorConfig(type="camera", solid_state=_profile())


def test_solid_state_context_schema_and_validation(tmp_path: Path) -> None:
    context = build_solid_state_context(
        {"livox0": _profile()},
        config_sha256="a" * 64,
        dataset_sha256="b" * 64,
        source_paths=["config.yaml", "dataset"],
        tool_version="0.4.0",
        git_commit="abc1234",
    )
    assert context is not None
    context.capture_windows["livox0"] = SolidStateCaptureWindow(
        start_timestamp_ns=1_000,
        end_timestamp_ns=2_000,
        point_time_field="time",
        point_time_offsets_available=True,
        point_time_min_s=-0.002,
        point_time_max_s=0.003,
        temperature=SolidStateTemperatureCondition(
            min_c=30.0,
            max_c=35.0,
            source="sensor_telemetry",
        ),
    )
    payload = context.model_dump(mode="json", exclude_none=True)
    jsonschema.validate(payload, solid_state_context_json_schema())

    artifact = tmp_path / "solid_state_context.yaml"
    write_mapping(artifact, payload)
    report = validate_file(artifact, "solid-state-context")
    assert report.valid is True
    assert report.schema_version == "slac.solid_state_lidar_context/v0.1"


def test_config_schema_accepts_solid_state_profile() -> None:
    config = CalibrationConfig(
        dataset=DatasetConfig(type="livox_pcd", path="missing"),
        sensors={"livox0": SensorConfig(type="lidar", solid_state=_profile())},
        frames={"livox0": FrameConfig(root=True)},
    )
    assert config.sensors["livox0"].solid_state is not None


def test_livox_evaluation_reports_distance_and_fov_coverage(tmp_path: Path) -> None:
    header = "\n".join(
        [
            "# .PCD v0.7 - Point Cloud Data file format",
            "VERSION 0.7",
            "FIELDS x y z intensity",
            "SIZE 4 4 4 4",
            "TYPE F F F F",
            "COUNT 1 1 1 1",
            "WIDTH 4",
            "HEIGHT 1",
            "VIEWPOINT 0 0 0 1 0 0 0",
            "POINTS 4",
            "DATA binary",
            "",
        ]
    ).encode("ascii")
    points = [
        (5.0, 0.0, 0.0, 1.0),
        (10.0, 10.0, 1.0, 1.0),
        (10.0, -10.0, -1.0, 1.0),
        (30.0, 0.0, 5.0, 1.0),
    ]
    for name in ("base_horizon_0001.pcd", "target_horizon_0002.pcd"):
        (tmp_path / name).write_bytes(
            header + b"".join(struct.pack("<ffff", *point) for point in points)
        )

    config = CalibrationConfig(
        dataset=DatasetConfig(type="livox_pcd", path=str(tmp_path)),
        sensors={
            "base": SensorConfig(type="lidar", solid_state=_profile()),
            "target": SensorConfig(type="lidar", solid_state=_profile()),
        },
        frames={"base": FrameConfig(root=True), "target": FrameConfig(parent="base")},
        evaluation=EvaluationConfig(
            solid_state=SolidStateEvaluationConfig(
                distance_bins_m=[0.0, 20.0, 40.0],
                min_points_per_distance_bin=1,
                min_distance_span_m=5.0,
                min_fov_azimuth_deg=20.0,
                min_fov_elevation_deg=5.0,
            )
        ),
    )
    inspection = inspect_dataset(config.dataset)
    metrics = solid_state_metrics_from_inspection(config, inspection)
    assert metrics["solid_state_distance_span_m"].grade == "pass"
    azimuth = metrics["solid_state_fov_azimuth_deg"].value
    elevation = metrics["solid_state_fov_elevation_deg"].value
    assert azimuth is not None and azimuth > 20.0
    assert elevation is not None and elevation > 5.0
    assert metrics["solid_state_distance_bin_coverage_fraction"].value == 1.0
    assert metrics["solid_state_temperature_span_c"].value == 0.0
    assert metrics["solid_state_temperature_span_c"].grade == "warn"


def test_rosbag2_livox_evaluation_uses_sampled_geometry_diagnostics() -> None:
    config = CalibrationConfig(
        dataset=DatasetConfig(type="rosbag2", path="missing-bag"),
        sensors={
            "livox": SensorConfig(
                type="lidar", topic="/livox/lidar", solid_state=_profile()
            )
        },
        frames={"livox": FrameConfig(root=True)},
        evaluation=EvaluationConfig(
            solid_state=SolidStateEvaluationConfig(
                distance_bins_m=[0.0, 10.0, 20.0, 40.0, 80.0],
                min_points_per_distance_bin=1,
                min_distance_span_m=5.0,
                min_fov_azimuth_deg=20.0,
                min_fov_elevation_deg=5.0,
            )
        ),
    )
    inspection = DatasetInspection(
        dataset_type="rosbag2",
        path="missing-bag",
        exists=True,
        diagnostics={
            "rosbag2": {
                "source": "rosbag2",
                "streams": [
                    {
                        "topic": "/livox/lidar",
                        "range_min_m": 4.0,
                        "range_max_m": 30.0,
                        "fov_azimuth_deg": 40.0,
                        "fov_elevation_deg": 10.0,
                        "point_time_available": True,
                        "sampled_point_time_min_s": 0.0,
                        "sampled_point_time_max_s": 0.1,
                        "distance_bin_edges_m": [0.0, 10.0, 20.0, 40.0, 80.0],
                        "distance_bin_counts": [10, 10, 10, 10],
                    }
                ],
            }
        },
    )

    metrics = solid_state_metrics_from_inspection(config, inspection)

    assert metrics["solid_state_distance_span_m"].grade == "pass"
    assert metrics["solid_state_fov_azimuth_deg"].value == pytest.approx(40.0)
    assert metrics["solid_state_fov_elevation_deg"].value == pytest.approx(10.0)
    assert metrics["solid_state_distance_bin_coverage_fraction"].value == 1.0


def test_offline_pipeline_persists_solid_state_context_and_provenance(
    tmp_path: Path,
) -> None:
    config = CalibrationConfig(
        project=ProjectConfig(output_dir=str(tmp_path / "output")),
        dataset=DatasetConfig(type="livox_pcd", path=str(tmp_path / "missing_dataset")),
        sensors={
            "base": SensorConfig(type="lidar", solid_state=_profile()),
            "target": SensorConfig(type="lidar", solid_state=_profile()),
        },
        frames={"base": FrameConfig(root=True), "target": FrameConfig(parent="base")},
    )
    config_path = tmp_path / "config.yaml"
    write_mapping(config_path, config.model_dump(mode="json", exclude_none=True))

    result = run_calibration(
        config_path,
        CalibrationRunOptions(output_dir=tmp_path / "output"),
    )

    assert result is not None
    assert result.solid_state is not None
    assert set(result.solid_state.sensors) == {"base", "target"}
    assert result.solid_state.provenance.source_sha256["config"]
    assert "solid_state_evaluation_config" in result.run.provenance
    assert "solid_state_profile_count" in result.metrics


def test_online_capture_window_records_point_time_observation() -> None:
    profile = _profile()
    config = CalibrationConfig(
        dataset=DatasetConfig(type="rosbag2", path="missing"),
        sensors={
            "base": SensorConfig(type="lidar", solid_state=profile),
            "target": SensorConfig(
                type="lidar",
                point_time_field="time",
                solid_state=profile,
            ),
        },
        frames={"base": FrameConfig(root=True), "target": FrameConfig(parent="base")},
    )
    context = build_solid_state_context(
        {"base": profile, "target": profile},
        config_sha256="a" * 64,
    )
    result = CalibrationResult(
        run=RunInfo(id="run", slac_version="test"),
        frame_graph=FrameGraphSnapshot(root="base", frames={"base": None, "target": "base"}),
        solid_state=context,
    )
    target_stream = [
        (0, (1.0, 0.0, 0.0), SE3.identity(), 0.0, 1_000_000_000),
        (0, (1.0, 0.0, 0.0), SE3.identity(), 0.0, 1_000_000_100),
    ]

    _attach_online_capture_window(
        result,
        config,
        "target",
        target_stream,
        {
                "rosbag2_first_target_timestamp_ns": 1_000_000_000,
                "rosbag2_replay_end_timestamp_ns": 1_000_000_100,
                "deskew_applied": True,
                "point_time_observed_by_sensor": {"target": True},
            },
    )

    assert result.solid_state is not None
    assert config.sensors["target"].solid_state is not None
    assert config.sensors["target"].solid_state.point_time_field == "time"
    window = result.solid_state.capture_windows["target"]
    assert window.point_time_offsets_available is True
    assert window.point_time_deskew_applied is True
    assert window.point_time_field == "time"
    assert window.capture_span_s == pytest.approx(1.0e-7)
