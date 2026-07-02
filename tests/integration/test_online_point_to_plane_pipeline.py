"""End-to-end tests for the online/streaming LiDAR point-to-plane pipeline."""

from pathlib import Path

import pytest

from calibrex.core.io import read_mapping
from calibrex.core.online_timeline import OnlineCalibrationTimelineArtifact
from calibrex.core.result import load_result
from calibrex.core.validation import validate_file
from calibrex.data.downloads import LIVOX_PAIR_DIRNAME
from calibrex.pipelines.online import (
    ONLINE_LIDAR_POINT_TO_PLANE_BACKEND,
    OnlineCalibrationRunOptions,
    run_online_calibration,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_A2D2_PAIR_DIR = _REPO_ROOT / "data" / "public" / "a2d2_lidar_pair"
_LIVOX_PAIR_DIR = _REPO_ROOT / "data" / "public" / LIVOX_PAIR_DIRNAME

requires_a2d2_pair = pytest.mark.skipif(
    len(list(_A2D2_PAIR_DIR.glob("*.npz"))) < 2 if _A2D2_PAIR_DIR.exists() else True,
    reason=(
        "A2D2 LiDAR pair sample is not downloaded; run "
        "tools/download_public_dataset.py a2d2_lidar_pair_sample"
    ),
)
requires_livox_pair = pytest.mark.skipif(
    len(list(_LIVOX_PAIR_DIR.glob("*.pcd"))) < 2 if _LIVOX_PAIR_DIR.exists() else True,
    reason=(
        "Livox horizon-horizon PCD pair sample is not downloaded; run "
        "tools/download_public_dataset.py livox_horizon_horizon_pcd_sample"
    ),
)


def _write_a2d2_online_config(tmp_path: Path, output_dir: Path) -> Path:
    config_path = tmp_path / "a2d2_online_config.yaml"
    config_path.write_text(
        f"""
schema_version: calibrex.config/v0.1
project:
  name: a2d2_online_point_to_plane
  output_dir: {output_dir}
dataset:
  type: a2d2_lidar
  path: {_A2D2_PAIR_DIR}
sensors:
  lidar_front_left:
    type: lidar
  lidar_front_right:
    type: lidar
frames:
  base_link:
    root: true
  lidar_front_left:
    parent: base_link
    transform:
      estimate: false
  lidar_front_right:
    parent: base_link
    transform:
      estimate: true
      prior_sigma:
        translation_m: 0.2
        rotation_deg: 5.0
pipeline:
  type: multi_sensor_slac
  factors:
    lidar_rig_point_to_plane:
      enabled: true
      options:
        voxel_size_m: 0.75
        correspondence_gate_m: 1.0
        max_target_points: 3000
solver:
  backend: native_lidar_point_to_plane
  max_iterations: 40
""",
        encoding="utf-8",
    )
    return config_path


def _write_livox_online_config(tmp_path: Path, output_dir: Path) -> Path:
    config_path = tmp_path / "livox_online_config.yaml"
    config_path.write_text(
        f"""
schema_version: calibrex.config/v0.1
project:
  name: livox_online_point_to_plane
  output_dir: {output_dir}
dataset:
  type: livox_pcd
  path: {_LIVOX_PAIR_DIR}
sensors:
  lidar_base:
    type: lidar
  lidar_target:
    type: lidar
frames:
  base_link:
    root: true
  lidar_base:
    parent: base_link
    transform:
      estimate: false
  lidar_target:
    parent: base_link
    transform:
      estimate: true
      prior_sigma:
        translation_m: 0.2
        rotation_deg: 5.0
pipeline:
  type: multi_sensor_slac
  factors:
    lidar_rig_point_to_plane:
      enabled: true
      options:
        voxel_size_m: 1.0
        correspondence_gate_m: 1.5
        max_target_points: 4000
solver:
  max_iterations: 40
""",
        encoding="utf-8",
    )
    return config_path


@requires_a2d2_pair
def test_online_point_to_plane_pipeline_on_a2d2_pair(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    config_path = _write_a2d2_online_config(tmp_path, output_dir)

    result = run_online_calibration(
        config_path,
        OnlineCalibrationRunOptions(batch_size=400, rolling_window=2000, holdout_ratio=0.2),
    )
    assert result is not None

    assert result.run.provenance["solver_adapter"] == ONLINE_LIDAR_POINT_TO_PLANE_BACKEND
    batch_count = result.run.provenance["online_batch_count"]
    assert batch_count >= 1
    assert (
        result.run.provenance["online_accepted_batch_count"]
        + result.run.provenance["online_rejected_batch_count"]
        + result.run.provenance["online_inconclusive_batch_count"]
        == batch_count
    )

    transform = result.transforms["T_base_link_lidar_front_right"]
    assert transform.provenance.producer == "calibrex_native"
    assert transform.provenance.execution_mode == "online_stream"
    assert transform.provenance.role_in_comparison == "output"
    assert transform.provenance.evidence_level == "algorithmically_refined"
    assert transform.provenance.tool_name == ONLINE_LIDAR_POINT_TO_PLANE_BACKEND

    # observability comes from the final batch's real factor evaluation
    assert result.observability.rank is not None

    # standard result artifacts plus the online timeline artifact are written
    result_path = output_dir / "result.yaml"
    assert result_path.exists()
    assert (output_dir / "report.html").exists()
    timeline_path = Path(result.run.provenance["online_timeline_path"])
    assert timeline_path.exists()

    saved = load_result(result_path)
    assert saved.run.provenance["solver_adapter"] == ONLINE_LIDAR_POINT_TO_PLANE_BACKEND

    timeline = OnlineCalibrationTimelineArtifact.model_validate(read_mapping(timeline_path))
    assert len(timeline.batches) == batch_count
    assert timeline.variable == "T_base_link_lidar_front_right"
    assert timeline.batch_size == 400
    report = validate_file(timeline_path, kind="online-timeline")
    assert report.valid


@requires_livox_pair
def test_online_point_to_plane_pipeline_on_livox_pair(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    config_path = _write_livox_online_config(tmp_path, output_dir)

    result = run_online_calibration(
        config_path,
        OnlineCalibrationRunOptions(batch_size=500, rolling_window=2000, holdout_ratio=0.2),
    )
    assert result is not None
    assert result.run.provenance["solver_adapter"] == ONLINE_LIDAR_POINT_TO_PLANE_BACKEND
    assert result.run.provenance["online_batch_count"] >= 1
    transform = result.transforms["T_base_link_lidar_target"]
    assert transform.provenance.execution_mode == "online_stream"
    timeline_path = Path(result.run.provenance["online_timeline_path"])
    assert timeline_path.exists()


def test_online_pipeline_reports_unavailable_without_data(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    config_path = tmp_path / "missing_data_config.yaml"
    config_path.write_text(
        f"""
schema_version: calibrex.config/v0.1
project:
  name: a2d2_online_missing_data
  output_dir: {output_dir}
dataset:
  type: a2d2_lidar
  path: {tmp_path / "missing_dataset"}
sensors:
  lidar_front_left:
    type: lidar
  lidar_front_right:
    type: lidar
frames:
  base_link:
    root: true
  lidar_front_left:
    parent: base_link
    transform:
      estimate: false
  lidar_front_right:
    parent: base_link
    transform:
      estimate: true
""",
        encoding="utf-8",
    )

    result = run_online_calibration(config_path, OnlineCalibrationRunOptions())
    assert result is not None
    assert result.run.provenance["solver_adapter_status"] == "unavailable"
    metric = result.metrics["online_lidar_point_to_plane_available"]
    assert metric.value == 0.0
    assert metric.grade == "warn"
