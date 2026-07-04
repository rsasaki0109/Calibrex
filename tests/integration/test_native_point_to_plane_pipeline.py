"""End-to-end tests for the native LiDAR point-to-plane solve path."""

from pathlib import Path

import pytest

from slac.core.result import load_result
from slac.pipelines.calibrate import CalibrationRunOptions, run_calibration

_REPO_ROOT = Path(__file__).resolve().parents[2]
_A2D2_PAIR_DIR = _REPO_ROOT / "data" / "public" / "a2d2_lidar_pair"

requires_a2d2_pair = pytest.mark.skipif(
    len(list(_A2D2_PAIR_DIR.glob("*.npz"))) < 2 if _A2D2_PAIR_DIR.exists() else True,
    reason=(
        "A2D2 LiDAR pair sample is not downloaded; run "
        "tools/download_public_dataset.py a2d2_lidar_pair_sample"
    ),
)


def _write_native_config(tmp_path: Path, output_dir: Path) -> Path:
    config_path = tmp_path / "a2d2_native_config.yaml"
    config_path.write_text(
        f"""
schema_version: slac.config/v0.1
project:
  name: a2d2_native_point_to_plane
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


@requires_a2d2_pair
def test_native_point_to_plane_pipeline_on_a2d2_pair(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    config_path = _write_native_config(tmp_path, output_dir)

    result = run_calibration(config_path, CalibrationRunOptions())
    assert result is not None

    # (a) the native solver ran and converged to optimized extrinsics
    assert result.run.provenance["solver_adapter"] == "native_lidar_point_to_plane"
    assert result.run.provenance["solver_adapter_status"] == "converged"
    solver_summary = result.run.provenance["native_lidar_point_to_plane_solver"]
    assert solver_summary["status"] == "converged"
    assert solver_summary["accepted_steps"] > 0
    assert solver_summary["final_rmse_m"] < solver_summary["initial_rmse_m"]

    transform = result.transforms["T_base_link_lidar_front_right"]
    assert transform.provenance.producer == "slac_native"
    assert transform.provenance.role_in_comparison == "output"
    assert transform.provenance.evidence_level == "algorithmically_refined"
    assert transform.provenance.tool_name == "native_lidar_point_to_plane"
    assert result.run.provenance["solver_adapter_applied_transforms"] == [
        "T_base_link_lidar_front_right"
    ]

    # (b) observability comes from the real factor evaluation, not the stub
    assert result.observability.rank == 6
    assert result.observability.condition_number is not None
    assert result.observability.condition_number > 0.0
    assert "uncomputed_alpha_backend" not in result.observability.weak_directions
    assert result.observability.grade == "pass"

    # (c) native point-to-plane metrics are surfaced in the result
    assert result.metrics["lidar_rig_point_to_plane_rank"].value == 6.0
    assert result.metrics["lidar_rig_point_to_plane_rmse_m"].value is not None
    condition_metric = result.metrics["lidar_rig_point_to_plane_condition_number"]
    assert condition_metric.value is not None
    assert condition_metric.value > 0.0
    assert result.metrics["lidar_rig_point_to_plane_weak_dof_count"].value == 0.0
    correspondence_metric = result.metrics[
        "native_lidar_point_to_plane_correspondence_count"
    ]
    assert correspondence_metric.value is not None
    assert correspondence_metric.value >= 6.0
    assert result.metrics["prototype_solver"].grade == "pass"

    # persisted result round-trips through the schema with the same evidence
    saved = load_result(output_dir / "result.yaml")
    assert saved.observability.rank == 6
    assert "uncomputed_alpha_backend" not in saved.observability.weak_directions
    assert saved.metrics["lidar_rig_point_to_plane_rank"].value == 6.0
    assert (output_dir / "report.html").exists()


def test_native_point_to_plane_backend_reports_unavailable_without_data(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "outputs"
    config_path = tmp_path / "missing_data_config.yaml"
    config_path.write_text(
        f"""
schema_version: slac.config/v0.1
project:
  name: a2d2_native_missing_data
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
solver:
  backend: native_lidar_point_to_plane
""",
        encoding="utf-8",
    )

    result = run_calibration(config_path, CalibrationRunOptions())
    assert result is not None
    assert result.run.provenance["solver_adapter_status"] == "unavailable"
    metric = result.metrics["native_lidar_point_to_plane_available"]
    assert metric.value == 0.0
    assert metric.grade == "warn"
    # the stub observability stays untouched when the native solve cannot run
    assert result.observability.rank is None
