"""Public TUM RGB-D frontend evidence for the multi-capture joint series."""

from pathlib import Path

import pytest

from calibrex.core.result import load_result
from calibrex.data.tum_rgbd import (
    associate_depth_groundtruth,
    read_groundtruth,
    read_image_index,
    read_tum_depth_png,
    sample_tum_depth_points,
)
from calibrex.pipelines.calibrate import CalibrationRunOptions, run_calibration

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TUM_ROOT = _REPO_ROOT / "data" / "public" / "rgbd_dataset_freiburg1_xyz"
_JOINT_CONFIG = (
    _REPO_ROOT
    / "examples"
    / "public_datasets"
    / "tum_rgbd_freiburg1_xyz"
    / "native_joint_slac_config.yaml"
)

requires_tum_xyz = pytest.mark.skipif(
    not (_TUM_ROOT / "depth.txt").exists() or not (_TUM_ROOT / "groundtruth.txt").exists(),
    reason=(
        "TUM fr1/xyz is not downloaded; run tools/download_public_dataset.py tum_rgbd_freiburg1_xyz"
    ),
)


@requires_tum_xyz
def test_public_tum_depth_and_groundtruth_frontend() -> None:
    depth = read_image_index(_TUM_ROOT / "depth.txt")
    trajectory = read_groundtruth(_TUM_ROOT / "groundtruth.txt")
    paired = associate_depth_groundtruth(depth, trajectory)

    assert len(depth) == 798
    assert len(trajectory) == 3000
    assert len(paired) == 796
    assert max(entry.absolute_time_delta_sec for entry in paired) < 0.011
    image = read_tum_depth_png(_TUM_ROOT / paired[0].depth.path)
    assert (image.width, image.height) == (640, 480)
    points = sample_tum_depth_points(image, max_points=1000)
    assert len(points) >= 500
    assert min(point[2] for point in points) >= 0.2
    assert max(point[2] for point in points) <= 5.0


@requires_tum_xyz
def test_public_tum_multicapture_joint_pipeline(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    result = run_calibration(
        _JOINT_CONFIG,
        CalibrationRunOptions(output_dir=output_dir),
    )

    assert result is not None
    assert result.run.provenance["solver_adapter"] == "native_tum_joint_slac"
    assert result.run.provenance["solver_adapter_status"] == "converged"
    assert result.run.provenance["native_tum_joint_slac_depth_index_sha256"]
    assert result.run.provenance["native_tum_joint_slac_groundtruth_sha256"]
    map_frames = set(result.run.provenance["native_tum_joint_slac_map_frame_ids"])
    query_frames = set(result.run.provenance["native_tum_joint_slac_query_frame_ids"])
    assert map_frames.isdisjoint(query_frames)
    solver = result.run.provenance["native_tum_joint_slac_solver"]
    assert set(solver["train_observation_groups"]).isdisjoint(solver["holdout_observation_groups"])
    rmse = result.metrics["tum_joint_point_to_plane_rmse_m"]
    assert rmse.train is not None and rmse.train < 0.05
    assert rmse.holdout is not None and rmse.holdout < 0.05
    assert rmse.grade == "pass"
    assert result.metrics["tum_joint_augmented_information_rank"].value == 56.0
    assert result.metrics["tum_joint_data_only_extrinsic_rank"].value == 6.0
    condition = result.metrics["tum_joint_data_only_extrinsic_condition_number"].value
    assert condition is not None and condition < 100.0
    assert result.metrics["tum_joint_data_only_shared_rank"].value == 8.0
    shared_condition = result.metrics["tum_joint_data_only_shared_condition_number"].value
    assert shared_condition is not None and shared_condition < 100.0
    scale_error = result.metrics["tum_joint_depth_scale_reference_error_percent"]
    bias_error = result.metrics["tum_joint_depth_bias_reference_error_m"]
    assert scale_error.value is not None and scale_error.value < 2.0
    assert scale_error.grade == "pass"
    assert bias_error.value is not None and bias_error.value > 0.06
    assert bias_error.grade == "fail"
    assert result.metrics["tum_joint_known_bad_detectable_fraction"].value == pytest.approx(0.8125)
    assert result.metrics["tum_joint_known_bad_detectable_fraction"].grade == "warn"
    assert result.metrics["tum_joint_replication_window_count"].value == 3.0
    assert result.metrics["tum_joint_replication_converged_fraction"].value == 1.0
    reference_fraction = result.metrics["tum_joint_replication_reference_pass_fraction"]
    assert reference_fraction.value == pytest.approx(1.0 / 3.0)
    assert reference_fraction.grade == "fail"
    scale_range = result.metrics["tum_joint_replication_depth_scale_range_percent"]
    bias_range = result.metrics["tum_joint_replication_depth_bias_range_m"]
    assert scale_range.value is not None and 2.0 < scale_range.value < 3.0
    assert bias_range.value is not None and 0.06 < bias_range.value < 0.07
    transfer_delta = result.metrics["tum_joint_cross_window_max_holdout_delta_rmse_m"]
    assert transfer_delta.value is not None and 0.015 < transfer_delta.value < 0.016
    assert transfer_delta.grade == "fail"
    assert result.metrics["tum_joint_cross_window_nondegrading_fraction"].value == pytest.approx(
        1.0 / 6.0
    )
    windows = result.run.provenance["native_tum_joint_slac_replication_windows"]
    assert [window["start_index"] for window in windows] == [60, 180, 300]
    assert all(window["selected_depth_files"] for window in windows)
    transfers = result.run.provenance["native_tum_joint_slac_cross_window_transfers"]
    assert len(transfers) == 6
    translation = result.metrics["tum_joint_identity_reference_translation_error_m"]
    rotation = result.metrics["tum_joint_identity_reference_rotation_error_deg"]
    assert translation.value is not None and translation.value > 0.10
    assert translation.grade == "fail"
    assert rotation.value is not None and 1.0 < rotation.value < 2.0
    assert rotation.grade == "warn"
    assert result.quality.grade == "fail"
    transform = result.transforms["T_trajectory_body_rgbd0"]
    assert transform.provenance.producer == "slac_native"
    assert transform.provenance.execution_mode == "offline_batch"
    assert transform.provenance.tool_name == "native_tum_joint_slac"
    saved = load_result(output_dir / "result.yaml")
    assert saved.run.provenance["native_tum_joint_slac_method"] == (
        "tum_multicapture_pose_extrinsic_depth/v0.3"
    )
    assert (
        saved.run.provenance["native_tum_joint_slac_data_only_shared_observability"][
            "information_rank"
        ]
        == 8
    )


def test_tum_joint_pipeline_reports_missing_public_download(tmp_path: Path) -> None:
    config = tmp_path / "missing_tum_joint.yaml"
    config.write_text(
        _JOINT_CONFIG.read_text(encoding="utf-8").replace(
            "data/public/rgbd_dataset_freiburg1_xyz",
            str(tmp_path / "missing-tum"),
        ),
        encoding="utf-8",
    )

    result = run_calibration(
        config,
        CalibrationRunOptions(output_dir=tmp_path / "missing-output"),
    )

    assert result is not None
    assert result.run.provenance["solver_adapter_status"] == "unavailable"
    assert result.metrics["native_tum_joint_slac_available"].value == 0.0
    assert result.observability.rank is None
