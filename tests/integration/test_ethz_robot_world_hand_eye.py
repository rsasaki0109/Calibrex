"""Public ETHZ evidence for Shah robot-world/hand-eye calibration."""

from pathlib import Path

import pytest

from calibrex.pipelines.calibrate import CalibrationRunOptions, run_calibration

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ARCHIVE = (
    _REPO_ROOT
    / "data"
    / "public"
    / "ethz_hand_eye_robot_arm_real"
    / "robot_arm_w_color_camera_real.zip"
)
_CONFIG = (
    _REPO_ROOT
    / "examples"
    / "public_datasets"
    / "ethz_hand_eye_robot_arm_real"
    / "config.yaml"
)

requires_ethz_hand_eye = pytest.mark.skipif(
    not _ARCHIVE.exists(),
    reason=(
        "ETHZ hand-eye archive is not downloaded; run "
        "tools/download_public_dataset.py ethz_hand_eye_robot_arm_real"
    ),
)


@requires_ethz_hand_eye
def test_public_ethz_shah_robot_world_hand_eye_pipeline(tmp_path: Path) -> None:
    result = run_calibration(
        _CONFIG,
        CalibrationRunOptions(output_dir=tmp_path / "outputs"),
    )

    assert result is not None
    assert result.run.provenance["solver_adapter"] == "native_hand_eye_comparison"
    assert result.run.provenance["solver_adapter_status"] == "inconclusive"
    assert result.run.provenance["data_verified"] is True
    assert "T_hand_eye" in result.transforms
    assert result.metrics["robot_world_hand_eye_pose_pair_count"].value == 1688.0
    assert (
        result.metrics["robot_world_hand_eye_shah_rotation_dominant_multiplicity"].value
        == 1.0
    )
    gap = result.metrics["robot_world_hand_eye_shah_rotation_normalized_gap"]
    assert gap.value is not None and 0.029 < gap.value < 0.030
    assert gap.grade == "pass"
    assert result.metrics["robot_world_hand_eye_shah_translation_rank"].value == 6.0
    condition = result.metrics["robot_world_hand_eye_shah_translation_condition_number"]
    assert condition.value is not None and 8.1 < condition.value < 8.3
    assert condition.grade == "pass"
    projection = result.metrics[
        "robot_world_hand_eye_shah_so3_projection_correction_frobenius_max"
    ]
    assert projection.value is not None and projection.value < 2.0e-4
    assert projection.grade == "pass"
    rotation = result.metrics["robot_world_hand_eye_shah_holdout_rotation_rmse_deg"]
    translation = result.metrics["robot_world_hand_eye_shah_holdout_translation_rmse_m"]
    assert rotation.value is not None and 0.58 < rotation.value < 0.59
    assert translation.value is not None and 0.010 < translation.value < 0.011
    assert rotation.grade == translation.grade == "pass"
    detection = result.metrics["robot_world_hand_eye_shah_known_bad_detectable_fraction"]
    assert detection.value == 1.0
    assert detection.grade == "pass"
    comparison = result.run.provenance["native_hand_eye_comparison"]
    shah = comparison["results"]["shah_robot_world_hand_eye"]
    assert shah["method"] == "shah_separable_robot_world_hand_eye_kronecker/v0.1"
    assert shah["paper"]["doi"] == "10.1115/1.4024473"
    assert len(shah["train_pair_ids"]) == 1350
    assert len(shah["holdout_pair_ids"]) == 338
    assert len(shah["known_bad_probes"]) == 24
    assert shah["transform_x"] is not None
    assert shah["transform_y"] is not None
    assert all(probe["detectable"] is True for probe in shah["known_bad_probes"])
