"""Public ETHZ evidence for independent robot-world/hand-eye baselines."""

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
    _REPO_ROOT / "examples" / "public_datasets" / "ethz_hand_eye_robot_arm_real" / "config.yaml"
)

requires_ethz_hand_eye = pytest.mark.skipif(
    not _ARCHIVE.exists(),
    reason=(
        "ETHZ hand-eye archive is not downloaded; run "
        "tools/download_public_dataset.py ethz_hand_eye_robot_arm_real"
    ),
)


@requires_ethz_hand_eye
def test_public_ethz_robot_world_hand_eye_pipeline(tmp_path: Path) -> None:
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
    assert result.metrics["robot_world_hand_eye_shah_rotation_dominant_multiplicity"].value == 1.0
    gap = result.metrics["robot_world_hand_eye_shah_rotation_normalized_gap"]
    assert gap.value is not None and 0.029 < gap.value < 0.030
    assert gap.grade == "pass"
    assert result.metrics["robot_world_hand_eye_shah_translation_rank"].value == 6.0
    condition = result.metrics["robot_world_hand_eye_shah_translation_condition_number"]
    assert condition.value is not None and 8.1 < condition.value < 8.3
    assert condition.grade == "pass"
    projection = result.metrics["robot_world_hand_eye_shah_so3_projection_correction_frobenius_max"]
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
    hand_eye_nonlinear = comparison["results"]["horaud_dornaika_nonlinear"]
    assert hand_eye_nonlinear["status"] == "converged"
    assert hand_eye_nonlinear["accepted_step_count"] == 8
    assert hand_eye_nonlinear["data_jacobian_rank"] == 6
    assert 4.9 < hand_eye_nonlinear["data_jacobian_condition_number"] < 5.1
    assert hand_eye_nonlinear["quaternion_unit_error"] < 1.0e-8
    assert hand_eye_nonlinear["final_cost"] <= hand_eye_nonlinear["initial_cost"]
    nonlinear_rotation = result.metrics[
        "hand_eye_horaud_dornaika_nonlinear_holdout_rotation_rmse_deg"
    ]
    nonlinear_translation = result.metrics[
        "hand_eye_horaud_dornaika_nonlinear_holdout_translation_rmse_m"
    ]
    assert nonlinear_rotation.value is not None and 0.98 < nonlinear_rotation.value < 0.99
    assert nonlinear_translation.value is not None and 0.018 < nonlinear_translation.value < 0.020
    assert nonlinear_rotation.grade == nonlinear_translation.grade == "pass"
    shah = comparison["results"]["shah_robot_world_hand_eye"]
    assert shah["method"] == "shah_separable_robot_world_hand_eye_kronecker/v0.1"
    assert shah["paper"]["doi"] == "10.1115/1.4024473"
    assert len(shah["train_pair_ids"]) == 1350
    assert len(shah["holdout_pair_ids"]) == 338
    assert len(shah["known_bad_probes"]) == 24
    assert shah["transform_x"] is not None
    assert shah["transform_y"] is not None
    assert all(probe["detectable"] is True for probe in shah["known_bad_probes"])
    li_rank = result.metrics["robot_world_hand_eye_li_linear_rank"]
    assert li_rank.value == 24.0
    assert li_rank.grade == "pass"
    li_condition = result.metrics["robot_world_hand_eye_li_linear_condition_number"]
    assert li_condition.value is not None and 28.3 < li_condition.value < 28.5
    assert li_condition.grade == "pass"
    li_residual = result.metrics["robot_world_hand_eye_li_raw_linear_residual_rmse"]
    assert li_residual.value is not None and 0.0055 < li_residual.value < 0.0056
    assert li_residual.grade == "warn"
    li_projection = result.metrics[
        "robot_world_hand_eye_li_so3_projection_correction_frobenius_max"
    ]
    assert li_projection.value is not None and 0.026 < li_projection.value < 0.027
    assert li_projection.grade == "pass"
    assert result.metrics["robot_world_hand_eye_li_shah_common_split_consistent"].value == 1.0
    li_rotation = result.metrics["robot_world_hand_eye_li_holdout_rotation_rmse_deg"]
    li_translation = result.metrics["robot_world_hand_eye_li_holdout_translation_rmse_m"]
    assert li_rotation.value is not None and 0.58 < li_rotation.value < 0.60
    assert li_translation.value is not None and 0.017 < li_translation.value < 0.018
    assert li_rotation.grade == li_translation.grade == "pass"
    li_detection = result.metrics["robot_world_hand_eye_li_known_bad_detectable_fraction"]
    assert li_detection.value == 1.0
    assert li_detection.grade == "pass"
    assert result.metrics["robot_world_hand_eye_li_shah_x_rotation_delta_deg"].value is not None
    assert result.metrics["robot_world_hand_eye_li_shah_x_translation_delta_m"].value is not None
    assert result.metrics["robot_world_hand_eye_li_shah_z_rotation_delta_deg"].value is not None
    assert result.metrics["robot_world_hand_eye_li_shah_z_translation_delta_m"].value is not None
    li = comparison["results"]["li_robot_world_hand_eye"]
    assert li["method"] == "li_wang_wu_simultaneous_robot_world_hand_eye_kronecker/v0.1"
    assert li["paper"]["doi"] == "10.5897/IJPS.9000501"
    assert len(li["train_pair_ids"]) == 1350
    assert len(li["holdout_pair_ids"]) == 338
    assert len(li["linear_singular_values"]) == 24
    assert len(li["known_bad_probes"]) == 24
    assert li["translation_recomputed_after_rotation_projection"] is False
    assert li["transform_x"] is not None
    assert li["transform_z"] is not None
    assert all(probe["detectable"] is True for probe in li["known_bad_probes"])
    assert result.metrics["robot_world_hand_eye_absolute_common_split_consistent"].value == 1.0
    dornaika_gap = result.metrics["robot_world_hand_eye_dornaika_horaud_rotation_normalized_gap"]
    assert dornaika_gap.value is not None and 0.029 < dornaika_gap.value < 0.030
    assert dornaika_gap.grade == "pass"
    sign_flips = result.metrics["robot_world_hand_eye_dornaika_horaud_sign_flip_count"]
    assert sign_flips.value == 405.0
    sign_fraction = result.metrics[
        "robot_world_hand_eye_dornaika_horaud_sign_synchronization_fraction"
    ]
    assert sign_fraction.value == 1.0
    assert sign_fraction.grade == "pass"
    dornaika_condition = result.metrics[
        "robot_world_hand_eye_dornaika_horaud_translation_condition_number"
    ]
    assert dornaika_condition.value is not None and 8.1 < dornaika_condition.value < 8.3
    assert dornaika_condition.grade == "pass"
    dornaika_rotation = result.metrics[
        "robot_world_hand_eye_dornaika_horaud_holdout_rotation_rmse_deg"
    ]
    dornaika_translation = result.metrics[
        "robot_world_hand_eye_dornaika_horaud_holdout_translation_rmse_m"
    ]
    assert dornaika_rotation.value is not None and 0.58 < dornaika_rotation.value < 0.59
    assert dornaika_translation.value is not None and 0.010 < dornaika_translation.value < 0.011
    assert dornaika_rotation.grade == dornaika_translation.grade == "pass"
    dornaika_detection = result.metrics[
        "robot_world_hand_eye_dornaika_horaud_known_bad_detectable_fraction"
    ]
    assert dornaika_detection.value == 1.0
    assert dornaika_detection.grade == "pass"
    dornaika = comparison["results"]["dornaika_horaud_robot_world_hand_eye"]
    assert dornaika["method"] == "dornaika_horaud_robot_world_hand_eye_closed_form/v0.1"
    assert dornaika["paper"]["doi"] == "10.1109/70.704233"
    assert len(dornaika["train_pair_ids"]) == 1350
    assert len(dornaika["holdout_pair_ids"]) == 338
    assert len(dornaika["rotation_singular_values"]) == 4
    assert len(dornaika["known_bad_probes"]) == 24
    assert dornaika["quaternion_sign_flip_count"] == 405
    assert dornaika["quaternion_sign_synchronization_fraction"] == 1.0
    assert dornaika["nonlinear_method_executed"] is False
    assert dornaika["transform_x"] is not None
    assert dornaika["transform_z"] is not None
    zhuang_rank = result.metrics["robot_world_hand_eye_zhuang_roth_sudhakar_rotation_rank"]
    assert zhuang_rank.value == 6.0
    assert zhuang_rank.grade == "pass"
    zhuang_condition = result.metrics[
        "robot_world_hand_eye_zhuang_roth_sudhakar_rotation_condition_number"
    ]
    assert zhuang_condition.value is not None and 21.9 < zhuang_condition.value < 22.1
    assert zhuang_condition.grade == "pass"
    zhuang_a_scalar = result.metrics[
        "robot_world_hand_eye_zhuang_roth_sudhakar_minimum_abs_a_scalar"
    ]
    assert zhuang_a_scalar.value is not None and 0.21 < zhuang_a_scalar.value < 0.22
    assert zhuang_a_scalar.grade == "pass"
    zhuang_z_scalar = result.metrics[
        "robot_world_hand_eye_zhuang_roth_sudhakar_recovered_abs_z_scalar"
    ]
    assert zhuang_z_scalar.value is not None and 0.70 < zhuang_z_scalar.value < 0.71
    assert zhuang_z_scalar.grade == "pass"
    zhuang_normalization = result.metrics[
        "robot_world_hand_eye_zhuang_roth_sudhakar_quaternion_normalization_disagreement"
    ]
    assert zhuang_normalization.value is not None
    assert zhuang_normalization.value < 0.0001
    assert zhuang_normalization.grade == "pass"
    zhuang_scalar_rmse = result.metrics[
        "robot_world_hand_eye_zhuang_roth_sudhakar_scalar_reconstruction_rmse"
    ]
    assert zhuang_scalar_rmse.value is not None
    assert 0.012 < zhuang_scalar_rmse.value < 0.013
    assert zhuang_scalar_rmse.grade == "warn"
    zhuang_sign_fraction = result.metrics[
        "robot_world_hand_eye_zhuang_roth_sudhakar_sign_synchronization_fraction"
    ]
    assert zhuang_sign_fraction.value == 1.0
    assert zhuang_sign_fraction.grade == "pass"
    zhuang_translation_condition = result.metrics[
        "robot_world_hand_eye_zhuang_roth_sudhakar_translation_condition_number"
    ]
    assert zhuang_translation_condition.value is not None
    assert 8.1 < zhuang_translation_condition.value < 8.3
    assert zhuang_translation_condition.grade == "pass"
    zhuang_rotation = result.metrics[
        "robot_world_hand_eye_zhuang_roth_sudhakar_holdout_rotation_rmse_deg"
    ]
    zhuang_translation = result.metrics[
        "robot_world_hand_eye_zhuang_roth_sudhakar_holdout_translation_rmse_m"
    ]
    assert zhuang_rotation.value is not None and 0.59 < zhuang_rotation.value < 0.60
    assert zhuang_translation.value is not None and 0.010 < zhuang_translation.value < 0.011
    assert zhuang_rotation.grade == zhuang_translation.grade == "pass"
    zhuang_detection = result.metrics[
        "robot_world_hand_eye_zhuang_roth_sudhakar_known_bad_detectable_fraction"
    ]
    assert zhuang_detection.value == 1.0
    assert zhuang_detection.grade == "pass"
    zhuang = comparison["results"]["zhuang_roth_sudhakar_robot_world_hand_eye"]
    assert zhuang["method"] == "zhuang_roth_sudhakar_robot_world_hand_eye_linear/v0.1"
    assert zhuang["paper"]["doi"] == "10.1109/70.313105"
    assert len(zhuang["train_pair_ids"]) == 1350
    assert len(zhuang["holdout_pair_ids"]) == 338
    assert len(zhuang["rotation_singular_values"]) == 6
    assert len(zhuang["known_bad_probes"]) == 24
    assert zhuang["quaternion_sign_flip_count"] == 405
    assert zhuang["quaternion_sign_synchronization_fraction"] == 1.0
    assert zhuang["external_code_executed"] is False
    assert zhuang["transform_x"] is not None
    assert zhuang["transform_z"] is not None
    assert all(probe["detectable"] is True for probe in zhuang["known_bad_probes"])
    nonlinear_nonincrease = result.metrics[
        "robot_world_hand_eye_dornaika_horaud_nonlinear_objective_nonincrease"
    ]
    assert nonlinear_nonincrease.value == 1.0
    assert nonlinear_nonincrease.grade == "pass"
    nonlinear_steps = result.metrics[
        "robot_world_hand_eye_dornaika_horaud_nonlinear_accepted_step_count"
    ]
    assert nonlinear_steps.value == 9.0
    nonlinear_residual = result.metrics[
        "robot_world_hand_eye_dornaika_horaud_nonlinear_final_data_residual_rmse"
    ]
    assert nonlinear_residual.value is not None
    assert 0.0055 < nonlinear_residual.value < 0.0057
    nonlinear_rank = result.metrics[
        "robot_world_hand_eye_dornaika_horaud_nonlinear_data_jacobian_rank"
    ]
    assert nonlinear_rank.value == 24.0
    assert nonlinear_rank.grade == "pass"
    nonlinear_condition = result.metrics[
        "robot_world_hand_eye_dornaika_horaud_nonlinear_data_jacobian_condition_number"
    ]
    assert nonlinear_condition.value is not None and 28.3 < nonlinear_condition.value < 28.5
    assert nonlinear_condition.grade == "pass"
    nonlinear_projection = result.metrics[
        "robot_world_hand_eye_dornaika_horaud_nonlinear_so3_projection_correction_frobenius_max"
    ]
    assert nonlinear_projection.value is not None
    assert nonlinear_projection.value < 1.0e-6
    assert nonlinear_projection.grade == "pass"
    nonlinear_rotation = result.metrics[
        "robot_world_hand_eye_dornaika_horaud_nonlinear_holdout_rotation_rmse_deg"
    ]
    nonlinear_translation = result.metrics[
        "robot_world_hand_eye_dornaika_horaud_nonlinear_holdout_translation_rmse_m"
    ]
    assert nonlinear_rotation.value is not None and 0.58 < nonlinear_rotation.value < 0.60
    assert nonlinear_translation.value is not None
    assert 0.010 < nonlinear_translation.value < 0.011
    assert nonlinear_rotation.grade == nonlinear_translation.grade == "pass"
    nonlinear_detection = result.metrics[
        "robot_world_hand_eye_dornaika_horaud_nonlinear_known_bad_detectable_fraction"
    ]
    assert nonlinear_detection.value == 1.0
    assert nonlinear_detection.grade == "pass"
    nonlinear = comparison["results"]["dornaika_horaud_nonlinear_robot_world_hand_eye"]
    assert nonlinear["method"] == "dornaika_horaud_robot_world_hand_eye_nonlinear/v0.1"
    assert nonlinear["paper"]["doi"] == "10.1109/70.704233"
    assert len(nonlinear["train_pair_ids"]) == 1350
    assert len(nonlinear["holdout_pair_ids"]) == 338
    assert len(nonlinear["data_jacobian_singular_values"]) == 24
    assert len(nonlinear["known_bad_probes"]) == 24
    assert nonlinear["accepted_step_count"] == 9
    assert nonlinear["initial_cost"] > nonlinear["final_cost"]
    assert nonlinear["external_code_executed"] is False
    assert nonlinear["transform_x"] is not None
    assert nonlinear["transform_z"] is not None
