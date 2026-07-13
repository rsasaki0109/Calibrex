from calibrex.core.config import CalibrationConfig
from calibrex.core.evidence import stable_artifact_id, validate_no_frame_overlap
from calibrex.core.result import (
    CalibrationResult,
    DegeneracyResult,
    FrameGraphSnapshot,
    MetricResult,
    ObservabilityResult,
    RunInfo,
    TransformResult,
)
from calibrex.data.base import StreamSummary
from calibrex.data.inspect import DatasetInspection
from calibrex.evaluation.compare import compare_results
from calibrex.evaluation.degeneracy import degeneracy_from_inspection
from calibrex.evaluation.holdout import split_indices
from calibrex.evaluation.lidar import lidar_metrics_from_inspection
from calibrex.evaluation.lidar_camera import lidar_camera_metrics_from_result
from calibrex.evaluation.metrics import evaluate_quality
from calibrex.evaluation.motion import motion_metrics_from_inspection
from calibrex.evaluation.registry import get_metric_definition, list_metric_definitions
from calibrex.evaluation.thresholds import apply_metric_thresholds
from calibrex.evaluation.timing import timing_metrics_from_inspection


def test_metric_thresholds_grade_holdout_values() -> None:
    metrics = {
        "reprojection_rmse_px": MetricResult(train=0.7, holdout=2.5),
        "dataset_exists": MetricResult(value=1.0),
    }
    apply_metric_thresholds(metrics, "default")
    assert metrics["reprojection_rmse_px"].grade == "fail"
    assert metrics["dataset_exists"].grade == "pass"


def test_joint_slac_thresholds_preserve_known_bad_warning() -> None:
    metrics = {
        "joint_slac_augmented_information_rank": MetricResult(value=12.0),
        "joint_slac_augmented_condition_number": MetricResult(value=80.8),
        "joint_slac_known_bad_detectable_fraction": MetricResult(value=0.75),
        "tum_joint_point_to_plane_rmse_m": MetricResult(holdout=0.0318),
        "tum_joint_data_only_extrinsic_rank": MetricResult(value=6.0),
        "tum_joint_data_only_shared_rank": MetricResult(value=8.0),
        "tum_joint_depth_scale_reference_error_percent": MetricResult(value=0.161),
        "tum_joint_depth_bias_reference_error_m": MetricResult(value=0.0755),
        "tum_joint_known_bad_detectable_fraction": MetricResult(value=0.75),
    }

    apply_metric_thresholds(metrics, "default")

    assert metrics["joint_slac_augmented_information_rank"].grade == "pass"
    assert metrics["joint_slac_augmented_condition_number"].grade == "pass"
    assert metrics["joint_slac_known_bad_detectable_fraction"].grade == "warn"
    assert metrics["tum_joint_point_to_plane_rmse_m"].grade == "pass"
    assert metrics["tum_joint_data_only_extrinsic_rank"].grade == "pass"
    assert metrics["tum_joint_data_only_shared_rank"].grade == "pass"
    assert metrics["tum_joint_depth_scale_reference_error_percent"].grade == "pass"
    assert metrics["tum_joint_depth_bias_reference_error_m"].grade == "fail"
    assert metrics["tum_joint_known_bad_detectable_fraction"].grade == "warn"


def test_evidence_leakage_validator_reports_frame_overlap() -> None:
    assert stable_artifact_id("slice", ["a", "b"]) == stable_artifact_id("slice", ["a", "b"])

    passed = validate_no_frame_overlap(
        map_frame_ids=["0000000000"],
        query_frame_ids=["0000000001"],
        checked_dependency_ids=["map0", "corr0"],
    )
    assert passed.status == "pass"
    assert passed.issue_count == 0
    assert passed.checked_dependency_ids == ("map0", "corr0")

    failed = validate_no_frame_overlap(
        map_frame_ids=["0000000000", "0000000001"],
        query_frame_ids=["0000000001"],
    )
    assert failed.status == "fail"
    assert failed.issue_count == 1
    assert failed.overlapping_frame_ids == ("0000000001",)


def test_autonomous_driving_threshold_profile_is_stricter() -> None:
    default_metrics = {"reprojection_rmse_px": MetricResult(value=0.9)}
    auto_metrics = {"reprojection_rmse_px": MetricResult(value=0.9)}
    apply_metric_thresholds(default_metrics, "default")
    apply_metric_thresholds(auto_metrics, "autonomous_driving")
    assert default_metrics["reprojection_rmse_px"].grade == "pass"
    assert auto_metrics["reprojection_rmse_px"].grade == "warn"


def test_evaluate_quality_recomputes_metric_grades() -> None:
    result = CalibrationResult(
        run=RunInfo(id="unit", slac_version="0.1.0"),
        frame_graph=FrameGraphSnapshot(root="base", frames={"base": None}),
        metrics={"lidar_point_to_plane_rmse_m": MetricResult(value=0.2, grade="pass")},
    )
    evaluated = evaluate_quality(result)
    assert evaluated.metrics["lidar_point_to_plane_rmse_m"].grade == "fail"
    assert evaluated.quality.grade == "fail"


def test_compare_results_reports_metric_and_transform_deltas() -> None:
    left = CalibrationResult(
        run=RunInfo(id="left", slac_version="0.1.0"),
        frame_graph=FrameGraphSnapshot(root="base", frames={"base": None, "lidar0": "base"}),
        transforms={
            "T_base_lidar0": TransformResult(
                parent="base",
                child="lidar0",
                translation_m=[0.0, 0.0, 0.0],
                rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            )
        },
        metrics={
            "lidar_point_to_plane_rmse_m": MetricResult(
                train=0.04,
                holdout=0.05,
                grade="warn",
                unit="m",
            ),
            "lidar_pair_source_voxel_recall_in_target": MetricResult(
                value=0.20,
                grade="warn",
            ),
            "lidar_pair_known_bad_detectable_fraction": MetricResult(
                value=0.25,
                grade="warn",
            ),
            "lidar_pair_known_bad_centroid_rmse_delta_max_m": MetricResult(
                value=0.01,
                grade="warn",
                unit="m",
            ),
        },
        observability=ObservabilityResult(
            rank=4,
            condition_number=1000.0,
            weak_directions=["yaw_lidar0"],
        ),
        degeneracy=DegeneracyResult(grade="warn", reason="limited yaw excitation"),
    )
    right = CalibrationResult(
        run=RunInfo(id="right", slac_version="0.1.0"),
        frame_graph=FrameGraphSnapshot(root="base", frames={"base": None, "lidar0": "base"}),
        transforms={
            "T_base_lidar0": TransformResult(
                parent="base",
                child="lidar0",
                translation_m=[0.1, 0.0, 0.0],
                rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            )
        },
        metrics={
            "lidar_point_to_plane_rmse_m": MetricResult(
                train=0.03,
                holdout=0.02,
                grade="pass",
                unit="m",
            ),
            "lidar_pair_source_voxel_recall_in_target": MetricResult(
                value=0.30,
                grade="pass",
            ),
            "lidar_pair_known_bad_detectable_fraction": MetricResult(
                value=0.75,
                grade="pass",
            ),
            "lidar_pair_known_bad_centroid_rmse_delta_max_m": MetricResult(
                value=0.04,
                grade="pass",
                unit="m",
            ),
        },
        observability=ObservabilityResult(rank=5, condition_number=500.0),
        degeneracy=DegeneracyResult(grade="pass"),
    )

    comparison = compare_results(left, right)

    metric = comparison.metrics["lidar_point_to_plane_rmse_m"]
    assert metric.preferred_field == "holdout"
    assert round(metric.delta_right_minus_left or 0.0, 6) == -0.03
    assert metric.preference == "lower"
    assert metric.winner == "right"
    assert comparison.metric_families["lidar"].right_better_count == 1
    known_bad_delta = comparison.metrics["lidar_pair_known_bad_centroid_rmse_delta_max_m"]
    assert known_bad_delta.preference == "higher"
    assert known_bad_delta.winner == "not_comparable"
    assert known_bad_delta.not_comparable_reason == (
        "LiDAR pair evidence protocol is not_comparable: "
        "neither result declares an evidence protocol"
    )
    evidence = {(item.family, item.check): item for item in comparison.evidence_comparisons}
    assert evidence[("lidar_pair", "Candidate Support")].winner == "not_comparable"
    assert evidence[("lidar_pair", "Candidate Support")].not_comparable_reason == (
        "LiDAR pair evidence protocol is not_comparable: "
        "neither result declares an evidence protocol"
    )
    assert evidence[("lidar_pair", "Known-Bad Controls")].winner == "not_comparable"
    assert evidence[("lidar_pair", "Decision Boundary")].winner == "not_comparable"
    transform = comparison.transform_groups["transforms"].comparisons[0]
    assert round(transform.translation_delta_m, 6) == 0.1
    assert transform.rotation_delta_deg == 0.0
    assert comparison.summary.max_translation_delta_m == 0.1
    assert comparison.observability.only_left_weak_directions == ["yaw_lidar0"]
    assert comparison.observability.rank_delta_right_minus_left == 1
    assert comparison.protocol_compatibility.status == "not_comparable"
    assert comparison.protocol_compatibility.reasons == [
        "neither result declares an evidence protocol"
    ]


def test_compare_results_reports_protocol_compatibility() -> None:
    left = CalibrationResult(
        run=RunInfo(
            id="left",
            slac_version="0.1.0",
            provenance=_livox_pair_provenance(metrics_origin="cached"),
        ),
        frame_graph=FrameGraphSnapshot(root="base", frames={"base": None}),
        metrics={
            "lidar_pair_holdout_point_to_plane_support_ratio": MetricResult(
                value=0.70,
                grade="pass",
            )
        },
    )
    right = CalibrationResult(
        run=RunInfo(
            id="right",
            slac_version="0.1.0",
            provenance=_livox_pair_provenance(metrics_origin="cached"),
        ),
        frame_graph=FrameGraphSnapshot(root="base", frames={"base": None}),
        metrics={
            "lidar_pair_holdout_point_to_plane_support_ratio": MetricResult(
                value=0.80,
                grade="pass",
            )
        },
    )

    comparison = compare_results(left, right)

    assert comparison.protocol_compatibility.status == "compatible"
    assert comparison.protocol_compatibility.shared_protocol_ids == [
        "livox_pair_single_pair_holdout_point_to_plane/v0.1"
    ]
    assert comparison.protocol_compatibility.left_protocols[0].metrics_origin == "cached"
    assert comparison.protocol_compatibility.left_protocols[0].support_population_id == (
        "livox_pair_support:train=base:holdout=target:eligible_points=1000:voxel_m=1:gate_m=1.5"
    )
    assert comparison.protocol_compatibility.left_protocols[0].challenge_id == (
        "livox_pair_mandatory_6dof_large_controls/v0.1"
    )
    assert (
        comparison.protocol_compatibility.left_protocols[
            0
        ].challenge_target_supported_detection_count
        == 8
    )
    assert (
        comparison.protocol_compatibility.left_protocols[0].mandatory_detection_by_dof_digest
        is not None
    )
    assert comparison.metrics["lidar_pair_holdout_point_to_plane_support_ratio"].winner == "right"

    support_changed = CalibrationResult(
        run=RunInfo(
            id="support-changed",
            slac_version="0.1.0",
            provenance=_livox_pair_provenance(metrics_origin="cached"),
        ),
        frame_graph=FrameGraphSnapshot(root="base", frames={"base": None}),
        metrics={
            "lidar_pair_holdout_point_to_plane_support_ratio": MetricResult(
                value=0.95,
                grade="pass",
            )
        },
    )
    support_changed.run.provenance["livox_pair_evidence"]["holdout_geometry"][
        "support_population_id"
    ] = "livox_pair_support:train=base:holdout=other:eligible_points=1000:voxel_m=1:gate_m=1.5"
    support_mismatch = compare_results(left, support_changed)

    assert support_mismatch.protocol_compatibility.status == "warning"
    assert support_mismatch.protocol_compatibility.reasons == [
        "livox_pair_single_pair_holdout_point_to_plane/v0.1: support population "
        "differs (livox_pair_support:train=base:holdout=target:"
        "eligible_points=1000:voxel_m=1:gate_m=1.5 vs "
        "livox_pair_support:train=base:holdout=other:eligible_points=1000:"
        "voxel_m=1:gate_m=1.5)"
    ]
    assert (
        support_mismatch.metrics["lidar_pair_holdout_point_to_plane_support_ratio"].winner
        == "not_comparable"
    )
    assert (
        support_mismatch.metrics[
            "lidar_pair_holdout_point_to_plane_support_ratio"
        ].not_comparable_reason
        == "LiDAR pair evidence protocol is warning: "
        "livox_pair_single_pair_holdout_point_to_plane/v0.1: "
        "support population differs "
        "(livox_pair_support:train=base:holdout=target:eligible_points=1000:"
        "voxel_m=1:gate_m=1.5 vs "
        "livox_pair_support:train=base:holdout=other:eligible_points=1000:"
        "voxel_m=1:gate_m=1.5)"
    )

    challenge_changed = CalibrationResult(
        run=RunInfo(
            id="challenge-changed",
            slac_version="0.1.0",
            provenance=_livox_pair_provenance(metrics_origin="cached"),
        ),
        frame_graph=FrameGraphSnapshot(root="base", frames={"base": None}),
        metrics={
            "lidar_pair_holdout_point_to_plane_support_ratio": MetricResult(
                value=0.95,
                grade="pass",
            )
        },
    )
    challenge_changed.run.provenance["livox_pair_evidence"]["known_bad_challenge"][
        "target_supported_detection_count"
    ] = 10
    challenge_mismatch = compare_results(left, challenge_changed)

    assert challenge_mismatch.protocol_compatibility.status == "warning"
    assert challenge_mismatch.protocol_compatibility.reasons == [
        "livox_pair_single_pair_holdout_point_to_plane/v0.1: "
        "challenge supported-detection target differs (8 vs 10)"
    ]
    assert (
        challenge_mismatch.metrics["lidar_pair_holdout_point_to_plane_support_ratio"].winner
        == "not_comparable"
    )

    dof_changed = CalibrationResult(
        run=RunInfo(
            id="dof-changed",
            slac_version="0.1.0",
            provenance=_livox_pair_provenance(metrics_origin="cached"),
        ),
        frame_graph=FrameGraphSnapshot(root="base", frames={"base": None}),
        metrics={
            "lidar_pair_holdout_point_to_plane_support_ratio": MetricResult(
                value=0.95,
                grade="pass",
            )
        },
    )
    dof_changed.run.provenance["livox_pair_evidence"]["known_bad_challenge"][
        "mandatory_detection_by_dof"
    ]["yaw_deg"]["supported_detection_count"] = 1
    dof_mismatch = compare_results(left, dof_changed)

    assert dof_mismatch.protocol_compatibility.status == "warning"
    assert dof_mismatch.protocol_compatibility.reasons == [
        "livox_pair_single_pair_holdout_point_to_plane/v0.1: "
        "mandatory detection-by-DoF summary differs"
    ]
    assert (
        dof_mismatch.metrics["lidar_pair_holdout_point_to_plane_support_ratio"].winner
        == "not_comparable"
    )

    right.run.provenance["metrics_origin"] = "recomputed"
    mixed = compare_results(left, right)

    assert mixed.protocol_compatibility.status == "warning"
    assert mixed.protocol_compatibility.reasons == [
        "livox_pair_single_pair_holdout_point_to_plane/v0.1: "
        "metrics origin differs (cached vs recomputed)"
    ]


def _livox_pair_provenance(metrics_origin: str) -> dict[str, object]:
    return {
        "metrics_origin": metrics_origin,
        "data_verified": False,
        "livox_pair_evidence": {
            "candidate_transform": "T_source_target",
            "transform_convention": (
                "T_source_target maps target PCD points into the source PCD frame"
            ),
            "known_bad_case_count": 24,
            "holdout_geometry": {
                "split_policy": "single_pair_source_map_target_query",
                "independent_holdout": False,
                "support_population_id": (
                    "livox_pair_support:train=base:holdout=target:"
                    "eligible_points=1000:voxel_m=1:gate_m=1.5"
                ),
                "support_definition": ("eligible population is every finite target PCD point"),
                "voxel_size_m": 1.0,
                "correspondence_gate_m": 1.5,
                "inlier_threshold_m": 0.25,
                "plane_normal_source": "pcd_normal_fields_or_local_fallback",
            },
            "known_bad_challenge": {
                "challenge_id": "livox_pair_mandatory_6dof_large_controls/v0.1",
                "composition": "T_test = Exp(xi_hat) * T_source_target",
                "tangent_frame": "source_lidar_frame",
                "mandatory_rotation_deg": 1.0,
                "mandatory_translation_m": 0.1,
                "mandatory_case_count": 12,
                "mandatory_detection_by_dof": {
                    "roll_deg": {
                        "case_count": 2,
                        "detected_count": 2,
                        "supported_detection_count": 2,
                        "support_collapse_count": 0,
                        "amounts": [1.0, -1.0],
                    },
                    "pitch_deg": {
                        "case_count": 2,
                        "detected_count": 2,
                        "supported_detection_count": 2,
                        "support_collapse_count": 0,
                        "amounts": [1.0, -1.0],
                    },
                    "yaw_deg": {
                        "case_count": 2,
                        "detected_count": 2,
                        "supported_detection_count": 2,
                        "support_collapse_count": 0,
                        "amounts": [1.0, -1.0],
                    },
                    "x_m": {
                        "case_count": 2,
                        "detected_count": 2,
                        "supported_detection_count": 2,
                        "support_collapse_count": 0,
                        "amounts": [0.1, -0.1],
                    },
                    "y_m": {
                        "case_count": 2,
                        "detected_count": 2,
                        "supported_detection_count": 2,
                        "support_collapse_count": 0,
                        "amounts": [0.1, -0.1],
                    },
                    "z_m": {
                        "case_count": 2,
                        "detected_count": 2,
                        "supported_detection_count": 2,
                        "support_collapse_count": 0,
                        "amounts": [0.1, -0.1],
                    },
                },
                "min_support_ratio": 0.1,
                "min_accepted_correspondence_count": 500.0,
                "target_supported_detection_count": 8,
            },
        },
    }


def test_metric_registry_contains_autonomous_metrics() -> None:
    names = {definition.name for definition in list_metric_definitions()}
    assert "nuscenes_reference_extrinsic_count" in names
    assert "candidate_extrinsic_import_count" in names
    assert "extrinsic_reference_comparison_count" in names
    assert "extrinsic_reference_translation_delta_max_m" in names
    assert "extrinsic_reference_rotation_delta_max_deg" in names
    assert "radar_lidar_velocity_consistency" in names
    assert "native_joint_slac_available" in names
    assert "joint_slac_point_to_plane_rmse_m" in names
    assert "joint_slac_known_bad_detectable_fraction" in names
    assert "native_tum_joint_slac_available" in names
    assert "tum_joint_point_to_plane_rmse_m" in names
    assert "tum_joint_schur_solver_used" in names
    assert "tum_joint_schur_eliminated_dimension" in names
    assert "tum_joint_schur_retained_dimension" in names
    assert "tum_joint_schur_step_count" in names
    assert "tum_joint_schur_linear_residual_inf" in names
    assert "tum_joint_schur_complement_condition_number" in names
    assert "robot_world_hand_eye_pose_pair_count" in names
    assert "robot_world_hand_eye_shah_rotation_dominant_multiplicity" in names
    assert "robot_world_hand_eye_shah_rotation_normalized_gap" in names
    assert "robot_world_hand_eye_shah_translation_rank" in names
    assert "robot_world_hand_eye_shah_translation_condition_number" in names
    assert "robot_world_hand_eye_shah_so3_projection_correction_frobenius_max" in names
    assert "robot_world_hand_eye_shah_holdout_rotation_rmse_deg" in names
    assert "robot_world_hand_eye_shah_holdout_translation_rmse_m" in names
    assert "robot_world_hand_eye_shah_known_bad_detectable_fraction" in names
    assert "robot_world_hand_eye_li_linear_rank" in names
    assert "robot_world_hand_eye_li_linear_condition_number" in names
    assert "robot_world_hand_eye_li_raw_linear_residual_rmse" in names
    assert "robot_world_hand_eye_li_so3_projection_correction_frobenius_max" in names
    assert "robot_world_hand_eye_li_shah_common_split_consistent" in names
    assert "robot_world_hand_eye_li_holdout_rotation_rmse_deg" in names
    assert "robot_world_hand_eye_li_holdout_translation_rmse_m" in names
    assert "robot_world_hand_eye_li_known_bad_detectable_fraction" in names
    assert "robot_world_hand_eye_li_shah_x_rotation_delta_deg" in names
    assert "robot_world_hand_eye_li_shah_x_translation_delta_m" in names
    assert "robot_world_hand_eye_li_shah_z_rotation_delta_deg" in names
    assert "robot_world_hand_eye_li_shah_z_translation_delta_m" in names
    assert "robot_world_hand_eye_absolute_common_split_consistent" in names
    assert "robot_world_hand_eye_dornaika_horaud_rotation_dominant_multiplicity" in names
    assert "robot_world_hand_eye_dornaika_horaud_rotation_normalized_gap" in names
    assert "robot_world_hand_eye_dornaika_horaud_rotation_objective" in names
    assert "robot_world_hand_eye_dornaika_horaud_quaternion_unit_error_max" in names
    assert "robot_world_hand_eye_dornaika_horaud_sign_flip_count" in names
    assert "robot_world_hand_eye_dornaika_horaud_sign_synchronization_fraction" in names
    assert "robot_world_hand_eye_dornaika_horaud_translation_rank" in names
    assert "robot_world_hand_eye_dornaika_horaud_translation_condition_number" in names
    assert "robot_world_hand_eye_dornaika_horaud_holdout_rotation_rmse_deg" in names
    assert "robot_world_hand_eye_dornaika_horaud_holdout_translation_rmse_m" in names
    assert "robot_world_hand_eye_dornaika_horaud_known_bad_detectable_fraction" in names
    assert "robot_world_hand_eye_dornaika_horaud_shah_x_rotation_delta_deg" in names
    assert "robot_world_hand_eye_dornaika_horaud_shah_x_translation_delta_m" in names
    assert "robot_world_hand_eye_dornaika_horaud_shah_z_rotation_delta_deg" in names
    assert "robot_world_hand_eye_dornaika_horaud_shah_z_translation_delta_m" in names
    assert "robot_world_hand_eye_dornaika_horaud_nonlinear_objective_nonincrease" in names
    assert "robot_world_hand_eye_dornaika_horaud_nonlinear_accepted_step_count" in names
    assert "robot_world_hand_eye_dornaika_horaud_nonlinear_final_data_residual_rmse" in names
    assert "robot_world_hand_eye_dornaika_horaud_nonlinear_data_jacobian_rank" in names
    assert "robot_world_hand_eye_dornaika_horaud_nonlinear_data_jacobian_condition_number" in names
    assert (
        "robot_world_hand_eye_dornaika_horaud_nonlinear_orthogonality_error_frobenius_max" in names
    )
    assert (
        "robot_world_hand_eye_dornaika_horaud_nonlinear_so3_projection_correction_frobenius_max"
        in names
    )
    assert "robot_world_hand_eye_dornaika_horaud_nonlinear_holdout_rotation_rmse_deg" in names
    assert "robot_world_hand_eye_dornaika_horaud_nonlinear_holdout_translation_rmse_m" in names
    assert "robot_world_hand_eye_dornaika_horaud_nonlinear_known_bad_detectable_fraction" in names
    assert (
        "robot_world_hand_eye_dornaika_horaud_nonlinear_closed_form_x_rotation_delta_deg" in names
    )
    assert (
        "robot_world_hand_eye_dornaika_horaud_nonlinear_closed_form_x_translation_delta_m" in names
    )
    assert (
        "robot_world_hand_eye_dornaika_horaud_nonlinear_closed_form_z_rotation_delta_deg" in names
    )
    assert (
        "robot_world_hand_eye_dornaika_horaud_nonlinear_closed_form_z_translation_delta_m" in names
    )
    assert "tum_joint_data_only_extrinsic_rank" in names
    assert "tum_joint_data_only_shared_rank" in names
    assert "tum_joint_depth_scale" in names
    assert "tum_joint_depth_bias_m" in names
    assert "tum_joint_depth_scale_reference_error_percent" in names
    assert "tum_joint_depth_bias_reference_error_m" in names
    assert "tum_joint_reassociation_window_count" in names
    assert "tum_joint_reassociation_train_pair_jaccard_min" in names
    assert "tum_joint_reassociation_holdout_pair_jaccard_min" in names
    assert "tum_joint_reassociation_known_bad_detectable_fraction_min" in names
    assert "tum_joint_reassociation_aware_known_bad_detectable_fraction_min" in names
    assert "tum_joint_reassociation_aware_probe_valid_fraction_min" in names
    assert "tum_joint_reassociation_aware_support_collapse_fraction_max" in names
    assert "tum_joint_xyz_ablation_window_count" in names
    assert "tum_joint_xyz_data_only_field_rank_min" in names
    assert "tum_joint_xyz_data_only_shared_rank_min" in names
    assert "tum_joint_xyz_known_bad_detectable_fraction_min" in names
    assert "tum_joint_xyz_cross_window_max_holdout_delta_rmse_m" in names
    assert "tum_joint_xyz_cross_window_nondegrading_fraction" in names
    assert "lidar_frame_coverage" in names
    assert "lidar_point_coverage" in names
    assert "lidar_spatial_coverage_m" in names
    assert "lidar_pair_shared_voxel_count" in names
    assert "lidar_pair_unmatched_source_voxel_count" in names
    assert "lidar_pair_unmatched_target_voxel_count" in names
    assert "lidar_pair_source_voxel_recall_in_target" in names
    assert "lidar_pair_target_voxel_recall_in_source" in names
    assert "lidar_pair_shared_voxel_centroid_rmse_m" in names
    assert "lidar_pair_holdout_point_to_plane_map_voxel_count" in names
    assert "lidar_pair_holdout_point_to_plane_eligible_point_count" in names
    assert "lidar_pair_holdout_point_to_plane_considered_point_count" in names
    assert "lidar_pair_holdout_point_to_plane_accepted_correspondence_count" in names
    assert "lidar_pair_holdout_point_to_plane_support_ratio" in names
    assert "lidar_pair_holdout_point_to_plane_matched_point_count" in names
    assert "lidar_pair_holdout_point_to_plane_unmatched_fraction" in names
    assert "lidar_pair_holdout_point_to_plane_median_abs_m" in names
    assert "lidar_pair_holdout_point_to_plane_p90_abs_m" in names
    assert "lidar_pair_holdout_point_to_plane_rmse_m" in names
    assert "lidar_pair_holdout_point_to_plane_inlier_fraction" in names
    assert "lidar_pair_known_bad_case_count" in names
    assert "lidar_pair_known_bad_detectable_fraction" in names
    assert "lidar_pair_known_bad_source_recall_delta_mean" in names
    assert "lidar_pair_known_bad_centroid_rmse_delta_mean_m" in names
    assert "lidar_pair_known_bad_centroid_rmse_delta_max_m" in names
    assert "lidar_pair_known_bad_point_to_plane_p90_delta_max_m" in names
    assert "lidar_pair_known_bad_point_to_plane_rmse_delta_mean_m" in names
    assert "lidar_world_map_point_to_plane_rmse_m" in names
    assert "lidar_world_map_point_to_plane_median_holdout_m" in names
    assert "lidar_world_map_point_to_plane_p95_holdout_m" in names
    assert "lidar_world_map_train_voxel_count" in names
    assert "lidar_world_map_holdout_residual_count" in names
    assert "lidar_world_map_leakage_issue_count" in names
    assert "lidar_world_map_stability_window_count" in names
    assert "lidar_world_map_stability_scored_window_count" in names
    assert "lidar_world_map_stability_holdout_rmse_spread_m" in names
    assert "lidar_world_map_perturbation_case_count" in names
    assert "lidar_world_map_perturbation_detectable_fraction" in names
    assert "lidar_world_map_perturbation_train_rmse_delta_mean_m" in names
    assert "lidar_world_map_perturbation_holdout_rmse_delta_mean_m" in names
    assert "lidar_world_map_perturbation_holdout_rmse_delta_max_m" in names
    assert "lidar_world_map_perturbation_p95_holdout_delta_mean_m" in names
    assert "lidar_world_map_sensitivity_roll_m" in names
    assert "lidar_world_map_sensitivity_pitch_m" in names
    assert "lidar_world_map_sensitivity_yaw_m" in names
    assert "lidar_world_map_sensitivity_x_m" in names
    assert "lidar_world_map_sensitivity_y_m" in names
    assert "lidar_world_map_sensitivity_z_m" in names
    assert "lidar_world_map_weak_dof_count" in names
    assert "lidar_world_map_min_dof_sensitivity_m" in names
    assert "lidar_rig_point_to_plane_residual_count" in names
    assert "lidar_rig_point_to_plane_rmse_m" in names
    assert "lidar_rig_point_to_plane_rank" in names
    assert "lidar_rig_point_to_plane_condition_number" in names
    assert "lidar_rig_point_to_plane_normalization_length_m" in names
    assert "lidar_rig_point_to_plane_weak_dof_count" in names
    assert "lidar_perturbation_case_count" in names
    assert "lidar_perturbation_detectable_fraction" in names
    assert "lidar_perturbation_train_rmse_delta_mean_m" in names
    assert "lidar_perturbation_holdout_rmse_delta_mean_m" in names
    assert "lidar_perturbation_holdout_rmse_delta_max_m" in names
    assert "lidar_local_planarity" in names
    assert "lidar_map_roughness_m" in names
    assert "lidar_map_sharpness" in names
    assert "vehicle_motion_duration_s" in names
    assert "vehicle_yaw_excitation_deg" in names
    assert "camera_lidar_timestamp_alignment_ms" in names
    assert "camera_lidar_capture_time_input_available" in names
    assert "camera_lidar_capture_pair_count" in names
    assert "camera_lidar_capture_point_count" in names
    assert "camera_lidar_point_time_span_s" in names
    assert "camera_lidar_camera_stamp_delta_ms" in names
    assert "camera_lidar_point_to_exposure_delta_ms" in names
    assert "camera_lidar_deskew_displacement_rmse_m" in names
    assert "camera_lidar_time_sensitivity_mps" in names
    assert "camera_lidar_time_observability_rank" in names
    assert "camera_lidar_time_known_bad_detectable_fraction" in names
    assert "camera_lidar_time_offset_estimated" in names
    assert "lidar_oxts_timestamp_alignment_ms" in names
    assert "koide_lidar_camera_adapter_available" in names
    assert "koide_lidar_camera_execution_success" in names
    assert "lidar_camera_overlay_score" in names
    assert "lidar_camera_projection_frame_count" in names
    assert "lidar_camera_projected_points" in names
    assert "lidar_camera_projection_ratio" in names
    assert "lidar_camera_projection_depth_median_m" in names
    assert "lidar_camera_projection_depth_span_m" in names
    assert "lidar_camera_projection_horizontal_coverage" in names
    assert "lidar_camera_projection_vertical_coverage" in names
    assert "lidar_camera_edge_alignment_score" in names
    assert "lidar_camera_edge_gradient_mean" in names
    assert "lidar_camera_depth_discontinuity_points" in names
    assert "lidar_camera_depth_edge_alignment_score" in names
    assert "lidar_camera_depth_edge_gradient_mean" in names
    assert "lidar_camera_perturbation_case_count" in names
    assert "lidar_camera_perturbation_detectable_fraction" in names
    assert "lidar_camera_perturbation_edge_delta_mean" in names
    assert "lidar_camera_perturbation_depth_edge_delta_mean" in names
    assert "lidar_camera_perturbation_projection_ratio_delta_mean" in names
    assert "lidar_camera_mutual_information_score" in names
    assert "lidar_camera_comparison_candidate_count" in names
    assert "lidar_camera_capture_time_policy_declared" in names
    assert "lidar_camera_capture_time_deskew_applied" in names
    assert "point_plane_center_rmse_m" in names
    assert "point_plane_known_bad_detectable_fraction" in names
    assert "point_plane_joint_rank" in names
    assert "point_plane_vs_plane_rotation_delta_deg" in names
    assert "horn_point_rmse_m" in names
    assert "horn_point_known_bad_detectable_fraction" in names
    assert "horn_point_joint_rank" in names
    assert "horn_point_quaternion_normalized_eigengap" in names
    assert "horn_point_vs_point_plane_rotation_delta_deg" in names
    definition = get_metric_definition("autonomous_driving_dynamic_holdout")
    assert definition is not None
    assert definition.family == "autonomous_driving"


def test_lidar_metrics_from_kitti_inspection_diagnostics() -> None:
    inspection = DatasetInspection(
        dataset_type="kitti_raw",
        path="/tmp/kitti",
        exists=True,
        diagnostics={
            "velodyne_points": {
                "frame_count": 12,
                "sampled_point_count": 25_000,
                "bounds_min_m": [-5.0, -3.0, -1.0],
                "bounds_max_m": [15.0, 7.0, 2.0],
                "local_planarity_mean": 0.8,
                "roughness_mean_m": 0.03,
                "map_sharpness_score": 0.77,
                "point_to_plane_rmse_train_m": 0.02,
                "point_to_plane_rmse_holdout_m": 0.04,
            },
            "lidar_world_map_consistency": {
                "status": "scored",
                "train_voxel_count": 8,
                "holdout_residual_count": 24,
                "point_to_plane_rmse_train_m": 0.03,
                "point_to_plane_rmse_holdout_m": 0.05,
                "point_to_plane_median_holdout_m": 0.02,
                "point_to_plane_p95_holdout_m": 0.12,
            },
        },
    )
    metrics = lidar_metrics_from_inspection(inspection)
    assert metrics["lidar_frame_coverage"].value == 12
    assert metrics["lidar_point_coverage"].value == 25_000
    assert round(metrics["lidar_spatial_coverage_m"].value or 0.0, 3) == 22.561
    assert metrics["lidar_local_planarity"].value == 0.8
    assert metrics["lidar_map_roughness_m"].value == 0.03
    assert metrics["lidar_map_sharpness"].value == 0.77
    assert metrics["lidar_point_to_plane_rmse_m"].train == 0.02
    assert metrics["lidar_point_to_plane_rmse_m"].holdout == 0.04
    assert metrics["lidar_world_map_train_voxel_count"].value == 8
    assert metrics["lidar_world_map_holdout_residual_count"].value == 24
    assert metrics["lidar_world_map_point_to_plane_rmse_m"].train == 0.03
    assert metrics["lidar_world_map_point_to_plane_rmse_m"].holdout == 0.05
    assert metrics["lidar_world_map_point_to_plane_median_holdout_m"].value == 0.02
    assert metrics["lidar_world_map_point_to_plane_p95_holdout_m"].value == 0.12


def test_lidar_metrics_from_livox_pcd_inspection_diagnostics() -> None:
    inspection = DatasetInspection(
        dataset_type="livox_pcd",
        path="data/public/livox_horizon_horizon_pair",
        exists=True,
        diagnostics={
            "livox_pcd": {
                "sampled_file_count": 2,
                "sampled_point_count": 64000,
                "bounds_min_m": [0.0, -4.0, -2.0],
                "bounds_max_m": [20.0, 5.0, 3.0],
                "pair_shared_voxel_count": 42,
                "pair_unmatched_source_voxel_count": 108,
                "pair_unmatched_target_voxel_count": 96,
                "pair_source_voxel_recall_in_target": 0.28,
                "pair_target_voxel_recall_in_source": 0.304,
                "pair_shared_voxel_centroid_rmse_m": 0.44,
            }
        },
    )

    metrics = lidar_metrics_from_inspection(inspection)

    assert metrics["lidar_frame_coverage"].value == 2.0
    assert metrics["lidar_point_coverage"].value == 64000.0
    assert metrics["lidar_pair_shared_voxel_count"].value == 42.0
    assert metrics["lidar_pair_unmatched_source_voxel_count"].value == 108.0
    assert metrics["lidar_pair_unmatched_target_voxel_count"].value == 96.0
    assert metrics["lidar_pair_source_voxel_recall_in_target"].value == 0.28
    assert metrics["lidar_pair_target_voxel_recall_in_source"].value == 0.304
    assert metrics["lidar_pair_shared_voxel_centroid_rmse_m"].value == 0.44


def test_autonomous_driving_lidar_coverage_thresholds_are_stricter() -> None:
    default_metrics = {"lidar_frame_coverage": MetricResult(value=20)}
    auto_metrics = {"lidar_frame_coverage": MetricResult(value=20)}
    apply_metric_thresholds(default_metrics, "default")
    apply_metric_thresholds(auto_metrics, "autonomous_driving")
    assert default_metrics["lidar_frame_coverage"].grade == "pass"
    assert auto_metrics["lidar_frame_coverage"].grade == "warn"


def test_motion_metrics_from_kitti_oxts_diagnostics() -> None:
    inspection = DatasetInspection(
        dataset_type="kitti_raw",
        path="/tmp/kitti",
        exists=True,
        diagnostics={
            "oxts_motion": {
                "duration_sec": 12.0,
                "mean_speed_mps": 3.0,
                "speed_range_mps": 2.5,
                "yaw_excitation_deg": 16.0,
                "pitch_excitation_deg": 1.2,
                "roll_excitation_deg": 0.8,
                "mean_acceleration_norm_mps2": 0.4,
            }
        },
    )
    metrics = motion_metrics_from_inspection(inspection)
    assert metrics["vehicle_motion_duration_s"].value == 12.0
    assert metrics["vehicle_mean_speed_mps"].value == 3.0
    assert metrics["vehicle_yaw_excitation_deg"].value == 16.0


def test_timing_metrics_from_kitti_timestamp_alignment_diagnostics() -> None:
    inspection = DatasetInspection(
        dataset_type="kitti_raw",
        path="/tmp/kitti",
        exists=True,
        diagnostics={
            "timestamp_alignment": {
                "camera_lidar_max_abs_dt_ms": 12.0,
                "lidar_oxts_max_abs_dt_ms": 18.0,
            }
        },
    )
    metrics = timing_metrics_from_inspection(inspection)
    assert metrics["camera_lidar_timestamp_alignment_ms"].value == 12.0
    assert metrics["lidar_oxts_timestamp_alignment_ms"].value == 18.0


def test_lidar_camera_metrics_from_result_scores_overlay_readiness() -> None:
    config = CalibrationConfig.model_validate(
        {
            "schema_version": "slac.config/v0.1",
            "dataset": {"type": "kitti_raw", "path": "/tmp/kitti"},
            "sensors": {
                "camera0": {"type": "camera"},
                "lidar0": {"type": "lidar"},
            },
            "frames": {
                "base": {"root": True},
                "camera0": {"parent": "base"},
                "lidar0": {"parent": "base"},
            },
            "pipeline": {
                "factors": {
                    "lidar_camera_mutual_information": {"enabled": True},
                }
            },
        }
    )
    result = CalibrationResult(
        run=RunInfo(id="unit", slac_version="0.1.0"),
        frame_graph=FrameGraphSnapshot(
            root="base",
            frames={"base": None, "camera0": "base", "lidar0": "base"},
        ),
        transforms={
            "T_base_camera0": TransformResult(
                parent="base",
                child="camera0",
                translation_m=[0.0, 0.0, 0.0],
                rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            ),
            "T_base_lidar0": TransformResult(
                parent="base",
                child="lidar0",
                translation_m=[1.0, 0.0, 0.0],
                rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            ),
        },
        metrics={"lidar_point_coverage": MetricResult(value=200_000.0)},
    )
    inspection = DatasetInspection(
        dataset_type="kitti_raw",
        path="/tmp/kitti",
        exists=True,
        streams=[
            StreamSummary("camera_left_color", "image", 20, sensor="camera0"),
            StreamSummary("velodyne_points", "pointcloud", 20, sensor="lidar0"),
        ],
        diagnostics={
            "timestamp_alignment": {
                "camera_lidar_pair_count": 20,
                "camera_lidar_max_abs_dt_ms": 5.0,
                "camera_timestamp_count": 20,
                "lidar_timestamp_count": 20,
            }
        },
    )

    metrics = lidar_camera_metrics_from_result(config, result, inspection)
    assert metrics["lidar_camera_transform_pairs"].value == 1.0
    assert metrics["lidar_camera_overlay_readiness"].value == 1.0
    assert (metrics["lidar_camera_overlay_score"].value or 0.0) > 0.9
    assert metrics["lidar_camera_projected_points"].grade == "warn"
    assert metrics["lidar_camera_mutual_information_score"].grade == "warn"


def test_lidar_camera_metrics_fail_without_temporal_pairs() -> None:
    config = CalibrationConfig.model_validate(
        {
            "schema_version": "slac.config/v0.1",
            "dataset": {"type": "kitti_raw", "path": "/tmp/kitti"},
            "sensors": {
                "camera0": {"type": "camera"},
                "lidar0": {"type": "lidar"},
            },
            "frames": {
                "base": {"root": True},
                "camera0": {"parent": "base"},
                "lidar0": {"parent": "base"},
            },
            "evaluation": {"metrics": ["lidar_camera_overlay_readiness"]},
        }
    )
    result = CalibrationResult(
        run=RunInfo(id="unit", slac_version="0.1.0"),
        frame_graph=FrameGraphSnapshot(
            root="base",
            frames={"base": None, "camera0": "base", "lidar0": "base"},
        ),
        transforms={
            "T_base_camera0": TransformResult(
                parent="base",
                child="camera0",
                translation_m=[0.0, 0.0, 0.0],
                rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            ),
            "T_base_lidar0": TransformResult(
                parent="base",
                child="lidar0",
                translation_m=[1.0, 0.0, 0.0],
                rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            ),
        },
        metrics={"lidar_point_coverage": MetricResult(value=200_000.0)},
    )
    inspection = DatasetInspection(
        dataset_type="kitti_raw",
        path="/tmp/kitti",
        exists=True,
        streams=[
            StreamSummary("camera_left_color", "image", 20, sensor="camera0"),
            StreamSummary("velodyne_points", "pointcloud", 20, sensor="lidar0"),
        ],
        diagnostics={"timestamp_alignment": {"camera_lidar_pair_count": 0}},
    )

    metrics = lidar_camera_metrics_from_result(config, result, inspection)
    assert metrics["lidar_camera_overlay_readiness"].grade == "fail"
    assert metrics["lidar_camera_overlay_score"].value == 0.0


def test_lidar_degeneracy_passes_for_rich_planar_structure() -> None:
    inspection = DatasetInspection(
        dataset_type="kitti_raw",
        path="/tmp/kitti",
        exists=True,
        diagnostics={
            "velodyne_points": {
                "frame_count": 120,
                "sampled_point_count": 300_000,
                "planarity_voxel_count": 20,
                "bounds_min_m": [-20.0, -10.0, -2.0],
                "bounds_max_m": [40.0, 10.0, 3.0],
                "local_planarity_mean": 0.6,
                "map_sharpness_score": 0.4,
                "point_to_plane_rmse_train_m": 0.02,
                "point_to_plane_rmse_holdout_m": 0.03,
            },
            "oxts_motion": {
                "packet_count": 120,
                "duration_sec": 35.0,
                "mean_speed_mps": 8.0,
                "speed_range_mps": 6.0,
                "yaw_excitation_deg": 45.0,
            },
            "timestamp_alignment": {
                "camera_lidar_pair_count": 120,
                "camera_lidar_max_abs_dt_ms": 5.0,
                "lidar_oxts_pair_count": 120,
                "lidar_oxts_max_abs_dt_ms": 5.0,
            },
        },
    )
    degeneracy = degeneracy_from_inspection(inspection)
    assert degeneracy.grade == "pass"
    assert degeneracy.reason is None


def test_lidar_degeneracy_fails_for_missing_points() -> None:
    inspection = DatasetInspection(
        dataset_type="kitti_raw",
        path="/tmp/kitti",
        exists=True,
        diagnostics={
            "velodyne_points": {
                "frame_count": 0,
                "sampled_point_count": 0,
                "planarity_voxel_count": 0,
            }
        },
    )
    degeneracy = degeneracy_from_inspection(inspection)
    assert degeneracy.grade == "fail"
    assert "no LiDAR frames" in (degeneracy.reason or "")


def test_lidar_degeneracy_warns_for_weak_geometry_and_holdout_gap() -> None:
    inspection = DatasetInspection(
        dataset_type="kitti_raw",
        path="/tmp/kitti",
        exists=True,
        diagnostics={
            "velodyne_points": {
                "frame_count": 20,
                "sampled_point_count": 50_000,
                "planarity_voxel_count": 2,
                "bounds_min_m": [-5.0, -5.0, 0.0],
                "bounds_max_m": [5.0, 5.0, 0.2],
                "local_planarity_mean": 0.08,
                "map_sharpness_score": 0.03,
                "point_to_plane_rmse_train_m": 0.02,
                "point_to_plane_rmse_holdout_m": 0.08,
            },
            "oxts_motion": {
                "packet_count": 2,
                "duration_sec": 2.0,
                "mean_speed_mps": 0.2,
                "speed_range_mps": 0.1,
                "yaw_excitation_deg": 0.5,
            },
            "timestamp_alignment": {
                "camera_lidar_pair_count": 2,
                "camera_lidar_max_abs_dt_ms": 120.0,
                "lidar_oxts_pair_count": 2,
                "lidar_oxts_max_abs_dt_ms": 80.0,
            },
        },
    )
    degeneracy = degeneracy_from_inspection(inspection)
    assert degeneracy.grade == "warn"
    assert "too few local planar" in (degeneracy.reason or "")
    assert "z_extent" in (degeneracy.reason or "")
    assert "holdout" in (degeneracy.reason or "")
    assert "limited yaw excitation" in (degeneracy.reason or "")
    assert "timestamp mismatch" in (degeneracy.reason or "")


def test_evaluate_quality_generates_lidar_collection_recommendations() -> None:
    result = CalibrationResult(
        run=RunInfo(id="unit", slac_version="0.1.0", domain="autonomous_driving"),
        frame_graph=FrameGraphSnapshot(root="base", frames={"base": None}),
        metrics={
            "lidar_frame_coverage": MetricResult(value=5.0),
            "lidar_local_planarity": MetricResult(value=0.08),
            "lidar_map_sharpness": MetricResult(value=0.03),
            "lidar_point_to_plane_rmse_m": MetricResult(train=0.02, holdout=0.08),
            "vehicle_motion_duration_s": MetricResult(value=2.0),
            "vehicle_mean_speed_mps": MetricResult(value=0.2),
            "vehicle_speed_range_mps": MetricResult(value=0.1),
            "vehicle_yaw_excitation_deg": MetricResult(value=0.5),
            "camera_lidar_timestamp_alignment_ms": MetricResult(value=120.0),
            "lidar_world_map_weak_dof_count": MetricResult(value=2.0, grade="warn"),
        },
        observability=ObservabilityResult(grade="pass"),
        degeneracy=DegeneracyResult(
            grade="warn",
            reason=(
                "too few local planar LiDAR neighborhoods; "
                "limited vertical LiDAR structure: z_extent=0.2 m; "
                "LiDAR holdout point-to-plane RMSE is much worse than train: ratio=4; "
                "limited yaw excitation: yaw_range=0.5 deg; "
                "large camera-LiDAR timestamp mismatch: 120 ms; "
                "weak LiDAR world-map DoF from perturbation sensitivity: z_lidar0"
            ),
        ),
    )
    evaluated = evaluate_quality(result)
    recommendations = evaluated.quality.recommendation
    assert any("longer fixed-LiDAR sequence" in item for item in recommendations)
    assert any("vertical structure" in item for item in recommendations)
    assert any("static planar surfaces" in item for item in recommendations)
    assert any("dynamic traffic out of holdout" in item for item in recommendations)
    assert any("longer driving segment" in item for item in recommendations)
    assert any("acceleration, braking" in item for item in recommendations)
    assert any("turns" in item for item in recommendations)
    assert any("timestamp synchronization" in item for item in recommendations)
    assert any("richer turns, grade changes" in item for item in recommendations)
    assert any("camera-lidar-radar timing" in item for item in recommendations)


def test_evaluate_quality_keeps_archive_recommendation_for_clean_result() -> None:
    result = CalibrationResult(
        run=RunInfo(id="unit", slac_version="0.1.0"),
        frame_graph=FrameGraphSnapshot(root="base", frames={"base": None}),
        observability=ObservabilityResult(grade="pass"),
        degeneracy=DegeneracyResult(grade="pass"),
    )
    evaluated = evaluate_quality(result)
    assert evaluated.quality.grade == "pass"
    assert evaluated.quality.recommendation == [
        "archive config, result, report, and dataset manifest for reproducibility"
    ]


def test_holdout_split_is_deterministic() -> None:
    first = split_indices(count=10, holdout_ratio=0.2, seed=7)
    second = split_indices(count=10, holdout_ratio=0.2, seed=7)
    assert first == second
    assert len(first[1]) == 2
    assert sorted(first[0] + first[1]) == list(range(10))
