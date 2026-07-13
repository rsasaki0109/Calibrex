"""Metric thresholds and grading profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from calibrex.core.result import Grade, MetricResult

ThresholdDirection = Literal["lower_is_better", "higher_is_better"]
ThresholdProfile = Literal["default", "autonomous_driving"]


@dataclass(frozen=True)
class MetricThreshold:
    """Threshold rule for a scalar metric."""

    metric: str
    pass_value: float
    warn_value: float
    direction: ThresholdDirection = "lower_is_better"
    unit: str | None = None

    def grade(self, value: float) -> Grade:
        """Grade a value with this threshold."""

        if self.direction == "lower_is_better":
            if value <= self.pass_value:
                return "pass"
            if value <= self.warn_value:
                return "warn"
            return "fail"
        if value >= self.pass_value:
            return "pass"
        if value >= self.warn_value:
            return "warn"
        return "fail"

    def reason(self, value: float, grade: Grade) -> str:
        """Return a user-facing grading reason."""

        unit = f" {self.unit}" if self.unit else ""
        if self.direction == "lower_is_better":
            return (
                f"{self.metric}={value:g}{unit}; pass <= {self.pass_value:g}{unit}, "
                f"warn <= {self.warn_value:g}{unit}"
            )
        return (
            f"{self.metric}={value:g}{unit}; pass >= {self.pass_value:g}{unit}, "
            f"warn >= {self.warn_value:g}{unit}"
        )


DEFAULT_THRESHOLDS: dict[str, MetricThreshold] = {
    "schema_validation": MetricThreshold(
        metric="schema_validation",
        pass_value=1.0,
        warn_value=0.5,
        direction="higher_is_better",
    ),
    "dataset_exists": MetricThreshold(
        metric="dataset_exists",
        pass_value=1.0,
        warn_value=0.5,
        direction="higher_is_better",
    ),
    "nuscenes_reference_extrinsic_count": MetricThreshold(
        metric="nuscenes_reference_extrinsic_count",
        pass_value=1.0,
        warn_value=0.0,
        direction="higher_is_better",
        unit="transforms",
    ),
    "extrinsic_reference_comparison_count": MetricThreshold(
        metric="extrinsic_reference_comparison_count",
        pass_value=1.0,
        warn_value=0.0,
        direction="higher_is_better",
        unit="pairs",
    ),
    "candidate_extrinsic_import_count": MetricThreshold(
        metric="candidate_extrinsic_import_count",
        pass_value=1.0,
        warn_value=0.0,
        direction="higher_is_better",
        unit="transforms",
    ),
    "extrinsic_reference_translation_delta_max_m": MetricThreshold(
        metric="extrinsic_reference_translation_delta_max_m",
        pass_value=0.05,
        warn_value=0.20,
        unit="m",
    ),
    "extrinsic_reference_rotation_delta_max_deg": MetricThreshold(
        metric="extrinsic_reference_rotation_delta_max_deg",
        pass_value=1.0,
        warn_value=5.0,
        unit="deg",
    ),
    "reprojection_rmse_px": MetricThreshold(
        metric="reprojection_rmse_px",
        pass_value=1.0,
        warn_value=2.0,
        unit="px",
    ),
    "lidar_point_to_plane_rmse_m": MetricThreshold(
        metric="lidar_point_to_plane_rmse_m",
        pass_value=0.03,
        warn_value=0.08,
        unit="m",
    ),
    "lidar_world_map_point_to_plane_rmse_m": MetricThreshold(
        metric="lidar_world_map_point_to_plane_rmse_m",
        pass_value=0.10,
        warn_value=0.30,
        unit="m",
    ),
    "lidar_world_map_point_to_plane_p95_holdout_m": MetricThreshold(
        metric="lidar_world_map_point_to_plane_p95_holdout_m",
        pass_value=0.30,
        warn_value=1.00,
        unit="m",
    ),
    "lidar_world_map_train_voxel_count": MetricThreshold(
        metric="lidar_world_map_train_voxel_count",
        pass_value=5.0,
        warn_value=1.0,
        direction="higher_is_better",
        unit="voxels",
    ),
    "lidar_world_map_holdout_residual_count": MetricThreshold(
        metric="lidar_world_map_holdout_residual_count",
        pass_value=20.0,
        warn_value=1.0,
        direction="higher_is_better",
        unit="points",
    ),
    "lidar_world_map_perturbation_case_count": MetricThreshold(
        metric="lidar_world_map_perturbation_case_count",
        pass_value=12.0,
        warn_value=1.0,
        direction="higher_is_better",
        unit="cases",
    ),
    "lidar_world_map_perturbation_detectable_fraction": MetricThreshold(
        metric="lidar_world_map_perturbation_detectable_fraction",
        pass_value=0.50,
        warn_value=0.10,
        direction="higher_is_better",
    ),
    "lidar_world_map_weak_dof_count": MetricThreshold(
        metric="lidar_world_map_weak_dof_count",
        pass_value=0.0,
        warn_value=6.0,
        unit="dof",
    ),
    "lidar_world_map_min_dof_sensitivity_m": MetricThreshold(
        metric="lidar_world_map_min_dof_sensitivity_m",
        pass_value=0.001,
        warn_value=0.0001,
        direction="higher_is_better",
        unit="m",
    ),
    "lidar_perturbation_case_count": MetricThreshold(
        metric="lidar_perturbation_case_count",
        pass_value=12.0,
        warn_value=1.0,
        direction="higher_is_better",
        unit="cases",
    ),
    "lidar_perturbation_detectable_fraction": MetricThreshold(
        metric="lidar_perturbation_detectable_fraction",
        pass_value=0.50,
        warn_value=0.10,
        direction="higher_is_better",
    ),
    "lidar_frame_coverage": MetricThreshold(
        metric="lidar_frame_coverage",
        pass_value=20.0,
        warn_value=1.0,
        direction="higher_is_better",
        unit="frames",
    ),
    "lidar_point_coverage": MetricThreshold(
        metric="lidar_point_coverage",
        pass_value=50_000.0,
        warn_value=1_000.0,
        direction="higher_is_better",
        unit="points",
    ),
    "lidar_spatial_coverage_m": MetricThreshold(
        metric="lidar_spatial_coverage_m",
        pass_value=20.0,
        warn_value=5.0,
        direction="higher_is_better",
        unit="m",
    ),
    "lidar_local_planarity": MetricThreshold(
        metric="lidar_local_planarity",
        pass_value=0.30,
        warn_value=0.10,
        direction="higher_is_better",
    ),
    "lidar_map_roughness_m": MetricThreshold(
        metric="lidar_map_roughness_m",
        pass_value=0.15,
        warn_value=0.40,
        unit="m",
    ),
    "lidar_map_sharpness": MetricThreshold(
        metric="lidar_map_sharpness",
        pass_value=0.20,
        warn_value=0.05,
        direction="higher_is_better",
    ),
    "lidar_pair_shared_voxel_count": MetricThreshold(
        metric="lidar_pair_shared_voxel_count",
        pass_value=20.0,
        warn_value=1.0,
        direction="higher_is_better",
        unit="voxels",
    ),
    "lidar_pair_source_voxel_recall_in_target": MetricThreshold(
        metric="lidar_pair_source_voxel_recall_in_target",
        pass_value=0.20,
        warn_value=0.05,
        direction="higher_is_better",
    ),
    "lidar_pair_target_voxel_recall_in_source": MetricThreshold(
        metric="lidar_pair_target_voxel_recall_in_source",
        pass_value=0.20,
        warn_value=0.05,
        direction="higher_is_better",
    ),
    "lidar_pair_shared_voxel_centroid_rmse_m": MetricThreshold(
        metric="lidar_pair_shared_voxel_centroid_rmse_m",
        pass_value=0.75,
        warn_value=1.50,
        unit="m",
    ),
    "lidar_pair_holdout_point_to_plane_map_voxel_count": MetricThreshold(
        metric="lidar_pair_holdout_point_to_plane_map_voxel_count",
        pass_value=20.0,
        warn_value=1.0,
        direction="higher_is_better",
        unit="voxels",
    ),
    "lidar_pair_holdout_point_to_plane_eligible_point_count": MetricThreshold(
        metric="lidar_pair_holdout_point_to_plane_eligible_point_count",
        pass_value=500.0,
        warn_value=20.0,
        direction="higher_is_better",
        unit="points",
    ),
    "lidar_pair_holdout_point_to_plane_considered_point_count": MetricThreshold(
        metric="lidar_pair_holdout_point_to_plane_considered_point_count",
        pass_value=500.0,
        warn_value=20.0,
        direction="higher_is_better",
        unit="points",
    ),
    "lidar_pair_holdout_point_to_plane_accepted_correspondence_count": MetricThreshold(
        metric="lidar_pair_holdout_point_to_plane_accepted_correspondence_count",
        pass_value=500.0,
        warn_value=20.0,
        direction="higher_is_better",
        unit="points",
    ),
    "lidar_pair_holdout_point_to_plane_support_ratio": MetricThreshold(
        metric="lidar_pair_holdout_point_to_plane_support_ratio",
        pass_value=0.20,
        warn_value=0.01,
        direction="higher_is_better",
    ),
    "lidar_pair_holdout_point_to_plane_matched_point_count": MetricThreshold(
        metric="lidar_pair_holdout_point_to_plane_matched_point_count",
        pass_value=500.0,
        warn_value=20.0,
        direction="higher_is_better",
        unit="points",
    ),
    "lidar_pair_holdout_point_to_plane_unmatched_fraction": MetricThreshold(
        metric="lidar_pair_holdout_point_to_plane_unmatched_fraction",
        pass_value=0.90,
        warn_value=0.98,
    ),
    "lidar_pair_holdout_point_to_plane_median_abs_m": MetricThreshold(
        metric="lidar_pair_holdout_point_to_plane_median_abs_m",
        pass_value=0.30,
        warn_value=0.75,
        unit="m",
    ),
    "lidar_pair_holdout_point_to_plane_p90_abs_m": MetricThreshold(
        metric="lidar_pair_holdout_point_to_plane_p90_abs_m",
        pass_value=1.00,
        warn_value=2.00,
        unit="m",
    ),
    "lidar_pair_holdout_point_to_plane_rmse_m": MetricThreshold(
        metric="lidar_pair_holdout_point_to_plane_rmse_m",
        pass_value=0.75,
        warn_value=1.50,
        unit="m",
    ),
    "lidar_pair_holdout_point_to_plane_inlier_fraction": MetricThreshold(
        metric="lidar_pair_holdout_point_to_plane_inlier_fraction",
        pass_value=0.20,
        warn_value=0.01,
        direction="higher_is_better",
    ),
    "lidar_pair_known_bad_case_count": MetricThreshold(
        metric="lidar_pair_known_bad_case_count",
        pass_value=1.0,
        warn_value=0.0,
        direction="higher_is_better",
        unit="cases",
    ),
    "lidar_pair_known_bad_detectable_fraction": MetricThreshold(
        metric="lidar_pair_known_bad_detectable_fraction",
        pass_value=0.50,
        warn_value=0.10,
        direction="higher_is_better",
    ),
    "lidar_pair_known_bad_mandatory_case_count": MetricThreshold(
        metric="lidar_pair_known_bad_mandatory_case_count",
        pass_value=12.0,
        warn_value=1.0,
        direction="higher_is_better",
        unit="cases",
    ),
    "lidar_pair_known_bad_mandatory_supported_detection_count": MetricThreshold(
        metric="lidar_pair_known_bad_mandatory_supported_detection_count",
        pass_value=8.0,
        warn_value=1.0,
        direction="higher_is_better",
        unit="cases",
    ),
    "lidar_pair_known_bad_mandatory_supported_detection_fraction": MetricThreshold(
        metric="lidar_pair_known_bad_mandatory_supported_detection_fraction",
        pass_value=0.66,
        warn_value=0.10,
        direction="higher_is_better",
    ),
    "lidar_pair_known_bad_mandatory_support_collapse_count": MetricThreshold(
        metric="lidar_pair_known_bad_mandatory_support_collapse_count",
        pass_value=0.0,
        warn_value=4.0,
        unit="cases",
    ),
    "lidar_pair_known_bad_source_recall_delta_mean": MetricThreshold(
        metric="lidar_pair_known_bad_source_recall_delta_mean",
        pass_value=0.0,
        warn_value=-0.05,
        direction="higher_is_better",
    ),
    "lidar_pair_known_bad_centroid_rmse_delta_mean_m": MetricThreshold(
        metric="lidar_pair_known_bad_centroid_rmse_delta_mean_m",
        pass_value=0.0,
        warn_value=-0.05,
        direction="higher_is_better",
        unit="m",
    ),
    "lidar_pair_known_bad_centroid_rmse_delta_max_m": MetricThreshold(
        metric="lidar_pair_known_bad_centroid_rmse_delta_max_m",
        pass_value=0.0,
        warn_value=-0.05,
        direction="higher_is_better",
        unit="m",
    ),
    "lidar_pair_known_bad_point_to_plane_p90_delta_max_m": MetricThreshold(
        metric="lidar_pair_known_bad_point_to_plane_p90_delta_max_m",
        pass_value=0.0,
        warn_value=-0.05,
        direction="higher_is_better",
        unit="m",
    ),
    "lidar_pair_known_bad_point_to_plane_rmse_delta_mean_m": MetricThreshold(
        metric="lidar_pair_known_bad_point_to_plane_rmse_delta_mean_m",
        pass_value=0.0,
        warn_value=-0.05,
        direction="higher_is_better",
        unit="m",
    ),
    "trajectory_consistency": MetricThreshold(
        metric="trajectory_consistency",
        pass_value=0.05,
        warn_value=0.15,
        unit="m",
    ),
    "vehicle_motion_duration_s": MetricThreshold(
        metric="vehicle_motion_duration_s",
        pass_value=10.0,
        warn_value=3.0,
        direction="higher_is_better",
        unit="s",
    ),
    "vehicle_mean_speed_mps": MetricThreshold(
        metric="vehicle_mean_speed_mps",
        pass_value=2.0,
        warn_value=0.5,
        direction="higher_is_better",
        unit="m/s",
    ),
    "vehicle_speed_range_mps": MetricThreshold(
        metric="vehicle_speed_range_mps",
        pass_value=2.0,
        warn_value=0.5,
        direction="higher_is_better",
        unit="m/s",
    ),
    "vehicle_yaw_excitation_deg": MetricThreshold(
        metric="vehicle_yaw_excitation_deg",
        pass_value=15.0,
        warn_value=3.0,
        direction="higher_is_better",
        unit="deg",
    ),
    "vehicle_pitch_excitation_deg": MetricThreshold(
        metric="vehicle_pitch_excitation_deg",
        pass_value=1.0,
        warn_value=0.2,
        direction="higher_is_better",
        unit="deg",
    ),
    "vehicle_roll_excitation_deg": MetricThreshold(
        metric="vehicle_roll_excitation_deg",
        pass_value=1.0,
        warn_value=0.2,
        direction="higher_is_better",
        unit="deg",
    ),
    "vehicle_mean_acceleration_norm_mps2": MetricThreshold(
        metric="vehicle_mean_acceleration_norm_mps2",
        pass_value=0.5,
        warn_value=0.1,
        direction="higher_is_better",
        unit="m/s^2",
    ),
    "camera_lidar_timestamp_alignment_ms": MetricThreshold(
        metric="camera_lidar_timestamp_alignment_ms",
        pass_value=20.0,
        warn_value=100.0,
        unit="ms",
    ),
    "lidar_oxts_timestamp_alignment_ms": MetricThreshold(
        metric="lidar_oxts_timestamp_alignment_ms",
        pass_value=20.0,
        warn_value=100.0,
        unit="ms",
    ),
    "condition_number": MetricThreshold(
        metric="condition_number",
        pass_value=1.0e6,
        warn_value=1.0e8,
    ),
    "open3d_slac_backend_available": MetricThreshold(
        metric="open3d_slac_backend_available",
        pass_value=1.0,
        warn_value=0.0,
        direction="higher_is_better",
    ),
    "native_joint_slac_available": MetricThreshold(
        metric="native_joint_slac_available",
        pass_value=1.0,
        warn_value=0.0,
        direction="higher_is_better",
    ),
    "native_tum_joint_slac_available": MetricThreshold(
        metric="native_tum_joint_slac_available",
        pass_value=1.0,
        warn_value=0.0,
        direction="higher_is_better",
    ),
    "tum_joint_map_frame_count": MetricThreshold(
        metric="tum_joint_map_frame_count",
        pass_value=3.0,
        warn_value=1.0,
        direction="higher_is_better",
        unit="frames",
    ),
    "tum_joint_query_frame_count": MetricThreshold(
        metric="tum_joint_query_frame_count",
        pass_value=8.0,
        warn_value=4.0,
        direction="higher_is_better",
        unit="frames",
    ),
    "tum_joint_correspondence_count": MetricThreshold(
        metric="tum_joint_correspondence_count",
        pass_value=1000.0,
        warn_value=100.0,
        direction="higher_is_better",
        unit="correspondences",
    ),
    "tum_joint_point_to_plane_rmse_m": MetricThreshold(
        metric="tum_joint_point_to_plane_rmse_m",
        pass_value=0.05,
        warn_value=0.10,
        unit="m",
    ),
    "tum_joint_data_only_extrinsic_rank": MetricThreshold(
        metric="tum_joint_data_only_extrinsic_rank",
        pass_value=6.0,
        warn_value=5.0,
        direction="higher_is_better",
    ),
    "tum_joint_data_only_extrinsic_condition_number": MetricThreshold(
        metric="tum_joint_data_only_extrinsic_condition_number",
        pass_value=1.0e6,
        warn_value=1.0e8,
    ),
    "tum_joint_data_only_shared_rank": MetricThreshold(
        metric="tum_joint_data_only_shared_rank",
        pass_value=8.0,
        warn_value=7.0,
        direction="higher_is_better",
    ),
    "tum_joint_data_only_shared_condition_number": MetricThreshold(
        metric="tum_joint_data_only_shared_condition_number",
        pass_value=1.0e6,
        warn_value=1.0e8,
    ),
    "tum_joint_depth_scale_reference_error_percent": MetricThreshold(
        metric="tum_joint_depth_scale_reference_error_percent",
        pass_value=2.0,
        warn_value=5.0,
        unit="percent",
    ),
    "tum_joint_depth_bias_reference_error_m": MetricThreshold(
        metric="tum_joint_depth_bias_reference_error_m",
        pass_value=0.03,
        warn_value=0.06,
        unit="m",
    ),
    "tum_joint_known_bad_detectable_fraction": MetricThreshold(
        metric="tum_joint_known_bad_detectable_fraction",
        pass_value=1.0,
        warn_value=0.5,
        direction="higher_is_better",
    ),
    "tum_joint_identity_reference_translation_error_m": MetricThreshold(
        metric="tum_joint_identity_reference_translation_error_m",
        pass_value=0.05,
        warn_value=0.10,
        unit="m",
    ),
    "tum_joint_identity_reference_rotation_error_deg": MetricThreshold(
        metric="tum_joint_identity_reference_rotation_error_deg",
        pass_value=1.0,
        warn_value=2.0,
        unit="deg",
    ),
    "joint_slac_augmented_information_rank": MetricThreshold(
        metric="joint_slac_augmented_information_rank",
        pass_value=12.0,
        warn_value=6.0,
        direction="higher_is_better",
    ),
    "joint_slac_augmented_condition_number": MetricThreshold(
        metric="joint_slac_augmented_condition_number",
        pass_value=1.0e6,
        warn_value=1.0e8,
    ),
    "joint_slac_known_bad_detectable_fraction": MetricThreshold(
        metric="joint_slac_known_bad_detectable_fraction",
        pass_value=1.0,
        warn_value=0.5,
        direction="higher_is_better",
    ),
    "rgbd_fragment_count": MetricThreshold(
        metric="rgbd_fragment_count",
        pass_value=1.0,
        warn_value=0.5,
        direction="higher_is_better",
    ),
    "pose_graph_edges": MetricThreshold(
        metric="pose_graph_edges",
        pass_value=1.0,
        warn_value=0.5,
        direction="higher_is_better",
    ),
    "koide_lidar_camera_adapter_available": MetricThreshold(
        metric="koide_lidar_camera_adapter_available",
        pass_value=1.0,
        warn_value=0.0,
        direction="higher_is_better",
    ),
    "koide_lidar_camera_input_ready": MetricThreshold(
        metric="koide_lidar_camera_input_ready",
        pass_value=1.0,
        warn_value=0.0,
        direction="higher_is_better",
    ),
    "koide_lidar_camera_result_available": MetricThreshold(
        metric="koide_lidar_camera_result_available",
        pass_value=1.0,
        warn_value=0.0,
        direction="higher_is_better",
    ),
    "koide_lidar_camera_execution_success": MetricThreshold(
        metric="koide_lidar_camera_execution_success",
        pass_value=1.0,
        warn_value=0.0,
        direction="higher_is_better",
    ),
    "lidar_camera_transform_pairs": MetricThreshold(
        metric="lidar_camera_transform_pairs",
        pass_value=1.0,
        warn_value=0.0,
        direction="higher_is_better",
        unit="pairs",
    ),
    "lidar_camera_overlay_readiness": MetricThreshold(
        metric="lidar_camera_overlay_readiness",
        pass_value=1.0,
        warn_value=0.0,
        direction="higher_is_better",
    ),
    "lidar_camera_overlay_score": MetricThreshold(
        metric="lidar_camera_overlay_score",
        pass_value=0.75,
        warn_value=0.40,
        direction="higher_is_better",
    ),
    "lidar_camera_projection_frame_count": MetricThreshold(
        metric="lidar_camera_projection_frame_count",
        pass_value=2.0,
        warn_value=0.0,
        direction="higher_is_better",
        unit="frames",
    ),
    "lidar_camera_projected_points": MetricThreshold(
        metric="lidar_camera_projected_points",
        pass_value=5.0,
        warn_value=1.0,
        direction="higher_is_better",
        unit="points",
    ),
    "lidar_camera_projection_ratio": MetricThreshold(
        metric="lidar_camera_projection_ratio",
        pass_value=0.05,
        warn_value=0.01,
        direction="higher_is_better",
    ),
    "lidar_camera_projection_horizontal_coverage": MetricThreshold(
        metric="lidar_camera_projection_horizontal_coverage",
        pass_value=0.15,
        warn_value=0.05,
        direction="higher_is_better",
    ),
    "lidar_camera_projection_vertical_coverage": MetricThreshold(
        metric="lidar_camera_projection_vertical_coverage",
        pass_value=0.15,
        warn_value=0.05,
        direction="higher_is_better",
    ),
    "lidar_camera_edge_alignment_score": MetricThreshold(
        metric="lidar_camera_edge_alignment_score",
        pass_value=0.20,
        warn_value=0.05,
        direction="higher_is_better",
    ),
    "lidar_camera_depth_discontinuity_points": MetricThreshold(
        metric="lidar_camera_depth_discontinuity_points",
        pass_value=3.0,
        warn_value=1.0,
        direction="higher_is_better",
        unit="points",
    ),
    "lidar_camera_depth_edge_alignment_score": MetricThreshold(
        metric="lidar_camera_depth_edge_alignment_score",
        pass_value=0.20,
        warn_value=0.05,
        direction="higher_is_better",
    ),
    "lidar_camera_perturbation_case_count": MetricThreshold(
        metric="lidar_camera_perturbation_case_count",
        pass_value=12.0,
        warn_value=1.0,
        direction="higher_is_better",
        unit="cases",
    ),
    "lidar_camera_perturbation_detectable_fraction": MetricThreshold(
        metric="lidar_camera_perturbation_detectable_fraction",
        pass_value=0.50,
        warn_value=0.10,
        direction="higher_is_better",
    ),
    "lidar_camera_perturbation_mandatory_case_count": MetricThreshold(
        metric="lidar_camera_perturbation_mandatory_case_count",
        pass_value=12.0,
        warn_value=1.0,
        direction="higher_is_better",
        unit="cases",
    ),
    "lidar_camera_perturbation_mandatory_detectable_count": MetricThreshold(
        metric="lidar_camera_perturbation_mandatory_detectable_count",
        pass_value=8.0,
        warn_value=1.0,
        direction="higher_is_better",
        unit="cases",
    ),
    "lidar_camera_mutual_information_score": MetricThreshold(
        metric="lidar_camera_mutual_information_score",
        pass_value=0.50,
        warn_value=0.10,
        direction="higher_is_better",
    ),
    "autonomous_driving_dynamic_holdout": MetricThreshold(
        metric="autonomous_driving_dynamic_holdout",
        pass_value=0.10,
        warn_value=0.25,
    ),
}

AUTONOMOUS_DRIVING_OVERRIDES: dict[str, MetricThreshold] = {
    "reprojection_rmse_px": MetricThreshold(
        metric="reprojection_rmse_px",
        pass_value=0.8,
        warn_value=1.5,
        unit="px",
    ),
    "lidar_point_to_plane_rmse_m": MetricThreshold(
        metric="lidar_point_to_plane_rmse_m",
        pass_value=0.02,
        warn_value=0.05,
        unit="m",
    ),
    "lidar_frame_coverage": MetricThreshold(
        metric="lidar_frame_coverage",
        pass_value=100.0,
        warn_value=10.0,
        direction="higher_is_better",
        unit="frames",
    ),
    "lidar_point_coverage": MetricThreshold(
        metric="lidar_point_coverage",
        pass_value=200_000.0,
        warn_value=20_000.0,
        direction="higher_is_better",
        unit="points",
    ),
    "lidar_spatial_coverage_m": MetricThreshold(
        metric="lidar_spatial_coverage_m",
        pass_value=40.0,
        warn_value=15.0,
        direction="higher_is_better",
        unit="m",
    ),
    "lidar_local_planarity": MetricThreshold(
        metric="lidar_local_planarity",
        pass_value=0.40,
        warn_value=0.20,
        direction="higher_is_better",
    ),
    "lidar_map_roughness_m": MetricThreshold(
        metric="lidar_map_roughness_m",
        pass_value=0.10,
        warn_value=0.25,
        unit="m",
    ),
    "lidar_map_sharpness": MetricThreshold(
        metric="lidar_map_sharpness",
        pass_value=0.30,
        warn_value=0.10,
        direction="higher_is_better",
    ),
    "vehicle_motion_duration_s": MetricThreshold(
        metric="vehicle_motion_duration_s",
        pass_value=30.0,
        warn_value=10.0,
        direction="higher_is_better",
        unit="s",
    ),
    "vehicle_mean_speed_mps": MetricThreshold(
        metric="vehicle_mean_speed_mps",
        pass_value=5.0,
        warn_value=1.0,
        direction="higher_is_better",
        unit="m/s",
    ),
    "vehicle_speed_range_mps": MetricThreshold(
        metric="vehicle_speed_range_mps",
        pass_value=5.0,
        warn_value=1.0,
        direction="higher_is_better",
        unit="m/s",
    ),
    "vehicle_yaw_excitation_deg": MetricThreshold(
        metric="vehicle_yaw_excitation_deg",
        pass_value=30.0,
        warn_value=10.0,
        direction="higher_is_better",
        unit="deg",
    ),
    "camera_lidar_timestamp_alignment_ms": MetricThreshold(
        metric="camera_lidar_timestamp_alignment_ms",
        pass_value=10.0,
        warn_value=50.0,
        unit="ms",
    ),
    "lidar_oxts_timestamp_alignment_ms": MetricThreshold(
        metric="lidar_oxts_timestamp_alignment_ms",
        pass_value=10.0,
        warn_value=50.0,
        unit="ms",
    ),
}


def thresholds_for_profile(profile: ThresholdProfile) -> dict[str, MetricThreshold]:
    """Return threshold mapping for a profile."""

    thresholds = dict(DEFAULT_THRESHOLDS)
    if profile == "autonomous_driving":
        thresholds.update(AUTONOMOUS_DRIVING_OVERRIDES)
    return thresholds


def profile_from_domain(domain: str) -> ThresholdProfile:
    """Map a result domain to a threshold profile."""

    if domain == "autonomous_driving":
        return "autonomous_driving"
    return "default"


def primary_metric_value(metric: MetricResult) -> float | None:
    """Return the scalar value used for threshold grading."""

    if metric.holdout is not None:
        return metric.holdout
    if metric.value is not None:
        return metric.value
    return metric.train


def apply_metric_thresholds(
    metrics: dict[str, MetricResult],
    profile: ThresholdProfile,
) -> dict[str, MetricResult]:
    """Apply known threshold rules in-place and return metrics."""

    thresholds = thresholds_for_profile(profile)
    policy_graded_metrics = {"radar_lidar_velocity_consistency"}
    for name, metric in metrics.items():
        if name in policy_graded_metrics:
            continue
        threshold = thresholds.get(name)
        value = primary_metric_value(metric)
        if threshold is None or value is None:
            continue
        grade = threshold.grade(value)
        metric.grade = grade
        metric.unit = metric.unit or threshold.unit
        metric.reason = threshold.reason(value, grade)
    return metrics
