"""Metric registry for documentation, reporting, and plugins."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class MetricDefinition:
    """Human-readable metric metadata."""

    name: str
    description: str
    unit: str | None = None
    family: str = "common"


_METRICS: dict[str, MetricDefinition] = {}


def register_metric(definition: MetricDefinition) -> MetricDefinition:
    """Register a metric definition."""

    if definition.name in _METRICS:
        msg = f"metric already registered: {definition.name}"
        raise ValueError(msg)
    _METRICS[definition.name] = definition
    return definition


def metric(
    name: str,
    description: str,
    unit: str | None = None,
    family: str = "common",
) -> Callable[[type[object]], type[object]]:
    """Class decorator for plugin-owned metric declarations."""

    def decorator(owner: type[object]) -> type[object]:
        register_metric(
            MetricDefinition(name=name, description=description, unit=unit, family=family)
        )
        return owner

    return decorator


def get_metric_definition(name: str) -> MetricDefinition | None:
    """Return metric metadata when registered."""

    return _METRICS.get(name)


def list_metric_definitions() -> list[MetricDefinition]:
    """List registered metric metadata."""

    return [definition for _, definition in sorted(_METRICS.items())]


def _register_builtin_metrics() -> None:
    builtins = [
        MetricDefinition(
            "schema_validation",
            "Config and result schema validation",
            family="common",
        ),
        MetricDefinition("dataset_exists", "Dataset path availability", family="data"),
        MetricDefinition(
            "nuscenes_reference_extrinsic_count",
            "nuScenes calibrated_sensor transforms imported as reference extrinsics",
            "transforms",
            "data",
        ),
        MetricDefinition(
            "extrinsic_reference_comparison_count",
            "Number of candidate extrinsics matched against reference extrinsics",
            "pairs",
            "extrinsic",
        ),
        MetricDefinition(
            "candidate_extrinsic_import_count",
            "Number of externally supplied candidate extrinsics imported",
            "transforms",
            "extrinsic",
        ),
        MetricDefinition(
            "extrinsic_reference_translation_delta_max_m",
            "Maximum translation delta between matched candidate and reference extrinsics",
            "m",
            "extrinsic",
        ),
        MetricDefinition(
            "extrinsic_reference_rotation_delta_max_deg",
            "Maximum rotation delta between matched candidate and reference extrinsics",
            "deg",
            "extrinsic",
        ),
        MetricDefinition("reprojection_rmse_px", "Camera reprojection RMSE", "px", "camera"),
        MetricDefinition(
            "lidar_point_to_plane_rmse_m",
            "LiDAR point-to-plane RMSE",
            "m",
            "lidar",
        ),
        MetricDefinition(
            "lidar_world_map_point_to_plane_rmse_m",
            "OXTS-projected LiDAR train/holdout map point-to-plane RMSE",
            "m",
            "lidar",
        ),
        MetricDefinition(
            "lidar_world_map_point_to_plane_median_holdout_m",
            "Median holdout point-to-plane residual against train LiDAR map",
            "m",
            "lidar",
        ),
        MetricDefinition(
            "lidar_world_map_point_to_plane_p95_holdout_m",
            "P95 holdout point-to-plane residual against train LiDAR map",
            "m",
            "lidar",
        ),
        MetricDefinition(
            "lidar_world_map_train_voxel_count",
            "Number of train voxel planes in OXTS-projected LiDAR map",
            "voxels",
            "lidar",
        ),
        MetricDefinition(
            "lidar_world_map_holdout_residual_count",
            "Number of holdout LiDAR points matched to train voxel planes",
            "points",
            "lidar",
        ),
        MetricDefinition(
            "lidar_world_map_leakage_issue_count",
            "Frame-level leakage issues between LiDAR world-map train and holdout artifacts",
            "issues",
            "lidar",
        ),
        MetricDefinition(
            "lidar_world_map_stability_window_count",
            "Temporal block windows evaluated for LiDAR world-map stability",
            "windows",
            "lidar",
        ),
        MetricDefinition(
            "lidar_world_map_stability_scored_window_count",
            "Temporal block windows with scored LiDAR world-map holdout residuals",
            "windows",
            "lidar",
        ),
        MetricDefinition(
            "lidar_world_map_stability_holdout_rmse_spread_m",
            "Spread of holdout RMSE across LiDAR world-map temporal windows",
            "m",
            "lidar",
        ),
        MetricDefinition(
            "lidar_world_map_perturbation_case_count",
            "Number of world-map LiDAR extrinsic perturbation cases scored",
            "cases",
            "lidar",
        ),
        MetricDefinition(
            "lidar_world_map_perturbation_detectable_fraction",
            "Fraction of known perturbations that worsened LiDAR world-map consistency",
            family="lidar",
        ),
        MetricDefinition(
            "lidar_world_map_perturbation_train_rmse_delta_mean_m",
            "Mean train world-map RMSE increase after known LiDAR perturbations",
            "m",
            "lidar",
        ),
        MetricDefinition(
            "lidar_world_map_perturbation_holdout_rmse_delta_mean_m",
            "Mean holdout world-map RMSE increase after known LiDAR perturbations",
            "m",
            "lidar",
        ),
        MetricDefinition(
            "lidar_world_map_perturbation_holdout_rmse_delta_max_m",
            "Worst holdout world-map RMSE increase after known LiDAR perturbations",
            "m",
            "lidar",
        ),
        MetricDefinition(
            "lidar_world_map_perturbation_p95_holdout_delta_mean_m",
            "Mean P95 holdout residual increase after known LiDAR perturbations",
            "m",
            "lidar",
        ),
        MetricDefinition(
            "lidar_world_map_sensitivity_roll_m",
            "Roll sensitivity from LiDAR world-map perturbation sweep",
            "m",
            "lidar",
        ),
        MetricDefinition(
            "lidar_world_map_sensitivity_pitch_m",
            "Pitch sensitivity from LiDAR world-map perturbation sweep",
            "m",
            "lidar",
        ),
        MetricDefinition(
            "lidar_world_map_sensitivity_yaw_m",
            "Yaw sensitivity from LiDAR world-map perturbation sweep",
            "m",
            "lidar",
        ),
        MetricDefinition(
            "lidar_world_map_sensitivity_x_m",
            "X translation sensitivity from LiDAR world-map perturbation sweep",
            "m",
            "lidar",
        ),
        MetricDefinition(
            "lidar_world_map_sensitivity_y_m",
            "Y translation sensitivity from LiDAR world-map perturbation sweep",
            "m",
            "lidar",
        ),
        MetricDefinition(
            "lidar_world_map_sensitivity_z_m",
            "Z translation sensitivity from LiDAR world-map perturbation sweep",
            "m",
            "lidar",
        ),
        MetricDefinition(
            "lidar_world_map_weak_dof_count",
            "Number of weak LiDAR DoF from world-map perturbation sensitivity",
            "dof",
            "lidar",
        ),
        MetricDefinition(
            "lidar_world_map_min_dof_sensitivity_m",
            "Minimum DoF sensitivity from LiDAR world-map perturbation sweep",
            "m",
            "lidar",
        ),
        MetricDefinition(
            "lidar_rig_point_to_plane_residual_count",
            "Native LiDAR rig point-to-plane factor residual count",
            "residuals",
            "lidar",
        ),
        MetricDefinition(
            "lidar_rig_point_to_plane_rmse_m",
            "Native LiDAR rig point-to-plane factor RMSE",
            "m",
            "lidar",
        ),
        MetricDefinition(
            "lidar_rig_point_to_plane_rank",
            "Rank of the native LiDAR rig point-to-plane normal equations",
            family="lidar",
        ),
        MetricDefinition(
            "lidar_rig_point_to_plane_condition_number",
            "Normalized local-curvature condition proxy for native LiDAR rig point-to-plane",
            family="lidar",
        ),
        MetricDefinition(
            "lidar_rig_point_to_plane_normalization_length_m",
            "Representative length used to normalize translation and rotation DoF",
            "m",
            "lidar",
        ),
        MetricDefinition(
            "lidar_rig_point_to_plane_weak_dof_count",
            "Weak DoF count from normalized native LiDAR rig point-to-plane local curvature",
            "dof",
            "lidar",
        ),
        MetricDefinition(
            "lidar_perturbation_case_count",
            "Number of known LiDAR extrinsic perturbation cases scored",
            "cases",
            "lidar",
        ),
        MetricDefinition(
            "lidar_perturbation_detectable_fraction",
            "Fraction of known LiDAR extrinsic perturbations that worsened point-to-plane RMSE",
            family="lidar",
        ),
        MetricDefinition(
            "lidar_perturbation_train_rmse_delta_mean_m",
            "Mean train point-to-plane RMSE increase after known LiDAR perturbations",
            "m",
            "lidar",
        ),
        MetricDefinition(
            "lidar_perturbation_holdout_rmse_delta_mean_m",
            "Mean holdout point-to-plane RMSE increase after known LiDAR perturbations",
            "m",
            "lidar",
        ),
        MetricDefinition(
            "lidar_perturbation_holdout_rmse_delta_max_m",
            "Worst holdout point-to-plane RMSE increase after known LiDAR perturbations",
            "m",
            "lidar",
        ),
        MetricDefinition(
            "lidar_frame_coverage",
            "Number of available LiDAR frames",
            "frames",
            "lidar",
        ),
        MetricDefinition(
            "lidar_point_coverage",
            "Number of sampled LiDAR points used for input coverage checks",
            "points",
            "lidar",
        ),
        MetricDefinition(
            "lidar_spatial_coverage_m",
            "Diagonal XYZ extent of sampled LiDAR points",
            "m",
            "lidar",
        ),
        MetricDefinition(
            "lidar_local_planarity",
            "Mean local LiDAR voxel planarity score",
            family="lidar",
        ),
        MetricDefinition(
            "lidar_map_roughness_m",
            "Mean local LiDAR voxel normal roughness proxy",
            "m",
            "lidar",
        ),
        MetricDefinition(
            "lidar_map_sharpness",
            "Mean local LiDAR map sharpness proxy",
            family="lidar",
        ),
        MetricDefinition(
            "lidar_pair_shared_voxel_count",
            "Coarse shared voxel count between two fixed LiDAR point clouds",
            "voxels",
            "lidar",
        ),
        MetricDefinition(
            "lidar_pair_unmatched_source_voxel_count",
            "Source LiDAR voxels without a target voxel at the configured voxel size",
            "voxels",
            "lidar",
        ),
        MetricDefinition(
            "lidar_pair_unmatched_target_voxel_count",
            "Target LiDAR voxels without a source voxel at the configured voxel size",
            "voxels",
            "lidar",
        ),
        MetricDefinition(
            "lidar_pair_source_voxel_recall_in_target",
            "Fraction of source LiDAR voxels also occupied by the target LiDAR",
            family="lidar",
        ),
        MetricDefinition(
            "lidar_pair_target_voxel_recall_in_source",
            "Fraction of target LiDAR voxels also occupied by the source LiDAR",
            family="lidar",
        ),
        MetricDefinition(
            "lidar_pair_shared_voxel_centroid_rmse_m",
            "Shared voxel centroid RMSE for fixed LiDAR pair evidence",
            "m",
            "lidar",
        ),
        MetricDefinition(
            "lidar_pair_holdout_point_to_plane_map_voxel_count",
            "Source-frame voxel-plane count for fixed LiDAR pair holdout evidence",
            "voxels",
            "lidar",
        ),
        MetricDefinition(
            "lidar_pair_holdout_point_to_plane_eligible_point_count",
            "Candidate-independent target point population for fixed LiDAR pair holdout evidence",
            "points",
            "lidar",
        ),
        MetricDefinition(
            "lidar_pair_holdout_point_to_plane_considered_point_count",
            "Target points considered for fixed LiDAR pair point-to-plane matching",
            "points",
            "lidar",
        ),
        MetricDefinition(
            "lidar_pair_holdout_point_to_plane_accepted_correspondence_count",
            "Accepted fixed LiDAR pair point-to-plane correspondences under "
            "the fixed support denominator",
            "points",
            "lidar",
        ),
        MetricDefinition(
            "lidar_pair_holdout_point_to_plane_support_ratio",
            "Accepted correspondences divided by the candidate-independent support population",
            family="lidar",
        ),
        MetricDefinition(
            "lidar_pair_holdout_point_to_plane_matched_point_count",
            "Transformed target points matched to source voxel planes",
            "points",
            "lidar",
        ),
        MetricDefinition(
            "lidar_pair_holdout_point_to_plane_unmatched_fraction",
            "Fraction of target points without a source voxel-plane correspondence",
            family="lidar",
        ),
        MetricDefinition(
            "lidar_pair_holdout_point_to_plane_median_abs_m",
            "Median absolute holdout point-to-plane residual for a fixed LiDAR pair",
            "m",
            "lidar",
        ),
        MetricDefinition(
            "lidar_pair_holdout_point_to_plane_p90_abs_m",
            "P90 absolute holdout point-to-plane residual for a fixed LiDAR pair",
            "m",
            "lidar",
        ),
        MetricDefinition(
            "lidar_pair_holdout_point_to_plane_rmse_m",
            "Holdout point-to-plane RMSE for a fixed LiDAR pair",
            "m",
            "lidar",
        ),
        MetricDefinition(
            "lidar_pair_holdout_point_to_plane_inlier_fraction",
            "Fraction of matched fixed LiDAR pair residuals below the inlier threshold",
            family="lidar",
        ),
        MetricDefinition(
            "lidar_pair_known_bad_case_count",
            "Number of known-bad LiDAR pair perturbation cases scored",
            "cases",
            "lidar",
        ),
        MetricDefinition(
            "lidar_pair_known_bad_detectable_fraction",
            "Fraction of known-bad LiDAR pair perturbations detected by pair evidence",
            family="lidar",
        ),
        MetricDefinition(
            "lidar_pair_known_bad_mandatory_case_count",
            "Number of mandatory large 6-DoF known-bad LiDAR pair controls materialized",
            "cases",
            "lidar",
        ),
        MetricDefinition(
            "lidar_pair_known_bad_mandatory_supported_detection_count",
            "Mandatory known-bad controls detected under sufficient LiDAR pair support",
            "cases",
            "lidar",
        ),
        MetricDefinition(
            "lidar_pair_known_bad_mandatory_supported_detection_fraction",
            "Fraction of mandatory known-bad controls detected under sufficient support",
            family="lidar",
        ),
        MetricDefinition(
            "lidar_pair_known_bad_mandatory_support_collapse_count",
            "Mandatory known-bad controls dominated by support collapse",
            "cases",
            "lidar",
        ),
        MetricDefinition(
            "lidar_pair_known_bad_source_recall_delta_mean",
            "Mean source voxel recall drop under known-bad LiDAR pair perturbations",
            family="lidar",
        ),
        MetricDefinition(
            "lidar_pair_known_bad_centroid_rmse_delta_mean_m",
            "Mean shared-voxel centroid RMSE increase under known-bad LiDAR pair perturbations",
            "m",
            "lidar",
        ),
        MetricDefinition(
            "lidar_pair_known_bad_centroid_rmse_delta_max_m",
            "Largest shared-voxel centroid RMSE increase under known-bad LiDAR pair perturbations",
            "m",
            "lidar",
        ),
        MetricDefinition(
            "lidar_pair_known_bad_point_to_plane_p90_delta_max_m",
            "Largest holdout point-to-plane P90 increase under known-bad LiDAR pair perturbations",
            "m",
            "lidar",
        ),
        MetricDefinition(
            "lidar_pair_known_bad_point_to_plane_rmse_delta_mean_m",
            "Mean holdout point-to-plane RMSE increase under known-bad LiDAR pair perturbations",
            "m",
            "lidar",
        ),
        MetricDefinition(
            "trajectory_consistency",
            "Trajectory consistency residual",
            "m",
            "motion",
        ),
        MetricDefinition(
            "vehicle_motion_duration_s",
            "Vehicle motion log duration from OXTS",
            "s",
            "motion",
        ),
        MetricDefinition(
            "vehicle_mean_speed_mps",
            "Mean vehicle speed from OXTS",
            "m/s",
            "motion",
        ),
        MetricDefinition(
            "vehicle_speed_range_mps",
            "Vehicle speed range from OXTS",
            "m/s",
            "motion",
        ),
        MetricDefinition(
            "vehicle_yaw_excitation_deg",
            "Vehicle yaw excitation from OXTS",
            "deg",
            "motion",
        ),
        MetricDefinition(
            "vehicle_pitch_excitation_deg",
            "Vehicle pitch excitation from OXTS",
            "deg",
            "motion",
        ),
        MetricDefinition(
            "vehicle_roll_excitation_deg",
            "Vehicle roll excitation from OXTS",
            "deg",
            "motion",
        ),
        MetricDefinition(
            "vehicle_mean_acceleration_norm_mps2",
            "Mean vehicle acceleration norm from OXTS",
            "m/s^2",
            "motion",
        ),
        MetricDefinition(
            "camera_lidar_timestamp_alignment_ms",
            "Max nearest timestamp delta between camera and LiDAR",
            "ms",
            "timing",
        ),
        MetricDefinition(
            "lidar_oxts_timestamp_alignment_ms",
            "Max nearest timestamp delta between LiDAR and OXTS",
            "ms",
            "timing",
        ),
        MetricDefinition("condition_number", "Optimization normal-equation condition number"),
        MetricDefinition(
            "open3d_slac_backend_available",
            "Open3D optional backend availability",
            family="backend",
        ),
        MetricDefinition(
            "native_joint_slac_available",
            "Native backend-neutral joint SLAC execution availability",
            family="backend",
        ),
        MetricDefinition(
            "native_tum_joint_slac_available",
            "Native TUM RGB-D multi-capture joint SLAC availability",
            family="backend",
        ),
        MetricDefinition(
            "tum_joint_map_frame_count",
            "Disjoint TUM RGB-D frames reserved to build the world plane map",
            "frames",
            "joint_slac",
        ),
        MetricDefinition(
            "tum_joint_query_frame_count",
            "TUM RGB-D query frames grouped for joint train/holdout",
            "frames",
            "joint_slac",
        ),
        MetricDefinition(
            "tum_joint_correspondence_count",
            "Multi-capture TUM point-to-world-plane correspondences",
            "correspondences",
            "joint_slac",
        ),
        MetricDefinition(
            "tum_joint_point_to_plane_rmse_m",
            "TUM joint point-to-plane RMSE excluding trajectory priors",
            "m",
            "joint_slac",
        ),
        MetricDefinition(
            "tum_joint_augmented_information_rank",
            "TUM joint pose/extrinsic rank including measured-pose priors",
            family="joint_slac",
        ),
        MetricDefinition(
            "tum_joint_data_only_extrinsic_rank",
            "TUM train-geometry shared-extrinsic rank with poses fixed",
            family="joint_slac",
        ),
        MetricDefinition(
            "tum_joint_data_only_extrinsic_condition_number",
            "TUM train-geometry normalized shared-extrinsic condition number",
            family="joint_slac",
        ),
        MetricDefinition(
            "tum_joint_data_only_shared_rank",
            "TUM train rank of shared extrinsic plus depth scale/bias",
            family="joint_slac",
        ),
        MetricDefinition(
            "tum_joint_data_only_shared_condition_number",
            "TUM train condition number of shared extrinsic plus depth scale/bias",
            family="joint_slac",
        ),
        MetricDefinition(
            "tum_joint_depth_scale",
            "Estimated multiplicative correction of nominal TUM metric depth",
            family="joint_slac",
        ),
        MetricDefinition(
            "tum_joint_depth_bias_m",
            "Estimated additive correction of nominal TUM metric depth",
            "m",
            "joint_slac",
        ),
        MetricDefinition(
            "tum_joint_depth_scale_reference_error_percent",
            "TUM depth multiplier error against the official pre-scaled reference",
            "percent",
            "joint_slac",
        ),
        MetricDefinition(
            "tum_joint_depth_bias_reference_error_m",
            "TUM depth bias error against the official zero-bias reference",
            "m",
            "joint_slac",
        ),
        MetricDefinition(
            "tum_joint_known_bad_detectable_fraction",
            "Fraction of signed shared-extrinsic perturbations detected on TUM holdout",
            family="joint_slac",
        ),
        MetricDefinition(
            "tum_joint_identity_reference_translation_error_m",
            "TUM shared mounting translation error against camera-frame identity",
            "m",
            "joint_slac",
        ),
        MetricDefinition(
            "tum_joint_identity_reference_rotation_error_deg",
            "TUM shared mounting rotation error against camera-frame identity",
            "deg",
            "joint_slac",
        ),
        MetricDefinition(
            "tum_joint_replication_window_count",
            "Number of independently constructed TUM temporal windows",
            "windows",
            "joint_slac",
        ),
        MetricDefinition(
            "tum_joint_replication_converged_fraction",
            "Fraction of configured TUM replication windows converged",
            family="joint_slac",
        ),
        MetricDefinition(
            "tum_joint_replication_reference_pass_fraction",
            "Fraction of TUM windows passing unchanged mounting and depth references",
            family="joint_slac",
        ),
        MetricDefinition(
            "tum_joint_replication_depth_scale_range_percent",
            "Range of estimated depth multipliers over TUM temporal windows",
            "percent",
            "joint_slac",
        ),
        MetricDefinition(
            "tum_joint_replication_depth_bias_range_m",
            "Range of estimated depth biases over TUM temporal windows",
            "m",
            "joint_slac",
        ),
        MetricDefinition(
            "tum_joint_cross_window_max_holdout_delta_rmse_m",
            "Worst holdout RMSE change after cross-window shared-parameter transfer",
            "m",
            "joint_slac",
        ),
        MetricDefinition(
            "tum_joint_cross_window_nondegrading_fraction",
            "Fraction of ordered cross-window transfers within the RMSE margin",
            family="joint_slac",
        ),
        MetricDefinition(
            "tum_joint_spatial_ablation_converged_fraction",
            "Fraction of TUM spatial-lattice ablation windows converged",
            family="joint_slac",
        ),
        MetricDefinition(
            "tum_joint_spatial_ablation_holdout_improved_fraction",
            "Fraction of windows where spatial lattice improves scalar holdout RMSE",
            family="joint_slac",
        ),
        MetricDefinition(
            "tum_joint_spatial_ablation_worst_holdout_delta_rmse_m",
            "Worst spatial-minus-scalar same-window holdout RMSE change",
            "m",
            "joint_slac",
        ),
        MetricDefinition(
            "tum_joint_spatial_lattice_max_abs_offset_m",
            "Largest absolute fitted trilinear ray-depth control offset",
            "m",
            "joint_slac",
        ),
        MetricDefinition(
            "tum_joint_spatial_constant_bias_range_m",
            "Range of separated constant depth bias over TUM temporal windows",
            "m",
            "joint_slac",
        ),
        MetricDefinition(
            "tum_joint_spatial_known_bad_detectable_fraction_min",
            "Minimum spatial-lattice known-bad detection fraction over windows",
            family="joint_slac",
        ),
        MetricDefinition(
            "tum_joint_spatial_cross_window_max_holdout_delta_rmse_m",
            "Worst cross-window spatial shared-parameter transfer RMSE change",
            "m",
            "joint_slac",
        ),
        MetricDefinition(
            "tum_joint_spatial_cross_window_nondegrading_fraction",
            "Fraction of spatial cross-window transfers within the RMSE margin",
            family="joint_slac",
        ),
        MetricDefinition(
            "tum_joint_rematch_pair_jaccard_min",
            "Minimum fixed/rematched query-voxel pair Jaccard over TUM windows",
            family="registration",
        ),
        MetricDefinition(
            "tum_joint_rematch_retained_query_fraction_min",
            "Minimum fraction of fixed holdout queries retained after rematching",
            family="registration",
        ),
        MetricDefinition(
            "tum_joint_rematch_same_target_fraction_min",
            "Minimum same-voxel fraction among retained TUM holdout queries",
            family="registration",
        ),
        MetricDefinition(
            "tum_joint_rematch_best_holdout_delta_rmse_m",
            "Best rematched-minus-fixed TUM holdout RMSE change",
            "m",
            "registration",
        ),
        MetricDefinition(
            "tum_joint_rematched_curvature_rank_min",
            "Minimum numerical rematched Hessian rank over TUM windows",
            family="observability",
        ),
        MetricDefinition(
            "tum_joint_rematched_negative_curvature_count_max",
            "Maximum rematched-objective negative curvature directions",
            family="observability",
        ),
        MetricDefinition(
            "tum_joint_curvature_relative_hessian_difference_max",
            "Maximum relative fixed/rematched Hessian Frobenius difference",
            family="observability",
        ),
        MetricDefinition(
            "tum_joint_multistart_converged_fraction",
            "Fraction of public TUM rematched multi-start searches converged",
            family="registration",
        ),
        MetricDefinition(
            "tum_joint_multistart_basin_count",
            "Distinct scale-normalized TUM rematched solution basins",
            "basins",
            "registration",
        ),
        MetricDefinition(
            "tum_joint_multistart_competitive_basin_count",
            "Objective-competitive TUM rematched solution basins",
            "basins",
            "registration",
        ),
        MetricDefinition(
            "tum_joint_multistart_ambiguity",
            "Binary separated competitive-basin ambiguity indicator",
            family="registration",
        ),
        MetricDefinition(
            "joint_slac_point_to_plane_rmse_m",
            "Joint pose-extrinsic point-to-plane RMSE excluding priors",
            "m",
            "joint_slac",
        ),
        MetricDefinition(
            "joint_slac_augmented_information_rank",
            "Joint information rank including declared gauge priors",
            family="joint_slac",
        ),
        MetricDefinition(
            "joint_slac_augmented_condition_number",
            "Joint Jacobian condition number including declared gauge priors",
            family="joint_slac",
        ),
        MetricDefinition(
            "joint_slac_known_bad_detectable_fraction",
            "Fraction of signed joint holdout perturbations detected",
            family="joint_slac",
        ),
        MetricDefinition(
            "joint_slac_vs_fixed_baseline_translation_m",
            "Translation delta from the fixed-trajectory native baseline",
            "m",
            "joint_slac",
        ),
        MetricDefinition(
            "joint_slac_vs_fixed_baseline_rotation_deg",
            "Rotation delta from the fixed-trajectory native baseline",
            "deg",
            "joint_slac",
        ),
        MetricDefinition("rgbd_fragment_count", "RGB-D fragment count", family="open3d"),
        MetricDefinition("pose_graph_edges", "Pose graph edge count", family="open3d"),
        MetricDefinition(
            "point_plane_center_rmse_m",
            "Board-centre closure RMSE for the independent Camera-LiDAR point+plane baseline",
            "m",
            "lidar_camera",
        ),
        MetricDefinition(
            "point_plane_normal_rmse_deg",
            "Board-normal closure RMSE for the independent Camera-LiDAR point+plane baseline",
            "deg",
            "lidar_camera",
        ),
        MetricDefinition(
            "point_plane_offset_rmse_m",
            "Board-plane offset closure RMSE for the Camera-LiDAR point+plane baseline",
            "m",
            "lidar_camera",
        ),
        MetricDefinition(
            "point_plane_known_bad_detectable_fraction",
            "Fraction of held-out point+plane 6-DoF controls detected",
            family="lidar_camera",
        ),
        MetricDefinition(
            "point_plane_joint_rank",
            "Rank of the Camera-LiDAR centre/normal six-DoF Jacobian",
            "rank",
            "lidar_camera",
        ),
        MetricDefinition(
            "point_plane_joint_condition_number",
            "Condition number of the Camera-LiDAR centre/normal six-DoF Jacobian",
            family="lidar_camera",
        ),
        MetricDefinition(
            "point_plane_vs_plane_translation_delta_m",
            "Translation delta between point+plane and plane-only Camera-LiDAR baselines",
            "m",
            "lidar_camera",
        ),
        MetricDefinition(
            "point_plane_vs_plane_rotation_delta_deg",
            "Rotation delta between point+plane and plane-only Camera-LiDAR baselines",
            "deg",
            "lidar_camera",
        ),
        MetricDefinition(
            "koide_lidar_camera_adapter_available",
            "Koide-style targetless LiDAR-camera adapter availability",
            family="backend",
        ),
        MetricDefinition(
            "koide_lidar_camera_input_ready",
            "Camera and LiDAR stream readiness for targetless LiDAR-camera calibration",
            family="lidar_camera",
        ),
        MetricDefinition(
            "koide_lidar_camera_result_available",
            "Precomputed targetless LiDAR-camera transform availability",
            family="lidar_camera",
        ),
        MetricDefinition(
            "koide_lidar_camera_execution_success",
            "External targetless LiDAR-camera adapter command execution status",
            family="backend",
        ),
        MetricDefinition(
            "lidar_camera_transform_pairs",
            "Number of camera-LiDAR transform pairs available for overlay evaluation",
            "pairs",
            "lidar_camera",
        ),
        MetricDefinition(
            "lidar_camera_overlay_readiness",
            "Camera-LiDAR stream, transform, and temporal-pair readiness",
            family="lidar_camera",
        ),
        MetricDefinition(
            "lidar_camera_overlay_score",
            "Alpha proxy score for projected LiDAR-on-camera overlay quality",
            family="lidar_camera",
        ),
        MetricDefinition(
            "lidar_camera_projection_frame_count",
            "Number of camera-LiDAR frame pairs used for projection metrics",
            "frames",
            "lidar_camera",
        ),
        MetricDefinition(
            "lidar_camera_projected_points",
            "Sampled LiDAR points projected into the camera image",
            "points",
            "lidar_camera",
        ),
        MetricDefinition(
            "lidar_camera_projection_ratio",
            "Fraction of sampled LiDAR points projected into the camera image",
            family="lidar_camera",
        ),
        MetricDefinition(
            "lidar_camera_projection_depth_median_m",
            "Median depth of projected LiDAR points",
            "m",
            "lidar_camera",
        ),
        MetricDefinition(
            "lidar_camera_projection_depth_span_m",
            "Depth span of projected LiDAR points",
            "m",
            "lidar_camera",
        ),
        MetricDefinition(
            "lidar_camera_projection_horizontal_coverage",
            "Normalized horizontal image span of projected LiDAR points",
            family="lidar_camera",
        ),
        MetricDefinition(
            "lidar_camera_projection_vertical_coverage",
            "Normalized vertical image span of projected LiDAR points",
            family="lidar_camera",
        ),
        MetricDefinition(
            "lidar_camera_edge_alignment_score",
            "Fraction of projected LiDAR points near image intensity edges",
            family="lidar_camera",
        ),
        MetricDefinition(
            "lidar_camera_edge_gradient_mean",
            "Mean image gradient at projected LiDAR points",
            family="lidar_camera",
        ),
        MetricDefinition(
            "lidar_camera_depth_discontinuity_points",
            "Projected LiDAR points with nearby depth jumps",
            "points",
            "lidar_camera",
        ),
        MetricDefinition(
            "lidar_camera_depth_edge_alignment_score",
            "Fraction of projected LiDAR depth jumps near image intensity edges",
            family="lidar_camera",
        ),
        MetricDefinition(
            "lidar_camera_depth_edge_gradient_mean",
            "Mean image gradient at projected LiDAR depth jumps",
            family="lidar_camera",
        ),
        MetricDefinition(
            "lidar_camera_perturbation_case_count",
            "Number of known KITTI camera-LiDAR extrinsic perturbation cases scored",
            "cases",
            "lidar_camera",
        ),
        MetricDefinition(
            "lidar_camera_perturbation_detectable_fraction",
            "Fraction of known extrinsic perturbations that worsened projection diagnostics",
            family="lidar_camera",
        ),
        MetricDefinition(
            "lidar_camera_perturbation_edge_delta_mean",
            "Mean edge-alignment score drop after known extrinsic perturbations",
            family="lidar_camera",
        ),
        MetricDefinition(
            "lidar_camera_perturbation_depth_edge_delta_mean",
            "Mean depth-edge score drop after known extrinsic perturbations",
            family="lidar_camera",
        ),
        MetricDefinition(
            "lidar_camera_perturbation_projection_ratio_delta_mean",
            "Mean projection-ratio drop after known extrinsic perturbations",
            family="lidar_camera",
        ),
        MetricDefinition(
            "lidar_camera_perturbation_mandatory_case_count",
            "Number of mandatory ±1 deg / ±0.10 m camera-LiDAR perturbation cases",
            "cases",
            "lidar_camera",
        ),
        MetricDefinition(
            "lidar_camera_perturbation_mandatory_detectable_count",
            "Mandatory camera-LiDAR perturbations detected under projection scoring",
            "cases",
            "lidar_camera",
        ),
        MetricDefinition(
            "lidar_camera_mutual_information_score",
            "Image-LiDAR mutual-information alignment score",
            family="lidar_camera",
        ),
        MetricDefinition(
            "lidar_camera_comparison_candidate_count",
            "Camera-LiDAR extrinsic candidates evaluated on the shared frame split",
            "candidates",
            "lidar_camera",
        ),
        MetricDefinition(
            "lidar_camera_capture_time_policy_declared",
            "Whether a valid structured LiDAR capture-time policy was declared",
            "bool",
            "lidar_camera",
        ),
        MetricDefinition(
            "lidar_camera_capture_time_deskew_applied",
            "Whether per-point LiDAR capture-time deskew was applied for camera projection",
            "bool",
            "lidar_camera",
        ),
        MetricDefinition(
            "lidar_imu_holdout_rotation_rate_rmse_dps",
            "Holdout rotation-rate RMSE between odometry and candidate-rotated IMU",
            "deg/s",
            "lidar_imu",
        ),
        MetricDefinition(
            "lidar_imu_known_bad_detectable_fraction",
            "Fraction of known-bad LiDAR-IMU rotation probes detected",
            family="lidar_imu",
        ),
        MetricDefinition(
            "lidar_imu_gravity_alignment_deg",
            "Angle between low-pass IMU acceleration and world gravity",
            "deg",
            "lidar_imu",
        ),
        MetricDefinition("observability_rank", "Estimated observability rank"),
        MetricDefinition(
            "radar_lidar_velocity_consistency",
            "Radar-LiDAR ego-motion velocity consistency",
            "m/s",
            "radar",
        ),
        MetricDefinition(
            "autonomous_driving_dynamic_holdout",
            "Dynamic-object holdout disagreement ratio",
            family="autonomous_driving",
        ),
    ]
    for definition in builtins:
        register_metric(definition)


_register_builtin_metrics()
