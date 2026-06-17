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
            "Diagonal condition estimate for native LiDAR rig point-to-plane Hessian",
            family="lidar",
        ),
        MetricDefinition(
            "lidar_rig_point_to_plane_weak_dof_count",
            "Weak DoF count from native LiDAR rig point-to-plane Hessian diagonal",
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
        MetricDefinition("rgbd_fragment_count", "RGB-D fragment count", family="open3d"),
        MetricDefinition("pose_graph_edges", "Pose graph edge count", family="open3d"),
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
            "lidar_camera_mutual_information_score",
            "Image-LiDAR mutual-information alignment score",
            family="lidar_camera",
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
