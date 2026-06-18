"""LiDAR-specific metric extraction helpers."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

from calibrex.core.config import CalibrationConfig
from calibrex.core.geometry import SE3, QuaternionXYZW
from calibrex.core.result import Grade, MetricResult
from calibrex.data.inspect import DatasetInspection
from calibrex.data.kitti import (
    summarize_lidar_world_map_consistency,
    summarize_velodyne_points,
)
from calibrex.graph.lidar_point_to_plane import LidarRigPointToPlaneEvaluation

_WORLD_MAP_WEAK_DOF_DELTA_M = 1.0e-3
_WORLD_MAP_DOF_METRICS = {
    "roll_deg": ("lidar_world_map_sensitivity_roll_m", "roll_lidar0"),
    "pitch_deg": ("lidar_world_map_sensitivity_pitch_m", "pitch_lidar0"),
    "yaw_deg": ("lidar_world_map_sensitivity_yaw_m", "yaw_lidar0"),
    "x_m": ("lidar_world_map_sensitivity_x_m", "x_lidar0"),
    "y_m": ("lidar_world_map_sensitivity_y_m", "y_lidar0"),
    "z_m": ("lidar_world_map_sensitivity_z_m", "z_lidar0"),
}


def lidar_metrics_from_inspection(
    inspection: DatasetInspection,
    config: CalibrationConfig | None = None,
) -> dict[str, MetricResult]:
    """Build LiDAR coverage metrics from dataset inspection diagnostics."""

    velodyne = inspection.diagnostics.get("velodyne_points")
    if not isinstance(velodyne, Mapping):
        return {}
    world_map = inspection.diagnostics.get("lidar_world_map_consistency")
    world_map_diagnostics = world_map if isinstance(world_map, Mapping) else None

    frame_count = _float_or_none(velodyne.get("frame_count"))
    sampled_point_count = _float_or_none(velodyne.get("sampled_point_count"))
    bounds_min = _vector3_or_none(velodyne.get("bounds_min_m"))
    bounds_max = _vector3_or_none(velodyne.get("bounds_max_m"))
    local_planarity = _float_or_none(velodyne.get("local_planarity_mean"))
    roughness = _float_or_none(velodyne.get("roughness_mean_m"))
    map_sharpness = _float_or_none(velodyne.get("map_sharpness_score"))
    point_to_plane_train = _float_or_none(velodyne.get("point_to_plane_rmse_train_m"))
    point_to_plane_holdout = _float_or_none(velodyne.get("point_to_plane_rmse_holdout_m"))

    metrics: dict[str, MetricResult] = {}
    if frame_count is not None:
        metrics["lidar_frame_coverage"] = MetricResult(
            value=frame_count,
            unit="frames",
            reason=f"inspected {frame_count:g} LiDAR frames",
        )
    if sampled_point_count is not None:
        metrics["lidar_point_coverage"] = MetricResult(
            value=sampled_point_count,
            unit="points",
            reason=f"sampled {sampled_point_count:g} LiDAR points",
        )
    if bounds_min is not None and bounds_max is not None:
        extent = math.dist(bounds_min, bounds_max)
        metrics["lidar_spatial_coverage_m"] = MetricResult(
            value=extent,
            unit="m",
            reason=f"sampled LiDAR XYZ extent is {extent:g} m",
        )
    if local_planarity is not None:
        metrics["lidar_local_planarity"] = MetricResult(
            value=local_planarity,
            reason=f"sampled local LiDAR planarity score is {local_planarity:g}",
        )
    if roughness is not None:
        metrics["lidar_map_roughness_m"] = MetricResult(
            value=roughness,
            unit="m",
            reason=f"sampled LiDAR map roughness proxy is {roughness:g} m",
        )
    if point_to_plane_train is not None or point_to_plane_holdout is not None:
        metrics["lidar_point_to_plane_rmse_m"] = MetricResult(
            train=point_to_plane_train,
            holdout=point_to_plane_holdout,
            unit="m",
            reason=(
                "sampled voxel-plane roughness is used as the alpha "
                "point-to-plane RMSE proxy"
            ),
        )
    if map_sharpness is not None:
        metrics["lidar_map_sharpness"] = MetricResult(
            value=map_sharpness,
            reason=f"sampled LiDAR map sharpness proxy is {map_sharpness:g}",
        )
    if world_map_diagnostics is not None:
        metrics.update(_lidar_world_map_metrics(world_map_diagnostics))
        metrics.update(
            _lidar_world_map_perturbation_metrics(
                inspection,
                config,
                world_map_diagnostics,
            )
        )
    metrics.update(
        _lidar_perturbation_metrics(
            inspection,
            config,
            baseline_train=point_to_plane_train,
            baseline_holdout=point_to_plane_holdout,
        )
    )
    return metrics


def lidar_rig_point_to_plane_metrics_from_evaluation(
    evaluation: LidarRigPointToPlaneEvaluation,
) -> dict[str, MetricResult]:
    """Build report-ready metrics from a native LiDAR point-to-plane factor."""

    residual_count_grade: Grade = "pass" if evaluation.residual_count > 0 else "warn"
    rank_grade: Grade = "pass" if evaluation.rank >= 6 else "warn"
    condition_grade: Grade = (
        "pass"
        if evaluation.normalized_condition_number_estimate is not None
        and evaluation.normalized_condition_number_estimate <= 1.0e8
        else "warn"
    )
    weak_dof_grade: Grade = "pass" if not evaluation.weak_directions else "warn"
    return {
        "lidar_rig_point_to_plane_residual_count": MetricResult(
            value=float(evaluation.residual_count),
            unit="residuals",
            grade=residual_count_grade,
            reason=f"native LiDAR rig point-to-plane residuals for {evaluation.variable}",
        ),
        "lidar_rig_point_to_plane_rmse_m": MetricResult(
            value=evaluation.rmse_m,
            unit="m",
            grade=residual_count_grade,
            reason=f"native LiDAR rig point-to-plane RMSE for {evaluation.variable}",
        ),
        "lidar_rig_point_to_plane_rank": MetricResult(
            value=float(evaluation.rank),
            grade=rank_grade,
            reason="rank of the native LiDAR rig point-to-plane normal equations",
        ),
        "lidar_rig_point_to_plane_condition_number": MetricResult(
            value=evaluation.normalized_condition_number_estimate,
            grade=condition_grade,
            reason=(
                "normalized local curvature proxy from native factor Hessian; "
                "dimensionless state=[translation/L, rotation], not a covariance"
            ),
        ),
        "lidar_rig_point_to_plane_normalization_length_m": MetricResult(
            value=evaluation.normalization_length_m,
            unit="m",
            grade="pass",
            reason="representative length L used to normalize translation and rotation DoF",
        ),
        "lidar_rig_point_to_plane_weak_dof_count": MetricResult(
            value=float(len(evaluation.weak_directions)),
            unit="dof",
            grade=weak_dof_grade,
            reason=(
                "weak native LiDAR point-to-plane DoF: "
                + ", ".join(evaluation.weak_directions)
                if evaluation.weak_directions
                else "normalized native LiDAR point-to-plane local curvature has no weak DoF"
            ),
        ),
    }


def _lidar_world_map_metrics(diagnostics: Mapping[object, object]) -> dict[str, MetricResult]:
    train_rmse = _float_or_none(diagnostics.get("point_to_plane_rmse_train_m"))
    holdout_rmse = _float_or_none(diagnostics.get("point_to_plane_rmse_holdout_m"))
    holdout_median = _float_or_none(diagnostics.get("point_to_plane_median_holdout_m"))
    holdout_p95 = _float_or_none(diagnostics.get("point_to_plane_p95_holdout_m"))
    train_voxels = _float_or_none(diagnostics.get("train_voxel_count"))
    holdout_residuals = _float_or_none(diagnostics.get("holdout_residual_count"))
    reason = diagnostics.get("reason")
    reason_text = str(reason) if reason is not None else None
    leakage = diagnostics.get("leakage_validation")
    leakage_diagnostics = leakage if isinstance(leakage, Mapping) else None
    metrics: dict[str, MetricResult] = {}
    if train_voxels is not None:
        metrics["lidar_world_map_train_voxel_count"] = MetricResult(
            value=train_voxels,
            unit="voxels",
            reason="train voxel-plane count from OXTS-projected LiDAR frames",
        )
    if holdout_residuals is not None:
        metrics["lidar_world_map_holdout_residual_count"] = MetricResult(
            value=holdout_residuals,
            unit="points",
            reason="holdout LiDAR points matched to train voxel planes",
        )
    if leakage_diagnostics is not None:
        issue_count = _float_or_none(leakage_diagnostics.get("issue_count"))
        status = str(leakage_diagnostics.get("status", "unknown"))
        metrics["lidar_world_map_leakage_issue_count"] = MetricResult(
            value=issue_count,
            unit="issues",
            grade="pass" if status == "pass" and issue_count == 0.0 else "fail",
            reason=str(leakage_diagnostics.get("reason", "world-map leakage validation")),
        )
    if train_rmse is not None or holdout_rmse is not None:
        metrics["lidar_world_map_point_to_plane_rmse_m"] = MetricResult(
            train=train_rmse,
            holdout=holdout_rmse,
            unit="m",
            reason=(
                "OXTS-projected train/holdout LiDAR point-to-plane consistency"
                if reason_text is None
                else reason_text
            ),
        )
    if holdout_median is not None:
        metrics["lidar_world_map_point_to_plane_median_holdout_m"] = MetricResult(
            value=holdout_median,
            unit="m",
            reason="median holdout point-to-plane residual against train LiDAR map",
        )
    if holdout_p95 is not None:
        metrics["lidar_world_map_point_to_plane_p95_holdout_m"] = MetricResult(
            value=holdout_p95,
            unit="m",
            reason="P95 holdout point-to-plane residual against train LiDAR map",
        )
    return metrics


def _lidar_world_map_perturbation_metrics(
    inspection: DatasetInspection,
    config: CalibrationConfig | None,
    diagnostics: Mapping[object, object],
) -> dict[str, MetricResult]:
    if config is None or inspection.dataset_type != "kitti_raw":
        return {}
    baseline_train = _float_or_none(diagnostics.get("point_to_plane_rmse_train_m"))
    baseline_holdout = _float_or_none(diagnostics.get("point_to_plane_rmse_holdout_m"))
    baseline_p95 = _float_or_none(diagnostics.get("point_to_plane_p95_holdout_m"))
    if baseline_train is None and baseline_holdout is None and baseline_p95 is None:
        return _unavailable_world_map_perturbation_metrics(
            "baseline OXTS-projected LiDAR map consistency is unavailable"
        )

    train_deltas: list[float] = []
    holdout_deltas: list[float] = []
    p95_deltas: list[float] = []
    dof_sensitivities: dict[str, list[float]] = {}
    measurable_cases = 0
    worsened_cases = 0
    for dof, perturbation in _perturbation_transforms(config):
        stats = summarize_lidar_world_map_consistency(
            inspection.path,
            t_ego_lidar=perturbation,
        )
        train_delta = _rmse_delta(
            baseline_train,
            stats.point_to_plane_rmse_train_m,
        )
        holdout_delta = _rmse_delta(
            baseline_holdout,
            stats.point_to_plane_rmse_holdout_m,
        )
        p95_delta = _rmse_delta(
            baseline_p95,
            stats.point_to_plane_p95_holdout_m,
        )
        if train_delta is not None:
            train_deltas.append(train_delta)
        if holdout_delta is not None:
            holdout_deltas.append(holdout_delta)
        if p95_delta is not None:
            p95_deltas.append(p95_delta)
        if train_delta is not None or holdout_delta is not None or p95_delta is not None:
            measurable_cases += 1
            dof_sensitivities.setdefault(dof, []).append(
                max(
                    abs(delta)
                    for delta in (train_delta, holdout_delta, p95_delta)
                    if delta is not None
                )
            )
            if any(
                delta is not None and delta > 1.0e-9
                for delta in (train_delta, holdout_delta, p95_delta)
            ):
                worsened_cases += 1

    if measurable_cases == 0:
        return _unavailable_world_map_perturbation_metrics(
            "perturbed OXTS-projected LiDAR map consistency could not be computed"
        )

    detectable_fraction = worsened_cases / measurable_cases
    grade: Grade = "pass" if detectable_fraction > 0.0 else "warn"
    metrics = {
        "lidar_world_map_perturbation_case_count": MetricResult(
            value=float(measurable_cases),
            unit="cases",
            grade="pass",
            reason=(
                "roll/pitch/yaw/x/y/z perturbation cases evaluated with "
                "OXTS-projected LiDAR train/holdout map consistency"
            ),
        ),
        "lidar_world_map_perturbation_detectable_fraction": MetricResult(
            value=detectable_fraction,
            grade=grade,
            reason=(
                "fraction of known LiDAR extrinsic perturbations that worsened "
                "OXTS-projected train, holdout, or P95 point-to-plane residuals"
            ),
        ),
        "lidar_world_map_perturbation_train_rmse_delta_mean_m": MetricResult(
            value=_mean_or_none(train_deltas),
            unit="m",
            grade=grade,
            reason=(
                "mean perturbed-minus-reference train RMSE against the "
                "OXTS-projected LiDAR map"
            ),
        ),
        "lidar_world_map_perturbation_holdout_rmse_delta_mean_m": MetricResult(
            value=_mean_or_none(holdout_deltas),
            unit="m",
            grade=grade,
            reason=(
                "mean perturbed-minus-reference holdout RMSE against the "
                "OXTS-projected LiDAR map"
            ),
        ),
        "lidar_world_map_perturbation_holdout_rmse_delta_max_m": MetricResult(
            value=max(holdout_deltas) if holdout_deltas else None,
            unit="m",
            grade=grade,
            reason="largest holdout RMSE increase among known world-map perturbations",
        ),
        "lidar_world_map_perturbation_p95_holdout_delta_mean_m": MetricResult(
            value=_mean_or_none(p95_deltas),
            unit="m",
            grade=grade,
            reason="mean perturbed-minus-reference P95 holdout residual",
        ),
    }
    metrics.update(_world_map_dof_sensitivity_metrics(dof_sensitivities))
    return metrics


def _unavailable_world_map_perturbation_metrics(reason: str) -> dict[str, MetricResult]:
    return {
        "lidar_world_map_perturbation_case_count": MetricResult(
            value=0.0,
            unit="cases",
            reason=reason,
        ),
        "lidar_world_map_perturbation_detectable_fraction": MetricResult(
            value=None,
            grade="warn",
            reason=reason,
        ),
        "lidar_world_map_perturbation_train_rmse_delta_mean_m": MetricResult(
            value=None,
            unit="m",
            grade="warn",
            reason=reason,
        ),
        "lidar_world_map_perturbation_holdout_rmse_delta_mean_m": MetricResult(
            value=None,
            unit="m",
            grade="warn",
            reason=reason,
        ),
        "lidar_world_map_perturbation_holdout_rmse_delta_max_m": MetricResult(
            value=None,
            unit="m",
            grade="warn",
            reason=reason,
        ),
        "lidar_world_map_perturbation_p95_holdout_delta_mean_m": MetricResult(
            value=None,
            unit="m",
            grade="warn",
            reason=reason,
        ),
        "lidar_world_map_weak_dof_count": MetricResult(
            value=None,
            unit="dof",
            grade="warn",
            reason=reason,
        ),
    }


def _world_map_dof_sensitivity_metrics(
    dof_sensitivities: dict[str, list[float]],
) -> dict[str, MetricResult]:
    metrics: dict[str, MetricResult] = {}
    weak_directions: list[str] = []
    min_sensitivity: float | None = None
    for dof, (metric_name, direction_name) in sorted(_WORLD_MAP_DOF_METRICS.items()):
        values = dof_sensitivities.get(dof, [])
        sensitivity = max(values) if values else None
        if sensitivity is not None:
            min_sensitivity = (
                sensitivity
                if min_sensitivity is None
                else min(min_sensitivity, sensitivity)
            )
        is_weak = sensitivity is None or sensitivity < _WORLD_MAP_WEAK_DOF_DELTA_M
        if is_weak:
            weak_directions.append(direction_name)
        metrics[metric_name] = MetricResult(
            value=sensitivity,
            unit="m",
            grade="warn" if is_weak else "pass",
            reason=(
                f"max absolute OXTS world-map residual change for {direction_name}; "
                f"weak if < {_WORLD_MAP_WEAK_DOF_DELTA_M:g} m"
            ),
        )

    weak_reason = (
        "weak LiDAR world-map DoF from perturbation sensitivity: "
        + ", ".join(weak_directions)
        if weak_directions
        else "all configured LiDAR world-map perturbation DoF exceeded sensitivity threshold"
    )
    metrics["lidar_world_map_weak_dof_count"] = MetricResult(
        value=float(len(weak_directions)),
        unit="dof",
        grade="warn" if weak_directions else "pass",
        reason=weak_reason,
    )
    metrics["lidar_world_map_min_dof_sensitivity_m"] = MetricResult(
        value=min_sensitivity,
        unit="m",
        grade=(
            "warn"
            if min_sensitivity is None or min_sensitivity < _WORLD_MAP_WEAK_DOF_DELTA_M
            else "pass"
        ),
        reason=(
            "minimum DoF sensitivity from OXTS-projected LiDAR map perturbation sweep"
        ),
    )
    return metrics


def _lidar_perturbation_metrics(
    inspection: DatasetInspection,
    config: CalibrationConfig | None,
    *,
    baseline_train: float | None,
    baseline_holdout: float | None,
) -> dict[str, MetricResult]:
    if config is None or inspection.dataset_type != "kitti_raw":
        return {}
    if baseline_train is None and baseline_holdout is None:
        return _unavailable_perturbation_metrics("baseline point-to-plane RMSE is unavailable")

    train_deltas: list[float] = []
    holdout_deltas: list[float] = []
    measurable_cases = 0
    worsened_cases = 0
    for _, perturbation in _perturbation_transforms(config):
        stats = summarize_velodyne_points(inspection.path, point_transform=perturbation)
        train_delta = _rmse_delta(
            baseline_train,
            stats.point_to_plane_rmse_train_m,
        )
        holdout_delta = _rmse_delta(
            baseline_holdout,
            stats.point_to_plane_rmse_holdout_m,
        )
        if train_delta is not None:
            train_deltas.append(train_delta)
        if holdout_delta is not None:
            holdout_deltas.append(holdout_delta)
        if train_delta is not None or holdout_delta is not None:
            measurable_cases += 1
            if (train_delta is not None and train_delta > 1.0e-9) or (
                holdout_delta is not None and holdout_delta > 1.0e-9
            ):
                worsened_cases += 1

    if measurable_cases == 0:
        return _unavailable_perturbation_metrics(
            "perturbed point-to-plane RMSE could not be computed"
        )

    detectable_fraction = worsened_cases / measurable_cases
    grade: Grade = "pass" if detectable_fraction > 0.0 else "warn"
    return {
        "lidar_perturbation_case_count": MetricResult(
            value=float(measurable_cases),
            unit="cases",
            grade="pass",
            reason=(
                "roll/pitch/yaw/x/y/z perturbation cases evaluated with "
                "LiDAR voxel-plane train/holdout roughness"
            ),
        ),
        "lidar_perturbation_detectable_fraction": MetricResult(
            value=detectable_fraction,
            grade=grade,
            reason=(
                "fraction of known LiDAR extrinsic perturbations that worsened "
                "train or holdout point-to-plane RMSE"
            ),
        ),
        "lidar_perturbation_train_rmse_delta_mean_m": MetricResult(
            value=_mean_or_none(train_deltas),
            unit="m",
            grade=grade,
            reason=(
                "mean perturbed-minus-reference train point-to-plane RMSE; "
                "positive means the reference calibration ranks better"
            ),
        ),
        "lidar_perturbation_holdout_rmse_delta_mean_m": MetricResult(
            value=_mean_or_none(holdout_deltas),
            unit="m",
            grade=grade,
            reason=(
                "mean perturbed-minus-reference holdout point-to-plane RMSE; "
                "positive means the reference calibration ranks better"
            ),
        ),
        "lidar_perturbation_holdout_rmse_delta_max_m": MetricResult(
            value=max(holdout_deltas) if holdout_deltas else None,
            unit="m",
            grade=grade,
            reason="largest holdout RMSE increase among known LiDAR perturbations",
        ),
    }


def _unavailable_perturbation_metrics(reason: str) -> dict[str, MetricResult]:
    return {
        "lidar_perturbation_case_count": MetricResult(
            value=0.0,
            unit="cases",
            reason=reason,
        ),
        "lidar_perturbation_detectable_fraction": MetricResult(
            value=None,
            grade="warn",
            reason=reason,
        ),
        "lidar_perturbation_train_rmse_delta_mean_m": MetricResult(
            value=None,
            unit="m",
            grade="warn",
            reason=reason,
        ),
        "lidar_perturbation_holdout_rmse_delta_mean_m": MetricResult(
            value=None,
            unit="m",
            grade="warn",
            reason=reason,
        ),
        "lidar_perturbation_holdout_rmse_delta_max_m": MetricResult(
            value=None,
            unit="m",
            grade="warn",
            reason=reason,
        ),
    }


def _rmse_delta(baseline: float | None, perturbed: float | None) -> float | None:
    if baseline is None or perturbed is None:
        return None
    return perturbed - baseline


def _perturbation_transforms(config: CalibrationConfig) -> list[tuple[str, SE3]]:
    transforms: list[tuple[str, SE3]] = []
    for angle_deg in config.evaluation.kitti.perturbation_rotation_deg:
        angle_rad = math.radians(angle_deg)
        transforms.extend(
            [
                ("roll_deg", _rotation_perturbation((1.0, 0.0, 0.0), angle_rad)),
                ("roll_deg", _rotation_perturbation((1.0, 0.0, 0.0), -angle_rad)),
                ("pitch_deg", _rotation_perturbation((0.0, 1.0, 0.0), angle_rad)),
                ("pitch_deg", _rotation_perturbation((0.0, 1.0, 0.0), -angle_rad)),
                ("yaw_deg", _rotation_perturbation((0.0, 0.0, 1.0), angle_rad)),
                ("yaw_deg", _rotation_perturbation((0.0, 0.0, 1.0), -angle_rad)),
            ]
        )
    for offset_m in config.evaluation.kitti.perturbation_translation_m:
        transforms.extend(
            [
                ("x_m", SE3((offset_m, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))),
                ("x_m", SE3((-offset_m, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))),
                ("y_m", SE3((0.0, offset_m, 0.0), (0.0, 0.0, 0.0, 1.0))),
                ("y_m", SE3((0.0, -offset_m, 0.0), (0.0, 0.0, 0.0, 1.0))),
                ("z_m", SE3((0.0, 0.0, offset_m), (0.0, 0.0, 0.0, 1.0))),
                ("z_m", SE3((0.0, 0.0, -offset_m), (0.0, 0.0, 0.0, 1.0))),
            ]
        )
    return transforms


def _rotation_perturbation(axis: tuple[float, float, float], angle_rad: float) -> SE3:
    return SE3((0.0, 0.0, 0.0), _axis_angle_quaternion(axis, angle_rad))


def _axis_angle_quaternion(axis: tuple[float, float, float], angle_rad: float) -> QuaternionXYZW:
    half_angle = angle_rad / 2.0
    scale = math.sin(half_angle)
    return (axis[0] * scale, axis[1] * scale, axis[2] * scale, math.cos(half_angle))


def _mean_or_none(values: Iterable[float]) -> float | None:
    values_list = list(values)
    if not values_list:
        return None
    return sum(values_list) / len(values_list)


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _vector3_or_none(value: object) -> tuple[float, float, float] | None:
    if not isinstance(value, list | tuple) or len(value) != 3:
        return None
    vector: list[float] = []
    for item in value:
        numeric = _float_or_none(item)
        if numeric is None:
            return None
        vector.append(numeric)
    return (vector[0], vector[1], vector[2])
