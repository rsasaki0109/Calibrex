"""LiDAR-specific metric extraction helpers."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from calibrex.core.config import CalibrationConfig
from calibrex.core.geometry import SE3, QuaternionXYZW, Vector3
from calibrex.core.report_artifacts import EvidenceCaseItem
from calibrex.core.result import Grade, MetricResult
from calibrex.data.inspect import DatasetInspection
from calibrex.data.kitti import (
    summarize_lidar_world_map_consistency,
    summarize_velodyne_points,
)
from calibrex.data.livox import (
    LivoxPairPointToPlaneStats,
    LivoxPCDDatasetStats,
    LivoxPointRecord,
    build_voxel_plane_map,
    nearest_voxel_plane,
    summarize_livox_pair_point_to_plane,
    summarize_livox_pcd,
)
from calibrex.graph.lidar_point_to_plane import (
    LidarPointToPlaneObservation,
    LidarRigPointToPlaneEvaluation,
)

_WORLD_MAP_WEAK_DOF_DELTA_M = 1.0e-3
_LIVOX_PAIR_MANDATORY_ROTATION_DEG = 1.0
_LIVOX_PAIR_MANDATORY_TRANSLATION_M = 0.10
_LIVOX_PAIR_MIN_SUPPORT_RATIO = 0.10
_LIVOX_PAIR_MIN_ACCEPTED_CORRESPONDENCE_COUNT = 500.0
_LIVOX_PAIR_MANDATORY_SUPPORTED_DETECTION_TARGET = 8
_WORLD_MAP_DOF_METRICS = {
    "roll_deg": ("lidar_world_map_sensitivity_roll_m", "roll_lidar0"),
    "pitch_deg": ("lidar_world_map_sensitivity_pitch_m", "pitch_lidar0"),
    "yaw_deg": ("lidar_world_map_sensitivity_yaw_m", "yaw_lidar0"),
    "x_m": ("lidar_world_map_sensitivity_x_m", "x_lidar0"),
    "y_m": ("lidar_world_map_sensitivity_y_m", "y_lidar0"),
    "z_m": ("lidar_world_map_sensitivity_z_m", "z_lidar0"),
}


@dataclass(frozen=True)
class LivoxPairEvidenceEvaluation:
    """Detailed Livox pair evidence evaluation."""

    metrics: dict[str, MetricResult]
    cases: list[EvidenceCaseItem]
    point_to_plane: LivoxPairPointToPlaneStats
    known_bad_challenge: dict[str, object]


def lidar_metrics_from_inspection(
    inspection: DatasetInspection,
    config: CalibrationConfig | None = None,
) -> dict[str, MetricResult]:
    """Build LiDAR coverage metrics from dataset inspection diagnostics."""

    velodyne = inspection.diagnostics.get("velodyne_points")
    if not isinstance(velodyne, Mapping):
        livox = inspection.diagnostics.get("livox_pcd")
        if isinstance(livox, Mapping):
            return _livox_pcd_metrics(livox)
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


def _livox_pcd_metrics(diagnostics: Mapping[str, object]) -> dict[str, MetricResult]:
    sampled_file_count = _float_or_none(diagnostics.get("sampled_file_count"))
    sampled_point_count = _float_or_none(diagnostics.get("sampled_point_count"))
    bounds_min = _vector3_or_none(diagnostics.get("bounds_min_m"))
    bounds_max = _vector3_or_none(diagnostics.get("bounds_max_m"))
    shared_voxel_count = _float_or_none(diagnostics.get("pair_shared_voxel_count"))
    unmatched_source_voxel_count = _float_or_none(
        diagnostics.get("pair_unmatched_source_voxel_count")
    )
    unmatched_target_voxel_count = _float_or_none(
        diagnostics.get("pair_unmatched_target_voxel_count")
    )
    source_recall = _float_or_none(diagnostics.get("pair_source_voxel_recall_in_target"))
    target_recall = _float_or_none(diagnostics.get("pair_target_voxel_recall_in_source"))
    centroid_rmse = _float_or_none(diagnostics.get("pair_shared_voxel_centroid_rmse_m"))

    metrics: dict[str, MetricResult] = {}
    if sampled_file_count is not None:
        metrics["lidar_frame_coverage"] = MetricResult(
            value=sampled_file_count,
            unit="frames",
            reason=f"inspected {sampled_file_count:g} Livox solid-state PCD frames",
        )
    if sampled_point_count is not None:
        metrics["lidar_point_coverage"] = MetricResult(
            value=sampled_point_count,
            unit="points",
            reason=f"sampled {sampled_point_count:g} Livox PCD points",
        )
    if bounds_min is not None and bounds_max is not None:
        extent = math.dist(bounds_min, bounds_max)
        metrics["lidar_spatial_coverage_m"] = MetricResult(
            value=extent,
            unit="m",
            reason=f"sampled Livox XYZ extent is {extent:g} m",
        )
    if shared_voxel_count is not None:
        metrics["lidar_pair_shared_voxel_count"] = MetricResult(
            value=shared_voxel_count,
            unit="voxels",
            grade="pass" if shared_voxel_count > 0 else "warn",
            reason="coarse shared voxel count between the first two Livox PCD frames",
        )
    if unmatched_source_voxel_count is not None:
        metrics["lidar_pair_unmatched_source_voxel_count"] = MetricResult(
            value=unmatched_source_voxel_count,
            unit="voxels",
            grade="pass",
            reason="source Livox voxels without a target voxel at the configured voxel size",
        )
    if unmatched_target_voxel_count is not None:
        metrics["lidar_pair_unmatched_target_voxel_count"] = MetricResult(
            value=unmatched_target_voxel_count,
            unit="voxels",
            grade="pass",
            reason="target Livox voxels without a source voxel at the configured voxel size",
        )
    if source_recall is not None:
        metrics["lidar_pair_source_voxel_recall_in_target"] = MetricResult(
            value=source_recall,
            grade="pass" if source_recall > 0.0 else "warn",
            reason=(
                "fraction of source Livox voxels that have a target voxel at the "
                "configured voxel size"
            ),
        )
    if target_recall is not None:
        metrics["lidar_pair_target_voxel_recall_in_source"] = MetricResult(
            value=target_recall,
            grade="pass" if target_recall > 0.0 else "warn",
            reason=(
                "fraction of target Livox voxels that have a source voxel at the "
                "configured voxel size"
            ),
        )
    if centroid_rmse is not None:
        metrics["lidar_pair_shared_voxel_centroid_rmse_m"] = MetricResult(
            value=centroid_rmse,
            unit="m",
            grade="pass",
            reason=(
                "coarse shared-voxel centroid RMSE for the first two Livox PCD frames"
            ),
        )
    return metrics


def _livox_pair_point_to_plane_metrics(
    stats: LivoxPairPointToPlaneStats,
) -> dict[str, MetricResult]:
    support_grade: Grade = "pass" if stats.matched_point_count > 0 else "warn"
    reason_suffix = (
        "single source/target PCD pair; limited holdout geometry evidence, "
        "not independent temporal validation"
    )
    return {
        "lidar_pair_holdout_point_to_plane_eligible_point_count": MetricResult(
            value=float(stats.eligible_point_count),
            unit="points",
            grade="pass" if stats.eligible_point_count > 0 else "warn",
            reason=(
                "candidate-independent target point population used as the "
                f"support denominator; {reason_suffix}"
            ),
        ),
        "lidar_pair_holdout_point_to_plane_considered_point_count": MetricResult(
            value=float(stats.considered_point_count),
            unit="points",
            grade="pass" if stats.considered_point_count > 0 else "warn",
            reason=(
                "target points considered for source-plane matching after decoding; "
                f"{reason_suffix}"
            ),
        ),
        "lidar_pair_holdout_point_to_plane_accepted_correspondence_count": MetricResult(
            value=float(stats.accepted_correspondence_count),
            unit="points",
            grade=support_grade,
            reason=(
                "candidate-dependent accepted point-to-plane correspondences under "
                f"the fixed support denominator; {reason_suffix}"
            ),
        ),
        "lidar_pair_holdout_point_to_plane_support_ratio": MetricResult(
            value=stats.support_ratio,
            grade=support_grade,
            reason=(
                "accepted correspondences divided by candidate-independent eligible "
                f"target points; {reason_suffix}"
            ),
        ),
        "lidar_pair_holdout_point_to_plane_map_voxel_count": MetricResult(
            value=float(stats.map_voxel_count),
            unit="voxels",
            grade=support_grade,
            reason=(
                f"source-frame voxel planes built for Livox pair point-to-plane; "
                f"{reason_suffix}"
            ),
        ),
        "lidar_pair_holdout_point_to_plane_matched_point_count": MetricResult(
            value=float(stats.matched_point_count),
            unit="points",
            grade=support_grade,
            reason=(
                "transformed target points matched to source voxel planes; "
                f"{reason_suffix}"
            ),
        ),
        "lidar_pair_holdout_point_to_plane_unmatched_fraction": MetricResult(
            value=stats.unmatched_fraction,
            grade=support_grade,
            reason=(
                "fraction of transformed target points without a source voxel-plane "
                f"correspondence; {reason_suffix}"
            ),
        ),
        "lidar_pair_holdout_point_to_plane_median_abs_m": MetricResult(
            value=stats.median_abs_m,
            unit="m",
            grade=support_grade,
            reason=f"median absolute target-to-source plane residual; {reason_suffix}",
        ),
        "lidar_pair_holdout_point_to_plane_p90_abs_m": MetricResult(
            value=stats.p90_abs_m,
            unit="m",
            grade=support_grade,
            reason=f"P90 absolute target-to-source plane residual; {reason_suffix}",
        ),
        "lidar_pair_holdout_point_to_plane_rmse_m": MetricResult(
            value=stats.rmse_m,
            unit="m",
            grade=support_grade,
            reason=f"RMSE target-to-source plane residual; {reason_suffix}",
        ),
        "lidar_pair_holdout_point_to_plane_inlier_fraction": MetricResult(
            value=stats.inlier_fraction,
            grade=support_grade,
            reason=(
                f"fraction of matched residuals <= {stats.inlier_threshold_m:g} m; "
                f"{reason_suffix}"
            ),
        ),
    }


def livox_pair_metrics_from_dataset(
    dataset_path: str,
    target_transform: SE3,
    config: CalibrationConfig,
) -> dict[str, MetricResult]:
    """Evaluate Livox pair evidence after applying a candidate target transform."""

    return livox_pair_evidence_from_dataset(
        dataset_path=dataset_path,
        target_transform=target_transform,
        config=config,
    ).metrics


def livox_pair_evidence_from_dataset(
    dataset_path: str,
    target_transform: SE3,
    config: CalibrationConfig,
) -> LivoxPairEvidenceEvaluation:
    """Evaluate Livox pair evidence and retain per-probe details."""

    baseline = summarize_livox_pcd(dataset_path, target_transform=target_transform)
    point_to_plane = summarize_livox_pair_point_to_plane(
        dataset_path,
        target_transform=target_transform,
    )
    metrics = _livox_pcd_metrics(baseline.as_dict())
    metrics.update(_livox_pair_point_to_plane_metrics(point_to_plane))
    known_bad_metrics, cases = _livox_pair_known_bad_evidence(
        dataset_path=dataset_path,
        target_transform=target_transform,
        baseline=baseline,
        baseline_point_to_plane=point_to_plane,
        config=config,
    )
    metrics.update(known_bad_metrics)
    return LivoxPairEvidenceEvaluation(
        metrics=metrics,
        cases=cases,
        point_to_plane=point_to_plane,
        known_bad_challenge=_livox_pair_known_bad_challenge_payload(cases),
    )


def _livox_pair_known_bad_evidence(
    *,
    dataset_path: str,
    target_transform: SE3,
    baseline: LivoxPCDDatasetStats,
    baseline_point_to_plane: LivoxPairPointToPlaneStats,
    config: CalibrationConfig,
) -> tuple[dict[str, MetricResult], list[EvidenceCaseItem]]:
    baseline_recall = baseline.pair_source_voxel_recall_in_target
    baseline_rmse = baseline.pair_shared_voxel_centroid_rmse_m
    baseline_p2p_p90 = baseline_point_to_plane.p90_abs_m
    baseline_p2p_rmse = baseline_point_to_plane.rmse_m
    if (
        baseline_recall is None
        and baseline_rmse is None
        and baseline_p2p_p90 is None
        and baseline_p2p_rmse is None
    ):
        return (
            _unavailable_livox_pair_known_bad_metrics(
                "baseline Livox pair evidence is unavailable"
            ),
            [],
        )

    recall_deltas: list[float] = []
    rmse_deltas: list[float] = []
    p2p_p90_deltas: list[float] = []
    p2p_rmse_deltas: list[float] = []
    cases: list[EvidenceCaseItem] = []
    measurable_cases = 0
    worsened_cases = 0
    for dof, amount, unit, perturbation in _perturbation_cases(config):
        perturbed_transform = perturbation.compose(target_transform)
        perturbed = summarize_livox_pcd(dataset_path, target_transform=perturbed_transform)
        perturbed_point_to_plane = summarize_livox_pair_point_to_plane(
            dataset_path,
            target_transform=perturbed_transform,
        )
        recall_delta = _recall_delta(
            baseline_recall,
            perturbed.pair_source_voxel_recall_in_target,
        )
        rmse_delta = _rmse_delta(
            baseline_rmse,
            perturbed.pair_shared_voxel_centroid_rmse_m,
        )
        p2p_p90_delta = _rmse_delta(
            baseline_p2p_p90,
            perturbed_point_to_plane.p90_abs_m,
        )
        p2p_rmse_delta = _rmse_delta(
            baseline_p2p_rmse,
            perturbed_point_to_plane.rmse_m,
        )
        support_ratio_delta = _ratio_delta(
            baseline_point_to_plane.support_ratio,
            perturbed_point_to_plane.support_ratio,
        )
        if recall_delta is not None:
            recall_deltas.append(recall_delta)
        if rmse_delta is not None:
            rmse_deltas.append(rmse_delta)
        if p2p_p90_delta is not None:
            p2p_p90_deltas.append(p2p_p90_delta)
        if p2p_rmse_delta is not None:
            p2p_rmse_deltas.append(p2p_rmse_delta)
        if any(
            value is not None
            for value in (recall_delta, rmse_delta, p2p_p90_delta, p2p_rmse_delta)
        ):
            measurable_cases += 1
            worsened = any(
                value is not None and value > 1.0e-9
                for value in (recall_delta, rmse_delta, p2p_p90_delta, p2p_rmse_delta)
            )
            if worsened:
                worsened_cases += 1
            cases.append(
                _livox_pair_known_bad_case(
                    dof=dof,
                    amount=amount,
                    unit=unit,
                    perturbed=perturbed,
                    perturbed_point_to_plane=perturbed_point_to_plane,
                    recall_delta=recall_delta,
                    rmse_delta=rmse_delta,
                    p2p_p90_delta=p2p_p90_delta,
                    p2p_rmse_delta=p2p_rmse_delta,
                    support_ratio_delta=support_ratio_delta,
                    worsened=worsened,
                )
            )

    if measurable_cases == 0:
        return (
            _unavailable_livox_pair_known_bad_metrics(
                "known-bad Livox pair perturbations could not be scored"
            ),
            [],
        )

    detectable_fraction = worsened_cases / measurable_cases
    grade: Grade = "pass" if detectable_fraction > 0.0 else "warn"
    challenge = _livox_pair_known_bad_challenge_payload(cases)
    mandatory_fraction = _float_or_none(
        challenge.get("mandatory_supported_detection_fraction")
    )
    mandatory_case_count = _float_or_none(challenge.get("mandatory_case_count"))
    mandatory_supported_detection_count = _float_or_none(
        challenge.get("mandatory_supported_detection_count")
    )
    mandatory_support_collapse_count = _float_or_none(
        challenge.get("mandatory_support_collapse_count")
    )
    mandatory_grade: Grade = (
        "pass"
        if mandatory_supported_detection_count is not None
        and mandatory_supported_detection_count
        >= _LIVOX_PAIR_MANDATORY_SUPPORTED_DETECTION_TARGET
        else "warn"
    )
    metrics = {
        "lidar_pair_known_bad_case_count": MetricResult(
            value=float(measurable_cases),
            unit="cases",
            grade="pass",
            reason=(
                "left-multiplied source-frame roll/pitch/yaw/x/y/z perturbation "
                "cases evaluated against Livox pair voxel and holdout plane evidence"
            ),
        ),
        "lidar_pair_known_bad_detectable_fraction": MetricResult(
            value=detectable_fraction,
            grade=grade,
            reason=(
                "fraction of known-bad Livox pair perturbations that reduced "
                "source voxel recall, increased shared-voxel centroid RMSE, or "
                "increased holdout point-to-plane residual"
            ),
        ),
        "lidar_pair_known_bad_source_recall_delta_mean": MetricResult(
            value=_mean_or_none(recall_deltas),
            grade=grade,
            reason=(
                "mean baseline-minus-perturbed source voxel recall; positive "
                "means the candidate ranks better than the known-bad controls"
            ),
        ),
        "lidar_pair_known_bad_centroid_rmse_delta_mean_m": MetricResult(
            value=_mean_or_none(rmse_deltas),
            unit="m",
            grade=grade,
            reason=(
                "mean perturbed-minus-baseline shared-voxel centroid RMSE; "
                "positive means the candidate ranks better than the known-bad controls"
            ),
        ),
        "lidar_pair_known_bad_centroid_rmse_delta_max_m": MetricResult(
            value=max(rmse_deltas) if rmse_deltas else None,
            unit="m",
            grade=grade,
            reason="largest shared-voxel centroid RMSE increase among known-bad controls",
        ),
        "lidar_pair_known_bad_point_to_plane_p90_delta_max_m": MetricResult(
            value=max(p2p_p90_deltas) if p2p_p90_deltas else None,
            unit="m",
            grade=grade,
            reason="largest holdout point-to-plane P90 increase among known-bad controls",
        ),
        "lidar_pair_known_bad_point_to_plane_rmse_delta_mean_m": MetricResult(
            value=_mean_or_none(p2p_rmse_deltas),
            unit="m",
            grade=grade,
            reason=(
                "mean perturbed-minus-baseline holdout point-to-plane RMSE; "
                "positive means the candidate ranks better than the known-bad controls"
            ),
        ),
        "lidar_pair_known_bad_mandatory_case_count": MetricResult(
            value=mandatory_case_count,
            unit="cases",
            grade="pass" if mandatory_case_count == 12.0 else "warn",
            reason=(
                "predeclared large ±1 deg and ±0.10 m known-bad control cases "
                "materialized for roll/pitch/yaw/x/y/z"
            ),
        ),
        "lidar_pair_known_bad_mandatory_supported_detection_count": MetricResult(
            value=mandatory_supported_detection_count,
            unit="cases",
            grade=mandatory_grade,
            reason=(
                "mandatory known-bad controls detected under sufficient support; "
                f"target >= {_LIVOX_PAIR_MANDATORY_SUPPORTED_DETECTION_TARGET} cases"
            ),
        ),
        "lidar_pair_known_bad_mandatory_supported_detection_fraction": MetricResult(
            value=mandatory_fraction,
            grade=mandatory_grade,
            reason=(
                "fraction of mandatory known-bad controls detected under sufficient "
                "support, excluding support-collapse-only detections"
            ),
        ),
        "lidar_pair_known_bad_mandatory_support_collapse_count": MetricResult(
            value=mandatory_support_collapse_count,
            unit="cases",
            grade="pass" if mandatory_support_collapse_count == 0.0 else "warn",
            reason=(
                "mandatory known-bad controls whose apparent detection is dominated "
                "by insufficient point-to-plane support"
            ),
        ),
    }
    return (metrics, cases)


def _livox_pair_known_bad_case(
    *,
    dof: str,
    amount: float,
    unit: str,
    perturbed: LivoxPCDDatasetStats,
    perturbed_point_to_plane: LivoxPairPointToPlaneStats,
    recall_delta: float | None,
    rmse_delta: float | None,
    p2p_p90_delta: float | None,
    p2p_rmse_delta: float | None,
    support_ratio_delta: float | None,
    worsened: bool,
) -> EvidenceCaseItem:
    return EvidenceCaseItem(
        family="lidar_pair",
        case_id=f"{dof}:{amount:+g}{unit}",
        check="Known-Bad Controls",
        status="pass" if worsened else "warn",
        dof=dof,
        amount=amount,
        unit=unit,
        convention="left-multiplied source-frame SE(3) perturbation",
        metric_values={
            "lidar_pair_source_voxel_recall_in_target": _float_or_none(
                perturbed.pair_source_voxel_recall_in_target
            ),
            "lidar_pair_shared_voxel_count": _float_or_none(
                perturbed.pair_shared_voxel_count
            ),
            "lidar_pair_shared_voxel_centroid_rmse_m": _float_or_none(
                perturbed.pair_shared_voxel_centroid_rmse_m
            ),
            "lidar_pair_holdout_point_to_plane_p90_abs_m": _float_or_none(
                perturbed_point_to_plane.p90_abs_m
            ),
            "lidar_pair_holdout_point_to_plane_rmse_m": _float_or_none(
                perturbed_point_to_plane.rmse_m
            ),
            "lidar_pair_holdout_point_to_plane_unmatched_fraction": _float_or_none(
                perturbed_point_to_plane.unmatched_fraction
            ),
            "lidar_pair_holdout_point_to_plane_support_ratio": _float_or_none(
                perturbed_point_to_plane.support_ratio
            ),
            "lidar_pair_holdout_point_to_plane_eligible_point_count": _float_or_none(
                perturbed_point_to_plane.eligible_point_count
            ),
            "lidar_pair_holdout_point_to_plane_accepted_correspondence_count": (
                _float_or_none(perturbed_point_to_plane.accepted_correspondence_count)
            ),
        },
        delta_values={
            "source_recall_delta": recall_delta,
            "centroid_rmse_delta_m": rmse_delta,
            "point_to_plane_p90_delta_m": p2p_p90_delta,
            "point_to_plane_rmse_delta_m": p2p_rmse_delta,
            "support_ratio_delta": support_ratio_delta,
        },
    )


def _livox_pair_known_bad_challenge_payload(
    cases: list[EvidenceCaseItem],
) -> dict[str, object]:
    mandatory_cases = [case for case in cases if _is_mandatory_livox_pair_case(case)]
    supported_detection_count = sum(
        1 for case in mandatory_cases if _is_supported_livox_pair_detection(case)
    )
    support_collapse_count = sum(
        1
        for case in mandatory_cases
        if case.status == "pass" and not _has_sufficient_livox_pair_support(case)
    )
    mandatory_count = len(mandatory_cases)
    return {
        "challenge_id": "livox_pair_mandatory_6dof_large_controls/v0.1",
        "composition": "T_test = Exp(xi_hat) * T_source_target",
        "tangent_frame": "source_lidar_frame",
        "mandatory_rotation_deg": _LIVOX_PAIR_MANDATORY_ROTATION_DEG,
        "mandatory_translation_m": _LIVOX_PAIR_MANDATORY_TRANSLATION_M,
        "mandatory_case_count": mandatory_count,
        "mandatory_supported_detection_count": supported_detection_count,
        "mandatory_supported_detection_fraction": (
            supported_detection_count / mandatory_count if mandatory_count else None
        ),
        "mandatory_detection_by_dof": _mandatory_detection_by_dof(mandatory_cases),
        "mandatory_support_collapse_count": support_collapse_count,
        "min_support_ratio": _LIVOX_PAIR_MIN_SUPPORT_RATIO,
        "min_accepted_correspondence_count": (
            _LIVOX_PAIR_MIN_ACCEPTED_CORRESPONDENCE_COUNT
        ),
        "target_supported_detection_count": (
            _LIVOX_PAIR_MANDATORY_SUPPORTED_DETECTION_TARGET
        ),
    }


def _mandatory_detection_by_dof(
    mandatory_cases: list[EvidenceCaseItem],
) -> dict[str, dict[str, object]]:
    by_dof: dict[str, dict[str, object]] = {}
    for case in mandatory_cases:
        if case.dof is None:
            continue
        state = by_dof.setdefault(
            case.dof,
            {
                "case_count": 0,
                "detected_count": 0,
                "supported_detection_count": 0,
                "support_collapse_count": 0,
                "amounts": [],
            },
        )
        state["case_count"] = _int_counter(state["case_count"]) + 1
        amounts = state["amounts"]
        if isinstance(amounts, list) and case.amount is not None:
            amounts.append(case.amount)
        if case.status != "pass":
            continue
        state["detected_count"] = _int_counter(state["detected_count"]) + 1
        if _has_sufficient_livox_pair_support(case):
            if _is_supported_livox_pair_detection(case):
                state["supported_detection_count"] = (
                    _int_counter(state["supported_detection_count"]) + 1
                )
        else:
            state["support_collapse_count"] = (
                _int_counter(state["support_collapse_count"]) + 1
            )
    return by_dof


def _int_counter(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


def _is_mandatory_livox_pair_case(case: EvidenceCaseItem) -> bool:
    if case.unit == "deg":
        return abs(case.amount or 0.0) >= _LIVOX_PAIR_MANDATORY_ROTATION_DEG
    if case.unit == "m":
        return abs(case.amount or 0.0) >= _LIVOX_PAIR_MANDATORY_TRANSLATION_M
    return False


def _has_sufficient_livox_pair_support(case: EvidenceCaseItem) -> bool:
    support_ratio = _float_or_none(
        case.metric_values.get("lidar_pair_holdout_point_to_plane_support_ratio")
    )
    accepted_count = _float_or_none(
        case.metric_values.get(
            "lidar_pair_holdout_point_to_plane_accepted_correspondence_count"
        )
    )
    if support_ratio is None or accepted_count is None:
        return False
    return (
        support_ratio >= _LIVOX_PAIR_MIN_SUPPORT_RATIO
        and accepted_count >= _LIVOX_PAIR_MIN_ACCEPTED_CORRESPONDENCE_COUNT
    )


def _is_supported_livox_pair_detection(case: EvidenceCaseItem) -> bool:
    if case.status != "pass" or not _has_sufficient_livox_pair_support(case):
        return False
    return any(
        (delta := _float_or_none(case.delta_values.get(delta_name))) is not None
        and delta > 1.0e-9
        for delta_name in (
            "centroid_rmse_delta_m",
            "point_to_plane_p90_delta_m",
            "point_to_plane_rmse_delta_m",
        )
    )


def _unavailable_livox_pair_known_bad_metrics(reason: str) -> dict[str, MetricResult]:
    return {
        "lidar_pair_known_bad_case_count": MetricResult(
            value=0.0,
            unit="cases",
            reason=reason,
        ),
        "lidar_pair_known_bad_detectable_fraction": MetricResult(
            value=None,
            grade="warn",
            reason=reason,
        ),
        "lidar_pair_known_bad_source_recall_delta_mean": MetricResult(
            value=None,
            grade="warn",
            reason=reason,
        ),
        "lidar_pair_known_bad_centroid_rmse_delta_mean_m": MetricResult(
            value=None,
            unit="m",
            grade="warn",
            reason=reason,
        ),
        "lidar_pair_known_bad_centroid_rmse_delta_max_m": MetricResult(
            value=None,
            unit="m",
            grade="warn",
            reason=reason,
        ),
        "lidar_pair_known_bad_point_to_plane_p90_delta_max_m": MetricResult(
            value=None,
            unit="m",
            grade="warn",
            reason=reason,
        ),
        "lidar_pair_known_bad_point_to_plane_rmse_delta_mean_m": MetricResult(
            value=None,
            unit="m",
            grade="warn",
            reason=reason,
        ),
        "lidar_pair_known_bad_mandatory_case_count": MetricResult(
            value=0.0,
            unit="cases",
            grade="warn",
            reason=reason,
        ),
        "lidar_pair_known_bad_mandatory_supported_detection_count": MetricResult(
            value=None,
            unit="cases",
            grade="warn",
            reason=reason,
        ),
        "lidar_pair_known_bad_mandatory_supported_detection_fraction": MetricResult(
            value=None,
            grade="warn",
            reason=reason,
        ),
        "lidar_pair_known_bad_mandatory_support_collapse_count": MetricResult(
            value=None,
            unit="cases",
            grade="warn",
            reason=reason,
        ),
    }


def build_rig_point_to_plane_observations(
    *,
    source_records: list[LivoxPointRecord],
    target_points: list[Vector3],
    initial_t_source_target: SE3,
    voxel_size_m: float = 1.0,
    correspondence_gate_m: float = 1.5,
) -> list[LidarPointToPlaneObservation]:
    """Build fixed-correspondence point-to-plane observations for a LiDAR pair.

    ``source_records`` provide the voxel-plane map expressed in the source (world)
    frame. Each target point ``p_target`` is matched to its nearest source plane
    after applying ``initial_t_source_target`` (``T_source_target``); the raw
    target-frame point is stored so the factor can re-apply the optimized
    extrinsic. ``t_world_ego`` is identity because the source frame is treated as
    the world frame for a single fixed-trajectory pair.
    """

    plane_map = build_voxel_plane_map(source_records, voxel_size_m)
    if not plane_map:
        return []
    observations: list[LidarPointToPlaneObservation] = []
    for point in target_points:
        world_point = initial_t_source_target.transform_point(point)
        plane = nearest_voxel_plane(
            world_point,
            plane_map,
            voxel_size_m=voxel_size_m,
            correspondence_gate_m=correspondence_gate_m,
        )
        if plane is None:
            continue
        normal = plane.normal
        if normal is None:
            continue
        observations.append(
            LidarPointToPlaneObservation(
                point_lidar_m=point,
                plane_point_world_m=plane.centroid,
                plane_normal_world=normal,
                t_world_ego=SE3.identity(),
            )
        )
    return observations


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
    stability = diagnostics.get("stability")
    stability_diagnostics = stability if isinstance(stability, Mapping) else None
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
    if stability_diagnostics is not None:
        stability_status = str(stability_diagnostics.get("status", "unknown"))
        stability_grade: Grade = "pass" if stability_status == "scored" else "warn"
        stability_reason = str(
            stability_diagnostics.get("reason", "world-map temporal window stability")
        )
        metrics["lidar_world_map_stability_window_count"] = MetricResult(
            value=_float_or_none(stability_diagnostics.get("window_count")),
            unit="windows",
            grade=stability_grade,
            reason=stability_reason,
        )
        metrics["lidar_world_map_stability_scored_window_count"] = MetricResult(
            value=_float_or_none(stability_diagnostics.get("scored_window_count")),
            unit="windows",
            grade=stability_grade,
            reason=stability_reason,
        )
        metrics["lidar_world_map_stability_holdout_rmse_spread_m"] = MetricResult(
            value=_float_or_none(stability_diagnostics.get("holdout_rmse_spread_m")),
            unit="m",
            grade=stability_grade,
            reason=stability_reason,
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


def _ratio_delta(baseline: float | None, perturbed: float | None) -> float | None:
    if baseline is None or perturbed is None:
        return None
    return perturbed - baseline


def _recall_delta(baseline: float | None, perturbed: float | None) -> float | None:
    if baseline is None or perturbed is None:
        return None
    return baseline - perturbed


def _perturbation_transforms(config: CalibrationConfig) -> list[tuple[str, SE3]]:
    return [(dof, transform) for dof, _amount, _unit, transform in _perturbation_cases(config)]


def _perturbation_cases(config: CalibrationConfig) -> list[tuple[str, float, str, SE3]]:
    transforms: list[tuple[str, float, str, SE3]] = []
    for angle_deg in config.evaluation.kitti.perturbation_rotation_deg:
        angle_rad = math.radians(angle_deg)
        transforms.extend(
            [
                (
                    "roll_deg",
                    angle_deg,
                    "deg",
                    _rotation_perturbation((1.0, 0.0, 0.0), angle_rad),
                ),
                (
                    "roll_deg",
                    -angle_deg,
                    "deg",
                    _rotation_perturbation((1.0, 0.0, 0.0), -angle_rad),
                ),
                (
                    "pitch_deg",
                    angle_deg,
                    "deg",
                    _rotation_perturbation((0.0, 1.0, 0.0), angle_rad),
                ),
                (
                    "pitch_deg",
                    -angle_deg,
                    "deg",
                    _rotation_perturbation((0.0, 1.0, 0.0), -angle_rad),
                ),
                (
                    "yaw_deg",
                    angle_deg,
                    "deg",
                    _rotation_perturbation((0.0, 0.0, 1.0), angle_rad),
                ),
                (
                    "yaw_deg",
                    -angle_deg,
                    "deg",
                    _rotation_perturbation((0.0, 0.0, 1.0), -angle_rad),
                ),
            ]
        )
    for offset_m in config.evaluation.kitti.perturbation_translation_m:
        transforms.extend(
            [
                (
                    "x_m",
                    offset_m,
                    "m",
                    SE3((offset_m, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
                ),
                (
                    "x_m",
                    -offset_m,
                    "m",
                    SE3((-offset_m, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
                ),
                (
                    "y_m",
                    offset_m,
                    "m",
                    SE3((0.0, offset_m, 0.0), (0.0, 0.0, 0.0, 1.0)),
                ),
                (
                    "y_m",
                    -offset_m,
                    "m",
                    SE3((0.0, -offset_m, 0.0), (0.0, 0.0, 0.0, 1.0)),
                ),
                (
                    "z_m",
                    offset_m,
                    "m",
                    SE3((0.0, 0.0, offset_m), (0.0, 0.0, 0.0, 1.0)),
                ),
                (
                    "z_m",
                    -offset_m,
                    "m",
                    SE3((0.0, 0.0, -offset_m), (0.0, 0.0, 0.0, 1.0)),
                ),
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
