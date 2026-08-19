"""Camera-LiDAR cross-modal candidate evaluation metrics."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import cast

from calibrex.core.config import CalibrationConfig
from calibrex.core.geometry import SE3, QuaternionXYZW
from calibrex.core.report_artifacts import EvidenceCaseItem
from calibrex.core.result import CalibrationResult, Grade, MetricResult, TransformResult
from calibrex.data.base import StreamSummary
from calibrex.data.inspect import DatasetInspection
from calibrex.data.kitti import (
    KITTILidarCameraProjection,
    find_camera_lidar_pairs,
    project_velodyne_to_camera,
    read_velodyne_to_camera_transform,
    score_lidar_camera_depth_edge_alignment,
    score_lidar_camera_edge_alignment,
)

LIDAR_CAMERA_FACTOR_NAMES = {
    "lidar_camera_mutual_information",
    "koide_lidar_camera",
    "direct_visual_lidar_calibration",
    "lidar_camera_targetless_baseline",
}
LIDAR_CAMERA_METRIC_PREFIXES = (
    "lidar_camera_",
    "koide_lidar_camera_",
)
_LIDAR_CAMERA_MANDATORY_ROTATION_DEG = 1.0
_LIDAR_CAMERA_MANDATORY_TRANSLATION_M = 0.10
_LIDAR_CAMERA_MANDATORY_CASE_COUNT = 12.0
_LIDAR_CAMERA_MANDATORY_DETECTABLE_TARGET = 8
_DELTA_EPSILON = 1.0e-9


@dataclass(frozen=True)
class _ProjectionMetricSample:
    projected_points: float
    projection_ratio: float
    depth_median_m: float | None
    depth_span_m: float | None
    horizontal_coverage: float
    vertical_coverage: float
    edge_alignment_score: float | None
    edge_gradient_mean: float | None
    depth_discontinuity_points: float | None
    depth_edge_alignment_score: float | None
    depth_edge_gradient_mean: float | None


@dataclass(frozen=True)
class _PerturbationCase:
    baseline_index: int
    dof: str
    amount: float
    sample: _ProjectionMetricSample


@dataclass(frozen=True)
class LidarCameraEvidencePayload:
    """Projection evidence metrics, cases, and protocol metadata."""

    metrics: dict[str, MetricResult]
    cases: list[EvidenceCaseItem]
    protocol: dict[str, object]
    evidence_gates: dict[str, float | int]


def lidar_camera_metrics_from_result(
    config: CalibrationConfig,
    result: CalibrationResult,
    inspection: DatasetInspection,
) -> dict[str, MetricResult]:
    """Build camera-LiDAR overlay and projection evidence metrics."""

    if not _uses_lidar_camera_evaluation(config):
        return {}

    camera_streams = _stream_names(inspection.streams, kind="camera")
    lidar_streams = _stream_names(inspection.streams, kind="lidar")
    transform_pairs = _transform_pairs(result)
    timestamp_pairs = _camera_lidar_pair_count(inspection)

    metrics = {
        "lidar_camera_transform_pairs": MetricResult(
            value=float(len(transform_pairs)),
            unit="pairs",
            reason=(
                f"{len(transform_pairs)} camera-LiDAR transform pair(s) are available"
            ),
        ),
        "lidar_camera_overlay_readiness": _overlay_readiness_metric(
            camera_streams,
            lidar_streams,
            transform_pairs,
            timestamp_pairs,
        ),
    }
    metrics["lidar_camera_overlay_score"] = _overlay_score_metric(
        result,
        inspection,
        metrics["lidar_camera_overlay_readiness"],
    )
    projection_payload = _projection_metrics(config, result, inspection)
    metrics.update(projection_payload.metrics)
    _store_lidar_camera_evidence_provenance(
        result,
        projection_payload,
        transform_pairs=transform_pairs,
    )
    if (
        _uses_mutual_information(config)
        and "lidar_camera_mutual_information_score" not in result.metrics
    ):
        metrics["lidar_camera_mutual_information_score"] = MetricResult(
            value=None,
            grade="warn",
            reason=(
                "mutual-information image/LiDAR intensity scoring is declared "
                "but not computed by the alpha backend"
            ),
        )
    return metrics


def _projection_metrics(
    config: CalibrationConfig,
    result: CalibrationResult,
    inspection: DatasetInspection,
) -> LidarCameraEvidencePayload:
    gates = _evidence_gate_options(config)
    if inspection.dataset_type != "kitti_raw":
        return LidarCameraEvidencePayload({}, [], {}, gates)
    max_pairs = config.evaluation.kitti.max_projection_pairs
    if max_pairs <= 0:
        return LidarCameraEvidencePayload(
            _unavailable_projection_metrics("KITTI projection evaluation is disabled"),
            [],
            {},
            gates,
        )
    frame_pairs = _configured_camera_lidar_frame_pairs(config, inspection, max_pairs=max_pairs)
    if not frame_pairs:
        return LidarCameraEvidencePayload(
            _unavailable_projection_metrics("no concrete camera-LiDAR frame pairs"),
            [],
            {},
            gates,
        )

    baseline_transform = _evaluation_camera_lidar_transform(config, result, inspection)
    samples: list[_ProjectionMetricSample] = []
    skipped_reasons: list[str] = []
    for pair in frame_pairs:
        sample = _projection_metric_sample(
            inspection,
            pair,
            max_points=config.evaluation.kitti.projection_sample_points,
            t_camera_lidar_override=baseline_transform,
        )
        if isinstance(sample, _ProjectionMetricSample):
            samples.append(sample)
        else:
            skipped_reasons.append(sample)

    if not samples:
        reason = (
            skipped_reasons[0]
            if skipped_reasons
            else "camera-LiDAR projection could not be computed"
        )
        return LidarCameraEvidencePayload(
            _unavailable_projection_metrics(reason),
            [],
            {},
            gates,
        )

    reason_suffix = (
        f"mean over {len(samples)} KITTI camera-LiDAR frame pair(s)"
        if len(samples) > 1
        else "selected KITTI camera-LiDAR frame pair"
    )
    metrics = {
        "lidar_camera_projection_frame_count": MetricResult(
            value=float(len(samples)),
            unit="frames",
            reason="KITTI camera-LiDAR frame pairs used for projection metrics",
        ),
        "lidar_camera_projected_points": _aggregate_sample_metric(
            samples,
            lambda sample: sample.projected_points,
            unit="points",
            reason=(
                "sampled KITTI Velodyne points projected into camera images; "
                f"{reason_suffix}"
            ),
        ),
        "lidar_camera_projection_ratio": _aggregate_sample_metric(
            samples,
            lambda sample: sample.projection_ratio,
            reason=(
                "fraction of sampled KITTI Velodyne points inside camera images; "
                f"{reason_suffix}"
            ),
        ),
        "lidar_camera_projection_depth_median_m": _aggregate_sample_metric(
            samples,
            lambda sample: sample.depth_median_m,
            unit="m",
            grade="pass",
            reason=f"median depth of projected KITTI Velodyne points; {reason_suffix}",
        ),
        "lidar_camera_projection_depth_span_m": _aggregate_sample_metric(
            samples,
            lambda sample: sample.depth_span_m,
            unit="m",
            grade="pass",
            reason=f"depth span of projected KITTI Velodyne points; {reason_suffix}",
        ),
        "lidar_camera_projection_horizontal_coverage": _aggregate_sample_metric(
            samples,
            lambda sample: sample.horizontal_coverage,
            reason=(
                "normalized horizontal image span of projected LiDAR points; "
                f"{reason_suffix}"
            ),
        ),
        "lidar_camera_projection_vertical_coverage": _aggregate_sample_metric(
            samples,
            lambda sample: sample.vertical_coverage,
            reason=(
                "normalized vertical image span of projected LiDAR points; "
                f"{reason_suffix}"
            ),
        ),
        "lidar_camera_edge_alignment_score": _aggregate_sample_metric(
            samples,
            lambda sample: sample.edge_alignment_score,
            reason=(
                "fraction of projected LiDAR points near image intensity edges; "
                f"{reason_suffix}"
            ),
        ),
        "lidar_camera_edge_gradient_mean": _aggregate_sample_metric(
            samples,
            lambda sample: sample.edge_gradient_mean,
            grade="pass",
            reason=(
                "mean image gradient sampled at projected LiDAR points; "
                f"{reason_suffix}"
            ),
        ),
        "lidar_camera_depth_discontinuity_points": _aggregate_sample_metric(
            samples,
            lambda sample: sample.depth_discontinuity_points,
            unit="points",
            reason=(
                "projected LiDAR points with nearby depth jumps; "
                f"{reason_suffix}"
            ),
        ),
        "lidar_camera_depth_edge_alignment_score": _aggregate_sample_metric(
            samples,
            lambda sample: sample.depth_edge_alignment_score,
            reason=(
                "fraction of projected LiDAR depth-discontinuity points near "
                f"image intensity edges; {reason_suffix}"
            ),
        ),
        "lidar_camera_depth_edge_gradient_mean": _aggregate_sample_metric(
            samples,
            lambda sample: sample.depth_edge_gradient_mean,
            grade="pass",
            reason=(
                "mean image gradient at projected LiDAR depth-discontinuity "
                f"points; {reason_suffix}"
            ),
        ),
    }
    perturbation_metrics, cases = _perturbation_metrics(
        config,
        inspection,
        frame_pairs,
        samples,
        baseline_transform=baseline_transform,
    )
    metrics.update(perturbation_metrics)
    case_count_metric = perturbation_metrics.get("lidar_camera_perturbation_case_count")
    case_count = (
        int(case_count_metric.value)
        if case_count_metric and case_count_metric.value is not None
        else 0
    )
    protocol = _lidar_camera_protocol_payload(
        config,
        frame_count=len(samples),
        case_count=case_count,
        gates=gates,
        use_frame_graph_candidate=config.evaluation.kitti.use_frame_graph_candidate,
    )
    return LidarCameraEvidencePayload(metrics, cases, protocol, gates)


def _unavailable_projection_metrics(reason: str) -> dict[str, MetricResult]:
    return {
        "lidar_camera_projection_frame_count": MetricResult(
            value=0.0,
            unit="frames",
            reason=reason,
        ),
        "lidar_camera_projected_points": MetricResult(
            value=None,
            unit="points",
            grade="warn",
            reason=reason,
        ),
        "lidar_camera_projection_ratio": MetricResult(
            value=None,
            grade="warn",
            reason=reason,
        ),
        "lidar_camera_projection_depth_median_m": MetricResult(
            value=None,
            unit="m",
            grade="warn",
            reason=reason,
        ),
        "lidar_camera_projection_depth_span_m": MetricResult(
            value=None,
            unit="m",
            grade="warn",
            reason=reason,
        ),
        "lidar_camera_projection_horizontal_coverage": MetricResult(
            value=None,
            grade="warn",
            reason=reason,
        ),
        "lidar_camera_projection_vertical_coverage": MetricResult(
            value=None,
            grade="warn",
            reason=reason,
        ),
        "lidar_camera_edge_alignment_score": MetricResult(
            value=None,
            grade="warn",
            reason=reason,
        ),
        "lidar_camera_edge_gradient_mean": MetricResult(
            value=None,
            grade="warn",
            reason=reason,
        ),
        "lidar_camera_depth_discontinuity_points": MetricResult(
            value=None,
            unit="points",
            grade="warn",
            reason=reason,
        ),
        "lidar_camera_depth_edge_alignment_score": MetricResult(
            value=None,
            grade="warn",
            reason=reason,
        ),
        "lidar_camera_depth_edge_gradient_mean": MetricResult(
            value=None,
            grade="warn",
            reason=reason,
        ),
        "lidar_camera_perturbation_case_count": MetricResult(
            value=0.0,
            unit="cases",
            reason=reason,
        ),
        "lidar_camera_perturbation_detectable_fraction": MetricResult(
            value=None,
            grade="warn",
            reason=reason,
        ),
        "lidar_camera_perturbation_edge_delta_mean": MetricResult(
            value=None,
            grade="warn",
            reason=reason,
        ),
        "lidar_camera_perturbation_depth_edge_delta_mean": MetricResult(
            value=None,
            grade="warn",
            reason=reason,
        ),
        "lidar_camera_perturbation_projection_ratio_delta_mean": MetricResult(
            value=None,
            grade="warn",
            reason=reason,
        ),
        "lidar_camera_perturbation_mandatory_case_count": MetricResult(
            value=0.0,
            unit="cases",
            reason=reason,
        ),
        "lidar_camera_perturbation_mandatory_detectable_count": MetricResult(
            value=None,
            grade="warn",
            reason=reason,
        ),
    }


def _projection_metric_sample(
    inspection: DatasetInspection,
    pair: Mapping[object, object],
    *,
    max_points: int,
    t_camera_lidar_override: SE3 | None = None,
) -> _ProjectionMetricSample | str:
    camera_path = pair.get("camera_path")
    lidar_path = pair.get("lidar_path")
    if not isinstance(camera_path, str) or not isinstance(lidar_path, str):
        return "camera-LiDAR frame pair paths are missing"

    projection = project_velodyne_to_camera(
        inspection.path,
        camera_path=camera_path,
        lidar_path=lidar_path,
        max_points=max_points,
        t_camera_lidar_override=t_camera_lidar_override,
    )
    if projection.status != "projected":
        return projection.reason or "camera-LiDAR projection could not be computed"

    ratio = (
        projection.projected_point_count / projection.sampled_point_count
        if projection.sampled_point_count
        else 0.0
    )
    edge_alignment = score_lidar_camera_edge_alignment(projection)
    depth_edge_alignment = score_lidar_camera_depth_edge_alignment(projection)
    horizontal, vertical = _projection_coverage_values(projection)
    return _ProjectionMetricSample(
        projected_points=float(projection.projected_point_count),
        projection_ratio=ratio,
        depth_median_m=_median(point.depth_m for point in projection.points),
        depth_span_m=_span(point.depth_m for point in projection.points),
        horizontal_coverage=horizontal,
        vertical_coverage=vertical,
        edge_alignment_score=(
            edge_alignment.edge_fraction if edge_alignment.status == "scored" else None
        ),
        edge_gradient_mean=(
            edge_alignment.mean_gradient if edge_alignment.status == "scored" else None
        ),
        depth_discontinuity_points=(
            float(depth_edge_alignment.discontinuity_point_count)
            if depth_edge_alignment.status == "scored"
            else None
        ),
        depth_edge_alignment_score=(
            depth_edge_alignment.edge_fraction
            if depth_edge_alignment.status == "scored"
            else None
        ),
        depth_edge_gradient_mean=(
            depth_edge_alignment.mean_gradient
            if depth_edge_alignment.status == "scored"
            else None
        ),
    )


def _perturbation_metrics(
    config: CalibrationConfig,
    inspection: DatasetInspection,
    frame_pairs: list[Mapping[object, object]],
    baseline_samples: list[_ProjectionMetricSample],
    *,
    baseline_transform: SE3 | None,
) -> tuple[dict[str, MetricResult], list[EvidenceCaseItem]]:
    if baseline_transform is None:
        reason = "KITTI Velodyne-camera transform is missing"
        return _unavailable_perturbation_metrics(reason), []

    cases: list[_PerturbationCase] = []
    skipped_reasons: list[str] = []
    for baseline_index, pair in enumerate(frame_pairs[: len(baseline_samples)]):
        for dof, delta in _perturbation_transforms(config):
            sample = _projection_metric_sample(
                inspection,
                pair,
                max_points=config.evaluation.kitti.projection_sample_points,
                t_camera_lidar_override=delta.compose(baseline_transform),
            )
            if isinstance(sample, _ProjectionMetricSample):
                cases.append(
                    _PerturbationCase(
                        baseline_index=baseline_index,
                        dof=dof,
                        amount=_perturbation_amount(delta),
                        sample=sample,
                    )
                )
            else:
                skipped_reasons.append(sample)

    if not cases:
        reason = skipped_reasons[0] if skipped_reasons else "no perturbation cases were scored"
        return _unavailable_perturbation_metrics(reason), []

    case_deltas = _perturbation_case_deltas(baseline_samples, cases)
    edge_deltas = [edge for edge, _, _ in case_deltas if edge is not None]
    depth_edge_deltas = [depth for _, depth, _ in case_deltas if depth is not None]
    projection_ratio_deltas = [ratio for _, _, ratio in case_deltas if ratio is not None]
    detectable = _detectable_fraction(case_deltas)
    mandatory_cases = [
        (case, deltas)
        for case, deltas in zip(cases, case_deltas, strict=True)
        if _is_mandatory_perturbation_case(case)
    ]
    mandatory_detectable_count = sum(
        1
        for _, deltas in mandatory_cases
        if any(delta is not None and delta > _DELTA_EPSILON for delta in deltas)
    )
    mandatory_grade: Grade = (
        "pass"
        if mandatory_detectable_count
        >= config.evaluation.kitti.evidence_gate_min_mandatory_detectable_count
        else "warn"
    )
    grade: Grade = "pass" if detectable and detectable > 0.0 else "warn"
    reason = (
        "known camera-frame perturbations of T_camera_lidar; "
        "positive deltas mean the reference projection score is better"
    )
    evidence_cases = _perturbation_evidence_cases(cases, case_deltas)
    return {
        "lidar_camera_perturbation_case_count": MetricResult(
            value=float(len(cases)),
            unit="cases",
            grade="pass",
            reason=(
                "roll/pitch/yaw/x/y/z perturbation cases evaluated across "
                f"{len(baseline_samples)} KITTI frame pair(s)"
            ),
        ),
        "lidar_camera_perturbation_detectable_fraction": MetricResult(
            value=detectable,
            grade=grade,
            reason=(
                "fraction of perturbation cases that worsened at least one "
                "projection, edge, or depth-edge score"
            ),
        ),
        "lidar_camera_perturbation_edge_delta_mean": MetricResult(
            value=_mean_or_none(edge_deltas),
            grade=grade,
            reason=reason,
        ),
        "lidar_camera_perturbation_depth_edge_delta_mean": MetricResult(
            value=_mean_or_none(depth_edge_deltas),
            grade=grade,
            reason=reason,
        ),
        "lidar_camera_perturbation_projection_ratio_delta_mean": MetricResult(
            value=_mean_or_none(projection_ratio_deltas),
            grade=grade,
            reason=reason,
        ),
        "lidar_camera_perturbation_mandatory_case_count": MetricResult(
            value=float(len(mandatory_cases)),
            unit="cases",
            grade=(
                "pass"
                if len(mandatory_cases) >= int(_LIDAR_CAMERA_MANDATORY_CASE_COUNT)
                else "warn"
            ),
            reason=(
                "predeclared large ±1 deg and ±0.10 m perturbation cases "
                "materialized for roll/pitch/yaw/x/y/z"
            ),
        ),
        "lidar_camera_perturbation_mandatory_detectable_count": MetricResult(
            value=float(mandatory_detectable_count),
            unit="cases",
            grade=mandatory_grade,
            reason=(
                "mandatory perturbation cases detected under projection scoring; "
                "target >= "
                f"{config.evaluation.kitti.evidence_gate_min_mandatory_detectable_count} cases"
            ),
        ),
    }, evidence_cases


def _unavailable_perturbation_metrics(reason: str) -> dict[str, MetricResult]:
    return {
        "lidar_camera_perturbation_case_count": MetricResult(
            value=0.0,
            unit="cases",
            reason=reason,
        ),
        "lidar_camera_perturbation_detectable_fraction": MetricResult(
            value=None,
            grade="warn",
            reason=reason,
        ),
        "lidar_camera_perturbation_edge_delta_mean": MetricResult(
            value=None,
            grade="warn",
            reason=reason,
        ),
        "lidar_camera_perturbation_depth_edge_delta_mean": MetricResult(
            value=None,
            grade="warn",
            reason=reason,
        ),
        "lidar_camera_perturbation_projection_ratio_delta_mean": MetricResult(
            value=None,
            grade="warn",
            reason=reason,
        ),
        "lidar_camera_perturbation_mandatory_case_count": MetricResult(
            value=0.0,
            unit="cases",
            reason=reason,
        ),
        "lidar_camera_perturbation_mandatory_detectable_count": MetricResult(
            value=None,
            grade="warn",
            reason=reason,
        ),
    }


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


def _perturbation_amount(transform: SE3) -> float:
    if transform.translation_m != (0.0, 0.0, 0.0):
        return math.sqrt(sum(value * value for value in transform.translation_m))
    x, y, z, w = transform.rotation_quat_xyzw
    vector_norm = math.sqrt(x * x + y * y + z * z)
    return math.degrees(2.0 * math.atan2(vector_norm, abs(w)))


def _perturbation_case_deltas(
    baseline_samples: list[_ProjectionMetricSample],
    cases: list[_PerturbationCase],
) -> list[tuple[float | None, float | None, float | None]]:
    if not baseline_samples:
        return []
    deltas: list[tuple[float | None, float | None, float | None]] = []
    for case in cases:
        baseline = baseline_samples[case.baseline_index]
        deltas.append(
            (
                _metric_delta(baseline.edge_alignment_score, case.sample.edge_alignment_score),
                _metric_delta(
                    baseline.depth_edge_alignment_score,
                    case.sample.depth_edge_alignment_score,
                ),
                _metric_delta(baseline.projection_ratio, case.sample.projection_ratio),
            )
        )
    return deltas


def _metric_delta(baseline: float | None, perturbed: float | None) -> float | None:
    if baseline is None or perturbed is None:
        return None
    return baseline - perturbed


def _detectable_fraction(
    case_deltas: list[tuple[float | None, float | None, float | None]],
) -> float | None:
    if not case_deltas:
        return None
    detectable = sum(
        1
        for deltas in case_deltas
        if any(delta is not None and delta > 1.0e-9 for delta in deltas)
    )
    measurable = sum(1 for deltas in case_deltas if any(delta is not None for delta in deltas))
    if measurable == 0:
        return None
    return detectable / measurable


def _mean_or_none(values: Iterable[float]) -> float | None:
    values_list = list(values)
    if not values_list:
        return None
    return _mean(values_list)


def _aggregate_sample_metric(
    samples: list[_ProjectionMetricSample],
    selector: Callable[[_ProjectionMetricSample], float | None],
    unit: str | None = None,
    grade: Grade = "warn",
    reason: str | None = None,
) -> MetricResult:
    values = [value for sample in samples if (value := selector(sample)) is not None]
    if not values:
        return MetricResult(value=None, unit=unit, grade=grade, reason=reason)
    if len(values) == 1:
        return MetricResult(value=values[0], unit=unit, grade=grade, reason=reason)
    return MetricResult(
        train=_mean(values[:-1]),
        holdout=values[-1],
        unit=unit,
        grade=grade,
        reason=reason,
    )


def _depth_edge_alignment_metrics(
    projection: KITTILidarCameraProjection,
) -> dict[str, MetricResult]:
    depth_edge_alignment = score_lidar_camera_depth_edge_alignment(projection)
    if depth_edge_alignment.status != "scored":
        reason = depth_edge_alignment.reason or "depth edge alignment could not be scored"
        return {
            "lidar_camera_depth_discontinuity_points": MetricResult(
                value=None,
                unit="points",
                grade="warn",
                reason=reason,
            ),
            "lidar_camera_depth_edge_alignment_score": MetricResult(
                value=None,
                grade="warn",
                reason=reason,
            ),
            "lidar_camera_depth_edge_gradient_mean": MetricResult(
                value=None,
                grade="warn",
                reason=reason,
            ),
        }
    return {
        "lidar_camera_depth_discontinuity_points": MetricResult(
            value=float(depth_edge_alignment.discontinuity_point_count),
            unit="points",
            reason=(
                "projected LiDAR points with nearby depth jumps "
                f"(>= {depth_edge_alignment.depth_jump_m:g} m)"
            ),
        ),
        "lidar_camera_depth_edge_alignment_score": MetricResult(
            value=depth_edge_alignment.edge_fraction,
            reason=(
                "fraction of projected LiDAR depth-discontinuity points near "
                f"image intensity edges (gradient >= {depth_edge_alignment.gradient_threshold:g})"
            ),
        ),
        "lidar_camera_depth_edge_gradient_mean": MetricResult(
            value=depth_edge_alignment.mean_gradient,
            grade="pass",
            reason="mean image gradient at projected LiDAR depth-discontinuity points",
        ),
    }


def _edge_alignment_metrics(
    projection: KITTILidarCameraProjection,
) -> dict[str, MetricResult]:
    edge_alignment = score_lidar_camera_edge_alignment(projection)
    if edge_alignment.status != "scored":
        reason = edge_alignment.reason or "edge alignment could not be scored"
        return {
            "lidar_camera_edge_alignment_score": MetricResult(
                value=None,
                grade="warn",
                reason=reason,
            ),
            "lidar_camera_edge_gradient_mean": MetricResult(
                value=None,
                grade="warn",
                reason=reason,
            ),
        }
    return {
        "lidar_camera_edge_alignment_score": MetricResult(
            value=edge_alignment.edge_fraction,
            reason=(
                "fraction of projected LiDAR points near image intensity edges "
                f"(gradient >= {edge_alignment.gradient_threshold:g})"
            ),
        ),
        "lidar_camera_edge_gradient_mean": MetricResult(
            value=edge_alignment.mean_gradient,
            grade="pass",
            reason="mean image gradient sampled at projected LiDAR points",
        ),
    }


def _projection_coverage_metrics(
    projection: KITTILidarCameraProjection,
) -> dict[str, MetricResult]:
    image_size = projection.image_size_px
    points = projection.points
    if image_size is None:
        reason = "image size is unavailable for projection coverage"
        return {
            "lidar_camera_projection_horizontal_coverage": MetricResult(
                value=None,
                grade="warn",
                reason=reason,
            ),
            "lidar_camera_projection_vertical_coverage": MetricResult(
                value=None,
                grade="warn",
                reason=reason,
            ),
        }
    width_px, height_px = image_size
    horizontal = _normalized_span((point.u_px for point in points), width_px)
    vertical = _normalized_span((point.v_px for point in points), height_px)
    return {
        "lidar_camera_projection_horizontal_coverage": MetricResult(
            value=horizontal,
            reason="normalized horizontal image span of projected LiDAR points",
        ),
        "lidar_camera_projection_vertical_coverage": MetricResult(
            value=vertical,
            reason="normalized vertical image span of projected LiDAR points",
        ),
    }


def _projection_coverage_values(projection: KITTILidarCameraProjection) -> tuple[float, float]:
    image_size = projection.image_size_px
    if image_size is None:
        return 0.0, 0.0
    width_px, height_px = image_size
    return (
        _normalized_span((point.u_px for point in projection.points), width_px),
        _normalized_span((point.v_px for point in projection.points), height_px),
    )


def _overlay_readiness_metric(
    camera_streams: tuple[str, ...],
    lidar_streams: tuple[str, ...],
    transform_pairs: tuple[tuple[str, str], ...],
    timestamp_pairs: float | None,
) -> MetricResult:
    missing: list[str] = []
    if not camera_streams:
        missing.append("camera streams")
    if not lidar_streams:
        missing.append("LiDAR streams")
    if not transform_pairs:
        missing.append("camera-LiDAR transform pairs")
    if timestamp_pairs == 0:
        missing.append("camera-LiDAR timestamp pairs")
    if missing:
        return MetricResult(
            value=0.0,
            grade="fail",
            reason="missing " + ", ".join(missing),
        )
    return MetricResult(
        value=1.0,
        grade="pass",
        reason=(
            "camera streams, LiDAR streams, transforms, and temporal pairing "
            "are ready for overlay evaluation"
        ),
    )


def _overlay_score_metric(
    result: CalibrationResult,
    inspection: DatasetInspection,
    readiness: MetricResult,
) -> MetricResult:
    if readiness.value != 1.0:
        return MetricResult(
            value=0.0,
            grade="fail",
            reason="overlay score cannot be computed until overlay readiness passes",
        )

    point_count = _metric_value(result.metrics.get("lidar_point_coverage"))
    max_dt_ms = _camera_lidar_max_dt_ms(inspection)
    pair_ratio = _camera_lidar_pair_ratio(inspection)
    if point_count is None or max_dt_ms is None:
        return MetricResult(
            value=None,
            grade="warn",
            reason=(
                "overlay score proxy needs LiDAR point coverage and camera-LiDAR "
                "timestamp alignment diagnostics"
            ),
        )

    point_score = _log_coverage_score(point_count, target=200_000.0)
    time_score = max(0.0, 1.0 - min(max_dt_ms, 100.0) / 100.0)
    score = (0.4 * point_score) + (0.4 * time_score) + (0.2 * pair_ratio)
    return MetricResult(
        value=score,
        reason=(
            "alpha overlay score proxy from LiDAR point coverage, camera-LiDAR "
            "timestamp alignment, and paired-frame ratio"
        ),
    )


def _stream_names(streams: list[StreamSummary], *, kind: str) -> tuple[str, ...]:
    names: list[str] = []
    for stream in streams:
        stream_name = stream.name.lower()
        stream_kind = stream.kind.lower()
        if (
            kind == "camera"
            and (stream_kind == "image" or "camera" in stream_name)
            and stream.message_count != 0
        ):
            names.append(stream.name)
        if kind == "lidar" and (
            stream_kind == "pointcloud" or "lidar" in stream_name or "velodyne" in stream_name
        ) and stream.message_count != 0:
            names.append(stream.name)
    return tuple(names)


def _transform_pairs(result: CalibrationResult) -> tuple[tuple[str, str], ...]:
    direct_pairs: set[tuple[str, str]] = set()
    by_parent: dict[str, dict[str, set[str]]] = {}
    for transform in result.transforms.values():
        parent = transform.parent
        child = transform.child
        if _is_camera(parent) and _is_lidar(child):
            direct_pairs.add((parent, child))
        if _is_lidar(parent) and _is_camera(child):
            direct_pairs.add((child, parent))
        bucket = by_parent.setdefault(parent, {"camera": set(), "lidar": set()})
        if _is_camera(child):
            bucket["camera"].add(child)
        if _is_lidar(child):
            bucket["lidar"].add(child)

    for bucket in by_parent.values():
        for camera in bucket["camera"]:
            for lidar in bucket["lidar"]:
                direct_pairs.add((camera, lidar))
    return tuple(sorted(direct_pairs))


def _uses_mutual_information(config: CalibrationConfig) -> bool:
    if "lidar_camera_mutual_information_score" in config.evaluation.metrics:
        return True
    factor = config.pipeline.factors.get("lidar_camera_mutual_information")
    return factor is not None and factor.enabled


def _uses_lidar_camera_evaluation(config: CalibrationConfig) -> bool:
    if any(
        metric_name.startswith(LIDAR_CAMERA_METRIC_PREFIXES)
        for metric_name in config.evaluation.metrics
    ):
        return True
    return any(
        factor.enabled
        for factor_name, factor in config.pipeline.factors.items()
        if factor_name in LIDAR_CAMERA_FACTOR_NAMES
    )


def _is_camera(name: str) -> bool:
    return name.lower().startswith("camera")


def _is_lidar(name: str) -> bool:
    lowered = name.lower()
    return lowered.startswith("lidar") or lowered.startswith("velodyne")


def _camera_lidar_pair_count(inspection: DatasetInspection) -> float | None:
    alignment = _timestamp_alignment(inspection)
    if alignment is None:
        return None
    return _float_or_none(alignment.get("camera_lidar_pair_count"))


def _camera_lidar_max_dt_ms(inspection: DatasetInspection) -> float | None:
    alignment = _timestamp_alignment(inspection)
    if alignment is None:
        return None
    return _float_or_none(alignment.get("camera_lidar_max_abs_dt_ms"))


def _camera_lidar_pair_ratio(inspection: DatasetInspection) -> float:
    alignment = _timestamp_alignment(inspection)
    if alignment is None:
        return 1.0
    pairs = _float_or_none(alignment.get("camera_lidar_pair_count"))
    camera_count = _float_or_none(alignment.get("camera_timestamp_count"))
    lidar_count = _float_or_none(alignment.get("lidar_timestamp_count"))
    if pairs is None:
        return 1.0
    denominator = max(value for value in [camera_count or 0.0, lidar_count or 0.0, 1.0])
    return max(0.0, min(1.0, pairs / denominator))


def _camera_lidar_frame_pairs(inspection: DatasetInspection) -> list[Mapping[object, object]]:
    pairs = inspection.diagnostics.get("camera_lidar_pairs")
    if not isinstance(pairs, list):
        return []
    return [pair for pair in pairs if isinstance(pair, Mapping)]


def _configured_camera_lidar_frame_pairs(
    config: CalibrationConfig,
    inspection: DatasetInspection,
    *,
    max_pairs: int,
) -> list[Mapping[object, object]]:
    inspected_pairs = _camera_lidar_frame_pairs(inspection)
    if len(inspected_pairs) >= max_pairs:
        return inspected_pairs[:max_pairs]
    loaded_pairs = find_camera_lidar_pairs(config.dataset.path, max_pairs=max_pairs)
    if loaded_pairs:
        loaded_mappings: list[Mapping[object, object]] = []
        for pair in loaded_pairs:
            loaded_mappings.append(cast(Mapping[object, object], pair.as_dict()))
        return loaded_mappings
    return inspected_pairs[:max_pairs]


def _timestamp_alignment(inspection: DatasetInspection) -> Mapping[object, object] | None:
    alignment = inspection.diagnostics.get("timestamp_alignment")
    return alignment if isinstance(alignment, Mapping) else None


def _metric_value(metric: MetricResult | None) -> float | None:
    if metric is None:
        return None
    if metric.holdout is not None:
        return metric.holdout
    if metric.value is not None:
        return metric.value
    return metric.train


def _log_coverage_score(value: float, *, target: float) -> float:
    if value <= 0.0:
        return 0.0
    return max(0.0, min(1.0, math.log10(value + 1.0) / math.log10(target + 1.0)))


def _mean(values: Iterable[float]) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return sum(values_list) / len(values_list)


def _median(values: Iterable[float]) -> float | None:
    sorted_values = sorted(values)
    count = len(sorted_values)
    if count == 0:
        return None
    midpoint = count // 2
    if count % 2:
        return sorted_values[midpoint]
    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2.0


def _span(values: Iterable[float]) -> float | None:
    sorted_values = sorted(values)
    if not sorted_values:
        return None
    return sorted_values[-1] - sorted_values[0]


def _normalized_span(values: Iterable[float], denominator: int) -> float:
    span = _span(values)
    if span is None or denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, span / float(denominator)))


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _evidence_gate_options(config: CalibrationConfig) -> dict[str, float | int]:
    kitti = config.evaluation.kitti
    return {
        "evidence_gate_min_edge_alignment_holdout": (
            kitti.evidence_gate_min_edge_alignment_holdout
        ),
        "evidence_gate_min_depth_edge_alignment_holdout": (
            kitti.evidence_gate_min_depth_edge_alignment_holdout
        ),
        "evidence_gate_min_perturbation_detectable_fraction": (
            kitti.evidence_gate_min_perturbation_detectable_fraction
        ),
        "evidence_gate_min_mandatory_detectable_count": (
            kitti.evidence_gate_min_mandatory_detectable_count
        ),
    }


def _evaluation_camera_lidar_transform(
    config: CalibrationConfig,
    result: CalibrationResult,
    inspection: DatasetInspection,
) -> SE3 | None:
    if config.evaluation.kitti.use_frame_graph_candidate:
        derived = camera_lidar_from_rig_candidates(result.candidate_extrinsics)
        if derived is not None:
            return derived
    return read_velodyne_to_camera_transform(inspection.path)


def camera_lidar_from_rig_candidates(
    candidates: dict[str, TransformResult],
) -> SE3 | None:
    camera = candidates.get("T_base_link_camera0")
    lidar = candidates.get("T_base_link_lidar0")
    if camera is None or lidar is None:
        return None
    return camera.as_se3().inverse().compose(lidar.as_se3())


def _store_lidar_camera_evidence_provenance(
    result: CalibrationResult,
    payload: LidarCameraEvidencePayload,
    *,
    transform_pairs: tuple[tuple[str, str], ...],
) -> None:
    if "lidar_camera_projection_frame_count" not in payload.metrics:
        return
    frame_count = payload.metrics["lidar_camera_projection_frame_count"].value
    if frame_count is None or frame_count <= 0.0:
        return
    candidate_transform = "T_base_link_lidar0"
    if transform_pairs:
        camera_name, lidar_name = transform_pairs[0]
        candidate_transform = f"T_{camera_name}_{lidar_name}"
    result.run.provenance["lidar_camera_evidence"] = {
        **payload.protocol,
        "candidate_transform": candidate_transform,
        "evidence_gates": payload.evidence_gates,
    }
    if payload.cases:
        existing_cases = result.run.provenance.get("evidence_cases")
        cases = existing_cases if isinstance(existing_cases, list) else []
        result.run.provenance["evidence_cases"] = [
            *cases,
            *(case.model_dump(mode="json") for case in payload.cases),
        ]


def _lidar_camera_protocol_payload(
    config: CalibrationConfig,
    *,
    frame_count: int,
    case_count: int,
    gates: dict[str, float | int],
    use_frame_graph_candidate: bool,
) -> dict[str, object]:
    split_policy = "temporal_tail_holdout" if frame_count > 1 else "single_frame"
    return {
        "protocol_id": "kitti_lidar_camera_projection_edge_holdout/v0.1",
        "split_policy": split_policy,
        "independent_holdout": frame_count > 1,
        "transform_convention": "T_camera_lidar maps Velodyne points into camera frame",
        "known_bad_perturbation": "right-composed camera-frame SE(3) controls",
        "known_bad_case_count": case_count,
        "known_bad_challenge": {
            "challenge_id": "kitti_lidar_camera_mandatory_6dof_large_controls/v0.1",
            "mandatory_rotation_deg": _LIDAR_CAMERA_MANDATORY_ROTATION_DEG,
            "mandatory_translation_m": _LIDAR_CAMERA_MANDATORY_TRANSLATION_M,
            "mandatory_case_count": int(_LIDAR_CAMERA_MANDATORY_CASE_COUNT),
            "target_mandatory_detectable_count": gates[
                "evidence_gate_min_mandatory_detectable_count"
            ],
        },
        "parameters": {
            "projection_frame_count": frame_count,
            "use_frame_graph_candidate": use_frame_graph_candidate,
            **gates,
        },
        "limitations": [
            "projection evidence observes image-plane alignment only; depth along "
            "the optical axis and baselines are coverage-limited",
            "depth-edge scores require projected LiDAR depth discontinuities",
        ],
    }


def _is_mandatory_perturbation_case(case: _PerturbationCase) -> bool:
    if case.dof.endswith("_deg"):
        return abs(case.amount) >= _LIDAR_CAMERA_MANDATORY_ROTATION_DEG
    if case.dof.endswith("_m"):
        return abs(case.amount) >= _LIDAR_CAMERA_MANDATORY_TRANSLATION_M
    return False


def _perturbation_evidence_cases(
    cases: list[_PerturbationCase],
    case_deltas: list[tuple[float | None, float | None, float | None]],
) -> list[EvidenceCaseItem]:
    evidence_cases: list[EvidenceCaseItem] = []
    for case, deltas in zip(cases, case_deltas, strict=True):
        edge_delta, depth_edge_delta, projection_ratio_delta = deltas
        detectable = any(
            delta is not None and delta > _DELTA_EPSILON for delta in deltas
        )
        unit = "deg" if case.dof.endswith("_deg") else "m"
        dof = case.dof.removesuffix("_deg").removesuffix("_m")
        evidence_cases.append(
            EvidenceCaseItem(
                family="lidar_camera",
                case_id=f"{case.dof}:{case.amount:+g}{unit}",
                check="Known-Bad Controls",
                status="pass" if detectable else "warn",
                dof=dof,
                amount=case.amount,
                unit=unit,
                convention="right-composed camera-frame SE(3) perturbation of T_camera_lidar",
                metric_values={
                    "lidar_camera_edge_alignment_score": case.sample.edge_alignment_score,
                    "lidar_camera_depth_edge_alignment_score": (
                        case.sample.depth_edge_alignment_score
                    ),
                    "lidar_camera_projection_ratio": case.sample.projection_ratio,
                },
                delta_values={
                    "edge_alignment_delta": edge_delta,
                    "depth_edge_alignment_delta": depth_edge_delta,
                    "projection_ratio_delta": projection_ratio_delta,
                },
            )
        )
    return evidence_cases
