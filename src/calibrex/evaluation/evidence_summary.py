"""Evidence summary builders shared by reports and comparisons."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

from calibrex.core.report_artifacts import EvidenceCaseItem, EvidenceSummaryItem
from calibrex.core.result import CalibrationResult, Grade, MetricResult
from calibrex.evaluation.thresholds import (
    ThresholdProfile,
    primary_metric_value,
    profile_from_domain,
    thresholds_for_profile,
)


def evidence_summaries_from_result(result: CalibrationResult) -> list[EvidenceSummaryItem]:
    """Return machine-readable evidence summary rows for a calibration result."""

    items = _lidar_pair_evidence_items(result)
    items.extend(_lidar_camera_evidence_items(result))
    items.extend(_lidar_imu_evidence_items(result))
    items.extend(_radar_lidar_evidence_items(result))
    items.extend(_temporal_evidence_items(result))
    return items


def evidence_decision_grade_from_result(result: CalibrationResult) -> Grade | None:
    """Return the worst decision-boundary grade across materialized evidence families."""

    decision_items = [
        item
        for item in evidence_summaries_from_result(result)
        if item.check == "Decision Boundary"
    ]
    if not decision_items:
        return None
    return _worst_grade(item.status for item in decision_items)


def evidence_cases_from_result(result: CalibrationResult) -> list[EvidenceCaseItem]:
    """Return machine-readable evidence case rows stored in result provenance."""

    raw_cases = result.run.provenance.get("evidence_cases")
    cases: list[EvidenceCaseItem] = []
    if isinstance(raw_cases, list):
        for raw_case in raw_cases:
            if not isinstance(raw_case, dict):
                continue
            cases.append(EvidenceCaseItem.model_validate(raw_case))
    cases.extend(_radar_lidar_evidence_cases(result))
    return cases


def _radar_lidar_evidence_items(result: CalibrationResult) -> list[EvidenceSummaryItem]:
    radar = result.run.provenance.get("radar_velocity_consistency")
    if not isinstance(radar, dict):
        return []
    assessment = radar.get("assessment")
    assessment_status = assessment.get("status") if isinstance(assessment, dict) else None
    grade_by_status: dict[object, Grade] = {
        "pass": "pass",
        "fail": "fail",
        "inconclusive": "warn",
    }
    decision_grade = grade_by_status.get(assessment_status, "warn")
    static_summary = radar.get("static_return_summary")
    holdout_static = (
        static_summary.get("holdout") if isinstance(static_summary, dict) else None
    )
    observability = radar.get("holdout_yaw_observability")
    sensor_statuses = [
        str(item.get("status"))
        for item in observability.values()
        if isinstance(item, dict)
    ] if isinstance(observability, dict) else []
    sensor_evaluations = radar.get("sensor_evaluations")
    sensor_evaluation_values = (
        [item for item in sensor_evaluations.values() if isinstance(item, dict)]
        if isinstance(sensor_evaluations, dict)
        else []
    )
    sensor_support_ok = (
        all(_radar_sensor_supports_holdout(item) for item in sensor_evaluation_values)
        if sensor_evaluation_values
        else (
            isinstance(holdout_static, dict)
            and holdout_static.get("status") == "supported"
            and bool(sensor_statuses)
            and all(status == "supported" for status in sensor_statuses)
        )
    )
    support_grade: Grade = (
        "pass"
        if sensor_support_ok
        else "warn"
    )
    challenge = radar.get("known_bad_rotation_probes")
    controls_grade: Grade = (
        "pass"
        if (
            all(_radar_sensor_controls_pass(item) for item in sensor_evaluation_values)
            if sensor_evaluation_values
            else (
                isinstance(challenge, dict)
                and challenge.get("supported_count") == challenge.get("required_count")
                and challenge.get("detectable_fraction") == 1.0
            )
        )
        else "warn"
    )
    residual_summary = radar.get("residual_summary")
    holdout_residual = (
        residual_summary.get("holdout")
        if isinstance(residual_summary, dict)
        else None
    )
    reason = assessment.get("reason") if isinstance(assessment, dict) else None
    return [
        EvidenceSummaryItem(
            family="radar_lidar",
            check="Candidate Support",
            status=support_grade,
            evidence=(
                f"holdout static returns {_mapping_value(holdout_static, 'static_return_count')}, "
                f"static fraction {_mapping_value(holdout_static, 'static_fraction')}; "
                f"sensor yaw observability {', '.join(sensor_statuses) or 'unavailable'}"
            ),
            interpretation=(
                "Holdout support can test radar yaw consistency."
                if support_grade == "pass"
                else "INCONCLUSIVE: holdout static support or yaw sensitivity is insufficient."
            ),
            metric_ids=["radar_lidar_velocity_consistency"],
        ),
        EvidenceSummaryItem(
            family="radar_lidar",
            check="Known-Bad Controls",
            status=controls_grade,
            evidence=(
                f"supported {_mapping_value(challenge, 'supported_count')} of "
                f"{_mapping_value(challenge, 'required_count')}; detectable fraction "
                f"{_mapping_value(challenge, 'detectable_fraction')}"
            ),
            interpretation=(
                "All mandatory yaw controls are detectable on shared holdout support."
                if controls_grade == "pass"
                else "INCONCLUSIVE: controls do not establish complete falsification power."
            ),
            metric_ids=["radar_lidar_velocity_consistency"],
        ),
        EvidenceSummaryItem(
            family="radar_lidar",
            check="Decision Boundary",
            status=decision_grade,
            evidence=(
                f"holdout median {_mapping_value(holdout_residual, 'median_abs_residual_mps')} "
                f"m/s; RMSE {_mapping_value(holdout_residual, 'rmse_mps')} m/s"
            ),
            interpretation=str(reason or "Radar yaw policy decision unavailable."),
            metric_ids=["radar_lidar_velocity_consistency"],
        ),
    ]


def _radar_lidar_evidence_cases(result: CalibrationResult) -> list[EvidenceCaseItem]:
    radar = result.run.provenance.get("radar_velocity_consistency")
    if not isinstance(radar, dict):
        return []
    challenge = radar.get("known_bad_rotation_probes")
    probes = challenge.get("probes") if isinstance(challenge, dict) else None
    if not isinstance(probes, list):
        return []
    cases: list[EvidenceCaseItem] = []
    for probe in probes:
        if not isinstance(probe, dict):
            continue
        supported = probe.get("supported") is True
        detectable = probe.get("detectable")
        status: Grade = "pass" if supported and detectable is True else "warn"
        baseline = _number_or_none(probe.get("comparison_baseline_median_abs_residual_mps"))
        perturbed = _number_or_none(probe.get("perturbed_median_abs_residual_mps"))
        amount = _number_or_none(probe.get("amount_deg"))
        cases.append(
            EvidenceCaseItem(
                family="radar_lidar",
                case_id=f"yaw_deg:{amount:+g}" if amount is not None else "yaw_deg:unknown",
                check="Known-Bad Controls",
                status=status,
                dof="yaw",
                amount=amount,
                unit="deg",
                convention="left-composed rotation about ego +Z",
                metric_values={
                    "baseline_median_abs_residual_mps": baseline,
                    "perturbed_median_abs_residual_mps": perturbed,
                },
                delta_values={
                    "median_abs_residual_delta_mps": (
                        perturbed - baseline
                        if perturbed is not None and baseline is not None
                        else None
                    )
                },
            )
        )
    return cases


def _lidar_pair_evidence_items(result: CalibrationResult) -> list[EvidenceSummaryItem]:
    support_metric = result.metrics.get("lidar_pair_source_voxel_recall_in_target")
    shared_metric = result.metrics.get("lidar_pair_shared_voxel_count")
    rmse_metric = result.metrics.get("lidar_pair_shared_voxel_centroid_rmse_m")
    p2p_median_metric = result.metrics.get("lidar_pair_holdout_point_to_plane_median_abs_m")
    p2p_p90_metric = result.metrics.get("lidar_pair_holdout_point_to_plane_p90_abs_m")
    p2p_support_ratio_metric = result.metrics.get(
        "lidar_pair_holdout_point_to_plane_support_ratio"
    )
    p2p_unmatched_metric = result.metrics.get(
        "lidar_pair_holdout_point_to_plane_unmatched_fraction"
    )
    known_bad_metric = result.metrics.get("lidar_pair_known_bad_detectable_fraction")
    max_delta_metric = result.metrics.get("lidar_pair_known_bad_centroid_rmse_delta_max_m")
    p2p_delta_metric = result.metrics.get(
        "lidar_pair_known_bad_point_to_plane_p90_delta_max_m"
    )
    mandatory_detection_metric = result.metrics.get(
        "lidar_pair_known_bad_mandatory_supported_detection_count"
    )
    mandatory_case_metric = result.metrics.get(
        "lidar_pair_known_bad_mandatory_case_count"
    )
    if (
        support_metric is None
        and shared_metric is None
        and rmse_metric is None
        and p2p_median_metric is None
        and p2p_p90_metric is None
        and known_bad_metric is None
        and max_delta_metric is None
        and p2p_delta_metric is None
        and mandatory_detection_metric is None
    ):
        return []

    support_grade: Grade = support_metric.grade if support_metric is not None else "warn"
    p2p_available = p2p_median_metric is not None or p2p_p90_metric is not None
    p2p_grade: Grade = p2p_p90_metric.grade if p2p_p90_metric is not None else "warn"
    known_bad_grade: Grade = known_bad_metric.grade if known_bad_metric is not None else "warn"
    decision_grade: Grade = (
        "pass"
        if support_grade == "pass"
        and (not p2p_available or p2p_grade == "pass")
        and known_bad_grade == "pass"
        else "warn"
    )
    known_bad_fraction = _fmt(known_bad_metric.value if known_bad_metric else None)
    known_bad_max_delta = _fmt(max_delta_metric.value if max_delta_metric else None)
    known_bad_p2p_delta = _fmt(p2p_delta_metric.value if p2p_delta_metric else None)
    mandatory_detection_count = _fmt(
        mandatory_detection_metric.value if mandatory_detection_metric else None
    )
    mandatory_case_count = _fmt(mandatory_case_metric.value if mandatory_case_metric else None)

    items = [
        EvidenceSummaryItem(
            family="lidar_pair",
            check="Candidate Support",
            status=support_grade,
            evidence=(
                f"source recall {_fmt(support_metric.value if support_metric else None)}, "
                f"shared voxels {_fmt(shared_metric.value if shared_metric else None)}, "
                f"centroid RMSE {_fmt(rmse_metric.value if rmse_metric else None)} m"
            ),
            interpretation=(
                "The candidate has voxel-level support under the declared real-data protocol."
            ),
            metric_ids=[
                "lidar_pair_source_voxel_recall_in_target",
                "lidar_pair_shared_voxel_count",
                "lidar_pair_shared_voxel_centroid_rmse_m",
            ],
        ),
    ]
    if p2p_available:
        items.append(
            EvidenceSummaryItem(
                family="lidar_pair",
                check="Holdout Geometry",
                status=p2p_grade,
                evidence=(
                    "median |p2plane| "
                    f"{_fmt(p2p_median_metric.value if p2p_median_metric else None)} m, "
                    f"P90 {_fmt(p2p_p90_metric.value if p2p_p90_metric else None)} m, "
                    "support "
                    f"{_fmt(_metric_value(p2p_support_ratio_metric))}, "
                    "unmatched "
                    f"{_fmt(p2p_unmatched_metric.value if p2p_unmatched_metric else None)}"
                ),
                interpretation=(
                    "Transformed target points are consistent with source-frame voxel planes."
                    if p2p_grade == "pass"
                    else "Point-to-plane support is weak or limited for this pair."
                ),
                metric_ids=[
                    "lidar_pair_holdout_point_to_plane_median_abs_m",
                    "lidar_pair_holdout_point_to_plane_p90_abs_m",
                    "lidar_pair_holdout_point_to_plane_support_ratio",
                    "lidar_pair_holdout_point_to_plane_unmatched_fraction",
                ],
            )
        )
    items.extend(
        [
        EvidenceSummaryItem(
            family="lidar_pair",
            check="Known-Bad Controls",
            status=known_bad_grade,
            evidence=(
                f"detectable fraction {known_bad_fraction}, "
                f"mandatory supported detections {mandatory_detection_count}/"
                f"{mandatory_case_count}, "
                f"max centroid RMSE delta {known_bad_max_delta} m, "
                f"max P90 point-to-plane delta {known_bad_p2p_delta} m"
            ),
            interpretation=(
                "Declared perturbations are distinguishable from the candidate."
                if known_bad_grade == "pass"
                else "Controls did not clearly separate from the candidate; evidence is weak."
            ),
            metric_ids=[
                "lidar_pair_known_bad_detectable_fraction",
                "lidar_pair_known_bad_mandatory_supported_detection_count",
                "lidar_pair_known_bad_mandatory_case_count",
                "lidar_pair_known_bad_centroid_rmse_delta_max_m",
                "lidar_pair_known_bad_point_to_plane_p90_delta_max_m",
            ],
        ),
        EvidenceSummaryItem(
            family="lidar_pair",
            check="Decision Boundary",
            status=decision_grade,
            evidence="candidate vs configured known-bad controls",
            interpretation=(
                "Supported by this evidence protocol, but not metrology ground truth."
                if decision_grade == "pass"
                else "Inconclusive under this evidence protocol; inspect support and controls."
            ),
            metric_ids=[
                "lidar_pair_source_voxel_recall_in_target",
                "lidar_pair_holdout_point_to_plane_p90_abs_m",
                "lidar_pair_known_bad_detectable_fraction",
            ],
        ),
        ]
    )
    return items


def _temporal_evidence_items(result: CalibrationResult) -> list[EvidenceSummaryItem]:
    separability_metric = result.metrics.get("online_temporal_separability")
    gate_metric = result.metrics.get("online_temporal_gate_verdict")
    offset_metric = result.metrics.get("online_temporal_estimated_offset_s")
    probe_metric = result.metrics.get("online_temporal_probe_detection_ratio")
    if separability_metric is None and gate_metric is None:
        return []

    temporal = result.run.provenance.get("temporal_evidence")
    adapted_flatness: float | None = None
    anchored_flatness: float | None = None
    separability_verdict: str | None = None
    anchor_mode: str | None = None
    injected_s: float | None = None
    if isinstance(temporal, dict):
        anchor_mode = temporal.get("time_offset_anchor") if isinstance(
            temporal.get("time_offset_anchor"), str
        ) else None
        injected_raw = temporal.get("time_offset_injected_s")
        injected_s = float(injected_raw) if isinstance(injected_raw, int | float) else None
        separability = temporal.get("separability")
        if isinstance(separability, dict):
            verdict_raw = separability.get("verdict")
            if isinstance(verdict_raw, str):
                separability_verdict = verdict_raw
            anchored_raw = separability.get("anchored_curve_flatness")
            adapted_raw = separability.get("adapted_curve_flatness")
            anchored_flatness = (
                float(anchored_raw) if isinstance(anchored_raw, int | float) else None
            )
            adapted_flatness = (
                float(adapted_raw) if isinstance(adapted_raw, int | float) else None
            )

    separability_grade: Grade = (
        separability_metric.grade if separability_metric is not None else "warn"
    )
    evidence_parts = [
        f"separability {separability_verdict or 'unavailable'}",
        f"anchored flatness {_fmt(anchored_flatness)}",
        f"adapted flatness {_fmt(adapted_flatness)}",
        f"anchor mode {anchor_mode or 'unknown'}",
    ]
    if injected_s is not None and abs(injected_s) > 1.0e-12:
        evidence_parts.append(f"injected offset {_fmt(injected_s)} s (validation-only)")

    metric_ids = ["online_temporal_separability"]
    if offset_metric is not None:
        metric_ids.append("online_temporal_estimated_offset_s")
    if probe_metric is not None:
        metric_ids.append("online_temporal_probe_detection_ratio")
    if gate_metric is not None:
        metric_ids.append("online_temporal_gate_verdict")

    return [
        EvidenceSummaryItem(
            family="temporal",
            check="Extrinsic/Temporal Separability",
            status=separability_grade,
            evidence=", ".join(evidence_parts),
            interpretation=(
                "Anchored holdout-RMSE curves separate extrinsic and temporal offsets under "
                "this motion profile."
                if separability_grade == "pass"
                else "Motion cannot excite δt or both curves remain flat; temporal verdict "
                "may be degenerate."
            ),
            metric_ids=metric_ids,
        ),
    ]


def _lidar_camera_evidence_items(result: CalibrationResult) -> list[EvidenceSummaryItem]:
    frame_metric = result.metrics.get("lidar_camera_projection_frame_count")
    projected_metric = result.metrics.get("lidar_camera_projected_points")
    ratio_metric = result.metrics.get("lidar_camera_projection_ratio")
    horizontal_metric = result.metrics.get("lidar_camera_projection_horizontal_coverage")
    vertical_metric = result.metrics.get("lidar_camera_projection_vertical_coverage")
    edge_metric = result.metrics.get("lidar_camera_edge_alignment_score")
    depth_edge_metric = result.metrics.get("lidar_camera_depth_edge_alignment_score")
    depth_points_metric = result.metrics.get("lidar_camera_depth_discontinuity_points")
    known_bad_metric = result.metrics.get("lidar_camera_perturbation_detectable_fraction")
    edge_delta_metric = result.metrics.get("lidar_camera_perturbation_edge_delta_mean")
    depth_edge_delta_metric = result.metrics.get(
        "lidar_camera_perturbation_depth_edge_delta_mean"
    )
    ratio_delta_metric = result.metrics.get(
        "lidar_camera_perturbation_projection_ratio_delta_mean"
    )
    mandatory_detection_metric = result.metrics.get(
        "lidar_camera_perturbation_mandatory_detectable_count"
    )
    mandatory_case_metric = result.metrics.get("lidar_camera_perturbation_mandatory_case_count")
    if (
        frame_metric is None
        and projected_metric is None
        and edge_metric is None
        and known_bad_metric is None
    ):
        return []

    gates = _lidar_camera_evidence_gates(result)
    profile = profile_from_domain(result.run.domain)
    support_grade = _lidar_camera_support_grade(
        projected_metric,
        ratio_metric,
        horizontal_metric,
        vertical_metric,
        profile=profile,
    )
    holdout_grade, holdout_evidence, depth_edge_available = _lidar_camera_holdout_grade(
        edge_metric,
        depth_edge_metric,
        depth_points_metric,
        gates=gates,
        profile=profile,
    )
    known_bad_grade = _lidar_camera_known_bad_grade(
        known_bad_metric,
        mandatory_detection_metric,
        mandatory_case_metric,
        gates=gates,
    )
    observability_text = _lidar_camera_observability_statement(
        result,
        horizontal_metric=horizontal_metric,
        vertical_metric=vertical_metric,
        depth_points_metric=depth_points_metric,
        depth_edge_available=depth_edge_available,
        mandatory_detection_metric=mandatory_detection_metric,
        mandatory_case_metric=mandatory_case_metric,
    )
    decision_grade: Grade = (
        "pass"
        if support_grade == "pass" and holdout_grade == "pass" and known_bad_grade == "pass"
        else "warn"
    )

    items = [
        EvidenceSummaryItem(
            family="lidar_camera",
            check="Candidate Support",
            status=support_grade,
            evidence=(
                "projected points train "
                f"{_fmt(projected_metric.train if projected_metric else None)} "
                f"holdout {_fmt(projected_metric.holdout if projected_metric else None)}, "
                "projection ratio train "
                f"{_fmt(ratio_metric.train if ratio_metric else None)} "
                f"holdout {_fmt(ratio_metric.holdout if ratio_metric else None)}, "
                f"h-coverage {_fmt(horizontal_metric.holdout if horizontal_metric else None)}, "
                f"v-coverage {_fmt(vertical_metric.holdout if vertical_metric else None)}"
            ),
            interpretation=(
                "Projected LiDAR points cover enough of the camera image for overlay evidence."
                if support_grade == "pass"
                else "Projection coverage is weak or points fall outside the image."
            ),
            metric_ids=[
                "lidar_camera_projected_points",
                "lidar_camera_projection_ratio",
                "lidar_camera_projection_horizontal_coverage",
                "lidar_camera_projection_vertical_coverage",
            ],
        ),
        EvidenceSummaryItem(
            family="lidar_camera",
            check="Holdout Edge Alignment",
            status=holdout_grade,
            evidence=holdout_evidence,
            interpretation=(
                "Holdout edge-alignment scores clear the recorded thresholds."
                if holdout_grade == "pass"
                else (
                    "Holdout edge-alignment is below threshold or unavailable for this sample."
                    if holdout_grade == "warn"
                    else "Holdout edge-alignment failed the recorded thresholds."
                )
            ),
            metric_ids=[
                "lidar_camera_edge_alignment_score",
                "lidar_camera_depth_edge_alignment_score",
                "lidar_camera_depth_discontinuity_points",
            ],
        ),
        EvidenceSummaryItem(
            family="lidar_camera",
            check="Known-Bad Controls",
            status=known_bad_grade,
            evidence=(
                "detectable fraction "
                f"{_fmt(known_bad_metric.value if known_bad_metric else None)} "
                f"(threshold >= "
                f"{_fmt(gates['evidence_gate_min_perturbation_detectable_fraction'])}), "
                "mandatory detections "
                f"{_fmt(mandatory_detection_metric.value if mandatory_detection_metric else None)}/"
                f"{_fmt(mandatory_case_metric.value if mandatory_case_metric else None)} "
                f"(threshold >= {_fmt(gates['evidence_gate_min_mandatory_detectable_count'])}), "
                f"mean edge delta {_fmt(edge_delta_metric.value if edge_delta_metric else None)}, "
                "mean depth-edge delta "
                f"{_fmt(depth_edge_delta_metric.value if depth_edge_delta_metric else None)}, "
                "mean projection-ratio delta "
                f"{_fmt(ratio_delta_metric.value if ratio_delta_metric else None)}"
            ),
            interpretation=(
                "Declared camera-frame perturbations are distinguishable from the candidate."
                if known_bad_grade == "pass"
                else "Controls did not clearly separate from the candidate; evidence is weak."
            ),
            metric_ids=[
                "lidar_camera_perturbation_detectable_fraction",
                "lidar_camera_perturbation_mandatory_detectable_count",
                "lidar_camera_perturbation_mandatory_case_count",
                "lidar_camera_perturbation_edge_delta_mean",
                "lidar_camera_perturbation_depth_edge_delta_mean",
                "lidar_camera_perturbation_projection_ratio_delta_mean",
            ],
        ),
        EvidenceSummaryItem(
            family="lidar_camera",
            check="Observability Statement",
            status="pass",
            evidence=observability_text,
            interpretation=(
                "What projection evidence can and cannot observe for this candidate."
            ),
            metric_ids=[
                "lidar_camera_projection_horizontal_coverage",
                "lidar_camera_projection_vertical_coverage",
                "lidar_camera_depth_discontinuity_points",
                "lidar_camera_perturbation_mandatory_detectable_count",
            ],
        ),
        EvidenceSummaryItem(
            family="lidar_camera",
            check="Decision Boundary",
            status=decision_grade,
            evidence=(
                "candidate vs configured camera-frame known-bad controls; edge holdout "
                f"threshold >= {_fmt(gates['evidence_gate_min_edge_alignment_holdout'])}"
            ),
            interpretation=(
                "Supported by this projection evidence protocol, but not metrology ground truth."
                if decision_grade == "pass"
                else "Inconclusive under this evidence protocol; inspect coverage and controls."
            ),
            metric_ids=[
                "lidar_camera_edge_alignment_score",
                "lidar_camera_perturbation_detectable_fraction",
                "lidar_camera_projection_ratio",
            ],
        ),
    ]
    return items


def _lidar_camera_evidence_gates(result: CalibrationResult) -> dict[str, float]:
    provenance = result.run.provenance.get("lidar_camera_evidence")
    if isinstance(provenance, dict):
        raw_gates = provenance.get("evidence_gates")
        if isinstance(raw_gates, dict):
            gates: dict[str, float] = {}
            for key in (
                "evidence_gate_min_edge_alignment_holdout",
                "evidence_gate_min_depth_edge_alignment_holdout",
                "evidence_gate_min_perturbation_detectable_fraction",
                "evidence_gate_min_mandatory_detectable_count",
            ):
                value = raw_gates.get(key)
                if isinstance(value, bool):
                    continue
                if isinstance(value, int | float):
                    gates[key] = float(value)
            if len(gates) == 4:
                return gates
    profile = profile_from_domain(result.run.domain)
    thresholds = thresholds_for_profile(profile)
    return {
        "evidence_gate_min_edge_alignment_holdout": thresholds[
            "lidar_camera_edge_alignment_score"
        ].pass_value,
        "evidence_gate_min_depth_edge_alignment_holdout": thresholds[
            "lidar_camera_depth_edge_alignment_score"
        ].pass_value,
        "evidence_gate_min_perturbation_detectable_fraction": thresholds[
            "lidar_camera_perturbation_detectable_fraction"
        ].pass_value,
        "evidence_gate_min_mandatory_detectable_count": 8.0,
    }


def _lidar_camera_support_grade(
    projected_metric: MetricResult | None,
    ratio_metric: MetricResult | None,
    horizontal_metric: MetricResult | None,
    vertical_metric: MetricResult | None,
    *,
    profile: ThresholdProfile,
) -> Grade:
    grades: list[Grade] = []
    for metric_id, metric in (
        ("lidar_camera_projected_points", projected_metric),
        ("lidar_camera_projection_ratio", ratio_metric),
        ("lidar_camera_projection_horizontal_coverage", horizontal_metric),
        ("lidar_camera_projection_vertical_coverage", vertical_metric),
    ):
        if metric is None:
            continue
        value = primary_metric_value(metric)
        if value is None:
            grades.append("warn")
            continue
        threshold = thresholds_for_profile(profile).get(metric_id)
        if threshold is None:
            grades.append(metric.grade)
            continue
        grades.append(threshold.grade(value))
    if not grades:
        return "warn"
    return _worst_grade(grades)


def _lidar_camera_holdout_grade(
    edge_metric: MetricResult | None,
    depth_edge_metric: MetricResult | None,
    depth_points_metric: MetricResult | None,
    *,
    gates: dict[str, float],
    profile: ThresholdProfile,
) -> tuple[Grade, str, bool]:
    if edge_metric is None or edge_metric.holdout is None:
        return (
            "warn",
            "holdout edge-alignment unavailable (insufficient projection frame pairs)",
            False,
        )
    edge_holdout = edge_metric.holdout
    edge_train = edge_metric.train
    edge_threshold = gates["evidence_gate_min_edge_alignment_holdout"]
    edge_grade: Grade = (
        "pass"
        if edge_holdout >= edge_threshold
        else (
            "warn"
            if edge_holdout
            >= thresholds_for_profile(profile)["lidar_camera_edge_alignment_score"].warn_value
            else "fail"
        )
    )
    depth_edge_available = (
        depth_points_metric is not None
        and (depth_points_metric.holdout or depth_points_metric.value or 0.0) > 0.0
        and depth_edge_metric is not None
        and depth_edge_metric.holdout is not None
    )
    evidence = (
        f"edge train {_fmt(edge_train)} holdout {_fmt(edge_holdout)} "
        f"(threshold >= {_fmt(edge_threshold)})"
    )
    if depth_edge_available:
        depth_holdout = depth_edge_metric.holdout if depth_edge_metric else None
        depth_threshold = gates["evidence_gate_min_depth_edge_alignment_holdout"]
        depth_grade = (
            "pass"
            if depth_holdout is not None and depth_holdout >= depth_threshold
            else "warn"
        )
        evidence += (
            f"; depth-edge holdout {_fmt(depth_holdout)} "
            f"(threshold >= {_fmt(depth_threshold)})"
        )
        holdout_grade = _worst_grade([edge_grade, depth_grade])
    else:
        evidence += "; depth-edge holdout unavailable (no depth discontinuities scored)"
        holdout_grade = edge_grade
    return holdout_grade, evidence, depth_edge_available


def _lidar_camera_known_bad_grade(
    known_bad_metric: MetricResult | None,
    mandatory_detection_metric: MetricResult | None,
    mandatory_case_metric: MetricResult | None,
    *,
    gates: dict[str, float],
) -> Grade:
    if known_bad_metric is None or known_bad_metric.value is None:
        return "warn"
    detectable_fraction = known_bad_metric.value
    if detectable_fraction < gates["evidence_gate_min_perturbation_detectable_fraction"]:
        return "warn" if detectable_fraction > 0.0 else "fail"
    mandatory_count = mandatory_detection_metric.value if mandatory_detection_metric else None
    mandatory_cases = mandatory_case_metric.value if mandatory_case_metric else None
    if mandatory_cases is not None and mandatory_cases < 12.0:
        return "warn"
    if mandatory_count is None:
        return "warn"
    if mandatory_count < gates["evidence_gate_min_mandatory_detectable_count"]:
        return "warn"
    return "pass"


def _lidar_camera_observability_statement(
    result: CalibrationResult,
    *,
    horizontal_metric: MetricResult | None,
    vertical_metric: MetricResult | None,
    depth_points_metric: MetricResult | None,
    depth_edge_available: bool,
    mandatory_detection_metric: MetricResult | None,
    mandatory_case_metric: MetricResult | None,
) -> str:
    horizontal = horizontal_metric.holdout if horizontal_metric else None
    vertical = vertical_metric.holdout if vertical_metric else None
    depth_points = depth_points_metric.holdout if depth_points_metric else None
    statements: list[str] = [
        "Observes image-plane alignment of projected LiDAR points against intensity edges",
    ]
    if horizontal is not None and vertical is not None:
        statements.append(
            f"horizontal coverage {_fmt(horizontal)} and vertical coverage {_fmt(vertical)} "
            "limit sensitivity to in-image motion and baselines"
        )
    if depth_edge_available:
        statements.append(
            f"depth-edge channel scored with {_fmt(depth_points)} discontinuity points"
        )
    else:
        statements.append(
            "depth-edge channel not scored because projected points lack nearby depth jumps"
        )
    mandatory_count = mandatory_detection_metric.value if mandatory_detection_metric else None
    mandatory_cases = mandatory_case_metric.value if mandatory_case_metric else None
    if mandatory_count is not None:
        case_total = int(mandatory_cases) if mandatory_cases is not None else 12
        statements.append(
            f"mandatory rotation/translation probes detected {int(mandatory_count)} of "
            f"{case_total} cases; coverage-limited DOFs may remain unobserved on sparse samples"
        )
    limitations = result.run.provenance.get("lidar_camera_evidence")
    if isinstance(limitations, dict):
        raw_limitations = limitations.get("limitations")
        if isinstance(raw_limitations, list):
            for item in raw_limitations:
                if isinstance(item, str):
                    statements.append(item)
    return "; ".join(statements)


def _lidar_imu_evidence_items(result: CalibrationResult) -> list[EvidenceSummaryItem]:
    interval_metric = result.metrics.get("lidar_imu_pose_interval_count")
    sample_metric = result.metrics.get("lidar_imu_imu_sample_count")
    excitation_metric = result.metrics.get("lidar_imu_odometry_angular_excitation_p95_dps")
    holdout_rmse_metric = result.metrics.get("lidar_imu_holdout_rotation_rate_rmse_dps")
    holdout_mean_metric = result.metrics.get("lidar_imu_holdout_rotation_rate_mean_abs_dps")
    known_bad_metric = result.metrics.get("lidar_imu_known_bad_detectable_fraction")
    mandatory_detection_metric = result.metrics.get(
        "lidar_imu_known_bad_mandatory_detectable_count"
    )
    mandatory_case_metric = result.metrics.get("lidar_imu_known_bad_mandatory_case_count")
    gravity_metric = result.metrics.get("lidar_imu_gravity_alignment_deg")
    if (
        interval_metric is None
        and holdout_rmse_metric is None
        and known_bad_metric is None
        and gravity_metric is None
    ):
        return []

    gates = _lidar_imu_evidence_gates(result)
    axis_excitation = _lidar_imu_axis_excitation(result)
    observability_note = _lidar_imu_axis_observability_note(axis_excitation, _gates=gates)
    support_grade = _lidar_imu_support_grade(excitation_metric, interval_metric, gates=gates)
    holdout_grade = _lidar_imu_holdout_grade(holdout_rmse_metric, gates=gates)
    known_bad_grade = _lidar_imu_known_bad_grade(
        known_bad_metric,
        mandatory_detection_metric,
        mandatory_case_metric,
        gates=gates,
    )
    gravity_grade = _lidar_imu_gravity_grade(gravity_metric, gates=gates)
    decision_grade: Grade = (
        "pass"
        if support_grade == "pass" and holdout_grade == "pass" and known_bad_grade == "pass"
        else "warn"
    )

    return [
        EvidenceSummaryItem(
            family="lidar_imu",
            check="Candidate Support",
            status=support_grade,
            evidence=(
                f"imu samples {_fmt(sample_metric.value if sample_metric else None)}, "
                f"pose intervals {_fmt(interval_metric.value if interval_metric else None)}, "
                f"odometry excitation p95 "
                f"{_fmt(excitation_metric.value if excitation_metric else None)} deg/s "
                f"(floor {_fmt(gates['min_excitation_p95_dps'])} deg/s); "
                f"per-axis p95 |ω| "
                f"x {_fmt(axis_excitation.get('x'))} "
                f"y {_fmt(axis_excitation.get('y'))} "
                f"z {_fmt(axis_excitation.get('z'))} deg/s"
            ),
            interpretation=(
                "Sufficient angular excitation and IMU samples for rotation-rate evidence. "
                f"{observability_note}"
                if support_grade == "pass"
                else (
                    "INCONCLUSIVE: angular excitation below the declared floor; rotation "
                    "evidence cannot falsify the candidate."
                    if excitation_metric is not None
                    and excitation_metric.value is not None
                    and excitation_metric.value < gates["min_excitation_p95_dps"]
                    else "IMU sample or interval support is weak."
                )
            ),
            metric_ids=[
                "lidar_imu_imu_sample_count",
                "lidar_imu_pose_interval_count",
                "lidar_imu_odometry_angular_excitation_p95_dps",
            ],
        ),
        EvidenceSummaryItem(
            family="lidar_imu",
            check="Holdout Rotation Consistency",
            status=holdout_grade,
            evidence=(
                "holdout RMSE "
                f"{_fmt(holdout_rmse_metric.holdout if holdout_rmse_metric else None)} "
                "deg/s (threshold <= "
                f"{_fmt(gates['imu_gate_max_holdout_rotation_rate_rmse_dps'])}), "
                "mean |Δω| "
                f"{_fmt(holdout_mean_metric.holdout if holdout_mean_metric else None)} deg/s"
            ),
            interpretation=(
                "Candidate-rotated IMU angular velocity matches odometry-derived body rates "
                "on holdout pose intervals."
                if holdout_grade == "pass"
                else "Holdout rotation-rate residual exceeds the declared gate or is unavailable."
            ),
            metric_ids=[
                "lidar_imu_holdout_rotation_rate_rmse_dps",
                "lidar_imu_holdout_rotation_rate_mean_abs_dps",
            ],
        ),
        EvidenceSummaryItem(
            family="lidar_imu",
            check="Known-Bad Controls",
            status=known_bad_grade,
            evidence=(
                f"detectable fraction "
                f"{_fmt(known_bad_metric.value if known_bad_metric else None)} "
                f"(threshold >= 0.5 detectable fraction; margin "
                f"{_fmt(gates['known_bad_detect_margin'])} relative RMSE increase), "
                "mandatory detected "
                f"{_fmt(mandatory_detection_metric.value if mandatory_detection_metric else None)}"
                " / "
                f"{_fmt(mandatory_case_metric.value if mandatory_case_metric else None)}"
            ),
            interpretation=(
                "Mandatory ±5°/±10° rotation probes worsen holdout rotation-rate RMSE."
                if known_bad_grade == "pass"
                else "Known-bad rotation probes did not reliably falsify the candidate."
            ),
            metric_ids=[
                "lidar_imu_known_bad_detectable_fraction",
                "lidar_imu_known_bad_mandatory_detectable_count",
                "lidar_imu_known_bad_mandatory_case_count",
            ],
        ),
        EvidenceSummaryItem(
            family="lidar_imu",
            check="Gravity Support",
            status=gravity_grade,
            evidence=(
                f"gravity alignment angle "
                f"{_fmt(gravity_metric.value if gravity_metric else None)} deg "
                f"(warn threshold <= {_fmt(gates['imu_gate_max_gravity_alignment_deg'])} deg)"
            ),
            interpretation=(
                "Low-pass accelerometer direction aligns with world gravity under the candidate "
                "(supporting-only; never fails the run)."
                if gravity_grade == "pass"
                else "Gravity alignment is weak or unavailable; treat as supporting context only."
            ),
            metric_ids=["lidar_imu_gravity_alignment_deg"],
        ),
        EvidenceSummaryItem(
            family="lidar_imu",
            check="Decision Boundary",
            status=decision_grade,
            evidence=(
                "pass requires support + holdout rotation consistency + known-bad controls; "
                f"holdout RMSE gate <= "
                f"{_fmt(gates['imu_gate_max_holdout_rotation_rate_rmse_dps'])} deg/s"
            ),
            interpretation=(
                "Rotation-rate evidence supports the candidate under ADR 0004 protocol shape."
                if decision_grade == "pass"
                else "Rotation evidence is inconclusive; inspect support, holdout RMSE, and probes."
            ),
            metric_ids=[
                "lidar_imu_holdout_rotation_rate_rmse_dps",
                "lidar_imu_known_bad_detectable_fraction",
                "lidar_imu_odometry_angular_excitation_p95_dps",
            ],
        ),
    ]


def _lidar_imu_evidence_gates(result: CalibrationResult) -> dict[str, float]:
    provenance = result.run.provenance.get("lidar_imu_evidence")
    if isinstance(provenance, dict):
        raw_gates = provenance.get("evidence_gates")
        if isinstance(raw_gates, dict):
            gates: dict[str, float] = {}
            for key in (
                "imu_gate_max_holdout_rotation_rate_rmse_dps",
                "imu_gate_max_gravity_alignment_deg",
                "known_bad_detect_margin",
                "min_excitation_p95_dps",
            ):
                value = raw_gates.get(key)
                if isinstance(value, bool):
                    continue
                if isinstance(value, int | float):
                    gates[key] = float(value)
            if len(gates) == 4:
                return gates
    return {
        "imu_gate_max_holdout_rotation_rate_rmse_dps": 5.0,
        "imu_gate_max_gravity_alignment_deg": 10.0,
        "known_bad_detect_margin": 0.20,
        "min_excitation_p95_dps": 5.0,
    }


def _lidar_imu_axis_excitation(result: CalibrationResult) -> dict[str, float | None]:
    provenance = result.run.provenance.get("lidar_imu_evidence")
    if not isinstance(provenance, dict):
        return {}
    raw = provenance.get("odometry_angular_excitation_p95_dps_per_axis")
    if not isinstance(raw, dict):
        return {}
    axis_excitation: dict[str, float | None] = {}
    for axis in ("x", "y", "z"):
        value = raw.get(axis)
        if isinstance(value, bool):
            axis_excitation[axis] = None
        elif isinstance(value, int | float):
            axis_excitation[axis] = float(value)
        else:
            axis_excitation[axis] = None
    return axis_excitation


def _lidar_imu_axis_observability_note(
    axis_excitation: dict[str, float | None],
    *,
    _gates: dict[str, float],
) -> str:
    if not axis_excitation:
        return (
            "Axis-observability: per-axis excitation split unavailable; known-bad probes "
            "about the dominant rotation axis may be weakly detectable."
        )
    axis_to_probe = {"x": "roll", "y": "pitch", "z": "yaw"}
    measured: dict[str, float] = {
        axis: value
        for axis in ("x", "y", "z")
        if (value := axis_excitation.get(axis)) is not None
    }
    if not measured:
        return (
            "Axis-observability: per-axis excitation split unavailable; known-bad probes "
            "about the dominant rotation axis may be weakly detectable."
        )
    dominant_axis = max(measured, key=lambda axis: measured[axis])
    weak_probe = axis_to_probe[dominant_axis]
    falsifiable = [
        probe_axis
        for probe_axis in ("roll", "pitch", "yaw")
        if probe_axis != weak_probe
    ]
    parts = [
        "Axis-observability: rotation probes about the dominant excitation axis are "
        "inherently weak (a misalignment about that axis leaves the dominant body-rate "
        "component unchanged).",
        f"Dominant excitation axis: {dominant_axis} "
        f"(p95 {measured[dominant_axis]:.1f} deg/s).",
    ]
    if falsifiable:
        parts.append(f"Falsifiable probe axes: {', '.join(falsifiable)}.")
    parts.append(f"Weakly detectable probe axis: {weak_probe}.")
    return " ".join(parts)


def _lidar_imu_support_grade(
    excitation_metric: MetricResult | None,
    interval_metric: MetricResult | None,
    *,
    gates: dict[str, float],
) -> Grade:
    excitation = excitation_metric.value if excitation_metric else None
    intervals = interval_metric.value if interval_metric else None
    if excitation is None or intervals is None:
        return "warn"
    if excitation < gates["min_excitation_p95_dps"]:
        return "warn"
    if intervals < 4.0:
        return "warn"
    return "pass"


def _lidar_imu_holdout_grade(
    holdout_rmse_metric: MetricResult | None,
    *,
    gates: dict[str, float],
) -> Grade:
    if holdout_rmse_metric is None or holdout_rmse_metric.holdout is None:
        return "warn"
    if holdout_rmse_metric.holdout <= gates["imu_gate_max_holdout_rotation_rate_rmse_dps"]:
        return "pass"
    return "warn"


def _lidar_imu_known_bad_grade(
    known_bad_metric: MetricResult | None,
    mandatory_detection_metric: MetricResult | None,
    mandatory_case_metric: MetricResult | None,
    *,
    gates: dict[str, float],
) -> Grade:
    _ = gates
    if known_bad_metric is None or known_bad_metric.value is None:
        return "warn"
    detectable_fraction = known_bad_metric.value
    if detectable_fraction >= 0.5:
        pass_grade = True
    elif detectable_fraction >= 0.1:
        return "warn"
    else:
        return "fail"
    mandatory_count = mandatory_detection_metric.value if mandatory_detection_metric else None
    mandatory_cases = mandatory_case_metric.value if mandatory_case_metric else None
    if mandatory_cases is not None and mandatory_cases < 12.0:
        return "warn"
    if mandatory_count is None or mandatory_count < 8.0:
        return "warn"
    return "pass" if pass_grade else "warn"


def _lidar_imu_gravity_grade(
    gravity_metric: MetricResult | None,
    *,
    gates: dict[str, float],
) -> Grade:
    if gravity_metric is None or gravity_metric.value is None:
        return "warn"
    if gravity_metric.value <= gates["imu_gate_max_gravity_alignment_deg"]:
        return "pass"
    return "warn"


def _worst_grade(grades: Iterable[str]) -> Grade:
    order = {"pass": 0, "warn": 1, "fail": 2}
    selected = "pass"
    for grade in grades:
        if order.get(grade, 1) > order.get(selected, 1):
            selected = grade
    return cast(Grade, selected)


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def _metric_value(metric: object) -> float | None:
    if metric is None:
        return None
    value = getattr(metric, "value", None)
    return value if isinstance(value, float | int) else None


def _number_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    return float(value) if isinstance(value, float | int) else None


def _mapping_value(mapping: object, key: str) -> object:
    return mapping.get(key) if isinstance(mapping, dict) else None


def _radar_sensor_supports_holdout(evaluation: dict[str, object]) -> bool:
    static_summary = evaluation.get("static_return_summary")
    holdout_static = (
        static_summary.get("holdout") if isinstance(static_summary, dict) else None
    )
    observability = evaluation.get("holdout_yaw_observability")
    return (
        isinstance(holdout_static, dict)
        and holdout_static.get("status") == "supported"
        and isinstance(observability, dict)
        and observability.get("status") == "supported"
    )


def _radar_sensor_controls_pass(evaluation: dict[str, object]) -> bool:
    challenge = evaluation.get("known_bad_rotation_probes")
    return (
        isinstance(challenge, dict)
        and challenge.get("supported_count") == challenge.get("required_count")
        and challenge.get("detectable_fraction") == 1.0
    )
