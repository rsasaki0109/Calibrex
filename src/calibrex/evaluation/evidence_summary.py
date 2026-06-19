"""Evidence summary builders shared by reports and comparisons."""

from __future__ import annotations

from calibrex.core.report_artifacts import EvidenceCaseItem, EvidenceSummaryItem
from calibrex.core.result import CalibrationResult, Grade


def evidence_summaries_from_result(result: CalibrationResult) -> list[EvidenceSummaryItem]:
    """Return machine-readable evidence summary rows for a calibration result."""

    return _lidar_pair_evidence_items(result)


def evidence_cases_from_result(result: CalibrationResult) -> list[EvidenceCaseItem]:
    """Return machine-readable evidence case rows stored in result provenance."""

    raw_cases = result.run.provenance.get("evidence_cases")
    if not isinstance(raw_cases, list):
        return []
    cases: list[EvidenceCaseItem] = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            continue
        cases.append(EvidenceCaseItem.model_validate(raw_case))
    return cases


def _lidar_pair_evidence_items(result: CalibrationResult) -> list[EvidenceSummaryItem]:
    support_metric = result.metrics.get("lidar_pair_source_voxel_recall_in_target")
    shared_metric = result.metrics.get("lidar_pair_shared_voxel_count")
    rmse_metric = result.metrics.get("lidar_pair_shared_voxel_centroid_rmse_m")
    p2p_median_metric = result.metrics.get("lidar_pair_holdout_point_to_plane_median_abs_m")
    p2p_p90_metric = result.metrics.get("lidar_pair_holdout_point_to_plane_p90_abs_m")
    p2p_unmatched_metric = result.metrics.get(
        "lidar_pair_holdout_point_to_plane_unmatched_fraction"
    )
    known_bad_metric = result.metrics.get("lidar_pair_known_bad_detectable_fraction")
    max_delta_metric = result.metrics.get("lidar_pair_known_bad_centroid_rmse_delta_max_m")
    if (
        support_metric is None
        and shared_metric is None
        and rmse_metric is None
        and p2p_median_metric is None
        and p2p_p90_metric is None
        and known_bad_metric is None
        and max_delta_metric is None
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
                f"max RMSE delta {known_bad_max_delta} m"
            ),
            interpretation=(
                "Declared perturbations are distinguishable from the candidate."
                if known_bad_grade == "pass"
                else "Controls did not clearly separate from the candidate; evidence is weak."
            ),
            metric_ids=[
                "lidar_pair_known_bad_detectable_fraction",
                "lidar_pair_known_bad_centroid_rmse_delta_max_m",
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


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"
