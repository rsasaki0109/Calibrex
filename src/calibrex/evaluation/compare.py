"""Result comparison helpers for calibration candidates and baselines."""

from __future__ import annotations

from collections.abc import Iterable
from math import acos, degrees, sqrt
from pathlib import Path
from typing import Literal

from pydantic import Field

from calibrex.core.geometry import normalize_quaternion_xyzw
from calibrex.core.report_artifacts import EvidenceSummaryItem
from calibrex.core.result import (
    CalibrationResult,
    Grade,
    MetricResult,
    StrictModel,
    TransformResult,
)
from calibrex.evaluation.evidence_summary import evidence_summaries_from_result
from calibrex.evaluation.metric_families import metric_family

COMPARISON_SCHEMA_VERSION: Literal["calibrex.comparison/v0.1"] = (
    "calibrex.comparison/v0.1"
)

MetricValueField = Literal["holdout", "value", "train", "none"]
MetricPreference = Literal["lower", "higher", "unknown"]
ComparisonWinner = Literal["left", "right", "tie", "not_comparable"]
ProtocolCompatibilityStatus = Literal["compatible", "warning", "not_comparable"]
MetricsOrigin = Literal["recomputed", "cached", "unknown"]


class ComparisonSide(StrictModel):
    """Summary of one compared result."""

    path: str | None = None
    run_id: str
    status: str
    grade: Grade
    domain: str
    metrics_origin: MetricsOrigin = "unknown"
    data_verified: bool | None = None
    computed_at: str | None = None


class MetricSide(StrictModel):
    """One side of a metric comparison."""

    train: float | None = None
    holdout: float | None = None
    value: float | None = None
    grade: Grade
    unit: str | None = None
    reason: str | None = None


class MetricComparison(StrictModel):
    """Comparison for one shared metric."""

    name: str
    family: str
    unit: str | None = None
    left: MetricSide
    right: MetricSide
    preferred_field: MetricValueField
    left_preferred: float | None = None
    right_preferred: float | None = None
    delta_right_minus_left: float | None = None
    preference: MetricPreference = "unknown"
    winner: ComparisonWinner = "not_comparable"


class MetricFamilyComparison(StrictModel):
    """Metric comparison rollup for one family."""

    family: str
    metric_count: int
    left_better_count: int = 0
    right_better_count: int = 0
    tied_count: int = 0
    not_comparable_count: int = 0
    metrics: list[str] = Field(default_factory=list)


class TransformSide(StrictModel):
    """One side of a transform comparison."""

    parent: str
    child: str
    translation_m: list[float]
    rotation_quat_xyzw: list[float]
    grade: Grade


class TransformComparison(StrictModel):
    """Comparison for one shared transform entry."""

    name: str
    left: TransformSide
    right: TransformSide
    parent: str
    child: str
    translation_delta_m: float
    rotation_delta_deg: float


class TransformGroupComparison(StrictModel):
    """Comparison for one transform namespace."""

    group: Literal["transforms", "candidate_extrinsics", "reference_extrinsics"]
    comparison_count: int
    only_left: list[str] = Field(default_factory=list)
    only_right: list[str] = Field(default_factory=list)
    max_translation_delta_m: float | None = None
    max_rotation_delta_deg: float | None = None
    comparisons: list[TransformComparison] = Field(default_factory=list)


class ObservabilityComparison(StrictModel):
    """Observability and weak-DoF comparison."""

    left_rank: int | None = None
    right_rank: int | None = None
    rank_delta_right_minus_left: int | None = None
    left_condition_number: float | None = None
    right_condition_number: float | None = None
    condition_number_delta_right_minus_left: float | None = None
    shared_weak_directions: list[str] = Field(default_factory=list)
    only_left_weak_directions: list[str] = Field(default_factory=list)
    only_right_weak_directions: list[str] = Field(default_factory=list)


class EvidenceSummarySide(StrictModel):
    """One side of an evidence summary comparison."""

    status: Grade
    evidence: str
    interpretation: str
    metric_ids: list[str] = Field(default_factory=list)


class EvidenceSummaryComparison(StrictModel):
    """Comparison for one evidence summary check."""

    family: str
    check: str
    left: EvidenceSummarySide | None = None
    right: EvidenceSummarySide | None = None
    winner: ComparisonWinner = "not_comparable"


class EvidenceProtocolSide(StrictModel):
    """One declared evidence protocol on one side of a comparison."""

    protocol_id: str
    family: str | None = None
    metrics_origin: str
    data_verified: bool | None = None
    split_policy: str | None = None
    independent_holdout: bool | None = None
    known_bad_case_count: int | None = None


class EvidenceProtocolCompatibility(StrictModel):
    """Whether two results were evaluated under comparable evidence protocols."""

    status: ProtocolCompatibilityStatus
    reasons: list[str] = Field(default_factory=list)
    shared_protocol_ids: list[str] = Field(default_factory=list)
    only_left_protocol_ids: list[str] = Field(default_factory=list)
    only_right_protocol_ids: list[str] = Field(default_factory=list)
    left_protocols: list[EvidenceProtocolSide] = Field(default_factory=list)
    right_protocols: list[EvidenceProtocolSide] = Field(default_factory=list)


class ComparisonSummary(StrictModel):
    """High-level comparison counters."""

    metric_comparison_count: int
    left_better_metric_count: int
    right_better_metric_count: int
    tied_metric_count: int
    not_comparable_metric_count: int
    transform_comparison_count: int
    max_translation_delta_m: float | None = None
    max_rotation_delta_deg: float | None = None


class ResultComparison(StrictModel):
    """Machine-readable comparison between two Calibrex results."""

    schema_version: Literal["calibrex.comparison/v0.1"] = COMPARISON_SCHEMA_VERSION
    left: ComparisonSide
    right: ComparisonSide
    summary: ComparisonSummary
    metric_families: dict[str, MetricFamilyComparison] = Field(default_factory=dict)
    metrics: dict[str, MetricComparison] = Field(default_factory=dict)
    only_left_metrics: list[str] = Field(default_factory=list)
    only_right_metrics: list[str] = Field(default_factory=list)
    transform_groups: dict[str, TransformGroupComparison] = Field(default_factory=dict)
    observability: ObservabilityComparison
    protocol_compatibility: EvidenceProtocolCompatibility
    evidence_comparisons: list[EvidenceSummaryComparison] = Field(default_factory=list)
    left_degeneracy_grade: Grade
    right_degeneracy_grade: Grade
    left_degeneracy_reason: str | None = None
    right_degeneracy_reason: str | None = None


def compare_results(
    left: CalibrationResult,
    right: CalibrationResult,
    *,
    left_path: str | Path | None = None,
    right_path: str | Path | None = None,
) -> ResultComparison:
    """Compare two Calibrex result files without changing either result."""

    metric_comparisons = _compare_metrics(left.metrics, right.metrics)
    transform_groups = {
        "transforms": _compare_transform_group("transforms", left.transforms, right.transforms),
        "candidate_extrinsics": _compare_transform_group(
            "candidate_extrinsics",
            left.candidate_extrinsics,
            right.candidate_extrinsics,
        ),
        "reference_extrinsics": _compare_transform_group(
            "reference_extrinsics",
            left.reference_extrinsics,
            right.reference_extrinsics,
        ),
    }
    metric_families = _metric_family_comparisons(metric_comparisons)
    summary = _comparison_summary(metric_comparisons, transform_groups)
    evidence_comparisons = _compare_evidence_summaries(left, right)

    return ResultComparison(
        left=_side(left, left_path),
        right=_side(right, right_path),
        summary=summary,
        metric_families=metric_families,
        metrics=metric_comparisons,
        only_left_metrics=sorted(set(left.metrics) - set(right.metrics)),
        only_right_metrics=sorted(set(right.metrics) - set(left.metrics)),
        transform_groups=transform_groups,
        observability=_compare_observability(left, right),
        protocol_compatibility=_compare_protocol_compatibility(left, right),
        evidence_comparisons=evidence_comparisons,
        left_degeneracy_grade=left.degeneracy.grade,
        right_degeneracy_grade=right.degeneracy.grade,
        left_degeneracy_reason=left.degeneracy.reason,
        right_degeneracy_reason=right.degeneracy.reason,
    )


def comparison_json_schema() -> dict[str, object]:
    """Return the JSON schema for comparison artifacts."""

    return ResultComparison.model_json_schema()


def _side(result: CalibrationResult, path: str | Path | None) -> ComparisonSide:
    provenance = result.run.provenance
    return ComparisonSide(
        path=str(path) if path is not None else None,
        run_id=result.run.id,
        status=result.run.status,
        grade=result.quality.grade,
        domain=result.run.domain,
        metrics_origin=_metrics_origin(provenance),
        data_verified=_bool_or_none(provenance.get("data_verified")),
        computed_at=_str_or_none(provenance.get("computed_at")),
    )


def _compare_metrics(
    left_metrics: dict[str, MetricResult],
    right_metrics: dict[str, MetricResult],
) -> dict[str, MetricComparison]:
    comparisons: dict[str, MetricComparison] = {}
    for name in sorted(set(left_metrics) & set(right_metrics)):
        left = left_metrics[name]
        right = right_metrics[name]
        preferred_field = _preferred_value_field(left, right)
        left_value = _metric_value(left, preferred_field)
        right_value = _metric_value(right, preferred_field)
        delta = None if left_value is None or right_value is None else right_value - left_value
        preference = _metric_preference(name)
        winner = _metric_winner(left_value, right_value, preference)
        comparisons[name] = MetricComparison(
            name=name,
            family=metric_family(name),
            unit=right.unit or left.unit,
            left=_metric_side(left),
            right=_metric_side(right),
            preferred_field=preferred_field,
            left_preferred=left_value,
            right_preferred=right_value,
            delta_right_minus_left=delta,
            preference=preference,
            winner=winner,
        )
    return comparisons


def _metric_side(metric: MetricResult) -> MetricSide:
    return MetricSide(
        train=metric.train,
        holdout=metric.holdout,
        value=metric.value,
        grade=metric.grade,
        unit=metric.unit,
        reason=metric.reason,
    )


def _preferred_value_field(left: MetricResult, right: MetricResult) -> MetricValueField:
    fields: tuple[MetricValueField, ...] = ("holdout", "value", "train")
    for field in fields:
        if _metric_value(left, field) is not None and _metric_value(right, field) is not None:
            return field
    return "none"


def _metric_value(metric: MetricResult, field: MetricValueField) -> float | None:
    if field == "holdout":
        return metric.holdout
    if field == "value":
        return metric.value
    if field == "train":
        return metric.train
    return None


def _metric_preference(name: str) -> MetricPreference:
    if name.startswith("lidar_pair_known_bad_") and "delta" in name:
        return "higher"
    lower_tokens = (
        "rmse",
        "residual",
        "error",
        "roughness",
        "condition_number",
        "timestamp_alignment",
        "alignment_ms",
        "time_offset",
        "delta_max",
        "delta_mean",
        "p95",
        "median_holdout",
        "point_to_plane",
        "unmatched",
    )
    higher_tokens = (
        "coverage",
        "count",
        "rank",
        "detectable_fraction",
        "inlier_fraction",
        "sharpness",
        "overlay_score",
        "alignment_score",
        "mutual_information_score",
        "projection_ratio",
        "duration",
        "excitation",
    )
    if any(token in name for token in lower_tokens):
        return "lower"
    if any(token in name for token in higher_tokens):
        return "higher"
    return "unknown"


def _metric_winner(
    left_value: float | None,
    right_value: float | None,
    preference: MetricPreference,
) -> ComparisonWinner:
    if left_value is None or right_value is None or preference == "unknown":
        return "not_comparable"
    if abs(left_value - right_value) <= 1.0e-12:
        return "tie"
    if preference == "lower":
        return "left" if left_value < right_value else "right"
    return "left" if left_value > right_value else "right"


def _metric_family_comparisons(
    comparisons: dict[str, MetricComparison],
) -> dict[str, MetricFamilyComparison]:
    families: dict[str, MetricFamilyComparison] = {}
    for comparison in comparisons.values():
        family = families.setdefault(
            comparison.family,
            MetricFamilyComparison(family=comparison.family, metric_count=0),
        )
        family.metric_count += 1
        family.metrics.append(comparison.name)
        if comparison.winner == "left":
            family.left_better_count += 1
        elif comparison.winner == "right":
            family.right_better_count += 1
        elif comparison.winner == "tie":
            family.tied_count += 1
        else:
            family.not_comparable_count += 1
    for family in families.values():
        family.metrics.sort()
    return dict(sorted(families.items()))


def _compare_transform_group(
    group: Literal["transforms", "candidate_extrinsics", "reference_extrinsics"],
    left_transforms: dict[str, TransformResult],
    right_transforms: dict[str, TransformResult],
) -> TransformGroupComparison:
    comparisons = [
        _compare_transform(name, left_transforms[name], right_transforms[name])
        for name in sorted(set(left_transforms) & set(right_transforms))
    ]
    return TransformGroupComparison(
        group=group,
        comparison_count=len(comparisons),
        only_left=sorted(set(left_transforms) - set(right_transforms)),
        only_right=sorted(set(right_transforms) - set(left_transforms)),
        max_translation_delta_m=_max_or_none(
            comparison.translation_delta_m for comparison in comparisons
        ),
        max_rotation_delta_deg=_max_or_none(
            comparison.rotation_delta_deg for comparison in comparisons
        ),
        comparisons=comparisons,
    )


def _compare_transform(
    name: str,
    left: TransformResult,
    right: TransformResult,
) -> TransformComparison:
    return TransformComparison(
        name=name,
        left=_transform_side(left),
        right=_transform_side(right),
        parent=left.parent if left.parent == right.parent else f"{left.parent} | {right.parent}",
        child=left.child if left.child == right.child else f"{left.child} | {right.child}",
        translation_delta_m=_translation_delta(left, right),
        rotation_delta_deg=_rotation_delta_deg(left, right),
    )


def _transform_side(transform: TransformResult) -> TransformSide:
    return TransformSide(
        parent=transform.parent,
        child=transform.child,
        translation_m=list(transform.translation_m),
        rotation_quat_xyzw=list(transform.rotation_quat_xyzw),
        grade=transform.quality.grade,
    )


def _translation_delta(left: TransformResult, right: TransformResult) -> float:
    return sqrt(
        sum(
            (float(right.translation_m[index]) - float(left.translation_m[index])) ** 2
            for index in range(3)
        )
    )


def _rotation_delta_deg(left: TransformResult, right: TransformResult) -> float:
    left_q = normalize_quaternion_xyzw(left.rotation_quat_xyzw)
    right_q = normalize_quaternion_xyzw(right.rotation_quat_xyzw)
    dot = abs(sum(left_q[index] * right_q[index] for index in range(4)))
    clamped = max(-1.0, min(1.0, dot))
    return degrees(2.0 * acos(clamped))


def _compare_observability(
    left: CalibrationResult,
    right: CalibrationResult,
) -> ObservabilityComparison:
    left_weak = set(left.observability.weak_directions)
    right_weak = set(right.observability.weak_directions)
    left_rank = left.observability.rank
    right_rank = right.observability.rank
    left_condition_number = left.observability.condition_number
    right_condition_number = right.observability.condition_number
    return ObservabilityComparison(
        left_rank=left_rank,
        right_rank=right_rank,
        rank_delta_right_minus_left=(
            None if left_rank is None or right_rank is None else right_rank - left_rank
        ),
        left_condition_number=left_condition_number,
        right_condition_number=right_condition_number,
        condition_number_delta_right_minus_left=(
            None
            if left_condition_number is None or right_condition_number is None
            else right_condition_number - left_condition_number
        ),
        shared_weak_directions=sorted(left_weak & right_weak),
        only_left_weak_directions=sorted(left_weak - right_weak),
        only_right_weak_directions=sorted(right_weak - left_weak),
    )


def _compare_evidence_summaries(
    left: CalibrationResult,
    right: CalibrationResult,
) -> list[EvidenceSummaryComparison]:
    left_items = _evidence_items_by_key(evidence_summaries_from_result(left))
    right_items = _evidence_items_by_key(evidence_summaries_from_result(right))
    comparisons: list[EvidenceSummaryComparison] = []
    for family, check in sorted(set(left_items) | set(right_items)):
        left_item = left_items.get((family, check))
        right_item = right_items.get((family, check))
        comparisons.append(
            EvidenceSummaryComparison(
                family=family,
                check=check,
                left=_evidence_side(left_item),
                right=_evidence_side(right_item),
                winner=_evidence_winner(left_item, right_item),
            )
        )
    return comparisons


def _evidence_items_by_key(
    items: list[EvidenceSummaryItem],
) -> dict[tuple[str, str], EvidenceSummaryItem]:
    return {(item.family, item.check): item for item in items}


def _evidence_side(item: EvidenceSummaryItem | None) -> EvidenceSummarySide | None:
    if item is None:
        return None
    return EvidenceSummarySide(
        status=item.status,
        evidence=item.evidence,
        interpretation=item.interpretation,
        metric_ids=list(item.metric_ids),
    )


def _evidence_winner(
    left: EvidenceSummaryItem | None,
    right: EvidenceSummaryItem | None,
) -> ComparisonWinner:
    if left is None or right is None:
        return "not_comparable"
    left_rank = _grade_rank(left.status)
    right_rank = _grade_rank(right.status)
    if left_rank == right_rank:
        return "tie"
    return "left" if left_rank > right_rank else "right"


def _compare_protocol_compatibility(
    left: CalibrationResult,
    right: CalibrationResult,
) -> EvidenceProtocolCompatibility:
    left_protocols = _evidence_protocol_sides(left)
    right_protocols = _evidence_protocol_sides(right)
    left_by_id = {protocol.protocol_id: protocol for protocol in left_protocols}
    right_by_id = {protocol.protocol_id: protocol for protocol in right_protocols}
    shared_ids = sorted(set(left_by_id) & set(right_by_id))
    only_left_ids = sorted(set(left_by_id) - set(right_by_id))
    only_right_ids = sorted(set(right_by_id) - set(left_by_id))
    reasons: list[str] = []
    if not shared_ids:
        if left_protocols or right_protocols:
            reasons.append("no shared evidence protocol IDs")
        else:
            reasons.append("neither result declares an evidence protocol")
        status: ProtocolCompatibilityStatus = "not_comparable"
    else:
        if only_left_ids:
            reasons.append(f"protocols only on left: {', '.join(only_left_ids)}")
        if only_right_ids:
            reasons.append(f"protocols only on right: {', '.join(only_right_ids)}")
        for protocol_id in shared_ids:
            left_protocol = left_by_id[protocol_id]
            right_protocol = right_by_id[protocol_id]
            reasons.extend(_protocol_mismatch_reasons(protocol_id, left_protocol, right_protocol))
        status = "compatible" if not reasons else "warning"
    return EvidenceProtocolCompatibility(
        status=status,
        reasons=reasons,
        shared_protocol_ids=shared_ids,
        only_left_protocol_ids=only_left_ids,
        only_right_protocol_ids=only_right_ids,
        left_protocols=left_protocols,
        right_protocols=right_protocols,
    )


def _evidence_protocol_sides(result: CalibrationResult) -> list[EvidenceProtocolSide]:
    provenance = result.run.provenance
    protocols: list[EvidenceProtocolSide] = []
    livox_pair = provenance.get("livox_pair_evidence")
    if isinstance(livox_pair, dict):
        holdout_geometry = livox_pair.get("holdout_geometry")
        holdout = holdout_geometry if isinstance(holdout_geometry, dict) else {}
        protocols.append(
            EvidenceProtocolSide(
                protocol_id="livox_pair_single_pair_holdout_point_to_plane/v0.1",
                family="lidar_pair",
                metrics_origin=_metrics_origin(provenance),
                data_verified=_bool_or_none(provenance.get("data_verified")),
                split_policy=_str_or_none(holdout.get("split_policy")),
                independent_holdout=_bool_or_none(holdout.get("independent_holdout")),
                known_bad_case_count=_int_or_none(livox_pair.get("known_bad_case_count")),
            )
        )
    return protocols


def _protocol_mismatch_reasons(
    protocol_id: str,
    left: EvidenceProtocolSide,
    right: EvidenceProtocolSide,
) -> list[str]:
    reasons: list[str] = []
    if left.metrics_origin != right.metrics_origin:
        reasons.append(
            f"{protocol_id}: metrics origin differs "
            f"({left.metrics_origin} vs {right.metrics_origin})"
        )
    if left.data_verified != right.data_verified:
        reasons.append(
            f"{protocol_id}: data verification differs "
            f"({left.data_verified} vs {right.data_verified})"
        )
    if left.split_policy != right.split_policy:
        reasons.append(
            f"{protocol_id}: split policy differs "
            f"({left.split_policy} vs {right.split_policy})"
        )
    if left.independent_holdout != right.independent_holdout:
        reasons.append(
            f"{protocol_id}: independent holdout differs "
            f"({left.independent_holdout} vs {right.independent_holdout})"
        )
    if left.known_bad_case_count != right.known_bad_case_count:
        reasons.append(
            f"{protocol_id}: known-bad case count differs "
            f"({left.known_bad_case_count} vs {right.known_bad_case_count})"
        )
    return reasons


def _metrics_origin(provenance: dict[str, object]) -> MetricsOrigin:
    value = provenance.get("metrics_origin", "recomputed")
    if value == "recomputed":
        return "recomputed"
    if value == "cached":
        return "cached"
    if value == "unknown":
        return "unknown"
    return "unknown"


def _str_or_none(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _bool_or_none(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None


def _grade_rank(grade: Grade) -> int:
    if grade == "pass":
        return 2
    if grade == "warn":
        return 1
    return 0


def _comparison_summary(
    metric_comparisons: dict[str, MetricComparison],
    transform_groups: dict[str, TransformGroupComparison],
) -> ComparisonSummary:
    transform_comparison_count = sum(group.comparison_count for group in transform_groups.values())
    max_translation_delta = _max_or_none(
        group.max_translation_delta_m
        for group in transform_groups.values()
        if group.max_translation_delta_m is not None
    )
    max_rotation_delta = _max_or_none(
        group.max_rotation_delta_deg
        for group in transform_groups.values()
        if group.max_rotation_delta_deg is not None
    )
    winners = [comparison.winner for comparison in metric_comparisons.values()]
    return ComparisonSummary(
        metric_comparison_count=len(metric_comparisons),
        left_better_metric_count=winners.count("left"),
        right_better_metric_count=winners.count("right"),
        tied_metric_count=winners.count("tie"),
        not_comparable_metric_count=winners.count("not_comparable"),
        transform_comparison_count=transform_comparison_count,
        max_translation_delta_m=max_translation_delta,
        max_rotation_delta_deg=max_rotation_delta,
    )


def _max_or_none(values: Iterable[float | None]) -> float | None:
    data = [float(value) for value in values if value is not None]
    if not data:
        return None
    return max(data)
