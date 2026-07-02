"""N-way report comparison assembling reference, candidate, and output results.

ADR 0004 (P1) calls for "common report comparison for dataset calibration,
perturbed candidates, Koide-style adapter output, and native Calibrex output"
in one artifact. `calibrex.evaluation.compare` only compares two results at a
time. This module assembles an arbitrary number of labeled results into one
machine-readable report by running the existing pairwise machinery
(`compare_results`) across every relevant pair and rolling the results up into
a cross-result ranking table per metric family, without reimplementing metric
preference, winner, or protocol-compatibility logic.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import Field

from calibrex.core.result import (
    CalibrationResult,
    EstimateEvidenceLevel,
    EstimateExecutionMode,
    EstimateProducer,
    EstimateRole,
    StrictModel,
    TransformResult,
)
from calibrex.evaluation.compare import (
    ComparisonSide,
    ProtocolCompatibilityStatus,
    ResultComparison,
    compare_results,
    comparison_side,
)

REPORT_COMPARISON_SCHEMA_VERSION: Literal["calibrex.report_comparison/v0.1"] = (
    "calibrex.report_comparison/v0.1"
)

ReportComparisonMode = Literal["all_pairs", "reference_vs_each"]
ProvenanceGroup = Literal[
    "transforms",
    "candidate_extrinsics",
    "reference_extrinsics",
    "none",
]


class ReportComparisonEntryProvenance(StrictModel):
    """Provenance summary for one labeled report-compare entry.

    Derived from `TransformEstimateProvenance` on the entry's transforms
    (preferring `transforms`, then `candidate_extrinsics`, then
    `reference_extrinsics`) so reference vs candidate vs external baseline vs
    native output is explicit rather than inferred from the label string.
    """

    provenance_group: ProvenanceGroup = "none"
    producers: list[EstimateProducer] = Field(default_factory=list)
    execution_modes: list[EstimateExecutionMode] = Field(default_factory=list)
    roles: list[EstimateRole] = Field(default_factory=list)
    evidence_levels: list[EstimateEvidenceLevel] = Field(default_factory=list)
    dominant_producer: EstimateProducer = "unknown"
    dominant_role: EstimateRole | None = None
    dominant_evidence_level: EstimateEvidenceLevel = "unknown"


class ReportComparisonEntry(StrictModel):
    """One labeled input to an N-way report comparison."""

    label: str
    path: str | None = None
    is_reference: bool = False
    side: ComparisonSide
    provenance: ReportComparisonEntryProvenance


class ReportComparisonPair(StrictModel):
    """One pairwise comparison computed with the existing compare machinery."""

    left_label: str
    right_label: str
    comparison: ResultComparison


class MetricFamilyRankingEntry(StrictModel):
    """One labeled entry's rollup within a metric family ranking."""

    label: str
    win_count: int = Field(default=0, ge=0)
    tie_count: int = Field(default=0, ge=0)
    loss_count: int = Field(default=0, ge=0)
    not_comparable_count: int = Field(default=0, ge=0)
    comparison_count: int = Field(default=0, ge=0)
    rank: int | None = None


class MetricFamilyRanking(StrictModel):
    """Cross-result ranking table for one metric family.

    Built entirely from `MetricFamilyComparison` rollups already produced by
    `compare_results` for each pair; no metric preference or winner logic is
    reimplemented here.
    """

    family: str
    metric_count: int = Field(default=0, ge=0)
    entries: list[MetricFamilyRankingEntry] = Field(default_factory=list)


class ReportComparisonSummary(StrictModel):
    """High-level counters for an N-way report comparison."""

    entry_count: int = Field(ge=2)
    comparison_mode: ReportComparisonMode
    reference_label: str | None = None
    pairwise_comparison_count: int = Field(ge=0)
    protocol_compatibility_status: ProtocolCompatibilityStatus
    compatible_pair_count: int = Field(default=0, ge=0)
    warning_pair_count: int = Field(default=0, ge=0)
    not_comparable_pair_count: int = Field(default=0, ge=0)
    not_comparable_pairs: list[str] = Field(default_factory=list)


class ReportComparison(StrictModel):
    """Machine-readable N-way comparison across labeled Calibrex results."""

    schema_version: Literal["calibrex.report_comparison/v0.1"] = (
        REPORT_COMPARISON_SCHEMA_VERSION
    )
    summary: ReportComparisonSummary
    entries: dict[str, ReportComparisonEntry]
    pairwise: dict[str, ReportComparisonPair] = Field(default_factory=dict)
    metric_family_rankings: dict[str, MetricFamilyRanking] = Field(default_factory=dict)


def compare_reports(
    labeled_results: list[tuple[str, CalibrationResult]],
    *,
    paths: dict[str, str | Path | None] | None = None,
    reference_label: str | None = None,
) -> ReportComparison:
    """Assemble an N-way report comparison from >=2 labeled results.

    Reuses `compare_results` for every compared pair so metric preference,
    winner, and evidence-protocol compatibility semantics stay identical to
    the two-way `calibrex compare` command. When `reference_label` is given,
    only reference-vs-each pairs are computed (mirroring `--enforce-compatible`
    against one baseline); otherwise every unordered pair is compared.
    """

    if len(labeled_results) < 2:
        msg = "report comparison requires at least two labeled results"
        raise ValueError(msg)
    labels = [label for label, _ in labeled_results]
    if len(set(labels)) != len(labels):
        msg = f"duplicate report-compare labels: {labels}"
        raise ValueError(msg)
    if reference_label is not None and reference_label not in labels:
        msg = f"reference label not found among entries: {reference_label}"
        raise ValueError(msg)

    results_by_label = dict(labeled_results)
    path_by_label = dict(paths or {})

    entries = {
        label: ReportComparisonEntry(
            label=label,
            path=_str_or_none(path_by_label.get(label)),
            is_reference=(label == reference_label),
            side=comparison_side(result, path_by_label.get(label)),
            provenance=_entry_provenance(result),
        )
        for label, result in labeled_results
    }

    pairs, mode = _comparison_pairs(labels, reference_label=reference_label)
    pairwise: dict[str, ReportComparisonPair] = {}
    for left_label, right_label in pairs:
        comparison = compare_results(
            results_by_label[left_label],
            results_by_label[right_label],
            left_path=path_by_label.get(left_label),
            right_path=path_by_label.get(right_label),
        )
        pairwise[_pair_key(left_label, right_label)] = ReportComparisonPair(
            left_label=left_label,
            right_label=right_label,
            comparison=comparison,
        )

    summary = _report_comparison_summary(
        labels,
        pairwise,
        mode=mode,
        reference_label=reference_label,
    )

    return ReportComparison(
        summary=summary,
        entries=entries,
        pairwise=pairwise,
        metric_family_rankings=_metric_family_rankings(labels, pairwise),
    )


def report_comparison_json_schema() -> dict[str, object]:
    """Return the JSON schema for N-way report-comparison artifacts."""

    return ReportComparison.model_json_schema()


def _comparison_pairs(
    labels: list[str],
    *,
    reference_label: str | None,
) -> tuple[list[tuple[str, str]], ReportComparisonMode]:
    if reference_label is not None:
        pairs = [
            (reference_label, other_label)
            for other_label in labels
            if other_label != reference_label
        ]
        return pairs, "reference_vs_each"
    pairs = [
        (labels[left_index], labels[right_index])
        for left_index in range(len(labels))
        for right_index in range(left_index + 1, len(labels))
    ]
    return pairs, "all_pairs"


def _pair_key(left_label: str, right_label: str) -> str:
    return f"{left_label}__{right_label}"


def _report_comparison_summary(
    labels: list[str],
    pairwise: dict[str, ReportComparisonPair],
    *,
    mode: ReportComparisonMode,
    reference_label: str | None,
) -> ReportComparisonSummary:
    statuses = [pair.comparison.protocol_compatibility.status for pair in pairwise.values()]
    compatible_count = statuses.count("compatible")
    warning_count = statuses.count("warning")
    not_comparable_count = statuses.count("not_comparable")
    if not_comparable_count:
        overall: ProtocolCompatibilityStatus = "not_comparable"
    elif warning_count:
        overall = "warning"
    else:
        overall = "compatible"
    not_comparable_pairs = sorted(
        key
        for key, pair in pairwise.items()
        if pair.comparison.protocol_compatibility.status == "not_comparable"
    )
    return ReportComparisonSummary(
        entry_count=len(labels),
        comparison_mode=mode,
        reference_label=reference_label,
        pairwise_comparison_count=len(pairwise),
        protocol_compatibility_status=overall,
        compatible_pair_count=compatible_count,
        warning_pair_count=warning_count,
        not_comparable_pair_count=not_comparable_count,
        not_comparable_pairs=not_comparable_pairs,
    )


def _metric_family_rankings(
    labels: list[str],
    pairwise: dict[str, ReportComparisonPair],
) -> dict[str, MetricFamilyRanking]:
    family_metric_names: dict[str, set[str]] = {}
    counters: dict[str, dict[str, Counter[str]]] = {}
    for pair in pairwise.values():
        for family, family_comparison in pair.comparison.metric_families.items():
            family_metric_names.setdefault(family, set()).update(family_comparison.metrics)
            family_counters = counters.setdefault(family, {})
            left_counter = family_counters.setdefault(pair.left_label, Counter())
            right_counter = family_counters.setdefault(pair.right_label, Counter())
            left_counter["win"] += family_comparison.left_better_count
            left_counter["loss"] += family_comparison.right_better_count
            left_counter["tie"] += family_comparison.tied_count
            left_counter["not_comparable"] += family_comparison.not_comparable_count
            left_counter["comparisons"] += family_comparison.metric_count
            right_counter["win"] += family_comparison.right_better_count
            right_counter["loss"] += family_comparison.left_better_count
            right_counter["tie"] += family_comparison.tied_count
            right_counter["not_comparable"] += family_comparison.not_comparable_count
            right_counter["comparisons"] += family_comparison.metric_count

    rankings: dict[str, MetricFamilyRanking] = {}
    for family, metric_names in family_metric_names.items():
        family_counters = counters.get(family, {})
        entries = [
            MetricFamilyRankingEntry(
                label=label,
                win_count=family_counters.get(label, Counter())["win"],
                tie_count=family_counters.get(label, Counter())["tie"],
                loss_count=family_counters.get(label, Counter())["loss"],
                not_comparable_count=family_counters.get(label, Counter())["not_comparable"],
                comparison_count=family_counters.get(label, Counter())["comparisons"],
            )
            for label in labels
        ]
        entries.sort(
            key=lambda entry: (
                entry.comparison_count == 0,
                -entry.win_count,
                entry.loss_count,
                entry.label,
            )
        )
        rank = 0
        for entry in entries:
            if entry.comparison_count == 0:
                entry.rank = None
                continue
            rank += 1
            entry.rank = rank
        rankings[family] = MetricFamilyRanking(
            family=family,
            metric_count=len(metric_names),
            entries=entries,
        )
    return dict(sorted(rankings.items()))


def _entry_provenance(result: CalibrationResult) -> ReportComparisonEntryProvenance:
    groups: tuple[tuple[ProvenanceGroup, dict[str, TransformResult]], ...] = (
        ("transforms", result.transforms),
        ("candidate_extrinsics", result.candidate_extrinsics),
        ("reference_extrinsics", result.reference_extrinsics),
    )
    unknown_producer: EstimateProducer = "unknown"
    unknown_evidence_level: EstimateEvidenceLevel = "unknown"
    for group_name, group in groups:
        if not group:
            continue
        provenances = [transform.provenance for transform in group.values()]
        producers = [provenance.producer for provenance in provenances]
        execution_modes = [provenance.execution_mode for provenance in provenances]
        roles = [
            provenance.role_in_comparison
            for provenance in provenances
            if provenance.role_in_comparison is not None
        ]
        evidence_levels = [provenance.evidence_level for provenance in provenances]
        return ReportComparisonEntryProvenance(
            provenance_group=group_name,
            producers=sorted(set(producers)),
            execution_modes=sorted(set(execution_modes)),
            roles=sorted(set(roles)),
            evidence_levels=sorted(set(evidence_levels)),
            dominant_producer=_mode_value(producers, default=unknown_producer),
            dominant_role=_mode_value(roles, default=None),
            dominant_evidence_level=_mode_value(
                evidence_levels,
                default=unknown_evidence_level,
            ),
        )
    return ReportComparisonEntryProvenance()


_StrValueT = TypeVar("_StrValueT", bound=str)
_DefaultT = TypeVar("_DefaultT")


def _mode_value(
    values: list[_StrValueT],
    *,
    default: _DefaultT,
) -> _StrValueT | _DefaultT:
    if not values:
        return default
    counts = Counter(values)
    max_count = max(counts.values())
    candidates = sorted(value for value, count in counts.items() if count == max_count)
    return candidates[0]


def _str_or_none(value: str | Path | None) -> str | None:
    if value is None:
        return None
    return str(value)
