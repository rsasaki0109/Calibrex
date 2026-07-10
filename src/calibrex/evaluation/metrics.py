"""Metric aggregation and PASS/WARN/FAIL quality gates."""

from __future__ import annotations

from typing import cast

from calibrex.core.result import CalibrationResult, Grade, MetricResult, QualitySummary
from calibrex.evaluation.evidence_summary import evidence_decision_grade_from_result
from calibrex.evaluation.recommendations import build_recommendations
from calibrex.evaluation.thresholds import (
    ThresholdProfile,
    apply_metric_thresholds,
    profile_from_domain,
)

GRADE_ORDER = {"pass": 0, "warn": 1, "fail": 2}


def worst_grade(grades: list[str]) -> str:
    """Return the worst grade from a list of grade strings."""

    if not grades:
        return "warn"
    return max(grades, key=lambda grade: GRADE_ORDER.get(grade, 1))


def evaluate_quality(
    result: CalibrationResult,
    strict: bool = False,
    threshold_profile: ThresholdProfile | None = None,
) -> CalibrationResult:
    """Aggregate metric, transform, timing, and degeneracy grades."""

    profile = threshold_profile or profile_from_domain(result.run.domain)
    apply_metric_thresholds(result.metrics, profile)

    grades: list[str] = [metric.grade for metric in result.metrics.values()]
    grades.extend(transform.quality.grade for transform in result.transforms.values())
    grades.extend(offset.quality.grade for offset in result.time_offsets.values())
    grades.append(result.observability.grade)
    grades.append(result.degeneracy.grade)
    evidence_grade = evidence_decision_grade_from_result(result)
    if evidence_grade is not None:
        grades.append(evidence_grade)

    warnings: list[str] = []
    failures: list[str] = []

    for name, metric in sorted(result.metrics.items()):
        _collect_metric_issue(name, metric, warnings, failures)
    for name, transform in sorted(result.transforms.items()):
        if transform.quality.grade == "warn":
            warnings.append(f"{name}: transform uncertainty is provisional")
        elif transform.quality.grade == "fail":
            failures.append(f"{name}: transform quality failed")
    for name, offset in sorted(result.time_offsets.items()):
        if offset.quality.grade == "warn":
            warnings.append(f"{name}: time offset uncertainty is provisional")
        elif offset.quality.grade == "fail":
            failures.append(f"{name}: time offset quality failed")
    if result.degeneracy.grade == "warn" and result.degeneracy.reason:
        warnings.append(result.degeneracy.reason)
    elif result.degeneracy.grade == "fail" and result.degeneracy.reason:
        failures.append(result.degeneracy.reason)
    if result.observability.grade == "warn":
        warnings.append("observability is not fully established")
    elif result.observability.grade == "fail":
        failures.append("observability check failed")
    if evidence_grade == "warn":
        warnings.append("evidence decision boundary is inconclusive under declared protocols")
    elif evidence_grade == "fail":
        failures.append("evidence decision boundary rejected the candidate")

    grade = worst_grade(grades)
    if strict and grade == "warn":
        grade = "fail"
        failures.extend(warnings)
        warnings = []

    result.quality = QualitySummary(
        grade=cast(Grade, grade),
        blocking_failures=dedupe(failures),
        warnings=dedupe(warnings),
        recommendation=build_recommendations(result),
    )
    return result


def dedupe(values: list[str]) -> list[str]:
    """Deduplicate while preserving order."""

    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output


def _collect_metric_issue(
    name: str,
    metric: MetricResult,
    warnings: list[str],
    failures: list[str],
) -> None:
    reason = metric.reason or f"{name} grade is {metric.grade}"
    if metric.grade == "warn":
        warnings.append(reason)
    elif metric.grade == "fail":
        failures.append(reason)

