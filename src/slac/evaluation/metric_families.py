"""Metric family helpers shared by reports and comparisons."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from slac.core.result import MetricResult
from slac.evaluation.registry import get_metric_definition


def metric_family(name: str) -> str:
    """Return the stable reporting family for a metric name."""

    definition = get_metric_definition(name)
    if definition is not None:
        return definition.family
    if name.startswith("extrinsic_reference_delta_"):
        return "extrinsic"
    if name.startswith("lidar_camera_") or name.startswith("koide_lidar_camera_"):
        return "lidar_camera"
    if name.startswith("lidar_"):
        return "lidar"
    if "timestamp" in name or "time_offset" in name:
        return "timing"
    if "observability" in name or "weak_dof" in name or name == "condition_number":
        return "observability"
    if name.startswith("camera_") or name.startswith("reprojection_"):
        return "camera"
    return "common"


def metric_family_payloads(metrics: dict[str, MetricResult]) -> dict[str, Any]:
    """Return report-ready family rollups for a metric mapping."""

    names_by_family: dict[str, list[str]] = {}
    for name in sorted(metrics):
        family = metric_family(name)
        names_by_family.setdefault(family, []).append(name)

    return {
        family: _metric_family_payload(family, names, metrics)
        for family, names in sorted(names_by_family.items())
    }


def grade_counts(grades: Iterable[str]) -> dict[str, int]:
    """Count pass/warn/fail grades."""

    counts = {"pass": 0, "warn": 0, "fail": 0}
    for grade in grades:
        if grade in counts:
            counts[grade] += 1
    return counts


def worst_grade(grades: Iterable[str]) -> str:
    """Return the worst grade in pass < warn < fail order."""

    severity = {"pass": 0, "warn": 1, "fail": 2}
    worst = "pass"
    for grade in grades:
        if severity.get(grade, -1) > severity[worst]:
            worst = grade
    return worst


def _metric_family_payload(
    family: str,
    names: list[str],
    metrics: dict[str, MetricResult],
) -> dict[str, Any]:
    family_metrics = [metrics[name] for name in names]
    return {
        "family": family,
        "metric_count": len(names),
        "grade_counts": grade_counts(metric.grade for metric in family_metrics),
        "worst_grade": worst_grade(metric.grade for metric in family_metrics),
        "holdout_metric_count": sum(1 for metric in family_metrics if metric.holdout is not None),
        "warn_or_fail_metrics": [
            name for name in names if metrics[name].grade in {"warn", "fail"}
        ],
        "metrics": names,
    }
