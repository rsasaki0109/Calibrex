"""Fit error radii on calibration draws and score a disjoint frozen cohort."""

from __future__ import annotations

import hashlib
import json
import math
from statistics import NormalDist
from typing import Any

from calibrex.core.empirical_intervals import (
    EmpiricalIntervalAxis,
    EmpiricalIntervalCoverageAxis,
    EmpiricalIntervalEvaluation,
    EmpiricalIntervalModel,
)
from calibrex.evaluation.uncertainty_coverage import UncertaintyCoverageReport, _wilson

LIMITATIONS = [
    "Marginal per-axis intervals; not simultaneous joint coverage or a covariance correction.",
    "Exchangeability of pipeline errors is assumed, not proved by a matching scope digest.",
    "Independent noise draws on one synthetic motion do not establish transfer to real rigs.",
    "Unavailable calibration draws act as infinite errors; insufficient order statistics abstain.",
    "Returned fits, including stalled fits and failed quality gates, remain in the cohort.",
    "Intervals do not replace observability, holdout, stationarity or deployment policy gates.",
    "Synthetic-bag world gravity is excluded: the local odometry gauge is not aligned to truth.",
]


def eligible_interval_axes(report: UncertaintyCoverageReport) -> list[str]:
    """Exclude bag gravity components whose world frame has no truth alignment.

    Relative mounting, body lever arm, IMU biases and time offset are invariant
    to the odometry world gauge. Gravity components are not. Known-plane runs
    have a declared common world frame and can retain gravity.
    """
    return [
        a.axis
        for a in report.axes
        if report.producer != "synthetic_bag" or not a.axis.startswith("gravity[")
    ]


def interval_scope(report: UncertaintyCoverageReport, protocol: dict[str, Any]) -> str:
    """Bind intervals to producer, options, exact solver source and declared axis set."""
    seeds = [t.seed for t in report.trials]
    if (
        protocol.get("producer") != report.producer
        or protocol.get("seeds") != seeds
        or protocol.get("source_sha256") != report.source_sha256
    ):
        raise ValueError("report and protocol disagree")
    for key in ("duration_s", "map_rounds"):
        if key not in protocol:
            raise ValueError(f"protocol lacks {key}")
    payload = {
        "producer": report.producer,
        "duration_s": protocol["duration_s"],
        "map_rounds": protocol["map_rounds"],
        "source_sha256": report.source_sha256,
        "axes": sorted(eligible_interval_axes(report)),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _check_report(report: UncertaintyCoverageReport) -> None:
    seeds = [t.seed for t in report.trials]
    axes = [a.axis for a in report.axes]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("unique, nonempty recording seeds are required")
    if not axes or len(set(axes)) != len(axes):
        raise ValueError("unique, nonempty axis set is required")
    digests = [t.artifact_sha256 for t in report.trials if t.artifact_sha256 is not None]
    if len(set(digests)) != len(digests):
        raise ValueError("duplicate recording artifact")
    if any(t.axes and not t.artifact_sha256 for t in report.trials):
        raise ValueError("reported errors require an artifact digest")
    if any(a.axis not in axes for t in report.trials for a in t.axes):
        raise ValueError("unexpected trial axis")


def fit_empirical_intervals(
    report: UncertaintyCoverageReport,
    *,
    scope_sha256: str,
    report_sha256: str,
    nominal: float = 0.95,
    command: list[str],
    source_sha256: dict[str, str],
    git_commit: str,
) -> EmpiricalIntervalModel:
    """Freeze finite-sample marginal order statistics using calibration errors only.

    Scores are absolute errors in the reported tangent coordinates. There is no
    bias subtraction fitted on the same scores and no Gaussian assumption.
    Missing axes/failed attempts are infinite scores, never silently dropped.
    """
    _check_report(report)
    if not 0 < nominal < 1:
        raise ValueError("nominal coverage must be in (0, 1)")
    rank = math.ceil((len(report.trials) + 1) * nominal)
    intervals = []
    for name in eligible_interval_axes(report):
        scores = sorted(abs(a.error) for t in report.trials for a in t.axes if a.axis == name)
        finite = rank <= len(scores)
        intervals.append(
            EmpiricalIntervalAxis(
                axis=name,
                available=len(scores),
                order_rank=rank,
                half_width=scores[rank - 1] if finite else None,
                status="finite" if finite else "inconclusive",
            )
        )
    return EmpiricalIntervalModel(
        nominal_coverage=nominal,
        scope_sha256=scope_sha256,
        calibration_report_sha256=report_sha256,
        calibration_seeds=[t.seed for t in report.trials],
        calibration_artifact_sha256=[
            t.artifact_sha256 for t in report.trials if t.artifact_sha256 is not None
        ],
        intervals=intervals,
        command=command,
        source_sha256=source_sha256,
        git_commit=git_commit,
        limitations=LIMITATIONS,
    )


def evaluate_empirical_intervals(
    model: EmpiricalIntervalModel,
    report: UncertaintyCoverageReport,
    *,
    scope_sha256: str,
    model_sha256: str,
    report_sha256: str,
    command: list[str],
    source_sha256: dict[str, str],
    git_commit: str,
    control_width_scale: float = 0.1,
) -> EmpiricalIntervalEvaluation:
    """Score frozen widths; reject leakage, incompatible scope and duplicate identities."""
    _check_report(report)
    if not 0 < control_width_scale < 1:
        raise ValueError("control width scale must be in (0, 1)")
    if model.scope_sha256 != scope_sha256:
        raise ValueError("evaluation scope differs from calibration scope")
    if model.calibration_report_sha256 == report_sha256:
        raise ValueError("calibration report cannot also be evaluation data")
    if set(model.calibration_seeds) & {t.seed for t in report.trials}:
        raise ValueError("calibration/evaluation seed overlap")
    if set(model.calibration_artifact_sha256) & {t.artifact_sha256 for t in report.trials}:
        raise ValueError("calibration/evaluation recording artifact overlap")
    if {a.axis for a in model.intervals} != set(eligible_interval_axes(report)):
        raise ValueError("evaluation axis set differs from calibration")
    z = NormalDist().inv_cdf((1 + model.nominal_coverage) / 2)
    summaries = []
    for interval in model.intervals:
        rows = [a for t in report.trials for a in t.axes if a.axis == interval.axis]
        width = interval.half_width
        n = len(rows) if width is not None else 0
        covered = sum(abs(a.error) <= width for a in rows) if width is not None else 0
        control = (
            sum(abs(a.error) <= width * control_width_scale for a in rows)
            if width is not None
            else 0
        )
        wilson = _wilson(covered, n) if n else None
        summaries.append(
            EmpiricalIntervalCoverageAxis(
                axis=interval.axis,
                attempts=len(report.trials),
                available=n,
                covered=covered,
                formal_covered=sum(abs(a.error) <= z * a.std for a in rows),
                shrunken_covered=control,
                half_width=width,
                mean_formal_half_width=z * sum(a.std for a in rows) / len(rows) if rows else None,
                conditional_coverage=covered / n if n else None,
                all_attempt_coverage=covered / len(report.trials),
                wilson_95=wilson,
                verdict=(
                    "undercoverage"
                    if wilson and wilson[1] < model.nominal_coverage
                    else "inconclusive"
                ),
            )
        )
    return EmpiricalIntervalEvaluation(
        model_sha256=model_sha256,
        evaluation_report_sha256=report_sha256,
        scope_sha256=scope_sha256,
        evaluation_seeds=[t.seed for t in report.trials],
        nominal_coverage=model.nominal_coverage,
        control_width_scale=control_width_scale,
        axes=summaries,
        command=command,
        source_sha256=source_sha256,
        git_commit=git_commit,
        limitations=[
            *LIMITATIONS,
            "Failure to reject undercoverage is not evidence of 95% coverage.",
        ],
    )
