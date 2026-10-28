"""Coverage uses independent trials, known probabilities and retained failures."""

from __future__ import annotations

from statistics import NormalDist

import pytest
from pydantic import ValidationError

from calibrex.evaluation.uncertainty_coverage import (
    CoverageAxisObservation,
    CoverageTrial,
    summarize_coverage,
)


def _trials(scale: float = 1.0) -> list[CoverageTrial]:
    # Deterministic equal-probability normal quantiles: exactly 190/200 covered.
    return [
        CoverageTrial(
            seed=i,
            fit_status="converged",
            policy_status="pass",
            axes=[
                CoverageAxisObservation(
                    axis="x", error=NormalDist().inv_cdf((i + 0.5) / 200), std=scale
                )
            ],
        )
        for i in range(200)
    ]


def test_known_normal_distribution_and_overconfidence_control() -> None:
    summary = summarize_coverage(_trials(), ["x"])[0]
    assert summary.covered == 190
    assert summary.wilson_95[0] < 0.95 < summary.wilson_95[1]
    assert summary.verdict == "inconclusive"
    assert summary.shrunken_covered < 40
    overconfident = summarize_coverage(_trials(0.1), ["x"])[0]
    assert overconfident.verdict == "undercoverage"


def test_failed_and_missing_axes_remain_in_denominator() -> None:
    trials = _trials()
    trials.append(
        CoverageTrial(
            seed=200,
            fit_status="exception",
            policy_status="failed",
            failure="failed map reconstruction",
        )
    )
    summary, missing = summarize_coverage(trials, ["x", "y"])
    assert summary.available == 200 and summary.attempts == 201
    assert summary.conditional_coverage == 0.95
    assert summary.all_attempt_coverage == 190 / 201
    assert missing.available == 0 and missing.conditional_coverage is None
    assert missing.all_attempt_coverage == 0 and missing.verdict == "inconclusive"


def test_stalled_fits_are_not_dropped() -> None:
    trial = CoverageTrial(
        seed=1,
        fit_status="stalled",
        policy_status="fail",
        axes=[CoverageAxisObservation(axis="x", error=10, std=0)],
    )
    summary = summarize_coverage([trial], ["x"])[0]
    assert summary.available == 1 and summary.covered == 0


def test_duplicate_seeds_and_invalid_observations_rejected() -> None:
    with pytest.raises(ValueError, match="unique seeds"):
        summarize_coverage([_trials()[0], _trials()[0]], ["x"])
    for std in (-1, float("nan"), float("inf")):
        with pytest.raises(ValidationError):
            CoverageAxisObservation(axis="x", error=0, std=std)
