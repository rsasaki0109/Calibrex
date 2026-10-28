"""Independent-recording uncertainty coverage, retaining every failed attempt.

This measures marginal Gaussian intervals against synthetic truth. It does not
calibrate intervals on evaluation data or mistake resamples of one recording
for independent experiments. A bag producer must rerun its complete front end.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Literal

from pydantic import Field, model_validator

from calibrex.core.result import StrictModel


class CoverageAxisObservation(StrictModel):
    """Error in the same tangent coordinates and units as the reported std."""

    axis: str
    error: float = Field(allow_inf_nan=False)
    std: float = Field(ge=0, allow_inf_nan=False)


class CoverageTrial(StrictModel):
    """One independently generated recording; failed fits remain in the cohort."""

    seed: int
    fit_status: str
    policy_status: str
    failure: str | None = None
    artifact_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    axes: list[CoverageAxisObservation] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_axes(self) -> CoverageTrial:
        """Reject duplicate measurements and ambiguous failed attempts."""
        names = [axis.axis for axis in self.axes]
        if len(set(names)) != len(names) or (self.failure is not None and self.axes):
            raise ValueError("axes must be unique; failed attempts cannot report intervals")
        return self


class CoverageAxisSummary(StrictModel):
    """Both conditional coverage and all-attempt interval delivery are reported."""

    axis: str
    attempts: int = Field(ge=1)
    available: int = Field(ge=0)
    covered: int = Field(ge=0)
    conditional_coverage: float | None
    all_attempt_coverage: float = Field(ge=0, le=1)
    wilson_95: tuple[float, float] | None
    mean_error: float | None
    rms_error: float | None
    mean_std: float | None
    shrunken_covered: int = Field(ge=0)
    verdict: Literal["undercoverage", "inconclusive"]


class UncertaintyCoverageReport(StrictModel):
    """A reproducible development measurement, never an accuracy certificate."""

    schema_version: Literal["slac.uncertainty_coverage/v0.1"] = "slac.uncertainty_coverage/v0.1"
    nominal_coverage: float = Field(gt=0, lt=1)
    control_std_scale: float = Field(gt=0, lt=1)
    producer: Literal["known_planes", "synthetic_bag"]
    source_sha256: dict[str, str]
    command: list[str]
    git_commit: str
    trials: list[CoverageTrial]
    axes: list[CoverageAxisSummary]
    limitations: list[str]


def summarize_coverage(
    trials: list[CoverageTrial],
    expected_axes: list[str],
    *,
    nominal: float = 0.95,
    control_std_scale: float = 0.1,
) -> list[CoverageAxisSummary]:
    """Compute per-axis coverage with Wilson uncertainty; never drop failed draws.

    Wilson bounds describe coverage conditional on an available interval. The
    all-attempt fraction additionally counts missing intervals as undelivered.
    Undercoverage is declared only if the upper Wilson bound is below nominal.
    Other outcomes remain inconclusive: absence of a rejection is not proof.
    """
    if not trials or len({t.seed for t in trials}) != len(trials):
        raise ValueError("coverage requires nonempty independent, unique seeds")
    if not expected_axes or len(set(expected_axes)) != len(expected_axes):
        raise ValueError("expected axes must be nonempty and unique")
    if not 0 < nominal < 1 or not 0 < control_std_scale < 1:
        raise ValueError("coverage and control scale must be in (0, 1)")
    if any(a.axis not in expected_axes for t in trials for a in t.axes):
        raise ValueError("unexpected axis")
    z = NormalDist().inv_cdf((1 + nominal) / 2)
    summaries = []
    for name in expected_axes:
        rows = [a for t in trials for a in t.axes if a.axis == name]
        n = len(rows)
        covered = sum(abs(a.error) <= z * a.std for a in rows)
        control = sum(abs(a.error) <= z * a.std * control_std_scale for a in rows)
        interval = _wilson(covered, n) if n else None
        summaries.append(
            CoverageAxisSummary(
                axis=name,
                attempts=len(trials),
                available=n,
                covered=covered,
                conditional_coverage=covered / n if n else None,
                all_attempt_coverage=covered / len(trials),
                wilson_95=interval,
                mean_error=sum(a.error for a in rows) / n if n else None,
                rms_error=math.sqrt(sum(a.error**2 for a in rows) / n) if n else None,
                mean_std=sum(a.std for a in rows) / n if n else None,
                shrunken_covered=control,
                verdict="undercoverage" if interval and interval[1] < nominal else "inconclusive",
            )
        )
    return summaries


def _wilson(k: int, n: int) -> tuple[float, float]:
    z = NormalDist().inv_cdf(0.975)
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def uncertainty_coverage_json_schema() -> dict[str, Any]:
    """Return the registered schema for independent-recording coverage evidence."""
    return UncertaintyCoverageReport.model_json_schema()
