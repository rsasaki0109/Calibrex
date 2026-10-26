"""Frozen marginal error intervals and disjoint evaluation evidence."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from calibrex.core.io import write_mapping
from calibrex.core.result import StrictModel


class EmpiricalIntervalAxis(StrictModel):
    """Absolute-error order statistic; unavailable calibration draws count as infinity."""

    axis: str
    available: int = Field(ge=0)
    order_rank: int = Field(ge=1)
    half_width: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    status: Literal["finite", "inconclusive"]

    @model_validator(mode="after")
    def check_interval(self) -> EmpiricalIntervalAxis:
        """An unbounded order statistic must never become a finite interval."""
        finite = self.order_rank <= self.available
        if (self.status == "finite") != finite or (self.half_width is not None) != finite:
            raise ValueError("finite intervals require an available order statistic")
        return self


class EmpiricalIntervalModel(StrictModel):
    """A frozen split-conformal marginal radius, not a corrected covariance matrix.

    At nominal p and n calibration attempts, the order rank is ceil((n+1)*p).
    Exchangeability of complete-pipeline calibration and future errors is an
    assumption, not established by this artifact or by matching scope hashes.
    """

    schema_version: Literal["slac.empirical_interval_model/v0.1"] = (
        "slac.empirical_interval_model/v0.1"
    )
    method: Literal["absolute_error_order_statistic/v0.1"] = "absolute_error_order_statistic/v0.1"
    nominal_coverage: float = Field(gt=0, lt=1, allow_inf_nan=False)
    scope_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_seeds: list[int] = Field(min_length=1)
    calibration_artifact_sha256: list[str]
    intervals: list[EmpiricalIntervalAxis] = Field(min_length=1)
    command: list[str]
    source_sha256: dict[str, str]
    git_commit: str
    limitations: list[str]

    @model_validator(mode="after")
    def check_cohort(self) -> EmpiricalIntervalModel:
        """Keep count, rank and independent calibration identity internally consistent."""
        n = len(self.calibration_seeds)
        rank = math.ceil((n + 1) * self.nominal_coverage)
        if len(set(self.calibration_seeds)) != n:
            raise ValueError("duplicate calibration seed")
        if len({a.axis for a in self.intervals}) != len(self.intervals):
            raise ValueError("duplicate interval axis")
        if any(a.order_rank != rank or a.available > n for a in self.intervals):
            raise ValueError("interval rank/count does not match calibration cohort")
        hashes = self.calibration_artifact_sha256
        if len(set(hashes)) != len(hashes) or len(hashes) > n:
            raise ValueError("calibration artifact identities must be unique")
        if any(len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in hashes):
            raise ValueError("invalid calibration artifact digest")
        return self

    def save(self, path: str | Path) -> None:
        """Save a schema-valid frozen interval model."""
        write_mapping(Path(path), self.model_dump(mode="json"))


class EmpiricalIntervalCoverageAxis(StrictModel):
    """Paired marginal interval coverage, retaining unavailable fits in attempts."""

    axis: str
    attempts: int = Field(ge=1)
    available: int = Field(ge=0)
    covered: int = Field(ge=0)
    formal_covered: int = Field(ge=0)
    shrunken_covered: int = Field(ge=0)
    half_width: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    mean_formal_half_width: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    conditional_coverage: float | None = Field(default=None, ge=0, le=1)
    all_attempt_coverage: float = Field(ge=0, le=1)
    wilson_95: tuple[float, float] | None
    verdict: Literal["undercoverage", "inconclusive"]

    @model_validator(mode="after")
    def check_counts(self) -> EmpiricalIntervalCoverageAxis:
        """Reject coverage summaries that omit failures or invent intervals."""
        if not (self.shrunken_covered <= self.covered <= self.available <= self.attempts):
            raise ValueError("inconsistent empirical coverage counts")
        if self.formal_covered > self.attempts:
            raise ValueError("formal coverage exceeds attempts")
        if self.half_width is None and self.available:
            raise ValueError("an unavailable interval cannot cover a recording")
        expected = self.covered / self.available if self.available else None
        if self.conditional_coverage != expected:
            raise ValueError("conditional coverage does not match counts")
        if self.all_attempt_coverage != self.covered / self.attempts:
            raise ValueError("all-attempt coverage does not match counts")
        if (self.wilson_95 is None) != (self.available == 0):
            raise ValueError("Wilson interval requires available observations")
        if self.wilson_95 is not None and not 0 <= self.wilson_95[0] <= self.wilson_95[1] <= 1:
            raise ValueError("invalid Wilson interval")
        return self


class EmpiricalIntervalEvaluation(StrictModel):
    """A separate cohort evaluated using previously frozen intervals."""

    schema_version: Literal["slac.empirical_interval_evaluation/v0.1"] = (
        "slac.empirical_interval_evaluation/v0.1"
    )
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scope_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_seeds: list[int] = Field(min_length=1)
    nominal_coverage: float = Field(gt=0, lt=1)
    control_width_scale: float = Field(gt=0, lt=1)
    axes: list[EmpiricalIntervalCoverageAxis]
    command: list[str]
    source_sha256: dict[str, str]
    git_commit: str
    limitations: list[str]

    @model_validator(mode="after")
    def check_cohort(self) -> EmpiricalIntervalEvaluation:
        """Require one retained count per unique independent recording and axis."""
        if len(set(self.evaluation_seeds)) != len(self.evaluation_seeds):
            raise ValueError("duplicate evaluation seed")
        if not self.axes or len({a.axis for a in self.axes}) != len(self.axes):
            raise ValueError("unique nonempty evaluation axes are required")
        if any(a.attempts != len(self.evaluation_seeds) for a in self.axes):
            raise ValueError("axis attempts do not match the evaluation cohort")
        return self

    def save(self, path: str | Path) -> None:
        """Save coverage without promoting it to a quality certificate."""
        write_mapping(Path(path), self.model_dump(mode="json"))


def empirical_interval_model_json_schema() -> dict[str, Any]:
    """Return the interval model JSON schema."""
    return EmpiricalIntervalModel.model_json_schema()


def empirical_interval_evaluation_json_schema() -> dict[str, Any]:
    """Return the interval evaluation JSON schema."""
    return EmpiricalIntervalEvaluation.model_json_schema()
