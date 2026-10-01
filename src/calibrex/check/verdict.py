"""The verdict rule of ``calibrex check``: candidate transform against an estimator.

Pure functions, no I/O. An estimator contributes per-axis estimates
(:class:`AxisEstimate`) and the candidate contributes the per-axis error
relative to them; this module turns those into per-axis judgements, a pair
verdict, and the overall verdict.

Rule (per judged axis ``i``, with ``delta_i`` = candidate minus estimate):

* ``tolerance_i = max(sigma_k * std_i, floor)`` with the rotation floor for
  rotation axes and the translation floor for translation axes;
* axis status: ``pass`` if ``|delta_i| <= tolerance_i``, ``fail`` if
  ``|delta_i| > 2 * tolerance_i``, otherwise ``warn``;
* pair verdict: ``fail`` if any axis fails, else ``warn`` if any axis warns,
  else ``pass``; ``inconclusive`` when no axis could be judged.

Only axes the estimator reports as estimated are judged; the others are
listed as unchecked.

Detection power (no re-solve): a deliberate error ``e`` on axis ``i`` makes
the error ``delta_i + e`` (either sign). It is flagged when that exceeds the
tolerance. Both signs are flagged exactly when ``e > tolerance_i + |delta_i|``,
which is recorded as ``detectable_error``; for rotation axes
``detects_perturbation`` says whether a probe of ``detection_probe_deg``
(default 1 deg) clears that bound.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from calibrex.core.calibration_check import (
    CheckAxisJudgement,
    CheckAxisName,
    CheckAxisStatus,
    CheckUncheckedAxis,
)

ROTATION_AXES: tuple[CheckAxisName, ...] = ("roll", "pitch", "yaw")
VERDICT_ORDER: tuple[str, ...] = ("pass", "inconclusive", "warn", "fail")
FAIL_FACTOR = 2.0


@dataclass(frozen=True)
class VerdictOptions:
    """Thresholds of the verdict rule."""

    sigma_k: float = 3.0
    rotation_floor_deg: float = 0.5
    translation_floor_m: float = 0.02
    detection_probe_deg: float = 1.0

    def __post_init__(self) -> None:
        if (
            min(
                self.sigma_k,
                self.rotation_floor_deg,
                self.translation_floor_m,
                self.detection_probe_deg,
            )
            <= 0.0
        ):
            raise ValueError("sigma_k, both floors and the detection probe must be positive")


@dataclass(frozen=True)
class AxisEstimate:
    """One axis as reported by an estimator, with the candidate's error on it.

    ``candidate_error`` is candidate minus estimate in ``unit`` (degrees for
    rotation axes, metres for translation axes), in the convention the
    estimator uses for ``std``.
    """

    name: CheckAxisName
    unit: Literal["deg", "m"]
    candidate_error: float
    std: float
    estimated: bool
    unchecked_reason: str | None = None
    unchecked_code: Literal["unobservable", "control_not_detected", "no_estimate"] | None = None


def judge_axis(estimate: AxisEstimate, options: VerdictOptions) -> CheckAxisJudgement:
    """Judge one estimated axis."""

    floor = options.rotation_floor_deg if estimate.unit == "deg" else options.translation_floor_m
    sigma_tolerance = options.sigma_k * estimate.std
    tolerance = max(sigma_tolerance, floor)
    error = abs(estimate.candidate_error)
    status: CheckAxisStatus
    if error <= tolerance:
        status = "pass"
    elif error > FAIL_FACTOR * tolerance:
        status = "fail"
    else:
        status = "warn"
    detectable = tolerance + error
    rotation = estimate.unit == "deg"
    return CheckAxisJudgement(
        name=estimate.name,
        unit=estimate.unit,
        candidate_error=estimate.candidate_error,
        estimate_std=estimate.std,
        tolerance=tolerance,
        tolerance_source="sigma" if sigma_tolerance > floor else "floor",
        ratio=error / tolerance,
        status=status,
        detectable_error=detectable,
        detection_probe=options.detection_probe_deg if rotation else None,
        detects_perturbation=detectable < options.detection_probe_deg if rotation else None,
    )


def worst_verdict(verdicts: Sequence[str]) -> str | None:
    """The worst of ``fail > warn > inconclusive > pass``; ``None`` for an empty list."""

    ranked = [verdict for verdict in verdicts if verdict in VERDICT_ORDER]
    if not ranked:
        return None
    return max(ranked, key=VERDICT_ORDER.index)


@dataclass(frozen=True)
class PairJudgement:
    """Per-axis judgements and the verdict derived from them."""

    verdict: Literal["pass", "warn", "fail", "inconclusive"]
    axes: tuple[CheckAxisJudgement, ...]
    unchecked: tuple[CheckUncheckedAxis, ...]

    @property
    def coverage(self) -> Literal["full", "partial"]:
        """``partial`` when any attempted axis is unchecked."""

        return "partial" if self.unchecked else "full"


def judge_pair(estimates: Sequence[AxisEstimate], options: VerdictOptions) -> PairJudgement:
    """Judge every estimated axis and list the rest as unchecked."""

    axes: list[CheckAxisJudgement] = []
    unchecked: list[CheckUncheckedAxis] = []
    for estimate in estimates:
        if estimate.estimated:
            axes.append(judge_axis(estimate, options))
        else:
            unchecked.append(
                CheckUncheckedAxis(
                    name=estimate.name,
                    unit=estimate.unit,
                    std=estimate.std,
                    reason=estimate.unchecked_reason or "not constrained by the data",
                    reason_code=estimate.unchecked_code or "unobservable",
                )
            )
    if not axes:
        return PairJudgement("inconclusive", (), tuple(unchecked))
    statuses = {axis.status for axis in axes}
    verdict: Literal["pass", "warn", "fail"] = (
        "fail" if "fail" in statuses else "warn" if "warn" in statuses else "pass"
    )
    return PairJudgement(verdict, tuple(axes), tuple(unchecked))
