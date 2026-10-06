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
    CheckFocalComponent,
    CheckFocalScale,
    CheckUncheckedAxis,
    CheckUncheckedFocal,
    FocalName,
)
from calibrex.core.excitation import AxisExcitation

ROTATION_AXES: tuple[CheckAxisName, ...] = ("roll", "pitch", "yaw")
VERDICT_ORDER: tuple[str, ...] = ("pass", "inconclusive", "warn", "fail")
FAIL_FACTOR = 2.0
RIGID_SCAN_ROTATION_FLOOR_DEG = 1.5
"""Default rotation floor of imu-lidar on clouds treated as rigid scans (no deskew)."""


FOCAL_SCALE_FLOOR = 0.005
"""Default relative floor of the camera-focal tolerance (0.5 %).

The Hilti 2022 forward cameras (exp21 cam0/cam1) agree with Kalibr's focal
lengths within 0.5 %; a floor below that would flag Kalibr itself. For every
camera the documented jackknife std is 0.2-1.3 %, so ``sigma_k * std`` (about
1.5-4 %) is the binding term and the floor only matters for a very precise run.
"""


@dataclass(frozen=True)
class VerdictOptions:
    """Thresholds of the verdict rule."""

    sigma_k: float = 3.0
    rotation_floor_deg: float = 0.5
    translation_floor_m: float = 0.02
    detection_probe_deg: float = 1.0
    focal_scale_floor: float = FOCAL_SCALE_FLOOR
    focal_detection_probe: float = 0.01

    def __post_init__(self) -> None:
        if (
            min(
                self.sigma_k,
                self.rotation_floor_deg,
                self.translation_floor_m,
                self.detection_probe_deg,
                self.focal_scale_floor,
                self.focal_detection_probe,
            )
            <= 0.0
        ):
            raise ValueError("sigma_k, the floors and the detection probes must be positive")


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
    floor: float | None = None
    """A floor for this axis (in ``unit``) that applies when larger than the option's floor."""
    excitation: AxisExcitation | None = None
    """Translation axes: why the axis is (un)observable on this recording."""


def judge_axis(estimate: AxisEstimate, options: VerdictOptions) -> CheckAxisJudgement:
    """Judge one estimated axis."""

    floor = options.rotation_floor_deg if estimate.unit == "deg" else options.translation_floor_m
    if estimate.floor is not None:
        floor = max(floor, estimate.floor)
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
                    excitation=estimate.excitation,
                )
            )
    if not axes:
        return PairJudgement("inconclusive", (), tuple(unchecked))
    statuses = {axis.status for axis in axes}
    verdict: Literal["pass", "warn", "fail"] = (
        "fail" if "fail" in statuses else "warn" if "warn" in statuses else "pass"
    )
    return PairJudgement(verdict, tuple(axes), tuple(unchecked))


@dataclass(frozen=True)
class FocalEstimate:
    """One focal length as reported by the camera-focal estimator.

    ``scale`` is ``estimate / candidate`` and ``scale_std`` its std.
    """

    name: FocalName
    candidate_px: float
    estimate_px: float
    estimate_std_px: float
    scale: float
    scale_std: float
    estimated: bool
    unchecked_reason: str | None = None
    unchecked_code: Literal["unobservable", "control_not_detected", "no_estimate"] | None = None


@dataclass(frozen=True)
class FocalRecord:
    """The focal-length estimates of one camera-focal run and the estimator's control."""

    components: tuple[FocalEstimate, ...]
    optical_axis_ratio: float | None = None
    optical_axis_ratio_std: float | None = None
    intrinsics_source: str | None = None


def judge_focal_component(estimate: FocalEstimate, options: VerdictOptions) -> CheckFocalComponent:
    """Judge one estimated focal length: ``|scale - 1|`` against ``max(k std, floor)``."""

    sigma_tolerance = options.sigma_k * estimate.scale_std
    tolerance = max(sigma_tolerance, options.focal_scale_floor)
    error = estimate.scale - 1.0
    status: CheckAxisStatus
    if abs(error) <= tolerance:
        status = "pass"
    elif abs(error) > FAIL_FACTOR * tolerance:
        status = "fail"
    else:
        status = "warn"
    detectable = tolerance + abs(error)
    return CheckFocalComponent(
        name=estimate.name,
        candidate_px=estimate.candidate_px,
        estimate_px=estimate.estimate_px,
        estimate_std_px=estimate.estimate_std_px,
        scale=estimate.scale,
        scale_std=estimate.scale_std,
        scale_error=error,
        tolerance=tolerance,
        tolerance_source="sigma" if sigma_tolerance > options.focal_scale_floor else "floor",
        ratio=abs(error) / tolerance,
        status=status,
        detectable_error=detectable,
        detection_probe=options.focal_detection_probe,
        detects_perturbation=detectable < options.focal_detection_probe,
    )


@dataclass(frozen=True)
class FocalJudgement:
    """Judged focal lengths and the verdict derived from them."""

    verdict: Literal["pass", "warn", "fail", "inconclusive"]
    record: CheckFocalScale

    @property
    def coverage(self) -> Literal["full", "partial"]:
        """``partial`` when a focal length was attempted but not judged."""

        return "partial" if self.record.unchecked else "full"


def judge_focal(record: FocalRecord, options: VerdictOptions) -> FocalJudgement:
    """Judge every estimated focal length and list the rest as unchecked."""

    components: list[CheckFocalComponent] = []
    unchecked: list[CheckUncheckedFocal] = []
    for estimate in record.components:
        if estimate.estimated:
            components.append(judge_focal_component(estimate, options))
        else:
            unchecked.append(
                CheckUncheckedFocal(
                    name=estimate.name,
                    candidate_px=estimate.candidate_px,
                    scale=estimate.scale if estimate.scale > 0.0 else None,
                    scale_std=estimate.scale_std if estimate.scale_std < float("inf") else None,
                    reason=estimate.unchecked_reason or "not constrained by the data",
                    reason_code=estimate.unchecked_code or "unobservable",
                )
            )
    scale_record = CheckFocalScale(
        components=components,
        unchecked=unchecked,
        optical_axis_ratio=record.optical_axis_ratio,
        optical_axis_ratio_std=record.optical_axis_ratio_std,
        floor=options.focal_scale_floor,
        intrinsics_source=record.intrinsics_source,
    )
    if not components:
        return FocalJudgement("inconclusive", scale_record)
    statuses = {item.status for item in components}
    verdict: Literal["pass", "warn", "fail"] = (
        "fail" if "fail" in statuses else "warn" if "warn" in statuses else "pass"
    )
    return FocalJudgement(verdict, scale_record)
