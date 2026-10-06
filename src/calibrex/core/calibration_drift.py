"""Schema-versioned calibration drift between recordings of one rig (``calibrex drift``).

``slac.calibration_drift/v0.1`` answers one question: did the extrinsic
calibration of a rig change between two or more recordings of it? Each bag is
estimated with ``calibrex estimate`` (the per-bag ``slac.bag_estimate/v0.1``
artifacts are referenced by path and digest, not embedded), and every axis that
the data *observed* in at least two bags is tested for consistency across them:

* pairwise differences ``d_ij = x_i - x_j`` with the z score
  ``d_ij / sqrt(s_i^2 + s_j^2)`` and the tolerance ``max(sigma_k * sqrt(s_i^2 +
  s_j^2), floor)`` (the floor is the minimum detectable change of the axis
  type, as in ``slac.calibration_check``);
* a chi-square homogeneity test of the inverse-variance weighted mean;
* for three or more bags, the bag(s) whose removal restores consistency.

An axis is ``drift`` only when a pairwise difference exceeds its tolerance and
the chi-square test rejects homogeneity; ``stable`` when no pair exceeds it;
``inconclusive`` when fewer than two bags observed the axis. The standard
deviations are the estimators' own (analytic or block jackknife) and are not
guaranteed to be calibrated; the floor is what protects against over-trusting a
small std.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from calibrex.core.bag_estimate import EstimateAxisStatus, EstimateOptions
from calibrex.core.calibration_check import CheckAxisName, CheckPairName, CheckProvenance
from calibrex.core.result import StrictModel

CALIBRATION_DRIFT_SCHEMA_VERSION: Literal["slac.calibration_drift/v0.1"] = (
    "slac.calibration_drift/v0.1"
)

DriftVerdict = Literal["stable", "inconclusive", "drift"]
FloorSource = Literal["rotation", "rigid_scan_rotation", "translation", "time_offset"]


class DriftThresholds(StrictModel):
    """The decision rule's settings."""

    sigma_k: float = Field(gt=0.0)
    rotation_floor_deg: float = Field(gt=0.0)
    rigid_scan_rotation_floor_deg: float = Field(gt=0.0)
    translation_floor_m: float = Field(gt=0.0)
    chi2_alpha: float = Field(gt=0.0, lt=1.0)
    time_offset_floors_s: dict[str, float] = Field(
        default_factory=dict,
        description="minimum detectable clock-offset change per pair type, in seconds; "
        "'default' applies to a pair type not listed",
    )


class DriftBagRef(StrictModel):
    """One recording and the per-bag estimate it was judged from."""

    name: str
    path: str
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_digest_scope: str
    estimate_path: str = Field(
        description="the bag's slac.bag_estimate artifact, relative to the drift artifact"
    )
    estimate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pair_statuses: dict[str, str] = Field(
        default_factory=dict, description="estimate status of each pair in this bag"
    )
    runtime_s: float = Field(ge=0.0)
    estimator_cache_hits: int = Field(ge=0)
    estimator_runs: int = Field(ge=0)


class DriftObservation(StrictModel):
    """One bag's estimate of one axis (or of the pair's clock offset, in seconds)."""

    bag: str
    value: float | None = None
    std: float | None = Field(default=None, ge=0.0)
    status: EstimateAxisStatus | Literal["pair_not_estimated"]
    used: bool = Field(description="true when the data observed the axis and it enters the tests")


class DriftPairwise(StrictModel):
    """Difference of two bags on one axis, ``bag_a`` minus ``bag_b``."""

    bag_a: str
    bag_b: str
    difference: float
    std: float = Field(ge=0.0)
    z: float
    tolerance: float = Field(gt=0.0)
    ratio: float = Field(ge=0.0, description="|difference| / tolerance; above 1 is a change")
    exceeds: bool


class DriftLeaveOneOut(StrictModel):
    """One bag against the weighted mean of the consensus bags (three or more bags).

    The consensus is the bags that remain after the deviating ones are removed; ``deviates``
    marks the removed bags.
    """

    bag: str
    delta: float
    std: float = Field(ge=0.0)
    z: float
    tolerance: float = Field(gt=0.0)
    deviates: bool


class DriftAxisRecord(StrictModel):
    """Consistency of one axis of one pair across bags."""

    name: CheckAxisName
    unit: Literal["deg", "m"]
    observations: list[DriftObservation]
    bags_observed: int = Field(ge=0)
    status: DriftVerdict
    reason: str | None = None
    weighted_mean: float | None = None
    weighted_mean_std: float | None = Field(default=None, ge=0.0)
    chi2: float | None = Field(default=None, ge=0.0)
    dof: int | None = Field(default=None, ge=0)
    p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    floor: float = Field(gt=0.0)
    floor_source: FloorSource
    minimum_detectable_change: float | None = Field(
        default=None,
        description="smallest pairwise tolerance over the bags that observed the axis: a "
        "change smaller than this cannot be flagged between those bags",
    )
    max_abs_difference: float | None = Field(default=None, ge=0.0)
    pairwise: list[DriftPairwise] = Field(default_factory=list)
    leave_one_out: list[DriftLeaveOneOut] = Field(default_factory=list)


class DriftTimeOffsetRecord(DriftAxisRecord):
    """Consistency of one pair's clock offset (seconds) across bags.

    The same record as an axis (observations, pairwise z with the floor, chi-square, leave-one-out)
    with ``name`` fixed to ``time_offset``. ``unit`` is ``s``, so it is not a rotation or
    translation axis.
    """

    name: Literal["time_offset"] = "time_offset"  # type: ignore[assignment]
    unit: Literal["s"] = "s"  # type: ignore[assignment]
    floor_source: FloorSource = "time_offset"


class DriftBagChange(StrictModel):
    """How far one bag sits from the others on the pair's rotation, when it deviates."""

    bag: str
    reference: Literal["other_bag", "mean_of_others"]
    rotation_delta_deg: float | None = Field(
        default=None,
        ge=0.0,
        description="angle of the rotation between this bag's estimated rotation and the "
        "reference, when all three rotation axes were observed in both",
    )
    axes_delta: dict[str, float] = Field(
        default_factory=dict, description="this bag minus the reference, per observed axis"
    )
    time_offset_delta_s: float | None = Field(
        default=None,
        description="this bag's clock offset minus the reference, when it was compared in both",
    )


class DriftPairRecord(StrictModel):
    """One sensor pair compared across bags."""

    pair: CheckPairName
    parent_frame: str | None = None
    child_frame: str | None = None
    verdict: DriftVerdict
    reason: str | None = None
    bags_compared: list[str] = Field(default_factory=list)
    axes: list[DriftAxisRecord] = Field(default_factory=list)
    time_offset: DriftTimeOffsetRecord | None = Field(
        default=None,
        description="the pair's clock offset compared across bags (same tests as an axis); "
        "absent when the pair type has no meaningful offset (see time_offset_skipped)",
    )
    time_offset_skipped: str | None = Field(
        default=None, description="why the clock offset of this pair type is not compared"
    )
    deviating_bags: list[str] = Field(
        default_factory=list,
        description="bags that disagree with the rest on a drifting axis; empty when two bags "
        "differ (which one moved cannot be told) or when the bags split into groups",
    )
    attribution: Literal["bag", "ambiguous"] | None = None
    changes: list[DriftBagChange] = Field(default_factory=list)


class DriftSummary(StrictModel):
    """Counts over the pair records."""

    bag_count: int = Field(ge=0)
    pair_count: int = Field(ge=0)
    verdict_counts: dict[str, int] = Field(default_factory=dict)
    axes_tested: int = Field(ge=0)
    axes_drifting: int = Field(ge=0)
    time_offsets_tested: int | None = Field(default=None, ge=0)
    time_offsets_drifting: int | None = Field(default=None, ge=0)


class CalibrationDriftArtifact(StrictModel):
    """Whether a rig's calibration changed between recordings, with its evidence."""

    schema_version: Literal["slac.calibration_drift/v0.1"] = CALIBRATION_DRIFT_SCHEMA_VERSION
    bags: list[DriftBagRef]
    vehicle_frame: str | None = None
    pairs: list[DriftPairRecord] = Field(default_factory=list)
    overall_verdict: DriftVerdict
    summary: DriftSummary
    thresholds: DriftThresholds
    options: EstimateOptions
    next_steps: list[str] = Field(default_factory=list)
    provenance: CheckProvenance


def calibration_drift_json_schema() -> dict[str, object]:
    """Return the standalone JSON schema for calibration drift artifacts."""

    return CalibrationDriftArtifact.model_json_schema()
