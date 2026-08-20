"""Schema-valid calibration lifecycle events for drift, adoption, and rollback."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from calibrex.core.io import write_mapping
from calibrex.core.result import ObservabilityResult, StrictModel, TransformResult

CALIBRATION_LIFECYCLE_SCHEMA_VERSION: Literal[
    "slac.calibration_lifecycle/v0.1"
] = "slac.calibration_lifecycle/v0.1"

LifecycleDecision = Literal["adopt", "reject", "hold", "rollback"]
LifecycleEventKind = Literal[
    "candidate_evaluated",
    "adopted",
    "rejected",
    "held",
    "drift_detected",
    "rolled_back",
]


class CalibrationLifecycleGateThresholds(StrictModel):
    """Adoption gate thresholds aligned with online holdout and observability."""

    max_holdout_rmse_m: float = Field(gt=0.0)
    min_holdout_observations: int = Field(ge=1)
    min_rank: int = Field(ge=1, le=6)
    max_rolling_regression_m: float = Field(ge=0.0)


class CalibrationLifecycleEvidenceWindow(StrictModel):
    """Observability and holdout evidence for one candidate evaluation window."""

    window_index: int = Field(ge=0)
    start_timestamp_ns: int | None = Field(default=None, ge=0)
    end_timestamp_ns: int | None = Field(default=None, ge=0)
    train_correspondence_count: int = Field(ge=0)
    holdout_correspondence_count: int = Field(ge=0)
    holdout_rmse_m: float | None = Field(default=None, ge=0.0)
    observability: ObservabilityResult = Field(default_factory=ObservabilityResult)
    drift_score: float | None = Field(default=None, ge=0.0)


class CalibrationLifecycleEvent(StrictModel):
    """One incumbent/candidate decision with installed-transform outcome."""

    event_index: int = Field(ge=0)
    event_kind: LifecycleEventKind
    incumbent_before: TransformResult
    candidate: TransformResult | None = None
    evidence_window: CalibrationLifecycleEvidenceWindow
    decision: LifecycleDecision
    decision_reason: str
    incumbent_after: TransformResult
    rollback_target_event_index: int | None = Field(default=None, ge=0)
    installed_transform_changed: bool


class CalibrationLifecycleProvenance(StrictModel):
    """Seeded synthetic lifecycle lineage."""

    generator: str
    generator_version: str
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    seed: int
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class CalibrationLifecycleArtifact(StrictModel):
    """Machine-readable adoption, rejection, and rollback timeline."""

    schema_version: Literal[
        "slac.calibration_lifecycle/v0.1"
    ] = CALIBRATION_LIFECYCLE_SCHEMA_VERSION
    lifecycle_id: str
    variable: str
    parent: str
    child: str
    seed: int
    thresholds: CalibrationLifecycleGateThresholds
    initial_incumbent: TransformResult
    final_incumbent: TransformResult
    events: list[CalibrationLifecycleEvent] = Field(min_length=1)
    adoption_count: int = Field(ge=0)
    rejection_count: int = Field(ge=0)
    hold_count: int = Field(ge=0)
    rollback_count: int = Field(ge=0)
    weak_observability_rejection_count: int = Field(ge=0)
    weak_observability_installed_delta_m: float = Field(ge=0.0)
    known_bad_weak_window_index: int = Field(ge=0)
    known_bad_holdout_window_index: int = Field(ge=0)
    rollback_source_event_index: int = Field(ge=0)
    policy_status: Literal["pass", "fail"]
    policy_reason: str
    provenance: CalibrationLifecycleProvenance

    @model_validator(mode="after")
    def check_events(self) -> CalibrationLifecycleArtifact:
        indices = [item.event_index for item in self.events]
        if indices != list(range(len(indices))):
            raise ValueError("lifecycle event indices must be contiguous from zero")
        if self.final_incumbent != self.events[-1].incumbent_after:
            raise ValueError("final_incumbent must match the last event outcome")
        return self

    def save(self, path: str | Path) -> None:
        """Save the lifecycle artifact as YAML or JSON."""

        write_mapping(
            Path(path),
            self.model_dump(mode="json", exclude_none=True),
        )


def decide_calibration_lifecycle_window(
    *,
    incumbent: TransformResult,
    candidate: TransformResult,
    evidence: CalibrationLifecycleEvidenceWindow,
    thresholds: CalibrationLifecycleGateThresholds,
    prior_holdout_rmse_m: float | None = None,
) -> tuple[LifecycleDecision, str, TransformResult, LifecycleEventKind]:
    """Apply adoption gates; weakly observable windows keep the incumbent."""

    if evidence.holdout_correspondence_count < thresholds.min_holdout_observations:
        return (
            "hold",
            (
                "too few holdout correspondences to score this window "
                f"({evidence.holdout_correspondence_count} < "
                f"{thresholds.min_holdout_observations})"
            ),
            incumbent,
            "held",
        )
    if evidence.holdout_rmse_m is None:
        return (
            "hold",
            "holdout RMSE is unavailable for this window",
            incumbent,
            "held",
        )
    if evidence.holdout_rmse_m > thresholds.max_holdout_rmse_m:
        return (
            "reject",
            (
                f"holdout RMSE {evidence.holdout_rmse_m:.4f} m exceeds the "
                f"{thresholds.max_holdout_rmse_m:.4f} m gate"
            ),
            incumbent,
            "rejected",
        )
    if (
        prior_holdout_rmse_m is not None
        and (evidence.holdout_rmse_m - prior_holdout_rmse_m)
        > thresholds.max_rolling_regression_m
    ):
        return (
            "reject",
            (
                f"holdout RMSE regressed by "
                f"{evidence.holdout_rmse_m - prior_holdout_rmse_m:.4f} m against "
                f"the prior baseline ({prior_holdout_rmse_m:.4f} m)"
            ),
            incumbent,
            "rejected",
        )
    observability = evidence.observability
    if observability.rank is not None and observability.rank < thresholds.min_rank:
        return (
            "reject",
            (
                f"window geometry is rank-deficient "
                f"(rank={observability.rank} < {thresholds.min_rank})"
            ),
            incumbent,
            "rejected",
        )
    if observability.weak_directions:
        return (
            "reject",
            (
                "window geometry has weak directions "
                f"{list(observability.weak_directions)}"
            ),
            incumbent,
            "rejected",
        )
    return (
        "adopt",
        "holdout and observability gates passed; adopting the candidate transform",
        candidate,
        "adopted",
    )


def apply_calibration_lifecycle_rollback(
    *,
    event_index: int,
    incumbent: TransformResult,
    rollback_target: TransformResult,
    evidence: CalibrationLifecycleEvidenceWindow,
    reason: str,
    rollback_target_event_index: int,
) -> CalibrationLifecycleEvent:
    """Restore a prior installed transform with explicit rollback provenance."""

    return CalibrationLifecycleEvent(
        event_index=event_index,
        event_kind="rolled_back",
        incumbent_before=incumbent,
        candidate=rollback_target,
        evidence_window=evidence,
        decision="rollback",
        decision_reason=reason,
        incumbent_after=rollback_target,
        rollback_target_event_index=rollback_target_event_index,
        installed_transform_changed=not _transforms_equal(incumbent, rollback_target),
    )


def _transforms_equal(left: TransformResult, right: TransformResult) -> bool:
    return (
        left.parent == right.parent
        and left.child == right.child
        and left.translation_m == right.translation_m
        and left.rotation_quat_xyzw == right.rotation_quat_xyzw
    )


def calibration_lifecycle_json_schema() -> dict[str, Any]:
    """Return the JSON schema for calibration lifecycle artifacts."""

    return CalibrationLifecycleArtifact.model_json_schema()
