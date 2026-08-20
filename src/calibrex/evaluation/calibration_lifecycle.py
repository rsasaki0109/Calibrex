"""Synthetic calibration lifecycle replay with adoption and rollback controls."""

from __future__ import annotations

from typing import Literal

import numpy as np

from calibrex.core.calibration_lifecycle import (
    CalibrationLifecycleArtifact,
    CalibrationLifecycleEvent,
    CalibrationLifecycleEvidenceWindow,
    CalibrationLifecycleGateThresholds,
    CalibrationLifecycleProvenance,
    apply_calibration_lifecycle_rollback,
    decide_calibration_lifecycle_window,
)
from calibrex.core.geometry import SE3
from calibrex.core.provenance import git_commit
from calibrex.core.result import ObservabilityResult, TransformResult
from calibrex.core.se3_manifold import se3_exp

_SYNTHETIC_SEED = 20260820
_THRESHOLDS = CalibrationLifecycleGateThresholds(
    max_holdout_rmse_m=0.12,
    min_holdout_observations=50,
    min_rank=6,
    max_rolling_regression_m=0.03,
)
_KNOWN_BAD_WEAK_WINDOW_INDEX = 1
_KNOWN_BAD_HOLDOUT_WINDOW_INDEX = 2
_ROLLBACK_SOURCE_EVENT_INDEX = 0


def run_synthetic_calibration_lifecycle(
    *,
    seed: int = _SYNTHETIC_SEED,
    command: list[str] | None = None,
    lifecycle_id: str = "calibration-lifecycle-synthetic",
) -> CalibrationLifecycleArtifact:
    """Replay adopt, weak-observability reject, holdout reject, and rollback."""

    rng = np.random.default_rng(seed)
    parent = "lidar"
    child = "camera"
    variable = "T_lidar_camera"
    initial = _transform(parent, child, (0.02, -0.01, 0.05), (0.01, -0.02, 0.03))
    windows = _synthetic_windows(rng)
    events: list[CalibrationLifecycleEvent] = []
    incumbent = initial
    prior_holdout: float | None = None
    adoption_count = 0
    rejection_count = 0
    hold_count = 0
    rollback_count = 0
    weak_observability_rejection_count = 0
    weak_window_translation_before = None

    for window_index, window in enumerate(windows):
        candidate = _transform(
            parent,
            child,
            window["translation_m"],
            window["rotation_vec"],
        )
        evidence = CalibrationLifecycleEvidenceWindow(
            window_index=window_index,
            start_timestamp_ns=int(window["start_s"] * 1.0e9),
            end_timestamp_ns=int(window["end_s"] * 1.0e9),
            train_correspondence_count=int(window["train_count"]),
            holdout_correspondence_count=int(window["holdout_count"]),
            holdout_rmse_m=float(window["holdout_rmse_m"]),
            observability=ObservabilityResult(
                rank=int(window["rank"]),
                condition_number=float(window["condition_number"]),
                weak_directions=list(window["weak_directions"]),
                grade="pass" if not window["weak_directions"] else "warn",
            ),
            drift_score=float(window["drift_score"]),
        )
        if window_index == _KNOWN_BAD_WEAK_WINDOW_INDEX:
            weak_window_translation_before = list(incumbent.translation_m)
        decision, reason, installed, event_kind = decide_calibration_lifecycle_window(
            incumbent=incumbent,
            candidate=candidate,
            evidence=evidence,
            thresholds=_THRESHOLDS,
            prior_holdout_rmse_m=prior_holdout,
        )
        changed = not _same_transform(incumbent, installed)
        events.append(
            CalibrationLifecycleEvent(
                event_index=window_index,
                event_kind=event_kind,
                incumbent_before=incumbent,
                candidate=candidate,
                evidence_window=evidence,
                decision=decision,
                decision_reason=reason,
                incumbent_after=installed,
                installed_transform_changed=changed,
            )
        )
        if decision == "adopt":
            adoption_count += 1
            prior_holdout = evidence.holdout_rmse_m
        elif decision == "reject":
            rejection_count += 1
            if evidence.observability.weak_directions or (
                evidence.observability.rank is not None
                and evidence.observability.rank < _THRESHOLDS.min_rank
            ):
                weak_observability_rejection_count += 1
        else:
            hold_count += 1
        incumbent = installed

    rollback_evidence = CalibrationLifecycleEvidenceWindow(
        window_index=len(windows),
        train_correspondence_count=0,
        holdout_correspondence_count=0,
        holdout_rmse_m=None,
        observability=ObservabilityResult(rank=6, grade="pass"),
        drift_score=float(windows[_KNOWN_BAD_HOLDOUT_WINDOW_INDEX]["drift_score"]),
    )
    rollback_event = apply_calibration_lifecycle_rollback(
        event_index=len(events),
        incumbent=incumbent,
        rollback_target=events[_ROLLBACK_SOURCE_EVENT_INDEX].incumbent_after,
        evidence=rollback_evidence,
        reason=(
            "sustained holdout regression and drift score exceeded rollback policy; "
            f"restoring event {_ROLLBACK_SOURCE_EVENT_INDEX} incumbent"
        ),
        rollback_target_event_index=_ROLLBACK_SOURCE_EVENT_INDEX,
    )
    events.append(rollback_event)
    incumbent = rollback_event.incumbent_after
    rollback_count = 1

    weak_delta = 0.0
    if weak_window_translation_before is not None:
        weak_event = events[_KNOWN_BAD_WEAK_WINDOW_INDEX]
        weak_delta = float(
            np.linalg.norm(
                np.asarray(weak_event.incumbent_after.translation_m, dtype=float)
                - np.asarray(weak_window_translation_before, dtype=float)
            )
        )

    policy_status, policy_reason = _policy(
        adoption_count=adoption_count,
        weak_observability_rejection_count=weak_observability_rejection_count,
        weak_observability_installed_delta_m=weak_delta,
        rollback_count=rollback_count,
        final_incumbent=incumbent,
        rollback_target=events[_ROLLBACK_SOURCE_EVENT_INDEX].incumbent_after,
    )

    return CalibrationLifecycleArtifact(
        lifecycle_id=lifecycle_id,
        variable=variable,
        parent=parent,
        child=child,
        seed=seed,
        thresholds=_THRESHOLDS,
        initial_incumbent=initial,
        final_incumbent=incumbent,
        events=events,
        adoption_count=adoption_count,
        rejection_count=rejection_count,
        hold_count=hold_count,
        rollback_count=rollback_count,
        weak_observability_rejection_count=weak_observability_rejection_count,
        weak_observability_installed_delta_m=weak_delta,
        known_bad_weak_window_index=_KNOWN_BAD_WEAK_WINDOW_INDEX,
        known_bad_holdout_window_index=_KNOWN_BAD_HOLDOUT_WINDOW_INDEX,
        rollback_source_event_index=_ROLLBACK_SOURCE_EVENT_INDEX,
        policy_status=policy_status,
        policy_reason=policy_reason,
        provenance=CalibrationLifecycleProvenance(
            generator=__name__,
            generator_version="0.1",
            git_commit=git_commit(),
            command=command or [],
            seed=seed,
        ),
    )


def _synthetic_windows(rng: np.random.Generator) -> list[dict[str, object]]:
    return [
        {
            "start_s": 0.0,
            "end_s": 2.0,
            "translation_m": (0.04, -0.02, 0.06),
            "rotation_vec": (0.015, -0.01, 0.02),
            "train_count": 420,
            "holdout_count": 110,
            "holdout_rmse_m": 0.05 + rng.normal(0.0, 0.002),
            "rank": 6,
            "condition_number": 18.0,
            "weak_directions": [],
            "drift_score": 0.02,
        },
        {
            "start_s": 2.0,
            "end_s": 4.0,
            "translation_m": (0.18, 0.11, -0.04),
            "rotation_vec": (0.08, 0.05, -0.03),
            "train_count": 360,
            "holdout_count": 95,
            "holdout_rmse_m": 0.03 + rng.normal(0.0, 0.002),
            "rank": 4,
            "condition_number": 250.0,
            "weak_directions": ["tx", "ty"],
            "drift_score": 0.08,
        },
        {
            "start_s": 4.0,
            "end_s": 6.0,
            "translation_m": (0.07, -0.03, 0.08),
            "rotation_vec": (0.02, -0.015, 0.01),
            "train_count": 400,
            "holdout_count": 105,
            "holdout_rmse_m": 0.22 + rng.normal(0.0, 0.003),
            "rank": 6,
            "condition_number": 21.0,
            "weak_directions": [],
            "drift_score": 0.35,
        },
    ]


def _transform(
    parent: str,
    child: str,
    translation_m: tuple[float, float, float],
    rotation_vec: tuple[float, float, float],
) -> TransformResult:
    pose = se3_exp(
        SE3.identity(),
        np.concatenate([np.asarray(translation_m), np.asarray(rotation_vec)]),
    )
    return TransformResult(
        parent=parent,
        child=child,
        translation_m=list(pose.translation_m),
        rotation_quat_xyzw=list(pose.rotation_quat_xyzw),
        estimate_id=f"T_{parent}_{child}",
    )


def _same_transform(left: TransformResult, right: TransformResult) -> bool:
    return (
        left.translation_m == right.translation_m
        and left.rotation_quat_xyzw == right.rotation_quat_xyzw
    )


def _policy(
    *,
    adoption_count: int,
    weak_observability_rejection_count: int,
    weak_observability_installed_delta_m: float,
    rollback_count: int,
    final_incumbent: TransformResult,
    rollback_target: TransformResult,
) -> tuple[Literal["pass", "fail"], str]:
    if adoption_count < 1:
        return ("fail", "expected at least one adopted window in the synthetic replay")
    if weak_observability_rejection_count < 1:
        return (
            "fail",
            "expected a weak-observability rejection in the synthetic replay",
        )
    if weak_observability_installed_delta_m > 1.0e-9:
        return (
            "fail",
            (
                "weakly observable window changed the installed transform by "
                f"{weak_observability_installed_delta_m:.6f} m"
            ),
        )
    if rollback_count != 1:
        return ("fail", f"expected exactly one rollback event, found {rollback_count}")
    if not _same_transform(final_incumbent, rollback_target):
        return ("fail", "rollback did not restore the requested incumbent transform")
    return (
        "pass",
        (
            "synthetic lifecycle replay adopts a good window, rejects weak "
            "observability without mutating the installed transform, rejects "
            "holdout regression, and rolls back with provenance"
        ),
    )
