"""Tests for calibration lifecycle adoption, rejection, and rollback artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from calibrex.cli.main import main
from calibrex.core.calibration_lifecycle import (
    CalibrationLifecycleEvidenceWindow,
    CalibrationLifecycleGateThresholds,
    decide_calibration_lifecycle_window,
)
from calibrex.core.geometry import SE3
from calibrex.core.result import ObservabilityResult, TransformResult
from calibrex.core.se3_manifold import se3_exp
from calibrex.core.validation import validate_file
from calibrex.evaluation.calibration_lifecycle import (
    run_synthetic_calibration_lifecycle,
)


def _transform(
    translation_m: tuple[float, float, float],
    rotation_vec: tuple[float, float, float],
) -> TransformResult:
    pose = se3_exp(
        SE3.identity(),
        np.concatenate([np.asarray(translation_m), np.asarray(rotation_vec)]),
    )
    return TransformResult(
        parent="lidar",
        child="camera",
        translation_m=list(pose.translation_m),
        rotation_quat_xyzw=list(pose.rotation_quat_xyzw),
    )


def test_weak_observability_rejects_without_changing_incumbent() -> None:
    incumbent = _transform((0.02, -0.01, 0.05), (0.01, -0.02, 0.03))
    candidate = _transform((0.15, 0.08, -0.02), (0.05, 0.03, -0.01))
    thresholds = CalibrationLifecycleGateThresholds(
        max_holdout_rmse_m=0.12,
        min_holdout_observations=50,
        min_rank=6,
        max_rolling_regression_m=0.03,
    )
    decision, reason, installed, event_kind = decide_calibration_lifecycle_window(
        incumbent=incumbent,
        candidate=candidate,
        evidence=CalibrationLifecycleEvidenceWindow(
            window_index=0,
            train_correspondence_count=300,
            holdout_correspondence_count=80,
            holdout_rmse_m=0.04,
            observability=ObservabilityResult(
                rank=4,
                weak_directions=["tx", "ty"],
                grade="warn",
            ),
        ),
        thresholds=thresholds,
        prior_holdout_rmse_m=0.05,
    )
    assert decision == "reject"
    assert event_kind == "rejected"
    assert "rank-deficient" in reason or "weak directions" in reason
    assert installed.translation_m == incumbent.translation_m
    assert installed.rotation_quat_xyzw == incumbent.rotation_quat_xyzw


def test_holdout_regression_rejects_candidate() -> None:
    incumbent = _transform((0.02, -0.01, 0.05), (0.01, -0.02, 0.03))
    candidate = _transform((0.04, -0.02, 0.06), (0.015, -0.01, 0.02))
    thresholds = CalibrationLifecycleGateThresholds(
        max_holdout_rmse_m=0.12,
        min_holdout_observations=50,
        min_rank=6,
        max_rolling_regression_m=0.03,
    )
    decision, reason, installed, _ = decide_calibration_lifecycle_window(
        incumbent=incumbent,
        candidate=candidate,
        evidence=CalibrationLifecycleEvidenceWindow(
            window_index=0,
            train_correspondence_count=320,
            holdout_correspondence_count=90,
            holdout_rmse_m=0.20,
            observability=ObservabilityResult(rank=6, grade="pass"),
        ),
        thresholds=thresholds,
    )
    assert decision == "reject"
    assert "holdout RMSE" in reason
    assert installed == incumbent


def test_synthetic_calibration_lifecycle_replay_and_controls(
    tmp_path: Path,
) -> None:
    artifact = run_synthetic_calibration_lifecycle(seed=20260820)
    output = tmp_path / "lifecycle.yaml"
    artifact.save(output)
    assert validate_file(output, kind="calibration-lifecycle").valid
    assert artifact.policy_status == "pass"
    assert artifact.adoption_count >= 1
    assert artifact.weak_observability_rejection_count >= 1
    assert artifact.weak_observability_installed_delta_m == 0.0
    assert artifact.rollback_count == 1
    assert artifact.final_incumbent.translation_m == artifact.events[
        artifact.rollback_source_event_index
    ].incumbent_after.translation_m


def test_cli_simulate_calibration_lifecycle(tmp_path: Path) -> None:
    output = tmp_path / "lifecycle.yaml"
    exit_code = main(
        [
            "lifecycle",
            "simulate",
            "--output",
            str(output),
            "--seed",
            "20260820",
            "--json",
        ]
    )
    assert exit_code == 0
    assert validate_file(output).valid
