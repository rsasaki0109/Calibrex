"""Scoring cannot improve a claim by dropping failed cases or inventing clock truth."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from tools.score_ins_lidar_sota import CaseSpec, score_artifacts, score_case

from calibrex.core.ins_lidar_hand_eye import load_ins_lidar_hand_eye

DEVELOPMENT = Path("docs/assets/kitti_ins_lidar_2011_09_26.yaml")


def native():
    return load_ins_lidar_hand_eye(DEVELOPMENT)


def spec(**overrides):
    artifact = native()
    return CaseSpec(
        **(
            {
                "case_id": "spent-development",
                "dataset_family": "kitti_raw",
                "rig_id": "kitti-2011-09-26",
                "drive_ids": artifact.provenance.drive_ids,
                "artifact_path": str(DEVELOPMENT),
                "artifact_sha256": hashlib.sha256(DEVELOPMENT.read_bytes()).hexdigest(),
            }
            | overrides
        )
    )


def test_saved_partial_development_result_scores_only_observed_rotation():
    row = score_case(native())
    assert row["completed"] and row["estimated_axes"] == 2
    assert row["rotation_axes"] == 3 and row["control_misses"] == 0
    assert max(row["errors_deg"]) == pytest.approx(0.12438429827378121)
    assert len(row["errors_deg"]) == 2  # No yaw, translations, scale or absolute clock error.


def test_missing_recording_stays_in_every_family_denominator():
    result = score_artifacts(
        [spec(), spec(case_id="failed", artifact_path=None, artifact_sha256=None)], Path(".")
    )
    family = result["families"]["kitti_raw"]
    assert family["recording_completion"] == 0.5
    assert family["axis_coverage"] == pytest.approx(2 / 6)
    assert result["declared_dataset_families"] == result["declared_independent_rigs"] == 1


def test_missing_all_results_has_no_invented_zero_error_or_perfect_detection():
    result = score_artifacts([spec(artifact_path=None, artifact_sha256=None)], Path("."))
    family = result["families"]["kitti_raw"]
    assert family["maximum_reference_difference_deg"] is None
    assert family["control_miss_rate"] is None and family["axis_coverage"] == 0


def test_prior_assistance_cannot_count_as_native_observability():
    result = native()
    result.options["priors"] = [{"dof": "yaw", "value": 0.0, "sigma": 0.1}]
    with pytest.raises(ValueError, match="must not use priors"):
        score_case(result)


@pytest.mark.parametrize("change", ["missing", "nonfinite", "wrong_amount", "undetected", "weak"])
def test_bad_control_counts_as_a_miss_even_if_other_fields_look_good(change):
    artifact = native()
    control = artifact.dofs[0].known_bad_control
    if change == "missing":
        artifact.dofs[0].known_bad_control = None
    elif change == "nonfinite":
        control.holdout_delta_chi2 = float("nan")
    elif change == "wrong_amount":
        control.amount = 0.1
    elif change == "undetected":
        control.detected = False
    else:
        control.holdout_delta_chi2 = 8.0
    assert score_case(artifact)["control_misses"] == 1


def test_missing_reference_and_jackknife_are_failures_not_dropped_axes():
    artifact = native()
    artifact.dofs[0].error_to_reference = None
    row = score_case(artifact)
    assert not row["completed"] and row["estimated_axes"] == 0
    artifact = native()
    artifact.jackknife_fits = 0
    assert not score_case(artifact)["completed"]


@pytest.mark.parametrize("overrides", [{"artifact_sha256": "0" * 64}, {"dataset_family": "boreas"}])
def test_mismatched_evidence_cannot_be_substituted(overrides):
    with pytest.raises(ValueError, match="mismatch"):
        score_artifacts([spec(**overrides)], Path("."))
