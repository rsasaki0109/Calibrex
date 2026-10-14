"""The KITTI camera-lidar v2 builder: its audit verdict equals the scorer's, gate by gate."""

import json
from pathlib import Path
from typing import Any

import pytest
from tests.unit.test_score_hilti_camera_camera import _load
from tests.unit.test_score_kitti_camera_lidar_v2 import (
    DRIVES,
    TRIALS,
    _all,
    _allowed,
    _drive,
    _missed,
    _run,
    _with_misses,
    scorer,
)

builder = _load("build_kitti_camera_lidar_v2_audit")


def _build(tmp_path: Path, drives: dict[str, Any]) -> Any:
    scores = scorer.score_drives(drives, list(drives))
    scores_path = tmp_path / "scores.json"
    scores_path.write_text(json.dumps(scores), encoding="utf-8")
    preregistration = tmp_path / "prereg.yaml"
    preregistration.write_text("claim: test\n", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    return scores, builder.build(scores, scores_path, preregistration, out)


def test_the_builder_and_the_scorer_share_their_thresholds() -> None:
    assert builder.AXIS_ERROR_MAX_DEG == scorer.AXIS_ERROR_MAX_DEG
    assert builder.TURNED_AXIS_ERROR_MAX_DEG == scorer.TURNED_AXIS_ERROR_MAX_DEG
    assert builder.UNJUDGED_AXES_MEAN_MAX == 3 * (1 - scorer.JUDGED_FRACTION_MIN)
    assert builder.MISSED_3DEG_MAX == scorer.MISSED_3DEG_MAX
    assert builder.MISSED_1DEG_MAX == scorer.MISSED_1DEG_MAX
    assert builder.SCORED_DRIVES_MIN == scorer.SCORED_DRIVES_MIN


def test_a_clean_set_is_supported_with_every_gate_achieved(tmp_path: Path) -> None:
    scores, result = _build(tmp_path, _all())
    assert scores["verdict"] == result.verdict == "supported"
    assert result.protocol_id == "kitti-camera-lidar-check-v2"
    assert [item.requirement_id for item in result.requirements] == list(scorer.REQUIREMENTS)
    assert {item.status for item in result.requirements} == {"achieved"}


def _inaccurate(drives: dict[str, Any]) -> None:
    error = scorer.AXIS_ERROR_MAX_DEG + 0.05
    drives[DRIVES[2]] = scorer.DriveInput(
        baseline=_run(errors_deg=(0.1, -error, 0.05), verdict="pass"),
        turned=drives[DRIVES[2]].turned,
    )


def _inaccurate_turned(drives: dict[str, Any]) -> None:
    error = scorer.TURNED_AXIS_ERROR_MAX_DEG + 0.1
    drives[DRIVES[2]] = _drive(y_1=_run("y:1", errors_deg=(error, 0.2, 0.05)))


def _false_alarm(drives: dict[str, Any]) -> None:
    drives[DRIVES[2]] = scorer.DriveInput(
        baseline=_run(verdict="warn"), turned=drives[DRIVES[2]].turned
    )


def _failed(drives: dict[str, Any]) -> None:
    drives[DRIVES[4]] = scorer.DriveInput(error="conversion failed")


def _low_coverage(drives: dict[str, Any]) -> None:
    for name in DRIVES:
        drives[name] = _drive(judged=("pitch",))


def _missed_3deg(drives: dict[str, Any]) -> None:
    drives.update(_with_misses(_allowed(scorer.MISSED_3DEG_MAX) + 1, "3", "inconclusive"))


def _judged_right_3deg(drives: dict[str, Any]) -> None:
    drives[DRIVES[9]] = _drive(y_3=_run("y:3", errors_deg=(0.1, 2.6, 0.05)))  # reads 0.4: pass


def _missed_1deg(drives: dict[str, Any]) -> None:
    drives.update(_with_misses(_allowed(scorer.MISSED_1DEG_MAX) + 1, "1", "pass"))


def _few_drives(drives: dict[str, Any]) -> None:
    for name in DRIVES[: len(DRIVES) - scorer.SCORED_DRIVES_MIN + 1]:
        drives[name] = scorer.DriveInput(
            baseline=scorer.RunInput(verdict="inconclusive", solver_status="insufficient_samples"),
            turned=drives[name].turned,
        )


def _nothing_judged(drives: dict[str, Any]) -> None:
    for name in DRIVES:
        drives[name] = _drive(judged=())


@pytest.mark.parametrize(
    ("change", "verdict", "requirement", "status"),
    [
        (_inaccurate, "refuted", "judged-axis-accuracy", "contradicted"),
        (_inaccurate_turned, "refuted", "turned-judged-axis-accuracy", "contradicted"),
        (_false_alarm, "refuted", "vendor-calibration-not-flagged", "contradicted"),
        (_failed, "refuted", "no-failed-drives", "contradicted"),
        (_low_coverage, "incomplete", "judged-axis-coverage", "incomplete"),
        (_missed_3deg, "refuted", "turn-3deg-flagged", "contradicted"),
        (_judged_right_3deg, "refuted", "turn-3deg-never-judged-right", "contradicted"),
        (_missed_1deg, "refuted", "turn-1deg-flagged", "contradicted"),
        (_few_drives, "incomplete", "scored-drives-coverage", "incomplete"),
        (_nothing_judged, "incomplete", "turn-3deg-flagged", "incomplete"),
    ],
)
def test_the_audit_follows_the_scorer(
    tmp_path: Path, change: Any, verdict: str, requirement: str, status: str
) -> None:
    drives = _all()
    change(drives)
    scores, result = _build(tmp_path, drives)
    expected = {"supported": "supported", "refuted": "refuted", "inconclusive": "incomplete"}
    assert expected[scores["verdict"]] == result.verdict == verdict
    assert {item.requirement_id: item.status for item in result.requirements}[requirement] == status


@pytest.mark.parametrize("extra", [0, 1])
def test_the_3deg_boundary_is_the_same_in_both(tmp_path: Path, extra: int) -> None:
    misses = _allowed(scorer.MISSED_3DEG_MAX) + extra
    scores, result = _build(tmp_path, _with_misses(misses, "3", "inconclusive"))
    assert scores["verdict"] == result.verdict == ("refuted" if extra else "supported")
    gate = {item.requirement_id: item for item in result.requirements}["turn-3deg-flagged"]
    assert gate.observed_value == misses / TRIALS


@pytest.mark.parametrize("extra", [0, 1])
def test_the_scored_drives_boundary_is_the_same_in_both(tmp_path: Path, extra: int) -> None:
    drives = _all()
    for name in DRIVES[: len(DRIVES) - scorer.SCORED_DRIVES_MIN + extra]:
        drives[name] = scorer.DriveInput(
            baseline=scorer.RunInput(verdict="inconclusive", solver_status="insufficient_samples"),
            turned=drives[name].turned,
        )
    scores, result = _build(tmp_path, drives)
    assert scores["verdict"] == ("inconclusive" if extra else "supported")
    assert result.verdict == ("incomplete" if extra else "supported")


def test_exactly_half_of_the_axes_judged_meets_coverage_in_both(tmp_path: Path) -> None:
    assert scorer.JUDGED_FRACTION_MIN == 0.5
    drives = {
        name: _drive(judged=("roll", "pitch") if index % 2 else ("yaw",))
        for index, name in enumerate(DRIVES)
    }
    scores, result = _build(tmp_path, drives)
    assert scores["judged_fraction"] == 0.5
    assert scores["verdict"] == result.verdict == "supported"
    coverage = {item.requirement_id: item for item in result.requirements}["judged-axis-coverage"]
    assert coverage.observed_value == 1.5


def test_the_claim_states_the_gates() -> None:
    claim = builder.CLAIM.format(drives=18)
    assert claim.startswith("On 18 KITTI raw drives of 2011_09_28, 2011_09_29, 2011_09_30 and")
    assert f"within {scorer.AXIS_ERROR_MAX_DEG:g} deg of it" in claim
    assert f"at least {round(100 * (1 - scorer.MISSED_3DEG_MAX))} percent" in claim
    assert "%" not in claim and "{" not in claim


def test_a_missed_trial_that_judges_another_axis_wrong_still_counts_once(tmp_path: Path) -> None:
    drives = _all()
    drives[DRIVES[6]] = _drive(z_3=_missed("z:3", "fail"))
    scores, result = _build(tmp_path, drives)
    assert scores["trials"]["3deg"]["flagged"] == TRIALS - 1
    gate = {item.requirement_id: item for item in result.requirements}["turn-3deg-flagged"]
    assert gate.observed_value == 1 / TRIALS
