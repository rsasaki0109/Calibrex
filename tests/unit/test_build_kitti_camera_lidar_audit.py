"""The KITTI camera-lidar builder: its audit verdict equals the scorer's, gate by gate."""

import json
from pathlib import Path
from typing import Any

import pytest
from tests.unit.test_score_hilti_camera_camera import _load
from tests.unit.test_score_kitti_camera_lidar import DRIVES, _all, _drive, _missed, _run, scorer

builder = _load("build_kitti_camera_lidar_audit")


def _build(tmp_path: Path, drives: dict[str, Any]) -> Any:
    scores = scorer.score_drives(drives, list(drives))
    scores_path = tmp_path / "scores.json"
    scores_path.write_text(json.dumps(scores), encoding="utf-8")
    preregistration = tmp_path / "prereg.yaml"
    preregistration.write_text("claim: test\n", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    return scores, builder.build(scores, scores_path, preregistration, out)


def test_a_clean_set_is_supported_with_every_gate_achieved(tmp_path: Path) -> None:
    scores, result = _build(tmp_path, _all())
    assert scores["verdict"] == result.verdict == "supported"
    assert result.protocol_id == "kitti-camera-lidar-check-v1"
    assert [item.requirement_id for item in result.requirements] == list(scorer.REQUIREMENTS)
    assert {item.status for item in result.requirements} == {"achieved"}


def _inaccurate(drives: dict[str, Any]) -> None:
    drives["d02"] = scorer.DriveInput(
        baseline=_run(errors_deg=(0.1, -0.8, 0.05), verdict="pass"), turned=drives["d02"].turned
    )


def _inaccurate_turned(drives: dict[str, Any]) -> None:
    drives["d02"] = _drive(y_1=_run("y:1", errors_deg=(1.6, 0.2, 0.05)))


def _false_alarm(drives: dict[str, Any]) -> None:
    drives["d02"] = scorer.DriveInput(baseline=_run(verdict="warn"), turned=drives["d02"].turned)


def _failed(drives: dict[str, Any]) -> None:
    drives["d04"] = scorer.DriveInput(error="conversion failed")


def _low_coverage(drives: dict[str, Any]) -> None:
    for name in DRIVES:
        drives[name] = _drive(judged=("pitch",))


def _missed_3deg(drives: dict[str, Any]) -> None:
    for name in DRIVES[:7]:
        drives[name] = _drive(z_3=_missed("z:3", "inconclusive"))


def _judged_right_3deg(drives: dict[str, Any]) -> None:
    drives["d09"] = _drive(y_3=_run("y:3", errors_deg=(0.1, 2.6, 0.05)))  # reads 0.4: pass


def _missed_1deg(drives: dict[str, Any]) -> None:
    for name in DRIVES:
        drives[name] = _drive(x_1=_missed("x:1", "pass"), y_1=_missed("y:1", "warn"))


def _few_drives(drives: dict[str, Any]) -> None:
    for name in DRIVES[:6]:
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


@pytest.mark.parametrize(("misses", "verdict"), [(6, "supported"), (7, "refuted")])
def test_the_3deg_boundary_is_the_same_in_both(tmp_path: Path, misses: int, verdict: str) -> None:
    drives = _all()
    for name in DRIVES[:misses]:
        drives[name] = _drive(z_3=_missed("z:3", "inconclusive"))
    scores, result = _build(tmp_path, drives)
    assert scores["verdict"] == result.verdict == verdict
    gate = {item.requirement_id: item for item in result.requirements}["turn-3deg-flagged"]
    assert gate.observed_value == misses / 60


def test_exactly_half_of_the_axes_judged_meets_coverage_in_both(tmp_path: Path) -> None:
    drives = {
        name: _drive(judged=("roll", "pitch") if index % 2 else ("yaw",))
        for index, name in enumerate(DRIVES)
    }
    scores, result = _build(tmp_path, drives)
    assert scores["judged_fraction"] == 0.5
    assert scores["verdict"] == result.verdict == "supported"
    coverage = {item.requirement_id: item for item in result.requirements}["judged-axis-coverage"]
    assert coverage.observed_value == 1.5
