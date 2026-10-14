"""The camera-IMU v3 builder: its audit verdict equals the scorer's, gate by gate."""

import json
from pathlib import Path
from typing import Any

import pytest
from tests.unit.test_score_hilti_camera_camera import _load
from tests.unit.test_score_hilti_camera_imu_v3 import RECORDINGS, UNJUDGED, _all, _unit, scorer

builder = _load("build_hilti_camera_imu_v3_audit")


def _build(tmp_path: Path, units: dict[str, dict[str, Any]], recordings: list[str]) -> Any:
    scores = scorer.score_units(units, recordings)
    scores_path = tmp_path / "scores.json"
    scores_path.write_text(json.dumps(scores), encoding="utf-8")
    preregistration = tmp_path / "prereg.yaml"
    preregistration.write_text("claim: test\n", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    return scores, builder.build(scores, scores_path, preregistration, out)


def test_a_clean_set_is_supported_with_every_gate_achieved(tmp_path: Path) -> None:
    scores, result = _build(tmp_path, _all(), RECORDINGS)
    assert scores["verdict"] == result.verdict == "supported"
    assert result.protocol_id == "hilti-camera-imu-v3"
    assert [item.requirement_id for item in result.requirements] == list(scorer.REQUIREMENTS)
    assert {item.status for item in result.requirements} == {"achieved"}
    coverage = {item.requirement_id: item for item in result.requirements}
    assert coverage["scored-recordings-coverage"].observed_value == 0.0


def _refuted(units: dict[str, dict[str, Any]]) -> None:
    units["cam0"]["r2"] = _unit(errors_deg=(0.1, -1.2, 0.1))
    units["cam1"]["r2"] = _unit("cam1", errors_deg=(0.1, -1.2, 0.1))


def _failed(units: dict[str, dict[str, Any]]) -> None:
    units["cam1"]["r4"] = scorer.UnitInput(error="artifact missing")


def _low_coverage(units: dict[str, dict[str, Any]]) -> None:
    judged = scorer.AxisInput()
    for camera in units:
        for recording in units[camera]:
            units[camera][recording] = _unit(camera, axes=(judged, UNJUDGED, UNJUDGED))


def _few_recordings(units: dict[str, dict[str, Any]]) -> None:
    for recording in ("r3", "r4", "r5"):
        units["cam1"][recording] = _unit("cam1", windows_used=3)


def _nothing_judged(units: dict[str, dict[str, Any]]) -> None:
    for camera in units:
        for recording in units[camera]:
            units[camera][recording] = _unit(camera, axes=(UNJUDGED,) * 3, time=UNJUDGED)


@pytest.mark.parametrize(
    ("change", "verdict", "requirement", "status"),
    [
        (_refuted, "refuted", "judged-axis-accuracy", "contradicted"),
        (_failed, "refuted", "no-failed-units", "contradicted"),
        (_low_coverage, "incomplete", "judged-axis-coverage", "incomplete"),
        (_few_recordings, "incomplete", "scored-recordings-coverage", "incomplete"),
        (_nothing_judged, "incomplete", "judged-axis-accuracy", "incomplete"),
    ],
)
def test_the_audit_follows_the_scorer(
    tmp_path: Path, change: Any, verdict: str, requirement: str, status: str
) -> None:
    units = _all()
    change(units)
    scores, result = _build(tmp_path, units, RECORDINGS)
    expected = {"supported": "supported", "refuted": "refuted", "inconclusive": "incomplete"}
    assert expected[scores["verdict"]] == result.verdict == verdict
    assert {item.requirement_id: item.status for item in result.requirements}[requirement] == status


def test_exactly_half_of_the_axes_judged_meets_coverage_in_both(tmp_path: Path) -> None:
    judged = scorer.AxisInput()
    units = _all()
    for recording in RECORDINGS:
        units["cam0"][recording] = _unit("cam0", axes=(judged, judged, UNJUDGED))
        units["cam1"][recording] = _unit("cam1", axes=(judged, UNJUDGED, UNJUDGED))
    scores, result = _build(tmp_path, units, RECORDINGS)
    assert scores["judged_fraction"] == 0.5
    assert scores["verdict"] == result.verdict == "supported"
    coverage = {item.requirement_id: item for item in result.requirements}["judged-axis-coverage"]
    assert coverage.observed_value == 1.5
