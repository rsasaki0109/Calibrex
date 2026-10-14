"""The camera-IMU v2 builder: v1's gates on five recordings, scored by the v1 scorer."""

import json
from pathlib import Path

from tests.unit.test_score_hilti_camera_imu import _all, _load, _unit, scorer

builder = _load("build_hilti_camera_imu_v2_audit")
RECORDINGS = ["exp15", "exp16", "exp23p0", "exp23p1", "exp23p2"]


def _run(tmp_path: Path, units: dict, recordings: list[str]):  # type: ignore[no-untyped-def]
    preregistration = tmp_path / "prereg.yaml"
    preregistration.write_text("claim: test\n", encoding="utf-8")
    scores_path = tmp_path / "scores.json"
    scores_path.write_text(json.dumps(scorer.score_units(units, recordings)), encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    return builder.build(scores, scores_path, preregistration, out)


def test_the_v2_builder_keeps_v1_s_gates_on_five_recordings(tmp_path: Path) -> None:
    result = _run(tmp_path, _all(RECORDINGS), RECORDINGS)
    assert result.verdict == "supported"
    assert result.protocol_id == "hilti-camera-imu-v2"
    assert (tmp_path / "out" / "hilti_camera_imu_v2_sota_audit.yaml").is_file()
    ids = [item.requirement_id for item in result.requirements]
    assert ids == [
        "rotation-vs-kalibr",
        "time-offset-vs-kalibr",
        "constrained",
        "no-failed-units",
        "cross-recording-consistency",
        "relative-cam0-cam1",
        "scored-recordings-coverage",
    ]


def test_two_scored_recordings_are_incomplete_and_a_bad_unit_refutes(tmp_path: Path) -> None:
    assert _run(tmp_path, _all(RECORDINGS[:2]), RECORDINGS[:2]).verdict == "incomplete"
    bad = _all(RECORDINGS)
    bad["cam0"]["exp23p1"] = _unit(0, offset_deg=1.2)
    assert _run(tmp_path, bad, RECORDINGS).verdict == "refuted"
