"""The post-hoc reading of the camera-camera audit: turned axis, judged or not, shifts."""

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from scipy.spatial.transform import Rotation
from tests.unit.test_score_hilti_camera_camera import _load, scorer

posthoc = _load("hilti_camera_camera_posthoc")
OPTIONS = {"max_std_deg": 0.1, "min_response": 0.5, "max_response": 1.5}
MISSED = "control_not_detected"


def _pair(
    run_dir: Path, run: str, pair: str, turn_deg: float, pitch_state: str, pitch: dict[str, float]
) -> dict[str, Any]:
    """A pair record whose candidate is turned ``turn_deg`` about the parent's y axis, with
    its evidence file. Roll is judged ``pass``; pitch is judged when ``pitch_state`` is a
    verdict and ``unchecked`` with that code otherwise; ``pitch`` is its evidence."""

    left, right = pair.split("-")
    judged = pitch_state in ("pass", "warn", "fail")
    evidence = f"{run}_evidence/{pair}.yaml"
    (run_dir / evidence).parent.mkdir(parents=True, exist_ok=True)
    roll = {**_estimate(0.0), "observable": True}
    axes = [
        {"name": "roll", "reason": "ok", **roll},
        {"name": "pitch", "observable": judged, "reason": "test", **pitch},
    ]
    (run_dir / evidence).write_text(
        yaml.safe_dump({"axes": axes, "options": OPTIONS, "recentred_by_deg": [0, turn_deg, 0]}),
        encoding="utf-8",
    )
    quat = Rotation.from_rotvec([0.0, turn_deg, 0.0], degrees=True).as_quat()
    verdicts = {"roll": "pass", **({"pitch": pitch_state} if judged else {})}
    return {
        "pair": "camera-camera",
        "frames": [scorer.frame(left), scorer.frame(right)],
        "status": "fail" if pitch_state == "fail" else "pass",
        "candidate_transform": {"rotation_quat_xyzw": quat.tolist()},
        "evidence_artifact": evidence,
        "axes": [
            {"name": name, "status": status, "candidate_error": 0.0, "tolerance": 0.5}
            for name, status in verdicts.items()
        ],
        "unchecked_axes": (
            [] if judged else [{"name": "pitch", "reason_code": pitch_state, "reason": "test"}]
        ),
    }


def _estimate(error: float, jackknife: float = 0.05, response: float = 1.0) -> dict[str, float]:
    # the estimator reports std_deg = hypot(jackknife, row-scale std)
    return {
        "candidate_error_deg": error,
        "std_deg": (jackknife**2 + 0.03**2) ** 0.5,
        "scale_std_deg": 0.03,
        "control_response": response,
    }


def _recording(directory: Path) -> None:
    directory.mkdir(parents=True)
    baseline = [
        _pair(directory, "baseline", "cam0-cam1", 0.0, "pass", _estimate(0.1)),
        _pair(directory, "baseline", "cam1-cam3", 0.0, "unobservable", _estimate(0.2, 0.2)),
        _pair(directory, "baseline", "cam1-cam4", 0.0, "pass", _estimate(-0.1)),
    ]
    # cam1 turned +1 deg about its y: cam0-cam1 (cam1 the child) reads +1 deg about the
    # parent's y; cam1-cam3 and cam1-cam4 (cam1 the parent) read -1 deg
    run = "inject_cam1_y+1"
    injected = [
        _pair(directory, run, "cam0-cam1", 1.0, "fail", _estimate(1.1)),
        _pair(directory, run, "cam1-cam3", -1.0, MISSED, _estimate(-0.75, 0.2, 0.3)),
        _pair(directory, run, "cam1-cam4", -1.0, "unobservable", _estimate(-0.1, 0.3)),
    ]
    group = {"frames": [scorer.frame(f"cam{k}") for k in range(5)], "status": "ambiguous"}
    runs = {
        "baseline": {"pairs": baseline},
        "inject_cam1_y+1": {"pairs": injected, "rotation_attribution": {"groups": [group]}},
    }
    for name, artifact in runs.items():
        (directory / f"{name}.json").write_text(json.dumps(artifact), encoding="utf-8")
    # the cam3 injection crashed: its pair-trial is missing, not counted as judged
    codes = {"baseline": 0, "inject_cam1_y+1": 0, "inject_cam3_y+1": 3}
    manifest = {"runs": {name: {"exit_code": code} for name, code in codes.items()}}
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_the_turn_is_read_in_the_pair_s_parent_axes() -> None:
    def record(rotvec_deg: list[float]) -> dict[str, Any]:
        quat = Rotation.from_rotvec(rotvec_deg, degrees=True).as_quat()
        return {"candidate_transform": {"rotation_quat_xyzw": quat.tolist()}}

    # a side camera 90 deg about the parent's y; the child turned 1 deg about its own y,
    # which is the parent's y as well
    base = Rotation.from_rotvec([0.0, 90.0, 0.0], degrees=True)
    turned = base * Rotation.from_rotvec([0.0, 1.0, 0.0], degrees=True)
    turn = posthoc.turn_in_pair_deg(
        record(list(base.as_rotvec(degrees=True))), record(list(turned.as_rotvec(degrees=True)))
    )
    assert turn == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)


def test_judged_unchecked_and_hidden_turned_axes_are_counted(tmp_path: Path) -> None:
    _recording(tmp_path / "a")
    result = posthoc.analyse(tmp_path, ["a"])
    summary = result["summary"]
    assert summary["pair_trials"] == 4
    assert summary["missing_injected_records"] == 1
    assert summary["turned_axis"] == {"pitch": 3}
    assert summary["turned_axis_judged"] == 1
    assert summary["judged_verdicts"] == {"fail": 1}
    assert summary["turned_axis_unchecked"] == 2
    assert summary["unchecked_jackknife_above_bound"] == 2
    assert summary["unchecked_control_failed"] == 1
    # cam1-cam3's estimate moved -0.95 deg with a -1 deg turn; cam1-cam4's did not move
    assert summary["unchecked_estimate_moved_with_the_turn"] == 1
    assert summary["unchecked_moved_abs_shift_deg"] == [0.95]
    # cam1-cam4 judged pitch on the baseline and lost it under the turn
    assert summary["unchecked_but_judged_on_the_baseline"] == 1
    assert summary["hidden_moved_with_the_turn"] == 0
    assert summary["hidden_jackknife_std_deg"] == [0.3]
    assert summary["ambiguous_trials_flagged_only_by"] == {"cam0-cam1": 1}
    assert summary["baseline_scored_units_with_turned_axis_unchecked"] == 1
    assert summary["their_abs_error_deg_max"] == 0.2
