"""The TUM-VI camera-imu audit harness: candidates, record reading and the pass rule."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation
from tools import tumvi_camera_imu_audit as harness

ROOT = Path(__file__).resolve().parents[2]
CAMCHAIN = ROOT / "docs/assets/tumvi_camera_focal/camchain.yaml"
DEV = ROOT / "docs/assets/tumvi_camera_imu/development"
# the Kalibr candidate's camera-imu ran in the camera-focal development run
KALIBR = ROOT / "docs/assets/tumvi_camera_focal/development/room1_kalibr.json"


def test_candidates_turn_only_the_rotation_about_one_camera_axis(tmp_path: Path) -> None:
    paths = harness.write_candidates(CAMCHAIN, tmp_path)
    kalibr = yaml.safe_load(CAMCHAIN.read_text())
    for name, axis in harness.VARIANTS.items():
        chain = yaml.safe_load(paths[name].read_text())
        for camera in harness.CAMERAS:
            assert chain[camera]["intrinsics"] == kalibr[camera]["intrinsics"]
            candidate = np.array(chain[camera]["T_cam_imu"])
            reference = np.array(kalibr[camera]["T_cam_imu"])
            # check's error: rotvec(R_candidate R_reference^T), in the camera frame
            error = np.degrees(
                Rotation.from_matrix(candidate[:3, :3] @ reference[:3, :3].T).as_rotvec()
            )
            expected = np.zeros(3) if axis is None else 2.0 * np.array(harness.AXIS_VECTORS[axis])
            assert error == pytest.approx(expected, abs=1e-9)
            # the IMU's position seen from the camera's rotated frame is unchanged
            assert -candidate[:3, :3].T @ candidate[:3, 3] == pytest.approx(
                -reference[:3, :3].T @ reference[:3, 3]
            )
    bad = tmp_path / "other.yaml"
    bad.write_text(CAMCHAIN.read_text() + "\n# edited\n")
    with pytest.raises(SystemExit, match="pinned"):
        harness.write_candidates(bad, tmp_path / "x")


def test_records_read_the_development_runs() -> None:
    kalibr = harness.imu_records(json.loads(KALIBR.read_text()))
    assert kalibr["cam0"]["status"] == "pass"
    assert kalibr["cam0"]["judged_axes"] == ["roll", "pitch", "yaw"]
    turned = harness.imu_records(json.loads((DEV / "room1_rx2.json").read_text()))
    for camera in harness.CAMERAS:
        assert harness.detected(turned[camera], "roll")
        assert not harness.detected(turned[camera], "yaw")
        assert turned[camera]["axes"]["roll"]["candidate_error"] == pytest.approx(2.0, abs=0.1)


def _case(status: str, failing: str | None = None, judged: bool = True) -> dict:
    axes = {
        name: {"status": "fail" if name == failing else "pass"} for name in harness.AXIS_VECTORS
    }
    return {"status": status, "judged_axes": list(axes) if judged else [], "axes": axes}


def test_the_pass_rule() -> None:
    sessions = list(harness.SESSIONS)
    good = {
        s: {
            "kalibr": {c: _case("pass") for c in harness.CAMERAS},
            "rx2": {c: _case("fail", "roll") for c in harness.CAMERAS},
            "ry2": {c: _case("fail", "pitch") for c in harness.CAMERAS},
            "rz2": {c: _case("inconclusive", judged=False) for c in harness.CAMERAS},
        }
        for s in sessions
    }
    verdict = harness.verdicts(good)
    assert verdict["honesty"]["verdict"] == "supported"
    assert verdict["detection"]["verdict"] == "supported"  # two thirds detected, none passed
    assert verdict["detection"]["perturbed_detected_fraction"] == pytest.approx(2 / 3)
    assert verdict["detection"]["perturbed_passes"] == 0
    few = json.loads(json.dumps(good))
    for s in sessions:
        few[s]["ry2"] = {c: _case("inconclusive", judged=False) for c in harness.CAMERAS}
    assert harness.verdicts(few)["detection"]["verdict"] == "refuted"  # one third detected


def test_detection_needs_the_fail_on_the_turned_axis() -> None:
    sessions = list(harness.SESSIONS)
    cases = {
        s: {
            "kalibr": {c: _case("pass") for c in harness.CAMERAS},
            "rx2": {c: _case("fail", "roll") for c in harness.CAMERAS},
            "ry2": {c: _case("fail", "yaw") for c in harness.CAMERAS},  # the wrong axis
            "rz2": {c: _case("fail", "yaw") for c in harness.CAMERAS},
        }
        for s in sessions
    }
    assert harness.verdicts(cases)["detection"]["perturbed_detected_fraction"] == pytest.approx(
        2 / 3
    )
    one_pass = json.loads(json.dumps(cases))
    one_pass[sessions[0]]["rz2"]["cam1"] = _case("pass")
    assert harness.verdicts(one_pass)["detection"]["verdict"] == "refuted"


def test_honesty_fails_on_a_false_fail_nothing_judged_or_a_missing_sequence() -> None:
    sessions = list(harness.SESSIONS)
    good = {
        s: {name: {c: _case("pass" if axis is None else "fail", axis) for c in harness.CAMERAS}
            for name, axis in harness.VARIANTS.items()}
        for s in sessions
    }  # fmt: skip
    assert harness.verdicts(good)["honesty"]["verdict"] == "supported"
    assert harness.verdicts(good)["detection"]["verdict"] == "supported"
    one_false_fail = json.loads(json.dumps(good))
    one_false_fail[sessions[0]]["kalibr"]["cam0"] = _case("fail", "roll")
    assert harness.verdicts(one_false_fail)["honesty"]["verdict"] == "refuted"
    nothing_judged = json.loads(json.dumps(good))
    for s in sessions:
        nothing_judged[s]["kalibr"] = {
            c: _case("inconclusive", judged=False) for c in harness.CAMERAS
        }
    assert harness.verdicts(nothing_judged)["honesty"]["verdict"] == "refuted"
    missing = {**good, sessions[0]: {"error": "x", **{name: {} for name in harness.VARIANTS}}}
    assert harness.verdicts(missing)["honesty"]["verdict"] == "refuted"
    assert harness.verdicts(missing)["detection"]["verdict"] == "refuted"
