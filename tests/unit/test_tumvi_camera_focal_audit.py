"""The TUM-VI camera-focal audit harness: candidates, record reading and the pass rule."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from tools import tumvi_camera_focal_audit as harness

ROOT = Path(__file__).resolve().parents[2]
CAMCHAIN = ROOT / "docs/assets/tumvi_camera_focal/camchain.yaml"
DEV = ROOT / "docs/assets/tumvi_camera_focal/development"


def test_candidates_scale_only_the_focal_lengths(tmp_path: Path) -> None:
    paths = harness.write_candidates(CAMCHAIN, tmp_path)
    kalibr = yaml.safe_load(CAMCHAIN.read_text())
    for name, scale in harness.VARIANTS.items():
        chain = yaml.safe_load(paths[name].read_text())
        for camera in harness.CAMERAS:
            fx, fy, cx, cy = chain[camera]["intrinsics"]
            kfx, kfy, kcx, kcy = kalibr[camera]["intrinsics"]
            assert (fx, fy) == pytest.approx((kfx * scale, kfy * scale))
            assert (cx, cy) == (kcx, kcy)
            assert chain[camera]["T_cam_imu"] == kalibr[camera]["T_cam_imu"]
    bad = tmp_path / "other.yaml"
    bad.write_text(CAMCHAIN.read_text() + "\n# edited\n")
    with pytest.raises(SystemExit, match="pinned"):
        harness.write_candidates(bad, tmp_path / "x")


def test_records_count_axes_only_when_check_gave_a_verdict() -> None:
    kalibr = harness.focal_records(json.loads((DEV / "room1_kalibr.json").read_text()))
    assert kalibr["cam0"]["status"] == "pass" and kalibr["cam0"]["judged_axes"] == ["fx", "fy"]
    # cam1's estimator failed its own held-out check: inconclusive, nothing judged
    assert kalibr["cam1"]["status"] == "inconclusive" and kalibr["cam1"]["judged_axes"] == []
    assert kalibr["cam1"]["axes"]["fx"]["scale"] == pytest.approx(0.9988, abs=1e-4)


def _case(status: str, judged: bool = True) -> dict:
    return {"status": status, "judged_axes": ["fx"] if judged else []}


def test_the_pass_rule() -> None:
    sessions = list(harness.SESSIONS)
    good = {
        s: {
            "kalibr": {c: _case("pass") for c in harness.CAMERAS},
            "plus3": {c: _case("fail") for c in harness.CAMERAS},
            "minus3": {c: _case("inconclusive", judged=False) for c in harness.CAMERAS},
        }
        for s in sessions
    }
    verdict = harness.verdicts(good)
    assert verdict["honesty"]["verdict"] == "supported"
    assert verdict["detection"]["verdict"] == "supported"  # half the perturbed cases fail
    one_false_fail = json.loads(json.dumps(good))
    one_false_fail[sessions[0]]["kalibr"]["cam0"] = _case("fail")
    assert harness.verdicts(one_false_fail)["honesty"]["verdict"] == "refuted"
    one_pass = json.loads(json.dumps(good))
    one_pass[sessions[0]]["minus3"]["cam1"] = _case("pass")
    assert harness.verdicts(one_pass)["detection"]["verdict"] == "refuted"
    nothing_judged = json.loads(json.dumps(good))
    for s in sessions:
        nothing_judged[s]["kalibr"] = {c: _case("inconclusive", False) for c in harness.CAMERAS}
    assert harness.verdicts(nothing_judged)["honesty"]["verdict"] == "refuted"
    missing = {**good, sessions[0]: {"error": "x", "kalibr": {}, "plus3": {}, "minus3": {}}}
    assert harness.verdicts(missing)["honesty"]["verdict"] == "refuted"
