"""The TUM-VI camera-imu v2 harness: --fail-on partial records and the pass rule."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tools import tumvi_camera_imu_v2_audit as harness

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "docs/assets/tumvi_camera_imu/evaluation"


def test_the_command_runs_check_with_fail_on_partial_and_a_shared_cache() -> None:
    command = harness.check_command(Path("BAG"), Path("C"), Path("O"), Path("CACHE"))
    assert command[command.index("--fail-on") + 1] == "partial"
    assert command[command.index("--cache-dir") + 1] == "CACHE"
    assert command[command.index("--pairs") + 1] == "camera-imu"
    assert "--no-cache" not in command


def test_a_partial_pass_is_flagged_and_a_full_pass_is_not() -> None:
    # v1's corridor2: cam1 left pitch unchecked; its pitch-turned case passed as partial
    turned = harness.imu_records(json.loads((V1 / "corridor2_ry2.json").read_text()))
    assert (turned["cam1"]["status"], turned["cam1"]["coverage"]) == ("pass", "partial")
    assert turned["cam1"]["unchecked_axes"] == ["pitch"]
    assert turned["cam1"]["exit_status"] == 1
    assert harness.detected(turned["cam0"], "pitch") and not harness.detected(
        turned["cam1"], "pitch"
    )
    kalibr = harness.imu_records(json.loads((V1 / "corridor1_kalibr.json").read_text()))
    assert harness.full_pass(kalibr["cam0"]) and kalibr["cam0"]["exit_status"] == 0


def _case(status: str, coverage: str = "full", failing: str | None = None) -> dict:
    axes = {
        name: {"status": "fail" if name == failing else "pass"} for name in harness.AXIS_VECTORS
    }
    partial = int(coverage == "partial")
    return {
        "status": status,
        "coverage": coverage,
        "exit_status": harness.check_exit_status(status, partial, "partial"),
        "judged_axes": list(axes),
        "axes": axes,
    }


def _scores(kalibr: dict, turned: dict[str, dict]) -> dict:
    return {
        s: {"kalibr": dict.fromkeys(harness.CAMERAS, kalibr)}
        | {v: dict.fromkeys(harness.CAMERAS, turned[v]) for v in ("rx2", "ry2", "rz2")}
        for s in harness.SESSIONS
    }


def test_the_pass_rule() -> None:
    turned = {"rx2": _case("fail", failing="roll"), "ry2": _case("fail", failing="pitch")}
    good = _scores(_case("pass"), {**turned, "rz2": _case("pass", "partial")})
    verdict = harness.verdicts(good)
    # a partial pass of a turned candidate is flagged under --fail-on partial
    assert verdict["detection"]["turned_unflagged"] == 0
    assert verdict["detection"]["turned_detected_fraction"] == pytest.approx(2 / 3)
    assert verdict["detection"]["verdict"] == "supported"
    assert verdict["honesty"]["verdict"] == "supported"
    unflagged = _scores(_case("pass"), {**turned, "rz2": _case("pass")})
    assert harness.verdicts(unflagged)["detection"]["verdict"] == "refuted"
    partial_kalibr = _scores(
        _case("pass", "partial"), {**turned, "rz2": _case("fail", failing="yaw")}
    )
    # Kalibr never fails, but a partial pass is not a full pass: honesty needs half full
    assert harness.verdicts(partial_kalibr)["honesty"]["verdict"] == "refuted"
    failed_kalibr = json.loads(json.dumps(good))
    failed_kalibr["slides1"]["kalibr"]["cam0"] = _case("fail", failing="roll")
    assert harness.verdicts(failed_kalibr)["honesty"]["verdict"] == "refuted"
    missing = {**good, "slides2": {"error": "x", **{name: {} for name in harness.VARIANTS}}}
    assert harness.verdicts(missing)["honesty"]["verdict"] == "refuted"
    assert harness.verdicts(missing)["detection"]["verdict"] == "refuted"


def test_v1_corridors_under_the_v2_rule() -> None:
    """Development: the five spent corridors, scored by v2's rule."""

    scores = {}
    for sequence in ("corridor1", "corridor2", "corridor3", "corridor4", "corridor5"):
        scores[sequence] = {
            name: harness.imu_records(json.loads((V1 / f"{sequence}_{name}.json").read_text()))
            for name in harness.VARIANTS
        }
    verdict = harness.verdicts(scores, sessions=dict.fromkeys(scores, (0, "")))
    assert verdict["honesty"]["kalibr_fails"] == 0
    assert verdict["honesty"]["kalibr_full_pass_fraction"] == pytest.approx(0.6)
    assert verdict["detection"]["turned_unflagged"] == 0
    assert verdict["detection"]["turned_detected_fraction"] == pytest.approx(26 / 30)
