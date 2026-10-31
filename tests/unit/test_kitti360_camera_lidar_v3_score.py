import pytest
from tools import kitti360_camera_lidar_v3_score as score

AXES = ("roll", "pitch", "yaw")


def _segment(verdict: str = "pass", turned: str = "fail", flagged: bool = True) -> dict:
    trial = {
        "counted": True,
        "flagged": flagged,
        "judged_right": False,
        "verdict": turned,
        "judged_errors_deg": {"roll": 0.3},
    }
    return {
        "state": "scored",
        "verdict": verdict,
        "axes": {name: {"judged": True, "error_deg": 0.2} for name in AXES},
        "trials": {f"{a}:{d}": dict(trial) for d in (3, 1) for a in "xyz"},
    }


PRIMARY = {
    "no-failed-segments": {"max": 0},
    "vendor-calibration-not-flagged": {"max": 0},
    "judged-axis-accuracy": {"max": 0.75},
    "judged-axis-coverage": {"min": 0.5},
    "turned-judged-axis-accuracy": {"max": 1.5},
    "turn-3deg-flagged": {"max": 0.1},
    "turn-3deg-never-judged-right": {"max": 0},
    "turn-1deg-flagged": {"max": 0.25},
    "scored-segments-coverage": {"min": 2},
}
SECONDARY = {
    "no-failed-segments": {"max": 0},
    "judged-axis-coverage": {"min": 0.5},
    "scored-segments-coverage": {"min": 2},
    "baseline-warned": {"max": 0.2},
    "judged-axis-accuracy": {"max": 0.75},
    "turned-judged-axis-accuracy": {"max": 1.5},
    "turn-3deg-run-flagged": {"max": 0.1},
    "turn-1deg-run-flagged": {"max": 0.25},
}


def test_all_passing_requirements_support_both_claims() -> None:
    scores = {"a": _segment(), "b": _segment()}
    observed = score.observe(scores, scores)
    assert score.judge(observed["primary"], PRIMARY)["verdict"] == "supported"
    assert score.judge(observed["secondary"], SECONDARY)["verdict"] == "supported"


def test_a_flagged_vendor_baseline_refutes_and_low_coverage_is_inconclusive() -> None:
    flagged = {"a": _segment(verdict="warn"), "b": _segment()}
    observed = score.observe(flagged, flagged)
    result = score.judge(observed["primary"], PRIMARY)
    assert result["verdict"] == "refuted"
    assert result["requirements"]["vendor-calibration-not-flagged"]["status"] == "fail"
    # The warning tolerates one warned baseline in two only above its allowance.
    assert score.judge(observed["secondary"], SECONDARY)["verdict"] == "refuted"
    few = {"a": _segment()}
    observed = score.observe(few, few)
    result = score.judge(observed["primary"], PRIMARY)
    assert result["verdict"] == "inconclusive"
    assert result["requirements"]["scored-segments-coverage"]["status"] == "fail"


def test_run_level_warning_counts_for_the_secondary_claim_only() -> None:
    unflagged_axis = {
        "a": _segment(turned="warn", flagged=False),
        "b": _segment(turned="warn", flagged=False),
    }
    observed = score.observe(unflagged_axis, unflagged_axis)
    assert observed["primary"]["turn-3deg-flagged"] == 1.0
    assert observed["secondary"]["turn-3deg-run-flagged"] == 0.0


def test_requirements_must_match_and_be_single_bounds() -> None:
    scores = {"a": _segment(), "b": _segment()}
    observed = score.observe(scores, scores)["primary"]
    with pytest.raises(ValueError, match="requirements differ"):
        score.judge(observed, {k: v for k, v in PRIMARY.items() if k != "turn-1deg-flagged"})
    with pytest.raises(ValueError, match="single min or max"):
        score.judge(observed, PRIMARY | {"turn-1deg-flagged": {"below": 0.25}})
