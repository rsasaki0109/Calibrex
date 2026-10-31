import json
from pathlib import Path

import pytest
from tools import kitti360_camera_lidar_v3_gates as gates

ROOT = Path(__file__).resolve().parents[2]


def test_the_rule_reproduces_v2s_frozen_thresholds_from_v2s_development_scores() -> None:
    scores = json.loads(
        (ROOT / "docs/assets/kitti_camera_lidar_v2/development/scores.json").read_text()
    )
    derived = gates.primary(scores["scores"])["gates"]
    assert derived["judged-axis-accuracy"]["threshold_deg"] == 0.75
    assert derived["turned-judged-axis-accuracy"]["threshold_deg"] == 1.5
    assert derived["judged-axis-coverage"]["threshold"] == 0.5
    assert derived["turn-3deg-flagged"]["development"] == "3 of 89"
    assert derived["turn-3deg-flagged"]["allowance"] == 0.1
    assert derived["turn-1deg-flagged"]["allowance"] == 0.25
    # v2 kept 12 of 18 evaluation drives; the same 1.0 / 1.5 rate keeps 28 of 42 segments.
    assert derived["scored-segments-coverage"]["threshold_segments"] == 28


@pytest.mark.parametrize(
    ("misses", "trials", "expected"),
    [(0, 30, 0.1), (0, 60, 0.05), (3, 89, 0.1), (1, 22, 0.2), (22, 22, 1.0)],
)
def test_binomial_allowance(misses: int, trials: int, expected: float) -> None:
    assert gates.allowance(misses, trials)["allowance"] == pytest.approx(expected)


def test_rounding_keeps_exact_steps() -> None:
    assert gates.round_up(0.75, 0.25) == 0.75
    assert gates.round_up(0.7500001, 0.25) == 1.0
    assert gates.round_down(0.5, 0.25) == 0.5
    with pytest.raises(ValueError):
        gates.error_gate(None)


def _trial(verdict: str, *, counted: bool = True, flagged: bool = False) -> dict:
    return {
        "counted": counted,
        "flagged": flagged,
        "judged_right": False,
        "verdict": verdict,
        "judged_errors_deg": {"pitch": 0.2},
    }


def test_secondary_counts_run_level_warnings_not_turned_axis_status() -> None:
    axes = {name: {"judged": True, "error_deg": 0.1} for name in ("roll", "pitch", "yaw")}
    trials = {"x:3": _trial("warn"), "y:3": _trial("pass"), "z:3": _trial("fail", flagged=True)}
    trials |= {"x:1": _trial("warn"), "y:1": _trial("warn"), "z:1": _trial("pass", counted=False)}
    scores = {
        "a": {"state": "scored", "verdict": "pass", "axes": axes, "trials": trials},
        "b": {"state": "scored", "verdict": "warn", "axes": axes, "trials": trials},
    }
    derived = gates.secondary(scores)["gates"]
    assert derived["baseline-warned"]["development"] == "1 of 2"
    assert derived["turn-3deg-run-flagged"]["development"] == "2 of 6"
    assert derived["turn-1deg-run-flagged"]["development"] == "0 of 4"
    assert derived["judged-axis-accuracy"]["threshold_deg"] == 0.25
