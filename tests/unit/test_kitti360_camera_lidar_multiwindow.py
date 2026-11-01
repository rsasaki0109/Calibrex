import json
from pathlib import Path

import pytest
from tools import kitti360_camera_lidar_multiwindow as mw

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "docs/assets/kitti_camera_lidar_v3/kitti360_dev_20261031"


def test_too_few_judging_windows_leave_the_axis_unchecked() -> None:
    assert mw.combine_axis([(0.1, 0.05)], 4) is None
    assert mw.combine_axis([(0.1, 0.05), (0.2, 0.05)], 4) is not None
    assert mw.combine_axis([(0.1, 0.05), (0.2, 0.05)], 5) is None


def test_median_resists_one_biased_window() -> None:
    result = mw.combine_axis([(0.05, 0.04), (0.0, 0.04), (-0.6, 0.04)], 3)
    assert result is not None
    assert result["candidate_error_deg"] == 0.0
    assert result["status"] == "pass"


def test_a_shared_offset_is_still_flagged_and_the_floor_holds() -> None:
    shared = [(1.0, 0.03), (1.05, 0.03), (0.95, 0.03)]
    result = mw.combine_axis(shared, 3)
    assert result is not None and result["tolerance_deg"] == pytest.approx(0.5)
    assert result["status"] == "warn"
    assert mw.combine_axis([(v + 0.3, s) for v, s in shared], 3)["status"] == "fail"


def test_between_window_spread_widens_the_tolerance() -> None:
    spread = [(-0.9, 0.03), (-0.6, 0.03), (0.0, 0.03), (0.6, 0.03), (0.9, 0.03)]
    result = mw.combine_axis(spread, 5)
    assert result is not None
    assert result["between_window_sd_deg"] > result["within_window_std_deg"]
    assert result["tolerance_deg"] > 0.5


def test_one_window_keeps_the_shipped_status() -> None:
    assert mw.combine_axis([(0.6, 0.04)], 1, "warn")["status"] == "warn"
    assert mw.combine_axis([], 1, None) is None


def test_single_window_measures_equal_the_v2_scorer_on_the_spent_segments() -> None:
    windows = mw.load_windows(DEV, "spent")
    result = mw.evaluate(windows, mw._tool("kitti360_camera_lidar_v3_batch").segments("all"))
    single = result["single_window"]
    gates = json.loads((DEV / "gates.json").read_text())["primary"]["metrics"]
    assert single["vendor_flagged"] == gates["flagged_baselines"] == ["0002-s04"]
    assert single["judged_axes"] == gates["judged_axes"]
    assert single["max_baseline_judged_error_deg"] == pytest.approx(
        gates["max_baseline_judged_error_deg"]
    )
    for degrees in (3, 1):
        trials = gates["trials"][f"{degrees}deg"]
        assert single["detection"][f"{degrees}deg"]["counted"] == trials["counted"]
        assert single["detection"][f"{degrees}deg"]["missed"] == (
            trials["counted"] - trials["flagged"]
        )
    assert result["triple_level"]["recordings"] == 7
    assert result["sequence_level"]["recordings"] == 2
