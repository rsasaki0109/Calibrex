import json
from pathlib import Path

import pytest
from tools import kitti_camera_lidar_v3_rule_screen as screen

BATCH = (
    Path(__file__).resolve().parents[2] / "docs/assets/kitti_camera_lidar_v3/spent_batch_20261031"
)


def _margin_reports() -> list[dict]:
    attempts = json.loads((BATCH / "summary.json").read_text())["attempts"]
    reports = []
    for drive, attempt in attempts.items():
        path = BATCH / "drives" / drive / f"attempt_{attempt}" / "comparison" / "comparison.json"
        reports.append(json.loads(path.read_text()))
    return reports


def test_screen_reproduces_the_published_margin_tradeoff_and_rejects_new_errors():
    result = screen.screen(_margin_reports())
    assert result["proposed_options"] == {"start_scan_min_control_t": 3.5}
    assert len(result["large_errors"]["current"]) == 10
    assert len(result["large_errors"]["proposed"]) == 5
    assert result["large_errors"]["new_under_proposed"] == [["2011_09_26_0036", "z:3", "yaw"]]
    checks = result["checks"]
    assert checks["turned_judged_accuracy"] and checks["turn_3deg_flagged"]
    assert checks["fewer_large_errors"] and not checks["no_new_large_errors"]
    assert result["verdict"] == "not_carried_forward"


def test_screen_requires_each_spent_drive_once_under_one_rule():
    reports = _margin_reports()
    with pytest.raises(ValueError, match="missing"):
        screen.screen(reports[:-1])
    with pytest.raises(ValueError, match="more than one report"):
        screen.screen([*reports, reports[0]])
    changed = json.loads(json.dumps(reports[0]))
    changed["proposed_options"] = {"start_agreement_deg": 0.5}
    with pytest.raises(ValueError, match="different proposed rules"):
        screen.screen([changed, *reports[1:]])
