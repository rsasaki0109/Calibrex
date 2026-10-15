"""The spent-data inventory must distinguish a saved-fit screen from a replay."""

import hashlib
import json
from pathlib import Path

import pytest
from tests.unit.test_score_hilti_camera_camera import ROOT, _load

development = _load("kitti_camera_lidar_v3_development")


def test_spent_55_drive_inventory_preserves_old_metrics_and_leaves_new_ones_unknown(
    tmp_path: Path,
) -> None:
    output = tmp_path / "inventory.json"
    assert development.main(["--root", str(ROOT), "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["analysis"] == "post hoc"
    assert report["status"] == "development_incomplete"
    assert "verdict" not in report
    assert report["proposed_rule_metrics"] is None
    assert report["coverage_change_percentage_points"] is None
    assert report["proposed_rule"]["default_changed"] is False
    metrics = report["current_v2_metrics"]
    assert (metrics["drives"], metrics["scored_drives"]) == (55, 52)
    assert metrics["flagged_baselines"] == []
    assert metrics["failed_drives"] == []
    assert len(metrics["inconclusive_drives"]) == 3
    assert (metrics["judged_axes"], metrics["scored_axes"], metrics["all_axes"]) == (125, 156, 165)
    assert metrics["judged_fraction_scored"] == 125 / 156
    assert metrics["judged_fraction_all"] == 125 / 165
    assert metrics["max_baseline_judged_error_deg"] == pytest.approx(0.45)
    assert metrics["max_turned_judged_error_deg"] == pytest.approx(1.502977473509024)
    assert metrics["trials"]["3deg"] == {
        "counted": 125,
        "flagged": 122,
        "flagged_fraction": 122 / 125,
        "judged_right": 0,
    }
    assert metrics["trials"]["1deg"]["flagged_fraction"] == 1.0
    diagnostic = report["saved_fit_diagnostics"]
    assert diagnostic["expected_evidence_runs"] == 385
    assert diagnostic["available_evidence_runs"] == 133
    assert sum(len(runs) for runs in diagnostic["missing_evidence_runs"].values()) == 252
    near_miss = next(
        fit
        for fit in diagnostic["selected_far_fits"]
        if fit["drive"] == "2011_09_28_0002" and fit["run"] == "inject_z+3"
    )
    assert not near_miss["saved_final_fit_clears_margin"]
    assert near_miss["axes_failing_proposed_margin"] == ["roll", "pitch"]
    assert near_miss["saved_judged_axis_control_t"]["roll"] == pytest.approx(2.6493418318574165)
    for relative, digest in report["provenance"]["source_files_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest


def test_inventory_refuses_duplicate_or_unlisted_drives(tmp_path: Path) -> None:
    source = tmp_path / "docs/assets/kitti_camera_lidar_v2"
    payload = {"drives": ["spent"], "scores": {"spent": {"state": "inconclusive"}}}
    for phase in development.PHASES:
        target = source / phase / "scores.json"
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="appears twice"):
        development.development_report(tmp_path)
    payload["drives"] = []
    (source / "development/scores.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="drive list differs"):
        development.development_report(tmp_path)


def test_inventory_refuses_a_weaker_proposal_before_reading_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="start_scan_min_control_t"):
        development.development_report(tmp_path, min_far_control_t=2.0)
