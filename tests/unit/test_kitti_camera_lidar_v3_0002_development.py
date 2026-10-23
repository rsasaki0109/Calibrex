"""Integrity and native/replay parity on the spent v2 gate-failing recording."""
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest
from tests.unit.test_score_hilti_camera_camera import _load

from calibrex.core.camera_lidar_edge import load_camera_lidar_edge

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs/assets/kitti_camera_lidar_v3/native_development_20110928_0002"
replay = _load("kitti_camera_lidar_v3_replay")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("directory", ["attempt_1", "confirmation_yaw3", "confirmation_pitch3"])
def test_published_receipts_and_sources_are_byte_bound(directory: str) -> None:
    root = ASSETS / directory
    manifest = _json(root / "manifest.json")
    jsonschema.validate(manifest, _json(ASSETS / "publication.schema.json"),
                        format_checker=jsonschema.FormatChecker())
    names = [item["path"] for item in manifest["files"]]
    assert len(names) == len(set(names))
    assert set(names) == {
        p.relative_to(root).as_posix() for p in root.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    for item in manifest["files"]:
        path = root / item["path"]
        assert path.resolve().is_relative_to(root.resolve())
        assert path.stat().st_size == item["bytes"] and _sha(path) == item["sha256"]
        assert path.suffix not in {".npz", ".png", ".bin", ".db3", ".zip"}
    assert manifest["provenance"]["generator_sha256"] == _sha(root / "publication-source.py.txt")
    report = _json(root / "report.json")
    jsonschema.validate(report, _json(root / "report.schema.json"))
    provenance = report["provenance"]
    assert provenance["development_only"] is True and provenance["new_accuracy_claim"] is False
    assert provenance["new_evaluation_started"] is False
    assert provenance["drive"] == "2011_09_28_0002"
    producer_hash = provenance.get("producer_sha256", provenance.get("generator_sha256"))
    assert producer_hash == _sha(root / "producer-source.py.txt")


def test_real_comparison_keeps_full_recipe_and_remains_incomplete() -> None:
    root = ASSETS / "attempt_1"
    report = _json(root / "report.json")
    assert (report["status"], report["stage"]) == ("PASS_PILOT_ONLY", "complete")
    fetch = _json(root / "fetch.json")
    assert fetch["size_bytes"] == 1458258535
    assert fetch["etag"] == "57c9f403568ce97a5aeb70180b029c93-174"
    assert fetch["max_frames"] == 600 and fetch["crc_verified"] is True
    assert fetch["frames"] == {"image_02": 376, "velodyne_points": 376, "oxts": 376}
    assert fetch["fetched_bytes"] <= report["provenance"]["fetch_cap_bytes"] == 2 * 1024**3
    assert report["details"]["fetch_sha256"] == _sha(root / "fetch.json")
    assert report["details"]["calibration"]["sha256"] == (
        "510b643c00adb7bc6b39616f1ed9b83cd0d689620d52ef5bc43ec019da6030d1"
    )
    for source, capsule in [("tools/kitti_camera_lidar_v3_replay.py", "replay"),
                            ("tools/kitti_drive_fetch.py", "fetch"),
                            ("src/calibrex/data/kitti_raw_to_rosbag2.py", "converter"),
                            ("src/calibrex/evaluation/camera_lidar_edge.py", "estimator")]:
        assert report["provenance"]["frozen_modules_sha256"][source] == _sha(
            root / f"{capsule}-source.py.txt",
        )
    comparison = _json(root / "comparison/comparison.json")
    assert comparison["status"] == "incomplete" and comparison["errors"] == {}
    assert comparison["drives"] == ["2011_09_28_0002"] and comparison["audit_thresholds"] is None
    current_metrics = comparison["metrics"]["current"]
    proposed_metrics = comparison["metrics"]["proposed"]
    assert current_metrics["max_turned_judged_error_deg"] > 1.5
    assert proposed_metrics["max_turned_judged_error_deg"] == pytest.approx(0.1)
    assert current_metrics["trials"]["3deg"]["flagged"] == 1
    assert proposed_metrics["trials"]["3deg"]["flagged"] == 0
    assert current_metrics["judged_axes"] == proposed_metrics["judged_axes"] == 1
    capture = _json(root / "capture/capture_manifest.json")
    assert capture["complete"] and all(r["ok"] for r in capture["runs"].values())
    assert set(capture["runs"]) == set(replay.RUNS)
    assert comparison["provenance"]["capture_manifests_sha256"] == {
        "2011_09_28_0002": _sha(root / "capture/capture_manifest.json"),
    }
    for run in replay.RUNS:
        snapshot = _json(root / f"capture/{run}/snapshot.json")
        assert snapshot["arrays_sha256"] == capture["files_sha256"][f"{run}/arrays.npz"]
        captured = load_camera_lidar_edge(root / f"capture/{run}/captured_edge.json")
        current = load_camera_lidar_edge(root / f"comparison/2011_09_28_0002/current/{run}.json")
        proposed = load_camera_lidar_edge(root / f"comparison/2011_09_28_0002/proposed/{run}.json")
        assert replay._same(captured.model_dump(exclude={"provenance"}),
                            current.model_dump(exclude={"provenance"}))
        assert current.provenance.input_sha256 == proposed.provenance.input_sha256


@pytest.mark.parametrize("control,run,selection", [
    ("yaw3", "inject_z+3", "Preselected"), ("pitch3", "inject_y+3", "Post hoc"),
])
def test_selected_controls_are_confirmed_by_the_native_driver(
    control: str, run: str, selection: str,
) -> None:
    root = ASSETS / f"confirmation_{control}"
    report = _json(root / "report.json")
    assert report["status"] == "PASS_SELECTED_CONTROL_ONLY" and report["error"] is None
    provenance = report["provenance"]
    paired = ASSETS / "attempt_1"
    assert provenance["paired_report_sha256"] == _sha(paired / "report.json")
    assert provenance["comparison_sha256"] == _sha(paired / "comparison/comparison.json")
    assert provenance["run"] == run and provenance["selection"].startswith(selection)
    for key, capsule in [("replay_tool_sha256", "replay"),
                         ("estimator_sha256", "estimator"), ("cli_sha256", "cli")]:
        assert provenance[key] == _sha(root / f"{capsule}-source.py.txt")
    expected = paired / f"comparison/2011_09_28_0002/proposed/{run}.json"
    assert report["details"]["expected_artifact_sha256"] == _sha(expected)
    artifacts = list((root / "evidence").glob("*.yaml"))
    assert len(artifacts) == 1
    assert report["details"]["actual_artifact_sha256"] == _sha(artifacts[0])
    actual = load_camera_lidar_edge(artifacts[0])
    replayed = load_camera_lidar_edge(expected)
    assert replay._same(actual.model_dump(exclude={"provenance"}),
                        replayed.model_dump(exclude={"provenance"}))
    assert report["details"]["replay_verdict_parity"] is True
    assert report["details"]["numerical_parity_atol"] == 1e-9
    assert report["details"]["verdict"] == "inconclusive"
    assert report["details"]["judged"] == {}
