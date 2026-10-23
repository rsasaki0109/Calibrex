"""Audit the real spent-data pilot's reports without loading sensor payloads."""
import hashlib
import json
from pathlib import Path

import jsonschema
from tests.unit.test_score_hilti_camera_camera import _load

from calibrex.core.camera_lidar_edge import load_camera_lidar_edge

ASSETS = Path(__file__).resolve().parents[2] / "docs/assets/kitti_camera_lidar_v3/native_pilot_0005"
replay = _load("kitti_camera_lidar_v3_replay")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_all_published_pilot_bytes_and_frozen_sources_are_bound() -> None:
    for attempt in ("attempt_1", "attempt_2"):
        root = ASSETS / attempt
        manifest = _json(root / "manifest.json")
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
            assert path.suffix not in {".npz", ".png", ".bin", ".db3", ".mcap", ".zip"}
        assert manifest["provenance"]["generator_sha256"] == _sha(
            root / "publication-source.py.txt",
        )
        report = _json(root / "report.json")
        jsonschema.validate(report, _json(root / "report.schema.json"))
        provenance = report["provenance"]
        assert provenance["development_only"] is True
        assert provenance["new_evaluation_started"] is False
        assert provenance["new_accuracy_claim"] is False
        assert provenance["drive"] == "2011_09_26_0005"
        assert provenance["producer_sha256"] == _sha(root / "producer-source.py.txt")
        for source, capsule in [("tools/kitti_camera_lidar_v3_replay.py", "replay"),
                                ("tools/kitti_drive_fetch.py", "fetch"),
                                ("src/calibrex/evaluation/camera_lidar_edge.py", "estimator")]:
            assert provenance["frozen_modules_sha256"][source] == _sha(
                root / f"{capsule}-source.py.txt",
            )
        converter = "src/calibrex/data/kitti_raw_to_rosbag2.py"
        if converter in provenance["frozen_modules_sha256"]:
            assert provenance["frozen_modules_sha256"][converter] == _sha(
                root / "converter-source.py.txt",
            )


def test_original_failure_and_crc_verified_retry_are_preserved() -> None:
    first = ASSETS / "attempt_1"
    second = ASSETS / "attempt_2"
    failed = _json(first / "report.json")
    retry = _json(second / "report.json")
    assert (failed["status"], failed["stage"]) == ("FAIL", "capture")
    capture = _json(first / "capture/capture_manifest.json")
    assert not capture["complete"]
    assert set(capture["runs"]) == set(replay.RUNS)
    assert all(not r["ok"] and "candidate differs" in r["error"] for r in capture["runs"].values())
    assert retry["details"]["prior_report_sha256"] == _sha(first / "report.json")
    assert retry["details"]["archive_pin"] == failed["details"]["archive_pin"]
    assert retry["details"]["fetched_bytes"] == 0
    assert retry["details"]["reused_acquisition_bytes"] == failed["details"]["fetched_bytes"]
    assert _sha(second / "fetch.json") == _sha(first / "fetch.json")
    fetch = _json(second / "fetch.json")
    assert fetch["crc_verified"] is True
    assert retry["details"]["reused_members_crc_verified"] == len(fetch["members"])
    assert fetch["frames"] == {"image_02": 154, "velodyne_points": 154, "oxts": 154}


def test_real_pilot_parity_is_not_full_development_or_a_new_audit() -> None:
    root = ASSETS / "attempt_2"
    report = _json(root / "report.json")
    assert (report["status"], report["stage"]) == ("PASS_PILOT_ONLY", "complete")
    assert report["details"]["capture_exit_code"] == 0
    comparison = _json(root / "comparison/comparison.json")
    assert comparison["status"] == "incomplete"
    assert comparison["drives"] == ["2011_09_26_0005"]
    assert comparison["errors"] == {} and comparison["audit_thresholds"] is None
    capture = _json(root / "capture/capture_manifest.json")
    assert capture["complete"] and all(r["ok"] for r in capture["runs"].values())
    assert capture["conversion_sha256"] == _sha(root / "capture/conversion.json")
    assert comparison["provenance"]["capture_manifests_sha256"] == {
        "2011_09_26_0005": _sha(root / "capture/capture_manifest.json"),
    }
    for run in replay.RUNS:
        snapshot = _json(root / f"capture/{run}/snapshot.json")
        assert snapshot["arrays_sha256"] == capture["files_sha256"][f"{run}/arrays.npz"]
        original = load_camera_lidar_edge(root / f"capture/{run}/captured_edge.json")
        current = load_camera_lidar_edge(root / f"comparison/2011_09_26_0005/current/{run}.json")
        proposed = load_camera_lidar_edge(root / f"comparison/2011_09_26_0005/proposed/{run}.json")
        assert replay._same(original.model_dump(exclude={"provenance"}),
                            current.model_dump(exclude={"provenance"}))
        assert current.provenance.input_sha256 == proposed.provenance.input_sha256
        assert current.options["start_scan_min_control_t"] is None
        assert proposed.options["start_scan_min_control_t"] == 3.5
