import hashlib
import json
from pathlib import Path

from tools import kitti_camera_lidar_v3_replay as replay

ROOT = Path(__file__).resolve().parents[2]
BATCH = ROOT / "docs/assets/kitti_camera_lidar_v3/spent_batch_20261031"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_published_receipts_match_their_manifests_and_hold_no_sensor_arrays():
    for part in ("drives", "smoke"):
        manifest = json.loads((BATCH / part / "manifest.json").read_text())
        listed = {item["path"]: item for item in manifest["files"]}
        on_disk = {
            p.relative_to(BATCH / part).as_posix()
            for p in (BATCH / part).rglob("*")
            if p.is_file() and p.name != "manifest.json"
        }
        assert set(listed) == on_disk
        for name, item in listed.items():
            path = BATCH / part / name
            assert path.stat().st_size == item["bytes"] and _sha(path) == item["sha256"]
        assert not list((BATCH / part).rglob("*.npz"))


def test_every_spent_drive_passes_on_its_latest_attempt_with_failures_retained():
    summary = json.loads((BATCH / "summary.json").read_text())
    assert summary["status"] == "complete" and summary["audit_thresholds"] is None
    assert set(summary["drive_status"]) == set(replay.spent_drives())
    for drive, latest in summary["attempts"].items():
        reports = [
            json.loads((BATCH / "drives" / drive / f"attempt_{n}" / "report.json").read_text())
            for n in range(1, latest + 1)
        ]
        assert reports[-1]["status"] == "PASS_DEVELOPMENT"
        assert all(r["status"] == "FAIL" and r["stage"] == "fetch" for r in reports[:-1])
        assert reports[-1]["provenance"]["development_only"]


def test_current_rule_replay_equals_the_committed_v2_aggregate():
    summary = json.loads((BATCH / "summary.json").read_text())
    inventory = json.loads(
        (ROOT / "docs/assets/kitti_camera_lidar_v3/development_inventory.json").read_text()
    )
    assert summary["metrics"]["current"] == inventory["current_v2_metrics"]
    proposed = summary["metrics"]["proposed"]
    assert proposed["trials"]["3deg"]["flagged"] == 114
    assert proposed["max_turned_judged_error_deg"] < 1.503


def test_pins_agree_with_the_earlier_native_pilots():
    pins = json.loads((BATCH / "pins.json").read_text())["drives"]
    assert (pins["2011_09_26_0005"]["size_bytes"], pins["2011_09_26_0005"]["etag"]) == (
        645940900,
        "6a880679a3e3eda8a0ef83abc5354eb1-78",
    )
    assert (pins["2011_09_28_0002"]["size_bytes"], pins["2011_09_28_0002"]["etag"]) == (
        1458258535,
        "57c9f403568ce97a5aeb70180b029c93-174",
    )
