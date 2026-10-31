import json
from pathlib import Path

import pytest
from tools import kitti_camera_lidar_v3_batch as batch
from tools import kitti_camera_lidar_v3_replay as replay

DRIVE = "2011_09_26_0005"
GIB = 1024**3


def _pins(tmp_path: Path) -> Path:
    meta = tmp_path / "meta"
    meta.mkdir()
    by_date: dict[str, dict] = {}
    for drive in replay.spent_drives():
        by_date.setdefault(drive[:10], {})[drive[11:]] = {
            "size_bytes": 10,
            "etag": "e",
            "frames": {"image_02": 125},
        }
    for date, records in by_date.items():
        (meta / f"{date}.json").write_text(json.dumps(records))
    path = tmp_path / "pins.json"
    batch.pin(meta, path)
    return path


def _fake_pipeline(monkeypatch, capture_code=0, compare_errors=None):
    fetch = batch._tool("kitti_drive_fetch")
    tool = batch._tool("kitti_camera_lidar_v3_replay")
    calls = []

    def fake_fetch(date, number, max_frames, out, expected, max_fetch_bytes):
        calls.append(("fetch", expected, max_fetch_bytes))
        (out / date / f"{date}_drive_{number}_sync").mkdir(parents=True)
        return {"fetched_bytes": 5, "frames": {"image_02": 125}, "members_sha256": "m"}

    class Converted:
        def __init__(self, path):
            self.provenance_path = path

    def fake_convert(source, bag, **kwargs):
        assert source.is_dir() and kwargs["camera"] == "image_02"
        bag.mkdir()
        (bag / "calibrex_conversion.json").write_text("{}")
        return Converted(bag / "calibrex_conversion.json")

    def fake_capture(bag, drive, output):
        assert bag.is_dir() and not (bag.parent / "raw").exists()
        (output / "baseline").mkdir(parents=True)
        (output / "baseline" / "arrays.npz").write_bytes(b"arrays")
        (output / "baseline.json").write_text("{}")
        return capture_code

    def fake_compare(snapshots, output, drives, margin):
        assert drives == [DRIVE] and margin == 3.5
        output.mkdir()
        scores = {rule: {DRIVE: {"state": "inconclusive"}} for rule in ("current", "proposed")}
        report = {"status": "incomplete", "errors": compare_errors or {}, "scores": scores}
        (output / "comparison.json").write_text(json.dumps(report))
        return report

    monkeypatch.setattr(fetch, "fetch_drive", fake_fetch)
    monkeypatch.setattr(tool, "capture_drive", fake_capture)
    monkeypatch.setattr(tool, "compare_drives", fake_compare)
    monkeypatch.setattr(
        "calibrex.data.kitti_raw_to_rosbag2.convert_kitti_raw_to_rosbag2", fake_convert
    )
    monkeypatch.setattr(batch, "_calibration", lambda work, date, sha: work)
    return calls


def test_pins_cover_exactly_the_spent_drives_and_pin_every_calibration_day(tmp_path):
    pins = json.loads(_pins(tmp_path).read_text())
    assert list(pins["drives"]) == list(replay.spent_drives())
    assert set(pins["calibration_sha256"]) == {d[:10] for d in replay.spent_drives()}
    (tmp_path / "meta" / "2011_09_29.json").unlink()
    with pytest.raises(ValueError, match="exactly the 55"):
        batch.pin(tmp_path / "meta", tmp_path / "other.json")


def test_successful_drive_deletes_raw_and_bag_and_keeps_arrays_with_room(tmp_path, monkeypatch):
    calls = _fake_pipeline(monkeypatch)
    pins = json.loads(_pins(tmp_path).read_text())
    report = batch.run_drive(pins, DRIVE, tmp_path / "work", free=lambda: 100 * GIB)
    out = tmp_path / "work" / DRIVE / "attempt_1"
    assert report.status == "PASS_DEVELOPMENT" and report.stage == "complete"
    assert calls == [("fetch", (10, "e"), 2 * GIB)]
    assert not (out / "raw").exists() and not (out / "bag").exists()
    assert report.details["arrays_kept"] == [f"snapshots/{DRIVE}/baseline/arrays.npz"]
    saved = json.loads((out / "report.json").read_text())
    assert saved["provenance"]["development_only"] and saved["status"] == "PASS_DEVELOPMENT"


def test_low_disk_deletes_arrays_but_records_their_digests(tmp_path, monkeypatch):
    _fake_pipeline(monkeypatch)
    pins = json.loads(_pins(tmp_path).read_text())
    free = iter([100 * GIB, 1 * GIB])
    report = batch.run_drive(pins, DRIVE, tmp_path / "work", free=lambda: next(free))
    assert report.status == "PASS_DEVELOPMENT"
    assert report.details["arrays_kept"] == []
    assert list(report.details["arrays_deleted"]) == [f"snapshots/{DRIVE}/baseline/arrays.npz"]


def test_capture_failure_is_retained_and_skipped_until_explicit_retry(tmp_path, monkeypatch):
    _fake_pipeline(monkeypatch, capture_code=1)
    pins_path = _pins(tmp_path)
    work = tmp_path / "work"
    monkeypatch.setattr(batch.shutil, "disk_usage", lambda path: type("U", (), {"free": 99 * GIB}))
    assert batch.run(pins_path, work, [DRIVE]) == 1
    first = json.loads((work / DRIVE / "attempt_1" / "report.json").read_text())
    assert first["status"] == "FAIL" and first["stage"] == "capture"
    assert (work / DRIVE / "attempt_1" / "failure.txt").is_file()
    assert not (work / DRIVE / "attempt_1" / "bag").exists()
    assert batch.run(pins_path, work, [DRIVE]) == 0  # skipped, never overwritten
    assert not (work / DRIVE / "attempt_2").exists()
    _fake_pipeline(monkeypatch)
    assert batch.run(pins_path, work, [DRIVE], retry_failed=True) == 0
    assert json.loads((work / DRIVE / "attempt_2" / "report.json").read_text())["status"] == (
        "PASS_DEVELOPMENT"
    )
    assert (work / DRIVE / "attempt_1" / "failure.txt").is_file()


def test_insufficient_disk_fails_before_any_fetch(tmp_path, monkeypatch):
    calls = _fake_pipeline(monkeypatch)
    pins = json.loads(_pins(tmp_path).read_text())
    report = batch.run_drive(pins, DRIVE, tmp_path / "work", free=lambda: 1 * GIB)
    assert report.status == "FAIL" and report.stage == "preflight" and calls == []


def test_publish_excludes_arrays_and_aggregate_counts_missing_drives(tmp_path, monkeypatch):
    _fake_pipeline(monkeypatch)
    pins = json.loads(_pins(tmp_path).read_text())
    work = tmp_path / "work"
    batch.run_drive(pins, DRIVE, work, free=lambda: 100 * GIB)
    (work / DRIVE / "attempt_1" / "fetch.json").write_text(
        json.dumps({"members": [1], "members_sha256": "m"})
    )
    dest = tmp_path / "published"
    assert batch.publish(work, dest) > 0
    published = dest / DRIVE / "attempt_1"
    assert not list(published.rglob("*.npz"))
    manifest = json.loads((dest / "manifest.json").read_text())
    assert {f["path"] for f in manifest["files"]} == {
        p.relative_to(dest).as_posix()
        for p in dest.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    assert manifest["provenance"]["generator_sha256"] == batch._sha(Path(batch.__file__))
    assert json.loads((published / "fetch.json").read_text()) == {"members_sha256": "m"}
    summary = batch.aggregate(dest, tmp_path / "summary.json")
    assert summary["status"] == "incomplete" and summary["audit_thresholds"] is None
    assert summary["attempts"][DRIVE] == 1 and summary["provenance"]["git_commit"]
    assert summary["drive_status"][DRIVE] == "PASS_DEVELOPMENT"
    assert summary["metrics"]["current"]["inconclusive_drives"] == [DRIVE]
    assert len(summary["metrics"]["current"]["failed_drives"]) == 54


def test_interrupted_attempt_is_retried_only_on_request_and_runner_is_exclusive(
    tmp_path, monkeypatch
):
    _fake_pipeline(monkeypatch)
    pins_path = _pins(tmp_path)
    work = tmp_path / "work"
    interrupted = work / DRIVE / "attempt_1"
    interrupted.mkdir(parents=True)
    (interrupted / "report.json").write_text(json.dumps({"status": "RUNNING"}))
    monkeypatch.setattr(batch.shutil, "disk_usage", lambda path: type("U", (), {"free": 99 * GIB}))
    assert batch.run(pins_path, work, [DRIVE]) == 0
    assert not (work / DRIVE / "attempt_2").exists()
    assert batch.run(pins_path, work, [DRIVE], retry_failed=True) == 0
    assert json.loads((interrupted / "report.json").read_text()) == {"status": "RUNNING"}
    assert (work / DRIVE / "attempt_2" / "report.json").is_file()
    import fcntl

    with (work / ".batch.lock").open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another batch runner"):
            batch.run(pins_path, work, [DRIVE])
    with pytest.raises(ValueError, match="distinct pinned"):
        batch.run(pins_path, work, [DRIVE, DRIVE])
