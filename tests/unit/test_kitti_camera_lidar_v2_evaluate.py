"""The KITTI camera-lidar v2 driver: dated drives, one calibration per day, and pins."""

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest
import yaml
from tests.unit.test_score_hilti_camera_camera import _load

driver = _load("kitti_camera_lidar_v2_evaluate")


def test_a_drive_is_named_by_its_day() -> None:
    assert driver.split_drive("2011_09_30_0028") == ("2011_09_30", "0028")
    with pytest.raises(ValueError, match="DATE_DRIVE"):
        driver.split_drive("0028")


class _Download:
    """Stands in for curl: writes a fixed calibration archive where ``-o`` points."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def run(self, label: str, command: list[str], log: Path) -> int:
        Path(command[command.index("-o") + 1]).write_bytes(self.payload)
        return 0


def test_a_calibration_archive_is_unpacked_only_when_it_matches_its_pin(tmp_path: Path) -> None:
    source = tmp_path / "calib.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("2011_09_28/calib_velo_to_cam.txt", "R: 1 0 0 0 1 0 0 0 1\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    download = _Download(source.read_bytes())

    def work(name: str) -> Path:
        path = tmp_path / name
        path.mkdir()
        return path

    pinned = driver.calibration(download, "2011_09_28", work("a"), digest, True)
    assert pinned["ok"] and pinned["sha256"] == digest
    assert (tmp_path / "a/2011_09_28/calib_velo_to_cam.txt").is_file()
    other = driver.calibration(download, "2011_09_28", work("b"), "0" * 64, True)
    assert not other["ok"] and "differs" in other["reason"]
    assert not (tmp_path / "b/2011_09_28").exists()
    unpinned = driver.calibration(download, "2011_09_28", work("c"), None, True)
    assert not unpinned["ok"] and "pins no SHA-256" in unpinned["reason"]
    # without a pre-registration (development runs) nothing is pinned
    assert driver.calibration(download, "2011_09_28", work("d"), None, False)["ok"]


class _Fetcher:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def drive_url(self, date: str, drive: str) -> str:
        return f"https://example.org/{date}_drive_{drive}_sync.zip"

    def fetch_drive(
        self, date: str, drive: str, max_frames: int, output_dir: Path, expected: Any = None
    ) -> dict[str, Any]:
        self.calls.append((date, drive, max_frames, expected))
        return {
            "url": self.drive_url(date, drive),
            "size_bytes": expected[0],
            "etag": expected[1],
            "max_frames": max_frames,
            "member_count": 1,
            "members_sha256": "f" * 64,
            "members": [{"name": "a", "bytes": 1, "crc32": 2}],
            "crc_verified": True,
        }


def _plan(tmp_path: Path) -> Path:
    plan = {
        "recordings": {
            "evaluation": ["2011_09_28_0002", "2011_09_30_0016"],
            "archives": {"2011_09_28_0002": {"size_bytes": 10, "etag": "a-1"}},
            "calibration_archive_sha256": {"2011_09_28": "1" * 64, "2011_09_30": "2" * 64},
        },
        "run_recipe": {"injections": ["x:3", "z:1"], "velodyne_time": "azimuth", "max_frames": 600},
    }
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(plan), encoding="utf-8")
    return path


def test_a_pre_registered_drive_without_a_pin_is_not_fetched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fetcher = _Fetcher()
    days: list[tuple[str, str | None, bool]] = []
    commands: list[list[str]] = []

    def calibration(record: Any, date: str, work: Path, pin: str | None, required: bool) -> Any:
        days.append((date, pin, required))
        return {"ok": True, "sha256": pin}

    monkeypatch.setattr(driver, "_fetch_module", lambda: fetcher)
    monkeypatch.setattr(driver, "calibration", calibration)
    monkeypatch.setattr(driver.Record, "run", lambda self, label, cmd, log: commands.append(cmd))
    work, runs = tmp_path / "work", tmp_path / "runs"
    arguments = ["--preregistration", str(_plan(tmp_path)), "--work-dir", str(work)]
    assert driver.main([*arguments, "--output-dir", str(runs)]) == 0
    assert days == [("2011_09_28", "1" * 64, True), ("2011_09_30", "2" * 64, True)]
    assert fetcher.calls == [("2011_09_28", "0002", 600, (10, "a-1"))]
    failed = json.loads((runs / "2011_09_30_0016/download.json").read_text())
    assert failed["ok"] is False and "pins no size and ETag" in failed["reason"]
    (command,) = commands
    assert command[command.index("--recording") + 1] == "2011_09_28_0002"
    assert command[command.index("--calib-dir") + 1] == str(work / "2011_09_28")
    assert command[command.index("--velodyne-time") + 1] == "azimuth"
    injections = [command[i + 1] for i, item in enumerate(command) if item == "--inject"]
    assert injections == ["x:3", "z:1"]
    fetched = json.loads((runs / "2011_09_28_0002/download.json").read_text())
    assert "members" not in fetched and fetched["members_sha256"] == "f" * 64
    assert json.loads((work / "2011_09_28_0002_members.json").read_text())[0]["name"] == "a"
    record = json.loads((work / "evaluate_record.json").read_text())
    assert [(entry["label"], entry["exit_code"]) for entry in record] == [
        ("fetch 2011_09_28_0002", 0),
        ("fetch 2011_09_30_0016", 1),
    ]
    assert "pins no size and ETag" in record[1]["detail"]


def test_nothing_runs_when_a_calibration_differs_from_its_pin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fetcher = _Fetcher()
    monkeypatch.setattr(driver, "_fetch_module", lambda: fetcher)
    monkeypatch.setattr(
        driver, "calibration", lambda *args: {"ok": args[1] != "2011_09_30", "reason": "differs"}
    )
    arguments = ["--preregistration", str(_plan(tmp_path)), "--work-dir", str(tmp_path / "w")]
    assert driver.main([*arguments, "--output-dir", str(tmp_path / "runs")]) == 1
    assert fetcher.calls == []
    days = json.loads((tmp_path / "runs/calibration_download.json").read_text())
    assert [days[date]["ok"] for date in sorted(days)] == [True, False]
