"""Case selection uses metadata counts, not previous calibration outcomes."""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.parse
import zipfile
from pathlib import Path

import pytest
from tools import ins_lidar_sota_inventory as inventory


def pool(tmp_path):
    root = tmp_path / "pool"
    for index in range(1, 19):
        directory = root / f"2011_09_29_{index:04d}"
        directory.mkdir(parents=True)
        record = {
            "size_bytes": 1000,
            "etag": "pin",
            "frames": {"oxts": 600, "velodyne_points": 600},
            "archive_frames": {"oxts": 1000, "velodyne_points": 1000},
            "irrelevant_method_result": "must not enter selection",
        }
        if index == 1:
            record["archive_frames"]["velodyne_points"] = 100
        (directory / "download.json").write_text(json.dumps(record))
    return root


def central(url, *args):
    archive = Path(urllib.parse.urlsplit(url).path).name.removesuffix("_sync.zip")
    prefix = f"{archive[:10]}/{archive}_sync/"
    infos = [
        zipfile.ZipInfo(prefix + f"{stream}/data/{index:010d}.{extension}")
        for stream, extension in (("oxts", "txt"), ("velodyne_points", "bin"))
        for index in range(600)
    ]
    for info in infos:
        info.CRC = 0
    return infos


def test_chronological_eligibility_and_member_pins_without_sensor_requests(tmp_path, monkeypatch):
    metadata = pool(tmp_path)
    monkeypatch.setattr(inventory.fetch, "head", lambda _: (1000, "pin"))
    monkeypatch.setattr(inventory.fetch, "central_directory", central)
    monkeypatch.setattr(inventory.fetch, "spans", lambda *args: [(0, 99)])
    monkeypatch.setattr(
        inventory.fetch, "fetch_range", lambda *args: pytest.fail("must not fetch members")
    )
    result = inventory.collect(metadata, source_root=tmp_path)
    assert [r.case_id for r in result.selected] == [f"2011_09_29_{i:04d}" for i in (2, 3, 4)]
    assert len(result.source_pool_metadata_sha256) == 18
    assert result.sensor_member_download_bytes == 0
    assert all(r.selected_frames == {"velodyne_points": 600, "oxts": 600} for r in result.selected)
    assert all(len(r.selected_members) == 1200 for r in result.selected)


def test_archive_pin_changes_fail_without_replacing_the_candidate(tmp_path, monkeypatch):
    metadata = pool(tmp_path)
    monkeypatch.setattr(inventory.fetch, "head", lambda _: (999, "different"))
    monkeypatch.setattr(
        inventory.fetch, "central_directory", lambda *args: pytest.fail("no payload")
    )
    with pytest.raises(ValueError, match="2011_09_29_0002: live archive pin"):
        inventory.collect(metadata, source_root=tmp_path)


def test_script_starts_outside_the_repository_without_pythonpath(tmp_path):
    result = subprocess.run(
        [sys.executable, str(Path(inventory.__file__).resolve()), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "ZIP metadata" in result.stdout
