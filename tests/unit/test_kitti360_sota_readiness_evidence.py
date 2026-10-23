"""Check published readiness receipts without reading bags or evaluation data."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import jsonschema
import pytest

ASSETS = Path(__file__).resolve().parents[2] / "docs/assets/kitti360_sota_readiness"


def _load(name: str) -> dict:
    return json.loads((ASSETS / name).read_text(encoding="utf-8"))


def _sha(name: str) -> str:
    return hashlib.sha256((ASSETS / name).read_bytes()).hexdigest()


def test_manifest_preserves_every_report_and_source() -> None:
    manifest = _load("manifest.json")
    jsonschema.validate(manifest, _load("manifest.schema.json"))
    files = {entry["path"]: entry for entry in manifest["files"]}
    assert len(files) == len(manifest["files"])
    assert set(files) == {
        p.name for p in ASSETS.iterdir() if p.name not in {"manifest.json", "manifest.schema.json"}
    }
    for name, entry in files.items():
        assert (ASSETS / name).stat().st_size == entry["bytes"]
        assert _sha(name) == entry["sha256"]
    assert manifest["provenance"]["generator_sha256"] == _sha("publication-source.py.txt")


@pytest.mark.parametrize("name", ["access", "metadata", "published-0000", "published-0002",
                                 "oxts-0000", "oxts-0002"])
def test_reports_are_schema_valid(name: str) -> None:
    jsonschema.validate(_load(f"{name}-report.json"), _load(f"{name}-report.schema.json"))


def test_failed_attempt_and_offline_recovery_remain_linked() -> None:
    report = _load("access-report.json")
    assert report["results"][0]["status"] == "FAIL"
    recovery = _load("distribution-script-recovery.json")
    assert recovery["status"] == "PASS_OFFLINE_RECOVERY"
    assert recovery["provenance"]["prior_report_sha256"] == _sha("access-report.json")
    assert recovery["provenance"]["script_zip_sha256"] == report["results"][0]["sha256"]
    assert recovery["provenance"]["sensor_payload_bytes"] == 0
    assert report["provenance"]["source_sha256"] == _sha("access-source.py.txt")


def test_metadata_ranges_exclude_sensor_payloads_and_bind_all_archives() -> None:
    report = _load("metadata-report.json")
    assert report["status"] == "METADATA_COMPLETE"
    assert len(report["archives"]) == 20
    provenance = report["provenance"]
    assert provenance["source_sha256"] == _sha("inventory-source.py.txt")
    assert provenance["metadata_tool_sha256"] == _sha("metadata-tool-source.py.txt")
    assert provenance["sensor_payload_bytes"] == 0
    assert provenance["evaluation_started"] is False
    assert provenance["exposure_audit_complete"] is False
    assert provenance["new_accuracy_claim"] is False
    bindings = _load("metadata-range-bindings.json")
    archives = {item["name"]: item for item in bindings["archives"]}
    assert len(archives) == len(bindings["archives"]) == 21
    assert set(archives) == {item["name"] for item in report["archives"]} | {"poses"}
    for item in report["archives"]:
        assert item["status"] == "PASS" and item["sensor_payload_bytes"] == 0
        for key in ["url", "size_bytes", "etag", "inventory_sha256"]:
            assert archives[item["name"]][key] == item[key]
    for item in archives.values():
        ranges = item["provenance"]["ranges"]
        assert [r["role"] for r in ranges] in [
            ["eocd", "central_directory"],
            ["eocd", "zip64_locator", "zip64_eocd", "central_directory"],
        ]
        for r in ranges:
            assert 0 <= r["start"] <= r["end"] < item["size_bytes"]
            assert re.fullmatch(r"[a-f0-9]{64}", r["sha256"])
        assert ranges[0]["end"] - ranges[0]["start"] + 1 == 22
    frames = _load("paired-frame-counts.json")
    assert frames["provenance"]["metadata_range_bindings_sha256"] == _sha(
        "metadata-range-bindings.json",
    )
    assert frames["provenance"]["excluded_spent_sequence"] == "0003"
    second = next(s for s in frames["sequences"] if s["sequence"] == "0002")
    assert second["image_first"] == second["lidar_first"] == 4391
    assert second["common_count"] == second["image_count"] == 14607


@pytest.mark.parametrize("mode,sequence,frames", [
    ("published", "0000", [1, 9, 14]), ("published", "0002", [4391, 4392, 4393]),
    ("oxts", "0000", [0, 1, 2]), ("oxts", "0002", [4391, 4392, 4393]),
])
def test_real_checks_are_spent_conversion_only(mode: str, sequence: str, frames: list[int]) -> None:
    report = _load(f"{mode}-{sequence}-report.json")
    assert report["status"] == "PASS" and report["error"] is None
    assert report["sequence"] == f"2013_05_28_drive_{sequence}_sync"
    assert report["selected_frame_ids"] == frames
    provenance = report["provenance"]
    source = "raw" if mode == "oxts" else "published"
    assert provenance["generator_sha256"] == _sha(f"{source}-development-source.py.txt")
    assert provenance["fresh_sensor_payload_bytes"] == 0
    assert provenance["new_accuracy_claim"] is False
    assert sum(item["bytes"] for item in report["inputs"]) <= 128 * 1024**2
    catalog = {a["name"]: a for a in _load("metadata-range-bindings.json")["archives"]}
    for item in report["inputs"]:
        assert item["crc_verified"] is True
        assert re.fullmatch(r"[a-f0-9]{64}", item["sha256"])
        archive = catalog[item["archive"]]
        assert item["url"] == archive["url"] and item["etag"] == archive["etag"]
    counts = report["conversion"]["message_counts"]
    assert counts == {"/tf_static": 1, "/camera/image_raw": 3, "/camera/camera_info": 3,
                      "/velodyne_points": 3, "/kitti360/odometry": 3}
    assert {t["topic"]: t["count"] for t in report["topics"]} == counts
    conversion = report["conversion"]["provenance"]
    if mode == "oxts":
        assert provenance["source_commit"] == "036c67ba7dc9eaba878d734430fbea6b8d70abbf"
        assert provenance["converter_source_sha256"] == _sha("oxts-converter-source.py.txt")
        assert conversion["motion"]["mode"] == "oxts"
        assert conversion["motion"]["origin_packet"].endswith("0000000000.txt")
        assert all(not i["member"].endswith("/poses.txt") for i in report["inputs"])
        assert any("/oxts/timestamps.txt" in i["member"] for i in report["inputs"])
    else:
        assert provenance["source_commit"] == "7f6528963bdaa02ab497e639702dc0df27b02b1f"
