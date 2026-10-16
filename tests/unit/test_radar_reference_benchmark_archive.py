from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import pytest
from tools.benchmark_radar_reference_indexing import audit

ARCHIVE = Path(__file__).resolve().parents[2] / "docs/assets/radar_reference_indexing"


def _write(path: Path, value: object) -> None:
    path.write_bytes((json.dumps(value, indent=2) + "\n").encode())


def test_complete_radar_reference_archive() -> None:
    report = audit(ARCHIVE)
    assert report == json.loads((ARCHIVE / "audit.json").read_bytes())
    assert report["verified_runs"] == 6
    assert report["profile_candidates_per_run"] == 201
    assert report["reference_count"] == 2762
    assert report["measurement_count"] == 2758
    assert report["complete_results_exact"]


def test_every_archived_byte_is_indexed() -> None:
    index = json.loads((ARCHIVE / "index.json").read_bytes())
    files = {
        path.relative_to(ARCHIVE).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in ARCHIVE.rglob("*")
        if path.is_file() and path.name != "index.json"
    }
    assert files == index["files"]
    assert index["provenance"]["protocol_sha256"] == hashlib.sha256(
        (ARCHIVE / "protocol.json").read_bytes()
    ).hexdigest()


@pytest.mark.parametrize("seconds", [0.0, -1.0, math.inf, math.nan])
def test_archive_rejects_invalid_timings(tmp_path: Path, seconds: float) -> None:
    target = tmp_path / "archive"
    shutil.copytree(ARCHIVE, target)
    attempts = json.loads((target / "attempts.json").read_bytes())
    attempts[0]["seconds"] = seconds
    _write(target / "attempts.json", attempts)
    with pytest.raises(ValueError, match="invalid whole-process timing"):
        audit(target)


def test_archive_rejects_duplicate_runs(tmp_path: Path) -> None:
    target = tmp_path / "archive"
    shutil.copytree(ARCHIVE, target)
    attempts = json.loads((target / "attempts.json").read_bytes())
    attempts[-1] = attempts[0]
    _write(target / "attempts.json", attempts)
    with pytest.raises(ValueError, match="missing, duplicate or reordered"):
        audit(target)


@pytest.mark.parametrize("part", ["candidate", "diagnostic", "missing_probe"])
def test_archive_checks_complete_results_beyond_file_hashes(tmp_path: Path, part: str) -> None:
    target = tmp_path / "archive"
    shutil.copytree(ARCHIVE, target)
    folder = target / "00-current"
    path = folder / "result.json"
    result = json.loads(path.read_bytes())
    if part == "candidate":
        result["candidates"][0]["mse"] += 1.0e-6
    elif part == "diagnostic":
        result["result"]["known_bad_probes"][0]["delta_mps"] += 1.0e-6
    else:
        result["result"]["known_bad_probes"].pop()
    _write(path, result)
    # Rebind the local digest to prove that the audit compares all run content.
    measurement = json.loads((folder / "measurement.json").read_bytes())
    measurement["result_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write(folder / "measurement.json", measurement)
    with pytest.raises(ValueError, match=r"complete results|complete its calibration and probes"):
        audit(target)


def test_archive_rejects_unfrozen_solver_import(tmp_path: Path) -> None:
    target = tmp_path / "archive"
    shutil.copytree(ARCHIVE, target)
    path = target / "00-current/measurement.json"
    measurement = json.loads(path.read_bytes())
    for name, item in measurement["imports"].items():
        if name.endswith("radar_spatiotemporal_lever_arm_solver"):
            item["sha256"] = "0" * 64
    _write(path, measurement)
    with pytest.raises(ValueError, match="unfrozen module"):
        audit(target)


def test_archive_rejects_changed_input(tmp_path: Path) -> None:
    target = tmp_path / "archive"
    shutil.copytree(ARCHIVE, target)
    path = target / "control.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match=r"artifact changed: control\.json"):
        audit(target)
