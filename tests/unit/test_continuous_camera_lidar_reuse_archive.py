from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import pytest
from jsonschema import ValidationError
from tools.benchmark_continuous_camera_lidar_reuse import audit

ARCHIVE = (
    Path(__file__).resolve().parents[2] / "docs/assets/continuous_camera_lidar_evaluation_reuse"
)


def _write(path: Path, value: object) -> None:
    path.write_bytes((json.dumps(value, indent=2) + "\n").encode())


@pytest.fixture
def archive_copy(tmp_path: Path) -> Path:
    target = tmp_path / "archive"
    shutil.copytree(ARCHIVE, target)
    return target


def test_complete_timed_camera_lidar_archive() -> None:
    report = audit(ARCHIVE)
    assert report == json.loads((ARCHIVE / "audit.json").read_bytes())
    assert report["verified_runs"] == 6 and report["complete_results_exact"]
    assert report["capture_count"] == 9 and report["correspondence_count"] == 9216
    assert report["objective_evaluations_per_run"] == 341
    assert report["projection_requests_per_run"] == 2462720
    assert report["deskew_computations_per_run"] == {"prior": 2462720, "current": 556032}


def test_every_timed_camera_archive_byte_is_indexed() -> None:
    index = json.loads((ARCHIVE / "index.json").read_bytes())
    assert index["files"] == {
        p.relative_to(ARCHIVE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in ARCHIVE.rglob("*")
        if p.is_file() and p.name != "index.json"
    }


@pytest.mark.parametrize("seconds", [0.0, -1.0, math.inf, math.nan])
def test_timed_archive_rejects_invalid_whole_timing(archive_copy: Path, seconds: float) -> None:
    path = archive_copy / "attempts.json"
    value = json.loads(path.read_bytes())
    value["runs"][0]["seconds"] = seconds
    _write(path, value)
    with pytest.raises(ValueError, match="invalid whole-process timing"):
        audit(archive_copy)


def test_timed_archive_requires_every_alternating_run(archive_copy: Path) -> None:
    path = archive_copy / "attempts.json"
    value = json.loads(path.read_bytes())
    value["runs"][-1] = value["runs"][0]
    _write(path, value)
    with pytest.raises(ValueError, match="duplicate or reordered"):
        audit(archive_copy)


@pytest.mark.parametrize("count", [0, True, 1.5])
def test_timed_archive_requires_positive_integer_work_counts(
    archive_copy: Path, count: object
) -> None:
    path = archive_copy / "00-current/measurement.json"
    value = json.loads(path.read_bytes())
    value["deskew_computations"] = count
    _write(path, value)
    with pytest.raises(ValidationError):
        audit(archive_copy)


def test_timed_archive_requires_identical_counts_between_repeats(archive_copy: Path) -> None:
    path = archive_copy / "00-current/measurement.json"
    value = json.loads(path.read_bytes())
    value["deskew_computations"] += 1
    _write(path, value)
    with pytest.raises(ValueError, match="counts differ between repeated runs"):
        audit(archive_copy)


@pytest.mark.parametrize("part", ["trace", "initial_state"])
def test_timed_archive_checks_complete_outputs_beyond_digest(archive_copy: Path, part: str) -> None:
    path = archive_copy / "00-current/result.json"
    value = json.loads(path.read_bytes())
    if part == "trace":
        value["trace"][0]["objective"] += 1.0e-6
    else:
        value["initial_time_offset_sec"] += 0.001
    _write(path, value)
    measurement = path.with_name("measurement.json")
    measured = json.loads(measurement.read_bytes())
    measured["result_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write(measurement, measured)
    with pytest.raises(ValueError, match=r"complete results|initial state"):
        audit(archive_copy)


def test_timed_archive_binds_original_public_points(archive_copy: Path) -> None:
    path = archive_copy / "source-points.json.gz"
    path.write_bytes(path.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="pinned artifact changed"):
        audit(archive_copy)


def test_timed_archive_binds_actual_source_identity(archive_copy: Path) -> None:
    path = archive_copy / "00-current/measurement.json"
    value = json.loads(path.read_bytes())
    value["provenance"]["git_commit"] = "0" * 40
    _write(path, value)
    with pytest.raises(ValueError, match=r"run provenance differs|Git provenance differs"):
        audit(archive_copy)


def test_timed_archive_requires_the_measured_solver_import(archive_copy: Path) -> None:
    path = archive_copy / "00-current/measurement.json"
    value = json.loads(path.read_bytes())
    del value["imports"]["calibrex.solvers.continuous_time_camera_lidar_solver"]
    _write(path, value)
    with pytest.raises(ValueError, match="measured solver import is missing"):
        audit(archive_copy)
