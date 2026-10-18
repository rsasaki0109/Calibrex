from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import pytest
from jsonschema import ValidationError
from tools.benchmark_continuous_camera_lidar_reuse import audit

ARCHIVE = Path(__file__).resolve().parents[2] / "docs/assets/continuous_camera_lidar_batching"


def _write(path: Path, value: object) -> None:
    path.write_bytes((json.dumps(value, indent=2) + "\n").encode())


@pytest.fixture
def archive_copy(tmp_path: Path) -> Path:
    target = tmp_path / "archive"
    shutil.copytree(ARCHIVE, target)
    return target


def test_complete_ordered_batch_archive() -> None:
    report = audit(ARCHIVE)
    assert report == json.loads((ARCHIVE / "audit.json").read_bytes())
    assert report["verified_runs"] == 6 and report["complete_results_exact"]
    assert report["capture_count"] == 9 and report["correspondence_count"] == 9216
    assert report["objective_evaluations_per_run"] == 341
    assert report["projection_requests_per_run"] == 2462720
    assert report["deskew_computations_per_run"] == {"prior": 556032, "current": 556032}
    assert report["work_counts_per_run"] == {
        "prior": {
            "scalar_deskew_points": 556032,
            "batched_deskew_points": 0,
            "batched_deskew_calls": 0,
            "scalar_projection_points": 2462720,
            "batched_projection_points": 0,
            "batched_projection_calls": 0,
        },
        "current": {
            "scalar_deskew_points": 0,
            "batched_deskew_points": 556032,
            "batched_deskew_calls": 543,
            "scalar_projection_points": 0,
            "batched_projection_points": 2462720,
            "batched_projection_calls": 2405,
        },
    }


def test_every_batch_archive_byte_is_indexed() -> None:
    index = json.loads((ARCHIVE / "index.json").read_bytes())
    assert index["files"] == {
        p.relative_to(ARCHIVE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in ARCHIVE.rglob("*")
        if p.is_file() and p.name != "index.json"
    }


@pytest.mark.parametrize(
    "counter",
    [
        "scalar_deskew_points",
        "batched_deskew_points",
        "batched_deskew_calls",
        "scalar_projection_points",
        "batched_projection_points",
        "batched_projection_calls",
    ],
)
def test_observed_work_must_match_independent_trace_replay(
    archive_copy: Path,
    counter: str,
) -> None:
    path = archive_copy / "00-current/measurement.json"
    measured = json.loads(path.read_bytes())
    measured["work_counts"][counter] += 1
    _write(path, measured)
    with pytest.raises(ValueError, match="observed work differs"):
        audit(archive_copy)


@pytest.mark.parametrize("seconds", [0.0, -1.0, math.inf, math.nan])
def test_batch_archive_requires_finite_positive_whole_timers(
    archive_copy: Path,
    seconds: float,
) -> None:
    path = archive_copy / "attempts.json"
    attempts = json.loads(path.read_bytes())
    attempts["runs"][0]["seconds"] = seconds
    _write(path, attempts)
    with pytest.raises(ValueError, match="invalid whole-process timing"):
        audit(archive_copy)


@pytest.mark.parametrize("count", [-1, True, 1.5])
def test_batch_counters_require_nonnegative_integers(archive_copy: Path, count: object) -> None:
    path = archive_copy / "00-current/measurement.json"
    measured = json.loads(path.read_bytes())
    measured["work_counts"]["batched_deskew_calls"] = count
    _write(path, measured)
    with pytest.raises(ValidationError):
        audit(archive_copy)


def test_batch_schema_requires_observed_counters(archive_copy: Path) -> None:
    path = archive_copy / "00-current/measurement.json"
    measured = json.loads(path.read_bytes())
    del measured["work_counts"]
    _write(path, measured)
    with pytest.raises(ValidationError):
        audit(archive_copy)


@pytest.mark.parametrize("part", ["trace", "initial_state"])
def test_batch_archive_checks_full_result_beyond_digest(archive_copy: Path, part: str) -> None:
    path = archive_copy / "00-current/result.json"
    result = json.loads(path.read_bytes())
    if part == "trace":
        result["trace"][0]["objective"] += 1.0e-6
    else:
        result["initial_time_offset_sec"] += 0.001
    _write(path, result)
    path = path.with_name("measurement.json")
    measured = json.loads(path.read_bytes())
    measured["result_sha256"] = hashlib.sha256(
        path.with_name("result.json").read_bytes()
    ).hexdigest()
    _write(path, measured)
    with pytest.raises(ValueError, match=r"complete results|initial state"):
        audit(archive_copy)


def test_batch_archive_requires_all_alternating_runs(archive_copy: Path) -> None:
    path = archive_copy / "attempts.json"
    attempts = json.loads(path.read_bytes())
    attempts["runs"][-1] = attempts["runs"][0]
    _write(path, attempts)
    with pytest.raises(ValueError, match="duplicate or reordered"):
        audit(archive_copy)


def test_batch_archive_binds_original_public_source_points(archive_copy: Path) -> None:
    path = archive_copy / "source-points.json.gz"
    path.write_bytes(path.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="pinned artifact changed"):
        audit(archive_copy)
