from __future__ import annotations

import gzip
import hashlib
import json
import math
import shutil
from pathlib import Path

import pytest
from jsonschema import ValidationError
from tools import benchmark_gnss_lever_arm_reuse as benchmark
from tools.benchmark_gnss_lever_arm_reuse import audit

ARCHIVE = Path(__file__).resolve().parents[2] / "docs/assets/gnss_lever_arm_evaluation_reuse"


def write(path: Path, value: object) -> None:
    path.write_bytes((json.dumps(value, indent=2) + "\n").encode())


@pytest.fixture
def archive_copy(tmp_path: Path) -> Path:
    target = tmp_path / "archive"
    shutil.copytree(ARCHIVE, target)
    return target


def test_complete_gnss_results_histories_and_work_are_exact() -> None:
    report = audit(ARCHIVE)
    assert report == json.loads((ARCHIVE / "audit.json").read_bytes())
    assert report["verified_runs"] == 6 and report["complete_results_and_histories_exact"]
    assert report["reference_pose_count"] == 2762 and report["window_count"] == 29
    assert report["train_windows"] == 19 and report["holdout_windows"] == 10
    assert report["jackknife_fits"] == 8
    assert report["work_per_run"] == {
        "prior": {
            "solve_calls": 9,
            "residual_requests": 1542,
            "least_squares_calls": 36,
            "lidar_poses_calls": 26029,
            "svd_calls": 26000,
            "profiled_windows": 26000,
            "pose_batch_calls": 0,
            "interpolated_epochs": 2368419,
        },
        "current": {
            "solve_calls": 9,
            "residual_requests": 1542,
            "least_squares_calls": 36,
            "lidar_poses_calls": 138,
            "svd_calls": 1258,
            "profiled_windows": 19532,
            "pose_batch_calls": 600,
            "interpolated_epochs": 935169,
        },
    }


def test_every_gnss_archive_byte_is_indexed() -> None:
    assert json.loads((ARCHIVE / "index.json").read_bytes())["files"] == {
        p.relative_to(ARCHIVE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in ARCHIVE.rglob("*")
        if p.is_file() and p.name != "index.json"
    }


def test_complete_archive_audits_from_an_lf_only_generator_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "tools/prepare_gnss_lever_arm_control.py"
    target.parent.mkdir()
    pinned = (ARCHIVE / "control-generator.py.txt").read_bytes()
    assert b"\r\n" in pinned
    target.write_bytes(pinned.replace(b"\r\n", b"\n"))
    assert target.read_bytes() != pinned
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    assert audit(ARCHIVE) == json.loads((ARCHIVE / "audit.json").read_bytes())


def test_observed_work_must_match_full_history_replay(archive_copy: Path) -> None:
    path = archive_copy / "00-current/measurement.json"
    value = json.loads(path.read_bytes())
    value["counts"]["pose_batch_calls"] += 1
    write(path, value)
    with pytest.raises(ValueError, match="observed work differs"):
        audit(archive_copy)


@pytest.mark.parametrize("seconds", [-1.0, math.nan])
def test_archive_requires_valid_whole_process_time(archive_copy: Path, seconds: float) -> None:
    path = archive_copy / "attempts.json"
    value = json.loads(path.read_bytes())
    value["runs"][0]["seconds"] = seconds
    write(path, value)
    with pytest.raises(ValueError, match="invalid whole-process timing"):
        audit(archive_copy)


def mutate_history(archive: Path, part: str) -> None:
    path = archive / "00-current/history.json.gz"
    value = json.loads(gzip.decompress(path.read_bytes()))
    if part == "residual":
        value["residuals"][0]["residual_sha256"] = "0" * 64
    elif part == "epochs":
        value["residuals"][0]["epoch_count"] -= 1
    else:
        value["least_squares"][0]["nfev"] += 1
    path.write_bytes(gzip.compress((json.dumps(value) + "\n").encode(), mtime=0))
    measured = json.loads(path.with_name("measurement.json").read_bytes())
    measured["history_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    write(path.with_name("measurement.json"), measured)


@pytest.mark.parametrize("part", ["residual", "optimizer"])
def test_archive_checks_complete_history_beyond_rebound_digest(
    archive_copy: Path, part: str
) -> None:
    mutate_history(archive_copy, part)
    with pytest.raises(ValueError, match="complete results or residual/optimizer histories differ"):
        audit(archive_copy)


def test_archive_checks_every_interpolated_epoch(archive_copy: Path) -> None:
    mutate_history(archive_copy, "epochs")
    with pytest.raises(ValueError, match="residual uses different windows/pairs"):
        audit(archive_copy)


def test_archive_requires_every_alternating_run(archive_copy: Path) -> None:
    path = archive_copy / "attempts.json"
    value = json.loads(path.read_bytes())
    value["runs"][-1] = value["runs"][0]
    write(path, value)
    with pytest.raises(ValueError, match="duplicate or reordered"):
        audit(archive_copy)


def test_archive_checks_actual_short_source_identity(archive_copy: Path) -> None:
    path = archive_copy / "00-current/measurement.json"
    value = json.loads(path.read_bytes())
    value["provenance"]["git_commit_short"] = "f" * 8
    write(path, value)
    with pytest.raises(ValueError, match="actual Git identities"):
        audit(archive_copy)


def test_archive_binds_original_raw_source_bytes(archive_copy: Path) -> None:
    path = archive_copy / "raw-oxts.tar.gz"
    path.write_bytes(path.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="pinned artifact changed"):
        audit(archive_copy)


def test_observed_work_requires_integer_counts(archive_copy: Path) -> None:
    path = archive_copy / "00-current/measurement.json"
    value = json.loads(path.read_bytes())
    value["counts"]["pose_batch_calls"] = True
    write(path, value)
    with pytest.raises(ValidationError):
        audit(archive_copy)
