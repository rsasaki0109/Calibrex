from __future__ import annotations

import gzip
import hashlib
import json
import math
import shutil
from pathlib import Path

import pytest
from jsonschema import ValidationError
from tools import benchmark_lidar_wheel_preparation as benchmark

ARCHIVE = Path(__file__).resolve().parents[2] / "docs/assets/lidar_wheel_evaluation_preparation"


def write(path: Path, value: dict) -> None:
    path.write_bytes((json.dumps(value, indent=2) + "\n").encode())


@pytest.fixture
def archive_copy(tmp_path: Path) -> Path:
    target = tmp_path / "archive"
    shutil.copytree(ARCHIVE, target)
    return target


def test_all_results_histories_and_replayed_work_match() -> None:
    report = benchmark.audit(ARCHIVE)
    assert report == json.loads((ARCHIVE / "audit.json").read_bytes())
    assert report["verified_runs"] == 6 and report["complete_results_and_histories_exact"]
    assert report["motion_count"] == 2760 and report["block_count"] == 29
    assert report["train_motions"] == 1824 and report["holdout_motions"] == 936
    assert report["jackknife_fits"] == 8
    assert report["work_per_run"]["prior"] == {
        "solve_calls": 9,
        "residual_requests": 411,
        "least_squares_calls": 9,
        "moving_calls": 420,
        "motion_selection_visits": 646689,
        "wheel_at_calls": 3120,
        "wheel_samples_queried": 4829055,
    }
    assert report["work_per_run"]["current"] == {
        "solve_calls": 9,
        "residual_requests": 411,
        "least_squares_calls": 9,
        "moving_calls": 17,
        "motion_selection_visits": 22968,
        "wheel_at_calls": 2836,
        "wheel_samples_queried": 4389516,
    }


def test_every_published_byte_is_indexed() -> None:
    assert json.loads((ARCHIVE / "index.json").read_bytes())["files"] == {
        p.relative_to(ARCHIVE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in ARCHIVE.rglob("*")
        if p.is_file() and p.name != "index.json"
    }


def test_audit_accepts_a_generator_checkout_with_other_newline_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = (ARCHIVE / "control-generator.py.txt").read_bytes()
    target = tmp_path / "tools/prepare_lidar_wheel_control.py"
    target.parent.mkdir()
    lf = source.replace(b"\r\n", b"\n")
    changed = lf if source != lf else lf.replace(b"\n", b"\r\n")
    assert changed != source
    target.write_bytes(changed)
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    assert benchmark.audit(ARCHIVE) == json.loads((ARCHIVE / "audit.json").read_bytes())


def test_observed_counts_must_match_complete_trace(archive_copy: Path) -> None:
    path = archive_copy / "00-current/measurement.json"
    value = json.loads(path.read_bytes())
    value["counts"]["wheel_at_calls"] += 1
    write(path, value)
    with pytest.raises(ValueError, match="observed work differs"):
        benchmark.audit(archive_copy)


@pytest.mark.parametrize("seconds", [-1.0, math.nan])
def test_whole_process_time_must_be_positive_and_finite(archive_copy: Path, seconds: float) -> None:
    path = archive_copy / "attempts.json"
    value = json.loads(path.read_bytes())
    value["runs"][0]["seconds"] = seconds
    write(path, value)
    with pytest.raises(ValueError, match="invalid whole-process timing"):
        benchmark.audit(archive_copy)


@pytest.mark.parametrize("part", ["residual", "optimizer", "rows", "jacobian", "clock", "control"])
def test_full_history_mutations_fail_even_after_rebinding_the_digest(
    archive_copy: Path, part: str
) -> None:
    folder = archive_copy / "00-current"
    path = folder / "history.json.gz"
    value = json.loads(gzip.decompress(path.read_bytes()))
    error = "complete results or residual/optimizer histories differ"
    if part == "residual":
        value["residuals"][0]["residual_sha256"] = "0" * 64
    elif part == "optimizer":
        value["least_squares"][0]["nfev"] += 1
    elif part == "rows":
        value["residuals"][0]["selected_count"] -= 1
        error = "residual uses different motions"
    elif part == "jacobian":
        value["least_squares"][0]["jac_shape"][0] -= 1
        error = "optimizer residual/Jacobian dimensions differ"
    elif part == "clock":
        for event in value["residuals"]:
            if event["fit"] == 0:
                event["params"][5] = 0.0
        error = "clock optimization was not exercised"
    else:
        value["residuals"] = [e for e in value["residuals"] if e["fit"] is not None]
        error = "held-out controls or median summaries differ"
    path.write_bytes(gzip.compress((json.dumps(value) + "\n").encode(), mtime=0))
    measured = json.loads((folder / "measurement.json").read_bytes())
    measured["history_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    write(folder / "measurement.json", measured)
    with pytest.raises(ValueError, match=error):
        benchmark.audit(archive_copy)


def test_every_alternating_run_is_required(archive_copy: Path) -> None:
    path = archive_copy / "attempts.json"
    value = json.loads(path.read_bytes())
    value["runs"][-1] = value["runs"][0]
    write(path, value)
    with pytest.raises(ValueError, match="duplicate or reordered"):
        benchmark.audit(archive_copy)


def test_actual_short_commit_must_match_the_source_pin(archive_copy: Path) -> None:
    path = archive_copy / "00-current/measurement.json"
    value = json.loads(path.read_bytes())
    value["provenance"]["git_commit_short"] = "f" * 8
    write(path, value)
    with pytest.raises(ValueError, match="actual Git identities"):
        benchmark.audit(archive_copy)


def test_original_raw_source_bytes_are_bound(archive_copy: Path) -> None:
    path = archive_copy / "raw-oxts.tar.gz"
    path.write_bytes(path.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="pinned artifact changed"):
        benchmark.audit(archive_copy)


def test_boolean_counts_are_rejected(archive_copy: Path) -> None:
    path = archive_copy / "00-current/measurement.json"
    value = json.loads(path.read_bytes())
    value["counts"]["moving_calls"] = True
    write(path, value)
    with pytest.raises(ValidationError):
        benchmark.audit(archive_copy)
