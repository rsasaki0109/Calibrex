from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import pytest
from tools.benchmark_radar_reference_indexing import audit

ARCHIVE = Path(__file__).resolve().parents[2] / "docs/assets/joint_radar_reference_indexing"


def _write(path: Path, value: object) -> None:
    path.write_bytes((json.dumps(value, indent=2) + "\n").encode())


@pytest.fixture
def archive_copy(tmp_path: Path) -> Path:
    target = tmp_path / "archive"
    shutil.copytree(ARCHIVE, target)
    return target


def test_complete_joint_radar_archive() -> None:
    report = audit(ARCHIVE)
    assert report == json.loads((ARCHIVE / "audit.json").read_bytes())
    assert report["solver"] == "joint" and report["verified_runs"] == 6
    assert report["reference_count"] == 2762 and report["measurement_count"] == 2758
    assert report["optimizer_iterations_per_run"] == 5
    assert report["eligible_measurements"] == 2099 and report["rejected_measurements"] == 659
    assert report["signed_probes"] == 14
    assert report["interpolation_calls_per_run"] == 171502
    prior = ARCHIVE.parent / "radar_reference_indexing/control.json"
    assert (ARCHIVE / "control.json").read_bytes() == prior.read_bytes()


def test_every_joint_archive_byte_is_indexed() -> None:
    index = json.loads((ARCHIVE / "index.json").read_bytes())
    files = {
        path.relative_to(ARCHIVE).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in ARCHIVE.rglob("*")
        if path.is_file() and path.name != "index.json"
    }
    assert files == index["files"]


@pytest.mark.parametrize("seconds", [0.0, -1.0, math.inf, math.nan])
def test_joint_archive_rejects_invalid_timing(archive_copy: Path, seconds: float) -> None:
    path = archive_copy / "attempts.json"
    records = json.loads(path.read_bytes())
    records[0]["seconds"] = seconds
    _write(path, records)
    with pytest.raises(ValueError, match="invalid whole-process timing"):
        audit(archive_copy)


def test_joint_archive_rejects_duplicate_runs(archive_copy: Path) -> None:
    path = archive_copy / "attempts.json"
    records = json.loads(path.read_bytes())
    records[-1] = records[0]
    _write(path, records)
    with pytest.raises(ValueError, match="missing, duplicate or reordered"):
        audit(archive_copy)


@pytest.mark.parametrize("part", ["history", "probe", "rejected_ids"])
def test_joint_archive_checks_all_results_beyond_the_digest(archive_copy: Path, part: str) -> None:
    folder = archive_copy / "00-current"
    path = folder / "result.json"
    value = json.loads(path.read_bytes())
    if part == "history":
        value["result"]["optimizer"]["history"][0]["train_rmse"] += 1.0e-6
    elif part == "probe":
        value["result"]["known_bad_probes"].pop()
    else:
        value["result"]["rejected_initial_residual_ids"].pop()
    _write(path, value)
    measured = json.loads((folder / "measurement.json").read_bytes())
    measured["result_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write(folder / "measurement.json", measured)
    with pytest.raises(
        ValueError, match=r"complete results|complete its calibration|input measurements"
    ):
        audit(archive_copy)


@pytest.mark.parametrize("calls", [0, 171503])
def test_joint_archive_rejects_changed_evaluation_counts(archive_copy: Path, calls: int) -> None:
    path = archive_copy / "00-current/measurement.json"
    measured = json.loads(path.read_bytes())
    measured["interpolation_calls"] = calls
    _write(path, measured)
    with pytest.raises(ValueError, match="interpolation call count"):
        audit(archive_copy)


def test_joint_archive_rejects_changed_control_generator(archive_copy: Path) -> None:
    path = archive_copy / "control-generator.py.txt"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="input generator snapshot changed"):
        audit(archive_copy)


def test_joint_archive_requires_the_measured_solver_import(archive_copy: Path) -> None:
    path = archive_copy / "00-current/measurement.json"
    measured = json.loads(path.read_bytes())
    name = "calibrex.solvers.radar_joint_spatiotemporal_solver"
    del measured["imports"][name]
    _write(path, measured)
    with pytest.raises(ValueError, match="measured solver import is missing"):
        audit(archive_copy)
