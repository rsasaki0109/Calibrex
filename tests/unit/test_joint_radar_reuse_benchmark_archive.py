from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from tools.benchmark_radar_reference_indexing import audit

ARCHIVE = Path(__file__).resolve().parents[2] / "docs/assets/joint_radar_evaluation_reuse"


def _write(path: Path, value: object) -> None:
    path.write_bytes((json.dumps(value, indent=2) + "\n").encode())


@pytest.fixture
def archive_copy(tmp_path: Path) -> Path:
    target = tmp_path / "archive"
    shutil.copytree(ARCHIVE, target)
    return target


def test_complete_joint_radar_reuse_archive() -> None:
    report = audit(ARCHIVE)
    assert report == json.loads((ARCHIVE / "audit.json").read_bytes())
    assert report["verified_runs"] == 6 and report["complete_results_exact"]
    assert report["reference_count"] == 2762 and report["measurement_count"] == 2758
    assert report["optimizer_iterations_per_run"] == 5 and report["signed_probes"] == 14
    assert report["eligible_measurements"] == 2099 and report["rejected_measurements"] == 659
    assert report["factor_evaluations_per_run"] == 162024
    assert report["interpolation_calls_per_run"] == {"prior": 171502, "current": 41799}
    assert report["transform_decodes_per_run"] == {"prior": 162037, "current": 91}
    assert (ARCHIVE / "control.json").read_bytes() == (
        ARCHIVE.parent / "joint_radar_reference_indexing/control.json"
    ).read_bytes()


def test_every_reuse_archive_byte_is_indexed() -> None:
    index = json.loads((ARCHIVE / "index.json").read_bytes())
    assert index["files"] == {
        p.relative_to(ARCHIVE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in ARCHIVE.rglob("*")
        if p.is_file() and p.name != "index.json"
    }


def test_reuse_archive_requires_work_counters(archive_copy: Path) -> None:
    path = archive_copy / "00-current/measurement.json"
    value = json.loads(path.read_bytes())
    del value["work_counters"]
    _write(path, value)
    with pytest.raises(ValueError, match="invalid joint work counters"):
        audit(archive_copy)


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_reuse_archive_rejects_invalid_counters(archive_copy: Path, count: object) -> None:
    path = archive_copy / "00-current/measurement.json"
    value = json.loads(path.read_bytes())
    value["work_counters"]["transform_decodes"] = count
    _write(path, value)
    with pytest.raises(ValueError, match="invalid joint work counters"):
        audit(archive_copy)


@pytest.mark.parametrize(
    "counter", ["interpolation_calls", "factor_evaluations", "transform_decodes"]
)
def test_reuse_archive_requires_identical_repeated_counts(archive_copy: Path, counter: str) -> None:
    path = archive_copy / "00-current/measurement.json"
    value = json.loads(path.read_bytes())
    target = value if counter == "interpolation_calls" else value["work_counters"]
    target[counter] += 1
    _write(path, value)
    with pytest.raises(ValueError, match="work counters differ between repeated runs"):
        audit(archive_copy)


def test_reuse_archive_preserves_factor_evaluation_budget(archive_copy: Path) -> None:
    for path in archive_copy.glob("*-current/measurement.json"):
        value = json.loads(path.read_bytes())
        value["work_counters"]["factor_evaluations"] -= 1
        _write(path, value)
    with pytest.raises(ValueError, match="factor evaluation counts differ"):
        audit(archive_copy)


@pytest.mark.parametrize("counter", ["interpolation_calls", "transform_decodes"])
def test_reuse_archive_requires_less_repeated_computation(archive_copy: Path, counter: str) -> None:
    for path in archive_copy.glob("*-current/measurement.json"):
        value = json.loads(path.read_bytes())
        target = value if counter == "interpolation_calls" else value["work_counters"]
        target[counter] = 171502 if counter == "interpolation_calls" else 162037
        _write(path, value)
    with pytest.raises(ValueError, match="reuse did not reduce"):
        audit(archive_copy)


def test_reuse_archive_checks_history_beyond_the_digest(archive_copy: Path) -> None:
    path = archive_copy / "00-current/result.json"
    value = json.loads(path.read_bytes())
    value["result"]["optimizer"]["history"][0]["step_norm"] += 1.0e-6
    _write(path, value)
    measurement = path.with_name("measurement.json")
    measured = json.loads(measurement.read_bytes())
    measured["result_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write(measurement, measured)
    with pytest.raises(ValueError, match="complete results"):
        audit(archive_copy)
