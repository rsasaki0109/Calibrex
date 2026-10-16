from __future__ import annotations

import gzip
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
from jsonschema import ValidationError
from tools.benchmark_trajectory_hand_eye_reuse import (
    PINNED,
    _same_reconstructed_geometry,
    audit,
    comparable,
    input_lineage,
)

ARCHIVE = Path(__file__).resolve().parents[2] / "docs/assets/trajectory_hand_eye_evaluation_reuse"


def _write(path: Path, value: object) -> None:
    path.write_bytes((json.dumps(value, indent=2) + "\n").encode())


@pytest.fixture
def archive_copy(tmp_path: Path) -> Path:
    target = tmp_path / "archive"
    shutil.copytree(ARCHIVE, target)
    return target


def test_complete_hand_eye_results_histories_and_work_are_exact() -> None:
    report = audit(ARCHIVE)
    assert report == json.loads((ARCHIVE / "audit.json").read_bytes())
    assert report["verified_runs"] == 6 and report["complete_results_and_histories_exact"]
    assert report["reference_pose_count"] == 2762 and report["sensor_pose_count"] == 2742
    assert report["motion_count"] == 788 and report["jackknife_fits"] == 38
    assert report["work_per_run"] == {
        "prior": {
            "solve_calls": 39,
            "normalized_requests": 17446,
            "least_squares_calls": 410,
            "poses_at_calls": 35000,
            "motion_sigmas_calls": 17460,
        },
        "current": {
            "solve_calls": 39,
            "normalized_requests": 17446,
            "least_squares_calls": 410,
            "poses_at_calls": 6430,
            "motion_sigmas_calls": 53,
        },
    }


def test_every_hand_eye_archive_byte_is_indexed() -> None:
    assert json.loads((ARCHIVE / "index.json").read_bytes())["files"] == {
        p.relative_to(ARCHIVE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in ARCHIVE.rglob("*")
        if p.is_file() and p.name != "index.json"
    }


def test_raw_geometry_reconstruction_allows_only_bounded_platform_roundoff() -> None:
    original = np.tile(np.eye(4), (2, 1, 1))
    changed = original.copy()
    changed[0, 0, 3] = 9.313225746154785e-10  # Observed Windows/glibc Mercator difference.
    changed[1, 0, 0] = np.nextafter(1.0, 0.0)
    assert _same_reconstructed_geometry(original, changed)
    for axis, amount in [((0, 0, 3), 1e-6), ((0, 0, 0), 1e-9), ((0, 3, 3), 1e-15)]:
        changed = original.copy()
        changed[axis] += amount
        assert not _same_reconstructed_geometry(original, changed)
    assert not _same_reconstructed_geometry(original, original[:1])
    changed = original.copy()
    changed[0, 1, 2] = math.nan
    assert not _same_reconstructed_geometry(original, changed)


@pytest.mark.parametrize("part", ["timestamp", "reference", "motion", "motion_id"])
def test_raw_lineage_still_rejects_changed_geometry_times_and_ids(part: str) -> None:
    problem = json.loads((ARCHIVE / "problem.json").read_bytes())
    if part == "timestamp":
        problem["reference_times_s"][3] += 1e-5
    elif part == "reference":
        problem["reference_poses"][3][0][3] += 1e-6
    elif part == "motion":
        problem["motions"][3]["motion"][0][3] += 1e-6
    else:
        problem["motions"][3]["motion_id"] += "-changed"
    with pytest.raises(ValueError, match=r"trajectory differs|sensor motions"):
        input_lineage(ARCHIVE, problem)


def test_observed_work_must_match_full_history_replay(archive_copy: Path) -> None:
    path = archive_copy / "00-current/measurement.json"
    value = json.loads(path.read_bytes())
    value["counts"]["poses_at_calls"] += 1
    _write(path, value)
    with pytest.raises(ValueError, match="observed work differs"):
        audit(archive_copy)


@pytest.mark.parametrize("seconds", [-1.0, math.nan])
def test_hand_eye_archive_requires_valid_whole_process_time(
    archive_copy: Path,
    seconds: float,
) -> None:
    path = archive_copy / "attempts.json"
    value = json.loads(path.read_bytes())
    value["runs"][0]["seconds"] = seconds
    _write(path, value)
    with pytest.raises(ValueError, match="invalid whole-process timing"):
        audit(archive_copy)


@pytest.mark.parametrize("part", ["residual", "optimizer"])
def test_archive_checks_complete_history_beyond_rebound_digest(
    archive_copy: Path, part: str
) -> None:
    path = archive_copy / "00-current/history.json.gz"
    value = json.loads(gzip.decompress(path.read_bytes()))
    if part == "residual":
        value["normalized"][0]["residual_sha256"] = "0" * 64
    else:
        value["least_squares"][0]["nfev"] += 1
    path.write_bytes(gzip.compress((json.dumps(value) + "\n").encode(), mtime=0))
    measurement = path.with_name("measurement.json")
    measured = json.loads(measurement.read_bytes())
    measured["history_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write(measurement, measured)
    with pytest.raises(ValueError, match="complete results or residual/optimizer histories differ"):
        audit(archive_copy)


def test_hand_eye_archive_requires_every_alternating_run(archive_copy: Path) -> None:
    path = archive_copy / "attempts.json"
    value = json.loads(path.read_bytes())
    value["runs"][-1] = value["runs"][0]
    _write(path, value)
    with pytest.raises(ValueError, match="duplicate or reordered"):
        audit(archive_copy)


def test_archive_checks_short_source_id_before_excluding_history_provenance(
    archive_copy: Path,
) -> None:
    path = archive_copy / "00-current/measurement.json"
    value = json.loads(path.read_bytes())
    value["provenance"]["git_commit_short"] = "f" * 8
    _write(path, value)
    with pytest.raises(ValueError, match="actual Git identities"):
        audit(archive_copy)


def test_hand_eye_archive_binds_original_raw_source_bytes(archive_copy: Path) -> None:
    path = archive_copy / "raw-oxts.tar.gz"
    path.write_bytes(path.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="pinned artifact changed"):
        audit(archive_copy)


def test_interrupted_cohort_preserves_every_pin_and_unchanged_core_and_input() -> None:
    earlier = ARCHIVE / "interrupted-v1"
    protocol = json.loads((earlier / "protocol.json").read_bytes())
    assert set(protocol["provenance"]["files"]) == PINNED
    for name, digest in protocol["provenance"]["files"].items():
        assert hashlib.sha256((earlier / name).read_bytes()).hexdigest() == digest
    assert (earlier / "problem.json").read_bytes() == (ARCHIVE / "problem.json").read_bytes()
    for variant in ("prior", "current"):
        assert (earlier / f"sources-{variant}-manifest.json").read_bytes() == (
            ARCHIVE / f"sources-{variant}-manifest.json"
        ).read_bytes()
    attempts = json.loads((earlier / "attempts.json").read_bytes())["runs"]
    assert len(attempts) == 1 and attempts[0]["exit_code"] == 0
    for name in ("result.json", "history.json.gz"):

        def read(folder: Path, filename: str = name) -> dict:
            raw = (folder / filename).read_bytes()
            return json.loads(gzip.decompress(raw) if filename.endswith(".gz") else raw)

        values = []
        for archive in (earlier, ARCHIVE):
            value = read(archive / "00-prior")
            if name.endswith(".gz"):
                assert (
                    value["provenance"]["protocol_sha256"]
                    == hashlib.sha256((archive / "protocol.json").read_bytes()).hexdigest()
                )
            value = comparable(value)
            if name.endswith(".gz"):
                # The two independently pinned cohorts have different protocols.
                value["provenance"].pop("protocol_sha256")
            values.append(value)
        assert values[0] == values[1]


def test_observed_work_requires_integer_counts(archive_copy: Path) -> None:
    path = archive_copy / "00-current/measurement.json"
    value = json.loads(path.read_bytes())
    value["counts"]["poses_at_calls"] = True
    _write(path, value)
    with pytest.raises(ValidationError):
        audit(archive_copy)
