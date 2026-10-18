"""Rejected screens stay replayable on CPU and resist rebound evidence changes."""

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "docs/assets/lidar_execution_screen"
GUARDED = """
import builtins, json, sys
from pathlib import Path
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] == 'cupy':
        raise AssertionError('CPU replay imported CUDA')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from tools.audit_lidar_execution_screen import audit_archive
print(json.dumps(audit_archive(Path(sys.argv[1]))))
"""


def _audit(root):
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
    return subprocess.run(
        [sys.executable, "-c", GUARDED, str(root)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def copied(tmp_path):
    root = tmp_path / "assets/lidar_execution_screen"
    shutil.copytree(ARCHIVE, root)
    for directory, names in {
        "resident_scan_validation": ("current.sources.tar.gz", "current.sources-manifest.json"),
        "cpu_map_query_cache": ("lazy.source-delta.tar.gz", "lazy.sources-manifest.json"),
        "cpu_map_bag_validation": ("current.source-delta.tar.gz", "current.sources-manifest.json"),
    }.items():
        destination = root.parent / directory
        destination.mkdir()
        for name in names:
            shutil.copyfile(ROOT / "docs/assets" / directory / name, destination / name)
    return root


def _rebind(root, name, value):
    raw = json.dumps(value, indent=2, allow_nan=False).encode() + b"\n"
    (root / name).write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    index = json.loads((root / "archive-index.json").read_bytes())
    index["files"][name] = digest
    (root / "archive-index.json").write_text(json.dumps(index), encoding="utf-8")
    return digest


def test_rejected_screens_replay_without_optional_runtime():
    assert (ROOT / "tools/audit_lidar_execution_screen.py").read_bytes() == (
        ARCHIVE / "auditor.py.txt"
    ).read_bytes()
    completed = _audit(ARCHIVE)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report == json.loads((ARCHIVE / "audit.json").read_bytes())
    assert report["decision"] == "no_candidate_adopted"
    assert report["complete_map_fits"] == 20 and report["gpu_timed_registrations"] == 288
    assert report["retained_gpu_warmups"] == 7 and report["registration_bit_comparisons"] == 271
    assert not report["gpu_runtime_imported"]
    assert report["cohorts"]["sample-parallel"]["ratios"]["samples"] == pytest.approx(
        1.1312772949467897
    )


def test_rebound_single_ulp_full_fit_change_is_rejected(copied):
    name = "sample-parallel/00-samples/measurement.json"
    value = json.loads((copied / name).read_bytes())
    covariance = value["full_fits"][0]["covariance"]
    covariance[0][0] = math.nextafter(covariance[0][0], math.inf)
    digest = _rebind(copied, name, value)
    receipt_name = "sample-parallel/00-samples/whole.json"
    receipt = json.loads((copied / receipt_name).read_bytes())
    receipt["measurement_sha256"] = digest
    _rebind(copied, receipt_name, receipt)
    completed = _audit(copied)
    assert completed.returncode != 0 and "physical fit/evidence bits differ" in completed.stderr


def test_rebound_registration_quality_change_is_rejected(copied):
    name = "normal-fusion/runs.json"
    value = json.loads((copied / name).read_bytes())
    value[8]["result"]["hessian_min_eigenvalue"] = math.nextafter(
        value[8]["result"]["hessian_min_eigenvalue"], math.inf
    )
    _rebind(copied, name, value)
    completed = _audit(copied)
    assert (
        completed.returncode != 0 and "registration physical bits/work differ" in completed.stderr
    )


def test_rebound_timings_cannot_replace_original_supervisor(copied):
    folder = copied / "normal-fusion"
    runs = json.loads((folder / "runs.json").read_bytes())
    for row in runs:
        if row["variant"] == "fused":
            for key in ("seconds", "target_seconds", "registration_seconds"):
                row[key] *= 0.5
    digest = _rebind(copied, "normal-fusion/runs.json", runs)
    import statistics

    summary = json.loads((folder / "summary.json").read_bytes())
    summary["runs_sha256"] = digest
    medians = {
        v: statistics.median(
            sum(r["seconds"] for r in runs if r["repeat"] == i and r["variant"] == v)
            for i in range(6)
        )
        for v in ("prior", "fused")
    }
    summary["medians_seconds"] = medians
    summary["ratio"] = medians["prior"] / medians["fused"]
    _rebind(copied, "normal-fusion/summary.json", summary)
    completed = _audit(copied)
    assert completed.returncode != 0 and "original supervisor differs" in completed.stderr


def test_rebound_source_manifest_is_rejected(copied):
    path = copied.parent / "cpu_map_bag_validation/current.sources-manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["calibrex/solvers/scan_to_scan_odometry.py"] = "0" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")
    index = json.loads((copied / "archive-index.json").read_bytes())
    index["source_dependencies"]["cpu_map_bag_validation"][path.name] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    (copied / "archive-index.json").write_text(json.dumps(index), encoding="utf-8")
    completed = _audit(copied)
    assert completed.returncode != 0 and "source manifest differs" in completed.stderr
