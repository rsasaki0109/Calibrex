"""Recompute the CPU screen and reject rebound physical/work/source claims."""

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
ARCHIVE = ROOT / "docs/assets/cpu_map_query_cache"
GUARDED = """
import builtins, json, sys
from pathlib import Path
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] == 'cupy':
        raise AssertionError('CPU evidence replay imported CUDA')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from tools.audit_cpu_map_query_cache import audit_archive
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
    root = tmp_path / "assets/cpu_map_query_cache"
    shutil.copytree(ARCHIVE, root)
    dependency = root.parent / "resident_scan_validation"
    dependency.mkdir()
    for name in ("current.sources.tar.gz", "current.sources-manifest.json"):
        shutil.copyfile(ROOT / "docs/assets/resident_scan_validation" / name, dependency / name)
    return root


def _rebind(root, name, value):
    raw = json.dumps(value, indent=2, allow_nan=False).encode() + b"\n"
    (root / name).write_bytes(raw)
    index = json.loads((root / "archive-index.json").read_bytes())
    index["files"][name] = hashlib.sha256(raw).hexdigest()
    (root / "archive-index.json").write_text(json.dumps(index), encoding="utf-8")


def test_complete_cpu_replay_retains_both_missed_gates_and_float_bits():
    result = _audit(ARCHIVE)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["complete_workers"] == 18
    assert report["complete_fits"] == 90 and report["fit_bit_comparisons"] == 60
    assert not report["gpu_runtime_imported"]
    first, second = report["cohorts"]["lazy-v1"], report["cohorts"]["certified-v2"]
    assert first["physical_bit_parity"] and second["physical_bit_parity"]
    assert first["metadata_only_comparator_correction"] and not first["original_quality_passed"]
    assert not first["advance"] and not second["advance"]
    assert first["prior_to_candidate"] == pytest.approx(1.9873817946551153)
    assert second["lazy_to_candidate"] < 1.0
    assert report == json.loads((ARCHIVE / "audit.json").read_bytes())


def test_single_ulp_covariance_change_is_rejected_with_rebound_checksum(copied):
    name = "certified-v2/00-lazy/measurement.json"
    record = json.loads((copied / name).read_bytes())
    matrix = record["evaluation"]["result"]["covariance"]
    matrix[0][0] = math.nextafter(matrix[0][0], math.inf)
    _rebind(copied, name, record)
    result = _audit(copied)
    assert (
        result.returncode != 0
        and "physical result, covariance or evidence bits differ" in result.stderr
    )


def test_rebound_original_advancement_claim_is_rejected(copied):
    name = "lazy-v1/summary.json"
    record = json.loads((copied / name).read_bytes())
    record["advance"] = True
    _rebind(copied, name, record)
    result = _audit(copied)
    assert result.returncode != 0 and "original decision differs" in result.stderr


def test_rebound_worker_budget_claim_is_rejected(copied):
    name = "certified-v2/00-certified/measurement.json"
    record = json.loads((copied / name).read_bytes())
    record["work"]["requested_workers"] = 4
    _rebind(copied, name, record)
    result = _audit(copied)
    assert result.returncode != 0 and "worker execution budget differs" in result.stderr


def test_rebound_source_manifest_is_rejected(copied):
    name = "lazy.sources-manifest.json"
    record = json.loads((copied / name).read_bytes())
    record["calibrex/solvers/lidar_lidar_map_solver.py"] = "0" * 64
    _rebind(copied, name, record)
    result = _audit(copied)
    assert result.returncode != 0 and "complete source manifest differs" in result.stderr
