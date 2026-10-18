"""Whole-bag evidence replays on CPU and rejects rebound work and timing claims."""

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
ARCHIVE = ROOT / "docs/assets/cpu_map_bag_validation"
GUARDED = """
import builtins, json, sys
from pathlib import Path
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] == 'cupy':
        raise AssertionError('CPU archive replay imported CUDA')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from tools.audit_cpu_map_bag import audit_archive
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
    root = tmp_path / "assets/cpu_map_bag_validation"
    shutil.copytree(ARCHIVE, root)
    for directory, names in {
        "resident_scan_validation": ("current.sources.tar.gz", "current.sources-manifest.json"),
        "cpu_map_query_cache": ("lazy.source-delta.tar.gz", "lazy.sources-manifest.json"),
    }.items():
        dependency = root.parent / directory
        dependency.mkdir()
        for name in names:
            shutil.copyfile(ROOT / "docs/assets" / directory / name, dependency / name)
    return root


def _rebind(root, name, value):
    raw = json.dumps(value, indent=2, allow_nan=False).encode() + b"\n"
    (root / name).write_bytes(raw)
    index = json.loads((root / "archive-index.json").read_bytes())
    index["files"][name] = hashlib.sha256(raw).hexdigest()
    (root / "archive-index.json").write_text(json.dumps(index), encoding="utf-8")
    return hashlib.sha256(raw).hexdigest()


def test_complete_bag_replay_checks_all_bits_and_keeps_densities_separate():
    result = _audit(ARCHIVE)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report == json.loads((ARCHIVE / "audit.json").read_bytes())
    assert report["complete_workers"] == 18 and report["complete_fits"] == 90
    assert report["fit_bit_comparisons"] == 60 and not report["gpu_runtime_imported"]
    coarse, dense = (report["scenarios"][name] for name in ("map-000", "map-001"))
    assert coarse["map_voxel_size_m"] == 0.2 and dense["map_voxel_size_m"] == 0.03
    assert coarse["whole"]["prior_to_parallel"] == pytest.approx(1.2486124256780935)
    assert dense["whole"]["prior_to_parallel"] == pytest.approx(1.6870537091557483)
    assert coarse["whole"]["parallel_faster_pairs"] == dense["whole"]["parallel_faster_pairs"] == 3


def test_single_ulp_fit_covariance_change_is_rejected_after_rebinding(copied):
    name = "map-001/00-parallel/measurement.json"
    record = json.loads((copied / name).read_bytes())
    covariance = record["fits"][0]["covariance"]
    covariance[0][0] = math.nextafter(covariance[0][0], math.inf)
    _rebind(copied, name, record)
    result = _audit(copied)
    assert result.returncode != 0 and "physical fit/evidence bits differ" in result.stderr


def test_rebound_whole_time_cannot_replace_the_original_triad(copied):
    name = "map-001/00-parallel/whole.json"
    record = json.loads((copied / name).read_bytes())
    record["seconds"] *= 0.95
    _rebind(copied, name, record)
    result = _audit(copied)
    assert result.returncode != 0 and "original triad binding differs" in result.stderr


def test_rebound_kernel_identity_is_checked_against_the_executed_source(copied):
    prefix = "map-000/00-lazy/"
    artifact = json.loads((copied / (prefix + "result.json")).read_bytes())
    identity = artifact["options"]["odometry_execution"]
    tokens = identity.split(";")
    artifact["options"]["odometry_execution"] = ";".join(
        "query_sha256=" + "0" * 64 if token.startswith("query_sha256=") else token
        for token in tokens
    )
    digest = _rebind(copied, prefix + "result.json", artifact)
    record = json.loads((copied / (prefix + "measurement.json")).read_bytes())
    record["artifact"] = artifact
    record["provenance"]["result_sha256"] = digest
    _rebind(copied, prefix + "measurement.json", record)
    result = _audit(copied)
    assert result.returncode != 0 and "common odometry kernel identity differs" in result.stderr


def test_rebound_current_manifest_cannot_change_the_frozen_source(copied):
    name = "current.sources-manifest.json"
    record = json.loads((copied / name).read_bytes())
    record["calibrex/evaluation/lidar_lidar_map.py"] = "0" * 64
    _rebind(copied, name, record)
    result = _audit(copied)
    assert result.returncode != 0 and "complete source manifest differs" in result.stderr
