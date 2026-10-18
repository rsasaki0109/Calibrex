"""Pipeline follow-ups preserve exact evidence and frozen numerical quality gates."""

import copy
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
ARCHIVE = ROOT / "docs/assets/lidar_pipeline_followup"
GUARDED = """
import builtins, json, sys
from pathlib import Path
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0]=='cupy':
        raise AssertionError('CPU archive replay imported CUDA')
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from tools.audit_lidar_pipeline_followup import audit_archive
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
    root = tmp_path / "assets/lidar_pipeline_followup"
    shutil.copytree(ARCHIVE, root)
    for directory, names in {
        "resident_scan_validation": ("current.sources.tar.gz", "current.sources-manifest.json"),
        "cpu_map_query_cache": ("lazy.source-delta.tar.gz", "lazy.sources-manifest.json"),
        "cpu_map_bag_validation": ("current.source-delta.tar.gz", "current.sources-manifest.json"),
    }.items():
        target = root.parent / directory
        target.mkdir()
        for name in names:
            shutil.copyfile(ROOT / "docs/assets" / directory / name, target / name)
    shutil.copytree(
        ROOT / "docs/assets/lidar_execution_screen/fixtures",
        root.parent / "lidar_execution_screen/fixtures",
    )
    return root


def _rebind(root, name, value):
    raw = json.dumps(value, indent=2, allow_nan=False).encode() + b"\n"
    (root / name).write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    index = json.loads((root / "archive-index.json").read_bytes())
    index["files"][name] = digest
    (root / "archive-index.json").write_text(json.dumps(index), encoding="utf-8")
    return digest


def test_complete_followup_replay_is_cpu_only_and_keeps_scopes_separate():
    assert (ROOT / "tools/audit_lidar_pipeline_followup.py").read_bytes() == (
        ARCHIVE / "auditor.py.txt"
    ).read_bytes()
    completed = _audit(ARCHIVE)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report == json.loads((ARCHIVE / "audit.json").read_bytes())
    assert report["timed_registrations"] == 480 and report["retained_warmups"] == 10
    assert report["registration_bit_comparisons"] == 360
    assert report["registration_tolerance_comparisons"] == 90
    assert report["complete_fits"] == 8 and report["diagnostic_fits"] == 4
    assert not report["gpu_runtime_imported"]
    assert report["whole"]["whole_ratio"] == pytest.approx(1.0288056314484246)


def test_rebound_whole_fit_covariance_change_is_rejected(copied):
    name = "whole/current/measurement.json"
    row = json.loads((copied / name).read_bytes())
    covariance = row["rounds"][0]["calibration_covariance"]
    covariance[0][0] = math.nextafter(covariance[0][0], math.inf)
    digest = _rebind(copied, name, row)
    name = "whole/current/whole.json"
    receipt = json.loads((copied / name).read_bytes())
    receipt["measurement_sha256"] = digest
    _rebind(copied, name, receipt)
    completed = _audit(copied)
    assert completed.returncode != 0 and "whole full-fit bits differ" in completed.stderr


def test_rebound_whole_time_cannot_replace_original_summary(copied):
    name = "whole/current/whole.json"
    receipt = json.loads((copied / name).read_bytes())
    receipt["seconds"] *= 1.01
    _rebind(copied, name, receipt)
    completed = _audit(copied)
    assert completed.returncode != 0 and "whole original summary differs" in completed.stderr


def test_duplicate_registration_cannot_hide_slow_observation(copied):
    name = "resident-tree/runs.json"
    runs = json.loads((copied / name).read_bytes())
    runs[1] = copy.deepcopy(runs[0])
    _rebind(copied, name, runs)
    completed = _audit(copied)
    assert (
        completed.returncode != 0
        and "observations missing, duplicated or reordered" in completed.stderr
    )


def test_frozen_protocol_refuses_looser_quality_threshold(copied):
    name = "persistent/protocol.json"
    protocol = json.loads((copied / name).read_bytes())
    protocol["quality"]["rtol"] = 1e-3
    _rebind(copied, name, protocol)
    completed = _audit(copied)
    assert completed.returncode != 0 and "original protocol differs" in completed.stderr


def test_persistent_quality_uses_existing_gpu_tolerance():
    from tools.audit_lidar_pipeline_followup import _close

    reference = json.loads((ARCHIVE / "persistent/runs.json").read_bytes())[0]["result"]
    near = copy.deepcopy(reference)
    near["rmse_m"] *= 1 + 1e-7
    _close(reference, near)
    distant = copy.deepcopy(reference)
    distant["rmse_m"] *= 1.01
    with pytest.raises(ValueError, match="persistent registration quality differs"):
        _close(reference, distant)


def test_persistent_convergence_decisions_remain_exact():
    from tools.audit_lidar_pipeline_followup import _close

    reference = json.loads((ARCHIVE / "persistent/runs.json").read_bytes())[0]["result"]
    changed = copy.deepcopy(reference)
    changed["iterations"] += 1
    with pytest.raises(ValueError, match="persistent registration decisions differ"):
        _close(reference, changed)
