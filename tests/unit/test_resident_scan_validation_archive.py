"""Replay the scan-validation speed evidence on CPU and reject rebound claims."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "docs/assets/resident_scan_validation"


def _run(path: Path) -> subprocess.CompletedProcess[str]:
    code = """
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith("cupy"):
        raise AssertionError("archive audit requested a GPU runtime")
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = ["audit", sys.argv[1], "--check"]
runpy.run_module("tools.audit_resident_scan_validation", run_name="__main__")
"""
    env = dict(os.environ)
    env.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
    return subprocess.run(
        [sys.executable, "-c", code, str(path)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )


def test_complete_scan_validation_cohort_replays_without_cuda() -> None:
    result = _run(ARCHIVE)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "confirmed"
    assert report["complete_workers"] == 6
    assert report["complete_fit_bit_comparisons"] == 12
    assert report["point_rows_bit_compared"] == 1188000
    assert report["faster_pairs"] >= 2
    assert not report["gpu_runtime_imported"]


@pytest.mark.parametrize("mutation", ["payload", "rebound-speed", "rebound-gates"])
def test_scan_validation_archive_rejects_tampering(tmp_path: Path, mutation: str) -> None:
    root = tmp_path / "archive"
    shutil.copytree(ARCHIVE, root)
    if mutation == "payload":
        path = root / "00-current/result.json"
        path.write_bytes(path.read_bytes() + b"\n")
        expected = "published asset checksum differs"
    else:
        name = "summary.json" if mutation == "rebound-speed" else "protocol.json"
        path = root / name
        value = json.loads(path.read_bytes())
        if mutation == "rebound-speed":
            value["prior_to_current_ratio"] *= 2
            expected = "reported ratios or gates differ"
        else:
            value["gates"]["median_prior_to_current_minimum"] = 0.5
            expected = "declared sources, input, options or gates differ"
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        index_path = root / "archive-index.json"
        index = json.loads(index_path.read_bytes())
        index["files"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if mutation == "rebound-gates":
            index["provenance"]["protocol_sha256"] = index["files"][name]
        index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    result = _run(root)
    assert result.returncode != 0
    assert expected in result.stderr
