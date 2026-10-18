"""Replay the complete publication with CUDA imports blocked and detect forged speed claims."""

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
ARCHIVE = ROOT / "docs/assets/resident_lidar_geometry"


def _run(path: Path) -> subprocess.CompletedProcess[str]:
    code = """
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith("cupy"):
        raise AssertionError("archive validation requested a GPU runtime")
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = ["audit", sys.argv[1], "--check"]
runpy.run_module("tools.audit_resident_lidar_geometry", run_name="__main__")
"""
    env = dict(os.environ)
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return subprocess.run(
        [sys.executable, "-c", code, str(path)], cwd=ROOT, env=env,
        capture_output=True, text=True, timeout=90,
    )


def test_complete_publication_recomputes_without_optional_gpu_runtime() -> None:
    result = _run(ARCHIVE)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "confirmed"
    assert report["complete_workers"] == 9
    assert report["cpu_gpu_fit_comparisons"] == 24
    assert report["gpu_bit_exact_fit_comparisons"] == 12
    assert report["gpu_bit_exact_point_rows"] == 1188000
    assert report["cpu_improved_runs"] == 3
    assert report["native_improved_runs"] == 2
    assert not report["gpu_runtime_imported"]


@pytest.mark.parametrize("mutation", ["payload", "rebound_speed"])
def test_publication_rejects_tampering_even_after_rebinding_summary_checksum(
    tmp_path: Path, mutation: str,
) -> None:
    archive = tmp_path / "archive"
    shutil.copytree(ARCHIVE, archive)
    if mutation == "payload":
        target = archive / "00-resident/result.json"
        target.write_bytes(target.read_bytes() + b"\n")
        message = "payload checksum differs"  # The outer checksum must fail first.
    else:
        target = archive / "summary.json"
        summary = json.loads(target.read_bytes())
        summary["native_to_resident"] *= 2
        target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        index_path = archive / "archive-index.json"
        index = json.loads(index_path.read_bytes())
        index["files"]["summary.json"] = hashlib.sha256(target.read_bytes()).hexdigest()
        index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        message = "ratios or gates differ"
    result = _run(archive)
    assert result.returncode != 0
    if mutation == "payload":
        assert "published asset checksum differs" in result.stderr
    else:
        assert message in result.stderr
