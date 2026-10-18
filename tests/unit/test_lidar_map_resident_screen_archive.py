"""Retain rejected performance evidence and reject rebound speed or source claims."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "docs/assets/lidar_map_resident_screen"


def _run(root: Path) -> subprocess.CompletedProcess[str]:
    code = """
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith("cupy"):
        raise AssertionError("screen audit requested a GPU")
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = ["audit", sys.argv[1], "--check"]
runpy.run_module("tools.audit_lidar_map_resident_screen", run_name="__main__")
"""
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
    return subprocess.run(
        [sys.executable, "-c", code, str(root)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )


def test_rejected_screen_replays_without_gpu() -> None:
    run = _run(ARCHIVE)
    assert run.returncode == 0, run.stderr
    report = json.loads(run.stdout)
    assert report["decision"] == "not_adopted" and not report["production_changed"]
    assert report["complete_workers"] == 9 and report["complete_fits"] == 45
    assert report["cpu_gpu_fit_comparisons"] == 30 and report["gpu_fit_bit_comparisons"] == 15
    assert report["candidate_faster_pairs"] == 0 and report["candidate_slowdown_fraction"] > 0
    assert not report["gpu_runtime_imported"]


@pytest.mark.parametrize("mutation", ["payload", "advance", "kernel", "source-delta"])
def test_rejected_screen_rejects_tampering(tmp_path: Path, mutation: str) -> None:
    root = tmp_path / "lidar_map_resident_screen"
    shutil.copytree(ARCHIVE, root)
    dependency = tmp_path / "resident_scan_validation"
    dependency.mkdir()
    for name in ("current.sources-manifest.json", "current.sources.tar.gz"):
        shutil.copyfile(ARCHIVE.parent / "resident_scan_validation" / name, dependency / name)
    index_path = root / "archive-index.json"
    index = json.loads(index_path.read_bytes())

    def rewrite(name: str, value: object) -> None:
        path = root / name
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        index["files"][name] = hashlib.sha256(path.read_bytes()).hexdigest()

    if mutation == "payload":
        path = root / "00-resident/result.json"
        path.write_bytes(path.read_bytes() + b"\n")
        message = "published asset checksum differs"
    elif mutation == "advance":
        value = json.loads((root / "summary.json").read_bytes())
        value["advance"] = True
        rewrite("summary.json", value)
        message = "reported ratios or advance gate differ"
    elif mutation == "kernel":
        result = json.loads((root / "00-resident/result.json").read_bytes())
        identity = result["options"]["odometry_execution"]
        fields = [
            "query_sha256=" + "0" * 64 if v.startswith("query_sha256=") else v
            for v in identity.split(";")
        ]
        result["options"]["odometry_execution"] = ";".join(fields)
        rewrite("00-resident/result.json", result)
        measurement = json.loads((root / "00-resident/measurement.json").read_bytes())
        measurement["provenance"]["result_sha256"] = index["files"]["00-resident/result.json"]
        rewrite("00-resident/measurement.json", measurement)
        message = "executed odometry kernels differ"
    else:
        path = root / "candidate.source-delta.tar.gz"
        contents = []
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                stream = archive.extractfile(member)
                assert stream is not None
                contents.append((member, stream.read()))
        with tarfile.open(path, "w:gz") as archive:
            for position, (member, raw) in enumerate(contents):
                if position == 0:
                    raw += b"\n"
                member.size = len(raw)
                archive.addfile(member, io.BytesIO(raw))
        index["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        message = "candidate complete source manifest differs"
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    run = _run(root)
    assert run.returncode != 0 and message in run.stderr
