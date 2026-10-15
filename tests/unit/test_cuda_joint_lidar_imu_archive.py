"""CPU-only integrity and real-work checks for the complete CUDA joint archive."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from tools.audit_bspline_lidar_imu_benchmark import audit

ROOT = Path(__file__).resolve().parents[2] / "docs/assets/cuda_joint_lidar_imu"


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _refresh(root: Path) -> None:
    path = root / "publication-index.json"
    index = json.loads(path.read_bytes())
    index["files"] = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*") if p.is_file() and p != path
    }
    _write(path, index)


def test_complete_cuda_joint_comparison_audits_without_gpu_or_bag() -> None:
    result = audit(ROOT)
    assert result["workers"] == 6
    assert result["quality_passed"]
    # The audit recomputes the recorded ratio; host-dependent speed targets
    # belong in the benchmark report rather than the integrity test.
    assert result["whole_speedup"] > 0


@pytest.mark.parametrize("mutation", ["work", "kernel", "geometry", "cpu_import", "rejection"])
def test_archive_rejects_false_execution_policy_or_rejection(
    tmp_path: Path, mutation: str,
) -> None:
    root = tmp_path / "archive"
    shutil.copytree(ROOT, root)
    if mutation in ("work", "kernel", "geometry"):
        path = root / "00-current/result.json"
        value = json.loads(path.read_bytes())
        report = value["options"]["initial_odometry_execution"][0]
        if mutation == "work":
            report["statistics"]["gpu_query_rows"] = 0
        elif mutation == "kernel":
            identity = report["executed_backends"][0]
            report["executed_backends"][0] = identity.split("system_sha256=")[0] + (
                "system_sha256=" + "0" * 64
            )
        else:
            value["options"]["odometry_options"]["motion_translation_quantum_m"] = 1e-5
        _write(path, value)
        measured = root / "00-current/measurement.json"
        data = json.loads(measured.read_bytes())
        data["provenance"]["result_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        _write(measured, data)
    elif mutation == "cpu_import":
        path = root / "00-prior/measurement.json"
        data = json.loads(path.read_bytes())
        manifest = json.loads((root / "sources-prior-manifest.json").read_bytes())
        name = "calibrex/solvers/cuda_scan_odometry_adapter.py"
        data["provenance"]["executed_sources"][name] = manifest[name]
        _write(path, data)
    else:
        path = root / "rejected-no-motion-grid/comparison.json"
        data = json.loads(path.read_bytes())
        data["passed"] = True
        _write(path, data)
    # Refresh outer hashes to exercise the independent semantic checks.
    _refresh(root)
    with pytest.raises(ValueError):
        audit(root)
