"""CPU-only integrity checks reject altered worker budgets and physical fields."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from tools.audit_bspline_lidar_imu_benchmark import audit

ROOT = Path(__file__).resolve().parents[2] / "docs/assets/cpu_parallel_scan_queries"


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def test_complete_cpu_thread_comparison_audits_without_cuda_or_bag() -> None:
    result = audit(ROOT)
    assert result["workers"] == 2
    assert result["quality_passed"]


@pytest.mark.parametrize("mutation", [
    "budget", "initial_backend", "metadata", "initial_rotation", "fit",
])
def test_rebound_hashes_cannot_hide_worker_or_numerical_changes(
    tmp_path: Path, mutation: str,
) -> None:
    root = tmp_path / "archive"
    shutil.copytree(ROOT, root)
    result_path = root / "00-current/result.json"
    record_path = root / "00-current/measurement.json"
    result = json.loads(result_path.read_bytes())
    record = json.loads(record_path.read_bytes())
    if mutation == "budget":
        result["options"]["odometry_options"]["query_workers"] = 4
    elif mutation == "initial_backend":
        result["options"]["initial_odometry_execution"][0]["requested_backend"] = "cuda"
    elif mutation == "metadata":
        record["provenance"]["cpu_query_workers"] = 4
    elif mutation == "initial_rotation":
        result["initial"]["rotation_vector_rad"][0] += 1e-12
    else:
        record["rounds"][0]["calibration_covariance"][0][0] += 1e-15
    _write(result_path, result)
    record["provenance"]["result_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    _write(record_path, record)
    path = root / "publication-index.json"
    index = json.loads(path.read_bytes())
    index["files"] = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*") if p.is_file() and p != path
    }
    _write(path, index)
    with pytest.raises(ValueError):
        audit(root)
