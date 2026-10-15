"""The rejected source comparison stays reproducible without CUDA or raw data."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from tools.audit_bspline_lidar_imu_benchmark import audit

ROOT = Path(__file__).resolve().parents[2] / "docs/assets/cpu_worker_reuse_rejected"


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def test_rejected_cpu_source_comparison_audits_without_cuda_or_bag() -> None:
    report = audit(ROOT)
    protocol = json.loads((ROOT / "protocol.json").read_bytes())
    assert protocol["comparison"] == "cpu-source"
    assert protocol["baseline_commit"] != protocol["current_commit"]
    assert protocol["cpu_query_workers"] == {"prior": 8, "current": 8}
    assert report["workers"] == 2 and report["quality_passed"]
    assert report["whole_speedup"] < 1.


@pytest.mark.parametrize("mutation", [
    "source_commit", "worker_budget", "initial_estimate", "fit_covariance", "ratio",
])
def test_rebound_hashes_cannot_hide_false_execution_quality_or_speed(
    tmp_path: Path, mutation: str,
) -> None:
    root = tmp_path / "archive"
    shutil.copytree(ROOT, root)
    result_path = root / "00-current/result.json"
    record_path = root / "00-current/measurement.json"
    result = json.loads(result_path.read_bytes())
    record = json.loads(record_path.read_bytes())
    if mutation == "source_commit":
        result["provenance"]["git_commit"] = "0" * 40
    elif mutation == "worker_budget":
        result["options"]["odometry_options"]["query_workers"] = 4
    elif mutation == "initial_estimate":
        result["initial"]["rotation_vector_rad"][0] += 1e-12
    elif mutation == "fit_covariance":
        record["rounds"][0]["calibration_covariance"][0][0] += 1e-15
    else:
        path = root / "summary.json"
        summary = json.loads(path.read_bytes())
        summary["whole_speedup"] *= 1.5
        _write(path, summary)
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
