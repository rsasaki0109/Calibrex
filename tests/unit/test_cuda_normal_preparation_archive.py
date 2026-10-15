"""CPU-only work and observation-integrity checks for fused normal preparation."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from tools.audit_bspline_lidar_imu_benchmark import audit

ROOT = Path(__file__).resolve().parents[2] / "docs/assets/cuda_normal_preparation"


def test_fused_normal_archive_recomputes_whole_and_component_ratios() -> None:
    result = audit(ROOT)
    assert result["workers"] == 6
    assert result["quality_passed"]
    assert result["whole_speedup"] > 0
    assert result["normal_speedup"] > 0


@pytest.mark.parametrize("mutation", ["kernel_work", "missing_work", "timing", "missing_timing"])
def test_archive_rejects_false_normal_work_or_timing(tmp_path: Path, mutation: str) -> None:
    root = tmp_path / "archive"
    shutil.copytree(ROOT, root)
    measurement = root / "00-current/measurement.json"
    record = json.loads(measurement.read_bytes())
    if mutation in ("kernel_work", "missing_work"):
        path = root / "00-current/result.json"
        value = json.loads(path.read_bytes())
        counters = value["options"]["initial_odometry_execution"][0]["statistics"]
        if mutation == "kernel_work":
            counters["gpu_normal_kernel_calls"] = 0
        else:
            counters.pop("gpu_normal_kernel_calls")
        path.write_text(json.dumps(value), encoding="utf-8")
        record["provenance"]["result_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif mutation == "timing":
        record["normal_preparation_seconds"] = record["pipeline_seconds"] + 1
    else:
        record.pop("normal_preparation_seconds")
        record.pop("normal_preparation_calls")
    measurement.write_text(json.dumps(record), encoding="utf-8")
    index_path = root / "publication-index.json"
    index = json.loads(index_path.read_bytes())
    index["files"] = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in root.rglob("*") if p.is_file() and p != index_path}
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="normal"):
        audit(root)
