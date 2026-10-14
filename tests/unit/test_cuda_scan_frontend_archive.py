"""CPU-only accounting, timing and publication checks for the complete prototype."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from tools.audit_bspline_lidar_imu_benchmark import audit

ROOT = Path(__file__).resolve().parents[2] / "docs/assets/cuda_scan_frontend"


def test_complete_gpu_frontend_archive_recomputes_every_scope() -> None:
    result = audit(ROOT)
    assert result["workers"] == 6 and result["quality_passed"]
    for name in ("whole_speedup", "spatial_build_speedup", "preprocessing_speedup"):
        assert result[name] > 0


@pytest.mark.parametrize("mutation", ["tree", "tree_kernel", "native_tree", "preprocess",
                                      "preprocess_kernel", "missing_preprocess",
                                      "spatial_timing", "missing_preprocessing_timing"])
def test_archive_rejects_rebound_false_gpu_frontend_evidence(
    tmp_path: Path, mutation: str,
) -> None:
    root = tmp_path / "archive"
    shutil.copytree(ROOT, root)
    measurement = root / "00-current/measurement.json"
    record = json.loads(measurement.read_bytes())
    if mutation.endswith("timing"):
        if mutation == "spatial_timing":
            record["spatial_build_seconds"] = record["pipeline_seconds"] + 1
        else:
            record.pop("preprocessing_seconds")
            record.pop("preprocessing_calls")
    else:
        result = root / "00-current/result.json"
        value = json.loads(result.read_bytes())
        counters = value["options"]["initial_odometry_execution"][0]["statistics"]
        if mutation == "missing_preprocess":
            counters.pop("gpu_preprocess_output_rows")
        elif mutation == "native_tree":
            counters["native_tree_builds"] = counters["gpu_tree_builds"] + 1
        else:
            key = {"tree": "gpu_tree_builds", "tree_kernel": "gpu_tree_build_kernel_calls",
                   "preprocess": "gpu_preprocess_calls",
                   "preprocess_kernel": "gpu_preprocess_kernel_calls"}[mutation]
            counters[key] = 0
        result.write_text(json.dumps(value), encoding="utf-8")
        record["provenance"]["result_sha256"] = hashlib.sha256(result.read_bytes()).hexdigest()
    measurement.write_text(json.dumps(record), encoding="utf-8")
    path = root / "publication-index.json"
    index = json.loads(path.read_bytes())
    index["files"] = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in root.rglob("*") if p.is_file() and p != path}
    path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match=r"construction|preprocessing|spatial_build"):
        audit(root)
