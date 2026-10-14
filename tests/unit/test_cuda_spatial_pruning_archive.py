"""CPU-only validation of the complete spatial-pruning measurement archive."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from tools.audit_bspline_lidar_imu_benchmark import audit

ROOT = Path(__file__).resolve().parents[2] / "docs/assets/cuda_spatial_pruning"


def test_complete_spatial_pruning_archive_audits_without_gpu_or_bag() -> None:
    result = audit(ROOT)
    assert result["workers"] == 6
    assert result["quality_passed"]
    assert result["whole_speedup"] > 0


def test_archive_rejects_missing_actual_tree_capture_accounting(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    shutil.copytree(ROOT, root)
    path = root / "00-current/result.json"
    result = json.loads(path.read_bytes())
    counters = result["options"]["initial_odometry_execution"][0]["statistics"]
    counters["serialized_tree_captures"] = counters["public_node_tree_captures"] = 0
    path.write_text(json.dumps(result), encoding="utf-8")
    measurement = root / "00-current/measurement.json"
    record = json.loads(measurement.read_bytes())
    record["provenance"]["result_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    measurement.write_text(json.dumps(record), encoding="utf-8")
    index_path = root / "publication-index.json"
    index = json.loads(index_path.read_bytes())
    index["files"] = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in root.rglob("*") if p.is_file() and p != index_path}
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="actual tree capture path"):
        audit(root)
