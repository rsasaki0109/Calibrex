"""The published complete acceleration evidence must remain auditable on CPU."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from tools.audit_bspline_lidar_imu_benchmark import audit

ROOT = Path(__file__).resolve().parents[2] / "docs/assets/bspline_lidar_imu_acceleration"


def _rewrite(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _refresh_index(root: Path) -> None:
    path = root / "publication-index.json"
    index = json.loads(path.read_bytes())
    index["files"] = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file() and p != path
    }
    _rewrite(path, index)


def test_complete_public_comparison_audits_without_bag_or_cuda() -> None:
    result = audit(ROOT)
    assert result["workers"] == 6
    assert result["quality_passed"]
    assert result["whole_speedup"] > 2.0
    assert result["fit_speedup"] > 4.0


def test_audit_allows_only_tiny_quaternion_recomputation_roundoff(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    shutil.copytree(ROOT, root)
    path = root / "summary.json"
    value = json.loads(path.read_bytes())
    value["quality"][0]["errors"]["rotation_deg"] += 1e-15
    _rewrite(path, value)
    _refresh_index(root)
    assert audit(root)["quality_passed"]


@pytest.mark.parametrize("mutation", ["result", "missing_worker", "import", "ratio", "covariance"])
def test_audit_rejects_changed_result_worker_source_ratio_or_covariance(
    tmp_path: Path, mutation: str
) -> None:
    root = tmp_path / "archive"
    shutil.copytree(ROOT, root)
    if mutation == "result":
        path = root / "00-current/result.json"
        value = json.loads(path.read_bytes())
        value["imu_clock_offset_s"] += 0.01
        _rewrite(path, value)
    elif mutation == "missing_worker":
        path = root / "attempts.json"
        _rewrite(path, json.loads(path.read_bytes())[:-1])
    elif mutation == "import":
        path = root / "00-current/measurement.json"
        value = json.loads(path.read_bytes())
        value["provenance"]["executed_sources"]["calibrex/core/continuous_time_bspline_fit.py"] = (
            "0" * 64
        )
        _rewrite(path, value)
    elif mutation == "ratio":
        path = root / "summary.json"
        value = json.loads(path.read_bytes())
        value["whole_speedup"] *= 2
        _rewrite(path, value)
    else:
        path = root / "00-current/measurement.json"
        value = json.loads(path.read_bytes())
        value["rounds"][0]["calibration_covariance"][0][0] += 1
        _rewrite(path, value)
    # Bypass the outer file hash to exercise the independent semantic guards.
    _refresh_index(root)
    with pytest.raises(ValueError):
        audit(root)
