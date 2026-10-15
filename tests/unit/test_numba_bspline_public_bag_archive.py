"""Complete decoded-bag evidence must retain its numerical and timing limits."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from tools.audit_bspline_lidar_imu_benchmark import audit

ROOT = Path(__file__).resolve().parents[2] / "docs/assets/numba_bspline_public_bag"


def test_both_full_bag_cohorts_are_auditable_without_recording_or_gpu() -> None:
    for root in (ROOT, ROOT / "variable-load-v1"):
        result = audit(root)
        assert result["workers"] == 6
        assert result["quality_passed"]
    # A larger ratio in the repeat cannot erase the earlier negative result.
    assert audit(ROOT / "variable-load-v1")["whole_speedup"] < 1


@pytest.mark.parametrize("mutation", ["backend", "cache", "fit", "cpu_sum"])
def test_bound_hashes_cannot_hide_runtime_or_fit_evidence_drift(
    tmp_path: Path, mutation: str,
) -> None:
    root = tmp_path / "archive"
    shutil.copytree(ROOT, root)
    path = root / "00-current/measurement.json"
    value = json.loads(path.read_bytes())
    if mutation == "backend":
        value["rounds"][0]["execution_backend"] = "numpy"
    elif mutation == "cache":
        value["provenance"]["numba_cache_initially_empty"] = False
    elif mutation == "fit":
        value["rounds"][0]["calibration"]["gyro_bias"][0] += .01
    else:
        value["fit_cpu_seconds"] += 1
    path.write_text(json.dumps(value), encoding="utf-8")
    index_path = root / "publication-index.json"
    index = json.loads(index_path.read_bytes())
    index["files"]["00-current/measurement.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError):
        audit(root)
