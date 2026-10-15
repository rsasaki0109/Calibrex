"""CPU-only exactness and actual-work guards for asynchronous normal evidence."""

from __future__ import annotations

import builtins
import hashlib
import json
import shutil
from pathlib import Path

import pytest
from tools.audit_bspline_lidar_imu_benchmark import audit

ROOT = Path(__file__).resolve().parents[2] / "docs/assets/cuda_normal_overlap"


def test_async_normal_comparison_audits_without_an_accelerator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = builtins.__import__

    def guarded(name: str, *args: object, **kwargs: object) -> object:
        assert not name.startswith(("cupy", "numba"))
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    protocol = json.loads((ROOT / "protocol.json").read_bytes())
    assert protocol["exact_preparation_outputs_required"]
    result = audit(ROOT)
    assert result["workers"] == 4 and result["quality_passed"]
    assert result["whole_speedup"] > 0


@pytest.mark.parametrize("mutation", ["batches", "missing", "work", "fit", "ratio"])
def test_rebound_async_archive_rejects_work_and_sub_tolerance_fit_changes(
    tmp_path: Path, mutation: str,
) -> None:
    root = tmp_path / "archive"
    shutil.copytree(ROOT, root)
    result_path = root / "00-current/result.json"
    record_path = root / "00-current/measurement.json"
    result = json.loads(result_path.read_bytes())
    record = json.loads(record_path.read_bytes())
    counters = result["options"]["segment_odometry_execution"][0]["statistics"]
    if mutation == "batches":
        counters["gpu_async_normal_batches"] = 0
    elif mutation == "missing":
        counters.pop("gpu_async_normal_batches")
    elif mutation == "work":
        counters["gpu_normal_rows"] += 1
    elif mutation == "fit":
        record["rounds"][0]["calibration_covariance"][0][0] += 1e-15
    else:
        path = root / "summary.json"
        summary = json.loads(path.read_bytes())
        summary["whole_speedup"] *= 2
        path.write_text(json.dumps(summary), encoding="utf-8")
    result_path.write_text(json.dumps(result), encoding="utf-8")
    record["provenance"]["result_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    record_path.write_text(json.dumps(record), encoding="utf-8")
    path = root / "publication-index.json"
    index = json.loads(path.read_bytes())
    index["files"] = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in root.rglob("*") if p.is_file() and p != path}
    path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError):
        audit(root)
