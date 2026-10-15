"""CPU-only exactness and actual-work guards for conservative spatial pruning."""

from __future__ import annotations

import builtins
import hashlib
import json
import shutil
from pathlib import Path

import pytest
from tools.audit_bspline_lidar_imu_benchmark import audit

ROOT = Path(__file__).resolve().parents[2] / "docs/assets/cuda_interval_pruning"


def test_interval_pruning_comparison_audits_without_an_accelerator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = builtins.__import__

    def guarded(name: str, *args: object, **kwargs: object) -> object:
        assert not name.startswith(("cupy", "numba"))
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    protocol = json.loads((ROOT / "protocol.json").read_bytes())
    assert protocol["exact_spatial_outputs_required"]
    result = audit(ROOT)
    assert result["workers"] == 4 and result["quality_passed"]
    assert result["earlier_guard"]["workers"] == 4
    assert result["earlier_guard"]["whole_speedup"] < 1
    assert result["whole_speedup"] > 0


@pytest.mark.parametrize("mutation", ["calls", "missing", "work", "fit", "ratio", "earlier_ratio"])
def test_rebound_interval_archive_rejects_work_and_sub_tolerance_fit_changes(
    tmp_path: Path, mutation: str,
) -> None:
    root = tmp_path / "archive"
    shutil.copytree(ROOT, root)
    result_path = root / "00-current/result.json"
    record_path = root / "00-current/measurement.json"
    result = json.loads(result_path.read_bytes())
    record = json.loads(record_path.read_bytes())
    counters = result["options"]["segment_odometry_execution"][0]["statistics"]
    if mutation == "calls":
        counters["gpu_interval_system_calls"] = 0
    elif mutation == "missing":
        counters.pop("gpu_interval_system_calls")
    elif mutation == "work":
        counters["gpu_normal_rows"] += 1
    elif mutation == "fit":
        record["rounds"][0]["calibration_covariance"][0][0] += 1e-15
    elif mutation == "ratio":
        path = root / "summary.json"
        summary = json.loads(path.read_bytes())
        summary["whole_speedup"] *= 2
        path.write_text(json.dumps(summary), encoding="utf-8")
    else:
        earlier = root / "earlier-guard-v1"
        path = earlier / "summary.json"
        summary = json.loads(path.read_bytes())
        summary["whole_speedup"] *= 2
        path.write_text(json.dumps(summary), encoding="utf-8")
        path = earlier / "publication-index.json"
        index = json.loads(path.read_bytes())
        index["files"] = {
            p.relative_to(earlier).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in earlier.rglob("*") if p.is_file() and p != path
        }
        path.write_text(json.dumps(index), encoding="utf-8")
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
