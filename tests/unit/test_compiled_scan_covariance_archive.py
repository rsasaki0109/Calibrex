"""CPU-only evidence audit binds compiled work before checking every physical field."""

from __future__ import annotations

import builtins
import hashlib
import json
import shutil
from pathlib import Path

import pytest
from tools.audit_bspline_lidar_imu_benchmark import audit

ROOT = Path(__file__).resolve().parents[2] / "docs/assets/cpu_fused_scan_covariance"


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def test_complete_cpu_covariance_comparison_audits_without_jit_gpu_or_bag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = builtins.__import__

    def guarded(name: str, *args: object, **kwargs: object) -> object:
        if name in ("numba", "cupy") or name.startswith(("numba.", "cupy.")):
            raise AssertionError("archive audit imported an optional accelerator")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    result = audit(ROOT)
    assert result["workers"] == 4
    assert result["quality_passed"]
    assert result["whole_speedup"] < 1


@pytest.mark.parametrize("mutation", [
    "backend", "identity", "zero_work", "counter_type", "warm_cache",
    "initial_rotation", "fit_covariance", "ratio", "validation_revision", "validation_log",
])
def test_rebound_hashes_cannot_hide_execution_or_physical_changes(
    tmp_path: Path, mutation: str,
) -> None:
    root = tmp_path / "archive"
    shutil.copytree(ROOT, root)
    result_path = root / "00-current/result.json"
    record_path = root / "00-current/measurement.json"
    summary_path = root / "summary.json"
    result = json.loads(result_path.read_bytes())
    record = json.loads(record_path.read_bytes())
    summary = json.loads(summary_path.read_bytes())
    report = result["options"]["initial_odometry_execution"][0]
    if mutation == "backend":
        result["options"]["odometry_options"]["normal_backend"] = "numpy"
    elif mutation == "identity":
        report["executed_backends"][0] = report["executed_backends"][0].replace(
            "fastmath=false", "fastmath=true",
        )
    elif mutation == "zero_work":
        report["statistics"]["numba_normal_covariance_rows"] = 0
    elif mutation == "counter_type":
        report["statistics"]["numba_normal_covariance_calls"] = True
    elif mutation == "warm_cache":
        record["provenance"]["numba_cache_initially_empty"] = False
    elif mutation == "initial_rotation":
        result["initial"]["rotation_vector_rad"][0] += 1e-12
    elif mutation == "fit_covariance":
        record["rounds"][0]["calibration_covariance"][0][0] += 1e-15
    elif mutation == "ratio":
        summary["whole_speedup"] *= 2
    elif mutation == "validation_revision":
        path = root / "prototype-validation/receipt.json"
        receipt = json.loads(path.read_bytes())
        receipt["provenance"]["source_commit"] = "0" * 40
        _write(path, receipt)
    else:
        path = root / "prototype-validation/pytest.log"
        path.write_text("invented validation output\n", encoding="utf-8")
    _write(result_path, result)
    record["provenance"]["result_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    _write(record_path, record)
    _write(summary_path, summary)
    path = root / "publication-index.json"
    index = json.loads(path.read_bytes())
    index["files"] = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*") if p.is_file() and p != path
    }
    _write(path, index)
    with pytest.raises(ValueError):
        audit(root)
