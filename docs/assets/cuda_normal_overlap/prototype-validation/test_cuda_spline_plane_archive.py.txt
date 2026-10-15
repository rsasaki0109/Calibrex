"""CPU-only source, execution and complete-fit guards for CUDA spline evidence."""

from __future__ import annotations

import builtins
import hashlib
import json
import shutil
from pathlib import Path

import pytest
from tools.audit_bspline_lidar_imu_benchmark import audit

ROOT = Path(__file__).resolve().parents[2] / "docs/assets/cuda_spline_planes"


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def test_full_cuda_plane_comparison_audits_without_an_accelerator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = builtins.__import__

    def guarded(name: str, *args: object, **kwargs: object) -> object:
        assert not name.startswith(("cupy", "numba"))
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    protocol = json.loads((ROOT / "protocol.json").read_bytes())
    assert protocol["fit_plane_backends"] == {"prior": "numpy", "current": "cuda"}
    result = audit(ROOT)
    assert result["workers"] == 4 and result["quality_passed"]
    assert result["fit_speedup"] > 1


@pytest.mark.parametrize("mutation", [
    "identity", "work", "row_count", "normal_calls", "calibration", "iterations", "ratio",
    "initial", "odometry_work",
])
def test_rebound_archive_rejects_changed_execution_work_fits_and_ratios(
    tmp_path: Path, mutation: str,
) -> None:
    root = tmp_path / "archive"
    shutil.copytree(ROOT, root)
    result_path = root / "00-current/result.json"
    record_path = root / "00-current/measurement.json"
    result = json.loads(result_path.read_bytes())
    record = json.loads(record_path.read_bytes())
    fitted = record["rounds"][0]
    report = result["options"]["fit_execution"][0]
    if mutation == "identity":
        fitted["execution_backend"] = fitted["execution_backend"].replace("sha256=", "sha256=b")
        report["executed_backend"] = fitted["execution_backend"]
    elif mutation in ("work", "normal_calls"):
        key = "gpu_plane_rows" if mutation == "work" else "gpu_plane_normal_calls"
        fitted["execution_statistics"][key] += 1
        report["statistics"] = fitted["execution_statistics"].copy()
    elif mutation == "row_count":
        result["rounds"][0]["plane_returns"] += 1
    elif mutation == "calibration":
        fitted["calibration"]["imu_clock_offset"][0] += 1e-5
    elif mutation == "iterations":
        fitted["iterations"] += 1
        result["rounds"][0]["iterations"] += 1
        fitted["execution_statistics"]["gpu_plane_normal_calls"] += 1
        report["statistics"] = fitted["execution_statistics"].copy()
    elif mutation == "initial":
        result["initial"]["rotation_vector_rad"][0] += 1e-12
    elif mutation == "odometry_work":
        result["options"]["initial_odometry_execution"][0]["statistics"]["gpu_query_calls"] += 1
    else:
        path = root / "summary.json"
        summary = json.loads(path.read_bytes())
        summary["fit_speedup"] *= 2
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
