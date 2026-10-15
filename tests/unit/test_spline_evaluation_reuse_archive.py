"""The source-bound comparison requires every physical and fitted field to be exact."""

from __future__ import annotations

import builtins
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from tools.audit_bspline_lidar_imu_benchmark import audit

ROOT = Path(__file__).resolve().parents[2] / "docs/assets/spline_evaluation_reuse"


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def test_complete_reuse_comparison_is_exact_and_audits_without_accelerators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = builtins.__import__

    def guarded(name: str, *args: object, **kwargs: object) -> object:
        if name in ("cupy", "numba") or name.startswith(("cupy.", "numba.")):
            raise AssertionError("audit imported an optional accelerator")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert json.loads((ROOT / "protocol.json").read_bytes())["exact_outputs_required"] is True
    result = audit(ROOT)
    assert result["workers"] == 4 and result["quality_passed"]
    assert result["fit_speedup"] > 1


@pytest.mark.parametrize("mutation", [
    "initial", "noise", "execution_work", "fit_covariance", "fit_cost", "ratio",
])
def test_rebound_hashes_cannot_hide_sub_tolerance_changes(
    tmp_path: Path, mutation: str,
) -> None:
    root = tmp_path / "archive"
    shutil.copytree(ROOT, root)
    result_path, record_path = root / "00-current/result.json", root / "00-current/measurement.json"
    result = json.loads(result_path.read_bytes())
    record = json.loads(record_path.read_bytes())
    if mutation == "initial":
        result["initial"]["rotation_vector_rad"][0] += 1e-12
    elif mutation == "noise":
        result["noise_sigmas"]["gyro"] += 1e-12
    elif mutation == "execution_work":
        result["options"]["initial_odometry_execution"][0]["statistics"]["gpu_query_calls"] += 1
    elif mutation == "fit_covariance":
        record["rounds"][0]["calibration_covariance"][0][0] += 1e-15
    elif mutation == "fit_cost":
        value = record["rounds"][0]["final_cost"]
        record["rounds"][0]["final_cost"] = float(np.nextafter(value, np.inf))
    else:
        path = root / "summary.json"
        summary = json.loads(path.read_bytes())
        summary["whole_speedup"] *= 2
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
