"""CPU-only checks of complete cold/warm execution and shared-path quality."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from tools.audit_cuda_map_pipeline import audit

ROOT = Path(__file__).resolve().parents[2] / "docs/assets/cuda_map_fusion"


def test_fused_map_archive_recomputes_compilation_and_full_quality() -> None:
    result = audit(ROOT)
    assert result["workers"] == 6 and result["quality_exact"]
    assert result["whole_speedup"] > 1.5 and result["api_speedup"] > 2
    assert result["compiled_programs"] == {"prior": [63] * 3, "current": [17] * 3}
    assert 1 < result["warm_api_speedup"] < 1.2
    assert result["joint_fit_rounds_exact"] == 4
    assert result["joint_signed_controls_exact"] == 32


@pytest.mark.parametrize(
    "mutation",
    [
        "fused_calls",
        "fused_rows",
        "system_rows",
        "compilation",
        "shader",
        "warm_cache",
        "warm_covariance",
        "warm_artifact",
        "joint_covariance",
    ],
)
def test_fused_archive_rejects_rebound_false_work_or_quality(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = tmp_path / "archive"
    shutil.copytree(ROOT, root)
    path = root / "00-current/measurement.json"
    value = json.loads(path.read_bytes())
    if mutation in ("fused_calls", "fused_rows"):
        key = (
            "gpu_map_fused_kernel_calls"
            if mutation == "fused_calls"
            else "gpu_map_fused_residual_rows"
        )
        value["statistics"][key] = 0
    elif mutation == "system_rows":
        value["statistics"]["gpu_map_fused_system_rows"] = (
            value["statistics"]["gpu_map_fused_residual_rows"] + 1
        )
    elif mutation == "compilation":
        value["frozen_provenance"]["nvrtc_compilation"]["seconds"] = value["api_seconds"] + 1
    elif mutation == "shader":
        identity = value["evaluation"]["result"]["execution_backend"]
        value["evaluation"]["result"]["execution_backend"] = identity[:-64] + "0" * 64
    elif mutation == "warm_cache":
        path = root / "warm/current/measurement.json"
        value = json.loads(path.read_bytes())
        value["rows"][1]["nvrtc_calls"] = 1
    elif mutation == "warm_covariance":
        path = root / "warm/current/01/measurement.json"
        value = json.loads(path.read_bytes())
        value["evaluation"]["result"]["covariance"][0][0] += 1e-5
    elif mutation == "warm_artifact":
        path = root / "warm/current/01/measurement.json"
        value = json.loads(path.read_bytes())
        value["artifact"]["holdout_median_residual_m"] += 1e-5
    else:
        path = root / "joint-validation/current/measurement.json"
        value = json.loads(path.read_bytes())
        value["rounds"][0]["calibration_covariance"][0][0] += 1e-5
    path.write_text(json.dumps(value), encoding="utf-8")
    path = root / "publication-index.json"
    index = json.loads(path.read_bytes())
    index["files"] = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file() and p != path
    }
    path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match=r"accounting|compilation|shader|warm|shared-path"):
        audit(root)
