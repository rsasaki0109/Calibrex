"""CPU-only complete-work, quality and initialization-evidence checks."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from tools.audit_cuda_fixed_neighbors import audit

ROOT = Path(__file__).resolve().parents[2] / "docs/assets/cuda_fixed_neighbors"


def test_rejected_fixed_query_archive_retains_complete_workers() -> None:
    result = audit(ROOT)
    assert result["workers"] == 6 and result["quality_exact"]
    assert result["whole_speedup"] < 1 and result["api_speedup"] < 1
    assert result["observations"] == 17
    assert result["foreign_samples"] == result["unknown_samples"] == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "query",
        "fixed",
        "cache",
        "threads",
        "imports",
        "covariance",
        "jackknife",
        "ratio",
        "profile_calls",
        "observer",
        "identity",
    ],
)
def test_archive_rejects_rebound_false_work_quality_or_profile(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = tmp_path / "archive"
    shutil.copytree(ROOT, root)
    path = root / "00-current/measurement.json"
    value = json.loads(path.read_bytes())
    if mutation == "query":
        value["statistics"]["gpu_query_rows"] = 1
    elif mutation == "fixed":
        value["statistics"]["gpu_fixed_query_calls"] = 0
    elif mutation == "cache":
        value["frozen_provenance"]["cuda_cache_initially_empty"] = False
    elif mutation == "threads":
        value["frozen_provenance"]["threads"]["OMP_NUM_THREADS"] = "8"
    elif mutation == "imports":
        value["frozen_provenance"]["executed_sources"].pop(
            "calibrex/solvers/cuda_lidar_map_adapter.py",
        )
    elif mutation == "covariance":
        value["evaluation"]["result"]["covariance"][0][0] += 1e-5
    elif mutation == "jackknife":
        value["evaluation"]["jackknife_fits"] -= 1
    elif mutation == "identity":
        value["evaluation"]["result"]["execution_backend"] = value["evaluation"]["result"][
            "execution_backend"
        ].replace("fp64=true", "fp64=false")
    elif mutation == "ratio":
        path = root / "summary.json"
        value = json.loads(path.read_bytes())
        value["whole_speedup"] = 3.7
    elif mutation == "profile_calls":
        path = root / "cold-profile/summary.json"
        value = json.loads(path.read_bytes())
        value["compiled_programs"] -= 1
    else:
        path = root / "observation.json"
        value = json.loads(path.read_bytes())
        value["samples"][0]["foreign"] = 1
    path.write_text(json.dumps(value), encoding="utf-8")
    path = root / "publication-index.json"
    index = json.loads(path.read_bytes())
    index["files"] = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file() and p != path
    }
    path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match=r"work|accounting|policy|imports|evaluation|ratio|compilation|interference|shader",
    ):
        audit(root)
