"""CPU-only checks for the fused map registration's complete public evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from tools.audit_cuda_map_preparation import audit

ROOT = Path(__file__).resolve().parents[2] / "docs/assets/cuda_map_registration_fusion"


@pytest.mark.parametrize("cohort, workers", [("cold", 6), ("retained", 8)])
def test_fused_registration_archive_recomputes_full_quality_and_timings(
    cohort: str, workers: int,
) -> None:
    summary = audit(ROOT / cohort)
    assert summary["workers"] == workers
    assert summary["quality_passed"]


@pytest.mark.parametrize("mutation", ["identity", "projection_rows", "residual_rows",
                                     "jacobian_calls", "jacobian_rows", "unchanged_work"])
def test_fused_archive_rejects_changed_code_or_work_after_hash_rebinding(
    tmp_path: Path, mutation: str,
) -> None:
    root = tmp_path / "archive"
    shutil.copytree(ROOT / "cold", root)
    path = root / "00-current/measurement.json"
    row = json.loads(path.read_bytes())
    if mutation == "identity":
        identity = row["evaluation"]["result"]["execution_backend"]
        start = identity.index("registration_sha256=") + len("registration_sha256=")
        identity = identity[:start] + "0" * 64
        row["evaluation"]["result"]["execution_backend"] = identity
        row["artifact"]["options"]["execution_backend"] = identity
    elif mutation == "projection_rows":
        row["statistics"]["gpu_projection_query_rows"] += 1
    elif mutation == "residual_rows":
        row["statistics"]["gpu_residual_kernel_rows"] = (
            row["statistics"]["gpu_projection_query_rows"] + 1
        )
    elif mutation in ("jacobian_calls", "jacobian_rows"):
        row["statistics"]["gpu_" + mutation.replace("jacobian_", "jacobian_kernel_")] = 0
    else:
        row["statistics"]["scalar_query_repairs"] += 1
    artifact_path = root / "00-current/result.json"
    artifact_path.write_text(json.dumps(row["artifact"]), encoding="utf-8")
    row["result_sha256"] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    path.write_text(json.dumps(row), encoding="utf-8")
    index_path = root / "publication-index.json"
    index = json.loads(index_path.read_bytes())
    index["files"] = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in root.rglob("*") if p.is_file() and p != index_path}
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match=r"registration|fused|fusion"):
        audit(root)
