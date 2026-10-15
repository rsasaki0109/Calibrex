"""Protect complete public map observations against plausible hidden changes."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from tools.audit_cuda_map_preparation import audit

ROOT = Path(__file__).resolve().parents[2] / "docs/assets/cuda_map_normal_cache"


@pytest.mark.parametrize("cohort, workers", [("cold", 6), ("retained", 8)])
def test_complete_archive_recomputes_quality_and_every_timing(cohort: str, workers: int) -> None:
    result = audit(ROOT / cohort)
    assert result["workers"] == workers
    assert result["quality_passed"]
    assert result["whole_speedup"] > 0


@pytest.mark.parametrize("mutation", ["timing", "normal_work", "query_work", "normal_identity",
                                     "baseline_identity", "import", "quality", "cache"])
def test_archive_rejects_changes_even_after_rebinding_file_hashes(
    tmp_path: Path, mutation: str,
) -> None:
    root = tmp_path / "archive"
    shutil.copytree(ROOT / "cold", root)
    worker = "00-prior" if mutation == "baseline_identity" else "00-current"
    path = root / worker / "measurement.json"
    row = json.loads(path.read_bytes())
    if mutation == "timing":
        attempts = json.loads((root / "attempts.json").read_bytes())
        attempts[0]["wall_seconds"] *= 1.01
        (root / "attempts.json").write_text(json.dumps(attempts), encoding="utf-8")
    elif mutation == "normal_work":
        row["statistics"]["gpu_normal_rows"] = 0
    elif mutation == "query_work":
        row["statistics"]["gpu_query_rows"] -= 1
    elif mutation in ("normal_identity", "baseline_identity"):
        identity = row["evaluation"]["result"]["execution_backend"]
        field = "query_sha256=" if mutation == "baseline_identity" else "normal_sha256="
        start = identity.index(field) + len(field)
        identity = identity[:start] + "0" * 64 + identity[start + 64:]
        row["evaluation"]["result"]["execution_backend"] = identity
        row["artifact"]["options"]["execution_backend"] = identity
    elif mutation == "import":
        key = "calibrex/solvers/cuda_lidar_map_adapter.py"
        row["provenance"]["executed_sources_sha256"][key] = "0" * 64
    elif mutation == "quality":
        row["evaluation"]["result"]["transform"][0][3] += 1e-5
    else:
        row["cache_initially_empty"] = False
    artifact_path = root / worker / "result.json"
    artifact_path.write_text(json.dumps(row["artifact"]), encoding="utf-8")
    row["result_sha256"] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    path.write_text(json.dumps(row), encoding="utf-8")
    index_path = root / "publication-index.json"
    index = json.loads(index_path.read_bytes())
    index["files"] = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in root.rglob("*") if p.is_file() and p != index_path}
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError):
        audit(root)


def test_audit_needs_neither_gpu_libraries_nor_public_raw_inputs() -> None:
    code = """
import importlib.abc
import sys
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('cupy', 'numba'):
            raise RuntimeError('audit imported optional GPU/JIT code')
sys.meta_path.insert(0, Block())
from tools.audit_cuda_map_preparation import audit
assert audit(Path(sys.argv[1]))['quality_passed']
"""
    subprocess.run([sys.executable, "-c", code, str(ROOT / "cold")], check=True)
