"""Audit full-fit publication independently of external sensor inputs or Numba."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from tools.audit_bspline_planes import audit

ARCHIVE = Path(__file__).resolve().parents[2] / "docs/assets/numba_bspline_planes"


def test_complete_fixed_input_fit_archive() -> None:
    result = audit(ARCHIVE)
    assert result["verified_pairs"] == 3
    assert result["verified_full_fit_comparisons"] == 6
    assert result["source_files"] == 377


@pytest.mark.parametrize("mutation", ["iteration", "speedup", "imports", "cohort"])
def test_rebound_hashes_cannot_hide_inconsistent_evidence(tmp_path: Path, mutation: str) -> None:
    root = tmp_path / "archive"
    shutil.copytree(ARCHIVE, root)
    relative = {
        "iteration": "00-numba_cpu/measurement.json",
        "speedup": "summary.json",
        "imports": "00-numba_cpu/measurement.json",
        "cohort": "attempts.json",
    }[mutation]
    record = json.loads((root / relative).read_bytes())
    if mutation == "iteration":
        record["runs"][1]["results"][0]["iterations"] += 1
    elif mutation == "speedup":
        record["speedups"]["warm"] += 1
    elif mutation == "imports":
        imports = record["provenance"]["executed_sources"]
        imports[next(iter(imports))] = "0" * 64
    else:
        record["attempts"].pop()
    path = root / relative
    path.write_text(json.dumps(record), encoding="utf-8")
    index = json.loads((root / "index.json").read_bytes())
    index["files"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "index.json").write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError):
        audit(root)
