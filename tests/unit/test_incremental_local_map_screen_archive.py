"""Integrity and full-result guards for the experimental local-map screen."""

from __future__ import annotations

import builtins
import hashlib
import json
import shutil
from pathlib import Path

import pytest
from tools.audit_incremental_local_map_screen import audit

ROOT = Path(__file__).resolve().parents[2] / "docs/assets/incremental_local_map_screen"


def test_local_map_screen_is_auditable_without_optional_runtimes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = builtins.__import__

    def guarded(name: str, *args: object, **kwargs: object) -> object:
        assert not name.startswith(("cupy", "numba"))
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    report = audit(ROOT)
    assert report["observations"] == 2478
    assert report["fresh_workers"] == 4
    assert report["exact_full_results_and_fits"]


@pytest.mark.parametrize("mutation", ["output", "missing", "cost", "work", "ratio"])
def test_rebound_local_map_archive_rejects_changed_outputs_fits_work_and_timing(
    tmp_path: Path, mutation: str,
) -> None:
    root = tmp_path / "archive"
    shutil.copytree(ROOT, root)
    if mutation in ("output", "missing"):
        path = root / "gpu/runs.json"
        value = json.loads(path.read_bytes())
        if mutation == "missing":
            value.pop()
        else:
            next(r for r in value if r["variant"] == "hash_cached")["output_sha256"] = "a" * 64
    elif mutation == "cost":
        path = root / "whole/00-current/measurement.json"
        value = json.loads(path.read_bytes())
        value["rounds"][0]["final_cost"] += 1e-9
    elif mutation == "work":
        path = root / "whole/00-current/candidate-work.json"
        value = json.loads(path.read_bytes())
        value["work"]["input_rows"] += 1
    else:
        path = root / "whole/summary.json"
        value = json.loads(path.read_bytes())
        value["whole_speedup"] *= 2
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    index_path = root / "publication-index.json"
    index = json.loads(index_path.read_bytes())
    index["files"] = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in root.rglob("*") if p.is_file() and p != index_path}
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        audit(root)
