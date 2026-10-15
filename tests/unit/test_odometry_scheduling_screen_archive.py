"""CPU-only integrity and numerical guards for rejected scheduling candidates."""

from __future__ import annotations

import builtins
import hashlib
import json
import shutil
from pathlib import Path

import pytest
from tools.audit_odometry_scheduling_screen import audit

ROOT = Path(__file__).resolve().parents[2] / "docs/assets/odometry_scheduling_screen"


def test_scheduling_screen_audits_every_observation_without_optional_runtimes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = builtins.__import__

    def guarded(name: str, *args: object, **kwargs: object) -> object:
        assert not name.startswith(("cupy", "numba"))
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    result = audit(ROOT)
    assert result["decision"] == "no_candidate_adopted"
    assert result["observations"] == 492
    assert len(result["comparisons"]) == 8


@pytest.mark.parametrize("mutation", ["physical", "work", "missing", "output", "ratio", "scope"])
def test_rebound_scheduling_archive_rejects_changed_results_work_sampling_and_scope(
    tmp_path: Path, mutation: str,
) -> None:
    root = tmp_path / "archive"
    shutil.copytree(ROOT, root)
    if mutation in ("physical", "work", "missing"):
        path = root / "traversal/runs.json"
        value = json.loads(path.read_bytes())
        record = next(r for r in value if r["variant"] == "heap")
        if mutation == "missing":
            value.pop()
        elif mutation == "work":
            record["work"]["gpu_query_rows"] += 1
        else:
            record["result"]["rmse_m"] += 1e-12
    elif mutation == "output":
        path = root / "placement/runs.json"
        value = json.loads(path.read_bytes())
        next(r for r in value if r["variant"] == "edges")["output_sha256"] = "a" * 64
    else:
        path = root / "screen.json"
        value = json.loads(path.read_bytes())
        if mutation == "ratio":
            value["comparisons"][0]["speedup"] *= 2
        else:
            value["cohorts"][0]["scope"] = "placement"
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    index_path = root / "publication-index.json"
    index = json.loads(index_path.read_bytes())
    index["provenance"]["screen_sha256"] = hashlib.sha256(
        (root / "screen.json").read_bytes(),
    ).hexdigest()
    index["files"] = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in root.rglob("*") if p.is_file() and p != index_path}
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        audit(root)
