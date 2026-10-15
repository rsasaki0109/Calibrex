"""Source/scope/numerical guards for unsuccessful CUDA odometry experiments."""

from __future__ import annotations

import builtins
import hashlib
import json
import shutil
from pathlib import Path

import pytest
from tools.audit_odometry_candidate_screen import audit

ROOT = Path(__file__).resolve().parents[2] / "docs/assets/odometry_candidate_screen"


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def test_rejected_candidates_audit_all_observations_without_optional_runtimes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = builtins.__import__

    def guarded(name: str, *args: object, **kwargs: object) -> object:
        assert not name.startswith(("cupy", "numba"))
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    result = audit(ROOT)
    assert result["decision"] == "no_candidate_adopted"
    assert result["observations"] == 744 and result["cohorts"] == 5
    assert len(result["comparisons"]) == 11


@pytest.mark.parametrize("mutation", [
    "ratio", "missing_run", "registration", "batched_iteration", "voxel_rows",
    "work", "warm_scope", "source", "fixture", "launcher",
])
def test_rebound_hashes_do_not_hide_changed_observations_sources_or_scope(
    tmp_path: Path, mutation: str,
) -> None:
    root = tmp_path / "archive"
    shutil.copytree(ROOT, root)
    if mutation == "ratio":
        path = root / "radix/summary.json"
        value = json.loads(path.read_bytes())
        value["speedups"]["radix"] *= 2
    elif mutation in ("missing_run", "registration", "work"):
        path = root / "layout/runs.json"
        value = json.loads(path.read_bytes())
        if mutation == "missing_run":
            value.pop()
        else:
            candidate = next(r for r in value if r["variant"] == "network")
            if mutation == "registration":
                candidate["result"]["rmse_m"] += 1e-12
            else:
                candidate["work"]["gpu_query_rows"] += 1
    elif mutation == "batched_iteration":
        path = root / "batching/runs.json"
        value = json.loads(path.read_bytes())
        next(r for r in value if r["variant"] == "8")["result"]["iterations"] += 1
    elif mutation == "voxel_rows":
        path = root / "voxels/runs.json"
        value = json.loads(path.read_bytes())
        next(r for r in value if r["variant"] == "current")["output_rows"] += 1
    elif mutation == "warm_scope":
        path = root / "screen.json"
        value = json.loads(path.read_bytes())
        value["cohorts"][0]["scopes"][0]["first_repeat"] = 0
    elif mutation == "source":
        path = root / "source-manifest.json"
        value = json.loads(path.read_bytes())
        value["calibrex/solvers/scan_to_scan_odometry.py"] = "b" * 64
    elif mutation == "fixture":
        path = root / "fixtures/registration-00.json"
        value = json.loads(path.read_bytes())
        value["fixture_sha256"] = "c" * 64
    else:
        path = root / "ordering/protocol.json"
        value = json.loads(path.read_bytes())
        value["launcher_sha256"] = "d" * 64
    _write(path, value)
    index_path = root / "publication-index.json"
    index = json.loads(index_path.read_bytes())
    index["provenance"]["screen_sha256"] = hashlib.sha256(
        (root / "screen.json").read_bytes(),
    ).hexdigest()
    index["files"] = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*") if p.is_file() and p != index_path
    }
    _write(index_path, index)
    with pytest.raises(ValueError):
        audit(root)
