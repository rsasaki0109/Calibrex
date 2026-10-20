"""Replay both complete cohorts, including the missed initial adoption gate."""

from __future__ import annotations

import ast
import hashlib
import json
import tarfile
from pathlib import Path

import jsonschema
import pytest
from tools import audit_screw_residual_archive as auditor

ASSETS = Path(__file__).resolve().parents[2] / "docs/assets/screw_interpolation_reuse"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _members(name: str) -> dict[str, bytes]:
    with tarfile.open(ASSETS / name) as archive:
        return {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}


@pytest.mark.parametrize("first", [False, True])
def test_complete_cohort_replays_all_bindings_outcomes_and_gate_decisions(
    first: bool,
    tmp_path: Path,
) -> None:
    inventory = json.loads(
        (ASSETS / ("first-attempt-inventory.json" if first else "inventory.json")).read_bytes()
    )
    name = inventory["archive"]["name"]
    raw = (ASSETS / name).read_bytes()
    assert len(raw) == inventory["archive"]["bytes"]
    assert _sha(raw) == inventory["archive"]["sha256"]
    files = _members(name)
    assert _sha(files["index.json"]) == inventory["index_sha256"]
    assert _sha(files["summary.json"]) == inventory["summary_sha256"]
    index = json.loads(files["index.json"])
    assert len(index["files"]) == inventory["indexed_files"]
    assert set(files) == set(index["files"]) | {"index.json"}
    for member, binding in index["files"].items():
        assert len(files[member]) == binding["size"]
        assert _sha(files[member]) == binding["sha256"]
    assert _sha(files["packaging.py.txt"]) == inventory["provenance"]["packaging_source_sha256"]
    result = auditor.audit(ASSETS / name, tmp_path / "audit")
    assert result["cases"] == json.loads(files["summary.json"])["cases"]
    assert result["cases"]["joint"]["passes_api_gate"]
    assert result["cases"]["geometry"]["passes_api_gate"] is (not first)
    for variant in ("baseline", "candidate"):
        for index, seed in enumerate((20261010, 20261011, 20261012)):
            matching = [
                n for n in files if n.startswith("joint-") and n.endswith(f"-{variant}/result.json")
            ]
            matching.sort()
            value = json.loads(files[matching[index]])
            assert value["seed"] == seed
            assert value["policy_status"] == "pass"
            assert len(value["controls"]) == 32 and all(c["detected"] for c in value["controls"])


@pytest.mark.parametrize("name", ["first-attempt.tar.gz", "complete-evidence.tar.gz"])
def test_frozen_runtime_changes_only_two_modules_and_preserves_public_signatures(name: str) -> None:
    files = _members(name)
    manifests = {v: json.loads(files[f"{v}/manifest.json"]) for v in ("baseline", "candidate")}
    assert len(manifests["baseline"]) == len(manifests["candidate"]) == 401
    changed = [
        p for p in manifests["baseline"] if manifests["baseline"][p] != manifests["candidate"][p]
    ]
    assert sorted(changed) == [
        "calibrex/core/continuous_time_sparse.py",
        "calibrex/core/se3_manifold.py",
    ]
    for module in changed:
        signatures = []
        for variant in ("baseline", "candidate"):
            tree = ast.parse(files[f"{variant}/src/{module}"])
            signatures.append(
                {
                    n.name: (ast.dump(n.args), ast.dump(n.returns))
                    for n in tree.body
                    if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")
                }
            )
        assert signatures[0] == signatures[1]


def test_retained_development_checks_bind_both_test_sources_and_profile_limits() -> None:
    value = json.loads((ASSETS / "development.json").read_bytes())
    jsonschema.validate(value, json.loads((ASSETS / "development.schema.json").read_bytes()))
    files = _members("complete-evidence.tar.gz")
    assert json.loads(files["development/record.json"]) == value
    for label in ("first", "final"):
        assert (
            _sha(files[f"development/{label}-reuse-tests.py.txt"])
            == value["test_source_sha256"][label]
        )
    checks = {c["name"]: c for c in value["checks"]}
    assert checks["reuse-tests-v1"] == {
        "name": "reuse-tests-v1",
        "tests": 45,
        "failure": 1,
        "error": 0,
        "skipped": 0,
    }
    assert checks["targeted-v2"] == {
        "name": "targeted-v2",
        "tests": 210,
        "failure": 0,
        "error": 0,
        "skipped": 0,
    }
    assert checks["reuse-tests-v4"]["tests"] == 62
    profile = json.loads(files["development/profile-v1.json"])
    jsonschema.validate(profile, json.loads(files["schemas/profile.schema.json"]))
    assert profile["exact_input_and_full_fit_parity"]
    assert profile["provenance"]["source_commit"] == value["source_commits"]["first"]
    assert "no speed claim" in profile["provenance"]["scope"]
