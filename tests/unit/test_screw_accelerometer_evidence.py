"""Retain the full fixed-source accelerometer development cohort and bindings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from calibrex.core.validation import validate_file

_ASSETS = Path(__file__).resolve().parents[2] / "docs/assets/screw_accelerometer_jacobians"
_SOURCE = "11cd1ca4e71779505ba071711eb2486b8f4f90d8"
_CASES = (
    (20261010, 29, "54b1b181283fe825912d6564f3220027a95f460e6a849df3708dd656692f8233"),
    (20261011, 22, "24d24beb478bbb98d46adc3d2f5c309f9ef649cde152474ab76583af534c6137"),
    (20261012, 25, "bd64c2b734f8211bd03690a0d8e0471af9e474e7761b3649b2ea1c912879d19d"),
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("seed,iterations,digest", _CASES)
def test_native_development_outcome_is_complete_and_valid(
    seed: int, iterations: int, digest: str,
) -> None:
    path = _ASSETS / f"seed-{seed}.json"
    assert _sha(path) == digest
    assert validate_file(path, kind="continuous-time-lidar-imu-joint").valid
    data = json.loads(path.read_text(encoding="utf-8"))
    assert (data["seed"], data["fit_status"], data["iterations"], data["policy_status"]) == (
        seed, "converged", iterations, "pass",
    )
    assert data["provenance"]["git_commit"] == _SOURCE[:8]
    assert data["provenance"]["seed"] == seed
    assert len(data["controls"]) == 32
    assert all(control["detected"] for control in data["controls"])


def test_receipt_binds_every_declared_outcome_and_original_launcher() -> None:
    receipt = json.loads((_ASSETS / "receipt.json").read_text(encoding="utf-8"))
    schema = json.loads((_ASSETS / "receipt.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(receipt, schema)
    assert receipt["source_commit"] == _SOURCE
    assert receipt["source_unchanged"] is True
    assert receipt["launcher_sha256"] == _sha(_ASSETS / "generator.py.txt")
    assert {path.name for path in _ASSETS.glob("seed-*.json")} == {
        f"seed-{seed}.json" for seed, *_ in _CASES
    }
    assert [row["seed"] for row in receipt["outcomes"]] == [row[0] for row in _CASES]
    for row in receipt["outcomes"]:
        assert row["returncode"] == 0
        assert row["artifact_sha256"] == _sha(_ASSETS / Path(row["artifact"]).name)
        assert row["command"][:3] == ["calibrex", "trajectory", "recover-lidar-imu-joint"]


def test_source_manifest_binding_and_complete_module_count() -> None:
    receipt = json.loads((_ASSETS / "receipt.json").read_text(encoding="utf-8"))
    path = _ASSETS / "source-manifest.json"
    assert _sha(path) == receipt["source_manifest_sha256"]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert len(manifest) == 401
    assert "src/calibrex/core/continuous_time_sparse.py" in manifest
    assert all(key.startswith("src/calibrex/") and len(value) == 64
               for key, value in manifest.items())
