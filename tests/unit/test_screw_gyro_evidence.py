"""Retain every gyro development outcome and its source/input/value bindings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from calibrex.core.validation import validate_file

_ASSETS = Path(__file__).resolve().parents[2] / "docs/assets/screw_gyro_jacobians"
_SOURCE = "8a6b69d84c13003cceddc3c4921985424068f442"
_BASE = "b733880e017877ed106ee80c4d7a6cfe2abe2bf2"
_CASES = (
    (20261010, 21, "2457f2389af7597c3b6052189c76a5c9befa6052e63316f5ba8b57f4252138dc"),
    (20261011, 23, "91dc8458687ec5af2cc7111e759b4e95943e4eb424a2a8f591c3974ebf86ea6b"),
    (20261012, 29, "eac4cc41fdd682b204e8eda1cbe5732804efc003c98af26a00cf77899c88bda5"),
)
_INPUTS = (
    "8e614fc61dca760efdb9580b98c9be7094c9b2532c4b4ca71e18612438ee087d",
    "767778193c2ec26d1a7d68bdd69d94d76a5a96990a553c3dc0837b85206b9fac",
    "92550880fb46e18bfd1024e0e42485fe822faa651cdabff9d8813bfcd19da82b",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(name: str) -> dict:
    return json.loads((_ASSETS / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("seed,iterations,digest", _CASES)
def test_native_development_outcome_is_complete_and_valid(
    seed: int, iterations: int, digest: str,
) -> None:
    path = _ASSETS / f"seed-{seed}.json"
    assert _sha(path) == digest
    assert validate_file(path, kind="continuous-time-lidar-imu-joint").valid
    data = _read(path.name)
    assert (data["seed"], data["fit_status"], data["iterations"], data["policy_status"]) == (
        seed, "converged", iterations, "pass",
    )
    assert data["provenance"]["git_commit"] == _SOURCE[:8]
    assert data["provenance"]["seed"] == seed
    assert len(data["controls"]) == 32
    assert all(control["detected"] for control in data["controls"])


def test_receipt_binds_complete_cohort_and_original_launcher() -> None:
    receipt = _read("receipt.json")
    jsonschema.validate(receipt, _read("receipt.schema.json"))
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


def test_source_manifest_binding_and_module_count() -> None:
    receipt = _read("receipt.json")
    assert _sha(_ASSETS / "source-manifest.json") == receipt["source_manifest_sha256"]
    manifest = _read("source-manifest.json")
    assert len(manifest) == 401
    assert "src/calibrex/core/continuous_time_sparse.py" in manifest
    assert all(key.startswith("src/calibrex/") and len(value) == 64
               for key, value in manifest.items())


def test_value_parity_binds_both_sources_fixture_and_producer() -> None:
    assert _sha(_ASSETS / "value-parity.json") == (
        "383ede2d1a9813c998951f70552e4cbba0d5789a64a037fe28a2c80a14b9625c"
    )
    record = _read("value-parity.json")
    jsonschema.validate(record, _read("value-parity.schema.json"))
    provenance = record["provenance"]
    assert provenance["source_commit"] == _SOURCE
    assert provenance["baseline_commit"] == _BASE
    assert provenance["launcher_sha256"] == _sha(_ASSETS / "verify-values.py.txt")
    assert provenance["fixture_source_sha256"] == _sha(_ASSETS / "fixture.py.txt")
    assert {(row["mounted"], tuple(row["body_window"])) for row in record["cases"]} == {
        (mounted, window) for mounted in (False, True)
        for window in ((0.2, 0.8), (0.4, 1.4), (1.0, 2.0), (1.2, 2.2), (2.1, 2.8))
    }
    assert len(record["cases"]) == 10
    for row in record["cases"]:
        assert row["identical"] is True
        assert row["baseline_residual"] == row["current_residual"]
        assert row["baseline_integrated_rotation"] == row["current_integrated_rotation"]


def test_input_comparison_binds_all_seeds_and_original_producer() -> None:
    assert _sha(_ASSETS / "input-comparison.json") == (
        "cece67044e73f5292fb63352cf418531ffe92c8925719599eb966b53e159276b"
    )
    record = _read("input-comparison.json")
    jsonschema.validate(record, _read("input-comparison.schema.json"))
    assert record["provenance"]["source_commit"] == _SOURCE
    assert record["provenance"]["launcher_sha256"] == _sha(_ASSETS / "verify-inputs.py.txt")
    assert [row["seed"] for row in record["cases"]] == [row[0] for row in _CASES]
    assert [row["canonical_input_options_sha256"] for row in record["cases"]] == list(_INPUTS)
    assert all(row["identical_to_archived_development_input"] for row in record["cases"])
