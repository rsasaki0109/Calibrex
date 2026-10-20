"""Bind complete historical/current prior-whitening results to frozen sources."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath

import jsonschema
import pytest

from calibrex.core.validation import validate_file

_ASSETS = Path(__file__).resolve().parents[2] / "docs/assets/prior_information_whitening"
_SOURCE = "ac7f90b6f584d23928a4ff56ed6ee1661e5f14b5"
_BASELINE = "8861ed65ebbcffb13a182ac4cd549db88ef763f2"
_ARCHIVE_SHA = "0d75dbd3d009b0c63fec818fe71ad8aa000b2c1c7ae8d016dcd64bfdebfe717a"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _receipt() -> dict:
    return json.loads((_ASSETS / "receipt.json").read_bytes())


@pytest.fixture(scope="module")
def records() -> dict[str, bytes]:
    path = _ASSETS / "complete-development.tar.gz"
    assert _sha(path.read_bytes()) == _ARCHIVE_SHA
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        assert len(members) == 49
        assert all(m.isfile() for m in members)
        assert all(
            not m.name.startswith("/") and ".." not in PurePosixPath(m.name).parts for m in members
        )
        return {m.name: archive.extractfile(m).read() for m in members}


def test_receipt_and_native_holdout_control_outcome() -> None:
    receipt = _receipt()
    schema = json.loads((_ASSETS / "receipt.schema.json").read_bytes())
    jsonschema.validate(receipt, schema)
    assert receipt["provenance"]["source_commit"] == _SOURCE
    assert receipt["baseline_source_commit"] == _BASELINE
    assert receipt["source_unchanged"] is True
    native = _ASSETS / "native-seed-20260820.json"
    assert (
        _sha(native.read_bytes())
        == ("b358e7cf3b32ae6790d5bf4b969bdd0ed2ff1d0bb252bb8fde9c9fea1c615541")
        == receipt["native"]["artifact_sha256"]
    )
    assert validate_file(native, kind="continuous-time-sliding-window").valid
    data = json.loads(native.read_bytes())
    assert data["seed"] == 20260820
    assert data["provenance"]["git_commit"] == _SOURCE[:8]
    assert data["policy_status"] == "pass"
    assert data["known_bad_skip_marginalization"] is True
    assert data["known_bad_rmse_delta_m"] > 0.01
    assert receipt["native"]["returncode"] == 0


def test_complete_archive_index_binds_every_original_byte(records: dict[str, bytes]) -> None:
    receipt = _receipt()
    assert receipt["archive_sha256"] == _ARCHIVE_SHA
    index = receipt["archive_members"]
    assert len(index) == len(records) == 49
    assert {row["name"] for row in index} == set(records)
    for row in index:
        assert len(records[row["name"]]) == row["size_bytes"]
        assert _sha(records[row["name"]]) == row["sha256"]
    assert (
        records["current/native-seed-20260820.json"]
        == (_ASSETS / "native-seed-20260820.json").read_bytes()
    )


def test_every_current_snapshot_is_schema_valid(records: dict[str, bytes]) -> None:
    pairs = [("current/input.json", "input")]
    pairs += [(f"current/phase-{i}/input.json", "input") for i in range(3)]
    pairs += [(f"current/phase-{i}/fit-result.json", "fit-result") for i in range(3)]
    pairs += [
        (f"current/{name}.json", "sliding-result")
        for name in ("sliding-result", "direct-api-result")
    ]
    pairs += [
        (f"current/{name}.json", name) for name in ("window-outcome", "retained-normal-inputs")
    ]
    for path, kind in pairs:
        value = json.loads(records[path])
        jsonschema.validate(value, json.loads(records[f"current/{kind}.schema.json"]))
        assert value["provenance"]["source_commit"] == _SOURCE


def test_both_historical_attempts_and_all_fit_results_are_retained(
    records: dict[str, bytes],
) -> None:
    root = "historical/window/"
    overall = json.loads(records[root + "outcomes.json"])
    assert overall["provenance"]["source_commit"] == _BASELINE
    assert overall["payload"]["source_unchanged"] is True
    assert (
        [row["provenance"]["implementation"] for row in overall["payload"]["outcomes"]]
        == [
            "production",
            "candidate",
        ]
        == _receipt()["historical_attempts_retained"]
    )
    for implementation in ("production", "candidate"):
        for phase in range(3):
            prefix = root + f"{implementation}/phase-{phase}/"
            data = json.loads(records[prefix + "fit-result.json"])
            jsonschema.validate(data, json.loads(records[prefix + "fit-result.schema.json"]))
            inputs = json.loads(records[prefix + "input.json"])
            jsonschema.validate(inputs, json.loads(records[root + "envelope.schema.json"]))
            assert inputs["provenance"]["source_commit"] == _BASELINE


def test_same_inputs_unaffected_phases_and_transparent_recording(
    records: dict[str, bytes],
) -> None:
    window = _receipt()["window"]
    assert window["existing_unit_gate_pass"] is True
    assert window["direct_api_result_identical"] is True
    assert [p["iterations"] for p in window["phases"]] == [7, 5, 7]
    for phase in range(3):
        old = json.loads(records[f"historical/window/production/phase-{phase}/input.json"])
        new = json.loads(records[f"current/phase-{phase}/input.json"])
        assert old["payload"] == new["payload"]
        if phase < 2:
            old_result = json.loads(
                records[f"historical/window/production/phase-{phase}/fit-result.json"]
            )
            new_result = json.loads(records[f"current/phase-{phase}/fit-result.json"])
            assert old_result["payload"] == new_result["payload"]
    recorded = json.loads(records["current/sliding-result.json"])["payload"]
    direct = json.loads(records["current/direct-api-result.json"])["payload"]
    assert recorded == direct
    # Aggregate sliding statistics currently copy batch fields. Compare the
    # actual per-window result, whose independently recomputed objective is kept.
    assert (
        window["phases"][2]["stored_final_objective"]
        == (window["phases"][2]["recomputed_objective"])
    )


def test_source_manifests_and_original_producer_bindings(records: dict[str, bytes]) -> None:
    receipt = _receipt()
    current = (_ASSETS / "source-manifest.json").read_bytes()
    assert current == records["meta/current-source-manifest.json"]
    assert _sha(current) == receipt["current_source_manifest_sha256"]
    baseline = records["meta/baseline-source-manifest.json"]
    assert _sha(baseline) == receipt["baseline_source_manifest_sha256"]
    new_map, old_map = json.loads(current), json.loads(baseline)
    assert len(new_map) == len(old_map) == 401
    assert (
        [name for name in new_map if new_map[name] != old_map[name]]
        == (receipt["changed_source_modules"])
        == ["src/calibrex/core/continuous_time_marginalization.py"]
    )
    assert (
        _sha(records["historical/baseline-marginalization.py.txt"].replace(b"\r\n", b"\n"))
        == old_map[receipt["changed_source_modules"][0]]
    )
    launcher = (_ASSETS / "generator.py.txt").read_bytes()
    assert launcher == records["meta/generator.py.txt"]
    assert _sha(launcher) == receipt["provenance"]["launcher_sha256"]
