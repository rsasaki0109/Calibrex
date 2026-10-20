"""Verify the complete, immutable final-gradient development comparison."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from typing import Any

import jsonschema
import pytest

from calibrex.core.continuous_time_lidar_imu_joint import ContinuousTimeLidarImuJointArtifact

_ASSETS = Path(__file__).resolve().parents[2] / "docs/assets/screw_final_gradient"
_SEEDS = (20261010, 20261011, 20261012)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read(name: str) -> Any:
    return json.loads((_ASSETS / name).read_bytes())


@pytest.fixture(scope="module")
def members() -> dict[str, bytes]:
    with tarfile.open(_ASSETS / "complete-development.tar.gz") as archive:
        return {member.name: archive.extractfile(member).read() for member in archive.getmembers()}


def test_receipt_and_complete_native_cohort_have_valid_schemas() -> None:
    receipt = _read("receipt.json")
    jsonschema.validate(receipt, _read("receipt.schema.json"))
    assert [pair["seed"] for pair in receipt["pairs"]] == list(_SEEDS)
    assert [pair["iterations"] for pair in receipt["pairs"]] == [21, 23, 29]
    for pair in receipt["pairs"]:
        raw = (_ASSETS / pair["current_artifact"]).read_bytes()
        assert _sha(raw) == pair["current_artifact_sha256"]
        artifact = json.loads(raw)
        jsonschema.validate(artifact, ContinuousTimeLidarImuJointArtifact.model_json_schema())
        assert artifact["provenance"]["git_commit"] == receipt["source_commit"][:8]
        assert artifact["provenance"]["seed"] == pair["seed"]
        assert artifact["policy_status"] == "pass"
        assert len(artifact["controls"]) == 32
        assert all(control["detected"] for control in artifact["controls"])


def test_archive_inventory_and_every_member_byte_are_bound(members: dict[str, bytes]) -> None:
    receipt = _read("receipt.json")
    raw = (_ASSETS / "complete-development.tar.gz").read_bytes()
    assert len(raw) == receipt["archive_bytes"]
    assert _sha(raw) == receipt["archive_sha256"]
    assert set(members) == {item["name"] for item in receipt["archive_members"]}
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        assert len(archive.getmembers()) == len(members)
    for item in receipt["archive_members"]:
        name = PurePosixPath(item["name"])
        assert not name.is_absolute() and ".." not in name.parts
        data = members[item["name"]]
        assert len(data) == item["size_bytes"] and _sha(data) == item["sha256"]


def test_all_paired_fit_records_validate_without_rewriting(members: dict[str, bytes]) -> None:
    for stage in ("baseline", "current"):
        root = f"development/{stage}"
        receipt = json.loads(members[root + "/receipt.json"])
        jsonschema.validate(receipt, json.loads(members[root + "/receipt.schema.json"]))
        assert receipt["source_unchanged"] and len(receipt["outcomes"]) == 3
        for seed in _SEEDS:
            directory = f"{root}/seed-{seed}"
            for kind in ("input", "fit", "model", "outcome"):
                data = json.loads(members[f"{directory}/{kind}.json"])
                jsonschema.validate(data, json.loads(members[f"{root}/{kind}.schema.json"]))
                assert data["provenance"]["seed"] == seed
            native = json.loads(members[f"{directory}/native.json"])
            jsonschema.validate(native, ContinuousTimeLidarImuJointArtifact.model_json_schema())


def test_paired_inputs_states_covariance_and_controls_match(members: dict[str, bytes]) -> None:
    for pair in _read("receipt.json")["pairs"]:
        seed = pair["seed"]
        records = {}
        for stage in ("baseline", "current"):
            records[stage] = {kind: json.loads(members[
                f"development/{stage}/seed-{seed}/{kind}.json"
            ]) for kind in ("input", "fit", "model", "native")}
        old, new = records["baseline"], records["current"]
        inputs = {key: old["input"][key] for key in ("problem", "options")}
        assert inputs == {key: new["input"][key] for key in inputs}
        assert _sha((json.dumps(inputs, indent=2) + "\n").encode()) == pair["input_sha256"]
        old_fit, new_fit = old["fit"]["payload"], new["fit"]["payload"]
        assert old_fit.pop("gradient_norm") == pair["previous_reported_gradient"]
        reported = new_fit.pop("gradient_norm")
        assert old_fit == new_fit
        assert reported == max(abs(value) for value in new["model"]["half_gradient"])
        assert reported == pair["current_reported_gradient"] == pair["actual_final_gradient"]
        for key in ("normal", "half_gradient"):
            assert old["model"][key] == new["model"][key]
        del old["native"]["provenance"], new["native"]["provenance"]
        assert old["native"] == new["native"]
        assert pair["current_assemblies"] == pair["previous_assemblies"]


def test_exact_maintained_tests_bind_the_red_and_green_counts(members: dict[str, bytes]) -> None:
    expected = {"baseline": 15, "current": 0}
    for stage, failures in expected.items():
        record = json.loads(members[f"development/{stage}-tests.json"])
        assert record["counts"] == {"tests": 24, "failures": failures, "errors": 0, "skipped": 0}
        assert record["provenance"]["test_source_sha256"] == _sha(members["maintained-test.py.txt"])
        xml = members[f"development/{stage}-tests.xml"]
        assert _sha(xml) == record["provenance"]["junit_sha256"]
        cases = list(ET.fromstring(xml).iter("testcase"))
        assert len(cases) == 24 and sum(c.find("failure") is not None for c in cases) == failures
        generator = f"development/{record['provenance']['generator']}"
        assert _sha(members[generator]) == record["provenance"]["launcher_sha256"]


def test_frozen_sources_producers_and_failed_history_are_retained(
    members: dict[str, bytes],
) -> None:
    receipt = _read("receipt.json")
    current_manifest = (_ASSETS / "source-manifest.json").read_bytes()
    assert current_manifest == members["development/current/source-manifest.json"]
    assert _sha(current_manifest) == receipt["source_manifest_sha256"]
    current = json.loads(current_manifest)
    baseline = json.loads(members["development/baseline/source-manifest.json"])
    assert len(current) == len(baseline) == receipt["normalized_source_modules"] == 401
    assert [p for p in baseline if baseline[p] != current[p]] == receipt["changed_source_modules"]
    assert receipt["changed_source_modules"] == ["src/calibrex/core/continuous_time_sparse.py"]
    assert _sha(members["development/baseline-source.py.txt"]) == baseline[
        "src/calibrex/core/continuous_time_sparse.py"
    ]
    producer = (_ASSETS / "generator.py.txt").read_bytes()
    assert producer == members["development/run.py"]
    assert _sha(producer) == receipt["cohort_generator_sha256"]
    assert _sha((_ASSETS / "package.py.txt").read_bytes()) == receipt["provenance"][
        "launcher_sha256"
    ]
    failed = json.loads(members["history/convergence-report-audit-v1/result.json"])
    fixed = json.loads(members["history/convergence-report-audit-v2/result.json"])
    assert len(failed["exceptions"]) == 6 and not failed["observations"]
    assert not fixed["exceptions"] and len(fixed["observations"]) == 6
    for name in receipt["historical_directories"]:
        assert (
            f"history/{name}/run.py" in members
            or f"history/{name}/run_report_candidate.py" in members
        )
