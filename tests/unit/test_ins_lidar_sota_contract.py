"""Audit contract, input custody and failure retention without evaluation data."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import ClassVar

import pytest
from tools import build_ins_lidar_sota_audit as builder
from tools import ins_lidar_sota_acquire as acquire
from tools import ins_lidar_sota_evaluate as runner
from tools.ins_lidar_sota_protocol import (
    EXPECTED_CASES,
    EvaluationCase,
    InsLidarPreregistration,
)
from tools.score_ins_lidar_sota import CaseSpec, score_case

from calibrex.core.ins_lidar_hand_eye import load_ins_lidar_hand_eye


@pytest.fixture
def registration():
    return InsLidarPreregistration(
        frozen_code_commit="a" * 40,
        metadata={f: {"path": f + ".json", "sha256": "b" * 64} for f in ("kitti_raw", "boreas")},
        evaluation=[
            EvaluationCase(case_id=c, dataset_family=f, rig_id=r) for c, f, r in EXPECTED_CASES
        ],
        calibration_archive_sha256=dict.fromkeys(("2011_09_29", "2011_09_30"), "c" * 64),
        development={},
        exposure={},
        run_recipe={},
        provenance={},
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("claim_text", "a weaker claim"),
        ("thresholds", {"maximum_reference_difference_deg": 0.75}),
        ("evaluation", []),
        ("metadata", {"kitti_raw": {"path": "../data", "sha256": "b" * 64}}),
    ],
)
def test_contract_rejects_claim_gate_case_or_pin_changes(registration, field, value):
    data = registration.model_dump()
    data[field] = value
    with pytest.raises(ValueError):
        InsLidarPreregistration.model_validate(data)


def test_uncertainty_cannot_be_relabelled_as_observed():
    artifact = load_ins_lidar_hand_eye(Path("docs/assets/kitti_ins_lidar_2011_09_26.yaml"))
    artifact.dofs[0].std_reported = 100.0
    assert score_case(artifact)["estimated_axes"] == 0


def test_input_custody_checks_hash_size_and_path(tmp_path):
    payload = tmp_path / "input"
    payload.write_bytes(b"native input")
    receipt = {
        "files": [
            {
                "path": "input",
                "bytes": payload.stat().st_size,
                "sha256": acquire.file_digest(payload),
            }
        ]
    }
    acquire.verify_input_receipt(receipt, tmp_path)
    payload.write_bytes(b"changed data")
    with pytest.raises(ValueError, match="digest mismatch"):
        acquire.verify_input_receipt(receipt, tmp_path)
    receipt["files"][0]["path"] = "../outside"
    with pytest.raises(ValueError, match="escapes"):
        acquire.verify_input_receipt(receipt, tmp_path)


class Response(io.BytesIO):
    status = 200
    headers: ClassVar[dict[str, str]] = {"ETag": '"pin"'}


def test_download_retry_retains_failure_and_reuses_verified_success(tmp_path, monkeypatch):
    calls = []

    def request(req, timeout):
        calls.append(req)
        if len(calls) == 1:
            raise OSError("transport interrupted")
        return Response(b"abc")

    monkeypatch.setattr(acquire.urllib.request, "urlopen", request)
    monkeypatch.setattr(acquire.time, "sleep", lambda _: None)
    path = tmp_path / "object"
    kwargs = {"size": 3, "etag": "pin", "cap": 4}
    result = acquire._download("https://example.test/object", path, **kwargs)
    assert result["sha256"] == hashlib.sha256(b"abc").hexdigest()
    history = json.loads(path.with_name("object.attempts.json").read_text())
    assert [a["status"] for a in history] == ["FAIL", "PASS"]
    assert acquire._download("https://example.test/object", path, **kwargs) == result
    assert len(calls) == 2
    path.write_bytes(b"bad")
    with pytest.raises(ValueError, match="changed on resume"):
        acquire._download("https://example.test/object", path, **kwargs)
    assert len(calls) == 2


def test_download_pin_mismatch_is_not_retried(tmp_path, monkeypatch):
    calls = []

    def request(req, timeout):
        calls.append(req)
        return Response(b"abc")

    monkeypatch.setattr(acquire.urllib.request, "urlopen", request)
    with pytest.raises(ValueError, match="ETag"):
        acquire._download(
            "https://example.test/object", tmp_path / "object", size=3, etag="wrong", cap=4
        )
    assert len(calls) == 1


@pytest.fixture
def audit_inputs(tmp_path, registration, monkeypatch):
    prereg = tmp_path / "registration.yaml"
    prereg.write_text("fixture registration")
    monkeypatch.setattr(builder, "load_preregistration", lambda *_: registration)
    monkeypatch.setattr(builder, "verify_frozen_source", lambda *a, **kw: "a" * 40)
    manifest = {
        "preregistration_sha256": acquire.file_digest(prereg),
        "frozen_code_commit": registration.frozen_code_commit,
        "cases": [],
        "input_data_verified": True,
        "input_receipts": {},
    }
    for index, declared in enumerate(registration.evaluation):
        artifact = load_ins_lidar_hand_eye(Path("docs/assets/kitti_ins_lidar_2011_09_26.yaml"))
        artifact.provenance.dataset_family = declared.dataset_family
        artifact.provenance.drive_ids = declared.drive_ids
        path = tmp_path / f"case-{index}.yaml"
        artifact.save(path)
        manifest["cases"].append(
            CaseSpec(
                **declared.model_dump(),
                drive_ids=declared.drive_ids,
                artifact_path=path.name,
                artifact_sha256=acquire.file_digest(path),
            ).model_dump()
        )
        work = tmp_path / declared.case_id
        work.mkdir()
        data = work / "input"
        data.write_text(declared.case_id)
        receipt = work / "acquisition.json"
        receipt.write_text(
            json.dumps(
                {
                    "case_id": declared.case_id,
                    "verified": True,
                    "files": [
                        {
                            "path": "input",
                            "bytes": data.stat().st_size,
                            "sha256": acquire.file_digest(data),
                        }
                    ],
                    "provenance": {
                        "frozen_code_commit": registration.frozen_code_commit,
                        "source_metadata_sha256": registration.metadata[
                            declared.dataset_family
                        ].sha256,
                    },
                }
            )
        )
        manifest["input_receipts"][declared.case_id] = {
            "path": str(receipt),
            "sha256": acquire.file_digest(receipt),
            "work_dir": str(work),
        }
    manifest_path = tmp_path / "run.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, prereg, manifest


def test_builder_uses_real_audit_engine_and_all_eight_gates(audit_inputs, tmp_path):
    manifest_path, prereg, _ = audit_inputs
    result = builder.build(manifest_path, prereg, tmp_path / "audit", source_root=tmp_path)
    assert result["verdict"] == "supported"
    assert result["achieved"] == result["requirements"] == 8


def test_missing_case_refutes_instead_of_shrinking_population(audit_inputs, tmp_path):
    manifest_path, prereg, manifest = audit_inputs
    manifest["cases"][0].update(artifact_path=None, artifact_sha256=None)
    manifest_path.write_text(json.dumps(manifest))
    result = builder.build(manifest_path, prereg, tmp_path / "audit", source_root=tmp_path)
    assert result["verdict"] == "refuted"
    assert len(result["scores"]["cases"]) == 4


def test_builder_rejects_dropped_or_corrupted_inputs(audit_inputs, tmp_path):
    manifest_path, prereg, manifest = audit_inputs
    (tmp_path / EXPECTED_CASES[0][0] / "input").write_text("changed")
    with pytest.raises(ValueError, match="digest mismatch"):
        builder.build(manifest_path, prereg, tmp_path / "audit", source_root=tmp_path)
    manifest["cases"].pop()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="dropped or substituted"):
        builder.build(manifest_path, prereg, tmp_path / "audit", source_root=tmp_path)


def test_runner_checks_registration_before_acquisition(tmp_path, monkeypatch):
    def reject(*_):
        raise ValueError("not merged")

    monkeypatch.setattr(runner, "verify_registered_main", reject)
    monkeypatch.setattr(runner, "acquire_case", lambda *_: pytest.fail("payload read"))
    with pytest.raises(ValueError, match="not merged"):
        runner.evaluate(tmp_path / "prereg", tmp_path / "work", tmp_path / "out")


def test_runner_does_not_repeat_interrupted_method(tmp_path, registration, monkeypatch):
    monkeypatch.setattr(runner, "verify_raw_budget", lambda *_: 0)
    monkeypatch.setattr(runner, "verify_registered_main", lambda *_: (registration, "a" * 40))
    monkeypatch.setattr(runner, "acquire_case", lambda *_: pytest.fail("case repeated"))
    prereg = tmp_path / "prereg"
    prereg.write_text("fixture")
    output = tmp_path / "output"
    output.mkdir()
    manifest = {
        "preregistration_sha256": acquire.file_digest(prereg),
        "frozen_code_commit": registration.frozen_code_commit,
        "cases": [
            CaseSpec(**c.model_dump(), drive_ids=c.drive_ids).model_dump()
            for c in registration.evaluation
        ],
        "input_receipts": {},
        "attempts": {c.case_id: {"status": "RUNNING"} for c in registration.evaluation},
    }
    (output / "run.json").write_text(json.dumps(manifest))
    result = runner.evaluate(prereg, tmp_path / "work", output)
    assert all(c["artifact_path"] is None for c in result["cases"])
    assert not result["input_data_verified"]
