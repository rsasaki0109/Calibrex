"""Bind post hoc conclusions to native artifacts and the acquired reference data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from calibrex.core.ins_lidar_hand_eye import load_ins_lidar_hand_eye

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs/assets/ins_lidar_registration_quality"
ORIGINAL = ROOT / "docs/assets/ins_lidar_sota_readiness/evaluation"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_translation_weighting_has_provenance_but_does_not_establish_rotation():
    report = json.loads((ASSETS / "weight-report.json").read_text(encoding="utf-8"))
    schema = json.loads((ASSETS / "weight-report.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(report, schema)
    assert digest(ASSETS / "weight-source.py.txt") == report["provenance"]["script_sha256"]
    assert report["provenance"]["new_accuracy_claim"] is False
    baseline = load_ins_lidar_hand_eye(ASSETS / "weight-baseline.yaml")
    candidate = load_ins_lidar_hand_eye(ASSETS / "weight-translation-x10.yaml")
    original = load_ins_lidar_hand_eye(ORIGINAL / "runs/2011_09_29_0071.yaml")
    assert baseline.dofs == original.dofs
    for label, name, artifact in [
        ("baseline", "weight-baseline.yaml", baseline),
        ("translation_noise_x10", "weight-translation-x10.yaml", candidate),
    ]:
        saved = report["cases"][label]
        assert saved["artifact_sha256"] == digest(ASSETS / name)
        assert artifact.provenance.git_commit == report["provenance"]["source_commit"]
        assert artifact.provenance.input_sha256 == report["provenance"]["input_sha256"]
        assert saved["rotation"] == [d.model_dump(mode="json") for d in artifact.dofs[:3]]
        assert all(d.status == "unobservable" for d in artifact.dofs[:3])
        assert artifact.options["observable_rotation_std_deg"] == 0.1
        assert artifact.options["priors"] == []
    options = dict(candidate.options)
    for key in ("translation_sigma_floor_m", "translation_sigma_per_m"):
        assert options[key] == pytest.approx(baseline.options[key] * 10)
        options[key] = baseline.options[key]
    assert options == baseline.options
    assert candidate.train_blocks == baseline.train_blocks
    assert candidate.holdout_blocks == baseline.holdout_blocks
    yaw = next(d for d in candidate.dofs if d.name == "yaw")
    assert yaw.std_reported > 1.0
    assert yaw.known_bad_control is not None and not yaw.known_bad_control.detected


def test_complete_reference_manifest_preserves_original_packets_and_bounds_acquisition():
    report = json.loads((ASSETS / "full-reference-report.json").read_text(encoding="utf-8"))
    schema = json.loads((ASSETS / "full-reference-report.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(report, schema)
    provenance = report["provenance"]
    for name, field in [
        ("full-reference-source.py.txt", "script_sha256"),
        ("full-reference-acquisition.json", "acquisition_sha256"),
        ("full-reference-input-sha256.json", "input_manifest_sha256"),
    ]:
        assert digest(ASSETS / name) == provenance[field]
    assert provenance["streams"] == ["oxts"]
    assert provenance["attempts"] == 1
    assert provenance["new_accuracy_claim"] is False
    acquisition = json.loads((ASSETS / "full-reference-acquisition.json").read_text())
    record = acquisition["record"]
    assert record["fetched_bytes"] <= provenance["selected_payload_cap_bytes"] == 2 * 1024**2
    assert record["crc_verified"]
    assert set(record["frames"]) == {"oxts"}
    manifest = json.loads((ASSETS / "full-reference-input-sha256.json").read_text())["files"]
    packets = {
        p.split("oxts/data/")[1]: value for p, value in manifest.items() if "oxts/data/" in p
    }
    old = json.loads((ORIGINAL / "inputs/2011_09_29_0071/acquisition.json").read_text())
    old_packets = {p["path"].split("oxts/data/")[1]: p["sha256"] for p in old["files"]
                   if "oxts/data/" in p["path"]}
    assert len(old_packets) == 600
    assert len(packets) == record["frames"]["oxts"] == 1059
    assert all(packets[name] == value for name, value in old_packets.items())
    first, complete = report["rows"]
    assert first["samples"] == 600 and complete["samples"] == 1059
    assert complete["duration_s"] > first["duration_s"]
    assert complete["yaw_span_deg"] == pytest.approx(first["yaw_span_deg"])
