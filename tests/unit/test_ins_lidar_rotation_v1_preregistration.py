"""The real registration must match the executable frozen contract and raw pins."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import jsonschema
from tools.ins_lidar_sota_evaluate import verify_raw_budget
from tools.ins_lidar_sota_protocol import (
    CLAIM,
    EXPECTED_CASES,
    AcquisitionReceipt,
    InsLidarPreregistration,
    RunManifest,
    load_preregistration,
)

from calibrex.core.ins_lidar_hand_eye import load_ins_lidar_hand_eye
from calibrex.evaluation.ins_lidar_hand_eye import InsLidarRunOptions

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs/assets/ins_lidar_sota_readiness"
REG = ROOT / "docs/benchmarks/ins_lidar_rotation_v1_preregistration.yaml"


def test_real_registration_matches_cases_default_options_and_gates():
    reg = load_preregistration(REG, ROOT)
    assert reg.claim_text == CLAIM
    assert tuple((c.case_id, c.dataset_family, c.rig_id) for c in reg.evaluation) == EXPECTED_CASES
    assert reg.run_recipe["options"] == json.loads(json.dumps(asdict(InsLidarRunOptions())))
    assert reg.run_recipe["max_frames"] == reg.acquisition.max_frames == 600
    assert reg.run_recipe["kitti_streams"] == ["velodyne_points", "oxts"]
    assert reg.acquisition.method_attempts == 1
    assert reg.acquisition.object_attempts == 2
    assert verify_raw_budget(reg, ROOT) <= 16 * 1024**3
    assert reg.exposure["complete"]
    assert not reg.exposure["evaluation_sensor_objects_read_before_registration"]


def test_thresholds_derive_from_pinned_genuine_development_not_generated_motion():
    reg = load_preregistration(REG, ROOT)
    development = reg.development["current_genuine"]
    path = ROOT / development["artifact"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == development["artifact_sha256"]
    artifact = load_ins_lidar_hand_eye(path)
    observed = [d for d in artifact.dofs if d.unit == "deg" and d.status == "estimated"]
    assert len(observed) == 2
    worst = max(abs(d.error_to_reference) for d in observed)
    assert reg.thresholds.maximum_reference_difference_deg == math.ceil(1.5 * worst / 0.25) * 0.25
    assert reg.thresholds.maximum_unobserved_axis_fraction == 0.5
    assert reg.thresholds.maximum_control_miss_fraction == 0.0
    assert all(d.known_bad_control.detected for d in observed)
    assert "stricter" in reg.development["threshold_derivation"]["coverage"]
    assert "stricter" in reg.development["threshold_derivation"]["controls"]


def test_committed_schemas_match_models_and_registration():
    for model, name in (
        (InsLidarPreregistration, "ins_lidar_rotation_v1_preregistration"),
        (AcquisitionReceipt, "acquisition"),
        (RunManifest, "run"),
    ):
        schema = json.loads((ASSETS / f"{name}.schema.json").read_text())
        assert schema == model.model_json_schema()
        jsonschema.Draft202012Validator.check_schema(schema)
    reg = load_preregistration(REG, ROOT)
    jsonschema.validate(reg.model_dump(mode="json"), InsLidarPreregistration.model_json_schema())
