"""The INS-LiDAR v2 pre-registration agrees with the frozen tools it names."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path

import yaml
from tools import ins_lidar_sota_v2_inventory as inventory
from tools import ins_lidar_sota_v2_protocol as protocol
from tools.boreas_ins_lidar_metadata import BoreasInsLidarMetadata
from tools.ins_lidar_sota_inventory import KittiInsLidarInventory

from calibrex.evaluation.ins_lidar_hand_eye import InsLidarRunOptions

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "docs/benchmarks/ins_lidar_rotation_v2_preregistration.yaml"
RAW = yaml.safe_load(PATH.read_text(encoding="utf-8"))
PLAN = protocol.load_preregistration(PATH, ROOT)  # validates the claim, cases and both pins


def test_the_cases_are_the_contracts_and_every_one_is_pinned_in_its_inventory() -> None:
    cases = [(c.case_id, c.dataset_family, c.rig_id) for c in PLAN.evaluation]
    assert tuple(cases) == protocol.EXPECTED_CASES
    # YAML 1.1 reads an unquoted 2011_09_30_0028 as a number: every ID must stay a string
    assert all(isinstance(c["case_id"], str) for c in RAW["evaluation"])
    kitti = KittiInsLidarInventory.model_validate_json(
        (ROOT / PLAN.metadata["kitti_raw"].path).read_text()
    )
    boreas = BoreasInsLidarMetadata.model_validate_json(
        (ROOT / PLAN.metadata["boreas"].path).read_text()
    )
    declared_kitti = [c for c, family, _ in cases if family == "kitti_raw"]
    declared_boreas = [c for c, family, _ in cases if family == "boreas"]
    assert [r.case_id for r in kitti.selected] == declared_kitti
    assert kitti.selection_rule == inventory.KITTI_SELECTION_RULE
    assert all(r.selected_frames == {"velodyne_points": 600, "oxts": 600} for r in kitti.selected)
    complete = [r.recording_id for r in boreas.recordings if not r.missing_required]
    assert complete == declared_boreas and boreas.selection_rule == inventory.BOREAS_SELECTION_RULE
    assert all(len(r.lidar) == 600 for r in boreas.recordings if not r.missing_required)
    assert kitti.sensor_member_download_bytes == boreas.sensor_object_download_bytes == 0


def test_no_declared_case_was_read_by_an_ins_lidar_method() -> None:
    exposure = PLAN.exposure
    declared = {c.case_id for c in PLAN.evaluation}
    assert not declared & {*exposure["spent_kitti"], *exposure["spent_boreas"]}
    assert not declared & set(inventory.V1_EVALUATION)
    assert exposure["evaluation_sensor_objects_read_before_registration"] is False
    assert "camera-lidar v2" in exposure["cross_pair_reads"]  # the KITTI cross-pair reads


def test_the_gates_are_the_tools_constants_and_no_looser_than_v1() -> None:
    thresholds = PLAN.thresholds
    assert thresholds.maximum_reference_difference_deg == protocol.ERROR_MAX_DEG == 0.5
    assert thresholds.maximum_unobserved_axis_fraction == protocol.UNOBSERVED_FRACTION_MAX == 0.5
    assert thresholds.maximum_control_miss_fraction == protocol.CONTROL_MISS_FRACTION_MAX == 0.0
    assert thresholds.maximum_method_failure_fraction == 0.0
    v1 = yaml.safe_load(
        (ROOT / "docs/benchmarks/ins_lidar_rotation_v1_preregistration.yaml").read_text()
    )["thresholds"]
    assert thresholds.maximum_reference_difference_deg <= v1["maximum_reference_difference_deg"]
    assert thresholds.maximum_unobserved_axis_fraction <= v1["maximum_unobserved_axis_fraction"]
    assert thresholds.maximum_control_miss_fraction <= v1["maximum_control_miss_fraction"]


def test_the_method_is_the_default_one_and_the_recipe_names_real_tools() -> None:
    recipe = PLAN.run_recipe
    defaults = json.loads(json.dumps(asdict(InsLidarRunOptions())))  # tuples as YAML lists
    assert recipe["options"] == defaults  # nothing tuned for the audit
    assert recipe["max_frames"] == PLAN.acquisition.max_frames == 600
    for key in ("runner", "builder"):
        script = recipe[key].split()[1]
        assert (ROOT / script).is_file()
        assert "ins_lidar_rotation_v2_preregistration.yaml" in recipe[key]
    assert re.fullmatch(r"[0-9a-f]{40}", PLAN.frozen_code_commit)
    assert "run again" in recipe["rerun_policy"] and "incomplete" in recipe["missing_case"]
    assert PLAN.acquisition.rerun_interrupted_method and PLAN.acquisition.delete_raw_after_method


def test_the_development_records_are_the_pinned_files() -> None:
    run = PLAN.development["run"]
    records = ROOT / run["records"]
    digest = hashlib.sha256((records / "runs/run.json").read_bytes()).hexdigest()
    assert digest == run["run_json_sha256"]
    digest = hashlib.sha256((records / "audit/scores.json").read_bytes()).hexdigest()
    assert digest == run["scores_json_sha256"]
    assert set(PLAN.development["recordings"]) == {c for c, _f, _r in protocol.DEVELOPMENT_CASES}
