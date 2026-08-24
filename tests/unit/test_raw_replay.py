"""Raw-capture replay and field-replacement pilot contract tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from calibrex.cli.main import main
from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.lifecycle_registry import load_registry
from calibrex.core.raw_replay import (
    ReplayDefinition,
    ReplayRegistryRequest,
    ReplayStageArtifact,
    load_replay_definition,
    plan_raw_replay,
    run_field_replacement_pilot,
    run_raw_replay,
    verify_raw_replay,
)

FIXTURE = Path("examples/raw_replay/synthetic/synthetic-definition.yaml")


def test_synthetic_raw_replay_clean_and_stage_digests(tmp_path: Path) -> None:
    output = tmp_path / "replay"
    plan = plan_raw_replay(FIXTURE, output=tmp_path / "plan.json")
    assert plan.status == "PASS"
    result = run_raw_replay(FIXTURE, output_directory=output)
    assert result.status == "PASS"
    assert result.decision == "ADOPT"
    assert result.artifact_sha256 != "0" * 64
    assert {stage.stage_kind for stage in result.stages} == {
        "verify_inputs",
        "readiness",
        "execute_calibration",
        "validate_result",
        "evaluate",
        "compare",
        "lifecycle",
        "autoware_plan",
    }
    verified = verify_raw_replay(output / "replay-result.json", definition=FIXTURE)
    assert verified.artifact_sha256 == result.artifact_sha256


def test_definition_provenance_does_not_depend_on_ambient_git_commit() -> None:
    definition = load_replay_definition(FIXTURE)
    assert definition.provenance.git_commit is None
    assert definition.with_artifact_digest().artifact_sha256 == definition.artifact_sha256


def test_generated_replay_provenance_binds_git_commit_when_available(tmp_path: Path) -> None:
    from calibrex.core.provenance import git_commit

    result = run_raw_replay(FIXTURE, output_directory=tmp_path / "replay")
    commit = git_commit()
    if commit is not None:
        assert result.generated_provenance.git_commit == commit
        assert all(
            ReplayStageArtifact.model_validate(read_mapping(path)).provenance.git_commit == commit
            for path in sorted((tmp_path / "replay" / "stages").glob("*.json"))
        )


def test_replay_source_drift_is_blocked(tmp_path: Path) -> None:
    copied = tmp_path / "fixture"
    shutil.copytree(FIXTURE.parent, copied)
    (copied / "raw-capture.bin").write_bytes(b"tampered synthetic capture")
    result = run_raw_replay(copied / FIXTURE.name, output_directory=tmp_path / "out")
    assert result.status == "BLOCKED"
    assert result.decision == "BLOCKED"
    assert "digest mismatch" in result.reason
    assert result.stages[0].stage_kind == "verify_inputs"


def test_replay_known_bad_and_weak_observability_never_adopt(tmp_path: Path) -> None:
    copied = tmp_path / "fixture"
    shutil.copytree(FIXTURE.parent, copied)
    candidate = read_mapping(copied / "candidate_result.yaml")
    candidate["observability"]["rank"] = 1
    candidate["observability"]["grade"] = "fail"
    write_mapping(copied / "candidate_result.yaml", candidate)
    # The modified candidate is intentionally no longer the declared input.
    result = run_raw_replay(copied / FIXTURE.name, output_directory=tmp_path / "out")
    assert result.status == "BLOCKED"
    assert result.decision == "BLOCKED"


def test_replay_cli_plan_run_verify(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    plan_path = tmp_path / "plan.json"
    output = tmp_path / "run"
    assert main(["replay", "plan", str(FIXTURE), "--output", str(plan_path), "--json"]) == 0
    capsys.readouterr()
    assert main(["replay", "run", str(FIXTURE), "--output-dir", str(output), "--json"]) == 0
    capsys.readouterr()
    assert (
        main(
            [
                "replay",
                "verify",
                str(output / "replay-result.json"),
                "--definition",
                str(FIXTURE),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True


def test_field_replacement_pilot_records_remount_and_holds_without_evidence(
    tmp_path: Path,
) -> None:
    copied = tmp_path / "fixture"
    shutil.copytree(FIXTURE.parent, copied)
    definition_payload = read_mapping(copied / FIXTURE.name)
    definition_payload["registry"] = ReplayRegistryRequest(
        registry_root=str(tmp_path / "registry"),
        vehicle_id="vehicle-fixture",
        sensor_kit_id="kit-fixture",
        edge_id="edge-lidar",
        parent_frame="base_link",
        child_frame="lidar_front",
        sensor_id="lidar-front",
        serial="LIDAR-BASE-001",
        model="synthetic-lidar",
        firmware="1.0",
        mount="mount-front",
        frame="lidar_front",
        replacement_serial="LIDAR-REPLACEMENT-002",
        replacement_mount="mount-front-remount",
        replacement_sensor_id="lidar-front-replacement",
    ).model_dump(mode="json")
    definition = ReplayDefinition.model_validate(definition_payload).with_artifact_digest()
    pilot_definition = copied / "pilot-definition.yaml"
    definition.save(pilot_definition)

    pilot = run_field_replacement_pilot(
        pilot_definition,
        output_directory=tmp_path / "pilot-output",
    )

    assert pilot.remount_count == 1
    assert pilot.package_mutated is False
    assert pilot.rollback.status == "PASS"
    assert pilot.decision != "ADOPT"
    state = load_registry(tmp_path / "registry").state()
    assert state.sensors["lidar-front"].state == "removed"
    assert state.sensors["lidar-front-replacement"].serial == "LIDAR-REPLACEMENT-002"
