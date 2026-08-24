"""Controlled, ROS-free downstream Autoware smoke adapter tests."""

from __future__ import annotations

import sys
from pathlib import Path

import jsonschema
import pytest
from tests.unit.test_autoware_promotion import _candidate, _package

from calibrex.export.autoware_promotion import (
    AutowarePromotionError,
    AutowarePromotionPolicy,
    apply_autoware_promotion,
    build_autoware_promotion_plan,
    rollback_autoware_promotion,
)
from calibrex.export.autoware_smoke import (
    AutowareSmokeCommand,
    AutowareSmokeConfig,
    AutowareSmokeError,
    AutowareSmokePolicy,
    autoware_smoke_json_schema,
    import_autoware_smoke,
    run_autoware_smoke,
)


def _production_plan(tmp_path: Path):
    roots, _, _, _ = _package(tmp_path)
    candidate = _candidate(tmp_path)
    return roots, build_autoware_promotion_plan(
        candidate,
        roots=roots,
        vehicle_id="vehicle_a",
        sensor_kit_id="sensor_kit_a",
        policy=AutowarePromotionPolicy(profile="production"),
    )


def _commands() -> list[AutowareSmokeCommand]:
    return [
        AutowareSmokeCommand(
            stage=stage,
            argv=[sys.executable, "-c", "print('controlled smoke')"],
        )
        for stage in ("xacro_urdf", "build_test", "tf_static", "sensor_launch")
    ]


def test_production_smoke_binds_and_allows_apply(tmp_path: Path) -> None:
    roots, plan = _production_plan(tmp_path)
    smoke = run_autoware_smoke(
        plan,
        config=AutowareSmokeConfig(
            mode="subprocess",
            profile="production",
            commands=_commands(),
            policy=AutowareSmokePolicy(profile="production"),
        ),
    )
    assert smoke.status == "PASS"
    assert smoke.admission_label == "production-admissible"
    plan_path = tmp_path / "promotion.yaml"
    smoke_path = tmp_path / "smoke.yaml"
    plan.save(plan_path)
    smoke.save(smoke_path)
    applied = apply_autoware_promotion(plan_path, smoke_artifact=smoke_path)
    assert applied.application.status == "applied"
    assert applied.admission_label == "production-admissible"
    assert applied.application.smoke_artifact_sha256 == smoke.artifact_sha256
    assert applied.application.output_manifest_sha256
    rolled_back = rollback_autoware_promotion(plan_path)
    assert rolled_back.application.status == "rolled_back"
    assert roots.workspace_root.exists()


def test_production_apply_requires_smoke_and_mismatches_are_blocked(tmp_path: Path) -> None:
    _, plan = _production_plan(tmp_path)
    with pytest.raises(AutowarePromotionError, match="smoke artifact"):
        apply_autoware_promotion(plan)
    smoke = run_autoware_smoke(
        plan,
        config=AutowareSmokeConfig(
            profile="production",
            commands=_commands(),
            policy=AutowareSmokePolicy(profile="production"),
        ),
    )
    changed = plan.model_copy(update={"promotion_id": "different"}).with_artifact_digest()
    with pytest.raises((AutowareSmokeError, AutowarePromotionError), match="promotion"):
        apply_autoware_promotion(changed, smoke_artifact=smoke)


def test_timeout_and_failed_stage_are_not_admissible(tmp_path: Path) -> None:
    _, plan = _production_plan(tmp_path)
    commands = _commands()
    commands[0] = AutowareSmokeCommand(
        stage="xacro_urdf",
        argv=[sys.executable, "-c", "import time; time.sleep(1)"],
    )
    smoke = run_autoware_smoke(
        plan,
        config=AutowareSmokeConfig(
            profile="production",
            commands=commands,
            policy=AutowareSmokePolicy(profile="production", stage_timeout_seconds=0.01),
        ),
    )
    assert smoke.status == "FAIL"
    assert smoke.admission_label == "not-admissible"
    assert smoke.stages[0].status in {"TIMEOUT", "FAIL"}


def test_precomputed_import_and_schema_are_valid(tmp_path: Path) -> None:
    _, plan = _production_plan(tmp_path)
    live = run_autoware_smoke(
        plan,
        config=AutowareSmokeConfig(
            profile="production",
            commands=_commands(),
            policy=AutowareSmokePolicy(profile="production"),
        ),
    )
    precomputed = live.model_copy(update={"mode": "precomputed"}).with_artifact_digest()
    path = tmp_path / "precomputed.yaml"
    precomputed.save(path)
    imported = import_autoware_smoke(path, promotion=plan)
    assert imported.artifact_sha256 == precomputed.artifact_sha256
    jsonschema.validate(
        imported.model_dump(mode="json", exclude_none=False), autoware_smoke_json_schema()
    )


def test_developer_smoke_is_explicitly_not_admissible(tmp_path: Path) -> None:
    _, plan = _production_plan(tmp_path)
    developer_plan = plan.model_copy(
        update={
            "policy": AutowarePromotionPolicy(profile="developer"),
        }
    ).with_artifact_digest()
    smoke = run_autoware_smoke(
        developer_plan,
        config=AutowareSmokeConfig(
            profile="developer",
            commands=_commands(),
            policy=AutowareSmokePolicy(profile="developer"),
        ),
    )
    assert smoke.status == "PASS"
    assert smoke.admission_label == "developer-only"
    applied = apply_autoware_promotion(developer_plan, smoke_artifact=smoke)
    assert applied.application.status == "applied"
