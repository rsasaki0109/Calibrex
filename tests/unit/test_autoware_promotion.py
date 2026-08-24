"""Package-scoped Autoware promotion contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest

import calibrex.export.autoware_promotion as promotion_module
from calibrex.cli.main import main
from calibrex.core.result import (
    CalibrationResult,
    FrameGraphSnapshot,
    QualitySummary,
    RunInfo,
    TransformQuality,
    TransformResult,
    build_result_provenance,
)
from calibrex.export.autoware_promotion import (
    AutowarePromotionError,
    AutowarePromotionPolicy,
    AutowarePromotionRoots,
    apply_autoware_promotion,
    build_autoware_promotion_plan,
    inspect_autoware_package,
    rollback_autoware_promotion,
)


def _package(tmp_path: Path) -> tuple[AutowarePromotionRoots, Path, Path, Path]:
    workspace = tmp_path / "autoware"
    source = workspace / "src" / "vehicle_sensor_kit_description"
    individual = workspace / "src" / "individual_params"
    source_config = source / "config"
    individual_config = individual / "config" / "vehicle_a" / "sensor_kit_a"
    urdf = source / "urdf"
    source_config.mkdir(parents=True)
    individual_config.mkdir(parents=True)
    urdf.mkdir(parents=True)
    (source_config / "sensor_kit_calibration.yaml").write_text(
        "sensor_kit_base_link:\n"
        "  lidar0_link:\n"
        "    x: 0.0\n"
        "    y: 0.0\n"
        "    z: 1.0\n"
        "    roll: 0.0\n"
        "    pitch: 0.0\n"
        "    yaw: 0.0\n",
        encoding="utf-8",
    )
    sensors_calibration = source_config / "sensors_calibration.yaml"
    sensors_calibration.write_text(
        "base_link:\n"
        "  sensor_kit_base_link:\n"
        "    x: 1.0\n"
        "    y: 0.0\n"
        "    z: 1.0\n"
        "    roll: 0.0\n"
        "    pitch: 0.0\n"
        "    yaw: 0.0\n",
        encoding="utf-8",
    )
    (individual_config / "sensor_kit_calibration.yaml").write_text(
        (source_config / "sensor_kit_calibration.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (urdf / "sensors.xacro").write_text(
        '<robot xmlns:xacro="http://ros.org/wiki/xacro">\n'
        '  <link name="base_link"/>\n'
        '  <link name="sensor_kit_base_link"/>\n'
        '  <link name="lidar0_link"/>\n'
        '  <joint name="kit_joint" type="fixed">\n'
        '    <parent link="base_link"/><child link="sensor_kit_base_link"/>\n'
        '    <origin xyz="1 0 1" rpy="0 0 0"/>\n'
        "  </joint>\n"
        '  <joint name="lidar_joint" type="fixed">\n'
        '    <parent link="sensor_kit_base_link"/><child link="lidar0_link"/>\n'
        '    <origin xyz="0 0 1" rpy="0 0 0"/>\n'
        "  </joint>\n"
        "</robot>\n",
        encoding="utf-8",
    )
    roots = AutowarePromotionRoots(
        workspace_root=workspace,
        package_root=workspace / "src",
        individual_params_root=individual,
        sensor_kit_description_root=source,
    )
    return (
        roots,
        source_config / "sensor_kit_calibration.yaml",
        sensors_calibration,
        individual_config / "sensor_kit_calibration.yaml",
    )


def _candidate(tmp_path: Path, *, grade: str = "pass") -> Path:
    result = CalibrationResult(
        run=RunInfo(
            id="run-1",
            slac_version="test",
            provenance=build_result_provenance(
                producer="calibrex",
                tool_name="calibrex.autoware-promotion-fixture",
                tool_version="test",
                command=["pytest", "tests/unit/test_autoware_promotion.py"],
                extra={
                    "fixture": "synthetic Autoware promotion candidate",
                    "provenance_note": (
                        "unit-test candidate has no external configuration or input "
                        "artifact; the builder records those unavailable digests explicitly"
                    ),
                },
            ),
        ),
        frame_graph=FrameGraphSnapshot(
            root="base_link",
            frames={
                "base_link": None,
                "sensor_kit_base_link": "base_link",
                "lidar0_link": "sensor_kit_base_link",
            },
        ),
        transforms={
            "T_sensor_kit_base_link_lidar0_link": TransformResult(
                parent="sensor_kit_base_link",
                child="lidar0_link",
                translation_m=[0.2, 0.0, 1.1],
                rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
                quality=TransformQuality(grade=grade),
            )
        },
        quality=QualitySummary(grade=grade),
    )
    path = tmp_path / f"candidate-{grade}.yaml"
    result.save(path)
    return path


def test_valid_plan_is_read_only_and_deterministic(tmp_path: Path) -> None:
    roots, calibration, _, individual = _package(tmp_path)
    candidate = _candidate(tmp_path)
    before = {path: path.read_bytes() for path in (calibration, individual)}
    policy = AutowarePromotionPolicy(
        required_files=[
            roots.relative(calibration),
            roots.relative(individual),
        ],
    )
    first = build_autoware_promotion_plan(
        candidate,
        roots=roots,
        vehicle_id="vehicle_a",
        sensor_kit_id="sensor_kit_a",
        policy=policy,
    )
    second = build_autoware_promotion_plan(
        candidate,
        roots=roots,
        vehicle_id="vehicle_a",
        sensor_kit_id="sensor_kit_a",
        policy=policy,
    )
    assert first.status == "PASS"
    assert first.decision == "ADOPT"
    assert first.generated_patches[0].unified_diff == second.generated_patches[0].unified_diff
    assert {path: path.read_bytes() for path in (calibration, individual)} == before


def test_apply_and_rollback_restore_exact_bytes(tmp_path: Path) -> None:
    roots, calibration, _, _ = _package(tmp_path)
    candidate = _candidate(tmp_path)
    plan_path = tmp_path / "promotion.yaml"
    artifact = build_autoware_promotion_plan(
        candidate,
        roots=roots,
        vehicle_id="vehicle_a",
        sensor_kit_id="sensor_kit_a",
        policy=AutowarePromotionPolicy(profile="developer"),
    )
    artifact.save(plan_path)
    original = calibration.read_bytes()
    applied = apply_autoware_promotion(plan_path)
    assert applied.application.status == "applied"
    assert calibration.read_bytes() != original
    rolled_back = rollback_autoware_promotion(plan_path)
    assert rolled_back.application.status == "rolled_back"
    assert calibration.read_bytes() == original


def test_stale_baseline_and_path_escape_are_blocked(tmp_path: Path) -> None:
    roots, calibration, _, _ = _package(tmp_path)
    candidate = _candidate(tmp_path)
    inspection = inspect_autoware_package(
        roots,
        vehicle_id="vehicle_a",
        sensor_kit_id="sensor_kit_a",
    )
    calibration.write_text(calibration.read_text(encoding="utf-8") + "# edit\n", encoding="utf-8")
    stale = build_autoware_promotion_plan(
        candidate,
        roots=roots,
        vehicle_id="vehicle_a",
        sensor_kit_id="sensor_kit_a",
        baseline_manifest_sha256=inspection.baseline_manifest_sha256,
    )
    assert stale.status == "BLOCKED"
    with pytest.raises(ValueError, match="within workspace_root"):
        AutowarePromotionRoots(
            workspace_root=tmp_path / "autoware",
            package_root=tmp_path,
        )


def test_non_admissible_result_does_not_promote(tmp_path: Path) -> None:
    roots, _, _, _ = _package(tmp_path)
    candidate = _candidate(tmp_path, grade="warn")
    artifact = build_autoware_promotion_plan(
        candidate,
        roots=roots,
        vehicle_id="vehicle_a",
        sensor_kit_id="sensor_kit_a",
    )
    assert artifact.status == "FAIL"
    assert artifact.decision == "DO_NOT_ADOPT"
    with pytest.raises(AutowarePromotionError, match="not admissible"):
        apply_autoware_promotion(artifact)


@pytest.mark.parametrize("failure", ["duplicate", "cycle", "disconnected", "missing"])
def test_package_frame_gate_reports_structural_failures(tmp_path: Path, failure: str) -> None:
    roots, _, sensors_calibration, individual = _package(tmp_path)
    if failure == "duplicate":
        individual.write_text(
            individual.read_text(encoding="utf-8").replace("z: 1.0", "z: 2.0"),
            encoding="utf-8",
        )
    elif failure == "cycle":
        sensors_calibration.write_text(
            sensors_calibration.read_text(encoding="utf-8")
            + "sensor_kit_base_link:\n"
            + "  base_link:\n"
            + "    x: 0.0\n    y: 0.0\n    z: 0.0\n"
            + "    roll: 0.0\n    pitch: 0.0\n    yaw: 0.0\n",
            encoding="utf-8",
        )
    elif failure == "disconnected":
        sensors_calibration.write_text(
            sensors_calibration.read_text(encoding="utf-8")
            + "orphan_parent:\n"
            + "  orphan_sensor:\n"
            + "    x: 0.0\n    y: 0.0\n    z: 0.0\n"
            + "    roll: 0.0\n    pitch: 0.0\n    yaw: 0.0\n",
            encoding="utf-8",
        )
    elif failure == "missing":
        sensors_calibration.write_text(
            sensors_calibration.read_text(encoding="utf-8").replace(
                "base_link:\n", "vehicle_origin:\n"
            ),
            encoding="utf-8",
        )
    inspection = inspect_autoware_package(
        roots,
        vehicle_id="vehicle_a",
        sensor_kit_id="sensor_kit_a",
    )
    codes = {item.code for item in inspection.diagnostics}
    expected = {
        "duplicate": "duplicate_child",
        "cycle": "frame_cycle",
        "disconnected": "disconnected_sensor",
        "missing": "missing_sensor_kit_base_edge",
    }[failure]
    assert expected in codes
    assert inspection.status == "FAIL"


def test_duplicate_yaml_key_and_allowlist_escape_are_rejected(tmp_path: Path) -> None:
    roots, _, _, _ = _package(tmp_path)
    assert roots.sensor_kit_description_root is not None
    bad = roots.sensor_kit_description_root / "config" / "bad_topics.yaml"
    bad.write_text("sensor:\n  topic: /points\n  topic: /other\n", encoding="utf-8")
    inspection = inspect_autoware_package(
        roots,
        vehicle_id="vehicle_a",
        sensor_kit_id="sensor_kit_a",
    )
    assert any(item.code == "invalid_yaml" for item in inspection.diagnostics)
    with pytest.raises(ValueError, match="workspace-relative"):
        AutowarePromotionPolicy(allowed_files=["../outside.yaml"])


def test_windows_style_relative_escape_is_rejected_on_every_host() -> None:
    with pytest.raises(ValueError, match="workspace-relative"):
        AutowarePromotionPolicy(allowed_files=[r"..\outside.yaml"])
    with pytest.raises(ValueError, match="backup_directory"):
        AutowarePromotionPolicy(backup_directory=r"..\backups")


def test_tampered_patch_inventory_is_refused_before_mutation(tmp_path: Path) -> None:
    roots, calibration, _, _ = _package(tmp_path)
    candidate = _candidate(tmp_path)
    artifact = build_autoware_promotion_plan(
        candidate,
        roots=roots,
        vehicle_id="vehicle_a",
        sensor_kit_id="sensor_kit_a",
        policy=AutowarePromotionPolicy(profile="developer"),
    )
    patch = artifact.generated_patches[0].model_copy(
        update={"replacement_text": artifact.generated_patches[0].replacement_text + "# tampered\n"}
    )
    tampered = artifact.model_copy(update={"generated_patches": [patch]}).with_artifact_digest()
    before = calibration.read_bytes()
    with pytest.raises(AutowarePromotionError, match=r"replacement digest|patch inventory"):
        apply_autoware_promotion(tampered)
    assert calibration.read_bytes() == before


def test_failed_apply_retains_backup_when_recovery_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots, _, _, _ = _package(tmp_path)
    candidate = _candidate(tmp_path)
    artifact = build_autoware_promotion_plan(
        candidate,
        roots=roots,
        vehicle_id="vehicle_a",
        sensor_kit_id="sensor_kit_a",
        policy=AutowarePromotionPolicy(profile="developer"),
    )

    def fail_after_apply(*args: object, **kwargs: object) -> None:
        raise AutowarePromotionError("injected post-apply failure")

    def fail_restore(*args: object, **kwargs: object) -> None:
        raise AutowarePromotionError("injected restore failure")

    monkeypatch.setattr(promotion_module, "_verify_applied_outputs", fail_after_apply)
    monkeypatch.setattr(promotion_module, "_restore_one", fail_restore)
    with pytest.raises(AutowarePromotionError, match="backup evidence was retained"):
        apply_autoware_promotion(artifact)
    backup_root = roots.workspace_root / ".calibrex" / "promotions" / artifact.promotion_id
    assert backup_root.is_dir()
    assert list(backup_root.glob("*.bak"))


def test_cli_plan_verify_apply_rollback_hierarchy(tmp_path: Path) -> None:
    roots, calibration, _, _ = _package(tmp_path)
    candidate = _candidate(tmp_path)
    plan = tmp_path / "promotion.yaml"
    arguments = [
        "autoware",
        "promotion",
        "plan",
        str(candidate),
        "--workspace-root",
        str(roots.workspace_root),
        "--package-root",
        str(roots.package_root),
        "--individual-params-root",
        str(roots.individual_params_root),
        "--sensor-kit-description-root",
        str(roots.sensor_kit_description_root),
        "--vehicle-id",
        "vehicle_a",
        "--sensor-kit-id",
        "sensor_kit_a",
        "--profile",
        "developer",
        "--output",
        str(plan),
    ]
    assert main(arguments) == 0
    plan_before_verify = plan.read_bytes()
    assert main(["autoware", "promotion", "verify", str(plan)]) == 0
    assert plan.read_bytes() == plan_before_verify
    verified = tmp_path / "promotion.verified.yaml"
    assert main(["autoware", "promotion", "verify", str(plan), "--output", str(verified)]) == 0
    original = calibration.read_bytes()
    assert main(["autoware", "promotion", "apply", str(verified)]) == 0
    assert calibration.read_bytes() != original
    assert main(["autoware", "promotion", "rollback", str(verified)]) == 0
    assert calibration.read_bytes() == original
