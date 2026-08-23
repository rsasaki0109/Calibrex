from __future__ import annotations

from pathlib import Path

from calibrex.cli.main import main
from calibrex.core.config import CalibrationConfig, load_config
from calibrex.core.koide_readiness import (
    KoideReadinessConfig,
    evaluate_koide_readiness,
    evaluate_koide_readiness_from_config,
    load_koide_readiness,
    validate_koide_readiness_for_execution,
)
from calibrex.core.koide_runner import KoideRunnerConfig, run_koide_workflow
from calibrex.core.provenance import sha256_path
from calibrex.data.inspect import DatasetInspection, inspect_dataset

CONFIG = Path("examples/public_datasets/kitti_lidar_camera_evidence/config.yaml")


def _ready_fixture() -> tuple[CalibrationConfig, DatasetInspection]:
    config = load_config(CONFIG)
    inspection = inspect_dataset(config.dataset)
    diagnostics = dict(inspection.diagnostics)
    lidar = dict(diagnostics["velodyne_points"])
    lidar.update(
        {
            "sampled_point_count": 2_000,
            "sampled_frame_count": 4,
            "sampled_nonfinite_xyz_count": 0,
            "sampled_nonfinite_intensity_count": 0,
            "has_intensity": True,
            "fov_azimuth_deg": 120.0,
            "fov_elevation_deg": 25.0,
            "planarity_voxel_count": 12,
        }
    )
    diagnostics["velodyne_points"] = lidar
    diagnostics.update(
        {
            "image_texture_score": 0.2,
            "duration_sec": 8.0,
            "static_start_duration_s": 1.0,
            "path_length_m": 2.0,
            "moving_object_risk": 0.1,
        }
    )
    inspection = inspection.__class__(
        dataset_type=inspection.dataset_type,
        path=inspection.path,
        exists=inspection.exists,
        manifest=inspection.manifest,
        streams=inspection.streams,
        warnings=[],
        diagnostics=diagnostics,
    )
    return config, inspection


def test_ready_fixture_has_no_false_ready_without_evidence() -> None:
    config, inspection = _ready_fixture()
    artifact = evaluate_koide_readiness(
        config,
        inspection,
        config_path=CONFIG,
        readiness=KoideReadinessConfig(
            profile="commercial",
            profile_declared=True,
            hardware_profile="velodyne",
            hardware_profile_declared=True,
        ),
    )
    assert artifact.status == "ready"
    assert all(check.status == "pass" for check in artifact.checks)
    assert artifact.provenance.config_sha256
    assert artifact.provenance.dataset_sha256
    assert artifact.provenance.manifest_sha256
    assert artifact.provenance.artifact_sha256
    assert all(check.observed_value is not None for check in artifact.checks)


def test_missing_intrinsics_blocks_even_when_streams_exist() -> None:
    config = load_config(CONFIG)
    payload = config.model_dump(mode="python")
    payload["sensors"]["camera0"]["intrinsics"] = {}
    config = CalibrationConfig.model_validate(payload)
    inspection = inspect_dataset(config.dataset)
    artifact = evaluate_koide_readiness(config, inspection, config_path=CONFIG)
    checks = {check.name: check for check in artifact.checks}
    assert artifact.status == "blocked"
    assert checks["camera_intrinsics"].status == "blocked"


def test_low_texture_and_density_are_blocked() -> None:
    config, inspection = _ready_fixture()
    diagnostics = dict(inspection.diagnostics)
    lidar = dict(diagnostics["velodyne_points"])
    lidar["sampled_point_count"] = 1
    diagnostics["velodyne_points"] = lidar
    diagnostics["image_texture_score"] = 0.0
    inspection = inspection.__class__(
        dataset_type=inspection.dataset_type,
        path=inspection.path,
        exists=inspection.exists,
        manifest=inspection.manifest,
        streams=inspection.streams,
        warnings=inspection.warnings,
        diagnostics=diagnostics,
    )
    artifact = evaluate_koide_readiness(config, inspection, config_path=CONFIG)
    checks = {check.name: check for check in artifact.checks}
    assert checks["point_density"].status == "blocked"
    assert checks["image_texture"].status == "blocked"


def test_missing_timestamp_evidence_is_unknown_and_never_ready() -> None:
    config, inspection = _ready_fixture()
    diagnostics = dict(inspection.diagnostics)
    diagnostics["timestamp_alignment"] = {}
    inspection = inspection.__class__(
        dataset_type=inspection.dataset_type,
        path=inspection.path,
        exists=inspection.exists,
        manifest=inspection.manifest,
        streams=inspection.streams,
        warnings=inspection.warnings,
        diagnostics=diagnostics,
    )
    artifact = evaluate_koide_readiness(
        config,
        inspection,
        config_path=CONFIG,
        readiness=KoideReadinessConfig(
            profile="commercial",
            profile_declared=True,
            hardware_profile="velodyne",
            hardware_profile_declared=True,
        ),
    )
    checks = {check.name: check for check in artifact.checks}
    assert checks["timestamp_overlap"].status == "unknown"
    assert checks["timestamp_synchronization"].status == "unknown"
    assert artifact.status == "warn"


def test_declared_hardware_profile_exposes_profile_guidance() -> None:
    config, inspection = _ready_fixture()
    artifact = evaluate_koide_readiness(
        config,
        inspection,
        config_path=CONFIG,
        readiness=KoideReadinessConfig(
            profile="commercial",
            profile_declared=True,
            hardware_profile="livox",
            hardware_profile_declared=True,
        ),
    )
    assert artifact.hardware_profile == "livox"
    assert artifact.hardware_profile_declared is True
    assert artifact.profile_guidance.minimum_static_start_duration_s == 1.0
    assert any(
        "point-time" in instruction for instruction in artifact.profile_guidance.instructions
    )


def test_artifact_save_load_and_digest_mismatch(tmp_path: Path) -> None:
    artifact = evaluate_koide_readiness_from_config(CONFIG)
    output = tmp_path / "readiness.json"
    artifact.save(output)
    loaded = load_koide_readiness(output)
    validate_koide_readiness_for_execution(output, strict=False)
    assert loaded.provenance.artifact_sha256
    file_digest = sha256_path(output)
    assert file_digest is not None
    output.write_text(output.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    try:
        validate_koide_readiness_for_execution(
            output,
            expected_sha256=file_digest,
            strict=False,
        )
    except ValueError as exc:
        assert "mismatch" in str(exc)
    else:
        raise AssertionError("changed readiness artifact must fail digest validation")


def test_doctor_koide_one_command_writes_schema_valid_artifact(tmp_path: Path) -> None:
    output = tmp_path / "koide-readiness.json"
    exit_code = main(
        [
            "doctor",
            "--workflow",
            "koide",
            "--config",
            str(CONFIG),
            "--output",
            str(output),
            "--json",
        ]
    )
    assert exit_code == 1  # bundled fixture is intentionally too small to be ready
    artifact = load_koide_readiness(output)
    assert artifact.status == "blocked"
    assert artifact.provenance.command[2:4] == ["--workflow", "koide"]


def test_strict_runner_refuses_blocked_readiness_before_external_stage(tmp_path: Path) -> None:
    readiness_path = tmp_path / "readiness.json"
    evaluate_koide_readiness_from_config(CONFIG).save(readiness_path)
    runner_config = KoideRunnerConfig(
        execution_mode="precomputed",
        result_path=tmp_path / "missing-calib.json",
        readiness_artifact_path=readiness_path,
        strict_readiness=True,
    )
    workflow = run_koide_workflow(runner_config)
    assert workflow.artifact.status == "unavailable"
    assert workflow.artifact.execution.attempted is False
    assert any("readiness" in warning.lower() for warning in workflow.artifact.warnings)
