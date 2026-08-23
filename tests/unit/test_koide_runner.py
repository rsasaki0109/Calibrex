from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from calibrex.core.koide_runner import (
    KoideContainerConfig,
    KoideRunnerConfig,
    KoideStageConfig,
    build_container_argv,
    run_koide_workflow,
)


def _native_result(path: Path) -> None:
    path.write_text(
        '{"results": {"T_lidar_camera": [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]}}',
        encoding="utf-8",
    )


def _config(tmp_path: Path, result_path: Path, *, mode: str = "subprocess") -> KoideRunnerConfig:
    stages = [
        KoideStageConfig(name="preprocess", argv=["fake-preprocess", "{dataset_path}"]),
        KoideStageConfig(name="initial_guess", argv=["fake-initial", "{initial_guess_path}"]),
        KoideStageConfig(name="calibrate", argv=["fake-calibrate", "{result_path}"]),
    ]
    kwargs: dict[str, object] = {
        "execution_mode": mode,
        "stages": stages,
        "dataset_path": tmp_path / "capture.mcap",
        "result_path": result_path,
        "input_paths": [tmp_path / "capture.mcap"],
        "lidar_frame": "lidar_front",
        "camera_frame": "camera_front",
        "tool_version": "fixture",
        "source_commit": "a" * 40,
        "training_isolation_declared": True,
    }
    return KoideRunnerConfig.model_validate(kwargs)


def test_multistage_runner_records_each_stage_and_digests(tmp_path: Path) -> None:
    result_path = tmp_path / "calib.json"
    input_path = tmp_path / "capture.mcap"
    input_path.write_bytes(b"fixture capture")
    config = _config(tmp_path, result_path)
    calls: list[list[str]] = []

    def fake_runner(
        args: list[str],
        *,
        capture_output: bool,
        text: bool,
        cwd: str | Path | None,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        del capture_output, text, cwd, timeout
        calls.append(args)
        if args[0] == "fake-calibrate":
            _native_result(result_path)
        return subprocess.CompletedProcess(args, 0, "stage ok", "")

    run = run_koide_workflow(config, process_runner=fake_runner)

    assert run.artifact.status == "success"
    assert [stage.name for stage in run.artifact.execution.stages] == [
        "preprocess",
        "initial_guess",
        "calibrate",
    ]
    assert all(stage.return_code == 0 for stage in run.artifact.execution.stages)
    assert all(stage.stdout_sha256 for stage in run.artifact.execution.stages)
    assert run.artifact.parsed_outputs.transforms["T_lidar_front_camera_front"]
    assert run.artifact.digests.input_sha256
    assert run.artifact.digests.config_sha256
    assert run.artifact.digests.output_sha256
    assert len(calls) == 3
    assert all(isinstance(argument, str) for call in calls for argument in call)


def test_stage_failure_short_circuits_following_stages(tmp_path: Path) -> None:
    result_path = tmp_path / "calib.json"
    config = _config(tmp_path, result_path)
    calls: list[str] = []

    def fake_runner(
        args: list[str],
        *,
        capture_output: bool,
        text: bool,
        cwd: str | Path | None,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        del capture_output, text, cwd, timeout
        calls.append(args[0])
        return subprocess.CompletedProcess(args, 9 if args[0] == "fake-initial" else 0, "", "bad")

    run = run_koide_workflow(config, process_runner=fake_runner)

    assert run.artifact.status == "failed"
    assert calls == ["fake-preprocess", "fake-initial"]
    assert [stage.status for stage in run.stages] == ["success", "failed", "skipped"]
    assert run.stages[-1].error == "prior stage failed"


def test_zero_exit_with_malformed_native_output_is_invalid_output(tmp_path: Path) -> None:
    result_path = tmp_path / "calib.json"
    result_path.write_text('{"results": []}', encoding="utf-8")
    config = _config(tmp_path, result_path)

    def fake_runner(
        args: list[str],
        *,
        capture_output: bool,
        text: bool,
        cwd: str | Path | None,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        del capture_output, text, cwd, timeout
        return subprocess.CompletedProcess(args, 0, "done", "")

    run = run_koide_workflow(config, process_runner=fake_runner)
    assert run.artifact.status == "invalid_output"
    assert run.artifact.execution.return_code == 0
    assert any("parse failed" in warning for warning in run.artifact.warnings)


def test_commercial_profile_rejects_automatic_initial_guess_and_superglue() -> None:
    with pytest.raises(ValidationError, match="automatic initial guess"):
        KoideRunnerConfig.model_validate(
            {
                "execution_mode": "precomputed",
                "profile": "commercial",
                "result_path": "calib.json",
                "automatic_initial_guess": True,
            }
        )

    with pytest.raises(ValueError, match="automatic initial guess"):
        KoideRunnerConfig.from_options(
            {
                "result_path": "calib.json",
                "provider": "superglue-provider",
            }
        )
    with pytest.raises(ValidationError, match="automatic initial guess"):
        KoideRunnerConfig.model_validate(
            {
                "execution_mode": "precomputed",
                "profile": "commercial",
                "result_path": "calib.json",
                "provider": "SuperGlue",
            }
        )


def test_research_profile_requires_provider_checkpoint_and_license() -> None:
    with pytest.raises(ValidationError, match="research-noncommercial"):
        KoideRunnerConfig.model_validate(
            {
                "execution_mode": "precomputed",
                "profile": "research-noncommercial",
                "result_path": "calib.json",
            }
        )


def test_research_checkpoint_digest_is_verified_before_execution(tmp_path: Path) -> None:
    checkpoint = tmp_path / "superglue-checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint bytes")
    config = KoideRunnerConfig.model_validate(
        {
            "execution_mode": "precomputed",
            "profile": "research-noncommercial",
            "result_path": tmp_path / "calib.json",
            "provider": "research-provider",
            "provider_license_spdx": "MIT",
            "checkpoint_path": checkpoint,
            "checkpoint_sha256": "0" * 64,
        }
    )

    run = run_koide_workflow(config)

    assert run.artifact.status == "unavailable"
    assert not run.artifact.execution.attempted
    assert any("checkpoint SHA-256 mismatch" in warning for warning in run.artifact.warnings)


def test_container_argv_is_digest_pinned_and_mounts_are_bounded(tmp_path: Path) -> None:
    config = KoideRunnerConfig(
        execution_mode="container",
        stages=[KoideStageConfig(name="calibrate", argv=["koide", "{result_path}"])],
        result_path=tmp_path / "out" / "calib.json",
        container=KoideContainerConfig(
            image_digest="ghcr.io/example/koide@sha256:" + "a" * 64,
            input_dir=tmp_path / "input",
            output_dir=tmp_path / "out",
            cpus=2.0,
            memory="4g",
        ),
    )
    argv = build_container_argv(config, config.stages[0])
    assert argv[:6] == ["docker", "run", "--rm", "--network", "none", "--read-only"]
    assert "--cap-drop=ALL" in argv
    assert "--security-opt=no-new-privileges" in argv
    assert "--cpus" in argv and "2" in argv
    assert "--memory" in argv and "4g" in argv
    assert any("dst=/calibrex/input,readonly" in item for item in argv)
    assert any("dst=/calibrex/output,rw" in item for item in argv)
    assert argv[-2:] == ["koide", "/calibrex/output/calib.json"]

    with pytest.raises(ValidationError, match="immutable"):
        KoideContainerConfig(
            image="ghcr.io/example/koide:latest",
            input_dir=tmp_path,
            output_dir=tmp_path,
        )


def test_timeout_is_recorded_with_stage_tail_and_digest(tmp_path: Path) -> None:
    config = _config(tmp_path, tmp_path / "calib.json")

    def fake_runner(
        args: list[str],
        *,
        capture_output: bool,
        text: bool,
        cwd: str | Path | None,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        del capture_output, text, cwd, timeout
        raise subprocess.TimeoutExpired(args, 0.01, output=b"out", stderr=b"timeout")

    run = run_koide_workflow(config, process_runner=fake_runner)
    assert run.artifact.status == "timeout"
    assert run.artifact.execution.timed_out is True
    assert run.artifact.execution.stages[0].timed_out is True
    assert run.artifact.execution.stages[0].stdout_sha256
    assert run.artifact.execution.stages[0].stderr_tail == "timeout"


def test_cli_run_koide_materializes_external_run_from_typed_config(tmp_path: Path) -> None:
    from calibrex.cli.main import main

    result_path = tmp_path / "calib.json"
    _native_result(result_path)
    config_path = tmp_path / "koide.yaml"
    config_path.write_text(
        f"""
schema_version: slac.config/v0.1
dataset:
  type: filesystem
  path: {tmp_path.as_posix()}
sensors:
  camera0:
    type: camera
  lidar0:
    type: lidar
frames:
  base:
    root: true
  camera0:
    parent: base
  lidar0:
    parent: base
pipeline:
  factors:
    koide_lidar_camera:
      enabled: true
      options:
        execution_mode: precomputed
        result_path: {result_path.as_posix()}
        lidar_frame: lidar0
        camera_frame: camera0
""".strip(),
        encoding="utf-8",
    )
    output_path = tmp_path / "external-run.yaml"
    assert main(
        [
            "external-run",
            "run-koide",
            str(config_path),
            "--output",
            str(output_path),
            "--json",
        ]
    ) == 0
    assert output_path.is_file()
