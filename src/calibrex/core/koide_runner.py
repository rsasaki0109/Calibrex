"""Safe, typed execution boundary for Koide's external workflow.

The upstream ``direct_visual_lidar_calibration`` project is intentionally not
imported here.  This module only builds argv vectors, executes an injected
process runner, and parses the small native ``calib.json`` result through the
ROS-independent importer.  It is therefore suitable for a commercial
Calibrex installation while keeping ROS, CUDA, GPL code, and Koide's source
outside ``src/calibrex``.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from pydantic import Field, field_validator, model_validator

from calibrex.core.external_run import (
    ExternalArtifactDigest,
    ExternalCalibrationRunArtifact,
    ExternalDataIsolation,
    ExternalExecution,
    ExternalParsedOutputs,
    ExternalRunDigests,
    ExternalRunProvenance,
    ExternalRunStatus,
    ExternalStageExecution,
    ExternalToolIdentity,
    ExternalTransformOutput,
)
from calibrex.core.geometry import SE3
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.result import StrictModel

if TYPE_CHECKING:
    from calibrex.importers.koide import KoideNativeCalibration

KOIDE_SOURCE_REPOSITORY = "https://github.com/koide3/direct_visual_lidar_calibration"

KoideExecutionMode = Literal["subprocess", "container", "precomputed"]
KoideProfile = Literal["commercial", "research-noncommercial"]
KoideStageName = Literal["preprocess", "initial_guess", "calibrate"]
KoideInitialGuessMode = Literal["manual", "precomputed", "automatic"]

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_DIGEST_RE = re.compile(r"^(?:[^@\s]+@)?sha256:[0-9a-f]{64}$")
_PATH_PLACEHOLDER_NAMES = {
    "dataset_path",
    "result_path",
    "output_dir",
    "input_dir",
    "config_path",
    "initial_guess_path",
    "camera_streams",
    "lidar_streams",
}


class KoideProcessRunner(Protocol):
    """Callable protocol used by :class:`KoideExternalRunner`.

    Tests can provide a fake executable without installing Docker, ROS, or
    Koide.  The default is :func:`subprocess.run`, always invoked with argv and
    never with ``shell=True``.
    """

    def __call__(
        self,
        args: Sequence[str],
        *,
        capture_output: bool,
        text: bool,
        cwd: str | Path | None,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]: ...


class KoideStageConfig(StrictModel):
    """One argv-only stage in the official preprocess/initialise/calibrate flow."""

    name: KoideStageName
    argv: list[str] = Field(min_length=1)
    enabled: bool = True
    timeout_seconds: float | None = Field(default=None, gt=0.0)

    @field_validator("argv")
    @classmethod
    def reject_nul_bytes(cls, value: list[str]) -> list[str]:
        if any("\x00" in argument for argument in value):
            raise ValueError("Koide stage argv must not contain NUL bytes")
        return value


class KoideContainerConfig(StrictModel):
    """Container isolation controls for a Koide stage."""

    engine: Literal["docker", "podman"] = "docker"
    image: str | None = None
    image_digest: str | None = None
    input_dir: Path
    output_dir: Path
    network_mode: Literal["none"] = "none"
    cpus: float | None = Field(default=None, gt=0.0)
    memory: str | None = None
    workdir: str = "/calibrex/output"

    @model_validator(mode="after")
    def validate_immutable_image(self) -> KoideContainerConfig:
        image = self.resolved_image
        if not _IMAGE_DIGEST_RE.fullmatch(image):
            raise ValueError(
                "Koide container image must be immutable (repository@sha256:<64 hex> "
                "or sha256:<64 hex>)"
            )
        if self.network_mode != "none":
            raise ValueError("Koide containers must use network_mode='none'")
        for path_name, path in (("input_dir", self.input_dir), ("output_dir", self.output_dir)):
            _validate_mount_path(path, path_name)
        return self

    @property
    def resolved_image(self) -> str:
        """Return the explicit immutable image reference."""

        value = self.image_digest or self.image
        if value is None:
            return ""
        return value.strip()


class KoideRunnerConfig(StrictModel):
    """Public, schema-validatable configuration for a Koide workflow.

    ``FactorConfig.options`` remains an open compatibility map.  Callers that
    want typed validation can use this model directly or
    :func:`koide_runner_config_from_options`; the Calibrex pipeline invokes the
    same validation for the Koide factor without changing other factor APIs.
    """

    execution_mode: KoideExecutionMode = "subprocess"
    profile: KoideProfile = "commercial"
    stages: list[KoideStageConfig] = Field(default_factory=list)
    result_path: Path | None = None
    dataset_path: Path | None = None
    input_paths: list[Path] = Field(default_factory=list)
    config_path: Path | None = None
    working_dir: Path | None = None
    timeout_seconds: float = Field(default=600.0, gt=0.0)
    tool_name: str = "direct_visual_lidar_calibration"
    tool_version: str | None = None
    source_repository: str = KOIDE_SOURCE_REPOSITORY
    source_commit: str | None = None
    license_spdx: str | None = "MIT"
    source_path: Path | None = None
    tool_path: Path | None = None
    camera_frame: str | None = None
    lidar_frame: str | None = None
    camera_streams: list[str] = Field(default_factory=list)
    lidar_streams: list[str] = Field(default_factory=list)
    initial_guess_mode: KoideInitialGuessMode = "manual"
    manual_initial_guess_path: Path | None = None
    precomputed_initial_guess_path: Path | None = None
    automatic_initial_guess: bool = False
    provider: str | None = None
    provider_license_spdx: str | None = None
    checkpoint_path: Path | None = None
    checkpoint_sha256: str | None = None
    container: KoideContainerConfig | None = None
    training_isolation_declared: bool = False
    training_isolation_evidence: str | None = None
    training_data_ids_sha256: str | None = None
    holdout_data_ids_sha256: str | None = None
    readiness_artifact_path: Path | None = None
    readiness_artifact_sha256: str | None = None
    strict_readiness: bool = False

    @field_validator("source_commit")
    @classmethod
    def normalize_source_commit(cls, value: str | None) -> str | None:
        return value.strip() if value is not None and value.strip() else None

    @field_validator(
        "checkpoint_sha256",
        "training_data_ids_sha256",
        "holdout_data_ids_sha256",
        "readiness_artifact_sha256",
    )
    @classmethod
    def validate_hex_digest(cls, value: str | None) -> str | None:
        if value is not None and not _DIGEST_RE.fullmatch(value.lower()):
            raise ValueError("digest must be a lowercase 64-character SHA-256")
        return value.lower() if value is not None else None

    @model_validator(mode="after")
    def enforce_profile_boundary(self) -> KoideRunnerConfig:
        # Do not search the serialized model for a field name here.  The
        # normal ``initial_guess`` stage and the compatibility field
        # ``automatic_initial_guess`` are both present in a serialized model
        # even when the former is a manual stage and the latter is ``False``.
        # Treat only an explicit opt-in (or a sensitive provider/value) as an
        # automatic initial-guess request.
        automatic = _uses_automatic_initial_guess(self)
        if self.profile == "commercial" and automatic:
            raise ValueError(
                "commercial Koide profile rejects automatic initial guess and SuperGlue; "
                "use initial_guess_mode='manual'/'precomputed' or explicitly opt into "
                "profile='research-noncommercial'"
            )
        if self.profile == "research-noncommercial":
            missing: list[str] = []
            if not self.provider:
                missing.append("provider")
            if not self.provider_license_spdx:
                missing.append("provider_license_spdx")
            if self.checkpoint_path is None:
                missing.append("checkpoint_path")
            if self.checkpoint_sha256 is None:
                missing.append("checkpoint_sha256")
            if missing:
                raise ValueError(
                    "research-noncommercial Koide profile requires provider, license, "
                    "checkpoint_path, and checkpoint_sha256 (missing: " + ", ".join(missing) + ")"
                )
        if self.execution_mode == "container" and self.container is None:
            raise ValueError("container execution requires a typed container configuration")
        if self.execution_mode != "container" and self.container is not None:
            raise ValueError("container configuration requires execution_mode='container'")
        if self.execution_mode == "precomputed":
            if self.result_path is None and self.stages:
                raise ValueError("precomputed Koide execution requires result_path")
            if self.stages:
                raise ValueError("precomputed Koide execution must not define executable stages")
        elif not self.stages and self.result_path is not None:
            raise ValueError(
                "subprocess/container Koide execution requires at least one stage; "
                "use precomputed mode for an existing calib.json"
            )
        if self.stages:
            order = {"preprocess": 0, "initial_guess": 1, "calibrate": 2}
            enabled = [stage for stage in self.stages if stage.enabled]
            stage_order = [order[stage.name] for stage in enabled]
            if stage_order != sorted(stage_order):
                raise ValueError(
                    "Koide stages must be ordered preprocess, initial_guess, calibrate"
                )
        return self

    @classmethod
    def from_options(
        cls,
        options: Mapping[str, Any],
        *,
        dataset_path: str | Path | None = None,
        result_path: str | Path | None = None,
        camera_frame: str | None = None,
        lidar_frame: str | None = None,
        camera_streams: Iterable[str] = (),
        lidar_streams: Iterable[str] = (),
    ) -> KoideRunnerConfig:
        """Build a typed config while retaining legacy factor option names."""

        raw_result = result_path or options.get("result_path") or options.get("slac_result")
        raw_dataset = dataset_path or options.get("dataset_path")
        raw_stages = _stages_from_options(options)
        mode_raw = options.get("execution_mode")
        if mode_raw is None:
            mode_raw = "precomputed" if raw_result and not raw_stages else "subprocess"
        container = _container_from_options(
            options,
            input_default=(Path(raw_dataset).parent if raw_dataset is not None else Path(".")),
            output_default=(Path(raw_result).parent if raw_result is not None else Path(".")),
        )
        kwargs: dict[str, Any] = {
            "execution_mode": mode_raw,
            "profile": options.get("profile", "commercial"),
            "stages": raw_stages,
            "result_path": raw_result,
            "dataset_path": raw_dataset,
            "input_paths": options.get("input_paths", options.get("input_artifacts", [])),
            "config_path": options.get("config_path"),
            "working_dir": options.get("working_dir", options.get("cwd")),
            "timeout_seconds": options.get("timeout_seconds", options.get("timeout_sec", 600.0)),
            "tool_name": options.get("tool_name", "direct_visual_lidar_calibration"),
            "tool_version": options.get("tool_version"),
            "source_repository": options.get("source_repository", KOIDE_SOURCE_REPOSITORY),
            "source_commit": options.get("source_commit"),
            "license_spdx": options.get("license_spdx", "MIT"),
            "source_path": options.get("source_path"),
            "tool_path": options.get("tool_path"),
            "camera_frame": options.get("camera_frame", camera_frame),
            "lidar_frame": options.get("lidar_frame", lidar_frame),
            "camera_streams": list(camera_streams),
            "lidar_streams": list(lidar_streams),
            "initial_guess_mode": options.get("initial_guess_mode", "manual"),
            "manual_initial_guess_path": options.get("manual_initial_guess_path"),
            "precomputed_initial_guess_path": options.get("precomputed_initial_guess_path"),
            "automatic_initial_guess": options.get(
                "automatic_initial_guess",
                _unsafe_initial_guess_requested(options),
            ),
            "provider": options.get("provider"),
            "provider_license_spdx": options.get("provider_license_spdx"),
            "checkpoint_path": options.get("checkpoint_path"),
            "checkpoint_sha256": options.get("checkpoint_sha256", options.get("checkpoint_digest")),
            "container": container,
            "training_isolation_declared": options.get("training_isolation_declared", False),
            "training_isolation_evidence": options.get("training_isolation_evidence"),
            "training_data_ids_sha256": options.get("training_data_ids_sha256"),
            "holdout_data_ids_sha256": options.get("holdout_data_ids_sha256"),
            "readiness_artifact_path": options.get("readiness_artifact_path"),
            "readiness_artifact_sha256": options.get("readiness_artifact_sha256"),
            "strict_readiness": options.get("strict_readiness", False),
        }
        return cls.model_validate(kwargs)


class KoideStageResult(StrictModel):
    """Execution result of one stage, including bounded process evidence."""

    name: str
    command: list[str] = Field(default_factory=list)
    status: Literal["success", "failed", "timeout", "unavailable", "skipped"]
    return_code: int | None = None
    timed_out: bool = False
    duration_seconds: float | None = Field(default=None, ge=0.0)
    stdout_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    stderr_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    stdout_tail: str | None = Field(default=None, max_length=4000)
    stderr_tail: str | None = Field(default=None, max_length=4000)
    error: str | None = None

    def as_external(self) -> ExternalStageExecution:
        """Convert to the schema-level stage contract."""

        return ExternalStageExecution(**self.model_dump(mode="python"))


class KoideWorkflowResult(StrictModel):
    """Typed runner result and schema-valid external-run evidence."""

    artifact: ExternalCalibrationRunArtifact
    stages: list[KoideStageResult] = Field(default_factory=list)

    @property
    def transforms(self) -> dict[str, SE3]:
        """Return parsed transforms for legacy solver consumers."""

        return {
            name: output.as_se3()
            for name, output in self.artifact.parsed_outputs.transforms.items()
        }


def koide_runner_config_from_options(
    options: Mapping[str, Any],
    **kwargs: Any,
) -> KoideRunnerConfig:
    """Validate legacy factor options as a first-class Koide config."""

    return KoideRunnerConfig.from_options(options, **kwargs)


def koide_runner_json_schema() -> dict[str, object]:
    """Return the generated JSON Schema for the public runner config."""

    return KoideRunnerConfig.model_json_schema()


def build_container_argv(
    config: KoideRunnerConfig,
    stage: KoideStageConfig,
    *,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> list[str]:
    """Build a safe Docker/Podman argv vector for one stage.

    The input bind is read-only, the output bind is the only writable host
    mount, the root filesystem is read-only, and networking is disabled.
    """

    if config.execution_mode != "container" or config.container is None:
        raise ValueError("build_container_argv requires execution_mode='container'")
    container = config.container
    source_input = Path(input_dir) if input_dir is not None else container.input_dir
    source_output = Path(output_dir) if output_dir is not None else container.output_dir
    _validate_mount_path(source_input, "input_dir")
    _validate_mount_path(source_output, "output_dir")
    command = _expand_stage_argv(
        stage.argv,
        config,
        values={
            "dataset_path": "/calibrex/input/dataset",
            "input_dir": "/calibrex/input",
            "output_dir": "/calibrex/output",
            "result_path": _container_result_path(config.result_path),
            "config_path": "/calibrex/input/config.yaml",
            "initial_guess_path": "/calibrex/input/initial_guess",
        },
    )
    argv = [
        container.engine,
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--mount",
        _mount_arg(source_input, "/calibrex/input", readonly=True),
        "--mount",
        _mount_arg(source_output, "/calibrex/output", readonly=False),
    ]
    if container.cpus is not None:
        argv.extend(["--cpus", f"{container.cpus:g}"])
    if container.memory is not None:
        argv.extend(["--memory", container.memory])
    argv.extend(["--workdir", container.workdir, container.resolved_image, *command])
    return argv


class KoideExternalRunner:
    """Execute a typed Koide workflow with bounded, reproducible evidence."""

    def __init__(self, process_runner: KoideProcessRunner | None = None) -> None:
        self._process_runner = process_runner or _subprocess_runner

    def run(
        self,
        config: KoideRunnerConfig,
        *,
        input_artifacts: Sequence[str | Path] = (),
    ) -> KoideWorkflowResult:
        """Run all configured stages or import a precomputed native result."""

        # Imported lazily so ``calibrex.core.config`` can validate Koide
        # options during package bootstrap without a package-version cycle.
        from calibrex import __version__
        from calibrex.importers.koide import parse_koide_calib_json

        input_paths = _unique_paths(
            [
                *config.input_paths,
                *input_artifacts,
                config.dataset_path,
                config.config_path,
                config.manual_initial_guess_path,
                config.precomputed_initial_guess_path,
                config.checkpoint_path,
                config.readiness_artifact_path,
            ]
        )
        artifacts = _digest_artifacts(input_paths, role="input")
        result_path = config.result_path
        stage_results: list[KoideStageResult] = []
        error: str | None = None

        preflight_errors = [
            error
            for error in (
                _readiness_error(config),
                _checkpoint_digest_error(config),
            )
            if error is not None
        ]
        preflight_error = "; ".join(preflight_errors) or None
        if preflight_error is not None:
            stage_results = []
            error = preflight_error
        elif config.execution_mode == "precomputed":
            stage_results = []
        elif config.execution_mode == "container":
            stage_results, error = self._execute_stages(config)
        else:
            stage_results, error = self._execute_stages(config)

        calibration: KoideNativeCalibration | None = None
        transforms: dict[str, SE3] = {}
        parse_error: str | None = None
        if error is None:
            if result_path is None:
                parse_error = "Koide workflow requires result_path for native output"
            elif not result_path.is_file():
                parse_error = f"Koide result does not exist: {result_path}"
            else:
                try:
                    if config.lidar_frame is None or config.camera_frame is None:
                        parse_error = (
                            "native Koide calib.json requires explicit lidar_frame and camera_frame"
                        )
                    else:
                        calibration = parse_koide_calib_json(
                            result_path,
                            lidar_frame=config.lidar_frame,
                            camera_frame=config.camera_frame,
                        )
                        transforms = calibration.transforms
                except ValueError as exc:
                    parse_error = str(exc)
                if parse_error is None and calibration is None:
                    parse_error = "Koide result parser returned no calibration"

        output_digest = _digest_artifact("output", result_path)
        if output_digest is not None:
            artifacts.append(output_digest)
        source_digest = _digest_path(config.source_path)
        tool_path = config.tool_path or _tool_path_from_stages(config, stage_results)
        tool_digest = _digest_path(tool_path)
        config_digest = _config_digest(config)
        input_digest = _combined_digest(artifacts, role="input")
        output_sha256 = output_digest.sha256 if output_digest else None
        status: ExternalRunStatus
        if preflight_error is not None:
            status = "unavailable"
        elif parse_error is not None:
            status = "invalid_output"
            error = parse_error
        elif error is not None:
            status = "timeout" if any(stage.timed_out for stage in stage_results) else "failed"
        elif config.execution_mode == "precomputed":
            status = "success"
        else:
            status = "success"

        run_warnings = _runner_warnings(config, artifacts, stage_results, error)
        if parse_error is not None:
            run_warnings.append(f"Koide native output parse failed: {parse_error}")
        parsed_transform_outputs: dict[str, ExternalTransformOutput] = {}
        for name, transform in transforms.items():
            frame_pair = _frames_for_transform(name, calibration)
            if frame_pair is None:
                continue
            parent, child = frame_pair
            parsed_transform_outputs[name] = ExternalTransformOutput(
                parent=parent,
                child=child,
                translation_m=list(transform.translation_m),
                rotation_quat_xyzw=list(transform.rotation_quat_xyzw),
            )
        parsed_outputs = ExternalParsedOutputs(
            transforms=parsed_transform_outputs,
            external_metrics=(
                {
                    "koide_native_format": calibration.native_format,
                    "koide_native_transform": calibration.native_transform_name,
                    "koide_native_declared_convention": calibration.declared_convention,
                }
                if calibration is not None
                else {}
            ),
        )
        container_digest = config.container.resolved_image if config.container else None
        commands = _stage_commands(config, stage_results)
        execution = ExternalExecution(
            mode=("container" if config.execution_mode == "container" else config.execution_mode),
            command=commands,
            working_directory=str(config.working_dir) if config.working_dir else None,
            container_digest=container_digest,
            timeout_seconds=config.timeout_seconds,
            attempted=bool(stage_results)
            and any(stage.status != "skipped" for stage in stage_results),
            return_code=_last_return_code(stage_results),
            timed_out=any(stage.timed_out for stage in stage_results),
            duration_seconds=sum(stage.duration_seconds or 0.0 for stage in stage_results),
            stdout_sha256=_combined_text_digest(stage_results, stream="stdout"),
            stderr_sha256=_combined_text_digest(stage_results, stream="stderr"),
            stdout_tail=_last_tail(stage_results, stream="stdout"),
            stderr_tail=_last_tail(stage_results, stream="stderr"),
            error=error,
            stages=[stage.as_external() for stage in stage_results],
            network_mode=config.container.network_mode if config.container else None,
            cpu_limit=config.container.cpus if config.container else None,
            memory_limit=config.container.memory if config.container else None,
            input_mounts=(["/calibrex/input:ro"] if config.container is not None else []),
            output_mount="/calibrex/output:rw" if config.container is not None else None,
        )
        artifact = ExternalCalibrationRunArtifact(
            run_id=f"koide:{config.tool_name}:{int(time.time())}",
            adapter_name="koide_lidar_camera",
            adapter_version="calibrex.koide_lidar_camera_runner/v0.1",
            tool=ExternalToolIdentity(
                name=config.tool_name,
                version=config.tool_version,
                source_repository=config.source_repository,
                source_commit=config.source_commit,
                license_spdx=config.license_spdx,
                license_boundary=(
                    "container"
                    if config.execution_mode == "container"
                    else "subprocess"
                    if config.execution_mode == "subprocess"
                    else "imported"
                ),
            ),
            execution=execution,
            artifacts=artifacts,
            digests=ExternalRunDigests(
                tool_sha256=tool_digest,
                source_sha256=source_digest,
                container_digest=container_digest,
                config_sha256=config_digest,
                input_sha256=input_digest,
                output_sha256=output_sha256,
            ),
            frame_convention=(
                calibration.declared_convention
                if calibration is not None
                else (
                    "Koide results.T_lidar_camera maps camera points into LiDAR; "
                    "explicit frame binding required"
                )
            ),
            time_convention="Koide calib.json contains no clock-offset estimate",
            train_data_isolation=ExternalDataIsolation(
                declared=config.training_isolation_declared,
                evidence=config.training_isolation_evidence,
                training_data_ids_sha256=config.training_data_ids_sha256,
                holdout_data_ids_sha256=config.holdout_data_ids_sha256,
            ),
            status=status,
            warnings=run_warnings,
            parsed_outputs=parsed_outputs,
            provenance=ExternalRunProvenance(
                calibrex_version=__version__,
                git_commit=git_commit(),
                source_artifact=str(result_path) if result_path else None,
                source_artifact_sha256=output_sha256,
            ),
        )
        return KoideWorkflowResult(artifact=artifact, stages=stage_results)

    def _execute_stages(
        self,
        config: KoideRunnerConfig,
    ) -> tuple[list[KoideStageResult], str | None]:
        results: list[KoideStageResult] = []
        failed = False
        for stage in config.stages:
            if not stage.enabled:
                results.append(
                    KoideStageResult(
                        name=stage.name,
                        command=[],
                        status="skipped",
                        error="disabled",
                    )
                )
                continue
            if failed:
                results.append(
                    KoideStageResult(
                        name=stage.name,
                        command=[],
                        status="skipped",
                        error="prior stage failed",
                    )
                )
                continue
            if config.execution_mode == "container":
                command = build_container_argv(config, stage)
                cwd: str | Path | None = None
            else:
                values = _host_placeholder_values(config)
                command = _expand_stage_argv(stage.argv, config, values=values)
                cwd = config.working_dir
            result = self._execute_one(
                stage,
                command,
                cwd=cwd,
                timeout=stage.timeout_seconds or config.timeout_seconds,
            )
            results.append(result)
            if result.status != "success":
                failed = True
        if failed:
            failed_stage = next(
                stage for stage in results if stage.status in {"failed", "timeout", "unavailable"}
            )
            return results, failed_stage.error or f"Koide stage {failed_stage.name} failed"
        return results, None

    def _execute_one(
        self,
        stage: KoideStageConfig,
        command: list[str],
        *,
        cwd: str | Path | None,
        timeout: float,
    ) -> KoideStageResult:
        executable = command[0] if command else ""
        if not executable:
            return KoideStageResult(
                name=stage.name,
                command=command,
                status="unavailable",
                error="empty stage argv",
            )
        started = time.perf_counter()
        try:
            completed = self._process_runner(
                command,
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = _to_text(exc.stdout) or ""
            stderr = _to_text(exc.stderr) or ""
            return KoideStageResult(
                name=stage.name,
                command=command,
                status="timeout",
                timed_out=True,
                stdout_sha256=_text_digest(stdout),
                stderr_sha256=_text_digest(stderr),
                stdout_tail=_tail(stdout),
                stderr_tail=_tail(stderr),
                duration_seconds=time.perf_counter() - started,
                error=f"stage timed out after {timeout:g} seconds",
            )
        except OSError as exc:
            return KoideStageResult(
                name=stage.name,
                command=command,
                status="unavailable",
                duration_seconds=time.perf_counter() - started,
                error=str(exc),
            )
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        success = completed.returncode == 0
        return KoideStageResult(
            name=stage.name,
            command=command,
            status="success" if success else "failed",
            return_code=completed.returncode,
            stdout_sha256=_text_digest(stdout),
            stderr_sha256=_text_digest(stderr),
            stdout_tail=_tail(stdout),
            stderr_tail=_tail(stderr),
            duration_seconds=time.perf_counter() - started,
            error=None if success else f"stage returned non-zero status {completed.returncode}",
        )


def run_koide_workflow(
    config: KoideRunnerConfig,
    *,
    input_artifacts: Sequence[str | Path] = (),
    process_runner: KoideProcessRunner | None = None,
) -> KoideWorkflowResult:
    """Convenience wrapper around :class:`KoideExternalRunner`."""

    return KoideExternalRunner(process_runner=process_runner).run(
        config,
        input_artifacts=input_artifacts,
    )


def _stages_from_options(options: Mapping[str, Any]) -> list[KoideStageConfig]:
    raw = options.get("stages")
    if raw is not None:
        if not isinstance(raw, list):
            raise ValueError("Koide stages must be a list of typed stage mappings")
        return [KoideStageConfig.model_validate(item) for item in raw]
    stages: list[KoideStageConfig] = []
    for name in ("preprocess", "initial_guess", "calibrate"):
        command = options.get(f"{name}_command")
        if command is None and isinstance(options.get("stage_commands"), Mapping):
            command = options["stage_commands"].get(name)
        if command is None and name == "calibrate":
            command = options.get("command")
        if command is None:
            continue
        argv = _argv_from_value(command)
        if argv:
            stages.append(KoideStageConfig(name=name, argv=argv))
    return stages


def _container_from_options(
    options: Mapping[str, Any],
    *,
    input_default: Path = Path("."),
    output_default: Path = Path("."),
) -> KoideContainerConfig | None:
    raw = options.get("container")
    if raw is None and any(
        key in options for key in ("container_engine", "container_digest", "image_digest")
    ):
        raw = {
            "engine": options.get("container_engine", "docker"),
            "image": options.get("container_image"),
            "image_digest": options.get("container_digest", options.get("image_digest")),
            "input_dir": options.get("input_dir", options.get("working_dir", input_default)),
            "output_dir": options.get("output_dir", options.get("working_dir", output_default)),
            "network_mode": options.get("network_mode", "none"),
            "cpus": options.get("cpu_limit", options.get("cpus")),
            "memory": options.get("memory_limit", options.get("memory")),
        }
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("Koide container options must be a mapping")
    normalized = dict(raw)
    normalized.setdefault("input_dir", input_default)
    normalized.setdefault("output_dir", output_default)
    return KoideContainerConfig.model_validate(normalized)


def _argv_from_value(value: object) -> list[str]:
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(part) for part in value]
    raise ValueError("Koide stage command must be an argv list or string")


def _unsafe_initial_guess_requested(options: Mapping[str, Any]) -> bool:
    """Detect explicit policy-sensitive values in legacy option mappings.

    This intentionally inspects values, rather than the serialized mapping:
    keys such as ``initial_guess`` and ``automatic_initial_guess`` are part of
    the backwards-compatible option vocabulary and must not be interpreted as
    an opt-in merely because they exist.  Boolean compatibility flags are
    honoured only when true; a string mode is accepted only when its value is
    exactly ``automatic``.
    """

    for key in ("automatic_initial_guess", "use_automatic_initial_guess"):
        value = options.get(key)
        if isinstance(value, bool) and value:
            return True
        if isinstance(value, str) and value.strip().lower() in {"true", "yes", "1"}:
            return True
    mode = options.get("initial_guess_mode")
    if isinstance(mode, str) and mode.strip().lower() == "automatic":
        return True
    return _contains_superglue_value(options)


def _uses_automatic_initial_guess(config: KoideRunnerConfig) -> bool:
    """Return whether a typed config opts into a non-commercial provider.

    Stage names are workflow labels, not provider selections.  In particular,
    a stage named ``initial_guess`` remains valid for a manual or precomputed
    initial guess.  Commercial mode blocks explicit automatic modes and any
    SuperGlue provider, checkpoint, or argv value.
    """

    if config.automatic_initial_guess or config.initial_guess_mode == "automatic":
        return True
    # Scan typed values rather than field names.  This covers provider names,
    # checkpoint/tool paths, stage argv, and future typed fields without
    # making a field named ``initial_guess`` or ``use_superglue`` unsafe by
    # itself.  A disabled/false flag is therefore harmless; a value that
    # actually points at SuperGlue is not.
    return _contains_superglue_value(config.model_dump(mode="python"))


def _contains_superglue_value(value: object) -> bool:
    """Recursively detect a SuperGlue provider/value, ignoring mapping keys."""

    if isinstance(value, Mapping):
        return any(_contains_superglue_value(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_superglue_value(item) for item in value)
    if isinstance(value, Path):
        return "superglue" in str(value).lower()
    return isinstance(value, str) and "superglue" in value.lower()


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _host_placeholder_values(config: KoideRunnerConfig) -> dict[str, str]:
    result = str(config.result_path) if config.result_path is not None else ""
    return {
        "dataset_path": str(config.dataset_path) if config.dataset_path else "",
        "result_path": result,
        "output_dir": str(config.result_path.parent) if config.result_path else "",
        "input_dir": str(config.dataset_path.parent) if config.dataset_path else "",
        "config_path": str(config.config_path) if config.config_path else "",
        "initial_guess_path": str(
            config.manual_initial_guess_path or config.precomputed_initial_guess_path or ""
        ),
        "camera_streams": ",".join(config.camera_streams),
        "lidar_streams": ",".join(config.lidar_streams),
    }


def _expand_stage_argv(
    argv: Sequence[str],
    config: KoideRunnerConfig,
    *,
    values: Mapping[str, str],
) -> list[str]:
    del config
    expanded: list[str] = []
    for argument in argv:
        value = argument
        for key in _PATH_PLACEHOLDER_NAMES:
            value = value.replace("{" + key + "}", values.get(key, ""))
        if "\x00" in value:
            raise ValueError("expanded Koide argv must not contain NUL bytes")
        expanded.append(value)
    return expanded


def _container_result_path(path: Path | None) -> str:
    return f"/calibrex/output/{path.name}" if path is not None else "/calibrex/output/calib.json"


def _mount_arg(source: Path, destination: str, *, readonly: bool) -> str:
    resolved = source.resolve()
    text = str(resolved)
    if any(character in text for character in (",", "\n", "\r")):
        raise ValueError("container mount paths must not contain commas or newlines")
    suffix = ",readonly" if readonly else ",rw"
    return f"type=bind,src={text},dst={destination}{suffix}"


def _validate_mount_path(path: Path, name: str) -> None:
    if any(character in str(path) for character in ("\x00", "\n", "\r", ",")):
        raise ValueError(f"container {name} contains an unsafe mount path")


def _subprocess_runner(
    args: Sequence[str],
    *,
    capture_output: bool,
    text: bool,
    cwd: str | Path | None,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        check=False,
        capture_output=capture_output,
        text=text,
        cwd=cwd,
        timeout=timeout,
        shell=False,
    )


def _digest_artifacts(
    paths: Iterable[Path],
    *,
    role: Literal["input", "output"],
) -> list[ExternalArtifactDigest]:
    artifacts: list[ExternalArtifactDigest] = []
    for path in paths:
        artifact = _digest_artifact(role, path)
        if artifact is not None and not any(
            existing.path == artifact.path for existing in artifacts
        ):
            artifacts.append(artifact)
    return artifacts


def _digest_artifact(
    role: Literal["input", "output"],
    path: Path | None,
) -> ExternalArtifactDigest | None:
    if path is None or not path.exists():
        return None
    digest = _digest_path(path)
    if digest is None:
        return None
    size = (
        path.stat().st_size
        if path.is_file()
        else sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    )
    media_type = "application/json" if path.suffix.lower() == ".json" else "application/yaml"
    return ExternalArtifactDigest(
        role=role,
        path=str(path),
        sha256=digest,
        size_bytes=size,
        media_type=media_type,
    )


def _digest_path(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    if path.is_file():
        return sha256_path(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = child.relative_to(path).as_posix().encode("utf-8")
        child_digest = sha256_path(child)
        if child_digest is None:
            continue
        digest.update(relative)
        digest.update(b"\0")
        digest.update(child_digest.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _config_digest(config: KoideRunnerConfig) -> str:
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _combined_digest(artifacts: Iterable[ExternalArtifactDigest], *, role: str) -> str | None:
    selected = sorted(
        (artifact.path, artifact.sha256) for artifact in artifacts if artifact.role == role
    )
    if not selected:
        return None
    payload = "\n".join(f"{path}\0{digest}" for path, digest in selected).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _unique_paths(values: Iterable[Path | str | None]) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for raw in values:
        if raw is None:
            continue
        path = Path(raw)
        key = str(path)
        if key not in seen:
            seen.add(key)
            paths.append(path)
    return paths


def _runner_warnings(
    config: KoideRunnerConfig,
    artifacts: Sequence[ExternalArtifactDigest],
    stages: Sequence[KoideStageResult],
    error: str | None,
) -> list[str]:
    warnings: list[str] = []
    if not any(artifact.role == "input" for artifact in artifacts):
        warnings.append("Koide run has no digest-bound input artifact")
    if config.tool_path is None and _tool_path_from_stages(config, stages) is None:
        warnings.append("Koide tool executable digest is not available")
    if config.source_path is None:
        warnings.append("Koide source tree digest is not declared")
    if config.tool_version is None:
        warnings.append("Koide tool version is not declared")
    if config.source_commit is None:
        warnings.append("Koide source commit is not declared")
    if not config.training_isolation_declared:
        warnings.append("external training-data isolation is not declared")
    if error:
        warnings.append(error)
    if any(stage.status == "skipped" for stage in stages):
        warnings.append("one or more Koide stages were skipped after an earlier failure")
    return warnings


def _readiness_error(config: KoideRunnerConfig) -> str | None:
    """Return a strict preflight error without importing readiness at module load."""

    if not config.strict_readiness:
        return None
    if config.readiness_artifact_path is None:
        return "strict Koide execution requires readiness_artifact_path"
    try:
        from calibrex.core.koide_readiness import validate_koide_readiness_for_execution

        validate_koide_readiness_for_execution(
            config.readiness_artifact_path,
            expected_sha256=config.readiness_artifact_sha256,
            strict=True,
        )
    except (OSError, ValueError) as exc:
        return f"Koide readiness preflight blocked execution: {exc}"
    return None


def _checkpoint_digest_error(config: KoideRunnerConfig) -> str | None:
    """Reject a missing or changed declared research checkpoint before running.

    The typed config validates the shape of ``checkpoint_sha256``.  This
    runtime check binds that declaration to the bytes that will be presented to
    the external provider; a syntactically valid digest must not be enough to
    pass a research/noncommercial execution gate.
    """

    expected = config.checkpoint_sha256
    required = config.profile == "research-noncommercial"
    if expected is None:
        return (
            "research-noncommercial Koide execution requires checkpoint_sha256"
            if required
            else None
        )
    if config.checkpoint_path is None:
        return "declared Koide checkpoint_sha256 has no checkpoint_path"
    observed = _digest_path(config.checkpoint_path)
    if observed is None:
        return f"Koide checkpoint is unavailable: {config.checkpoint_path}"
    if observed != expected.lower():
        return (
            "Koide checkpoint SHA-256 mismatch: "
            f"expected {expected.lower()}, observed {observed}"
        )
    return None


def _stage_commands(config: KoideRunnerConfig, stages: Sequence[KoideStageResult]) -> list[str]:
    if stages:
        return list(stages[0].command)
    return []


def _tool_path_from_stages(
    config: KoideRunnerConfig,
    stages: Sequence[KoideStageResult],
) -> Path | None:
    """Infer a host executable digest when an argv stage names a real file."""

    del config
    for stage in stages:
        if not stage.command:
            continue
        candidate = Path(stage.command[0])
        if candidate.is_file():
            return candidate
    return None


def _last_return_code(stages: Sequence[KoideStageResult]) -> int | None:
    for stage in reversed(stages):
        if stage.return_code is not None:
            return stage.return_code
    return None


def _combined_text_digest(
    stages: Sequence[KoideStageResult],
    *,
    stream: Literal["stdout", "stderr"],
) -> str | None:
    values = [getattr(stage, f"{stream}_sha256") for stage in stages]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return hashlib.sha256("\n".join(values).encode("ascii")).hexdigest()


def _last_tail(
    stages: Sequence[KoideStageResult],
    *,
    stream: Literal["stdout", "stderr"],
) -> str | None:
    for stage in reversed(stages):
        value = getattr(stage, f"{stream}_tail")
        if value:
            return cast(str, value)
    return None


def _text_digest(value: str | None) -> str | None:
    return hashlib.sha256(value.encode("utf-8")).hexdigest() if value is not None else None


def _tail(value: str | None, *, limit: int = 4000) -> str | None:
    return value[-limit:] if value else None


def _to_text(value: str | bytes | None) -> str | None:
    if value is None:
        return None
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def _frames_for_transform(
    name: str,
    calibration: KoideNativeCalibration | None,
) -> tuple[str, str] | None:
    if calibration is not None:
        forward = f"T_{calibration.lidar_frame}_{calibration.camera_frame}"
        inverse = f"T_{calibration.camera_frame}_{calibration.lidar_frame}"
        if name == forward:
            return calibration.lidar_frame, calibration.camera_frame
        if name == inverse:
            return calibration.camera_frame, calibration.lidar_frame
    if not name.startswith("T_"):
        return None
    body = name[2:]
    parent, separator, child = body.partition("_")
    return (parent, child) if separator and parent and child else None


__all__ = [
    "KoideContainerConfig",
    "KoideExecutionMode",
    "KoideExternalRunner",
    "KoideInitialGuessMode",
    "KoideProcessRunner",
    "KoideProfile",
    "KoideRunnerConfig",
    "KoideStageConfig",
    "KoideStageName",
    "KoideStageResult",
    "KoideWorkflowResult",
    "build_container_argv",
    "koide_runner_config_from_options",
    "koide_runner_json_schema",
    "run_koide_workflow",
]
