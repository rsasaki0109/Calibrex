"""Koide-style targetless LiDAR-camera adapter boundary.

This module intentionally does not vendor or import the external implementation.
It records readiness, license boundary, and optional precomputed outputs in the
Calibrex result schema so strong external baselines can be compared safely.
"""

from __future__ import annotations

import hashlib
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from yaml import YAMLError

from calibrex import __version__
from calibrex.core.config import CalibrationConfig, FactorConfig
from calibrex.core.external_run import (
    ExternalArtifactDigest,
    ExternalCalibrationRunArtifact,
    ExternalDataIsolation,
    ExternalExecution,
    ExternalExecutionMode,
    ExternalParsedOutputs,
    ExternalRunProvenance,
    ExternalRunStatus,
    ExternalTransformOutput,
)
from calibrex.core.external_run import (
    ExternalToolIdentity as ExternalRunToolIdentity,
)
from calibrex.core.frames import FrameGraph
from calibrex.core.geometry import SE3
from calibrex.core.io import read_mapping
from calibrex.core.koide_runner import (
    KoideRunnerConfig,
    run_koide_workflow,
)
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.result import MetricResult
from calibrex.data.base import StreamSummary
from calibrex.data.inspect import DatasetInspection
from calibrex.importers.koide import KoideNativeCalibration, parse_koide_calib_payload
from calibrex.solvers.base import SolverAdapter, SolverAdapterResult

ADAPTER_FACTOR_NAMES = (
    "koide_lidar_camera",
    "direct_visual_lidar_calibration",
    "lidar_camera_targetless_baseline",
)


@dataclass(frozen=True)
class KoideLidarCameraInputs:
    """Input readiness summary for a targetless LiDAR-camera adapter."""

    camera_streams: tuple[str, ...]
    lidar_streams: tuple[str, ...]
    command: str | None
    command_available: bool
    result_path: str | None
    result_exists: bool
    configured_factor: str | None
    execute: bool
    timeout_sec: float
    working_dir: str | None
    camera_frame: str | None = None
    lidar_frame: str | None = None
    frame_binding_source: str | None = None

    @property
    def has_camera(self) -> bool:
        return bool(self.camera_streams)

    @property
    def has_lidar(self) -> bool:
        return bool(self.lidar_streams)

    @property
    def input_ready(self) -> bool:
        return self.has_camera and self.has_lidar


@dataclass(frozen=True)
class AdapterExecutionResult:
    """Captured external process execution summary."""

    requested: bool = False
    attempted: bool = False
    returncode: int | None = None
    timed_out: bool = False
    stdout_tail: str | None = None
    stderr_tail: str | None = None
    stdout_sha256: str | None = None
    stderr_sha256: str | None = None
    duration_seconds: float | None = None
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.attempted and not self.timed_out and self.returncode == 0

    def as_dict(self) -> dict[str, object]:
        """Return a JSON/YAML-friendly execution summary."""

        return {
            "requested": self.requested,
            "attempted": self.attempted,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "stdout_sha256": self.stdout_sha256,
            "stderr_sha256": self.stderr_sha256,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
        }


@dataclass(frozen=True)
class KoideExternalToolIdentity:
    """Reproducible identity and output lineage for the external baseline."""

    tool_name: str
    tool_version: str | None
    source_repository: str | None
    source_commit: str | None
    license_spdx: str | None
    adapter_version: str
    command: str | None
    result_path: str | None
    result_sha256: str | None
    result_size_bytes: int | None
    training_isolation_declared: bool
    training_isolation_evidence: str | None

    @property
    def missing_required_fields(self) -> tuple[str, ...]:
        fields = {
            "tool_version": self.tool_version,
            "source_repository": self.source_repository,
            "source_commit": self.source_commit,
            "license_spdx": self.license_spdx,
            "result_sha256": self.result_sha256,
        }
        return tuple(name for name, value in fields.items() if value is None)

    @property
    def complete(self) -> bool:
        return not self.missing_required_fields

    def as_dict(self) -> dict[str, object]:
        return {
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "source_repository": self.source_repository,
            "source_commit": self.source_commit,
            "license_spdx": self.license_spdx,
            "adapter_version": self.adapter_version,
            "command": self.command,
            "result_path": self.result_path,
            "result_sha256": self.result_sha256,
            "result_size_bytes": self.result_size_bytes,
            "training_isolation_declared": self.training_isolation_declared,
            "training_isolation_evidence": self.training_isolation_evidence,
            "identity_declaration_source": "factor.options plus hashed result payload",
            "missing_required_fields": list(self.missing_required_fields),
            "complete": self.complete,
        }


class KoideLidarCameraSolver(SolverAdapter):
    """Optional boundary for Koide-style targetless LiDAR-camera calibration."""

    backend = "koide_lidar_camera"

    def solve(
        self,
        config: CalibrationConfig,
        frame_graph: FrameGraph,
        inspection: DatasetInspection,
    ) -> SolverAdapterResult:
        """Summarize adapter readiness and load optional external transforms."""

        del frame_graph
        inputs = _summarize_inputs(config, inspection)
        if _first_class_runner_requested(config):
            return _solve_with_first_class_runner(config, inspection, inputs)
        execution = _execute_command(config, inputs) if inputs.execute else AdapterExecutionResult()
        native_calibration: KoideNativeCalibration | None = None
        try:
            transforms, native_calibration = _load_transforms_with_metadata(
                inputs.result_path,
                lidar_frame=inputs.lidar_frame,
                camera_frame=inputs.camera_frame,
                configured_frames=set(config.frames),
            )
        except (OSError, ValueError) as exc:
            transforms = {}
            execution = _with_output_error(execution, exc)
        result_exists = _path_exists(inputs.result_path)
        identity = _external_tool_identity(config, inputs)
        metrics = {
            "koide_lidar_camera_adapter_available": MetricResult(
                value=1.0 if inputs.command_available or result_exists else 0.0,
                grade="pass" if inputs.command_available or result_exists else "warn",
                reason=(
                    "external command or precomputed result is available"
                    if inputs.command_available or result_exists
                    else "configure an external command or result_path for the adapter"
                ),
            ),
            "koide_lidar_camera_input_ready": MetricResult(
                value=1.0 if inputs.input_ready else 0.0,
                grade="pass" if inputs.input_ready else "fail",
                reason=(
                    "camera and LiDAR streams are available"
                    if inputs.input_ready
                    else "camera and LiDAR streams are required"
                ),
            ),
            "koide_lidar_camera_result_available": MetricResult(
                value=1.0 if transforms else 0.0,
                grade="pass" if transforms else "warn",
                reason=(
                    "precomputed LiDAR-camera transform was loaded"
                    if transforms
                    else "no precomputed external transform was loaded"
                ),
            ),
            "koide_lidar_camera_execution_success": _execution_metric(execution),
            "koide_lidar_camera_provenance_complete": MetricResult(
                value=1.0 if identity.complete else 0.0,
                grade="pass" if identity.complete else "warn",
                reason=(
                    "external tool identity and result digest are complete"
                    if identity.complete
                    else "missing external provenance: "
                    + ", ".join(identity.missing_required_fields)
                ),
            ),
        }
        warnings = _warnings(inputs, transforms, execution, result_exists)
        status = _status(inputs, transforms, execution, result_exists)
        external_run = _external_run_artifact(
            config=config,
            inspection=inspection,
            inputs=inputs,
            execution=execution,
            identity=identity,
            transforms=transforms,
            status=status,
            warnings=warnings,
            native_calibration=native_calibration,
        )
        native_provenance = (
            native_calibration.provenance(
                inputs.result_path or "",
                identity.result_sha256,
            )
            if native_calibration is not None
            else None
        )
        return SolverAdapterResult(
            backend=self.backend,
            available=inputs.command_available or result_exists,
            status=status,
            metrics=metrics,
            transforms=transforms,
            provenance={
                "koide_lidar_camera_status": status,
                "koide_lidar_camera_factor": inputs.configured_factor,
                "koide_lidar_camera_command": inputs.command,
                "koide_lidar_camera_command_available": inputs.command_available,
                "koide_lidar_camera_result_path": inputs.result_path,
                "koide_lidar_camera_result_exists": result_exists,
                "koide_lidar_camera_execute": inputs.execute,
                "koide_lidar_camera_timeout_sec": inputs.timeout_sec,
                "koide_lidar_camera_working_dir": inputs.working_dir,
                "koide_lidar_camera_camera_frame": inputs.camera_frame,
                "koide_lidar_camera_lidar_frame": inputs.lidar_frame,
                "koide_lidar_camera_frame_binding_source": inputs.frame_binding_source,
                "koide_lidar_camera_execution": execution.as_dict(),
                "koide_lidar_camera_camera_streams": list(inputs.camera_streams),
                "koide_lidar_camera_lidar_streams": list(inputs.lidar_streams),
                "koide_lidar_camera_training_isolation_declared": bool(
                    _adapter_options(config).get("training_isolation_declared", False)
                ),
                "koide_lidar_camera_loaded_transforms": {
                    name: {"convention": "T_parent_child", **transform.as_dict()}
                    for name, transform in sorted(transforms.items())
                },
                "koide_lidar_camera_native_import": native_provenance,
                "koide_lidar_camera_tool_identity": identity.as_dict(),
                "external_calibration_run": external_run.model_dump(
                    mode="json",
                    exclude_none=True,
                ),
                "external_calibration_run_schema_version": external_run.schema_version,
                "license_boundary": (
                    "external subprocess/precomputed-result adapter; no external "
                    "calibration code is copied into Calibrex core"
                ),
            },
            warnings=list(external_run.warnings),
            external_run=external_run,
        )


def _summarize_inputs(
    config: CalibrationConfig,
    inspection: DatasetInspection,
) -> KoideLidarCameraInputs:
    factor_name, factor = _adapter_factor(config)
    options = factor.options if factor is not None else {}
    command = _command_from_options(options)
    result_path = _result_path_from_options(options)
    working_dir = _working_dir_from_options(options)
    camera_frame, lidar_frame, frame_binding_source = _resolve_frame_bindings(
        options,
        inspection.streams,
    )
    return KoideLidarCameraInputs(
        camera_streams=_stream_names(inspection.streams, kind="camera"),
        lidar_streams=_stream_names(inspection.streams, kind="lidar"),
        command=command,
        command_available=_command_available(command),
        result_path=str(result_path) if result_path is not None else None,
        result_exists=result_path.exists() if result_path is not None else False,
        configured_factor=factor_name,
        execute=_bool_option(options, "execute", default=False),
        timeout_sec=_float_option(options, "timeout_sec", default=600.0),
        working_dir=str(working_dir) if working_dir is not None else None,
        camera_frame=camera_frame,
        lidar_frame=lidar_frame,
        frame_binding_source=frame_binding_source,
    )


def _first_class_runner_requested(config: CalibrationConfig) -> bool:
    """Return whether typed Koide workflow options are present.

    Legacy ``command``/``execute`` options continue through the established
    single-process path.  Explicit workflow/container/profile options opt into
    the stricter multi-stage runner.
    """

    options = _adapter_options(config)
    return any(
        key in options
        for key in (
            "execution_mode",
            "profile",
            "stages",
            "preprocess_command",
            "initial_guess_command",
            "calibrate_command",
            "stage_commands",
            "container",
            "container_digest",
            "image_digest",
            "readiness_artifact_path",
            "readiness_artifact_sha256",
            "strict_readiness",
        )
    )


def _solve_with_first_class_runner(
    config: CalibrationConfig,
    inspection: DatasetInspection,
    inputs: KoideLidarCameraInputs,
) -> SolverAdapterResult:
    """Run the typed workflow and preserve the legacy adapter result shape."""

    options = _adapter_options(config)
    runner_config = KoideRunnerConfig.from_options(
        options,
        dataset_path=config.dataset.path,
        result_path=inputs.result_path,
        camera_frame=inputs.camera_frame,
        lidar_frame=inputs.lidar_frame,
        camera_streams=inputs.camera_streams,
        lidar_streams=inputs.lidar_streams,
    )
    input_artifacts: list[str | Path] = []
    if inspection.manifest is not None:
        input_artifacts.append(inspection.manifest)
    workflow = run_koide_workflow(
        runner_config,
        input_artifacts=input_artifacts,
    )
    artifact = workflow.artifact
    warnings = list(artifact.warnings)
    readiness_gate_failed = runner_config.strict_readiness and any(
        "readiness" in warning.lower() for warning in warnings
    )
    effective_input_ready = inputs.input_ready and not readiness_gate_failed
    if readiness_gate_failed:
        warnings.append("strict Koide readiness gate rejected the input artifact")
    elif not inputs.input_ready:
        warnings.append("camera and LiDAR input streams are required before Koide execution")
    transforms = workflow.transforms
    digest_fields = artifact.digests.model_dump(mode="python")
    required_digest_fields = ("config_sha256", "input_sha256", "output_sha256")
    provenance_complete = all(digest_fields.get(field) for field in required_digest_fields)
    metrics = {
        "koide_lidar_camera_adapter_available": MetricResult(
            value=1.0 if artifact.status not in {"unavailable", "not_executed"} else 0.0,
            grade="pass" if artifact.status not in {"unavailable", "not_executed"} else "warn",
            reason=f"typed Koide runner status: {artifact.status}",
        ),
        "koide_lidar_camera_input_ready": MetricResult(
            value=1.0 if effective_input_ready else 0.0,
            grade="pass" if effective_input_ready else "fail",
            reason=(
                "camera and LiDAR streams are available"
                if effective_input_ready
                and not readiness_gate_failed
                else "strict Koide readiness gate rejected the input artifact"
                if readiness_gate_failed
                else "camera and LiDAR streams are required"
            ),
        ),
        "koide_lidar_camera_result_available": MetricResult(
            value=1.0 if transforms else 0.0,
            grade="pass" if transforms else "warn",
            reason=(
                "typed Koide native output was parsed"
                if transforms
                else "no readable Koide transform"
            ),
        ),
        "koide_lidar_camera_execution_success": MetricResult(
            value=1.0 if artifact.status == "success" else 0.0,
            grade="pass" if artifact.status == "success" else "fail",
            reason=f"typed Koide runner status: {artifact.status}",
        ),
        "koide_lidar_camera_provenance_complete": MetricResult(
            value=1.0 if provenance_complete else 0.0,
            grade="pass" if provenance_complete else "warn",
            reason=(
                "typed runner config/input/output digests are complete"
                if provenance_complete
                else "typed runner requires config, input, and output digests"
            ),
        ),
    }
    provenance = dict(artifact.to_solver_adapter_result().provenance)
    provenance.update(
        {
            "koide_lidar_camera_status": artifact.status,
            "koide_lidar_camera_execution_mode": runner_config.execution_mode,
            "koide_lidar_camera_profile": runner_config.profile,
            "koide_lidar_camera_stage_results": [
                stage.model_dump(mode="json", exclude_none=True) for stage in workflow.stages
            ],
            "koide_lidar_camera_runner_config": runner_config.model_dump(mode="json"),
            "koide_lidar_camera_input_ready": effective_input_ready,
            "koide_lidar_camera_frame_binding_source": inputs.frame_binding_source,
            "koide_lidar_camera_tool_identity": {
                "tool_name": artifact.tool.name,
                "tool_version": artifact.tool.version,
                "source_repository": artifact.tool.source_repository,
                "source_commit": artifact.tool.source_commit,
                "license_spdx": artifact.tool.license_spdx,
                "adapter_version": artifact.adapter_version,
                "digests": digest_fields,
                "complete": provenance_complete,
            },
            "license_boundary": (
                "external typed subprocess/container/precomputed adapter; no external "
                "calibration code is copied into Calibrex core"
            ),
        }
    )
    return SolverAdapterResult(
        backend="koide_lidar_camera",
        available=artifact.status not in {"unavailable", "not_executed"},
        status=artifact.status,
        metrics=metrics,
        transforms=transforms,
        provenance=provenance,
        warnings=warnings,
        external_run=artifact,
    )


def _adapter_factor(config: CalibrationConfig) -> tuple[str | None, FactorConfig | None]:
    for name in ADAPTER_FACTOR_NAMES:
        factor = config.pipeline.factors.get(name)
        if factor is not None and factor.enabled:
            return name, factor
    return None, None


def _adapter_options(config: CalibrationConfig) -> dict[str, Any]:
    _name, factor = _adapter_factor(config)
    return factor.options if factor is not None else {}


def _resolve_frame_bindings(
    options: dict[str, Any],
    streams: list[StreamSummary],
) -> tuple[str | None, str | None, str | None]:
    """Resolve explicit native frame names from options or selected streams.

    A stream's configured ``sensor`` name is an explicit binding supplied by
    the dataset/configuration.  It is used only when exactly one candidate is
    selected; no positional ``camera0``/``lidar0`` convention is guessed.
    """

    camera_frame = _text_option(options, "camera_frame")
    lidar_frame = _text_option(options, "lidar_frame")
    sources: list[str] = []
    if camera_frame is not None:
        sources.append("adapter_options.camera_frame")
    if lidar_frame is not None:
        sources.append("adapter_options.lidar_frame")

    camera_stream = _text_option(options, "camera_stream") or _text_option(
        options, "camera_topic"
    )
    lidar_stream = _text_option(options, "lidar_stream") or _text_option(
        options, "lidar_topic"
    )
    camera_frame, camera_source = _frame_from_stream_binding(
        streams,
        kind="camera",
        selected_name=camera_stream,
        existing=camera_frame,
    )
    lidar_frame, lidar_source = _frame_from_stream_binding(
        streams,
        kind="lidar",
        selected_name=lidar_stream,
        existing=lidar_frame,
    )
    if camera_source is not None:
        sources.append(camera_source)
    if lidar_source is not None:
        sources.append(lidar_source)
    source = "+".join(dict.fromkeys(sources)) if sources else None
    return camera_frame, lidar_frame, source


def _frame_from_stream_binding(
    streams: list[StreamSummary],
    *,
    kind: str,
    selected_name: str | None,
    existing: str | None,
) -> tuple[str | None, str | None]:
    if existing is not None:
        return existing, None
    candidates = [stream for stream in streams if _stream_matches_kind(stream, kind)]
    if selected_name is not None:
        candidates = [
            stream
            for stream in candidates
            if stream.name == selected_name or stream.topic == selected_name
        ]
    frame_names = sorted(
        {stream.sensor.strip() for stream in candidates if stream.sensor and stream.sensor.strip()}
    )
    if len(frame_names) != 1:
        return None, None
    return frame_names[0], "configured_stream_sensor"


def _stream_matches_kind(stream: StreamSummary, kind: str) -> bool:
    name = stream.name.lower()
    stream_kind = stream.kind.lower()
    if kind == "camera":
        return (
            stream_kind == "image" or "camera" in name
        ) and stream.message_count != 0
    return (
        stream_kind == "pointcloud" or "lidar" in name or "velodyne" in name
    ) and stream.message_count != 0


def _external_tool_identity(
    config: CalibrationConfig,
    inputs: KoideLidarCameraInputs,
) -> KoideExternalToolIdentity:
    options = _adapter_options(config)
    result_path = Path(inputs.result_path) if inputs.result_path is not None else None
    result_exists = result_path is not None and result_path.is_file()
    return KoideExternalToolIdentity(
        tool_name=_text_option(options, "tool_name")
        or "direct_visual_lidar_calibration",
        tool_version=_text_option(options, "tool_version"),
        source_repository=_text_option(options, "source_repository"),
        source_commit=_text_option(options, "source_commit"),
        license_spdx=_text_option(options, "license_spdx"),
        # Keep the established solver artifact version stable; native format
        # details are carried in the explicit native-import provenance below.
        adapter_version="calibrex.koide_lidar_camera_adapter/v0.2",
        command=inputs.command,
        result_path=inputs.result_path,
        result_sha256=sha256_path(result_path) if result_exists and result_path else None,
        result_size_bytes=result_path.stat().st_size if result_exists and result_path else None,
        training_isolation_declared=_bool_option(
            options, "training_isolation_declared", default=False
        ),
        training_isolation_evidence=_text_option(options, "training_isolation_evidence"),
    )


def _external_run_artifact(
    *,
    config: CalibrationConfig,
    inspection: DatasetInspection,
    inputs: KoideLidarCameraInputs,
    execution: AdapterExecutionResult,
    identity: KoideExternalToolIdentity,
    transforms: dict[str, SE3],
    status: str,
    warnings: list[str],
    native_calibration: KoideNativeCalibration | None,
) -> ExternalCalibrationRunArtifact:
    result_path = Path(inputs.result_path) if inputs.result_path is not None else None
    options = _adapter_options(config)
    artifacts: list[ExternalArtifactDigest] = []
    if inspection.manifest is not None:
        manifest_digest = _external_artifact_digest(
            "input",
            Path(inspection.manifest),
            media_type="application/yaml",
        )
        if manifest_digest is not None:
            artifacts.append(manifest_digest)
    # Preserve explicit input artifacts from both the typed runner vocabulary
    # and the older adapter option name.  The CLI appends ``--input-artifact``
    # values here before invoking the legacy-compatible solver path.
    for input_path in _configured_input_paths(options):
        if any(artifact.path == str(input_path) for artifact in artifacts):
            continue
        input_digest = _external_artifact_digest(
            "input",
            input_path,
            media_type=(
                "application/json"
                if input_path.suffix.lower() == ".json"
                else "application/yaml"
            ),
        )
        if input_digest is not None:
            artifacts.append(input_digest)
    if result_path is not None:
        output_digest = _external_artifact_digest(
            "output",
            result_path,
            media_type=(
                "application/json"
                if result_path.suffix.lower() == ".json"
                else "application/yaml"
            ),
        )
        if output_digest is not None:
            artifacts.append(output_digest)
    run_warnings = list(warnings)
    if not any(artifact.role == "input" for artifact in artifacts):
        run_warnings.append("external run has no digest-bound input artifact")

    mode: ExternalExecutionMode = (
        "subprocess"
        if inputs.command is not None
        else "imported"
        if native_calibration is not None
        else "precomputed"
        if result_path is not None
        else "imported"
    )
    command = (
        shlex.split(_format_command(inputs.command, config, inputs))
        if inputs.command is not None
        else []
    )
    status_by_legacy: dict[str, ExternalRunStatus] = {
        "result_loaded": "success",
        "executed_no_result": "invalid_output",
        "execution_failed": "timeout" if execution.timed_out else "failed",
        "execution_blocked": "unavailable",
        "ready": "not_executed",
        "not_executed": "not_executed",
    }
    external_status = status_by_legacy.get(status, "failed")
    parsed_transforms = {
        name: ExternalTransformOutput(
            parent=frames[0],
            child=frames[1],
            translation_m=list(transform.translation_m),
            rotation_quat_xyzw=list(transform.rotation_quat_xyzw),
        )
        for name, transform in sorted(transforms.items())
        if (frames := _transform_frames(name, config)) is not None
    }
    output_sha256 = sha256_path(result_path) if result_path is not None else None
    return ExternalCalibrationRunArtifact(
        run_id=f"{config.project.name}:koide_lidar_camera",
        adapter_name="koide_lidar_camera",
        adapter_version=identity.adapter_version,
        tool=ExternalRunToolIdentity(
            name=identity.tool_name,
            version=identity.tool_version,
            source_repository=identity.source_repository,
            source_commit=identity.source_commit,
            license_spdx=identity.license_spdx,
            license_boundary="subprocess" if inputs.command is not None else "imported",
        ),
        execution=ExternalExecution(
            mode=mode,
            command=command,
            working_directory=inputs.working_dir,
            timeout_seconds=inputs.timeout_sec if inputs.command is not None else None,
            attempted=execution.attempted,
            return_code=execution.returncode,
            timed_out=execution.timed_out,
            duration_seconds=execution.duration_seconds,
            stdout_sha256=execution.stdout_sha256,
            stderr_sha256=execution.stderr_sha256,
            stdout_tail=execution.stdout_tail,
            stderr_tail=execution.stderr_tail,
            error=execution.error,
        ),
        artifacts=artifacts,
        frame_convention=(
            native_calibration.declared_convention
            if native_calibration is not None
            else "T_parent_child"
        ),
        time_convention=(
            f"{config.dataset.time_base}; Koide adapter imports no clock-offset estimate"
        ),
        train_data_isolation=ExternalDataIsolation(
            declared=identity.training_isolation_declared,
            evidence=identity.training_isolation_evidence,
        ),
        status=external_status,
        warnings=run_warnings,
        parsed_outputs=ExternalParsedOutputs(
            transforms=parsed_transforms,
            external_metrics=(
                _native_external_metrics(native_calibration, result_path, output_sha256)
                if native_calibration is not None
                else {}
            ),
        ),
        provenance=ExternalRunProvenance(
            calibrex_version=__version__,
            git_commit=git_commit(),
            source_artifact=str(result_path) if result_path is not None else None,
            source_artifact_sha256=output_sha256,
        ),
    )


def _external_artifact_digest(
    role: Literal["input", "output"],
    path: Path,
    *,
    media_type: str,
) -> ExternalArtifactDigest | None:
    digest = sha256_path(path)
    if digest is None or not path.is_file():
        return None
    return ExternalArtifactDigest(
        role=role,
        path=str(path),
        sha256=digest,
        size_bytes=path.stat().st_size,
        media_type=media_type,
    )


def _native_external_metrics(
    calibration: KoideNativeCalibration,
    result_path: Path | None,
    result_sha256: str | None,
) -> dict[str, float | str | bool | None]:
    """Expose native-format metadata through the generic external-run schema."""

    return {
        "koide_native_format": "direct_visual_lidar_calibration.calib.json",
        "koide_native_source_path": str(result_path) if result_path is not None else None,
        "koide_native_source_sha256": result_sha256,
        "koide_native_declared_transform": "T_lidar_camera",
        "koide_native_declared_convention": calibration.declared_convention,
        "koide_lidar_frame": calibration.lidar_frame,
        "koide_camera_frame": calibration.camera_frame,
        "koide_calibrex_forward_transform": (
            f"T_{calibration.lidar_frame}_{calibration.camera_frame}"
        ),
        "koide_calibrex_inverse_transform": (
            f"T_{calibration.camera_frame}_{calibration.lidar_frame}"
        ),
    }


def _configured_input_paths(options: dict[str, Any]) -> tuple[Path, ...]:
    """Return explicit input artifacts declared in legacy or typed options."""

    raw = options.get("input_paths", options.get("input_artifacts", ()))
    if isinstance(raw, (str, Path)):
        return (Path(raw),)
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(Path(str(value)) for value in raw if value is not None)


def _transform_frames(
    name: str,
    config: CalibrationConfig,
) -> tuple[str, str] | None:
    frame_names = sorted(config.frames, key=len, reverse=True)
    for parent in frame_names:
        prefix = f"T_{parent}_"
        if not name.startswith(prefix):
            continue
        child = name[len(prefix) :]
        if child in config.frames:
            return parent, child
    return None


def _text_option(options: dict[str, Any], key: str) -> str | None:
    value = options.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _stream_names(streams: list[StreamSummary], *, kind: str) -> tuple[str, ...]:
    names: list[str] = []
    for stream in streams:
        stream_name = stream.name.lower()
        stream_kind = stream.kind.lower()
        if (
            kind == "camera"
            and (stream_kind == "image" or "camera" in stream_name)
            and stream.message_count != 0
        ):
            names.append(stream.name)
        if kind == "lidar" and (
            stream_kind == "pointcloud" or "lidar" in stream_name or "velodyne" in stream_name
        ) and stream.message_count != 0:
            names.append(stream.name)
    return tuple(names)


def _command_from_options(options: dict[str, Any]) -> str | None:
    raw = options.get("command")
    if raw is None:
        return None
    if isinstance(raw, str):
        stripped = raw.strip()
        return stripped or None
    if isinstance(raw, list | tuple):
        parts = [str(part) for part in raw if str(part)]
        return " ".join(shlex.quote(part) for part in parts) if parts else None
    return str(raw)


def _result_path_from_options(options: dict[str, Any]) -> Path | None:
    raw = options.get("result_path") or options.get("slac_result") or options.get(
        "output_result"
    )
    if raw is None:
        return None
    return Path(str(raw))


def _working_dir_from_options(options: dict[str, Any]) -> Path | None:
    raw = options.get("working_dir") or options.get("cwd")
    if raw is None:
        return None
    return Path(str(raw))


def _bool_option(options: dict[str, Any], key: str, *, default: bool) -> bool:
    value = options.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _float_option(options: dict[str, Any], key: str, *, default: float) -> float:
    value = options.get(key, default)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(parsed, 0.1)


def _command_available(command: str | None) -> bool:
    if command is None:
        return False
    parts = shlex.split(command)
    if not parts:
        return False
    executable = parts[0]
    return shutil.which(executable) is not None or Path(executable).exists()


def _execute_command(
    config: CalibrationConfig,
    inputs: KoideLidarCameraInputs,
) -> AdapterExecutionResult:
    if inputs.command is None:
        return AdapterExecutionResult(requested=True, error="command is not configured")
    if not inputs.command_available:
        return AdapterExecutionResult(requested=True, error="command executable is not available")
    if not inputs.input_ready:
        return AdapterExecutionResult(
            requested=True,
            error="camera and LiDAR input streams are not both available",
        )

    command = _format_command(inputs.command, config, inputs)
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            shlex.split(command),
            check=False,
            capture_output=True,
            cwd=inputs.working_dir,
            text=True,
            timeout=inputs.timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _to_text(exc.stdout)
        stderr = _to_text(exc.stderr)
        return AdapterExecutionResult(
            requested=True,
            attempted=True,
            timed_out=True,
            stdout_tail=_tail(stdout),
            stderr_tail=_tail(stderr),
            stdout_sha256=_sha256_text(stdout),
            stderr_sha256=_sha256_text(stderr),
            duration_seconds=time.perf_counter() - started,
            error=f"external command timed out after {inputs.timeout_sec:g} seconds",
        )
    except OSError as exc:
        return AdapterExecutionResult(
            requested=True,
            attempted=True,
            duration_seconds=time.perf_counter() - started,
            error=str(exc),
        )
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    return AdapterExecutionResult(
        requested=True,
        attempted=True,
        returncode=completed.returncode,
        stdout_tail=_tail(stdout),
        stderr_tail=_tail(stderr),
        stdout_sha256=_sha256_text(stdout),
        stderr_sha256=_sha256_text(stderr),
        duration_seconds=time.perf_counter() - started,
        error=None if completed.returncode == 0 else "external command returned non-zero status",
    )


def _format_command(
    command: str,
    config: CalibrationConfig,
    inputs: KoideLidarCameraInputs,
) -> str:
    replacements = {
        "dataset_path": config.dataset.path,
        "result_path": inputs.result_path or "",
        "camera_streams": ",".join(inputs.camera_streams),
        "lidar_streams": ",".join(inputs.lidar_streams),
    }
    formatted = command
    for key, value in replacements.items():
        formatted = formatted.replace("{" + key + "}", value)
    return formatted


def _tail(value: str | None, *, limit: int = 4000) -> str | None:
    if not value:
        return None
    return value[-limit:]


def _to_text(value: str | bytes | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _sha256_text(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_transforms(result_path: str | None) -> dict[str, SE3]:
    """Load either native Koide or the legacy normalized transform mapping.

    This compatibility wrapper intentionally retains the old private helper
    signature; the solver uses :func:`_load_transforms_with_metadata` so native
    parser provenance is not discarded.
    """

    transforms, _native = _load_transforms_with_metadata(
        result_path,
        lidar_frame=None,
        camera_frame=None,
    )
    return transforms


def _load_transforms_with_metadata(
    result_path: str | None,
    *,
    lidar_frame: str | None,
    camera_frame: str | None,
    configured_frames: set[str] | None = None,
) -> tuple[dict[str, SE3], KoideNativeCalibration | None]:
    """Load a result and return parsed transforms plus native metadata."""

    if result_path is None:
        return {}, None
    path = Path(result_path)
    if not path.exists():
        return {}, None
    try:
        payload = read_mapping(path)
    except (OSError, YAMLError, ValueError) as exc:
        raise ValueError(f"could not read Koide result {path}: {exc}") from exc
    if "results" in payload:
        if lidar_frame is None or camera_frame is None:
            raise ValueError(
                "native Koide calib.json requires explicit lidar_frame and "
                "camera_frame from adapter options or uniquely configured streams"
            )
        if configured_frames is not None and (
            lidar_frame not in configured_frames or camera_frame not in configured_frames
        ):
            raise ValueError(
                "native Koide frame binding must name frames declared in the "
                "Calibrex configuration: "
                f"lidar_frame={lidar_frame!r}, camera_frame={camera_frame!r}"
            )
        native = parse_koide_calib_payload(
            payload,
            lidar_frame=lidar_frame,
            camera_frame=camera_frame,
        )
        return native.transforms, native

    transforms = _read_transform_mapping_payload(payload)
    t_camera_lidar = transforms.get("T_camera0_lidar0")
    if t_camera_lidar is None:
        t_lidar_camera = transforms.get("T_lidar0_camera0")
        t_camera_lidar = t_lidar_camera.inverse() if t_lidar_camera is not None else None
    if t_camera_lidar is None:
        return transforms, None
    transforms.setdefault("T_camera0_lidar0", t_camera_lidar)
    transforms.setdefault("T_lidar0_camera0", t_camera_lidar.inverse())
    return transforms, None


def _read_transform_mapping(path: Path) -> dict[str, SE3]:
    payload = read_mapping(path)
    return _read_transform_mapping_payload(payload)


def _read_transform_mapping_payload(payload: dict[str, Any]) -> dict[str, SE3]:
    raw_transforms = payload.get("transforms", payload)
    if not isinstance(raw_transforms, dict):
        return {}
    transforms: dict[str, SE3] = {}
    for name, raw_transform in raw_transforms.items():
        if not isinstance(name, str) or not isinstance(raw_transform, dict):
            continue
        transform = _transform_from_mapping(raw_transform)
        if transform is None:
            continue
        transform_name = _normalize_transform_name(name, raw_transform)
        if transform_name is not None:
            transforms[transform_name] = transform
    return transforms


def _normalize_transform_name(name: str, payload: dict[str, Any]) -> str | None:
    parent = payload.get("parent")
    child = payload.get("child")
    if isinstance(parent, str) and isinstance(child, str):
        return f"T_{parent}_{child}"
    if name.startswith("T_"):
        return name
    return None


def _transform_from_mapping(payload: dict[str, Any]) -> SE3 | None:
    translation = payload.get("translation_m") or payload.get("translation")
    rotation = payload.get("rotation_quat_xyzw") or payload.get("quaternion_xyzw")
    if not isinstance(translation, list) or not isinstance(rotation, list):
        return None
    try:
        return SE3.from_lists(translation, rotation)
    except ValueError:
        return None


def _execution_metric(execution: AdapterExecutionResult) -> MetricResult:
    if not execution.requested:
        return MetricResult(
            value=None,
            grade="warn",
            reason="external execution was not requested",
        )
    if execution.success:
        return MetricResult(
            value=1.0,
            grade="pass",
            reason="external command completed successfully",
        )
    return MetricResult(
        value=0.0,
        grade="fail",
        reason=execution.error or "external command did not complete successfully",
    )


def _with_output_error(
    execution: AdapterExecutionResult,
    error: OSError | ValueError,
) -> AdapterExecutionResult:
    """Preserve execution facts while recording an unreadable output."""

    return AdapterExecutionResult(
        requested=execution.requested,
        attempted=execution.attempted,
        returncode=execution.returncode,
        timed_out=execution.timed_out,
        stdout_tail=execution.stdout_tail,
        stderr_tail=execution.stderr_tail,
        stdout_sha256=execution.stdout_sha256,
        stderr_sha256=execution.stderr_sha256,
        duration_seconds=execution.duration_seconds,
        error=f"external result could not be parsed: {error}",
    )


def _warnings(
    inputs: KoideLidarCameraInputs,
    transforms: dict[str, SE3],
    execution: AdapterExecutionResult,
    result_exists: bool,
) -> list[str]:
    warnings: list[str] = []
    if not inputs.has_camera:
        warnings.append("provide camera images for the targetless LiDAR-camera adapter")
    if not inputs.has_lidar:
        warnings.append("provide LiDAR point clouds for the targetless LiDAR-camera adapter")
    if inputs.result_exists and (
        inputs.camera_frame is None or inputs.lidar_frame is None
    ):
        warnings.append(
            "native Koide calib.json requires explicit camera_frame and lidar_frame "
            "from adapter options or uniquely configured streams"
        )
    if not inputs.command_available and not result_exists:
        warnings.append(
            "configure Koide-style external command or provide a precomputed result_path"
        )
    if execution.error:
        warnings.append(execution.error)
    if execution.success and not transforms:
        warnings.append("external command succeeded but no readable transform was loaded")
    if result_exists and not transforms:
        warnings.append("precomputed adapter result did not contain readable transforms")
    return warnings


def _status(
    inputs: KoideLidarCameraInputs,
    transforms: dict[str, SE3],
    execution: AdapterExecutionResult,
    result_exists: bool,
) -> str:
    if transforms:
        return "result_loaded"
    if result_exists:
        return "executed_no_result"
    if execution.requested:
        if execution.success:
            return "executed_no_result"
        if execution.attempted:
            return "execution_failed"
        return "execution_blocked"
    if inputs.command_available and inputs.input_ready:
        return "ready"
    return "not_executed"


def _path_exists(path: str | None) -> bool:
    return Path(path).exists() if path is not None else False
