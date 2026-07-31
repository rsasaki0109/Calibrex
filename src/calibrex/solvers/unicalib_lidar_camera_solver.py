"""Optional UniCalib LiDAR-camera adapter with no learned-runtime dependency."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

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
    ExternalToolIdentity,
    ExternalTransformOutput,
)
from calibrex.core.frames import FrameGraph
from calibrex.core.geometry import SE3
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.result import Grade, MetricResult
from calibrex.data.base import StreamSummary
from calibrex.data.inspect import DatasetInspection
from calibrex.solvers.base import SolverAdapter, SolverAdapterResult
from calibrex.solvers.external_adapter import (
    ExternalProcessResult,
    external_command_argv,
    external_command_available,
    load_external_transform_mapping,
    run_external_command,
)

UNICALIB_ADAPTER_VERSION = "calibrex.unicalib_lidar_camera_adapter/v0.1"
UNICALIB_FACTOR_NAMES = ("unicalib_lidar_camera", "unicalib")


class UniCalibLidarCameraSolver(SolverAdapter):
    """Execute or import UniCalib behind a schema-versioned adapter boundary."""

    backend = "unicalib_lidar_camera"

    def solve(
        self,
        config: CalibrationConfig,
        frame_graph: FrameGraph,
        inspection: DatasetInspection,
    ) -> SolverAdapterResult:
        """Run/import UniCalib and emit independently evaluable transforms."""

        del frame_graph
        factor_name, factor = _factor(config)
        options = factor.options if factor is not None else {}
        camera_streams = _streams(inspection.streams, kind="camera")
        lidar_streams = _streams(inspection.streams, kind="lidar")
        input_ready = bool(camera_streams and lidar_streams)
        result_path = _path_option(options, "result_path")
        command = _formatted_command(
            external_command_argv(options.get("command")),
            config=config,
            result_path=result_path,
            camera_streams=camera_streams,
            lidar_streams=lidar_streams,
        )
        execute = _bool_option(options, "execute", default=False)
        timeout = _float_option(options, "timeout_sec", default=600.0)
        execution = (
            run_external_command(
                command,
                working_directory=_text_option(options, "working_dir"),
                timeout_seconds=timeout,
                readiness_error=(
                    None
                    if input_ready
                    else "camera and LiDAR input streams are not both available"
                ),
            )
            if execute
            else ExternalProcessResult(requested=False)
        )
        transforms, parse_warning = _load_result(result_path)
        if "T_camera0_lidar0" in transforms:
            transforms.setdefault(
                "T_lidar0_camera0", transforms["T_camera0_lidar0"].inverse()
            )
        elif "T_lidar0_camera0" in transforms:
            transforms.setdefault(
                "T_camera0_lidar0", transforms["T_lidar0_camera0"].inverse()
            )
        warnings = _warnings(
            input_ready=input_ready,
            command=command,
            execute=execute,
            execution=execution,
            result_path=result_path,
            transforms=transforms,
            parse_warning=parse_warning,
        )
        artifact = _artifact(
            config=config,
            factor_name=factor_name,
            options=options,
            command=command,
            timeout=timeout,
            execution=execution,
            result_path=result_path,
            transforms=transforms,
            warnings=warnings,
        )
        artifact_path = _text_option(options, "external_run_output")
        if artifact_path is not None:
            artifact.save(artifact_path)
        available = external_command_available(command) or (
            result_path is not None and result_path.is_file()
        )
        status = _solver_status(execution, transforms)
        provenance_complete = _provenance_complete(artifact)
        return SolverAdapterResult(
            backend=self.backend,
            available=available,
            status=status,
            metrics={
                "unicalib_adapter_available": _binary_metric(
                    available,
                    passed="UniCalib command or precomputed result is available",
                    failed="configure UniCalib command or result_path",
                    false_grade="warn",
                ),
                "unicalib_input_ready": _binary_metric(
                    input_ready,
                    passed="camera and LiDAR streams are available",
                    failed="camera and LiDAR streams are required",
                ),
                "unicalib_result_available": _binary_metric(
                    bool(transforms),
                    passed="UniCalib transform was loaded",
                    failed="no readable UniCalib transform was loaded",
                    false_grade="warn",
                ),
                "unicalib_execution_success": _execution_metric(execution),
                "unicalib_provenance_complete": _binary_metric(
                    provenance_complete,
                    passed="UniCalib identity, license, commit, and digests are complete",
                    failed="UniCalib provenance is incomplete",
                    false_grade="warn",
                ),
            },
            transforms=transforms,
            provenance={
                "unicalib_status": status,
                "unicalib_factor": factor_name,
                "unicalib_execution": artifact.execution.model_dump(
                    mode="json", exclude_none=True
                ),
                "unicalib_tool_identity": artifact.tool.model_dump(
                    mode="json", exclude_none=True
                ),
                "external_calibration_run": artifact.model_dump(
                    mode="json", exclude_none=True
                ),
                "external_calibration_run_path": artifact_path,
                "license_boundary": (
                    "MIT external subprocess/precomputed-result adapter; UniCalib "
                    "runtime and weights are not imported into Calibrex core"
                ),
            },
            warnings=warnings,
        )


def _artifact(
    *,
    config: CalibrationConfig,
    factor_name: str | None,
    options: dict[str, Any],
    command: tuple[str, ...],
    timeout: float,
    execution: ExternalProcessResult,
    result_path: Path | None,
    transforms: dict[str, SE3],
    warnings: list[str],
) -> ExternalCalibrationRunArtifact:
    tool = ExternalToolIdentity(
        name=_text_option(options, "tool_name") or "UniCalib",
        version=_text_option(options, "tool_version"),
        source_repository=_text_option(options, "source_repository")
        or "https://github.com/han-15/UniCalib",
        source_commit=_text_option(options, "source_commit"),
        license_spdx=_text_option(options, "license_spdx") or "MIT",
        license_boundary=("subprocess" if execution.requested else "imported"),
    )
    artifacts: list[ExternalArtifactDigest] = []
    artifact_warnings = list(warnings)
    for input_path in _path_list_option(options, "input_paths"):
        artifact = _artifact_digest(input_path, role="input")
        if artifact is None:
            artifact_warnings.append(
                f"external-run file does not exist or is not regular: {input_path}"
            )
        else:
            artifacts.append(artifact)
    digest_failed = False
    if result_path is not None and result_path.is_file():
        output = _artifact_digest(result_path, role="output")
        if output is not None:
            artifacts.append(output)
        expected_digest = _text_option(options, "expected_result_sha256")
        digest_failed = expected_digest is not None and (
            output is None or output.sha256 != expected_digest
        )
        if digest_failed:
            artifact_warnings.append("UniCalib result digest mismatch")
    if not any(item.role == "input" for item in artifacts):
        artifact_warnings.append("no digest-pinned input_paths were declared")
    parsed_transforms = {
        name: ExternalTransformOutput(
            parent=frames[0],
            child=frames[1],
            translation_m=list(transform.translation_m),
            rotation_quat_xyzw=list(transform.rotation_quat_xyzw),
        )
        for name, transform in sorted(transforms.items())
        if (frames := _frames(name)) is not None
    }
    external_metrics: dict[str, float | str | bool | None] = {
        name: float(value)
        for name, value in _mapping_option(options, "external_metrics").items()
        if isinstance(name, str) and isinstance(value, int | float)
    }
    mode = _text_option(options, "execution_mode")
    if mode not in {"subprocess", "container", "precomputed", "imported"}:
        mode = "subprocess" if execution.requested else "precomputed"
    status = (
        "digest_mismatch"
        if digest_failed
        else "timeout"
        if execution.timed_out
        else "failed"
        if execution.requested and execution.attempted and not execution.success
        else "unavailable"
        if execution.requested and not execution.success
        else "success"
        if parsed_transforms
        else "not_executed"
    )
    output_sha256 = sha256_path(result_path) if result_path is not None else None
    return ExternalCalibrationRunArtifact(
        run_id=_text_option(options, "external_run_id")
        or f"unicalib-{factor_name or 'adapter'}",
        adapter_name="unicalib_lidar_camera",
        adapter_version=UNICALIB_ADAPTER_VERSION,
        tool=tool,
        execution=ExternalExecution(
            mode=cast(ExternalExecutionMode, mode),
            command=list(command),
            working_directory=_text_option(options, "working_dir"),
            container_digest=_text_option(options, "container_digest"),
            timeout_seconds=timeout if execution.requested else None,
            attempted=execution.attempted,
            return_code=execution.returncode,
            timed_out=execution.timed_out,
            duration_seconds=execution.duration_seconds,
            stdout_tail=execution.stdout_tail,
            stderr_tail=execution.stderr_tail,
            error=execution.error,
        ),
        artifacts=artifacts,
        frame_convention="T_parent_child",
        time_convention=_text_option(options, "time_convention")
        or "static transform; no time offset estimated",
        train_data_isolation=ExternalDataIsolation(
            declared=_bool_option(options, "training_isolation_declared", default=False),
            evidence=_text_option(options, "training_isolation_evidence"),
            training_data_ids_sha256=_text_option(options, "training_data_sha256"),
            holdout_data_ids_sha256=_text_option(options, "dataset_sha256"),
        ),
        status=cast(ExternalRunStatus, status),
        warnings=list(dict.fromkeys(artifact_warnings)),
        parsed_outputs=ExternalParsedOutputs(
            transforms=parsed_transforms,
            external_metrics=external_metrics,
        ),
        provenance=ExternalRunProvenance(
            calibrex_version=__version__,
            git_commit=git_commit(),
            source_artifact=str(result_path) if result_path is not None else None,
            source_artifact_sha256=output_sha256,
        ),
    )


def _factor(config: CalibrationConfig) -> tuple[str | None, FactorConfig | None]:
    for name in UNICALIB_FACTOR_NAMES:
        factor = config.pipeline.factors.get(name)
        if factor is not None and factor.enabled:
            return name, factor
    return None, None


def _streams(streams: list[StreamSummary], *, kind: str) -> tuple[str, ...]:
    names = []
    for stream in streams:
        stream_name = stream.name.lower()
        stream_kind = stream.kind.lower()
        is_kind = (
            stream_kind == "image" or "camera" in stream_name
            if kind == "camera"
            else stream_kind == "pointcloud"
            or "lidar" in stream_name
            or "velodyne" in stream_name
        )
        if is_kind and stream.message_count != 0:
            names.append(stream.name)
    return tuple(names)


def _formatted_command(
    command: tuple[str, ...],
    *,
    config: CalibrationConfig,
    result_path: Path | None,
    camera_streams: tuple[str, ...],
    lidar_streams: tuple[str, ...],
) -> tuple[str, ...]:
    values = {
        "dataset_path": config.dataset.path,
        "result_path": str(result_path) if result_path is not None else "",
        "camera_streams": ",".join(camera_streams),
        "lidar_streams": ",".join(lidar_streams),
    }
    output = []
    for part in command:
        for key, value in values.items():
            part = part.replace("{" + key + "}", value)
        output.append(part)
    return tuple(output)


def _load_result(path: Path | None) -> tuple[dict[str, SE3], str | None]:
    if path is None or not path.exists():
        return {}, None
    try:
        return load_external_transform_mapping(path), None
    except (OSError, ValueError) as exc:
        return {}, f"malformed UniCalib result: {exc}"


def _warnings(
    *,
    input_ready: bool,
    command: tuple[str, ...],
    execute: bool,
    execution: ExternalProcessResult,
    result_path: Path | None,
    transforms: dict[str, SE3],
    parse_warning: str | None,
) -> list[str]:
    warnings = []
    if not input_ready:
        warnings.append("provide camera images and LiDAR point clouds")
    if not external_command_available(command) and (
        result_path is None or not result_path.is_file()
    ):
        warnings.append("configure UniCalib command or provide result_path")
    if execute and not execution.success:
        warnings.append(execution.error or "UniCalib command did not complete successfully")
    if parse_warning is not None:
        warnings.append(parse_warning)
    if execution.success and not transforms:
        warnings.append("UniCalib command succeeded but no readable transform was loaded")
    return warnings


def _solver_status(
    execution: ExternalProcessResult,
    transforms: dict[str, SE3],
) -> str:
    if transforms:
        return "result_loaded"
    if execution.requested and execution.attempted:
        return "execution_failed"
    if execution.requested:
        return "execution_blocked"
    return "not_executed"


def _frames(name: str) -> tuple[str, str] | None:
    parts = name[2:].split("_", maxsplit=1) if name.startswith("T_") else []
    return (parts[0], parts[1]) if len(parts) == 2 and all(parts) else None


def _artifact_digest(
    path: Path,
    *,
    role: Literal["input", "output"],
) -> ExternalArtifactDigest | None:
    digest = sha256_path(path)
    if digest is None or not path.is_file():
        return None
    return ExternalArtifactDigest(
        role=role,
        path=str(path),
        sha256=digest,
        size_bytes=path.stat().st_size,
    )


def _provenance_complete(artifact: ExternalCalibrationRunArtifact) -> bool:
    return bool(
        artifact.tool.version
        and artifact.tool.source_commit
        and artifact.tool.license_spdx
        and artifact.artifacts
        and artifact.status != "digest_mismatch"
    )


def _binary_metric(
    passed_value: bool,
    *,
    passed: str,
    failed: str,
    false_grade: Grade = "fail",
) -> MetricResult:
    return MetricResult(
        value=1.0 if passed_value else 0.0,
        grade="pass" if passed_value else false_grade,
        reason=passed if passed_value else failed,
    )


def _execution_metric(execution: ExternalProcessResult) -> MetricResult:
    if not execution.requested:
        return MetricResult(
            value=None,
            grade="warn",
            reason="external execution was not requested",
        )
    return _binary_metric(
        execution.success,
        passed="UniCalib command completed successfully",
        failed=execution.error or "UniCalib command failed",
    )


def _text_option(options: dict[str, Any], key: str) -> str | None:
    value = options.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _path_option(options: dict[str, Any], key: str) -> Path | None:
    text = _text_option(options, key)
    return Path(text) if text is not None else None


def _path_list_option(options: dict[str, Any], key: str) -> tuple[Path, ...]:
    value = options.get(key, [])
    if isinstance(value, str):
        return (Path(value),) if value.strip() else ()
    if not isinstance(value, list | tuple):
        return ()
    return tuple(Path(str(item)) for item in value if str(item).strip())


def _mapping_option(options: dict[str, Any], key: str) -> dict[str, Any]:
    value = options.get(key, {})
    return value if isinstance(value, dict) else {}


def _bool_option(options: dict[str, Any], key: str, *, default: bool) -> bool:
    value = options.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _float_option(options: dict[str, Any], key: str, *, default: float) -> float:
    try:
        return max(float(options.get(key, default)), 0.1)
    except (TypeError, ValueError):
        return default
