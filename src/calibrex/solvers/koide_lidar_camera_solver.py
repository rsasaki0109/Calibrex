"""Koide-style targetless LiDAR-camera adapter boundary.

This module intentionally does not vendor or import the external implementation.
It records readiness, license boundary, and optional precomputed outputs in the
Calibrex result schema so strong external baselines can be compared safely.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from calibrex.core.config import CalibrationConfig, FactorConfig
from calibrex.core.frames import FrameGraph
from calibrex.core.geometry import SE3
from calibrex.core.io import read_mapping
from calibrex.core.provenance import sha256_path
from calibrex.core.result import MetricResult
from calibrex.data.base import StreamSummary
from calibrex.data.inspect import DatasetInspection
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
        execution = _execute_command(config, inputs) if inputs.execute else AdapterExecutionResult()
        transforms = _load_transforms(inputs.result_path)
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
        status = _status(inputs, transforms, execution)
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
                "koide_lidar_camera_tool_identity": identity.as_dict(),
                "license_boundary": (
                    "external subprocess/precomputed-result adapter; no external "
                    "calibration code is copied into Calibrex core"
                ),
            },
            warnings=warnings,
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


def _external_tool_identity(
    config: CalibrationConfig,
    inputs: KoideLidarCameraInputs,
) -> KoideExternalToolIdentity:
    options = _adapter_options(config)
    result_path = Path(inputs.result_path) if inputs.result_path is not None else None
    result_exists = result_path is not None and result_path.is_file()
    return KoideExternalToolIdentity(
        tool_name=_text_option(options, "tool_name") or "koide_lidar_camera",
        tool_version=_text_option(options, "tool_version"),
        source_repository=_text_option(options, "source_repository"),
        source_commit=_text_option(options, "source_commit"),
        license_spdx=_text_option(options, "license_spdx"),
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
        return AdapterExecutionResult(
            requested=True,
            attempted=True,
            timed_out=True,
            stdout_tail=_tail(_to_text(exc.stdout)),
            stderr_tail=_tail(_to_text(exc.stderr)),
            error=f"external command timed out after {inputs.timeout_sec:g} seconds",
        )
    except OSError as exc:
        return AdapterExecutionResult(
            requested=True,
            attempted=True,
            error=str(exc),
        )
    return AdapterExecutionResult(
        requested=True,
        attempted=True,
        returncode=completed.returncode,
        stdout_tail=_tail(completed.stdout),
        stderr_tail=_tail(completed.stderr),
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


def _load_transforms(result_path: str | None) -> dict[str, SE3]:
    if result_path is None:
        return {}
    path = Path(result_path)
    if not path.exists():
        return {}
    transforms = _read_transform_mapping(path)
    t_camera_lidar = transforms.get("T_camera0_lidar0")
    if t_camera_lidar is None:
        t_lidar_camera = transforms.get("T_lidar0_camera0")
        t_camera_lidar = t_lidar_camera.inverse() if t_lidar_camera is not None else None
    if t_camera_lidar is None:
        return transforms
    transforms.setdefault("T_camera0_lidar0", t_camera_lidar)
    transforms.setdefault("T_lidar0_camera0", t_camera_lidar.inverse())
    return transforms


def _read_transform_mapping(path: Path) -> dict[str, SE3]:
    payload = read_mapping(path)
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
    if not inputs.command_available and not result_exists:
        warnings.append(
            "configure Koide-style external command or provide a precomputed result_path"
        )
    if execution.requested and not execution.success:
        warnings.append(execution.error or "external command did not complete successfully")
    if execution.success and not transforms:
        warnings.append("external command succeeded but no readable transform was loaded")
    if result_exists and not transforms:
        warnings.append("precomputed adapter result did not contain readable transforms")
    return warnings


def _status(
    inputs: KoideLidarCameraInputs,
    transforms: dict[str, SE3],
    execution: AdapterExecutionResult,
) -> str:
    if transforms:
        return "result_loaded"
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
