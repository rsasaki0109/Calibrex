"""ROS-independent process and transform helpers for external solver adapters."""

from __future__ import annotations

import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from calibrex.core.geometry import SE3
from calibrex.core.io import read_mapping


@dataclass(frozen=True)
class ExternalProcessResult:
    """Captured result of an external process without importing its runtime."""

    requested: bool = True
    attempted: bool = False
    returncode: int | None = None
    timed_out: bool = False
    stdout_tail: str | None = None
    stderr_tail: str | None = None
    error: str | None = None
    duration_seconds: float | None = None

    @property
    def success(self) -> bool:
        """Return whether the command was attempted and exited successfully."""

        return self.attempted and not self.timed_out and self.returncode == 0


def external_command_argv(value: object) -> tuple[str, ...]:
    """Normalize a string/list command without enabling a shell."""

    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(shlex.split(value))
    if isinstance(value, list | tuple):
        return tuple(str(part) for part in value if str(part))
    return (str(value),)


def external_command_available(argv: tuple[str, ...]) -> bool:
    """Return whether an external command's executable can be resolved."""

    if not argv:
        return False
    executable = argv[0]
    return shutil.which(executable) is not None or Path(executable).is_file()


def run_external_command(
    argv: tuple[str, ...],
    *,
    working_directory: str | Path | None,
    timeout_seconds: float,
    readiness_error: str | None = None,
) -> ExternalProcessResult:
    """Run an argv-only subprocess and capture bounded output tails."""

    if not argv:
        return ExternalProcessResult(error="command is not configured")
    if not external_command_available(argv):
        return ExternalProcessResult(error="command executable is not available")
    if readiness_error is not None:
        return ExternalProcessResult(error=readiness_error)
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            list(argv),
            check=False,
            capture_output=True,
            cwd=working_directory,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return ExternalProcessResult(
            attempted=True,
            timed_out=True,
            stdout_tail=_tail(_to_text(exc.stdout)),
            stderr_tail=_tail(_to_text(exc.stderr)),
            error=f"external command timed out after {timeout_seconds:g} seconds",
            duration_seconds=time.perf_counter() - started,
        )
    except OSError as exc:
        return ExternalProcessResult(
            attempted=True,
            error=str(exc),
            duration_seconds=time.perf_counter() - started,
        )
    return ExternalProcessResult(
        attempted=True,
        returncode=completed.returncode,
        stdout_tail=_tail(completed.stdout),
        stderr_tail=_tail(completed.stderr),
        error=(
            None
            if completed.returncode == 0
            else "external command returned non-zero status"
        ),
        duration_seconds=time.perf_counter() - started,
    )


def load_external_transform_mapping(path: str | Path) -> dict[str, SE3]:
    """Load Calibrex-style transform mappings from YAML or JSON."""

    payload = read_mapping(Path(path))
    raw_transforms = payload.get("transforms", payload)
    if not isinstance(raw_transforms, dict):
        raise ValueError("external result transforms must be a mapping")
    transforms: dict[str, SE3] = {}
    for name, raw_transform in raw_transforms.items():
        if not isinstance(name, str) or not isinstance(raw_transform, dict):
            continue
        normalized_name = _transform_name(name, raw_transform)
        transform = _transform_from_mapping(raw_transform)
        if normalized_name is not None and transform is not None:
            transforms[normalized_name] = transform
    if not transforms:
        raise ValueError("external result did not contain a readable transform")
    return transforms


def _transform_name(name: str, payload: dict[str, Any]) -> str | None:
    parent = payload.get("parent")
    child = payload.get("child")
    if isinstance(parent, str) and isinstance(child, str):
        return f"T_{parent}_{child}"
    return name if name.startswith("T_") else None


def _transform_from_mapping(payload: dict[str, Any]) -> SE3 | None:
    translation = payload.get("translation_m") or payload.get("translation")
    rotation = payload.get("rotation_quat_xyzw") or payload.get("quaternion_xyzw")
    if not isinstance(translation, list) or not isinstance(rotation, list):
        return None
    try:
        return SE3.from_lists(translation, rotation)
    except ValueError:
        return None


def _tail(value: str | None, *, limit: int = 4000) -> str | None:
    return value[-limit:] if value else None


def _to_text(value: str | bytes | None) -> str | None:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
