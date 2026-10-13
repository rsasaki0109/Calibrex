"""License-safe optional GICP and NDT registration adapter boundaries."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import shlex
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3, quaternion_xyzw_from_rotation_matrix
from calibrex.core.io import read_mapping
from calibrex.core.provenance import sha256_path
from calibrex.solvers.robust_point_to_point_icp_solver import (
    IcpCandidateEvaluation,
    IcpPoint,
    RobustPointToPointIcpOptions,
    evaluate_icp_candidate,
    split_icp_source_points,
)

RegistrationAdapterStatus = Literal[
    "converged", "result_loaded", "unavailable", "not_executed", "failed"
]
NdtExecutionStatus = Literal["completed", "timeout", "os_error", "not_requested"]


@dataclass(frozen=True)
class NdtExecutionEvidence:
    """Bounded subprocess evidence without embedding arbitrary tool output."""

    status: NdtExecutionStatus
    returncode: int | None = None
    elapsed_sec: float | None = None
    stdout_size_bytes: int = 0
    stderr_size_bytes: int = 0
    stdout_sha256: str | None = None
    stderr_sha256: str | None = None

    def as_dict(self) -> dict[str, object]:
        return self.__dict__


@dataclass(frozen=True)
class RegistrationAdapterResult:
    backend: str
    status: RegistrationAdapterStatus
    available: bool
    transform_target_source: SE3 | None
    common_evaluation: IcpCandidateEvaluation | None
    raw_metrics: dict[str, float | int | str | bool | None]
    provenance: dict[str, object]
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "status": self.status,
            "available": self.available,
            "transform_target_source": (
                self.transform_target_source.as_dict()
                if self.transform_target_source is not None
                else None
            ),
            "common_evaluation": (
                self.common_evaluation.as_dict() if self.common_evaluation is not None else None
            ),
            "raw_metrics": self.raw_metrics,
            "provenance": self.provenance,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class Open3DGeneralizedIcpOptions:
    correspondence_distance_m: float = 1.0
    max_iterations: int = 50


class Open3DGeneralizedIcpAdapter:
    """Run Open3D GICP on train points and recompute Calibrex common evidence."""

    backend = "open3d_generalized_icp"

    def solve(
        self,
        source_points: list[IcpPoint],
        target_points: list[IcpPoint],
        initial_transform: SE3 | None = None,
        *,
        common_options: RobustPointToPointIcpOptions | None = None,
        options: Open3DGeneralizedIcpOptions | None = None,
    ) -> RegistrationAdapterResult:
        registration_options = options or Open3DGeneralizedIcpOptions()
        evaluation_options = common_options or RobustPointToPointIcpOptions()
        input_provenance = _point_input_provenance(source_points, target_points)
        available = importlib.util.find_spec("open3d") is not None
        if not available:
            return RegistrationAdapterResult(
                backend=self.backend,
                status="unavailable",
                available=False,
                transform_target_source=None,
                common_evaluation=None,
                raw_metrics={},
                provenance={
                    "license_spdx": "MIT",
                    "adapter_boundary": "optional Python import; no Open3D code vendored",
                    "inputs": input_provenance,
                    "options": registration_options.__dict__,
                },
                warnings=("install calibrex[open3d] to execute Generalized ICP",),
            )
        train_source, _holdout = split_icp_source_points(source_points, evaluation_options)
        if not train_source or not target_points:
            return RegistrationAdapterResult(
                self.backend,
                "failed",
                True,
                None,
                None,
                {},
                {
                    "license_spdx": "MIT",
                    "inputs": input_provenance,
                    "options": registration_options.__dict__,
                },
                ("GICP requires non-empty train source and target clouds",),
            )
        open3d = importlib.import_module("open3d")
        source_cloud = _open3d_cloud(open3d, train_source)
        target_cloud = _open3d_cloud(open3d, target_points)
        initial = initial_transform or SE3.identity()
        result = open3d.pipelines.registration.registration_generalized_icp(
            source_cloud,
            target_cloud,
            registration_options.correspondence_distance_m,
            _matrix4(initial),
            open3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            open3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=registration_options.max_iterations
            ),
        )
        transform = _se3_from_matrix4(np.asarray(result.transformation, dtype=float))
        common = evaluate_icp_candidate(source_points, target_points, transform, evaluation_options)
        version = str(getattr(open3d, "__version__", "unknown"))
        return RegistrationAdapterResult(
            backend=self.backend,
            status="converged",
            available=True,
            transform_target_source=transform,
            common_evaluation=common,
            raw_metrics={
                "open3d_fitness": float(result.fitness),
                "open3d_inlier_rmse": float(result.inlier_rmse),
            },
            provenance={
                "tool_name": "Open3D",
                "tool_version": version,
                "license_spdx": "MIT",
                "backend_metric_policy": (
                    "Open3D fitness retained as raw provenance; comparisons use "
                    "Calibrex spatial holdout"
                ),
                "train_source_count": len(train_source),
                "external_code_vendored": False,
                "inputs": input_provenance,
                "train_source_sha256": _point_cloud_sha256(train_source),
                "options": registration_options.__dict__,
            },
        )


@dataclass(frozen=True)
class NdtSubprocessOptions:
    result_path: Path
    command: tuple[str, ...] = ()
    execute: bool = False
    timeout_sec: float = 600.0
    working_dir: Path | None = None
    tool_name: str = "external_ndt"
    tool_version: str = "unknown"
    license_spdx: str = "unknown"
    training_isolation_declared: bool = False


class NdtSubprocessAdapter:
    """Load or execute an external PCL/Autoware NDT result without linking code."""

    backend = "ndt_subprocess"

    def solve(
        self,
        source_points: list[IcpPoint],
        target_points: list[IcpPoint],
        options: NdtSubprocessOptions,
        *,
        common_options: RobustPointToPointIcpOptions | None = None,
    ) -> RegistrationAdapterResult:
        execution = (
            _execute_ndt(options) if options.execute else NdtExecutionEvidence("not_requested")
        )
        provenance = _ndt_provenance(options, execution, source_points, target_points)
        transform = _load_registration_transform(options.result_path)
        if transform is None:
            status: RegistrationAdapterStatus = "failed" if options.execute else "not_executed"
            return RegistrationAdapterResult(
                backend=self.backend,
                status=status,
                available=bool(options.command) or options.result_path.exists(),
                transform_target_source=None,
                common_evaluation=None,
                raw_metrics={"returncode": execution.returncode},
                provenance=provenance,
                warnings=("external NDT transform result is unavailable or invalid",),
            )
        evaluation_options = common_options or RobustPointToPointIcpOptions()
        common = evaluate_icp_candidate(source_points, target_points, transform, evaluation_options)
        warnings: tuple[str, ...] = ()
        if not options.training_isolation_declared:
            warnings += ("external NDT train/holdout isolation is not declared",)
        if options.license_spdx == "unknown":
            warnings += ("external NDT license SPDX is unknown",)
        return RegistrationAdapterResult(
            backend=self.backend,
            status="result_loaded",
            available=True,
            transform_target_source=transform,
            common_evaluation=common,
            raw_metrics={"returncode": execution.returncode},
            provenance=provenance,
            warnings=warnings,
        )


def _open3d_cloud(open3d: Any, points: list[IcpPoint]) -> Any:
    cloud = open3d.geometry.PointCloud()
    cloud.points = open3d.utility.Vector3dVector(
        np.asarray([point.position_m for point in points], dtype=float)
    )
    return cloud


def _matrix4(transform: SE3) -> NDArray[np.float64]:
    x, y, z, w = transform.rotation_quat_xyzw
    rotation: NDArray[np.float64] = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    matrix: NDArray[np.float64] = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = transform.translation_m
    return matrix


def _se3_from_matrix4(matrix: NDArray[np.float64]) -> SE3:
    if matrix.shape != (4, 4):
        raise ValueError("registration transform must be 4x4")
    return SE3(
        (float(matrix[0, 3]), float(matrix[1, 3]), float(matrix[2, 3])),
        quaternion_xyzw_from_rotation_matrix(matrix[:3, :3].reshape(-1)),
    )


def _execute_ndt(options: NdtSubprocessOptions) -> NdtExecutionEvidence:
    if not options.command:
        return NdtExecutionEvidence("os_error")
    command = [part.format(result_path=str(options.result_path)) for part in options.command]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            cwd=options.working_dir,
            timeout=options.timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        return _execution_evidence(
            "timeout",
            None,
            time.monotonic() - started,
            exc.stdout or b"",
            exc.stderr or b"",
        )
    except OSError:
        return NdtExecutionEvidence("os_error", elapsed_sec=time.monotonic() - started)
    return _execution_evidence(
        "completed",
        completed.returncode,
        time.monotonic() - started,
        completed.stdout,
        completed.stderr,
    )


def _load_registration_transform(path: Path) -> SE3 | None:
    if not path.exists():
        return None
    try:
        payload = read_mapping(path)
    except Exception:
        return None
    raw: object = payload.get("transform_target_source", payload.get("transform"))
    if not isinstance(raw, dict):
        return None
    translation = raw.get("translation_m")
    rotation = raw.get("rotation_quat_xyzw")
    if not isinstance(translation, list) or not isinstance(rotation, list):
        return None
    try:
        return SE3.from_lists(translation, rotation)
    except ValueError:
        return None


def _ndt_provenance(
    options: NdtSubprocessOptions,
    execution: NdtExecutionEvidence,
    source_points: list[IcpPoint],
    target_points: list[IcpPoint],
) -> dict[str, object]:
    return {
        "tool_name": options.tool_name,
        "tool_version": options.tool_version,
        "license_spdx": options.license_spdx,
        "adapter_boundary": "external subprocess/precomputed transform",
        "command": shlex.join(options.command) if options.command else None,
        "result_path": str(options.result_path),
        "result_sha256": (
            sha256_path(options.result_path) if options.result_path.exists() else None
        ),
        "training_isolation_declared": options.training_isolation_declared,
        "external_code_vendored": False,
        "backend_metric_policy": "external scores are not treated as common metrics",
        "inputs": _point_input_provenance(source_points, target_points),
        "execution": execution.as_dict(),
        "options": {
            "execute": options.execute,
            "timeout_sec": options.timeout_sec,
            "working_dir": str(options.working_dir) if options.working_dir else None,
        },
    }


def _execution_evidence(
    status: NdtExecutionStatus,
    returncode: int | None,
    elapsed_sec: float,
    stdout: bytes,
    stderr: bytes,
) -> NdtExecutionEvidence:
    return NdtExecutionEvidence(
        status=status,
        returncode=returncode,
        elapsed_sec=elapsed_sec,
        stdout_size_bytes=len(stdout),
        stderr_size_bytes=len(stderr),
        stdout_sha256=hashlib.sha256(stdout).hexdigest(),
        stderr_sha256=hashlib.sha256(stderr).hexdigest(),
    )


def _point_input_provenance(
    source_points: list[IcpPoint], target_points: list[IcpPoint]
) -> dict[str, object]:
    return {
        "source_count": len(source_points),
        "source_sha256": _point_cloud_sha256(source_points),
        "target_count": len(target_points),
        "target_sha256": _point_cloud_sha256(target_points),
        "hash_policy": "ordered point ID UTF-8 with NUL separator and big-endian float64 xyz",
    }


def _point_cloud_sha256(points: list[IcpPoint]) -> str:
    digest = hashlib.sha256()
    for point in points:
        digest.update(point.point_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(struct.pack(">ddd", *point.position_m))
    return digest.hexdigest()
