"""Materialize provenance-complete Camera--LiDAR baseline readiness audits."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from calibrex import __version__
from calibrex.core.camera_lidar_artifacts import (
    CameraLidarCalibrationProblem,
    load_camera_lidar_benchmark_protocol,
    load_camera_lidar_problem,
)
from calibrex.core.external_run import (
    ExternalArtifactDigest,
    ExternalCalibrationRunArtifact,
    ExternalDataIsolation,
    ExternalExecution,
    ExternalParsedOutputs,
    ExternalRunProvenance,
    ExternalToolIdentity,
)
from calibrex.core.provenance import git_commit, sha256_path

CAMERA_LIDAR_EXTERNAL_BASELINE_AUDIT_VERSION = (
    "calibrex.camera_lidar_external_baseline_audit/v0.1"
)
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class CameraLidarExternalBaselineAuditArtifacts:
    """Paths and typed records from one external-baseline readiness audit."""

    koide_path: Path
    unicalib_path: Path
    koide: ExternalCalibrationRunArtifact
    unicalib: ExternalCalibrationRunArtifact


def materialize_camera_lidar_external_baseline_audit(
    problem_path: str | Path,
    protocol_path: str | Path,
    *,
    output_directory: str | Path,
    koide_source_commit: str,
    unicalib_source_commit: str,
    command: Sequence[str] = (),
) -> CameraLidarExternalBaselineAuditArtifacts:
    """Audit Koide and UniCalib boundaries without inventing run results.

    The emitted external-run artifacts deliberately remain ``not_executed``.
    A comparable baseline requires a separately pinned command/container,
    output transform, and Calibrex-side evaluation on this exact protocol.
    """

    _validate_commit("Koide", koide_source_commit)
    _validate_commit("UniCalib", unicalib_source_commit)
    problem_file = Path(problem_path).resolve()
    protocol_file = Path(protocol_path).resolve()
    problem = load_camera_lidar_problem(problem_file)
    protocol = load_camera_lidar_benchmark_protocol(protocol_file)
    problem_digest = _required_digest(problem_file)
    protocol_digest = _required_digest(protocol_file)
    if protocol.problem_sha256 != problem_digest:
        raise ValueError("external baseline audit protocol does not bind to problem")
    if protocol.frame_ids != [item.frame_id for item in problem.observations]:
        raise ValueError("external baseline audit protocol frame coverage differs")

    output_root = Path(output_directory).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    inputs = [
        _input_artifact(problem_file, problem_digest),
        _input_artifact(protocol_file, protocol_digest),
    ]
    koide = _audit_record(
        problem=problem,
        problem_file=problem_file,
        problem_digest=problem_digest,
        protocol_id=protocol.protocol_id,
        tool_name="direct_visual_lidar_calibration",
        adapter_name="koide_lidar_camera",
        adapter_version="calibrex.koide_lidar_camera_adapter/v0.2",
        repository="https://github.com/koide3/direct_visual_lidar_calibration",
        source_commit=koide_source_commit,
        inputs=inputs,
        isolation=ExternalDataIsolation(
            declared=True,
            evidence=(
                "classical direct-registration baseline; no learned calibration "
                "checkpoint is selected by this audit"
            ),
        ),
        warnings=[
            "readiness audit only; no command, container digest, or transform output was supplied",
            "upstream is ROS/PCL/GTSAM/Ceres based and remains behind a subprocess boundary",
            "no upstream result is bound to the frozen KITTI-family 25-frame protocol",
            "external objective/runtime values are non-comparable until "
            "Calibrex recomputes the frozen metrics",
        ],
        command=command,
    )
    unicalib_warnings = [
        "readiness audit only; no command, container digest, checkpoint "
        "digest, or transform output was supplied",
        "training/checkpoint data IDs are not digest-bound, so evaluation isolation is unresolved",
        "the CUDA visibility extension and learned runtime remain behind a subprocess boundary",
        "external t_Mean/R_Mean values are non-comparable until Calibrex "
        "recomputes the frozen metrics",
    ]
    if problem.dataset_family == "KITTI-360":
        unicalib_warnings.append(
            "the audited problem uses image_03 MEI fisheye, while the upstream "
            "documented KITTI-360 layout uses image_00/data_rect"
        )
    unicalib = _audit_record(
        problem=problem,
        problem_file=problem_file,
        problem_digest=problem_digest,
        protocol_id=protocol.protocol_id,
        tool_name="UniCalib",
        adapter_name="unicalib_lidar_camera",
        adapter_version="calibrex.unicalib_lidar_camera_adapter/v0.1",
        repository="https://github.com/han-15/UniCalib",
        source_commit=unicalib_source_commit,
        inputs=inputs,
        isolation=ExternalDataIsolation(
            declared=False,
            evidence=(
                "upstream repository was audited, but no checkpoint identity or "
                "training/evaluation ID manifests were supplied"
            ),
        ),
        warnings=unicalib_warnings,
        command=command,
    )
    koide_path = output_root / "koide.external-run.yaml"
    unicalib_path = output_root / "unicalib.external-run.yaml"
    koide.save(koide_path)
    unicalib.save(unicalib_path)
    return CameraLidarExternalBaselineAuditArtifacts(
        koide_path=koide_path,
        unicalib_path=unicalib_path,
        koide=koide,
        unicalib=unicalib,
    )


def _audit_record(
    *,
    problem: CameraLidarCalibrationProblem,
    problem_file: Path,
    problem_digest: str,
    protocol_id: str,
    tool_name: str,
    adapter_name: str,
    adapter_version: str,
    repository: str,
    source_commit: str,
    inputs: list[ExternalArtifactDigest],
    isolation: ExternalDataIsolation,
    warnings: list[str],
    command: Sequence[str],
) -> ExternalCalibrationRunArtifact:
    reference = problem.reference_transform_camera_lidar
    audit_command = list(command)
    return ExternalCalibrationRunArtifact(
        run_id=f"{problem.dataset_id}:{adapter_name}:readiness-audit",
        adapter_name=adapter_name,
        adapter_version=adapter_version,
        tool=ExternalToolIdentity(
            name=tool_name,
            version=f"git-{source_commit[:12]}",
            source_repository=repository,
            source_commit=source_commit,
            license_spdx="MIT",
            license_boundary="subprocess",
        ),
        execution=ExternalExecution(
            mode="subprocess",
            command=audit_command,
            attempted=False,
        ),
        artifacts=[item.model_copy(deep=True) for item in inputs],
        frame_convention=(
            "T_parent_child; required output "
            f"T_{reference.parent}_{reference.child}"
        ),
        time_convention=problem.time_convention,
        train_data_isolation=isolation,
        status="not_executed",
        warnings=[
            *warnings,
            f"required Calibrex protocol: {protocol_id}",
            "this artifact is not an accuracy or runtime result",
        ],
        parsed_outputs=ExternalParsedOutputs(),
        provenance=ExternalRunProvenance(
            calibrex_version=__version__,
            git_commit=git_commit(),
            source_artifact=str(problem_file),
            source_artifact_sha256=problem_digest,
        ),
    )


def _input_artifact(path: Path, digest: str) -> ExternalArtifactDigest:
    media_type: Literal["application/yaml"] = "application/yaml"
    return ExternalArtifactDigest(
        role="input",
        path=str(path),
        sha256=digest,
        size_bytes=path.stat().st_size,
        media_type=media_type,
    )


def _validate_commit(name: str, value: str) -> None:
    if _COMMIT_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} source commit must be 40 lowercase hex characters")


def _required_digest(path: Path) -> str:
    digest = sha256_path(path)
    if digest is None or not path.is_file():
        raise ValueError(f"external baseline audit input is unreadable: {path}")
    return digest
