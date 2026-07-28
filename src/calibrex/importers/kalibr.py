"""Import Kalibr camchain YAML without importing or executing Kalibr."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from yaml import YAMLError

from calibrex import __version__
from calibrex.core.external_run import (
    ExternalArtifactDigest,
    ExternalCalibrationRunArtifact,
    ExternalDataIsolation,
    ExternalExecution,
    ExternalParsedOutputs,
    ExternalRunProvenance,
    ExternalRunStatus,
    ExternalToolIdentity,
    ExternalTransformOutput,
)
from calibrex.core.geometry import quaternion_xyzw_from_rotation_matrix
from calibrex.core.io import read_mapping
from calibrex.core.provenance import git_commit, sha256_path

KALIBR_ADAPTER_VERSION = "calibrex.kalibr_camchain_importer/v0.1"
KALIBR_SOURCE_REPOSITORY = "https://github.com/ethz-asl/kalibr"


def import_kalibr_camchain(
    path: str | Path,
    *,
    input_artifacts: tuple[str | Path, ...] = (),
    expected_source_sha256: str | None = None,
    tool_version: str | None = None,
    source_commit: str | None = None,
    license_spdx: str | None = "BSD-4-Clause",
    training_isolation_declared: bool = False,
    training_isolation_evidence: str | None = None,
) -> ExternalCalibrationRunArtifact:
    """Convert a Kalibr camchain YAML into a generic external-run artifact."""

    source = Path(path)
    source_sha256 = sha256_path(source)
    artifacts = [
        artifact
        for input_path in input_artifacts
        if (artifact := _digest_artifact("input", Path(input_path))) is not None
    ]
    output_digest = _digest_artifact("output", source)
    if output_digest is not None:
        artifacts.append(output_digest)
    warnings = _identity_warnings(
        tool_version=tool_version,
        source_commit=source_commit,
    )
    if not any(artifact.role == "input" for artifact in artifacts):
        warnings.append("Kalibr import has no digest-bound fitting input artifact")

    status: ExternalRunStatus
    parsed_outputs = ExternalParsedOutputs()
    if source_sha256 is None:
        status = "unavailable"
        warnings.append(f"Kalibr camchain does not exist: {source}")
    elif (
        expected_source_sha256 is not None
        and source_sha256 != expected_source_sha256.lower()
    ):
        status = "digest_mismatch"
        warnings.append(
            "Kalibr camchain SHA-256 mismatch: "
            f"expected {expected_source_sha256.lower()}, observed {source_sha256}"
        )
    else:
        try:
            payload = read_mapping(source)
            parsed_outputs = _parse_camchain(payload)
        except (OSError, ValueError, YAMLError) as exc:
            status = "invalid_output"
            warnings.append(f"Kalibr camchain could not be parsed: {exc}")
        else:
            status = "success"

    return ExternalCalibrationRunArtifact(
        run_id=f"kalibr:{source.stem}",
        adapter_name="kalibr_camchain",
        adapter_version=KALIBR_ADAPTER_VERSION,
        tool=ExternalToolIdentity(
            name="kalibr",
            version=tool_version,
            source_repository=KALIBR_SOURCE_REPOSITORY,
            source_commit=source_commit,
            license_spdx=license_spdx,
            license_boundary="imported",
        ),
        execution=ExternalExecution(mode="imported"),
        artifacts=artifacts,
        frame_convention=(
            "T_parent_child; Kalibr T_cam_imu maps IMU into camera and "
            "T_cn_cnm1 maps the previous camera into the current camera"
        ),
        time_convention=(
            "timeshift_cam_imu seconds; t_imu = t_cam + timeshift_cam_imu"
        ),
        train_data_isolation=ExternalDataIsolation(
            declared=training_isolation_declared,
            evidence=training_isolation_evidence,
        ),
        status=status,
        warnings=warnings,
        parsed_outputs=parsed_outputs,
        provenance=ExternalRunProvenance(
            calibrex_version=__version__,
            git_commit=git_commit(),
            source_artifact=str(source),
            source_artifact_sha256=source_sha256,
        ),
    )


def _parse_camchain(payload: dict[str, object]) -> ExternalParsedOutputs:
    cameras = [
        (name, value)
        for name, value in sorted(payload.items(), key=lambda item: _camera_sort_key(item[0]))
        if name.startswith("cam") and isinstance(value, dict)
    ]
    if not cameras:
        msg = "Kalibr camchain contains no camN mappings"
        raise ValueError(msg)

    transforms: dict[str, ExternalTransformOutput] = {}
    time_offsets: dict[str, float] = {}
    intrinsics: dict[str, dict[str, float | str | list[float]]] = {}
    previous_camera: str | None = None
    for camera_name, raw_camera in cameras:
        camera = _string_mapping(raw_camera, context=camera_name)
        intrinsics[camera_name] = _camera_intrinsics(camera_name, camera)
        if "T_cam_imu" in camera:
            transforms[f"T_{camera_name}_imu"] = _transform_output(
                camera["T_cam_imu"],
                parent=camera_name,
                child="imu",
                field=f"{camera_name}.T_cam_imu",
            )
        if "T_cn_cnm1" in camera:
            if previous_camera is None:
                msg = f"{camera_name}.T_cn_cnm1 has no previous camera"
                raise ValueError(msg)
            transforms[f"T_{camera_name}_{previous_camera}"] = _transform_output(
                camera["T_cn_cnm1"],
                parent=camera_name,
                child=previous_camera,
                field=f"{camera_name}.T_cn_cnm1",
            )
        if "timeshift_cam_imu" in camera:
            shift = _finite_float(
                camera["timeshift_cam_imu"],
                field=f"{camera_name}.timeshift_cam_imu",
            )
            time_offsets[f"dt_imu_minus_{camera_name}"] = shift
        previous_camera = camera_name

    return ExternalParsedOutputs(
        transforms=transforms,
        time_offsets_seconds=time_offsets,
        intrinsics=intrinsics,
    )


def _camera_intrinsics(
    camera_name: str,
    camera: dict[str, object],
) -> dict[str, float | str | list[float]]:
    output: dict[str, float | str | list[float]] = {}
    for key in ("camera_model", "distortion_model", "rostopic"):
        value = camera.get(key)
        if value is not None:
            if not isinstance(value, str):
                msg = f"{camera_name}.{key} must be a string"
                raise ValueError(msg)
            output[key] = value
    for key in ("intrinsics", "distortion_coeffs", "resolution"):
        value = camera.get(key)
        if value is not None:
            output[key] = _finite_vector(value, field=f"{camera_name}.{key}")
    return output


def _transform_output(
    value: object,
    *,
    parent: str,
    child: str,
    field: str,
) -> ExternalTransformOutput:
    matrix = _matrix4(value, field=field)
    rotation = [
        matrix[0][0],
        matrix[0][1],
        matrix[0][2],
        matrix[1][0],
        matrix[1][1],
        matrix[1][2],
        matrix[2][0],
        matrix[2][1],
        matrix[2][2],
    ]
    return ExternalTransformOutput(
        parent=parent,
        child=child,
        translation_m=[matrix[0][3], matrix[1][3], matrix[2][3]],
        rotation_quat_xyzw=list(quaternion_xyzw_from_rotation_matrix(rotation)),
    )


def _matrix4(value: object, *, field: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 4:
        msg = f"{field} must be a 4x4 matrix"
        raise ValueError(msg)
    matrix: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            msg = f"{field} must be a 4x4 matrix"
            raise ValueError(msg)
        matrix.append([_finite_float(item, field=field) for item in row])
    if any(abs(value - expected) > 1.0e-9 for value, expected in zip(
        matrix[3],
        [0.0, 0.0, 0.0, 1.0],
        strict=True,
    )):
        msg = f"{field} homogeneous last row must be [0, 0, 0, 1]"
        raise ValueError(msg)
    return matrix


def _finite_vector(value: object, *, field: str) -> list[float]:
    if not isinstance(value, list):
        msg = f"{field} must be a list"
        raise ValueError(msg)
    return [_finite_float(item, field=field) for item in value]


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{field} must contain finite numbers"
        raise ValueError(msg)
    number = float(value)
    if not math.isfinite(number):
        msg = f"{field} must contain finite numbers"
        raise ValueError(msg)
    return number


def _string_mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        msg = f"{context} must be a string-keyed mapping"
        raise ValueError(msg)
    return {str(key): item for key, item in value.items()}


def _camera_sort_key(name: str) -> tuple[int, str]:
    suffix = name[3:] if name.startswith("cam") else ""
    return (int(suffix) if suffix.isdigit() else 1_000_000, name)


def _digest_artifact(
    role: Literal["input", "output"],
    path: Path,
) -> ExternalArtifactDigest | None:
    digest = sha256_path(path)
    if digest is None or not path.is_file():
        return None
    return ExternalArtifactDigest(
        role=role,
        path=str(path),
        sha256=digest,
        size_bytes=path.stat().st_size,
        media_type="application/yaml",
    )


def _identity_warnings(
    *,
    tool_version: str | None,
    source_commit: str | None,
) -> list[str]:
    warnings: list[str] = []
    if tool_version is None:
        warnings.append("Kalibr tool version is not declared")
    if source_commit is None:
        warnings.append("Kalibr source commit is not declared")
    return warnings
