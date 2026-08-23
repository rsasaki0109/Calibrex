"""Import the official ``direct_visual_lidar_calibration`` result format.

The upstream Koide implementation is deliberately not imported here.  This
module is a small, ROS-independent file-format adapter for the JSON/YAML
``calib.json`` output written by
``koide3/direct_visual_lidar_calibration``.

The native result contains ``results.T_lidar_camera`` as
``[x, y, z, qx, qy, qz, qw]``.  It maps a point in the camera frame into the
LiDAR frame.  Calibrex represents the same transform as
``T_<lidar>_<camera>`` and always materializes its mathematically exact inverse
``T_<camera>_<lidar>`` as well.  Frame names are required from the caller;
there is intentionally no ``camera0``/``lidar0`` fallback in this adapter.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
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
    ExternalRunDigests,
    ExternalRunProvenance,
    ExternalRunStatus,
    ExternalToolIdentity,
    ExternalTransformOutput,
)
from calibrex.core.geometry import SE3
from calibrex.core.io import read_mapping
from calibrex.core.provenance import git_commit, sha256_path

KOIDE_ADAPTER_VERSION = "calibrex.koide_calib_json_importer/v0.1"
KOIDE_SOURCE_REPOSITORY = (
    "https://github.com/koide3/direct_visual_lidar_calibration"
)
KOIDE_NATIVE_FORMAT = "direct_visual_lidar_calibration.calib.json"
KOIDE_NATIVE_TRANSFORM_NAME: Literal["T_lidar_camera"] = "T_lidar_camera"


@dataclass(frozen=True)
class KoideNativeCalibration:
    """A strictly parsed Koide native calibration and its frame declaration.

    ``transforms`` contains both ``T_<lidar>_<camera>`` from the native
    estimate and the inverse ``T_<camera>_<lidar>``.  Each transform follows
    the Calibrex ``T_parent_child`` convention.
    """

    transforms: dict[str, SE3]
    lidar_frame: str
    camera_frame: str
    native_transform_name: Literal["T_lidar_camera"] = KOIDE_NATIVE_TRANSFORM_NAME
    native_format: str = KOIDE_NATIVE_FORMAT
    declared_convention: str = (
        "results.T_lidar_camera [x,y,z,qx,qy,qz,qw] maps camera-frame points "
        "to the LiDAR frame: p_lidar = R_lidar_camera p_camera + t_lidar_camera"
    )

    def provenance(self, source_path: str | Path, source_sha256: str | None) -> dict[str, object]:
        """Return JSON-friendly parser provenance for a result artifact."""

        return {
            "native_format": self.native_format,
            "native_transform_name": self.native_transform_name,
            "source_path": str(source_path),
            "source_sha256": source_sha256,
            "declared_convention": self.declared_convention,
            "lidar_frame": self.lidar_frame,
            "camera_frame": self.camera_frame,
            "calibrex_forward_transform": f"T_{self.lidar_frame}_{self.camera_frame}",
            "calibrex_inverse_transform": f"T_{self.camera_frame}_{self.lidar_frame}",
        }


def parse_koide_calib_payload(
    payload: Mapping[str, object],
    *,
    lidar_frame: str,
    camera_frame: str,
) -> KoideNativeCalibration:
    """Parse a Koide native payload with an explicit frame binding.

    Parameters
    ----------
    payload:
        Top-level mapping loaded from the official ``calib.json`` file.
    lidar_frame, camera_frame:
        Calibrex frame names corresponding to the upstream LiDAR and camera.
        Both are mandatory because the upstream field names are generic.

    Raises
    ------
    ValueError
        If the native structure, transform vector, numeric values, quaternion,
        or frame binding is invalid.  The caller must not silently substitute
        another frame convention after this error.
    """

    lidar = _required_frame_name(lidar_frame, field="lidar_frame")
    camera = _required_frame_name(camera_frame, field="camera_frame")
    if lidar == camera:
        raise ValueError("lidar_frame and camera_frame must be different")

    results = payload.get("results")
    if not isinstance(results, Mapping):
        raise ValueError("Koide calib.json must contain a 'results' mapping")
    if KOIDE_NATIVE_TRANSFORM_NAME not in results:
        raise ValueError(
            "Koide calib.json results must contain 'T_lidar_camera'"
        )

    raw_transform = results[KOIDE_NATIVE_TRANSFORM_NAME]
    values = _finite_vector(
        raw_transform,
        length=7,
        field="results.T_lidar_camera",
    )
    translation = values[:3]
    quaternion = values[3:]
    norm = math.sqrt(sum(value * value for value in quaternion))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError(
            "results.T_lidar_camera quaternion norm must be finite and non-zero"
        )

    native = SE3.from_lists(translation, quaternion)
    forward_name = f"T_{lidar}_{camera}"
    inverse_name = f"T_{camera}_{lidar}"
    return KoideNativeCalibration(
        transforms={forward_name: native, inverse_name: native.inverse()},
        lidar_frame=lidar,
        camera_frame=camera,
    )


def parse_koide_calib_json(
    path: str | Path,
    *,
    lidar_frame: str,
    camera_frame: str,
) -> KoideNativeCalibration:
    """Read and strictly parse one official Koide ``calib.json`` file.

    The function raises :class:`ValueError` for every malformed native result,
    including a missing file, so subprocess adapters can materialize the error
    as an ``invalid_output`` external-run status.
    """

    source = Path(path)
    if not source.is_file():
        raise ValueError(f"Koide calib.json does not exist: {source}")
    try:
        payload = read_mapping(source)
    except (OSError, YAMLError, ValueError) as exc:
        raise ValueError(f"could not read Koide calib.json {source}: {exc}") from exc
    return parse_koide_calib_payload(
        payload,
        lidar_frame=lidar_frame,
        camera_frame=camera_frame,
    )


def import_koide_result(
    path: str | Path,
    *,
    lidar_frame: str,
    camera_frame: str,
    input_artifacts: tuple[str | Path, ...] = (),
    expected_source_sha256: str | None = None,
    tool_version: str | None = None,
    source_commit: str | None = None,
    license_spdx: str | None = "MIT",
    training_isolation_declared: bool = False,
    training_isolation_evidence: str | None = None,
    training_data_ids_sha256: str | None = None,
    holdout_data_ids_sha256: str | None = None,
) -> ExternalCalibrationRunArtifact:
    """Convert a Koide native result into a typed external-run artifact.

    Invalid native data is represented as ``status='invalid_output'`` while
    preserving the source path and digest.  For callers that need a strict
    exception, use :func:`parse_koide_calib_json` directly.
    """

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

    warnings = _identity_warnings(tool_version=tool_version, source_commit=source_commit)
    if not any(artifact.role == "input" for artifact in artifacts):
        warnings.append("Koide import has no digest-bound fitting input artifact")
    input_sha256 = _combined_digest(artifacts, role="input")

    status: ExternalRunStatus
    parsed_outputs = ExternalParsedOutputs(
        external_metrics=_native_metadata(
            source=source,
            source_sha256=source_sha256,
            calibration=None,
        )
    )
    calibration: KoideNativeCalibration | None = None
    if source_sha256 is None:
        status = "unavailable"
        warnings.append(f"Koide calib.json does not exist: {source}")
    elif (
        expected_source_sha256 is not None
        and source_sha256 != expected_source_sha256.lower()
    ):
        status = "digest_mismatch"
        warnings.append(
            "Koide result SHA-256 mismatch: "
            f"expected {expected_source_sha256.lower()}, observed {source_sha256}"
        )
    else:
        try:
            calibration = parse_koide_calib_json(
                source,
                lidar_frame=lidar_frame,
                camera_frame=camera_frame,
            )
        except ValueError as exc:
            status = "invalid_output"
            warnings.append(f"Koide result could not be parsed: {exc}")
        else:
            status = "success"
            parsed_outputs = ExternalParsedOutputs(
                transforms={
                    name: ExternalTransformOutput(
                        parent=_koide_transform_frames(name, calibration)[0],
                        child=_koide_transform_frames(name, calibration)[1],
                        translation_m=list(transform.translation_m),
                        rotation_quat_xyzw=list(transform.rotation_quat_xyzw),
                    )
                    for name, transform in calibration.transforms.items()
                },
                external_metrics=_native_metadata(
                    source=source,
                    source_sha256=source_sha256,
                    calibration=calibration,
                ),
            )

    frame_convention = (
        calibration.declared_convention
        if calibration is not None
        else (
            "Koide results.T_lidar_camera [x,y,z,qx,qy,qz,qw] maps camera "
            "points to LiDAR; explicit Calibrex frame binding is required"
        )
    )
    return ExternalCalibrationRunArtifact(
        run_id=f"koide:{source.stem}",
        adapter_name="koide_calib_json",
        adapter_version=KOIDE_ADAPTER_VERSION,
        tool=ExternalToolIdentity(
            name="direct_visual_lidar_calibration",
            version=tool_version,
            source_repository=KOIDE_SOURCE_REPOSITORY,
            source_commit=source_commit,
            license_spdx=license_spdx,
            license_boundary="imported",
        ),
        execution=ExternalExecution(mode="imported"),
        artifacts=artifacts,
        digests=ExternalRunDigests(
            input_sha256=input_sha256,
            output_sha256=output_digest.sha256 if output_digest is not None else None,
        ),
        frame_convention=frame_convention,
        time_convention="Koide calib.json contains no clock-offset estimate",
        train_data_isolation=ExternalDataIsolation(
            declared=training_isolation_declared,
            evidence=training_isolation_evidence,
            training_data_ids_sha256=training_data_ids_sha256,
            holdout_data_ids_sha256=holdout_data_ids_sha256,
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


def _required_frame_name(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty explicit frame name")
    return value.strip()


def _finite_vector(value: object, *, length: int, field: str) -> list[float]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a {length}-element numeric list")
    if len(value) != length:
        raise ValueError(f"{field} must contain exactly {length} values")
    parsed: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{field}[{index}] must be a finite number")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{field}[{index}] must be a finite number")
        parsed.append(number)
    return parsed


def _parent_child(name: str) -> tuple[str, str]:
    if not name.startswith("T_"):
        raise ValueError(f"invalid Calibrex transform name: {name}")
    body = name[2:]
    parent, separator, child = body.partition("_")
    if not separator or not parent or not child:
        raise ValueError(f"invalid Calibrex transform name: {name}")
    return parent, child


def _koide_transform_frames(
    name: str,
    calibration: KoideNativeCalibration,
) -> tuple[str, str]:
    """Resolve native transform names without splitting frame underscores."""

    forward_name = f"T_{calibration.lidar_frame}_{calibration.camera_frame}"
    inverse_name = f"T_{calibration.camera_frame}_{calibration.lidar_frame}"
    if name == forward_name:
        return calibration.lidar_frame, calibration.camera_frame
    if name == inverse_name:
        return calibration.camera_frame, calibration.lidar_frame
    return _parent_child(name)


def _native_metadata(
    *,
    source: Path,
    source_sha256: str | None,
    calibration: KoideNativeCalibration | None,
) -> dict[str, float | str | bool | None]:
    """Return scalar metadata accepted by the generic parsed-output schema."""

    metadata: dict[str, float | str | bool | None] = {
        "koide_native_format": KOIDE_NATIVE_FORMAT,
        "koide_native_source_path": str(source),
        "koide_native_source_sha256": source_sha256,
        "koide_native_declared_transform": KOIDE_NATIVE_TRANSFORM_NAME,
        "koide_native_declared_convention": (
            "results.T_lidar_camera [x,y,z,qx,qy,qz,qw] maps camera-frame "
            "points into the LiDAR frame"
        ),
    }
    if calibration is not None:
        metadata.update(
            {
                "koide_lidar_frame": calibration.lidar_frame,
                "koide_camera_frame": calibration.camera_frame,
                "koide_calibrex_forward_transform": (
                    f"T_{calibration.lidar_frame}_{calibration.camera_frame}"
                ),
                "koide_calibrex_inverse_transform": (
                    f"T_{calibration.camera_frame}_{calibration.lidar_frame}"
                ),
            }
        )
    return metadata


def _digest_artifact(role: Literal["input", "output"], path: Path) -> ExternalArtifactDigest | None:
    digest = sha256_path(path)
    if digest is None or not path.is_file():
        return None
    media_type = "application/json" if path.suffix.lower() == ".json" else "application/yaml"
    return ExternalArtifactDigest(
        role=role,
        path=str(path),
        sha256=digest,
        size_bytes=path.stat().st_size,
        media_type=media_type,
    )


def _identity_warnings(*, tool_version: str | None, source_commit: str | None) -> list[str]:
    warnings: list[str] = []
    if not tool_version:
        warnings.append("Koide tool version is not declared")
    if not source_commit:
        warnings.append("Koide source commit is not declared")
    return warnings


def _combined_digest(
    artifacts: list[ExternalArtifactDigest], *, role: Literal["input", "output"]
) -> str | None:
    """Bind an external artifact inventory to one deterministic digest."""

    selected = sorted(
        (artifact.path, artifact.sha256)
        for artifact in artifacts
        if artifact.role == role
    )
    if not selected:
        return None
    payload = "\n".join(f"{path}\0{digest}" for path, digest in selected).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "KOIDE_ADAPTER_VERSION",
    "KOIDE_NATIVE_FORMAT",
    "KOIDE_NATIVE_TRANSFORM_NAME",
    "KOIDE_SOURCE_REPOSITORY",
    "KoideNativeCalibration",
    "import_koide_result",
    "parse_koide_calib_json",
    "parse_koide_calib_payload",
]
