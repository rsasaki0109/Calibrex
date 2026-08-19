"""Import iKalibr calibration result without importing or executing iKalibr.

iKalibr is a unified targetless spatiotemporal calibration framework for
resilient integrated inertial systems (IMU, LiDAR, Camera, Radar).
Reference: S. Chen et al., IEEE T-RO 2025, arXiv:2407.11420.
Repository: https://github.com/Unsigned-Long/iKalibr (BSD-3-Clause)

iKalibr serialises its CalibParam structure via the cereal library into YAML,
JSON, XML, or binary.  This adapter reads the **YAML or JSON** result file
(``ikalibr-prog-result.yaml`` / ``ikalibr-prog-result.json``) and converts
it to a generic ``slac.external_calibration_run/v0.1`` artifact.

Frame convention (iKalibr)
--------------------------
All extrinsics are expressed as transformations **from sensor frame to the
reference IMU body frame** (``Br``):

* ``SO3_LkToBr`` — rotation component of T_Br_Lk (quaternion, xyzw order
  in Sophus/cereal storage).
* ``POS_LkInBr`` — translation: position of LiDAR origin expressed in Br.
* ``SO3_CmToBr`` / ``POS_CmInBr`` — same for camera (Cm).
* ``SO3_RjToBr`` / ``POS_RjInBr`` — radar.
* ``SO3_BiToBr`` / ``POS_BiInBr`` — additional IMU.
* ``SO3_DnToBr`` / ``POS_DnInBr`` — depth camera.
* ``SO3_EsToBr`` / ``POS_EsToBr`` — event camera.

Time convention (iKalibr)
--------------------------
``TO_XkToBr`` — time offset (seconds) such that
``t_Br = t_Xk + TO_XkToBr``.

cereal YAML storage format
--------------------------
Rotations (Sophus::SO3d) are stored as a unit quaternion with coefficients
in xyzw order (Eigen internal layout)::

    SO3_LkToBr:
      /lidar_topic:
        value0: x
        value1: y
        value2: z
        value3: w  # real part

Translations (Eigen::Vector3d)::

    POS_LkInBr:
      /lidar_topic:
        value0: x
        value1: y
        value2: z

License boundary
----------------
No iKalibr C++ source or Python bindings are imported.  iKalibr must be
run separately (via ROS subprocess or Docker container) and the resulting
artifact file passed to this function.
"""

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

IKALIBR_ADAPTER_VERSION = "calibrex.ikalibr_result_importer/v0.1"
IKALIBR_SOURCE_REPOSITORY = "https://github.com/Unsigned-Long/iKalibr"

_SENSOR_MAP: dict[str, tuple[str, str]] = {
    "Lk": ("SO3_LkToBr", "POS_LkInBr"),
    "Cm": ("SO3_CmToBr", "POS_CmInBr"),
    "Rj": ("SO3_RjToBr", "POS_RjInBr"),
    "Bi": ("SO3_BiToBr", "POS_BiInBr"),
    "Dn": ("SO3_DnToBr", "POS_DnInBr"),
    "Es": ("SO3_EsToBr", "POS_EsInBr"),
}
_TEMPORAL_MAP: dict[str, str] = {
    "Lk": "TO_LkToBr",
    "Cm": "TO_CmToBr",
    "Rj": "TO_RjToBr",
    "Bi": "TO_BiToBr",
    "Dn": "TO_DnToBr",
    "Es": "TO_EsToBr",
}


def import_ikalibr_result(
    path: str | Path,
    *,
    input_artifacts: tuple[str | Path, ...] = (),
    expected_source_sha256: str | None = None,
    tool_version: str | None = None,
    source_commit: str | None = None,
    license_spdx: str | None = "BSD-3-Clause",
    training_isolation_declared: bool = False,
    training_isolation_evidence: str | None = None,
) -> ExternalCalibrationRunArtifact:
    """Convert an iKalibr YAML/JSON result file into a generic external-run artifact.

    Parameters
    ----------
    path:
        Path to the iKalibr result file (``ikalibr-prog-result.yaml`` or
        ``ikalibr-prog-result.json``).
    input_artifacts:
        Optional paths to input files used during iKalibr calibration
        (rosbag, config file).  Each existing file is digest-bound.
    expected_source_sha256:
        If provided, the importer raises ``digest_mismatch`` status when the
        file SHA-256 does not match.
    tool_version:
        iKalibr version string (e.g. ``"2.1.0"``).  Recorded as a warning
        if absent.
    source_commit:
        iKalibr git commit hash used to produce the result.
    license_spdx:
        SPDX identifier for iKalibr.  Defaults to ``"BSD-3-Clause"``.
    training_isolation_declared:
        Set True when the calibration run used data that was isolated from
        any Calibrex model training data.
    training_isolation_evidence:
        Free-form description of how isolation was verified.
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
    warnings = _identity_warnings(
        tool_version=tool_version,
        source_commit=source_commit,
    )
    if not any(artifact.role == "input" for artifact in artifacts):
        warnings.append("iKalibr import has no digest-bound fitting input artifact")

    status: ExternalRunStatus
    parsed_outputs = ExternalParsedOutputs()
    if source_sha256 is None:
        status = "unavailable"
        warnings.append(f"iKalibr result file does not exist: {source}")
    elif (
        expected_source_sha256 is not None
        and source_sha256 != expected_source_sha256.lower()
    ):
        status = "digest_mismatch"
        warnings.append(
            "iKalibr result SHA-256 mismatch: "
            f"expected {expected_source_sha256.lower()}, observed {source_sha256}"
        )
    else:
        try:
            payload = read_mapping(source)
            parsed_outputs = _parse_result(payload)
        except (OSError, ValueError, YAMLError) as exc:
            status = "invalid_output"
            warnings.append(f"iKalibr result could not be parsed: {exc}")
        else:
            status = "success"

    return ExternalCalibrationRunArtifact(
        run_id=f"ikalibr:{source.stem}",
        adapter_name="ikalibr_result",
        adapter_version=IKALIBR_ADAPTER_VERSION,
        tool=ExternalToolIdentity(
            name="ikalibr",
            version=tool_version,
            source_repository=IKALIBR_SOURCE_REPOSITORY,
            source_commit=source_commit,
            license_spdx=license_spdx,
            license_boundary="imported",
        ),
        execution=ExternalExecution(mode="imported"),
        artifacts=artifacts,
        frame_convention=(
            "T_Br_Xk: extrinsics map sensor frame Xk into reference IMU body frame Br. "
            "SO3_XkToBr is the rotation quaternion (xyzw) of that transform; "
            "POS_XkInBr is the position of Xk's origin expressed in Br."
        ),
        time_convention=(
            "TO_XkToBr seconds: t_Br = t_Xk + TO_XkToBr"
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


def _parse_result(payload: dict[str, object]) -> ExternalParsedOutputs:
    """Parse the top-level ``CalibParam`` mapping from an iKalibr YAML/JSON file."""
    root = payload.get("CalibParam")
    if not isinstance(root, dict) or not all(isinstance(k, str) for k in root):
        msg = "iKalibr result must have a top-level 'CalibParam' mapping"
        raise ValueError(msg)
    root = {str(k): v for k, v in root.items()}

    extri_raw = root.get("EXTRI")
    if not isinstance(extri_raw, dict):
        msg = "CalibParam.EXTRI must be a mapping"
        raise ValueError(msg)
    extri = {str(k): v for k, v in extri_raw.items()}

    temporal_raw = root.get("TEMPORAL", {})
    if not isinstance(temporal_raw, dict):
        temporal_raw = {}
    temporal = {str(k): v for k, v in temporal_raw.items()}

    transforms: dict[str, ExternalTransformOutput] = {}
    time_offsets: dict[str, float] = {}

    for sensor_tag, (so3_key, pos_key) in _SENSOR_MAP.items():
        so3_map = extri.get(so3_key, {})
        pos_map = extri.get(pos_key, {})
        if not isinstance(so3_map, dict) or not isinstance(pos_map, dict):
            continue
        for topic in so3_map:
            so3_entry = so3_map[topic]
            pos_entry = pos_map.get(topic)
            if so3_entry is None or pos_entry is None:
                continue
            try:
                quat_xyzw = _parse_so3d(so3_entry, field=f"{so3_key}.{topic}")
                translation = _parse_vector3d(pos_entry, field=f"{pos_key}.{topic}")
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
            transform_key = f"T_Br_{sensor_tag}_{_topic_to_id(topic)}"
            transforms[transform_key] = ExternalTransformOutput(
                parent="Br",
                child=f"{sensor_tag}_{_topic_to_id(topic)}",
                translation_m=translation,
                rotation_quat_xyzw=quat_xyzw,
            )

    to_key = _TEMPORAL_MAP.get("Lk", "TO_LkToBr")
    for sensor_tag, temporal_key in _TEMPORAL_MAP.items():
        to_map = temporal.get(temporal_key, {})
        if not isinstance(to_map, dict):
            continue
        for topic, offset in to_map.items():
            try:
                offset_s = _finite_float(offset, field=f"{temporal_key}.{topic}")
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
            offset_key = f"TO_{sensor_tag}_{_topic_to_id(topic)}_to_Br_s"
            time_offsets[offset_key] = offset_s

    return ExternalParsedOutputs(
        transforms=transforms,
        time_offsets_seconds=time_offsets,
    )


def _parse_so3d(value: object, *, field: str) -> list[float]:
    """Extract xyzw quaternion from a cereal-serialized Sophus::SO3d.

    cereal YAML format::

        value0: x   # imaginary
        value1: y   # imaginary
        value2: z   # imaginary
        value3: w   # real

    The function also tolerates a plain 4-element list in [x, y, z, w] order
    (as produced by the JSON format after conversion).
    """
    if isinstance(value, dict):
        str_value = {str(k): v for k, v in value.items()}
        if all(f"value{i}" in str_value for i in range(4)):
            coeffs = [
                _finite_float(str_value[f"value{i}"], field=field) for i in range(4)
            ]
        elif all(k in str_value for k in ("x", "y", "z", "w")):
            coeffs = [
                _finite_float(str_value[k], field=field) for k in ("x", "y", "z", "w")
            ]
        else:
            msg = f"{field}: expected cereal SO3d mapping with value0..value3 or x,y,z,w"
            raise ValueError(msg)
    elif isinstance(value, (list, tuple)) and len(value) == 4:
        coeffs = [_finite_float(v, field=field) for v in value]
    else:
        msg = f"{field}: expected SO3d mapping or 4-element list, got {type(value).__name__}"
        raise ValueError(msg)
    norm = math.sqrt(sum(c * c for c in coeffs))
    if not math.isfinite(norm) or abs(norm - 1.0) > 0.01:
        msg = f"{field}: quaternion norm {norm:.6f} is not close to 1"
        raise ValueError(msg)
    return [c / norm for c in coeffs]


def _parse_vector3d(value: object, *, field: str) -> list[float]:
    """Extract [x, y, z] from a cereal-serialized Eigen::Vector3d.

    cereal YAML format::

        value0: x
        value1: y
        value2: z

    Also accepts a plain 3-element list.
    """
    if isinstance(value, dict):
        str_value = {str(k): v for k, v in value.items()}
        if all(f"value{i}" in str_value for i in range(3)):
            return [_finite_float(str_value[f"value{i}"], field=field) for i in range(3)]
        if all(k in str_value for k in ("x", "y", "z")):
            return [_finite_float(str_value[k], field=field) for k in ("x", "y", "z")]
        msg = f"{field}: expected cereal Vector3d mapping with value0..value2 or x,y,z"
        raise ValueError(msg)
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return [_finite_float(v, field=field) for v in value]
    msg = f"{field}: expected Vector3d mapping or 3-element list, got {type(value).__name__}"
    raise ValueError(msg)


def _topic_to_id(topic: str) -> str:
    """Sanitise a ROS topic name to a valid key fragment."""
    return topic.lstrip("/").replace("/", "_").replace("-", "_") or "sensor"


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{field} must contain a finite number, got {type(value).__name__}"
        raise ValueError(msg)
    number = float(value)
    if not math.isfinite(number):
        msg = f"{field} must contain a finite number"
        raise ValueError(msg)
    return number


def _digest_artifact(
    role: Literal["input", "output"],
    path: Path,
) -> ExternalArtifactDigest | None:
    digest = sha256_path(path)
    if digest is None or not path.is_file():
        return None
    suffix = path.suffix.lower()
    media_type = (
        "application/yaml"
        if suffix in (".yaml", ".yml")
        else "application/json"
        if suffix == ".json"
        else "application/octet-stream"
    )
    return ExternalArtifactDigest(
        role=role,
        path=str(path),
        sha256=digest,
        size_bytes=path.stat().st_size,
        media_type=media_type,
    )


def _identity_warnings(
    *,
    tool_version: str | None,
    source_commit: str | None,
) -> list[str]:
    warnings: list[str] = []
    if tool_version is None:
        warnings.append("iKalibr tool version is not declared")
    if source_commit is None:
        warnings.append("iKalibr source commit is not declared")
    return warnings
