"""ROS-independent Autoware sensor-kit calibration export.

The adapter deliberately emits the small, stable subset consumed by
Autoware's ``sensor_kit_calibration.yaml`` files and a ROS 2 launch snippet
for the same edges.  No ROS package is imported here.  Calibrex's transform
convention is explicit throughout: ``T_parent_child`` maps a point from the
child frame into the parent frame, and quaternions are always ``xyzw``.

The direct YAML document contains an Autoware-compatible mapping under the
selected parent frame (for example ``sensor_kit_base_link:``).  It also keeps
the historical ``sensors`` list used by Calibrex 0.4.  Provenance and output
digests are namespaced under ``calibrex`` so an Autoware consumer can select
the parent-frame mapping without depending on Calibrex internals.  A
schema-valid manifest can be written next to the YAML by using
``write_autoware_export``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import ConfigDict, Field, field_validator, model_validator

from calibrex import __version__
from calibrex.core.exceptions import CalibrexError
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.result import CalibrationResult, Grade, StrictModel, TransformResult

AUTOWARE_EXPORT_SCHEMA_VERSION: Literal["slac.autoware_export/v0.1"] = (
    "slac.autoware_export/v0.1"
)
# Keep the historical Calibrex format marker for callers that used the 0.4
# exporter.  The concrete Autoware protocol is recorded separately in
# ``protocol_version`` and the direct parent-frame mapping.
AUTOWARE_FORMAT: Literal["slac.autoware/v0.1"] = "slac.autoware/v0.1"
AUTOWARE_PROTOCOL_VERSION = "autoware.sensor_kit_calibration/v1"
_SHA256_RE = r"^[0-9a-f]{64}$"


class AutowareExportError(CalibrexError):
    """Raised when an export cannot be made unambiguous and schema-valid."""


class AutowareExportConfig(StrictModel):
    """Resolved and user-selectable export policy.

    ``base_frame`` and ``sensor_frames`` are optional at the input boundary.
    The builder resolves them from a result frame graph only when that
    resolution is unique; mapping-only exports must provide a unique parent
    or an explicit base frame.  ``invert_transforms`` names source transform
    keys that need to be inverted before export.
    """

    model_config = ConfigDict(extra="forbid")

    base_frame: str | None = None
    sensor_frames: list[str] = Field(default_factory=list)
    transform_names: list[str] = Field(default_factory=list)
    invert_transforms: list[str] = Field(default_factory=list)
    protocol_version: str = AUTOWARE_PROTOCOL_VERSION
    include_static_tf: bool = True
    include_legacy_transforms: bool = True

    @field_validator("base_frame")
    @classmethod
    def _validate_optional_frame(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("base_frame must not be empty")
        return value

    @field_validator("sensor_frames", "transform_names", "invert_transforms")
    @classmethod
    def _validate_names(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("frame and transform names must not be empty")
        if len(value) != len(set(value)):
            raise ValueError("frame and transform names must be unique")
        return value


class AutowareSensorCalibration(StrictModel):
    """One sensor pose in the format consumed by Autoware."""

    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


class AutowareTransformRecord(StrictModel):
    """Audit record retaining the source edge and any exact inversion."""

    name: str
    source_parent_frame: str
    source_child_frame: str
    parent_frame: str
    child_frame: str
    inverse_applied: bool = False
    convention: Literal["T_parent_child"] = "T_parent_child"
    translation_m: list[float] = Field(min_length=3, max_length=3)
    rotation_quat_xyzw: list[float] = Field(min_length=4, max_length=4)
    rotation_rpy_rad: list[float] = Field(min_length=3, max_length=3)
    quality_grade: Grade = "warn"


class AutowareStaticTransform(StrictModel):
    """One static TF edge and its ROS 2 CLI arguments."""

    parent_frame: str
    child_frame: str
    convention: Literal["T_parent_child"] = "T_parent_child"
    translation_m: list[float] = Field(min_length=3, max_length=3)
    rotation_quat_xyzw: list[float] = Field(min_length=4, max_length=4)
    quaternion_order: Literal["xyzw"] = "xyzw"
    command: list[str] = Field(min_length=1)


class AutowareExportProvenance(StrictModel):
    """Lineage and digest fields carried by the export artifact."""

    producer: Literal["calibrex"] = "calibrex"
    tool_name: str = "calibrex.autoware-export"
    tool_version: str = __version__
    git_commit: str | None = None
    protocol_version: str = AUTOWARE_PROTOCOL_VERSION
    command: list[str] = Field(default_factory=list)
    source_result_path: str | None = None
    source_result_sha256: str | None = Field(default=None, pattern=_SHA256_RE)
    # Alias retained for generic adapter consumers that use ``source_sha256``.
    source_sha256: str | None = Field(default=None, pattern=_SHA256_RE)
    input_sha256: dict[str, str] = Field(default_factory=dict)
    source_run_id: str | None = None
    source_quality_grade: Grade | None = None
    artifact_sha256: str | None = Field(default=None, pattern=_SHA256_RE)
    artifact_digest_scope: str = (
        "canonical export document excluding provenance.artifact_sha256 and artifact_sha256"
    )

    @field_validator("input_sha256")
    @classmethod
    def _validate_input_digests(cls, value: dict[str, str]) -> dict[str, str]:
        for name, digest in value.items():
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"input_sha256[{name!r}] must be a 64-character SHA-256")
            try:
                int(digest, 16)
            except ValueError as exc:
                raise ValueError(f"input_sha256[{name!r}] is not hexadecimal") from exc
        return {name: digest.lower() for name, digest in value.items()}


class AutowareExportArtifact(StrictModel):
    """Typed, schema-valid representation of an Autoware export bundle."""

    # The dynamic parent-frame key is retained in the serialized document,
    # while this stable ``calibration`` field makes the artifact easy to
    # validate and consume without knowing the chosen frame name in advance.
    model_config = ConfigDict(extra="allow")

    schema_version: Literal["slac.autoware_export/v0.1"] = AUTOWARE_EXPORT_SCHEMA_VERSION
    format: Literal["slac.autoware/v0.1"] = AUTOWARE_FORMAT
    calibration: dict[str, dict[str, AutowareSensorCalibration]]
    sensors: list[dict[str, Any]] = Field(default_factory=list)
    transforms: list[AutowareTransformRecord] = Field(default_factory=list)
    static_tf: list[AutowareStaticTransform] = Field(default_factory=list)
    frame_convention: Literal["T_parent_child"] = "T_parent_child"
    quaternion_order: Literal["xyzw"] = "xyzw"
    translation_unit: Literal["m"] = "m"
    rotation_unit: Literal["rad"] = "rad"
    export_config: AutowareExportConfig
    provenance: AutowareExportProvenance
    artifact_sha256: str = Field(pattern=_SHA256_RE)
    source_run: str | None = None
    quality_grade: Grade | None = None

    @model_validator(mode="after")
    def _validate_calibration(self) -> AutowareExportArtifact:
        if len(self.calibration) != 1:
            raise ValueError("Autoware export must contain exactly one parent frame")
        parent = next(iter(self.calibration))
        if not parent.strip():
            raise ValueError("Autoware parent frame must not be empty")
        if not self.calibration[parent]:
            raise ValueError("Autoware export must contain at least one sensor")
        return self

    def autoware_mapping(self) -> dict[str, Any]:
        """Return the direct Autoware YAML mapping plus Calibrex metadata."""

        parent, sensors = next(iter(self.calibration.items()))
        # Parent-first order is intentional: the first mapping is directly
        # usable as sensor_kit_calibration.yaml by existing Autoware tools.
        document: dict[str, Any] = {
            parent: {
                name: sensor.model_dump(mode="json")
                for name, sensor in sorted(sensors.items())
            },
            "calibration": {
                parent: {
                    name: sensor.model_dump(mode="json")
                    for name, sensor in sorted(sensors.items())
                }
            },
            "schema_version": self.schema_version,
            "format": self.format,
            "sensors": self.sensors,
            "transforms": [item.model_dump(mode="json") for item in self.transforms],
            "static_tf": [item.model_dump(mode="json") for item in self.static_tf],
            "frame_convention": self.frame_convention,
            "quaternion_order": self.quaternion_order,
            "translation_unit": self.translation_unit,
            "rotation_unit": self.rotation_unit,
            "export_config": self.export_config.model_dump(mode="json"),
            "provenance": self.provenance.model_dump(mode="json"),
            "artifact_sha256": self.artifact_sha256,
            "source_run": self.source_run,
            "quality_grade": self.quality_grade,
        }
        document["calibrex"] = {
            "schema_version": self.schema_version,
            "frame_convention": self.frame_convention,
            "quaternion_order": self.quaternion_order,
            "translation_unit": self.translation_unit,
            "rotation_unit": self.rotation_unit,
            "export_config": self.export_config.model_dump(mode="json"),
            "provenance": self.provenance.model_dump(mode="json"),
            "artifact_sha256": self.artifact_sha256,
        }
        # Nulls are intentionally omitted to keep the direct Autoware mapping
        # small and preserve the historical output shape.
        return cast(dict[str, Any], _drop_none(document))

    def static_tf_launch(self) -> str:
        """Return a deterministic ROS 2 ``tf2_ros`` launch Python snippet."""

        lines = [
            '"""Generated by Calibrex; do not edit by hand.',
            "",
            "The arguments use ROS 2 static_transform_publisher's explicit",
            "quaternion form. Calibrex source convention is T_parent_child",
            "(child points mapped into parent), quaternion order xyzw.",
            '"""',
            "",
            "from launch import LaunchDescription",
            "from launch_ros.actions import Node",
            "",
            "",
            "def generate_launch_description():",
            "    return LaunchDescription([",
        ]
        for index, transform in enumerate(self.static_tf):
            lines.extend(
                [
                    "        Node(",
                    '            package="tf2_ros",',
                    '            executable="static_transform_publisher",',
                    f'            name="calibrex_static_tf_{index:03d}",',
                    "            arguments=[",
                    *(f"                {argument!r}," for argument in transform.command),
                    "            ],",
                    '            output="screen",',
                    "        ),",
                ]
            )
        lines.extend(["    ])", ""])
        return "\n".join(lines)


def autoware_export_json_schema() -> dict[str, Any]:
    """Return the JSON schema for the export artifact and manifest."""

    return AutowareExportArtifact.model_json_schema()


def build_autoware_export(
    source: CalibrationResult | Mapping[str, TransformResult | Mapping[str, Any]],
    *,
    config: AutowareExportConfig | None = None,
    source_path: str | Path | None = None,
    source_run: str | None = None,
    quality_grade: Grade | None = None,
    input_paths: Mapping[str, str | Path] | None = None,
    input_sha256: Mapping[str, str] | None = None,
    command: Sequence[str] = (),
    tool_name: str = "calibrex.autoware-export",
    tool_version: str = __version__,
    git_commit_override: str | None = None,
) -> AutowareExportArtifact:
    """Build a validated Autoware export from a result or transform mapping.

    The function resolves only direct ``T_base_sensor`` edges.  A source edge
    in the opposite direction is accepted only when its name is explicitly
    listed in ``config.invert_transforms``.  In every case the output edge is
    checked to be ``T_base_sensor`` after inversion; this prevents silently
    publishing an inverse in the wrong direction.
    """

    resolved_config = config or AutowareExportConfig()
    transforms, result = _coerce_source(source)
    if not transforms:
        raise AutowareExportError("Autoware export requires at least one transform")

    base_frame = resolved_config.base_frame
    if base_frame is None and result is not None:
        base_frame = result.frame_graph.root
    if base_frame is None:
        parents = {transform.parent for transform in transforms.values()}
        if len(parents) != 1:
            names = ", ".join(sorted(parents)) or "none"
            raise AutowareExportError(
                "base frame is ambiguous for a transform mapping; "
                f"observed parent frames: {names}; pass config.base_frame"
            )
        base_frame = next(iter(parents))
    if not base_frame.strip():
        raise AutowareExportError("base frame must not be empty")

    selected = _select_transforms(transforms, resolved_config)
    records = _resolve_records(selected, base_frame, resolved_config)
    if not records:
        raise AutowareExportError(
            f"no transform resolves to base frame {base_frame!r}; "
            "check frame names and T_parent_child direction"
        )
    output_children = [item.child_frame for item in records]
    if len(output_children) != len(set(output_children)):
        duplicates = sorted({name for name in output_children if output_children.count(name) > 1})
        raise AutowareExportError(
            "ambiguous Autoware sensor frame(s): " + ", ".join(duplicates)
        )
    requested_sensors = set(resolved_config.sensor_frames)
    if requested_sensors and requested_sensors != set(output_children):
        missing = sorted(requested_sensors - set(output_children))
        extra = sorted(set(output_children) - requested_sensors)
        detail = []
        if missing:
            detail.append("missing=" + ",".join(missing))
        if extra:
            detail.append("unexpected=" + ",".join(extra))
        raise AutowareExportError(
            "sensor frame selection does not match resolved transforms (" + "; ".join(detail) + ")"
        )

    records = sorted(records, key=lambda item: (item.child_frame, item.name))
    calibration = {
        base_frame: {
            record.child_frame: AutowareSensorCalibration(
                x=record.translation_m[0],
                y=record.translation_m[1],
                z=record.translation_m[2],
                roll=record.rotation_rpy_rad[0],
                pitch=record.rotation_rpy_rad[1],
                yaw=record.rotation_rpy_rad[2],
            )
            for record in records
        }
    }

    static_tf = (
        [_static_tf_for(record) for record in records]
        if resolved_config.include_static_tf
        else []
    )
    if result is not None:
        if source_run is None:
            source_run = result.run.id
        if quality_grade is None:
            quality_grade = result.quality.grade

    source_digest = _source_digest(source, source_path)
    digests: dict[str, str] = {}
    if source_digest is not None:
        digests["source_result"] = source_digest
    if result is not None:
        if result.run.config_sha256:
            digests["config"] = result.run.config_sha256
        if result.run.dataset_sha256:
            digests["dataset"] = result.run.dataset_sha256
    if input_sha256:
        digests.update({key: value.lower() for key, value in input_sha256.items()})
    if input_paths:
        for key, value in sorted(input_paths.items()):
            digest = sha256_path(Path(value))
            if digest is None:
                raise AutowareExportError(f"input path does not exist: {value}")
            digests[key] = digest
    provenance = AutowareExportProvenance(
        tool_name=tool_name,
        tool_version=tool_version,
        git_commit=git_commit_override if git_commit_override is not None else git_commit(),
        protocol_version=resolved_config.protocol_version,
        source_result_path=Path(source_path).as_posix() if source_path is not None else None,
        source_result_sha256=source_digest,
        source_sha256=source_digest,
        input_sha256=digests,
        command=list(command),
        source_run_id=source_run,
        source_quality_grade=quality_grade,
    )
    resolved_config = resolved_config.model_copy(update={"base_frame": base_frame})
    payload_without_digest = _artifact_payload(
        calibration=calibration,
        records=records,
        static_tf=static_tf,
        config=resolved_config,
        provenance=provenance,
        source_run=source_run,
        quality_grade=quality_grade,
        artifact_sha256=None,
    )
    artifact_digest = _canonical_sha256(_drop_none(payload_without_digest))
    provenance = provenance.model_copy(update={"artifact_sha256": artifact_digest})
    payload = _artifact_payload(
        calibration=calibration,
        records=records,
        static_tf=static_tf,
        config=resolved_config,
        provenance=provenance,
        source_run=source_run,
        quality_grade=quality_grade,
        artifact_sha256=artifact_digest,
    )
    return AutowareExportArtifact.model_validate(payload)


def write_autoware_export(
    artifact: AutowareExportArtifact,
    output: str | Path,
    *,
    static_tf_output: str | Path | None = None,
    manifest_output: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Atomically write calibration YAML, static-TF launch, and manifest.

    ``output`` is always the direct Autoware calibration document.  If the
    optional sidecar paths are supplied, all targets are checked before any
    write; existing files require ``overwrite=True``.  The manifest records
    byte-level digests for every emitted file and is itself deterministic.
    """

    output_path = Path(output)
    static_path = Path(static_tf_output) if static_tf_output is not None else None
    manifest_path = Path(manifest_output) if manifest_output is not None else None
    targets = [output_path]
    if static_path is not None:
        targets.append(static_path)
    if manifest_path is not None:
        targets.append(manifest_path)
    normalized = [path.resolve() for path in targets]
    if len(normalized) != len(set(normalized)):
        raise AutowareExportError("Autoware export output paths must be distinct")
    if not overwrite:
        existing = [path for path in targets if path.exists()]
        if existing:
            raise AutowareExportError(
                "refusing to overwrite existing output(s): "
                + ", ".join(str(path) for path in existing)
                + "; pass --force or overwrite=True"
            )

    calibration_text = _serialize_mapping(artifact.autoware_mapping(), output_path)
    static_text = artifact.static_tf_launch() if static_path is not None else None
    calibration_digest = _sha256_text(calibration_text)
    static_digest = _sha256_text(static_text)
    manifest_text: str | None = None
    if manifest_path is not None:
        manifest_payload = artifact.autoware_mapping()
        manifest_payload.update(
            {
                "calibration_output": output_path.as_posix(),
                "calibration_sha256": calibration_digest,
                "static_tf_output": static_path.as_posix() if static_path is not None else None,
                "static_tf_sha256": static_digest,
                "calibration_digest_scope": "sha256 of emitted UTF-8 calibration bytes",
                "static_tf_digest_scope": "sha256 of emitted UTF-8 launch bytes",
            }
        )
        manifest_payload = cast(dict[str, Any], _drop_none(manifest_payload))
        manifest_payload["manifest_sha256"] = _canonical_sha256(manifest_payload)
        manifest_payload["manifest_digest_scope"] = (
            "canonical manifest mapping excluding manifest_sha256 and this scope"
        )
        manifest_text = _serialize_mapping(_drop_none(manifest_payload), manifest_path)

    # Every call is atomic at the individual-file level.  All existence and
    # parent checks happen before writing, so a normal failure cannot clobber
    # an incumbent artifact.
    _atomic_write(output_path, calibration_text, overwrite=overwrite)
    if static_path is not None and static_text is not None:
        _atomic_write(static_path, static_text, overwrite=overwrite)
    if manifest_path is not None and manifest_text is not None:
        _atomic_write(manifest_path, manifest_text, overwrite=overwrite)
    paths = {"calibration": output_path}
    if static_path is not None:
        paths["static_tf"] = static_path
    if manifest_path is not None:
        paths["manifest"] = manifest_path
    return paths


def export_autoware_yaml(
    result: CalibrationResult,
    output: str | Path,
    *,
    config: AutowareExportConfig | None = None,
    source_path: str | Path | None = None,
    input_paths: Mapping[str, str | Path] | None = None,
    command: Sequence[str] = (),
    static_tf_output: str | Path | None = None,
    manifest_output: str | Path | None = None,
    overwrite: bool = False,
) -> AutowareExportArtifact:
    """Backward-compatible result export with optional production sidecars."""

    artifact = build_autoware_export(
        result,
        config=config,
        source_path=source_path,
        source_run=result.run.id,
        quality_grade=result.quality.grade,
        input_paths=input_paths,
        command=command,
    )
    write_autoware_export(
        artifact,
        output,
        static_tf_output=static_tf_output,
        manifest_output=manifest_output,
        overwrite=overwrite,
    )
    return artifact


def export_autoware_transforms(
    transforms: Mapping[str, TransformResult],
    output: str | Path,
    *,
    source_run: str | None = None,
    quality_grade: Grade | None = None,
    config: AutowareExportConfig | None = None,
    source_path: str | Path | None = None,
    input_paths: Mapping[str, str | Path] | None = None,
    command: Sequence[str] = (),
    static_tf_output: str | Path | None = None,
    manifest_output: str | Path | None = None,
    overwrite: bool = False,
) -> AutowareExportArtifact:
    """Backward-compatible transform-map export with strict frame checks."""

    artifact = build_autoware_export(
        transforms,
        config=config,
        source_path=source_path,
        source_run=source_run,
        quality_grade=quality_grade,
        input_paths=input_paths,
        command=command,
    )
    write_autoware_export(
        artifact,
        output,
        static_tf_output=static_tf_output,
        manifest_output=manifest_output,
        overwrite=overwrite,
    )
    return artifact


def _coerce_source(
    source: CalibrationResult | Mapping[str, TransformResult | Mapping[str, Any]],
) -> tuple[dict[str, TransformResult], CalibrationResult | None]:
    if isinstance(source, CalibrationResult):
        return dict(source.transforms), source
    transforms: dict[str, TransformResult] = {}
    for name, raw in source.items():
        if isinstance(raw, TransformResult):
            transforms[name] = raw
            continue
        if not isinstance(raw, Mapping):
            raise AutowareExportError(f"transform {name!r} is not a mapping")
        try:
            transforms[name] = TransformResult.model_validate(dict(raw))
        except Exception as exc:
            raise AutowareExportError(f"invalid transform {name!r}: {exc}") from exc
    return transforms, None


def _select_transforms(
    transforms: dict[str, TransformResult], config: AutowareExportConfig
) -> dict[str, TransformResult]:
    if config.transform_names:
        missing = [name for name in config.transform_names if name not in transforms]
        if missing:
            raise AutowareExportError(
                "requested transform name(s) are missing: " + ", ".join(missing)
            )
        return {name: transforms[name] for name in config.transform_names}
    return dict(transforms)


def _resolve_records(
    transforms: dict[str, TransformResult],
    base_frame: str,
    config: AutowareExportConfig,
) -> list[AutowareTransformRecord]:
    records: list[AutowareTransformRecord] = []
    inverse_names = set(config.invert_transforms)
    for name, transform in sorted(transforms.items()):
        _validate_transform(name, transform)
        should_invert = name in inverse_names
        if transform.parent != base_frame and transform.child != base_frame:
            # A non-base edge is not a sensor-kit calibration edge.  Silently
            # dropping a specifically requested edge would make a typo look
            # like a successful export.  Unrequested edges are harmless when
            # an explicit sensor-frame allowlist is supplied.
            if config.transform_names or (
                config.sensor_frames
                and (
                    transform.parent in config.sensor_frames
                    or transform.child in config.sensor_frames
                )
            ):
                raise AutowareExportError(
                    f"transform {name!r} ({transform.parent}->{transform.child}) does not "
                    f"touch base frame {base_frame!r}"
                )
            continue
        if transform.parent == base_frame and transform.child == base_frame:
            raise AutowareExportError(f"transform {name!r} has identical parent and child frame")
        if transform.child == base_frame:
            if not should_invert:
                # Reverse edges are never guessed.  Explicit sensor selection
                # is still rejected here so the caller must document inverse.
                continue
            resolved = transform.as_se3().inverse()
            parent_frame, child_frame = base_frame, transform.parent
        elif should_invert:
            resolved = transform.as_se3().inverse()
            parent_frame, child_frame = transform.child, transform.parent
        else:
            resolved = transform.as_se3()
            parent_frame, child_frame = transform.parent, transform.child
        if parent_frame != base_frame:
            raise AutowareExportError(
                f"transform {name!r} resolves to parent {parent_frame!r}, expected {base_frame!r}"
            )
        if config.sensor_frames and child_frame not in config.sensor_frames:
            continue
        records.append(
            AutowareTransformRecord(
                name=name,
                source_parent_frame=transform.parent,
                source_child_frame=transform.child,
                parent_frame=parent_frame,
                child_frame=child_frame,
                inverse_applied=should_invert,
                translation_m=list(resolved.translation_m),
                rotation_quat_xyzw=list(resolved.rotation_quat_xyzw),
                rotation_rpy_rad=list(_rpy_from_quaternion(resolved.rotation_quat_xyzw)),
                quality_grade=transform.quality.grade,
            )
        )
    unknown = sorted(inverse_names - set(transforms))
    if unknown:
        raise AutowareExportError(
            "requested inverse transform name(s) are missing: " + ", ".join(unknown)
        )
    return records


def _validate_transform(name: str, transform: TransformResult) -> None:
    if transform.convention != "T_parent_child":
        raise AutowareExportError(
            f"transform {name!r} uses unsupported convention {transform.convention!r}; "
            "expected T_parent_child"
        )
    if not transform.parent.strip() or not transform.child.strip():
        raise AutowareExportError(f"transform {name!r} has an empty frame name")
    if transform.parent == transform.child:
        raise AutowareExportError(f"transform {name!r} has identical parent and child frame")
    values = [*transform.translation_m, *transform.rotation_quat_xyzw]
    if not all(math.isfinite(float(value)) for value in values):
        raise AutowareExportError(f"transform {name!r} contains non-finite values")
    norm = math.sqrt(sum(float(value) ** 2 for value in transform.rotation_quat_xyzw))
    if norm <= 1.0e-12:
        raise AutowareExportError(f"transform {name!r} has a zero quaternion")
    # TransformResult's geometry helper normalizes values.  For export we do
    # not silently turn a scaled/non-rigid quaternion into a rigid transform.
    if abs(norm - 1.0) > 1.0e-3:
        raise AutowareExportError(
            f"transform {name!r} has a non-rigid quaternion norm {norm:.9g}; expected 1"
        )


def _rpy_from_quaternion(quaternion: Sequence[float]) -> tuple[float, float, float]:
    x, y, z, w = (float(value) for value in quaternion)
    sin_roll_cos_pitch = 2.0 * (w * x + y * z)
    cos_roll_cos_pitch = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sin_roll_cos_pitch, cos_roll_cos_pitch)
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sin_pitch)))
    sin_yaw_cos_pitch = 2.0 * (w * z + x * y)
    cos_yaw_cos_pitch = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch)
    return roll, pitch, yaw


def _static_tf_for(record: AutowareTransformRecord) -> AutowareStaticTransform:
    command = [
        "--x",
        _float_text(record.translation_m[0]),
        "--y",
        _float_text(record.translation_m[1]),
        "--z",
        _float_text(record.translation_m[2]),
        "--qx",
        _float_text(record.rotation_quat_xyzw[0]),
        "--qy",
        _float_text(record.rotation_quat_xyzw[1]),
        "--qz",
        _float_text(record.rotation_quat_xyzw[2]),
        "--qw",
        _float_text(record.rotation_quat_xyzw[3]),
        "--frame-id",
        record.parent_frame,
        "--child-frame-id",
        record.child_frame,
    ]
    return AutowareStaticTransform(
        parent_frame=record.parent_frame,
        child_frame=record.child_frame,
        translation_m=list(record.translation_m),
        rotation_quat_xyzw=list(record.rotation_quat_xyzw),
        command=command,
    )


def _artifact_payload(
    *,
    calibration: dict[str, dict[str, AutowareSensorCalibration]],
    records: list[AutowareTransformRecord],
    static_tf: list[AutowareStaticTransform],
    config: AutowareExportConfig,
    provenance: AutowareExportProvenance,
    source_run: str | None,
    quality_grade: Grade | None,
    artifact_sha256: str | None,
) -> dict[str, Any]:
    legacy = [
        {
            "transform": item.name,
            "parent_frame": item.parent_frame,
            "child_frame": item.child_frame,
            "translation": {
                "x": item.translation_m[0],
                "y": item.translation_m[1],
                "z": item.translation_m[2],
            },
            "rotation_xyzw": {
                "x": item.rotation_quat_xyzw[0],
                "y": item.rotation_quat_xyzw[1],
                "z": item.rotation_quat_xyzw[2],
                "w": item.rotation_quat_xyzw[3],
            },
        }
        for item in records
    ]
    return {
        "schema_version": AUTOWARE_EXPORT_SCHEMA_VERSION,
        "format": AUTOWARE_FORMAT,
        "calibration": {
            parent: {
                name: sensor.model_dump(mode="json")
                for name, sensor in sorted(sensors.items())
            }
            for parent, sensors in calibration.items()
        },
        "sensors": legacy if config.include_legacy_transforms else [],
        "transforms": [item.model_dump(mode="json") for item in records],
        "static_tf": [item.model_dump(mode="json") for item in static_tf],
        "frame_convention": "T_parent_child",
        "quaternion_order": "xyzw",
        "translation_unit": "m",
        "rotation_unit": "rad",
        "export_config": config.model_dump(mode="json"),
        "provenance": provenance.model_dump(mode="json"),
        "artifact_sha256": artifact_sha256,
        "source_run": source_run,
        "quality_grade": quality_grade,
    }


def _source_digest(
    source: CalibrationResult | Mapping[str, TransformResult | Mapping[str, Any]],
    source_path: str | Path | None,
) -> str | None:
    if source_path is not None:
        digest = sha256_path(Path(source_path))
        if digest is None:
            raise AutowareExportError(f"source result path does not exist: {source_path}")
        return digest
    if isinstance(source, CalibrationResult):
        return _canonical_sha256(source.model_dump(mode="json", exclude_none=False))
    canonical = {
        name: dict(value) if isinstance(value, Mapping) else value.model_dump(mode="json")
        for name, value in sorted(source.items())
    }
    return _canonical_sha256(canonical)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256_text(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _float_text(value: float) -> str:
    return format(float(value), ".17g")


def _serialize_mapping(mapping: dict[str, Any], path: Path) -> str:
    if path.suffix.lower() == ".json":
        return json.dumps(mapping, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return yaml.safe_dump(mapping, sort_keys=False, allow_unicode=False)


def _atomic_write(path: Path, text: str, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise AutowareExportError(
            f"refusing to overwrite existing output {path}; pass --force or overwrite=True"
        )
    handle: int | None = None
    temporary: Path | None = None
    try:
        handle, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            handle = None
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    except OSError as exc:
        raise AutowareExportError(f"could not atomically write {path}: {exc}") from exc
    finally:
        if handle is not None:
            os.close(handle)
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink()


def _drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _drop_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_drop_none(item) for item in value]
    return value
