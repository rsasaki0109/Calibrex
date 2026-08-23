"""Product-grade input readiness checks for the Koide external provider.

This module is deliberately a ROS-independent adapter boundary.  It consumes
the portable :class:`~calibrex.data.inspect.DatasetInspection` surface and
declared Calibrex configuration; it never imports ROS, a vendor driver, or the
upstream Koide implementation.  Missing evidence is represented as
``unknown`` on the individual check and is never promoted to a false-ready
decision.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field, field_validator

from calibrex import __version__
from calibrex.core.config import CalibrationConfig
from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.result import StrictModel
from calibrex.data.base import StreamSummary
from calibrex.data.inspect import DatasetInspection
from calibrex.data.kitti import read_png_luminance
from calibrex.data.manifest import DatasetManifest, StreamManifest, find_manifest, load_manifest

KOIDE_READINESS_SCHEMA_VERSION: Literal["slac.koide_readiness/v0.1"] = "slac.koide_readiness/v0.1"

KoideReadinessStatus = Literal["ready", "warn", "blocked"]
KoideReadinessCheckStatus = Literal["pass", "warn", "blocked", "unknown"]
KoideReadinessProfile = Literal["commercial", "research-noncommercial"]
KoideHardwareProfile = Literal["generic", "livox", "ouster", "velodyne"]

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class KoideReadinessThresholds(StrictModel):
    """Explicit, reproducible gates used by the Koide preflight.

    These thresholds are intentionally separate from evaluation/holdout
    thresholds.  They answer only whether a capture contains enough usable
    evidence to attempt fitting.
    """

    max_sync_delta_ms: float = Field(default=50.0, gt=0.0)
    min_points_per_frame: int = Field(default=100, ge=1)
    min_sampled_point_count: int = Field(default=500, ge=1)
    min_fov_azimuth_deg: float = Field(default=45.0, ge=0.0, le=360.0)
    min_fov_elevation_deg: float = Field(default=10.0, ge=0.0, le=180.0)
    min_scene_geometry_count: int = Field(default=3, ge=1)
    min_image_texture_score: float = Field(default=0.03, ge=0.0)
    min_capture_duration_s: float = Field(default=3.0, ge=0.0)
    min_static_start_duration_s: float = Field(default=0.5, ge=0.0)
    min_motion_path_length_m: float = Field(default=0.5, ge=0.0)
    max_moving_object_risk: float = Field(default=0.5, ge=0.0, le=1.0)


class KoideReadinessConfig(StrictModel):
    """Typed configuration for a Koide readiness run."""

    profile: KoideReadinessProfile = "commercial"
    profile_declared: bool = False
    hardware_profile: KoideHardwareProfile = "generic"
    hardware_profile_declared: bool = False
    camera_streams: list[str] = Field(default_factory=list)
    lidar_streams: list[str] = Field(default_factory=list)
    thresholds: KoideReadinessThresholds = Field(default_factory=KoideReadinessThresholds)
    protocol_version: str = "koide-targetless-input/v0.1"


class KoideReadinessCheck(StrictModel):
    """One auditable input check and its action-oriented evidence."""

    name: str = Field(min_length=1)
    status: KoideReadinessCheckStatus
    observed_value: Any
    evidence: dict[str, Any]
    action: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    provenance: list[str] = Field(min_length=1)
    required_for_ready: bool = True


class KoideProfileGuidance(StrictModel):
    """Minimum capture guidance for a declared LiDAR hardware profile."""

    hardware_profile: KoideHardwareProfile
    source: str
    minimum_capture_duration_s: float = Field(ge=0.0)
    minimum_static_start_duration_s: float = Field(ge=0.0)
    minimum_points_per_frame: int = Field(ge=1)
    minimum_azimuth_fov_deg: float = Field(ge=0.0, le=360.0)
    minimum_elevation_fov_deg: float = Field(ge=0.0, le=180.0)
    instructions: list[str] = Field(min_length=1)


class KoideReadinessProvenance(StrictModel):
    """Complete lineage for the preflight decision."""

    producer: Literal["calibrex"] = "calibrex"
    calibrex_version: str = __version__
    tool_name: str = "calibrex.koide-readiness"
    tool_version: str = __version__
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    protocol_version: str = "koide-targetless-input/v0.1"
    config_path: str | None = None
    config_sha256: str | None = None
    dataset_path: str
    dataset_sha256: str | None = None
    manifest_path: str | None = None
    manifest_sha256: str | None = None
    input_digests: dict[str, str] = Field(default_factory=dict)
    artifact_sha256: str | None = None
    artifact_digest_scope: str = "canonical artifact excluding provenance.artifact_sha256"

    @field_validator("config_sha256", "dataset_sha256", "manifest_sha256", "artifact_sha256")
    @classmethod
    def validate_optional_digest(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256_RE.fullmatch(value):
            raise ValueError("digest must be a 64-character SHA-256")
        return value.lower() if value is not None else None

    @field_validator("input_digests")
    @classmethod
    def validate_input_digests(cls, value: dict[str, str]) -> dict[str, str]:
        invalid = [key for key, digest in value.items() if not _SHA256_RE.fullmatch(digest)]
        if invalid:
            raise ValueError(f"invalid input digest(s): {', '.join(invalid)}")
        return {key: digest.lower() for key, digest in value.items()}


class KoideReadinessArtifact(StrictModel):
    """Schema-valid, evidence-bearing Koide input decision."""

    schema_version: Literal["slac.koide_readiness/v0.1"] = KOIDE_READINESS_SCHEMA_VERSION
    status: KoideReadinessStatus
    profile: KoideReadinessProfile
    profile_declared: bool
    hardware_profile: KoideHardwareProfile
    hardware_profile_declared: bool
    dataset_path: str
    camera_streams: list[str] = Field(default_factory=list)
    lidar_streams: list[str] = Field(default_factory=list)
    thresholds: KoideReadinessThresholds
    profile_guidance: KoideProfileGuidance
    checks: list[KoideReadinessCheck] = Field(min_length=1)
    recommendations: list[str] = Field(min_length=1)
    provenance: KoideReadinessProvenance

    def with_artifact_digest(self) -> KoideReadinessArtifact:
        """Return a copy carrying a deterministic self-digest."""

        payload = self.model_dump(mode="json", exclude_none=True)
        provenance = cast(dict[str, Any], payload["provenance"])
        provenance.pop("artifact_sha256", None)
        canonical = _canonical_json(payload)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return self.model_copy(
            update={"provenance": self.provenance.model_copy(update={"artifact_sha256": digest})}
        )

    def verify_artifact_digest(self) -> None:
        """Raise when a recorded readiness self-digest is stale."""

        recorded = self.provenance.artifact_sha256
        if recorded is None:
            raise ValueError("Koide readiness artifact has no self-digest")
        expected = self.with_artifact_digest().provenance.artifact_sha256
        if expected != recorded:
            raise ValueError(
                "Koide readiness artifact self-digest mismatch: "
                f"declared={recorded}, expected={expected}"
            )

    def save(self, path: str | Path) -> None:
        """Persist a digest-complete YAML or JSON artifact."""

        artifact = self.with_artifact_digest()
        write_mapping(Path(path), artifact.model_dump(mode="json", exclude_none=False))


def koide_readiness_json_schema() -> dict[str, Any]:
    """Return the generated JSON Schema for the artifact."""

    return KoideReadinessArtifact.model_json_schema()


def load_koide_readiness(path: str | Path) -> KoideReadinessArtifact:
    """Load a schema-valid Koide readiness artifact."""

    artifact = KoideReadinessArtifact.model_validate(read_mapping(Path(path)))
    artifact.verify_artifact_digest()
    return artifact


def evaluate_koide_readiness(
    config: CalibrationConfig,
    inspection: DatasetInspection,
    *,
    config_path: str | Path | None = None,
    command: Sequence[str] = (),
    readiness: KoideReadinessConfig | None = None,
) -> KoideReadinessArtifact:
    """Evaluate only evidence observable from config, manifest, and inspection.

    The function is intentionally deterministic apart from ``generated_at`` and
    the current git revision.  It does not run a solver and does not claim
    calibration quality or holdout performance.
    """

    factor_options = _koide_options(config)
    policy = readiness or _readiness_config_from_options(factor_options)
    profile_declared = policy.profile_declared or (
        "profile" in policy.model_fields_set and "profile_declared" not in policy.model_fields_set
    )
    hardware_profile_declared = policy.hardware_profile_declared or (
        "hardware_profile" in policy.model_fields_set
        and "hardware_profile_declared" not in policy.model_fields_set
    )
    dataset_path = Path(config.dataset.path)
    manifest_path = (
        Path(inspection.manifest)
        if inspection.manifest is not None and Path(inspection.manifest).exists()
        else _find_manifest_near(dataset_path)
    )
    manifest = _load_manifest_safely(manifest_path)
    stream_records = _stream_records(inspection, manifest)
    camera_streams = _select_stream_names(
        stream_records,
        kind="camera",
        explicit=policy.camera_streams,
        configured_topics=_configured_topics(config, kind="camera"),
    )
    lidar_streams = _select_stream_names(
        stream_records,
        kind="lidar",
        explicit=policy.lidar_streams,
        configured_topics=_configured_topics(config, kind="lidar"),
    )
    digests, config_digest = _input_digests(
        config,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        config_path=Path(config_path) if config_path is not None else None,
        factor_options=factor_options,
    )
    provenance_paths = _provenance_labels(digests)
    thresholds = policy.thresholds
    guidance = _profile_guidance(policy.hardware_profile, thresholds)
    checks: list[KoideReadinessCheck] = []

    checks.append(
        _check(
            "dataset_exists",
            "pass" if inspection.exists and dataset_path.exists() else "blocked",
            {"exists": inspection.exists and dataset_path.exists(), "path": str(dataset_path)},
            {"dataset_path": str(dataset_path), "inspection_exists": inspection.exists},
            "provide a readable dataset path and rerun calibrex doctor --workflow koide",
            provenance_paths,
            "dataset path is available"
            if inspection.exists and dataset_path.exists()
            else "dataset path is missing",
        )
    )
    checks.append(
        _stream_check(
            "camera_stream",
            camera_streams,
            stream_records,
            provenance_paths,
            "declare a camera image stream with non-zero samples",
        )
    )
    checks.append(
        _stream_check(
            "lidar_stream",
            lidar_streams,
            stream_records,
            provenance_paths,
            "declare a LiDAR point-cloud stream with non-zero samples",
        )
    )
    checks.append(
        _check(
            "fitting_inputs",
            (
                "pass"
                if inspection.exists
                and camera_streams
                and lidar_streams
                and _selected_streams_known(stream_records, camera_streams)
                and _selected_streams_known(stream_records, lidar_streams)
                else "unknown"
                if inspection.exists and camera_streams and lidar_streams
                else "blocked"
            ),
            {
                "dataset": inspection.exists,
                "camera_streams": camera_streams,
                "lidar_streams": lidar_streams,
            },
            {"stream_count": len(stream_records), "source": "DatasetInspection/manifest"},
            "provide both camera and LiDAR fitting streams before invoking Koide",
            provenance_paths,
            "camera and LiDAR fitting inputs are present"
            if (
                inspection.exists
                and camera_streams
                and lidar_streams
                and _selected_streams_known(stream_records, camera_streams)
                and _selected_streams_known(stream_records, lidar_streams)
            )
            else "result fitting inputs are incomplete",
        )
    )

    camera_sensor = _configured_sensor(config, kind="camera")
    lidar_sensor = _configured_sensor(config, kind="lidar")
    intrinsics = _camera_intrinsics(camera_sensor, inspection)
    checks.append(
        _check(
            "camera_intrinsics",
            intrinsics[0],
            intrinsics[1],
            intrinsics[2],
            "declare finite fx/fy/cx/cy or provide a decoded CameraInfo evidence record",
            _extend_provenance(provenance_paths, "config.sensors.camera.intrinsics"),
            intrinsics[3],
        )
    )
    frames = _frame_bindings(config, camera_sensor, lidar_sensor, stream_records)
    checks.append(
        _check(
            "frame_binding",
            frames[0],
            frames[1],
            frames[2],
            "declare camera and LiDAR frame_id values and bind them in the Koide options",
            _extend_provenance(provenance_paths, "config.frames/manifest.stream.frame_id"),
            frames[3],
        )
    )

    timestamp = _timestamp_evidence(
        inspection, stream_records, camera_streams, lidar_streams, thresholds
    )
    checks.append(
        _check(
            "timestamp_overlap",
            timestamp[0],
            timestamp[1],
            timestamp[2],
            "record overlapping camera and LiDAR timestamps in the manifest or bag inspection",
            provenance_paths,
            timestamp[6],
        )
    )
    checks.append(
        _check(
            "timestamp_synchronization",
            timestamp[3],
            timestamp[4],
            timestamp[5],
            (
                "provide nearest-pair synchronization evidence below "
                f"{thresholds.max_sync_delta_ms:g} ms"
            ),
            provenance_paths,
            timestamp[7],
        )
    )

    lidar_evidence = _lidar_evidence(inspection, stream_records, lidar_streams, thresholds)
    for name, item in (
        ("lidar_intensity", lidar_evidence[0]),
        ("lidar_finite_points", lidar_evidence[1]),
        ("point_density", lidar_evidence[2]),
        ("lidar_fov", lidar_evidence[3]),
        ("scene_geometry", lidar_evidence[4]),
    ):
        checks.append(_check(name, item[0], item[1], item[2], item[3], provenance_paths, item[4]))

    intensity_finite = _intensity_finite_evidence(inspection, lidar_streams)
    checks.append(
        _check(
            "lidar_intensity_finite",
            intensity_finite[0],
            intensity_finite[1],
            intensity_finite[2],
            intensity_finite[3],
            provenance_paths,
            intensity_finite[4],
        )
    )

    texture = _image_texture(inspection, manifest, camera_streams, thresholds)
    checks.append(
        _check(
            "image_texture",
            texture[0],
            texture[1],
            texture[2],
            texture[3],
            provenance_paths,
            texture[4],
        )
    )
    capture = _capture_evidence(inspection, stream_records, thresholds)
    checks.append(
        _check(
            "capture_duration",
            capture[0],
            capture[1],
            capture[2],
            capture[3],
            provenance_paths,
            capture[4],
        )
    )
    checks.append(
        _check(
            "static_start",
            capture[5],
            capture[6],
            capture[7],
            capture[8],
            provenance_paths,
            capture[9],
        )
    )
    checks.append(
        _check(
            "motion_suitability",
            capture[10],
            capture[11],
            capture[12],
            capture[13],
            provenance_paths,
            capture[14],
        )
    )
    moving = _moving_object_evidence(inspection, thresholds)
    checks.append(
        _check(
            "moving_object_risk",
            moving[0],
            moving[1],
            moving[2],
            moving[3],
            provenance_paths,
            moving[4],
        )
    )
    profile_status: KoideReadinessCheckStatus = "pass" if profile_declared else "warn"
    checks.append(
        _check(
            "execution_profile_declared",
            profile_status,
            {"profile": policy.profile, "declared": profile_declared},
            {"source": "Koide readiness options"},
            "declare profile=commercial or profile=research-noncommercial explicitly",
            _extend_provenance(provenance_paths, "factor.options.profile"),
            "execution profile is explicitly declared"
            if profile_declared
            else "execution profile was not declared; commercial-safe default is recorded",
        )
    )
    hardware_profile_status: KoideReadinessCheckStatus = (
        "pass" if hardware_profile_declared else "warn"
    )
    checks.append(
        _check(
            "hardware_profile_declared",
            hardware_profile_status,
            {
                "hardware_profile": policy.hardware_profile,
                "declared": hardware_profile_declared,
            },
            {"source": "Koide readiness options"},
            (
                "declare hardware_profile=livox, ouster, or velodyne to select "
                "profile-specific guidance"
            ),
            _extend_provenance(provenance_paths, "factor.options.hardware_profile"),
            "hardware profile is explicitly declared"
            if hardware_profile_declared
            else "hardware profile was not declared; generic guidance is shown",
        )
    )

    status: KoideReadinessStatus
    if any(check.status == "blocked" for check in checks):
        status = "blocked"
    elif any(check.status in {"warn", "unknown"} for check in checks):
        status = "warn"
    else:
        status = "ready"
    recommendations = _recommendations(checks, guidance)
    provenance = KoideReadinessProvenance(
        command=list(command),
        protocol_version=policy.protocol_version,
        tool_name="calibrex.koide-readiness",
        tool_version=__version__,
        config_path=str(config_path) if config_path is not None else None,
        config_sha256=(
            sha256_path(Path(config_path)) if config_path is not None else config_digest
        ),
        dataset_path=str(dataset_path),
        dataset_sha256=digests.get(str(dataset_path)),
        manifest_path=str(manifest_path) if manifest_path is not None else None,
        manifest_sha256=(digests.get(str(manifest_path)) if manifest_path is not None else None),
        input_digests=digests,
        git_commit=git_commit(),
    )
    artifact = KoideReadinessArtifact(
        status=status,
        profile=policy.profile,
        profile_declared=profile_declared,
        hardware_profile=policy.hardware_profile,
        hardware_profile_declared=hardware_profile_declared,
        dataset_path=str(dataset_path),
        camera_streams=camera_streams,
        lidar_streams=lidar_streams,
        thresholds=thresholds,
        profile_guidance=guidance,
        checks=checks,
        recommendations=recommendations,
        provenance=provenance,
    )
    return artifact.with_artifact_digest()


def evaluate_koide_readiness_from_config(
    config_path: str | Path,
    *,
    command: Sequence[str] = (),
) -> KoideReadinessArtifact:
    """Load a CalibrationConfig, inspect its dataset, and evaluate readiness."""

    from calibrex.core.config import load_config
    from calibrex.data.inspect import inspect_dataset

    path = Path(config_path)
    config = load_config(path)
    inspection = inspect_dataset(config.dataset)
    return evaluate_koide_readiness(config, inspection, config_path=path, command=command)


def validate_koide_readiness_for_execution(
    artifact: KoideReadinessArtifact | str | Path,
    *,
    expected_sha256: str | None = None,
    strict: bool = True,
) -> None:
    """Raise when a readiness artifact is blocked or no longer digest-valid."""

    loaded = (
        load_koide_readiness(artifact)
        if not isinstance(artifact, KoideReadinessArtifact)
        else artifact
    )
    if strict and loaded.status == "blocked":
        raise ValueError("Koide execution is blocked by the readiness artifact")
    if expected_sha256 is not None:
        actual_file = (
            sha256_path(Path(artifact))
            if not isinstance(artifact, KoideReadinessArtifact)
            else None
        )
        canonical = loaded.with_artifact_digest().provenance.artifact_sha256
        expected = expected_sha256.lower()
        if expected not in {actual_file, canonical}:
            raise ValueError("Koide readiness artifact SHA-256 mismatch")
    recorded = loaded.provenance.artifact_sha256
    if (
        recorded is not None
        and loaded.with_artifact_digest().provenance.artifact_sha256 != recorded
    ):
        raise ValueError("Koide readiness artifact self-digest mismatch")
    for path_text, expected in loaded.provenance.input_digests.items():
        actual = sha256_path(Path(path_text))
        if actual is None or actual != expected:
            raise ValueError(f"Koide readiness input digest mismatch: {path_text}")


def _check(
    name: str,
    status: KoideReadinessCheckStatus,
    observed: Any,
    evidence: dict[str, Any],
    action: str,
    provenance: list[str],
    reason: str,
) -> KoideReadinessCheck:
    return KoideReadinessCheck(
        name=name,
        status=status,
        observed_value=observed,
        evidence=evidence,
        action=action,
        reason=reason,
        provenance=provenance or ["no evidence source"],
    )


def _stream_check(
    name: str,
    selected: list[str],
    records: Mapping[str, StreamManifest | StreamSummary],
    provenance: list[str],
    action: str,
) -> KoideReadinessCheck:
    observed = {"selected": selected, "available": sorted(records)}
    if not selected:
        return _check(
            name,
            "blocked",
            observed,
            {"stream_records": len(records)},
            action,
            provenance,
            "required stream is absent",
        )
    uncertain = [name for name in selected if _stream_count(records[name]) is None]
    if uncertain:
        return _check(
            name,
            "unknown",
            observed,
            {"uncertain_streams": uncertain},
            action,
            provenance,
            "stream was discovered but its non-zero sample count is unknown",
        )
    return _check(
        name,
        "pass",
        observed,
        {"nonzero_streams": selected},
        action,
        provenance,
        "required stream is present",
    )


def _koide_options(config: CalibrationConfig) -> dict[str, Any]:
    for name in (
        "koide_lidar_camera",
        "direct_visual_lidar_calibration",
        "lidar_camera_targetless_baseline",
    ):
        factor = config.pipeline.factors.get(name)
        if factor is not None and factor.enabled:
            return dict(factor.options)
    return {}


def _readiness_config_from_options(options: Mapping[str, Any]) -> KoideReadinessConfig:
    raw = options.get("readiness")
    values = dict(raw) if isinstance(raw, Mapping) else {}
    for key in (
        "profile",
        "hardware_profile",
        "camera_streams",
        "lidar_streams",
        "protocol_version",
    ):
        if key in options and key not in values:
            values[key] = options[key]
    values["profile_declared"] = "profile" in options or "profile" in values
    values["hardware_profile_declared"] = (
        "hardware_profile" in options or "hardware_profile" in values
    )
    if "thresholds" not in values and isinstance(options.get("readiness_thresholds"), Mapping):
        values["thresholds"] = options["readiness_thresholds"]
    return KoideReadinessConfig.model_validate(values)


def _configured_sensor(config: CalibrationConfig, *, kind: Literal["camera", "lidar"]) -> Any:
    for sensor in config.sensors.values():
        if sensor.type == kind:
            return sensor
    return None


def _configured_topics(config: CalibrationConfig, *, kind: Literal["camera", "lidar"]) -> list[str]:
    return [
        sensor.topic for sensor in config.sensors.values() if sensor.type == kind and sensor.topic
    ]


def _stream_records(
    inspection: DatasetInspection,
    manifest: DatasetManifest | None,
) -> dict[str, StreamManifest | StreamSummary]:
    records: dict[str, StreamManifest | StreamSummary] = {}
    for stream in inspection.streams:
        if stream.message_count is None or stream.message_count > 0:
            records[stream.name] = stream
            if stream.topic:
                records.setdefault(stream.topic, stream)
    if manifest is not None:
        for name, manifest_stream in manifest.streams.items():
            if manifest_stream.count is None or manifest_stream.count > 0:
                records.setdefault(name, manifest_stream)
                if manifest_stream.topic:
                    # Keep the manifest's topic alias even when the reader
                    # discovered the same stream under a friendly name such
                    # (for example KITTI ``camera_left_color`` vs ``image_02``).
                    records[manifest_stream.topic] = manifest_stream
    return records


def _select_stream_names(
    records: Mapping[str, StreamManifest | StreamSummary],
    *,
    kind: Literal["camera", "lidar"],
    explicit: Sequence[str],
    configured_topics: Sequence[str],
) -> list[str]:
    if explicit:
        return sorted({name for name in explicit if name in records})
    selected: list[str] = []
    for name, stream in records.items():
        stream_kind = stream.kind
        topic = stream.topic
        sensor = stream.sensor
        text = f"{name} {topic or ''} {sensor or ''}".lower()
        is_camera = stream_kind in {"image", "rgbd"} or "camera" in text or "image" in text
        is_lidar = stream_kind == "pointcloud" or any(
            token in text for token in ("lidar", "velodyne", "ouster", "livox", "pointcloud")
        )
        if ((kind == "camera" and is_camera) or (kind == "lidar" and is_lidar)) and (
            not configured_topics or name in configured_topics or topic in configured_topics
        ):
            selected.append(name)
    return sorted(set(selected))


def _camera_intrinsics(
    sensor: Any, inspection: DatasetInspection
) -> tuple[KoideReadinessCheckStatus, Any, dict[str, Any], str]:
    intrinsics = getattr(sensor, "intrinsics", None)
    values = (
        {key: getattr(intrinsics, key, None) for key in ("fx", "fy", "cx", "cy")}
        if intrinsics is not None
        else {}
    )
    finite = all(
        isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values.values()
    )
    fx = _number(values.get("fx"))
    fy = _number(values.get("fy"))
    positive = finite and fx is not None and fy is not None and fx > 0.0 and fy > 0.0
    if positive:
        return (
            "pass",
            values,
            {"source": "config.sensors.*.intrinsics", "camera_info": False},
            "camera intrinsics are finite and usable",
        )
    camera_info = _find_key(inspection.diagnostics, ("camera_info", "camerainfo", "intrinsics"))
    camera_info_values = _camera_info_values(camera_info)
    if camera_info_values is not None:
        return (
            "pass",
            camera_info_values,
            {"source": "DatasetInspection.diagnostics.camera_info"},
            "decoded CameraInfo evidence is usable",
        )
    return (
        "blocked",
        values or None,
        {"source": "no finite config intrinsics or CameraInfo evidence"},
        "camera intrinsics/CameraInfo evidence is missing",
    )


def _frame_bindings(
    config: CalibrationConfig,
    camera_sensor: Any,
    lidar_sensor: Any,
    records: Mapping[str, StreamManifest | StreamSummary],
) -> tuple[KoideReadinessCheckStatus, dict[str, Any], dict[str, Any], str]:
    camera = _text(getattr(camera_sensor, "frame_id", None))
    lidar = _text(getattr(lidar_sensor, "frame_id", None))
    source = "config.sensors.*.frame_id"
    if camera is None:
        camera = _manifest_frame(records, "camera")
        source = "manifest.stream.frame_id"
    if lidar is None:
        lidar = _manifest_frame(records, "lidar")
        source = "manifest.stream.frame_id"
    valid = (
        camera is not None
        and lidar is not None
        and camera != lidar
        and camera in config.frames
        and lidar in config.frames
    )
    return (
        "pass" if valid else "blocked",
        {"camera_frame": camera, "lidar_frame": lidar},
        {"source": source, "config_frame_count": len(config.frames)},
        "camera and LiDAR frames are explicitly bound"
        if valid
        else "camera/LiDAR frame binding is missing or not declared in config.frames",
    )


def _manifest_frame(records: Mapping[str, StreamManifest | StreamSummary], kind: str) -> str | None:
    values: set[str] = set()
    for stream in records.values():
        if (kind == "camera" and stream.kind not in {"image", "rgbd"}) or (
            kind == "lidar" and stream.kind != "pointcloud"
        ):
            continue
        frame = getattr(stream, "frame_id", None)
        if frame:
            values.add(frame)
    return next(iter(values)) if len(values) == 1 else None


def _timestamp_evidence(
    inspection: DatasetInspection,
    records: Mapping[str, StreamManifest | StreamSummary],
    camera_streams: Sequence[str],
    lidar_streams: Sequence[str],
    thresholds: KoideReadinessThresholds,
) -> tuple[
    KoideReadinessCheckStatus,
    Any,
    dict[str, Any],
    KoideReadinessCheckStatus,
    Any,
    dict[str, Any],
    str,
    str,
]:
    diagnostics = inspection.diagnostics
    alignment = _find_key(diagnostics, ("timestamp_alignment", "synchronization", "sync"))
    if not isinstance(alignment, Mapping):
        flat_alignment = {
            key: _find_key(diagnostics, (key,))
            for key in (
                "camera_lidar_pair_count",
                "camera_lidar_mean_abs_dt_ms",
                "camera_lidar_max_abs_dt_ms",
            )
        }
        alignment = {key: value for key, value in flat_alignment.items() if value is not None}
    if isinstance(alignment, Mapping):
        pairs = _number(alignment.get("camera_lidar_pair_count"))
        mean_delta = _number(alignment.get("camera_lidar_mean_abs_dt_ms"))
        max_delta = _number(alignment.get("camera_lidar_max_abs_dt_ms"))
        overlap_status: KoideReadinessCheckStatus = "pass" if pairs and pairs > 0 else "unknown"
        sync_status: KoideReadinessCheckStatus = (
            "unknown"
            if max_delta is None
            else ("pass" if max_delta <= thresholds.max_sync_delta_ms else "blocked")
        )
        return (
            overlap_status,
            {"pair_count": pairs},
            dict(alignment),
            sync_status,
            {"mean_abs_dt_ms": mean_delta, "max_abs_dt_ms": max_delta},
            dict(alignment),
            "timestamp overlap is evidenced by nearest camera/LiDAR pairs"
            if pairs
            else "timestamp overlap evidence is unavailable",
            "timestamp synchronization is within the declared gate"
            if sync_status == "pass"
            else "timestamp synchronization evidence is missing or exceeds the gate",
        )
    stats = _diagnostic_streams(diagnostics)
    camera_ranges = [_timestamp_range(stats.get(name)) for name in camera_streams]
    lidar_ranges = [_timestamp_range(stats.get(name)) for name in lidar_streams]
    ranges = [item for item in (*camera_ranges, *lidar_ranges) if item is not None]
    overlap = _range_overlap(camera_ranges, lidar_ranges)
    if overlap is not None and overlap > 0.0:
        range_overlap_status: KoideReadinessCheckStatus = "pass"
        observed: Any = {"overlap_s": overlap, "stream_ranges": ranges}
        evidence = {"source": "inspection diagnostics stream timestamp ranges"}
    else:
        range_overlap_status = "unknown"
        observed = {"overlap_s": overlap, "stream_ranges": ranges}
        evidence = {"source": "no camera/LiDAR timestamp range evidence"}
    return (
        range_overlap_status,
        observed,
        evidence,
        "unknown",
        None,
        evidence,
        "timestamp overlap is evidenced by stream ranges"
        if range_overlap_status == "pass"
        else "timestamp overlap evidence is unavailable",
        "nearest-pair synchronization evidence is unavailable",
    )


def _lidar_evidence(
    inspection: DatasetInspection,
    records: Mapping[str, StreamManifest | StreamSummary],
    lidar_streams: Sequence[str],
    thresholds: KoideReadinessThresholds,
) -> tuple[tuple[KoideReadinessCheckStatus, Any, dict[str, Any], str, str], ...]:
    diagnostics = inspection.diagnostics
    stats = _diagnostic_streams(diagnostics)
    candidates = [stats.get(name) for name in lidar_streams if stats.get(name) is not None]
    candidate = candidates[0] if candidates else _first_lidar_diagnostic(diagnostics)
    fields = _stream_fields(records, lidar_streams)
    has_intensity = _first_value(
        candidate,
        ("has_intensity", "intensity_available", "reflectivity_available"),
    )
    if has_intensity is None and fields:
        has_intensity = any(
            field.lower() in {"intensity", "reflectivity", "reflectance"} for field in fields
        )
    intensity_status: KoideReadinessCheckStatus = "pass" if has_intensity is True else "warn"
    intensity_reason = (
        "LiDAR intensity/reflectivity is available"
        if has_intensity is True
        else "LiDAR intensity evidence is unavailable"
        if has_intensity is None
        else "LiDAR intensity field is absent; provider compatibility must be reviewed"
    )
    nonfinite = _first_value(
        candidate,
        (
            "sampled_nonfinite_xyz_count",
            "nonfinite_xyz_count",
            "sampled_nonfinite_point_count",
            "nonfinite_point_count",
            "xyz_nonfinite_count",
        ),
    )
    finite_status: KoideReadinessCheckStatus = (
        "pass"
        if nonfinite is not None and float(nonfinite) == 0.0
        else "blocked"
        if nonfinite is not None
        else "unknown"
    )
    finite_reason = (
        "sampled xyz values are finite"
        if finite_status == "pass"
        else "non-finite xyz samples were observed"
        if finite_status == "blocked"
        else "finite xyz evidence is unavailable"
    )
    point_count = _number(
        _first_value(
            candidate,
            (
                "sampled_point_count",
                "sampled_points",
                "point_count",
                "total_point_count",
                "sampled_raw_point_count",
                "point_density",
            ),
        )
    )
    frame_count = _number(
        _first_value(
            candidate,
            ("sampled_message_count", "sampled_frame_count", "sampled_frames", "frame_count"),
        )
    )
    explicit_per_frame = _number(
        _first_value(
            candidate,
            ("points_per_frame", "sampled_points_per_frame", "mean_points_per_frame"),
        )
    )
    per_frame = (
        explicit_per_frame
        if explicit_per_frame is not None
        else point_count / frame_count
        if point_count is not None and frame_count and frame_count > 0
        else point_count
    )
    density_status: KoideReadinessCheckStatus = (
        "unknown"
        if point_count is None
        else "pass"
        if point_count >= thresholds.min_sampled_point_count
        and (per_frame is None or per_frame >= thresholds.min_points_per_frame)
        else "blocked"
    )
    density_reason = (
        "LiDAR point density meets the declared gate"
        if density_status == "pass"
        else "sampled LiDAR point density is below the declared gate"
        if density_status == "blocked"
        else "LiDAR point density evidence is unavailable"
    )
    azimuth = _number(
        _first_value(candidate, ("fov_azimuth_deg", "azimuth_fov_deg", "fov_azimuth"))
    )
    elevation = _number(
        _first_value(candidate, ("fov_elevation_deg", "elevation_fov_deg", "fov_elevation"))
    )
    fov_status: KoideReadinessCheckStatus = (
        "unknown"
        if azimuth is None or elevation is None
        else "pass"
        if azimuth >= thresholds.min_fov_azimuth_deg
        and elevation >= thresholds.min_fov_elevation_deg
        else "blocked"
    )
    fov_reason = (
        "LiDAR FOV meets the declared gate"
        if fov_status == "pass"
        else "LiDAR FOV is below the declared gate"
        if fov_status == "blocked"
        else "LiDAR FOV evidence is unavailable"
    )
    geometry_count = _number(
        _first_value(
            candidate,
            (
                "planarity_voxel_count",
                "plane_count",
                "normal_rank",
                "geometry_count",
                "surface_count",
                "scene_geometry_count",
            ),
        )
    )
    geometry_status: KoideReadinessCheckStatus = (
        "unknown"
        if geometry_count is None
        else "pass"
        if geometry_count >= thresholds.min_scene_geometry_count
        else "blocked"
    )
    geometry_reason = (
        "scene geometry contains enough independent structure"
        if geometry_status == "pass"
        else "scene geometry is too sparse or degenerate"
        if geometry_status == "blocked"
        else "scene geometry evidence is unavailable"
    )
    provenance = {"source": "inspection diagnostics/manifest", "lidar_streams": list(lidar_streams)}
    return (
        (
            intensity_status,
            {"has_intensity": has_intensity},
            provenance,
            "declare/verify intensity or reflectivity fields for the selected LiDAR",
            intensity_reason,
        ),
        (
            finite_status,
            {"sampled_nonfinite_xyz_count": nonfinite},
            provenance,
            "remove or repair non-finite xyz samples before fitting",
            finite_reason,
        ),
        (
            density_status,
            {
                "sampled_point_count": point_count,
                "sampled_frame_count": frame_count,
                "points_per_frame": per_frame,
            },
            provenance,
            (
                f"capture at least {thresholds.min_points_per_frame} points per frame and "
                f"{thresholds.min_sampled_point_count} sampled points"
            ),
            density_reason,
        ),
        (
            fov_status,
            {"fov_azimuth_deg": azimuth, "fov_elevation_deg": elevation},
            provenance,
            (
                f"capture at least {thresholds.min_fov_azimuth_deg:g}° azimuth and "
                f"{thresholds.min_fov_elevation_deg:g}° elevation FOV"
            ),
            fov_reason,
        ),
        (
            geometry_status,
            {"scene_geometry_count": geometry_count},
            provenance,
            (
                f"include at least {thresholds.min_scene_geometry_count} independent "
                "surfaces/geometry directions"
            ),
            geometry_reason,
        ),
    )


def _intensity_finite_evidence(
    inspection: DatasetInspection,
    lidar_streams: Sequence[str],
) -> tuple[KoideReadinessCheckStatus, Any, dict[str, Any], str, str]:
    """Check finite intensity/reflectivity samples when a decoder exposes them."""

    diagnostics = inspection.diagnostics
    stats = _diagnostic_streams(diagnostics)
    candidate = next((stats.get(name) for name in lidar_streams if stats.get(name)), None)
    if candidate is None:
        candidate = _first_lidar_diagnostic(diagnostics)
    nonfinite = _first_value(
        candidate,
        (
            "sampled_nonfinite_intensity_count",
            "nonfinite_intensity_count",
            "nonfinite_reflectivity_count",
            "sampled_intensity_nonfinite_count",
        ),
    )
    count = _number(nonfinite)
    if count is None:
        return (
            "unknown",
            {"sampled_nonfinite_intensity_count": None},
            {"source": "no finite intensity/reflectivity evidence"},
            "provide finite intensity/reflectivity samples or explicitly document their absence",
            "finite intensity/reflectivity evidence is unavailable",
        )
    status: KoideReadinessCheckStatus = "pass" if count == 0.0 else "blocked"
    return (
        status,
        {"sampled_nonfinite_intensity_count": count},
        {"source": "inspection diagnostics"},
        "remove non-finite intensity/reflectivity values before fitting",
        "sampled intensity/reflectivity values are finite"
        if status == "pass"
        else "non-finite intensity/reflectivity samples were observed",
    )


def _image_texture(
    inspection: DatasetInspection,
    manifest: DatasetManifest | None,
    camera_streams: Sequence[str],
    thresholds: KoideReadinessThresholds,
) -> tuple[KoideReadinessCheckStatus, Any, dict[str, Any], str, str]:
    diagnostics = inspection.diagnostics
    value = _number(
        _find_key(
            diagnostics,
            (
                "image_texture_score",
                "texture_score",
                "texture",
                "edge_gradient_mean",
                "gradient_mean",
            ),
        )
    )
    source = "inspection diagnostics"
    if value is None and manifest is not None:
        value = _sample_manifest_texture(manifest, camera_streams)
        source = "manifest image samples" if value is not None else source
    if value is None:
        return (
            "unknown",
            None,
            {"source": "no image texture evidence"},
            "provide decoded image texture/gradient evidence",
            "image texture evidence is unavailable",
        )
    status: KoideReadinessCheckStatus = (
        "pass" if value >= thresholds.min_image_texture_score else "blocked"
    )
    return (
        status,
        {"texture_score": value},
        {"source": source},
        "capture textured views with visible edges and repeat the readiness check",
        "image texture meets the declared gate"
        if status == "pass"
        else "image texture is below the declared gate",
    )


def _capture_evidence(
    inspection: DatasetInspection,
    records: Mapping[str, StreamManifest | StreamSummary],
    thresholds: KoideReadinessThresholds,
) -> tuple[Any, ...]:
    diagnostics = inspection.diagnostics
    duration = _number(
        _find_key(
            diagnostics,
            (
                "duration_sec",
                "duration_s",
                "time_span_s",
                "capture_duration_s",
                "capture_duration",
            ),
        )
    )
    if duration is None:
        motions = _find_key(diagnostics, ("odometry_motion", "oxts_motion"))
        if isinstance(motions, Mapping):
            duration = _number(_first_value(motions, ("duration_sec", "time_span_s")))
            if duration is None:
                for value in motions.values():
                    if isinstance(value, Mapping):
                        duration = _number(_first_value(value, ("duration_sec", "time_span_s")))
                        if duration is not None:
                            break
    duration_status: KoideReadinessCheckStatus = (
        "unknown"
        if duration is None
        else "pass"
        if duration >= thresholds.min_capture_duration_s
        else "blocked"
    )
    static = _number(
        _find_key(
            diagnostics,
            (
                "static_start_duration_s",
                "initial_static_duration_s",
                "static_duration_s",
                "static_start_s",
                "initial_static_sec",
            ),
        )
    )
    static_status: KoideReadinessCheckStatus = (
        "unknown"
        if static is None
        else "pass"
        if static >= thresholds.min_static_start_duration_s
        else "blocked"
    )
    path = _number(
        _find_key(
            diagnostics,
            (
                "path_length_m",
                "motion_path_length_m",
                "motion_path_m",
                "translation_distance_m",
                "distance_m",
                "translation_m",
            ),
        )
    )
    motion_flag = _find_key(diagnostics, ("motion_suitability", "motion_ready"))
    motion_status: KoideReadinessCheckStatus = (
        "unknown"
        if path is None and not isinstance(motion_flag, bool)
        else "pass"
        if motion_flag is True or (path is not None and path >= thresholds.min_motion_path_length_m)
        else "blocked"
    )
    common = {
        "source": "inspection diagnostics/stream timestamp ranges",
        "stream_count": len(records),
    }
    return (
        duration_status,
        {"duration_s": duration},
        common,
        f"capture at least {thresholds.min_capture_duration_s:g} seconds",
        "capture duration meets the declared gate"
        if duration_status == "pass"
        else "capture duration is below the declared gate"
        if duration_status == "blocked"
        else "capture duration evidence is unavailable",
        static_status,
        {"static_start_duration_s": static},
        common,
        (
            "hold the rig static for at least "
            f"{thresholds.min_static_start_duration_s:g} seconds before motion"
        ),
        "static-start duration meets the declared gate"
        if static_status == "pass"
        else "static-start duration is below the declared gate"
        if static_status == "blocked"
        else "static-start evidence is unavailable",
        motion_status,
        {"path_length_m": path, "motion_suitability": motion_flag},
        common,
        (
            "include at least "
            f"{thresholds.min_motion_path_length_m:g} m of controlled motion or "
            "declare motion_suitability=true"
        ),
        "capture motion meets the declared gate"
        if motion_status == "pass"
        else "motion is insufficient for targetless fitting"
        if motion_status == "blocked"
        else "motion suitability evidence is unavailable",
    )


def _moving_object_evidence(
    inspection: DatasetInspection, thresholds: KoideReadinessThresholds
) -> tuple[KoideReadinessCheckStatus, Any, dict[str, Any], str, str]:
    value = _number(
        _find_key(
            inspection.diagnostics,
            (
                "moving_object_risk",
                "moving_object_fraction",
                "dynamic_fraction",
                "dynamic_object_fraction",
                "dynamic_risk",
            ),
        )
    )
    if value is None:
        return (
            "unknown",
            None,
            {"source": "no dynamic-scene evidence"},
            "provide a moving-object risk estimate or capture a static scene",
            "moving-object risk evidence is unavailable",
        )
    status: KoideReadinessCheckStatus = (
        "pass" if value <= thresholds.max_moving_object_risk else "blocked"
    )
    return (
        status,
        {"moving_object_risk": value},
        {"source": "inspection diagnostics"},
        "mask moving objects or recapture in a static scene",
        "moving-object risk is within the declared gate"
        if status == "pass"
        else "moving-object risk exceeds the declared gate",
    )


def _profile_guidance(
    profile: KoideHardwareProfile, thresholds: KoideReadinessThresholds
) -> KoideProfileGuidance:
    guidance: dict[KoideHardwareProfile, tuple[float, float, int, float, float, list[str]]] = {
        "generic": (
            thresholds.min_capture_duration_s,
            thresholds.min_static_start_duration_s,
            thresholds.min_points_per_frame,
            thresholds.min_fov_azimuth_deg,
            thresholds.min_fov_elevation_deg,
            [
                "use a rigid mount",
                "capture overlapping textured geometry",
                "keep the first segment static",
            ],
        ),
        "livox": (
            5.0,
            1.0,
            150,
            60.0,
            12.0,
            [
                "preserve point-time/offset_time fields",
                "rotate or translate slowly across walls and ground",
                "avoid sparse single-plane scenes",
            ],
        ),
        "ouster": (
            3.0,
            0.5,
            500,
            90.0,
            20.0,
            [
                "retain per-scan timestamps",
                "include broad azimuth overlap",
                "avoid clipping the camera FOV",
            ],
        ),
        "velodyne": (
            3.0,
            0.5,
            1000,
            90.0,
            15.0,
            [
                "use several scans with timestamp overlap",
                "include vertical structure and road-side edges",
                "avoid ego-motion blur at capture start",
            ],
        ),
    }
    values = guidance[profile]
    return KoideProfileGuidance(
        hardware_profile=profile,
        source="calibrex Koide profile guidance v0.1",
        minimum_capture_duration_s=values[0],
        minimum_static_start_duration_s=values[1],
        minimum_points_per_frame=values[2],
        minimum_azimuth_fov_deg=values[3],
        minimum_elevation_fov_deg=values[4],
        instructions=values[5],
    )


def _recommendations(
    checks: Sequence[KoideReadinessCheck], guidance: KoideProfileGuidance
) -> list[str]:
    messages = [check.action for check in checks if check.status in {"blocked", "warn", "unknown"}]
    if not messages:
        messages = ["readiness is ready; execute Koide only with the recorded artifact"]
    messages.extend(guidance.instructions[:2])
    return list(dict.fromkeys(messages))


def _input_digests(
    config: CalibrationConfig,
    *,
    dataset_path: Path,
    manifest_path: Path | None,
    config_path: Path | None,
    factor_options: Mapping[str, Any],
) -> tuple[dict[str, str], str]:
    paths: list[Path] = [dataset_path]
    if config_path is not None:
        paths.append(config_path)
    if manifest_path is not None:
        paths.append(manifest_path)
    for raw in (factor_options.get("input_paths", factor_options.get("input_artifacts", [])),):
        if isinstance(raw, (str, Path)):
            paths.append(Path(raw))
        elif isinstance(raw, Sequence):
            paths.extend(Path(str(item)) for item in raw)
    digests: dict[str, str] = {}
    for path in paths:
        digest = sha256_path(path)
        if digest is not None:
            digests[str(path)] = digest
    payload = _canonical_json(config.model_dump(mode="json"))
    config_digest = (
        sha256_path(config_path)
        if config_path is not None
        else hashlib.sha256(payload.encode("utf-8")).hexdigest()
    )
    return digests, config_digest or hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _provenance_labels(digests: Mapping[str, str]) -> list[str]:
    return [f"{path}@sha256:{digest}" for path, digest in sorted(digests.items())] or [
        "inspection diagnostics (no file digest available)"
    ]


def _extend_provenance(values: Sequence[str], item: str) -> list[str]:
    return list(dict.fromkeys((*values, item)))


def _load_manifest_safely(path: Path | None) -> DatasetManifest | None:
    if path is None:
        return None
    try:
        return load_manifest(path)
    except (OSError, ValueError):
        return None


def _find_manifest_near(path: Path) -> Path | None:
    """Find a manifest at the dataset or one of its declared parent roots."""

    direct = find_manifest(path)
    if direct is not None:
        return direct
    current = path if path.is_dir() else path.parent
    for parent in (current, *current.parents):
        candidate = find_manifest(parent)
        if candidate is not None:
            return candidate
    return None


def _stream_fields(
    records: Mapping[str, StreamManifest | StreamSummary], names: Sequence[str]
) -> list[str]:
    values: list[str] = []
    for name in names:
        stream = records.get(name)
        if isinstance(stream, StreamManifest):
            values.extend(stream.fields)
    return values


def _stream_count(stream: StreamManifest | StreamSummary) -> int | None:
    return stream.count if isinstance(stream, StreamManifest) else stream.message_count


def _selected_streams_known(
    records: Mapping[str, StreamManifest | StreamSummary],
    selected: Sequence[str],
) -> bool:
    return all(_stream_count(records[name]) is not None for name in selected)


def _diagnostic_streams(diagnostics: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            streams = value.get("streams")
            if isinstance(streams, list):
                for item in streams:
                    if isinstance(item, Mapping):
                        name = _text(item.get("topic")) or _text(item.get("name"))
                        if name:
                            result[name] = item
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(diagnostics)
    return result


def _first_lidar_diagnostic(diagnostics: Mapping[str, Any]) -> Mapping[str, Any] | None:
    streams = _diagnostic_streams(diagnostics)
    for value in streams.values():
        text = f"{value.get('topic', '')} {value.get('message_type', '')}".lower()
        if (
            "lidar" in text
            or "pointcloud" in text
            or "velodyne" in text
            or "livox" in text
            or "ouster" in text
        ):
            return value
    for key in ("velodyne_points", "livox_pcd", "lidar"):
        candidate = diagnostics.get(key)
        if isinstance(candidate, Mapping):
            return candidate
    nested = _find_key(
        diagnostics,
        ("lidar_evidence", "pointcloud_evidence", "scene_geometry"),
    )
    if isinstance(nested, Mapping):
        return nested
    return None


def _timestamp_range(value: Mapping[str, Any] | None) -> tuple[float, float] | None:
    if value is None:
        return None
    first = _number(value.get("first_timestamp_ns"))
    last = _number(value.get("last_timestamp_ns"))
    if first is None or last is None:
        return None
    scale = 1_000_000_000.0 if max(abs(first), abs(last)) > 1_000_000.0 else 1.0
    return first / scale, last / scale


def _range_overlap(
    camera: Sequence[tuple[float, float] | None], lidar: Sequence[tuple[float, float] | None]
) -> float | None:
    values = [item for item in (*camera, *lidar) if item is not None]
    if len(values) < 2:
        return None
    camera_start = max(item[0] for item in camera if item is not None)
    camera_end = min(item[1] for item in camera if item is not None)
    lidar_start = max(item[0] for item in lidar if item is not None)
    lidar_end = min(item[1] for item in lidar if item is not None)
    return max(0.0, min(camera_end, lidar_end) - max(camera_start, lidar_start))


def _sample_manifest_texture(
    manifest: DatasetManifest, camera_streams: Sequence[str]
) -> float | None:
    for name in camera_streams:
        stream = manifest.streams.get(name)
        if stream is None or stream.path is None:
            continue
        base = Path(stream.path)
        matches = sorted(base.parent.glob(base.name)) if base.parent.exists() else []
        for image_path in matches[:2]:
            image = read_png_luminance(image_path)
            if image is None or image.width < 2 or image.height < 2:
                continue
            differences: list[float] = []
            for row in image.rows:
                differences.extend(
                    abs(row[index + 1] - row[index]) for index in range(len(row) - 1)
                )
            for y in range(image.height - 1):
                differences.extend(
                    abs(image.rows[y + 1][x] - image.rows[y][x]) for x in range(image.width)
                )
            if differences:
                return sum(differences) / len(differences) / 255.0
    return None


def _find_key(value: Any, keys: Iterable[str]) -> Any:
    wanted = {key.lower() for key in keys}
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in wanted:
                return item
        for item in value.values():
            found = _find_key(item, wanted)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_key(item, wanted)
            if found is not None:
                return found
    return None


def _first_value(value: Mapping[str, Any] | None, keys: Iterable[str]) -> Any:
    if value is None:
        return None
    for key in keys:
        if key in value:
            return value[key]
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_number(value: Any) -> bool:
    return _number(value) is not None


def _camera_info_values(value: Any) -> dict[str, float] | None:
    """Normalize direct or ROS-style ``CameraInfo.K`` evidence."""

    if not isinstance(value, Mapping):
        return None
    direct = {key: _number(value.get(key)) for key in ("fx", "fy", "cx", "cy")}
    if all(item is not None for item in direct.values()):
        return cast(dict[str, float], direct)
    matrix = value.get("K", value.get("k", value.get("camera_matrix")))
    if isinstance(matrix, Mapping):
        matrix = matrix.get("data")
    if isinstance(matrix, Sequence) and not isinstance(matrix, (str, bytes)) and len(matrix) >= 9:
        values = {
            "fx": _number(matrix[0]),
            "fy": _number(matrix[4]),
            "cx": _number(matrix[2]),
            "cy": _number(matrix[5]),
        }
        if all(item is not None for item in values.values()):
            return cast(dict[str, float], values)
    return None


def _text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _canonical_json(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


__all__ = [
    "KOIDE_READINESS_SCHEMA_VERSION",
    "KoideHardwareProfile",
    "KoideProfileGuidance",
    "KoideReadinessArtifact",
    "KoideReadinessCheck",
    "KoideReadinessCheckStatus",
    "KoideReadinessConfig",
    "KoideReadinessProvenance",
    "KoideReadinessStatus",
    "KoideReadinessThresholds",
    "evaluate_koide_readiness",
    "evaluate_koide_readiness_from_config",
    "koide_readiness_json_schema",
    "load_koide_readiness",
    "validate_koide_readiness_for_execution",
]
