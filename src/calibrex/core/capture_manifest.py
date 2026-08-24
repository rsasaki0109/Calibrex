"""Schema-versioned capture/session inventory and readiness evidence.

This module is deliberately an adapter boundary.  The public models describe
capture identity, source integrity, stream metadata, and cross-stream frame /
clock bindings without importing ROS (or any vendor message package).  The
``build_capture_manifest`` helper uses the pure-Python readers in
``calibrex.data`` when they are available and preserves an explicit
``unsupported``/``unknown`` state when a payload cannot be decoded.

The distinction between *unknown* and *false* is important here: a bag that
contains Image, CameraInfo, RadarScan, or vendor topics which this installation
cannot decode is never reported as ``ready`` merely because a file and a topic
row exist. Exact standard RadarScan types are supported only after a sampled
payload validates successfully.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from pydantic import AliasChoices, Field, StringConstraints, field_validator, model_validator

from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.mcap_integrity import McapIntegrityEvidence
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.result import StrictModel

CAPTURE_MANIFEST_SCHEMA_VERSION: Literal["slac.capture_manifest/v0.1"] = (
    "slac.capture_manifest/v0.1"
)

CaptureStatus = Literal["ready", "warn", "blocked"]
CheckStatus = Literal["pass", "warn", "blocked"]
EvidenceStatus = Literal["known", "unknown", "not_applicable"]
SourceFormat = Literal["rosbag1", "rosbag2", "mcap", "files"]
DecodeStatus = Literal["supported", "unsupported", "unknown"]
CalibrationStatus = Literal[
    "calibrated",
    "uncalibrated",
    "mixed",
    "unknown",
    "not_applicable",
]
SensorType = Literal["camera", "camera_info", "lidar", "imu", "radar", "odometry", "other"]
StreamKind = Literal[
    "image",
    "camera_info",
    "depth_image",
    "rgbd",
    "pointcloud",
    "imu",
    "radar",
    "odometry",
    "metadata",
    "trajectory",
    "file",
    "other",
]

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
Sha256Digest = Annotated[str, StringConstraints(pattern=_SHA256_PATTERN)]


def _reject_ambiguous_aliases(
    value: Any,
    aliases: Sequence[tuple[str, str]],
) -> Any:
    """Reject payloads that provide both spellings of one public field.

    The v0.1 reader accepts a small set of migration aliases, but serialized
    artifacts always use the canonical field name.  Accepting both spellings
    in one payload would otherwise make the winner depend on pydantic's
    validation order and would make provenance difficult to audit.
    """

    if not isinstance(value, Mapping):
        return value
    collisions = [
        f"{canonical}/{alias}"
        for canonical, alias in aliases
        if canonical in value and alias in value
    ]
    if collisions:
        raise ValueError("ambiguous field aliases: " + ", ".join(collisions))
    return value


def _validate_sha256(value: str | None, *, field_name: str) -> str | None:
    """Validate a digest as exactly 64 lowercase hexadecimal characters."""

    if value is None:
        return None
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field_name} must be a lowercase 64-character SHA-256 digest")
    return value


def _require_unique_values(field_name: str, values: Sequence[str]) -> None:
    """Reject empty or duplicate declaration values."""

    empty = [str(index) for index, value in enumerate(values) if not value]
    if empty:
        raise ValueError(f"{field_name} contains empty values at indexes: " + ", ".join(empty))
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"{field_name} contains duplicate IDs: " + ", ".join(duplicates))


def _require_unique_ids(kind: str, values: Sequence[str]) -> None:
    """Reject missing or duplicate inventory IDs."""

    _require_unique_values(f"{kind} IDs", values)


class CaptureQoS(StrictModel):
    """Transport QoS inventory with explicit unknown values."""

    status: EvidenceStatus = "unknown"
    reliability: str | None = None
    durability: str | None = None
    history: str | None = None
    depth: int | None = Field(default=None, ge=0)
    liveliness: str | None = None
    deadline_ns: int | None = Field(default=None, ge=0)
    lifespan_ns: int | None = Field(default=None, ge=0)
    source: str | None = None


class SensorIdentity(StrictModel):
    """Physical sensor identity and mounting/frame lineage.

    ``type`` is intentionally a small core vocabulary.  Vendor-specific
    message names belong in :class:`StreamInventory.message_type`, not in the
    core API.
    """

    sensor_id: str = ""
    type: SensorType = Field(
        default="other",
        validation_alias=AliasChoices("type", "sensor_type"),
        serialization_alias="type",
    )
    serial: str | None = Field(
        default=None,
        validation_alias=AliasChoices("serial", "serial_number"),
        serialization_alias="serial",
    )
    model: str | None = None
    firmware: str | None = None
    mount_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("mount_id", "mount"),
        serialization_alias="mount_id",
    )
    frame_id: str | None = None
    manufacturer: str | None = None
    adapter: str | None = None
    identity_status: Literal["complete", "partial", "unknown"] = "unknown"
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def reject_alias_collisions(cls, value: Any) -> Any:
        return _reject_ambiguous_aliases(
            value,
            (("serial", "serial_number"), ("mount_id", "mount")),
        )

    @model_validator(mode="after")
    def infer_identity_status(self) -> SensorIdentity:
        if all(
            value
            for value in (
                self.sensor_id,
                self.type != "other",
                self.serial,
                self.model,
                self.firmware,
                self.mount_id,
                self.frame_id,
            )
        ):
            self.identity_status = "complete"
        elif any(
            value
            for value in (
                self.sensor_id,
                self.type != "other",
                self.serial,
                self.model,
                self.firmware,
                self.mount_id,
                self.frame_id,
            )
        ):
            self.identity_status = "partial"
        else:
            self.identity_status = "unknown"
        return self


class SourceInventory(StrictModel):
    """Immutable identity and container evidence for one capture source."""

    source_id: str = "source-0"
    path: str = ""
    format: SourceFormat = Field(
        default="files",
        validation_alias=AliasChoices("format", "source_format"),
        serialization_alias="format",
    )
    exists: bool = False
    sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    size_bytes: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices("size_bytes", "size"),
        serialization_alias="size_bytes",
    )
    file_count: int | None = Field(default=None, ge=0)
    storage_identifier: str | None = Field(
        default=None,
        validation_alias=AliasChoices("storage_identifier", "storage"),
        serialization_alias="storage_identifier",
    )
    storage_status: EvidenceStatus = "unknown"
    index_status: EvidenceStatus = Field(
        default="unknown",
        validation_alias=AliasChoices("index_status", "index"),
        serialization_alias="index_status",
    )
    crc_status: EvidenceStatus = Field(
        default="unknown",
        validation_alias=AliasChoices("crc_status", "crc"),
        serialization_alias="crc_status",
    )
    mcap_integrity: McapIntegrityEvidence | None = Field(
        default=None,
        validation_alias=AliasChoices("mcap_integrity", "integrity"),
        serialization_alias="mcap_integrity",
    )
    metadata_path: str | None = None
    metadata_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    compression_format: str | None = None
    compression_mode: str | None = None
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def reject_alias_collisions(cls, value: Any) -> Any:
        return _reject_ambiguous_aliases(
            value,
            (
                ("format", "source_format"),
                ("size_bytes", "size"),
                ("storage_identifier", "storage"),
                ("index_status", "index"),
                ("crc_status", "crc"),
                ("mcap_integrity", "integrity"),
            ),
        )

    @field_validator("sha256", "metadata_sha256")
    @classmethod
    def validate_digests(cls, value: str | None, info: Any) -> str | None:
        return _validate_sha256(value, field_name=str(info.field_name))

    @property
    def source_format(self) -> SourceFormat:
        """Compatibility spelling for callers that avoid ``format``."""

        return self.format

    @property
    def storage(self) -> str | None:
        """Compatibility spelling for the storage identifier."""

        return self.storage_identifier

    @property
    def index(self) -> EvidenceStatus:
        """Return the explicit index evidence state."""

        return self.index_status

    @property
    def crc(self) -> EvidenceStatus:
        """Return the explicit CRC evidence state."""

        return self.crc_status

    @property
    def integrity(self) -> McapIntegrityEvidence | None:
        """Compatibility spelling for MCAP integrity evidence."""

        return self.mcap_integrity


class StreamInventory(StrictModel):
    """Portable inventory for one topic, file stream, or timestamped source."""

    stream_id: str = ""
    name: str | None = None
    kind: StreamKind = "other"
    topic: str | None = None
    path: str | None = None
    sensor_id: str | None = None
    message_type: str | None = None
    serialization_format: str | None = Field(
        default=None,
        validation_alias=AliasChoices("serialization_format", "serialization"),
        serialization_alias="serialization_format",
    )
    clock_domain: str | None = Field(
        default=None,
        validation_alias=AliasChoices("clock_domain", "clock"),
        serialization_alias="clock_domain",
    )
    timestamp_source: str | None = None
    frame_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("frame_id", "frame"),
        serialization_alias="frame_id",
    )
    qos: CaptureQoS | None = None
    message_count: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices("message_count", "count"),
        serialization_alias="message_count",
    )
    first_timestamp_ns: int | None = None
    last_timestamp_ns: int | None = None
    time_span_ns: int | None = Field(default=None, ge=0)
    time_span_s: float | None = Field(default=None, ge=0.0)
    rate_hz: float | None = Field(default=None, ge=0.0)
    missing_count: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices("missing_count", "missing"),
        serialization_alias="missing_count",
    )
    duplicate_count: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices("duplicate_count", "duplicate"),
        serialization_alias="duplicate_count",
    )
    out_of_order_count: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices("out_of_order_count", "out_of_order"),
        serialization_alias="out_of_order_count",
    )
    decode_status: DecodeStatus = "unknown"
    calibration_status: CalibrationStatus = "not_applicable"
    # RadarScan evidence is optional for non-radar streams and populated only
    # after a valid sampled decode of the exact official message type.
    radar_return_count: int | None = Field(default=None, ge=0)
    radar_range_min_m: float | None = Field(default=None, ge=0.0)
    radar_range_max_m: float | None = Field(default=None, ge=0.0)
    radar_azimuth_span_rad: float | None = Field(default=None, ge=0.0)
    radar_elevation_span_rad: float | None = Field(default=None, ge=0.0)
    radar_doppler_span_mps: float | None = Field(default=None, ge=0.0)
    radar_duplicate_return_count: int | None = Field(default=None, ge=0)
    radar_diversity_status: Literal[
        "strong", "weak", "empty", "unknown", "not_applicable"
    ] = "not_applicable"
    frame_conflict: bool = False
    capabilities: list[str] = Field(default_factory=list)
    unknown_fields: list[str] = Field(default_factory=list)
    source_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def reject_alias_collisions(cls, value: Any) -> Any:
        return _reject_ambiguous_aliases(
            value,
            (
                ("serialization_format", "serialization"),
                ("clock_domain", "clock"),
                ("frame_id", "frame"),
                ("message_count", "count"),
                ("missing_count", "missing"),
                ("duplicate_count", "duplicate"),
                ("out_of_order_count", "out_of_order"),
            ),
        )

    @field_validator("source_sha256")
    @classmethod
    def validate_digest(cls, value: str | None) -> str | None:
        return _validate_sha256(value, field_name="source_sha256")

    @model_validator(mode="after")
    def fill_stream_id(self) -> StreamInventory:
        if not self.stream_id:
            self.stream_id = self.topic or self.path or self.name or "stream-unknown"
        if (
            self.first_timestamp_ns is not None
            and self.last_timestamp_ns is not None
            and self.first_timestamp_ns > self.last_timestamp_ns
        ):
            raise ValueError(
                f"stream {self.stream_id!r} first_timestamp_ns must be <= last_timestamp_ns"
            )
        if (
            self.time_span_ns is None
            and self.first_timestamp_ns is not None
            and self.last_timestamp_ns is not None
        ):
            self.time_span_ns = self.last_timestamp_ns - self.first_timestamp_ns
        elif (
            self.time_span_ns is not None
            and self.first_timestamp_ns is not None
            and self.last_timestamp_ns is not None
            and self.time_span_ns != self.last_timestamp_ns - self.first_timestamp_ns
        ):
            raise ValueError(
                f"stream {self.stream_id!r} time_span_ns is inconsistent with timestamps"
            )
        if self.time_span_s is None and self.time_span_ns is not None:
            self.time_span_s = self.time_span_ns / 1_000_000_000.0
        elif self.time_span_s is not None and self.time_span_ns is not None:
            expected_span_s = self.time_span_ns / 1_000_000_000.0
            if not math.isclose(
                self.time_span_s,
                expected_span_s,
                rel_tol=1e-6,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"stream {self.stream_id!r} time_span_s is inconsistent with time_span_ns"
                )
        if self.time_span_s is not None and not math.isfinite(self.time_span_s):
            raise ValueError(f"stream {self.stream_id!r} time_span_s must be finite")
        if (
            self.rate_hz is None
            and self.message_count is not None
            and self.time_span_s is not None
            and self.time_span_s > 0
        ):
            self.rate_hz = self.message_count / self.time_span_s
        elif (
            self.rate_hz is not None
            and self.message_count is not None
            and self.time_span_s is not None
            and self.time_span_s > 0
        ):
            expected_rate_hz = self.message_count / self.time_span_s
            if not math.isclose(
                self.rate_hz,
                expected_rate_hz,
                rel_tol=0.02,
                abs_tol=max(1e-9, expected_rate_hz * 0.02),
            ):
                raise ValueError(
                    f"stream {self.stream_id!r} rate_hz is inconsistent with count/span"
                )
        elif self.rate_hz is not None and self.time_span_s == 0 and self.message_count is not None:
            raise ValueError(
                f"stream {self.stream_id!r} rate_hz cannot be stated for a zero time span"
            )
        if self.rate_hz is not None and not math.isfinite(self.rate_hz):
            raise ValueError(f"stream {self.stream_id!r} rate_hz must be finite")
        for field_name in (
            "radar_range_min_m",
            "radar_range_max_m",
            "radar_azimuth_span_rad",
            "radar_elevation_span_rad",
            "radar_doppler_span_mps",
        ):
            value = getattr(self, field_name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"stream {self.stream_id!r} {field_name} must be finite")
        if (
            self.radar_range_min_m is not None
            and self.radar_range_max_m is not None
            and self.radar_range_min_m > self.radar_range_max_m
        ):
            raise ValueError(
                f"stream {self.stream_id!r} radar range minimum exceeds maximum"
            )
        return self

    @property
    def serialization(self) -> str | None:
        """Compatibility spelling for serialization format."""

        return self.serialization_format

    @property
    def clock(self) -> str | None:
        """Compatibility spelling for clock domain."""

        return self.clock_domain


class ClockBinding(StrictModel):
    """Evidence that two streams share or are related by a clock."""

    binding_id: str = ""
    source_stream: str
    target_stream: str
    source_clock_domain: str | None = None
    target_clock_domain: str | None = None
    offset_ns: float | None = None
    relation: Literal["same_clock", "offset", "unknown"] = "unknown"
    status: EvidenceStatus = "unknown"
    evidence: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class FrameBinding(StrictModel):
    """Evidence that a stream is expressed in a named frame relationship."""

    binding_id: str = ""
    source_stream: str
    target_stream: str
    source_frame: str | None = None
    target_frame: str | None = None
    relation: Literal["same_frame", "transform", "unknown"] = "unknown"
    status: EvidenceStatus = "unknown"
    evidence: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class CrossStreamBinding(StrictModel):
    """Generic JSON-safe clock/frame binding for adapter authors."""

    binding_id: str = ""
    kind: Literal["clock", "frame"]
    source_stream: str
    target_stream: str
    source: str | None = None
    target: str | None = None
    relation: str = "unknown"
    status: EvidenceStatus = "unknown"
    offset_ns: float | None = None
    evidence: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class CaptureCheck(StrictModel):
    """One deterministic readiness check and its reason."""

    check_id: str
    status: CheckStatus
    subject: str
    reason: str
    observed: dict[str, Any] = Field(default_factory=dict)
    action: str | None = None


class CaptureAction(StrictModel):
    """Actionable remediation or evidence-collection step."""

    action_id: str
    priority: Literal["required", "recommended", "informational"] = "required"
    message: str
    subject: str | None = None
    command: list[str] | None = None


class CaptureManifestProvenance(StrictModel):
    """Generation and input identity for a capture manifest."""

    command: list[str] = Field(default_factory=list)
    tool_name: str = Field(
        default="calibrex",
        validation_alias=AliasChoices("tool_name", "tool"),
        serialization_alias="tool_name",
    )
    tool_version: str = Field(
        default="unknown",
        validation_alias=AliasChoices("tool_version", "version"),
        serialization_alias="tool_version",
    )
    git_commit: str | None = Field(
        default=None,
        validation_alias=AliasChoices("git_commit", "git_revision"),
        serialization_alias="git_commit",
    )
    config_path: str | None = None
    config_sha256: str | None = Field(
        default=None,
        pattern=_SHA256_PATTERN,
        validation_alias=AliasChoices("config_sha256", "config_digest"),
        serialization_alias="config_sha256",
    )
    source_sha256: dict[str, Sha256Digest] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("source_sha256", "source_digests"),
        serialization_alias="source_sha256",
    )
    adapter_versions: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def reject_alias_collisions(cls, value: Any) -> Any:
        return _reject_ambiguous_aliases(
            value,
            (
                ("tool_name", "tool"),
                ("tool_version", "version"),
                ("git_commit", "git_revision"),
                ("config_sha256", "config_digest"),
                ("source_sha256", "source_digests"),
            ),
        )

    @field_validator("config_sha256")
    @classmethod
    def validate_config_digest(cls, value: str | None) -> str | None:
        return _validate_sha256(value, field_name="config_sha256")

    @field_validator("source_sha256")
    @classmethod
    def validate_source_digests(cls, value: dict[str, Sha256Digest]) -> dict[str, Sha256Digest]:
        invalid: list[str] = []
        for name, digest in value.items():
            try:
                _validate_sha256(digest, field_name=f"source_sha256[{name!r}]")
            except ValueError:
                invalid.append(name)
        if invalid:
            raise ValueError(
                "source_sha256 contains invalid lowercase SHA-256 values: "
                + repr(invalid)
            )
        if any(not isinstance(name, str) or not name.strip() for name in value):
            raise ValueError("source_sha256 keys must be non-empty source IDs or paths")
        return value

    @property
    def source_digests(self) -> dict[str, Sha256Digest]:
        """Compatibility spelling for source digest mapping."""

        return self.source_sha256

    @property
    def tool(self) -> str:
        """Compatibility spelling for tool name."""

        return self.tool_name

    @property
    def version(self) -> str:
        """Compatibility spelling for tool version."""

        return self.tool_version


CAPTURE_MANIFEST_VERIFICATION_SCHEMA_VERSION: Literal[
    "slac.capture_manifest.verification/v0.1"
] = "slac.capture_manifest.verification/v0.1"
InputVerificationStatus = Literal["verified", "missing", "mismatch", "unknown"]


class CaptureInputVerification(StrictModel):
    """Recomputed digest evidence for one manifest input path."""

    role: Literal["source", "config", "metadata"]
    identifier: str
    path: str
    expected_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    observed_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    status: InputVerificationStatus
    reason: str

    @field_validator("expected_sha256", "observed_sha256")
    @classmethod
    def validate_digests(cls, value: str | None, info: Any) -> str | None:
        return _validate_sha256(value, field_name=str(info.field_name))


class CaptureManifestVerification(StrictModel):
    """Typed production verification report for a capture manifest.

    ``load_capture_manifest`` intentionally performs only schema and
    self-digest verification so a manifest can be moved with its source.  This
    report is the explicit, fail-closed operation that revalidates every
    source/config path still available from the manifest's directory.
    """

    schema_version: Literal[
        "slac.capture_manifest.verification/v0.1"
    ] = CAPTURE_MANIFEST_VERIFICATION_SCHEMA_VERSION
    manifest_path: str
    self_digest_status: InputVerificationStatus
    expected_artifact_sha256: Sha256Digest
    observed_artifact_sha256: Sha256Digest
    inputs: list[CaptureInputVerification] = Field(default_factory=list)
    valid: bool = False
    summary: str = ""

    @property
    def source_inputs(self) -> list[CaptureInputVerification]:
        """Return source and metadata checks in declaration order."""

        return [item for item in self.inputs if item.role in {"source", "metadata"}]

    @property
    def config_input(self) -> CaptureInputVerification | None:
        """Return the optional config check, if the manifest declares one."""

        return next((item for item in self.inputs if item.role == "config"), None)


class CaptureManifest(StrictModel):
    """Schema-validating capture/session inventory and trust decision."""

    schema_version: Literal["slac.capture_manifest/v0.1"] = CAPTURE_MANIFEST_SCHEMA_VERSION
    capture_id: str | None = None
    session_id: str | None = None
    vehicle_id: str | None = None
    sensor_kit_id: str | None = None
    sources: list[SourceInventory] = Field(
        default_factory=list,
        validation_alias=AliasChoices("sources", "source_inventory"),
        serialization_alias="sources",
    )
    sensors: list[SensorIdentity] = Field(default_factory=list)
    streams: list[StreamInventory] = Field(default_factory=list)
    readiness_profile: Literal["strict", "declared"] = "strict"
    required_streams: list[str] = Field(default_factory=list)
    optional_streams: list[str] = Field(default_factory=list)
    required_stream_kinds: list[StreamKind] = Field(default_factory=list)
    clock_bindings: list[ClockBinding] = Field(default_factory=list)
    frame_bindings: list[FrameBinding] = Field(default_factory=list)
    bindings: list[CrossStreamBinding] = Field(default_factory=list)
    checks: list[CaptureCheck] = Field(default_factory=list)
    actions: list[CaptureAction] = Field(default_factory=list)
    status: CaptureStatus = "blocked"
    summary: str = ""
    provenance: CaptureManifestProvenance = Field(default_factory=CaptureManifestProvenance)
    artifact_sha256: str = Field(
        pattern=_SHA256_PATTERN,
        validation_alias=AliasChoices("artifact_sha256", "self_sha256"),
        serialization_alias="artifact_sha256",
    )

    @model_validator(mode="before")
    @classmethod
    def reject_alias_collisions(cls, value: Any) -> Any:
        return _reject_ambiguous_aliases(
            value,
            (("sources", "source_inventory"), ("artifact_sha256", "self_sha256")),
        )

    @field_validator("artifact_sha256")
    @classmethod
    def validate_artifact_digest(cls, value: str) -> str:
        validated = _validate_sha256(value, field_name="artifact_sha256")
        assert validated is not None
        return validated

    @model_validator(mode="after")
    def validate_manifest_references(self) -> CaptureManifest:
        """Reject duplicate IDs, dangling references, and ambiguous policy lists."""

        _require_unique_ids("source", [item.source_id for item in self.sources])
        _require_unique_ids("sensor", [item.sensor_id for item in self.sensors])
        _require_unique_ids("stream", [item.stream_id for item in self.streams])
        _require_unique_values("required_streams", self.required_streams)
        _require_unique_values("optional_streams", self.optional_streams)
        overlap = sorted(set(self.required_streams) & set(self.optional_streams))
        if overlap:
            raise ValueError(
                "required_streams and optional_streams overlap: " + ", ".join(overlap)
            )
        sensor_ids = {item.sensor_id for item in self.sensors}
        unknown_sensor_refs = sorted(
            {stream.sensor_id for stream in self.streams if stream.sensor_id is not None}
            - sensor_ids
        )
        if unknown_sensor_refs:
            raise ValueError(
                "stream.sensor_id references undeclared sensor IDs: "
                + ", ".join(unknown_sensor_refs)
            )
        stream_ids = {item.stream_id for item in self.streams}
        declared_policy_ids = set(self.required_streams) | set(self.optional_streams)
        unknown_policy_ids = sorted(declared_policy_ids - stream_ids)
        if unknown_policy_ids:
            raise ValueError(
                "readiness policy references undeclared stream IDs: "
                + ", ".join(unknown_policy_ids)
            )
        for binding_group_name, bindings in (
            ("clock_bindings", self.clock_bindings),
            ("frame_bindings", self.frame_bindings),
            ("bindings", self.bindings),
        ):
            for binding in bindings:
                missing = sorted(
                    {binding.source_stream, binding.target_stream} - stream_ids
                )
                if missing:
                    raise ValueError(
                        f"{binding_group_name} binding {binding.binding_id!r} "
                        "references undeclared stream IDs: "
                        + ", ".join(missing)
                    )
        if self.readiness_profile == "declared" and not (
            self.required_streams or self.required_stream_kinds
        ):
            raise ValueError(
                "declared readiness_profile requires required_streams or required_stream_kinds"
            )
        if self.readiness_profile == "strict" and (
            self.required_streams or self.optional_streams or self.required_stream_kinds
        ):
            raise ValueError(
                "strict readiness_profile cannot carry declared stream policy; "
                "use readiness_profile='declared'"
            )
        return self

    @model_validator(mode="before")
    @classmethod
    def normalize_single_source(cls, value: Any) -> Any:
        if isinstance(value, Mapping) and "source" in value and "sources" not in value:
            normalized = dict(value)
            normalized["sources"] = [normalized.pop("source")]
            return normalized
        return value

    @property
    def source(self) -> SourceInventory | None:
        """Return the first source for single-source callers."""

        return self.sources[0] if self.sources else None

    @property
    def source_inventory(self) -> list[SourceInventory]:
        """Compatibility spelling for the source inventory list."""

        return self.sources

    @property
    def self_sha256(self) -> str:
        """Compatibility spelling for the canonical artifact digest."""

        return self.artifact_sha256

    def with_artifact_digest(self) -> CaptureManifest:
        """Return a copy with the canonical digest over all other fields."""

        payload = self.model_dump(mode="json", exclude_none=False)
        payload.pop("artifact_sha256", None)
        digest = _canonical_sha256(payload)
        return self.model_copy(update={"artifact_sha256": digest})

    def verify_artifact_digest(self) -> None:
        """Raise when the serialized artifact content no longer matches itself."""

        expected = self.with_artifact_digest().artifact_sha256
        if self.artifact_sha256 != expected:
            raise ValueError(
                "capture manifest self-digest mismatch: "
                f"declared={self.artifact_sha256}, expected={expected}"
            )

    def save(self, path: str | Path) -> None:
        """Write a digest-complete YAML or JSON capture manifest."""

        write_mapping(
            Path(path),
            self.with_artifact_digest().model_dump(mode="json", exclude_none=False),
        )


# Natural names used by downstream integrations during the v0.1 migration.
CaptureSessionManifest = CaptureManifest
CaptureManifestArtifact = CaptureManifest
CaptureSourceInventory = SourceInventory
CaptureStreamInventory = StreamInventory
SensorKitIdentity = SensorIdentity


def capture_manifest_json_schema() -> dict[str, Any]:
    """Return the generated JSON Schema for capture manifests."""

    return CaptureManifest.model_json_schema()


def capture_session_manifest_json_schema() -> dict[str, Any]:
    """Compatibility alias for :func:`capture_manifest_json_schema`."""

    return capture_manifest_json_schema()


def capture_manifest_verification_json_schema() -> dict[str, Any]:
    """Return the generated JSON Schema for input verification reports."""

    return CaptureManifestVerification.model_json_schema()


def load_capture_manifest(
    path: str | Path,
    *,
    verify: bool = True,
    verify_inputs: bool = False,
) -> CaptureManifest:
    """Load a capture manifest.

    ``verify`` checks the portable artifact self-digest.  ``verify_inputs`` is
    intentionally opt-in because relative source/config paths may not travel
    with the manifest; use :func:`verify_capture_manifest_inputs` for a typed,
    fail-closed production report.
    """

    manifest_path = Path(path)
    manifest = CaptureManifest.model_validate(read_mapping(manifest_path))
    if verify:
        manifest.verify_artifact_digest()
    if verify_inputs:
        verification = verify_capture_manifest_inputs(manifest_path)
        if not verification.valid:
            raise ValueError(verification.summary)
    return manifest


def verify_capture_manifest(path: str | Path) -> CaptureManifest:
    """Load and verify a capture manifest, returning the typed artifact."""

    return load_capture_manifest(path, verify=True)


def verify_capture_manifest_inputs(
    path: str | Path,
) -> CaptureManifestVerification:
    """Recompute all declared source/config paths and fail closed on drift.

    Paths are resolved relative to the manifest file, which makes the result
    deterministic when a capture bundle is moved as a directory.  A missing
    path, absent expected digest, or digest mismatch is represented as a typed
    finding and makes ``valid`` false.  Self-digest mismatch is also reported
    instead of being hidden behind an exception so ``capture verify`` can emit
    machine-readable evidence.
    """

    manifest_path = Path(path)
    manifest = CaptureManifest.model_validate(read_mapping(manifest_path))
    expected_self = manifest.with_artifact_digest().artifact_sha256
    self_status: InputVerificationStatus = (
        "verified" if manifest.artifact_sha256 == expected_self else "mismatch"
    )
    inputs: list[CaptureInputVerification] = []
    for source in manifest.sources:
        expected = source.sha256 or manifest.provenance.source_sha256.get(source.source_id)
        if expected is None:
            expected = manifest.provenance.source_sha256.get(source.path)
        inputs.append(
            _verify_input_path(
                role="source",
                identifier=source.source_id,
                path=source.path,
                expected_sha256=expected,
                base_dir=manifest_path.parent,
            )
        )
        if source.metadata_path is not None:
            inputs.append(
                _verify_input_path(
                    role="metadata",
                    identifier=source.source_id,
                    path=source.metadata_path,
                    expected_sha256=source.metadata_sha256,
                    base_dir=manifest_path.parent,
                )
            )
    if manifest.provenance.config_path is not None:
        inputs.append(
            _verify_input_path(
                role="config",
                identifier="config",
                path=manifest.provenance.config_path,
                expected_sha256=manifest.provenance.config_sha256,
                base_dir=manifest_path.parent,
            )
        )
    failed_inputs = [item for item in inputs if item.status != "verified"]
    valid = self_status == "verified" and not failed_inputs
    if valid:
        summary = "capture manifest self-digest and all declared inputs verified"
    else:
        failures = [
            f"{item.role}:{item.identifier}:{item.status}" for item in failed_inputs
        ]
        if self_status != "verified":
            failures.insert(0, f"artifact:self:{self_status}")
        summary = "capture manifest verification failed: " + ", ".join(failures)
    return CaptureManifestVerification(
        manifest_path=manifest_path.as_posix(),
        self_digest_status=self_status,
        expected_artifact_sha256=expected_self,
        observed_artifact_sha256=manifest.artifact_sha256,
        inputs=inputs,
        valid=valid,
        summary=summary,
    )


def _verify_input_path(
    *,
    role: Literal["source", "config", "metadata"],
    identifier: str,
    path: str,
    expected_sha256: str | None,
    base_dir: Path,
) -> CaptureInputVerification:
    resolved_path = _resolve_manifest_path(base_dir, path)
    if not path or not resolved_path.exists():
        return CaptureInputVerification(
            role=role,
            identifier=identifier,
            path=path,
            expected_sha256=expected_sha256,
            observed_sha256=None,
            status="missing",
            reason=f"declared {role} path is missing: {resolved_path}",
        )
    observed = sha256_path(resolved_path)
    if expected_sha256 is None:
        return CaptureInputVerification(
            role=role,
            identifier=identifier,
            path=path,
            expected_sha256=None,
            observed_sha256=observed,
            status="unknown",
            reason=f"{role} exists but no expected SHA-256 is recorded",
        )
    if observed != expected_sha256:
        return CaptureInputVerification(
            role=role,
            identifier=identifier,
            path=path,
            expected_sha256=expected_sha256,
            observed_sha256=observed,
            status="mismatch",
            reason=(
                f"{role} SHA-256 mismatch: expected {expected_sha256}, observed {observed}"
            ),
        )
    return CaptureInputVerification(
        role=role,
        identifier=identifier,
        path=path,
        expected_sha256=expected_sha256,
        observed_sha256=observed,
        status="verified",
        reason=f"{role} SHA-256 matches recorded digest",
    )


def _resolve_manifest_path(base_dir: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base_dir / candidate


def inspect_capture(
    path: str | Path,
    *,
    source_format: str | None = None,
    capture_id: str | None = None,
    session_id: str | None = None,
    vehicle_id: str | None = None,
    sensor_kit_id: str | None = None,
    sensors: Sequence[SensorIdentity | Mapping[str, Any]] = (),
    config_path: str | Path | None = None,
    config_sha256: str | None = None,
    command: Sequence[str] | None = None,
    tool_name: str = "calibrex",
    tool_version: str | None = None,
    readiness_profile: Literal["strict", "declared"] = "strict",
    required_streams: Sequence[str] = (),
    optional_streams: Sequence[str] = (),
    required_stream_kinds: Sequence[StreamKind] = (),
    sample_limit: int = 4,
) -> CaptureManifest:
    """Inspect a supported bag or plain-file source into a trust artifact.

    This function is the programmatic entry point used by ``capture inspect``.
    It never turns a decode failure into an empty successful stream; the
    resulting stream remains ``unknown`` or ``unsupported`` and the top-level
    status is downgraded accordingly.
    """

    source_path = Path(path)
    fmt = _normalize_source_format(source_format, source_path)
    source = _build_source_inventory(source_path, fmt)
    streams: list[StreamInventory]
    adapter_versions: dict[str, str] = {}
    inspection_notes: list[str] = []
    try:
        if fmt == "rosbag1":
            streams, notes = _inventory_rosbag1(source_path, sample_limit=sample_limit)
        elif fmt == "rosbag2":
            streams, notes = _inventory_rosbag2(source_path, sample_limit=sample_limit)
        elif fmt == "mcap":
            streams, notes = _inventory_rosbag2(
                source_path,
                sample_limit=sample_limit,
                force_format="mcap",
            )
        else:
            streams, notes = _inventory_files(source_path)
        inspection_notes.extend(notes)
    except Exception as exc:  # adapters must not hide a blocked source
        streams = []
        inspection_notes.append(f"source inspection failed: {type(exc).__name__}: {exc}")

    normalized_sensors = _normalize_sensors(sensors)
    if not normalized_sensors:
        normalized_sensors = _infer_sensor_identities(streams)
    for stream in streams:
        if stream.sensor_id is None and len(normalized_sensors) == 1:
            stream.sensor_id = normalized_sensors[0].sensor_id or None

    clock_bindings, frame_bindings, generic_bindings = _infer_bindings(streams)
    config_digest = config_sha256
    config_name: str | None = str(config_path) if config_path is not None else None
    if config_path is not None and config_digest is None:
        config_digest = sha256_path(Path(config_path))

    source_digests: dict[str, str] = {}
    if source.sha256 is not None:
        source_digests[source.source_id] = source.sha256
    provenance = CaptureManifestProvenance(
        command=list(command)
        if command is not None
        else ["calibrex", "capture", "inspect", str(path)],
        tool_name=tool_name,
        tool_version=tool_version or _calibrex_version(),
        git_commit=git_commit(),
        config_path=config_name,
        config_sha256=config_digest,
        source_sha256=source_digests,
        adapter_versions=adapter_versions,
        notes=inspection_notes,
    )
    checks, actions, status, summary = _evaluate_capture(
        source=source,
        streams=streams,
        sensors=normalized_sensors,
        capture_id=capture_id,
        session_id=session_id,
        vehicle_id=vehicle_id,
        sensor_kit_id=sensor_kit_id,
        clock_bindings=clock_bindings,
        frame_bindings=frame_bindings,
        readiness_profile=readiness_profile,
        required_streams=required_streams,
        optional_streams=optional_streams,
        required_stream_kinds=required_stream_kinds,
    )
    artifact = CaptureManifest(
        capture_id=capture_id,
        session_id=session_id,
        vehicle_id=vehicle_id,
        sensor_kit_id=sensor_kit_id,
        sources=[source],
        sensors=normalized_sensors,
        streams=sorted(streams, key=lambda item: item.stream_id),
        readiness_profile=readiness_profile,
        required_streams=sorted(set(required_streams)),
        optional_streams=sorted(set(optional_streams)),
        required_stream_kinds=sorted(set(required_stream_kinds)),
        clock_bindings=sorted(clock_bindings, key=lambda item: item.binding_id),
        frame_bindings=sorted(frame_bindings, key=lambda item: item.binding_id),
        bindings=sorted(generic_bindings, key=lambda item: item.binding_id),
        checks=checks,
        actions=actions,
        status=status,
        summary=summary,
        provenance=provenance,
        artifact_sha256="0" * 64,
    )
    return artifact.with_artifact_digest()


def build_capture_manifest(*args: Any, **kwargs: Any) -> CaptureManifest:
    """Compatibility alias for :func:`inspect_capture`."""

    return inspect_capture(*args, **kwargs)


def inspect_capture_manifest(*args: Any, **kwargs: Any) -> CaptureManifest:
    """Compatibility alias for :func:`inspect_capture`."""

    return inspect_capture(*args, **kwargs)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _calibrex_version() -> str:
    try:
        from calibrex import __version__

        return str(__version__)
    except Exception:  # pragma: no cover - defensive package bootstrap path
        return "unknown"


def _normalize_source_format(source_format: str | None, path: Path) -> SourceFormat:
    if source_format is not None and source_format not in {"", "auto"}:
        normalized = source_format.replace("-", "_").lower()
        if normalized in {"rosbag1", "rosbag2", "mcap", "files", "filesystem"}:
            return "files" if normalized == "filesystem" else cast(SourceFormat, normalized)
        raise ValueError(f"unsupported capture source format: {source_format}")
    suffix = path.suffix.lower()
    if suffix == ".bag":
        return "rosbag1"
    if suffix == ".mcap":
        return "mcap"
    if suffix == ".db3":
        return "rosbag2"
    if path.is_dir() and (
        (path / "metadata.yaml").is_file()
        or any(path.glob("*.db3"))
        or any(path.glob("*.mcap"))
    ):
        return "rosbag2"
    return "files"


def _path_size(path: Path) -> tuple[int | None, int | None]:
    if not path.exists():
        return None, None
    if path.is_file():
        return path.stat().st_size, 1
    total = 0
    count = 0
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        total += child.stat().st_size
        count += 1
    return total, count


def _build_source_inventory(path: Path, source_format: SourceFormat) -> SourceInventory:
    size_bytes, file_count = _path_size(path)
    digest = sha256_path(path)
    metadata_path: Path | None = None
    storage_identifier: str | None = None
    compression_format: str | None = None
    compression_mode: str | None = None
    mcap_integrity: McapIntegrityEvidence | None = None
    notes: list[str] = []
    if source_format == "rosbag2" and path.is_dir() and (path / "metadata.yaml").is_file():
        metadata_path = path / "metadata.yaml"
    if source_format == "rosbag2":
        notes.extend(
            [
                "portable adapter does not independently verify storage index integrity",
                (
                    "portable adapter does not verify per-record/container CRC unless an "
                    "adapter exposes it"
                ),
            ]
        )
        try:
            from calibrex.data.rosbag2 import _load_metadata, resolve_storage

            storage_path, storage_identifier = resolve_storage(path)
            if path.is_dir() and metadata_path is not None:
                metadata = _load_metadata(metadata_path)
                compression_format = metadata.get("compression_format") or None
                compression_mode = metadata.get("compression_mode") or None
            if storage_path != path and path.is_file():
                storage_identifier = storage_identifier or path.suffix.lstrip(".")
        except Exception as exc:
            notes.append(f"storage metadata unavailable: {type(exc).__name__}: {exc}")
    elif source_format == "rosbag1":
        storage_identifier = "rosbag1"
    elif source_format == "mcap":
        storage_identifier = "mcap"
        try:
            from calibrex.data.mcap import inspect_mcap_integrity

            mcap_integrity = inspect_mcap_integrity(path)
            notes.extend(
                f"MCAP integrity: {message}"
                for message in [*mcap_integrity.errors, *mcap_integrity.warnings]
            )
            if mcap_integrity.unsupported_compressions:
                notes.append(
                    "MCAP unsupported chunk compression: "
                    + ", ".join(mcap_integrity.unsupported_compressions)
                )
        except Exception as exc:
            # Integrity evidence is part of the trust boundary.  Preserve an
            # explicit blocked evidence object rather than silently leaving
            # old optimistic ``unknown`` fields in place.
            mcap_integrity = McapIntegrityEvidence(
                status="blocked",
                errors=[f"MCAP integrity inspection failed: {type(exc).__name__}: {exc}"],
            )
            notes.extend(mcap_integrity.errors)
    return SourceInventory(
        path=str(path),
        format=source_format,
        exists=path.exists(),
        sha256=digest,
        size_bytes=size_bytes,
        file_count=file_count,
        storage_identifier=storage_identifier,
        storage_status="known" if storage_identifier is not None else "unknown",
        index_status=(
            "known"
            if mcap_integrity is not None and mcap_integrity.index_status == "known"
            else "not_applicable"
            if source_format == "files"
            else "unknown"
        ),
        crc_status=(
            "known"
            if mcap_integrity is not None and mcap_integrity.crc_status == "known"
            else "not_applicable"
            if source_format == "files"
            else "unknown"
        ),
        metadata_path=str(metadata_path) if metadata_path is not None else None,
        metadata_sha256=sha256_path(metadata_path) if metadata_path is not None else None,
        compression_format=compression_format,
        compression_mode=compression_mode,
        mcap_integrity=mcap_integrity,
        notes=notes,
    )


def _stream_kind(message_type: str | None, *, path: str | None = None) -> StreamKind:
    token = (message_type or "").lower()
    path_token = (path or "").lower()
    if (
        "camera/info" in token
        or "camera_info" in token
        or "camerainfo" in token
    ):
        return "camera_info"
    if "image" in token or path_token.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        return "image"
    if "pointcloud" in token or "point_cloud" in token or path_token.endswith(
        (".pcd", ".ply", ".las", ".laz")
    ):
        return "pointcloud"
    if token.endswith("/imu") or token.endswith(".imu") or "imu" in token:
        return "imu"
    if "radar" in token or "radar" in path_token:
        return "radar"
    if "odometry" in token or "pose" in token or "odom" in token:
        return "odometry"
    if path_token.endswith((".yaml", ".yml", ".json", ".csv", ".txt")):
        return "metadata"
    return "other"


def _capability(message_type: str | None) -> tuple[DecodeStatus, list[str]]:
    token = message_type or ""
    supported = {
        "sensor_msgs/PointCloud2",
        "sensor_msgs/msg/PointCloud2",
        "livox_ros_driver/CustomMsg",
        "livox_interfaces/msg/CustomMsg",
        "livox_ros_driver/msg/CustomMsg",
        "livox_ros_driver2/msg/CustomMsg",
        "geometry_msgs/PoseStamped",
        "nav_msgs/msg/Odometry",
        "sensor_msgs/msg/Imu",
        "sensor_msgs/Image",
        "sensor_msgs/msg/Image",
        "sensor_msgs/CameraInfo",
        "sensor_msgs/msg/CameraInfo",
        "radar_msgs/RadarScan",
        "radar_msgs/msg/RadarScan",
    }
    if token in supported:
        return "supported", [_stream_kind(token)]
    if token:
        return "unsupported", []
    return "unknown", []


def _parse_rosbag2_qos(raw: str | None) -> CaptureQoS | None:
    """Parse only explicitly named QoS values exposed by SQLite metadata.

    rosbag2 stores ``offered_qos_profiles`` as a serialized ROS QoS object.
    Numeric enum values are deliberately not translated without a ROS type
    package; retaining them as unknown avoids claiming a policy we cannot
    prove.  A partial result is still useful when fields such as depth are
    directly represented.
    """

    if not raw or not raw.strip():
        return None
    try:
        import yaml

        payload = yaml.safe_load(raw)
    except Exception:
        return CaptureQoS(status="unknown", source="rosbag2 offered_qos_profiles (unparsed)")
    if not isinstance(payload, Mapping):
        return CaptureQoS(status="unknown", source="rosbag2 offered_qos_profiles (unparsed)")
    values: dict[str, Any] = {}
    for field_name in (
        "reliability",
        "durability",
        "history",
        "liveliness",
    ):
        value = payload.get(field_name)
        if isinstance(value, str) and value.strip():
            values[field_name] = value.strip()
    depth = payload.get("depth")
    if isinstance(depth, int) and depth >= 0:
        values["depth"] = depth
    if not values:
        return CaptureQoS(status="unknown", source="rosbag2 offered_qos_profiles (unparsed)")
    return CaptureQoS(
        status="known",
        reliability=cast(str | None, values.get("reliability")),
        durability=cast(str | None, values.get("durability")),
        history=cast(str | None, values.get("history")),
        depth=cast(int | None, values.get("depth")),
        liveliness=cast(str | None, values.get("liveliness")),
        source="rosbag2 offered_qos_profiles",
    )


def _inventory_rosbag2(
    path: Path,
    *,
    sample_limit: int,
    force_format: str | None = None,
) -> tuple[list[StreamInventory], list[str]]:
    from calibrex.data.rosbag2 import decode_rosbag2_message

    states: dict[str, dict[str, Any]] = {}
    notes: list[str] = []
    try:
        iterator = _iter_rosbag2_inventory_records(path)
        for connection, timestamp_ns, data in iterator:
            state = states.setdefault(
                connection.topic,
                {
                    "message_type": connection.message_type,
                    "serialization": connection.serialization_format,
                    "count": 0,
                    "first": None,
                    "last": None,
                    "previous": None,
                    "duplicate": 0,
                    "out_of_order": 0,
                    "frame": None,
                    "frame_ids": set(),
                    "frame_conflict": False,
                    "sample_count": 0,
                    "radar_return_count": 0,
                    "radar_seen": False,
                    "radar_range_min_m": None,
                    "radar_range_max_m": None,
                    "radar_azimuth_span_rad": None,
                    "radar_elevation_span_rad": None,
                    "radar_doppler_span_mps": None,
                    "radar_duplicate_return_count": 0,
                    "radar_diversity_status": None,
                    "capability_status": _capability(connection.message_type)[0],
                    # An exact built-in type is only "supported" after a
                    # sampled payload validates successfully.  This prevents
                    # sample_limit=0 or an empty stream from becoming a false
                    # positive readiness claim.
                    "decode_status": (
                        "unknown"
                        if _capability(connection.message_type)[0] == "supported"
                        else _capability(connection.message_type)[0]
                    ),
                    "capabilities": _capability(connection.message_type)[1],
                    "calibration_status": "not_applicable",
                    "unknown": set(),
                    "notes": [],
                    "qos": _parse_rosbag2_qos(connection.offered_qos_profiles),
                },
            )
            state["count"] += 1
            if state["first"] is None or timestamp_ns < state["first"]:
                state["first"] = int(timestamp_ns)
            if state["last"] is None or timestamp_ns > state["last"]:
                state["last"] = int(timestamp_ns)
            previous = state["previous"]
            if previous is not None:
                if timestamp_ns == previous:
                    state["duplicate"] += 1
                elif timestamp_ns < previous:
                    state["out_of_order"] += 1
            state["previous"] = int(timestamp_ns)
            if (
                state["sample_count"] < max(0, sample_limit)
                and state["capability_status"] == "supported"
            ):
                state["sample_count"] += 1
                try:
                    message = decode_rosbag2_message(
                        connection.topic,
                        connection.message_type,
                        int(timestamp_ns),
                        bytes(data),
                    )
                    _record_decoded_message(state, message)
                    state["decode_status"] = "supported"
                except Exception as exc:
                    state["decode_status"] = "unknown"
                    state["unknown"].add("decode")
                    if _stream_kind(connection.message_type) == "camera_info":
                        state["calibration_status"] = "unknown"
                    if len(notes) < 20:
                        notes.append(
                            f"{connection.topic}: decode unavailable: "
                            f"{type(exc).__name__}: {exc}"
                        )
    except Exception as exc:
        notes.append(f"rosbag2 traversal failed: {type(exc).__name__}: {exc}")
    return _states_to_streams(states), notes


def _iter_rosbag2_inventory_records(
    path: Path,
) -> Iterator[tuple[Any, int, bytes]]:
    """Yield rosbag2 records in storage order for duplicate/order evidence.

    The calibration reader intentionally orders SQLite records by timestamp so
    replay is stable.  Inventory needs the original insertion order as well,
    otherwise an out-of-order recording could be silently reported as clean.
    MCAP records already retain their serialized order through the native
    adapter iterator.
    """

    from calibrex.data.rosbag2 import (
        Rosbag2Connection,
        _message_decompressor_from_path,
        iter_messages,
        resolve_storage,
    )

    storage_path, storage_identifier = resolve_storage(path)
    if storage_identifier != "sqlite3":
        yield from iter_messages(path)
        return
    import sqlite3

    decompressor = _message_decompressor_from_path(path)
    connection = sqlite3.connect(f"file:{storage_path}?mode=ro", uri=True)
    try:
        try:
            rows = connection.execute(
                "SELECT id, name, type, serialization_format, offered_qos_profiles FROM topics"
            ).fetchall()
            has_qos_profiles = True
        except sqlite3.OperationalError:
            rows = connection.execute(
                "SELECT id, name, type, serialization_format FROM topics"
            ).fetchall()
            has_qos_profiles = False
        topics: dict[int, Rosbag2Connection] = {}
        for row in rows:
            topic_id, name, message_type, serialization = row[:4]
            qos_profiles = row[4] if has_qos_profiles and len(row) > 4 else None
            topics[int(topic_id)] = Rosbag2Connection(
                topic_id=int(topic_id),
                topic=str(name),
                message_type=str(message_type),
                serialization_format=str(serialization),
                offered_qos_profiles=(str(qos_profiles) if qos_profiles else None),
            )
        try:
            messages = connection.execute(
                "SELECT topic_id, timestamp, data FROM messages ORDER BY id ASC"
            )
        except sqlite3.OperationalError:
            # A few hand-authored bags omit the id column; the existing reader
            # still provides deterministic timestamp-order inspection there.
            yield from iter_messages(path)
            return
        for topic_id, timestamp_ns, payload in messages:
            resolved = topics.get(int(topic_id))
            if resolved is None:
                continue
            data = bytes(payload)
            if decompressor is not None:
                data = decompressor(data)
            yield resolved, int(timestamp_ns), data
    finally:
        connection.close()


def _inventory_rosbag1(
    path: Path,
    *,
    sample_limit: int,
) -> tuple[list[StreamInventory], list[str]]:
    from calibrex.data.rosbag1 import decode_rosbag1_message, iter_messages

    states: dict[str, dict[str, Any]] = {}
    notes: list[str] = []
    try:
        for connection, timestamp_ns, data in iter_messages(path):
            state = states.setdefault(
                connection.topic,
                {
                    "message_type": connection.message_type,
                    "serialization": "rosmsg",
                    "count": 0,
                    "first": None,
                    "last": None,
                    "previous": None,
                    "duplicate": 0,
                    "out_of_order": 0,
                    "frame": None,
                    "frame_ids": set(),
                    "frame_conflict": False,
                    "sample_count": 0,
                    "radar_return_count": 0,
                    "radar_seen": False,
                    "radar_range_min_m": None,
                    "radar_range_max_m": None,
                    "radar_azimuth_span_rad": None,
                    "radar_elevation_span_rad": None,
                    "radar_doppler_span_mps": None,
                    "radar_duplicate_return_count": 0,
                    "radar_diversity_status": None,
                    "capability_status": _capability(connection.message_type)[0],
                    "decode_status": (
                        "unknown"
                        if _capability(connection.message_type)[0] == "supported"
                        else _capability(connection.message_type)[0]
                    ),
                    "capabilities": _capability(connection.message_type)[1],
                    "calibration_status": "not_applicable",
                    "unknown": set(),
                    "notes": [],
                },
            )
            state["count"] += 1
            if state["first"] is None or timestamp_ns < state["first"]:
                state["first"] = int(timestamp_ns)
            if state["last"] is None or timestamp_ns > state["last"]:
                state["last"] = int(timestamp_ns)
            previous = state["previous"]
            if previous is not None:
                if timestamp_ns == previous:
                    state["duplicate"] += 1
                elif timestamp_ns < previous:
                    state["out_of_order"] += 1
            state["previous"] = int(timestamp_ns)
            if (
                state["sample_count"] < max(0, sample_limit)
                and state["capability_status"] == "supported"
            ):
                state["sample_count"] += 1
                try:
                    message = decode_rosbag1_message(
                        connection.topic,
                        connection.message_type,
                        int(timestamp_ns),
                        bytes(data),
                        include_image_data=False,
                    )
                    _record_decoded_message(state, message)
                    state["decode_status"] = "supported"
                except Exception as exc:
                    state["decode_status"] = "unknown"
                    state["unknown"].add("decode")
                    if _stream_kind(connection.message_type) == "camera_info":
                        state["calibration_status"] = "unknown"
                    if len(notes) < 20:
                        notes.append(
                            f"{connection.topic}: decode unavailable: "
                            f"{type(exc).__name__}: {exc}"
                        )
    except Exception as exc:
        notes.append(f"rosbag1 traversal failed: {type(exc).__name__}: {exc}")
    return _states_to_streams(states), notes


def _record_decoded_message(state: dict[str, Any], message: Any) -> None:
    """Add bounded, schema-safe metadata from one decoded adapter message."""

    frame_id = getattr(message, "frame_id", None)
    if frame_id:
        normalized_frame = str(frame_id)
        state.setdefault("frame_ids", set()).add(normalized_frame)
        state["frame_conflict"] = len(state["frame_ids"]) > 1
        if state.get("frame") is None:
            state["frame"] = normalized_frame
    else:
        state["unknown"].add("frame_id")
    if hasattr(message, "diversity_summary") and hasattr(message, "return_count"):
        # Duck typing keeps this core module independent of ROS and allows
        # future radar adapters to expose the same evidence contract.
        evidence = message.diversity_summary()
        state["radar_seen"] = True
        provenance = getattr(message, "source_spec_provenance", {})
        if isinstance(provenance, Mapping):
            source_type = provenance.get(
                "type", getattr(message, "source_spec", "unknown")
            )
            source_note = (
                f"radar_msgs source spec {source_type} "
                f"commit {provenance.get('commit', 'unknown')} "
                f"license {provenance.get('license', 'unknown')}"
            )
            if source_note not in state["notes"]:
                state["notes"].append(source_note)
        state["radar_return_count"] = int(state.get("radar_return_count", 0)) + int(
            evidence.get("return_count", 0)
        )
        for field_name in (
            "radar_range_min_m",
            "radar_range_max_m",
            "radar_azimuth_span_rad",
            "radar_elevation_span_rad",
            "radar_doppler_span_mps",
        ):
            key = field_name.removeprefix("radar_")
            value = evidence.get(key)
            if value is None:
                continue
            current = state.get(field_name)
            if current is None:
                state[field_name] = float(value)
            elif field_name.endswith("_min_m"):
                state[field_name] = min(float(current), float(value))
            elif field_name.endswith("_max_m"):
                state[field_name] = max(float(current), float(value))
            else:
                state[field_name] = max(float(current), float(value))
        state["radar_duplicate_return_count"] = int(
            state.get("radar_duplicate_return_count", 0)
        ) + int(evidence.get("duplicate_return_count", 0))
        diversity = str(evidence.get("status", "unknown"))
        previous = state.get("radar_diversity_status")
        if (
            previous is None
            or diversity == "strong"
            or (previous != "strong" and diversity == "weak")
        ):
            state["radar_diversity_status"] = diversity
    calibration_status = getattr(message, "calibration_status", None)
    if calibration_status is None:
        return
    current = str(state.get("calibration_status", "not_applicable"))
    if current == "not_applicable":
        state["calibration_status"] = str(calibration_status)
    elif current != calibration_status:
        state["calibration_status"] = "mixed"


def _states_to_streams(states: Mapping[str, Mapping[str, Any]]) -> list[StreamInventory]:
    streams: list[StreamInventory] = []
    for topic in sorted(states):
        state = states[topic]
        first = cast(int | None, state["first"])
        last = cast(int | None, state["last"])
        span_ns = max(0, last - first) if first is not None and last is not None else None
        span_s = span_ns / 1_000_000_000.0 if span_ns is not None else None
        count = int(state["count"])
        streams.append(
            StreamInventory(
                stream_id=topic,
                name=topic,
                kind=_stream_kind(cast(str | None, state["message_type"])),
                topic=topic,
                message_type=state["message_type"],
                serialization_format=state["serialization"],
                clock_domain="record_time_ns",
                timestamp_source="bag_record_time",
                frame_id=state["frame"],
                qos=cast(CaptureQoS | None, state.get("qos")),
                message_count=count,
                first_timestamp_ns=first,
                last_timestamp_ns=last,
                time_span_ns=span_ns,
                time_span_s=span_s,
                rate_hz=count / span_s if span_s and span_s > 0 else None,
                missing_count=None,
                duplicate_count=int(state["duplicate"]),
                out_of_order_count=int(state["out_of_order"]),
                decode_status=state["decode_status"],
                calibration_status=cast(
                    CalibrationStatus,
                    state.get("calibration_status", "not_applicable"),
                ),
                radar_return_count=(
                    int(state.get("radar_return_count", 0))
                    if state.get("radar_seen")
                    else None
                ),
                radar_range_min_m=(
                    state.get("radar_range_min_m") if state.get("radar_seen") else None
                ),
                radar_range_max_m=(
                    state.get("radar_range_max_m") if state.get("radar_seen") else None
                ),
                radar_azimuth_span_rad=(
                    state.get("radar_azimuth_span_rad")
                    if state.get("radar_seen")
                    else None
                ),
                radar_elevation_span_rad=(
                    state.get("radar_elevation_span_rad")
                    if state.get("radar_seen")
                    else None
                ),
                radar_doppler_span_mps=(
                    state.get("radar_doppler_span_mps")
                    if state.get("radar_seen")
                    else None
                ),
                radar_duplicate_return_count=(
                    int(state.get("radar_duplicate_return_count", 0))
                    if state.get("radar_seen")
                    else None
                ),
                radar_diversity_status=cast(
                    Literal["strong", "weak", "empty", "unknown", "not_applicable"],
                    state.get("radar_diversity_status")
                    if state.get("radar_seen")
                    else "not_applicable",
                ),
                frame_conflict=bool(state.get("frame_conflict", False)),
                capabilities=list(state["capabilities"]),
                notes=list(state.get("notes", [])),
                unknown_fields=sorted(
                    set(state["unknown"])
                    | {"missing_count"}
                    | ({"qos"} if state.get("qos") is None else set())
                    | ({"frame_id"} if state["frame"] is None else set())
                    | ({"frame_id_conflict"} if state.get("frame_conflict") else set())
                ),
            )
        )
    return streams


def _inventory_files(path: Path) -> tuple[list[StreamInventory], list[str]]:
    if not path.exists():
        return [], ["source path does not exist"]
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    manifest = None
    if path.is_dir():
        try:
            from calibrex.data.manifest import find_manifest, load_manifest

            manifest_path = find_manifest(path)
            if manifest_path is not None:
                manifest = load_manifest(manifest_path)
        except Exception as exc:
            return _file_streams(files), [
                f"dataset manifest unavailable: {type(exc).__name__}: {exc}"
            ]
    if manifest is not None:
        streams: list[StreamInventory] = []
        for name, item in sorted(manifest.streams.items()):
            matches = sorted(path.glob(item.path)) if item.path else []
            count = item.count if item.count is not None else len(matches)
            stream_path = item.path
            stream_digest = sha256_path(matches[0]) if len(matches) == 1 else None
            stream = StreamInventory(
                stream_id=name,
                name=name,
                kind=cast(StreamKind, item.kind),
                topic=item.topic,
                path=stream_path,
                sensor_id=item.sensor,
                frame_id=item.frame_id,
                message_count=count,
                decode_status=(
                    "unsupported"
                    if item.kind in {"image", "radar"}
                    else "unknown"
                ),
                unknown_fields=["message_type", "clock_domain", "qos"],
                source_sha256=stream_digest,
            )
            streams.append(stream)
        return streams, []
    return _file_streams(files), []


def _file_streams(files: Sequence[Path]) -> list[StreamInventory]:
    streams: list[StreamInventory] = []
    for file_path in files:
        kind = _stream_kind(None, path=str(file_path))
        suffix = file_path.suffix.lower()
        # A file extension is not sufficient evidence of a decodable sensor
        # stream.  PCD/PLY get an unknown capability; camera/radar payloads are
        # explicitly unsupported until an adapter claims them.
        decode_status: DecodeStatus = "unsupported" if kind in {"image", "radar"} else "unknown"
        streams.append(
            StreamInventory(
                stream_id=file_path.as_posix(),
                name=file_path.name,
                kind=kind,
                path=file_path.as_posix(),
                message_count=1,
                decode_status=decode_status,
                unknown_fields=["message_type", "clock_domain", "frame_id", "qos"],
                source_sha256=sha256_path(file_path),
                notes=[f"suffix={suffix or '<none>'}"],
            )
        )
    return streams


def _normalize_sensors(
    sensors: Sequence[SensorIdentity | Mapping[str, Any]],
) -> list[SensorIdentity]:
    normalized: list[SensorIdentity] = []
    for sensor in sensors:
        normalized.append(
            sensor if isinstance(sensor, SensorIdentity) else SensorIdentity.model_validate(sensor)
        )
    return sorted(normalized, key=lambda item: item.sensor_id or item.type)


def _infer_sensor_identities(streams: Sequence[StreamInventory]) -> list[SensorIdentity]:
    inferred: list[SensorIdentity] = []
    for stream in streams:
        sensor_type: SensorType
        if stream.kind == "pointcloud":
            sensor_type = "lidar"
        elif stream.kind == "image" or stream.kind == "camera_info":
            sensor_type = "camera"
        elif stream.kind == "imu":
            sensor_type = "imu"
        elif stream.kind == "radar":
            sensor_type = "radar"
        elif stream.kind == "odometry":
            sensor_type = "odometry"
        else:
            sensor_type = "other"
        inferred.append(
            SensorIdentity(
                sensor_id=stream.stream_id,
                type=sensor_type,
                frame_id=stream.frame_id,
                identity_status="partial",
                notes=["inferred from stream; serial/model/firmware/mount are not source evidence"],
            )
        )
    unique: dict[str, SensorIdentity] = {
        item.sensor_id: item for item in inferred if item.sensor_id
    }
    return [unique[key] for key in sorted(unique)]


def _infer_bindings(
    streams: Sequence[StreamInventory],
) -> tuple[list[ClockBinding], list[FrameBinding], list[CrossStreamBinding]]:
    ordered = sorted(streams, key=lambda item: item.stream_id)
    clocks: list[ClockBinding] = []
    frames: list[FrameBinding] = []
    generic: list[CrossStreamBinding] = []
    for index, source in enumerate(ordered):
        for target in ordered[index + 1 :]:
            binding_base = f"{source.stream_id}->{target.stream_id}"
            if source.clock_domain is not None and target.clock_domain is not None:
                same = source.clock_domain == target.clock_domain
                relation: Literal["same_clock", "offset", "unknown"] = (
                    "same_clock" if same else "unknown"
                )
                status: EvidenceStatus = "known" if same else "unknown"
                clocks.append(
                    ClockBinding(
                        binding_id=f"clock:{binding_base}",
                        source_stream=source.stream_id,
                        target_stream=target.stream_id,
                        source_clock_domain=source.clock_domain,
                        target_clock_domain=target.clock_domain,
                        relation=relation,
                        status=status,
                        evidence=["stream clock_domain fields"],
                    )
                )
                generic.append(
                    CrossStreamBinding(
                        binding_id=f"clock:{binding_base}",
                        kind="clock",
                        source_stream=source.stream_id,
                        target_stream=target.stream_id,
                        source=source.clock_domain,
                        target=target.clock_domain,
                        relation=relation,
                        status=status,
                        evidence=["stream clock_domain fields"],
                    )
                )
            if source.frame_id is not None and target.frame_id is not None:
                same_frame = source.frame_id == target.frame_id
                relation_frame: Literal["same_frame", "transform", "unknown"] = (
                    "same_frame" if same_frame else "unknown"
                )
                status_frame: EvidenceStatus = "known" if same_frame else "unknown"
                frames.append(
                    FrameBinding(
                        binding_id=f"frame:{binding_base}",
                        source_stream=source.stream_id,
                        target_stream=target.stream_id,
                        source_frame=source.frame_id,
                        target_frame=target.frame_id,
                        relation=relation_frame,
                        status=status_frame,
                        evidence=["stream frame_id fields"],
                    )
                )
                generic.append(
                    CrossStreamBinding(
                        binding_id=f"frame:{binding_base}",
                        kind="frame",
                        source_stream=source.stream_id,
                        target_stream=target.stream_id,
                        source=source.frame_id,
                        target=target.frame_id,
                        relation=relation_frame,
                        status=status_frame,
                        evidence=["stream frame_id fields"],
                    )
                )
    return clocks, frames, generic


def _evaluate_capture(
    *,
    source: SourceInventory,
    streams: Sequence[StreamInventory],
    sensors: Sequence[SensorIdentity],
    capture_id: str | None,
    session_id: str | None,
    vehicle_id: str | None,
    sensor_kit_id: str | None,
    clock_bindings: Sequence[ClockBinding],
    frame_bindings: Sequence[FrameBinding],
    readiness_profile: Literal["strict", "declared"],
    required_streams: Sequence[str],
    optional_streams: Sequence[str],
    required_stream_kinds: Sequence[StreamKind],
) -> tuple[list[CaptureCheck], list[CaptureAction], CaptureStatus, str]:
    checks: list[CaptureCheck] = []
    actions: list[CaptureAction] = []

    def add(
        check_id: str,
        status: CheckStatus,
        subject: str,
        reason: str,
        *,
        observed: dict[str, Any] | None = None,
        action: str | None = None,
        priority: Literal["required", "recommended", "informational"] = "required",
    ) -> None:
        checks.append(
            CaptureCheck(
                check_id=check_id,
                status=status,
                subject=subject,
                reason=reason,
                observed=observed or {},
                action=action,
            )
        )
        if status != "pass" and action:
            actions.append(
                CaptureAction(
                    action_id=check_id,
                    priority=priority,
                    message=action,
                    subject=subject,
                )
            )

    add(
        "source.exists",
        "pass" if source.exists else "blocked",
        source.path or "source",
        "source exists" if source.exists else "source path does not exist",
        observed={"exists": source.exists},
        action="Provide the original capture file or directory.",
    )
    add(
        "source.sha256",
        "pass" if source.sha256 is not None else "warn",
        source.path or "source",
        "source SHA-256 is recorded" if source.sha256 else "source SHA-256 is unknown",
        observed={"sha256": source.sha256},
        action="Hash the source before calibration and preserve the digest.",
        priority="recommended",
    )
    identity_ids = {
        "capture_id": capture_id,
        "session_id": session_id,
        "vehicle_id": vehicle_id,
        "sensor_kit_id": sensor_kit_id,
    }
    missing_ids = [name for name, value in identity_ids.items() if not value]
    add(
        "identity.ids",
        "pass" if not missing_ids else "blocked",
        "capture identity",
        "capture/session/vehicle/sensor-kit IDs are complete"
        if not missing_ids
        else "missing identity: " + ", ".join(missing_ids),
        observed={"missing": missing_ids},
        action="Provide stable capture, session, vehicle, and sensor-kit IDs.",
    )
    incomplete_sensors = [
        sensor.sensor_id or "unknown"
        for sensor in sensors
        if sensor.identity_status != "complete"
    ]
    add(
        "identity.sensors",
        "pass" if sensors and not incomplete_sensors else ("blocked" if not sensors else "warn"),
        "sensor identity",
        "all sensor identities include type/serial/model/firmware/mount/frame"
        if sensors and not incomplete_sensors
        else (
            "no sensor identities were supplied"
            if not sensors
            else "sensor identity is partial or inferred: " + ", ".join(incomplete_sensors)
        ),
        observed={"sensor_count": len(sensors), "incomplete": incomplete_sensors},
        action="Record each physical sensor serial, model, firmware, mount, and frame.",
    )
    stream_by_id = {stream.stream_id: stream for stream in streams}
    stream_ids = set(stream_by_id)
    requested_required = set(required_streams)
    requested_optional = set(optional_streams)
    unknown_policy_ids = sorted((requested_required | requested_optional) - stream_ids)
    critical_streams = {
        stream.stream_id
        for stream in streams
        if stream.kind in {"image", "camera_info", "radar"}
    }
    if readiness_profile == "strict":
        required_ids = stream_ids
        optional_ids: set[str] = set()
        policy_status: CheckStatus = "pass"
        policy_reason = "strict profile treats every discovered stream as required"
    else:
        required_ids = requested_required | {
            stream.stream_id
            for stream in streams
            if stream.kind in set(required_stream_kinds)
        }
        optional_ids = requested_optional | (stream_ids - required_ids - critical_streams)
        undeclared_critical = sorted(critical_streams - required_ids - requested_optional)
        if unknown_policy_ids:
            policy_status = "blocked"
            policy_reason = (
                "readiness policy references unknown stream(s): "
                + ", ".join(unknown_policy_ids)
            )
        elif undeclared_critical:
            policy_status = "blocked"
            policy_reason = (
                "critical Image/CameraInfo/Radar stream(s) must be explicitly required "
                "or optional: "
                + ", ".join(undeclared_critical)
            )
        elif not required_ids:
            policy_status = "blocked"
            policy_reason = "declared profile has no required stream"
        else:
            policy_status = "pass"
            policy_reason = (
                f"declared profile requires {len(required_ids)} stream(s); "
                f"{len(optional_ids)} optional stream(s) are non-gating"
            )
    add(
        "streams.policy",
        policy_status,
        "required stream declaration",
        policy_reason,
        observed={
            "profile": readiness_profile,
            "required": sorted(required_ids),
            "optional": sorted(optional_ids),
            "required_kinds": sorted(set(required_stream_kinds)),
        },
        action=(
            "Declare required_streams/required_stream_kinds and explicitly mark "
            "non-required critical streams optional."
        )
        if policy_status == "blocked"
        else None,
    )
    required = [stream_by_id[stream_id] for stream_id in sorted(required_ids)]
    optional = [stream_by_id[stream_id] for stream_id in sorted(optional_ids)]
    supported = [stream for stream in required if stream.decode_status == "supported"]
    unsupported = [stream for stream in required if stream.decode_status == "unsupported"]
    unknown = [stream for stream in required if stream.decode_status == "unknown"]
    optional_unsupported = [
        stream for stream in optional if stream.decode_status in {"unsupported", "unknown"}
    ]
    if not streams:
        stream_status: CheckStatus = "blocked"
        stream_reason = "no streams were discovered"
    elif not required:
        stream_status = "blocked"
        stream_reason = "no required stream is available for calibration intake"
    elif unsupported or unknown:
        stream_status = "blocked"
        stream_reason = (
            f"{len(supported)} required stream(s) supported; "
            f"{len(unsupported)} required stream(s) unsupported and "
            f"{len(unknown)} required stream(s) unknown"
        )
    elif policy_status == "blocked":
        stream_status = "blocked"
        stream_reason = "stream capability cannot pass until the readiness policy is fixed"
    else:
        stream_status = "pass"
        optional_suffix = (
            f"; {len(optional_unsupported)} optional stream(s) are non-gating"
            if optional_unsupported
            else ""
        )
        stream_reason = (
            f"{len(supported)} required stream(s) have supported decoders" + optional_suffix
        )
    add(
        "streams.capability",
        stream_status,
        "stream decoder capability",
        stream_reason,
        observed={
            "total": len(streams),
            "supported": len(supported),
            "unsupported": len(unsupported),
            "unknown": len(unknown),
            "optional_unsupported": len(optional_unsupported),
        },
        action="Install or register an adapter for every required Image/CameraInfo/Radar stream."
        if unsupported or unknown or not streams
        else None,
    )
    required_camera_info = [
        stream for stream in required if stream.kind == "camera_info"
    ]
    optional_camera_info = [
        stream for stream in optional if stream.kind == "camera_info"
    ]
    camera_info_unusable = [
        stream
        for stream in [*required_camera_info, *optional_camera_info]
        if stream.calibration_status in {"uncalibrated", "mixed", "unknown"}
    ]
    required_camera_info_unusable = [
        stream for stream in required_camera_info if stream in camera_info_unusable
    ]
    if required_camera_info_unusable:
        camera_info_status: CheckStatus = "blocked"
        camera_info_reason = (
            "required CameraInfo stream(s) are not calibrated: "
            + ", ".join(stream.stream_id for stream in required_camera_info_unusable)
        )
    elif camera_info_unusable:
        camera_info_status = "warn"
        camera_info_reason = (
            "optional CameraInfo stream(s) are not calibrated: "
            + ", ".join(stream.stream_id for stream in camera_info_unusable)
        )
    else:
        camera_info_status = "pass"
        camera_info_reason = (
            "all required CameraInfo streams expose usable intrinsics"
            if required_camera_info
            else "no CameraInfo calibration evidence is required"
        )
    add(
        "camera_info.calibration",
        camera_info_status,
        "CameraInfo calibration readiness",
        camera_info_reason,
        observed={
            "required": {
                stream.stream_id: stream.calibration_status
                for stream in required_camera_info
            },
            "optional": {
                stream.stream_id: stream.calibration_status
                for stream in optional_camera_info
            },
        },
        action=(
            "Provide calibrated CameraInfo intrinsics (K/P) or declare this stream "
            "optional before calibration adoption."
        )
        if camera_info_status != "pass"
        else None,
    )
    required_radar = [stream for stream in required if stream.kind == "radar"]
    optional_radar = [stream for stream in optional if stream.kind == "radar"]
    required_radar_weak = [
        stream
        for stream in required_radar
        if stream.radar_diversity_status in {"empty", "weak", "unknown", "not_applicable"}
    ]
    optional_radar_weak = [
        stream
        for stream in optional_radar
        if stream.radar_diversity_status in {"empty", "weak", "unknown", "not_applicable"}
    ]
    if required_radar_weak:
        radar_status: CheckStatus = "blocked"
        radar_reason = (
            "required RadarScan stream(s) have empty or weak return diversity: "
            + ", ".join(
                f"{stream.stream_id}={stream.radar_diversity_status}"
                for stream in required_radar_weak
            )
        )
    elif optional_radar_weak:
        radar_status = "warn"
        radar_reason = (
            "optional RadarScan stream(s) have empty or weak return diversity: "
            + ", ".join(
                f"{stream.stream_id}={stream.radar_diversity_status}"
                for stream in optional_radar_weak
            )
        )
    else:
        radar_status = "pass"
        radar_reason = (
            "all required RadarScan streams expose non-degenerate return diversity"
            if required_radar
            else "no RadarScan diversity evidence is required"
        )
    add(
        "radar.diversity",
        radar_status,
        "RadarScan return diversity",
        radar_reason,
        observed={
            "required": {
                stream.stream_id: {
                    "status": stream.radar_diversity_status,
                    "return_count": stream.radar_return_count,
                    "range_min_m": stream.radar_range_min_m,
                    "range_max_m": stream.radar_range_max_m,
                    "azimuth_span_rad": stream.radar_azimuth_span_rad,
                    "elevation_span_rad": stream.radar_elevation_span_rad,
                    "doppler_span_mps": stream.radar_doppler_span_mps,
                }
                for stream in required_radar
            },
            "optional": {
                stream.stream_id: {
                    "status": stream.radar_diversity_status,
                    "return_count": stream.radar_return_count,
                    "range_min_m": stream.radar_range_min_m,
                    "range_max_m": stream.radar_range_max_m,
                    "azimuth_span_rad": stream.radar_azimuth_span_rad,
                    "elevation_span_rad": stream.radar_elevation_span_rad,
                    "doppler_span_mps": stream.radar_doppler_span_mps,
                }
                for stream in optional_radar
            },
        },
        action=(
            "Record a RadarScan with multiple finite returns spanning range, angle, "
            "or Doppler, or declare the radar stream optional before adoption."
        )
        if radar_status != "pass"
        else None,
    )
    missing_frame = [
        stream.stream_id
        for stream in required
        if stream.frame_id is None or not stream.frame_id or stream.frame_conflict
    ]
    frame_conflicts = [stream.stream_id for stream in required if stream.frame_conflict]
    add(
        "streams.frame",
        "blocked" if frame_conflicts else "pass" if required and not missing_frame else "warn",
        "stream frame binding",
        "all required streams expose a frame_id"
        if required and not missing_frame
        else (
            "frame_id conflicts across sampled messages for: "
            + ", ".join(frame_conflicts)
            if frame_conflicts
            else "frame_id is unknown for: "
            + ", ".join(missing_frame or ["all required streams"])
        ),
        observed={"missing": missing_frame, "conflicts": frame_conflicts},
        action=(
            "Preserve one stable frame_id per stream; resolve the conflict before "
            "calibration intake."
            if frame_conflicts
            else "Preserve frame_id in the source message or provide an explicit frame binding."
        ),
        priority="recommended",
    )
    missing_clock = [stream.stream_id for stream in required if stream.clock_domain is None]
    add(
        "streams.clock",
        "pass" if required and not missing_clock else "warn",
        "stream clock binding",
        "all required streams expose a clock domain"
        if required and not missing_clock
        else "clock domain is unknown for: " + ", ".join(missing_clock or ["all required streams"]),
        observed={"missing": missing_clock},
        action="Declare the source timestamp clock and any inter-stream offset.",
        priority="recommended",
    )
    required_ids_for_binding = {stream.stream_id for stream in required}
    unknown_frame_bindings = [
        binding
        for binding in frame_bindings
        if binding.status != "known"
        and {binding.source_stream, binding.target_stream} <= required_ids_for_binding
    ]
    unknown_clock_bindings = [
        binding
        for binding in clock_bindings
        if binding.status != "known"
        and {binding.source_stream, binding.target_stream} <= required_ids_for_binding
    ]
    if len(required) > 1 and (
        not clock_bindings
        or not frame_bindings
        or unknown_frame_bindings
        or unknown_clock_bindings
    ):
        add(
            "bindings.cross_stream",
            "warn",
            "cross-stream binding",
            "multi-stream source lacks complete clock/frame binding evidence",
            observed={
                "clock_bindings": len(clock_bindings),
                "frame_bindings": len(frame_bindings),
                "unknown_clock_bindings": [
                    binding.binding_id for binding in unknown_clock_bindings
                ],
                "unknown_frame_bindings": [
                    binding.binding_id for binding in unknown_frame_bindings
                ],
            },
            action="Supply explicit cross-stream clock and frame bindings before adoption.",
        )
    else:
        add(
            "bindings.cross_stream",
            "pass" if required else "warn",
            "cross-stream binding",
            "binding evidence is sufficient for the discovered stream count"
            if required
            else "binding cannot be evaluated without required streams",
            observed={"clock_bindings": len(clock_bindings), "frame_bindings": len(frame_bindings)},
            priority="recommended",
        )
    if source.mcap_integrity is not None:
        integrity = source.mcap_integrity
        add(
            "source.mcap_integrity",
            integrity.status,
            source.path or "source",
            (
                "MCAP framing, link, and available CRC evidence passed"
                if integrity.status == "pass"
                else "MCAP integrity evidence requires review"
                if integrity.status == "warn"
                else "MCAP integrity evidence is blocking"
            ),
            observed=integrity.model_dump(mode="json", exclude_none=False),
            action=(
                "Repair the MCAP framing/CRC/link mismatch before calibration intake."
                if integrity.status == "blocked"
                else "Review MCAP summary/index and unknown CRC evidence before adoption."
            ),
            priority="required" if integrity.status == "blocked" else "recommended",
        )
    for evidence_name, evidence_status in (
        ("source.index", source.index_status),
        ("source.crc", source.crc_status),
    ):
        add(
            evidence_name,
            "pass" if evidence_status in {"known", "not_applicable"} else "warn",
            source.path or "source",
            f"{evidence_name.split('.')[-1]} evidence is {evidence_status}",
            observed={"status": evidence_status},
            action=(
                f"Preserve verifiable {evidence_name.split('.')[-1].upper()} evidence "
                "for the source."
            )
            if evidence_status == "unknown"
            else None,
            priority="recommended",
        )
    statuses = {check.status for check in checks}
    status: CaptureStatus = (
        "blocked" if "blocked" in statuses else "warn" if "warn" in statuses else "ready"
    )
    summary = {
        "ready": "capture inventory is ready for calibration intake",
        "warn": "capture inventory is usable only with review of warnings",
        "blocked": (
            "capture inventory is blocked until required identity/source evidence "
            "is supplied"
        ),
    }[status]
    return checks, actions, status, summary
