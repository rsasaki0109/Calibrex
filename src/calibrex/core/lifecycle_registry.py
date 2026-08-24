"""Filesystem-first, append-only calibration lifecycle registry.

The registry is deliberately independent of ROS.  It stores a small immutable
event log (``events.jsonl``), a materialised state projection, and a digest
bound head under one caller supplied directory.  The state projection is
cache-like: the event log is the source of truth and :func:`verify_registry`
replays it before accepting the projection.

This module is an additive v0.2 companion to ``calibration_lifecycle/v0.1``.
The older synthetic replay remains unchanged and can be used to exercise the
adoption policy without a filesystem registry.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, NoReturn, Protocol, cast

from pydantic import Field, field_validator

from calibrex.core.capture_manifest import load_capture_manifest, verify_capture_manifest_inputs
from calibrex.core.exceptions import CalibrexError
from calibrex.core.io import read_mapping, write_mapping_atomic
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.result import StrictModel, load_result

LIFECYCLE_REGISTRY_SCHEMA_VERSION: Literal["slac.calibration_lifecycle_registry/v0.2"] = (
    "slac.calibration_lifecycle_registry/v0.2"
)
LIFECYCLE_EVENT_SCHEMA_VERSION: Literal["slac.calibration_lifecycle_event/v0.2"] = (
    "slac.calibration_lifecycle_event/v0.2"
)
LIFECYCLE_EVALUATION_SCHEMA_VERSION: Literal["slac.calibration_lifecycle_evaluation/v0.2"] = (
    "slac.calibration_lifecycle_evaluation/v0.2"
)
REGISTRY_SCHEMA_VERSION = LIFECYCLE_REGISTRY_SCHEMA_VERSION
EVENT_SCHEMA_VERSION = LIFECYCLE_EVENT_SCHEMA_VERSION
EVALUATION_SCHEMA_VERSION = LIFECYCLE_EVALUATION_SCHEMA_VERSION

ZERO_SHA256 = "0" * 64
_SHA256_PATTERN = r"^[0-9a-f]{64}$"

REGISTRY_MANIFEST_FILENAME = "registry.json"
REGISTRY_STATE_FILENAME = "state.json"
REGISTRY_EVENTS_FILENAME = "events.jsonl"
REGISTRY_HEAD_FILENAME = "head.json"
REGISTRY_LOCK_FILENAME = ".registry.lock"
REGISTRY_PENDING_FILENAME = ".registry.pending.json"

LifecycleEventType = Literal[
    "register",
    "install",
    "capture",
    "evaluate",
    "hold",
    "promote",
    "reject",
    "rollback",
    "remove",
]
LifecycleStatus = Literal[
    "PASS",
    "WARN",
    "FAIL",
    "INCONCLUSIVE",
    "BLOCKED",
    "REGISTERED",
    "INSTALLED",
    "CAPTURED",
    "PROMOTED",
    "ROLLED_BACK",
    "REMOVED",
]
LifecycleAdmission = Literal[
    "ADOPT",
    "DO_NOT_ADOPT",
    "HOLD",
    "BLOCKED",
    "ROLLBACK",
    "NOT_APPLICABLE",
]
SensorState = Literal["registered", "installed", "removed"]
EdgeState = Literal["registered", "candidate", "active", "retired"]


class LifecycleRegistryError(CalibrexError):
    """Raised when a registry operation cannot be admitted safely."""


class LifecycleRegistryVerificationError(LifecycleRegistryError):
    """Raised by strict verification when the chain or source is invalid."""


class LifecycleRegistryConcurrencyError(LifecycleRegistryError):
    """Raised when an append observes a changed head or cannot acquire a lock."""


class LifecycleRollbackAdapter(Protocol):
    """Narrow adapter boundary for an optional downstream rollback operation."""

    def __call__(self, edge: CalibrationEdge, target: CalibrationEdge) -> None:
        """Apply an already guarded rollback to an external deployment."""


class LifecycleProvenance(StrictModel):
    """Operator and tool lineage for a registry event or generated artifact."""

    operator: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tool: str = "calibrex.lifecycle"
    tool_version: str = "0.2"
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)


class LifecycleArtifactRef(StrictModel):
    """Exact filesystem reference retained in the event log."""

    kind: str = Field(min_length=1)
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    schema_version: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)

    @property
    def digest(self) -> str:
        """Compatibility spelling for callers using ``digest``."""

        return self.sha256


class LifecycleArtifactSet(StrictModel):
    """Digest references attached to one candidate/decision."""

    incumbent_transform: LifecycleArtifactRef | None = None
    incumbent_result: LifecycleArtifactRef | None = None
    candidate_transform: LifecycleArtifactRef | None = None
    candidate_result: LifecycleArtifactRef | None = None
    candidate_pilot: LifecycleArtifactRef | None = None
    promotion: LifecycleArtifactRef | None = None
    smoke: LifecycleArtifactRef | None = None
    capture: LifecycleArtifactRef | None = None
    evidence: LifecycleArtifactRef | None = None
    assessment: LifecycleArtifactRef | None = None
    policy: LifecycleArtifactRef | None = None
    evaluation: LifecycleArtifactRef | None = None

    def refs(self) -> list[LifecycleArtifactRef]:
        """Return declared references in stable field order."""

        return [
            value
            for value in (
                self.incumbent_transform,
                self.incumbent_result,
                self.candidate_transform,
                self.candidate_result,
                self.candidate_pilot,
                self.promotion,
                self.smoke,
                self.capture,
                self.evidence,
                self.assessment,
                self.policy,
                self.evaluation,
            )
            if value is not None
        ]

    def by_kind(self) -> dict[str, LifecycleArtifactRef]:
        """Return non-null references keyed by their artifact role."""

        return {
            name: value
            for name in type(self).model_fields
            if isinstance(value := getattr(self, name), LifecycleArtifactRef)
        }


class VehicleRecord(StrictModel):
    """Stable vehicle identity used to scope every edge and sensor."""

    vehicle_id: str = Field(min_length=1)
    name: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    status: Literal["active", "retired"] = "active"


class SensorKitRecord(StrictModel):
    """Vehicle-local sensor-kit identity."""

    sensor_kit_id: str = Field(min_length=1)
    vehicle_id: str = Field(min_length=1)
    name: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    status: Literal["active", "retired"] = "active"


class PhysicalSensorRecord(StrictModel):
    """Physical sensor identity and mount lifecycle."""

    sensor_id: str = Field(min_length=1)
    vehicle_id: str = Field(min_length=1)
    sensor_kit_id: str = Field(min_length=1)
    serial: str = Field(min_length=1)
    model: str = Field(min_length=1)
    firmware: str = Field(min_length=1)
    mount: str = Field(min_length=1)
    frame: str | None = None
    state: SensorState = "registered"
    installed_at: str | None = None
    removed_at: str | None = None
    remount_count: int = Field(default=0, ge=0)
    install_count: int = Field(default=0, ge=0)
    # The projection sequence is used to reject a stale install/remove
    # command that was prepared before another operator changed this sensor.
    updated_sequence: int = Field(default=-1, ge=-1)
    provenance: LifecycleProvenance | None = None

    @property
    def mount_id(self) -> str:
        """Compatibility spelling for the physical mounting identity."""

        return self.mount


class CalibrationEdge(StrictModel):
    """One vehicle-local parent/child calibration edge."""

    edge_id: str = Field(min_length=1)
    vehicle_id: str = Field(min_length=1)
    sensor_kit_id: str = Field(min_length=1)
    parent_frame: str = Field(min_length=1)
    child_frame: str = Field(min_length=1)
    state: EdgeState = "registered"
    incumbent_transform: LifecycleArtifactRef | None = None
    incumbent_result: LifecycleArtifactRef | None = None
    candidate_transform: LifecycleArtifactRef | None = None
    candidate_result: LifecycleArtifactRef | None = None
    candidate_pilot: LifecycleArtifactRef | None = None
    promotion: LifecycleArtifactRef | None = None
    smoke: LifecycleArtifactRef | None = None
    capture: LifecycleArtifactRef | None = None
    evidence: LifecycleArtifactRef | None = None
    assessment: LifecycleArtifactRef | None = None
    policy: LifecycleArtifactRef | None = None
    last_evaluation_status: LifecycleStatus | None = None
    last_admission: LifecycleAdmission | None = None
    updated_sequence: int = Field(default=-1, ge=-1)
    updated_at: str | None = None

    @property
    def incumbent_transform_sha256(self) -> str | None:
        """Digest convenience accessor."""

        return self.incumbent_transform.sha256 if self.incumbent_transform else None

    @property
    def incumbent_result_sha256(self) -> str | None:
        """Digest convenience accessor."""

        return self.incumbent_result.sha256 if self.incumbent_result else None

    @property
    def candidate_result_sha256(self) -> str | None:
        """Digest convenience accessor."""

        return self.candidate_result.sha256 if self.candidate_result else None

    @property
    def candidate_transform_sha256(self) -> str | None:
        """Digest convenience accessor."""

        return self.candidate_transform.sha256 if self.candidate_transform else None

    @property
    def candidate_pilot_sha256(self) -> str | None:
        """Digest convenience accessor."""

        return self.candidate_pilot.sha256 if self.candidate_pilot else None

    @property
    def promotion_sha256(self) -> str | None:
        """Digest convenience accessor."""

        return self.promotion.sha256 if self.promotion else None

    @property
    def smoke_sha256(self) -> str | None:
        """Digest convenience accessor."""

        return self.smoke.sha256 if self.smoke else None

    @property
    def capture_sha256(self) -> str | None:
        """Digest convenience accessor."""

        return self.capture.sha256 if self.capture else None

    @property
    def evidence_sha256(self) -> str | None:
        """Digest convenience accessor."""

        return self.evidence.sha256 if self.evidence else None

    @property
    def policy_sha256(self) -> str | None:
        """Digest convenience accessor."""

        return self.policy.sha256 if self.policy else None


class CaptureRecord(StrictModel):
    """Capture references retained by the projection."""

    capture_id: str = Field(min_length=1)
    vehicle_id: str
    sensor_kit_id: str
    status: str
    artifact: LifecycleArtifactRef
    sequence: int = Field(ge=0)


class LifecycleRegistryState(StrictModel):
    """Materialised state reconstructed from immutable events."""

    schema_version: Literal["slac.calibration_lifecycle_registry_state/v0.2"] = (
        "slac.calibration_lifecycle_registry_state/v0.2"
    )
    registry_id: str = Field(min_length=1)
    last_sequence: int = Field(default=-1, ge=-1)
    vehicles: dict[str, VehicleRecord] = Field(default_factory=dict)
    sensor_kits: dict[str, SensorKitRecord] = Field(default_factory=dict)
    sensors: dict[str, PhysicalSensorRecord] = Field(default_factory=dict)
    edges: dict[str, CalibrationEdge] = Field(default_factory=dict)
    captures: dict[str, CaptureRecord] = Field(default_factory=dict)


class RegistryManifest(StrictModel):
    """Immutable registry identity and creation provenance."""

    schema_version: Literal["slac.calibration_lifecycle_registry/v0.2"] = (
        LIFECYCLE_REGISTRY_SCHEMA_VERSION
    )
    registry_id: str = Field(min_length=1)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    provenance: LifecycleProvenance


class RegistryHead(StrictModel):
    """Digest-bound checkpoint for the event log and projection."""

    schema_version: Literal["slac.calibration_lifecycle_head/v0.2"] = (
        "slac.calibration_lifecycle_head/v0.2"
    )
    registry_id: str = Field(min_length=1)
    sequence: int = Field(ge=-1)
    event_sha256: str = Field(pattern=_SHA256_PATTERN)
    state_sha256: str = Field(pattern=_SHA256_PATTERN)
    checkpoint_sha256: str = Field(pattern=_SHA256_PATTERN)
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class LifecycleEvent(StrictModel):
    """Immutable tamper-evident registry event."""

    schema_version: Literal["slac.calibration_lifecycle_event/v0.2"] = (
        LIFECYCLE_EVENT_SCHEMA_VERSION
    )
    registry_id: str = Field(min_length=1)
    sequence: int = Field(ge=0)
    event_type: LifecycleEventType
    status: LifecycleStatus
    admission: LifecycleAdmission = "NOT_APPLICABLE"
    vehicle_id: str | None = None
    sensor_kit_id: str | None = None
    sensor_id: str | None = None
    edge_ids: list[str] = Field(default_factory=list)
    provenance: LifecycleProvenance
    artifacts: LifecycleArtifactSet = Field(default_factory=LifecycleArtifactSet)
    payload: dict[str, Any] = Field(default_factory=dict)
    previous_event_sha256: str = Field(default=ZERO_SHA256, pattern=_SHA256_PATTERN)
    event_sha256: str = Field(default=ZERO_SHA256, pattern=_SHA256_PATTERN)

    @field_validator("edge_ids")
    @classmethod
    def _unique_edge_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("event edge_ids must be unique")
        if any(not item for item in value):
            raise ValueError("event edge_ids must not contain empty IDs")
        return value

    @property
    def operator(self) -> str:
        """Convenience access to event operator provenance."""

        return self.provenance.operator

    @property
    def reason(self) -> str:
        """Convenience access to event reason provenance."""

        return self.provenance.reason

    @property
    def timestamp(self) -> str:
        """Convenience access to event timestamp provenance."""

        return self.provenance.timestamp

    @property
    def tool(self) -> str:
        """Convenience access to tool provenance."""

        return self.provenance.tool

    @property
    def git_commit(self) -> str | None:
        """Convenience access to source commit provenance."""

        return self.provenance.git_commit

    @property
    def event_kind(self) -> LifecycleEventType:
        """Compatibility spelling shared with the synthetic v0.1 artifact."""

        return self.event_type


class LifecycleEvaluationArtifact(StrictModel):
    """Fail-closed evaluation decision retained by an evaluate event."""

    schema_version: Literal["slac.calibration_lifecycle_evaluation/v0.2"] = (
        LIFECYCLE_EVALUATION_SCHEMA_VERSION
    )
    evaluation_id: str = Field(min_length=1)
    registry_id: str = Field(min_length=1)
    vehicle_id: str = Field(min_length=1)
    sensor_kit_id: str = Field(min_length=1)
    edge_ids: list[str] = Field(min_length=1)
    status: Literal["PASS", "WARN", "FAIL", "INCONCLUSIVE", "BLOCKED"]
    admission: LifecycleAdmission
    reason: str = Field(min_length=1)
    gates: dict[str, Literal["PASS", "WARN", "FAIL", "INCONCLUSIVE", "BLOCKED"]] = Field(
        default_factory=dict
    )
    artifacts: LifecycleArtifactSet
    provenance: LifecycleProvenance
    artifact_sha256: str = Field(default=ZERO_SHA256, pattern=_SHA256_PATTERN)

    @field_validator("edge_ids")
    @classmethod
    def _unique_edge_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("evaluation edge_ids must be unique")
        if any(not item for item in value):
            raise ValueError("evaluation edge_ids must not contain empty IDs")
        return value

    def with_artifact_digest(self) -> LifecycleEvaluationArtifact:
        """Return a copy with a deterministic self-digest."""

        payload = self.model_dump(mode="json", exclude_none=False)
        payload.pop("artifact_sha256", None)
        digest = _canonical_sha256(payload)
        return self.model_copy(update={"artifact_sha256": digest})

    def verify_artifact_digest(self) -> None:
        """Raise if the self-digest does not match the canonical payload."""

        expected = self.with_artifact_digest().artifact_sha256
        if expected != self.artifact_sha256:
            raise LifecycleRegistryVerificationError(
                "evaluation self-digest mismatch: "
                f"declared={self.artifact_sha256}, expected={expected}"
            )

    def save(self, path: str | Path) -> None:
        """Write a digest-complete YAML or JSON evaluation artifact."""

        write_mapping_atomic(Path(path), self.with_artifact_digest().model_dump(mode="json"))


class RegistryVerificationArtifact(StrictModel):
    """Schema-valid result of checking a registry and all source artifacts."""

    schema_version: Literal["slac.calibration_lifecycle_registry_verification/v0.2"] = (
        "slac.calibration_lifecycle_registry_verification/v0.2"
    )
    registry_id: str
    root: str
    valid: bool
    head_sequence: int = Field(ge=-1)
    event_count: int = Field(ge=0)
    checked_artifact_count: int = Field(default=0, ge=0)
    errors: list[str] = Field(default_factory=list)
    provenance: LifecycleProvenance


class LifecycleRegistryStatus(StrictModel):
    """Compact status projection suitable for CLI/API clients."""

    schema_version: Literal["slac.calibration_lifecycle_status/v0.2"] = (
        "slac.calibration_lifecycle_status/v0.2"
    )
    registry_id: str
    head_sequence: int = Field(ge=-1)
    event_count: int = Field(ge=0)
    vehicle_count: int = Field(ge=0)
    sensor_kit_count: int = Field(ge=0)
    sensor_count: int = Field(ge=0)
    active_sensor_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    active_edge_count: int = Field(ge=0)
    last_event_sha256: str = Field(pattern=_SHA256_PATTERN)
    provenance: LifecycleProvenance


class LifecycleRegistry:
    """Filesystem-backed registry with serialized append operations."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    @property
    def manifest_path(self) -> Path:
        return self.root / REGISTRY_MANIFEST_FILENAME

    @property
    def state_path(self) -> Path:
        return self.root / REGISTRY_STATE_FILENAME

    @property
    def events_path(self) -> Path:
        return self.root / REGISTRY_EVENTS_FILENAME

    @property
    def head_path(self) -> Path:
        return self.root / REGISTRY_HEAD_FILENAME

    @property
    def pending_path(self) -> Path:
        """Crash-recovery marker for an append whose projection was incomplete."""

        return self.root / REGISTRY_PENDING_FILENAME

    @classmethod
    def init(
        cls,
        root: str | Path,
        *,
        registry_id: str = "calibrex-registry",
        operator: str = "unknown",
        reason: str = "initialize calibration lifecycle registry",
        timestamp: str | None = None,
        command: Sequence[str] | None = None,
    ) -> LifecycleRegistry:
        """Create an empty registry without overwriting an existing one."""

        registry = cls(root)
        registry.root.mkdir(parents=True, exist_ok=True)
        # Initialization is itself a compare-and-swap operation.  Without the
        # lock, two first writers could each pass the existence check and
        # interleave manifest/events/state writes into an unverifiable tree.
        with registry._lock():
            if any(
                path.exists()
                for path in (
                    registry.manifest_path,
                    registry.state_path,
                    registry.events_path,
                    registry.head_path,
                )
            ):
                raise LifecycleRegistryError(f"registry already exists: {registry.root}")
            provenance = _provenance(
                operator=operator,
                reason=reason,
                timestamp=timestamp,
                command=command,
            )
            manifest = RegistryManifest(registry_id=registry_id, provenance=provenance)
            state = LifecycleRegistryState(registry_id=registry_id)
            write_mapping_atomic(registry.manifest_path, manifest.model_dump(mode="json"))
            registry.events_path.write_text("", encoding="utf-8", newline="\n")
            write_mapping_atomic(registry.state_path, state.model_dump(mode="json"))
            registry._write_head(-1, ZERO_SHA256, state)
        return registry

    @classmethod
    def open(cls, root: str | Path) -> LifecycleRegistry:
        """Open an existing registry after checking required files."""

        registry = cls(root)
        if not registry.manifest_path.is_file():
            raise LifecycleRegistryError(f"registry manifest is missing: {registry.manifest_path}")
        if (
            not registry.events_path.is_file()
            or not registry.state_path.is_file()
            or not registry.head_path.is_file()
        ):
            raise LifecycleRegistryError(f"registry files are incomplete: {registry.root}")
        return registry

    load = open
    initialize = init

    def manifest(self) -> RegistryManifest:
        """Load the immutable registry manifest."""

        try:
            return RegistryManifest.model_validate(read_mapping(self.manifest_path))
        except Exception as exc:
            raise LifecycleRegistryVerificationError(f"invalid registry manifest: {exc}") from exc

    def state(self) -> LifecycleRegistryState:
        """Load the materialised state projection."""

        try:
            return LifecycleRegistryState.model_validate(read_mapping(self.state_path))
        except Exception as exc:
            raise LifecycleRegistryVerificationError(f"invalid registry state: {exc}") from exc

    def head(self) -> RegistryHead:
        """Load the digest-bound checkpoint."""

        try:
            return RegistryHead.model_validate(read_mapping(self.head_path))
        except Exception as exc:
            raise LifecycleRegistryVerificationError(f"invalid registry head: {exc}") from exc

    def events(self) -> list[LifecycleEvent]:
        """Load immutable events in file order."""

        events: list[LifecycleEvent] = []
        with self.events_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    events.append(LifecycleEvent.model_validate(json.loads(line)))
                except Exception as exc:
                    raise LifecycleRegistryVerificationError(
                        f"invalid event at line {line_number}: {exc}"
                    ) from exc
        return events

    def append_event(
        self,
        event: LifecycleEvent,
        *,
        expected_sequence: int | None = None,
        expected_head_sha256: str | None = None,
    ) -> LifecycleEvent:
        """Append one event atomically under an exclusive cross-platform lock.

        The caller supplies event content with any sequence/hash placeholders;
        this method assigns the next sequence and both chain digests.  A stale
        expected sequence/head is rejected before the file is touched.
        """

        with self._lock():
            self._recover_pending_unlocked()
            self._verify_unlocked(verify_sources=False)
            current_events = self.events()
            current_sequence = len(current_events) - 1
            current_head = self.head()
            if expected_sequence is not None and expected_sequence != current_sequence:
                raise LifecycleRegistryConcurrencyError(
                    "stale registry sequence: "
                    f"expected {expected_sequence}, current {current_sequence}"
                )
            if (
                expected_head_sha256 is not None
                and expected_head_sha256 != current_head.event_sha256
            ):
                raise LifecycleRegistryConcurrencyError(
                    "stale registry head: expected "
                    f"{expected_head_sha256}, current {current_head.event_sha256}"
                )
            if event.registry_id != self.manifest().registry_id:
                raise LifecycleRegistryError("event registry_id does not match registry manifest")
            for ref in event.artifacts.refs():
                _verify_artifact_ref(self.root, ref)
            sequence = current_sequence + 1
            previous = current_events[-1].event_sha256 if current_events else ZERO_SHA256
            candidate = event.model_copy(
                update={
                    "sequence": sequence,
                    "previous_event_sha256": previous,
                    "event_sha256": ZERO_SHA256,
                }
            )
            digest = _event_digest(candidate)
            committed = candidate.model_copy(update={"event_sha256": digest})
            pending = {
                "event": committed.model_dump(mode="json", exclude_none=False),
                "state": self._replay([*current_events, committed]).model_dump(
                    mode="json", exclude_none=False
                ),
            }
            # The pending marker makes a state/head write failure recoverable:
            # the immutable event line is never removed or rewritten.
            write_mapping_atomic(self.pending_path, pending)
            line = _canonical_json(committed.model_dump(mode="json", exclude_none=False)) + "\n"
            with self.events_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
            state = self._replay(self.events())
            write_mapping_atomic(self.state_path, state.model_dump(mode="json"))
            self._write_head(sequence, digest, state)
            with suppress(FileNotFoundError):
                self.pending_path.unlink()
            return committed

    append = append_event

    def verify(
        self, *, verify_sources: bool = True, strict: bool = False
    ) -> RegistryVerificationArtifact:
        """Verify chain, head, state projection, and source artifact digests."""

        errors: list[str] = []
        event_count = 0
        sequence = -1
        registry_id = "unknown"
        checked = 0
        try:
            registry_id = self.manifest().registry_id
            self._verify_unlocked(verify_sources=verify_sources)
            events = self.events()
            event_count = len(events)
            sequence = len(events) - 1
            if verify_sources:
                checked = sum(len(event.artifacts.refs()) for event in events)
        except Exception as exc:
            errors.append(str(exc))
            try:
                event_count = len(self.events())
                sequence = event_count - 1
            except Exception:
                pass
        artifact = RegistryVerificationArtifact(
            registry_id=registry_id,
            root=str(self.root),
            valid=not errors,
            head_sequence=sequence,
            event_count=event_count,
            checked_artifact_count=checked,
            errors=errors,
            provenance=_provenance(operator="system", reason="verify lifecycle registry"),
        )
        if strict and not artifact.valid:
            raise LifecycleRegistryVerificationError("; ".join(errors))
        return artifact

    def recover(self) -> RegistryVerificationArtifact:
        """Repair a projection interrupted after an event-line append.

        Recovery is only accepted when the pending event is exactly the next
        valid chain element.  It never removes or rewrites an event line.
        """

        with self._lock():
            self._recover_pending_unlocked()
            self._verify_unlocked(verify_sources=True)
        return self.verify(verify_sources=True)

    def status(self) -> LifecycleRegistryStatus:
        """Return a typed status summary after strict verification."""

        self._verify_unlocked(verify_sources=True)
        manifest = self.manifest()
        state = self.state()
        head = self.head()
        events = self.events()
        return LifecycleRegistryStatus(
            registry_id=manifest.registry_id,
            head_sequence=head.sequence,
            event_count=len(events),
            vehicle_count=len(state.vehicles),
            sensor_kit_count=len(state.sensor_kits),
            sensor_count=len(state.sensors),
            active_sensor_count=sum(
                1 for item in state.sensors.values() if item.state == "installed"
            ),
            edge_count=len(state.edges),
            active_edge_count=sum(1 for item in state.edges.values() if item.state == "active"),
            last_event_sha256=head.event_sha256,
            provenance=_provenance(operator="system", reason="read lifecycle registry status"),
        )

    def _write_head(self, sequence: int, event_sha256: str, state: LifecycleRegistryState) -> None:
        payload = {
            "schema_version": "slac.calibration_lifecycle_head/v0.2",
            "registry_id": state.registry_id,
            "sequence": sequence,
            "event_sha256": event_sha256,
            "state_sha256": _state_digest(state),
            "checkpoint_sha256": ZERO_SHA256,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        payload["checkpoint_sha256"] = _canonical_sha256(
            {key: value for key, value in payload.items() if key != "checkpoint_sha256"}
        )
        write_mapping_atomic(self.head_path, payload)

    def _verify_unlocked(self, *, verify_sources: bool) -> None:
        if self.pending_path.exists():
            raise LifecycleRegistryVerificationError(
                "registry has an incomplete append transaction; call recover()"
            )
        manifest = self.manifest()
        events = self.events()
        seen_sequences: set[int] = set()
        seen_hashes: set[str] = set()
        previous = ZERO_SHA256
        for expected_sequence, event in enumerate(events):
            if event.registry_id != manifest.registry_id:
                raise LifecycleRegistryVerificationError(
                    f"event {event.sequence} belongs to registry {event.registry_id!r}"
                )
            if event.sequence in seen_sequences:
                raise LifecycleRegistryVerificationError(
                    f"duplicate event sequence {event.sequence}"
                )
            if event.event_sha256 in seen_hashes:
                raise LifecycleRegistryVerificationError(
                    f"duplicate event digest {event.event_sha256}"
                )
            if event.sequence != expected_sequence:
                raise LifecycleRegistryVerificationError(
                    f"event sequence fork/gap: expected {expected_sequence}, got {event.sequence}"
                )
            if event.previous_event_sha256 != previous:
                raise LifecycleRegistryVerificationError(
                    f"event {event.sequence} previous digest does not extend the chain"
                )
            expected_digest = _event_digest(event)
            if event.event_sha256 != expected_digest:
                raise LifecycleRegistryVerificationError(
                    f"event {event.sequence} digest mismatch: "
                    f"declared={event.event_sha256}, expected={expected_digest}"
                )
            seen_sequences.add(event.sequence)
            seen_hashes.add(event.event_sha256)
            if verify_sources:
                for ref in event.artifacts.refs():
                    _verify_artifact_ref(self.root, ref)
            previous = event.event_sha256
        state = self.state()
        replayed = self._replay(events)
        if _canonical_json(state.model_dump(mode="json")) != _canonical_json(
            replayed.model_dump(mode="json")
        ):
            raise LifecycleRegistryVerificationError("state projection does not match event replay")
        head = self.head()
        expected_sequence = len(events) - 1
        if head.sequence != expected_sequence:
            raise LifecycleRegistryVerificationError(
                f"stale registry head sequence: {head.sequence} != {expected_sequence}"
            )
        if head.event_sha256 != previous:
            raise LifecycleRegistryVerificationError("stale registry head event digest")
        if head.registry_id != manifest.registry_id:
            raise LifecycleRegistryVerificationError("registry head identity mismatch")
        if head.state_sha256 != _state_digest(state):
            raise LifecycleRegistryVerificationError("registry state digest mismatch")
        expected_checkpoint = _canonical_sha256(
            {
                key: value
                for key, value in head.model_dump(mode="json").items()
                if key != "checkpoint_sha256"
            }
        )
        if head.checkpoint_sha256 != expected_checkpoint:
            raise LifecycleRegistryVerificationError("registry checkpoint digest mismatch")

    def _recover_pending_unlocked(self) -> None:
        if not self.pending_path.exists():
            return
        try:
            pending = read_mapping(self.pending_path)
            committed = LifecycleEvent.model_validate(pending["event"])
        except Exception as exc:
            raise LifecycleRegistryVerificationError(
                f"invalid pending append transaction: {exc}"
            ) from exc
        if _event_digest(committed) != committed.event_sha256:
            raise LifecycleRegistryVerificationError(
                "pending append transaction event digest mismatch"
            )
        current_events = self.events()
        expected_state = self._replay([*current_events, committed])
        try:
            pending_state = LifecycleRegistryState.model_validate(pending["state"])
        except Exception as exc:
            raise LifecycleRegistryVerificationError(
                f"invalid pending append state: {exc}"
            ) from exc
        if _canonical_json(pending_state.model_dump(mode="json")) != _canonical_json(
            expected_state.model_dump(mode="json")
        ):
            raise LifecycleRegistryVerificationError(
                "pending append transaction state does not match event replay"
            )
        expected_sequence = len(current_events)
        expected_previous = current_events[-1].event_sha256 if current_events else ZERO_SHA256
        if (
            committed.sequence == expected_sequence
            and committed.previous_event_sha256 == expected_previous
        ):
            # The marker was durable but the immutable line was not appended;
            # discard the uncommitted intent and let the caller retry.
            with suppress(FileNotFoundError):
                self.pending_path.unlink()
            return
        last = current_events[-1] if current_events else None
        if (
            last is None
            or last.event_sha256 != committed.event_sha256
            or last.sequence != committed.sequence
        ):
            raise LifecycleRegistryVerificationError(
                "pending append transaction does not match the event-log tail"
            )
        write_mapping_atomic(self.state_path, expected_state.model_dump(mode="json"))
        self._write_head(last.sequence, last.event_sha256, expected_state)
        with suppress(FileNotFoundError):
            self.pending_path.unlink()

    def _replay(self, events: Sequence[LifecycleEvent]) -> LifecycleRegistryState:
        registry_id = self.manifest().registry_id
        state = LifecycleRegistryState(registry_id=registry_id)
        for event in events:
            _apply_event(state, event)
            state.last_sequence = event.sequence
        return state

    @contextmanager
    def _lock(self, *, timeout_s: float = 10.0) -> Iterator[None]:
        """Acquire an O_EXCL lockfile, portable across Windows and POSIX."""

        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / REGISTRY_LOCK_FILENAME
        deadline = time.monotonic() + timeout_s
        descriptor: int | None = None
        while descriptor is None:
            try:
                descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii", errors="ignore"))
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise LifecycleRegistryConcurrencyError(
                        f"could not acquire registry lock within {timeout_s:g}s: {lock_path}"
                    ) from None
                time.sleep(0.01)
        try:
            yield
        finally:
            if descriptor is not None:
                os.close(descriptor)
            with suppress(FileNotFoundError):
                lock_path.unlink()


def init_registry(
    root: str | Path,
    *,
    registry_id: str = "calibrex-registry",
    operator: str = "unknown",
    reason: str = "initialize calibration lifecycle registry",
    timestamp: str | None = None,
    command: Sequence[str] | None = None,
) -> LifecycleRegistry:
    """Functional alias for :meth:`LifecycleRegistry.init`."""

    return LifecycleRegistry.init(
        root,
        registry_id=registry_id,
        operator=operator,
        reason=reason,
        timestamp=timestamp,
        command=command,
    )


def load_registry(root: str | Path) -> LifecycleRegistry:
    """Functional alias for :meth:`LifecycleRegistry.open`."""

    return LifecycleRegistry.open(root)


def verify_registry(
    root: str | Path,
    *,
    verify_sources: bool = True,
    strict: bool = False,
) -> RegistryVerificationArtifact:
    """Verify a registry from its filesystem root."""

    return LifecycleRegistry.open(root).verify(verify_sources=verify_sources, strict=strict)


def initialize_registry(
    root: str | Path,
    *,
    registry_id: str = "calibrex-registry",
    operator: str = "unknown",
    reason: str = "initialize calibration lifecycle registry",
    timestamp: str | None = None,
    command: Sequence[str] | None = None,
) -> LifecycleRegistry:
    """Compatibility alias for :func:`init_registry`."""

    return init_registry(
        root,
        registry_id=registry_id,
        operator=operator,
        reason=reason,
        timestamp=timestamp,
        command=command,
    )


def register_vehicle(
    root: str | Path,
    vehicle_id: str,
    *,
    name: str | None = None,
    metadata: Mapping[str, str] | None = None,
    operator: str = "unknown",
    reason: str = "register vehicle",
    timestamp: str | None = None,
    command: Sequence[str] | None = None,
) -> LifecycleEvent:
    """Register a vehicle identity."""

    registry = LifecycleRegistry.open(root)
    state = registry.state()
    if vehicle_id in state.vehicles:
        raise LifecycleRegistryError(f"vehicle already registered: {vehicle_id}")
    vehicle = VehicleRecord(vehicle_id=vehicle_id, name=name, metadata=dict(metadata or {}))
    event = _event(
        registry,
        event_type="register",
        status="REGISTERED",
        operator=operator,
        reason=reason,
        timestamp=timestamp,
        command=command,
        vehicle_id=vehicle_id,
        payload={"entity_type": "vehicle", "vehicle": vehicle.model_dump(mode="json")},
    )
    return registry.append_event(event)


def register_sensor_kit(
    root: str | Path,
    sensor_kit_id: str,
    vehicle_id: str,
    *,
    name: str | None = None,
    metadata: Mapping[str, str] | None = None,
    operator: str = "unknown",
    reason: str = "register sensor kit",
    timestamp: str | None = None,
    command: Sequence[str] | None = None,
) -> LifecycleEvent:
    """Register a sensor-kit identity under one vehicle."""

    registry = LifecycleRegistry.open(root)
    state = registry.state()
    if sensor_kit_id in state.sensor_kits:
        raise LifecycleRegistryError(f"sensor kit already registered: {sensor_kit_id}")
    if vehicle_id not in state.vehicles:
        raise LifecycleRegistryError(f"unknown vehicle: {vehicle_id}")
    kit = SensorKitRecord(
        sensor_kit_id=sensor_kit_id,
        vehicle_id=vehicle_id,
        name=name,
        metadata=dict(metadata or {}),
    )
    event = _event(
        registry,
        event_type="register",
        status="REGISTERED",
        operator=operator,
        reason=reason,
        timestamp=timestamp,
        command=command,
        vehicle_id=vehicle_id,
        sensor_kit_id=sensor_kit_id,
        payload={"entity_type": "sensor_kit", "sensor_kit": kit.model_dump(mode="json")},
    )
    return registry.append_event(event)


def register_sensor(
    root: str | Path,
    *,
    sensor_id: str,
    vehicle_id: str,
    sensor_kit_id: str,
    serial: str,
    model: str,
    firmware: str,
    mount: str,
    frame: str | None = None,
    install: bool = False,
    operator: str = "unknown",
    reason: str = "register physical sensor",
    timestamp: str | None = None,
    command: Sequence[str] | None = None,
) -> LifecycleEvent:
    """Register a physical sensor, optionally as an initial installation."""

    registry = LifecycleRegistry.open(root)
    state = registry.state()
    if sensor_id in state.sensors:
        raise LifecycleRegistryError(f"sensor already registered: {sensor_id}")
    vehicle = state.vehicles.get(vehicle_id)
    if vehicle is None:
        vehicle = VehicleRecord(vehicle_id=vehicle_id)
    kit = state.sensor_kits.get(sensor_kit_id)
    if kit is not None and kit.vehicle_id != vehicle_id:
        raise LifecycleRegistryError(f"unknown or cross-vehicle sensor kit: {sensor_kit_id}")
    if kit is None:
        kit = SensorKitRecord(sensor_kit_id=sensor_kit_id, vehicle_id=vehicle_id)
    provenance = _provenance(
        operator=operator,
        reason=reason,
        timestamp=timestamp,
        command=command,
    )
    sensor = PhysicalSensorRecord(
        sensor_id=sensor_id,
        vehicle_id=vehicle_id,
        sensor_kit_id=sensor_kit_id,
        serial=serial,
        model=model,
        firmware=firmware,
        mount=mount,
        frame=frame,
        state="installed" if install else "registered",
        installed_at=provenance.timestamp if install else None,
        install_count=1 if install else 0,
        provenance=provenance,
    )
    event = _event(
        registry,
        event_type="register",
        status="INSTALLED" if install else "REGISTERED",
        operator=operator,
        reason=reason,
        timestamp=timestamp,
        command=command,
        vehicle_id=vehicle_id,
        sensor_kit_id=sensor_kit_id,
        sensor_id=sensor_id,
        payload={
            "entity_type": "sensor",
            "vehicle": vehicle.model_dump(mode="json"),
            "sensor_kit": kit.model_dump(mode="json"),
            "sensor": sensor.model_dump(mode="json"),
        },
    )
    return registry.append_event(event)


def install_sensor(
    root: str | Path,
    *,
    sensor_id: str,
    serial: str,
    model: str,
    firmware: str,
    mount: str,
    frame: str | None = None,
    operator: str = "unknown",
    reason: str = "install or remount physical sensor",
    timestamp: str | None = None,
    command: Sequence[str] | None = None,
) -> LifecycleEvent:
    """Install/remount a sensor only when its physical identity matches."""

    registry = LifecycleRegistry.open(root)
    state = registry.state()
    current = state.sensors.get(sensor_id)
    if current is None:
        raise LifecycleRegistryError(f"unknown sensor: {sensor_id}")
    mismatches = [
        label
        for label, expected, observed in (
            ("serial", current.serial, serial),
            ("model", current.model, model),
            ("firmware", current.firmware, firmware),
            ("mount", current.mount, mount),
        )
        if expected != observed
    ]
    if mismatches:
        raise LifecycleRegistryError(
            f"sensor identity mismatch for {sensor_id}: {', '.join(mismatches)}"
        )
    provenance = _provenance(
        operator=operator,
        reason=reason,
        timestamp=timestamp,
        command=command,
    )
    updated = current.model_copy(
        update={
            "state": "installed",
            "frame": frame or current.frame,
            "installed_at": provenance.timestamp,
            "removed_at": None,
            "remount_count": current.remount_count + (1 if current.state == "removed" else 0),
            "install_count": current.install_count + 1,
            "provenance": provenance,
        }
    )
    event = _event(
        registry,
        event_type="install",
        status="INSTALLED",
        operator=operator,
        reason=reason,
        timestamp=timestamp,
        command=command,
        vehicle_id=current.vehicle_id,
        sensor_kit_id=current.sensor_kit_id,
        sensor_id=sensor_id,
        payload={
            "base_sensor_sequence": current.updated_sequence,
            "sensor": updated.model_dump(mode="json"),
        },
    )
    return registry.append_event(event)


def remove_sensor(
    root: str | Path,
    *,
    sensor_id: str,
    operator: str = "unknown",
    reason: str = "remove physical sensor",
    timestamp: str | None = None,
    command: Sequence[str] | None = None,
) -> LifecycleEvent:
    """Record removal without deleting the physical sensor history."""

    registry = LifecycleRegistry.open(root)
    current = registry.state().sensors.get(sensor_id)
    if current is None:
        raise LifecycleRegistryError(f"unknown sensor: {sensor_id}")
    provenance = _provenance(operator=operator, reason=reason, timestamp=timestamp, command=command)
    updated = current.model_copy(update={"state": "removed", "removed_at": provenance.timestamp})
    event = _event(
        registry,
        event_type="remove",
        status="REMOVED",
        operator=operator,
        reason=reason,
        timestamp=timestamp,
        command=command,
        vehicle_id=current.vehicle_id,
        sensor_kit_id=current.sensor_kit_id,
        sensor_id=sensor_id,
        payload={
            "base_sensor_sequence": current.updated_sequence,
            "sensor": updated.model_dump(mode="json"),
        },
    )
    return registry.append_event(event)


def register_calibration_edge(
    root: str | Path,
    *,
    edge_id: str,
    vehicle_id: str,
    sensor_kit_id: str,
    parent_frame: str,
    child_frame: str,
    incumbent_transform: str | Path | None = None,
    incumbent_result: str | Path | None = None,
    operator: str = "unknown",
    reason: str = "register calibration edge",
    timestamp: str | None = None,
    command: Sequence[str] | None = None,
) -> LifecycleEvent:
    """Register one calibration edge and optional incumbent artifacts."""

    registry = LifecycleRegistry.open(root)
    state = registry.state()
    if edge_id in state.edges:
        raise LifecycleRegistryError(f"calibration edge already registered: {edge_id}")
    if vehicle_id not in state.vehicles:
        raise LifecycleRegistryError(f"unknown vehicle: {vehicle_id}")
    kit = state.sensor_kits.get(sensor_kit_id)
    if kit is None or kit.vehicle_id != vehicle_id:
        raise LifecycleRegistryError(f"unknown or cross-vehicle sensor kit: {sensor_kit_id}")
    artifacts = LifecycleArtifactSet(
        incumbent_transform=_artifact_ref(
            registry.root, incumbent_transform, "transform", "incumbent_transform"
        )
        if incumbent_transform is not None
        else None,
        incumbent_result=_artifact_ref(
            registry.root, incumbent_result, "result", "incumbent_result"
        )
        if incumbent_result is not None
        else None,
    )
    edge = CalibrationEdge(
        edge_id=edge_id,
        vehicle_id=vehicle_id,
        sensor_kit_id=sensor_kit_id,
        parent_frame=parent_frame,
        child_frame=child_frame,
        state="active"
        if artifacts.incumbent_result or artifacts.incumbent_transform
        else "registered",
        incumbent_transform=artifacts.incumbent_transform,
        incumbent_result=artifacts.incumbent_result,
    )
    event = _event(
        registry,
        event_type="register",
        status="REGISTERED",
        operator=operator,
        reason=reason,
        timestamp=timestamp,
        command=command,
        vehicle_id=vehicle_id,
        sensor_kit_id=sensor_kit_id,
        edge_ids=[edge_id],
        artifacts=artifacts,
        payload={"entity_type": "edge", "edge": edge.model_dump(mode="json", exclude_none=False)},
    )
    return registry.append_event(event)


def record_capture(
    root: str | Path,
    *,
    capture_manifest: str | Path,
    operator: str = "unknown",
    reason: str = "record calibration capture",
    timestamp: str | None = None,
    command: Sequence[str] | None = None,
) -> LifecycleEvent:
    """Record a digest-verified capture manifest and exact vehicle identity."""

    registry = LifecycleRegistry.open(root)
    ref = _artifact_ref(registry.root, capture_manifest, "capture_manifest", "capture")
    path = _resolve_source(registry.root, capture_manifest)
    try:
        manifest = load_capture_manifest(path, verify=True)
        verification = verify_capture_manifest_inputs(path)
        if not verification.valid:
            raise LifecycleRegistryError(verification.summary)
        vehicle_id = manifest.vehicle_id
        sensor_kit_id = manifest.sensor_kit_id
        capture_id = manifest.capture_id or manifest.session_id
        manifest_status = manifest.status
    except Exception as exc:
        raise LifecycleRegistryError(f"capture manifest is not valid: {exc}") from exc
    if not vehicle_id or not sensor_kit_id or not capture_id:
        raise LifecycleRegistryError(
            "capture manifest must declare capture, vehicle, and sensor-kit IDs"
        )
    state = registry.state()
    _require_identity(state, vehicle_id, sensor_kit_id)
    for manifest_sensor in manifest.sensors:
        if not manifest_sensor.sensor_id:
            continue
        registered = state.sensors.get(manifest_sensor.sensor_id)
        if registered is None:
            raise LifecycleRegistryError(
                f"capture references unknown physical sensor: {manifest_sensor.sensor_id}"
            )
        if registered.vehicle_id != vehicle_id or registered.sensor_kit_id != sensor_kit_id:
            raise LifecycleRegistryError(
                "capture sensor crosses vehicle or sensor-kit boundary: "
                f"{manifest_sensor.sensor_id}"
            )
        identity_mismatches = [
            label
            for label, expected, observed in (
                ("serial", registered.serial, manifest_sensor.serial),
                ("model", registered.model, manifest_sensor.model),
                ("firmware", registered.firmware, manifest_sensor.firmware),
                ("mount", registered.mount, manifest_sensor.mount_id),
            )
            if observed is not None and expected != observed
        ]
        if identity_mismatches:
            raise LifecycleRegistryError(
                f"capture physical sensor identity mismatch for {manifest_sensor.sensor_id}: "
                + ", ".join(identity_mismatches)
            )
    event = _event(
        registry,
        event_type="capture",
        status="CAPTURED" if manifest_status == "ready" else "BLOCKED",
        admission="NOT_APPLICABLE" if manifest_status == "ready" else "BLOCKED",
        operator=operator,
        reason=reason,
        timestamp=timestamp,
        command=command,
        vehicle_id=vehicle_id,
        sensor_kit_id=sensor_kit_id,
        artifacts=LifecycleArtifactSet(capture=ref),
        payload={"capture_id": capture_id, "capture_status": manifest_status},
    )
    return registry.append_event(event)


def evaluate_lifecycle(
    root: str | Path,
    *,
    edge_ids: Sequence[str] | None = None,
    edge_id: str | None = None,
    capture_manifest: str | Path,
    candidate_result: str | Path | None = None,
    candidate_transform: str | Path | None = None,
    candidate_pilot: str | Path | None = None,
    evidence: str | Path | None = None,
    assessment: str | Path | None = None,
    promotion: str | Path | None = None,
    smoke: str | Path | None = None,
    policy: str | Path | None = None,
    output: str | Path | None = None,
    operator: str = "unknown",
    reason: str = "evaluate calibration candidate",
    timestamp: str | None = None,
    command: Sequence[str] | None = None,
) -> LifecycleEvaluationArtifact:
    """Evaluate candidate evidence and append evaluate/hold/reject events.

    Every input is copied into an exact digest reference.  Missing, malformed,
    weak, or ``INCONCLUSIVE`` evidence can only produce a non-admissible
    decision; this function never changes an edge incumbent.
    """

    registry = LifecycleRegistry.open(root)
    ids = list(edge_ids or ([] if edge_id is None else [edge_id]))
    if not ids:
        raise LifecycleRegistryError("at least one calibration edge must be declared")
    if len(set(ids)) != len(ids):
        raise LifecycleRegistryError("duplicate calibration edge IDs")
    state = registry.state()
    edges = [_edge_for_state(state, item) for item in ids]
    if (
        len({item.vehicle_id for item in edges}) != 1
        or len({item.sensor_kit_id for item in edges}) != 1
    ):
        raise LifecycleRegistryError("evaluation cannot span vehicles or sensor kits")
    vehicle_id = edges[0].vehicle_id
    sensor_kit_id = edges[0].sensor_kit_id
    artifacts = LifecycleArtifactSet(
        capture=_artifact_ref(registry.root, capture_manifest, "capture_manifest", "capture"),
        candidate_result=_artifact_ref(
            registry.root, candidate_result, "result", "candidate_result"
        )
        if candidate_result is not None
        else None,
        candidate_transform=_artifact_ref(
            registry.root, candidate_transform, "transform", "candidate_transform"
        )
        if candidate_transform is not None
        else None,
        candidate_pilot=_artifact_ref(registry.root, candidate_pilot, "pilot", "candidate_pilot")
        if candidate_pilot is not None
        else None,
        evidence=_artifact_ref(registry.root, evidence, "evidence", "evidence")
        if evidence is not None
        else None,
        assessment=_artifact_ref(registry.root, assessment, "assessment", "assessment")
        if assessment is not None
        else None,
        promotion=_artifact_ref(registry.root, promotion, "promotion", "promotion")
        if promotion is not None
        else None,
        smoke=_artifact_ref(registry.root, smoke, "smoke", "smoke") if smoke is not None else None,
        policy=_artifact_ref(registry.root, policy, "policy", "policy")
        if policy is not None
        else None,
    )
    status, admission, gates, decision_reason = _evaluate_inputs(
        registry.root,
        edges,
        artifacts,
        candidate_result=candidate_result,
        capture_manifest=capture_manifest,
        evidence=evidence,
        assessment=assessment,
        promotion=promotion,
        smoke=smoke,
    )
    provenance = _provenance(operator=operator, reason=reason, timestamp=timestamp, command=command)
    evaluation = LifecycleEvaluationArtifact(
        evaluation_id=_evaluation_id(
            registry.manifest().registry_id, ids, artifacts, provenance.timestamp
        ),
        registry_id=registry.manifest().registry_id,
        vehicle_id=vehicle_id,
        sensor_kit_id=sensor_kit_id,
        edge_ids=ids,
        status=status,
        admission=admission,
        reason=decision_reason,
        gates=gates,
        artifacts=artifacts,
        provenance=provenance,
    ).with_artifact_digest()
    event_artifacts = artifacts
    if output is not None:
        evaluation.save(output)
        # Keep the evaluation file's self-digest independent of a reference
        # to itself.  The immutable event still records that exact file ref.
        event_artifacts = artifacts.model_copy(
            update={
                "evaluation": _artifact_ref(registry.root, output, "evaluation", "evaluation"),
            }
        )
    event_type: LifecycleEventType = "evaluate"
    event_status: LifecycleStatus = status
    if admission == "HOLD":
        event_type, event_status = "hold", "WARN" if status == "PASS" else status
    elif admission in {"DO_NOT_ADOPT", "BLOCKED"}:
        event_type = "reject"
    updates = {
        item.edge_id: _candidate_edge(
            item, event_artifacts, status, admission, sequence=-1, timestamp=provenance.timestamp
        )
        for item in edges
    }
    event = _event(
        registry,
        event_type=event_type,
        status=event_status,
        admission=admission,
        operator=operator,
        reason=decision_reason,
        timestamp=timestamp,
        command=command,
        vehicle_id=vehicle_id,
        sensor_kit_id=sensor_kit_id,
        edge_ids=ids,
        artifacts=event_artifacts,
        payload={
            "evaluation": evaluation.model_dump(mode="json", exclude_none=False),
            "edge_updates": {
                key: value.model_dump(mode="json", exclude_none=False)
                for key, value in updates.items()
            },
            "base_edge_sequences": {
                key: edge.updated_sequence for key, edge in zip(ids, edges, strict=True)
            },
            "incumbent_changed": False,
        },
    )
    registry.append_event(event)
    # The event is immutable; a second event is not needed just to materialise
    # the sequence, and replay uses the event sequence as its authoritative value.
    return evaluation.model_copy(
        update={
            "artifacts": evaluation.artifacts,
            "provenance": evaluation.provenance,
        }
    )


def promote_lifecycle(
    root: str | Path,
    *,
    edge_ids: Sequence[str] | None = None,
    edge_id: str | None = None,
    evaluation: str | Path | LifecycleEvaluationArtifact | None = None,
    promotion: str | Path | None = None,
    smoke: str | Path | None = None,
    operator: str = "unknown",
    reason: str = "promote evaluated calibration candidate",
    timestamp: str | None = None,
    command: Sequence[str] | None = None,
) -> LifecycleEvent:
    """Promote only a PASS/ADOPT evaluation with exact source artifacts."""

    registry = LifecycleRegistry.open(root)
    ids = list(edge_ids or ([] if edge_id is None else [edge_id]))
    if not ids:
        raise LifecycleRegistryError("at least one calibration edge must be declared")
    state = registry.state()
    edges = [_edge_for_state(state, item) for item in ids]
    if (
        len({item.vehicle_id for item in edges}) != 1
        or len({item.sensor_kit_id for item in edges}) != 1
    ):
        raise LifecycleRegistryError("promotion cannot span vehicles or sensor kits")
    evaluation_ref: LifecycleArtifactRef | None = None
    if isinstance(evaluation, (str, Path)):
        evaluation_ref = _artifact_ref(registry.root, evaluation, "evaluation", "evaluation")
    evaluation_artifact = _load_evaluation(registry.root, evaluation, ids, registry)
    if (
        evaluation_artifact.vehicle_id != edges[0].vehicle_id
        or evaluation_artifact.sensor_kit_id != edges[0].sensor_kit_id
    ):
        raise LifecycleRegistryError("evaluation vehicle/sensor-kit identity does not match edges")
    if evaluation_artifact.status != "PASS" or evaluation_artifact.admission != "ADOPT":
        raise LifecycleRegistryError(
            "promotion requires evaluation status PASS and admission ADOPT"
        )
    artifacts = evaluation_artifact.artifacts
    if promotion is not None:
        promotion_ref = _artifact_ref(registry.root, promotion, "promotion", "promotion")
        if artifacts.promotion is not None and promotion_ref.sha256 != artifacts.promotion.sha256:
            raise LifecycleRegistryError(
                "promotion artifact differs from evaluated promotion digest"
            )
        artifacts = artifacts.model_copy(update={"promotion": promotion_ref})
    if smoke is not None:
        smoke_ref = _artifact_ref(registry.root, smoke, "smoke", "smoke")
        if artifacts.smoke is not None and smoke_ref.sha256 != artifacts.smoke.sha256:
            raise LifecycleRegistryError("smoke artifact differs from evaluated smoke digest")
        artifacts = artifacts.model_copy(update={"smoke": smoke_ref})
    if evaluation_ref is not None:
        artifacts = artifacts.model_copy(update={"evaluation": evaluation_ref})
    _require_promotable_artifacts(registry.root, artifacts)
    _verify_promotion_and_smoke(registry.root, artifacts)
    updates: dict[str, CalibrationEdge] = {}
    for edge in edges:
        updates[edge.edge_id] = edge.model_copy(
            update={
                "state": "active",
                "incumbent_transform": artifacts.candidate_transform or edge.incumbent_transform,
                "incumbent_result": artifacts.candidate_result or edge.incumbent_result,
                "candidate_transform": artifacts.candidate_transform,
                "candidate_result": artifacts.candidate_result,
                "candidate_pilot": artifacts.candidate_pilot,
                "promotion": artifacts.promotion,
                "smoke": artifacts.smoke,
                "capture": artifacts.capture,
                "evidence": artifacts.evidence,
                "assessment": artifacts.assessment,
                "policy": artifacts.policy,
                "last_evaluation_status": "PASS",
                "last_admission": "ADOPT",
                "updated_at": _timestamp(timestamp),
            }
        )
    event = _event(
        registry,
        event_type="promote",
        status="PROMOTED",
        admission="ADOPT",
        operator=operator,
        reason=reason,
        timestamp=timestamp,
        command=command,
        vehicle_id=edges[0].vehicle_id,
        sensor_kit_id=edges[0].sensor_kit_id,
        edge_ids=ids,
        artifacts=artifacts,
        payload={
            "evaluation": evaluation_artifact.model_dump(mode="json", exclude_none=False),
            "incumbent_before": {
                key: edge.model_dump(mode="json", exclude_none=False)
                for key, edge in zip(ids, edges, strict=True)
            },
            "edge_updates": {
                key: value.model_dump(mode="json", exclude_none=False)
                for key, value in updates.items()
            },
            "base_edge_sequences": {
                key: edge.updated_sequence for key, edge in zip(ids, edges, strict=True)
            },
            "declared_edge_ids": ids,
            "incumbent_changed": True,
        },
    )
    return registry.append_event(event)


def rollback_lifecycle(
    root: str | Path,
    *,
    edge_id: str,
    target_sequence: int | None = None,
    target_event_sha256: str | None = None,
    promotion: str | Path | None = None,
    rollback_adapter: LifecycleRollbackAdapter | None = None,
    operator: str = "unknown",
    reason: str = "rollback calibration edge to prior incumbent",
    timestamp: str | None = None,
    command: Sequence[str] | None = None,
) -> LifecycleEvent:
    """Append a guarded rollback event; no historical event is deleted."""

    registry = LifecycleRegistry.open(root)
    state = registry.state()
    edge = _edge_for_state(state, edge_id)
    events = registry.events()
    candidates = [
        event for event in events if event.event_type == "promote" and edge_id in event.edge_ids
    ]
    if target_sequence is not None:
        candidates = [event for event in candidates if event.sequence == target_sequence]
    if target_event_sha256 is not None:
        candidates = [event for event in candidates if event.event_sha256 == target_event_sha256]
    if not candidates:
        raise LifecycleRegistryError(
            "rollback target must reference a prior promote event for this edge"
        )
    target_event = candidates[-1]
    target_payload = cast(dict[str, Any], target_event.payload.get("incumbent_before", {}))
    raw_target = target_payload.get(edge_id)
    if not isinstance(raw_target, dict):
        raise LifecycleRegistryError("rollback target does not contain the declared edge incumbent")
    target = CalibrationEdge.model_validate(raw_target)
    # A rollback that points at a deleted or modified incumbent is not a safe
    # recovery target.  Verify both sides before invoking an optional external
    # adapter, so a downstream package cannot be changed from unverifiable
    # source bytes.
    for ref in (
        edge.incumbent_transform,
        edge.incumbent_result,
        target.incumbent_transform,
        target.incumbent_result,
    ):
        if ref is not None:
            _verify_artifact_ref(registry.root, ref)
    if promotion is not None:
        promotion_ref = _artifact_ref(registry.root, promotion, "promotion", "rollback_promotion")
        _verify_promotion_rollback(registry.root, promotion_ref)
    if rollback_adapter is not None:
        try:
            rollback_adapter(edge, target)
        except Exception as exc:
            raise LifecycleRegistryError(f"guarded rollback adapter failed: {exc}") from exc
    provenance = _provenance(operator=operator, reason=reason, timestamp=timestamp, command=command)
    updated = edge.model_copy(
        update={
            "state": "active"
            if target.incumbent_result or target.incumbent_transform
            else "registered",
            "incumbent_transform": target.incumbent_transform,
            "incumbent_result": target.incumbent_result,
            "candidate_transform": edge.candidate_transform,
            "candidate_result": edge.candidate_result,
            "last_evaluation_status": "PASS",
            "last_admission": "ROLLBACK",
            "updated_at": provenance.timestamp,
        }
    )
    artifacts = LifecycleArtifactSet(
        incumbent_transform=edge.incumbent_transform,
        incumbent_result=edge.incumbent_result,
        candidate_transform=target.incumbent_transform,
        candidate_result=target.incumbent_result,
        promotion=_artifact_ref(registry.root, promotion, "promotion", "rollback_promotion")
        if promotion is not None
        else None,
    )
    event = _event(
        registry,
        event_type="rollback",
        status="ROLLED_BACK",
        admission="ROLLBACK",
        operator=operator,
        reason=reason,
        timestamp=timestamp,
        command=command,
        vehicle_id=edge.vehicle_id,
        sensor_kit_id=edge.sensor_kit_id,
        edge_ids=[edge_id],
        artifacts=artifacts,
        payload={
            "target_event_sequence": target_event.sequence,
            "target_event_sha256": target_event.event_sha256,
            "incumbent_before": edge.model_dump(mode="json", exclude_none=False),
            "rollback_target": target.model_dump(mode="json", exclude_none=False),
            "edge_updates": {edge_id: updated.model_dump(mode="json", exclude_none=False)},
            "base_edge_sequences": {edge_id: edge.updated_sequence},
            "incumbent_changed": edge != updated,
        },
    )
    return registry.append_event(event)


def lifecycle_registry_json_schema() -> dict[str, Any]:
    """Return the generated schema for the registry manifest."""

    return RegistryManifest.model_json_schema()


def lifecycle_event_json_schema() -> dict[str, Any]:
    """Return the generated schema for lifecycle events."""

    return LifecycleEvent.model_json_schema()


def lifecycle_evaluation_json_schema() -> dict[str, Any]:
    """Return the generated schema for evaluation artifacts."""

    return LifecycleEvaluationArtifact.model_json_schema()


def lifecycle_registry_state_json_schema() -> dict[str, Any]:
    """Return the generated schema for the materialised registry projection."""

    return LifecycleRegistryState.model_json_schema()


def lifecycle_head_json_schema() -> dict[str, Any]:
    """Return the generated schema for the registry checkpoint."""

    return RegistryHead.model_json_schema()


def lifecycle_registry_verification_json_schema() -> dict[str, Any]:
    """Return the generated schema for verification reports."""

    return RegistryVerificationArtifact.model_json_schema()


def lifecycle_status_json_schema() -> dict[str, Any]:
    """Return the generated schema for status summaries."""

    return LifecycleRegistryStatus.model_json_schema()


def _event(
    registry: LifecycleRegistry,
    *,
    event_type: LifecycleEventType,
    status: LifecycleStatus,
    operator: str,
    reason: str,
    timestamp: str | None,
    command: Sequence[str] | None,
    admission: LifecycleAdmission = "NOT_APPLICABLE",
    vehicle_id: str | None = None,
    sensor_kit_id: str | None = None,
    sensor_id: str | None = None,
    edge_ids: Sequence[str] = (),
    artifacts: LifecycleArtifactSet | None = None,
    payload: Mapping[str, Any] | None = None,
) -> LifecycleEvent:
    """Build an event before append assigns sequence and chain hashes."""

    return LifecycleEvent(
        registry_id=registry.manifest().registry_id,
        sequence=0,
        event_type=event_type,
        status=status,
        admission=admission,
        vehicle_id=vehicle_id,
        sensor_kit_id=sensor_kit_id,
        sensor_id=sensor_id,
        edge_ids=list(edge_ids),
        provenance=_provenance(
            operator=operator,
            reason=reason,
            timestamp=timestamp,
            command=command,
        ),
        artifacts=artifacts or LifecycleArtifactSet(),
        payload=dict(payload or {}),
    )


def _provenance(
    *,
    operator: str,
    reason: str,
    timestamp: str | None = None,
    command: Sequence[str] | None = None,
) -> LifecycleProvenance:
    return LifecycleProvenance(
        operator=operator,
        reason=reason,
        timestamp=timestamp or _timestamp(None),
        git_commit=git_commit(),
        command=list(command or []),
    )


def _timestamp(value: str | None) -> str:
    return value or datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _event_digest(event: LifecycleEvent) -> str:
    payload = event.model_dump(mode="json", exclude_none=False)
    payload["event_sha256"] = ZERO_SHA256
    return _canonical_sha256(payload)


def _state_digest(state: LifecycleRegistryState) -> str:
    return _canonical_sha256(state.model_dump(mode="json", exclude_none=False))


def _resolve_source(root: Path, path: str | Path) -> Path:
    raw = Path(path)
    root_resolved = root.resolve()
    if raw.is_absolute():
        resolved = raw.resolve()
        # Absolute paths are supported for external evidence, but an absolute
        # path lexically inside the registry must not escape through a symlink.
        lexical = Path(os.path.abspath(raw))
        try:
            lexical.relative_to(root_resolved)
        except ValueError:
            pass
        else:
            try:
                resolved.relative_to(root_resolved)
            except ValueError as exc:
                raise LifecycleRegistryError(f"source path escapes registry root: {path}") from exc
    else:
        # Relative source paths are registry-root relative and may not escape
        # the exact root.  External inputs can be passed as absolute paths.
        if any(part == ".." for part in raw.parts):
            raise LifecycleRegistryError(f"source path escapes registry root: {path}")
        resolved = (root_resolved / raw).resolve()
        try:
            resolved.relative_to(root_resolved)
        except ValueError as exc:
            raise LifecycleRegistryError(f"source path escapes registry root: {path}") from exc
    if not resolved.is_file():
        raise LifecycleRegistryError(f"source artifact does not exist: {resolved}")
    return resolved


def _artifact_ref(
    root: Path, path: str | Path | None, kind: str, role: str
) -> LifecycleArtifactRef:
    if path is None:
        raise LifecycleRegistryError(f"{role} artifact path is required")
    resolved = _resolve_source(root, path)
    digest = sha256_path(resolved)
    if digest is None:
        raise LifecycleRegistryError(f"cannot digest source artifact: {resolved}")
    schema_version: str | None = None
    try:
        payload = read_mapping(resolved)
        value = payload.get("schema_version")
        schema_version = value if isinstance(value, str) else None
    except Exception:
        pass
    return LifecycleArtifactRef(
        kind=kind,
        path=_portable_path(root, resolved),
        sha256=digest,
        schema_version=schema_version,
        size_bytes=resolved.stat().st_size,
    )


def _portable_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _verify_artifact_ref(root: Path, ref: LifecycleArtifactRef) -> None:
    path = _resolve_source(root, ref.path)
    observed = sha256_path(path)
    if observed != ref.sha256:
        raise LifecycleRegistryVerificationError(
            f"artifact digest mismatch for {ref.kind}: {path} "
            f"declared={ref.sha256}, observed={observed}"
        )
    if ref.size_bytes is not None and path.stat().st_size != ref.size_bytes:
        raise LifecycleRegistryVerificationError(f"artifact size mismatch for {ref.kind}: {path}")


def _require_identity(state: LifecycleRegistryState, vehicle_id: str, sensor_kit_id: str) -> None:
    if vehicle_id not in state.vehicles:
        raise LifecycleRegistryError(f"unknown vehicle: {vehicle_id}")
    kit = state.sensor_kits.get(sensor_kit_id)
    if kit is None or kit.vehicle_id != vehicle_id:
        raise LifecycleRegistryError(f"unknown or cross-vehicle sensor kit: {sensor_kit_id}")


def _edge_for_state(state: LifecycleRegistryState, edge_id: str) -> CalibrationEdge:
    try:
        return state.edges[edge_id]
    except KeyError as exc:
        raise LifecycleRegistryError(f"unknown calibration edge: {edge_id}") from exc


def _candidate_edge(
    edge: CalibrationEdge,
    artifacts: LifecycleArtifactSet,
    status: LifecycleStatus,
    admission: LifecycleAdmission,
    *,
    sequence: int,
    timestamp: str,
) -> CalibrationEdge:
    return edge.model_copy(
        update={
            "state": "candidate",
            "candidate_transform": artifacts.candidate_transform,
            "candidate_result": artifacts.candidate_result,
            "candidate_pilot": artifacts.candidate_pilot,
            "promotion": artifacts.promotion,
            "smoke": artifacts.smoke,
            "capture": artifacts.capture,
            "evidence": artifacts.evidence,
            "assessment": artifacts.assessment,
            "policy": artifacts.policy,
            "last_evaluation_status": status,
            "last_admission": admission,
            "updated_sequence": sequence,
            "updated_at": timestamp,
        }
    )


def _evaluate_inputs(
    root: Path,
    edges: Sequence[CalibrationEdge],
    artifacts: LifecycleArtifactSet,
    *,
    candidate_result: str | Path | None,
    capture_manifest: str | Path,
    evidence: str | Path | None,
    assessment: str | Path | None,
    promotion: str | Path | None,
    smoke: str | Path | None,
) -> tuple[
    Literal["PASS", "WARN", "FAIL", "INCONCLUSIVE", "BLOCKED"],
    LifecycleAdmission,
    dict[str, Literal["PASS", "WARN", "FAIL", "INCONCLUSIVE", "BLOCKED"]],
    str,
]:
    gates: dict[str, Literal["PASS", "WARN", "FAIL", "INCONCLUSIVE", "BLOCKED"]] = {}
    try:
        manifest_path = _resolve_source(root, capture_manifest)
        manifest = load_capture_manifest(manifest_path, verify=True)
        vehicle_id = manifest.vehicle_id
        sensor_kit_id = manifest.sensor_kit_id
        if not vehicle_id or not sensor_kit_id:
            raise LifecycleRegistryError("capture identity is incomplete")
        if vehicle_id != edges[0].vehicle_id or sensor_kit_id != edges[0].sensor_kit_id:
            raise LifecycleRegistryError("capture vehicle/sensor-kit identity does not match edge")
        verification = verify_capture_manifest_inputs(manifest_path)
        if not verification.valid:
            gates["capture"] = "BLOCKED"
            return "BLOCKED", "BLOCKED", gates, verification.summary
        capture_status = str(manifest.status).lower()
        gates["capture"] = (
            "PASS"
            if capture_status == "ready"
            else "WARN"
            if capture_status == "warn"
            else "BLOCKED"
        )
        if gates["capture"] != "PASS":
            return (
                "BLOCKED",
                "BLOCKED",
                gates,
                f"capture manifest is not READY (status={manifest.status})",
            )
    except Exception as exc:
        gates["capture"] = "BLOCKED"
        return "BLOCKED", "BLOCKED", gates, f"capture readiness is unavailable: {exc}"
    if (
        candidate_result is None
        and artifacts.candidate_transform is None
        and artifacts.candidate_pilot is None
    ):
        gates["candidate"] = "INCONCLUSIVE"
        return (
            "INCONCLUSIVE",
            "HOLD",
            gates,
            "candidate result, transform, or pilot artifact is missing",
        )
    if candidate_result is not None:
        try:
            payload = read_mapping(_resolve_source(root, candidate_result))
            schema_version = payload.get("schema_version")
            if not isinstance(schema_version, str):
                raise ValueError("candidate result has no schema_version")
            # A native Calibrex result is a public schema, not merely a
            # mapping carrying a plausible version string.  Keep the generic
            # adapter path for provider-specific result schemas, while
            # strictly validating the core result contract when recognized.
            if schema_version == "slac.result/v0.1":
                load_result(_resolve_source(root, candidate_result))
            gates["candidate"] = "PASS"
        except Exception as exc:
            gates["candidate"] = "BLOCKED"
            return "BLOCKED", "BLOCKED", gates, f"candidate result is invalid: {exc}"
    else:
        gates["candidate"] = "PASS"
    if evidence is None:
        gates["evidence"] = "INCONCLUSIVE"
        return "INCONCLUSIVE", "HOLD", gates, "independent evidence artifact is missing"
    gates["evidence"] = _artifact_status(root, evidence, default="INCONCLUSIVE")
    if gates["evidence"] != "PASS":
        return (
            gates["evidence"],
            "HOLD" if gates["evidence"] in {"WARN", "INCONCLUSIVE"} else "DO_NOT_ADOPT",
            gates,
            "evidence is not PASS",
        )
    if assessment is None:
        gates["assessment"] = "INCONCLUSIVE"
        return "INCONCLUSIVE", "HOLD", gates, "assessment artifact is missing"
    gates["assessment"] = _artifact_status(root, assessment, default="INCONCLUSIVE")
    if gates["assessment"] != "PASS":
        return (
            gates["assessment"],
            "HOLD" if gates["assessment"] in {"WARN", "INCONCLUSIVE"} else "DO_NOT_ADOPT",
            gates,
            "assessment is not PASS",
        )
    if promotion is not None:
        gates["promotion"] = _artifact_status(root, promotion, default="INCONCLUSIVE")
        if gates["promotion"] != "PASS":
            return gates["promotion"], "DO_NOT_ADOPT", gates, "promotion artifact is not PASS/ADOPT"
    else:
        gates["promotion"] = "INCONCLUSIVE"
    if smoke is not None:
        gates["smoke"] = _artifact_status(root, smoke, default="INCONCLUSIVE")
        if gates["smoke"] != "PASS":
            return (
                gates["smoke"],
                "DO_NOT_ADOPT",
                gates,
                "Autoware smoke artifact is not PASS/admissible",
            )
    else:
        gates["smoke"] = "INCONCLUSIVE"
    if promotion is None or smoke is None:
        return "INCONCLUSIVE", "HOLD", gates, "promotion and smoke evidence are required for ADOPT"
    return (
        "PASS",
        "ADOPT",
        gates,
        "capture, candidate, evidence, assessment, promotion, and smoke gates passed",
    )


def _artifact_status(
    root: Path,
    path: str | Path,
    *,
    default: Literal["PASS", "WARN", "FAIL", "INCONCLUSIVE", "BLOCKED"],
) -> Literal["PASS", "WARN", "FAIL", "INCONCLUSIVE", "BLOCKED"]:
    try:
        payload = read_mapping(_resolve_source(root, path))
    except Exception:
        return "BLOCKED"
    value = payload.get("status")
    if isinstance(value, str):
        normalized = value.upper()
        if normalized in {"PASS", "WARN", "FAIL", "INCONCLUSIVE", "BLOCKED"}:
            return cast(Literal["PASS", "WARN", "FAIL", "INCONCLUSIVE", "BLOCKED"], normalized)
        if normalized in {"READY", "SUCCESS", "CONVERGED", "ADOPT"}:
            return "PASS"
    quality = payload.get("quality")
    if isinstance(quality, Mapping):
        grade = quality.get("grade")
        if isinstance(grade, str) and grade.lower() == "pass":
            return "PASS"
    decision = payload.get("decision")
    if isinstance(decision, str) and decision.upper() == "ADOPT":
        return "PASS"
    return default


def _require_promotable_artifacts(root: Path, artifacts: LifecycleArtifactSet) -> None:
    required = {
        "capture": artifacts.capture,
        "candidate_result": artifacts.candidate_result,
        "evidence": artifacts.evidence,
        "assessment": artifacts.assessment,
        "promotion": artifacts.promotion,
        "smoke": artifacts.smoke,
    }
    missing = [name for name, ref in required.items() if ref is None]
    if missing:
        raise LifecycleRegistryError("promotion is missing exact artifacts: " + ", ".join(missing))
    for ref in artifacts.refs():
        _verify_artifact_ref(root, ref)


def _verify_promotion_and_smoke(root: Path, artifacts: LifecycleArtifactSet) -> None:
    assert artifacts.promotion is not None
    assert artifacts.smoke is not None
    promotion_path = _resolve_source(root, artifacts.promotion.path)
    promotion_payload = read_mapping(promotion_path)
    status = str(promotion_payload.get("status", "")).upper()
    decision = str(promotion_payload.get("decision", "")).upper()
    if status != "PASS" or decision != "ADOPT":
        raise LifecycleRegistryError("promotion requires exact status PASS and decision ADOPT")
    promotion_plan: Any | None = None
    if promotion_payload.get("schema_version") == "slac.autoware_promotion/v0.1":
        try:
            # Import at the adapter boundary only when an Autoware artifact is
            # actually supplied.  The registry core remains ROS-independent,
            # while native promotion self-digests and strict models are still
            # honored for interoperability.
            from calibrex.export.autoware_promotion import load_autoware_promotion

            promotion_plan = load_autoware_promotion(promotion_path)
        except Exception as exc:
            raise LifecycleRegistryError(f"invalid Autoware promotion artifact: {exc}") from exc
        if promotion_plan.status != "PASS" or promotion_plan.decision != "ADOPT":
            raise LifecycleRegistryError("promotion requires exact status PASS and decision ADOPT")
    smoke_path = _resolve_source(root, artifacts.smoke.path)
    smoke_payload = read_mapping(smoke_path)
    if str(smoke_payload.get("status", "")).upper() != "PASS":
        raise LifecycleRegistryError("smoke evidence must have status PASS")
    admission = str(smoke_payload.get("admission_label", "")).lower()
    if admission not in {"production-admissible", "developer-only"}:
        raise LifecycleRegistryError("smoke evidence is not admissible")
    if smoke_payload.get("schema_version") == "slac.autoware_smoke/v0.1":
        try:
            from calibrex.export.autoware_smoke import load_autoware_smoke, verify_autoware_smoke

            smoke = load_autoware_smoke(smoke_path)
            verify_autoware_smoke(smoke, promotion_plan, require_pass=False)
        except Exception as exc:
            raise LifecycleRegistryError(f"invalid Autoware smoke artifact: {exc}") from exc


def _verify_promotion_rollback(root: Path, ref: LifecycleArtifactRef) -> None:
    _verify_artifact_ref(root, ref)
    payload = read_mapping(_resolve_source(root, ref.path))
    if str(payload.get("status", "")).upper() not in {"PASS", "APPLIED"}:
        raise LifecycleRegistryError(
            "guarded Autoware rollback requires a PASS/APPLIED promotion artifact"
        )


def _load_evaluation(
    root: Path,
    value: str | Path | LifecycleEvaluationArtifact | None,
    ids: Sequence[str],
    registry: LifecycleRegistry,
) -> LifecycleEvaluationArtifact:
    if isinstance(value, LifecycleEvaluationArtifact):
        artifact = value
        try:
            artifact.verify_artifact_digest()
        except Exception as exc:
            raise LifecycleRegistryError(f"invalid evaluation artifact: {exc}") from exc
    elif value is not None:
        path = _resolve_source(root, value)
        try:
            artifact = LifecycleEvaluationArtifact.model_validate(read_mapping(path))
            artifact.verify_artifact_digest()
        except Exception as exc:
            raise LifecycleRegistryError(f"invalid evaluation artifact: {exc}") from exc
    else:
        matching = [
            event
            for event in registry.events()
            if event.event_type in {"evaluate", "hold", "reject"}
            and set(ids).issubset(event.edge_ids)
        ]
        if not matching:
            raise LifecycleRegistryError("no prior evaluation event for promotion")
        try:
            artifact = LifecycleEvaluationArtifact.model_validate(
                matching[-1].payload["evaluation"]
            )
            artifact.verify_artifact_digest()
        except Exception as exc:
            raise LifecycleRegistryError(f"prior evaluation event is malformed: {exc}") from exc
    if artifact.registry_id != registry.manifest().registry_id or artifact.edge_ids != list(ids):
        raise LifecycleRegistryError("evaluation identity does not match declared promotion edges")
    return artifact


def _evaluation_id(
    registry_id: str, edge_ids: Sequence[str], artifacts: LifecycleArtifactSet, timestamp: str
) -> str:
    digest = _canonical_sha256(
        {
            "registry_id": registry_id,
            "edge_ids": list(edge_ids),
            "artifacts": artifacts.model_dump(mode="json", exclude_none=False),
            "timestamp": timestamp,
        }
    )
    return f"evaluation-{digest[:16]}"


def _apply_event(state: LifecycleRegistryState, event: LifecycleEvent) -> None:
    """Apply one event while enforcing immutable identity and stale-write guards.

    The event hash chain proves that a log was not changed accidentally, but it
    does not by itself prove that a newly appended event is semantically safe.
    These checks make replay fail closed for duplicate registrations, cross-
    vehicle references, and commands prepared from an old projection.  Older
    hand-authored v0.2 events without the optional ``base_*`` fields remain
    readable; new public operations always emit the guards.
    """

    payload = event.payload
    entity_type = payload.get("entity_type")

    def _fail(message: str) -> NoReturn:
        raise LifecycleRegistryVerificationError(
            f"event {event.sequence} semantic validation failed: {message}"
        )

    def _mapping(name: str, *, required: bool = False) -> Mapping[str, Any] | None:
        value = payload.get(name)
        if value is None and not required:
            return None
        if not isinstance(value, Mapping):
            _fail(f"payload.{name} must be an object")
        return cast(Mapping[str, Any], value)

    def _same(left: StrictModel, right: StrictModel) -> bool:
        return _canonical_json(left.model_dump(mode="json", exclude_none=False)) == _canonical_json(
            right.model_dump(mode="json", exclude_none=False)
        )

    # Sensor registration may carry the vehicle and kit records needed to
    # bootstrap an empty registry. Existing identities are immutable: a
    # concurrent or forged event may repeat an exact context, but may not
    # silently replace metadata or move it to another vehicle.
    raw_vehicle = _mapping("vehicle")
    vehicle: VehicleRecord | None = None
    if raw_vehicle is not None:
        try:
            vehicle = VehicleRecord.model_validate(raw_vehicle)
        except Exception as exc:
            _fail(f"invalid vehicle payload: {exc}")
        if event.vehicle_id is not None and event.vehicle_id != vehicle.vehicle_id:
            _fail("event vehicle_id does not match payload vehicle")
        existing_vehicle = state.vehicles.get(vehicle.vehicle_id)
        if existing_vehicle is None:
            state.vehicles[vehicle.vehicle_id] = vehicle
        elif not _same(existing_vehicle, vehicle):
            _fail(f"vehicle identity/metadata conflict: {vehicle.vehicle_id}")
        elif entity_type == "vehicle":
            _fail(f"vehicle already registered: {vehicle.vehicle_id}")

    raw_kit = _mapping("sensor_kit")
    kit: SensorKitRecord | None = None
    if raw_kit is not None:
        try:
            kit = SensorKitRecord.model_validate(raw_kit)
        except Exception as exc:
            _fail(f"invalid sensor_kit payload: {exc}")
        if event.sensor_kit_id is not None and event.sensor_kit_id != kit.sensor_kit_id:
            _fail("event sensor_kit_id does not match payload sensor_kit")
        known_vehicle = state.vehicles.get(kit.vehicle_id)
        if known_vehicle is None:
            _fail(f"sensor kit references unknown vehicle: {kit.vehicle_id}")
        existing_kit = state.sensor_kits.get(kit.sensor_kit_id)
        if existing_kit is None:
            state.sensor_kits[kit.sensor_kit_id] = kit
        elif not _same(existing_kit, kit):
            _fail(f"sensor-kit identity/metadata conflict: {kit.sensor_kit_id}")
        elif entity_type == "sensor_kit":
            _fail(f"sensor kit already registered: {kit.sensor_kit_id}")

    if entity_type == "vehicle":
        if vehicle is None:
            _fail("vehicle registration is missing payload.vehicle")
    elif entity_type == "sensor_kit":
        if kit is None:
            _fail("sensor-kit registration is missing payload.sensor_kit")
    elif entity_type == "sensor":
        raw_sensor = _mapping("sensor", required=True)
        try:
            sensor = PhysicalSensorRecord.model_validate(raw_sensor)
        except Exception as exc:
            _fail(f"invalid sensor payload: {exc}")
        if event.sensor_id is not None and event.sensor_id != sensor.sensor_id:
            _fail("event sensor_id does not match payload sensor")
        if state.vehicles.get(sensor.vehicle_id) is None:
            _fail(f"sensor references unknown vehicle: {sensor.vehicle_id}")
        sensor_kit = state.sensor_kits.get(sensor.sensor_kit_id)
        if sensor_kit is None or sensor_kit.vehicle_id != sensor.vehicle_id:
            _fail(f"sensor crosses vehicle/sensor-kit boundary: {sensor.sensor_id}")
        if sensor.sensor_id in state.sensors:
            _fail(f"sensor already registered: {sensor.sensor_id}")
        state.sensors[sensor.sensor_id] = sensor.model_copy(
            update={"updated_sequence": event.sequence}
        )
    elif entity_type == "edge":
        raw_edge = _mapping("edge", required=True)
        try:
            edge = CalibrationEdge.model_validate(raw_edge)
        except Exception as exc:
            _fail(f"invalid edge payload: {exc}")
        if event.edge_ids and edge.edge_id not in event.edge_ids:
            _fail("event edge_ids does not contain payload edge")
        if edge.vehicle_id not in state.vehicles:
            _fail(f"edge references unknown vehicle: {edge.vehicle_id}")
        edge_kit = state.sensor_kits.get(edge.sensor_kit_id)
        if edge_kit is None or edge_kit.vehicle_id != edge.vehicle_id:
            _fail(f"edge crosses vehicle/sensor-kit boundary: {edge.edge_id}")
        if edge.edge_id in state.edges:
            _fail(f"calibration edge already registered: {edge.edge_id}")
        state.edges[edge.edge_id] = edge.model_copy(update={"updated_sequence": event.sequence})

    if event.event_type in {"install", "remove"}:
        raw_sensor = _mapping("sensor", required=True)
        try:
            sensor = PhysicalSensorRecord.model_validate(raw_sensor)
        except Exception as exc:
            _fail(f"invalid sensor update payload: {exc}")
        if event.sensor_id != sensor.sensor_id:
            _fail("sensor update event identity does not match payload")
        current = state.sensors.get(sensor.sensor_id)
        if current is None:
            _fail(f"sensor update references unknown sensor: {sensor.sensor_id}")
        if any(
            expected != observed
            for expected, observed in (
                (current.vehicle_id, sensor.vehicle_id),
                (current.sensor_kit_id, sensor.sensor_kit_id),
                (current.serial, sensor.serial),
                (current.model, sensor.model),
                (current.firmware, sensor.firmware),
                (current.mount, sensor.mount),
            )
        ):
            _fail(f"physical sensor identity changed: {sensor.sensor_id}")
        base_sequence = payload.get("base_sensor_sequence")
        if base_sequence is not None and (
            not isinstance(base_sequence, int) or current.updated_sequence != base_sequence
        ):
            _fail(
                f"stale sensor update for {sensor.sensor_id}: "
                f"expected sequence {base_sequence}, current {current.updated_sequence}"
            )
        state.sensors[sensor.sensor_id] = sensor.model_copy(
            update={"updated_sequence": event.sequence}
        )

    if event.event_type in {"evaluate", "hold", "reject", "promote", "rollback"}:
        raw_evaluation = payload.get("evaluation")
        if raw_evaluation is not None:
            try:
                evaluation = LifecycleEvaluationArtifact.model_validate(raw_evaluation)
                evaluation.verify_artifact_digest()
            except Exception as exc:
                _fail(f"invalid evaluation payload: {exc}")
            if evaluation.registry_id != state.registry_id:
                _fail("evaluation registry_id does not match registry")
            if evaluation.edge_ids != event.edge_ids:
                _fail("evaluation edge_ids do not match event edge_ids")

        updates = payload.get("edge_updates")
        if updates is not None and not isinstance(updates, Mapping):
            _fail("payload.edge_updates must be an object")
        if isinstance(updates, Mapping):
            update_ids = [key for key in updates if isinstance(key, str)]
            if len(update_ids) != len(updates):
                _fail("edge_updates keys must be strings")
            if event.edge_ids and set(update_ids) != set(event.edge_ids):
                _fail("edge_updates keys do not match event edge_ids")
            bases = payload.get("base_edge_sequences")
            if bases is not None and not isinstance(bases, Mapping):
                _fail("payload.base_edge_sequences must be an object")
            for edge_id, raw in updates.items():
                if not isinstance(edge_id, str) or not isinstance(raw, Mapping):
                    _fail("edge_updates must map string IDs to objects")
                try:
                    updated_edge = CalibrationEdge.model_validate(raw)
                except Exception as exc:
                    _fail(f"invalid edge update {edge_id}: {exc}")
                if updated_edge.edge_id != edge_id:
                    _fail(f"edge update key does not match edge_id: {edge_id}")
                current_edge = state.edges.get(edge_id)
                if current_edge is None:
                    _fail(f"edge update references unknown edge: {edge_id}")
                if any(
                    expected != observed
                    for expected, observed in (
                        (current_edge.vehicle_id, updated_edge.vehicle_id),
                        (current_edge.sensor_kit_id, updated_edge.sensor_kit_id),
                        (current_edge.parent_frame, updated_edge.parent_frame),
                        (current_edge.child_frame, updated_edge.child_frame),
                    )
                ):
                    _fail(f"edge identity changed: {edge_id}")
                if isinstance(bases, Mapping) and edge_id in bases:
                    base_sequence = bases[edge_id]
                    if not isinstance(base_sequence, int) or (
                        current_edge.updated_sequence != base_sequence
                    ):
                        _fail(
                            f"stale edge update for {edge_id}: expected sequence "
                            f"{base_sequence}, current {current_edge.updated_sequence}"
                        )
                state.edges[edge_id] = updated_edge.model_copy(
                    update={"updated_sequence": event.sequence}
                )

    if event.event_type == "capture":
        raw_capture_id = payload.get("capture_id")
        if not isinstance(raw_capture_id, str) or not raw_capture_id:
            _fail("capture event is missing capture_id")
        capture_id = raw_capture_id
        capture_ref = event.artifacts.capture
        vehicle_id = event.vehicle_id
        sensor_kit_id = event.sensor_kit_id
        if capture_ref is None or not vehicle_id or not sensor_kit_id:
            _fail("capture event is missing identity or capture artifact")
        if capture_id in state.captures:
            _fail(f"capture already recorded: {capture_id}")
        _require_identity(state, vehicle_id, sensor_kit_id)
        state.captures[capture_id] = CaptureRecord(
            capture_id=capture_id,
            vehicle_id=vehicle_id,
            sensor_kit_id=sensor_kit_id,
            status=str(payload.get("capture_status", "unknown")),
            artifact=capture_ref,
            sequence=event.sequence,
        )


__all__ = [
    "CALIBRATION_LIFECYCLE_REGISTRY_SCHEMA_VERSION",
    "EVALUATION_SCHEMA_VERSION",
    "EVENT_SCHEMA_VERSION",
    "LIFECYCLE_EVALUATION_SCHEMA_VERSION",
    "LIFECYCLE_EVENT_SCHEMA_VERSION",
    "LIFECYCLE_REGISTRY_SCHEMA_VERSION",
    "REGISTRY_SCHEMA_VERSION",
    "ZERO_SHA256",
    "CalibrationEdge",
    "CaptureRecord",
    "LifecycleAdmission",
    "LifecycleArtifactRef",
    "LifecycleArtifactSet",
    "LifecycleEvaluationArtifact",
    "LifecycleEvent",
    "LifecycleEventType",
    "LifecycleProvenance",
    "LifecycleRegistry",
    "LifecycleRegistryConcurrencyError",
    "LifecycleRegistryError",
    "LifecycleRegistryState",
    "LifecycleRegistryStatus",
    "LifecycleRegistryVerificationError",
    "LifecycleRollbackAdapter",
    "LifecycleStatus",
    "PhysicalSensorRecord",
    "RegistryHead",
    "RegistryManifest",
    "RegistryVerificationArtifact",
    "SensorKitRecord",
    "VehicleRecord",
    "evaluate_lifecycle",
    "init_registry",
    "initialize_registry",
    "install_sensor",
    "lifecycle_evaluation_json_schema",
    "lifecycle_event_json_schema",
    "lifecycle_head_json_schema",
    "lifecycle_registry_json_schema",
    "lifecycle_registry_state_json_schema",
    "lifecycle_registry_verification_json_schema",
    "lifecycle_status_json_schema",
    "load_registry",
    "promote_lifecycle",
    "record_capture",
    "register_calibration_edge",
    "register_sensor",
    "register_sensor_kit",
    "register_vehicle",
    "remove_sensor",
    "rollback_lifecycle",
    "verify_registry",
]

# Historical spelling used by a few downstream clients.
CALIBRATION_LIFECYCLE_REGISTRY_SCHEMA_VERSION = LIFECYCLE_REGISTRY_SCHEMA_VERSION
