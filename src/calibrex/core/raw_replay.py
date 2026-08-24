"""Digest-bound raw-capture replay and field-replacement orchestration.

The replay surface is intentionally a small, ROS-independent contract.  A
capture manifest and every file used by a replay are treated as immutable
inputs; providers are reached through either the in-process calibration API or
an argv-only adapter.  The module records a plan, stage evidence, the final
decision, and all lifecycle actions in one self-digested artifact.

This is an orchestration boundary, not a second calibration solver.  It is
safe to use from CI because the default path never mutates an Autoware
workspace and ``precomputed`` mode verifies the declared candidate bytes and
the replay protocol before accepting them.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field, field_validator, model_validator

from calibrex import __version__
from calibrex.core.capture_manifest import (
    load_capture_manifest,
    verify_capture_manifest_inputs,
)
from calibrex.core.exceptions import CalibrexError
from calibrex.core.io import read_mapping, write_mapping_atomic
from calibrex.core.lifecycle_registry import (
    LifecycleRegistryError,
    evaluate_lifecycle,
    init_registry,
    install_sensor,
    record_capture,
    register_calibration_edge,
    register_sensor,
    register_sensor_kit,
    register_vehicle,
    remove_sensor,
    verify_registry,
)
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.result import CalibrationResult, MetricResult, StrictModel, load_result
from calibrex.evaluation.compare import ResultComparison, compare_results
from calibrex.pipelines.calibrate import CalibrationRunOptions, run_calibration

RAW_REPLAY_DEFINITION_SCHEMA_VERSION: Literal["slac.raw_replay_definition/v0.1"] = (
    "slac.raw_replay_definition/v0.1"
)
RAW_REPLAY_PLAN_SCHEMA_VERSION: Literal["slac.raw_replay_plan/v0.1"] = "slac.raw_replay_plan/v0.1"
RAW_REPLAY_STAGE_SCHEMA_VERSION: Literal["slac.raw_replay_stage/v0.1"] = (
    "slac.raw_replay_stage/v0.1"
)
RAW_REPLAY_RESULT_SCHEMA_VERSION: Literal["slac.raw_replay_result/v0.1"] = (
    "slac.raw_replay_result/v0.1"
)
RAW_REPLAY_COMPARISON_SCHEMA_VERSION: Literal["slac.raw_replay_comparison/v0.1"] = (
    "slac.raw_replay_comparison/v0.1"
)
FIELD_REPLACEMENT_PILOT_SCHEMA_VERSION: Literal["slac.field_replacement_pilot/v0.1"] = (
    "slac.field_replacement_pilot/v0.1"
)

ReplayStatus = Literal["PASS", "WARN", "FAIL", "BLOCKED"]
ReplayDecision = Literal["ADOPT", "HOLD", "REJECT", "BLOCKED", "NOT_APPLICABLE"]
ReplayStageStatus = ReplayStatus
ReplayExecutionMode = Literal["in_process", "argv", "precomputed"]
ReplayInputRole = Literal[
    "capture_manifest",
    "source",
    "config",
    "baseline",
    "candidate",
    "holdout",
    "known_bad",
    "environment",
    "tool",
    "container",
    "promotion",
    "smoke",
]
ReplayStageKind = Literal[
    "verify_inputs",
    "readiness",
    "execute_calibration",
    "validate_result",
    "evaluate",
    "compare",
    "lifecycle",
    "autoware_plan",
    "rollback_simulation",
]

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_STAGE_ORDER: tuple[ReplayStageKind, ...] = (
    "verify_inputs",
    "readiness",
    "execute_calibration",
    "validate_result",
    "evaluate",
    "compare",
    "lifecycle",
    "autoware_plan",
)
_REQUIRED_STAGE_ORDER: tuple[ReplayStageKind, ...] = _STAGE_ORDER[:6]


class RawReplayError(CalibrexError):
    """Raised when a replay definition or artifact violates its contract."""


class ReplayDigestRef(StrictModel):
    """Portable digest reference to one replay input or generated artifact."""

    role: str = Field(min_length=1)
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    size_bytes: int | None = Field(default=None, ge=0)
    schema_version: str | None = None


class ReplayInput(StrictModel):
    """One immutable file bound to a replay definition."""

    input_id: str = Field(min_length=1)
    role: ReplayInputRole
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    schema_version: str | None = None
    required: bool = True
    notes: list[str] = Field(default_factory=list)


class ReplayEnvironment(StrictModel):
    """Execution environment identity; no ambient state is trusted silently."""

    environment_id: str = Field(default="local", min_length=1)
    platform: str = Field(default_factory=platform.platform)
    python_version: str = Field(default_factory=lambda: platform.python_version())
    calibrex_version: str = __version__
    git_commit: str | None = None
    environment_sha256: str = Field(pattern=_SHA256_PATTERN)
    container_image: str | None = None
    container_digest: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    network: Literal["none", "host"] = "none"
    environment_inputs: list[str] = Field(default_factory=list)


class ReplayTool(StrictModel):
    """Provider/adapter identity, including the immutable tool digest."""

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    adapter_version: str = Field(min_length=1)
    tool_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_commit: str | None = None
    license_spdx: str | None = None
    command: list[str] = Field(default_factory=list)
    executable_path: str | None = None
    executable_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)

    @field_validator("command")
    @classmethod
    def validate_argv(cls, value: list[str]) -> list[str]:
        if any(not isinstance(item, str) or not item or "\x00" in item for item in value):
            raise ValueError("tool command must contain non-empty argv strings without NUL")
        return list(value)

    @model_validator(mode="after")
    def validate_executable_binding(self) -> ReplayTool:
        """Require a digest whenever an external executable is declared."""

        if self.executable_path is not None and self.executable_sha256 is None:
            raise ValueError("tool executable_sha256 is required with executable_path")
        return self


class ReplayProtocol(StrictModel):
    """Frozen split, seed, provider, and stage protocol."""

    protocol_id: str = Field(min_length=1)
    protocol_version: str = "v0.1"
    deterministic_seed: int = 0
    split_seed: int = 0
    holdout_ratio: float = Field(default=0.2, gt=0.0, lt=1.0)
    execution_mode: ReplayExecutionMode = "precomputed"
    provider: str = "calibrex.native"
    stages: list[ReplayStageKind] = Field(default_factory=lambda: list(_STAGE_ORDER))
    stage_timeout_seconds: dict[str, float] = Field(default_factory=dict)
    require_independent_holdout: bool = True
    require_known_bad: bool = True
    require_observability: bool = True
    allow_autoware_plan: bool = False
    mutate_autoware: bool = False

    @field_validator("stages")
    @classmethod
    def validate_stages(cls, value: list[ReplayStageKind]) -> list[ReplayStageKind]:
        if len(value) != len(set(value)):
            raise ValueError("replay stages must be unique")
        if tuple(value[: len(_REQUIRED_STAGE_ORDER)]) != _REQUIRED_STAGE_ORDER:
            raise ValueError("replay stages must begin with the exact required stage order")
        if any(item not in _STAGE_ORDER for item in value):
            raise ValueError(f"unsupported replay stage; expected {_STAGE_ORDER}")
        return list(value)

    @model_validator(mode="after")
    def validate_timeouts(self) -> ReplayProtocol:
        unknown = sorted(set(self.stage_timeout_seconds) - set(self.stages))
        if unknown:
            raise ValueError(
                "stage_timeout_seconds references undeclared stage(s): " + ", ".join(unknown)
            )
        for stage, seconds in self.stage_timeout_seconds.items():
            if seconds <= 0.0 or seconds > 86_400.0:
                raise ValueError(f"stage timeout for {stage} must be in (0, 86400]")
        if self.execution_mode == "precomputed" and self.mutate_autoware:
            raise ValueError("precomputed replay may not mutate Autoware")
        return self


class ReplayBudgets(StrictModel):
    """Metric and transform acceptance budgets frozen before execution."""

    holdout_metric: str = "holdout_rmse"
    holdout_field: Literal["value", "holdout", "train"] = "value"
    max_holdout_value: float | None = Field(default=None, ge=0.0)
    known_bad_metric: str = "holdout_rmse"
    known_bad_field: Literal["value", "holdout", "train"] = "value"
    min_known_bad_delta: float | None = Field(default=None, ge=0.0)
    max_transform_translation_delta_m: float | None = Field(default=None, ge=0.0)
    max_transform_rotation_delta_deg: float | None = Field(default=None, ge=0.0)
    min_observability_rank: int = Field(default=1, ge=0, le=6)
    require_candidate_quality_pass: bool = True


class ReplayRegistryRequest(StrictModel):
    """Optional field-pilot lifecycle scope."""

    registry_root: str = Field(min_length=1)
    vehicle_id: str = Field(min_length=1)
    sensor_kit_id: str = Field(min_length=1)
    edge_id: str = Field(min_length=1)
    parent_frame: str = Field(min_length=1)
    child_frame: str = Field(min_length=1)
    sensor_id: str = Field(min_length=1)
    serial: str = Field(min_length=1)
    model: str = Field(min_length=1)
    firmware: str = Field(min_length=1)
    mount: str = Field(min_length=1)
    frame: str | None = None
    replacement_serial: str | None = None
    replacement_mount: str | None = None
    replacement_sensor_id: str | None = None
    operator: str = "ci"
    simulate_rollback: bool = True

    @model_validator(mode="after")
    def validate_replacement_identity(self) -> ReplayRegistryRequest:
        """Keep physical replacement identity distinct from a remount."""

        replacement_changed = (
            self.replacement_serial is not None
            and self.replacement_serial != self.serial
        ) or (
            self.replacement_mount is not None
            and self.replacement_mount != self.mount
        )
        if replacement_changed and self.replacement_sensor_id == self.sensor_id:
            raise ValueError("a physical replacement requires a distinct replacement_sensor_id")
        if not replacement_changed and self.replacement_sensor_id is not None:
            raise ValueError("replacement_sensor_id requires a changed serial or mount")
        return self


class ReplayAutowareRequest(StrictModel):
    """Read-only package plan request; apply is deliberately not represented."""

    workspace_root: str = Field(min_length=1)
    package_root: str = Field(min_length=1)
    vehicle_id: str = Field(min_length=1)
    sensor_kit_id: str = Field(min_length=1)
    baseline_manifest_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    candidate_path: str | None = None
    output_path: str | None = None


class ReplayProvenance(StrictModel):
    """Complete provenance shared by every generated replay artifact."""

    producer: Literal["calibrex"] = "calibrex"
    tool_name: str = "calibrex.raw-replay"
    tool_version: str = __version__
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_commit: str | None = Field(default_factory=git_commit)
    command: list[str] = Field(default_factory=list)
    source_sha256: dict[str, str] = Field(default_factory=dict)
    config_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    baseline_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    candidate_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    holdout_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    known_bad_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    evidence_sha256: dict[str, str] = Field(default_factory=dict)
    environment_sha256: str = Field(pattern=_SHA256_PATTERN)
    tool_sha256: str = Field(pattern=_SHA256_PATTERN)
    container_digest: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    artifact_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    artifact_digest_scope: str = (
        "canonical artifact excluding artifact_sha256 and provenance.artifact_sha256/generated_at"
    )


class ReplayDefinition(StrictModel):
    """Schema-valid, digest-bound declaration for one replay."""

    schema_version: Literal["slac.raw_replay_definition/v0.1"] = (
        RAW_REPLAY_DEFINITION_SCHEMA_VERSION
    )
    replay_id: str = Field(min_length=1)
    description: str = ""
    capture_manifest: ReplayInput
    inputs: list[ReplayInput] = Field(default_factory=list)
    config: ReplayInput
    baseline: ReplayInput | None = None
    candidate: ReplayInput | None = None
    holdout: ReplayInput | None = None
    known_bad: ReplayInput | None = None
    protocol: ReplayProtocol
    budgets: ReplayBudgets = Field(default_factory=ReplayBudgets)
    tool: ReplayTool
    environment: ReplayEnvironment
    registry: ReplayRegistryRequest | None = None
    autoware: ReplayAutowareRequest | None = None
    provenance: ReplayProvenance
    artifact_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_definition(self) -> ReplayDefinition:
        declared = [self.capture_manifest, self.config, *self.inputs]
        for item in (self.baseline, self.candidate, self.holdout, self.known_bad):
            if item is not None:
                declared.append(item)
        ids = [item.input_id for item in declared]
        if len(ids) != len(set(ids)):
            raise ValueError("replay input IDs must be unique")
        if self.capture_manifest.role != "capture_manifest":
            raise ValueError("capture_manifest input must have role capture_manifest")
        if self.config.role != "config":
            raise ValueError("config input must have role config")
        for attr, role in (
            ("baseline", "baseline"),
            ("candidate", "candidate"),
            ("holdout", "holdout"),
            ("known_bad", "known_bad"),
        ):
            item = getattr(self, attr)
            if item is not None and item.role != role:
                raise ValueError(f"{attr} input must have role {role}")
        if self.protocol.execution_mode == "precomputed" and self.candidate is None:
            raise ValueError("precomputed replay requires a digest-bound candidate input")
        if self.protocol.execution_mode == "argv" and self.candidate is None:
            raise ValueError("argv replay requires a digest-bound candidate output input")
        if self.protocol.require_independent_holdout and self.holdout is None:
            raise ValueError("replay protocol requires an independent holdout input")
        if self.protocol.require_known_bad and self.known_bad is None:
            raise ValueError("replay protocol requires a known_bad input")
        if self.protocol.require_independent_holdout and self.budgets.max_holdout_value is None:
            raise ValueError("independent holdout requires an explicit max_holdout_value budget")
        if self.protocol.require_known_bad and self.budgets.min_known_bad_delta is None:
            raise ValueError("known_bad control requires an explicit min_known_bad_delta budget")
        if self.protocol.require_independent_holdout and self.budgets.holdout_field == "train":
            raise ValueError("independent holdout gate cannot use a train metric")
        if self.protocol.require_known_bad and self.budgets.known_bad_field == "train":
            raise ValueError("known_bad control gate cannot use a train metric")
        for item, label in (
            (self.holdout, "holdout"),
            (self.known_bad, "known_bad"),
        ):
            if item is not None and not item.required:
                raise ValueError(f"{label} input must be required for fail-closed replay")
        named_inputs = [
            item
            for item in (self.capture_manifest, self.config, self.baseline, self.candidate,
                         self.holdout, self.known_bad)
            if item is not None
        ]
        if self.holdout is not None and any(
            self.holdout.sha256 == item.sha256
            for item in named_inputs
            if item is not self.holdout
        ):
            raise ValueError("independent holdout digest must not reuse another replay input")
        if self.known_bad is not None and any(
            self.known_bad.sha256 == item.sha256
            for item in named_inputs
            if item is not self.known_bad
        ):
            raise ValueError("known_bad digest must not reuse another replay input")
        if self.protocol.allow_autoware_plan and self.autoware is None:
            raise ValueError("allow_autoware_plan requires an autoware plan request")
        if self.protocol.mutate_autoware:
            raise ValueError(
                "raw replay never permits package mutation; apply is a separate command"
            )
        expected_provenance = {
            "config_sha256": self.config.sha256,
            "baseline_sha256": self.baseline.sha256 if self.baseline else None,
            "candidate_sha256": self.candidate.sha256 if self.candidate else None,
            "holdout_sha256": self.holdout.sha256 if self.holdout else None,
            "known_bad_sha256": self.known_bad.sha256 if self.known_bad else None,
            "environment_sha256": self.environment.environment_sha256,
            "tool_sha256": self.tool.tool_sha256,
        }
        for field_name, expected in expected_provenance.items():
            observed = getattr(self.provenance, field_name)
            if observed != expected:
                raise ValueError(
                    f"definition provenance.{field_name} must match declared replay input"
                )
        for item in (self.capture_manifest, *self.inputs):
            if item.role in {"capture_manifest", "source"} and (
                self.provenance.source_sha256.get(item.input_id) != item.sha256
            ):
                raise ValueError(
                    f"definition provenance.source_sha256 is missing or drifted for {item.input_id}"
                )
        return self

    def with_artifact_digest(self) -> ReplayDefinition:
        payload = self.model_dump(mode="json", exclude_none=False)
        payload.pop("artifact_sha256", None)
        provenance = cast(dict[str, Any], payload["provenance"])
        # Generation time is audit metadata, not identity.  Excluding it
        # keeps a definition digest stable when a YAML definition relies on
        # the provenance timestamp default.
        provenance.pop("generated_at", None)
        provenance.pop("artifact_sha256", None)
        digest = _canonical_sha256(payload)
        return self.model_copy(
            update={
                "artifact_sha256": digest,
                "provenance": self.provenance.model_copy(update={"artifact_sha256": digest}),
            }
        )

    def verify_artifact_digest(self) -> None:
        expected = self.with_artifact_digest().artifact_sha256
        if expected != self.artifact_sha256:
            raise RawReplayError(
                "replay definition self-digest mismatch: "
                f"declared={self.artifact_sha256}, expected={expected}"
            )
        if self.provenance.artifact_sha256 != self.artifact_sha256:
            raise RawReplayError("replay definition provenance artifact digest is not bound")

    def save(self, path: str | Path) -> None:
        artifact = self.with_artifact_digest()
        write_mapping_atomic(Path(path), artifact.model_dump(mode="json", exclude_none=False))


class ReplayStageArtifact(StrictModel):
    """One independently verifiable stage output."""

    schema_version: Literal["slac.raw_replay_stage/v0.1"] = RAW_REPLAY_STAGE_SCHEMA_VERSION
    replay_id: str = Field(min_length=1)
    stage_id: str = Field(min_length=1)
    stage_kind: ReplayStageKind
    status: ReplayStageStatus
    reason: str = ""
    checks: dict[str, ReplayStageStatus] = Field(default_factory=dict)
    observations: dict[str, Any] = Field(default_factory=dict)
    actions: list[str] = Field(default_factory=list)
    provenance: ReplayProvenance
    artifact_sha256: str = Field(pattern=_SHA256_PATTERN)

    def with_artifact_digest(self) -> ReplayStageArtifact:
        payload = self.model_dump(mode="json", exclude_none=False)
        payload.pop("artifact_sha256", None)
        provenance = cast(dict[str, Any], payload["provenance"])
        provenance.pop("artifact_sha256", None)
        provenance.pop("generated_at", None)
        digest = _canonical_sha256(payload)
        return self.model_copy(
            update={
                "artifact_sha256": digest,
                "provenance": self.provenance.model_copy(update={"artifact_sha256": digest}),
            }
        )

    def verify_artifact_digest(self) -> None:
        expected = self.with_artifact_digest().artifact_sha256
        if expected != self.artifact_sha256:
            raise RawReplayError(f"replay stage self-digest mismatch for {self.stage_id}")
        if self.provenance.artifact_sha256 != self.artifact_sha256:
            raise RawReplayError(f"replay stage provenance digest is not bound for {self.stage_id}")

    def save(self, path: str | Path) -> None:
        artifact = self.with_artifact_digest()
        write_mapping_atomic(Path(path), artifact.model_dump(mode="json", exclude_none=False))


class ReplayStageResult(StrictModel):
    """Timing and digest reference for one stage in the top-level result."""

    stage_id: str = Field(min_length=1)
    stage_kind: ReplayStageKind
    status: ReplayStageStatus
    timeout_seconds: float | None = Field(default=None, gt=0.0)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    artifact: ReplayDigestRef | None = None
    reason: str = ""
    actions: list[str] = Field(default_factory=list)


class ReplayGate(StrictModel):
    """One explicit decision gate with observed value and rationale."""

    gate_id: str = Field(min_length=1)
    status: ReplayStatus
    expected: str | float | int | bool | None = None
    observed: str | float | int | bool | None = None
    reason: str


class ReplayProvenanceSummary(StrictModel):
    """Digest references carried by the final replay result."""

    source: list[ReplayDigestRef] = Field(default_factory=list)
    config: ReplayDigestRef | None = None
    baseline: ReplayDigestRef | None = None
    candidate: ReplayDigestRef | None = None
    holdout: ReplayDigestRef | None = None
    known_bad: ReplayDigestRef | None = None
    environment: ReplayDigestRef | None = None
    tool: ReplayDigestRef | None = None
    container: ReplayDigestRef | None = None
    evidence: list[ReplayDigestRef] = Field(default_factory=list)


class ReplayResult(StrictModel):
    """Top-level replay artifact: all stages, gates, actions, and advice."""

    schema_version: Literal["slac.raw_replay_result/v0.1"] = RAW_REPLAY_RESULT_SCHEMA_VERSION
    replay_id: str = Field(min_length=1)
    definition_sha256: str = Field(pattern=_SHA256_PATTERN)
    status: ReplayStatus
    decision: ReplayDecision
    reason: str
    stages: list[ReplayStageResult] = Field(min_length=1)
    gates: list[ReplayGate] = Field(default_factory=list)
    metrics: dict[str, float | None] = Field(default_factory=dict)
    transform_deltas: dict[str, dict[str, float]] = Field(default_factory=dict)
    artifacts: dict[str, ReplayDigestRef] = Field(default_factory=dict)
    lifecycle_actions: list[str] = Field(default_factory=list)
    operator_recommendations: list[str] = Field(default_factory=list)
    provenance: ReplayProvenanceSummary
    generated_provenance: ReplayProvenance
    artifact_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_result(self) -> ReplayResult:
        stage_ids = [item.stage_id for item in self.stages]
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError("replay stage IDs must be unique")
        if any(item.artifact is None for item in self.stages):
            raise ValueError("every replay stage must carry a digest-bound artifact")
        artifact_refs = {
            (ref.role, ref.path, ref.sha256, ref.size_bytes, ref.schema_version)
            for ref in self.artifacts.values()
        }
        for item in self.stages:
            if item.artifact is None:
                continue
            ref_key = (
                item.artifact.role,
                item.artifact.path,
                item.artifact.sha256,
                item.artifact.size_bytes,
                item.artifact.schema_version,
            )
            if ref_key not in artifact_refs:
                raise ValueError("every replay stage artifact must be present in artifacts")
        expected_decisions: dict[ReplayStatus, set[ReplayDecision]] = {
            "PASS": {"ADOPT", "HOLD"},
            "WARN": {"HOLD", "NOT_APPLICABLE"},
            "FAIL": {"REJECT", "HOLD"},
            "BLOCKED": {"BLOCKED"},
        }
        if self.decision not in expected_decisions[self.status]:
            raise ValueError(
                f"replay decision {self.decision} is inconsistent with status {self.status}"
            )
        return self

    def with_artifact_digest(self) -> ReplayResult:
        payload = self.model_dump(mode="json", exclude_none=False)
        payload.pop("artifact_sha256", None)
        generated = cast(dict[str, Any], payload["generated_provenance"])
        generated.pop("artifact_sha256", None)
        generated.pop("generated_at", None)
        digest = _canonical_sha256(payload)
        return self.model_copy(
            update={
                "artifact_sha256": digest,
                "generated_provenance": self.generated_provenance.model_copy(
                    update={"artifact_sha256": digest}
                ),
            }
        )

    def verify_artifact_digest(self) -> None:
        expected = self.with_artifact_digest().artifact_sha256
        if expected != self.artifact_sha256:
            raise RawReplayError(
                "replay result self-digest mismatch: "
                f"declared={self.artifact_sha256}, expected={expected}"
            )
        if self.generated_provenance.artifact_sha256 != self.artifact_sha256:
            raise RawReplayError("replay result provenance artifact digest is not bound")

    def save(self, path: str | Path) -> None:
        artifact = self.with_artifact_digest()
        write_mapping_atomic(Path(path), artifact.model_dump(mode="json", exclude_none=False))


class ReplayPlan(StrictModel):
    """Read-only plan artifact produced before execution."""

    schema_version: Literal["slac.raw_replay_plan/v0.1"] = RAW_REPLAY_PLAN_SCHEMA_VERSION
    replay_id: str = Field(min_length=1)
    definition_sha256: str = Field(pattern=_SHA256_PATTERN)
    status: ReplayStatus
    reason: str
    stages: list[ReplayStageResult] = Field(min_length=1)
    input_digests: list[ReplayDigestRef] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    operator_recommendations: list[str] = Field(default_factory=list)
    provenance: ReplayProvenance
    artifact_sha256: str = Field(pattern=_SHA256_PATTERN)

    def with_artifact_digest(self) -> ReplayPlan:
        payload = self.model_dump(mode="json", exclude_none=False)
        payload.pop("artifact_sha256", None)
        provenance = cast(dict[str, Any], payload["provenance"])
        provenance.pop("artifact_sha256", None)
        provenance.pop("generated_at", None)
        digest = _canonical_sha256(payload)
        return self.model_copy(
            update={
                "artifact_sha256": digest,
                "provenance": self.provenance.model_copy(update={"artifact_sha256": digest}),
            }
        )

    def verify_artifact_digest(self) -> None:
        if self.with_artifact_digest().artifact_sha256 != self.artifact_sha256:
            raise RawReplayError("replay plan self-digest mismatch")
        if self.provenance.artifact_sha256 != self.artifact_sha256:
            raise RawReplayError("replay plan provenance artifact digest is not bound")

    def save(self, path: str | Path) -> None:
        artifact = self.with_artifact_digest()
        write_mapping_atomic(Path(path), artifact.model_dump(mode="json", exclude_none=False))


class ReplayComparison(StrictModel):
    """Digest-bound comparison between two top-level replay results."""

    schema_version: Literal["slac.raw_replay_comparison/v0.1"] = (
        RAW_REPLAY_COMPARISON_SCHEMA_VERSION
    )
    left: ReplayDigestRef
    right: ReplayDigestRef
    status: ReplayStatus
    decision: Literal["BETTER", "WORSE", "TIE", "NOT_COMPARABLE"]
    metric_deltas: dict[str, float | None] = Field(default_factory=dict)
    transform_deltas: dict[str, dict[str, float]] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    provenance: ReplayProvenance
    artifact_sha256: str = Field(pattern=_SHA256_PATTERN)

    def with_artifact_digest(self) -> ReplayComparison:
        payload = self.model_dump(mode="json", exclude_none=False)
        payload.pop("artifact_sha256", None)
        provenance = cast(dict[str, Any], payload["provenance"])
        provenance.pop("artifact_sha256", None)
        provenance.pop("generated_at", None)
        digest = _canonical_sha256(payload)
        return self.model_copy(
            update={
                "artifact_sha256": digest,
                "provenance": self.provenance.model_copy(update={"artifact_sha256": digest}),
            }
        )

    def verify_artifact_digest(self) -> None:
        if self.with_artifact_digest().artifact_sha256 != self.artifact_sha256:
            raise RawReplayError("replay comparison self-digest mismatch")
        if self.provenance.artifact_sha256 != self.artifact_sha256:
            raise RawReplayError("replay comparison provenance artifact digest is not bound")

    def save(self, path: str | Path) -> None:
        artifact = self.with_artifact_digest()
        write_mapping_atomic(Path(path), artifact.model_dump(mode="json", exclude_none=False))


class RollbackSimulation(StrictModel):
    """Explicitly dry-run rollback evidence for a field replacement."""

    attempted: bool = False
    package_mutated: bool = False
    status: Literal["PASS", "WARN", "BLOCKED", "NOT_RUN"] = "NOT_RUN"
    reason: str = ""
    target_event_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_rollback_state(self) -> RollbackSimulation:
        if self.attempted and self.status == "NOT_RUN":
            raise ValueError("attempted rollback simulation cannot be NOT_RUN")
        if self.package_mutated:
            raise ValueError("raw replay rollback simulation may not mutate a package")
        return self


class FieldReplacementPilot(StrictModel):
    """Vehicle/sensor replacement evidence paired with a replay result."""

    schema_version: Literal["slac.field_replacement_pilot/v0.1"] = (
        FIELD_REPLACEMENT_PILOT_SCHEMA_VERSION
    )
    pilot_id: str = Field(min_length=1)
    replay_id: str = Field(min_length=1)
    registry_root: str = Field(min_length=1)
    vehicle_id: str = Field(min_length=1)
    sensor_kit_id: str = Field(min_length=1)
    edge_id: str = Field(min_length=1)
    sensor_id: str = Field(min_length=1)
    baseline_serial: str = Field(min_length=1)
    replacement_serial: str = Field(min_length=1)
    baseline_mount: str = Field(min_length=1)
    replacement_mount: str = Field(min_length=1)
    remount_count: int = Field(default=0, ge=0)
    candidate: ReplayDigestRef | None = None
    lifecycle_event_digests: list[str] = Field(default_factory=list)
    status: ReplayStatus
    decision: ReplayDecision
    reason: str
    rollback: RollbackSimulation
    package_mutated: bool = False
    operator_recommendations: list[str] = Field(default_factory=list)
    provenance: ReplayProvenance
    artifact_sha256: str = Field(pattern=_SHA256_PATTERN)

    def with_artifact_digest(self) -> FieldReplacementPilot:
        payload = self.model_dump(mode="json", exclude_none=False)
        payload.pop("artifact_sha256", None)
        provenance = cast(dict[str, Any], payload["provenance"])
        provenance.pop("artifact_sha256", None)
        provenance.pop("generated_at", None)
        digest = _canonical_sha256(payload)
        return self.model_copy(
            update={
                "artifact_sha256": digest,
                "provenance": self.provenance.model_copy(update={"artifact_sha256": digest}),
            }
        )

    def verify_artifact_digest(self) -> None:
        if self.with_artifact_digest().artifact_sha256 != self.artifact_sha256:
            raise RawReplayError("field replacement pilot self-digest mismatch")
        if self.provenance.artifact_sha256 != self.artifact_sha256:
            raise RawReplayError("field replacement pilot provenance digest is not bound")

    def save(self, path: str | Path) -> None:
        artifact = self.with_artifact_digest()
        write_mapping_atomic(Path(path), artifact.model_dump(mode="json", exclude_none=False))


def replay_definition_json_schema() -> dict[str, Any]:
    """Return the replay-definition schema."""

    return ReplayDefinition.model_json_schema()


def replay_plan_json_schema() -> dict[str, Any]:
    """Return the replay-plan schema."""

    return ReplayPlan.model_json_schema()


def replay_stage_json_schema() -> dict[str, Any]:
    """Return the replay-stage schema."""

    return ReplayStageArtifact.model_json_schema()


def replay_result_json_schema() -> dict[str, Any]:
    """Return the top-level replay-result schema."""

    return ReplayResult.model_json_schema()


def replay_comparison_json_schema() -> dict[str, Any]:
    """Return the replay-comparison schema."""

    return ReplayComparison.model_json_schema()


def field_replacement_pilot_json_schema() -> dict[str, Any]:
    """Return the field-replacement pilot schema."""

    return FieldReplacementPilot.model_json_schema()


def load_replay_definition(path: str | Path) -> ReplayDefinition:
    """Load and verify a digest-complete replay definition."""

    definition_path = Path(path)
    try:
        definition = ReplayDefinition.model_validate(read_mapping(definition_path))
        definition.verify_artifact_digest()
    except Exception as exc:
        raise RawReplayError(f"invalid replay definition {definition_path}: {exc}") from exc
    return definition


def load_replay_result(path: str | Path) -> ReplayResult:
    """Load and verify a top-level replay result."""

    result_path = Path(path)
    try:
        result = ReplayResult.model_validate(read_mapping(result_path))
        result.verify_artifact_digest()
    except Exception as exc:
        raise RawReplayError(f"invalid replay result {result_path}: {exc}") from exc
    return result


def load_replay_plan(path: str | Path) -> ReplayPlan:
    """Load and verify a replay plan."""

    plan_path = Path(path)
    try:
        plan = ReplayPlan.model_validate(read_mapping(plan_path))
        plan.verify_artifact_digest()
    except Exception as exc:
        raise RawReplayError(f"invalid replay plan {plan_path}: {exc}") from exc
    return plan


def load_field_replacement_pilot(path: str | Path) -> FieldReplacementPilot:
    """Load and verify a field-replacement pilot artifact."""

    pilot_path = Path(path)
    try:
        pilot = FieldReplacementPilot.model_validate(read_mapping(pilot_path))
        pilot.verify_artifact_digest()
    except Exception as exc:
        raise RawReplayError(f"invalid field replacement pilot {pilot_path}: {exc}") from exc
    return pilot


def plan_raw_replay(
    definition: ReplayDefinition | str | Path,
    *,
    output: str | Path | None = None,
    command: Sequence[str] = (),
) -> ReplayPlan:
    """Create a read-only plan and verify every declared input digest."""

    definition_path: Path | None = None
    if isinstance(definition, ReplayDefinition):
        replay_definition = definition
    else:
        definition_path = Path(definition)
        replay_definition = load_replay_definition(definition_path)
    base = definition_path.parent if definition_path is not None else Path.cwd()
    refs, errors = _verify_definition_inputs(replay_definition, base)
    status: ReplayStatus = "PASS" if not errors else "BLOCKED"
    actions = (
        ["do not execute calibration until every input digest and capture readiness check passes"]
        if errors
        else [
            "execute the declared stages with the frozen seed and protocol",
            "retain the result and all stage artifacts under the output directory",
        ]
    )
    plan = ReplayPlan(
        replay_id=replay_definition.replay_id,
        definition_sha256=replay_definition.artifact_sha256,
        status=status,
        reason="all declared replay inputs are present and digest-verified"
        if not errors
        else "; ".join(errors),
        stages=[
            ReplayStageResult(
                stage_id=stage,
                stage_kind=stage,
                status="PASS" if not errors else "BLOCKED",
                timeout_seconds=replay_definition.protocol.stage_timeout_seconds.get(stage),
                reason="planned" if not errors else "blocked by input verification",
            )
            for stage in replay_definition.protocol.stages
        ],
        input_digests=refs,
        actions=actions,
        operator_recommendations=(
            ["source/config/protocol drift must be resolved before rerun"]
            if errors
            else ["CI plan is read-only; Autoware apply is never implicit"]
        ),
        provenance=_provenance_for_definition(replay_definition, command=list(command)),
        artifact_sha256="0" * 64,
    ).with_artifact_digest()
    if output is not None:
        plan.save(output)
    return plan


def run_raw_replay(
    definition: ReplayDefinition | str | Path,
    *,
    output_directory: str | Path,
    command: Sequence[str] = (),
    strict: bool = True,
) -> ReplayResult:
    """Run a replay with fail-closed gates and no package mutation by default."""

    definition_path: Path | None = None
    if isinstance(definition, ReplayDefinition):
        replay_definition = definition
    else:
        definition_path = Path(definition)
        replay_definition = load_replay_definition(definition_path)
    base = definition_path.parent if definition_path is not None else Path.cwd()
    output = Path(output_directory)
    _validate_replay_output_scope(replay_definition, base=base, output=output)
    output.mkdir(parents=True, exist_ok=True)
    stage_directory = output / "stages"
    stage_directory.mkdir(parents=True, exist_ok=True)
    refs, input_errors = _verify_definition_inputs(replay_definition, base)
    stage_results: list[ReplayStageResult] = []
    stage_artifacts: dict[str, ReplayDigestRef] = {}
    gates: list[ReplayGate] = []
    metrics: dict[str, float | None] = {}
    transform_deltas: dict[str, dict[str, float]] = {}
    lifecycle_actions: list[str] = []
    recommendations: list[str] = []
    candidate_result_path: Path | None = None
    baseline_result_path: Path | None = None
    known_bad_result_path: Path | None = None
    candidate_result: CalibrationResult | None = None
    baseline_result: CalibrationResult | None = None
    known_bad_result: CalibrationResult | None = None
    status: ReplayStatus = "PASS"
    decision: ReplayDecision = "ADOPT"
    reason = "replay passed all declared gates"

    def emit_stage(
        stage_kind: ReplayStageKind,
        stage_status: ReplayStageStatus,
        *,
        stage_reason: str,
        checks: Mapping[str, ReplayStageStatus] | None = None,
        observations: Mapping[str, Any] | None = None,
        actions: Sequence[str] = (),
    ) -> None:
        stage_id = f"{replay_definition.replay_id}-{stage_kind}"
        artifact = ReplayStageArtifact(
            replay_id=replay_definition.replay_id,
            stage_id=stage_id,
            stage_kind=stage_kind,
            status=stage_status,
            reason=stage_reason,
            checks=dict(checks or {}),
            observations=dict(observations or {}),
            actions=list(actions),
            provenance=_provenance_for_definition(replay_definition, command=list(command)),
            artifact_sha256="0" * 64,
        ).with_artifact_digest()
        artifact_path = stage_directory / f"{len(stage_results):02d}-{stage_kind}.json"
        artifact.save(artifact_path)
        digest = sha256_path(artifact_path)
        if digest is None:
            raise RawReplayError(f"stage artifact was not written: {artifact_path}")
        ref = ReplayDigestRef(
            role=f"stage:{stage_kind}",
            path=_portable_path(output, artifact_path),
            sha256=digest,
            size_bytes=artifact_path.stat().st_size,
            schema_version=RAW_REPLAY_STAGE_SCHEMA_VERSION,
        )
        stage_artifacts[stage_kind] = ref
        stage_results.append(
            ReplayStageResult(
                stage_id=stage_id,
                stage_kind=stage_kind,
                status=stage_status,
                timeout_seconds=replay_definition.protocol.stage_timeout_seconds.get(stage_kind),
                artifact=ref,
                reason=stage_reason,
                actions=list(actions),
            )
        )

    # Stage 1: source/config/tool/environment and self-digest verification.
    if input_errors:
        status, decision = "BLOCKED", "BLOCKED"
        reason = "; ".join(input_errors)
        emit_stage(
            "verify_inputs",
            "BLOCKED",
            stage_reason=reason,
            actions=["fix declared input/protocol drift and rerun"],
        )
        for remaining in replay_definition.protocol.stages[1:]:
            emit_stage(remaining, "BLOCKED", stage_reason="blocked by verify_inputs stage")
        result = _build_replay_result(
            replay_definition,
            status=status,
            decision=decision,
            reason=reason,
            stages=stage_results,
            gates=gates,
            metrics=metrics,
            transform_deltas=transform_deltas,
            artifacts={**stage_artifacts},
            lifecycle_actions=lifecycle_actions,
            recommendations=["source/config/protocol drift must be resolved before adoption"],
            refs=refs,
            command=command,
            artifact_base=output,
        )
        result.save(output / "replay-result.json")
        return result
    emit_stage(
        "verify_inputs",
        "PASS",
        stage_reason="all source, config, tool, environment, and definition digests verified",
        checks={"inputs": "PASS"},
    )

    # Stage 2: capture manifest and readiness are independent gates.  A
    # manifest that is merely syntactically valid is not sufficient.
    capture_path = _resolve(base, replay_definition.capture_manifest.path)
    try:
        capture = load_capture_manifest(capture_path, verify=True)
        capture_verification = verify_capture_manifest_inputs(capture_path)
        if not capture_verification.valid:
            raise RawReplayError(capture_verification.summary)
        if capture.status != "ready":
            raise RawReplayError(f"capture status is {capture.status}, not ready")
        if (
            capture.provenance.config_sha256
            and capture.provenance.config_sha256 != replay_definition.config.sha256
        ):
            raise RawReplayError("capture manifest config digest differs from replay definition")
        emit_stage(
            "readiness",
            "PASS",
            stage_reason="capture manifest and raw inputs are ready",
            checks={"capture_manifest": "PASS", "readiness": "PASS"},
            observations={"capture_status": capture.status, "capture_id": capture.capture_id},
        )
    except Exception as exc:
        status, decision = "BLOCKED", "BLOCKED"
        reason = f"capture readiness blocked: {exc}"
        emit_stage(
            "readiness",
            "BLOCKED",
            stage_reason=reason,
            checks={"capture_manifest": "BLOCKED"},
            actions=["repair or recapture raw inputs; do not run a solver"],
        )
        for remaining in replay_definition.protocol.stages[2:]:
            emit_stage(remaining, "BLOCKED", stage_reason="blocked by readiness stage")
        result = _build_replay_result(
            replay_definition,
            status=status,
            decision=decision,
            reason=reason,
            stages=stage_results,
            gates=gates,
            metrics=metrics,
            transform_deltas=transform_deltas,
            artifacts={**stage_artifacts},
            lifecycle_actions=lifecycle_actions,
            recommendations=[
                "capture is not ready; recapture with independent holdout and known-bad controls"
            ],
            refs=refs,
            command=command,
            artifact_base=output,
        )
        result.save(output / "replay-result.json")
        return result

    # Stage 3: execute or import a candidate.  External commands are argv
    # only and have a strict timeout.  Precomputed mode verifies the candidate
    # bytes and never silently substitutes another result.
    try:
        candidate_result_path, candidate_result = _execute_candidate(
            replay_definition,
            base=base,
            output=output,
            command=command,
        )
        candidate_label = (
            replay_definition.candidate.path
            if replay_definition.candidate is not None
            else "candidate-result.yaml"
        )
        emit_stage(
            "execute_calibration",
            "PASS",
            stage_reason=f"candidate available at {candidate_label}",
            checks={"execution": "PASS"},
            observations={"execution_mode": replay_definition.protocol.execution_mode},
        )
    except Exception as exc:
        status, decision = "BLOCKED", "BLOCKED"
        reason = f"calibration execution blocked: {exc}"
        emit_stage(
            "execute_calibration",
            "BLOCKED",
            stage_reason=reason,
            checks={"execution": "BLOCKED"},
            actions=["provide a digest-bound precomputed result or fix the provider adapter"],
        )
        for remaining in replay_definition.protocol.stages[3:]:
            emit_stage(remaining, "BLOCKED", stage_reason="blocked by execution stage")
        result = _build_replay_result(
            replay_definition,
            status=status,
            decision=decision,
            reason=reason,
            stages=stage_results,
            gates=gates,
            metrics=metrics,
            transform_deltas=transform_deltas,
            artifacts={**stage_artifacts},
            lifecycle_actions=lifecycle_actions,
            recommendations=["execution did not produce a trusted candidate"],
            refs=refs,
            command=command,
            artifact_base=output,
        )
        result.save(output / "replay-result.json")
        return result

    # Stage 4: schema and protocol binding for candidate/baseline/known-bad.
    try:
        assert candidate_result is not None and candidate_result_path is not None
        source_digests = {
            replay_definition.capture_manifest.sha256,
            *(
                source.sha256
                for source in capture.sources
                if source.sha256 is not None
            ),
        }
        _validate_result_protocol(
            candidate_result,
            replay_definition,
            source_digests,
            require_controls=True,
        )
        if replay_definition.baseline is not None:
            baseline_result_path = _resolve(base, replay_definition.baseline.path)
            baseline_result = load_result(baseline_result_path)
            _validate_result_protocol(baseline_result, replay_definition, source_digests)
        if replay_definition.known_bad is not None:
            known_bad_result_path = _resolve(base, replay_definition.known_bad.path)
            known_bad_result = load_result(known_bad_result_path)
            _validate_result_protocol(known_bad_result, replay_definition, source_digests)
        emit_stage(
            "validate_result",
            "PASS",
            stage_reason=(
                "candidate, baseline, and known-bad artifacts are schema-valid and protocol-bound"
            ),
            checks={"candidate_schema": "PASS", "protocol_binding": "PASS"},
        )
    except Exception as exc:
        status, decision = "BLOCKED", "BLOCKED"
        reason = f"result validation blocked: {exc}"
        emit_stage(
            "validate_result",
            "BLOCKED",
            stage_reason=reason,
            checks={"candidate_schema": "BLOCKED"},
            actions=["regenerate the candidate with the exact replay definition and protocol"],
        )
        for remaining in replay_definition.protocol.stages[4:]:
            emit_stage(remaining, "BLOCKED", stage_reason="blocked by result validation stage")
        result = _build_replay_result(
            replay_definition,
            status=status,
            decision=decision,
            reason=reason,
            stages=stage_results,
            gates=gates,
            metrics=metrics,
            transform_deltas=transform_deltas,
            artifacts={**stage_artifacts},
            lifecycle_actions=lifecycle_actions,
            recommendations=["result provenance/protocol binding is incomplete"],
            refs=refs,
            command=command,
            artifact_base=output,
        )
        result.save(output / "replay-result.json")
        return result

    # Stage 5: independent holdout, known-bad, observability, and candidate
    # quality gates.  Missing evidence is BLOCKED rather than a soft warning.
    assert candidate_result is not None
    try:
        _evaluate_candidate_gates(
            replay_definition,
            candidate_result,
            known_bad_result,
            gates,
            metrics,
        )
        failing = [gate for gate in gates if gate.status == "FAIL"]
        blocking = [gate for gate in gates if gate.status == "BLOCKED"]
        if blocking:
            status, decision = "BLOCKED", "BLOCKED"
            reason = "; ".join(gate.reason for gate in blocking)
        elif failing:
            status, decision = "FAIL", "REJECT"
            reason = "; ".join(gate.reason for gate in failing)
        else:
            status, decision = "PASS", "ADOPT"
            reason = "holdout, known-bad, observability, and candidate-quality gates passed"
        emit_stage(
            "evaluate",
            status,
            stage_reason=reason,
            checks={gate.gate_id: gate.status for gate in gates},
            observations=metrics,
            actions=[]
            if status == "PASS"
            else ["hold or reject candidate; incumbent remains unchanged"],
        )
    except Exception as exc:
        status, decision = "BLOCKED", "BLOCKED"
        reason = f"evaluation blocked: {exc}"
        emit_stage(
            "evaluate",
            "BLOCKED",
            stage_reason=reason,
            checks={"evaluation": "BLOCKED"},
            actions=["restore independent holdout and known-bad evidence"],
        )
    # Stage 6: compare candidate to the incumbent.  Comparison is allowed to
    # be unavailable, but adoption is not when a baseline was declared and
    # cannot be compared.
    try:
        if baseline_result is not None and candidate_result is not None:
            comparison = compare_results(
                baseline_result,
                candidate_result,
                left_path=baseline_result_path,
                right_path=candidate_result_path,
            )
            transform_deltas = _comparison_transform_deltas(comparison)
            _apply_transform_budgets(replay_definition.budgets, transform_deltas, gates)
            if any(gate.status == "FAIL" for gate in gates):
                status, decision = "FAIL", "REJECT"
                reason = "; ".join(gate.reason for gate in gates if gate.status == "FAIL")
            elif any(gate.status == "BLOCKED" for gate in gates):
                status, decision = "BLOCKED", "BLOCKED"
                reason = "; ".join(gate.reason for gate in gates if gate.status == "BLOCKED")
            emit_stage(
                "compare",
                status,
                stage_reason="candidate compared to incumbent",
                checks={"baseline_comparison": "PASS" if status in {"PASS", "WARN"} else status},
                observations={"transform_deltas": transform_deltas},
            )
        elif replay_definition.baseline is None:
            emit_stage(
                "compare",
                "PASS",
                stage_reason="no incumbent declared; comparison not applicable",
                checks={"baseline_comparison": "PASS"},
            )
        else:
            raise RawReplayError("baseline was declared but could not be loaded")
    except Exception as exc:
        if status == "PASS":
            status, decision = "BLOCKED", "BLOCKED"
        reason = f"comparison blocked: {exc}"
        emit_stage(
            "compare",
            "BLOCKED",
            stage_reason=reason,
            checks={"baseline_comparison": "BLOCKED"},
            actions=["provide a digest-verified incumbent or explicitly declare no baseline"],
        )

    # Stage 7 lifecycle and optional package plan.  The lifecycle adapter is
    # invoked only when the definition asks for it.  All operations are
    # filesystem event appends; no Autoware package writes occur here.
    if "lifecycle" in replay_definition.protocol.stages:
        if replay_definition.registry is None:
            emit_stage(
                "lifecycle",
                "PASS",
                stage_reason="lifecycle not requested",
                checks={"registry": "PASS"},
            )
        else:
            lifecycle_status, lifecycle_decision, lifecycle_actions = _run_lifecycle_stage(
                replay_definition, base, output, candidate_result_path, status, decision, command
            )
            if lifecycle_status != "PASS" or lifecycle_decision != "ADOPT":
                if status == "BLOCKED" or lifecycle_status == "BLOCKED":
                    status, decision = "BLOCKED", "BLOCKED"
                elif status == "FAIL" or lifecycle_status == "FAIL":
                    status, decision = "FAIL", "REJECT"
                elif lifecycle_status == "WARN" and status == "PASS":
                    status, decision = "WARN", "HOLD"
                reason = "; ".join(
                    [reason, "lifecycle registry did not admit adoption"]
                )
            emit_stage(
                "lifecycle",
                lifecycle_status,
                stage_reason="; ".join(lifecycle_actions),
                checks={"registry": lifecycle_status},
                actions=lifecycle_actions,
            )
    if "autoware_plan" in replay_definition.protocol.stages:
        if replay_definition.autoware is None or not replay_definition.protocol.allow_autoware_plan:
            emit_stage(
                "autoware_plan",
                "PASS",
                stage_reason="Autoware plan not requested; CI remains read-only",
                checks={"package_mutation": "PASS"},
            )
        else:
            try:
                plan_ref = _run_autoware_plan_stage(
                    replay_definition, base, output, candidate_result_path, command
                )
                stage_artifacts["autoware_promotion"] = plan_ref
                emit_stage(
                    "autoware_plan",
                    "PASS",
                    stage_reason="read-only Autoware plan created",
                    checks={"package_mutation": "PASS"},
                    observations={"plan": plan_ref.path},
                )
            except Exception as exc:
                if status == "PASS":
                    status, decision = "BLOCKED", "BLOCKED"
                reason = f"Autoware plan blocked: {exc}"
                emit_stage(
                    "autoware_plan",
                    "BLOCKED",
                    stage_reason=reason,
                    checks={"package_mutation": "BLOCKED"},
                    actions=["inspect package scope and candidate frame bindings"],
                )

    # Ensure every declared stage has an artifact, even if a caller removed an
    # optional stage from a future protocol.  The result remains schema-valid.
    for stage in replay_definition.protocol.stages:
        if not any(item.stage_kind == stage for item in stage_results):
            emit_stage(stage, "BLOCKED", stage_reason="stage was not reached")
    if status == "PASS" and decision == "ADOPT":
        recommendations.append(
            "candidate may be promoted only through an explicit, separately verified Autoware plan"
        )
    else:
        recommendations.append(
            "do not mutate the incumbent or Autoware package; retain the full replay evidence"
        )
    result = _build_replay_result(
        replay_definition,
        status=status,
        decision=decision,
        reason=reason,
        stages=stage_results,
        gates=gates,
        metrics=metrics,
        transform_deltas=transform_deltas,
        artifacts={**stage_artifacts},
        lifecycle_actions=lifecycle_actions,
        recommendations=recommendations,
        refs=refs,
        command=command,
        artifact_base=output,
        candidate_path=candidate_result_path,
        baseline_path=baseline_result_path,
    )
    result.save(output / "replay-result.json")
    return result


def verify_raw_replay(path: str | Path, *, definition: str | Path | None = None) -> ReplayResult:
    """Verify a replay result, all stage artifacts, and optional definition binding."""

    result_path = Path(path)
    result = load_replay_result(result_path)
    replay_definition: ReplayDefinition | None = None
    definition_base: Path | None = None
    if definition is not None:
        replay_definition = load_replay_definition(definition)
        definition_base = Path(definition).parent
        if result.definition_sha256 != replay_definition.artifact_sha256:
            raise RawReplayError("replay result definition digest differs from supplied definition")
        _, input_errors = _verify_definition_inputs(replay_definition, definition_base)
        if input_errors:
            raise RawReplayError(
                "replay definition/input verification failed: " + "; ".join(input_errors)
            )
        _verify_result_definition_binding(result, replay_definition)
    refs_to_verify = [
        *result.artifacts.values(),
        *(stage.artifact for stage in result.stages if stage.artifact is not None),
    ]
    seen_refs: set[tuple[str, str]] = set()
    for ref in refs_to_verify:
        ref_key = (ref.path, ref.sha256)
        if ref_key in seen_refs:
            continue
        seen_refs.add(ref_key)
        if ref.path.startswith("<declared:"):
            continue
        if (
            ref.role == "candidate_result"
            and replay_definition is not None
            and replay_definition.candidate is not None
        ):
            artifact_path = _resolve(
                definition_base or result_path.parent, replay_definition.candidate.path
            )
        elif (
            ref.role == "baseline"
            and replay_definition is not None
            and replay_definition.baseline is not None
        ):
            artifact_path = _resolve(
                definition_base or result_path.parent, replay_definition.baseline.path
            )
        else:
            artifact_path = _resolve(result_path.parent, ref.path)
        observed = sha256_path(artifact_path)
        if observed != ref.sha256:
            raise RawReplayError(f"replay artifact digest mismatch: {ref.path}")
        if ref.role.startswith("stage:"):
            try:
                stage = ReplayStageArtifact.model_validate(read_mapping(artifact_path))
                stage.verify_artifact_digest()
            except Exception as exc:
                raise RawReplayError(f"invalid replay stage artifact {ref.path}: {exc}") from exc
    return result


def _verify_result_definition_binding(
    result: ReplayResult, definition: ReplayDefinition
) -> None:
    """Verify that result references still identify the supplied definition."""

    expected_refs = {
        "config": definition.config.sha256,
        "baseline": definition.baseline.sha256 if definition.baseline is not None else None,
        "candidate": definition.candidate.sha256 if definition.candidate is not None else None,
        "holdout": definition.holdout.sha256 if definition.holdout is not None else None,
        "known_bad": definition.known_bad.sha256 if definition.known_bad is not None else None,
        "environment": definition.environment.environment_sha256,
        "tool": definition.tool.tool_sha256,
    }
    observed_refs = {
        "config": result.provenance.config,
        "baseline": result.provenance.baseline,
        "candidate": result.provenance.candidate,
        "holdout": result.provenance.holdout,
        "known_bad": result.provenance.known_bad,
        "environment": result.provenance.environment,
        "tool": result.provenance.tool,
    }
    for name, expected in expected_refs.items():
        observed = observed_refs[name]
        if expected is None:
            # In-process replay may generate a candidate that was not a
            # declared input.  A baseline, by contrast, must be absent when
            # it was not declared.
            if name == "candidate" and observed is not None:
                continue
            if observed is not None:
                raise RawReplayError(f"replay result unexpectedly declares {name} input")
            continue
        if observed is None or observed.sha256 != expected:
            raise RawReplayError(f"replay result {name} digest is missing or drifted")

    source_digests = {ref.sha256 for ref in result.provenance.source}
    required_sources = [definition.capture_manifest, *definition.inputs]
    for item in required_sources:
        if item.role not in {"capture_manifest", "source"}:
            continue
        if item.sha256 not in source_digests:
            raise RawReplayError(
                f"replay result source digest is missing or drifted for {item.input_id}"
            )
    artifact_candidate = result.artifacts.get("candidate_result")
    if definition.candidate is not None and (
        artifact_candidate is None or artifact_candidate.sha256 != definition.candidate.sha256
    ):
        raise RawReplayError("replay candidate artifact digest is missing or drifted")
    artifact_baseline = result.artifacts.get("baseline_result")
    if definition.baseline is not None and (
        artifact_baseline is None or artifact_baseline.sha256 != definition.baseline.sha256
    ):
        raise RawReplayError("replay baseline artifact digest is missing or drifted")


def compare_raw_replays(
    left: ReplayResult | str | Path,
    right: ReplayResult | str | Path,
    *,
    output: str | Path | None = None,
) -> ReplayComparison:
    """Compare two digest-complete replay artifacts without mutation."""

    left_result = left if isinstance(left, ReplayResult) else load_replay_result(left)
    right_result = right if isinstance(right, ReplayResult) else load_replay_result(right)
    left_path = Path(left) if isinstance(left, (str, Path)) else None
    right_path = Path(right) if isinstance(right, (str, Path)) else None
    metric_deltas: dict[str, float | None] = {}
    for name in sorted(set(left_result.metrics) | set(right_result.metrics)):
        lvalue, rvalue = left_result.metrics.get(name), right_result.metrics.get(name)
        metric_deltas[name] = None if lvalue is None or rvalue is None else rvalue - lvalue
    transform_deltas = dict(right_result.transform_deltas)
    if left_result.status == "BLOCKED" or right_result.status == "BLOCKED":
        status: ReplayStatus = "BLOCKED"
        decision: Literal["BETTER", "WORSE", "TIE", "NOT_COMPARABLE"] = "NOT_COMPARABLE"
        reasons = ["blocked replay cannot support a comparison"]
    else:
        status = "PASS"
        positive = [value for value in metric_deltas.values() if value is not None and value > 0.0]
        negative = [value for value in metric_deltas.values() if value is not None and value < 0.0]
        if positive and not negative:
            decision = "WORSE"
        elif negative and not positive:
            decision = "BETTER"
        else:
            decision = "TIE"
        reasons = ["comparison is descriptive; metric direction is provider-specific"]
    left_ref = ReplayDigestRef(
        role="replay_result",
        path=str(left_path) if left_path else "<memory>",
        sha256=left_result.artifact_sha256,
        schema_version=RAW_REPLAY_RESULT_SCHEMA_VERSION,
    )
    right_ref = ReplayDigestRef(
        role="replay_result",
        path=str(right_path) if right_path else "<memory>",
        sha256=right_result.artifact_sha256,
        schema_version=RAW_REPLAY_RESULT_SCHEMA_VERSION,
    )
    comparison = ReplayComparison(
        left=left_ref,
        right=right_ref,
        status=status,
        decision=decision,
        metric_deltas=metric_deltas,
        transform_deltas=transform_deltas,
        reasons=reasons,
        provenance=_provenance_for_result(left_result, command=["calibrex", "replay", "compare"]),
        artifact_sha256="0" * 64,
    ).with_artifact_digest()
    if output is not None:
        comparison.save(output)
    return comparison


def run_field_replacement_pilot(
    definition: ReplayDefinition | str | Path,
    *,
    output_directory: str | Path,
    command: Sequence[str] = (),
) -> FieldReplacementPilot:
    """Run a replay and record a vehicle/sensor replacement lifecycle pilot.

    Existing registries are opened and verified by the lifecycle API.  A new
    registry is initialized when the declared root does not exist.  The pilot
    may append lifecycle events, but never calls an Autoware apply operation.
    """

    definition_path = Path(definition) if isinstance(definition, (str, Path)) else None
    replay_definition = (
        definition
        if isinstance(definition, ReplayDefinition)
        else load_replay_definition(definition)
    )
    if replay_definition.registry is None:
        raise RawReplayError("field replacement pilot requires definition.registry")
    registry_request = replay_definition.registry
    base = definition_path.parent if definition_path is not None else Path.cwd()
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    replay_result = run_raw_replay(
        definition_path or replay_definition,
        output_directory=output,
        command=command,
    )
    candidate_ref = replay_result.artifacts.get("candidate_result")
    candidate_path = (
        _resolve(base, replay_definition.candidate.path)
        if replay_definition.candidate is not None
        else output / "candidate-result.yaml"
    )
    events: list[str] = []
    lifecycle_status: ReplayStatus = replay_result.status
    lifecycle_decision: ReplayDecision = replay_result.decision
    lifecycle_reason = replay_result.reason
    registry_root = _resolve(base, registry_request.registry_root)
    try:
        if not (registry_root / "registry.json").exists():
            init_registry(
                registry_root,
                registry_id=f"field-pilot-{registry_request.vehicle_id}",
                operator=registry_request.operator,
                command=list(command),
            )
        # Registration is idempotent at the pilot boundary: an already
        # registered identity is retained and not rewritten.
        from calibrex.core.lifecycle_registry import load_registry

        state = load_registry(registry_root).state()
        if registry_request.vehicle_id not in state.vehicles:
            event = register_vehicle(
                registry_root,
                registry_request.vehicle_id,
                operator=registry_request.operator,
                command=list(command),
            )
            events.append(event.event_sha256)
        elif state.vehicles[registry_request.vehicle_id].status != "active":
            raise LifecycleRegistryError(
                f"vehicle is not active: {registry_request.vehicle_id}"
            )
        state = load_registry(registry_root).state()
        if registry_request.sensor_kit_id not in state.sensor_kits:
            event = register_sensor_kit(
                registry_root,
                registry_request.sensor_kit_id,
                registry_request.vehicle_id,
                operator=registry_request.operator,
                command=list(command),
            )
            events.append(event.event_sha256)
        elif (
            state.sensor_kits[registry_request.sensor_kit_id].vehicle_id
            != registry_request.vehicle_id
        ):
            raise LifecycleRegistryError("sensor kit belongs to another vehicle")
        state = load_registry(registry_root).state()
        if registry_request.sensor_id not in state.sensors:
            event = register_sensor(
                registry_root,
                sensor_id=registry_request.sensor_id,
                vehicle_id=registry_request.vehicle_id,
                sensor_kit_id=registry_request.sensor_kit_id,
                serial=registry_request.serial,
                model=registry_request.model,
                firmware=registry_request.firmware,
                mount=registry_request.mount,
                frame=registry_request.frame,
                install=True,
                operator=registry_request.operator,
                command=list(command),
            )
            events.append(event.event_sha256)
        else:
            baseline = state.sensors[registry_request.sensor_id]
            expected_identity = (
                registry_request.vehicle_id,
                registry_request.sensor_kit_id,
                registry_request.serial,
                registry_request.model,
                registry_request.firmware,
                registry_request.mount,
            )
            observed_identity = (
                baseline.vehicle_id,
                baseline.sensor_kit_id,
                baseline.serial,
                baseline.model,
                baseline.firmware,
                baseline.mount,
            )
            if observed_identity != expected_identity:
                raise LifecycleRegistryError(
                    f"baseline sensor identity mismatch for {registry_request.sensor_id}"
                )
        replacement_serial = registry_request.replacement_serial or registry_request.serial
        replacement_mount = registry_request.replacement_mount or registry_request.mount
        replacement_changed = (
            replacement_serial != registry_request.serial
            or replacement_mount != registry_request.mount
        )
        if replacement_changed:
            # A serial/mount change is a new physical identity.  Preserve the
            # baseline sensor history, then register the replacement under an
            # explicit ID; install_sensor intentionally rejects identity drift.
            state = load_registry(registry_root).state()
            baseline = state.sensors[registry_request.sensor_id]
            if baseline.state != "removed":
                event = remove_sensor(
                    registry_root,
                    sensor_id=registry_request.sensor_id,
                    operator=registry_request.operator,
                    reason="field replacement baseline removal",
                    command=list(command),
                )
                events.append(event.event_sha256)
            replacement_id = (
                registry_request.replacement_sensor_id
                or f"{registry_request.sensor_id}-replacement"
            )
            state = load_registry(registry_root).state()
            replacement = state.sensors.get(replacement_id)
            if replacement is None:
                event = register_sensor(
                    registry_root,
                    sensor_id=replacement_id,
                    vehicle_id=registry_request.vehicle_id,
                    sensor_kit_id=registry_request.sensor_kit_id,
                    serial=replacement_serial,
                    model=registry_request.model,
                    firmware=registry_request.firmware,
                    mount=replacement_mount,
                    frame=registry_request.frame,
                    install=True,
                    operator=registry_request.operator,
                    reason="field replacement installation",
                    command=list(command),
                )
                events.append(event.event_sha256)
            elif (
                replacement.serial != replacement_serial
                or replacement.mount != replacement_mount
                or replacement.model != registry_request.model
                or replacement.firmware != registry_request.firmware
            ):
                raise LifecycleRegistryError(
                    f"replacement sensor identity mismatch for {replacement_id}"
                )
            elif replacement.state == "removed":
                event = install_sensor(
                    registry_root,
                    sensor_id=replacement_id,
                    serial=replacement_serial,
                    model=registry_request.model,
                    firmware=registry_request.firmware,
                    mount=replacement_mount,
                    frame=registry_request.frame,
                    operator=registry_request.operator,
                    reason="field replacement remount",
                    command=list(command),
                )
                events.append(event.event_sha256)
        else:
            state = load_registry(registry_root).state()
            baseline = state.sensors[registry_request.sensor_id]
            if baseline.state == "removed":
                event = install_sensor(
                    registry_root,
                    sensor_id=registry_request.sensor_id,
                    serial=registry_request.serial,
                    model=registry_request.model,
                    firmware=registry_request.firmware,
                    mount=registry_request.mount,
                    frame=registry_request.frame,
                    operator=registry_request.operator,
                    reason="field sensor remount",
                    command=list(command),
                )
                events.append(event.event_sha256)
        state = load_registry(registry_root).state()
        if registry_request.edge_id not in state.edges:
            event = register_calibration_edge(
                registry_root,
                edge_id=registry_request.edge_id,
                vehicle_id=registry_request.vehicle_id,
                sensor_kit_id=registry_request.sensor_kit_id,
                parent_frame=registry_request.parent_frame,
                child_frame=registry_request.child_frame,
                operator=registry_request.operator,
                command=list(command),
            )
            events.append(event.event_sha256)
        else:
            edge = state.edges[registry_request.edge_id]
            if (
                edge.vehicle_id != registry_request.vehicle_id
                or edge.sensor_kit_id != registry_request.sensor_kit_id
                or edge.parent_frame != registry_request.parent_frame
                or edge.child_frame != registry_request.child_frame
            ):
                raise LifecycleRegistryError(
                    f"calibration edge identity mismatch for {registry_request.edge_id}"
                )
        # Record the capture only once.  It is allowed to be BLOCKED, but that
        # fact remains visible in the append-only registry.
        state = load_registry(registry_root).state()
        capture_id = _capture_id_from_definition(replay_definition, base=base)
        if capture_id not in state.captures:
            try:
                event = record_capture(
                    registry_root,
                    capture_manifest=_resolve(base, replay_definition.capture_manifest.path),
                    operator=registry_request.operator,
                    command=list(command),
                )
                events.append(event.event_sha256)
            except LifecycleRegistryError as exc:
                events.append(f"capture lifecycle record blocked: {exc}")
        # A replay result is always an evaluation input.  The lifecycle API
        # itself decides whether candidate evidence is admissible.
        if (
            candidate_ref is not None
            and candidate_path.exists()
            and replay_result.status in {"PASS", "WARN", "FAIL"}
        ):
            evaluation_path = output / "lifecycle-evaluation.json"
            evaluation = evaluate_lifecycle(
                registry_root,
                edge_id=registry_request.edge_id,
                capture_manifest=_resolve(base, replay_definition.capture_manifest.path),
                candidate_result=candidate_path,
                output=evaluation_path,
                operator=registry_request.operator,
                reason="raw replay field replacement evaluation",
                command=list(command),
            )
            events.extend(_event_digests_from_evaluation(registry_root, evaluation.evaluation_id))
            lifecycle_status, lifecycle_decision = _lifecycle_decision(
                evaluation.status, evaluation.admission
            )
            lifecycle_reason = evaluation.reason
            if lifecycle_decision == "ADOPT" and replay_result.decision != "ADOPT":
                lifecycle_status, lifecycle_decision = replay_result.status, replay_result.decision
                lifecycle_reason = (
                    "replay gates did not admit adoption; lifecycle evaluation cannot upgrade "
                    "the replay decision"
                )
    except Exception as exc:
        lifecycle_status, lifecycle_decision = "BLOCKED", "BLOCKED"
        lifecycle_reason = f"field replacement lifecycle blocked: {exc}"
        events.append(lifecycle_reason)
    try:
        verification = verify_registry(registry_root, verify_sources=True, strict=True)
        if not verification.valid:
            raise LifecycleRegistryError(
                "registry verification failed: " + "; ".join(verification.errors)
            )
    except Exception as exc:
        lifecycle_status, lifecycle_decision = "BLOCKED", "BLOCKED"
        lifecycle_reason = f"field replacement registry verification blocked: {exc}"
        events.append(lifecycle_reason)
    replacement_serial = registry_request.replacement_serial or registry_request.serial
    replacement_mount = registry_request.replacement_mount or registry_request.mount
    rollback = RollbackSimulation(
        attempted=registry_request.simulate_rollback,
        package_mutated=False,
        status="PASS" if registry_request.simulate_rollback else "NOT_RUN",
        reason=(
            "rollback simulation records no package mutation; an explicit prior "
            "promotion is required for an actual rollback"
        )
        if registry_request.simulate_rollback
        else "rollback simulation disabled",
    )
    pilot = FieldReplacementPilot(
        pilot_id=f"field-replacement-{replay_definition.replay_id}",
        replay_id=replay_definition.replay_id,
        registry_root=str(registry_root),
        vehicle_id=registry_request.vehicle_id,
        sensor_kit_id=registry_request.sensor_kit_id,
        edge_id=registry_request.edge_id,
        sensor_id=registry_request.sensor_id,
        baseline_serial=registry_request.serial,
        replacement_serial=replacement_serial,
        baseline_mount=registry_request.mount,
        replacement_mount=replacement_mount,
        remount_count=1
        if replacement_serial != registry_request.serial
        or replacement_mount != registry_request.mount
        else 0,
        candidate=candidate_ref,
        lifecycle_event_digests=events,
        status=lifecycle_status,
        decision=lifecycle_decision,
        reason=lifecycle_reason,
        rollback=rollback,
        package_mutated=False,
        operator_recommendations=[
            "replace/remount identity is recorded; do not apply a package patch "
            "until PASS/ADOPT promotion and smoke evidence exist"
        ],
        provenance=_provenance_for_definition(replay_definition, command=list(command)),
        artifact_sha256="0" * 64,
    ).with_artifact_digest()
    pilot.save(output / "field-replacement-pilot.json")
    return pilot


def _build_replay_result(
    definition: ReplayDefinition,
    *,
    status: ReplayStatus,
    decision: ReplayDecision,
    reason: str,
    stages: Sequence[ReplayStageResult],
    gates: Sequence[ReplayGate],
    metrics: Mapping[str, float | None],
    transform_deltas: Mapping[str, Mapping[str, float]],
    artifacts: Mapping[str, ReplayDigestRef],
    lifecycle_actions: Sequence[str],
    recommendations: Sequence[str],
    refs: Sequence[ReplayDigestRef],
    command: Sequence[str],
    candidate_path: Path | None = None,
    baseline_path: Path | None = None,
    candidate_ref_path: str | None = None,
    baseline_ref_path: str | None = None,
    artifact_base: Path | None = None,
) -> ReplayResult:
    source_refs = [
        ref
        for ref in refs
        if ref.role
        not in {
            "config",
            "baseline",
            "candidate",
            "holdout",
            "known_bad",
            "environment",
            "tool",
            "container",
        }
    ]
    config_ref = next((ref for ref in refs if ref.role == "config"), None)
    baseline_ref = next((ref for ref in refs if ref.role == "baseline"), None)
    candidate_ref = artifacts.get("candidate_result")
    declared_candidate_ref = next((ref for ref in refs if ref.role == "candidate"), None)
    if candidate_ref is None and declared_candidate_ref is not None:
        candidate_ref = declared_candidate_ref.model_copy(update={"role": "candidate_result"})
    if candidate_ref is None and candidate_path is not None:
        digest = sha256_path(candidate_path)
        if digest is not None:
            ref_base = artifact_base or Path.cwd()
            candidate_ref = ReplayDigestRef(
                role="candidate_result",
                path=candidate_ref_path or _portable_path(ref_base, candidate_path),
                sha256=digest,
                size_bytes=candidate_path.stat().st_size if candidate_path.is_file() else None,
                schema_version="slac.result/v0.1",
            )
    declared_baseline_ref = next((ref for ref in refs if ref.role == "baseline"), None)
    if declared_baseline_ref is not None:
        baseline_ref = declared_baseline_ref
    if baseline_ref is None and baseline_path is not None:
        digest = sha256_path(baseline_path)
        if digest is not None:
            ref_base = artifact_base or Path.cwd()
            baseline_ref = ReplayDigestRef(
                role="baseline",
                path=baseline_ref_path or _portable_path(ref_base, baseline_path),
                sha256=digest,
                size_bytes=baseline_path.stat().st_size if baseline_path.is_file() else None,
                schema_version="slac.result/v0.1",
            )
    all_artifacts = dict(artifacts)
    if candidate_ref is not None:
        all_artifacts["candidate_result"] = candidate_ref
    if baseline_ref is not None:
        all_artifacts["baseline_result"] = baseline_ref
    generated = _provenance_for_definition(
        definition,
        command=list(command),
        candidate_sha256=candidate_ref.sha256 if candidate_ref else None,
        baseline_sha256=baseline_ref.sha256 if baseline_ref else None,
        evidence_sha256={
            key: ref.sha256
            for key, ref in all_artifacts.items()
            if ref.role.startswith("stage:") or ref.role == "autoware_promotion"
        },
    )
    environment_ref = ReplayDigestRef(
        role="environment",
        path=f"<declared:{definition.environment.environment_id}>",
        sha256=definition.environment.environment_sha256,
    )
    tool_ref = ReplayDigestRef(
        role="tool",
        path=f"<declared:{definition.tool.name}@{definition.tool.version}>",
        sha256=definition.tool.tool_sha256,
    )
    container_ref = (
        ReplayDigestRef(
            role="container",
            path=f"<declared:{definition.environment.container_image or 'container'}>",
            sha256=definition.environment.container_digest.removeprefix("sha256:"),
        )
        if definition.environment.container_digest is not None
        else None
    )
    return ReplayResult(
        replay_id=definition.replay_id,
        definition_sha256=definition.artifact_sha256,
        status=status,
        decision=decision,
        reason=reason,
        stages=list(stages),
        gates=list(gates),
        metrics=dict(metrics),
        transform_deltas={key: dict(value) for key, value in transform_deltas.items()},
        artifacts=all_artifacts,
        lifecycle_actions=list(lifecycle_actions),
        operator_recommendations=list(recommendations),
        provenance=ReplayProvenanceSummary(
            source=source_refs,
            config=config_ref,
            baseline=baseline_ref,
            candidate=candidate_ref,
            holdout=next((ref for ref in refs if ref.role == "holdout"), None),
            known_bad=next((ref for ref in refs if ref.role == "known_bad"), None),
            environment=environment_ref,
            tool=tool_ref,
            container=container_ref,
            evidence=[
                ref
                for ref in all_artifacts.values()
                if ref.role.startswith("stage:") or ref.role == "autoware_promotion"
            ],
        ),
        generated_provenance=generated,
        artifact_sha256="0" * 64,
    ).with_artifact_digest()


def _verify_definition_inputs(
    definition: ReplayDefinition, base: Path
) -> tuple[list[ReplayDigestRef], list[str]]:
    inputs = [definition.capture_manifest, definition.config, *definition.inputs]
    for item in (
        definition.baseline,
        definition.candidate,
        definition.holdout,
        definition.known_bad,
    ):
        if item is not None:
            inputs.append(item)
    refs: list[ReplayDigestRef] = []
    errors: list[str] = []
    for item in inputs:
        path = _resolve(base, item.path)
        observed = sha256_path(path)
        if observed is None:
            required = item.required or item.role in {
                "capture_manifest",
                "config",
                "baseline",
                "candidate",
                "holdout",
                "known_bad",
            }
            if required:
                errors.append(f"missing {item.role} input {item.input_id}: {item.path}")
            continue
        if observed != item.sha256:
            errors.append(
                f"{item.role} input digest mismatch for {item.input_id}: "
                f"expected {item.sha256}, observed {observed}"
            )
        refs.append(
            ReplayDigestRef(
                role=item.role,
                path=_portable_path(base, path),
                sha256=observed,
                size_bytes=path.stat().st_size if path.is_file() else None,
                schema_version=item.schema_version,
            )
        )
        if item.role == "capture_manifest":
            # The manifest is an inventory, not a substitute for its raw
            # source digests.  Surface each source in the replay plan/result
            # and fail the plan when a source has drifted.
            try:
                manifest = load_capture_manifest(path, verify=True)
                verification = verify_capture_manifest_inputs(path)
                if not verification.valid:
                    errors.append(f"capture input verification failed: {verification.summary}")
                if manifest.status != "ready":
                    errors.append(f"capture status is {manifest.status}, not ready")
                if (
                    manifest.provenance.config_sha256 is not None
                    and manifest.provenance.config_sha256 != definition.config.sha256
                ):
                    errors.append("capture manifest config digest differs from replay definition")
                for source in manifest.sources:
                    source_path = _resolve(path.parent, source.path)
                    source_digest = sha256_path(source_path)
                    if source_digest is None:
                        errors.append(f"missing capture source {source.source_id}: {source.path}")
                        continue
                    if source.sha256 is not None and source_digest != source.sha256:
                        errors.append(f"capture source digest mismatch for {source.source_id}")
                    refs.append(
                        ReplayDigestRef(
                            role="source",
                            path=_portable_path(base, source_path),
                            sha256=source_digest,
                            size_bytes=source_path.stat().st_size
                            if source_path.is_file()
                            else None,
                        )
                    )
                    if source.metadata_path is not None:
                        metadata_path = _resolve(path.parent, source.metadata_path)
                        metadata_digest = sha256_path(metadata_path)
                        if metadata_digest is None:
                            errors.append(
                                f"missing capture metadata {source.source_id}: "
                                f"{source.metadata_path}"
                            )
                        elif source.metadata_sha256 is not None and (
                            metadata_digest != source.metadata_sha256
                        ):
                            errors.append(
                                f"capture metadata digest mismatch for {source.source_id}"
                            )
                        if metadata_digest is not None:
                            refs.append(
                                ReplayDigestRef(
                                    role="capture_metadata",
                                    path=_portable_path(base, metadata_path),
                                    sha256=metadata_digest,
                                    size_bytes=metadata_path.stat().st_size
                                    if metadata_path.is_file()
                                    else None,
                                )
                            )
                if manifest.provenance.config_path is not None:
                    manifest_config_path = _resolve(
                        path.parent, manifest.provenance.config_path
                    )
                    manifest_config_digest = sha256_path(manifest_config_path)
                    if manifest_config_digest is not None:
                        refs.append(
                            ReplayDigestRef(
                                role="capture_config",
                                path=_portable_path(base, manifest_config_path),
                                sha256=manifest_config_digest,
                                size_bytes=manifest_config_path.stat().st_size
                                if manifest_config_path.is_file()
                                else None,
                            )
                        )
            except Exception as exc:
                errors.append(f"capture manifest input verification failed: {exc}")
    # A declared executable/environment input is also digest checked.  The
    # declaration's digest remains part of provenance even when the path is
    # not available on a different CI runner.
    for path_text, expected, _role in (
        (definition.tool.executable_path, definition.tool.executable_sha256, "tool"),
    ):
        if path_text is None:
            continue
        observed = sha256_path(_resolve(base, path_text))
        if observed != expected:
            errors.append(
                f"tool executable digest mismatch: expected {expected}, observed {observed}"
            )
    return refs, errors


def _execute_candidate(
    definition: ReplayDefinition, *, base: Path, output: Path, command: Sequence[str]
) -> tuple[Path, CalibrationResult]:
    mode = definition.protocol.execution_mode
    if mode == "precomputed":
        assert definition.candidate is not None
        path = _resolve(base, definition.candidate.path)
        result = load_result(path)
        if definition.candidate.sha256 != sha256_path(path):
            raise RawReplayError("precomputed candidate digest changed after input verification")
        return path, result
    if mode == "in_process":
        in_process_result = run_calibration(
            _resolve(base, definition.config.path),
            CalibrationRunOptions(
                output_dir=output / "calibration", seed=definition.protocol.deterministic_seed
            ),
        )
        if in_process_result is None:
            raise RawReplayError("in-process calibration returned no result")
        in_process_result.run.provenance["replay_definition_sha256"] = definition.artifact_sha256
        in_process_result.run.provenance["replay_protocol_id"] = definition.protocol.protocol_id
        in_process_result.run.provenance["replay_seed"] = definition.protocol.deterministic_seed
        in_process_result.run.provenance[
            "replay_capture_manifest_sha256"
        ] = definition.capture_manifest.sha256
        if definition.holdout is not None:
            in_process_result.run.provenance["replay_holdout_sha256"] = definition.holdout.sha256
        if definition.known_bad is not None:
            in_process_result.run.provenance["replay_known_bad_sha256"] = (
                definition.known_bad.sha256
            )
        path = output / "candidate-result.yaml"
        in_process_result.save(path)
        return path, in_process_result
    if not definition.tool.command:
        raise RawReplayError("argv execution requires tool.command")
    argv = list(definition.tool.command)
    if any(not item or "\x00" in item for item in argv):
        raise RawReplayError("argv execution rejected empty/NUL command argument")
    if definition.tool.executable_path is not None:
        declared_executable = _resolve(base, definition.tool.executable_path)
        resolved_command = Path(argv[0])
        if not resolved_command.is_absolute():
            found = shutil.which(argv[0])
            resolved_command = Path(found) if found is not None else _resolve(base, argv[0])
        if resolved_command.resolve() != declared_executable.resolve():
            raise RawReplayError(
                "argv executable does not match the digest-bound executable_path"
            )
    timeout = definition.protocol.stage_timeout_seconds.get("execute_calibration", 3_600.0)
    completed = subprocess.run(
        argv,
        cwd=base,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        shell=False,
    )
    (output / "provider.stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output / "provider.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RawReplayError(f"provider exited with code {completed.returncode}")
    if definition.candidate is None:
        raise RawReplayError("argv execution requires a candidate result declaration")
    path = _resolve(base, definition.candidate.path)
    return path, load_result(path)


def _validate_result_protocol(
    result: CalibrationResult,
    definition: ReplayDefinition,
    source_digests: set[str],
    *,
    require_controls: bool = False,
) -> None:
    """Require result metadata to remain bound to the replay contract."""

    provenance = result.run.provenance
    declared_definition = provenance.get("replay_definition_sha256")
    # An in-process/argv provider can bind the result to the exact definition
    # after the definition digest is known.  A precomputed result is an
    # immutable input to the definition, so requiring the definition digest
    # inside that input would create an impossible self-referential hash.  In
    # that mode the candidate file digest and protocol identifier are the
    # binding; a supplied definition digest is still checked strictly.
    if declared_definition is not None and declared_definition != definition.artifact_sha256:
        raise RawReplayError("candidate replay_definition_sha256 is missing or drifted")
    declared_protocol = provenance.get("replay_protocol_id")
    if declared_protocol != definition.protocol.protocol_id:
        raise RawReplayError("candidate replay protocol ID is missing or drifted")
    if result.run.config_sha256 != definition.config.sha256:
        raise RawReplayError("candidate config digest differs from replay definition")
    dataset_digest = result.run.dataset_sha256
    if dataset_digest is None or dataset_digest not in source_digests:
        raise RawReplayError("candidate dataset digest is missing or not bound to capture sources")
    declared_capture = provenance.get("replay_capture_manifest_sha256")
    if require_controls and declared_capture != definition.capture_manifest.sha256:
        raise RawReplayError("candidate capture manifest digest is missing or drifted")
    if declared_capture is not None and declared_capture != definition.capture_manifest.sha256:
        raise RawReplayError("candidate capture manifest digest is missing or drifted")
    declared_holdout = provenance.get("replay_holdout_sha256")
    if require_controls and declared_holdout != (
        definition.holdout.sha256 if definition.holdout is not None else None
    ):
        raise RawReplayError("candidate holdout digest is missing or drifted")
    if declared_holdout is not None and (
        definition.holdout is None or declared_holdout != definition.holdout.sha256
    ):
        raise RawReplayError("candidate holdout digest is missing or drifted")
    declared_known_bad = provenance.get("replay_known_bad_sha256")
    if require_controls and declared_known_bad != (
        definition.known_bad.sha256 if definition.known_bad is not None else None
    ):
        raise RawReplayError("candidate known_bad digest is missing or drifted")
    if declared_known_bad is not None and (
        definition.known_bad is None or declared_known_bad != definition.known_bad.sha256
    ):
        raise RawReplayError("candidate known_bad digest is missing or drifted")


def _evaluate_candidate_gates(
    definition: ReplayDefinition,
    candidate: CalibrationResult,
    known_bad: CalibrationResult | None,
    gates: list[ReplayGate],
    metrics: dict[str, float | None],
) -> None:
    holdout = _metric_value(
        candidate.metrics.get(definition.budgets.holdout_metric), definition.budgets.holdout_field
    )
    metrics[definition.budgets.holdout_metric] = holdout
    if definition.protocol.require_independent_holdout and definition.holdout is None:
        gates.append(
            ReplayGate(
                gate_id="independent_holdout",
                status="BLOCKED",
                expected=True,
                observed=False,
                reason="independent holdout input is missing",
            )
        )
    elif holdout is None:
        gates.append(
            ReplayGate(
                gate_id="independent_holdout",
                status="BLOCKED",
                expected="finite metric",
                observed=None,
                reason=f"candidate metric {definition.budgets.holdout_metric} is missing",
            )
        )
    elif (
        definition.budgets.max_holdout_value is not None
        and holdout > definition.budgets.max_holdout_value
    ):
        gates.append(
            ReplayGate(
                gate_id="independent_holdout",
                status="FAIL",
                expected=definition.budgets.max_holdout_value,
                observed=holdout,
                reason="candidate holdout metric exceeds budget",
            )
        )
    else:
        gates.append(
            ReplayGate(
                gate_id="independent_holdout",
                status="PASS",
                expected=definition.budgets.max_holdout_value,
                observed=holdout,
                reason="candidate holdout metric passed",
            )
        )
    bad_value = (
        _metric_value(
            known_bad.metrics.get(definition.budgets.known_bad_metric),
            definition.budgets.known_bad_field,
        )
        if known_bad is not None
        else None
    )
    metrics["known_bad_value"] = bad_value
    delta = None if holdout is None or bad_value is None else bad_value - holdout
    metrics["known_bad_delta"] = delta
    if definition.protocol.require_known_bad and known_bad is None:
        gates.append(
            ReplayGate(
                gate_id="known_bad_controls",
                status="BLOCKED",
                expected=True,
                observed=False,
                reason="known-bad control input is missing",
            )
        )
    elif delta is None:
        gates.append(
            ReplayGate(
                gate_id="known_bad_controls",
                status="BLOCKED",
                expected="finite delta",
                observed=None,
                reason="known-bad metric comparison is unavailable",
            )
        )
    elif (
        definition.budgets.min_known_bad_delta is not None
        and delta < definition.budgets.min_known_bad_delta
    ):
        gates.append(
            ReplayGate(
                gate_id="known_bad_controls",
                status="FAIL",
                expected=definition.budgets.min_known_bad_delta,
                observed=delta,
                reason="known-bad control was not separated from candidate",
            )
        )
    else:
        gates.append(
            ReplayGate(
                gate_id="known_bad_controls",
                status="PASS",
                expected=definition.budgets.min_known_bad_delta,
                observed=delta,
                reason="known-bad control was separated",
            )
        )
    rank = candidate.observability.rank
    condition_number = candidate.observability.condition_number
    if definition.protocol.require_observability and (
        rank is None
        or rank < 0
        or rank > 6
        or rank < definition.budgets.min_observability_rank
        or condition_number is not None
        and (not math.isfinite(condition_number) or condition_number <= 0.0)
        or candidate.observability.grade != "pass"
    ):
        gates.append(
            ReplayGate(
                gate_id="observability",
                status="BLOCKED",
                expected=definition.budgets.min_observability_rank,
                observed=rank,
                reason="observability is weak or unavailable",
            )
        )
    else:
        gates.append(
            ReplayGate(
                gate_id="observability",
                status="PASS",
                expected=definition.budgets.min_observability_rank,
                observed=rank,
                reason="observability passed",
            )
        )
    quality = candidate.quality.grade
    if definition.budgets.require_candidate_quality_pass and quality != "pass":
        gates.append(
            ReplayGate(
                gate_id="candidate_quality",
                status="FAIL",
                expected="pass",
                observed=quality,
                reason="candidate quality is not PASS",
            )
        )
    else:
        gates.append(
            ReplayGate(
                gate_id="candidate_quality",
                status="PASS",
                expected="pass",
                observed=quality,
                reason="candidate quality passed",
            )
        )


def _apply_transform_budgets(
    budgets: ReplayBudgets, deltas: Mapping[str, Mapping[str, float]], gates: list[ReplayGate]
) -> None:
    translation = max(
        (abs(value.get("translation_delta_m", 0.0)) for value in deltas.values()), default=0.0
    )
    rotation = max(
        (abs(value.get("rotation_delta_deg", 0.0)) for value in deltas.values()), default=0.0
    )
    if (
        budgets.max_transform_translation_delta_m is not None
        and translation > budgets.max_transform_translation_delta_m
    ):
        gates.append(
            ReplayGate(
                gate_id="transform_translation_budget",
                status="FAIL",
                expected=budgets.max_transform_translation_delta_m,
                observed=translation,
                reason="candidate transform moved beyond translation budget",
            )
        )
    else:
        gates.append(
            ReplayGate(
                gate_id="transform_translation_budget",
                status="PASS",
                expected=budgets.max_transform_translation_delta_m,
                observed=translation,
                reason="candidate transform translation budget passed",
            )
        )
    if (
        budgets.max_transform_rotation_delta_deg is not None
        and rotation > budgets.max_transform_rotation_delta_deg
    ):
        gates.append(
            ReplayGate(
                gate_id="transform_rotation_budget",
                status="FAIL",
                expected=budgets.max_transform_rotation_delta_deg,
                observed=rotation,
                reason="candidate transform moved beyond rotation budget",
            )
        )
    else:
        gates.append(
            ReplayGate(
                gate_id="transform_rotation_budget",
                status="PASS",
                expected=budgets.max_transform_rotation_delta_deg,
                observed=rotation,
                reason="candidate transform rotation budget passed",
            )
        )


def _comparison_transform_deltas(comparison: ResultComparison) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for group in comparison.transform_groups.values():
        for item in group.comparisons:
            output[f"{group.group}:{item.name}"] = {
                "translation_delta_m": item.translation_delta_m,
                "rotation_delta_deg": item.rotation_delta_deg,
            }
    return output


def _run_lifecycle_stage(
    definition: ReplayDefinition,
    base: Path,
    output: Path,
    candidate_path: Path | None,
    status: ReplayStatus,
    decision: ReplayDecision,
    command: Sequence[str],
) -> tuple[ReplayStatus, ReplayDecision, list[str]]:
    request = definition.registry
    assert request is not None
    actions: list[str] = []
    root = _resolve(base, request.registry_root)
    try:
        from calibrex.core.lifecycle_registry import load_registry

        if not (root / "registry.json").exists():
            init_registry(
                root,
                registry_id=f"replay-{request.vehicle_id}",
                operator=request.operator,
                command=list(command),
            )
        state = load_registry(root).state()
        if request.vehicle_id not in state.vehicles:
            register_vehicle(
                root, request.vehicle_id, operator=request.operator, command=list(command)
            )
        elif state.vehicles[request.vehicle_id].status != "active":
            raise LifecycleRegistryError(f"vehicle is not active: {request.vehicle_id}")
        state = load_registry(root).state()
        if request.sensor_kit_id not in state.sensor_kits:
            register_sensor_kit(
                root,
                request.sensor_kit_id,
                request.vehicle_id,
                operator=request.operator,
                command=list(command),
            )
        elif (
            state.sensor_kits[request.sensor_kit_id].vehicle_id != request.vehicle_id
        ):
            raise LifecycleRegistryError("sensor kit belongs to another vehicle")
        state = load_registry(root).state()
        if request.sensor_id not in state.sensors:
            register_sensor(
                root,
                sensor_id=request.sensor_id,
                vehicle_id=request.vehicle_id,
                sensor_kit_id=request.sensor_kit_id,
                serial=request.serial,
                model=request.model,
                firmware=request.firmware,
                mount=request.mount,
                frame=request.frame,
                install=True,
                operator=request.operator,
                command=list(command),
            )
        else:
            sensor = state.sensors[request.sensor_id]
            expected_identity = (
                request.vehicle_id,
                request.sensor_kit_id,
                request.serial,
                request.model,
                request.firmware,
                request.mount,
            )
            observed_identity = (
                sensor.vehicle_id,
                sensor.sensor_kit_id,
                sensor.serial,
                sensor.model,
                sensor.firmware,
                sensor.mount,
            )
            if observed_identity != expected_identity:
                raise LifecycleRegistryError(
                    f"baseline sensor identity mismatch for {request.sensor_id}"
                )
            if sensor.state != "installed":
                raise LifecycleRegistryError(
                    f"baseline sensor is not installed: {request.sensor_id}"
                )
        state = load_registry(root).state()
        if request.edge_id not in state.edges:
            register_calibration_edge(
                root,
                edge_id=request.edge_id,
                vehicle_id=request.vehicle_id,
                sensor_kit_id=request.sensor_kit_id,
                parent_frame=request.parent_frame,
                child_frame=request.child_frame,
                operator=request.operator,
                command=list(command),
            )
        else:
            edge = state.edges[request.edge_id]
            if (
                edge.vehicle_id != request.vehicle_id
                or edge.sensor_kit_id != request.sensor_kit_id
                or edge.parent_frame != request.parent_frame
                or edge.child_frame != request.child_frame
            ):
                raise LifecycleRegistryError(
                    f"calibration edge identity mismatch for {request.edge_id}"
                )
        capture_path = _resolve(base, definition.capture_manifest.path)
        state = load_registry(root).state()
        capture_id = _capture_id_from_definition(definition, base=base)
        existing_capture = state.captures.get(capture_id)
        if existing_capture is None:
            record_capture(
                root,
                capture_manifest=capture_path,
                operator=request.operator,
                command=list(command),
            )
        else:
            capture_digest = sha256_path(capture_path)
            if (
                existing_capture.vehicle_id != request.vehicle_id
                or existing_capture.sensor_kit_id != request.sensor_kit_id
                or existing_capture.artifact.sha256 != capture_digest
            ):
                raise LifecycleRegistryError(
                    f"capture identity or digest mismatch for {capture_id}"
                )
        if candidate_path is not None and status in {"PASS", "WARN", "FAIL"}:
            evaluation = evaluate_lifecycle(
                root,
                edge_id=request.edge_id,
                capture_manifest=capture_path,
                candidate_result=candidate_path,
                output=output / "lifecycle-evaluation.json",
                operator=request.operator,
                reason="raw replay lifecycle evaluation",
                command=list(command),
            )
            actions.append(f"evaluate:{evaluation.status}/{evaluation.admission}")
            actions.extend(
                f"event:{digest}" for digest in _event_digests_from_evaluation(
                    root, evaluation.evaluation_id
                )
            )
            if evaluation.status == "PASS" and evaluation.admission == "ADOPT":
                actions.append(
                    "promotion withheld: explicit Autoware promotion and smoke "
                    "artifacts were not supplied"
                )
            lifecycle_status, lifecycle_decision = _lifecycle_decision(
                evaluation.status, evaluation.admission
            )
        else:
            actions.append("candidate not admissible; lifecycle evaluation withheld")
            return "BLOCKED", "BLOCKED", actions
        return lifecycle_status, lifecycle_decision, actions
    except Exception as exc:
        return "BLOCKED", "BLOCKED", [f"lifecycle stage blocked: {exc}"]


def _lifecycle_decision(
    status: Literal["PASS", "WARN", "FAIL", "INCONCLUSIVE", "BLOCKED"],
    admission: str,
) -> tuple[ReplayStatus, ReplayDecision]:
    """Map the lifecycle registry's fail-closed result to replay semantics."""

    if status == "PASS" and admission == "ADOPT":
        return "PASS", "ADOPT"
    if status in {"PASS", "WARN", "INCONCLUSIVE"} and admission in {"HOLD", "NOT_APPLICABLE"}:
        return "WARN", "HOLD"
    if status == "FAIL" or admission == "DO_NOT_ADOPT":
        return "FAIL", "REJECT"
    return "BLOCKED", "BLOCKED"


def _run_autoware_plan_stage(
    definition: ReplayDefinition,
    base: Path,
    output: Path,
    candidate_path: Path | None,
    command: Sequence[str],
) -> ReplayDigestRef:
    request = definition.autoware
    if request is None:
        raise RawReplayError("Autoware plan request is missing")
    if candidate_path is None:
        raise RawReplayError("Autoware plan requires a candidate result")
    from calibrex.export.autoware_promotion import (
        AutowarePromotionRoots,
        build_autoware_promotion_plan,
    )

    plan = build_autoware_promotion_plan(
        candidate_path,
        roots=AutowarePromotionRoots(
            workspace_root=_resolve(base, request.workspace_root),
            package_root=_resolve(base, request.package_root),
        ),
        vehicle_id=request.vehicle_id,
        sensor_kit_id=request.sensor_kit_id,
        baseline_manifest_sha256=request.baseline_manifest_sha256,
        command=list(command),
    )
    plan_path = (
        _resolve(base, request.output_path)
        if request.output_path is not None
        else (output / "autoware-promotion-plan.json").resolve()
    )
    try:
        plan_path.resolve().relative_to(output.resolve())
    except ValueError as exc:
        raise RawReplayError(
            "Autoware promotion plan output must remain inside the replay output directory"
        ) from exc
    plan.save(plan_path)
    if plan.status != "PASS" or plan.decision != "ADOPT":
        raise RawReplayError(
            "Autoware promotion plan is not admissible: "
            f"status={plan.status}, decision={plan.decision}, reason={plan.reason}"
        )
    digest = sha256_path(plan_path)
    if digest is None:
        raise RawReplayError("Autoware plan was not written")
    return ReplayDigestRef(
        role="autoware_promotion",
        path=_portable_path(output, plan_path),
        sha256=digest,
        schema_version="slac.autoware_promotion/v0.1",
    )


def _validate_replay_output_scope(
    definition: ReplayDefinition, *, base: Path, output: Path
) -> None:
    """Reject replay output paths that could mutate an Autoware workspace."""

    request = definition.autoware
    if request is None or not definition.protocol.allow_autoware_plan:
        return
    output_resolved = output.resolve()
    for label, root_text in (
        ("workspace", request.workspace_root),
        ("package", request.package_root),
    ):
        root = _resolve(base, root_text)
        try:
            output_resolved.relative_to(root)
        except ValueError:
            continue
        raise RawReplayError(
            f"replay output directory must not be inside the Autoware {label}: {output_resolved}"
        )


def _metric_value(metric: MetricResult | None, field: str) -> float | None:
    if metric is None:
        return None
    value = getattr(metric, field)
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _provenance_for_definition(
    definition: ReplayDefinition,
    *,
    command: list[str],
    candidate_sha256: str | None = None,
    baseline_sha256: str | None = None,
    evidence_sha256: Mapping[str, str] | None = None,
) -> ReplayProvenance:
    source_sha256 = {definition.capture_manifest.input_id: definition.capture_manifest.sha256}
    source_sha256.update(
        {
            item.input_id: item.sha256
            for item in definition.inputs
            if item.role in {"source", "capture_manifest"}
        }
    )
    return ReplayProvenance(
        command=command,
        source_sha256=source_sha256,
        config_sha256=definition.config.sha256,
        baseline_sha256=baseline_sha256
        or (definition.baseline.sha256 if definition.baseline else None),
        candidate_sha256=candidate_sha256
        or (definition.candidate.sha256 if definition.candidate else None),
        holdout_sha256=definition.holdout.sha256 if definition.holdout else None,
        known_bad_sha256=definition.known_bad.sha256 if definition.known_bad else None,
        evidence_sha256=dict(evidence_sha256 or {}),
        environment_sha256=definition.environment.environment_sha256,
        tool_sha256=definition.tool.tool_sha256,
        container_digest=definition.environment.container_digest,
    )


def _provenance_for_result(result: ReplayResult, *, command: list[str]) -> ReplayProvenance:
    generated = result.generated_provenance
    return generated.model_copy(update={"command": command})


def _capture_id_from_definition(definition: ReplayDefinition, *, base: Path | None = None) -> str:
    capture_path = _resolve(base or Path.cwd(), definition.capture_manifest.path)
    try:
        manifest = load_capture_manifest(capture_path, verify=True)
        value = manifest.capture_id or manifest.session_id
    except Exception:
        value = None
    return value or definition.capture_manifest.input_id


def _event_digests_from_evaluation(root: Path, evaluation_id: str) -> list[str]:
    try:
        from calibrex.core.lifecycle_registry import load_registry

        return [
            event.event_sha256
            for event in load_registry(root).events()
            if event.payload.get("evaluation", {}).get("evaluation_id") == evaluation_id
        ]
    except Exception:
        return []


def _resolve(base: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return (
        candidate.expanduser().resolve()
        if candidate.is_absolute()
        else (base / candidate).resolve()
    )


def _portable_path(base: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "FIELD_REPLACEMENT_PILOT_SCHEMA_VERSION",
    "RAW_REPLAY_COMPARISON_SCHEMA_VERSION",
    "RAW_REPLAY_DEFINITION_SCHEMA_VERSION",
    "RAW_REPLAY_PLAN_SCHEMA_VERSION",
    "RAW_REPLAY_RESULT_SCHEMA_VERSION",
    "RAW_REPLAY_STAGE_SCHEMA_VERSION",
    "FieldReplacementPilot",
    "FieldReplacementPilotArtifact",
    "RawReplayError",
    "ReplayAutowareRequest",
    "ReplayBudgets",
    "ReplayComparison",
    "ReplayDefinition",
    "ReplayDigestRef",
    "ReplayEnvironment",
    "ReplayGate",
    "ReplayInput",
    "ReplayPlan",
    "ReplayProtocol",
    "ReplayProvenance",
    "ReplayProvenanceSummary",
    "ReplayRegistryRequest",
    "ReplayResult",
    "ReplayStageArtifact",
    "ReplayStageResult",
    "ReplayTool",
    "compare_raw_replays",
    "field_replacement_pilot_json_schema",
    "load_field_replacement_pilot",
    "load_replay_definition",
    "load_replay_plan",
    "load_replay_result",
    "plan_raw_replay",
    "replay_comparison_json_schema",
    "replay_definition_json_schema",
    "replay_plan_json_schema",
    "replay_result_json_schema",
    "replay_stage_json_schema",
    "run_field_replacement_pilot",
    "run_raw_replay",
    "verify_raw_replay",
]

# A descriptive alias used by integrations that call this a pilot artifact.
FieldReplacementPilotArtifact = FieldReplacementPilot
