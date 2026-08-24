"""Digest-bound, read-only multi-LiDAR field replacement service.

The service is intentionally an orchestration boundary.  It compares a
baseline and candidate frame graph, verifies the bytes and self-digests of
every declared artifact, and produces a READY/HOLD decision.  It never edits
an Autoware package or a lifecycle registry.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import AliasChoices, ConfigDict, Field, field_validator, model_validator

from calibrex import __version__
from calibrex.core.exceptions import CalibrexError
from calibrex.core.geometry import SE3
from calibrex.core.io import read_mapping, write_mapping_atomic
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.raw_replay import ReplayResult, load_replay_result
from calibrex.core.result import StrictModel, TransformResult

MULTI_LIDAR_SERVICE_PLAN_SCHEMA_VERSION: Literal[
    "slac.multi_lidar_service_plan/v0.1"
] = "slac.multi_lidar_service_plan/v0.1"
MULTI_LIDAR_SERVICE_EVALUATION_SCHEMA_VERSION: Literal[
    "slac.multi_lidar_service_evaluation/v0.1"
] = "slac.multi_lidar_service_evaluation/v0.1"

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_ZERO_SHA256 = "0" * 64

ServiceCheckStatus = Literal["PASS", "WARN", "FAIL", "BLOCKED"]
ServiceReadiness = Literal["READY", "HOLD"]


class MultiLidarServiceError(CalibrexError):
    """Raised when a multi-LiDAR service artifact is malformed."""


class MultiLidarServiceProvenance(StrictModel):
    """Provenance carried by each generated service artifact."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    producer: Literal["calibrex"] = "calibrex"
    tool_name: str = "calibrex.multi-lidar-service"
    tool_version: str = __version__
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_commit: str | None = Field(default_factory=git_commit)
    command: list[str] = Field(default_factory=list)
    input_sha256: dict[str, str] = Field(default_factory=dict)
    artifact_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    artifact_digest_scope: str = (
        "canonical artifact excluding artifact_sha256 and "
        "provenance.artifact_sha256/generated_at"
    )


class MultiLidarSensorIdentity(StrictModel):
    """Stable physical identity for a sensor before or after replacement."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    sensor_id: str = Field(min_length=1)
    serial: str = Field(min_length=1)
    model: str = Field(min_length=1)
    firmware: str = Field(min_length=1)
    mount: str = Field(min_length=1)
    frame: str | None = None
    sensor_kit_id: str | None = None

    @property
    def mount_id(self) -> str:
        """Compatibility spelling used by lifecycle records."""

        return self.mount


class MultiLidarArtifactRef(StrictModel):
    """A filesystem or inline artifact bound by SHA-256."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    role: str = Field(default="artifact", min_length=1)
    path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("path", "artifact_path"),
    )
    sha256: str = Field(
        pattern=_SHA256_PATTERN,
        validation_alias=AliasChoices("sha256", "digest"),
    )
    size_bytes: int | None = Field(default=None, ge=0)
    schema_version: str | None = None
    artifact_sha256: str | None = Field(
        default=None,
        pattern=_SHA256_PATTERN,
        validation_alias=AliasChoices("artifact_sha256", "self_digest"),
    )
    payload: dict[str, Any] | None = None

    @property
    def digest(self) -> str:
        """Return the declared byte digest."""

        return self.sha256


class MultiLidarEdgeArtifactRef(MultiLidarArtifactRef):
    """One vehicle-local graph edge and its immutable artifact reference."""

    edge_id: str = Field(min_length=1)
    parent_frame: str = Field(
        min_length=1,
        validation_alias=AliasChoices("parent_frame", "parent"),
    )
    child_frame: str = Field(
        min_length=1,
        validation_alias=AliasChoices("child_frame", "child"),
    )
    transform: TransformResult | None = None


class MultiLidarServicePlan(StrictModel):
    """Immutable scope and input declaration for a replacement evaluation."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["slac.multi_lidar_service_plan/v0.1"] = (
        MULTI_LIDAR_SERVICE_PLAN_SCHEMA_VERSION
    )
    plan_id: str = Field(
        min_length=1,
        validation_alias=AliasChoices("plan_id", "service_id", "id"),
    )
    vehicle_id: str = Field(min_length=1)
    sensor_kit_id: str | None = None
    old_sensor: MultiLidarSensorIdentity = Field(
        validation_alias=AliasChoices(
            "old_sensor", "target_old_sensor", "old_sensor_identity"
        )
    )
    new_sensor: MultiLidarSensorIdentity = Field(
        validation_alias=AliasChoices(
            "new_sensor", "target_new_sensor", "new_sensor_identity"
        )
    )
    baseline_edges: list[MultiLidarEdgeArtifactRef] = Field(
        default_factory=list,
        validation_alias=AliasChoices("baseline_edges", "baseline_edge_artifacts"),
    )
    candidate_edges: list[MultiLidarEdgeArtifactRef] = Field(
        default_factory=list,
        validation_alias=AliasChoices("candidate_edges", "candidate_edge_artifacts"),
    )
    mutable_incident_edge_ids: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "mutable_incident_edge_ids", "incident_edge_ids", "allowlisted_edge_ids"
        ),
    )
    candidate_replay: MultiLidarArtifactRef | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "candidate_replay", "candidate_replay_result", "replay_result"
        ),
    )
    lifecycle_artifacts: dict[str, MultiLidarArtifactRef] = Field(default_factory=dict)
    autoware_artifacts: dict[str, MultiLidarArtifactRef] = Field(default_factory=dict)
    graph_root: str | None = None
    cycle_closure_budget_m: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("cycle_closure_budget_m", "cycle_closure_budget"),
    )
    cycle_closure_budget_deg: float = Field(default=0.0, ge=0.0)
    provenance: MultiLidarServiceProvenance = Field(
        default_factory=MultiLidarServiceProvenance
    )
    artifact_sha256: str = Field(default=_ZERO_SHA256, pattern=_SHA256_PATTERN)

    @field_validator("baseline_edges", "candidate_edges", mode="before")
    @classmethod
    def _coerce_edge_mapping(
        cls,
        value: object,
    ) -> object:
        """Accept edge maps as a convenience while serializing stable lists."""

        if not isinstance(value, Mapping):
            return value
        edges: list[dict[str, Any]] = []
        for edge_id, raw in value.items():
            if not isinstance(raw, Mapping):
                raise ValueError("edge mapping values must be mappings")
            item = dict(raw)
            item.setdefault("edge_id", str(edge_id))
            edges.append(item)
        return edges

    @model_validator(mode="after")
    def validate_plan(self) -> MultiLidarServicePlan:
        """Reject ambiguous scope and duplicate edge identities early."""

        if self.old_sensor.sensor_id == self.new_sensor.sensor_id:
            raise ValueError("old_sensor and new_sensor must have distinct sensor_id values")
        for label, edges in (
            ("baseline", self.baseline_edges),
            ("candidate", self.candidate_edges),
        ):
            ids = [edge.edge_id for edge in edges]
            if len(ids) != len(set(ids)):
                raise ValueError(f"{label}_edges must contain unique edge_id values")
        if len(self.mutable_incident_edge_ids) != len(set(self.mutable_incident_edge_ids)):
            raise ValueError("mutable_incident_edge_ids must be unique")
        if any(not value for value in self.mutable_incident_edge_ids):
            raise ValueError("mutable_incident_edge_ids must not contain empty IDs")
        return self

    @property
    def service_id(self) -> str:
        """Compatibility spelling for the plan identifier."""

        return self.plan_id

    @property
    def target_old_sensor(self) -> MultiLidarSensorIdentity:
        """Return the old target identity."""

        return self.old_sensor

    @property
    def target_new_sensor(self) -> MultiLidarSensorIdentity:
        """Return the new target identity."""

        return self.new_sensor

    @property
    def incident_edge_ids(self) -> list[str]:
        """Compatibility spelling for the exact mutation allowlist."""

        return list(self.mutable_incident_edge_ids)

    @property
    def baseline_edge_artifacts(self) -> list[MultiLidarEdgeArtifactRef]:
        """Compatibility spelling for baseline edge references."""

        return list(self.baseline_edges)

    @property
    def candidate_edge_artifacts(self) -> list[MultiLidarEdgeArtifactRef]:
        """Compatibility spelling for candidate edge references."""

        return list(self.candidate_edges)

    def with_artifact_digest(self) -> MultiLidarServicePlan:
        """Return a copy with the canonical self-digest and bound provenance."""

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
        """Raise when the plan or its provenance self-digest is tampered."""

        expected = self.with_artifact_digest().artifact_sha256
        if expected != self.artifact_sha256:
            raise MultiLidarServiceError(
                f"multi-LiDAR plan self-digest mismatch: declared={self.artifact_sha256}, "
                f"expected={expected}"
            )
        if self.provenance.artifact_sha256 != self.artifact_sha256:
            raise MultiLidarServiceError("multi-LiDAR plan provenance digest is not bound")

    def save(self, path: str | Path) -> None:
        """Write a digest-complete YAML or JSON plan."""

        artifact = self.with_artifact_digest()
        write_mapping_atomic(Path(path), artifact.model_dump(mode="json", exclude_none=False))


class MultiLidarReplayEvidence(StrictModel):
    """Candidate replay decision and its digest-bound source."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    role: str = "candidate_replay"
    path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("path", "artifact_path"),
    )
    sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    artifact_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    status: Literal["PASS", "WARN", "FAIL", "BLOCKED"]
    decision: Literal["ADOPT", "HOLD", "REJECT", "BLOCKED", "NOT_APPLICABLE"]
    gates: dict[str, ServiceCheckStatus] = Field(default_factory=dict)
    source_sha256: dict[str, str] = Field(default_factory=dict)


class MultiLidarGraphCheck(StrictModel):
    """Connectedness and cycle-closure observations for candidate edges."""

    status: ServiceCheckStatus
    connected: bool
    root: str | None = None
    frame_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    cycle_count: int = Field(default=0, ge=0)
    max_cycle_translation_error_m: float | None = Field(default=None, ge=0.0)
    max_cycle_rotation_error_deg: float | None = Field(default=None, ge=0.0)
    cycle_closure_budget_m: float = Field(default=0.0, ge=0.0)
    cycle_closure_budget_deg: float = Field(default=0.0, ge=0.0)
    reasons: list[str] = Field(default_factory=list)


class MultiLidarServiceEvaluation(StrictModel):
    """Digest-bound READY/HOLD decision for a multi-LiDAR replacement."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["slac.multi_lidar_service_evaluation/v0.1"] = (
        MULTI_LIDAR_SERVICE_EVALUATION_SCHEMA_VERSION
    )
    evaluation_id: str = Field(min_length=1)
    plan_sha256: str = Field(pattern=_SHA256_PATTERN)
    status: ServiceReadiness
    decision: ServiceReadiness
    reasons: list[str] = Field(default_factory=list)
    changed_edge_ids: list[str] = Field(default_factory=list)
    unallowlisted_changed_edge_ids: list[str] = Field(default_factory=list)
    digest_checks: dict[str, ServiceCheckStatus] = Field(default_factory=dict)
    gates: dict[str, ServiceCheckStatus] = Field(default_factory=dict)
    graph: MultiLidarGraphCheck
    candidate_replay: MultiLidarReplayEvidence | None = None
    referenced_artifacts: list[MultiLidarArtifactRef] = Field(default_factory=list)
    provenance: MultiLidarServiceProvenance = Field(
        default_factory=MultiLidarServiceProvenance
    )
    artifact_sha256: str = Field(default=_ZERO_SHA256, pattern=_SHA256_PATTERN)

    @property
    def readiness(self) -> ServiceReadiness:
        """Compatibility spelling for the final service decision."""

        return self.status

    @property
    def failure_reasons(self) -> list[str]:
        """Compatibility spelling for HOLD explanations."""

        return list(self.reasons)

    def with_artifact_digest(self) -> MultiLidarServiceEvaluation:
        """Return a copy with the canonical self-digest and bound provenance."""

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
        """Raise when the evaluation or provenance self-digest is tampered."""

        expected = self.with_artifact_digest().artifact_sha256
        if expected != self.artifact_sha256:
            raise MultiLidarServiceError(
                f"multi-LiDAR evaluation self-digest mismatch: declared={self.artifact_sha256}, "
                f"expected={expected}"
            )
        if self.provenance.artifact_sha256 != self.artifact_sha256:
            raise MultiLidarServiceError("multi-LiDAR evaluation provenance digest is not bound")

    def save(self, path: str | Path) -> None:
        """Write a digest-complete YAML or JSON evaluation."""

        artifact = self.with_artifact_digest()
        write_mapping_atomic(Path(path), artifact.model_dump(mode="json", exclude_none=False))


class MultiLidarServiceVerification(StrictModel):
    """Non-throwing verification summary returned by the service API."""

    path: str
    kind: Literal["plan", "evaluation", "unknown"]
    valid: bool
    checked_digest_count: int = Field(default=0, ge=0)
    errors: list[str] = Field(default_factory=list)


def multi_lidar_service_plan_json_schema() -> dict[str, Any]:
    """Return the generated plan JSON schema."""

    return MultiLidarServicePlan.model_json_schema()


def multi_lidar_service_evaluation_json_schema() -> dict[str, Any]:
    """Return the generated evaluation JSON schema."""

    return MultiLidarServiceEvaluation.model_json_schema()


def load_multi_lidar_service_plan(path: str | Path) -> MultiLidarServicePlan:
    """Load and verify a plan, including all referenced source bytes."""

    plan_path = Path(path)
    try:
        plan = MultiLidarServicePlan.model_validate(read_mapping(plan_path))
        plan.verify_artifact_digest()
        errors, _ = _verify_plan_inputs(plan, plan_path.parent, require_provenance=True)
    except Exception as exc:
        if isinstance(exc, MultiLidarServiceError):
            raise
        raise MultiLidarServiceError(f"invalid multi-LiDAR plan {plan_path}: {exc}") from exc
    if errors:
        raise MultiLidarServiceError("; ".join(errors))
    return plan


def load_multi_lidar_service_evaluation(path: str | Path) -> MultiLidarServiceEvaluation:
    """Load and verify an evaluation self-digest."""

    evaluation_path = Path(path)
    try:
        evaluation = MultiLidarServiceEvaluation.model_validate(read_mapping(evaluation_path))
        evaluation.verify_artifact_digest()
    except Exception as exc:
        if isinstance(exc, MultiLidarServiceError):
            raise
        raise MultiLidarServiceError(
            f"invalid multi-LiDAR evaluation {evaluation_path}: {exc}"
        ) from exc
    return evaluation


def build_multi_lidar_service_plan(
    definition: MultiLidarServicePlan | str | Path,
    *,
    output: str | Path | None = None,
    command: Sequence[str] = (),
) -> MultiLidarServicePlan:
    """Materialize a read-only plan after verifying declared source digests."""

    definition_path: Path | None = None
    if isinstance(definition, MultiLidarServicePlan):
        plan = definition
    else:
        definition_path = Path(definition)
        try:
            plan = MultiLidarServicePlan.model_validate(read_mapping(definition_path))
        except Exception as exc:
            raise MultiLidarServiceError(f"invalid multi-LiDAR plan input: {exc}") from exc
    if plan.artifact_sha256 != _ZERO_SHA256:
        plan.verify_artifact_digest()
    source_base = (
        definition_path.parent
        if definition_path is not None
        else (Path(output).parent if output is not None else Path.cwd())
    )
    output_base = Path(output).parent if output is not None else source_base
    errors, input_digests = _verify_plan_inputs(plan, source_base)
    if errors:
        raise MultiLidarServiceError("; ".join(errors))
    plan = _rebase_plan_references(plan, source_base, output_base)
    provenance = plan.provenance.model_copy(
        update={
            "command": list(command) or plan.provenance.command,
            "input_sha256": {**plan.provenance.input_sha256, **input_digests},
        }
    )
    result = plan.model_copy(update={"provenance": provenance}).with_artifact_digest()
    if output is not None:
        result.save(output)
    return result


def plan_multi_lidar_service(
    definition: MultiLidarServicePlan | str | Path,
    *,
    output: str | Path | None = None,
    command: Sequence[str] = (),
) -> MultiLidarServicePlan:
    """Compatibility alias for :func:`build_multi_lidar_service_plan`."""

    return build_multi_lidar_service_plan(definition, output=output, command=command)


def create_multi_lidar_service_plan(
    definition: MultiLidarServicePlan | str | Path,
    *,
    output: str | Path | None = None,
    command: Sequence[str] = (),
) -> MultiLidarServicePlan:
    """Compatibility alias for callers that use a create verb."""

    return build_multi_lidar_service_plan(definition, output=output, command=command)


def evaluate_multi_lidar_service(
    plan: MultiLidarServicePlan | str | Path,
    *,
    candidate_replay: ReplayResult | str | Path | Mapping[str, Any] | None = None,
    output: str | Path | None = None,
    evaluation_id: str | None = None,
    command: Sequence[str] = (),
) -> MultiLidarServiceEvaluation:
    """Evaluate a plan fail-closed and emit READY or HOLD.

    Verification failures are represented as HOLD reasons so CI can preserve a
    digest-bound evaluation artifact instead of losing the audit record to an
    exception.  The function only writes ``output`` when explicitly requested.
    """

    plan_path: Path | None = None
    errors: list[str] = []
    if isinstance(plan, MultiLidarServicePlan):
        plan_model = plan
        base_dir = Path.cwd()
    else:
        plan_path = Path(plan)
        base_dir = plan_path.parent
        try:
            plan_model = MultiLidarServicePlan.model_validate(read_mapping(plan_path))
        except Exception as exc:
            raise MultiLidarServiceError(f"invalid multi-LiDAR plan input: {exc}") from exc
    if plan_model.artifact_sha256 == _ZERO_SHA256:
        errors.append("plan artifact_sha256 is unset")
    else:
        try:
            plan_model.verify_artifact_digest()
        except MultiLidarServiceError as exc:
            errors.append(str(exc))
    input_errors, input_digests = _verify_plan_inputs(
        plan_model,
        base_dir,
        require_provenance=True,
    )
    errors.extend(input_errors)

    replay = _load_replay_evidence(
        candidate_replay
        if candidate_replay is not None
        else plan_model.candidate_replay,
        base_dir,
        errors,
    )
    digest_checks: dict[str, ServiceCheckStatus] = {
        "plan": "PASS" if not any("plan" in reason for reason in errors) else "FAIL",
        "inputs": "PASS" if not input_errors else "FAIL",
    }
    if replay is None:
        digest_checks["candidate_replay"] = "FAIL"
    elif replay.sha256 is not None and replay.path is not None:
        digest_checks["candidate_replay"] = (
            "PASS"
            if not any("candidate replay" in reason for reason in errors)
            else "FAIL"
        )
    changed, unallowlisted = _changed_edges(plan_model)
    declared_edge_ids = {
        edge.edge_id for edge in (*plan_model.baseline_edges, *plan_model.candidate_edges)
    }
    unknown_allowlisted = sorted(
        set(plan_model.mutable_incident_edge_ids) - declared_edge_ids
    )
    if unknown_allowlisted:
        errors.append(
            "mutable_incident_edge_ids references unknown edge(s): "
            + ", ".join(unknown_allowlisted)
        )
    if unallowlisted:
        errors.append(
            "changed edge(s) are outside mutable_incident_edge_ids: "
            + ", ".join(unallowlisted)
        )
    digest_checks["edge_allowlist"] = (
        "PASS" if not unallowlisted and not unknown_allowlisted else "FAIL"
    )

    graph = _evaluate_graph(plan_model, base_dir)
    errors.extend(graph.reasons)
    digest_checks["graph"] = graph.status

    gates: dict[str, ServiceCheckStatus] = {}
    if replay is None:
        gates.update(
            {
                "candidate_replay": "FAIL",
                "holdout": "FAIL",
                "known_bad": "FAIL",
                "observability": "FAIL",
            }
        )
    else:
        if replay.status != "PASS" or replay.decision != "ADOPT":
            errors.append(
                "candidate replay must have status PASS and decision ADOPT; "
                f"observed {replay.status}/{replay.decision}"
            )
        gates["candidate_replay"] = (
            "PASS" if replay.status == "PASS" and replay.decision == "ADOPT" else "FAIL"
        )
        for gate_name in ("holdout", "known_bad", "observability"):
            gate_status = _replay_gate_status(replay, gate_name)
            gates[gate_name] = gate_status
            if gate_status != "PASS":
                errors.append(f"candidate replay {gate_name} gate is {gate_status}, expected PASS")

    all_pass = not errors and all(value == "PASS" for value in gates.values())
    status: ServiceReadiness = "READY" if all_pass else "HOLD"
    plan_sha256 = plan_model.artifact_sha256
    if plan_sha256 == _ZERO_SHA256:
        plan_sha256 = plan_model.with_artifact_digest().artifact_sha256
    provenance = MultiLidarServiceProvenance(
        command=list(command),
        input_sha256={"plan": plan_sha256, **input_digests},
    )
    evaluation = MultiLidarServiceEvaluation(
        evaluation_id=evaluation_id or f"{plan_model.plan_id}-evaluation",
        plan_sha256=plan_sha256,
        status=status,
        decision=status,
        reasons=_unique(errors),
        changed_edge_ids=changed,
        unallowlisted_changed_edge_ids=unallowlisted,
        digest_checks=digest_checks,
        gates=gates,
        graph=graph,
        candidate_replay=replay,
        referenced_artifacts=[
            *plan_model.baseline_edges,
            *plan_model.candidate_edges,
            *([plan_model.candidate_replay] if plan_model.candidate_replay else []),
        ],
        provenance=provenance,
    ).with_artifact_digest()
    if output is not None:
        evaluation.save(output)
    return evaluation


def evaluate_multi_lidar_service_plan(
    plan: MultiLidarServicePlan | str | Path,
    *,
    candidate_replay: ReplayResult | str | Path | Mapping[str, Any] | None = None,
    output: str | Path | None = None,
    evaluation_id: str | None = None,
    command: Sequence[str] = (),
) -> MultiLidarServiceEvaluation:
    """Compatibility alias for callers that name the evaluated plan."""

    return evaluate_multi_lidar_service(
        plan,
        candidate_replay=candidate_replay,
        output=output,
        evaluation_id=evaluation_id,
        command=command,
    )


def verify_multi_lidar_service(
    path: str | Path,
    *,
    plan: str | Path | None = None,
) -> MultiLidarServiceVerification:
    """Verify a plan/evaluation and all digest-bound source references."""

    artifact_path = Path(path)
    errors: list[str] = []
    checked = 0
    kind: Literal["plan", "evaluation", "unknown"] = "unknown"
    try:
        payload = read_mapping(artifact_path)
        schema_version = payload.get("schema_version")
        if schema_version == MULTI_LIDAR_SERVICE_PLAN_SCHEMA_VERSION:
            kind = "plan"
            model = MultiLidarServicePlan.model_validate(payload)
            model.verify_artifact_digest()
            source_errors, digests = _verify_plan_inputs(
                model,
                artifact_path.parent,
                require_provenance=True,
            )
            errors.extend(source_errors)
            checked = len(digests)
        elif schema_version == MULTI_LIDAR_SERVICE_EVALUATION_SCHEMA_VERSION:
            kind = "evaluation"
            evaluation = MultiLidarServiceEvaluation.model_validate(payload)
            evaluation.verify_artifact_digest()
            source_base = artifact_path.parent
            plan_model: MultiLidarServicePlan | None = None
            if plan is not None:
                plan_model = load_multi_lidar_service_plan(plan)
                source_base = Path(plan).parent
            for index, ref in enumerate(evaluation.referenced_artifacts):
                source_errors, _ = _verify_ref(
                    ref,
                    source_base,
                    f"evaluation referenced_artifact[{index}]",
                )
                errors.extend(source_errors)
            checked = len(evaluation.referenced_artifacts)
            if plan_model is not None and evaluation.plan_sha256 != plan_model.artifact_sha256:
                errors.append("evaluation plan_sha256 does not match supplied plan")
        else:
            errors.append(f"unsupported schema_version {schema_version!r}")
    except Exception as exc:
        errors.append(str(exc))
    return MultiLidarServiceVerification(
        path=artifact_path.as_posix(),
        kind=kind,
        valid=not errors,
        checked_digest_count=checked,
        errors=_unique(errors),
    )


def verify_multi_lidar_service_artifact(
    path: str | Path,
    *,
    plan: str | Path | None = None,
) -> MultiLidarServiceVerification:
    """Compatibility alias for :func:`verify_multi_lidar_service`."""

    return verify_multi_lidar_service(path, plan=plan)


def _load_replay_evidence(
    source: ReplayResult | str | Path | Mapping[str, Any] | MultiLidarArtifactRef | None,
    base_dir: Path,
    errors: list[str],
) -> MultiLidarReplayEvidence | None:
    if source is None:
        errors.append("candidate replay result is required")
        return None
    if isinstance(source, MultiLidarArtifactRef):
        if source.path is None:
            if source.payload is None:
                errors.append("candidate replay reference has no path or payload")
                return None
            raw: Mapping[str, Any] = source.payload
            source_path: Path | None = None
            declared_sha = source.sha256
        else:
            source_path = _resolve_path(source.path, base_dir)
            declared_sha = source.sha256
            try:
                actual = sha256_path(source_path)
                if actual != declared_sha:
                    errors.append(
                        "candidate replay digest mismatch: "
                        f"declared={declared_sha}, actual={actual}"
                    )
                raw = read_mapping(source_path)
            except Exception as exc:
                errors.append(f"candidate replay input cannot be read: {exc}")
                return None
        return _replay_evidence_from_mapping(raw, source_path, declared_sha, base_dir, errors)
    if isinstance(source, ReplayResult):
        return _replay_evidence_from_model(source, None, None, errors)
    if isinstance(source, Mapping):
        return _replay_evidence_from_mapping(source, None, None, base_dir, errors)
    source_path = _resolve_path(str(source), base_dir)
    try:
        actual = sha256_path(source_path)
        raw = read_mapping(source_path)
    except Exception as exc:
        errors.append(f"candidate replay input cannot be read: {exc}")
        return None
    if actual is None:
        errors.append("candidate replay input does not exist")
        return None
    return _replay_evidence_from_mapping(raw, source_path, actual, source_path.parent, errors)


def _replay_evidence_from_model(
    result: ReplayResult,
    path: Path | None,
    declared_sha: str | None,
    errors: list[str],
) -> MultiLidarReplayEvidence:
    try:
        result.verify_artifact_digest()
    except Exception as exc:
        errors.append(f"candidate replay self-digest mismatch: {exc}")
    if path is not None and declared_sha is not None:
        actual = sha256_path(path)
        if actual != declared_sha:
            errors.append(
                "candidate replay digest mismatch: "
                f"declared={declared_sha}, actual={actual}"
            )
    stage_refs = [stage.artifact for stage in result.stages if stage.artifact is not None]
    for ref in stage_refs:
        if ref is None:
            continue
        ref_path = _resolve_path(ref.path, path.parent if path else Path.cwd())
        actual = sha256_path(ref_path)
        if actual != ref.sha256:
            errors.append(f"candidate replay stage digest mismatch for {ref.path}")
    for name, ref in result.artifacts.items():
        ref_path = _resolve_path(ref.path, path.parent if path else Path.cwd())
        actual = sha256_path(ref_path)
        if actual != ref.sha256:
            errors.append(f"candidate replay artifact digest mismatch for {name}")
    gates = {gate.gate_id: gate.status for gate in result.gates}
    return MultiLidarReplayEvidence(
        path=path.as_posix() if path else None,
        sha256=declared_sha,
        artifact_sha256=result.artifact_sha256,
        status=result.status,
        decision=result.decision,
        gates=gates,
    )


def _replay_evidence_from_mapping(
    raw: Mapping[str, Any],
    path: Path | None,
    declared_sha: str | None,
    base_dir: Path,
    errors: list[str],
) -> MultiLidarReplayEvidence | None:
    try:
        if raw.get("schema_version") == "slac.raw_replay_result/v0.1":
            result = (
                load_replay_result(path)
                if path is not None
                else ReplayResult.model_validate(raw)
            )
            return _replay_evidence_from_model(result, path, declared_sha, errors)
        status = raw.get("status")
        decision = raw.get("decision")
        if status not in {"PASS", "WARN", "FAIL", "BLOCKED"}:
            raise ValueError("replay status must be PASS, WARN, FAIL, or BLOCKED")
        if decision not in {"ADOPT", "HOLD", "REJECT", "BLOCKED", "NOT_APPLICABLE"}:
            raise ValueError("replay decision is invalid")
        raw_artifact_sha = raw.get("artifact_sha256")
        if isinstance(raw_artifact_sha, str):
            expected = _self_digest_mapping(raw)
            if expected != raw_artifact_sha:
                errors.append("candidate replay self-digest mismatch")
        raw_gates = raw.get("gates", {})
        gates: dict[str, ServiceCheckStatus] = {}
        if isinstance(raw_gates, Mapping):
            for key, value in raw_gates.items():
                if value in {"PASS", "WARN", "FAIL", "BLOCKED"}:
                    gates[str(key)] = cast(ServiceCheckStatus, value)
        elif isinstance(raw_gates, list):
            for value in raw_gates:
                if isinstance(value, Mapping) and value.get("gate_id") and value.get("status"):
                    gates[str(value["gate_id"])] = cast(ServiceCheckStatus, value["status"])
        return MultiLidarReplayEvidence(
            path=path.as_posix() if path else None,
            sha256=declared_sha,
            artifact_sha256=raw_artifact_sha
            if isinstance(raw_artifact_sha, str)
            else None,
            status=cast(Literal["PASS", "WARN", "FAIL", "BLOCKED"], status),
            decision=cast(
                Literal["ADOPT", "HOLD", "REJECT", "BLOCKED", "NOT_APPLICABLE"],
                decision,
            ),
            gates=gates,
            source_sha256={
                str(key): str(value)
                for key, value in raw.items()
                if str(key).endswith("_sha256") and isinstance(value, str)
            },
        )
    except Exception as exc:
        errors.append(f"invalid candidate replay result: {exc}")
        return None


def _verify_plan_inputs(
    plan: MultiLidarServicePlan,
    base_dir: Path,
    *,
    require_provenance: bool = False,
) -> tuple[list[str], dict[str, str]]:
    errors: list[str] = []
    digests: dict[str, str] = {}
    refs: list[tuple[str, MultiLidarArtifactRef]] = []
    refs.extend((f"baseline_edge:{edge.edge_id}", edge) for edge in plan.baseline_edges)
    refs.extend((f"candidate_edge:{edge.edge_id}", edge) for edge in plan.candidate_edges)
    if plan.candidate_replay is not None:
        refs.append(("candidate_replay", plan.candidate_replay))
    refs.extend((f"lifecycle:{key}", ref) for key, ref in plan.lifecycle_artifacts.items())
    refs.extend((f"autoware:{key}", ref) for key, ref in plan.autoware_artifacts.items())
    for name, ref in refs:
        ref_errors, digest = _verify_ref(ref, base_dir, name)
        errors.extend(ref_errors)
        if digest is not None:
            digests[name] = digest
            declared = plan.provenance.input_sha256.get(name)
            if require_provenance and declared is not None and declared != digest:
                errors.append(
                    f"{name} provenance input digest mismatch: "
                    f"declared={declared}, actual={digest}"
                )
    return _unique(errors), digests


def _rebase_plan_references(
    plan: MultiLidarServicePlan,
    source_base: Path,
    output_base: Path,
) -> MultiLidarServicePlan:
    """Rebase source paths so an emitted plan is evaluable from its directory."""

    def rebase(ref: MultiLidarArtifactRef) -> MultiLidarArtifactRef:
        if ref.path is None:
            return ref
        source_path = _resolve_path(ref.path, source_base)
        if not source_path.exists():
            return ref
        try:
            portable = os.path.relpath(source_path, output_base)
        except ValueError:
            portable = str(source_path.resolve())
        return ref.model_copy(update={"path": Path(portable).as_posix()})

    return plan.model_copy(
        update={
            "baseline_edges": [rebase(edge) for edge in plan.baseline_edges],
            "candidate_edges": [rebase(edge) for edge in plan.candidate_edges],
            "candidate_replay": (
                rebase(plan.candidate_replay) if plan.candidate_replay is not None else None
            ),
            "lifecycle_artifacts": {
                key: rebase(ref) for key, ref in plan.lifecycle_artifacts.items()
            },
            "autoware_artifacts": {
                key: rebase(ref) for key, ref in plan.autoware_artifacts.items()
            },
        }
    )


def _verify_ref(
    ref: MultiLidarArtifactRef,
    base_dir: Path,
    name: str,
) -> tuple[list[str], str | None]:
    if ref.path is None:
        if ref.payload is not None:
            inline_digest = _canonical_sha256(ref.payload)
            if inline_digest != ref.sha256:
                return [
                    f"{name} inline digest mismatch: declared={ref.sha256}, actual={inline_digest}"
                ], None
            return [], inline_digest
        return [], None
    path = _resolve_path(ref.path, base_dir)
    byte_digest = sha256_path(path)
    if byte_digest is None:
        return [f"{name} path does not exist: {ref.path}"], None
    errors: list[str] = []
    if byte_digest != ref.sha256:
        errors.append(f"{name} digest mismatch: declared={ref.sha256}, actual={byte_digest}")
    if ref.size_bytes is not None and path.is_file() and path.stat().st_size != ref.size_bytes:
        errors.append(f"{name} size mismatch for {ref.path}")
    if ref.artifact_sha256 is not None and path.suffix.lower() in {".json", ".yaml", ".yml"}:
        try:
            raw = read_mapping(path)
            expected = _self_digest_mapping(raw)
            declared = raw.get("artifact_sha256")
            if isinstance(declared, str) and expected != declared:
                errors.append(f"{name} referenced artifact self-digest mismatch")
            if expected != ref.artifact_sha256:
                errors.append(f"{name} self-digest does not match declared artifact_sha256")
        except Exception as exc:
            errors.append(f"{name} self-digest cannot be checked: {exc}")
    return errors, byte_digest


def _changed_edges(plan: MultiLidarServicePlan) -> tuple[list[str], list[str]]:
    baseline = {edge.edge_id: edge for edge in plan.baseline_edges}
    candidate = {edge.edge_id: edge for edge in plan.candidate_edges}
    changed: list[str] = []
    for edge_id in sorted(set(baseline) | set(candidate)):
        left = baseline.get(edge_id)
        right = candidate.get(edge_id)
        if left is None or right is None:
            changed.append(edge_id)
            continue
        if (
            left.sha256 != right.sha256
            or left.parent_frame != right.parent_frame
            or left.child_frame != right.child_frame
        ):
            changed.append(edge_id)
    allowlist = set(plan.mutable_incident_edge_ids)
    return changed, [edge_id for edge_id in changed if edge_id not in allowlist]


def _evaluate_graph(
    plan: MultiLidarServicePlan,
    base_dir: Path | None = None,
) -> MultiLidarGraphCheck:
    edges = plan.candidate_edges
    nodes = {node for edge in edges for node in (edge.parent_frame, edge.child_frame)}
    if not edges:
        return MultiLidarGraphCheck(
            status="FAIL",
            connected=False,
            root=plan.graph_root,
            frame_count=0,
            edge_count=0,
            reasons=["candidate graph has no edges"],
            cycle_closure_budget_m=plan.cycle_closure_budget_m,
            cycle_closure_budget_deg=plan.cycle_closure_budget_deg,
        )
    root = plan.graph_root or min(nodes)
    reasons: list[str] = []
    if root not in nodes:
        reasons.append(f"graph_root {root!r} is not present in candidate graph")
    adjacency: dict[str, list[tuple[str, SE3, str]]] = {node: [] for node in nodes}
    missing_transform = False
    for edge in edges:
        transform = _edge_transform(edge, base_dir)
        if transform is None:
            missing_transform = True
            transform = SE3.identity()
        adjacency[edge.parent_frame].append((edge.child_frame, transform, edge.edge_id))
        adjacency[edge.child_frame].append((edge.parent_frame, transform.inverse(), edge.edge_id))
    reachable: set[str] = set()
    if root in nodes:
        stack = [root]
        while stack:
            current = stack.pop()
            if current in reachable:
                continue
            reachable.add(current)
            stack.extend(neighbor for neighbor, _, _ in adjacency[current])
    connected = reachable == nodes and not reasons
    if not connected:
        disconnected = sorted(nodes - reachable)
        suffix = f": {disconnected}" if disconnected else ""
        reasons.append("candidate graph is disconnected" + suffix)

    transforms: dict[str, SE3] = {}
    tree_edges: set[str] = set()
    for component_root in sorted(nodes):
        if component_root in transforms:
            continue
        transforms[component_root] = SE3.identity()
        stack = [component_root]
        while stack:
            current = stack.pop()
            for neighbor, relative, edge_id in adjacency[current]:
                if neighbor not in transforms:
                    transforms[neighbor] = transforms[current].compose(relative)
                    tree_edges.add(edge_id)
                    stack.append(neighbor)
    cycles = [edge for edge in edges if edge.edge_id not in tree_edges]
    max_translation: float | None = 0.0 if cycles else None
    max_rotation: float | None = 0.0 if cycles else None
    if cycles and missing_transform:
        reasons.append("cycle closure cannot be checked because an edge transform is missing")
    elif cycles:
        for edge in cycles:
            left = transforms[edge.parent_frame]
            right = transforms[edge.child_frame]
            implied = left.inverse().compose(right)
            observed = _edge_transform(edge, base_dir)
            if observed is None:
                continue
            residual = implied.inverse().compose(observed)
            translation_error = math.sqrt(sum(value * value for value in residual.translation_m))
            rotation_error = math.degrees(
                2.0 * math.acos(min(1.0, max(0.0, abs(residual.rotation_quat_xyzw[3]))))
            )
            max_translation = max(max_translation or 0.0, translation_error)
            max_rotation = max(max_rotation or 0.0, rotation_error)
        if (max_translation or 0.0) > plan.cycle_closure_budget_m:
            reasons.append(
                f"cycle translation closure {max_translation:.6g} m exceeds "
                f"budget {plan.cycle_closure_budget_m:.6g} m"
            )
        if (max_rotation or 0.0) > plan.cycle_closure_budget_deg:
            reasons.append(
                f"cycle rotation closure {max_rotation:.6g} deg exceeds "
                f"budget {plan.cycle_closure_budget_deg:.6g} deg"
            )
    status: ServiceCheckStatus = "PASS" if not reasons else "FAIL"
    return MultiLidarGraphCheck(
        status=status,
        connected=connected,
        root=root,
        frame_count=len(nodes),
        edge_count=len(edges),
        cycle_count=len(cycles),
        max_cycle_translation_error_m=max_translation,
        max_cycle_rotation_error_deg=max_rotation,
        cycle_closure_budget_m=plan.cycle_closure_budget_m,
        cycle_closure_budget_deg=plan.cycle_closure_budget_deg,
        reasons=reasons,
    )


def _edge_transform(edge: MultiLidarEdgeArtifactRef, base_dir: Path | None = None) -> SE3 | None:
    if edge.transform is not None:
        return edge.transform.as_se3()
    if edge.path is not None and base_dir is not None:
        try:
            raw = read_mapping(_resolve_path(edge.path, base_dir))
        except (OSError, ValueError):
            return None
        candidate: Any = raw.get("transform", raw)
        if isinstance(candidate, Mapping):
            translation = candidate.get("translation_m")
            rotation = candidate.get("rotation_quat_xyzw")
            if isinstance(translation, list) and isinstance(rotation, list):
                try:
                    return SE3.from_lists(translation, rotation)
                except ValueError:
                    return None
            initial = candidate.get("initial")
            if isinstance(initial, Mapping):
                translation = initial.get("translation_m")
                rotation = initial.get("rotation_quat_xyzw")
                if isinstance(translation, list) and isinstance(rotation, list):
                    try:
                        return SE3.from_lists(translation, rotation)
                    except ValueError:
                        return None
    return None


def _replay_gate_status(replay: MultiLidarReplayEvidence, name: str) -> ServiceCheckStatus:
    for gate_id, status in replay.gates.items():
        normalized = gate_id.lower().replace("-", "_").replace(" ", "_")
        if name in normalized:
            return status
    return "FAIL"


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    relative = base_dir / path
    return relative if relative.exists() else path


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _self_digest_mapping(value: Mapping[str, Any]) -> str:
    payload = json.loads(json.dumps(value))
    if isinstance(payload, dict):
        payload.pop("artifact_sha256", None)
        provenance = payload.get("provenance")
        if isinstance(provenance, dict):
            provenance.pop("artifact_sha256", None)
            provenance.pop("generated_at", None)
    return _canonical_sha256(cast(Mapping[str, Any], payload))


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


# Concise compatibility aliases used by downstream callers.
MultiLidarSensor = MultiLidarSensorIdentity
MultiLidarEdgeArtifact = MultiLidarEdgeArtifactRef
MultiLidarServiceArtifactRef = MultiLidarArtifactRef
MultiLidarPlan = MultiLidarServicePlan
MultiLidarEvaluation = MultiLidarServiceEvaluation
MultiLidarServicePlanArtifact = MultiLidarServicePlan
MultiLidarServiceEvaluationArtifact = MultiLidarServiceEvaluation
MultiLidarServicePlanProvenance = MultiLidarServiceProvenance
