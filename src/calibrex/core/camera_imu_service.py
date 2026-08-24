"""Digest-bound, read-only camera--IMU replacement service.

This module is the small v0.1 admission boundary for replacing a camera and
IMU pair.  It deliberately does not know how to edit a lifecycle registry or
an Autoware package.  A plan names immutable inputs and evidence; evaluation
only emits a digest-bound ``READY``/``HOLD`` record.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import AliasChoices, ConfigDict, Field, field_validator, model_validator

from calibrex import __version__
from calibrex.core.exceptions import CalibrexError
from calibrex.core.io import read_mapping, write_mapping_atomic
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.raw_replay import ReplayResult, load_replay_result
from calibrex.core.result import StrictModel

CAMERA_IMU_SERVICE_PLAN_SCHEMA_VERSION: Literal["slac.camera_imu_service_plan/v0.1"] = (
    "slac.camera_imu_service_plan/v0.1"
)
CAMERA_IMU_SERVICE_EVALUATION_SCHEMA_VERSION: Literal[
    "slac.camera_imu_service_evaluation/v0.1"
] = "slac.camera_imu_service_evaluation/v0.1"

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_ZERO_SHA256 = "0" * 64

ServiceCheckStatus = Literal["PASS", "WARN", "FAIL", "BLOCKED"]
ServiceReadiness = Literal["READY", "HOLD"]
MetricSplit = Literal["train", "holdout", "known_bad"]
MetricStatus = Literal["PASS", "WARN", "FAIL", "BLOCKED"]

# The role names are intentionally explicit.  A caller may require a stricter
# subset/superset in a plan, but an unqualified plan is never allowed to skip
# the three independent decision controls and the protocol/capture binding.
DEFAULT_REQUIRED_EVIDENCE_ROLES: tuple[str, ...] = (
    "protocol",
    "capture",
    "holdout",
    "known_bad",
    "observability",
)


class CameraImuServiceError(CalibrexError):
    """Raised when a camera--IMU service artifact is malformed."""


class CameraImuServiceProvenance(StrictModel):
    """Portable provenance carried by generated plan/evaluation artifacts."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    producer: Literal["calibrex"] = "calibrex"
    tool_name: str = "calibrex.camera-imu-service"
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


class SensorIdentity(StrictModel):
    """Stable identity for one camera or IMU before/after replacement."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    sensor_id: str = Field(min_length=1)
    serial: str = Field(min_length=1)
    model: str = Field(min_length=1)
    firmware: str = Field(min_length=1)
    mount: str = Field(min_length=1)
    frame: str | None = None
    sensor_type: Literal["camera", "imu"] | None = Field(
        default=None,
        validation_alias=AliasChoices("sensor_type", "type", "kind"),
    )
    sensor_kit_id: str | None = None


class DigestRef(StrictModel):
    """A portable filesystem or inline value bound by SHA-256."""

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


class MetricObservation(StrictModel):
    """One named metric with an explicit split, status, and derivation."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    metric_id: str = Field(min_length=1)
    value: float
    unit: str = Field(min_length=1)
    split: MetricSplit
    status: MetricStatus = "PASS"
    provenance: dict[str, Any] = Field(default_factory=dict)
    protocol_id: str | None = None
    capture_id: str | None = None
    evidence_role: str | None = None


class CameraImuReplayEvidence(StrictModel):
    """Digest and gate summary copied from a replay result."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    role: str = "candidate_replay"
    path: str | None = None
    sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    artifact_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    status: Literal["PASS", "WARN", "FAIL", "BLOCKED"]
    decision: Literal["ADOPT", "HOLD", "REJECT", "BLOCKED", "NOT_APPLICABLE"]
    gates: dict[str, ServiceCheckStatus] = Field(default_factory=dict)
    source_sha256: dict[str, str] = Field(default_factory=dict)


class CameraImuServicePlan(StrictModel):
    """Immutable scope, inputs, evidence, and budgets for one evaluation."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["slac.camera_imu_service_plan/v0.1"] = (
        CAMERA_IMU_SERVICE_PLAN_SCHEMA_VERSION
    )
    plan_id: str = Field(
        min_length=1,
        validation_alias=AliasChoices("plan_id", "service_id", "id"),
    )
    vehicle_id: str = Field(min_length=1)
    sensor_kit_id: str | None = None
    old_camera: SensorIdentity = Field(
        validation_alias=AliasChoices("old_camera", "old_camera_sensor")
    )
    new_camera: SensorIdentity = Field(
        validation_alias=AliasChoices("new_camera", "new_camera_sensor")
    )
    old_imu: SensorIdentity = Field(
        validation_alias=AliasChoices("old_imu", "old_imu_sensor")
    )
    new_imu: SensorIdentity = Field(
        validation_alias=AliasChoices("new_imu", "new_imu_sensor")
    )
    candidate_replay: DigestRef | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "candidate_replay", "candidate_replay_result", "replay_result"
        ),
    )
    required_evidence_roles: list[str] = Field(
        default_factory=lambda: list(DEFAULT_REQUIRED_EVIDENCE_ROLES),
        validation_alias=AliasChoices("required_evidence_roles", "required_roles"),
    )
    evidence: dict[str, DigestRef] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("evidence", "evidence_artifacts"),
    )
    metric_observations: list[MetricObservation] = Field(
        default_factory=list,
        validation_alias=AliasChoices("metric_observations", "metrics", "observations"),
    )
    mutable_components: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("mutable_components", "allowlisted_components"),
    )
    observed_changed_components: list[str] = Field(default_factory=list)
    baseline_components: dict[str, Any] = Field(default_factory=dict)
    candidate_components: dict[str, Any] = Field(default_factory=dict)

    # Budgets are kept as scalar fields in v0.1 so that policy diffs are
    # obvious in review and old JSON tooling can inspect them without refs.
    max_clock_offset_abs_s: float = Field(default=0.0, ge=0.0)
    max_rotation_holdout_rmse_rad: float = Field(default=0.0, ge=0.0)
    max_lever_arm_holdout_rmse_m_s2: float = Field(default=0.0, ge=0.0)
    max_bias_norm: float = Field(default=0.0, ge=0.0)
    min_known_bad_delta: float = Field(default=0.0, ge=0.0)
    min_observability_rank: float = Field(default=0.0, ge=0.0)

    protocol_id: str | None = None
    capture_id: str | None = None
    lifecycle_artifacts: dict[str, DigestRef] = Field(default_factory=dict)
    autoware_artifacts: dict[str, DigestRef] = Field(default_factory=dict)
    provenance: CameraImuServiceProvenance = Field(
        default_factory=CameraImuServiceProvenance
    )
    artifact_sha256: str = Field(default=_ZERO_SHA256, pattern=_SHA256_PATTERN)

    @field_validator("evidence", mode="before")
    @classmethod
    def _coerce_evidence_mapping(cls, value: object) -> object:
        """Accept a list of role-bearing refs while serializing a stable map."""

        if isinstance(value, Mapping):
            return value
        if isinstance(value, list):
            result: dict[str, object] = {}
            for item in value:
                if not isinstance(item, Mapping):
                    raise ValueError("evidence list items must be mappings")
                role = item.get("role")
                if not isinstance(role, str) or not role:
                    raise ValueError("evidence list items require a role")
                result[role] = dict(item)
            return result
        return value

    @model_validator(mode="after")
    def validate_plan(self) -> CameraImuServicePlan:
        """Reject ambiguous identities and duplicate role/component declarations."""

        sensors = (self.old_camera, self.new_camera, self.old_imu, self.new_imu)
        if len({sensor.sensor_id for sensor in sensors}) != len(sensors):
            raise ValueError("old/new camera and IMU sensor_id values must be distinct")
        if self.old_camera.sensor_type not in {None, "camera"}:
            raise ValueError("old_camera.sensor_type must be camera")
        if self.new_camera.sensor_type not in {None, "camera"}:
            raise ValueError("new_camera.sensor_type must be camera")
        if self.old_imu.sensor_type not in {None, "imu"}:
            raise ValueError("old_imu.sensor_type must be imu")
        if self.new_imu.sensor_type not in {None, "imu"}:
            raise ValueError("new_imu.sensor_type must be imu")
        if len(self.required_evidence_roles) != len(set(self.required_evidence_roles)):
            raise ValueError("required_evidence_roles must be unique")
        if any(not role for role in self.required_evidence_roles):
            raise ValueError("required_evidence_roles must not contain empty roles")
        if len(self.mutable_components) != len(set(self.mutable_components)):
            raise ValueError("mutable_components must be unique")
        if any(not component for component in self.mutable_components):
            raise ValueError("mutable_components must not contain empty component IDs")
        return self

    def with_artifact_digest(self) -> CameraImuServicePlan:
        """Return a copy with its canonical self-digest bound into provenance."""

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
        """Raise if the plan or its provenance self-digest was tampered with."""

        expected = self.with_artifact_digest().artifact_sha256
        if expected != self.artifact_sha256:
            raise CameraImuServiceError(
                f"camera-IMU plan self-digest mismatch: declared={self.artifact_sha256}, "
                f"expected={expected}"
            )
        if self.provenance.artifact_sha256 != self.artifact_sha256:
            raise CameraImuServiceError("camera-IMU plan provenance digest is not bound")

    def save(self, path: str | Path) -> None:
        """Write a digest-complete YAML or JSON plan."""

        artifact = self.with_artifact_digest()
        write_mapping_atomic(Path(path), artifact.model_dump(mode="json", exclude_none=False))


class CameraImuServiceEvaluation(StrictModel):
    """Digest-bound READY/HOLD decision for a camera--IMU replacement."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["slac.camera_imu_service_evaluation/v0.1"] = (
        CAMERA_IMU_SERVICE_EVALUATION_SCHEMA_VERSION
    )
    evaluation_id: str = Field(min_length=1)
    plan_sha256: str = Field(pattern=_SHA256_PATTERN)
    status: ServiceReadiness
    decision: ServiceReadiness
    reasons: list[str] = Field(default_factory=list)
    digest_checks: dict[str, ServiceCheckStatus] = Field(default_factory=dict)
    gates: dict[str, ServiceCheckStatus] = Field(default_factory=dict)
    required_evidence_roles: list[str] = Field(default_factory=list)
    missing_evidence_roles: list[str] = Field(default_factory=list)
    metric_checks: dict[str, ServiceCheckStatus] = Field(default_factory=dict)
    metric_observations: list[MetricObservation] = Field(default_factory=list)
    changed_components: list[str] = Field(default_factory=list)
    unallowlisted_changed_components: list[str] = Field(default_factory=list)
    bindings: dict[str, str] = Field(default_factory=dict)
    candidate_replay: CameraImuReplayEvidence | None = None
    referenced_artifacts: list[DigestRef] = Field(default_factory=list)
    provenance: CameraImuServiceProvenance = Field(
        default_factory=CameraImuServiceProvenance
    )
    artifact_sha256: str = Field(default=_ZERO_SHA256, pattern=_SHA256_PATTERN)

    def with_artifact_digest(self) -> CameraImuServiceEvaluation:
        """Return a copy with its canonical self-digest bound into provenance."""

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
        """Raise if the evaluation or its provenance self-digest was tampered with."""

        expected = self.with_artifact_digest().artifact_sha256
        if expected != self.artifact_sha256:
            raise CameraImuServiceError(
                "camera-IMU evaluation self-digest mismatch: "
                f"declared={self.artifact_sha256}, expected={expected}"
            )
        if self.provenance.artifact_sha256 != self.artifact_sha256:
            raise CameraImuServiceError("camera-IMU evaluation provenance digest is not bound")

    def save(self, path: str | Path) -> None:
        """Write a digest-complete YAML or JSON evaluation."""

        artifact = self.with_artifact_digest()
        write_mapping_atomic(Path(path), artifact.model_dump(mode="json", exclude_none=False))


class CameraImuServiceVerification(StrictModel):
    """Non-throwing verification summary returned by :func:`verify_camera_imu_service`."""

    path: str
    kind: Literal["plan", "evaluation", "unknown"]
    valid: bool
    checked_digest_count: int = Field(default=0, ge=0)
    errors: list[str] = Field(default_factory=list)


def camera_imu_service_plan_json_schema() -> dict[str, Any]:
    """Return the generated plan JSON schema."""

    return CameraImuServicePlan.model_json_schema()


def camera_imu_service_evaluation_json_schema() -> dict[str, Any]:
    """Return the generated evaluation JSON schema."""

    return CameraImuServiceEvaluation.model_json_schema()


def load_camera_imu_service_plan(path: str | Path) -> CameraImuServicePlan:
    """Load a plan and verify its self-digest and all source references."""

    plan_path = Path(path)
    try:
        plan = CameraImuServicePlan.model_validate(read_mapping(plan_path))
        plan.verify_artifact_digest()
        errors, _ = _verify_plan_inputs(plan, plan_path.parent, require_provenance=True)
    except Exception as exc:
        if isinstance(exc, CameraImuServiceError):
            raise
        raise CameraImuServiceError(f"invalid camera-IMU plan {plan_path}: {exc}") from exc
    if errors:
        raise CameraImuServiceError("; ".join(errors))
    return plan


def load_camera_imu_service_evaluation(path: str | Path) -> CameraImuServiceEvaluation:
    """Load and verify an evaluation self-digest."""

    evaluation_path = Path(path)
    try:
        evaluation = CameraImuServiceEvaluation.model_validate(read_mapping(evaluation_path))
        evaluation.verify_artifact_digest()
    except Exception as exc:
        if isinstance(exc, CameraImuServiceError):
            raise
        raise CameraImuServiceError(
            f"invalid camera-IMU evaluation {evaluation_path}: {exc}"
        ) from exc
    return evaluation


def build_camera_imu_service_plan(
    definition: CameraImuServicePlan | str | Path,
    *,
    output: str | Path | None = None,
    command: Sequence[str] = (),
) -> CameraImuServicePlan:
    """Verify declared inputs, rebase paths, and materialize a plan."""

    definition_path: Path | None = None
    if isinstance(definition, CameraImuServicePlan):
        plan = definition
    else:
        definition_path = Path(definition)
        try:
            plan = CameraImuServicePlan.model_validate(read_mapping(definition_path))
        except Exception as exc:
            raise CameraImuServiceError(f"invalid camera-IMU plan input: {exc}") from exc
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
        raise CameraImuServiceError("; ".join(errors))
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


def plan_camera_imu_service(
    definition: CameraImuServicePlan | str | Path,
    *,
    output: str | Path | None = None,
    command: Sequence[str] = (),
) -> CameraImuServicePlan:
    """Compatibility alias for :func:`build_camera_imu_service_plan`."""

    return build_camera_imu_service_plan(definition, output=output, command=command)


def evaluate_camera_imu_service(
    plan: CameraImuServicePlan | str | Path,
    *,
    candidate_replay: ReplayResult | str | Path | Mapping[str, Any] | None = None,
    output: str | Path | None = None,
    evaluation_id: str | None = None,
    command: Sequence[str] = (),
) -> CameraImuServiceEvaluation:
    """Evaluate all v0.1 gates fail-closed and optionally write an artifact."""

    plan_path: Path | None = None
    errors: list[str] = []
    if isinstance(plan, CameraImuServicePlan):
        plan_model = plan
        base_dir = Path.cwd()
    else:
        plan_path = Path(plan)
        base_dir = plan_path.parent
        try:
            plan_model = CameraImuServicePlan.model_validate(read_mapping(plan_path))
        except Exception as exc:
            raise CameraImuServiceError(f"invalid camera-IMU plan input: {exc}") from exc

    if plan_model.artifact_sha256 == _ZERO_SHA256:
        errors.append("plan artifact_sha256 is unset")
    else:
        try:
            plan_model.verify_artifact_digest()
        except CameraImuServiceError as exc:
            errors.append(str(exc))

    input_errors, input_digests = _verify_plan_inputs(
        plan_model, base_dir, require_provenance=True
    )
    errors.extend(input_errors)

    replay_source: ReplayResult | str | Path | Mapping[str, Any] | DigestRef | None
    replay_source = (
        candidate_replay if candidate_replay is not None else plan_model.candidate_replay
    )
    replay = _load_replay_evidence(replay_source, base_dir, errors)
    digest_checks: dict[str, ServiceCheckStatus] = {
        "plan": "PASS" if not any("plan" in reason for reason in errors) else "FAIL",
        "inputs": "PASS" if not input_errors else "FAIL",
        "candidate_replay": "PASS" if replay is not None else "FAIL",
    }

    gates: dict[str, ServiceCheckStatus] = {}
    if replay is None:
        gates = {
            "candidate_replay": "FAIL",
            "holdout": "FAIL",
            "known_bad": "FAIL",
            "observability": "FAIL",
        }
    else:
        replay_pass = replay.status == "PASS" and replay.decision == "ADOPT"
        gates["candidate_replay"] = "PASS" if replay_pass else "FAIL"
        if not replay_pass:
            errors.append(
                "candidate replay must have status PASS and decision ADOPT; "
                f"observed {replay.status}/{replay.decision}"
            )
        for gate_name in ("holdout", "known_bad", "observability"):
            gate_status = _replay_gate_status(replay, gate_name)
            gates[gate_name] = gate_status
            if gate_status != "PASS":
                errors.append(
                    f"candidate replay {gate_name} gate is {gate_status}, expected PASS"
                )

    loaded_evidence, missing_roles = _load_required_evidence(plan_model, base_dir, errors)
    if missing_roles:
        errors.append("required evidence role(s) missing: " + ", ".join(missing_roles))
    digest_checks["evidence"] = "PASS" if not missing_roles else "FAIL"

    observations = _collect_observations(plan_model, loaded_evidence)
    bindings, binding_errors = _check_bindings(plan_model, loaded_evidence, observations)
    errors.extend(binding_errors)
    digest_checks["bindings"] = "PASS" if not binding_errors else "FAIL"
    metric_checks, metric_errors = _evaluate_metrics(plan_model, observations)
    errors.extend(metric_errors)
    digest_checks["metrics"] = "PASS" if not metric_errors else "FAIL"

    changed = _changed_components(plan_model, loaded_evidence)
    allowlisted = set(plan_model.mutable_components)
    unallowlisted = [component for component in changed if component not in allowlisted]
    if unallowlisted:
        errors.append(
            "observed changed component(s) are outside mutable_components: "
            + ", ".join(unallowlisted)
        )
    digest_checks["component_allowlist"] = "PASS" if not unallowlisted else "FAIL"

    all_pass = not errors and all(value == "PASS" for value in gates.values())
    status: ServiceReadiness = "READY" if all_pass else "HOLD"
    plan_sha256 = plan_model.artifact_sha256
    if plan_sha256 == _ZERO_SHA256:
        plan_sha256 = plan_model.with_artifact_digest().artifact_sha256
    provenance = CameraImuServiceProvenance(
        command=list(command),
        input_sha256={"plan": plan_sha256, **input_digests},
    )
    referenced = [
        *plan_model.evidence.values(),
        *([plan_model.candidate_replay] if plan_model.candidate_replay else []),
        *plan_model.lifecycle_artifacts.values(),
        *plan_model.autoware_artifacts.values(),
    ]
    evaluation = CameraImuServiceEvaluation(
        evaluation_id=evaluation_id or f"{plan_model.plan_id}-evaluation",
        plan_sha256=plan_sha256,
        status=status,
        decision=status,
        reasons=_unique(errors),
        digest_checks=digest_checks,
        gates=gates,
        required_evidence_roles=list(plan_model.required_evidence_roles),
        missing_evidence_roles=missing_roles,
        metric_checks=metric_checks,
        metric_observations=observations,
        changed_components=changed,
        unallowlisted_changed_components=unallowlisted,
        bindings=bindings,
        candidate_replay=replay,
        referenced_artifacts=referenced,
        provenance=provenance,
    ).with_artifact_digest()
    if output is not None:
        evaluation.save(output)
    return evaluation


def evaluate_camera_imu_service_plan(
    plan: CameraImuServicePlan | str | Path,
    *,
    candidate_replay: ReplayResult | str | Path | Mapping[str, Any] | None = None,
    output: str | Path | None = None,
    evaluation_id: str | None = None,
    command: Sequence[str] = (),
) -> CameraImuServiceEvaluation:
    """Compatibility alias for callers that name the evaluated plan."""

    return evaluate_camera_imu_service(
        plan,
        candidate_replay=candidate_replay,
        output=output,
        evaluation_id=evaluation_id,
        command=command,
    )


def verify_camera_imu_service(
    path: str | Path,
    *,
    plan: str | Path | None = None,
) -> CameraImuServiceVerification:
    """Verify a plan/evaluation self-digest and all digest-bound references."""

    artifact_path = Path(path)
    errors: list[str] = []
    checked = 0
    kind: Literal["plan", "evaluation", "unknown"] = "unknown"
    try:
        payload = read_mapping(artifact_path)
        schema_version = payload.get("schema_version")
        if schema_version == CAMERA_IMU_SERVICE_PLAN_SCHEMA_VERSION:
            kind = "plan"
            model = CameraImuServicePlan.model_validate(payload)
            model.verify_artifact_digest()
            source_errors, digests = _verify_plan_inputs(
                model, artifact_path.parent, require_provenance=True
            )
            errors.extend(source_errors)
            checked = len(digests)
        elif schema_version == CAMERA_IMU_SERVICE_EVALUATION_SCHEMA_VERSION:
            kind = "evaluation"
            evaluation = CameraImuServiceEvaluation.model_validate(payload)
            evaluation.verify_artifact_digest()
            source_base = artifact_path.parent
            plan_model: CameraImuServicePlan | None = None
            if plan is not None:
                plan_model = load_camera_imu_service_plan(plan)
                source_base = Path(plan).parent
            for index, ref in enumerate(evaluation.referenced_artifacts):
                source_errors, _ = _verify_ref(
                    ref, source_base, f"evaluation referenced_artifact[{index}]"
                )
                errors.extend(source_errors)
            checked = len(evaluation.referenced_artifacts)
            if plan_model is not None and evaluation.plan_sha256 != plan_model.artifact_sha256:
                errors.append("evaluation plan_sha256 does not match supplied plan")
        else:
            errors.append(f"unsupported schema_version {schema_version!r}")
    except Exception as exc:
        errors.append(str(exc))
    return CameraImuServiceVerification(
        path=artifact_path.as_posix(),
        kind=kind,
        valid=not errors,
        checked_digest_count=checked,
        errors=_unique(errors),
    )


def verify_camera_imu_service_artifact(
    path: str | Path,
    *,
    plan: str | Path | None = None,
) -> CameraImuServiceVerification:
    """Compatibility alias for :func:`verify_camera_imu_service`."""

    return verify_camera_imu_service(path, plan=plan)


def _load_required_evidence(
    plan: CameraImuServicePlan,
    base_dir: Path,
    errors: list[str],
) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    loaded: dict[str, Mapping[str, Any]] = {}
    missing: list[str] = []
    by_normalized = {_normalize_role(key): (key, ref) for key, ref in plan.evidence.items()}
    for role in plan.required_evidence_roles:
        item = by_normalized.get(_normalize_role(role))
        if item is None:
            missing.append(role)
            continue
        source_role, ref = item
        raw = _load_ref_mapping(ref, base_dir, f"evidence:{source_role}", errors)
        if raw is not None:
            loaded[role] = raw
    return loaded, missing


def _load_ref_mapping(
    ref: DigestRef,
    base_dir: Path,
    name: str,
    errors: list[str],
) -> Mapping[str, Any] | None:
    if ref.path is None:
        if ref.payload is None:
            errors.append(f"{name} has no path or payload")
            return None
        if _canonical_sha256(ref.payload) != ref.sha256:
            errors.append(f"{name} inline digest mismatch")
        if not isinstance(ref.payload, Mapping):
            errors.append(f"{name} payload must be a mapping")
            return None
        return ref.payload
    path = _resolve_path(ref.path, base_dir)
    try:
        raw = read_mapping(path)
    except Exception as exc:
        errors.append(f"{name} cannot be read: {exc}")
        return None
    source_errors, _ = _verify_ref(ref, base_dir, name)
    errors.extend(source_errors)
    return raw


def _load_replay_evidence(
    source: ReplayResult | str | Path | Mapping[str, Any] | DigestRef | None,
    base_dir: Path,
    errors: list[str],
) -> CameraImuReplayEvidence | None:
    if source is None:
        errors.append("candidate replay result is required")
        return None
    path: Path | None = None
    declared_sha: str | None = None
    raw: Mapping[str, Any] | None = None
    if isinstance(source, DigestRef):
        declared_sha = source.sha256
        if source.path is not None:
            path = _resolve_path(source.path, base_dir)
            ref_errors, _ = _verify_ref(source, base_dir, "candidate replay")
            errors.extend(ref_errors)
            try:
                raw = read_mapping(path)
            except Exception as exc:
                errors.append(f"candidate replay cannot be read: {exc}")
                return None
        elif isinstance(source.payload, Mapping):
            raw = source.payload
        else:
            errors.append("candidate replay reference has no path or mapping payload")
            return None
    elif isinstance(source, ReplayResult):
        return _replay_evidence_from_model(source, None, None, errors)
    elif isinstance(source, Mapping):
        raw = source
    else:
        path = _resolve_path(str(source), base_dir)
        declared_sha = sha256_path(path)
        try:
            raw = read_mapping(path)
        except Exception as exc:
            errors.append(f"candidate replay cannot be read: {exc}")
            return None
    if raw is None:
        return None
    return _replay_evidence_from_mapping(raw, path, declared_sha, errors)


def _replay_evidence_from_mapping(
    raw: Mapping[str, Any],
    path: Path | None,
    declared_sha: str | None,
    errors: list[str],
) -> CameraImuReplayEvidence | None:
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
        if isinstance(raw_artifact_sha, str) and _self_digest_mapping(raw) != raw_artifact_sha:
            errors.append("candidate replay self-digest mismatch")
        gates = _gate_mapping(raw.get("gates"))
        return CameraImuReplayEvidence(
            path=path.as_posix() if path else None,
            sha256=declared_sha,
            artifact_sha256=raw_artifact_sha if isinstance(raw_artifact_sha, str) else None,
            status=cast(Literal["PASS", "WARN", "FAIL", "BLOCKED"], status),
            decision=cast(
                Literal["ADOPT", "HOLD", "REJECT", "BLOCKED", "NOT_APPLICABLE"], decision
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


def _replay_evidence_from_model(
    result: ReplayResult,
    path: Path | None,
    declared_sha: str | None,
    errors: list[str],
) -> CameraImuReplayEvidence:
    try:
        result.verify_artifact_digest()
    except Exception as exc:
        errors.append(f"candidate replay self-digest mismatch: {exc}")
    result_base = path.parent if path is not None else Path.cwd()
    for stage in result.stages:
        if stage.artifact is None:
            continue
        actual = sha256_path(_resolve_path(stage.artifact.path, result_base))
        if actual != stage.artifact.sha256:
            errors.append(f"candidate replay stage digest mismatch for {stage.stage_id}")
    for name, ref in result.artifacts.items():
        actual = sha256_path(_resolve_path(ref.path, result_base))
        if actual != ref.sha256:
            errors.append(f"candidate replay artifact digest mismatch for {name}")
    return CameraImuReplayEvidence(
        path=path.as_posix() if path else None,
        sha256=declared_sha,
        artifact_sha256=result.artifact_sha256,
        status=result.status,
        decision=result.decision,
        gates={gate.gate_id: gate.status for gate in result.gates},
    )


def _verify_plan_inputs(
    plan: CameraImuServicePlan,
    base_dir: Path,
    *,
    require_provenance: bool = False,
) -> tuple[list[str], dict[str, str]]:
    errors: list[str] = []
    digests: dict[str, str] = {}
    refs: list[tuple[str, DigestRef]] = []
    refs.extend((f"evidence:{key}", ref) for key, ref in plan.evidence.items())
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
                    f"{name} provenance input digest mismatch: declared={declared}, actual={digest}"
                )
    return _unique(errors), digests


def _rebase_plan_references(
    plan: CameraImuServicePlan,
    source_base: Path,
    output_base: Path,
) -> CameraImuServicePlan:
    """Rebase all relative refs so an emitted plan works from its own folder."""

    def rebase(ref: DigestRef) -> DigestRef:
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
            "evidence": {key: rebase(ref) for key, ref in plan.evidence.items()},
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
    ref: DigestRef,
    base_dir: Path,
    name: str,
) -> tuple[list[str], str | None]:
    if ref.path is None:
        if ref.payload is None:
            return [f"{name} has no path or payload"], None
        inline_digest = _canonical_sha256(ref.payload)
        if inline_digest != ref.sha256:
            return [
                f"{name} inline digest mismatch: declared={ref.sha256}, actual={inline_digest}"
            ], None
        return [], inline_digest
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


def _collect_observations(
    plan: CameraImuServicePlan,
    evidence: Mapping[str, Mapping[str, Any]],
) -> list[MetricObservation]:
    observations = list(plan.metric_observations)
    for role, raw in evidence.items():
        value = raw.get("metric_observations", raw.get("metrics", raw.get("observations")))
        if isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping):
                    try:
                        observations.append(
                            MetricObservation.model_validate({**item, "evidence_role": role})
                        )
                    except Exception:
                        # Invalid evidence is recorded by the metric evaluator
                        # through the missing required metric reason.
                        continue
    return observations


def _check_bindings(
    plan: CameraImuServicePlan,
    evidence: Mapping[str, Mapping[str, Any]],
    observations: Sequence[MetricObservation],
) -> tuple[dict[str, str], list[str]]:
    bindings: dict[str, str] = {}
    if plan.protocol_id is not None:
        bindings["protocol_id"] = plan.protocol_id
    if plan.capture_id is not None:
        bindings["capture_id"] = plan.capture_id
    candidates: list[dict[str, str]] = []
    for raw in evidence.values():
        found = _binding_fields(raw)
        if found:
            candidates.append(found)
            for key, value in found.items():
                bindings.setdefault(key, value)
    for observation in observations:
        found = dict(_binding_fields(observation.provenance))
        if observation.protocol_id is not None:
            found["protocol_id"] = observation.protocol_id
        if observation.capture_id is not None:
            found["capture_id"] = observation.capture_id
        if found:
            candidates.append(found)
    errors: list[str] = []
    for key in ("protocol_id", "capture_id"):
        if key not in bindings:
            errors.append(f"{key} binding is missing")
    expected = {
        key: value
        for key, value in bindings.items()
        if key in {"protocol_id", "capture_id"}
    }
    for index, found in enumerate(candidates):
        for key, expected_value in expected.items():
            if found.get(key) != expected_value:
                errors.append(
                    f"evidence binding mismatch for {key}: expected {expected_value!r}, "
                    f"observed {found.get(key)!r} in item {index}"
                )
    return bindings, _unique(errors)


def _evaluate_metrics(
    plan: CameraImuServicePlan,
    observations: Sequence[MetricObservation],
) -> tuple[dict[str, ServiceCheckStatus], list[str]]:
    aliases: dict[str, tuple[set[str], set[str], set[str]]] = {
        "clock_offset": (
            {"clock_offset", "clock_offset_s", "time_offset", "time_offset_s"},
            {"s", "sec", "second", "seconds"},
            {"train", "holdout"},
        ),
        "rotation_holdout_rmse": (
            {"rotation_holdout_rmse", "rotation_holdout_rmse_rad", "rotation_rmse"},
            {"rad", "radian", "radians"},
            {"holdout"},
        ),
        "lever_arm_holdout_rmse": (
            {
                "lever_arm_holdout_rmse",
                "lever_arm_holdout_rmse_m_s2",
                "lever_arm_rmse_m_s2",
            },
            {"m/s^2", "m/s²", "m s^-2", "m_s2"},
            {"holdout"},
        ),
        "bias_norm": (
            {"bias_norm", "bias_norm_m_s2", "imu_bias_norm"},
            {"m/s^2", "m/s²", "m s^-2", "m_s2", "rad/s", "mixed"},
            {"train", "holdout"},
        ),
        "known_bad_delta": (
            {"known_bad_delta", "known_bad_improvement", "known_bad_margin"},
            {"", "delta", "unitless", "1"},
            {"known_bad"},
        ),
        "observability_rank": (
            {"observability_rank", "rank", "observability"},
            {"", "rank", "unitless", "1"},
            {"train", "holdout"},
        ),
    }
    checks: dict[str, ServiceCheckStatus] = {}
    errors: list[str] = []
    values: dict[str, MetricObservation] = {}
    for name, (metric_ids, units, splits) in aliases.items():
        matches = [
            item
            for item in observations
            if _normalize_role(item.metric_id) in {_normalize_role(value) for value in metric_ids}
        ]
        valid = [
            item
            for item in matches
            if item.unit.lower() in units and item.split in splits and item.status == "PASS"
        ]
        if not valid:
            checks[name] = "FAIL"
            errors.append(
                f"required metric {name} is missing or has invalid unit/split/status"
            )
            continue
        values[name] = valid[-1]
        checks[name] = "PASS"
    if "clock_offset" in values and abs(values["clock_offset"].value) > plan.max_clock_offset_abs_s:
        checks["clock_offset"] = "FAIL"
        errors.append(
            f"clock offset {abs(values['clock_offset'].value):.6g} s exceeds "
            f"budget {plan.max_clock_offset_abs_s:.6g} s"
        )
    if (
        "rotation_holdout_rmse" in values
        and values["rotation_holdout_rmse"].value > plan.max_rotation_holdout_rmse_rad
    ):
        checks["rotation_holdout_rmse"] = "FAIL"
        errors.append("rotation holdout RMSE exceeds budget")
    if (
        "lever_arm_holdout_rmse" in values
        and values["lever_arm_holdout_rmse"].value > plan.max_lever_arm_holdout_rmse_m_s2
    ):
        checks["lever_arm_holdout_rmse"] = "FAIL"
        errors.append("lever-arm holdout RMSE exceeds budget")
    if "bias_norm" in values and values["bias_norm"].value > plan.max_bias_norm:
        checks["bias_norm"] = "FAIL"
        errors.append("IMU bias norm exceeds budget")
    if "known_bad_delta" in values and values["known_bad_delta"].value < plan.min_known_bad_delta:
        checks["known_bad_delta"] = "FAIL"
        errors.append("known-bad delta is below minimum")
    if (
        "observability_rank" in values
        and values["observability_rank"].value < plan.min_observability_rank
    ):
        checks["observability_rank"] = "FAIL"
        errors.append("observability rank is below minimum")
    return checks, _unique(errors)


def _changed_components(
    plan: CameraImuServicePlan,
    evidence: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    changed = list(plan.observed_changed_components)
    for key in sorted(set(plan.baseline_components) | set(plan.candidate_components)):
        if plan.baseline_components.get(key) != plan.candidate_components.get(key):
            changed.append(key)
    for raw in evidence.values():
        value = raw.get("observed_changed_components", raw.get("changed_components"))
        if isinstance(value, list):
            changed.extend(str(item) for item in value)
    return sorted(set(changed))


def _binding_fields(value: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, raw in value.items():
        normalized = _normalize_role(str(key))
        if normalized in {
            "protocol_id",
            "protocol_sha256",
            "protocol_digest",
            "capture_id",
            "capture_sha256",
            "capture_digest",
        } and isinstance(raw, (str, int, float, bool)):
            result[normalized] = str(raw)
        elif isinstance(raw, Mapping) and normalized in {"provenance", "binding", "bindings"}:
            result.update(_binding_fields(raw))
    return result


def _gate_mapping(value: object) -> dict[str, ServiceCheckStatus]:
    result: dict[str, ServiceCheckStatus] = {}
    if isinstance(value, Mapping):
        for key, raw in value.items():
            if raw in {"PASS", "WARN", "FAIL", "BLOCKED"}:
                result[str(key)] = cast(ServiceCheckStatus, raw)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, Mapping) and item.get("gate_id") and item.get("status") in {
                "PASS",
                "WARN",
                "FAIL",
                "BLOCKED",
            }:
                result[str(item["gate_id"])] = cast(ServiceCheckStatus, item["status"])
    return result


def _replay_gate_status(replay: CameraImuReplayEvidence, name: str) -> ServiceCheckStatus:
    for gate_id, status in replay.gates.items():
        if name in _normalize_role(gate_id):
            return status
    return "FAIL"


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    relative = base_dir / path
    return relative if relative.exists() else path


def _canonical_sha256(value: object) -> str:
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
    return _canonical_sha256(payload)


def _normalize_role(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


# Concise aliases used by downstream callers and older integration code.
CameraImuSensorIdentity = SensorIdentity
CameraImuDigestRef = DigestRef
CameraImuMetricObservation = MetricObservation
CameraImuServicePlanArtifact = CameraImuServicePlan
CameraImuServiceEvaluationArtifact = CameraImuServiceEvaluation
