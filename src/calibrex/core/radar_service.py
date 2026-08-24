"""Digest-bound, read-only radar replacement and reverification service.

The radar service is deliberately an admission boundary, not a calibration
solver.  It binds an old/new automotive radar identity to captured
``radar_msgs/RadarScan`` evidence and cross-modal radar--LiDAR/camera metrics,
then emits a self-digested ``READY`` or ``HOLD`` record.  Lifecycle and
Autoware references are verified read-only; this module never edits either
system.
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
from calibrex.core.camera_imu_service import DigestRef, MetricObservation
from calibrex.core.exceptions import CalibrexError
from calibrex.core.io import read_mapping, write_mapping_atomic
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.raw_replay import ReplayResult, load_replay_result
from calibrex.core.result import StrictModel

RADAR_SERVICE_PLAN_SCHEMA_VERSION: Literal["slac.radar_service_plan/v0.1"] = (
    "slac.radar_service_plan/v0.1"
)
RADAR_SERVICE_EVALUATION_SCHEMA_VERSION: Literal["slac.radar_service_evaluation/v0.1"] = (
    "slac.radar_service_evaluation/v0.1"
)

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_ZERO_SHA256 = "0" * 64

ServiceCheckStatus = Literal["PASS", "WARN", "FAIL", "BLOCKED"]
ServiceReadiness = Literal["READY", "HOLD"]

# These controls are the minimum independent evidence roles.  A plan can add
# ``radar_lidar_association`` and ``radar_camera`` (or a local equivalent)
# when cross-modal artifacts are required for a particular vehicle kit.
DEFAULT_REQUIRED_EVIDENCE_ROLES: tuple[str, ...] = (
    "protocol",
    "capture",
    "holdout",
    "known_bad",
    "observability",
)


class RadarServiceError(CalibrexError):
    """Raised when a radar service artifact is malformed."""


class RadarServiceProvenance(StrictModel):
    """Portable provenance carried by generated radar service artifacts."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    producer: Literal["calibrex"] = "calibrex"
    tool_name: str = "calibrex.radar-service"
    tool_version: str = __version__
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    # Input definitions must not silently inherit checkout state.  Generation
    # paths set this field explicitly, preserving reproducible self-digests.
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    input_sha256: dict[str, str] = Field(default_factory=dict)
    artifact_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    artifact_digest_scope: str = (
        "canonical artifact excluding artifact_sha256 and "
        "provenance.artifact_sha256/generated_at"
    )


class RadarSensorIdentity(StrictModel):
    """Stable physical identity for a radar before or after replacement."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    sensor_id: str = Field(
        min_length=1,
        validation_alias=AliasChoices("sensor_id", "radar_id", "id"),
    )
    serial: str = Field(min_length=1)
    model: str = Field(min_length=1)
    firmware: str = Field(min_length=1)
    mount: str = Field(min_length=1)
    frame: str | None = None
    sensor_type: Literal["radar"] | None = Field(
        default=None,
        validation_alias=AliasChoices("sensor_type", "type", "kind"),
    )
    sensor_kit_id: str | None = None


class RadarReplayEvidence(StrictModel):
    """Digest and gate summary copied from a raw replay result."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    role: str = "candidate_replay"
    path: str | None = None
    sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    artifact_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    status: Literal["PASS", "WARN", "FAIL", "BLOCKED"]
    decision: Literal["ADOPT", "HOLD", "REJECT", "BLOCKED", "NOT_APPLICABLE"]
    gates: dict[str, ServiceCheckStatus] = Field(default_factory=dict)
    source_sha256: dict[str, str] = Field(default_factory=dict)


class RadarServicePlan(StrictModel):
    """Immutable radar replacement scope, inputs, evidence, and budgets."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["slac.radar_service_plan/v0.1"] = (
        RADAR_SERVICE_PLAN_SCHEMA_VERSION
    )
    plan_id: str = Field(
        min_length=1,
        validation_alias=AliasChoices("plan_id", "service_id", "id"),
    )
    vehicle_id: str = Field(min_length=1)
    sensor_kit_id: str | None = None
    old_radar: RadarSensorIdentity = Field(
        validation_alias=AliasChoices("old_radar", "old_radar_sensor", "old_sensor")
    )
    new_radar: RadarSensorIdentity = Field(
        validation_alias=AliasChoices("new_radar", "new_radar_sensor", "new_sensor")
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

    # Explicit convention fields prevent a replacement from silently
    # changing the meaning of a signed Doppler velocity or frame transform.
    doppler_sign_convention: str = Field(
        default="positive_away",
        min_length=1,
        validation_alias=AliasChoices(
            "doppler_sign_convention", "doppler_convention", "doppler_sign"
        ),
    )
    frame_convention: str = Field(
        default="radar_sensor_frame_x_forward_y_left_z_up",
        min_length=1,
        validation_alias=AliasChoices("frame_convention", "radar_frame_convention"),
    )

    max_abs_range_bias_m: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("max_abs_range_bias_m", "max_range_bias_m"),
    )
    max_abs_azimuth_bias_deg: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("max_abs_azimuth_bias_deg", "max_azimuth_bias_deg"),
    )
    max_doppler_scale_error_ratio: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices(
            "max_doppler_scale_error_ratio", "max_doppler_scale_error"
        ),
    )
    max_time_offset_abs_s: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("max_time_offset_abs_s", "max_time_offset_s"),
    )
    max_radar_lidar_association_rmse_m: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices(
            "max_radar_lidar_association_rmse_m",
            "max_association_rmse_m",
            "max_radar_lidar_rmse_m",
        ),
    )
    min_fov_overlap_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("min_fov_overlap_ratio", "fov_overlap_ratio_min"),
    )
    min_known_bad_delta: float = Field(default=0.0, ge=0.0)
    min_observability_rank: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("min_observability_rank", "observability_rank_min"),
    )

    protocol_id: str | None = None
    capture_id: str | None = None
    lifecycle_artifacts: dict[str, DigestRef] = Field(default_factory=dict)
    autoware_artifacts: dict[str, DigestRef] = Field(default_factory=dict)
    provenance: RadarServiceProvenance = Field(default_factory=RadarServiceProvenance)
    artifact_sha256: str = Field(default=_ZERO_SHA256, pattern=_SHA256_PATTERN)

    @field_validator("evidence", mode="before")
    @classmethod
    def _coerce_evidence_mapping(cls, value: object) -> object:
        """Accept a role-bearing list while serializing a stable role map."""

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
    def validate_plan(self) -> RadarServicePlan:
        """Reject ambiguous radar identities and duplicate declarations."""

        if self.old_radar.sensor_id == self.new_radar.sensor_id:
            raise ValueError("old_radar and new_radar must have distinct sensor_id values")
        for label, sensor in (("old_radar", self.old_radar), ("new_radar", self.new_radar)):
            if sensor.sensor_type not in {None, "radar"}:
                raise ValueError(f"{label}.sensor_type must be radar")
        if len(self.required_evidence_roles) != len(set(self.required_evidence_roles)):
            raise ValueError("required_evidence_roles must be unique")
        if any(not role for role in self.required_evidence_roles):
            raise ValueError("required_evidence_roles must not contain empty roles")
        if len(self.mutable_components) != len(set(self.mutable_components)):
            raise ValueError("mutable_components must be unique")
        if any(not component for component in self.mutable_components):
            raise ValueError("mutable_components must not contain empty component IDs")
        return self

    def with_artifact_digest(self) -> RadarServicePlan:
        """Return a copy with its canonical self-digest bound in provenance."""

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
        """Raise when the plan or bound provenance self-digest was tampered with."""

        expected = self.with_artifact_digest().artifact_sha256
        if expected != self.artifact_sha256:
            raise RadarServiceError(
                f"radar plan self-digest mismatch: declared={self.artifact_sha256}, "
                f"expected={expected}"
            )
        if self.provenance.artifact_sha256 != self.artifact_sha256:
            raise RadarServiceError("radar plan provenance digest is not bound")

    def save(self, path: str | Path) -> None:
        """Write a digest-complete YAML or JSON plan."""

        artifact = self.with_artifact_digest()
        write_mapping_atomic(Path(path), artifact.model_dump(mode="json", exclude_none=False))


class RadarServiceEvaluation(StrictModel):
    """Digest-bound READY/HOLD decision for a radar replacement."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["slac.radar_service_evaluation/v0.1"] = (
        RADAR_SERVICE_EVALUATION_SCHEMA_VERSION
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
    doppler_sign_convention: str = Field(min_length=1)
    frame_convention: str = Field(min_length=1)
    candidate_replay: RadarReplayEvidence | None = None
    referenced_artifacts: list[DigestRef] = Field(default_factory=list)
    provenance: RadarServiceProvenance = Field(default_factory=RadarServiceProvenance)
    artifact_sha256: str = Field(default=_ZERO_SHA256, pattern=_SHA256_PATTERN)

    def with_artifact_digest(self) -> RadarServiceEvaluation:
        """Return a copy with its canonical self-digest bound in provenance."""

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
        """Raise when the evaluation or provenance self-digest was tampered with."""

        expected = self.with_artifact_digest().artifact_sha256
        if expected != self.artifact_sha256:
            raise RadarServiceError(
                "radar evaluation self-digest mismatch: "
                f"declared={self.artifact_sha256}, expected={expected}"
            )
        if self.provenance.artifact_sha256 != self.artifact_sha256:
            raise RadarServiceError("radar evaluation provenance digest is not bound")

    def save(self, path: str | Path) -> None:
        """Write a digest-complete YAML or JSON evaluation."""

        artifact = self.with_artifact_digest()
        write_mapping_atomic(Path(path), artifact.model_dump(mode="json", exclude_none=False))


class RadarServiceVerification(StrictModel):
    """Non-throwing verification summary returned by verify_radar_service."""

    path: str
    kind: Literal["plan", "evaluation", "unknown"]
    valid: bool
    checked_digest_count: int = Field(default=0, ge=0)
    errors: list[str] = Field(default_factory=list)


def radar_service_plan_json_schema() -> dict[str, Any]:
    """Return the generated plan JSON schema."""

    return RadarServicePlan.model_json_schema()


def radar_service_evaluation_json_schema() -> dict[str, Any]:
    """Return the generated evaluation JSON schema."""

    return RadarServiceEvaluation.model_json_schema()


def load_radar_service_plan(path: str | Path) -> RadarServicePlan:
    """Load a plan and verify its self-digest and all source references."""

    plan_path = Path(path)
    try:
        plan = RadarServicePlan.model_validate(read_mapping(plan_path))
        plan.verify_artifact_digest()
        errors, _ = _verify_plan_inputs(plan, plan_path.parent, require_provenance=True)
    except Exception as exc:
        if isinstance(exc, RadarServiceError):
            raise
        raise RadarServiceError(f"invalid radar plan {plan_path}: {exc}") from exc
    if errors:
        raise RadarServiceError("; ".join(errors))
    return plan


def load_radar_service_evaluation(path: str | Path) -> RadarServiceEvaluation:
    """Load and verify an evaluation self-digest."""

    evaluation_path = Path(path)
    try:
        evaluation = RadarServiceEvaluation.model_validate(read_mapping(evaluation_path))
        evaluation.verify_artifact_digest()
    except Exception as exc:
        if isinstance(exc, RadarServiceError):
            raise
        raise RadarServiceError(f"invalid radar evaluation {evaluation_path}: {exc}") from exc
    return evaluation


def build_radar_service_plan(
    definition: RadarServicePlan | str | Path,
    *,
    output: str | Path | None = None,
    command: Sequence[str] = (),
) -> RadarServicePlan:
    """Verify declared inputs, rebase paths, and materialize a plan."""

    definition_path: Path | None = None
    if isinstance(definition, RadarServicePlan):
        plan = definition
    else:
        definition_path = Path(definition)
        try:
            plan = RadarServicePlan.model_validate(read_mapping(definition_path))
        except Exception as exc:
            raise RadarServiceError(f"invalid radar plan input: {exc}") from exc
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
        raise RadarServiceError("; ".join(errors))
    plan = _rebase_plan_references(plan, source_base, output_base)
    provenance = plan.provenance.model_copy(
        update={
            "command": list(command) or plan.provenance.command,
            "input_sha256": {**plan.provenance.input_sha256, **input_digests},
            "git_commit": git_commit(),
        }
    )
    result = plan.model_copy(update={"provenance": provenance}).with_artifact_digest()
    if output is not None:
        result.save(output)
    return result


def plan_radar_service(
    definition: RadarServicePlan | str | Path,
    *,
    output: str | Path | None = None,
    command: Sequence[str] = (),
) -> RadarServicePlan:
    """Compatibility alias for build_radar_service_plan."""

    return build_radar_service_plan(definition, output=output, command=command)


def evaluate_radar_service(
    plan: RadarServicePlan | str | Path,
    *,
    candidate_replay: ReplayResult | str | Path | Mapping[str, Any] | None = None,
    output: str | Path | None = None,
    evaluation_id: str | None = None,
    command: Sequence[str] = (),
) -> RadarServiceEvaluation:
    """Evaluate all radar v0.1 gates fail-closed and optionally save output."""

    plan_path: Path | None = None
    errors: list[str] = []
    if isinstance(plan, RadarServicePlan):
        plan_model = plan
        base_dir = Path.cwd()
    else:
        plan_path = Path(plan)
        base_dir = plan_path.parent
        try:
            plan_model = RadarServicePlan.model_validate(read_mapping(plan_path))
        except Exception as exc:
            raise RadarServiceError(f"invalid radar plan input: {exc}") from exc

    if plan_model.artifact_sha256 == _ZERO_SHA256:
        errors.append("plan artifact_sha256 is unset")
    else:
        try:
            plan_model.verify_artifact_digest()
        except RadarServiceError as exc:
            errors.append(str(exc))

    input_errors, input_digests = _verify_plan_inputs(
        plan_model, base_dir, require_provenance=True
    )
    errors.extend(input_errors)
    replay_source: ReplayResult | str | Path | Mapping[str, Any] | DigestRef | None = (
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
    convention_errors = _check_conventions(plan_model, loaded_evidence, observations)
    errors.extend(convention_errors)
    digest_checks["conventions"] = "PASS" if not convention_errors else "FAIL"

    all_pass = not errors and all(value == "PASS" for value in gates.values())
    status: ServiceReadiness = "READY" if all_pass else "HOLD"
    plan_sha256 = plan_model.artifact_sha256
    if plan_sha256 == _ZERO_SHA256:
        plan_sha256 = plan_model.with_artifact_digest().artifact_sha256
    provenance = RadarServiceProvenance(
        git_commit=git_commit(),
        command=list(command),
        input_sha256={"plan": plan_sha256, **input_digests},
    )
    referenced = [
        *plan_model.evidence.values(),
        *([plan_model.candidate_replay] if plan_model.candidate_replay else []),
        *plan_model.lifecycle_artifacts.values(),
        *plan_model.autoware_artifacts.values(),
    ]
    evaluation = RadarServiceEvaluation(
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
        doppler_sign_convention=plan_model.doppler_sign_convention,
        frame_convention=plan_model.frame_convention,
        candidate_replay=replay,
        referenced_artifacts=referenced,
        provenance=provenance,
    ).with_artifact_digest()
    if output is not None:
        # References in an evaluation are interpreted relative to the plan
        # directory when a plan path is supplied.  Rebase them for a sibling
        # or otherwise different output directory so verify remains portable.
        if plan_path is not None and Path(output).parent != plan_path.parent:
            evaluation = evaluation.model_copy(
                update={
                    "referenced_artifacts": _rebase_refs_for_output(
                        evaluation.referenced_artifacts,
                        plan_path.parent,
                        Path(output).parent,
                    )
                }
            ).with_artifact_digest()
        evaluation.save(output)
    return evaluation


def evaluate_radar_service_plan(
    plan: RadarServicePlan | str | Path,
    *,
    candidate_replay: ReplayResult | str | Path | Mapping[str, Any] | None = None,
    output: str | Path | None = None,
    evaluation_id: str | None = None,
    command: Sequence[str] = (),
) -> RadarServiceEvaluation:
    """Compatibility alias for callers that name the evaluated plan."""

    return evaluate_radar_service(
        plan,
        candidate_replay=candidate_replay,
        output=output,
        evaluation_id=evaluation_id,
        command=command,
    )


def verify_radar_service(
    path: str | Path,
    *,
    plan: str | Path | None = None,
) -> RadarServiceVerification:
    """Verify a plan/evaluation self-digest and all digest-bound references."""

    artifact_path = Path(path)
    errors: list[str] = []
    checked = 0
    kind: Literal["plan", "evaluation", "unknown"] = "unknown"
    try:
        payload = read_mapping(artifact_path)
        schema_version = payload.get("schema_version")
        if schema_version == RADAR_SERVICE_PLAN_SCHEMA_VERSION:
            kind = "plan"
            model = RadarServicePlan.model_validate(payload)
            model.verify_artifact_digest()
            source_errors, digests = _verify_plan_inputs(
                model, artifact_path.parent, require_provenance=True
            )
            errors.extend(source_errors)
            checked = len(digests)
        elif schema_version == RADAR_SERVICE_EVALUATION_SCHEMA_VERSION:
            kind = "evaluation"
            evaluation = RadarServiceEvaluation.model_validate(payload)
            evaluation.verify_artifact_digest()
            source_base = artifact_path.parent
            plan_model: RadarServicePlan | None = None
            if plan is not None:
                plan_path = Path(plan)
                plan_model = load_radar_service_plan(plan_path)
                source_base = plan_path.parent
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
    return RadarServiceVerification(
        path=artifact_path.as_posix(),
        kind=kind,
        valid=not errors,
        checked_digest_count=checked,
        errors=_unique(errors),
    )


def verify_radar_service_artifact(
    path: str | Path,
    *,
    plan: str | Path | None = None,
) -> RadarServiceVerification:
    """Compatibility alias for verify_radar_service."""

    return verify_radar_service(path, plan=plan)


def _load_required_evidence(
    plan: RadarServicePlan,
    base_dir: Path,
    errors: list[str],
) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    loaded: dict[str, Mapping[str, Any]] = {}
    missing: list[str] = []
    by_normalized = {_normalize_role(key): (key, ref) for key, ref in plan.evidence.items()}
    for role in plan.required_evidence_roles:
        item = next(
            (
                by_normalized[candidate]
                for candidate in _role_candidates(role)
                if candidate in by_normalized
            ),
            None,
        )
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
) -> RadarReplayEvidence | None:
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
) -> RadarReplayEvidence | None:
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
        return RadarReplayEvidence(
            path=path.as_posix() if path else None,
            sha256=declared_sha,
            artifact_sha256=raw_artifact_sha if isinstance(raw_artifact_sha, str) else None,
            status=cast(Literal["PASS", "WARN", "FAIL", "BLOCKED"], status),
            decision=cast(
                Literal["ADOPT", "HOLD", "REJECT", "BLOCKED", "NOT_APPLICABLE"], decision
            ),
            gates=_gate_mapping(raw.get("gates")),
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
) -> RadarReplayEvidence:
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
    return RadarReplayEvidence(
        path=path.as_posix() if path else None,
        sha256=declared_sha,
        artifact_sha256=result.artifact_sha256,
        status=result.status,
        decision=result.decision,
        gates={gate.gate_id: gate.status for gate in result.gates},
    )


def _verify_plan_inputs(
    plan: RadarServicePlan,
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
            if require_provenance and declared is None:
                errors.append(f"{name} provenance input digest is missing")
            elif require_provenance and declared != digest:
                errors.append(
                    f"{name} provenance input digest mismatch: declared={declared}, actual={digest}"
                )
    return _unique(errors), digests


def _rebase_plan_references(
    plan: RadarServicePlan,
    source_base: Path,
    output_base: Path,
) -> RadarServicePlan:
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


def _rebase_refs_for_output(
    refs: Sequence[DigestRef], source_base: Path, output_base: Path
) -> list[DigestRef]:
    result: list[DigestRef] = []
    for ref in refs:
        if ref.path is None:
            result.append(ref)
            continue
        source_path = _resolve_path(ref.path, source_base)
        if not source_path.exists():
            result.append(ref)
            continue
        try:
            portable = os.path.relpath(source_path, output_base)
        except ValueError:
            portable = str(source_path.resolve())
        result.append(ref.model_copy(update={"path": Path(portable).as_posix()}))
    return result


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
    plan: RadarServicePlan,
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
                        continue
    return observations


def _check_bindings(
    plan: RadarServicePlan,
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
        found = _binding_fields(observation.provenance)
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
    expected = {key: bindings[key] for key in ("protocol_id", "capture_id") if key in bindings}
    for index, found in enumerate(candidates):
        for key, expected_value in expected.items():
            if found.get(key) != expected_value:
                errors.append(
                    f"evidence binding mismatch for {key}: expected {expected_value!r}, "
                    f"observed {found.get(key)!r} in item {index}"
                )
    return bindings, _unique(errors)


def _evaluate_metrics(
    plan: RadarServicePlan,
    observations: Sequence[MetricObservation],
) -> tuple[dict[str, ServiceCheckStatus], list[str]]:
    aliases: dict[str, tuple[set[str], set[str], set[str]]] = {
        "abs_range_bias": (
            {
                "range_bias",
                "range_bias_m",
                "abs_range_bias",
                "abs_range_bias_m",
                "range_bias_abs_m",
            },
            {"m", "meter", "meters"},
            {"holdout", "train"},
        ),
        "abs_azimuth_bias": (
            {
                "azimuth_bias",
                "azimuth_bias_deg",
                "abs_azimuth_bias",
                "abs_azimuth_bias_deg",
                "azimuth_bias_abs_deg",
            },
            {"deg", "degree", "degrees"},
            {"holdout", "train"},
        ),
        "doppler_scale_error": (
            {"doppler_scale_error", "doppler_scale_error_ratio", "doppler_scale"},
            {"ratio", "unitless", "1", ""},
            {"holdout", "train"},
        ),
        "time_offset": (
            {
                "time_offset",
                "time_offset_s",
                "time_offset_abs_s",
                "clock_offset",
                "clock_offset_s",
            },
            {"s", "sec", "second", "seconds"},
            {"holdout", "train"},
        ),
        "radar_lidar_association_rmse": (
            {
                "radar_lidar_association_rmse",
                "radar_lidar_association_rmse_m",
                "association_rmse",
                "association_rmse_m",
                "radar_lidar_rmse_m",
            },
            {"m", "meter", "meters"},
            {"holdout"},
        ),
        "fov_overlap_ratio": (
            {"fov_overlap", "fov_overlap_ratio", "field_of_view_overlap_ratio"},
            {"ratio", "unitless", "1", ""},
            {"holdout", "train"},
        ),
        "known_bad_delta": (
            {"known_bad_delta", "known_bad_improvement", "known_bad_margin"},
            {"", "delta", "unitless", "1", "ratio"},
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
            if _normalize_role(item.metric_id)
            in {_normalize_role(value) for value in metric_ids}
        ]
        valid = [
            item
            for item in matches
            if (
                item.unit.lower() in units
                and item.split in splits
                and item.status == "PASS"
                and math.isfinite(item.value)
            )
        ]
        if not valid:
            checks[name] = "FAIL"
            errors.append(
                f"required metric {name} is missing or has invalid unit/split/status"
            )
            continue
        values[name] = valid[-1]
        checks[name] = "PASS"

    upper_bounds = (
        ("abs_range_bias", plan.max_abs_range_bias_m, "range bias", True),
        ("abs_azimuth_bias", plan.max_abs_azimuth_bias_deg, "azimuth bias", True),
        ("doppler_scale_error", plan.max_doppler_scale_error_ratio, "Doppler scale error", True),
        ("time_offset", plan.max_time_offset_abs_s, "time offset", True),
        (
            "radar_lidar_association_rmse",
            plan.max_radar_lidar_association_rmse_m,
            "radar-LiDAR association RMSE",
            False,
        ),
    )
    for name, budget, label, absolute in upper_bounds:
        item = values.get(name)
        if item is not None and (abs(item.value) if absolute else item.value) > budget:
            checks[name] = "FAIL"
            observed_value = abs(item.value) if absolute else item.value
            errors.append(
                f"{label} {observed_value:.6g} exceeds budget {budget:.6g}"
            )
    for name, observed, minimum, label in (
        (
            "fov_overlap_ratio",
            values.get("fov_overlap_ratio"),
            plan.min_fov_overlap_ratio,
            "FOV overlap ratio",
        ),
        (
            "known_bad_delta",
            values.get("known_bad_delta"),
            plan.min_known_bad_delta,
            "known-bad delta",
        ),
        (
            "observability_rank",
            values.get("observability_rank"),
            plan.min_observability_rank,
            "observability rank",
        ),
    ):
        if observed is not None and observed.value < minimum:
            checks[name] = "FAIL"
            errors.append(f"{label} {observed.value:.6g} is below minimum {minimum:.6g}")
    fov = values.get("fov_overlap_ratio")
    if fov is not None and not 0.0 <= fov.value <= 1.0:
        checks["fov_overlap_ratio"] = "FAIL"
        errors.append(f"FOV overlap ratio {fov.value:.6g} is outside [0, 1]")
    return checks, _unique(errors)


def _check_conventions(
    plan: RadarServicePlan,
    evidence: Mapping[str, Mapping[str, Any]],
    observations: Sequence[MetricObservation],
) -> list[str]:
    errors: list[str] = []
    values: list[Mapping[str, Any]] = [
        *evidence.values(),
        *(item.provenance for item in observations),
    ]
    for index, value in enumerate(values):
        found = _convention_fields(value)
        for key, expected in (
            ("doppler_sign_convention", plan.doppler_sign_convention),
            ("frame_convention", plan.frame_convention),
        ):
            if key in found and found[key] != expected:
                errors.append(
                    f"{key} mismatch: expected {expected!r}, observed "
                    f"{found[key]!r} in item {index}"
                )
    return _unique(errors)


def _changed_components(
    plan: RadarServicePlan,
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


def _convention_fields(value: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, raw in value.items():
        normalized = _normalize_role(str(key))
        if normalized in {"doppler_sign_convention", "doppler_convention", "doppler_sign"}:
            if isinstance(raw, (str, int, float, bool)):
                result["doppler_sign_convention"] = str(raw)
        elif normalized in {"frame_convention", "radar_frame_convention"}:
            if isinstance(raw, (str, int, float, bool)):
                result["frame_convention"] = str(raw)
        elif isinstance(raw, Mapping) and normalized in {"provenance", "binding", "bindings"}:
            result.update(_convention_fields(raw))
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


def _replay_gate_status(replay: RadarReplayEvidence, name: str) -> ServiceCheckStatus:
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


def _role_candidates(value: str) -> tuple[str, ...]:
    """Return stable aliases for common cross-modal evidence role spellings."""

    normalized = _normalize_role(value)
    aliases = {
        "radar_lidar": "radar_lidar_association",
        "radar_lidar_cross_modal": "radar_lidar_association",
        "radar_camera_cross_modal": "radar_camera",
    }
    canonical = aliases.get(normalized, normalized)
    return (normalized, canonical) if canonical != normalized else (normalized,)


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


# Concise aliases used by downstream adapters and integration code.
SensorIdentity = RadarSensorIdentity
RadarIdentity = RadarSensorIdentity
RadarServiceSensorIdentity = RadarSensorIdentity
RadarDigestRef = DigestRef
RadarArtifactRef = DigestRef
RadarServiceDigestRef = DigestRef
RadarMetricObservation = MetricObservation
RadarServiceMetricObservation = MetricObservation
RadarServicePlanArtifact = RadarServicePlan
RadarServiceEvaluationArtifact = RadarServiceEvaluation
