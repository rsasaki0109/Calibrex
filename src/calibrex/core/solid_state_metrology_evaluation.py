"""Schema-valid physical ground-truth evaluation for solid-state LiDAR."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator

from calibrex.core.geometry import SE3
from calibrex.core.provenance import sha256_path
from calibrex.core.result import StrictModel, TransformResult

SOLID_STATE_METROLOGY_EVALUATION_SCHEMA_VERSION: Literal[
    "slac.solid_state_metrology_evaluation/v0.1"
] = "slac.solid_state_metrology_evaluation/v0.1"

MetrologyExtrinsicMethod = Literal[
    "surveyed_rig",
    "optical_tracker",
    "robot_arm",
    "calibration_target",
    "mechanical_cad",
    "unknown",
]
MetrologyClockMethod = Literal[
    "hardware_trigger",
    "common_pps",
    "ptp",
    "external_timebase",
    "timestamp_injection",
    "unknown",
]
MetrologySessionStatus = Literal["usable", "rejected"]
MetrologyStatus = Literal["planned", "inconclusive", "pass", "fail"]
MetrologyDecision = Literal["collect", "review", "pass", "fail"]
MetrologyEvidenceRole = Literal[
    "reference",
    "session_reference",
    "session_capture",
    "estimate",
]
MetrologyEvidenceSourceStatus = Literal[
    "verified",
    "missing",
    "missing_digest",
    "mismatch",
]

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _validate_sha256_map(value: dict[str, str]) -> dict[str, str]:
    invalid = [key for key, digest in value.items() if not _SHA256_RE.fullmatch(digest)]
    if invalid:
        raise ValueError(f"source_sha256 contains invalid SHA-256 values: {invalid}")
    return value


class SolidStateMetrologyEvaluationProtocol(StrictModel):
    """Predeclared physical evidence requirements for one sensor pair."""

    name: str
    source_sensor: str
    target_sensor: str
    frame_convention: Literal["T_parent_child"] = "T_parent_child"
    required_extrinsic_method: MetrologyExtrinsicMethod = "surveyed_rig"
    required_clock_method: MetrologyClockMethod = "hardware_trigger"
    minimum_usable_sessions: int = Field(default=3, ge=1)
    minimum_remounts: int = Field(default=2, ge=1)
    require_downstream_metric: bool = True
    require_per_session_reference: bool = True
    notes: list[str] = Field(default_factory=list)


class SolidStateMetrologyEvaluationThresholds(StrictModel):
    """Absolute accuracy gates and maximum reference uncertainty."""

    max_rotation_error_deg: float = Field(default=0.5, gt=0.0)
    max_translation_error_m: float = Field(default=0.01, gt=0.0)
    max_time_offset_error_sec: float = Field(default=0.001, gt=0.0)
    max_reference_rotation_uncertainty_deg: float = Field(default=0.1, gt=0.0)
    max_reference_translation_uncertainty_m: float = Field(default=0.002, gt=0.0)
    max_reference_time_uncertainty_sec: float = Field(default=0.0002, gt=0.0)


class SolidStateMetrologyReference(StrictModel):
    """Independently measured spatial and temporal reference values."""

    extrinsic_method: MetrologyExtrinsicMethod = "unknown"
    clock_method: MetrologyClockMethod = "unknown"
    independent_of_solver: bool = False
    transform: TransformResult | None = None
    time_offset_sec: float | None = None
    rotation_uncertainty_deg: float | None = Field(default=None, ge=0.0)
    translation_uncertainty_m: float | None = Field(default=None, ge=0.0)
    time_uncertainty_sec: float | None = Field(default=None, ge=0.0)
    source_paths: list[str] = Field(default_factory=list)
    source_sha256: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    @field_validator("source_sha256")
    @classmethod
    def validate_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        return _validate_sha256_map(value)


class SolidStateMetrologyEstimate(StrictModel):
    """One solver estimate tied to a captured and remounted session."""

    id: str
    session_id: str
    transform: TransformResult | None = None
    time_offset_sec: float | None = None
    source_path: str | None = None
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    notes: list[str] = Field(default_factory=list)


class SolidStateMetrologySession(StrictModel):
    """One capture session, including its mechanical remount identity."""

    id: str
    remount_id: str
    reference: SolidStateMetrologyReference | None = None
    capture_path: str | None = None
    capture_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-fA-F]{64}$",
    )
    status: MetrologySessionStatus = "usable"
    notes: list[str] = Field(default_factory=list)


class SolidStateMetrologyDownstreamMetric(StrictModel):
    """A held-out physical-use metric not consumed by the solver."""

    name: str
    value: float | None = None
    baseline_value: float | None = None
    unit: str
    lower_is_better: bool = True
    held_out: bool = False
    independent_of_solver: bool = False
    max_value: float | None = None
    notes: list[str] = Field(default_factory=list)


class SolidStateMetrologyRunMetric(StrictModel):
    """Absolute errors and gate outcome for one solver estimate."""

    estimate_id: str
    session_id: str
    remount_id: str | None = None
    rotation_error_deg: float | None = Field(default=None, ge=0.0)
    translation_error_m: float | None = Field(default=None, ge=0.0)
    time_offset_error_sec: float | None = Field(default=None, ge=0.0)
    gate_passed: bool = False
    reasons: list[str] = Field(default_factory=list)


class SolidStateMetrologyEvaluationMetrics(StrictModel):
    """Aggregate metrics retained alongside every per-session result."""

    run_count: int = Field(default=0, ge=0)
    evaluated_run_count: int = Field(default=0, ge=0)
    passing_run_count: int = Field(default=0, ge=0)
    usable_session_count: int = Field(default=0, ge=0)
    remount_count: int = Field(default=0, ge=0)
    max_rotation_error_deg: float | None = Field(default=None, ge=0.0)
    max_translation_error_m: float | None = Field(default=None, ge=0.0)
    max_time_offset_error_sec: float | None = Field(default=None, ge=0.0)
    downstream_metric_passed: bool | None = None
    run_metrics: list[SolidStateMetrologyRunMetric] = Field(default_factory=list)


class SolidStateMetrologyEvidenceSourceCheck(StrictModel):
    """One digest check for a declared physical-evidence source."""

    role: MetrologyEvidenceRole
    owner_id: str
    path: str
    declared_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-fA-F]{64}$",
    )
    observed_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-fA-F]{64}$",
    )
    status: MetrologyEvidenceSourceStatus
    reason: str | None = None


class SolidStateMetrologyEvidenceIntegrity(StrictModel):
    """Recomputed source and relationship checks for a physical packet."""

    checked: bool = False
    passed: bool = False
    source_checks: list[SolidStateMetrologyEvidenceSourceCheck] = Field(
        default_factory=list
    )
    issues: list[str] = Field(default_factory=list)


class SolidStateMetrologyEvaluationProvenance(StrictModel):
    """Digest-bound provenance for the generated physical evaluation."""

    generator: str
    generator_version: str
    source_sha256: dict[str, str] = Field(default_factory=dict, min_length=1)
    command: list[str] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    notes: list[str] = Field(default_factory=list)

    @field_validator("source_sha256")
    @classmethod
    def validate_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        return _validate_sha256_map(value)


class SolidStateMetrologyEvaluationArtifact(StrictModel):
    """Physical ground-truth evaluation or an explicitly incomplete template."""

    schema_version: Literal[
        "slac.solid_state_metrology_evaluation/v0.1"
    ] = SOLID_STATE_METROLOGY_EVALUATION_SCHEMA_VERSION
    evaluation_id: str
    protocol: SolidStateMetrologyEvaluationProtocol
    reference: SolidStateMetrologyReference
    sessions: list[SolidStateMetrologySession] = Field(default_factory=list)
    estimates: list[SolidStateMetrologyEstimate] = Field(default_factory=list)
    downstream_metric: SolidStateMetrologyDownstreamMetric | None = None
    thresholds: SolidStateMetrologyEvaluationThresholds
    metrics: SolidStateMetrologyEvaluationMetrics = Field(
        default_factory=SolidStateMetrologyEvaluationMetrics
    )
    evidence_integrity: SolidStateMetrologyEvidenceIntegrity = Field(
        default_factory=SolidStateMetrologyEvidenceIntegrity
    )
    status: MetrologyStatus = "planned"
    decision: MetrologyDecision = "collect"
    reasons: list[str] = Field(default_factory=list)
    provenance: SolidStateMetrologyEvaluationProvenance


def _rotation_error_deg(reference: SE3, estimate: SE3) -> float:
    dot = abs(
        sum(
            reference_value * estimate_value
            for reference_value, estimate_value in zip(
                reference.rotation_quat_xyzw,
                estimate.rotation_quat_xyzw,
                strict=True,
            )
        )
    )
    return math.degrees(2.0 * math.acos(min(1.0, max(-1.0, dot))))


def _translation_error_m(reference: SE3, estimate: SE3) -> float:
    return math.dist(reference.translation_m, estimate.translation_m)


def _append_unique(reasons: list[str], value: str) -> None:
    if value not in reasons:
        reasons.append(value)


def _source_check(
    *,
    role: MetrologyEvidenceRole,
    owner_id: str,
    path: str,
    declared_sha256: str | None,
    base_dir: Path,
) -> SolidStateMetrologyEvidenceSourceCheck:
    """Recompute one declared source digest relative to the packet directory."""

    source_path = Path(path)
    if not source_path.is_absolute():
        source_path = base_dir / source_path
    observed_sha256 = sha256_path(source_path)
    if declared_sha256 is None:
        status: MetrologyEvidenceSourceStatus = "missing_digest"
        reason = "source_digest_missing"
    elif observed_sha256 is None:
        status = "missing"
        reason = "source_path_missing"
    elif observed_sha256.lower() != declared_sha256.lower():
        status = "mismatch"
        reason = "source_digest_mismatch"
    else:
        status = "verified"
        reason = None
    return SolidStateMetrologyEvidenceSourceCheck(
        role=role,
        owner_id=owner_id,
        path=path,
        declared_sha256=declared_sha256,
        observed_sha256=observed_sha256,
        status=status,
        reason=reason,
    )


def _append_reference_source_checks(
    *,
    reference: SolidStateMetrologyReference,
    role: MetrologyEvidenceRole,
    owner_id: str,
    base_dir: Path,
    issues: list[str],
    source_checks: list[SolidStateMetrologyEvidenceSourceCheck],
) -> None:
    """Append digest checks for one global or per-session reference."""

    prefix = "session_reference" if role == "session_reference" else "reference"
    reference_paths = set(reference.source_paths)
    if len(reference_paths) != len(reference.source_paths):
        _append_unique(issues, f"duplicate_{prefix}_source_path")
    if not reference.source_paths:
        _append_unique(issues, f"{prefix}_source_paths_missing")
    for source_path in reference.source_paths:
        declared_sha256 = reference.source_sha256.get(source_path)
        if declared_sha256 is None:
            _append_unique(issues, f"{prefix}_source_digest_missing")
        source_checks.append(
            _source_check(
                role=role,
                owner_id=owner_id,
                path=source_path,
                declared_sha256=declared_sha256,
                base_dir=base_dir,
            )
        )
    if set(reference.source_sha256) - reference_paths:
        _append_unique(issues, f"{prefix}_source_digest_without_path")


def verify_solid_state_metrology_evidence(
    artifact: SolidStateMetrologyEvaluationArtifact,
    *,
    base_dir: Path,
) -> SolidStateMetrologyEvidenceIntegrity:
    """Verify declared physical sources and cross-record relationships.

    Relative source paths are resolved against ``base_dir``.  The function is
    intentionally separate from the numerical evaluator so a caller can keep
    the evidence check reproducible and visible in the saved artifact.
    """

    issues: list[str] = []
    source_checks: list[SolidStateMetrologyEvidenceSourceCheck] = []

    session_ids = [session.id for session in artifact.sessions]
    if len(session_ids) != len(set(session_ids)):
        issues.append("duplicate_session_id")
    estimate_ids = [estimate.id for estimate in artifact.estimates]
    if len(estimate_ids) != len(set(estimate_ids)):
        issues.append("duplicate_estimate_id")

    _append_reference_source_checks(
        reference=artifact.reference,
        role="reference",
        owner_id="reference",
        base_dir=base_dir,
        issues=issues,
        source_checks=source_checks,
    )

    known_session_ids = set(session_ids)
    for session in artifact.sessions:
        if session.reference is not None:
            _append_reference_source_checks(
                reference=session.reference,
                role="session_reference",
                owner_id=session.id,
                base_dir=base_dir,
                issues=issues,
                source_checks=source_checks,
            )
        elif (
            session.status == "usable"
            and artifact.protocol.require_per_session_reference
        ):
            _append_unique(issues, "session_reference_missing")
        if session.capture_path is not None:
            source_checks.append(
                _source_check(
                    role="session_capture",
                    owner_id=session.id,
                    path=session.capture_path,
                    declared_sha256=session.capture_sha256,
                    base_dir=base_dir,
                )
            )
            if session.capture_sha256 is None:
                _append_unique(issues, "session_capture_digest_missing")
        elif session.capture_sha256 is not None:
            _append_unique(issues, "session_capture_path_missing")
        elif session.status == "usable":
            _append_unique(issues, "session_capture_path_missing")
            _append_unique(issues, "session_capture_digest_missing")

    for estimate in artifact.estimates:
        if estimate.session_id not in known_session_ids:
            _append_unique(issues, "estimate_session_link_missing")
        if estimate.source_path is not None:
            source_checks.append(
                _source_check(
                    role="estimate",
                    owner_id=estimate.id,
                    path=estimate.source_path,
                    declared_sha256=estimate.source_sha256,
                    base_dir=base_dir,
                )
            )
            if estimate.source_sha256 is None:
                _append_unique(issues, "estimate_source_digest_missing")
        elif estimate.source_sha256 is not None:
            _append_unique(issues, "estimate_source_path_missing")
        else:
            _append_unique(issues, "estimate_source_path_missing")
            _append_unique(issues, "estimate_source_digest_missing")

    for source_check in source_checks:
        if source_check.status == "missing":
            _append_unique(issues, "evidence_source_missing")
        elif source_check.status == "missing_digest":
            _append_unique(issues, "evidence_source_digest_missing")
        elif source_check.status == "mismatch":
            _append_unique(issues, "evidence_source_digest_mismatch")

    passed = bool(source_checks) and not issues and all(
        source_check.status == "verified" for source_check in source_checks
    )
    return SolidStateMetrologyEvidenceIntegrity(
        checked=True,
        passed=passed,
        source_checks=source_checks,
        issues=issues,
    )


def _evidence_integrity_reasons(
    artifact: SolidStateMetrologyEvaluationArtifact,
) -> list[str]:
    """Turn a persisted integrity report into numerical-gate reasons."""

    integrity = artifact.evidence_integrity
    if not integrity.checked:
        return ["evidence_integrity_not_checked"] if artifact.estimates else []
    reasons: list[str] = []
    if not integrity.passed:
        reasons.append("evidence_integrity_failed")
    if not integrity.source_checks:
        reasons.append("evidence_integrity_source_checks_missing")
    if any(check.status != "verified" for check in integrity.source_checks):
        reasons.append("evidence_source_integrity_failed")
    for issue in integrity.issues:
        _append_unique(reasons, issue)
    return reasons


def _reference_reasons_for(
    reference: SolidStateMetrologyReference,
    protocol: SolidStateMetrologyEvaluationProtocol,
    thresholds: SolidStateMetrologyEvaluationThresholds,
    *,
    prefix: str = "",
) -> list[str]:
    reasons: list[str] = []

    def add(reason: str) -> None:
        reasons.append(f"{prefix}{reason}")

    if not reference.independent_of_solver:
        add("reference_not_declared_independent_of_solver")
    if reference.extrinsic_method != protocol.required_extrinsic_method:
        add("extrinsic_reference_method_does_not_match_protocol")
    if reference.clock_method != protocol.required_clock_method:
        add("clock_reference_method_does_not_match_protocol")
    if reference.transform is None:
        add("missing_reference_transform")
    elif reference.transform.provenance.evidence_level != "independently_measured":
        add("reference_transform_provenance_is_not_independent")
    if reference.time_offset_sec is None:
        add("missing_reference_time_offset")
    if not reference.source_paths or not reference.source_sha256:
        add("missing_reference_source_provenance")
    if reference.rotation_uncertainty_deg is None:
        add("missing_reference_rotation_uncertainty")
    elif (
        reference.rotation_uncertainty_deg
        > thresholds.max_reference_rotation_uncertainty_deg
    ):
        add("reference_rotation_uncertainty_exceeds_threshold")
    if reference.translation_uncertainty_m is None:
        add("missing_reference_translation_uncertainty")
    elif (
        reference.translation_uncertainty_m
        > thresholds.max_reference_translation_uncertainty_m
    ):
        add("reference_translation_uncertainty_exceeds_threshold")
    if reference.time_uncertainty_sec is None:
        add("missing_reference_time_uncertainty")
    elif reference.time_uncertainty_sec > thresholds.max_reference_time_uncertainty_sec:
        add("reference_time_uncertainty_exceeds_threshold")
    return reasons


def _reference_reasons(
    artifact: SolidStateMetrologyEvaluationArtifact,
) -> list[str]:
    """Validate the packet-level reference declaration."""

    return _reference_reasons_for(
        artifact.reference,
        artifact.protocol,
        artifact.thresholds,
    )


def _downstream_metric_state(
    artifact: SolidStateMetrologyEvaluationArtifact,
) -> tuple[bool | None, list[str]]:
    if not artifact.protocol.require_downstream_metric:
        return True, []
    metric = artifact.downstream_metric
    if metric is None:
        return None, ["missing_downstream_metric"]
    reasons: list[str] = []
    if metric.value is None:
        reasons.append("missing_downstream_metric_value")
    if not metric.held_out:
        reasons.append("downstream_metric_is_not_held_out")
    if not metric.independent_of_solver:
        reasons.append("downstream_metric_is_not_independent_of_solver")
    if metric.max_value is None:
        reasons.append("missing_downstream_metric_threshold")
    if reasons:
        return None, reasons
    assert metric.value is not None
    assert metric.max_value is not None
    passed = (
        metric.value <= metric.max_value
        if metric.lower_is_better
        else metric.value >= metric.max_value
    )
    return passed, [] if passed else ["downstream_metric_exceeds_threshold"]


def evaluate_solid_state_metrology(
    artifact: SolidStateMetrologyEvaluationArtifact,
) -> SolidStateMetrologyEvaluationArtifact:
    """Evaluate independent reference errors and physical evidence gates."""

    thresholds = artifact.thresholds
    session_by_id = {session.id: session for session in artifact.sessions}
    usable_sessions = [
        session for session in artifact.sessions if session.status == "usable"
    ]
    remount_ids = {session.remount_id for session in usable_sessions}
    reasons = _reference_reasons(artifact)
    if artifact.protocol.require_per_session_reference:
        for session in usable_sessions:
            if session.reference is None:
                _append_unique(reasons, "missing_session_reference")
            else:
                for reason in _reference_reasons_for(
                    session.reference,
                    artifact.protocol,
                    thresholds,
                    prefix="session_",
                ):
                    _append_unique(reasons, reason)
    for reason in _evidence_integrity_reasons(artifact):
        _append_unique(reasons, reason)
    if len(usable_sessions) < artifact.protocol.minimum_usable_sessions:
        _append_unique(reasons, "insufficient_usable_sessions")
    if len(remount_ids) < artifact.protocol.minimum_remounts:
        _append_unique(reasons, "insufficient_remounts")
    downstream_passed, downstream_reasons = _downstream_metric_state(artifact)
    for reason in downstream_reasons:
        _append_unique(reasons, reason)

    run_metrics: list[SolidStateMetrologyRunMetric] = []
    for estimate in artifact.estimates:
        estimate_session = session_by_id.get(estimate.session_id)
        run_reasons: list[str] = []
        session_reference: SolidStateMetrologyReference | None = None
        if estimate_session is None:
            run_reasons.append("estimate_references_unknown_session")
        elif estimate_session.status != "usable":
            run_reasons.append("estimate_references_rejected_session")
        elif (
            estimate_session.reference is None
            and artifact.protocol.require_per_session_reference
        ):
            run_reasons.append("missing_session_reference")
        else:
            session_reference = estimate_session.reference or artifact.reference
        if session_reference is None:
            run_reasons.append("missing_reference_transform")
            run_reasons.append("missing_reference_time_offset")
        elif session_reference.transform is None:
            run_reasons.append("missing_reference_transform")
        if session_reference is not None and session_reference.time_offset_sec is None:
            run_reasons.append("missing_reference_time_offset")
        if estimate.transform is None:
            run_reasons.append("missing_estimate_transform")
        if estimate.time_offset_sec is None:
            run_reasons.append("missing_estimate_time_offset")

        rotation_error: float | None = None
        translation_error: float | None = None
        time_error: float | None = None
        if (
            session_reference is not None
            and session_reference.transform is not None
            and estimate.transform is not None
        ):
            rotation_error = _rotation_error_deg(
                session_reference.transform.as_se3(), estimate.transform.as_se3()
            )
            translation_error = _translation_error_m(
                session_reference.transform.as_se3(), estimate.transform.as_se3()
            )
            if rotation_error > thresholds.max_rotation_error_deg:
                run_reasons.append("rotation_error_exceeds_threshold")
            if translation_error > thresholds.max_translation_error_m:
                run_reasons.append("translation_error_exceeds_threshold")
        if (
            session_reference is not None
            and session_reference.time_offset_sec is not None
            and estimate.time_offset_sec is not None
        ):
            time_error = abs(
                estimate.time_offset_sec - session_reference.time_offset_sec
            )
            if time_error > thresholds.max_time_offset_error_sec:
                run_reasons.append("time_offset_error_exceeds_threshold")
        run_metrics.append(
            SolidStateMetrologyRunMetric(
                estimate_id=estimate.id,
                session_id=estimate.session_id,
                remount_id=(
                    estimate_session.remount_id
                    if estimate_session is not None
                    else None
                ),
                rotation_error_deg=rotation_error,
                translation_error_m=translation_error,
                time_offset_error_sec=time_error,
                gate_passed=not run_reasons,
                reasons=run_reasons,
            )
        )

    evaluated = [
        metric
        for metric in run_metrics
        if metric.rotation_error_deg is not None
        and metric.translation_error_m is not None
        and metric.time_offset_error_sec is not None
    ]
    for metric in run_metrics:
        for reason in metric.reasons:
            _append_unique(reasons, reason)
    if not artifact.estimates:
        _append_unique(reasons, "missing_estimates")
        status: MetrologyStatus = "planned"
        decision: MetrologyDecision = "collect"
    elif len(evaluated) != len(run_metrics):
        _append_unique(reasons, "incomplete_estimate_evaluation")
        status = "inconclusive"
        decision = "review"
    elif reasons:
        has_only_accuracy_failures = all(
            reason
            in {
                "rotation_error_exceeds_threshold",
                "translation_error_exceeds_threshold",
                "time_offset_error_exceeds_threshold",
                "downstream_metric_exceeds_threshold",
            }
            for reason in reasons
        )
        status = "fail" if has_only_accuracy_failures else "inconclusive"
        decision = "fail" if status == "fail" else "review"
    else:
        status = "pass" if all(metric.gate_passed for metric in run_metrics) else "fail"
        decision = status

    rotation_errors = [
        metric.rotation_error_deg
        for metric in evaluated
        if metric.rotation_error_deg is not None
    ]
    translation_errors = [
        metric.translation_error_m
        for metric in evaluated
        if metric.translation_error_m is not None
    ]
    time_errors = [
        metric.time_offset_error_sec
        for metric in evaluated
        if metric.time_offset_error_sec is not None
    ]
    metrics = SolidStateMetrologyEvaluationMetrics(
        run_count=len(run_metrics),
        evaluated_run_count=len(evaluated),
        passing_run_count=sum(metric.gate_passed for metric in evaluated),
        usable_session_count=len(usable_sessions),
        remount_count=len(remount_ids),
        max_rotation_error_deg=max(rotation_errors) if rotation_errors else None,
        max_translation_error_m=(
            max(translation_errors) if translation_errors else None
        ),
        max_time_offset_error_sec=max(time_errors) if time_errors else None,
        downstream_metric_passed=downstream_passed,
        run_metrics=run_metrics,
    )
    evidence_integrity = artifact.evidence_integrity
    if evidence_integrity.checked:
        evidence_integrity = evidence_integrity.model_copy(
            update={
                "passed": bool(evidence_integrity.source_checks)
                and not evidence_integrity.issues
                and all(
                    check.status == "verified"
                    for check in evidence_integrity.source_checks
                )
            }
        )
    return artifact.model_copy(
        update={
            "metrics": metrics,
            "evidence_integrity": evidence_integrity,
            "status": status,
            "decision": decision,
            "reasons": reasons,
        }
    )


def solid_state_metrology_evaluation_json_schema() -> dict[str, Any]:
    """Return the JSON schema for physical ground-truth evaluation artifacts."""

    return SolidStateMetrologyEvaluationArtifact.model_json_schema()
