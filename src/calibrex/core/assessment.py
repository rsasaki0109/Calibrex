"""Falsification assessment artifacts for calibration evidence."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.report_artifacts import (
    EvidenceCaseItem,
    EvidenceProtocolItem,
    EvidenceSummaryItem,
    ReportEvidenceArtifact,
)
from calibrex.core.result import StrictModel

ASSESSMENT_SCHEMA_VERSION: Literal["calibrex.assessment/v0.1"] = (
    "calibrex.assessment/v0.1"
)
FALSIFICATION_POLICY_ID = "calibrex.falsification.lidar_pair/v0.1"

AssessmentStatus = Literal["pass", "fail", "inconclusive"]
AssessmentScalar = str | float | int | bool | None

_MIN_MAP_VOXEL_COUNT = 20
_MIN_MATCHED_POINT_COUNT = 500
_MIN_ELIGIBLE_POINT_COUNT = 500
_MIN_ACCEPTED_CORRESPONDENCE_COUNT = 500
_MIN_SUPPORT_RATIO = 0.10
_MAX_UNMATCHED_FRACTION = 0.90
_MIN_KNOWN_BAD_CASE_COUNT = 1
_MIN_KNOWN_BAD_PASS_FRACTION = 0.50
_MIN_SUPPORTED_KNOWN_BAD_PASS_FRACTION = 0.50
_MIN_MANDATORY_KNOWN_BAD_CASE_COUNT = 12
_MIN_MANDATORY_SUPPORTED_DETECTION_COUNT = 8
_DELTA_EPSILON = 1.0e-9


def _default_policy_parameters() -> dict[str, AssessmentScalar]:
    return {
        "min_map_voxel_count": _MIN_MAP_VOXEL_COUNT,
        "min_matched_point_count": _MIN_MATCHED_POINT_COUNT,
        "min_eligible_point_count": _MIN_ELIGIBLE_POINT_COUNT,
        "min_accepted_correspondence_count": _MIN_ACCEPTED_CORRESPONDENCE_COUNT,
        "min_support_ratio": _MIN_SUPPORT_RATIO,
        "max_unmatched_fraction": _MAX_UNMATCHED_FRACTION,
        "min_known_bad_case_count": _MIN_KNOWN_BAD_CASE_COUNT,
        "min_known_bad_pass_fraction": _MIN_KNOWN_BAD_PASS_FRACTION,
        "min_supported_known_bad_pass_fraction": (
            _MIN_SUPPORTED_KNOWN_BAD_PASS_FRACTION
        ),
        "min_mandatory_known_bad_case_count": _MIN_MANDATORY_KNOWN_BAD_CASE_COUNT,
        "min_mandatory_supported_detection_count": (
            _MIN_MANDATORY_SUPPORTED_DETECTION_COUNT
        ),
    }


class AssessmentSourceEvidence(StrictModel):
    """Evidence artifact assessed by a policy."""

    path: str | None = None
    sha256: str | None = None
    schema_version: str
    run_id: str
    metrics_origin: str
    data_verified: bool | None = None


class AssessmentPolicy(StrictModel):
    """Versioned policy metadata used to assess an evidence artifact."""

    policy_id: str = FALSIFICATION_POLICY_ID
    status_semantics: str = (
        "pass means all required gates passed; fail means at least one gate "
        "rejected the candidate; inconclusive means the evidence cannot support "
        "or reject the candidate under this policy"
    )
    parameters: dict[str, AssessmentScalar] = Field(
        default_factory=_default_policy_parameters
    )


class AssessmentRuleResult(StrictModel):
    """One policy gate applied to an evidence artifact."""

    rule_id: str
    status: AssessmentStatus
    reason: str
    metric_ids: list[str] = Field(default_factory=list)
    observed: dict[str, AssessmentScalar] = Field(default_factory=dict)
    thresholds: dict[str, AssessmentScalar] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)


class AssessmentArtifact(StrictModel):
    """Machine-readable falsification assessment."""

    schema_version: Literal["calibrex.assessment/v0.1"] = ASSESSMENT_SCHEMA_VERSION
    source_evidence: AssessmentSourceEvidence
    policy: AssessmentPolicy = Field(default_factory=AssessmentPolicy)
    status: AssessmentStatus
    reason: str
    rules: list[AssessmentRuleResult] = Field(default_factory=list)


def assessment_json_schema() -> dict[str, Any]:
    """Return the JSON schema for assessment artifacts."""

    return AssessmentArtifact.model_json_schema()


def assess_evidence_file(
    path: str | Path,
    *,
    policy: AssessmentPolicy | None = None,
) -> AssessmentArtifact:
    """Assess a report evidence artifact from disk."""

    evidence_path = Path(path)
    payload = read_mapping(evidence_path)
    evidence = ReportEvidenceArtifact.model_validate(payload)
    return assess_report_evidence(
        evidence,
        source_path=evidence_path,
        source_sha256=_sha256_file(evidence_path),
        policy=policy,
    )


def write_assessment_from_evidence(
    evidence_path: str | Path,
    assessment_path: str | Path,
    *,
    policy: AssessmentPolicy | None = None,
) -> AssessmentArtifact:
    """Assess evidence and write an immutable assessment artifact."""

    evidence_file = Path(evidence_path)
    assessment_file = Path(assessment_path)
    evidence = ReportEvidenceArtifact.model_validate(read_mapping(evidence_file))
    assessment = assess_report_evidence(
        evidence,
        source_path=_portable_path(assessment_file.parent, evidence_file),
        source_sha256=_sha256_file(evidence_file),
        policy=policy,
    )
    write_mapping(assessment_file, assessment.model_dump(mode="json", exclude_none=True))
    return assessment


def assess_report_evidence(
    evidence: ReportEvidenceArtifact,
    *,
    source_path: str | Path | None = None,
    source_sha256: str | None = None,
    policy: AssessmentPolicy | None = None,
) -> AssessmentArtifact:
    """Apply the built-in falsification policy to report evidence."""

    active_policy = policy or AssessmentPolicy()
    rules = [
        _materialization_rule(evidence),
        _protocol_declaration_rule(evidence, active_policy),
        _holdout_support_rule(evidence, active_policy),
        _holdout_independence_rule(evidence),
        _known_bad_rule(evidence, active_policy),
        _summary_decision_rule(evidence),
    ]
    status = _overall_status(rules)
    reason = _overall_reason(status, rules)
    return AssessmentArtifact(
        source_evidence=AssessmentSourceEvidence(
            path=str(source_path) if source_path is not None else None,
            sha256=source_sha256,
            schema_version=evidence.schema_version,
            run_id=evidence.run.id,
            metrics_origin=evidence.materialization.metrics_origin,
            data_verified=evidence.materialization.data_verified,
        ),
        policy=active_policy,
        status=status,
        reason=reason,
        rules=rules,
    )


def _materialization_rule(evidence: ReportEvidenceArtifact) -> AssessmentRuleResult:
    origin = evidence.materialization.metrics_origin
    data_verified = evidence.materialization.data_verified
    input_file_count = len(evidence.input_files)
    input_digest_count = sum(
        1
        for input_file in evidence.input_files
        if input_file.sha256 and input_file.size_bytes is not None
    )
    observed: dict[str, AssessmentScalar] = {
        "metrics_origin": origin,
        "data_verified": data_verified,
        "input_file_count": input_file_count,
        "input_digest_count": input_digest_count,
    }
    if origin == "recomputed" and data_verified is True and input_digest_count > 0:
        return AssessmentRuleResult(
            rule_id="raw_recomputation",
            status="pass",
            reason="evidence was materialized from recomputed metrics with verified raw inputs",
            observed=observed,
        )
    return AssessmentRuleResult(
        rule_id="raw_recomputation",
        status="inconclusive",
        reason="raw observations were not verified as recomputed for this evidence",
        observed=observed,
        evidence_refs=["materialization", "input_files"],
    )


def _protocol_declaration_rule(
    evidence: ReportEvidenceArtifact,
    policy: AssessmentPolicy,
) -> AssessmentRuleResult:
    protocol = _lidar_pair_protocol(evidence)
    if protocol is None:
        return AssessmentRuleResult(
            rule_id="protocol_declared",
            status="inconclusive",
            reason="no lidar_pair evidence protocol is declared",
            evidence_refs=["protocols"],
        )
    observed: dict[str, AssessmentScalar] = {
        "protocol_id": protocol.protocol_id,
        "candidate_transform": protocol.candidate_transform,
        "transform_convention": protocol.transform_convention,
        "known_bad_case_count": protocol.known_bad_case_count,
    }
    if not protocol.candidate_transform or not protocol.transform_convention:
        return AssessmentRuleResult(
            rule_id="protocol_declared",
            status="inconclusive",
            reason="candidate transform or transform convention is missing",
            observed=observed,
            evidence_refs=[protocol.protocol_id],
        )
    min_known_bad_case_count = _policy_number(
        policy,
        "min_known_bad_case_count",
        _MIN_KNOWN_BAD_CASE_COUNT,
    )
    case_count = protocol.known_bad_case_count
    if case_count is None or case_count < min_known_bad_case_count:
        return AssessmentRuleResult(
            rule_id="protocol_declared",
            status="fail",
            reason="declared known-bad perturbation case count is below policy minimum",
            observed=observed,
            thresholds={"min_known_bad_case_count": min_known_bad_case_count},
            evidence_refs=[protocol.protocol_id],
        )
    return AssessmentRuleResult(
        rule_id="protocol_declared",
        status="pass",
        reason="lidar_pair evidence protocol declares transform convention and controls",
        observed=observed,
        thresholds={"min_known_bad_case_count": min_known_bad_case_count},
        evidence_refs=[protocol.protocol_id],
    )


def _holdout_support_rule(
    evidence: ReportEvidenceArtifact,
    policy: AssessmentPolicy,
) -> AssessmentRuleResult:
    protocol = _lidar_pair_protocol(evidence)
    if protocol is None:
        return AssessmentRuleResult(
            rule_id="holdout_support_gate",
            status="inconclusive",
            reason="holdout support cannot be assessed without a lidar_pair protocol",
            evidence_refs=["protocols"],
        )
    map_voxels = _number(protocol.parameters.get("map_voxel_count"))
    matched_points = _number(protocol.parameters.get("matched_point_count"))
    eligible_points = _number(protocol.parameters.get("eligible_point_count"))
    accepted_correspondences = _number(
        protocol.parameters.get("accepted_correspondence_count")
    )
    support_ratio = _number(protocol.parameters.get("support_ratio"))
    unmatched_fraction = _number(protocol.parameters.get("unmatched_fraction"))
    observed: dict[str, AssessmentScalar] = {
        "map_voxel_count": map_voxels,
        "matched_point_count": matched_points,
        "eligible_point_count": eligible_points,
        "accepted_correspondence_count": accepted_correspondences,
        "support_ratio": support_ratio,
        "unmatched_fraction": unmatched_fraction,
    }
    min_map_voxel_count = _policy_number(
        policy,
        "min_map_voxel_count",
        _MIN_MAP_VOXEL_COUNT,
    )
    min_matched_point_count = _policy_number(
        policy,
        "min_matched_point_count",
        _MIN_MATCHED_POINT_COUNT,
    )
    min_eligible_point_count = _policy_number(
        policy,
        "min_eligible_point_count",
        _MIN_ELIGIBLE_POINT_COUNT,
    )
    min_accepted_correspondence_count = _policy_number(
        policy,
        "min_accepted_correspondence_count",
        _MIN_ACCEPTED_CORRESPONDENCE_COUNT,
    )
    min_support_ratio = _policy_number(policy, "min_support_ratio", _MIN_SUPPORT_RATIO)
    max_unmatched_fraction = _policy_number(
        policy,
        "max_unmatched_fraction",
        _MAX_UNMATCHED_FRACTION,
    )
    thresholds: dict[str, AssessmentScalar] = {
        "min_map_voxel_count": min_map_voxel_count,
        "min_matched_point_count": min_matched_point_count,
        "min_eligible_point_count": min_eligible_point_count,
        "min_accepted_correspondence_count": min_accepted_correspondence_count,
        "min_support_ratio": min_support_ratio,
        "max_unmatched_fraction": max_unmatched_fraction,
    }
    if (
        map_voxels is None
        or matched_points is None
        or eligible_points is None
        or accepted_correspondences is None
        or support_ratio is None
        or unmatched_fraction is None
    ):
        return AssessmentRuleResult(
            rule_id="holdout_support_gate",
            status="inconclusive",
            reason=(
                "holdout support parameters are incomplete; candidate-independent "
                "eligible population, accepted correspondences, and support ratio "
                "are required"
            ),
            observed=observed,
            thresholds=thresholds,
            evidence_refs=[protocol.protocol_id],
        )
    if (
        map_voxels < min_map_voxel_count
        or matched_points < min_matched_point_count
        or unmatched_fraction > max_unmatched_fraction
        or eligible_points < min_eligible_point_count
        or accepted_correspondences < min_accepted_correspondence_count
        or support_ratio < min_support_ratio
    ):
        return AssessmentRuleResult(
            rule_id="holdout_support_gate",
            status="fail",
            reason="holdout support is below the falsification policy gate",
            metric_ids=[
                "lidar_pair_holdout_point_to_plane_map_voxel_count",
                "lidar_pair_holdout_point_to_plane_matched_point_count",
                "lidar_pair_holdout_point_to_plane_eligible_point_count",
                "lidar_pair_holdout_point_to_plane_accepted_correspondence_count",
                "lidar_pair_holdout_point_to_plane_support_ratio",
                "lidar_pair_holdout_point_to_plane_unmatched_fraction",
            ],
            observed=observed,
            thresholds=thresholds,
            evidence_refs=[protocol.protocol_id],
        )
    return AssessmentRuleResult(
        rule_id="holdout_support_gate",
        status="pass",
        reason="holdout support satisfies voxel, match, fixed-denominator, and unmatched gates",
        metric_ids=[
            "lidar_pair_holdout_point_to_plane_map_voxel_count",
            "lidar_pair_holdout_point_to_plane_matched_point_count",
            "lidar_pair_holdout_point_to_plane_eligible_point_count",
            "lidar_pair_holdout_point_to_plane_accepted_correspondence_count",
            "lidar_pair_holdout_point_to_plane_support_ratio",
            "lidar_pair_holdout_point_to_plane_unmatched_fraction",
        ],
        observed=observed,
        thresholds=thresholds,
        evidence_refs=[protocol.protocol_id],
    )


def _holdout_independence_rule(evidence: ReportEvidenceArtifact) -> AssessmentRuleResult:
    protocol = _lidar_pair_protocol(evidence)
    if protocol is None:
        return AssessmentRuleResult(
            rule_id="holdout_independence",
            status="inconclusive",
            reason="holdout independence cannot be assessed without a lidar_pair protocol",
            evidence_refs=["protocols"],
        )
    if protocol.independent_holdout is True:
        return AssessmentRuleResult(
            rule_id="holdout_independence",
            status="pass",
            reason="protocol declares an independent holdout split",
            observed={"independent_holdout": True, "split_policy": protocol.split_policy},
            evidence_refs=[protocol.protocol_id],
        )
    return AssessmentRuleResult(
        rule_id="holdout_independence",
        status="inconclusive",
        reason="protocol does not provide an independent holdout split",
        observed={
            "independent_holdout": protocol.independent_holdout,
            "split_policy": protocol.split_policy,
        },
        evidence_refs=[protocol.protocol_id],
    )


def _known_bad_rule(
    evidence: ReportEvidenceArtifact,
    policy: AssessmentPolicy,
) -> AssessmentRuleResult:
    protocol = _lidar_pair_protocol(evidence)
    cases = _lidar_pair_known_bad_cases(evidence)
    summary = _summary(evidence, family="lidar_pair", check="Known-Bad Controls")
    case_count = protocol.known_bad_case_count if protocol else None
    pass_count = sum(1 for case in cases if case.status == "pass")
    pass_fraction = pass_count / len(cases) if cases else None
    support_scored_cases = [case for case in cases if _case_has_support_metrics(case)]
    supported_detection_count = sum(
        1
        for case in support_scored_cases
        if _case_is_supported_known_bad_detection(case, policy)
    )
    supported_pass_fraction = (
        supported_detection_count / len(support_scored_cases)
        if support_scored_cases
        else None
    )
    support_collapse_count = sum(
        1
        for case in support_scored_cases
        if not _case_has_sufficient_support(case, policy)
    )
    mandatory_case_count = _number_from_protocol_parameters(
        protocol,
        "mandatory_case_count",
    )
    mandatory_supported_detection_count = _number_from_protocol_parameters(
        protocol,
        "mandatory_supported_detection_count",
    )
    mandatory_support_collapse_count = _number_from_protocol_parameters(
        protocol,
        "mandatory_support_collapse_count",
    )
    challenge_id = _string_from_protocol_parameters(protocol, "challenge_id")
    min_known_bad_case_count = _policy_number(
        policy,
        "min_known_bad_case_count",
        _MIN_KNOWN_BAD_CASE_COUNT,
    )
    min_known_bad_pass_fraction = _policy_number(
        policy,
        "min_known_bad_pass_fraction",
        _MIN_KNOWN_BAD_PASS_FRACTION,
    )
    min_supported_known_bad_pass_fraction = _policy_number(
        policy,
        "min_supported_known_bad_pass_fraction",
        _MIN_SUPPORTED_KNOWN_BAD_PASS_FRACTION,
    )
    min_mandatory_known_bad_case_count = _policy_number(
        policy,
        "min_mandatory_known_bad_case_count",
        _MIN_MANDATORY_KNOWN_BAD_CASE_COUNT,
    )
    min_mandatory_supported_detection_count = _policy_number(
        policy,
        "min_mandatory_supported_detection_count",
        _MIN_MANDATORY_SUPPORTED_DETECTION_COUNT,
    )
    min_accepted_correspondence_count = _policy_number(
        policy,
        "min_accepted_correspondence_count",
        _MIN_ACCEPTED_CORRESPONDENCE_COUNT,
    )
    min_support_ratio = _policy_number(policy, "min_support_ratio", _MIN_SUPPORT_RATIO)
    observed: dict[str, AssessmentScalar] = {
        "declared_known_bad_case_count": case_count,
        "materialized_case_count": len(cases),
        "materialized_pass_fraction": pass_fraction,
        "support_scored_case_count": len(support_scored_cases),
        "supported_detection_count": supported_detection_count,
        "supported_pass_fraction": supported_pass_fraction,
        "support_collapse_count": support_collapse_count,
        "challenge_id": challenge_id,
        "mandatory_case_count": mandatory_case_count,
        "mandatory_supported_detection_count": mandatory_supported_detection_count,
        "mandatory_support_collapse_count": mandatory_support_collapse_count,
        "summary_status": summary.status if summary else None,
    }
    thresholds: dict[str, AssessmentScalar] = {
        "min_known_bad_case_count": min_known_bad_case_count,
        "min_materialized_pass_fraction": min_known_bad_pass_fraction,
        "min_supported_known_bad_pass_fraction": min_supported_known_bad_pass_fraction,
        "min_mandatory_known_bad_case_count": min_mandatory_known_bad_case_count,
        "min_mandatory_supported_detection_count": (
            min_mandatory_supported_detection_count
        ),
        "min_accepted_correspondence_count": min_accepted_correspondence_count,
        "min_support_ratio": min_support_ratio,
    }
    if protocol is None or case_count is None or case_count < min_known_bad_case_count:
        return AssessmentRuleResult(
            rule_id="known_bad_controls",
            status="inconclusive",
            reason="known-bad controls are not declared with enough detail",
            observed=observed,
            thresholds=thresholds,
            evidence_refs=["protocols", "cases"],
        )
    if not cases:
        return AssessmentRuleResult(
            rule_id="known_bad_controls",
            status="inconclusive",
            reason="known-bad controls are declared but no case evidence is materialized",
            observed=observed,
            thresholds=thresholds,
            evidence_refs=[protocol.protocol_id, "cases"],
        )
    if summary is not None and summary.status == "fail":
        return AssessmentRuleResult(
            rule_id="known_bad_controls",
            status="fail",
            reason="known-bad summary explicitly failed",
            metric_ids=summary.metric_ids,
            observed=observed,
            thresholds=thresholds,
            evidence_refs=[protocol.protocol_id, "Known-Bad Controls"],
        )
    if pass_fraction is not None and pass_fraction < min_known_bad_pass_fraction:
        return AssessmentRuleResult(
            rule_id="known_bad_controls",
            status="fail",
            reason="too few materialized known-bad controls were detected",
            metric_ids=summary.metric_ids if summary else [],
            observed=observed,
            thresholds=thresholds,
            evidence_refs=[protocol.protocol_id, "cases"],
        )
    if (
        supported_pass_fraction is not None
        and supported_pass_fraction < min_supported_known_bad_pass_fraction
    ):
        return AssessmentRuleResult(
            rule_id="known_bad_controls",
            status="fail",
            reason=(
                "known-bad controls did not separate under enough supported geometry; "
                "detections may be dominated by support collapse"
            ),
            metric_ids=summary.metric_ids if summary else [],
            observed=observed,
            thresholds=thresholds,
            evidence_refs=[protocol.protocol_id, "cases"],
        )
    if challenge_id is not None:
        if (
            mandatory_case_count is None
            or mandatory_case_count < min_mandatory_known_bad_case_count
        ):
            return AssessmentRuleResult(
                rule_id="known_bad_controls",
                status="inconclusive",
                reason="mandatory known-bad challenge is incomplete",
                metric_ids=summary.metric_ids if summary else [],
                observed=observed,
                thresholds=thresholds,
                evidence_refs=[protocol.protocol_id, "cases"],
            )
        if (
            mandatory_supported_detection_count is None
            or mandatory_supported_detection_count
            < min_mandatory_supported_detection_count
        ):
            return AssessmentRuleResult(
                rule_id="known_bad_controls",
                status="fail",
                reason=(
                    "mandatory known-bad controls were not rejected under enough "
                    "supported geometry"
                ),
                metric_ids=summary.metric_ids if summary else [],
                observed=observed,
                thresholds=thresholds,
                evidence_refs=[protocol.protocol_id, "cases"],
            )
    if summary is not None and summary.status == "warn":
        return AssessmentRuleResult(
            rule_id="known_bad_controls",
            status="inconclusive",
            reason="known-bad controls are only weakly separated from the candidate",
            metric_ids=summary.metric_ids,
            observed=observed,
            thresholds=thresholds,
            evidence_refs=[protocol.protocol_id, "Known-Bad Controls"],
        )
    return AssessmentRuleResult(
        rule_id="known_bad_controls",
        status="pass",
        reason="predeclared known-bad controls are distinguishable from the candidate",
        metric_ids=summary.metric_ids if summary else [],
        observed=observed,
        thresholds=thresholds,
        evidence_refs=[protocol.protocol_id, "Known-Bad Controls"],
    )


def _summary_decision_rule(evidence: ReportEvidenceArtifact) -> AssessmentRuleResult:
    summary = _summary(evidence, family="lidar_pair", check="Decision Boundary")
    if summary is None:
        return AssessmentRuleResult(
            rule_id="decision_boundary_summary",
            status="inconclusive",
            reason="no lidar_pair decision boundary summary is available",
            evidence_refs=["summaries"],
        )
    if summary.status == "fail":
        status: AssessmentStatus = "fail"
        reason = "evidence summary rejected the candidate"
    elif summary.status == "pass":
        status = "pass"
        reason = "evidence summary supports the candidate under its protocol"
    else:
        status = "inconclusive"
        reason = "evidence summary is not strong enough to support or reject the candidate"
    return AssessmentRuleResult(
        rule_id="decision_boundary_summary",
        status=status,
        reason=reason,
        metric_ids=summary.metric_ids,
        observed={"summary_status": summary.status},
        evidence_refs=["Decision Boundary"],
    )


def _lidar_pair_protocol(evidence: ReportEvidenceArtifact) -> EvidenceProtocolItem | None:
    for protocol in evidence.protocols:
        if protocol.family == "lidar_pair":
            return protocol
    return None


def _summary(
    evidence: ReportEvidenceArtifact,
    *,
    family: str,
    check: str,
) -> EvidenceSummaryItem | None:
    for summary in evidence.summaries:
        if summary.family == family and summary.check == check:
            return summary
    return None


def _lidar_pair_known_bad_cases(evidence: ReportEvidenceArtifact) -> list[EvidenceCaseItem]:
    return [
        case
        for case in evidence.cases
        if case.family == "lidar_pair" and case.check == "Known-Bad Controls"
    ]


def _case_has_support_metrics(case: EvidenceCaseItem) -> bool:
    return (
        "lidar_pair_holdout_point_to_plane_support_ratio" in case.metric_values
        or "lidar_pair_holdout_point_to_plane_accepted_correspondence_count"
        in case.metric_values
    )


def _case_has_sufficient_support(
    case: EvidenceCaseItem,
    policy: AssessmentPolicy,
) -> bool:
    support_ratio = _number(
        case.metric_values.get("lidar_pair_holdout_point_to_plane_support_ratio")
    )
    accepted_count = _number(
        case.metric_values.get(
            "lidar_pair_holdout_point_to_plane_accepted_correspondence_count"
        )
    )
    if support_ratio is None or accepted_count is None:
        return False
    min_support_ratio = _policy_number(policy, "min_support_ratio", _MIN_SUPPORT_RATIO)
    min_accepted_correspondence_count = _policy_number(
        policy,
        "min_accepted_correspondence_count",
        _MIN_ACCEPTED_CORRESPONDENCE_COUNT,
    )
    return (
        support_ratio >= min_support_ratio
        and accepted_count >= min_accepted_correspondence_count
    )


def _case_is_supported_known_bad_detection(
    case: EvidenceCaseItem,
    policy: AssessmentPolicy,
) -> bool:
    if case.status != "pass" or not _case_has_sufficient_support(case, policy):
        return False
    delta_names = (
        "centroid_rmse_delta_m",
        "point_to_plane_p90_delta_m",
        "point_to_plane_rmse_delta_m",
    )
    return any(
        (delta := _number(case.delta_values.get(delta_name))) is not None
        and delta > _DELTA_EPSILON
        for delta_name in delta_names
    )


def _number_from_protocol_parameters(
    protocol: EvidenceProtocolItem | None,
    key: str,
) -> float | None:
    if protocol is None:
        return None
    return _number(protocol.parameters.get(key))


def _string_from_protocol_parameters(
    protocol: EvidenceProtocolItem | None,
    key: str,
) -> str | None:
    if protocol is None:
        return None
    value = protocol.parameters.get(key)
    return value if isinstance(value, str) else None


def _policy_number(
    policy: AssessmentPolicy,
    key: str,
    default: int | float,
) -> float:
    value = policy.parameters.get(key)
    if isinstance(value, bool):
        return float(default)
    if isinstance(value, int | float):
        return float(value)
    return float(default)


def _overall_status(rules: list[AssessmentRuleResult]) -> AssessmentStatus:
    if any(rule.status == "fail" for rule in rules):
        return "fail"
    if any(rule.status == "inconclusive" for rule in rules):
        return "inconclusive"
    return "pass"


def _overall_reason(status: AssessmentStatus, rules: list[AssessmentRuleResult]) -> str:
    counts = {
        "pass": sum(1 for rule in rules if rule.status == "pass"),
        "fail": sum(1 for rule in rules if rule.status == "fail"),
        "inconclusive": sum(1 for rule in rules if rule.status == "inconclusive"),
    }
    if status == "pass":
        return f"all {len(rules)} falsification policy gates passed"
    if status == "fail":
        return (
            f"{counts['fail']} falsification policy gate(s) failed; "
            f"{counts['inconclusive']} inconclusive"
        )
    return (
        f"{counts['inconclusive']} falsification policy gate(s) were inconclusive; "
        f"{counts['fail']} failed"
    )


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(base_dir: Path, path: Path) -> str:
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return str(path)
