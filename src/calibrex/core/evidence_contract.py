"""Versioned evidence protocol and policy artifacts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from calibrex.core.assessment import AssessmentArtifact, AssessmentPolicy
from calibrex.core.report_artifacts import EvidenceProtocolItem, ReportRunInfo
from calibrex.core.result import StrictModel

PROTOCOL_SCHEMA_VERSION: Literal["calibrex.protocol/v0.1"] = "calibrex.protocol/v0.1"
POLICY_SCHEMA_VERSION: Literal["calibrex.policy/v0.1"] = "calibrex.policy/v0.1"

PolicyGateDisposition = Literal["pass", "fail", "inconclusive", "not_applicable"]
PolicyScalar = str | float | int | bool | None

_STATUS_ON_FAILURE_BY_RULE: dict[str, PolicyGateDisposition] = {
    "raw_recomputation": "inconclusive",
    "protocol_declared": "fail",
    "holdout_support_gate": "fail",
    "holdout_independence": "inconclusive",
    "known_bad_controls": "fail",
    "decision_boundary_summary": "fail",
}


class ProtocolArtifact(StrictModel):
    """Machine-readable evidence protocol declarations for one run."""

    schema_version: Literal["calibrex.protocol/v0.1"] = PROTOCOL_SCHEMA_VERSION
    run: ReportRunInfo
    protocols: list[EvidenceProtocolItem] = Field(default_factory=list)


class PolicyGateDefinition(StrictModel):
    """One declared assessment gate in a policy artifact."""

    rule_id: str
    status_on_pass: PolicyGateDisposition = "pass"
    status_on_failure: PolicyGateDisposition
    status_on_missing_evidence: PolicyGateDisposition = "inconclusive"
    metric_ids: list[str] = Field(default_factory=list)
    thresholds: dict[str, PolicyScalar] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)


class PolicyArtifact(StrictModel):
    """Machine-readable assessment policy declarations for one run."""

    schema_version: Literal["calibrex.policy/v0.1"] = POLICY_SCHEMA_VERSION
    run: ReportRunInfo
    policy: AssessmentPolicy = Field(default_factory=AssessmentPolicy)
    gates: list[PolicyGateDefinition] = Field(default_factory=list)


def protocol_json_schema() -> dict[str, Any]:
    """Return the JSON schema for protocol artifacts."""

    return ProtocolArtifact.model_json_schema()


def policy_json_schema() -> dict[str, Any]:
    """Return the JSON schema for policy artifacts."""

    return PolicyArtifact.model_json_schema()


def policy_artifact_from_assessment(
    assessment: AssessmentArtifact,
    *,
    run: ReportRunInfo,
) -> PolicyArtifact:
    """Build a policy artifact from an applied assessment."""

    return PolicyArtifact(
        run=run,
        policy=assessment.policy,
        gates=[
            PolicyGateDefinition(
                rule_id=rule.rule_id,
                status_on_failure=_STATUS_ON_FAILURE_BY_RULE.get(
                    rule.rule_id,
                    "inconclusive",
                ),
                metric_ids=list(rule.metric_ids),
                thresholds=dict(rule.thresholds),
                evidence_refs=list(rule.evidence_refs),
            )
            for rule in assessment.rules
        ],
    )
