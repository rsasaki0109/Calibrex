from calibrex.core.assessment import assess_report_evidence
from calibrex.core.report_artifacts import (
    EvidenceMaterializationInfo,
    ReportEvidenceArtifact,
    ReportRunInfo,
)


def test_raw_recomputation_requires_verified_raw_inputs() -> None:
    assert _raw_recomputation_status(data_verified=True) == "pass"
    assert _raw_recomputation_status(data_verified=False) == "inconclusive"
    assert _raw_recomputation_status(data_verified=None) == "inconclusive"


def _raw_recomputation_status(*, data_verified: bool | None) -> str:
    evidence = ReportEvidenceArtifact(
        run=ReportRunInfo(
            id="assessment-unit",
            status="success",
            domain="robotics",
            calibrex_version="0.1.0",
            created_at="2026-06-19T00:00:00Z",
        ),
        materialization=EvidenceMaterializationInfo(
            metrics_origin="recomputed",
            data_verified=data_verified,
        ),
    )
    assessment = assess_report_evidence(evidence)
    for rule in assessment.rules:
        if rule.rule_id == "raw_recomputation":
            return rule.status
    raise AssertionError("raw_recomputation rule was not emitted")
