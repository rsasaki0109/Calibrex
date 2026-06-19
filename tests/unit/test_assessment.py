from calibrex.core.assessment import assess_report_evidence
from calibrex.core.report_artifacts import (
    EvidenceInputFileItem,
    EvidenceMaterializationInfo,
    ReportEvidenceArtifact,
    ReportRunInfo,
)


def test_raw_recomputation_requires_verified_raw_inputs() -> None:
    assert _raw_recomputation_status(data_verified=True, input_files=True) == "pass"
    assert _raw_recomputation_status(data_verified=True, input_files=False) == "inconclusive"
    assert _raw_recomputation_status(data_verified=False, input_files=True) == "inconclusive"
    assert _raw_recomputation_status(data_verified=None, input_files=True) == "inconclusive"


def _raw_recomputation_status(*, data_verified: bool | None, input_files: bool) -> str:
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
        input_files=[
            EvidenceInputFileItem(
                path="raw.pcd",
                sha256="0" * 64,
                size_bytes=128,
            )
        ]
        if input_files
        else [],
    )
    assessment = assess_report_evidence(evidence)
    for rule in assessment.rules:
        if rule.rule_id == "raw_recomputation":
            return rule.status
    raise AssertionError("raw_recomputation rule was not emitted")
