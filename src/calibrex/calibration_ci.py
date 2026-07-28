"""Calibration CI orchestration and schema-versioned decision artifacts."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import Field

from calibrex.core.assessment import (
    AssessmentArtifact,
    AssessmentStatus,
    assess_evidence_file,
)
from calibrex.core.evidence_bundle import verify_evidence_bundle
from calibrex.core.evidence_contract import PolicyArtifact
from calibrex.core.io import read_mapping, write_mapping, write_text
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.result import StrictModel, load_result
from calibrex.evaluation.compare import (
    ProtocolCompatibilityStatus,
    compare_results,
)
from calibrex.visualization.comparison_table import write_comparison_table
from calibrex.visualization.evidence_card import write_evidence_card
from calibrex.visualization.report import write_evidence_contract_artifacts

CALIBRATION_CI_SCHEMA_VERSION: Literal["slac.calibration_ci/v0.1"] = (
    "slac.calibration_ci/v0.1"
)
CalibrationCICheckStatus = Literal["pass", "fail", "inconclusive", "skipped"]


class CalibrationCIInput(StrictModel):
    """One digest-bound result consumed by a CI decision."""

    role: Literal["candidate", "baseline"]
    path: str
    sha256: str
    schema_version: str
    run_id: str


class CalibrationCICheck(StrictModel):
    """One independently inspectable CI gate."""

    check_id: str
    status: CalibrationCICheckStatus
    reason: str
    enforced: bool = True
    artifact: str | None = None


class CalibrationCIPolicyInput(StrictModel):
    """Digest-bound policy used for the CI assessment."""

    path: str
    sha256: str
    schema_version: str
    policy_id: str


class CalibrationCIArtifacts(StrictModel):
    """Files materialized by a CI run."""

    evidence: str
    assessment: str
    protocol: str
    evidence_policy: str
    ci_policy: str
    transforms: str
    bundle: str
    verification: str
    evidence_card: str
    comparison: str | None = None
    comparison_table: str | None = None
    markdown_summary: str


class CalibrationCIProvenance(StrictModel):
    """Execution lineage for a CI decision."""

    producer: Literal["calibrex"] = "calibrex"
    calibrex_version: str
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    github_repository: str | None = None
    github_ref: str | None = None
    github_sha: str | None = None
    github_run_id: str | None = None
    github_action_repository: str | None = None
    github_action_ref: str | None = None


class CalibrationCIArtifact(StrictModel):
    """Portable decision emitted by ``calibrex ci`` and the GitHub Action."""

    schema_version: Literal["slac.calibration_ci/v0.1"] = CALIBRATION_CI_SCHEMA_VERSION
    status: AssessmentStatus
    candidate: CalibrationCIInput
    baseline: CalibrationCIInput | None = None
    policy: CalibrationCIPolicyInput
    assessment_status: AssessmentStatus
    protocol_compatibility: ProtocolCompatibilityStatus | None = None
    checks: list[CalibrationCICheck] = Field(default_factory=list)
    artifacts: CalibrationCIArtifacts
    provenance: CalibrationCIProvenance


def calibration_ci_json_schema() -> dict[str, object]:
    """Return the JSON schema for Calibration CI artifacts."""

    return CalibrationCIArtifact.model_json_schema()


def run_calibration_ci(
    candidate_path: str | Path,
    *,
    output_dir: str | Path,
    calibrex_version: str,
    command: list[str],
    baseline_path: str | Path | None = None,
    policy_path: str | Path | None = None,
    enforce_protocol: bool = True,
) -> CalibrationCIArtifact:
    """Validate, assess, compare, and render one calibration candidate."""

    candidate_file = Path(candidate_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    candidate = load_result(candidate_file)
    candidate_input = _ci_input("candidate", candidate_file, candidate.run.id)

    evidence_path = output_path / "evidence.json"
    contract_paths = write_evidence_contract_artifacts(
        candidate,
        evidence_path=evidence_path,
        output_dir=output_path,
    )
    checks = [
        CalibrationCICheck(
            check_id="candidate-schema",
            status="pass",
            reason=f"validated {candidate.schema_version}",
            artifact=str(candidate_file),
        )
    ]

    assessment_path = Path(contract_paths["assessment"])
    ci_policy_path = Path(contract_paths["policy"])
    policy_source_path = ci_policy_path
    if policy_path is not None:
        policy_source_path = Path(policy_path)
        policy_artifact = PolicyArtifact.model_validate(read_mapping(policy_source_path))
        ci_policy_path = output_path / "ci-policy.json"
        write_mapping(
            ci_policy_path,
            policy_artifact.model_dump(mode="json", exclude_none=True),
        )
        assessment = assess_evidence_file(
            evidence_path,
            policy=policy_artifact.policy,
        )
        assessment_path = output_path / "ci-assessment.json"
        write_mapping(
            assessment_path,
            assessment.model_dump(mode="json", exclude_none=True),
        )
    else:
        assessment = AssessmentArtifact.model_validate(read_mapping(assessment_path))
        policy_artifact = PolicyArtifact.model_validate(read_mapping(ci_policy_path))
    checks.append(
        CalibrationCICheck(
            check_id="falsification-assessment",
            status=assessment.status,
            reason=assessment.reason,
            artifact=str(assessment_path),
        )
    )

    verification = verify_evidence_bundle(contract_paths["bundle"])
    checks.append(
        CalibrationCICheck(
            check_id="bundle-integrity",
            status="pass" if verification.valid else "fail",
            reason=(
                "all evidence bundle digests and schema bindings verified"
                if verification.valid
                else "; ".join(verification.issues)
            ),
            artifact=contract_paths["verification"],
        )
    )

    evidence_card_path = output_path / "evidence-card.svg"
    write_evidence_card(candidate, evidence_card_path, source_path=candidate_file)

    baseline_input: CalibrationCIInput | None = None
    compatibility: ProtocolCompatibilityStatus | None = None
    comparison_path: Path | None = None
    comparison_table_path: Path | None = None
    if baseline_path is not None:
        baseline_file = Path(baseline_path)
        baseline = load_result(baseline_file)
        baseline_input = _ci_input("baseline", baseline_file, baseline.run.id)
        comparison = compare_results(
            baseline,
            candidate,
            left_path=baseline_file,
            right_path=candidate_file,
        )
        comparison_path = output_path / "comparison.json"
        write_mapping(
            comparison_path,
            comparison.model_dump(mode="json", exclude_none=True),
        )
        comparison_table_path = output_path / "comparison-table.svg"
        write_comparison_table(
            baseline,
            candidate,
            comparison_table_path,
            left_path=baseline_file,
            right_path=candidate_file,
            left_label="baseline",
            right_label="candidate",
        )
        compatibility = comparison.protocol_compatibility.status
        checks.append(
            CalibrationCICheck(
                check_id="protocol-compatibility",
                status="pass" if compatibility == "compatible" else "fail",
                reason=(
                    "candidate and baseline use compatible evidence protocols"
                    if compatibility == "compatible"
                    else "; ".join(comparison.protocol_compatibility.reasons)
                ),
                enforced=enforce_protocol,
                artifact=str(comparison_path),
            )
        )

    status = _overall_status(
        assessment.status,
        verification_valid=verification.valid,
        compatibility=compatibility,
        enforce_protocol=enforce_protocol,
    )
    summary_path = output_path / "summary.md"
    artifact = CalibrationCIArtifact(
        status=status,
        candidate=candidate_input,
        baseline=baseline_input,
        policy=_ci_policy_input(policy_source_path, policy_artifact),
        assessment_status=assessment.status,
        protocol_compatibility=compatibility,
        checks=checks,
        artifacts=CalibrationCIArtifacts(
            evidence=contract_paths["evidence"],
            assessment=str(assessment_path),
            protocol=contract_paths["protocol"],
            evidence_policy=contract_paths["policy"],
            ci_policy=str(ci_policy_path),
            transforms=contract_paths["transforms"],
            bundle=contract_paths["bundle"],
            verification=contract_paths["verification"],
            evidence_card=str(evidence_card_path),
            comparison=str(comparison_path) if comparison_path is not None else None,
            comparison_table=(
                str(comparison_table_path) if comparison_table_path is not None else None
            ),
            markdown_summary=str(summary_path),
        ),
        provenance=_ci_provenance(
            calibrex_version=calibrex_version,
            command=command,
        ),
    )
    write_text(summary_path, render_calibration_ci_markdown(artifact))
    write_mapping(
        output_path / "calibration-ci.json",
        artifact.model_dump(mode="json", exclude_none=True),
    )
    return artifact


def render_calibration_ci_markdown(artifact: CalibrationCIArtifact) -> str:
    """Render a compact GitHub Step Summary from a CI decision."""

    icon = {"pass": "✅", "fail": "❌", "inconclusive": "⚠️"}[artifact.status]
    lines = [
        "# Calibrex Calibration CI",
        "",
        f"## {icon} {artifact.status.upper()}",
        "",
        f"- Candidate: `{artifact.candidate.path}`",
        f"- Run: `{artifact.candidate.run_id}`",
        f"- Assessment: **{artifact.assessment_status.upper()}**",
    ]
    if artifact.baseline is not None:
        lines.append(f"- Baseline: `{artifact.baseline.path}`")
        compatibility = (artifact.protocol_compatibility or "unknown").upper()
        lines.append(
            f"- Protocol compatibility: **{compatibility}**"
        )
    lines.extend(
        [
            "",
            "| Check | Status | Enforced | Reason |",
            "|---|:---:|:---:|---|",
        ]
    )
    for check in artifact.checks:
        reason = check.reason.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| `{check.check_id}` | {check.status.upper()} | "
            f"{'yes' if check.enforced else 'no'} | {reason} |"
        )
    lines.extend(
        [
            "",
            "Generated artifacts are schema-valid and bound to their inputs by SHA-256.",
            "",
        ]
    )
    return "\n".join(lines)


def _ci_input(
    role: Literal["candidate", "baseline"],
    path: Path,
    run_id: str,
) -> CalibrationCIInput:
    digest = sha256_path(path)
    if digest is None:
        msg = f"could not digest CI input: {path}"
        raise ValueError(msg)
    payload = read_mapping(path)
    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, str):
        msg = f"CI input has no string schema_version: {path}"
        raise ValueError(msg)
    return CalibrationCIInput(
        role=role,
        path=str(path),
        sha256=digest,
        schema_version=schema_version,
        run_id=run_id,
    )


def _overall_status(
    assessment_status: AssessmentStatus,
    *,
    verification_valid: bool,
    compatibility: ProtocolCompatibilityStatus | None,
    enforce_protocol: bool,
) -> AssessmentStatus:
    if not verification_valid:
        return "fail"
    if enforce_protocol and compatibility is not None and compatibility != "compatible":
        return "fail"
    return assessment_status


def _ci_policy_input(
    path: Path,
    policy: PolicyArtifact,
) -> CalibrationCIPolicyInput:
    digest = sha256_path(path)
    if digest is None:
        msg = f"could not digest CI policy: {path}"
        raise ValueError(msg)
    return CalibrationCIPolicyInput(
        path=str(path),
        sha256=digest,
        schema_version=policy.schema_version,
        policy_id=policy.policy.policy_id,
    )


def _ci_provenance(
    *,
    calibrex_version: str,
    command: list[str],
) -> CalibrationCIProvenance:
    return CalibrationCIProvenance(
        calibrex_version=calibrex_version,
        git_commit=git_commit(),
        command=command,
        github_repository=os.environ.get("GITHUB_REPOSITORY"),
        github_ref=os.environ.get("GITHUB_REF"),
        github_sha=os.environ.get("GITHUB_SHA"),
        github_run_id=os.environ.get("GITHUB_RUN_ID"),
        github_action_repository=os.environ.get("GITHUB_ACTION_REPOSITORY"),
        github_action_ref=os.environ.get("GITHUB_ACTION_REF"),
    )
