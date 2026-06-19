"""Evidence bundle manifests and integrity verification."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from calibrex.core.exceptions import CalibrexError
from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.report_artifacts import ReportEvidenceArtifact
from calibrex.core.result import StrictModel

EVIDENCE_BUNDLE_SCHEMA_VERSION: Literal["calibrex.evidence_bundle/v0.1"] = (
    "calibrex.evidence_bundle/v0.1"
)
EVIDENCE_BUNDLE_VERIFICATION_SCHEMA_VERSION: Literal[
    "calibrex.evidence_bundle.verification/v0.1"
] = "calibrex.evidence_bundle.verification/v0.1"
BundleArtifactKind = Literal[
    "assessment",
    "report-html",
    "report-summary",
    "report-metrics",
    "report-observability",
    "report-degeneracy",
    "report-evidence",
    "unknown",
]
VerificationClaimScope = Literal[
    "bundle_manifest",
    "artifact_digest",
    "artifact_schema",
    "run_consistency",
    "assessment_source",
    "source_evidence_link",
    "input_file_digest",
]
VerificationClaimStatus = Literal["ok", "failed", "skipped"]


class VerificationClaim(StrictModel):
    """One machine-readable verification check performed on a bundle."""

    scope: VerificationClaimScope
    subject: str
    status: VerificationClaimStatus
    method: str
    expected: dict[str, Any] = Field(default_factory=dict)
    observed: dict[str, Any] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)


class VerificationClaimSummary(StrictModel):
    """Rollup counts for verification claims."""

    total: int = Field(ge=0)
    ok: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    skipped: int = Field(default=0, ge=0)
    by_scope: dict[str, int] = Field(default_factory=dict)


class EvidenceBundleArtifact(StrictModel):
    """One immutable artifact tracked by an evidence bundle."""

    path: str
    kind: BundleArtifactKind = "unknown"
    sha256: str
    size_bytes: int
    schema_version: str | None = None


class EvidenceBundleManifest(StrictModel):
    """Digest manifest for report and evidence artifacts."""

    schema_version: Literal["calibrex.evidence_bundle/v0.1"] = (
        EVIDENCE_BUNDLE_SCHEMA_VERSION
    )
    run_id: str
    created_at: str = ""
    primary_evidence_path: str
    artifact_count: int
    artifacts: list[EvidenceBundleArtifact]


class EvidenceBundleSource(StrictModel):
    """Digest identity of the bundle verified by a verification artifact."""

    path: str
    sha256: str
    size_bytes: int = Field(ge=0)
    schema_version: str
    run_id: str


class EvidenceBundleVerification(StrictModel):
    """Machine-readable verification result for an evidence bundle."""

    schema_version: Literal["calibrex.evidence_bundle.verification/v0.1"] = (
        EVIDENCE_BUNDLE_VERIFICATION_SCHEMA_VERSION
    )
    path: str
    source_bundle: EvidenceBundleSource
    valid: bool
    issue_count: int
    issues: list[str]
    artifact_count: int
    checked_artifacts: list[str]
    input_file_count: int = Field(default=0, ge=0)
    checked_input_file_count: int = Field(default=0, ge=0)
    checked_input_files: list[str] = Field(default_factory=list)
    verification_summary: VerificationClaimSummary
    verification_claims: list[VerificationClaim] = Field(default_factory=list)


def evidence_bundle_json_schema() -> dict[str, Any]:
    """Return the JSON schema for evidence bundle manifests."""

    return EvidenceBundleManifest.model_json_schema()


def evidence_bundle_verification_json_schema() -> dict[str, Any]:
    """Return the JSON schema for evidence bundle verification artifacts."""

    return EvidenceBundleVerification.model_json_schema()


def write_evidence_bundle(
    bundle_path: Path,
    *,
    run_id: str,
    primary_evidence_path: Path,
    artifacts: list[tuple[Path, BundleArtifactKind]],
) -> EvidenceBundleManifest:
    """Write an evidence bundle manifest for already materialized artifacts."""

    entries = [
        _bundle_artifact(bundle_path.parent, path, kind)
        for path, kind in artifacts
        if path.exists()
    ]
    manifest = EvidenceBundleManifest(
        run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        primary_evidence_path=_portable_path(bundle_path.parent, primary_evidence_path),
        artifact_count=len(entries),
        artifacts=entries,
    )
    write_mapping(bundle_path, manifest.model_dump(mode="json", exclude_none=True))
    return manifest


def load_evidence_bundle(path: str | Path) -> EvidenceBundleManifest:
    """Load and validate an evidence bundle manifest."""

    bundle_path = Path(path)
    try:
        return EvidenceBundleManifest.model_validate(read_mapping(bundle_path))
    except Exception as exc:
        raise CalibrexError(f"invalid evidence bundle {bundle_path}: {exc}") from exc


def verify_evidence_bundle(path: str | Path) -> EvidenceBundleVerification:
    """Verify artifact digests and basic run consistency for an evidence bundle."""

    bundle_path = Path(path)
    manifest = load_evidence_bundle(bundle_path)
    bundle_sha256, bundle_size_bytes = _sha256_file(bundle_path)
    base_dir = bundle_path.parent
    issues: list[str] = []
    verification_claims: list[VerificationClaim] = []
    checked: list[str] = []
    artifact_paths = {artifact.path for artifact in manifest.artifacts}
    artifact_digests = {artifact.path: artifact.sha256 for artifact in manifest.artifacts}
    if manifest.primary_evidence_path not in artifact_paths:
        issue = (
            f"primary evidence {manifest.primary_evidence_path!r} is not listed in artifacts"
        )
        issues.append(issue)
        _append_claim(
            verification_claims,
            scope="bundle_manifest",
            subject=str(bundle_path),
            status="failed",
            method="primary_evidence_membership",
            expected={"primary_evidence_path": manifest.primary_evidence_path},
            observed={"artifact_paths": sorted(artifact_paths)},
            issues=[issue],
        )
    else:
        _append_claim(
            verification_claims,
            scope="bundle_manifest",
            subject=str(bundle_path),
            status="ok",
            method="primary_evidence_membership",
            expected={"primary_evidence_path": manifest.primary_evidence_path},
            observed={"artifact_count": len(manifest.artifacts)},
        )
    for artifact in manifest.artifacts:
        checked.append(artifact.path)
        artifact_path = _resolve_artifact_path(base_dir, artifact.path)
        if not artifact_path.exists():
            issue = f"{artifact.path}: missing artifact"
            issues.append(issue)
            _append_claim(
                verification_claims,
                scope="artifact_digest",
                subject=artifact.path,
                status="failed",
                method="sha256_and_size",
                expected={
                    "sha256": artifact.sha256,
                    "size_bytes": artifact.size_bytes,
                },
                observed={"exists": False},
                issues=[issue],
            )
            continue
        digest, size_bytes = _sha256_file(artifact_path)
        digest_issues: list[str] = []
        if digest != artifact.sha256:
            issue = f"{artifact.path}: sha256 mismatch"
            issues.append(issue)
            digest_issues.append(issue)
        if size_bytes != artifact.size_bytes:
            issue = f"{artifact.path}: size mismatch"
            issues.append(issue)
            digest_issues.append(issue)
        _append_claim(
            verification_claims,
            scope="artifact_digest",
            subject=artifact.path,
            status="ok" if not digest_issues else "failed",
            method="sha256_and_size",
            expected={
                "sha256": artifact.sha256,
                "size_bytes": artifact.size_bytes,
            },
            observed={
                "exists": True,
                "sha256": digest,
                "size_bytes": size_bytes,
            },
            issues=digest_issues,
        )
        if artifact.schema_version is not None:
            _verify_schema_version(
                artifact=artifact,
                artifact_path=artifact_path,
                issues=issues,
                verification_claims=verification_claims,
            )
        _verify_run_id(
            artifact=artifact,
            artifact_path=artifact_path,
            expected_run_id=manifest.run_id,
            issues=issues,
            verification_claims=verification_claims,
        )
        if artifact.kind == "assessment":
            _verify_assessment_source(
                artifact=artifact,
                artifact_path=artifact_path,
                primary_evidence_path=manifest.primary_evidence_path,
                primary_evidence_sha256=artifact_digests.get(manifest.primary_evidence_path),
                issues=issues,
                verification_claims=verification_claims,
            )
        elif artifact.kind in {
            "report-summary",
            "report-metrics",
            "report-observability",
            "report-degeneracy",
        }:
            _verify_source_evidence_link(
                artifact=artifact,
                artifact_path=artifact_path,
                primary_evidence_path=manifest.primary_evidence_path,
                primary_evidence_sha256=artifact_digests.get(manifest.primary_evidence_path),
                issues=issues,
                verification_claims=verification_claims,
            )
    input_file_count, checked_input_files = _verify_primary_evidence_input_files(
        base_dir=base_dir,
        primary_evidence_path=manifest.primary_evidence_path,
        issues=issues,
        verification_claims=verification_claims,
    )
    return EvidenceBundleVerification(
        path=str(bundle_path),
        source_bundle=EvidenceBundleSource(
            path=str(bundle_path),
            sha256=bundle_sha256,
            size_bytes=bundle_size_bytes,
            schema_version=manifest.schema_version,
            run_id=manifest.run_id,
        ),
        valid=not issues,
        issue_count=len(issues),
        issues=issues,
        artifact_count=len(manifest.artifacts),
        checked_artifacts=checked,
        input_file_count=input_file_count,
        checked_input_file_count=len(checked_input_files),
        checked_input_files=checked_input_files,
        verification_summary=_verification_claim_summary(verification_claims),
        verification_claims=verification_claims,
    )


def _bundle_artifact(
    base_dir: Path,
    path: Path,
    kind: BundleArtifactKind,
) -> EvidenceBundleArtifact:
    digest, size_bytes = _sha256_file(path)
    schema_version = _schema_version(path)
    return EvidenceBundleArtifact(
        path=_portable_path(base_dir, path),
        kind=kind,
        sha256=digest,
        size_bytes=size_bytes,
        schema_version=schema_version,
    )


def _portable_path(base_dir: Path, path: Path) -> str:
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return str(path)


def _resolve_artifact_path(base_dir: Path, path: str) -> Path:
    artifact_path = Path(path)
    if artifact_path.is_absolute():
        return artifact_path
    return base_dir / artifact_path


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size_bytes += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size_bytes


def _verification_claim_summary(
    claims: list[VerificationClaim],
) -> VerificationClaimSummary:
    by_scope: dict[str, int] = {}
    ok = 0
    failed = 0
    skipped = 0
    for claim in claims:
        by_scope[claim.scope] = by_scope.get(claim.scope, 0) + 1
        if claim.status == "ok":
            ok += 1
        elif claim.status == "failed":
            failed += 1
        else:
            skipped += 1
    return VerificationClaimSummary(
        total=len(claims),
        ok=ok,
        failed=failed,
        skipped=skipped,
        by_scope=dict(sorted(by_scope.items())),
    )


def _append_claim(
    claims: list[VerificationClaim],
    *,
    scope: VerificationClaimScope,
    subject: str,
    status: VerificationClaimStatus,
    method: str,
    expected: dict[str, Any] | None = None,
    observed: dict[str, Any] | None = None,
    issues: list[str] | None = None,
) -> None:
    claims.append(
        VerificationClaim(
            scope=scope,
            subject=subject,
            status=status,
            method=method,
            expected=expected or {},
            observed=observed or {},
            issues=issues or [],
        )
    )


def _schema_version(path: Path) -> str | None:
    if path.suffix.lower() not in {".json", ".yaml", ".yml"}:
        return None
    try:
        value = read_mapping(path).get("schema_version")
    except Exception:
        return None
    return value if isinstance(value, str) else None


def _verify_schema_version(
    *,
    artifact: EvidenceBundleArtifact,
    artifact_path: Path,
    issues: list[str],
    verification_claims: list[VerificationClaim],
) -> None:
    observed = _schema_version(artifact_path)
    claim_issues: list[str] = []
    if observed != artifact.schema_version:
        issue = (
            f"{artifact.path}: schema_version mismatch "
            f"({observed!r} != {artifact.schema_version!r})"
        )
        issues.append(issue)
        claim_issues.append(issue)
    _append_claim(
        verification_claims,
        scope="artifact_schema",
        subject=artifact.path,
        status="ok" if not claim_issues else "failed",
        method="schema_version_field",
        expected={"schema_version": artifact.schema_version},
        observed={"schema_version": observed},
        issues=claim_issues,
    )


def _verify_run_id(
    *,
    artifact: EvidenceBundleArtifact,
    artifact_path: Path,
    expected_run_id: str,
    issues: list[str],
    verification_claims: list[VerificationClaim],
) -> None:
    if artifact.kind == "report-html":
        _append_claim(
            verification_claims,
            scope="run_consistency",
            subject=artifact.path,
            status="skipped",
            method="run_id_field",
            expected={"run_id": expected_run_id},
            observed={"reason": "html artifacts do not expose structured run metadata"},
        )
        return
    try:
        payload = read_mapping(artifact_path)
    except Exception:
        _append_claim(
            verification_claims,
            scope="run_consistency",
            subject=artifact.path,
            status="skipped",
            method="run_id_field",
            expected={"run_id": expected_run_id},
            observed={"reason": "artifact could not be parsed as structured data"},
        )
        return
    run = payload.get("run")
    if not isinstance(run, dict):
        _append_claim(
            verification_claims,
            scope="run_consistency",
            subject=artifact.path,
            status="skipped",
            method="run_id_field",
            expected={"run_id": expected_run_id},
            observed={"reason": "artifact has no run object"},
        )
        return
    observed = run.get("id")
    claim_issues: list[str] = []
    if observed != expected_run_id:
        issue = (
            f"{artifact.path}: run id mismatch ({observed!r} != {expected_run_id!r})"
        )
        issues.append(issue)
        claim_issues.append(issue)
    _append_claim(
        verification_claims,
        scope="run_consistency",
        subject=artifact.path,
        status="ok" if not claim_issues else "failed",
        method="run_id_field",
        expected={"run_id": expected_run_id},
        observed={"run_id": observed},
        issues=claim_issues,
    )


def _verify_assessment_source(
    *,
    artifact: EvidenceBundleArtifact,
    artifact_path: Path,
    primary_evidence_path: str,
    primary_evidence_sha256: str | None,
    issues: list[str],
    verification_claims: list[VerificationClaim],
) -> None:
    try:
        payload = read_mapping(artifact_path)
    except Exception:
        _append_claim(
            verification_claims,
            scope="assessment_source",
            subject=artifact.path,
            status="skipped",
            method="source_evidence_pointer",
            expected={
                "path": primary_evidence_path,
                "sha256": primary_evidence_sha256,
            },
            observed={"reason": "assessment artifact could not be parsed"},
        )
        return
    source = payload.get("source_evidence")
    if not isinstance(source, dict):
        issue = f"{artifact.path}: missing source_evidence"
        issues.append(issue)
        _append_claim(
            verification_claims,
            scope="assessment_source",
            subject=artifact.path,
            status="failed",
            method="source_evidence_pointer",
            expected={
                "path": primary_evidence_path,
                "sha256": primary_evidence_sha256,
            },
            observed={"source_evidence": None},
            issues=[issue],
        )
        return
    observed_path = source.get("path")
    claim_issues: list[str] = []
    if observed_path is not None and observed_path != primary_evidence_path:
        issue = (
            f"{artifact.path}: source evidence path mismatch "
            f"({observed_path!r} != {primary_evidence_path!r})"
        )
        issues.append(issue)
        claim_issues.append(issue)
    observed_digest = source.get("sha256")
    if primary_evidence_sha256 is not None and observed_digest != primary_evidence_sha256:
        issue = (
            f"{artifact.path}: source evidence sha256 mismatch "
            f"({observed_digest!r} != {primary_evidence_sha256!r})"
        )
        issues.append(issue)
        claim_issues.append(issue)
    _append_claim(
        verification_claims,
        scope="assessment_source",
        subject=artifact.path,
        status="ok" if not claim_issues else "failed",
        method="source_evidence_pointer",
        expected={
            "path": primary_evidence_path,
            "sha256": primary_evidence_sha256,
        },
        observed={
            "path": observed_path,
            "sha256": observed_digest,
        },
        issues=claim_issues,
    )


def _verify_source_evidence_link(
    *,
    artifact: EvidenceBundleArtifact,
    artifact_path: Path,
    primary_evidence_path: str,
    primary_evidence_sha256: str | None,
    issues: list[str],
    verification_claims: list[VerificationClaim],
) -> None:
    try:
        payload = read_mapping(artifact_path)
    except Exception:
        _append_claim(
            verification_claims,
            scope="source_evidence_link",
            subject=artifact.path,
            status="skipped",
            method="source_evidence_pointer",
            expected={
                "path": primary_evidence_path,
                "sha256": primary_evidence_sha256,
            },
            observed={"reason": "artifact could not be parsed"},
        )
        return
    source = payload.get("source_evidence")
    if not isinstance(source, dict):
        issue = f"{artifact.path}: missing source_evidence"
        issues.append(issue)
        _append_claim(
            verification_claims,
            scope="source_evidence_link",
            subject=artifact.path,
            status="failed",
            method="source_evidence_pointer",
            expected={
                "path": primary_evidence_path,
                "sha256": primary_evidence_sha256,
            },
            observed={"source_evidence": None},
            issues=[issue],
        )
        return
    observed_path = source.get("path")
    observed_digest = source.get("sha256")
    claim_issues: list[str] = []
    if observed_path != primary_evidence_path:
        issue = (
            f"{artifact.path}: source evidence path mismatch "
            f"({observed_path!r} != {primary_evidence_path!r})"
        )
        issues.append(issue)
        claim_issues.append(issue)
    if primary_evidence_sha256 is not None and observed_digest != primary_evidence_sha256:
        issue = (
            f"{artifact.path}: source evidence sha256 mismatch "
            f"({observed_digest!r} != {primary_evidence_sha256!r})"
        )
        issues.append(issue)
        claim_issues.append(issue)
    _append_claim(
        verification_claims,
        scope="source_evidence_link",
        subject=artifact.path,
        status="ok" if not claim_issues else "failed",
        method="source_evidence_pointer",
        expected={
            "path": primary_evidence_path,
            "sha256": primary_evidence_sha256,
        },
        observed={
            "path": observed_path,
            "sha256": observed_digest,
        },
        issues=claim_issues,
    )


def _verify_primary_evidence_input_files(
    *,
    base_dir: Path,
    primary_evidence_path: str,
    issues: list[str],
    verification_claims: list[VerificationClaim],
) -> tuple[int, list[str]]:
    evidence_path = _resolve_artifact_path(base_dir, primary_evidence_path)
    if not evidence_path.exists():
        _append_claim(
            verification_claims,
            scope="input_file_digest",
            subject=primary_evidence_path,
            status="skipped",
            method="sha256_and_size",
            observed={"reason": "primary evidence artifact is missing"},
        )
        return 0, []
    try:
        evidence = ReportEvidenceArtifact.model_validate(read_mapping(evidence_path))
    except Exception as exc:
        issue = f"{primary_evidence_path}: invalid evidence artifact ({exc})"
        issues.append(issue)
        _append_claim(
            verification_claims,
            scope="input_file_digest",
            subject=primary_evidence_path,
            status="skipped",
            method="sha256_and_size",
            observed={"reason": "primary evidence artifact is invalid"},
            issues=[issue],
        )
        return 0, []
    checked: list[str] = []
    evidence_dir = evidence_path.parent
    if not evidence.input_files:
        _append_claim(
            verification_claims,
            scope="input_file_digest",
            subject=primary_evidence_path,
            status="skipped",
            method="sha256_and_size",
            observed={"input_file_count": 0},
        )
    for input_file in evidence.input_files:
        checked.append(input_file.path)
        path = _resolve_input_file_path(
            bundle_dir=base_dir,
            evidence_dir=evidence_dir,
            path=input_file.path,
        )
        expected = {
            "path": input_file.path,
            "sha256": input_file.sha256,
            "size_bytes": input_file.size_bytes,
        }
        claim_issues: list[str] = []
        if not path.exists():
            issue = f"{primary_evidence_path}: input file {input_file.path}: missing"
            issues.append(issue)
            claim_issues.append(issue)
            _append_claim(
                verification_claims,
                scope="input_file_digest",
                subject=input_file.path,
                status="failed",
                method="sha256_and_size",
                expected=expected,
                observed={"exists": False},
                issues=claim_issues,
            )
            continue
        digest, size_bytes = _sha256_file(path)
        if input_file.sha256 is not None and digest != input_file.sha256:
            issue = (
                f"{primary_evidence_path}: input file {input_file.path}: sha256 mismatch"
            )
            issues.append(issue)
            claim_issues.append(issue)
        if input_file.size_bytes is not None and size_bytes != input_file.size_bytes:
            issue = (
                f"{primary_evidence_path}: input file {input_file.path}: size mismatch"
            )
            issues.append(issue)
            claim_issues.append(issue)
        _append_claim(
            verification_claims,
            scope="input_file_digest",
            subject=input_file.path,
            status="ok" if not claim_issues else "failed",
            method="sha256_and_size",
            expected=expected,
            observed={
                "exists": True,
                "sha256": digest,
                "size_bytes": size_bytes,
            },
            issues=claim_issues,
        )
    return len(evidence.input_files), checked


def _resolve_input_file_path(*, bundle_dir: Path, evidence_dir: Path, path: str) -> Path:
    input_path = Path(path)
    if input_path.is_absolute():
        return input_path
    if input_path.exists():
        return input_path
    bundle_relative = bundle_dir / input_path
    if bundle_relative.exists():
        return bundle_relative
    return evidence_dir / input_path
