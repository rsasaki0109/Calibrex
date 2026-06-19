"""Evidence bundle manifests and integrity verification."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from calibrex.core.exceptions import CalibrexError
from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.result import StrictModel

EVIDENCE_BUNDLE_SCHEMA_VERSION: Literal["calibrex.evidence_bundle/v0.1"] = (
    "calibrex.evidence_bundle/v0.1"
)
BundleArtifactKind = Literal[
    "report-html",
    "report-summary",
    "report-metrics",
    "report-observability",
    "report-degeneracy",
    "report-evidence",
    "unknown",
]


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


class EvidenceBundleVerification(StrictModel):
    """Machine-readable verification result for an evidence bundle."""

    path: str
    valid: bool
    issue_count: int
    issues: list[str]
    artifact_count: int
    checked_artifacts: list[str]


def evidence_bundle_json_schema() -> dict[str, Any]:
    """Return the JSON schema for evidence bundle manifests."""

    return EvidenceBundleManifest.model_json_schema()


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
    base_dir = bundle_path.parent
    issues: list[str] = []
    checked: list[str] = []
    artifact_paths = {artifact.path for artifact in manifest.artifacts}
    if manifest.primary_evidence_path not in artifact_paths:
        issues.append(
            f"primary evidence {manifest.primary_evidence_path!r} is not listed in artifacts"
        )
    for artifact in manifest.artifacts:
        checked.append(artifact.path)
        artifact_path = _resolve_artifact_path(base_dir, artifact.path)
        if not artifact_path.exists():
            issues.append(f"{artifact.path}: missing artifact")
            continue
        digest, size_bytes = _sha256_file(artifact_path)
        if digest != artifact.sha256:
            issues.append(f"{artifact.path}: sha256 mismatch")
        if size_bytes != artifact.size_bytes:
            issues.append(f"{artifact.path}: size mismatch")
        if artifact.schema_version is not None:
            _verify_schema_version(
                artifact=artifact,
                artifact_path=artifact_path,
                issues=issues,
            )
        _verify_run_id(
            artifact=artifact,
            artifact_path=artifact_path,
            expected_run_id=manifest.run_id,
            issues=issues,
        )
    return EvidenceBundleVerification(
        path=str(bundle_path),
        valid=not issues,
        issue_count=len(issues),
        issues=issues,
        artifact_count=len(manifest.artifacts),
        checked_artifacts=checked,
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
) -> None:
    observed = _schema_version(artifact_path)
    if observed != artifact.schema_version:
        issues.append(
            f"{artifact.path}: schema_version mismatch "
            f"({observed!r} != {artifact.schema_version!r})"
        )


def _verify_run_id(
    *,
    artifact: EvidenceBundleArtifact,
    artifact_path: Path,
    expected_run_id: str,
    issues: list[str],
) -> None:
    if artifact.kind == "report-html":
        return
    try:
        payload = read_mapping(artifact_path)
    except Exception:
        return
    run = payload.get("run")
    if not isinstance(run, dict):
        return
    observed = run.get("id")
    if observed != expected_run_id:
        issues.append(
            f"{artifact.path}: run id mismatch ({observed!r} != {expected_run_id!r})"
        )
