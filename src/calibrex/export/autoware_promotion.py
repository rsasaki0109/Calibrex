"""Safe, ROS-independent promotion gates for Autoware sensor-kit packages.

This module is the package boundary around :mod:`calibrex.export.autoware`.
It intentionally does not import ROS, ``ament`` or ``xacro``.  Autoware
packages are ordinary files at this boundary: YAML is parsed with a strict
duplicate-key loader and URDF/Xacro is inspected with the Python XML parser.
The optional ROS smoke test belongs to a downstream adapter.

The promotion workflow is deliberately two phase::

    inspect/plan (read only) -> verify -> apply -> rollback

``plan`` produces a deterministic set of file replacements and a digest of
the package state observed while making the plan.  ``apply`` refuses a stale
baseline, backs up only files in the explicitly resolved workspace scope, and
atomically replaces every file.  ``rollback`` restores only the files listed
in that application inventory and detects an intervening edit before doing so.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal, cast

import yaml
from pydantic import ConfigDict, Field, field_validator, model_validator

from calibrex import __version__
from calibrex.core.exceptions import CalibrexError
from calibrex.core.io import read_mapping, write_mapping_atomic
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.result import StrictModel, TransformResult, load_result
from calibrex.evaluation.koide_pilot import KoidePilotArtifact, load_koide_pilot

AUTOWARE_PROMOTION_SCHEMA_VERSION: Literal["slac.autoware_promotion/v0.1"] = (
    "slac.autoware_promotion/v0.1"
)
AUTOWARE_PROMOTION_PROTOCOL_ID = "autoware_sensor_kit_promotion/v0.1"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"

PromotionStatus = Literal["PASS", "WARN", "FAIL", "BLOCKED"]
PromotionDecision = Literal["ADOPT", "ADOPT_WITH_WARNING", "DO_NOT_ADOPT", "BLOCKED"]
PromotionApplicationStatus = Literal["planned", "applied", "rolled_back"]
PromotionCandidateKind = Literal["koide-pilot", "result"]
PromotionFileKind = Literal[
    "sensor_kit_calibration",
    "sensors_calibration",
    "urdf",
    "xacro",
    "sensor_parameters",
    "topic_parameters",
    "other",
]


class AutowarePromotionError(CalibrexError):
    """Raised when a package cannot be safely inspected or promoted."""


class AutowarePromotionRoots(StrictModel):
    """Explicit filesystem scope for one promotion.

    All paths are resolved before use.  ``workspace_root`` is the outer safety
    boundary; the package roots identify the package locations within it.  A
    caller may point ``package_root`` at a workspace checkout when a real
    Autoware meta-repository contains several sibling packages.
    """

    model_config = ConfigDict(extra="forbid")

    workspace_root: Path
    package_root: Path
    individual_params_root: Path | None = None
    sensor_kit_description_root: Path | None = None

    @model_validator(mode="after")
    def _resolve_and_validate(self) -> AutowarePromotionRoots:
        workspace = self.workspace_root.expanduser().resolve()
        package = self.package_root.expanduser().resolve()
        object.__setattr__(self, "workspace_root", workspace)
        object.__setattr__(self, "package_root", package)
        if not workspace.exists() or not workspace.is_dir():
            raise ValueError(f"workspace_root is not an existing directory: {workspace}")
        if not package.exists() or not package.is_dir():
            raise ValueError(f"package_root is not an existing directory: {package}")
        if not _is_relative_to(package, workspace):
            raise ValueError("package_root must be within workspace_root")
        for name in ("individual_params_root", "sensor_kit_description_root"):
            value = getattr(self, name)
            if value is None:
                continue
            resolved = value.expanduser().resolve()
            if not resolved.exists() or not resolved.is_dir():
                raise ValueError(f"{name} is not an existing directory: {resolved}")
            if not _is_relative_to(resolved, workspace):
                raise ValueError(f"{name} must be within workspace_root")
            object.__setattr__(self, name, resolved)
        return self

    def relative(self, path: Path) -> str:
        """Return a stable workspace-relative path after scope checking."""

        _assert_safe_path(self.workspace_root, path, allow_missing=False)
        resolved = path.expanduser().resolve()
        if not _is_relative_to(resolved, self.workspace_root):
            raise AutowarePromotionError(
                f"path escapes explicit workspace_root {self.workspace_root}: {path}"
            )
        return resolved.relative_to(self.workspace_root).as_posix()

    def resolve_relative(self, relative: str) -> Path:
        """Resolve a workspace-relative path without permitting traversal."""

        _validate_relative_path_text(relative, field_name="relative package path")
        candidate = (self.workspace_root / relative).resolve()
        if not _is_relative_to(candidate, self.workspace_root):
            raise AutowarePromotionError(
                f"relative package path escapes workspace_root: {relative!r}"
            )
        _assert_safe_path(self.workspace_root, self.workspace_root / relative, allow_missing=True)
        return candidate


class AutowarePromotionPolicy(StrictModel):
    """Admission and package validation policy.

    Empty allowlists mean that the package-aware defaults are used.  Supplying
    an allowlist makes it explicit which frames, topics, or files the caller
    permits a candidate to affect.
    """

    model_config = ConfigDict(extra="forbid")

    base_frame: str = "base_link"
    sensor_kit_base_frame: str = "sensor_kit_base_link"
    allowed_frames: list[str] = Field(default_factory=list)
    allowed_topics: list[str] = Field(default_factory=list)
    allowed_files: list[str] = Field(default_factory=list)
    required_files: list[str] = Field(default_factory=list)
    allow_warnings: bool = False
    require_generic_result_pass: bool = True
    require_koide_pass_adopt: bool = True
    require_sensor_kit_base_edge: bool = True
    require_urdf_or_xacro: bool = True
    validate_topic_frame_bindings: bool = True
    backup_directory: str = ".calibrex/promotions"
    # Production is the safe default.  A local developer must explicitly opt
    # out; ``apply_autoware_promotion`` enforces the smoke gate immediately
    # before mutation for every production/admissible plan.
    profile: Literal["production", "developer"] = "production"
    require_smoke_for_apply: bool = False

    @field_validator("base_frame", "sensor_kit_base_frame")
    @classmethod
    def _nonempty_frame(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("frame names must not be empty")
        return value.strip()

    @field_validator("allowed_frames", "allowed_topics", "allowed_files", "required_files")
    @classmethod
    def _unique_strings(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("allowlist entries must not be empty")
        if len(value) != len(set(value)):
            raise ValueError("allowlist entries must be unique")
        return value

    @field_validator("allowed_files", "required_files")
    @classmethod
    def _relative_file_names(cls, value: list[str]) -> list[str]:
        for item in value:
            try:
                _validate_relative_path_text(item, field_name="file allowlist entry")
            except AutowarePromotionError as exc:
                raise ValueError(
                    f"file allowlist entries must be workspace-relative: {exc}"
                ) from exc
        return value

    @field_validator("backup_directory")
    @classmethod
    def _relative_backup_directory(cls, value: str) -> str:
        try:
            _validate_relative_path_text(value, field_name="backup_directory")
        except AutowarePromotionError as exc:
            raise ValueError(str(exc)) from exc
        return PurePosixPath(value).as_posix()

    @model_validator(mode="after")
    def _production_requires_smoke(self) -> AutowarePromotionPolicy:
        if self.profile == "production" and not self.require_smoke_for_apply:
            object.__setattr__(self, "require_smoke_for_apply", True)
        return self


class AutowarePromotionCandidate(StrictModel):
    """Digest-bound candidate input used by a promotion plan."""

    model_config = ConfigDict(extra="forbid")

    path: Path
    kind: PromotionCandidateKind
    sha256: str = Field(pattern=_SHA256_PATTERN)
    artifact_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    source_run_id: str | None = None
    quality_grade: Literal["pass", "warn", "fail"] | None = None
    adoption_decision: PromotionDecision | None = None


class AutowarePromotionFile(StrictModel):
    """One inspected package file and its content digest."""

    model_config = ConfigDict(extra="forbid")

    path: str
    kind: PromotionFileKind
    root: Literal[
        "workspace_root", "package_root", "individual_params_root", "sensor_kit_description_root"
    ]
    exists: bool = True
    sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    size_bytes: int = Field(default=0, ge=0)


class AutowarePromotionPatch(StrictModel):
    """Deterministic text replacement generated by ``plan``."""

    model_config = ConfigDict(extra="forbid")

    path: str
    operation: Literal["replace", "create"] = "replace"
    before_exists: bool = True
    before_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    after_sha256: str = Field(pattern=_SHA256_PATTERN)
    replacement_text: str
    unified_diff: str


class AutowarePromotionRollback(StrictModel):
    """One exact backup entry produced by an applied promotion."""

    model_config = ConfigDict(extra="forbid")

    path: str
    backup_path: str
    before_exists: bool
    before_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    after_sha256: str = Field(pattern=_SHA256_PATTERN)
    backup_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)


class AutowarePromotionDiagnostic(StrictModel):
    """Machine-readable package or candidate gate result."""

    model_config = ConfigDict(extra="forbid")

    code: str
    severity: Literal["info", "warning", "error"]
    message: str
    path: str | None = None


class AutowarePromotionProvenance(StrictModel):
    """Lineage and self-digest scope for every promotion artifact."""

    model_config = ConfigDict(extra="forbid")

    producer: Literal["calibrex"] = "calibrex"
    tool_name: str = "calibrex.autoware-promotion"
    tool_version: str = __version__
    protocol_id: str = AUTOWARE_PROMOTION_PROTOCOL_ID
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    artifact_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    artifact_digest_scope: str = (
        "canonical promotion artifact excluding artifact_sha256, "
        "provenance.artifact_sha256, and provenance.generated_at"
    )


class AutowarePromotionApplication(StrictModel):
    """Apply/rollback state retained in the promotion artifact."""

    model_config = ConfigDict(extra="forbid")

    status: PromotionApplicationStatus = "planned"
    backup_root: str | None = None
    applied_at: str | None = None
    rolled_back_at: str | None = None
    rollback_inventory: list[AutowarePromotionRollback] = Field(default_factory=list)
    smoke_artifact_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    output_manifest_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    output_file_digests: dict[str, str] = Field(default_factory=dict)
    admission_label: Literal["production-admissible", "developer-only", "not-admissible"] = (
        "developer-only"
    )


class AutowarePackageInspection(StrictModel):
    """Schema-valid read-only inventory and diagnostics for a package."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["slac.autoware_package_inspection/v0.1"] = (
        "slac.autoware_package_inspection/v0.1"
    )
    vehicle_id: str
    sensor_kit_id: str
    roots: AutowarePromotionRoots
    files: list[AutowarePromotionFile] = Field(default_factory=list)
    frame_edges: list[AutowareFrameEdge] = Field(default_factory=list)
    frames: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    diagnostics: list[AutowarePromotionDiagnostic] = Field(default_factory=list)
    status: PromotionStatus
    baseline_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    provenance: AutowarePromotionProvenance
    artifact_sha256: str = Field(pattern=_SHA256_PATTERN)

    def with_artifact_digest(self) -> AutowarePackageInspection:
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
        expected = self.with_artifact_digest().artifact_sha256
        if expected != self.artifact_sha256:
            raise AutowarePromotionError(
                "package inspection self-digest mismatch: "
                f"declared={self.artifact_sha256}, expected={expected}"
            )
        if self.provenance.artifact_sha256 != self.artifact_sha256:
            raise AutowarePromotionError("package inspection provenance digest mismatch")


class AutowareFrameEdge(StrictModel):
    """One declarative parent-to-child edge observed in package files."""

    model_config = ConfigDict(extra="forbid")

    parent: str
    child: str
    source_path: str
    source_kind: Literal["yaml", "xml"]
    translation_m: list[float] = Field(min_length=3, max_length=3)
    rotation_rpy_rad: list[float] = Field(min_length=3, max_length=3)
    quaternion_xyzw: list[float] | None = Field(default=None, min_length=4, max_length=4)
    quaternion_order: Literal["xyzw"] | None = None


class AutowarePromotionArtifact(StrictModel):
    """Complete, schema-valid plan/verification/application contract."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["slac.autoware_promotion/v0.1"] = AUTOWARE_PROMOTION_SCHEMA_VERSION
    promotion_id: str
    vehicle_id: str
    sensor_kit_id: str
    roots: AutowarePromotionRoots
    policy: AutowarePromotionPolicy
    candidate: AutowarePromotionCandidate
    baseline_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    package_manifest: list[AutowarePromotionFile] = Field(default_factory=list)
    status: PromotionStatus
    decision: PromotionDecision
    reason: str = Field(min_length=1)
    admission_label: Literal["production-admissible", "developer-only", "not-admissible"] = (
        "developer-only"
    )
    diagnostics: list[AutowarePromotionDiagnostic] = Field(default_factory=list)
    generated_patches: list[AutowarePromotionPatch] = Field(default_factory=list)
    rollback_inventory: list[AutowarePromotionRollback] = Field(default_factory=list)
    application: AutowarePromotionApplication = Field(default_factory=AutowarePromotionApplication)
    provenance: AutowarePromotionProvenance
    artifact_sha256: str = Field(pattern=_SHA256_PATTERN)

    def with_artifact_digest(self) -> AutowarePromotionArtifact:
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
        expected = self.with_artifact_digest().artifact_sha256
        if expected != self.artifact_sha256:
            raise AutowarePromotionError(
                "promotion artifact self-digest mismatch: "
                f"declared={self.artifact_sha256}, expected={expected}"
            )
        if self.provenance.artifact_sha256 != self.artifact_sha256:
            raise AutowarePromotionError("promotion provenance artifact_sha256 mismatch")

    def save(self, path: str | Path) -> None:
        """Write a digest-complete YAML/JSON promotion artifact."""

        write_mapping_atomic(
            Path(path), self.with_artifact_digest().model_dump(mode="json", exclude_none=False)
        )


def autoware_promotion_json_schema() -> dict[str, Any]:
    """Return the generated schema for promotion artifacts."""

    return AutowarePromotionArtifact.model_json_schema()


def autoware_package_inspection_json_schema() -> dict[str, Any]:
    """Return the generated schema for package inspection artifacts."""

    return AutowarePackageInspection.model_json_schema()


def load_autoware_promotion(path: str | Path) -> AutowarePromotionArtifact:
    """Load and verify a promotion artifact."""

    artifact = AutowarePromotionArtifact.model_validate(read_mapping(Path(path)))
    artifact.verify_artifact_digest()
    return artifact


def load_autoware_package_inspection(path: str | Path) -> AutowarePackageInspection:
    """Load and verify a package inspection artifact."""

    artifact = AutowarePackageInspection.model_validate(read_mapping(Path(path)))
    artifact.verify_artifact_digest()
    return artifact


def inspect_autoware_package(
    roots: AutowarePromotionRoots,
    *,
    vehicle_id: str,
    sensor_kit_id: str,
    policy: AutowarePromotionPolicy | None = None,
    command: Sequence[str] = (),
) -> AutowarePackageInspection:
    """Read and validate an Autoware package without importing ROS."""

    resolved_policy = policy or AutowarePromotionPolicy()
    files = _discover_package_files(roots, resolved_policy)
    diagnostics: list[AutowarePromotionDiagnostic] = []
    edges: list[AutowareFrameEdge] = []
    frames: set[str] = set()
    topics: set[str] = set()

    calibration_yaml_paths = [
        item for item in files if item.kind in {"sensor_kit_calibration", "sensors_calibration"}
    ]
    parameter_yaml_paths = [
        item for item in files if item.kind in {"sensor_parameters", "topic_parameters"}
    ]
    for item in calibration_yaml_paths:
        path = roots.resolve_relative(item.path)
        if not path.exists():
            diagnostics.append(
                _error("missing_file", f"required package file is missing: {item.path}", item.path)
            )
            continue
        try:
            payload = _read_strict_yaml(path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            diagnostics.append(
                _error("invalid_yaml", f"could not parse {item.path}: {exc}", item.path)
            )
            continue
        parsed, parse_diagnostics = _parse_calibration_yaml(payload, item.path)
        edges.extend(parsed)
        diagnostics.extend(parse_diagnostics)
        for edge in parsed:
            frames.update((edge.parent, edge.child))

    xml_paths = [item for item in files if item.kind in {"urdf", "xacro"}]
    if resolved_policy.require_urdf_or_xacro and not xml_paths:
        diagnostics.append(_error("missing_description", "no sensor-kit URDF/Xacro was found"))
    for item in xml_paths:
        path = roots.resolve_relative(item.path)
        if not path.exists():
            diagnostics.append(
                _error("missing_file", f"description file is missing: {item.path}", item.path)
            )
            continue
        parsed, xml_frames, xml_diagnostics = _parse_urdf_or_xacro(path, item.path)
        edges.extend(parsed)
        frames.update(xml_frames)
        diagnostics.extend(xml_diagnostics)

    # Parameter bindings are scanned after calibration and XML declarations so
    # a lexically early params.yaml cannot look like a false frame mismatch.
    for item in parameter_yaml_paths:
        path = roots.resolve_relative(item.path)
        if not path.exists():
            diagnostics.append(
                _error("missing_file", f"required package file is missing: {item.path}", item.path)
            )
            continue
        try:
            payload = _read_strict_yaml(path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            diagnostics.append(
                _error("invalid_yaml", f"could not parse {item.path}: {exc}", item.path)
            )
            continue
        if resolved_policy.validate_topic_frame_bindings:
            yaml_topics, yaml_frames, binding_diagnostics = _scan_parameter_yaml(
                payload, item.path, frames
            )
            topics.update(yaml_topics)
            frames.update(yaml_frames)
            diagnostics.extend(binding_diagnostics)

    # Validate calibration and XML graph constraints using all observed edges.
    diagnostics.extend(_validate_frame_graph(edges, frames, roots, resolved_policy))
    if resolved_policy.allowed_frames:
        allowed = set(resolved_policy.allowed_frames)
        for frame in sorted(frames):
            if frame not in allowed:
                diagnostics.append(
                    _error("frame_not_allowed", f"frame {frame!r} is not in allowed_frames")
                )
    if resolved_policy.allowed_topics:
        allowed_topics = set(resolved_policy.allowed_topics)
        for topic in sorted(topics):
            if topic not in allowed_topics:
                diagnostics.append(
                    _error("topic_not_allowed", f"topic {topic!r} is not in allowed_topics")
                )

    # Required files are workspace-relative and are intentionally checked
    # after discovery so a caller can make the allowed inventory explicit.
    discovered_paths = {item.path for item in files if item.exists}
    for required in resolved_policy.required_files:
        try:
            roots.resolve_relative(required)
        except AutowarePromotionError as exc:
            diagnostics.append(_error("path_escape", str(exc), required))
            continue
        if required not in discovered_paths:
            diagnostics.append(
                _error(
                    "required_file_missing",
                    f"required file was not discovered: {required}",
                    required,
                )
            )

    status = _status_from_diagnostics(diagnostics)
    manifest_digest = _manifest_digest(files)
    provenance = AutowarePromotionProvenance(command=list(command), git_commit=git_commit())
    inspection = AutowarePackageInspection(
        vehicle_id=vehicle_id,
        sensor_kit_id=sensor_kit_id,
        roots=roots,
        files=files,
        frame_edges=edges,
        frames=sorted(frames),
        topics=sorted(topics),
        diagnostics=diagnostics,
        status=status,
        baseline_manifest_sha256=manifest_digest,
        provenance=provenance,
        artifact_sha256="0" * 64,
    ).with_artifact_digest()
    return inspection


def build_autoware_promotion_plan(
    candidate_path: str | Path,
    *,
    roots: AutowarePromotionRoots,
    vehicle_id: str,
    sensor_kit_id: str,
    baseline_manifest_sha256: str | None = None,
    policy: AutowarePromotionPolicy | None = None,
    command: Sequence[str] = (),
) -> AutowarePromotionArtifact:
    """Build a deterministic, read-only promotion plan.

    The package is never written by this function.  A stale expected baseline
    turns the plan into ``BLOCKED`` and is also rechecked by ``apply``.
    """

    resolved_policy = policy or AutowarePromotionPolicy()
    inspection = inspect_autoware_package(
        roots,
        vehicle_id=vehicle_id,
        sensor_kit_id=sensor_kit_id,
        policy=resolved_policy,
        command=command,
    )
    candidate = _load_promotion_candidate(Path(candidate_path), roots)
    diagnostics = list(inspection.diagnostics)
    if (
        baseline_manifest_sha256 is not None
        and baseline_manifest_sha256 != inspection.baseline_manifest_sha256
    ):
        diagnostics.append(
            _error(
                "stale_baseline",
                "package baseline digest differs from the expected digest supplied by the caller",
            )
        )
    transforms = _candidate_transforms(Path(candidate_path), candidate)
    _verify_candidate_digest(Path(candidate_path), candidate.sha256)
    if not transforms:
        diagnostics.append(
            _error("candidate_without_transforms", "candidate contains no usable transforms")
        )
    patches: list[AutowarePromotionPatch] = []
    if not any(item.severity == "error" for item in diagnostics):
        try:
            patches = _build_patches(roots, inspection, transforms, resolved_policy)
        except AutowarePromotionError as exc:
            diagnostics.append(_error("patch_generation_failed", str(exc)))

    status, decision, reason = _promotion_decision(
        candidate,
        diagnostics,
        policy=resolved_policy,
    )
    provenance = AutowarePromotionProvenance(command=list(command), git_commit=git_commit())
    artifact = AutowarePromotionArtifact(
        promotion_id=_promotion_id(
            vehicle_id, sensor_kit_id, candidate.sha256, inspection.baseline_manifest_sha256
        ),
        vehicle_id=vehicle_id,
        sensor_kit_id=sensor_kit_id,
        roots=roots,
        policy=resolved_policy,
        candidate=candidate,
        baseline_manifest_sha256=inspection.baseline_manifest_sha256,
        package_manifest=inspection.files,
        status=status,
        decision=decision,
        reason=reason,
        admission_label=(
            "not-admissible" if resolved_policy.profile == "production" else "developer-only"
        ),
        diagnostics=diagnostics,
        generated_patches=patches,
        provenance=provenance,
        artifact_sha256="0" * 64,
    ).with_artifact_digest()
    return artifact


def verify_autoware_promotion(
    plan: AutowarePromotionArtifact | str | Path,
    *,
    command: Sequence[str] = (),
) -> AutowarePromotionArtifact:
    """Re-read the package and candidate, returning a refreshed verification.

    Verification does not write the package.  It preserves the original plan
    patch inventory only when the baseline and generated bytes still match.
    """

    artifact = (
        plan if isinstance(plan, AutowarePromotionArtifact) else load_autoware_promotion(plan)
    )
    artifact.verify_artifact_digest()
    inspection = inspect_autoware_package(
        artifact.roots,
        vehicle_id=artifact.vehicle_id,
        sensor_kit_id=artifact.sensor_kit_id,
        policy=artifact.policy,
        command=command,
    )
    diagnostics = list(inspection.diagnostics)
    if inspection.baseline_manifest_sha256 != artifact.baseline_manifest_sha256:
        diagnostics.append(
            _error("stale_baseline", "package baseline changed since plan generation")
        )
    candidate = _load_promotion_candidate(artifact.candidate.path, artifact.roots)
    if (
        candidate.sha256 != artifact.candidate.sha256
        or candidate.model_dump(mode="json", exclude_none=False)
        != artifact.candidate.model_dump(mode="json", exclude_none=False)
    ):
        diagnostics.append(
            _error("stale_candidate", "candidate bytes changed since plan generation")
        )
    transforms = _candidate_transforms(artifact.candidate.path, candidate)
    _verify_candidate_digest(artifact.candidate.path, candidate.sha256)
    patches: list[AutowarePromotionPatch] = []
    if not any(item.severity == "error" for item in diagnostics):
        try:
            patches = _build_patches(artifact.roots, inspection, transforms, artifact.policy)
        except AutowarePromotionError as exc:
            diagnostics.append(_error("patch_generation_failed", str(exc)))
    status, decision, reason = _promotion_decision(candidate, diagnostics, policy=artifact.policy)
    refreshed = artifact.model_copy(
        update={
            "status": status,
            "decision": decision,
            "reason": reason,
            "diagnostics": diagnostics,
            "generated_patches": patches,
            "package_manifest": inspection.files,
            "application": artifact.application.model_copy(update={"status": "planned"}),
            "provenance": artifact.provenance.model_copy(
                update={"command": list(command) or artifact.provenance.command}
            ),
        }
    )
    return refreshed.with_artifact_digest()


def apply_autoware_promotion(
    plan: AutowarePromotionArtifact | str | Path,
    *,
    smoke_artifact: Any | None = None,
    output_path: str | Path | None = None,
) -> AutowarePromotionArtifact:
    """Atomically apply an admissible plan and record exact backups.

    A production plan cannot be applied without a PASS smoke artifact bound to
    the exact plan self-digest, candidate digest, and package baseline.  The
    smoke artifact is revalidated and the package is inspected again directly
    before mutation.  Developer plans may explicitly opt out, but their
    application is never represented as production-admissible evidence.
    """

    from calibrex.export.autoware_smoke import (
        AutowareSmokeArtifact,
        AutowareSmokeError,
        load_autoware_smoke,
        verify_autoware_smoke,
    )

    artifact = (
        plan if isinstance(plan, AutowarePromotionArtifact) else load_autoware_promotion(plan)
    )
    artifact.verify_artifact_digest()
    if artifact.status != "PASS" or artifact.decision != "ADOPT":
        raise AutowarePromotionError(
            f"promotion is not admissible: status={artifact.status}, decision={artifact.decision}"
        )
    if artifact.application.status != "planned":
        raise AutowarePromotionError(
            f"promotion application is already {artifact.application.status}"
        )
    smoke: AutowareSmokeArtifact | None = None
    if smoke_artifact is not None:
        try:
            smoke = (
                smoke_artifact
                if isinstance(smoke_artifact, AutowareSmokeArtifact)
                else load_autoware_smoke(smoke_artifact)
            )
            verify_autoware_smoke(
                smoke,
                artifact,
                require_pass=artifact.policy.profile == "production"
                or artifact.policy.require_smoke_for_apply,
            )
        except (AutowareSmokeError, ValueError, OSError) as exc:
            raise AutowarePromotionError(f"invalid smoke artifact: {exc}") from exc
    if artifact.policy.profile == "production" or artifact.policy.require_smoke_for_apply:
        if smoke is None:
            raise AutowarePromotionError(
                "production/admissible promotion requires a PASS smoke artifact; "
                "run `calibrex autoware smoke run` or explicitly use a developer profile"
            )
    elif smoke is not None:
        # A developer smoke result is useful evidence, but can never elevate
        # a developer plan to production/admissible status.
        verify_autoware_smoke(smoke, artifact)
    current = inspect_autoware_package(
        artifact.roots,
        vehicle_id=artifact.vehicle_id,
        sensor_kit_id=artifact.sensor_kit_id,
        policy=artifact.policy,
        command=("calibrex", "autoware", "promotion", "apply"),
    )
    if current.baseline_manifest_sha256 != artifact.baseline_manifest_sha256:
        raise AutowarePromotionError(
            "stale baseline: package changed since plan generation; rebuild the plan"
        )
    current_candidate = _load_promotion_candidate(artifact.candidate.path, artifact.roots)
    if (
        current_candidate.sha256 != artifact.candidate.sha256
        or current_candidate.model_dump(mode="json", exclude_none=False)
        != artifact.candidate.model_dump(mode="json", exclude_none=False)
    ):
        raise AutowarePromotionError(
            "stale candidate: candidate changed after smoke/plan verification; rebuild the plan"
        )
    if _manifest_digest(current.files) != _manifest_digest(artifact.package_manifest):
        raise AutowarePromotionError(
            "promotion package manifest is stale or tampered; rebuild the plan"
        )
    current_transforms = _candidate_transforms(current_candidate.path, current_candidate)
    _verify_candidate_digest(current_candidate.path, current_candidate.sha256)
    try:
        expected_patches = _build_patches(
            artifact.roots, current, current_transforms, artifact.policy
        )
    except AutowarePromotionError as exc:
        raise AutowarePromotionError(
            f"promotion patch inventory is no longer valid: {exc}"
        ) from exc
    if _patch_payloads(expected_patches) != _patch_payloads(artifact.generated_patches):
        raise AutowarePromotionError(
            "promotion patch inventory is stale or tampered; rebuild the plan"
        )
    if smoke is not None and (
        smoke.workspace_baseline_manifest_sha256 != current.baseline_manifest_sha256
    ):
        # Rebind against the freshly inspected baseline immediately before the
        # patch-input check.  This closes the stale-workspace window between a
        # smoke import and the filesystem mutation.
        raise AutowarePromotionError(
            "stale workspace: smoke baseline differs from the current package baseline"
        )
    if smoke is not None:
        try:
            # Re-load path-backed evidence and verify its self-digest a second
            # time at the mutation boundary.  A caller cannot replace a smoke
            # file after the first check without being detected here.
            latest_smoke = (
                load_autoware_smoke(smoke_artifact)
                if isinstance(smoke_artifact, (str, Path))
                else smoke
            )
            smoke = verify_autoware_smoke(
                latest_smoke,
                artifact,
                require_pass=artifact.policy.profile == "production"
                or artifact.policy.require_smoke_for_apply,
            )
        except (AutowareSmokeError, ValueError, OSError) as exc:
            raise AutowarePromotionError(f"smoke changed before apply: {exc}") from exc
    _verify_patch_inputs(artifact.roots, artifact.generated_patches)
    effective_output = output_path if output_path is not None else plan
    if isinstance(effective_output, (str, Path)):
        _validate_artifact_output_path(artifact, Path(effective_output))
    backup_candidate = (
        artifact.roots.workspace_root / artifact.policy.backup_directory / artifact.promotion_id
    )
    _assert_safe_path(artifact.roots.workspace_root, backup_candidate, allow_missing=True)
    backup_root = backup_candidate.resolve()
    if not _is_relative_to(backup_root, artifact.roots.workspace_root):
        raise AutowarePromotionError("backup directory escapes workspace_root")
    _assert_safe_path(artifact.roots.workspace_root, backup_root, allow_missing=True)
    if backup_root.exists():
        raise AutowarePromotionError(f"backup directory already exists: {backup_root}")
    rollback: list[AutowarePromotionRollback] = []
    created_backups: list[Path] = []
    try:
        _mkdir_safe(artifact.roots.workspace_root, backup_candidate)
        for index, patch in enumerate(artifact.generated_patches):
            target = artifact.roots.resolve_relative(patch.path)
            if not _is_relative_to(target, artifact.roots.workspace_root):
                raise AutowarePromotionError(f"patch path escapes workspace_root: {patch.path}")
            _assert_safe_path(artifact.roots.workspace_root, target, allow_missing=True)
            backup_path = backup_root / f"{index:04d}-{Path(patch.path).name}.bak"
            _assert_safe_path(artifact.roots.workspace_root, backup_path, allow_missing=True)
            if target.exists():
                if not target.is_file():
                    raise AutowarePromotionError(f"patch target is not a file: {patch.path}")
                _atomic_replace_bytes(backup_path, target.read_bytes())
                created_backups.append(backup_path)
                _assert_safe_path(artifact.roots.workspace_root, backup_path, allow_missing=False)
                backup_sha = sha256_path(backup_path)
                if backup_sha is None:
                    raise AutowarePromotionError(
                        f"backup disappeared immediately after creation: {patch.path}"
                    )
                before_exists = True
            else:
                backup_sha = None
                before_exists = False
            rollback.append(
                AutowarePromotionRollback(
                    path=patch.path,
                    backup_path=artifact.roots.relative(backup_path),
                    before_exists=before_exists,
                    before_sha256=patch.before_sha256,
                    after_sha256=patch.after_sha256,
                    backup_sha256=backup_sha,
                )
            )
            _atomic_replace_text(target, patch.replacement_text)
        _verify_applied_outputs(artifact.roots, rollback)
        application = AutowarePromotionApplication(
            status="applied",
            backup_root=artifact.roots.relative(backup_root),
            applied_at=datetime.now(timezone.utc).isoformat(),
            rollback_inventory=rollback,
            smoke_artifact_sha256=smoke.artifact_sha256 if smoke is not None else None,
            output_manifest_sha256=_canonical_sha256(
                {item.path: item.after_sha256 for item in rollback}
            ),
            output_file_digests={item.path: item.after_sha256 for item in rollback},
            admission_label=(
                "production-admissible"
                if smoke is not None and artifact.policy.profile == "production"
                else "developer-only"
            ),
        )
        updated = artifact.model_copy(
            update={
                "application": application,
                "rollback_inventory": rollback,
                "admission_label": application.admission_label,
            }
        ).with_artifact_digest()
        if output_path is not None:
            updated.save(output_path)
        elif isinstance(plan, (str, Path)):
            updated.save(plan)
        return updated
    except Exception as exc:
        # Restore only entries already backed up/applied.  Never recursively
        # delete the backup root: if restoration or cleanup fails, the exact
        # backup evidence remains available for manual recovery.
        rollback_errors: list[str] = []
        for item in reversed(rollback):
            try:
                _restore_one(artifact.roots, item, verify_current=False)
            except Exception as restore_exc:
                rollback_errors.append(f"{item.path}: {restore_exc}")
        cleanup_error: str | None = None
        if not rollback_errors:
            try:
                _cleanup_created_backups(
                    artifact.roots.workspace_root, backup_root, created_backups
                )
            except Exception as cleanup_exc:
                cleanup_error = str(cleanup_exc)
        if rollback_errors or cleanup_error:
            details = [*rollback_errors]
            if cleanup_error:
                details.append(f"backup cleanup: {cleanup_error}")
            raise AutowarePromotionError(
                "promotion failed and backup evidence was retained: " + "; ".join(details)
            ) from exc
        raise


def rollback_autoware_promotion(
    plan: AutowarePromotionArtifact | str | Path,
    *,
    output_path: str | Path | None = None,
) -> AutowarePromotionArtifact:
    """Restore an applied promotion after checking every current digest."""

    artifact = (
        plan if isinstance(plan, AutowarePromotionArtifact) else load_autoware_promotion(plan)
    )
    artifact.verify_artifact_digest()
    if artifact.application.status != "applied":
        raise AutowarePromotionError("only an applied promotion can be rolled back")
    # Preflight every entry before restoring any file.  A missing backup or an
    # intervening edit therefore cannot leave a partially rolled-back package.
    for item in artifact.rollback_inventory:
        _verify_rollback_entry(artifact.roots, item)
    for item in artifact.rollback_inventory:
        _restore_one(artifact.roots, item, verify_current=False)
    for item in artifact.rollback_inventory:
        _verify_restored_entry(artifact.roots, item)
    updated = artifact.model_copy(
        update={
            "application": artifact.application.model_copy(
                update={
                    "status": "rolled_back",
                    "rolled_back_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        }
    ).with_artifact_digest()
    if output_path is not None:
        updated.save(output_path)
    elif isinstance(plan, (str, Path)):
        updated.save(plan)
    return updated


def _load_promotion_candidate(
    path: Path, roots: AutowarePromotionRoots
) -> AutowarePromotionCandidate:
    del roots  # Candidate artifacts may be handed off outside the ROS workspace.
    _assert_no_link_components(path)
    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise AutowarePromotionError(f"candidate does not exist: {path}")
    raw_digest = sha256_path(resolved)
    if raw_digest is None:
        raise AutowarePromotionError(f"could not digest candidate: {path}")
    try:
        payload = read_mapping(resolved)
    except Exception as exc:
        raise AutowarePromotionError(
            f"candidate is not a YAML/JSON artifact: {path}: {exc}"
        ) from exc
    schema_version = payload.get("schema_version")
    if schema_version == "slac.koide_pilot/v0.1":
        try:
            pilot = load_koide_pilot(resolved)
        except Exception as exc:
            raise AutowarePromotionError(f"invalid Koide pilot candidate: {exc}") from exc
        candidate = AutowarePromotionCandidate(
            path=resolved,
            kind="koide-pilot",
            sha256=raw_digest,
            artifact_sha256=pilot.artifact_sha256,
            source_run_id=pilot.pilot_id,
            quality_grade="pass"
            if pilot.status == "PASS"
            else "warn"
            if pilot.status == "WARN"
            else "fail",
            adoption_decision=_koide_decision(pilot),
        )
        _verify_candidate_digest(resolved, raw_digest)
        return candidate
    if schema_version == "slac.result/v0.1":
        try:
            result = load_result(resolved)
        except Exception as exc:
            raise AutowarePromotionError(f"invalid Calibrex result candidate: {exc}") from exc
        candidate = AutowarePromotionCandidate(
            path=resolved,
            kind="result",
            sha256=raw_digest,
            artifact_sha256=_canonical_sha256(result.model_dump(mode="json", exclude_none=False)),
            source_run_id=result.run.id,
            quality_grade=result.quality.grade,
            adoption_decision=None,
        )
        _verify_candidate_digest(resolved, raw_digest)
        return candidate
    raise AutowarePromotionError(
        "unsupported promotion candidate "
        f"schema_version={schema_version!r}; expected slac.result/v0.1 "
        "or slac.koide_pilot/v0.1"
    )


def _verify_candidate_digest(path: Path, expected: str) -> None:
    """Ensure candidate bytes did not change during a read/parse operation."""

    _assert_no_link_components(path)
    observed = sha256_path(path)
    if observed != expected:
        raise AutowarePromotionError(f"candidate changed while it was being read: {path}")


def _candidate_transforms(
    path: Path, candidate: AutowarePromotionCandidate
) -> list[TransformResult]:
    if candidate.kind == "koide-pilot":
        pilot = load_koide_pilot(path)
        if pilot.candidate_transform_camera_lidar is None:
            return []
        transform = pilot.candidate_transform_camera_lidar
        return [
            TransformResult(
                parent=transform.parent,
                child=transform.child,
                translation_m=list(transform.translation_m),
                rotation_quat_xyzw=list(transform.rotation_quat_xyzw),
            )
        ]
    result = load_result(path)
    transforms: list[TransformResult] = []
    for mapping in (result.transforms, result.candidate_extrinsics):
        transforms.extend(mapping.values())
    return _dedupe_transforms(transforms)


def _dedupe_transforms(transforms: Sequence[TransformResult]) -> list[TransformResult]:
    seen: set[tuple[str, str, tuple[float, ...], tuple[float, ...]]] = set()
    output: list[TransformResult] = []
    for transform in transforms:
        key = (
            transform.parent,
            transform.child,
            tuple(float(value) for value in transform.translation_m),
            tuple(float(value) for value in transform.rotation_quat_xyzw),
        )
        if key not in seen:
            seen.add(key)
            output.append(transform)
    return output


def _build_patches(
    roots: AutowarePromotionRoots,
    inspection: AutowarePackageInspection,
    transforms: Sequence[TransformResult],
    policy: AutowarePromotionPolicy,
) -> list[AutowarePromotionPatch]:
    file_records = [
        item
        for item in inspection.files
        if item.kind in {"sensor_kit_calibration", "sensors_calibration"} and item.exists
    ]
    if not file_records:
        raise AutowarePromotionError("no calibration YAML files are available for patching")
    grouped: dict[str, list[TransformResult]] = {}
    for transform in transforms:
        _validate_candidate_transform(transform, policy)
        matches = [
            item
            for item in file_records
            if _transform_belongs_to_file(transform, item.kind, policy)
        ]
        if not matches:
            # A candidate with a frame pair not present in the package is not
            # silently added; callers must explicitly include the target file.
            continue
        for item in matches:
            grouped.setdefault(item.path, []).append(transform)
    if not grouped:
        raise AutowarePromotionError(
            "candidate transforms do not match any package calibration edge; "
            "ensure frame names and base/sensor-kit direction are explicit"
        )
    patches: list[AutowarePromotionPatch] = []
    for relative, selected in sorted(grouped.items()):
        if policy.allowed_files and relative not in set(policy.allowed_files):
            raise AutowarePromotionError(f"patch target is not in allowed_files: {relative}")
        target = roots.resolve_relative(relative)
        before = _read_text_exact(target)
        payload = _read_strict_yaml(target)
        for transform in selected:
            _replace_calibration_edge(payload, transform)
        replacement = _serialize_yaml(payload)
        if replacement == before:
            continue
        patch = AutowarePromotionPatch(
            path=relative,
            operation="replace",
            before_exists=True,
            before_sha256=_sha256_bytes(before.encode("utf-8")),
            after_sha256=_sha256_bytes(replacement.encode("utf-8")),
            replacement_text=replacement,
            unified_diff="".join(
                difflib.unified_diff(
                    before.splitlines(keepends=True),
                    replacement.splitlines(keepends=True),
                    fromfile=relative,
                    tofile=relative,
                    lineterm="",
                )
            ),
        )
        patches.append(patch)
    if not patches:
        raise AutowarePromotionError("candidate produces no package changes")
    return patches


def _replace_calibration_edge(payload: dict[str, Any], transform: TransformResult) -> None:
    parent = transform.parent
    child = transform.child
    if parent not in payload or not isinstance(payload[parent], dict):
        raise AutowarePromotionError(f"calibration parent frame is not declared: {parent}")
    if child not in payload[parent] or not isinstance(payload[parent][child], dict):
        raise AutowarePromotionError(f"calibration child frame is not declared: {child}")
    record = cast(dict[str, Any], payload[parent][child])
    record.update(
        {
            "x": float(transform.translation_m[0]),
            "y": float(transform.translation_m[1]),
            "z": float(transform.translation_m[2]),
            "roll": _rpy_from_quaternion(transform.rotation_quat_xyzw)[0],
            "pitch": _rpy_from_quaternion(transform.rotation_quat_xyzw)[1],
            "yaw": _rpy_from_quaternion(transform.rotation_quat_xyzw)[2],
        }
    )


def _transform_belongs_to_file(
    transform: TransformResult,
    kind: PromotionFileKind,
    policy: AutowarePromotionPolicy,
) -> bool:
    if kind == "sensors_calibration":
        return (
            transform.parent == policy.base_frame or transform.child == policy.sensor_kit_base_frame
        )
    if kind == "sensor_kit_calibration":
        return transform.parent == policy.sensor_kit_base_frame
    return False


def _validate_candidate_transform(
    transform: TransformResult, policy: AutowarePromotionPolicy
) -> None:
    if transform.convention != "T_parent_child":
        raise AutowarePromotionError("candidate transform must use T_parent_child")
    if (
        not transform.parent.strip()
        or not transform.child.strip()
        or transform.parent == transform.child
    ):
        raise AutowarePromotionError("candidate transform has invalid frame direction")
    if policy.allowed_frames:
        allowed = set(policy.allowed_frames)
        if transform.parent not in allowed or transform.child not in allowed:
            raise AutowarePromotionError(
                f"candidate transform {transform.parent}->{transform.child} uses a "
                "frame outside allowed_frames"
            )
    values = [*transform.translation_m, *transform.rotation_quat_xyzw]
    if not all(math.isfinite(float(value)) for value in values):
        raise AutowarePromotionError("candidate transform contains non-finite values")
    norm = math.sqrt(sum(float(value) ** 2 for value in transform.rotation_quat_xyzw))
    if abs(norm - 1.0) > 1.0e-3:
        raise AutowarePromotionError("candidate transform quaternion is not unit length")


def _promotion_decision(
    candidate: AutowarePromotionCandidate,
    diagnostics: Sequence[AutowarePromotionDiagnostic],
    *,
    policy: AutowarePromotionPolicy,
) -> tuple[PromotionStatus, PromotionDecision, str]:
    errors = [item.message for item in diagnostics if item.severity == "error"]
    warnings = [item.message for item in diagnostics if item.severity == "warning"]
    if errors:
        if any(item.code in {"stale_baseline", "stale_candidate"} for item in diagnostics):
            return "BLOCKED", "BLOCKED", "; ".join(errors)
        return "FAIL", "DO_NOT_ADOPT", "; ".join(errors)
    if (
        candidate.kind == "koide-pilot"
        and policy.require_koide_pass_adopt
        and (candidate.quality_grade != "pass" or candidate.adoption_decision != "ADOPT")
    ):
        return "FAIL", "DO_NOT_ADOPT", "Koide promotion requires a PASS/ADOPT pilot"
    if (
        candidate.kind == "result"
        and policy.require_generic_result_pass
        and candidate.quality_grade != "pass"
    ):
        return "FAIL", "DO_NOT_ADOPT", "generic result promotion requires quality grade pass"
    if warnings:
        if not policy.allow_warnings:
            return "WARN", "DO_NOT_ADOPT", "; ".join(warnings)
        return "WARN", "ADOPT_WITH_WARNING", "; ".join(warnings)
    return "PASS", "ADOPT", "candidate and package promotion gates passed"


def _discover_package_files(
    roots: AutowarePromotionRoots,
    policy: AutowarePromotionPolicy,
) -> list[AutowarePromotionFile]:
    candidates: dict[str, tuple[PromotionFileKind, str]] = {}
    root_specs: list[tuple[str, Path]] = [("package_root", roots.package_root)]
    if roots.individual_params_root is not None:
        root_specs.append(("individual_params_root", roots.individual_params_root))
    if roots.sensor_kit_description_root is not None:
        root_specs.append(("sensor_kit_description_root", roots.sensor_kit_description_root))
    for root_name, root in root_specs:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            suffix = path.suffix.lower()
            name = path.name.lower()
            if suffix not in {".yaml", ".yml", ".xml", ".urdf", ".xacro"}:
                continue
            if name == "sensor_kit_calibration.yaml" or name == "sensor_kit_calibration.yml":
                kind: PromotionFileKind = "sensor_kit_calibration"
            elif name == "sensors_calibration.yaml" or name == "sensors_calibration.yml":
                kind = "sensors_calibration"
            elif (
                suffix in {".xml", ".urdf"}
                or name.endswith(".urdf.xacro")
                or name.endswith(".xacro")
            ):
                kind = "urdf" if suffix in {".xml", ".urdf"} else "xacro"
            elif "topic" in name:
                kind = "topic_parameters"
            else:
                kind = "sensor_parameters"
            relative = roots.relative(path)
            previous = candidates.get(relative)
            if previous is None or _file_kind_priority(kind) < _file_kind_priority(previous[0]):
                candidates[relative] = (kind, root_name)
    output: list[AutowarePromotionFile] = []
    for relative, (kind, root_name) in sorted(candidates.items()):
        path = roots.resolve_relative(relative)
        digest = sha256_path(path)
        output.append(
            AutowarePromotionFile(
                path=relative,
                kind=kind,
                root=cast(Any, root_name),
                exists=path.exists(),
                sha256=digest,
                size_bytes=path.stat().st_size if path.exists() else 0,
            )
        )
    return output


def _file_kind_priority(kind: PromotionFileKind) -> int:
    return {
        "sensor_kit_calibration": 0,
        "sensors_calibration": 1,
        "urdf": 2,
        "xacro": 3,
        "sensor_parameters": 4,
        "topic_parameters": 5,
        "other": 6,
    }[kind]


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys instead of last-wins."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ValueError("Autoware YAML mapping keys must be strings")
        if key in mapping:
            raise ValueError(f"duplicate YAML mapping key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _read_strict_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.load(stream, Loader=_UniqueKeyLoader)
    if not isinstance(payload, dict):
        raise ValueError("Autoware YAML document must be a mapping")
    return cast(dict[str, Any], payload)


def _parse_calibration_yaml(
    payload: Mapping[str, Any], source_path: str
) -> tuple[list[AutowareFrameEdge], list[AutowarePromotionDiagnostic]]:
    edges: list[AutowareFrameEdge] = []
    diagnostics: list[AutowarePromotionDiagnostic] = []
    for parent, children in payload.items():
        if parent in {"calibrex", "schema_version", "format", "provenance"}:
            continue
        if not isinstance(parent, str) or not isinstance(children, Mapping):
            continue
        for child, raw in children.items():
            if not isinstance(child, str) or not isinstance(raw, Mapping):
                continue
            try:
                edge = _edge_from_mapping(parent, child, raw, source_path)
            except ValueError as exc:
                diagnostics.append(
                    _error("invalid_transform", f"{source_path}: {exc}", source_path)
                )
                continue
            edges.append(edge)
    return edges, diagnostics


def _edge_from_mapping(
    parent: str, child: str, raw: Mapping[str, Any], source_path: str
) -> AutowareFrameEdge:
    try:
        xyz = [float(raw[key]) for key in ("x", "y", "z")]
        rpy = [float(raw[key]) for key in ("roll", "pitch", "yaw")]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("calibration edge requires numeric x/y/z/roll/pitch/yaw") from exc
    if not all(math.isfinite(value) for value in [*xyz, *rpy]):
        raise ValueError("calibration edge contains non-finite values")
    quaternion: list[float] | None = None
    order: Literal["xyzw"] | None = None
    if "quaternion" in raw or "rotation_quat_xyzw" in raw:
        value = raw.get("quaternion", raw.get("rotation_quat_xyzw"))
        if isinstance(value, Mapping):
            try:
                quaternion = [float(value[key]) for key in ("x", "y", "z", "w")]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("quaternion mapping requires x/y/z/w") from exc
        elif isinstance(value, Sequence) and not isinstance(value, str) and len(value) == 4:
            try:
                quaternion = [float(item) for item in value]
            except (TypeError, ValueError) as exc:
                raise ValueError("quaternion entries must be numeric") from exc
        else:
            raise ValueError("quaternion must be a four-element xyzw sequence")
        declared = raw.get("quaternion_order", raw.get("rotation_order"))
        if declared != "xyzw":
            raise ValueError("quaternion order must be explicitly declared as xyzw")
        order = "xyzw"
        norm = math.sqrt(sum(value * value for value in quaternion))
        if abs(norm - 1.0) > 1.0e-3:
            raise ValueError("quaternion is not unit length")
    return AutowareFrameEdge(
        parent=parent,
        child=child,
        source_path=source_path,
        source_kind="yaml",
        translation_m=xyz,
        rotation_rpy_rad=rpy,
        quaternion_xyzw=quaternion,
        quaternion_order=order,
    )


def _parse_urdf_or_xacro(
    path: Path, source_path: str
) -> tuple[list[AutowareFrameEdge], set[str], list[AutowarePromotionDiagnostic]]:
    edges: list[AutowareFrameEdge] = []
    frames: set[str] = set()
    diagnostics: list[AutowarePromotionDiagnostic] = []
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        return (
            [],
            set(),
            [_error("invalid_xml", f"could not parse {source_path}: {exc}", source_path)],
        )
    for element in root.iter():
        if _local_name(element.tag) == "link":
            name = element.attrib.get("name")
            if name:
                if name in frames:
                    diagnostics.append(
                        _error("duplicate_frame", f"duplicate XML link frame: {name}", source_path)
                    )
                frames.add(name)
        if _local_name(element.tag) != "joint":
            continue
        parent_element = next((item for item in element if _local_name(item.tag) == "parent"), None)
        child_element = next((item for item in element if _local_name(item.tag) == "child"), None)
        if parent_element is None or child_element is None:
            diagnostics.append(
                _error("invalid_joint", f"joint in {source_path} lacks parent/child", source_path)
            )
            continue
        parent = parent_element.attrib.get("link")
        child = child_element.attrib.get("link")
        if not parent or not child:
            diagnostics.append(
                _error(
                    "invalid_joint", f"joint in {source_path} has empty parent/child", source_path
                )
            )
            continue
        frames.update((parent, child))
        origin = next((item for item in element if _local_name(item.tag) == "origin"), None)
        xyz = _xml_vector(origin.attrib.get("xyz") if origin is not None else None, 3)
        rpy = _xml_vector(origin.attrib.get("rpy") if origin is not None else None, 3)
        if xyz is None or rpy is None:
            # Xacro substitutions are legal before expansion; retain graph
            # evidence while reporting that numerical XML validation is deferred.
            diagnostics.append(
                AutowarePromotionDiagnostic(
                    code="symbolic_origin",
                    severity="info",
                    message=(
                        f"joint {parent}->{child} has symbolic/missing origin; "
                        "ROS/Xacro expansion remains a downstream check"
                    ),
                    path=source_path,
                )
            )
            xyz = [0.0, 0.0, 0.0]
            rpy = [0.0, 0.0, 0.0]
        edges.append(
            AutowareFrameEdge(
                parent=parent,
                child=child,
                source_path=source_path,
                source_kind="xml",
                translation_m=xyz,
                rotation_rpy_rad=rpy,
            )
        )
    return edges, frames, diagnostics


def _scan_parameter_yaml(
    payload: Mapping[str, Any], source_path: str, known_frames: set[str]
) -> tuple[set[str], set[str], list[AutowarePromotionDiagnostic]]:
    topics: set[str] = set()
    frames: set[str] = set()
    diagnostics: list[AutowarePromotionDiagnostic] = []

    def visit(value: Any, key: str | None = None) -> None:
        if isinstance(value, Mapping):
            for name, child in value.items():
                visit(child, str(name))
            return
        if isinstance(value, list):
            for child in value:
                visit(child, key)
            return
        if not isinstance(value, str):
            return
        lower = (key or "").lower()
        if "topic" in lower:
            topics.add(value)
        if lower in {"frame", "frame_id", "child_frame", "parent_frame", "sensor_frame"}:
            frames.add(value)
            if (
                known_frames
                and value not in known_frames
                and "${" not in value
                and "$" not in value
            ):
                diagnostics.append(
                    _error(
                        "topic_frame_mismatch",
                        f"{source_path}: declarative frame {value!r} is not in the "
                        "package frame graph",
                        source_path,
                    )
                )

    visit(payload)
    return topics, frames, diagnostics


def _validate_frame_graph(
    edges: Sequence[AutowareFrameEdge],
    frames: set[str],
    roots: AutowarePromotionRoots,
    policy: AutowarePromotionPolicy,
) -> list[AutowarePromotionDiagnostic]:
    diagnostics: list[AutowarePromotionDiagnostic] = []
    calibration_edges = [edge for edge in edges if edge.source_kind == "yaml"]
    child_edges: dict[str, list[AutowareFrameEdge]] = {}
    for edge in edges:
        child_edges.setdefault(edge.child, []).append(edge)
    for child, declarations in sorted(child_edges.items()):
        if len(declarations) > 1:
            # Autoware's individual_params package intentionally mirrors the
            # sensor-kit calibration YAML.  Identical declarations are one
            # logical edge; conflicting declarations are an ambiguity.
            fingerprints = {
                (
                    edge.parent,
                    tuple(edge.translation_m),
                    tuple(edge.rotation_rpy_rad),
                    tuple(edge.quaternion_xyzw or ()),
                )
                for edge in declarations
            }
            if len(fingerprints) > 1:
                diagnostics.append(
                    _error(
                        "duplicate_child",
                        "frame child "
                        f"{child!r} is declared more than once with conflicting transforms",
                        ",".join(sorted({edge.source_path for edge in declarations})),
                    )
                )
    adjacency: dict[str, list[str]] = {}
    for edge in edges:
        adjacency.setdefault(edge.parent, []).append(edge.child)
    cycle = _find_cycle(adjacency)
    if cycle:
        diagnostics.append(_error("frame_cycle", "frame graph cycle: " + " -> ".join(cycle)))
    base = policy.base_frame
    kit = policy.sensor_kit_base_frame
    if policy.require_sensor_kit_base_edge and not any(
        edge.parent == base and edge.child == kit for edge in calibration_edges
    ):
        diagnostics.append(
            _error(
                "missing_sensor_kit_base_edge",
                f"missing required {base}->{kit} calibration edge",
            )
        )
    reachable = _reachable(adjacency, base)
    sensor_frames = {edge.child for edge in calibration_edges}
    for frame in sorted(sensor_frames):
        if frame not in reachable:
            diagnostics.append(
                _error(
                    "disconnected_sensor", f"sensor frame {frame!r} is disconnected from {base!r}"
                )
            )
    # If XML declares an independent root, it is a graph error unless it is a
    # symbolic xacro fragment (which has already emitted a warning).
    if frames and base not in frames:
        diagnostics.append(
            _error("missing_base_frame", f"base frame {base!r} is not declared in package files")
        )
    return diagnostics


def _find_cycle(adjacency: Mapping[str, Sequence[str]]) -> list[str] | None:
    visiting: set[str] = set()
    visited: set[str] = set()
    stack: list[str] = []

    def visit(node: str) -> list[str] | None:
        if node in visiting:
            try:
                return [*stack[stack.index(node) :], node]
            except ValueError:
                return [node, node]
        if node in visited:
            return None
        visiting.add(node)
        stack.append(node)
        for child in adjacency.get(node, ()):
            found = visit(child)
            if found:
                return found
        stack.pop()
        visiting.remove(node)
        visited.add(node)
        return None

    for node in sorted(
        set(adjacency) | {child for values in adjacency.values() for child in values}
    ):
        found = visit(node)
        if found:
            return found
    return None


def _reachable(adjacency: Mapping[str, Sequence[str]], root: str) -> set[str]:
    reached: set[str] = set()
    todo = [root]
    while todo:
        node = todo.pop()
        if node in reached:
            continue
        reached.add(node)
        todo.extend(adjacency.get(node, ()))
    return reached


def _verify_patch_inputs(
    roots: AutowarePromotionRoots, patches: Sequence[AutowarePromotionPatch]
) -> None:
    for patch in patches:
        replacement_digest = _sha256_bytes(patch.replacement_text.encode("utf-8"))
        if replacement_digest != patch.after_sha256:
            raise AutowarePromotionError(f"patch replacement digest mismatch: {patch.path}")
        if patch.before_exists and patch.before_sha256 is None:
            raise AutowarePromotionError(f"patch lacks a before digest: {patch.path}")
        if not patch.before_exists and patch.before_sha256 is not None:
            raise AutowarePromotionError(
                f"create patch unexpectedly carries a before digest: {patch.path}"
            )
        target = roots.resolve_relative(patch.path)
        _assert_safe_path(roots.workspace_root, target, allow_missing=True)
        if patch.before_exists:
            if not target.exists() or not target.is_file():
                raise AutowarePromotionError(f"patch input disappeared: {patch.path}")
            digest = sha256_path(target)
            if digest != patch.before_sha256:
                raise AutowarePromotionError(f"stale patch input: {patch.path}")
        elif target.exists():
            raise AutowarePromotionError(f"new patch target unexpectedly exists: {patch.path}")


def _patch_payloads(patches: Sequence[AutowarePromotionPatch]) -> list[dict[str, Any]]:
    return [
        patch.model_dump(mode="json", exclude_none=False)
        for patch in sorted(patches, key=lambda item: item.path)
    ]


def _restore_one(
    roots: AutowarePromotionRoots, item: AutowarePromotionRollback, *, verify_current: bool
) -> None:
    target = roots.resolve_relative(item.path)
    if verify_current:
        current = sha256_path(target)
        if current != item.after_sha256:
            raise AutowarePromotionError(
                f"rollback refused because target changed after apply: {item.path}"
            )
    if item.before_exists:
        backup = roots.resolve_relative(item.backup_path)
        _assert_safe_path(roots.workspace_root, target, allow_missing=True)
        _assert_safe_path(roots.workspace_root, backup, allow_missing=False)
        if not backup.exists() or not backup.is_file():
            raise AutowarePromotionError(f"rollback backup is missing: {item.backup_path}")
        if sha256_path(backup) != item.backup_sha256:
            raise AutowarePromotionError(f"rollback backup digest mismatch: {item.backup_path}")
        _atomic_replace_bytes(target, backup.read_bytes())
    else:
        if target.exists():
            _assert_safe_path(roots.workspace_root, target, allow_missing=False)
            target.unlink()


def _verify_rollback_entry(roots: AutowarePromotionRoots, item: AutowarePromotionRollback) -> None:
    """Validate one rollback entry without changing the filesystem."""

    target = roots.resolve_relative(item.path)
    _assert_safe_path(roots.workspace_root, target, allow_missing=True)
    current = sha256_path(target)
    if current != item.after_sha256:
        raise AutowarePromotionError(
            f"rollback refused because target changed after apply: {item.path}"
        )
    if item.before_exists:
        backup = roots.resolve_relative(item.backup_path)
        _assert_safe_path(roots.workspace_root, backup, allow_missing=False)
        if not backup.exists() or not backup.is_file():
            raise AutowarePromotionError(f"rollback backup is missing: {item.backup_path}")
        if sha256_path(backup) != item.backup_sha256:
            raise AutowarePromotionError(f"rollback backup digest mismatch: {item.backup_path}")


def _verify_restored_entry(roots: AutowarePromotionRoots, item: AutowarePromotionRollback) -> None:
    target = roots.resolve_relative(item.path)
    _assert_safe_path(roots.workspace_root, target, allow_missing=True)
    observed = sha256_path(target)
    expected = item.backup_sha256 if item.before_exists else None
    if observed != expected:
        raise AutowarePromotionError(
            f"rollback did not restore the exact target bytes: {item.path}"
        )


def _atomic_replace_text(path: Path, text: str) -> None:
    _atomic_replace_bytes(path, text.encode("utf-8"))


def _atomic_replace_bytes(path: Path, data: bytes) -> None:
    if not path.parent.exists() or not path.parent.is_dir():
        raise AutowarePromotionError(f"atomic target parent does not exist: {path.parent}")
    handle: int | None = None
    temporary: Path | None = None
    try:
        handle, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(handle, "wb") as stream:
            handle = None
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if handle is not None:
            os.close(handle)
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink()


def _mkdir_safe(root: Path, target: Path) -> None:
    """Create a directory tree while rejecting links/reparse points."""

    root = root.resolve()
    target_text = target
    _assert_safe_path(root, target_text, allow_missing=True)
    try:
        relative = target_text.relative_to(root)
    except ValueError as exc:
        raise AutowarePromotionError(f"directory is outside workspace_root: {target}") from exc
    current = root
    for component in relative.parts:
        current = current / component
        if current.exists() or current.is_symlink():
            _assert_safe_path(root, current, allow_missing=False)
            if not current.is_dir():
                raise AutowarePromotionError(
                    f"promotion directory component is not a directory: {current}"
                )
            continue
        try:
            current.mkdir()
        except FileExistsError:
            # A concurrent creator is acceptable only if it created a real,
            # in-scope directory rather than a link.
            _assert_safe_path(root, current, allow_missing=False)
        _assert_safe_path(root, current, allow_missing=False)


def _validate_artifact_output_path(artifact: AutowarePromotionArtifact, output_path: Path) -> None:
    """Prevent artifact persistence from overwriting a managed target file."""

    _assert_no_link_components(output_path.parent)
    resolved = output_path.expanduser().resolve(strict=False)
    for patch in artifact.generated_patches:
        target = artifact.roots.resolve_relative(patch.path)
        if resolved == target:
            raise AutowarePromotionError(
                f"promotion artifact output would overwrite managed target: {output_path}"
            )


def _verify_applied_outputs(
    roots: AutowarePromotionRoots, inventory: Sequence[AutowarePromotionRollback]
) -> None:
    for item in inventory:
        target = roots.resolve_relative(item.path)
        _assert_safe_path(roots.workspace_root, target, allow_missing=False)
        if not target.is_file() or sha256_path(target) != item.after_sha256:
            raise AutowarePromotionError(
                f"applied target digest mismatch after replacement: {item.path}"
            )


def _cleanup_created_backups(root: Path, backup_root: Path, created: Sequence[Path]) -> None:
    """Remove only files created by this failed apply, then its empty root."""

    for path in reversed(created):
        _assert_safe_path(root, path, allow_missing=False)
        if not path.is_file():
            raise AutowarePromotionError(f"backup cleanup target is not a file: {path}")
        path.unlink()
    _assert_safe_path(root, backup_root, allow_missing=False)
    backup_root.rmdir()


def _assert_safe_path(root: Path, path: Path, *, allow_missing: bool) -> None:
    """Reject symlink/reparse components inside a promotion scope.

    ``Path.resolve`` alone is insufficient on Windows: a junction or a
    symlink can change its target between resolution and an operation.  The
    explicit component walk is conservative and also protects backup files.
    """

    root_resolved = root.resolve()
    path_resolved = path.resolve(strict=False)
    if not _is_relative_to(path_resolved, root_resolved):
        raise AutowarePromotionError(f"path escapes workspace_root: {path}")
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise AutowarePromotionError(f"path is outside workspace_root: {path}") from exc
    current = root
    for component in relative.parts:
        current = current / component
        if not current.exists() and not current.is_symlink():
            if allow_missing:
                break
            raise AutowarePromotionError(f"path does not exist: {path}")
        if current.is_symlink():
            raise AutowarePromotionError(
                f"symlink path is not permitted in promotion scope: {path}"
            )
        try:
            attributes = current.stat().st_file_attributes
        except (AttributeError, OSError):
            attributes = 0
        if attributes & 0x400:
            raise AutowarePromotionError(
                f"reparse path is not permitted in promotion scope: {path}"
            )


def _serialize_yaml(payload: Mapping[str, Any]) -> str:
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)


def _read_text_exact(path: Path) -> str:
    """Read text without platform newline translation for byte-level digests."""

    with path.open("r", encoding="utf-8", newline="") as stream:
        return stream.read()


def _manifest_digest(files: Sequence[AutowarePromotionFile]) -> str:
    payload = [
        item.model_dump(mode="json", exclude_none=False)
        for item in sorted(files, key=lambda item: item.path)
    ]
    return _canonical_sha256(payload)


def _promotion_id(
    vehicle_id: str, sensor_kit_id: str, candidate_sha: str, baseline_sha: str
) -> str:
    return (
        "promotion-"
        + _canonical_sha256(
            {
                "vehicle_id": vehicle_id,
                "sensor_kit_id": sensor_kit_id,
                "candidate": candidate_sha,
                "baseline": baseline_sha,
            }
        )[:20]
    )


def _koide_decision(pilot: KoidePilotArtifact) -> PromotionDecision:
    if pilot.status == "PASS" and pilot.adoption_decision == "ADOPT":
        return "ADOPT"
    if pilot.status == "WARN":
        return "ADOPT_WITH_WARNING"
    if pilot.status in {"BLOCKED", "INCONCLUSIVE"} or pilot.adoption_decision == "BLOCKED":
        return "BLOCKED"
    return "DO_NOT_ADOPT"


def _status_from_diagnostics(diagnostics: Sequence[AutowarePromotionDiagnostic]) -> PromotionStatus:
    if any(item.severity == "error" for item in diagnostics):
        return "FAIL"
    if any(item.severity == "warning" for item in diagnostics):
        return "WARN"
    return "PASS"


def _error(code: str, message: str, path: str | None = None) -> AutowarePromotionDiagnostic:
    return AutowarePromotionDiagnostic(code=code, severity="error", message=message, path=path)


def _xml_vector(value: str | None, length: int) -> list[float] | None:
    if value is None:
        return None
    parts = value.split()
    if len(parts) != length or any("${" in part or "$" in part for part in parts):
        return None
    try:
        values = [float(part) for part in parts]
    except ValueError:
        return None
    return values if all(math.isfinite(item) for item in values) else None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_relative_path_text(value: str, *, field_name: str) -> None:
    """Reject traversal in either native path syntax.

    Artifacts can be generated on Windows and verified on Linux (and vice
    versa).  Checking only ``Path.parts`` therefore leaves a Windows
    ``..\\outside`` traversal invisible on POSIX.  Both grammars are checked
    and drive/UNC roots are rejected regardless of the host platform.
    """

    if not isinstance(value, str) or not value.strip():
        raise AutowarePromotionError(f"{field_name} must be a non-empty relative path")
    posix = PurePosixPath(value.replace("\\", "/"))
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part == ".." for part in posix.parts)
        or any(part == ".." for part in windows.parts)
    ):
        raise AutowarePromotionError(
            f"{field_name} must not be absolute or traverse parent directories"
        )


def _assert_no_link_components(path: Path) -> None:
    """Reject symlink/junction components before trusting an external file."""

    absolute = path.expanduser().absolute()
    ancestors = [absolute, *absolute.parents]
    for current in reversed(ancestors):
        if current.is_symlink():
            raise AutowarePromotionError(
                f"symlink paths are not permitted for promotion evidence: {path}"
            )
        try:
            attributes = current.stat().st_file_attributes
        except (AttributeError, OSError):
            attributes = 0
        if attributes & 0x400:
            raise AutowarePromotionError(
                f"reparse paths are not permitted for promotion evidence: {path}"
            )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _rpy_from_quaternion(quaternion: Sequence[float]) -> tuple[float, float, float]:
    x, y, z, w = (float(value) for value in quaternion)
    sin_roll_cos_pitch = 2.0 * (w * x + y * z)
    cos_roll_cos_pitch = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sin_roll_cos_pitch, cos_roll_cos_pitch)
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sin_pitch)))
    sin_yaw_cos_pitch = 2.0 * (w * z + x * y)
    cos_yaw_cos_pitch = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch)
    return roll, pitch, yaw


__all__ = [
    "AUTOWARE_PROMOTION_PROTOCOL_ID",
    "AUTOWARE_PROMOTION_SCHEMA_VERSION",
    "AutowareFrameEdge",
    "AutowarePackageInspection",
    "AutowarePromotionApplication",
    "AutowarePromotionArtifact",
    "AutowarePromotionCandidate",
    "AutowarePromotionDiagnostic",
    "AutowarePromotionError",
    "AutowarePromotionFile",
    "AutowarePromotionPatch",
    "AutowarePromotionPolicy",
    "AutowarePromotionProvenance",
    "AutowarePromotionRollback",
    "AutowarePromotionRoots",
    "apply_autoware_promotion",
    "autoware_package_inspection_json_schema",
    "autoware_promotion_json_schema",
    "build_autoware_promotion_plan",
    "inspect_autoware_package",
    "load_autoware_package_inspection",
    "load_autoware_promotion",
    "rollback_autoware_promotion",
    "verify_autoware_promotion",
]
