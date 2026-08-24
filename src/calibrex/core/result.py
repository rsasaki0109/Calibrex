"""Typed calibration result schema."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
)

from calibrex.core.exceptions import ResultError
from calibrex.core.geometry import SE3
from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.solid_state import SolidStateLidarCalibrationContext

RESULT_SCHEMA_VERSION: Literal["slac.result/v0.1"] = "slac.result/v0.1"
RESULT_PROVENANCE_VERSION: Literal["slac.result.provenance/v0.1"] = (
    "slac.result.provenance/v0.1"
)
Grade = Literal["pass", "warn", "fail"]
EstimateProducer = Literal[
    "slac_native",
    "external_tool",
    "human",
    "dataset_provider",
    "factory",
    "unknown",
]
EstimateExecutionMode = Literal[
    "offline_batch",
    "sliding_window",
    "online_stream",
    "manual",
    "imported",
    "dataset_reference",
    "unknown",
]
EstimateRole = Literal[
    "initial",
    "candidate",
    "selected_reference",
    "output",
    "comparison_baseline",
]
EstimateEvidenceLevel = Literal[
    "synthetic_truth",
    "independently_measured",
    "dataset_provided",
    "factory_provided",
    "algorithmically_refined",
    "imported_without_documented_derivation",
    "unknown",
]


class StrictModel(BaseModel):
    """Base model with stable, explicit fields."""

    model_config = ConfigDict(extra="forbid")


class ResultProvenance(StrictModel):
    """Versioned generation metadata for a calibration result.

    Result provenance deliberately allows additional pipeline-specific fields.
    The fields below are the stable minimum required to identify how a new
    result was produced.  Legacy v0.1 files may contain an unstructured map;
    those files remain readable through :func:`load_result`, but are not
    production-admissible until explicitly migrated.
    """

    model_config = ConfigDict(extra="allow")

    provenance_version: Literal["slac.result.provenance/v0.1"] = (
        RESULT_PROVENANCE_VERSION
    )
    producer: str = Field(min_length=1, pattern=r"\S")
    tool_name: str = Field(min_length=1, pattern=r"\S")
    tool_version: str = Field(min_length=1, pattern=r"\S")
    generated_at: str = Field(min_length=1, pattern=r"\S")
    git_commit: str | None = None
    git_commit_unavailable_reason: str | None = None
    command: str | list[str]
    config_sha256: str | None = Field(pattern=r"^[0-9a-fA-F]{64}$")
    input_sha256: str | None = Field(pattern=r"^[0-9a-fA-F]{64}$")
    config_digest_unavailable_reason: str | None = None
    input_digest_unavailable_reason: str | None = None

    @field_validator(
        "producer",
        "tool_name",
        "tool_version",
        "generated_at",
        mode="before",
    )
    @classmethod
    def require_non_empty_text(cls, value: object) -> str:
        """Reject blank identity and timestamp fields."""

        if not isinstance(value, str) or not value.strip():
            raise ValueError("must be a non-empty string")
        return value

    @field_validator("command", mode="before")
    @classmethod
    def require_command(cls, value: object) -> str | list[str]:
        """Reject an omitted or empty recorded command."""

        if isinstance(value, str):
            if not value.strip():
                raise ValueError("must be a non-empty command")
            return value
        if isinstance(value, list) and value and all(
            isinstance(item, str) and item.strip() for item in value
        ):
            return value
        raise ValueError("must be a non-empty string or argv list")

    @field_validator(
        "git_commit",
        "git_commit_unavailable_reason",
        "config_digest_unavailable_reason",
        "input_digest_unavailable_reason",
        mode="before",
    )
    @classmethod
    def require_non_empty_optional_text(cls, value: object) -> str | None:
        """Reject blank optional identity and unavailable-reason fields."""

        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must be null or a non-empty string")
        return value

    @model_validator(mode="after")
    def require_commit_or_reason(self) -> ResultProvenance:
        """Require an honest reason when the repository commit is unavailable."""

        if not self.git_commit and not self.git_commit_unavailable_reason:
            raise ValueError(
                "git_commit or git_commit_unavailable_reason is required for result provenance"
            )
        if not self.config_sha256 and not self.config_digest_unavailable_reason:
            raise ValueError(
                "config_sha256 or config_digest_unavailable_reason is required "
                "for result provenance"
            )
        if not self.input_sha256 and not self.input_digest_unavailable_reason:
            raise ValueError(
                "input_sha256 or input_digest_unavailable_reason is required for result provenance"
            )
        return self


ResultProvenanceStatus = Literal["unvalidated", "production", "legacy"]


def result_provenance_issues(value: Mapping[str, Any] | object) -> list[str]:
    """Return production-provenance validation issues without mutating ``value``.

    This helper is intentionally separate from ``load_result``.  Loading a
    historical artifact must not silently add ambient git, clock, or input
    metadata to its payload.
    """

    if not isinstance(value, Mapping):
        return ["run.provenance must be a mapping"]
    if not value:
        return ["run.provenance is empty; legacy results are not production-admissible"]
    if "provenance_version" not in value:
        return [
            "run.provenance has no provenance_version; this is a legacy result "
            "and is not production-admissible"
        ]
    try:
        ResultProvenance.model_validate(value)
    except ValidationError as exc:
        return [
            f"run.provenance.{'.'.join(str(part) for part in error['loc'])}: "
            f"{error['msg']}"
            for error in exc.errors()
        ]
    return []


def is_legacy_result_provenance(value: Mapping[str, Any] | object) -> bool:
    """Return whether a result payload is eligible for the read-only legacy path."""

    return isinstance(value, Mapping) and "provenance_version" not in value


def validate_result_provenance(value: Mapping[str, Any]) -> ResultProvenance:
    """Validate and return the typed provenance contract for a new result."""

    return ResultProvenance.model_validate(value)


def build_result_provenance(
    *,
    producer: str,
    tool_name: str,
    tool_version: str,
    command: str | Sequence[str],
    config_sha256: str | None = None,
    input_sha256: str | None = None,
    config_paths: Sequence[str | Path] = (),
    input_paths: Sequence[str | Path] = (),
    git_commit: str | None = None,
    git_commit_unavailable_reason: str | None = None,
    generated_at: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build explicit, validated result-level provenance.

    ``config_paths`` and ``input_paths`` are convenience inputs for callers
    that have not already computed SHA-256 values.  The helper computes no
    metadata when loading a result; it is intended for an active generator
    boundary only.  A missing commit or digest is represented by an explicit
    reason rather than an invented value.
    """

    from calibrex.core.provenance import git_commit as current_git_commit
    from calibrex.core.provenance import sha256_path

    if config_sha256 is None:
        config_sha256 = _digest_paths(config_paths, sha256_path)
    if input_sha256 is None:
        input_sha256 = _digest_paths(input_paths, sha256_path)
    if git_commit is None and git_commit_unavailable_reason is None:
        git_commit = current_git_commit()
    if git_commit is None and git_commit_unavailable_reason is None:
        git_commit_unavailable_reason = "git commit is unavailable in the execution environment"
    config_reason = (
        None
        if config_sha256
        else "configuration digest was unavailable at generation time"
    )
    input_reason = (
        None if input_sha256 else "input digest was unavailable at generation time"
    )
    normalized_command: str | list[str] = (
        command if isinstance(command, str) else list(command)
    )
    payload: dict[str, Any] = {
        "provenance_version": RESULT_PROVENANCE_VERSION,
        "producer": producer,
        "tool_name": tool_name,
        "tool_version": tool_version,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_commit_unavailable_reason": git_commit_unavailable_reason,
        "command": normalized_command,
        "config_sha256": config_sha256,
        "input_sha256": input_sha256,
        "config_digest_unavailable_reason": config_reason,
        "input_digest_unavailable_reason": input_reason,
    }
    if extra:
        collisions = set(payload).intersection(extra)
        if collisions:
            names = ", ".join(sorted(collisions))
            raise ValueError(f"extra provenance cannot override required keys: {names}")
        payload.update(extra)
    validate_result_provenance(payload)
    return payload


def _digest_paths(
    paths: Sequence[str | Path],
    digest_path: Any,
) -> str | None:
    """Return a deterministic digest for one or more declared paths."""

    declared = [Path(path) for path in paths]
    if not declared:
        return None
    if len(declared) == 1:
        return cast(str | None, digest_path(declared[0]))
    import hashlib

    digest = hashlib.sha256()
    for path in sorted(declared, key=lambda item: str(item)):
        value = digest_path(path)
        if value is None:
            return None
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


class RunInfo(StrictModel):
    id: str
    slac_version: str
    git_commit: str | None = None
    config_sha256: str | None = None
    dataset_sha256: str | None = None
    status: Literal["success", "warning", "failed", "dry_run"] = "success"
    domain: str = "robotics"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    provenance: dict[str, Any] = Field(default_factory=dict)


class CovarianceInfo(StrictModel):
    order: list[str]
    matrix: list[list[float]]


class TransformQuality(StrictModel):
    grade: Grade = "warn"
    std_translation_m: list[float] = Field(default_factory=list)
    std_rotation_deg: list[float] = Field(default_factory=list)


class TransformEstimateProvenance(StrictModel):
    producer: EstimateProducer = "unknown"
    execution_mode: EstimateExecutionMode = "unknown"
    role_in_comparison: EstimateRole | None = None
    evidence_level: EstimateEvidenceLevel = "unknown"
    source: str | None = None
    source_path: str | None = None
    tool_name: str | None = None
    tool_version: str | None = None
    source_commit: str | None = None
    license_spdx: str | None = None
    adapter_version: str | None = None
    command: str | None = None
    notes: list[str] = Field(default_factory=list)


class TransformResult(StrictModel):
    convention: Literal["T_parent_child"] = "T_parent_child"
    parent: str
    child: str
    translation_m: list[float] = Field(min_length=3, max_length=3)
    rotation_quat_xyzw: list[float] = Field(min_length=4, max_length=4)
    covariance: CovarianceInfo | None = None
    quality: TransformQuality = Field(default_factory=TransformQuality)
    estimate_id: str | None = None
    provenance: TransformEstimateProvenance = Field(
        default_factory=TransformEstimateProvenance
    )

    def as_se3(self) -> SE3:
        """Return this transform as an `SE3` value."""

        return SE3.from_lists(self.translation_m, self.rotation_quat_xyzw)


ExtrinsicEstimate = TransformResult


class TimeOffsetQuality(StrictModel):
    grade: Grade = "warn"


class TimeOffsetResult(StrictModel):
    seconds: float
    std_seconds: float | None = None
    quality: TimeOffsetQuality = Field(default_factory=TimeOffsetQuality)


class MetricResult(StrictModel):
    train: float | None = None
    holdout: float | None = None
    value: float | None = None
    grade: Grade = "warn"
    unit: str | None = None
    reason: str | None = None


class ObservabilityResult(StrictModel):
    rank: int | None = None
    condition_number: float | None = None
    weak_directions: list[str] = Field(default_factory=list)
    grade: Grade = "warn"


class DegeneracyResult(StrictModel):
    grade: Grade = "warn"
    reason: str | None = None


class QualitySummary(StrictModel):
    grade: Grade = "warn"
    blocking_failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    recommendation: list[str] = Field(default_factory=list)


class ArtifactSet(StrictModel):
    html_report: str | None = None
    trajectory_plot: str | None = None
    rig_3d_viewer: str | None = None
    camera_lidar_overlay: str | None = None
    residual_histogram: str | None = None


class ExportSet(StrictModel):
    ros_tf: str | None = None
    urdf: str | None = None
    kalibr_yaml: str | None = None
    autoware: str | None = None


class FrameGraphSnapshot(StrictModel):
    root: str
    convention: Literal["T_parent_child"] = "T_parent_child"
    frames: dict[str, str | None]


class CalibrationResult(StrictModel):
    schema_version: Literal["slac.result/v0.1"] = RESULT_SCHEMA_VERSION
    run: RunInfo
    frame_graph: FrameGraphSnapshot
    solid_state: SolidStateLidarCalibrationContext | None = Field(
        default=None,
        description=(
            "optional solid-state LiDAR acquisition, intrinsic-calibration, "
            "temperature, and timing context"
        ),
    )
    candidate_extrinsics: dict[str, TransformResult] = Field(default_factory=dict)
    reference_extrinsics: dict[str, TransformResult] = Field(default_factory=dict)
    transforms: dict[str, TransformResult] = Field(default_factory=dict)
    time_offsets: dict[str, TimeOffsetResult] = Field(default_factory=dict)
    metrics: dict[str, MetricResult] = Field(default_factory=dict)
    observability: ObservabilityResult = Field(default_factory=ObservabilityResult)
    degeneracy: DegeneracyResult = Field(default_factory=DegeneracyResult)
    quality: QualitySummary = Field(default_factory=QualitySummary)
    artifacts: ArtifactSet = Field(default_factory=ArtifactSet)
    export: ExportSet = Field(default_factory=ExportSet)
    _provenance_status: ResultProvenanceStatus = PrivateAttr(default="unvalidated")

    @property
    def provenance_status(self) -> ResultProvenanceStatus:
        """Return the in-memory provenance/admissibility status."""

        if self._provenance_status == "unvalidated":
            return "production" if not result_provenance_issues(self.run.provenance) else "legacy"
        return self._provenance_status

    @property
    def production_valid(self) -> bool:
        """Return whether this result carries the complete new provenance contract."""

        return not result_provenance_issues(self.run.provenance)

    def save(self, path: str | Path, *, allow_legacy: bool | None = None) -> None:
        """Save this result as YAML or JSON.

        New or in-memory results are fail-closed at this boundary. A result
        loaded through the explicit legacy compatibility path may be written
        back unchanged; it is never silently upgraded with ambient metadata.
        Callers that intentionally persist a hand-authored legacy fixture may
        pass ``allow_legacy=True`` explicitly.
        """

        if allow_legacy is None:
            # A loaded legacy result is the one compatibility exception. New
            # in-memory results must opt in explicitly with ``allow_legacy``.
            allow_legacy = self._provenance_status == "legacy"
        save_result(self, path, allow_legacy=allow_legacy)


def save_result(
    result: CalibrationResult,
    path: str | Path,
    *,
    allow_legacy: bool = False,
) -> None:
    """Persist a result, enforcing provenance unless legacy is explicit."""

    issues = result_provenance_issues(result.run.provenance)
    if issues:
        if not allow_legacy or not is_legacy_result_provenance(result.run.provenance):
            detail = "; ".join(issues)
            raise ResultError(
                "result save blocked: production provenance is incomplete; "
                f"{detail}"
            )
        result._provenance_status = "legacy"
    else:
        result._provenance_status = "production"
    write_mapping(Path(path), result.model_dump(mode="json", exclude_none=True))


def load_result(
    path: str | Path,
    *,
    allow_legacy: bool = True,
) -> CalibrationResult:
    """Load and validate a Calibrex result without fabricating provenance.

    The default compatibility path accepts historical v0.1 payloads whose
    provenance lacks the version marker. Such a result is marked ``legacy``
    in memory and remains non-production-admissible. A payload that declares
    the new marker but fails its required fields is rejected even when legacy
    loading is enabled.
    """

    result_path = Path(path)
    try:
        result = CalibrationResult.model_validate(read_mapping(result_path))
        issues = result_provenance_issues(result.run.provenance)
        if issues:
            if not allow_legacy or not is_legacy_result_provenance(result.run.provenance):
                detail = "; ".join(issues)
                raise ResultError(f"result provenance is not valid: {detail}")
            result._provenance_status = "legacy"
        else:
            result._provenance_status = "production"
        return result
    except Exception as exc:
        if isinstance(exc, ResultError):
            raise
        raise ResultError(f"invalid result {result_path}: {exc}") from exc


def result_json_schema() -> dict[str, Any]:
    """Return the JSON schema for result files."""

    schema = CalibrationResult.model_json_schema()
    definitions = schema.setdefault("$defs", {})
    definitions["ResultProvenance"] = ResultProvenance.model_json_schema()
    provenance_definition = definitions["ResultProvenance"]
    if isinstance(provenance_definition, dict):
        required = provenance_definition.setdefault("required", [])
        if isinstance(required, list) and "provenance_version" not in required:
            required.insert(0, "provenance_version")
        properties = provenance_definition.get("properties")
        if isinstance(properties, dict):
            command_schema = properties.get("command")
            if isinstance(command_schema, dict):
                variants = command_schema.get("anyOf")
                if isinstance(variants, list):
                    for variant in variants:
                        if not isinstance(variant, dict):
                            continue
                        if variant.get("type") == "string":
                            variant.update({"minLength": 1, "pattern": r"\S"})
                        elif variant.get("type") == "array":
                            variant["minItems"] = 1
                            items = variant.get("items")
                            if isinstance(items, dict):
                                items.update({"minLength": 1, "pattern": r"\S"})
            for field_name in (
                "git_commit",
                "git_commit_unavailable_reason",
                "config_digest_unavailable_reason",
                "input_digest_unavailable_reason",
            ):
                field_schema = properties.get(field_name)
                if not isinstance(field_schema, dict):
                    continue
                variants = field_schema.get("anyOf")
                if not isinstance(variants, list):
                    continue
                for variant in variants:
                    if isinstance(variant, dict) and variant.get("type") == "string":
                        variant.update({"minLength": 1, "pattern": r"\S"})
        provenance_definition["allOf"] = [
            {
                "anyOf": [
                    {
                        "required": ["git_commit"],
                        "properties": {
                            "git_commit": {
                                "type": "string",
                                "minLength": 1,
                                "pattern": r"\S",
                            }
                        },
                    },
                    {
                        "required": ["git_commit_unavailable_reason"],
                        "properties": {
                            "git_commit_unavailable_reason": {
                                "type": "string",
                                "minLength": 1,
                                "pattern": r"\S",
                            }
                        },
                    },
                ]
            },
            {
                "anyOf": [
                    {
                        "required": ["config_sha256"],
                        "properties": {
                            "config_sha256": {
                                "type": "string",
                                "pattern": r"^[0-9a-fA-F]{64}$",
                            }
                        },
                    },
                    {
                        "required": ["config_digest_unavailable_reason"],
                        "properties": {
                            "config_digest_unavailable_reason": {
                                "type": "string",
                                "minLength": 1,
                                "pattern": r"\S",
                            }
                        },
                    },
                ]
            },
            {
                "anyOf": [
                    {
                        "required": ["input_sha256"],
                        "properties": {
                            "input_sha256": {
                                "type": "string",
                                "pattern": r"^[0-9a-fA-F]{64}$",
                            }
                        },
                    },
                    {
                        "required": ["input_digest_unavailable_reason"],
                        "properties": {
                            "input_digest_unavailable_reason": {
                                "type": "string",
                                "minLength": 1,
                                "pattern": r"\S",
                            }
                        },
                    },
                ]
            },
        ]
    run_schema = definitions.get("RunInfo")
    if isinstance(run_schema, dict):
        properties = run_schema.get("properties")
        if isinstance(properties, dict) and isinstance(properties.get("provenance"), dict):
            properties["provenance"] = {
                "description": (
                    "Versioned provenance for new results. Legacy v0.1 maps are "
                    "readable but are not production-admissible."
                ),
                "oneOf": [
                    {"$ref": "#/$defs/ResultProvenance"},
                    {
                        "type": "object",
                        "minProperties": 1,
                        "not": {"required": ["provenance_version"]},
                        "additionalProperties": True,
                    },
                ],
            }
    return schema
