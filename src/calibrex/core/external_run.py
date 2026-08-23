"""Schema-versioned contract for external calibration tool execution and import."""

from __future__ import annotations

import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field, model_validator

from calibrex.core.geometry import SE3
from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.result import (
    MetricResult,
    StrictModel,
    TransformEstimateProvenance,
)

if TYPE_CHECKING:
    from calibrex.solvers.base import SolverAdapterResult

EXTERNAL_RUN_SCHEMA_VERSION: Literal["slac.external_calibration_run/v0.1"] = (
    "slac.external_calibration_run/v0.1"
)
ExternalExecutionMode = Literal["subprocess", "container", "precomputed", "imported"]
ExternalLicenseBoundary = Literal["in_process", "subprocess", "container", "imported"]
ExternalRunStatus = Literal[
    "success",
    "not_executed",
    "unavailable",
    "failed",
    "timeout",
    "invalid_output",
    "digest_mismatch",
]

ExternalStageStatus = Literal[
    "pending",
    "running",
    "success",
    "failed",
    "timeout",
    "unavailable",
    "skipped",
]


class ExternalArtifactDigest(StrictModel):
    """Digest and role of one external-run input or output."""

    role: Literal["input", "output"]
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    media_type: str | None = None


class ExternalToolIdentity(StrictModel):
    """External tool identity and legal boundary declaration."""

    name: str
    version: str | None = None
    source_repository: str | None = None
    source_commit: str | None = None
    license_spdx: str | None = None
    license_boundary: ExternalLicenseBoundary

    @model_validator(mode="after")
    def keep_copyleft_out_of_process(self) -> ExternalToolIdentity:
        """Reject an in-process GPL boundary."""

        license_id = (self.license_spdx or "").upper()
        if ("GPL-" in license_id or license_id.startswith("GPL")) and (
            self.license_boundary == "in_process"
        ):
            msg = "GPL external tools must use a subprocess, container, or import boundary"
            raise ValueError(msg)
        return self


class ExternalStageExecution(StrictModel):
    """Bounded facts for one stage of a multi-stage external workflow.

    Stage stdout/stderr are intentionally retained only as a short tail and a
    SHA-256 digest.  This keeps an evidence artifact useful for debugging while
    avoiding unbounded or potentially sensitive process logs.
    """

    name: str
    command: list[str] = Field(default_factory=list)
    status: ExternalStageStatus
    return_code: int | None = None
    timed_out: bool = False
    duration_seconds: float | None = Field(default=None, ge=0.0)
    stdout_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    stderr_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    stdout_tail: str | None = Field(default=None, max_length=4000)
    stderr_tail: str | None = Field(default=None, max_length=4000)
    error: str | None = None


class ExternalExecution(StrictModel):
    """Bounded execution facts without unbounded external process output."""

    mode: ExternalExecutionMode
    command: list[str] = Field(default_factory=list)
    working_directory: str | None = None
    container_digest: str | None = Field(
        default=None,
        pattern=r"^(?:[^@\s]+@)?sha256:[0-9a-f]{64}$",
    )
    timeout_seconds: float | None = Field(default=None, gt=0.0)
    attempted: bool = False
    return_code: int | None = None
    timed_out: bool = False
    duration_seconds: float | None = Field(default=None, ge=0.0)
    stdout_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    stderr_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    stdout_tail: str | None = Field(default=None, max_length=4000)
    stderr_tail: str | None = Field(default=None, max_length=4000)
    error: str | None = None
    stages: list[ExternalStageExecution] = Field(default_factory=list)
    network_mode: str | None = None
    cpu_limit: float | None = Field(default=None, gt=0.0)
    memory_limit: str | None = None
    input_mounts: list[str] = Field(default_factory=list)
    output_mount: str | None = None

    @model_validator(mode="after")
    def require_container_digest(self) -> ExternalExecution:
        """Require an immutable image identity for container execution."""

        if self.mode == "container" and not self.container_digest:
            msg = "container execution requires container_digest"
            raise ValueError(msg)
        return self


class ExternalRunDigests(StrictModel):
    """Content identities needed to reproduce an external calibration run."""

    tool_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    container_digest: str | None = Field(
        default=None,
        pattern=r"^(?:[^@\s]+@)?sha256:[0-9a-f]{64}$",
    )
    config_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    input_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    output_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class ExternalDataIsolation(StrictModel):
    """Declaration that Calibrex holdout data was isolated from external fitting."""

    declared: bool
    evidence: str | None = None
    training_data_ids_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    holdout_data_ids_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )


class ExternalTransformOutput(StrictModel):
    """One parsed transform emitted by an external tool."""

    convention: Literal["T_parent_child"] = "T_parent_child"
    parent: str
    child: str
    translation_m: list[float] = Field(min_length=3, max_length=3)
    rotation_quat_xyzw: list[float] = Field(min_length=4, max_length=4)

    def as_se3(self) -> SE3:
        """Return the transform as an ``SE3`` value."""

        return SE3.from_lists(self.translation_m, self.rotation_quat_xyzw)


class ExternalParsedOutputs(StrictModel):
    """Typed outputs parsed independently of external fitness metrics."""

    transforms: dict[str, ExternalTransformOutput] = Field(default_factory=dict)
    time_offsets_seconds: dict[str, float] = Field(default_factory=dict)
    intrinsics: dict[str, dict[str, float | str | list[float]]] = Field(
        default_factory=dict
    )
    external_metrics: dict[str, float | str | bool | None] = Field(default_factory=dict)
    external_metrics_comparable: Literal[False] = False


class ExternalRunProvenance(StrictModel):
    """Lineage of the adapter artifact itself."""

    producer: Literal["calibrex"] = "calibrex"
    calibrex_version: str
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_commit: str | None = None
    source_artifact: str | None = None
    source_artifact_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )


class ExternalCalibrationRunArtifact(StrictModel):
    """Generic, reusable record of an external calibration run or import."""

    schema_version: Literal["slac.external_calibration_run/v0.1"] = (
        EXTERNAL_RUN_SCHEMA_VERSION
    )
    run_id: str
    adapter_name: str
    adapter_version: str
    tool: ExternalToolIdentity
    execution: ExternalExecution
    artifacts: list[ExternalArtifactDigest] = Field(default_factory=list)
    digests: ExternalRunDigests = Field(default_factory=ExternalRunDigests)
    frame_convention: str
    time_convention: str
    train_data_isolation: ExternalDataIsolation
    status: ExternalRunStatus
    warnings: list[str] = Field(default_factory=list)
    parsed_outputs: ExternalParsedOutputs = Field(default_factory=ExternalParsedOutputs)
    provenance: ExternalRunProvenance

    @model_validator(mode="after")
    def make_boundary_risks_explicit(self) -> ExternalCalibrationRunArtifact:
        """Materialize warnings for unknown licenses and undeclared isolation."""

        if self.tool.license_spdx is None and not any(
            "license" in warning.lower() for warning in self.warnings
        ):
            self.warnings.append("external tool license is unknown")
        if not self.train_data_isolation.declared and not any(
            "isolation" in warning.lower() for warning in self.warnings
        ):
            self.warnings.append("external training-data isolation is not declared")
        return self

    def save(self, path: str | Path) -> None:
        """Write the external-run artifact as YAML or JSON."""

        write_mapping(
            Path(path),
            self.model_dump(mode="json", exclude_none=True),
        )

    def to_solver_adapter_result(self) -> SolverAdapterResult:
        """Convert to the legacy adapter result without losing the typed artifact."""

        from calibrex.solvers.base import SolverAdapterResult

        transforms = {
            name: transform.as_se3()
            for name, transform in self.parsed_outputs.transforms.items()
        }
        available = self.status not in {"unavailable", "not_executed"}
        return SolverAdapterResult(
            backend=self.adapter_name,
            available=available,
            status=self.status,
            metrics={
                "external_run_available": MetricResult(
                    value=1.0 if available else 0.0,
                    grade="pass" if available else "warn",
                    reason=f"external run status: {self.status}",
                ),
                "external_run_success": MetricResult(
                    value=1.0 if self.status == "success" else 0.0,
                    grade="pass" if self.status == "success" else "fail",
                    reason=f"external run status: {self.status}",
                ),
            },
            transforms=transforms,
            provenance={
                "external_calibration_run": self.model_dump(mode="json", exclude_none=True),
                "external_calibration_run_schema_version": self.schema_version,
            },
            warnings=list(self.warnings),
            external_run=self,
        )

    def transform_provenance(self, transform_name: str) -> TransformEstimateProvenance:
        """Build result-level provenance for one parsed external transform."""

        output_digests = [
            artifact
            for artifact in self.artifacts
            if artifact.role == "output"
        ]
        notes = [
            f"external_run_id={self.run_id}",
            f"external_run_schema={self.schema_version}",
            "external metrics are non-comparable; Calibrex evidence is evaluated independently",
        ]
        if output_digests:
            notes.append(f"output_sha256={output_digests[0].sha256}")
        return TransformEstimateProvenance(
            producer="external_tool",
            execution_mode=(
                "imported"
                if self.execution.mode in {"imported", "precomputed"}
                else "offline_batch"
            ),
            role_in_comparison="output",
            evidence_level="algorithmically_refined",
            source=self.tool.source_repository or "external_calibration_run",
            source_path=(
                output_digests[0].path
                if output_digests
                else self.provenance.source_artifact
            ),
            tool_name=self.tool.name,
            tool_version=self.tool.version,
            source_commit=self.tool.source_commit,
            license_spdx=self.tool.license_spdx,
            adapter_version=self.adapter_version,
            command=shlex.join(self.execution.command) if self.execution.command else None,
            notes=[f"transform={transform_name}", *notes],
        )


def external_run_json_schema() -> dict[str, object]:
    """Return the generic external-run JSON Schema."""

    return ExternalCalibrationRunArtifact.model_json_schema()


def load_external_run(path: str | Path) -> ExternalCalibrationRunArtifact:
    """Load and validate an external-run artifact."""

    return ExternalCalibrationRunArtifact.model_validate(read_mapping(Path(path)))
