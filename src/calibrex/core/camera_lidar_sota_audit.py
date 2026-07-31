"""Schema-valid Camera--LiDAR SOTA claim audit protocol and result."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.result import StrictModel

CAMERA_LIDAR_SOTA_AUDIT_PROTOCOL_SCHEMA_VERSION: Literal[
    "slac.camera_lidar_sota_audit_protocol/v0.1"
] = "slac.camera_lidar_sota_audit_protocol/v0.1"
CAMERA_LIDAR_SOTA_AUDIT_RESULT_SCHEMA_VERSION: Literal[
    "slac.camera_lidar_sota_audit_result/v0.1"
] = "slac.camera_lidar_sota_audit_result/v0.1"
CameraLidarSotaCategory = Literal[
    "training_free_targetless",
    "learned_targetless",
    "spatiotemporal_targetless",
    "production_target_based_reference",
]


class CameraLidarSotaAuditProvenance(StrictModel):
    """Generator/command lineage for a claim protocol or result."""

    generator: str
    generator_version: str
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    source_sha256: dict[str, str] = Field(default_factory=dict)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @model_validator(mode="after")
    def check_digests(self) -> CameraLidarSotaAuditProvenance:
        """Validate all declared SHA-256 values."""

        invalid = [
            name
            for name, digest in self.source_sha256.items()
            if len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ]
        if invalid:
            raise ValueError(
                "invalid audit provenance SHA-256: " + ", ".join(invalid)
            )
        return self


class CameraLidarClaimRequirement(StrictModel):
    """One frozen numerical or completion gate."""

    requirement_id: str
    phase: str
    description: str
    dataset_family: str | None = None
    independent_rig: bool = False
    rig_id: str | None = None
    evidence_path: str | None = None
    evidence_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    method_id: str | None = None
    reference_method_id: str | None = None
    metric: str | None = None
    statistic: Literal[
        "metric_mean",
        "metric_median",
        "metric_p90",
        "metric_p95",
        "metric_maximum",
        "failure_rate",
        "runtime_mean",
        "runtime_p95",
        "peak_memory_mean",
        "peak_memory_p95",
        "paired_improvement_ci95_low",
        "paired_improvement_ci95_high",
    ] | None = None
    comparison: Literal[
        "greater_equal",
        "greater_than",
        "less_equal",
        "less_than",
        "equal",
    ] | None = None
    threshold: float | None = None
    required: bool = True

    @model_validator(mode="after")
    def check_locator(self) -> CameraLidarClaimRequirement:
        """Require a complete locator when benchmark evidence is declared."""

        if self.threshold is not None and not math.isfinite(self.threshold):
            raise ValueError("audit threshold must be finite")
        locator = (self.method_id, self.metric, self.statistic)
        numeric_gate = (self.comparison, self.threshold)
        if self.evidence_path is None:
            if any(value is not None for value in (*locator, *numeric_gate)):
                raise ValueError(
                    "requirement without evidence cannot declare a locator/gate"
                )
        elif (
            self.evidence_sha256 is None
            or self.method_id is None
            or self.statistic is None
            or self.comparison is None
            or self.threshold is None
            or (
                self.statistic
                in {
                    "metric_mean",
                    "metric_median",
                    "metric_p90",
                    "metric_p95",
                    "metric_maximum",
                    "paired_improvement_ci95_low",
                    "paired_improvement_ci95_high",
                }
                and self.metric is None
            )
        ):
            raise ValueError(
                "benchmark evidence requires digest, method, statistic, and gate"
            )
        paired = self.statistic in {
            "paired_improvement_ci95_low",
            "paired_improvement_ci95_high",
        }
        if paired and self.reference_method_id is None:
            raise ValueError(
                "paired confidence requirements need reference_method_id"
            )
        if not paired and self.reference_method_id is not None:
            raise ValueError(
                "reference_method_id is only valid for paired confidence gates"
            )
        if self.independent_rig and self.rig_id is None:
            raise ValueError("independent rig requirements require rig_id")
        return self


class CameraLidarSotaAuditProtocol(StrictModel):
    """Frozen requirements for one explicitly scoped SOTA claim."""

    schema_version: Literal[
        "slac.camera_lidar_sota_audit_protocol/v0.1"
    ] = CAMERA_LIDAR_SOTA_AUDIT_PROTOCOL_SCHEMA_VERSION
    protocol_id: str
    declared_category: CameraLidarSotaCategory
    claim_text: str
    minimum_dataset_families: int = Field(default=2, ge=1)
    minimum_independent_rigs: int = Field(default=1, ge=1)
    requirements: list[CameraLidarClaimRequirement] = Field(min_length=1)
    provenance: CameraLidarSotaAuditProvenance

    @model_validator(mode="after")
    def check_unique_requirements(self) -> CameraLidarSotaAuditProtocol:
        """Require unique stable requirement IDs."""

        identifiers = [item.requirement_id for item in self.requirements]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("SOTA audit requirement IDs must be unique")
        return self

    def save(self, path: str | Path) -> None:
        """Save the frozen audit protocol."""

        _save(self, path)


class CameraLidarClaimRequirementResult(StrictModel):
    """Observed result for one frozen requirement."""

    requirement_id: str
    required: bool
    status: Literal["achieved", "contradicted", "incomplete", "missing"]
    observed_value: float | None = None
    threshold: float | None = None
    reason: str
    evidence_path: str | None = None
    evidence_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )


class CameraLidarSotaAuditResult(StrictModel):
    """Machine-readable support/refutation decision for a SOTA claim."""

    schema_version: Literal[
        "slac.camera_lidar_sota_audit_result/v0.1"
    ] = CAMERA_LIDAR_SOTA_AUDIT_RESULT_SCHEMA_VERSION
    audit_id: str
    protocol_id: str
    declared_category: CameraLidarSotaCategory
    claim_text: str
    verdict: Literal["supported", "refuted", "incomplete"]
    achieved_dataset_families: list[str]
    achieved_independent_rig_count: int = Field(ge=0)
    minimum_dataset_families: int = Field(ge=1)
    minimum_independent_rigs: int = Field(ge=1)
    requirements: list[CameraLidarClaimRequirementResult] = Field(min_length=1)
    summary: str
    provenance: CameraLidarSotaAuditProvenance

    @model_validator(mode="after")
    def check_supported_claim(self) -> CameraLidarSotaAuditResult:
        """Require the verdict to match required gates and coverage."""

        required = [item for item in self.requirements if item.required]
        contradicted = any(item.status == "contradicted" for item in required)
        all_achieved = all(item.status == "achieved" for item in required)
        coverage = (
            len(set(self.achieved_dataset_families))
            >= self.minimum_dataset_families
            and self.achieved_independent_rig_count
            >= self.minimum_independent_rigs
        )
        if self.verdict == "supported" and not (all_achieved and coverage):
            raise ValueError("supported SOTA verdict requires every coverage/gate")
        if self.verdict == "refuted" and not contradicted:
            raise ValueError(
                "refuted SOTA verdict requires a contradicted required gate"
            )
        if self.verdict == "incomplete" and (
            contradicted or (all_achieved and coverage)
        ):
            raise ValueError(
                "incomplete SOTA verdict requires unresolved evidence or coverage"
            )
        return self

    def save(self, path: str | Path) -> None:
        """Save the SOTA audit result."""

        _save(self, path)


def camera_lidar_sota_audit_protocol_json_schema() -> dict[str, Any]:
    """Return the SOTA audit protocol JSON schema."""

    return CameraLidarSotaAuditProtocol.model_json_schema()


def camera_lidar_sota_audit_result_json_schema() -> dict[str, Any]:
    """Return the SOTA audit result JSON schema."""

    return CameraLidarSotaAuditResult.model_json_schema()


def load_camera_lidar_sota_audit_protocol(
    path: str | Path,
) -> CameraLidarSotaAuditProtocol:
    """Load and validate a frozen SOTA audit protocol."""

    return CameraLidarSotaAuditProtocol.model_validate(read_mapping(Path(path)))


def load_camera_lidar_sota_audit_result(
    path: str | Path,
) -> CameraLidarSotaAuditResult:
    """Load and validate a SOTA audit result."""

    return CameraLidarSotaAuditResult.model_validate(read_mapping(Path(path)))


def _save(model: StrictModel, path: str | Path) -> None:
    write_mapping(
        Path(path),
        model.model_dump(mode="json", exclude_none=True),
    )
