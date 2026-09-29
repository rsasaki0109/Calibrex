"""Sensor-pair-agnostic, schema-valid state-of-the-art claim audits.

A SOTA claim is only as good as the evidence frozen before it is evaluated.
A protocol fixes, per claim scope (which modalities, which estimated
quantities, which method category), every numerical gate and the SHA-256 of
every benchmark artifact that can satisfy it.  The audit then reports
``supported`` only when every required gate and the declared dataset-family and
independent-rig coverage are achieved, ``refuted`` when any required gate is
contradicted, and ``incomplete`` otherwise.  The result model re-checks that
rule, so an artifact cannot carry an unsupported ``supported`` label.

The Camera--LiDAR audit in :mod:`calibrex.core.camera_lidar_sota_audit` is a
specialization of these models and shares the same evaluation engine.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.result import StrictModel

SOTA_AUDIT_PROTOCOL_SCHEMA_VERSION: Literal["slac.sota_audit_protocol/v0.1"] = (
    "slac.sota_audit_protocol/v0.1"
)
SOTA_AUDIT_RESULT_SCHEMA_VERSION: Literal["slac.sota_audit_result/v0.1"] = (
    "slac.sota_audit_result/v0.1"
)

SotaModality = Literal[
    "camera",
    "lidar",
    "imu",
    "gnss",
    "ins",
    "radar",
    "rgbd",
    "wheel_odometry",
    "vehicle",
    "robot_arm",
]
SotaQuantity = Literal[
    "rotation",
    "translation",
    "time_offset",
    "lever_arm",
    "intrinsics",
    "scale",
    "gravity",
    "bias",
]
SotaVerdict = Literal["supported", "refuted", "incomplete"]
SotaRequirementStatus = Literal["achieved", "contradicted", "incomplete", "missing"]
SotaStatistic = Literal[
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
]
SotaComparison = Literal[
    "greater_equal",
    "greater_than",
    "less_equal",
    "less_than",
    "equal",
]

METRIC_STATISTICS: frozenset[str] = frozenset(
    {
        "metric_mean",
        "metric_median",
        "metric_p90",
        "metric_p95",
        "metric_maximum",
        "paired_improvement_ci95_low",
        "paired_improvement_ci95_high",
    }
)
PAIRED_STATISTICS: frozenset[str] = frozenset(
    {"paired_improvement_ci95_low", "paired_improvement_ci95_high"}
)


class SotaAuditProvenance(StrictModel):
    """Generator/command lineage for a claim protocol or result."""

    generator: str
    generator_version: str
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    source_sha256: dict[str, str] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @model_validator(mode="after")
    def check_digests(self) -> SotaAuditProvenance:
        """Validate all declared SHA-256 values."""

        invalid = [
            name
            for name, digest in self.source_sha256.items()
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)
        ]
        if invalid:
            raise ValueError("invalid audit provenance SHA-256: " + ", ".join(invalid))
        return self


class SotaClaimRequirement(StrictModel):
    """One frozen numerical or completion gate."""

    requirement_id: str
    phase: str
    description: str
    dataset_family: str | None = None
    independent_rig: bool = False
    rig_id: str | None = None
    evidence_path: str | None = None
    evidence_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    method_id: str | None = None
    reference_method_id: str | None = None
    metric: str | None = None
    statistic: SotaStatistic | None = None
    comparison: SotaComparison | None = None
    threshold: float | None = None
    required: bool = True

    @model_validator(mode="after")
    def check_locator(self) -> SotaClaimRequirement:
        """Require a complete locator when benchmark evidence is declared."""

        if self.threshold is not None and not math.isfinite(self.threshold):
            raise ValueError("audit threshold must be finite")
        locator = (self.method_id, self.metric, self.statistic)
        numeric_gate = (self.comparison, self.threshold)
        if self.evidence_path is None:
            if any(value is not None for value in (*locator, *numeric_gate)):
                raise ValueError("requirement without evidence cannot declare a locator/gate")
        elif (
            self.evidence_sha256 is None
            or self.method_id is None
            or self.statistic is None
            or self.comparison is None
            or self.threshold is None
            or (self.statistic in METRIC_STATISTICS and self.metric is None)
        ):
            raise ValueError("benchmark evidence requires digest, method, statistic, and gate")
        paired = self.statistic in PAIRED_STATISTICS
        if paired and self.reference_method_id is None:
            raise ValueError("paired confidence requirements need reference_method_id")
        if not paired and self.reference_method_id is not None:
            raise ValueError("reference_method_id is only valid for paired confidence gates")
        if self.independent_rig and self.rig_id is None:
            raise ValueError("independent rig requirements require rig_id")
        return self


class SotaClaimRequirementResult(StrictModel):
    """Observed result for one frozen requirement."""

    requirement_id: str
    required: bool
    status: SotaRequirementStatus
    observed_value: float | None = None
    threshold: float | None = None
    reason: str
    evidence_path: str | None = None
    evidence_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class SotaClaimScope(StrictModel):
    """What a SOTA claim is about: modalities, estimated quantities, category.

    ``modalities`` lists the sensor types involved and may repeat a type, as in
    ``["lidar", "lidar"]`` for a multi-LiDAR claim or ``["camera"]`` for
    intrinsics.
    """

    modalities: list[SotaModality] = Field(min_length=1)
    quantities: list[SotaQuantity] = Field(min_length=1)
    category: str = Field(pattern=r"^[a-z][a-z0-9_]*$")

    @model_validator(mode="after")
    def check_unique_quantities(self) -> SotaClaimScope:
        """Reject repeated quantities."""

        if len(set(self.quantities)) != len(self.quantities):
            raise ValueError("claim scope quantities must be unique")
        return self

    @property
    def pair_id(self) -> str:
        """Return an order-independent identifier such as ``gnss-lidar``."""

        return "-".join(sorted(self.modalities))


class SotaAuditProtocol(StrictModel):
    """Frozen requirements for one explicitly scoped SOTA claim."""

    schema_version: Literal["slac.sota_audit_protocol/v0.1"] = SOTA_AUDIT_PROTOCOL_SCHEMA_VERSION
    protocol_id: str
    scope: SotaClaimScope
    claim_text: str
    minimum_dataset_families: int = Field(default=2, ge=1)
    minimum_independent_rigs: int = Field(default=1, ge=1)
    requirements: list[SotaClaimRequirement] = Field(min_length=1)
    provenance: SotaAuditProvenance

    @model_validator(mode="after")
    def check_unique_requirements(self) -> SotaAuditProtocol:
        """Require unique stable requirement IDs."""

        check_unique_requirement_ids(self.requirements)
        return self

    def save(self, path: str | Path) -> None:
        """Save the frozen audit protocol."""

        save_sota_model(self, path)


class SotaAuditResult(StrictModel):
    """Machine-readable support/refutation decision for a SOTA claim."""

    schema_version: Literal["slac.sota_audit_result/v0.1"] = SOTA_AUDIT_RESULT_SCHEMA_VERSION
    audit_id: str
    protocol_id: str
    scope: SotaClaimScope
    claim_text: str
    verdict: SotaVerdict
    achieved_dataset_families: list[str]
    achieved_independent_rig_count: int = Field(ge=0)
    minimum_dataset_families: int = Field(ge=1)
    minimum_independent_rigs: int = Field(ge=1)
    requirements: list[SotaClaimRequirementResult] = Field(min_length=1)
    summary: str
    provenance: SotaAuditProvenance

    @model_validator(mode="after")
    def check_supported_claim(self) -> SotaAuditResult:
        """Require the verdict to match required gates and coverage."""

        check_verdict_consistency(
            self.verdict,
            self.requirements,
            achieved_dataset_families=self.achieved_dataset_families,
            achieved_independent_rig_count=self.achieved_independent_rig_count,
            minimum_dataset_families=self.minimum_dataset_families,
            minimum_independent_rigs=self.minimum_independent_rigs,
        )
        return self

    def save(self, path: str | Path) -> None:
        """Save the SOTA audit result."""

        save_sota_model(self, path)


def check_unique_requirement_ids(requirements: Sequence[SotaClaimRequirement]) -> None:
    """Raise when two frozen requirements share an ID."""

    identifiers = [item.requirement_id for item in requirements]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("SOTA audit requirement IDs must be unique")


def check_verdict_consistency(
    verdict: SotaVerdict,
    requirements: Sequence[SotaClaimRequirementResult],
    *,
    achieved_dataset_families: Sequence[str],
    achieved_independent_rig_count: int,
    minimum_dataset_families: int,
    minimum_independent_rigs: int,
) -> None:
    """Raise unless ``verdict`` follows from the gates and coverage."""

    required = [item for item in requirements if item.required]
    contradicted = any(item.status == "contradicted" for item in required)
    all_achieved = all(item.status == "achieved" for item in required)
    coverage = (
        len(set(achieved_dataset_families)) >= minimum_dataset_families
        and achieved_independent_rig_count >= minimum_independent_rigs
    )
    if verdict == "supported" and not (all_achieved and coverage):
        raise ValueError("supported SOTA verdict requires every coverage/gate")
    if verdict == "refuted" and not contradicted:
        raise ValueError("refuted SOTA verdict requires a contradicted required gate")
    if verdict == "incomplete" and (contradicted or (all_achieved and coverage)):
        raise ValueError("incomplete SOTA verdict requires unresolved evidence or coverage")


def sota_audit_protocol_json_schema() -> dict[str, Any]:
    """Return the pair-agnostic SOTA audit protocol JSON schema."""

    return SotaAuditProtocol.model_json_schema()


def sota_audit_result_json_schema() -> dict[str, Any]:
    """Return the pair-agnostic SOTA audit result JSON schema."""

    return SotaAuditResult.model_json_schema()


def load_sota_audit_protocol(path: str | Path) -> SotaAuditProtocol:
    """Load and validate a frozen pair-agnostic SOTA audit protocol."""

    return SotaAuditProtocol.model_validate(read_mapping(Path(path)))


def load_sota_audit_result(path: str | Path) -> SotaAuditResult:
    """Load and validate a pair-agnostic SOTA audit result."""

    return SotaAuditResult.model_validate(read_mapping(Path(path)))


def save_sota_model(model: StrictModel, path: str | Path) -> None:
    """Write a SOTA audit model without ``None`` fields."""

    write_mapping(Path(path), model.model_dump(mode="json", exclude_none=True))
