"""Schema-valid Camera--LiDAR SOTA claim audit protocol and result.

These models specialize the pair-agnostic audit in
:mod:`calibrex.core.sota_audit`: requirements, requirement results, and
provenance are the same models under their original names, so the published
Camera--LiDAR schemas are unchanged, and :meth:`as_generic` converts protocols
and results into the pair-agnostic form used by the SOTA leaderboard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final, Literal

from pydantic import Field, model_validator

from calibrex.core.io import read_mapping
from calibrex.core.result import StrictModel
from calibrex.core.sota_audit import (
    SotaAuditProtocol,
    SotaAuditProvenance,
    SotaAuditResult,
    SotaClaimRequirement,
    SotaClaimRequirementResult,
    SotaClaimScope,
    SotaQuantity,
    check_unique_requirement_ids,
    check_verdict_consistency,
    save_sota_model,
)

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

_CATEGORY_QUANTITIES: Final[dict[str, list[SotaQuantity]]] = {
    "training_free_targetless": ["rotation", "translation"],
    "learned_targetless": ["rotation", "translation"],
    "spatiotemporal_targetless": ["rotation", "translation", "time_offset"],
    "production_target_based_reference": ["rotation", "translation"],
}


class CameraLidarSotaAuditProvenance(SotaAuditProvenance):
    """Generator/command lineage for a claim protocol or result."""


class CameraLidarClaimRequirement(SotaClaimRequirement):
    """One frozen numerical or completion gate."""


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

        check_unique_requirement_ids(self.requirements)
        return self

    def as_generic(self) -> SotaAuditProtocol:
        """Return the equivalent pair-agnostic protocol."""

        return SotaAuditProtocol(
            protocol_id=self.protocol_id,
            scope=camera_lidar_claim_scope(self.declared_category),
            claim_text=self.claim_text,
            minimum_dataset_families=self.minimum_dataset_families,
            minimum_independent_rigs=self.minimum_independent_rigs,
            requirements=[
                SotaClaimRequirement.model_validate(item.model_dump())
                for item in self.requirements
            ],
            provenance=SotaAuditProvenance.model_validate(
                self.provenance.model_dump()
            ),
        )

    def save(self, path: str | Path) -> None:
        """Save the frozen audit protocol."""

        save_sota_model(self, path)


class CameraLidarClaimRequirementResult(SotaClaimRequirementResult):
    """Observed result for one frozen requirement."""


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

        check_verdict_consistency(
            self.verdict,
            self.requirements,
            achieved_dataset_families=self.achieved_dataset_families,
            achieved_independent_rig_count=self.achieved_independent_rig_count,
            minimum_dataset_families=self.minimum_dataset_families,
            minimum_independent_rigs=self.minimum_independent_rigs,
        )
        return self

    def as_generic(self) -> SotaAuditResult:
        """Return the equivalent pair-agnostic result."""

        payload = self.model_dump(mode="python", exclude={"schema_version", "declared_category"})
        payload["scope"] = camera_lidar_claim_scope(self.declared_category)
        return SotaAuditResult.model_validate(payload)

    def save(self, path: str | Path) -> None:
        """Save the SOTA audit result."""

        save_sota_model(self, path)


def camera_lidar_claim_scope(category: CameraLidarSotaCategory) -> SotaClaimScope:
    """Return the pair-agnostic scope of a Camera--LiDAR claim category."""

    return SotaClaimScope(
        modalities=["camera", "lidar"],
        quantities=list(_CATEGORY_QUANTITIES[category]),
        category=category,
    )


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
