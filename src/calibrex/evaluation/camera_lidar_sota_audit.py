"""Execute a frozen Camera--LiDAR SOTA claim audit.

Evaluation runs through the pair-agnostic engine in
:mod:`calibrex.evaluation.sota_audit`; this module only converts to and from
the Camera--LiDAR artifact models.
"""

from __future__ import annotations

from pathlib import Path

from calibrex.core.camera_lidar_sota_audit import (
    CameraLidarClaimRequirement,
    CameraLidarClaimRequirementResult,
    CameraLidarSotaAuditProvenance,
    CameraLidarSotaAuditResult,
    load_camera_lidar_sota_audit_protocol,
)
from calibrex.evaluation.sota_audit import (
    evaluate_sota_protocol,
    evaluate_sota_requirement,
    required_sha256,
)


def audit_camera_lidar_sota_claim(
    protocol_path: str | Path,
    *,
    audit_id: str | None = None,
    command: list[str] | None = None,
) -> CameraLidarSotaAuditResult:
    """Evaluate all frozen requirements without tuning or omission."""

    path = Path(protocol_path)
    protocol = load_camera_lidar_sota_audit_protocol(path)
    generic = evaluate_sota_protocol(
        protocol.as_generic(),
        protocol_sha256=required_sha256(path),
        base_path=path.parent,
        audit_id=audit_id,
        command=command,
        generator=__name__,
    )
    payload = generic.model_dump(mode="python", exclude={"schema_version", "scope"})
    payload["declared_category"] = protocol.declared_category
    payload["requirements"] = [
        CameraLidarClaimRequirementResult.model_validate(item)
        for item in payload["requirements"]
    ]
    payload["provenance"] = CameraLidarSotaAuditProvenance.model_validate(
        payload["provenance"]
    )
    return CameraLidarSotaAuditResult.model_validate(payload)


def _evaluate_requirement(
    requirement: CameraLidarClaimRequirement,
    *,
    base_path: Path,
) -> CameraLidarClaimRequirementResult:
    return CameraLidarClaimRequirementResult.model_validate(
        evaluate_sota_requirement(requirement, base_path=base_path).model_dump()
    )
