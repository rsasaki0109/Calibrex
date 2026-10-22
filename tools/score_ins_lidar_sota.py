"""Failure-aware per-axis INS--LiDAR audit scoring; the audit engine owns verdicts.

Count all three rotation axes of every declared case in coverage, including
failed/empty runs. Only data-estimated axes count; priors are forbidden. The
clock offset and scale are nuisance estimates, not accuracy claims here.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from calibrex.core.ins_lidar_hand_eye import InsLidarHandEyeArtifact, load_ins_lidar_hand_eye

ROTATION_AXES = ("roll", "pitch", "yaw")


class CaseSpec(BaseModel):
    """Frozen case identity. A missing method result remains in the denominator."""

    model_config = ConfigDict(extra="forbid")
    case_id: str
    dataset_family: str
    rig_id: str
    drive_ids: list[str] = Field(min_length=1)
    artifact_path: str | None = None
    artifact_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


def score_case(artifact: InsLidarHandEyeArtifact | None) -> dict[str, Any]:
    """Return finite statistics, retaining failures and missing reference/control evidence."""
    result: dict[str, Any] = {
        "completed": False,
        "rotation_axes": 3,
        "estimated_axes": 0,
        "errors_deg": [],
        "control_misses": 0,
        "reasons": [],
    }
    if artifact is None:
        result["reasons"].append("missing method result")
        return result
    if artifact.options.get("priors") != [] or any(d.status == "prior" for d in artifact.dofs):
        raise ValueError("audit method must not use priors")
    if artifact.solver_status != "converged":
        result["reasons"].append("method did not converge")
        return result
    if (
        artifact.jackknife_fits < 3
        or not artifact.holdout_blocks
        or artifact.holdout.motion_count < 1
    ):
        result["reasons"].append("missing jackknife or held-out evidence")
        return result
    records = [d for d in artifact.dofs if d.name in ROTATION_AXES]
    if len(records) != 3 or len({d.name for d in records}) != 3:
        result["reasons"].append("missing or duplicate rotation records")
        return result
    observed = [d for d in records if d.status == "estimated"]
    for dof in observed:
        if dof.error_to_reference is None or not math.isfinite(dof.error_to_reference):
            result["reasons"].append(f"{dof.name}: missing finite reference difference")
            return result
        if dof.reference_value is None or not math.isfinite(dof.reference_value):
            result["reasons"].append(f"{dof.name}: missing finite reference value")
            return result
        if not math.isfinite(dof.std_reported):
            result["reasons"].append(f"{dof.name}: non-finite uncertainty")
            return result
    result["completed"] = True
    result["estimated_axes"] = len(observed)
    result["errors_deg"] = [abs(d.error_to_reference) for d in observed]
    for dof in observed:
        control = dof.known_bad_control
        detected = (
            control is not None
            and control.unit == "deg"
            and control.amount == 1.0
            and control.detected
            and math.isfinite(control.holdout_delta_chi2)
            and control.holdout_delta_chi2 >= 9.0
        )
        result["control_misses"] += int(not detected)
    return result


def score_artifacts(cases: list[CaseSpec], base_dir: Path) -> dict[str, Any]:
    """Score a digest-pinned case manifest without deriving or changing any thresholds.

    A corrupt or mismatched artifact raises rather than being silently scored
    as another recording. A declared missing result is an explicit method
    failure. The caller records provenance and feeds these metrics to the
    frozen SOTA protocol; this function never labels a claim supported.
    """
    if not cases or len({c.case_id for c in cases}) != len(cases):
        raise ValueError("non-empty unique declared cases required")
    rows = []
    for case in cases:
        artifact = None
        if case.artifact_path is not None:
            path = base_dir / case.artifact_path
            if case.artifact_sha256 is None:
                raise ValueError(f"{case.case_id}: artifact digest required")
            if hashlib.sha256(path.read_bytes()).hexdigest() != case.artifact_sha256:
                raise ValueError(f"{case.case_id}: artifact digest mismatch")
            artifact = load_ins_lidar_hand_eye(path)
            if (
                artifact.provenance.dataset_family != case.dataset_family
                or artifact.provenance.drive_ids != case.drive_ids
            ):
                raise ValueError(f"{case.case_id}: recording identity mismatch")
        elif case.artifact_sha256 is not None:
            raise ValueError("missing artifact cannot declare a result digest")
        rows.append({"case": case.model_dump(mode="json"), **score_case(artifact)})
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[row["case"]["dataset_family"]].append(row)
    families = {}
    for family, group in sorted(by_family.items()):
        errors = [e for row in group for e in row["errors_deg"]]
        estimated = sum(row["estimated_axes"] for row in group)
        misses = sum(row["control_misses"] for row in group)
        families[family] = {
            "declared_recordings": len(group),
            "recording_completion": sum(row["completed"] for row in group) / len(group),
            "axis_coverage": estimated / (3 * len(group)),
            "estimated_axes": estimated,
            "control_misses": misses,
            "control_miss_rate": misses / estimated if estimated else None,
            "maximum_reference_difference_deg": max(errors) if errors else None,
        }
    return {
        "cases": rows,
        "families": families,
        "declared_dataset_families": len(by_family),
        "declared_independent_rigs": len({c.rig_id for c in cases}),
    }
