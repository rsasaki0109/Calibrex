"""Standalone transform estimate artifacts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from slac.core.report_artifacts import ReportRunInfo
from slac.core.result import CalibrationResult, EstimateRole, StrictModel, TransformResult

TRANSFORMS_SCHEMA_VERSION: Literal["slac.transforms/v0.1"] = (
    "slac.transforms/v0.1"
)


class TransformEstimateSet(StrictModel):
    """Named group of transform estimates with a comparison role."""

    role: EstimateRole
    transforms: dict[str, TransformResult] = Field(default_factory=dict)


class TransformArtifact(StrictModel):
    """Machine-readable transform estimates for one calibration run."""

    schema_version: Literal["slac.transforms/v0.1"] = TRANSFORMS_SCHEMA_VERSION
    run: ReportRunInfo
    convention: Literal["T_parent_child"] = "T_parent_child"
    estimate_sets: dict[str, TransformEstimateSet] = Field(default_factory=dict)


def transform_artifact_json_schema() -> dict[str, Any]:
    """Return the JSON schema for standalone transform artifacts."""

    return TransformArtifact.model_json_schema()


def transform_artifact_from_result(
    result: CalibrationResult,
    *,
    run: ReportRunInfo,
) -> TransformArtifact:
    """Build a standalone transform artifact from a calibration result."""

    estimate_sets: dict[str, TransformEstimateSet] = {}
    if result.candidate_extrinsics:
        estimate_sets["candidate_extrinsics"] = TransformEstimateSet(
            role="candidate",
            transforms=result.candidate_extrinsics,
        )
    if result.reference_extrinsics:
        estimate_sets["reference_extrinsics"] = TransformEstimateSet(
            role="selected_reference",
            transforms=result.reference_extrinsics,
        )
    if result.transforms:
        estimate_sets["output_transforms"] = TransformEstimateSet(
            role="output",
            transforms=result.transforms,
        )
    return TransformArtifact(run=run, estimate_sets=estimate_sets)
