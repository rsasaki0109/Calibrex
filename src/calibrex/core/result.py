"""Typed calibration result schema."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from calibrex.core.exceptions import ResultError
from calibrex.core.geometry import SE3
from calibrex.core.io import read_mapping, write_mapping

RESULT_SCHEMA_VERSION: Literal["calibrex.result/v0.1"] = "calibrex.result/v0.1"
Grade = Literal["pass", "warn", "fail"]


class StrictModel(BaseModel):
    """Base model with stable, explicit fields."""

    model_config = ConfigDict(extra="forbid")


class RunInfo(StrictModel):
    id: str
    calibrex_version: str
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


class TransformResult(StrictModel):
    convention: Literal["T_parent_child"] = "T_parent_child"
    parent: str
    child: str
    translation_m: list[float] = Field(min_length=3, max_length=3)
    rotation_quat_xyzw: list[float] = Field(min_length=4, max_length=4)
    covariance: CovarianceInfo | None = None
    quality: TransformQuality = Field(default_factory=TransformQuality)

    def as_se3(self) -> SE3:
        """Return this transform as an `SE3` value."""

        return SE3.from_lists(self.translation_m, self.rotation_quat_xyzw)


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
    schema_version: Literal["calibrex.result/v0.1"] = RESULT_SCHEMA_VERSION
    run: RunInfo
    frame_graph: FrameGraphSnapshot
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

    def save(self, path: str | Path) -> None:
        """Save this result as YAML or JSON."""

        write_mapping(Path(path), self.model_dump(mode="json", exclude_none=True))


def load_result(path: str | Path) -> CalibrationResult:
    """Load and validate a Calibrex result."""

    result_path = Path(path)
    try:
        return CalibrationResult.model_validate(read_mapping(result_path))
    except Exception as exc:
        raise ResultError(f"invalid result {result_path}: {exc}") from exc


def result_json_schema() -> dict[str, Any]:
    """Return the JSON schema for result files."""

    return CalibrationResult.model_json_schema()
