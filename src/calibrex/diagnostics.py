"""Schema-versioned environment and dataset diagnostics."""

from __future__ import annotations

import importlib.util
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, cast

from pydantic import Field
from yaml import YAMLError

from calibrex.core.config import DatasetConfig, DatasetType
from calibrex.core.io import read_mapping
from calibrex.core.provenance import git_commit
from calibrex.core.result import DegeneracyResult, MetricResult, StrictModel
from calibrex.data.inspect import DatasetInspection, inspect_dataset
from calibrex.evaluation.degeneracy import degeneracy_from_inspection
from calibrex.evaluation.lidar import lidar_metrics_from_inspection
from calibrex.evaluation.motion import motion_metrics_from_inspection
from calibrex.evaluation.recommendations import build_inspection_recommendations
from calibrex.evaluation.thresholds import apply_metric_thresholds
from calibrex.evaluation.timing import timing_metrics_from_inspection

DOCTOR_SCHEMA_VERSION: Literal["slac.doctor/v0.1"] = "slac.doctor/v0.1"
DoctorStatus = Literal["pass", "warn", "fail"]
WorkflowStatus = Literal["available", "conditional"]


class DoctorDependency(StrictModel):
    """Availability of one runtime dependency."""

    available: bool
    optional: bool = False


class DoctorEnvironment(StrictModel):
    """Runtime environment recorded by a doctor run."""

    calibrex_version: str
    python: str
    platform: str
    dependencies: dict[str, DoctorDependency]
    core_ros_independent: bool = True


class DoctorDataset(StrictModel):
    """Dataset inspection embedded in the doctor artifact."""

    dataset_type: DatasetType
    path: str
    exists: bool
    manifest: str | None = None
    streams: list[dict[str, object | None]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    diagnostics: dict[str, object] = Field(default_factory=dict)


class DoctorQuality(StrictModel):
    """Provisional quality evidence derived without running a solver."""

    metrics: dict[str, MetricResult] = Field(default_factory=dict)
    degeneracy: DegeneracyResult
    recommendations: list[str] = Field(default_factory=list)


class DoctorWorkflow(StrictModel):
    """A workflow that the inspected inputs can support."""

    workflow_id: str
    status: WorkflowStatus
    reason: str
    next_command: str | None = None


class DoctorProvenance(StrictModel):
    """Lineage of a generated doctor artifact."""

    producer: Literal["calibrex"] = "calibrex"
    calibrex_version: str
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    dataset_type_source: Literal["none", "explicit", "inferred"] = "none"
    dataset_digest: str | None = None
    dataset_digest_status: Literal["not_applicable", "not_computed"] = "not_applicable"


class DoctorArtifact(StrictModel):
    """Portable environment and input-readiness diagnosis."""

    schema_version: Literal["slac.doctor/v0.1"] = DOCTOR_SCHEMA_VERSION
    status: DoctorStatus
    environment: DoctorEnvironment
    dataset: DoctorDataset | None = None
    quality: DoctorQuality | None = None
    workflows: list[DoctorWorkflow] = Field(default_factory=list)
    provenance: DoctorProvenance


def doctor_json_schema() -> dict[str, object]:
    """Return the JSON schema for doctor artifacts."""

    return DoctorArtifact.model_json_schema()


def infer_dataset_type(path: Path) -> DatasetType:
    """Infer a supported dataset type from a path without opening ROS middleware."""

    suffix = path.suffix.lower()
    if suffix == ".mcap":
        return "mcap"
    if suffix == ".bag":
        return "rosbag1"
    if path.is_dir():
        for declaration in (path / "config.yaml", path / "config.json"):
            declared_type = _declared_dataset_type(declaration, nested=True)
            if declared_type is not None:
                return declared_type
        for declaration in (path / "manifest.yaml", path / "manifest.json"):
            declared_type = _declared_dataset_type(declaration, nested=False)
            if declared_type is not None:
                return declared_type
        if (path / "metadata.yaml").is_file() or (path / "metadata.json").is_file():
            return "rosbag2"
        if (path / "velodyne_points").is_dir() and (path / "oxts").is_dir():
            return "kitti_raw"
        if (path / "rgb.txt").is_file() and (path / "depth.txt").is_file():
            return "tum_rgbd"
        if (path / "v1.0-mini").is_dir() or (path / "sample_data.json").is_file():
            return "nuscenes"
        if any(path.glob("*.pcd")):
            return "livox_pcd"
        if any(path.rglob("*.npz")):
            return "a2d2_lidar"
    return "filesystem"


def build_doctor_artifact(
    *,
    calibrex_version: str,
    command: list[str],
    path: Path | None = None,
    dataset_type: DatasetType | None = None,
    sample_limit: int | None = None,
) -> DoctorArtifact:
    """Diagnose the runtime and, optionally, one dataset path."""

    environment = _environment(calibrex_version)
    provenance = DoctorProvenance(
        calibrex_version=calibrex_version,
        git_commit=git_commit(),
        command=command,
        dataset_type_source=(
            "none" if path is None else "explicit" if dataset_type is not None else "inferred"
        ),
        dataset_digest_status="not_applicable" if path is None else "not_computed",
    )
    if path is None:
        return DoctorArtifact(
            status="pass",
            environment=environment,
            provenance=provenance,
        )

    resolved_type = dataset_type or infer_dataset_type(path)
    inspection = inspect_dataset(
        DatasetConfig(
            type=resolved_type,
            path=str(path),
            sample_limit=sample_limit,
        )
    )
    quality = _quality(inspection)
    status: DoctorStatus = "pass"
    if not inspection.exists or quality.degeneracy.grade == "fail":
        status = "fail"
    elif inspection.warnings or quality.degeneracy.grade == "warn":
        status = "warn"
    return DoctorArtifact(
        status=status,
        environment=environment,
        dataset=DoctorDataset.model_validate(inspection.as_dict()),
        quality=quality,
        workflows=_workflows(inspection),
        provenance=provenance,
    )


def _environment(calibrex_version: str) -> DoctorEnvironment:
    dependencies = {
        "pydantic": DoctorDependency(available=_has_module("pydantic")),
        "yaml": DoctorDependency(available=_has_module("yaml")),
        "jsonschema": DoctorDependency(available=_has_module("jsonschema")),
        "mcap": DoctorDependency(available=_has_module("mcap"), optional=True),
        "open3d": DoctorDependency(available=_has_module("open3d"), optional=True),
    }
    return DoctorEnvironment(
        calibrex_version=calibrex_version,
        python=platform.python_version(),
        platform=platform.platform(),
        dependencies=dependencies,
    )


def _quality(inspection: DatasetInspection) -> DoctorQuality:
    domain = "autonomous_driving" if inspection.dataset_type == "kitti_raw" else "robotics"
    metrics = lidar_metrics_from_inspection(inspection)
    metrics.update(motion_metrics_from_inspection(inspection))
    metrics.update(timing_metrics_from_inspection(inspection))
    apply_metric_thresholds(
        metrics,
        "autonomous_driving" if domain == "autonomous_driving" else "default",
    )
    degeneracy = degeneracy_from_inspection(inspection)
    return DoctorQuality(
        metrics=metrics,
        degeneracy=degeneracy,
        recommendations=build_inspection_recommendations(
            metrics,
            degeneracy,
            domain=domain,
        ),
    )


def _workflows(inspection: DatasetInspection) -> list[DoctorWorkflow]:
    kinds = {stream.kind for stream in inspection.streams}
    pointcloud_count = sum(stream.kind == "pointcloud" for stream in inspection.streams)
    workflows: list[DoctorWorkflow] = []
    if pointcloud_count >= 2 or inspection.dataset_type in {"a2d2_lidar", "livox_pcd"}:
        workflows.append(
            DoctorWorkflow(
                workflow_id="lidar-lidar-evidence",
                status="available",
                reason="at least two LiDAR streams or a supported LiDAR pair dataset were found",
            )
        )
    elif pointcloud_count == 1:
        workflows.append(
            DoctorWorkflow(
                workflow_id="lidar-lidar-evidence",
                status="conditional",
                reason="a second overlapping LiDAR stream is required",
            )
        )
    if (
        inspection.dataset_type == "kitti_raw"
        or ("pointcloud" in kinds and ({"camera", "image"} & kinds))
    ):
        workflows.append(
            DoctorWorkflow(
                workflow_id="camera-lidar-evidence",
                status="available",
                reason="camera and LiDAR inputs were found",
                next_command="calibrex init camera-lidar-imu --output calibrex.yaml",
            )
        )
    if inspection.dataset_type == "tum_rgbd" or "rgbd" in kinds:
        workflows.append(
            DoctorWorkflow(
                workflow_id="rgbd-joint-slac",
                status="available",
                reason="associated RGB-D inputs were found",
            )
        )
    if inspection.dataset_type == "nuscenes" or "radar" in kinds:
        workflows.append(
            DoctorWorkflow(
                workflow_id="radar-extrinsic-evidence",
                status="available",
                reason="radar-capable dataset inputs were found",
            )
        )
    return workflows


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _declared_dataset_type(path: Path, *, nested: bool) -> DatasetType | None:
    if not path.is_file():
        return None
    try:
        payload = read_mapping(path)
    except (OSError, ValueError, YAMLError):
        return None
    candidate: object = payload
    if nested:
        candidate = payload.get("dataset")
        if not isinstance(candidate, dict):
            return None
    value = candidate.get("type") if isinstance(candidate, dict) else None
    supported = {
        "a2d2_lidar",
        "filesystem",
        "kitti_raw",
        "livox_pcd",
        "mcap",
        "nuscenes",
        "rosbag1",
        "rosbag2",
        "tum_rgbd",
    }
    if isinstance(value, str) and value in supported:
        return cast(DatasetType, value)
    return None
