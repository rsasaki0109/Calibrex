"""Schema-versioned environment and dataset readiness for ``calibrex doctor``."""

from __future__ import annotations

import importlib
import importlib.util
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, cast

from pydantic import Field
from yaml import YAMLError

from calibrex.core.config import DatasetConfig, DatasetType
from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.provenance import git_commit
from calibrex.core.result import DegeneracyResult, MetricResult, StrictModel
from calibrex.data.inspect import DatasetInspection, inspect_dataset
from calibrex.evaluation.degeneracy import degeneracy_from_inspection
from calibrex.evaluation.lidar import lidar_metrics_from_inspection
from calibrex.evaluation.motion import motion_metrics_from_inspection
from calibrex.evaluation.recommendations import build_inspection_recommendations
from calibrex.evaluation.thresholds import apply_metric_thresholds
from calibrex.evaluation.timing import timing_metrics_from_inspection
from calibrex.init_templates import resolve_sensor_template_name

ENVIRONMENT_READINESS_SCHEMA_VERSION: Literal[
    "slac.environment_readiness/v0.1"
] = "slac.environment_readiness/v0.1"
ReadinessStatus = Literal["pass", "warn", "fail"]
WorkflowStatus = Literal["available", "conditional"]

_SENSOR_TEMPLATE_PREFIX = "examples/sensor_templates"


class ReadinessDependency(StrictModel):
    """Availability and version of one runtime dependency."""

    available: bool
    optional: bool = False
    version: str | None = None


class ReadinessEnvironment(StrictModel):
    """Runtime environment recorded by a readiness check."""

    calibrex_version: str
    python_version: str
    platform: str
    dependencies: dict[str, ReadinessDependency]
    core_ros_independent: bool = True


class ReadinessDatasetCheck(StrictModel):
    """Dataset path inspection embedded in a readiness artifact."""

    dataset_type: DatasetType
    path: str
    exists: bool
    manifest: str | None = None
    streams: list[dict[str, object | None]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    diagnostics: dict[str, object] = Field(default_factory=dict)


class ReadinessQuality(StrictModel):
    """Provisional quality evidence derived without running a solver."""

    metrics: dict[str, MetricResult] = Field(default_factory=dict)
    degeneracy: DegeneracyResult
    recommendations: list[str] = Field(default_factory=list)


class WorkflowSuggestion(StrictModel):
    """A suggested calibration workflow and starter template."""

    workflow_id: str
    status: WorkflowStatus
    reason: str
    template_path: str
    next_command: str | None = None


class ReadinessProvenance(StrictModel):
    """Lineage of a generated environment-readiness artifact."""

    producer: Literal["calibrex"] = "calibrex"
    calibrex_version: str
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    dataset_type_source: Literal["none", "explicit", "inferred"] = "none"
    dataset_digest: str | None = None
    dataset_digest_status: Literal["not_applicable", "not_computed"] = (
        "not_applicable"
    )


class EnvironmentReadinessArtifact(StrictModel):
    """Portable environment and input-readiness diagnosis."""

    schema_version: Literal[
        "slac.environment_readiness/v0.1"
    ] = ENVIRONMENT_READINESS_SCHEMA_VERSION
    status: ReadinessStatus
    environment: ReadinessEnvironment
    dataset: ReadinessDatasetCheck | None = None
    quality: ReadinessQuality | None = None
    workflow_suggestions: list[WorkflowSuggestion] = Field(default_factory=list)
    provenance: ReadinessProvenance

    def save(self, path: str | Path) -> None:
        """Persist this artifact as schema-valid YAML or JSON."""

        write_mapping(
            Path(path),
            self.model_dump(mode="json", exclude_none=True),
        )


def environment_readiness_json_schema() -> dict[str, object]:
    """Return the JSON schema for environment-readiness artifacts."""

    return EnvironmentReadinessArtifact.model_json_schema()


def load_environment_readiness(path: str | Path) -> EnvironmentReadinessArtifact:
    """Load a schema-valid environment-readiness artifact from disk."""

    payload = read_mapping(Path(path))
    return EnvironmentReadinessArtifact.model_validate(payload)


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


def build_environment_readiness_artifact(
    *,
    calibrex_version: str,
    command: list[str],
    path: Path | None = None,
    dataset_type: DatasetType | None = None,
    sample_limit: int | None = None,
) -> EnvironmentReadinessArtifact:
    """Diagnose the runtime and, optionally, one dataset path."""

    environment = _environment(calibrex_version)
    provenance = ReadinessProvenance(
        calibrex_version=calibrex_version,
        git_commit=git_commit(),
        command=command,
        dataset_type_source=(
            "none"
            if path is None
            else "explicit"
            if dataset_type is not None
            else "inferred"
        ),
        dataset_digest_status="not_applicable" if path is None else "not_computed",
    )
    if path is None:
        return EnvironmentReadinessArtifact(
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
    status: ReadinessStatus = "pass"
    if not inspection.exists or quality.degeneracy.grade == "fail":
        status = "fail"
    elif inspection.warnings or quality.degeneracy.grade == "warn":
        status = "warn"
    return EnvironmentReadinessArtifact(
        status=status,
        environment=environment,
        dataset=ReadinessDatasetCheck.model_validate(inspection.as_dict()),
        quality=quality,
        workflow_suggestions=_workflow_suggestions(inspection),
        provenance=provenance,
    )


def _environment(calibrex_version: str) -> ReadinessEnvironment:
    dependencies = {
        "pydantic": _dependency("pydantic"),
        "yaml": _dependency("yaml"),
        "jsonschema": _dependency("jsonschema"),
        "mcap": _dependency("mcap", optional=True),
        "open3d": _dependency("open3d", optional=True),
    }
    return ReadinessEnvironment(
        calibrex_version=calibrex_version,
        python_version=platform.python_version(),
        platform=platform.platform(),
        dependencies=dependencies,
    )


_PACKAGE_DISTRIBUTIONS: dict[str, str] = {
    "yaml": "PyYAML",
}


def _dependency(name: str, *, optional: bool = False) -> ReadinessDependency:
    if importlib.util.find_spec(name) is None:
        return ReadinessDependency(available=False, optional=optional, version=None)
    return ReadinessDependency(
        available=True,
        optional=optional,
        version=_installed_version(name),
    )


def _installed_version(module_name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version as pkg_version

    distribution = _PACKAGE_DISTRIBUTIONS.get(module_name, module_name)
    try:
        return pkg_version(distribution)
    except PackageNotFoundError:
        return None


def _quality(inspection: DatasetInspection) -> ReadinessQuality:
    domain = "autonomous_driving" if inspection.dataset_type == "kitti_raw" else "robotics"
    metrics = lidar_metrics_from_inspection(inspection)
    metrics.update(motion_metrics_from_inspection(inspection))
    metrics.update(timing_metrics_from_inspection(inspection))
    apply_metric_thresholds(
        metrics,
        "autonomous_driving" if domain == "autonomous_driving" else "default",
    )
    degeneracy = degeneracy_from_inspection(inspection)
    return ReadinessQuality(
        metrics=metrics,
        degeneracy=degeneracy,
        recommendations=build_inspection_recommendations(
            metrics,
            degeneracy,
            domain=domain,
        ),
    )


def _template_path(template_name: str) -> str:
    canonical = resolve_sensor_template_name(template_name)
    return f"{_SENSOR_TEMPLATE_PREFIX}/{canonical}"


def _init_command(template_name: str, output: str = "my_calib/config.yaml") -> str:
    return f"calibrex init --template {template_name} --output {output}"


def _lidar_lidar_template(dataset_type: str) -> str:
    if dataset_type == "rosbag1":
        return "velodyne_vlp16_pair_rosbag1"
    if dataset_type == "rosbag2":
        return "velodyne_vlp16_pair_rosbag2"
    return "velodyne_vlp16_pair_rosbag2"


def _lidar_lidar_next_command(dataset_type: str) -> str:
    template = _lidar_lidar_template(dataset_type)
    return (
        f"{_init_command(template)} && "
        "# edit dataset.path and sensor topics, then "
        "calibrex calibrate my_calib/config.yaml"
    )


def _workflow_suggestions(
    inspection: DatasetInspection,
) -> list[WorkflowSuggestion]:
    kinds = {stream.kind for stream in inspection.streams}
    pointcloud_count = sum(stream.kind == "pointcloud" for stream in inspection.streams)
    suggestions: list[WorkflowSuggestion] = []
    bag_dataset = inspection.dataset_type in {"rosbag1", "rosbag2"}
    if (
        pointcloud_count >= 2
        or inspection.dataset_type in {"a2d2_lidar", "livox_pcd"}
        or (bag_dataset and pointcloud_count >= 2)
    ):
        template = _lidar_lidar_template(inspection.dataset_type)
        suggestions.append(
            WorkflowSuggestion(
                workflow_id="lidar-lidar-evidence",
                status="available",
                reason="at least two LiDAR streams or a supported LiDAR pair dataset were found",
                template_path=_template_path(template),
                next_command=_lidar_lidar_next_command(inspection.dataset_type),
            )
        )
    elif bag_dataset:
        template = _lidar_lidar_template(inspection.dataset_type)
        suggestions.append(
            WorkflowSuggestion(
                workflow_id="lidar-lidar-evidence",
                status="conditional",
                reason=(
                    f"{inspection.dataset_type} bag detected; confirm two PointCloud2 "
                    "topics are present then edit the template"
                ),
                template_path=_template_path(template),
                next_command=_lidar_lidar_next_command(inspection.dataset_type),
            )
        )
    elif pointcloud_count == 1:
        template = _lidar_lidar_template(inspection.dataset_type)
        suggestions.append(
            WorkflowSuggestion(
                workflow_id="lidar-lidar-evidence",
                status="conditional",
                reason="a second overlapping LiDAR stream is required",
                template_path=_template_path(template),
            )
        )
    if (
        inspection.dataset_type == "kitti_raw"
        or ("pointcloud" in kinds and ({"camera", "image"} & kinds))
    ):
        template = "spinning_lidar_camera_planar_board"
        suggestions.append(
            WorkflowSuggestion(
                workflow_id="camera-lidar-evidence",
                status="available",
                reason="camera and LiDAR inputs were found",
                template_path=_template_path(template),
                next_command=_init_command(template, "calibrex.yaml"),
            )
        )
    if inspection.dataset_type == "tum_rgbd" or "rgbd" in kinds:
        suggestions.append(
            WorkflowSuggestion(
                workflow_id="rgbd-joint-slac",
                status="available",
                reason="associated RGB-D inputs were found",
                template_path=_template_path("spinning_lidar_camera_planar_board"),
            )
        )
    if inspection.dataset_type == "nuscenes" or "radar" in kinds:
        suggestions.append(
            WorkflowSuggestion(
                workflow_id="radar-extrinsic-evidence",
                status="available",
                reason="radar-capable dataset inputs were found",
                template_path=_template_path("velodyne_vlp16_pair_rosbag2"),
            )
        )
    return suggestions


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
