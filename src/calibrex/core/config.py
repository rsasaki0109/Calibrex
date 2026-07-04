"""Typed calibration config schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from calibrex.core.exceptions import ConfigError
from calibrex.core.io import read_mapping

CONFIG_SCHEMA_VERSION: Literal["calibrex.config/v0.1"] = "calibrex.config/v0.1"

SensorType = Literal["camera", "lidar", "imu", "radar"]
DatasetType = Literal[
    "a2d2_lidar",
    "filesystem",
    "kitti_raw",
    "livox_pcd",
    "mcap",
    "nuscenes",
    "rosbag1",
    "rosbag2",
    "tum_rgbd",
]
DomainType = Literal["robotics", "autonomous_driving", "industrial", "research", "other"]


class StrictModel(BaseModel):
    """Base model with stable, explicit fields."""

    model_config = ConfigDict(extra="forbid")


class ProjectConfig(StrictModel):
    name: str = "calibrex_project"
    description: str | None = None
    output_dir: str = "outputs/default"
    domain: DomainType = "robotics"


class DatasetConfig(StrictModel):
    type: DatasetType
    path: str
    time_base: str = "sensor_time_ns"
    odometry_topic: str | None = Field(
        default=None,
        description=(
            "optional nav_msgs/msg/Odometry topic for online motion compensation "
            "from rosbag1/rosbag2 replays; unset preserves the static-rig path"
        ),
    )
    sample_limit: int | None = Field(
        default=None,
        ge=1,
        description=(
            "override the number of frames/files sampled for public-dataset diagnostics "
            "(a2d2_lidar, livox_pcd, kitti_raw); unset preserves each dataset type's "
            "current default"
        ),
    )


class CameraIntrinsicsConfig(StrictModel):
    estimate: bool = False
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    distortion_model: str = "none"
    distortion: list[float] = Field(default_factory=list)


class ImuNoiseConfig(StrictModel):
    gyro_noise_density: float | None = None
    accel_noise_density: float | None = None
    gyro_random_walk: float | None = None
    accel_random_walk: float | None = None


class SensorConfig(StrictModel):
    type: SensorType
    model: str = "generic"
    topic: str | None = None
    frame_id: str | None = None
    camera_info_topic: str | None = None
    intrinsics: CameraIntrinsicsConfig | None = None
    fields: list[str] = Field(default_factory=list)
    noise: ImuNoiseConfig | None = None
    point_time_field: str | None = Field(
        default=None,
        description=(
            "optional PointCloud2 field name for per-point capture-time offsets "
            "relative to the message stamp (seconds or integer nanoseconds); "
            "enables per-point deskew during rosbag2 online motion compensation"
        ),
    )

    @model_validator(mode="after")
    def validate_sensor_specifics(self) -> SensorConfig:
        if self.type == "camera" and self.intrinsics is None:
            self.intrinsics = CameraIntrinsicsConfig()
        if self.type == "lidar" and not self.fields:
            self.fields = ["x", "y", "z"]
        return self


class TransformInitialConfig(StrictModel):
    translation: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        min_length=3,
        max_length=3,
    )
    rotation_quat_xyzw: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0, 1.0],
        min_length=4,
        max_length=4,
    )


class TransformPriorSigmaConfig(StrictModel):
    translation_m: float | None = None
    rotation_deg: float | None = None


class TransformConfig(StrictModel):
    estimate: bool = True
    initial: TransformInitialConfig = Field(default_factory=TransformInitialConfig)
    prior_sigma: TransformPriorSigmaConfig | None = None


class FrameConfig(StrictModel):
    root: bool = False
    parent: str | None = None
    transform: TransformConfig | None = None

    @model_validator(mode="after")
    def validate_parent(self) -> FrameConfig:
        if not self.root and self.parent is None:
            msg = "non-root frames require parent"
            raise ValueError(msg)
        return self


class TimeOffsetConfig(StrictModel):
    estimate: bool = False
    initial_sec: float = 0.0
    prior_sigma_sec: float | None = None


class FactorConfig(StrictModel):
    enabled: bool = True
    weight: float | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class PipelineConfig(StrictModel):
    type: str = "multi_sensor_slac"
    frontends: list[str] = Field(default_factory=list)
    factors: dict[str, FactorConfig] = Field(default_factory=dict)


class SolverConfig(StrictModel):
    backend: str = "scipy"
    max_iterations: int = Field(default=50, ge=0)
    robust_loss: str = "huber"
    convergence_tolerance: float = Field(default=1.0e-6, gt=0.0)
    seed: int | None = None


class KITTIEvaluationConfig(StrictModel):
    max_projection_pairs: int = Field(default=3, ge=0, le=500)
    projection_sample_points: int = Field(default=800, ge=1, le=200_000)
    perturbation_rotation_deg: list[float] = Field(default_factory=lambda: [0.5, 1.0])
    perturbation_translation_m: list[float] = Field(default_factory=lambda: [0.05, 0.10])

    @field_validator("perturbation_rotation_deg", "perturbation_translation_m")
    @classmethod
    def validate_positive_perturbations(cls, value: list[float]) -> list[float]:
        if any(amount <= 0.0 for amount in value):
            msg = "perturbation values must be positive"
            raise ValueError(msg)
        return value


class EvaluationConfig(StrictModel):
    holdout_ratio: float = Field(default=0.2, ge=0.0, le=0.9)
    metrics: list[str] = Field(default_factory=list)
    strict: bool = False
    kitti: KITTIEvaluationConfig = Field(default_factory=KITTIEvaluationConfig)


class OutputsConfig(StrictModel):
    result: str = "result.yaml"
    report: str = "report.html"
    artifacts_dir: str = "artifacts"


class CalibrationConfig(StrictModel):
    schema_version: Literal["calibrex.config/v0.1"] = CONFIG_SCHEMA_VERSION
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    dataset: DatasetConfig
    sensors: dict[str, SensorConfig]
    frames: dict[str, FrameConfig]
    time_offsets: dict[str, TimeOffsetConfig] = Field(default_factory=dict)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    outputs: OutputsConfig = Field(default_factory=OutputsConfig)

    @field_validator("sensors", "frames")
    @classmethod
    def require_non_empty_mapping(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not value:
            msg = "mapping must not be empty"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def validate_sensor_frames(self) -> CalibrationConfig:
        missing = sorted(set(self.sensors) - set(self.frames))
        if missing:
            msg = f"sensor frames missing from frames section: {', '.join(missing)}"
            raise ValueError(msg)
        return self

    @property
    def output_dir(self) -> Path:
        return Path(self.project.output_dir)


def load_config(path: str | Path) -> CalibrationConfig:
    """Load and validate a Calibrex config."""

    config_path = Path(path)
    try:
        return CalibrationConfig.model_validate(read_mapping(config_path))
    except Exception as exc:
        raise ConfigError(f"invalid config {config_path}: {exc}") from exc


def config_json_schema() -> dict[str, Any]:
    """Return the JSON schema for config files."""

    return CalibrationConfig.model_json_schema()
