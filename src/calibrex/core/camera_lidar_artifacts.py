"""Versioned problem, protocol, and trace artifacts for camera--LiDAR research."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.result import StrictModel, TransformResult
from calibrex.data.depth import DepthFileReference

CAMERA_LIDAR_PROBLEM_SCHEMA_VERSION: Literal[
    "slac.camera_lidar_problem/v0.1"
] = "slac.camera_lidar_problem/v0.1"
CAMERA_LIDAR_BENCHMARK_PROTOCOL_SCHEMA_VERSION: Literal[
    "slac.camera_lidar_benchmark_protocol/v0.1"
] = "slac.camera_lidar_benchmark_protocol/v0.1"
CALIBRATION_CANDIDATE_TRACE_SCHEMA_VERSION: Literal[
    "slac.calibration_candidate_trace/v0.1"
] = "slac.calibration_candidate_trace/v0.1"
BULLSEYE_PLOT_SCHEMA_VERSION: Literal[
    "slac.bullseye_plot/v0.1"
] = "slac.bullseye_plot/v0.1"
Sha256 = str


class CameraLidarArtifactProvenance(StrictModel):
    """Generator and source digests shared by camera--LiDAR artifacts."""

    generator: str
    generator_version: str
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    config_sha256: Sha256 | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    source_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class CameraLidarObservationBinding(StrictModel):
    """Binding between one depth observation and one LiDAR scan."""

    frame_id: str
    split_id: str
    depth_observation_frame_id: str
    lidar: DepthFileReference
    lidar_capture_time_ns: int


class CameraLidarCalibrationProblem(StrictModel):
    """Digest-pinned solver-neutral camera--LiDAR calibration problem."""

    schema_version: Literal[
        "slac.camera_lidar_problem/v0.1"
    ] = CAMERA_LIDAR_PROBLEM_SCHEMA_VERSION
    problem_id: str
    dataset_id: str
    dataset_family: str
    sequence_id: str
    depth_provider_path: str
    depth_provider_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    observations: list[CameraLidarObservationBinding] = Field(min_length=1)
    reference_transform_camera_lidar: TransformResult
    initial_transform_camera_lidar: TransformResult
    frame_convention: Literal["T_parent_child"] = "T_parent_child"
    time_convention: str
    rotation_bound_deg: float = Field(gt=0.0)
    translation_bound_m: float = Field(ge=0.0)
    provenance: CameraLidarArtifactProvenance

    @model_validator(mode="after")
    def check_frames(self) -> CameraLidarCalibrationProblem:
        """Require unique frame IDs and a camera-from-LiDAR transform."""

        frame_ids = [item.frame_id for item in self.observations]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("camera-LiDAR problem frame IDs must be unique")
        for transform in (
            self.reference_transform_camera_lidar,
            self.initial_transform_camera_lidar,
        ):
            if transform.convention != "T_parent_child":
                raise ValueError("camera-LiDAR transforms must use T_parent_child")
        return self

    def save(self, path: str | Path) -> None:
        """Save the problem artifact as YAML or JSON."""

        _save(self, path)


class CameraLidarPerturbation(StrictModel):
    """One immutable perturbation trial relative to dataset reference."""

    trial_id: str
    rotation_deg_xyz: list[float] = Field(min_length=3, max_length=3)
    translation_m_xyz: list[float] = Field(min_length=3, max_length=3)


class CameraLidarHitDefinition(StrictModel):
    """Prespecified success criterion matching the paper protocol."""

    rotation_error_max_deg: float = Field(gt=0.0)
    translation_error_max_m: float = Field(gt=0.0)
    comparison: Literal["strict_less_than"] = "strict_less_than"


class CameraLidarIsolationDeclaration(StrictModel):
    """Train/validation/test selection and leakage declaration."""

    provider_selection_split: str
    threshold_selection_split: str
    evaluation_split: str
    test_data_used_for_selection: bool = False
    evidence: str


class CameraLidarBenchmarkProtocol(StrictModel):
    """Immutable Borer-style perturbation and evaluation protocol."""

    schema_version: Literal[
        "slac.camera_lidar_benchmark_protocol/v0.1"
    ] = CAMERA_LIDAR_BENCHMARK_PROTOCOL_SCHEMA_VERSION
    protocol_id: str
    primary_source: str
    dataset_id: str
    problem_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    degrees_of_freedom: Literal["rotation_only", "six_dof"]
    frame_sampling: Literal["uniform_over_sequence"]
    frame_ids: list[str] = Field(min_length=1)
    perturbation_method: Literal["fibonacci_sphere"]
    perturbation_count: int = Field(gt=0)
    rotation_magnitude_deg: float = Field(gt=0.0)
    translation_magnitude_m: float = Field(ge=0.0)
    perturbations: list[CameraLidarPerturbation] = Field(min_length=1)
    hit: CameraLidarHitDefinition
    objective: Literal["per_frame_mean_mutual_information"]
    histogram_bins: int = Field(ge=8)
    min_visible_points: int = Field(ge=4)
    visibility: Literal["z_buffer_nearest_range"]
    optimizer: str
    optimizer_options: dict[str, float | int | str | bool]
    isolation: CameraLidarIsolationDeclaration
    metric_definitions: dict[str, str]
    provenance: CameraLidarArtifactProvenance

    @model_validator(mode="after")
    def check_explicit_protocol(self) -> CameraLidarBenchmarkProtocol:
        """Keep declared counts, frames, and DoF consistent."""

        if len(self.perturbations) != self.perturbation_count:
            raise ValueError("perturbation_count does not match perturbations")
        if len(self.frame_ids) != len(set(self.frame_ids)):
            raise ValueError("protocol frame IDs must be unique")
        if self.degrees_of_freedom == "rotation_only" and (
            self.translation_magnitude_m != 0.0
            or any(
                any(value != 0.0 for value in item.translation_m_xyz)
                for item in self.perturbations
            )
        ):
            raise ValueError("rotation-only protocol cannot perturb translation")
        return self

    def save(self, path: str | Path) -> None:
        """Save the protocol artifact as YAML or JSON."""

        _save(self, path)


class CalibrationCandidateEvaluation(StrictModel):
    """One objective evaluation in an optimizer trace."""

    evaluation: int = Field(gt=0)
    parameters: dict[str, float]
    objective: float
    mutual_information: float
    normalized_mutual_information: float
    evaluated_frame_count: int = Field(ge=0)
    accepted: bool


class CalibrationCandidateOutcome(StrictModel):
    """Independent error and hit decision for one candidate."""

    rotation_error_deg: float = Field(ge=0.0)
    translation_error_m: float = Field(ge=0.0)
    hit: bool
    hit_reason: str


class CalibrationCandidateTrace(StrictModel):
    """Schema-valid complete optimizer trace for one perturbation trial."""

    schema_version: Literal[
        "slac.calibration_candidate_trace/v0.1"
    ] = CALIBRATION_CANDIDATE_TRACE_SCHEMA_VERSION
    trace_id: str
    trial_id: str
    problem_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    solver: str
    solver_version: str
    status: str
    initial_transform_camera_lidar: TransformResult
    output_transform_camera_lidar: TransformResult
    evaluations: list[CalibrationCandidateEvaluation]
    stopping_reason: str
    runtime_seconds: float = Field(ge=0.0)
    outcome: CalibrationCandidateOutcome
    warnings: list[str] = Field(default_factory=list)
    provenance: CameraLidarArtifactProvenance

    @model_validator(mode="after")
    def check_evaluation_order(self) -> CalibrationCandidateTrace:
        """Require contiguous, unique evaluation numbers."""

        numbers = [item.evaluation for item in self.evaluations]
        if numbers and numbers != list(range(1, len(numbers) + 1)):
            raise ValueError("candidate evaluations must be contiguous from one")
        return self

    def save(self, path: str | Path) -> None:
        """Save the trace artifact as YAML or JSON."""

        _save(self, path)


class BullseyeTrialPoint(StrictModel):
    """Initial-to-output rotation vector for one Bull's Eye trial."""

    trial_id: str
    initial_rotation_deg_xyz: list[float] = Field(min_length=3, max_length=3)
    output_rotation_error_deg_xyz: list[float] = Field(min_length=3, max_length=3)
    hit: bool


class BullseyePlotArtifact(StrictModel):
    """Machine-readable source of a deterministic Bull's Eye SVG."""

    schema_version: Literal["slac.bullseye_plot/v0.1"] = BULLSEYE_PLOT_SCHEMA_VERSION
    plot_id: str
    problem_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    trace_sha256: dict[str, Sha256]
    svg_path: str
    svg_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    panels: list[Literal["roll_pitch", "roll_yaw", "pitch_yaw"]]
    hit_radius_deg: float = Field(gt=0.0)
    points: list[BullseyeTrialPoint] = Field(min_length=1)
    provenance: CameraLidarArtifactProvenance

    def save(self, path: str | Path) -> None:
        """Save the Bull's Eye sidecar as YAML or JSON."""

        _save(self, path)


def camera_lidar_problem_json_schema() -> dict[str, Any]:
    """Return the camera--LiDAR problem JSON schema."""

    return CameraLidarCalibrationProblem.model_json_schema()


def camera_lidar_benchmark_protocol_json_schema() -> dict[str, Any]:
    """Return the camera--LiDAR benchmark protocol JSON schema."""

    return CameraLidarBenchmarkProtocol.model_json_schema()


def calibration_candidate_trace_json_schema() -> dict[str, Any]:
    """Return the calibration candidate trace JSON schema."""

    return CalibrationCandidateTrace.model_json_schema()


def bullseye_plot_json_schema() -> dict[str, Any]:
    """Return the Bull's Eye plot sidecar JSON schema."""

    return BullseyePlotArtifact.model_json_schema()


def load_camera_lidar_problem(path: str | Path) -> CameraLidarCalibrationProblem:
    """Load and validate a camera--LiDAR problem."""

    return CameraLidarCalibrationProblem.model_validate(read_mapping(Path(path)))


def load_camera_lidar_benchmark_protocol(
    path: str | Path,
) -> CameraLidarBenchmarkProtocol:
    """Load and validate a camera--LiDAR benchmark protocol."""

    return CameraLidarBenchmarkProtocol.model_validate(read_mapping(Path(path)))


def load_calibration_candidate_trace(
    path: str | Path,
) -> CalibrationCandidateTrace:
    """Load and validate a candidate trace."""

    return CalibrationCandidateTrace.model_validate(read_mapping(Path(path)))


def load_bullseye_plot(path: str | Path) -> BullseyePlotArtifact:
    """Load and validate a Bull's Eye plot sidecar."""

    return BullseyePlotArtifact.model_validate(read_mapping(Path(path)))


def _save(model: StrictModel, path: str | Path) -> None:
    write_mapping(
        Path(path),
        model.model_dump(mode="json", exclude_none=True),
    )
