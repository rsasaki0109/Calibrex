"""Schema-valid continuous-time camera--LiDAR problems and results."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.probabilistic_correspondence import (
    CorrespondenceProviderIdentity,
)
from calibrex.core.result import StrictModel, TransformResult
from calibrex.data.depth import DepthCameraIntrinsics

CONTINUOUS_TIME_CAMERA_LIDAR_PROBLEM_SCHEMA_VERSION: Literal[
    "slac.continuous_time_camera_lidar_problem/v0.1"
] = "slac.continuous_time_camera_lidar_problem/v0.1"
CONTINUOUS_TIME_CAMERA_LIDAR_RESULT_SCHEMA_VERSION: Literal[
    "slac.continuous_time_camera_lidar_result/v0.1"
] = "slac.continuous_time_camera_lidar_result/v0.1"


class ContinuousTimeArtifactProvenance(StrictModel):
    """Generator and immutable input lineage."""

    generator: str
    generator_version: str
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class ContinuousTimeBodyTwist(StrictModel):
    """Body twist expressed in the declared body frame."""

    linear_velocity_body_mps: list[float] = Field(min_length=3, max_length=3)
    angular_velocity_body_radps: list[float] = Field(
        min_length=3, max_length=3
    )

    @model_validator(mode="after")
    def check_finite(self) -> ContinuousTimeBodyTwist:
        """Reject non-finite trajectory samples."""

        if not all(
            math.isfinite(value)
            for value in (
                self.linear_velocity_body_mps
                + self.angular_velocity_body_radps
            )
        ):
            raise ValueError("body twist must be finite")
        return self


class ContinuousTimeTrajectoryPoseArtifact(StrictModel):
    """One timestamped ``T_world_body`` trajectory knot."""

    timestamp_sec: float
    transform_world_body: TransformResult

    @model_validator(mode="after")
    def check_timestamp(self) -> ContinuousTimeTrajectoryPoseArtifact:
        """Reject non-finite trajectory timestamps."""

        if not math.isfinite(self.timestamp_sec):
            raise ValueError("trajectory timestamp must be finite")
        return self


class ContinuousTimeTrajectoryArtifact(StrictModel):
    """Provenance-pinned adapter input for non-constant body motion."""

    adapter: Literal["piecewise_se3_linear_slerp/v0.1"] = (
        "piecewise_se3_linear_slerp/v0.1"
    )
    world_frame: str
    body_frame: str
    interpolation: Literal["linear_translation_shortest_arc_slerp"] = (
        "linear_translation_shortest_arc_slerp"
    )
    extrapolation: Literal["reject"] = "reject"
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    poses: list[ContinuousTimeTrajectoryPoseArtifact] = Field(min_length=2)

    @model_validator(mode="after")
    def check_trajectory(self) -> ContinuousTimeTrajectoryArtifact:
        """Require ordered poses with consistent frame semantics."""

        timestamps = [item.timestamp_sec for item in self.poses]
        if any(
            right <= left
            for left, right in pairwise(timestamps)
        ):
            raise ValueError(
                "trajectory timestamps must be strictly increasing"
            )
        for pose in self.poses:
            if (
                pose.transform_world_body.parent != self.world_frame
                or pose.transform_world_body.child != self.body_frame
            ):
                raise ValueError("trajectory pose frame IDs are inconsistent")
        return self


class ContinuousTimeImageCorrespondence(StrictModel):
    """One per-point firing time and probabilistic image observation."""

    correspondence_id: str
    point_lidar_m: list[float] = Field(min_length=3, max_length=3)
    capture_offset_sec: float
    image_mean_px: list[float] = Field(min_length=2, max_length=2)
    image_covariance_px2: list[float] = Field(min_length=4, max_length=4)
    outlier_probability: float = Field(ge=0.0, le=1.0)
    reliability: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_distribution(self) -> ContinuousTimeImageCorrespondence:
        """Require finite values and positive-definite covariance."""

        values = (
            *self.point_lidar_m,
            self.capture_offset_sec,
            *self.image_mean_px,
            *self.image_covariance_px2,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("timed correspondence values must be finite")
        a, b, c, d = self.image_covariance_px2
        if not math.isclose(b, c, rel_tol=1.0e-9, abs_tol=1.0e-12):
            raise ValueError("image covariance must be symmetric")
        if a <= 0.0 or d <= 0.0 or a * d - b * c <= 0.0:
            raise ValueError("image covariance must be positive definite")
        return self


class ContinuousTimeCaptureArtifact(StrictModel):
    """One camera exposure with timed LiDAR points and supplied motion."""

    capture_id: str
    lidar_message_stamp_sec: float
    camera_exposure_time_sec: float
    camera_frame: str
    lidar_frame: str
    body_frame: str
    camera: DepthCameraIntrinsics
    transform_body_lidar: TransformResult
    twist: ContinuousTimeBodyTwist
    rolling_shutter_readout_sec: float = Field(default=0.0, ge=0.0)
    rolling_shutter_direction: Literal[
        "top_to_bottom", "bottom_to_top"
    ] = "top_to_bottom"
    correspondences: list[ContinuousTimeImageCorrespondence] = Field(
        min_length=4
    )

    @model_validator(mode="after")
    def check_capture(self) -> ContinuousTimeCaptureArtifact:
        """Check frame convention, timestamps, and correspondence IDs."""

        if not math.isfinite(self.lidar_message_stamp_sec) or not math.isfinite(
            self.camera_exposure_time_sec
        ):
            raise ValueError("capture timestamps must be finite")
        if (
            self.transform_body_lidar.parent != self.body_frame
            or self.transform_body_lidar.child != self.lidar_frame
        ):
            raise ValueError("transform_body_lidar frame IDs are inconsistent")
        identifiers = [item.correspondence_id for item in self.correspondences]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("correspondence IDs must be unique within a capture")
        if self.camera.projection != "pinhole":
            raise ValueError("continuous-time artifact v0.1 supports pinhole")
        return self


class ContinuousTimeCapturePolicyArtifact(StrictModel):
    """Explicit per-point timestamp and clock convention."""

    point_offset_unit: Literal["seconds"]
    stamp_reference: Literal["scan_start", "scan_midpoint", "scan_end"]
    sensor_time_offset_sec: float
    time_convention: Literal[
        "lidar_sensor_time + dt_lidar = camera_reference_time"
    ] = "lidar_sensor_time + dt_lidar = camera_reference_time"


class ContinuousTimeCameraLidarProblemArtifact(StrictModel):
    """Digest-pinned joint extrinsic/clock-refinement input."""

    schema_version: Literal[
        "slac.continuous_time_camera_lidar_problem/v0.1"
    ] = CONTINUOUS_TIME_CAMERA_LIDAR_PROBLEM_SCHEMA_VERSION
    problem_id: str
    dataset_id: str
    dataset_license_spdx: str | None = None
    split_id: str
    initial_transform_camera_lidar: TransformResult
    reference_transform_camera_lidar: TransformResult | None = None
    reference_time_offset_sec: float | None = None
    capture_time_policy: ContinuousTimeCapturePolicyArtifact
    body_trajectory: ContinuousTimeTrajectoryArtifact | None = None
    captures: list[ContinuousTimeCaptureArtifact] = Field(min_length=2)
    correspondence_provider: CorrespondenceProviderIdentity | None = None
    split_policy: Literal["deterministic_disjoint_holdout"]
    options: dict[str, int | float | str | bool]
    provenance: ContinuousTimeArtifactProvenance

    @model_validator(mode="after")
    def check_problem(self) -> ContinuousTimeCameraLidarProblemArtifact:
        """Require consistent frames and unique capture IDs."""

        identifiers = [item.capture_id for item in self.captures]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("continuous-time capture IDs must be unique")
        camera_frames = {item.camera_frame for item in self.captures}
        lidar_frames = {item.lidar_frame for item in self.captures}
        body_frames = {item.body_frame for item in self.captures}
        if (
            len(camera_frames) != 1
            or len(lidar_frames) != 1
            or len(body_frames) != 1
        ):
            raise ValueError(
                "all captures must use one camera/LiDAR/body frame tuple"
            )
        if (
            self.initial_transform_camera_lidar.parent not in camera_frames
            or self.initial_transform_camera_lidar.child not in lidar_frames
        ):
            raise ValueError("initial transform frame IDs are inconsistent")
        if (
            (self.reference_transform_camera_lidar is None)
            != (self.reference_time_offset_sec is None)
        ):
            raise ValueError(
                "reference extrinsic and time offset must be supplied together"
            )
        if self.reference_transform_camera_lidar is not None and (
            self.reference_transform_camera_lidar.parent not in camera_frames
            or self.reference_transform_camera_lidar.child not in lidar_frames
        ):
            raise ValueError("reference transform frame IDs are inconsistent")
        if (
            self.reference_time_offset_sec is not None
            and not math.isfinite(self.reference_time_offset_sec)
        ):
            raise ValueError("reference time offset must be finite")
        if (
            self.body_trajectory is not None
            and self.body_trajectory.body_frame not in body_frames
        ):
            raise ValueError(
                "trajectory body frame is inconsistent with captures"
            )
        return self

    def save(self, path: str | Path) -> None:
        """Save the continuous-time problem as YAML or JSON."""

        _save(self, path)


class ContinuousTimeEvaluationArtifact(StrictModel):
    """Schema-valid uncertainty-normalized evaluation."""

    objective: float
    weighted_reprojection_rmse_px: float | None = Field(default=None, ge=0.0)
    mean_mahalanobis_error: float | None = Field(default=None, ge=0.0)
    valid_correspondence_count: int = Field(ge=0)
    skipped_correspondence_count: int = Field(ge=0)


class ContinuousTimeIterationArtifact(StrictModel):
    """One retained optimizer candidate."""

    evaluation: int = Field(gt=0)
    delta_rotation_deg_xyz: list[float] = Field(min_length=3, max_length=3)
    delta_translation_m_xyz: list[float] = Field(min_length=3, max_length=3)
    time_offset_sec: float
    objective: float
    accepted: bool


class ContinuousTimeCameraLidarResultArtifact(StrictModel):
    """Schema-valid joint extrinsic/time result and disjoint evidence."""

    schema_version: Literal[
        "slac.continuous_time_camera_lidar_result/v0.1"
    ] = CONTINUOUS_TIME_CAMERA_LIDAR_RESULT_SCHEMA_VERSION
    result_id: str
    status: Literal[
        "converged",
        "max_evaluations",
        "insufficient_correspondences",
        "missing_holdout",
        "degenerate_motion",
        "at_bound",
    ]
    reason: str
    initial_transform_camera_lidar: TransformResult
    transform_camera_lidar: TransformResult
    initial_time_offset_sec: float
    estimated_time_offset_sec: float
    reference_transform_camera_lidar: TransformResult | None = None
    reference_time_offset_sec: float | None = None
    initial_rotation_error_deg: float | None = Field(default=None, ge=0.0)
    final_rotation_error_deg: float | None = Field(default=None, ge=0.0)
    initial_translation_error_m: float | None = Field(default=None, ge=0.0)
    final_translation_error_m: float | None = Field(default=None, ge=0.0)
    initial_time_offset_error_sec: float | None = Field(default=None, ge=0.0)
    final_time_offset_error_sec: float | None = Field(default=None, ge=0.0)
    train_capture_ids: list[str]
    holdout_capture_ids: list[str]
    initial_train_evaluation: ContinuousTimeEvaluationArtifact
    final_train_evaluation: ContinuousTimeEvaluationArtifact
    initial_holdout_evaluation: ContinuousTimeEvaluationArtifact
    final_holdout_evaluation: ContinuousTimeEvaluationArtifact
    time_sensitivity_px_per_sec: float = Field(ge=0.0)
    time_observability_rank: int = Field(ge=0, le=1)
    trajectory_model: Literal["constant_body_twist", "piecewise_se3"] = (
        "constant_body_twist"
    )
    trace: list[ContinuousTimeIterationArtifact]
    options: dict[str, int | float | str | bool]
    method: Literal[
        "continuous_time_probabilistic_camera_lidar/v0.1"
    ] = "continuous_time_probabilistic_camera_lidar/v0.1"
    clock_convention: Literal[
        "lidar_sensor_time + dt_lidar = camera_reference_time"
    ] = "lidar_sensor_time + dt_lidar = camera_reference_time"
    provenance: ContinuousTimeArtifactProvenance

    @model_validator(mode="after")
    def check_result(self) -> ContinuousTimeCameraLidarResultArtifact:
        """Require disjoint splits and contiguous trace evaluations."""

        if set(self.train_capture_ids).intersection(self.holdout_capture_ids):
            raise ValueError("train and holdout capture IDs must be disjoint")
        reference_values = (
            self.reference_transform_camera_lidar,
            self.reference_time_offset_sec,
            self.initial_rotation_error_deg,
            self.final_rotation_error_deg,
            self.initial_translation_error_m,
            self.final_translation_error_m,
            self.initial_time_offset_error_sec,
            self.final_time_offset_error_sec,
        )
        if any(value is not None for value in reference_values) and any(
            value is None for value in reference_values
        ):
            raise ValueError(
                "continuous-time reference and all error fields are one block"
            )
        evaluations = [item.evaluation for item in self.trace]
        if evaluations and evaluations != list(range(1, len(evaluations) + 1)):
            raise ValueError("continuous-time trace must be contiguous from one")
        return self

    def save(self, path: str | Path) -> None:
        """Save the continuous-time result as YAML or JSON."""

        _save(self, path)


def continuous_time_camera_lidar_problem_json_schema() -> dict[str, Any]:
    """Return the continuous-time problem JSON schema."""

    return ContinuousTimeCameraLidarProblemArtifact.model_json_schema()


def continuous_time_camera_lidar_result_json_schema() -> dict[str, Any]:
    """Return the continuous-time result JSON schema."""

    return ContinuousTimeCameraLidarResultArtifact.model_json_schema()


def load_continuous_time_camera_lidar_problem(
    path: str | Path,
) -> ContinuousTimeCameraLidarProblemArtifact:
    """Load and validate a continuous-time problem."""

    return ContinuousTimeCameraLidarProblemArtifact.model_validate(
        read_mapping(Path(path))
    )


def load_continuous_time_camera_lidar_result(
    path: str | Path,
) -> ContinuousTimeCameraLidarResultArtifact:
    """Load and validate a continuous-time result."""

    return ContinuousTimeCameraLidarResultArtifact.model_validate(
        read_mapping(Path(path))
    )


def _save(model: StrictModel, path: str | Path) -> None:
    write_mapping(
        Path(path),
        model.model_dump(mode="json", exclude_none=True),
    )
