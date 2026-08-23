"""Frozen protocol for external Camera--LiDAR pose-initializer benchmarks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from calibrex.core.camera_lidar_artifacts import (
    CameraLidarArtifactProvenance,
    CameraLidarHitDefinition,
    CameraLidarIsolationDeclaration,
    CameraLidarPerturbation,
)
from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.result import StrictModel

CAMERA_LIDAR_POSE_INITIALIZER_PROTOCOL_SCHEMA_VERSION: Literal[
    "slac.camera_lidar_pose_initializer_protocol/v0.1"
] = "slac.camera_lidar_pose_initializer_protocol/v0.1"

CameraLidarPoseInitializerPartition = Literal["development", "evaluation"]


class CameraLidarPoseInitializerProvider(StrictModel):
    """Content-addressed identity for one external pose initializer."""

    provider_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    source_repository: str = Field(min_length=1)
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    license_spdx: str = Field(min_length=1)
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter_version: str = Field(min_length=1)
    adapter_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    neighbor_backend: Literal["torch", "cuda_extension"]
    training_datasets: list[str] = Field(min_length=1)
    evaluation_overlap: Literal["none", "possible", "unknown"]


class CameraLidarPoseInitializerBenchmarkProtocol(StrictModel):
    """Immutable frames, perturbations, provider, and leakage declaration."""

    schema_version: Literal["slac.camera_lidar_pose_initializer_protocol/v0.1"] = (
        CAMERA_LIDAR_POSE_INITIALIZER_PROTOCOL_SCHEMA_VERSION
    )
    protocol_id: str = Field(min_length=1)
    problem_id: str = Field(min_length=1)
    problem_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_id: str = Field(min_length=1)
    dataset_family: Literal["kitti_raw", "kitti_360"]
    dataset_license_spdx: str = Field(min_length=1)
    sequence_id: str = Field(min_length=1)
    camera_stream: Literal["image_00", "image_01", "image_02", "image_03"]
    partition: CameraLidarPoseInitializerPartition
    frame_sampling: str = Field(min_length=1)
    frame_ids: list[str] = Field(min_length=1)
    perturbation_convention: Literal["left_camera_frame_se3"] = "left_camera_frame_se3"
    perturbation_count: int = Field(gt=0)
    perturbations: list[CameraLidarPerturbation] = Field(min_length=1)
    aggregation: Literal["translation_mean_markley_quaternion_mean"] = (
        "translation_mean_markley_quaternion_mean"
    )
    hit: CameraLidarHitDefinition
    provider: CameraLidarPoseInitializerProvider
    isolation: CameraLidarIsolationDeclaration
    failure_policy: str = Field(min_length=1)
    metric_definitions: dict[str, str]
    provenance: CameraLidarArtifactProvenance

    @model_validator(mode="after")
    def check_frozen_matrix(self) -> CameraLidarPoseInitializerBenchmarkProtocol:
        """Require a closed trial matrix and explicit evaluation isolation."""

        if len(self.frame_ids) != len(set(self.frame_ids)):
            raise ValueError("pose-initializer protocol frame IDs must be unique")
        if len(self.perturbations) != self.perturbation_count:
            raise ValueError("perturbation_count does not match perturbations")
        trial_ids = [item.trial_id for item in self.perturbations]
        if len(trial_ids) != len(set(trial_ids)):
            raise ValueError("pose-initializer perturbation trial IDs must be unique")
        for item in self.perturbations:
            values = [*item.rotation_deg_xyz, *item.translation_m_xyz]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("pose-initializer perturbations must be finite")
            if not any(abs(value) > 0.0 for value in values):
                raise ValueError("pose-initializer perturbations must be non-zero")
        if self.isolation.evaluation_split != self.partition:
            raise ValueError("isolation evaluation_split must match protocol partition")
        if self.partition == "evaluation" and (
            self.isolation.test_data_used_for_selection
            or self.isolation.provider_selection_split == "evaluation"
            or self.isolation.threshold_selection_split == "evaluation"
        ):
            raise ValueError("evaluation protocol cannot declare evaluation-data selection")
        required_metrics = {"rotation_error_deg", "translation_error_m", "hit"}
        if set(self.metric_definitions) != required_metrics:
            raise ValueError(
                "pose-initializer metric definitions must contain exactly "
                "rotation_error_deg, translation_error_m, and hit"
            )
        return self

    def save(self, path: str | Path) -> None:
        """Save the frozen protocol as YAML or JSON."""

        write_mapping(
            Path(path),
            self.model_dump(mode="json", exclude_none=True),
        )


def load_camera_lidar_pose_initializer_protocol(
    path: str | Path,
) -> CameraLidarPoseInitializerBenchmarkProtocol:
    """Load and validate a frozen pose-initializer protocol."""

    return CameraLidarPoseInitializerBenchmarkProtocol.model_validate(read_mapping(Path(path)))


def camera_lidar_pose_initializer_protocol_json_schema() -> dict[str, Any]:
    """Return the standalone JSON Schema for pose-initializer protocols."""

    return CameraLidarPoseInitializerBenchmarkProtocol.model_json_schema()
