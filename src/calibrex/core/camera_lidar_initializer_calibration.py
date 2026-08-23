"""Schema-valid development lock for aggregate Camera--LiDAR PnP."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from calibrex.core.geometry import SE3
from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.probabilistic_correspondence import (
    CorrespondenceProviderIdentity,
    ProbabilisticPnpToolIdentity,
)
from calibrex.core.result import StrictModel, TransformResult

CAMERA_LIDAR_INITIALIZER_CALIBRATION_SCHEMA_VERSION: Literal[
    "slac.camera_lidar_initializer_calibration/v0.1"
] = "slac.camera_lidar_initializer_calibration/v0.1"
CAMERA_LIDAR_INITIALIZER_SELECTION_RULE_ID: Literal[
    "highest_confidence_then_lowest_ransac_threshold_passing_recovery/v0.1"
] = "highest_confidence_then_lowest_ransac_threshold_passing_recovery/v0.1"
CAMERA_LIDAR_AGGREGATE_FRAME_SELECTION_RULE_ID: Literal[
    "minimum_effective_confidence_and_frame_correspondence_support/v0.1"
] = "minimum_effective_confidence_and_frame_correspondence_support/v0.1"
CAMERA_LIDAR_AGGREGATE_PNP_METHOD: Literal[
    "opencv_probabilistic_aggregate_pnp_ransac/v0.3"
] = "opencv_probabilistic_aggregate_pnp_ransac/v0.3"

InitializerCalibrationStatus = Literal[
    "converged",
    "unavailable",
    "failed",
    "insufficient_correspondences",
    "unsupported_camera",
]


class CameraLidarInitializerRecoveryGate(StrictModel):
    """Prespecified development recovery and reproducibility requirements."""

    rotation_error_max_deg: float = Field(gt=0.0)
    translation_error_max_m: float = Field(gt=0.0)
    minimum_selected_frame_count: int = Field(ge=1)
    minimum_selected_correspondence_count: int = Field(ge=4)
    minimum_ransac_inlier_count: int = Field(ge=4)
    maximum_ransac_inlier_reprojection_rmse_px: float = Field(gt=0.0)
    maximum_seed_rotation_delta_deg: float = Field(ge=0.0)
    maximum_seed_translation_delta_m: float = Field(ge=0.0)


class CameraLidarInitializerSeedResult(StrictModel):
    """One deterministic OpenCV seed outcome for a grid candidate."""

    random_seed: int
    status: InitializerCalibrationStatus
    source_frame_ids: list[str] = Field(default_factory=list)
    selected_correspondence_count: int = Field(ge=0)
    ransac_inlier_count: int = Field(ge=0)
    ransac_inlier_reprojection_rmse_px: float | None = Field(
        default=None, ge=0.0
    )
    rotation_error_deg: float | None = Field(default=None, ge=0.0)
    translation_error_m: float | None = Field(default=None, ge=0.0)
    transform_camera_lidar: TransformResult | None = None
    reason: str

    @model_validator(mode="after")
    def check_outcome(self) -> CameraLidarInitializerSeedResult:
        """Require complete recovery metrics exactly for converged outcomes."""

        if len(self.source_frame_ids) != len(set(self.source_frame_ids)):
            raise ValueError("initializer source frame IDs must be unique")
        if self.ransac_inlier_count > self.selected_correspondence_count:
            raise ValueError("initializer RANSAC inliers exceed selected matches")
        converged = self.status == "converged"
        values = (
            self.ransac_inlier_reprojection_rmse_px,
            self.rotation_error_deg,
            self.translation_error_m,
            self.transform_camera_lidar,
        )
        if converged and any(value is None for value in values):
            raise ValueError("converged initializer seed metrics are incomplete")
        if not converged and any(value is not None for value in values):
            raise ValueError("non-converged initializer seed has pose metrics")
        return self


class CameraLidarInitializerCalibrationCandidate(StrictModel):
    """All seed outcomes and hard-gate summary for one solver setting."""

    candidate_id: str
    minimum_confidence: float = Field(ge=0.0, le=1.0)
    ransac_reprojection_threshold_px: float = Field(gt=0.0)
    seed_results: list[CameraLidarInitializerSeedResult] = Field(min_length=1)
    all_seeds_converged: bool
    minimum_selected_frame_count: int = Field(ge=0)
    minimum_selected_correspondence_count: int = Field(ge=0)
    minimum_ransac_inlier_count: int = Field(ge=0)
    maximum_ransac_inlier_reprojection_rmse_px: float | None = Field(
        default=None, ge=0.0
    )
    maximum_rotation_error_deg: float | None = Field(default=None, ge=0.0)
    maximum_translation_error_m: float | None = Field(default=None, ge=0.0)
    maximum_pairwise_seed_rotation_delta_deg: float | None = Field(
        default=None, ge=0.0
    )
    maximum_pairwise_seed_translation_delta_m: float | None = Field(
        default=None, ge=0.0
    )
    gate_pass: bool
    gate_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_summary(self) -> CameraLidarInitializerCalibrationCandidate:
        """Keep candidate summaries equivalent to their seed-level evidence."""

        seeds = [item.random_seed for item in self.seed_results]
        if len(seeds) != len(set(seeds)):
            raise ValueError("initializer candidate random seeds must be unique")
        expected_id = camera_lidar_initializer_candidate_id(
            self.minimum_confidence,
            self.ransac_reprojection_threshold_px,
        )
        if self.candidate_id != expected_id:
            raise ValueError("initializer candidate ID differs from its settings")
        all_converged = all(item.status == "converged" for item in self.seed_results)
        if self.all_seeds_converged != all_converged:
            raise ValueError("all_seeds_converged differs from seed outcomes")
        expected_minima = (
            min(len(item.source_frame_ids) for item in self.seed_results),
            min(item.selected_correspondence_count for item in self.seed_results),
            min(item.ransac_inlier_count for item in self.seed_results),
        )
        declared_minima = (
            self.minimum_selected_frame_count,
            self.minimum_selected_correspondence_count,
            self.minimum_ransac_inlier_count,
        )
        if declared_minima != expected_minima:
            raise ValueError("initializer candidate minima differ from seed evidence")
        summaries = (
            self.maximum_ransac_inlier_reprojection_rmse_px,
            self.maximum_rotation_error_deg,
            self.maximum_translation_error_m,
            self.maximum_pairwise_seed_rotation_delta_deg,
            self.maximum_pairwise_seed_translation_delta_m,
        )
        if all_converged and any(item is None for item in summaries):
            raise ValueError("converged initializer candidate summary is incomplete")
        if not all_converged and any(item is not None for item in summaries):
            raise ValueError("non-converged initializer candidate has pose summaries")
        if all_converged:
            expected = _candidate_maxima(self.seed_results)
            for label, declared, measured in zip(
                (
                    "inlier RMSE",
                    "rotation error",
                    "translation error",
                    "seed rotation delta",
                    "seed translation delta",
                ),
                summaries,
                expected,
                strict=True,
            ):
                if declared is None or not math.isclose(
                    declared, measured, rel_tol=0.0, abs_tol=1.0e-12
                ):
                    raise ValueError(
                        f"initializer candidate maximum {label} differs from seeds"
                    )
        if self.gate_pass == bool(self.gate_reasons):
            raise ValueError("gate_pass must be true exactly with no gate reasons")
        if len(self.gate_reasons) != len(set(self.gate_reasons)):
            raise ValueError("initializer candidate gate reasons must be unique")
        return self


class CameraLidarLockedInitializerOptions(StrictModel):
    """Exact provider-independent aggregate PnP options frozen on development."""

    method: Literal[
        "opencv_probabilistic_aggregate_pnp_ransac/v0.3"
    ] = CAMERA_LIDAR_AGGREGATE_PNP_METHOD
    adapter_version: str
    opencv_version: str
    frame_selection_rule_id: Literal[
        "minimum_effective_confidence_and_frame_correspondence_support/v0.1"
    ] = CAMERA_LIDAR_AGGREGATE_FRAME_SELECTION_RULE_ID
    initialization_source: Literal["none"] = "none"
    minimum_confidence: float = Field(ge=0.0, le=1.0)
    minimum_correspondences: int = Field(ge=4)
    minimum_frame_correspondences: int = Field(ge=4)
    minimum_frames: int = Field(ge=1)
    ransac_reprojection_threshold_px: float = Field(gt=0.0)
    ransac_confidence: float = Field(gt=0.0, lt=1.0)
    ransac_iterations: int = Field(ge=1)
    mahalanobis_inlier_threshold: float = Field(gt=0.0)
    evaluation_random_seeds: list[int] = Field(min_length=1)

    @field_validator("evaluation_random_seeds")
    @classmethod
    def check_seeds(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)):
            raise ValueError("locked initializer seeds must be unique")
        return value


class CameraLidarInitializerCalibrationProvenance(StrictModel):
    """Digest-bound inputs and exact command for an initializer lock."""

    generator: str
    generator_version: str
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    source_paths: list[str] = Field(min_length=2)
    source_sha256: dict[str, str] = Field(min_length=2)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    notes: list[str] = Field(default_factory=list)

    @field_validator("source_sha256")
    @classmethod
    def check_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        invalid = [
            key
            for key, digest in value.items()
            if len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ]
        if invalid:
            raise ValueError(f"invalid source SHA-256 values: {invalid}")
        return value


class CameraLidarInitializerCalibrationArtifact(StrictModel):
    """Reference-scored development grid and immutable aggregate-PnP lock."""

    schema_version: Literal[
        "slac.camera_lidar_initializer_calibration/v0.1"
    ] = CAMERA_LIDAR_INITIALIZER_CALIBRATION_SCHEMA_VERSION
    status: Literal["locked", "rejected"]
    calibration_id: str
    dataset_id: str
    sequence_id: str
    split_id: Literal["development"]
    problem_id: str
    problem_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    correspondence_artifact_id: str
    correspondence_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider: CorrespondenceProviderIdentity
    tool: ProbabilisticPnpToolIdentity
    method: Literal[
        "opencv_probabilistic_aggregate_pnp_ransac/v0.3"
    ] = CAMERA_LIDAR_AGGREGATE_PNP_METHOD
    adapter_version: str
    evaluation_dataset_ids_excluded: list[str] = Field(min_length=1)
    selection_rule_id: Literal[
        "highest_confidence_then_lowest_ransac_threshold_passing_recovery/v0.1"
    ] = CAMERA_LIDAR_INITIALIZER_SELECTION_RULE_ID
    frame_selection_rule_id: Literal[
        "minimum_effective_confidence_and_frame_correspondence_support/v0.1"
    ] = CAMERA_LIDAR_AGGREGATE_FRAME_SELECTION_RULE_ID
    random_seeds: list[int] = Field(min_length=1)
    minimum_correspondences: int = Field(ge=4)
    minimum_frame_correspondences: int = Field(ge=4)
    minimum_frames: int = Field(ge=1)
    ransac_confidence: float = Field(gt=0.0, lt=1.0)
    ransac_iterations: int = Field(ge=1)
    mahalanobis_inlier_threshold: float = Field(gt=0.0)
    recovery_gate: CameraLidarInitializerRecoveryGate
    candidates: list[CameraLidarInitializerCalibrationCandidate] = Field(
        min_length=2
    )
    selected_candidate_id: str | None = None
    locked_initializer_options: CameraLidarLockedInitializerOptions | None = None
    reference_pose_used_for_development_calibration: Literal[True] = True
    evaluation_data_used_for_selection: Literal[False] = False
    release_sota_claim_allowed: Literal[False] = False
    provenance: CameraLidarInitializerCalibrationProvenance

    @model_validator(mode="after")
    def check_lock(self) -> CameraLidarInitializerCalibrationArtifact:
        """Enforce the deterministic selection rule and absence of eval leakage."""

        if len(self.random_seeds) != len(set(self.random_seeds)):
            raise ValueError("initializer calibration seeds must be unique")
        if len(self.evaluation_dataset_ids_excluded) != len(
            set(self.evaluation_dataset_ids_excluded)
        ):
            raise ValueError("excluded initializer evaluation datasets must be unique")
        if self.dataset_id in self.evaluation_dataset_ids_excluded:
            raise ValueError("development dataset cannot be an excluded evaluation set")
        grid = [
            (item.minimum_confidence, item.ransac_reprojection_threshold_px)
            for item in self.candidates
        ]
        if grid != sorted(grid) or len(grid) != len(set(grid)):
            raise ValueError("initializer candidate grid must be unique and ascending")
        for candidate in self.candidates:
            if [item.random_seed for item in candidate.seed_results] != self.random_seeds:
                raise ValueError("candidate seed order differs from calibration seeds")
            reasons = camera_lidar_initializer_gate_reasons(
                candidate, self.recovery_gate
            )
            if candidate.gate_reasons != reasons:
                raise ValueError("candidate gate reasons differ from recovery gate")
        passing = [item for item in self.candidates if item.gate_pass]
        if self.status == "rejected":
            if passing:
                raise ValueError("rejected initializer calibration has passing candidates")
            if (
                self.selected_candidate_id is not None
                or self.locked_initializer_options is not None
            ):
                raise ValueError("rejected initializer calibration cannot carry a lock")
            return self
        if not passing:
            raise ValueError("locked initializer calibration requires a passing candidate")
        selected = sorted(
            passing,
            key=lambda item: (
                -item.minimum_confidence,
                item.ransac_reprojection_threshold_px,
            ),
        )[0]
        if self.selected_candidate_id != selected.candidate_id:
            raise ValueError("selected initializer candidate violates selection rule")
        locked = self.locked_initializer_options
        if locked is None:
            raise ValueError("locked initializer calibration requires runtime options")
        expected = (
            selected.minimum_confidence,
            selected.ransac_reprojection_threshold_px,
            self.minimum_correspondences,
            self.minimum_frame_correspondences,
            self.minimum_frames,
            self.ransac_confidence,
            self.ransac_iterations,
            self.mahalanobis_inlier_threshold,
            self.random_seeds,
            self.adapter_version,
        )
        declared = (
            locked.minimum_confidence,
            locked.ransac_reprojection_threshold_px,
            locked.minimum_correspondences,
            locked.minimum_frame_correspondences,
            locked.minimum_frames,
            locked.ransac_confidence,
            locked.ransac_iterations,
            locked.mahalanobis_inlier_threshold,
            locked.evaluation_random_seeds,
            locked.adapter_version,
        )
        if declared != expected:
            raise ValueError("locked initializer options differ from selected grid")
        return self

    def save(self, path: str | Path) -> None:
        """Save this calibration lock as schema-valid YAML or JSON."""

        write_mapping(Path(path), self.model_dump(mode="json", exclude_none=True))


def camera_lidar_initializer_candidate_id(
    minimum_confidence: float,
    ransac_reprojection_threshold_px: float,
) -> str:
    """Return the canonical identifier for one initializer grid setting."""

    return (
        f"confidence={minimum_confidence:.12g};"
        f"ransac_px={ransac_reprojection_threshold_px:.12g}"
    )


def camera_lidar_initializer_gate_reasons(
    candidate: CameraLidarInitializerCalibrationCandidate,
    gate: CameraLidarInitializerRecoveryGate,
) -> list[str]:
    """Evaluate one candidate against the prespecified recovery gate."""

    reasons: list[str] = []
    if not candidate.all_seeds_converged:
        reasons.append("seed_convergence_failed")
    if candidate.minimum_selected_frame_count < gate.minimum_selected_frame_count:
        reasons.append("minimum_selected_frame_count_failed")
    if (
        candidate.minimum_selected_correspondence_count
        < gate.minimum_selected_correspondence_count
    ):
        reasons.append("minimum_selected_correspondence_count_failed")
    if candidate.minimum_ransac_inlier_count < gate.minimum_ransac_inlier_count:
        reasons.append("minimum_ransac_inlier_count_failed")
    if (
        candidate.maximum_ransac_inlier_reprojection_rmse_px is None
        or candidate.maximum_ransac_inlier_reprojection_rmse_px
        > gate.maximum_ransac_inlier_reprojection_rmse_px
    ):
        reasons.append("maximum_ransac_inlier_rmse_failed")
    if (
        candidate.maximum_rotation_error_deg is None
        or candidate.maximum_rotation_error_deg >= gate.rotation_error_max_deg
    ):
        reasons.append("rotation_recovery_failed")
    if (
        candidate.maximum_translation_error_m is None
        or candidate.maximum_translation_error_m >= gate.translation_error_max_m
    ):
        reasons.append("translation_recovery_failed")
    if (
        candidate.maximum_pairwise_seed_rotation_delta_deg is None
        or candidate.maximum_pairwise_seed_rotation_delta_deg
        > gate.maximum_seed_rotation_delta_deg
    ):
        reasons.append("seed_rotation_stability_failed")
    if (
        candidate.maximum_pairwise_seed_translation_delta_m is None
        or candidate.maximum_pairwise_seed_translation_delta_m
        > gate.maximum_seed_translation_delta_m
    ):
        reasons.append("seed_translation_stability_failed")
    return reasons


def load_camera_lidar_initializer_calibration(
    path: str | Path,
) -> CameraLidarInitializerCalibrationArtifact:
    """Load and validate a development aggregate-PnP lock."""

    return CameraLidarInitializerCalibrationArtifact.model_validate(
        read_mapping(Path(path))
    )


def camera_lidar_initializer_calibration_json_schema() -> dict[str, Any]:
    """Return the standalone JSON Schema for aggregate-PnP locks."""

    return CameraLidarInitializerCalibrationArtifact.model_json_schema()


def _candidate_maxima(
    seed_results: list[CameraLidarInitializerSeedResult],
) -> tuple[float, float, float, float, float]:
    inlier_rmse = [
        _required_float(item.ransac_inlier_reprojection_rmse_px)
        for item in seed_results
    ]
    rotation_errors = [
        _required_float(item.rotation_error_deg) for item in seed_results
    ]
    translation_errors = [
        _required_float(item.translation_error_m) for item in seed_results
    ]
    transforms = [
        item.transform_camera_lidar.as_se3()
        for item in seed_results
        if item.transform_camera_lidar is not None
    ]
    rotation_deltas: list[float] = [0.0]
    translation_deltas: list[float] = [0.0]
    for left_index, left in enumerate(transforms):
        for right in transforms[left_index + 1 :]:
            rotation_delta, translation_delta = _transform_errors(left, right)
            rotation_deltas.append(rotation_delta)
            translation_deltas.append(translation_delta)
    return (
        max(inlier_rmse),
        max(rotation_errors),
        max(translation_errors),
        max(rotation_deltas),
        max(translation_deltas),
    )


def _transform_errors(left: SE3, right: SE3) -> tuple[float, float]:
    relative = right.compose(left.inverse())
    quaternion = relative.rotation_quat_xyzw
    rotation = 2.0 * math.degrees(
        math.acos(min(1.0, max(-1.0, abs(quaternion[3]))))
    )
    translation = math.sqrt(
        sum(
            (left.translation_m[index] - right.translation_m[index]) ** 2
            for index in range(3)
        )
    )
    return rotation, translation


def _required_float(value: float | None) -> float:
    if value is None:
        raise ValueError("converged initializer metric is absent")
    return value
