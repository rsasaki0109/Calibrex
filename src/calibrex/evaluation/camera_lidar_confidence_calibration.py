"""Lock probabilistic Camera--LiDAR thresholds on a development sequence."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from calibrex.core.camera_lidar_artifacts import load_camera_lidar_problem
from calibrex.core.camera_lidar_confidence_calibration import (
    CameraLidarConfidenceCalibrationArtifact,
    CameraLidarConfidenceCalibrationCandidate,
    CameraLidarConfidenceCalibrationProvenance,
    CameraLidarLockedRefinementOptions,
)
from calibrex.core.camera_lidar_correspondence_quality import (
    CameraLidarCorrespondenceQualitySummary,
    CameraLidarDistributionSummary,
    CameraLidarQualityPartition,
)
from calibrex.core.geometry import SE3
from calibrex.core.probabilistic_correspondence import (
    ProbabilisticCorrespondenceFrame,
    load_probabilistic_correspondence,
)
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.evaluation.camera_lidar_correspondence_quality import (
    summarize_camera_lidar_correspondence_quality_at_pose,
)
from calibrex.evaluation.holdout import split_indices
from calibrex.solvers.probabilistic_camera_lidar_refiner import (
    ProbabilisticCameraLidarRefinementOptions,
    project_camera_point,
)

CAMERA_LIDAR_CONFIDENCE_CALIBRATION_VERSION = (
    "calibrex.camera_lidar_confidence_calibration/v0.2"
)
DEFAULT_CONFIDENCE_THRESHOLDS: tuple[float, ...] = (
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
)


@dataclass(frozen=True)
class _GeometricDiagnostics:
    accepted_count: int
    inlier_count: int
    inlier_count_by_frame: dict[str, int]
    reprojection_error_px: CameraLidarDistributionSummary


def calibrate_camera_lidar_confidence(
    correspondence_path: str | Path,
    problem_path: str | Path,
    *,
    thresholds: Sequence[float] = DEFAULT_CONFIDENCE_THRESHOLDS,
    calibration_split_seeds: Sequence[int] = tuple(range(10)),
    evaluation_split_seeds: Sequence[int] = tuple(range(5)),
    holdout_ratio: float = 0.25,
    minimum_train_correspondences: int = 24,
    minimum_holdout_correspondences: int = 8,
    minimum_frame_correspondences: int = 4,
    minimum_frame_support_rate: float = 0.95,
    minimum_full_rank_frame_rate: float = 0.95,
    development_reprojection_inlier_threshold_px: float = 8.0,
    minimum_geometric_inlier_rate: float = 0.25,
    minimum_frame_geometric_inlier_count: int = 4,
    minimum_frame_geometric_support_rate: float = 0.95,
    evaluation_dataset_ids_excluded: Sequence[str],
    calibration_id: str | None = None,
    command: list[str] | None = None,
) -> CameraLidarConfidenceCalibrationArtifact:
    """Select the highest geometry-safe threshold without reading evaluation data."""

    correspondence_file = Path(correspondence_path).resolve()
    problem_file = Path(problem_path).resolve()
    correspondence = load_probabilistic_correspondence(correspondence_file)
    problem = load_camera_lidar_problem(problem_file)
    correspondence_digest = _required_digest(correspondence_file)
    problem_digest = _required_digest(problem_file)
    ordered_thresholds = _validate_thresholds(thresholds)
    calibration_seeds = _validate_seeds(calibration_split_seeds, "calibration")
    locked_seeds = _validate_seeds(evaluation_split_seeds, "evaluation")
    if not 0.0 < holdout_ratio < 1.0:
        raise ValueError("holdout_ratio must be in (0, 1)")
    if not 0.0 < minimum_frame_support_rate <= 1.0:
        raise ValueError("minimum_frame_support_rate must be in (0, 1]")
    if not 0.0 < minimum_full_rank_frame_rate <= 1.0:
        raise ValueError("minimum_full_rank_frame_rate must be in (0, 1]")
    if development_reprojection_inlier_threshold_px <= 0.0:
        raise ValueError("development reprojection threshold must be positive")
    if not 0.0 < minimum_geometric_inlier_rate <= 1.0:
        raise ValueError("minimum_geometric_inlier_rate must be in (0, 1]")
    if minimum_frame_geometric_inlier_count < 1:
        raise ValueError("minimum_frame_geometric_inlier_count must be positive")
    if not 0.0 < minimum_frame_geometric_support_rate <= 1.0:
        raise ValueError("minimum_frame_geometric_support_rate must be in (0, 1]")
    if problem.dataset_id != correspondence.dataset_id:
        raise ValueError("problem and correspondence dataset IDs differ")
    if problem.sequence_id != correspondence.sequence_id:
        raise ValueError("problem and correspondence sequence IDs differ")
    problem_frame_ids = [item.frame_id for item in problem.observations]
    frames = sorted(correspondence.frames, key=lambda item: item.frame_id)
    correspondence_frame_ids = [item.frame_id for item in frames]
    if sorted(problem_frame_ids) != correspondence_frame_ids:
        raise ValueError("problem and correspondence frame IDs differ")
    problem_splits = {item.split_id for item in problem.observations}
    if problem_splits != {"development"} or correspondence.split_id != "development":
        raise ValueError("confidence calibration accepts development split inputs only")
    reference = problem.reference_transform_camera_lidar
    if any(
        frame.camera_frame != reference.parent or frame.lidar_frame != reference.child
        for frame in frames
    ):
        raise ValueError("reference transform frames differ from correspondences")

    candidates: list[CameraLidarConfidenceCalibrationCandidate] = []
    for threshold in ordered_thresholds:
        summaries = []
        for seed in calibration_seeds:
            train_indices, holdout_indices = split_indices(
                len(frames), holdout_ratio, seed
            )
            partitions: dict[str, CameraLidarQualityPartition] = {
                frames[index].frame_id: "train" for index in train_indices
            }
            partitions.update(
                {frames[index].frame_id: "holdout" for index in holdout_indices}
            )
            summaries.append(
                summarize_camera_lidar_correspondence_quality_at_pose(
                    frames,
                    partitions,
                    reference.as_se3(),
                    minimum_confidence=threshold,
                    minimum_train_correspondences=minimum_train_correspondences,
                    minimum_holdout_correspondences=minimum_holdout_correspondences,
                    minimum_frame_correspondences=minimum_frame_correspondences,
                )
            )
        representative = summaries[0]
        minimum_train = min(
            item.train_accepted_correspondence_count for item in summaries
        )
        minimum_holdout = min(
            item.holdout_accepted_correspondence_count for item in summaries
        )
        geometry = _geometric_diagnostics(
            frames,
            reference.as_se3(),
            minimum_confidence=threshold,
            inlier_threshold_px=development_reprojection_inlier_threshold_px,
        )
        if geometry.accepted_count != representative.accepted_correspondence_count:
            raise ValueError(
                "development geometric diagnostics disagree with support diagnostics"
            )
        minimum_train_geometry = min(
            sum(
                geometry.inlier_count_by_frame[frames[index].frame_id]
                for index in split_indices(len(frames), holdout_ratio, seed)[0]
            )
            for seed in calibration_seeds
        )
        minimum_holdout_geometry = min(
            sum(
                geometry.inlier_count_by_frame[frames[index].frame_id]
                for index in split_indices(len(frames), holdout_ratio, seed)[1]
            )
            for seed in calibration_seeds
        )
        frames_meeting_geometry = sum(
            count >= minimum_frame_geometric_inlier_count
            for count in geometry.inlier_count_by_frame.values()
        )
        frame_geometric_rate = frames_meeting_geometry / len(frames)
        geometric_inlier_rate = (
            geometry.inlier_count / geometry.accepted_count
            if geometry.accepted_count
            else 0.0
        )
        reasons = _gate_reasons(
            representative,
            minimum_train=minimum_train,
            minimum_holdout=minimum_holdout,
            minimum_train_correspondences=minimum_train_correspondences,
            minimum_holdout_correspondences=minimum_holdout_correspondences,
            minimum_frame_support_rate=minimum_frame_support_rate,
            minimum_full_rank_frame_rate=minimum_full_rank_frame_rate,
            minimum_train_geometric_inlier_count=minimum_train_geometry,
            minimum_holdout_geometric_inlier_count=minimum_holdout_geometry,
            geometric_inlier_rate=geometric_inlier_rate,
            minimum_geometric_inlier_rate=minimum_geometric_inlier_rate,
            frame_geometric_support_rate=frame_geometric_rate,
            minimum_frame_geometric_support_rate=(
                minimum_frame_geometric_support_rate
            ),
        )
        candidates.append(
            CameraLidarConfidenceCalibrationCandidate(
                minimum_confidence=threshold,
                accepted_correspondence_count=(
                    representative.accepted_correspondence_count
                ),
                accepted_correspondence_rate=(
                    representative.accepted_correspondence_rate
                ),
                frames_meeting_minimum_count=(
                    representative.frames_meeting_minimum_count
                ),
                frames_meeting_minimum_rate=(
                    representative.frames_meeting_minimum_rate
                ),
                full_rank_frame_count=representative.full_rank_frame_count,
                full_rank_frame_rate=(
                    representative.full_rank_frame_count
                    / representative.frame_count
                ),
                minimum_train_correspondence_count_across_seeds=minimum_train,
                minimum_holdout_correspondence_count_across_seeds=minimum_holdout,
                aggregate_observability=representative.aggregate_observability,
                geometric_inlier_count=geometry.inlier_count,
                geometric_inlier_rate=geometric_inlier_rate,
                frames_meeting_geometric_minimum_count=frames_meeting_geometry,
                frames_meeting_geometric_minimum_rate=frame_geometric_rate,
                minimum_train_geometric_inlier_count_across_seeds=(
                    minimum_train_geometry
                ),
                minimum_holdout_geometric_inlier_count_across_seeds=(
                    minimum_holdout_geometry
                ),
                reprojection_error_px=geometry.reprojection_error_px,
                gate_pass=not reasons,
                gate_reasons=reasons,
            )
        )
    passing = [item.minimum_confidence for item in candidates if item.gate_pass]
    selected_confidence = max(passing) if passing else None
    solver_options = (
        ProbabilisticCameraLidarRefinementOptions(
            minimum_confidence=selected_confidence,
            holdout_ratio=holdout_ratio,
            minimum_train_correspondences=minimum_train_correspondences,
            minimum_holdout_correspondences=minimum_holdout_correspondences,
        )
        if selected_confidence is not None
        else None
    )
    excluded = list(evaluation_dataset_ids_excluded)
    source_paths = [str(correspondence_file), str(problem_file)]
    return CameraLidarConfidenceCalibrationArtifact(
        status="locked" if selected_confidence is not None else "rejected",
        calibration_id=(
            calibration_id
            or f"{problem.dataset_id}-probabilistic-confidence-v0.7"
        ),
        dataset_id=problem.dataset_id,
        sequence_id=problem.sequence_id,
        split_id="development",
        problem_id=problem.problem_id,
        problem_sha256=problem_digest,
        correspondence_artifact_id=correspondence.artifact_id,
        correspondence_artifact_sha256=correspondence_digest,
        provider=correspondence.provider,
        evaluation_dataset_ids_excluded=excluded,
        selection_rule_id=(
            "highest_threshold_passing_support_observability_and_geometry/v0.2"
        ),
        calibration_holdout_ratio=holdout_ratio,
        calibration_split_seeds=calibration_seeds,
        minimum_frame_correspondence_count=minimum_frame_correspondences,
        minimum_frame_support_rate=minimum_frame_support_rate,
        minimum_full_rank_frame_rate=minimum_full_rank_frame_rate,
        development_reprojection_inlier_threshold_px=(
            development_reprojection_inlier_threshold_px
        ),
        minimum_geometric_inlier_rate=minimum_geometric_inlier_rate,
        minimum_frame_geometric_inlier_count=(
            minimum_frame_geometric_inlier_count
        ),
        minimum_frame_geometric_support_rate=(
            minimum_frame_geometric_support_rate
        ),
        candidates=candidates,
        selected_minimum_confidence=selected_confidence,
        locked_refinement_options=(
            _locked_options(solver_options, locked_seeds)
            if solver_options is not None
            else None
        ),
        provenance=CameraLidarConfidenceCalibrationProvenance(
            generator="calibrex.evaluation.camera_lidar_confidence_calibration",
            generator_version=CAMERA_LIDAR_CONFIDENCE_CALIBRATION_VERSION,
            git_commit=git_commit(),
            command=command or [],
            source_paths=source_paths,
            source_sha256={
                str(correspondence_file): correspondence_digest,
                str(problem_file): problem_digest,
            },
            notes=[
                "only the declared development sequence was used for selection",
                "the reference pose is used only for development support and "
                "reprojection geometry",
                "a rejected calibration is retained as schema-valid evidence and "
                "cannot be used at runtime",
                "runtime candidate acceptance remains training-fit-only",
                "runtime holdout evidence never selects the published pose",
                "optimizer and acceptance thresholds are prespecified safety policy",
                "excluded evaluation dataset artifacts were not opened",
            ],
        ),
    )


def refinement_options_from_camera_lidar_confidence_calibration(
    calibration: CameraLidarConfidenceCalibrationArtifact,
    *,
    split_seed: int,
) -> ProbabilisticCameraLidarRefinementOptions:
    """Materialize exact runtime options from a validated development lock."""

    if calibration.schema_version != (
        "slac.camera_lidar_confidence_calibration/v0.2"
    ):
        raise ValueError(
            "legacy support-only confidence calibrations cannot be used at runtime"
        )
    locked = calibration.locked_refinement_options
    if calibration.status != "locked" or locked is None:
        raise ValueError("confidence calibration was rejected on development geometry")
    if split_seed not in locked.evaluation_split_seeds:
        raise ValueError(
            f"split seed {split_seed} is absent from the confidence calibration lock"
        )
    return ProbabilisticCameraLidarRefinementOptions(
        holdout_ratio=locked.holdout_ratio,
        split_seed=split_seed,
        minimum_confidence=locked.minimum_confidence,
        minimum_train_correspondences=locked.minimum_train_correspondences,
        minimum_holdout_correspondences=locked.minimum_holdout_correspondences,
        rotation_bound_deg=locked.rotation_bound_deg,
        translation_bound_m=locked.translation_bound_m,
        initial_rotation_step_deg=locked.initial_rotation_step_deg,
        initial_translation_step_m=locked.initial_translation_step_m,
        minimum_rotation_step_deg=locked.minimum_rotation_step_deg,
        minimum_translation_step_m=locked.minimum_translation_step_m,
        max_evaluations=locked.max_evaluations,
        cauchy_scale=locked.cauchy_scale,
        use_covariance=locked.use_covariance,
        use_outlier_probability=locked.use_outlier_probability,
        use_reliability=locked.use_reliability,
        acceptance_policy_id=locked.acceptance_policy_id,
        minimum_absolute_train_objective_improvement=(
            locked.minimum_absolute_train_objective_improvement
        ),
        minimum_relative_train_objective_improvement=(
            locked.minimum_relative_train_objective_improvement
        ),
        maximum_train_correspondence_loss_fraction=(
            locked.maximum_train_correspondence_loss_fraction
        ),
        maximum_accepted_bound_fraction=locked.maximum_accepted_bound_fraction,
    )


def _geometric_diagnostics(
    frames: Sequence[ProbabilisticCorrespondenceFrame],
    reference_transform_camera_lidar: SE3,
    *,
    minimum_confidence: float,
    inlier_threshold_px: float,
) -> _GeometricDiagnostics:
    residuals: list[float] = []
    inlier_count_by_frame: dict[str, int] = {}
    for frame in frames:
        frame_inliers = 0
        for item in frame.correspondences:
            confidence = item.reliability * (1.0 - item.outlier_probability)
            if confidence < minimum_confidence:
                continue
            point = reference_transform_camera_lidar.transform_point(
                item.point_lidar_m
            )
            if point[2] <= 1.0e-9:
                continue
            u_value, v_value = project_camera_point(point, frame)
            if not math.isfinite(u_value) or not math.isfinite(v_value):
                continue
            if not (
                0.0 <= u_value < frame.intrinsics.width
                and 0.0 <= v_value < frame.intrinsics.height
            ):
                continue
            residual = math.hypot(
                u_value - item.image_mean_px[0],
                v_value - item.image_mean_px[1],
            )
            residuals.append(residual)
            if residual <= inlier_threshold_px:
                frame_inliers += 1
        inlier_count_by_frame[frame.frame_id] = frame_inliers
    inlier_count = sum(inlier_count_by_frame.values())
    return _GeometricDiagnostics(
        accepted_count=len(residuals),
        inlier_count=inlier_count,
        inlier_count_by_frame=inlier_count_by_frame,
        reprojection_error_px=_distribution(residuals),
    )


def _distribution(values: Sequence[float]) -> CameraLidarDistributionSummary:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return CameraLidarDistributionSummary(count=0)
    return CameraLidarDistributionSummary(
        count=len(ordered),
        minimum=ordered[0],
        p10=_nearest_rank(ordered, 0.10),
        median=float(median(ordered)),
        p90=_nearest_rank(ordered, 0.90),
        p95=_nearest_rank(ordered, 0.95),
        maximum=ordered[-1],
        mean=sum(ordered) / len(ordered),
    )


def _nearest_rank(ordered: Sequence[float], quantile: float) -> float:
    rank = max(1, math.ceil(quantile * len(ordered)))
    return float(ordered[rank - 1])


def _gate_reasons(
    summary: CameraLidarCorrespondenceQualitySummary,
    *,
    minimum_train: int,
    minimum_holdout: int,
    minimum_train_correspondences: int,
    minimum_holdout_correspondences: int,
    minimum_frame_support_rate: float,
    minimum_full_rank_frame_rate: float,
    minimum_train_geometric_inlier_count: int,
    minimum_holdout_geometric_inlier_count: int,
    geometric_inlier_rate: float,
    minimum_geometric_inlier_rate: float,
    frame_geometric_support_rate: float,
    minimum_frame_geometric_support_rate: float,
) -> list[str]:
    reasons: list[str] = []
    if minimum_train < minimum_train_correspondences:
        reasons.append("minimum_train_support_failed")
    if minimum_holdout < minimum_holdout_correspondences:
        reasons.append("minimum_holdout_support_failed")
    if summary.frames_meeting_minimum_rate < minimum_frame_support_rate:
        reasons.append("frame_support_rate_failed")
    full_rank_rate = summary.full_rank_frame_count / summary.frame_count
    if full_rank_rate < minimum_full_rank_frame_rate:
        reasons.append("full_rank_frame_rate_failed")
    if summary.aggregate_observability.rank < 6:
        reasons.append("aggregate_information_rank_failed")
    condition = summary.aggregate_observability.condition_number
    if condition is not None and condition > 1.0e8:
        reasons.append("aggregate_information_condition_failed")
    if minimum_train_geometric_inlier_count < minimum_train_correspondences:
        reasons.append("minimum_train_geometric_inliers_failed")
    if minimum_holdout_geometric_inlier_count < minimum_holdout_correspondences:
        reasons.append("minimum_holdout_geometric_inliers_failed")
    if geometric_inlier_rate < minimum_geometric_inlier_rate:
        reasons.append("geometric_inlier_rate_failed")
    if frame_geometric_support_rate < minimum_frame_geometric_support_rate:
        reasons.append("frame_geometric_support_rate_failed")
    return reasons


def _locked_options(
    options: ProbabilisticCameraLidarRefinementOptions,
    evaluation_split_seeds: list[int],
) -> CameraLidarLockedRefinementOptions:
    return CameraLidarLockedRefinementOptions(
        minimum_confidence=options.minimum_confidence,
        holdout_ratio=options.holdout_ratio,
        evaluation_split_seeds=evaluation_split_seeds,
        minimum_train_correspondences=options.minimum_train_correspondences,
        minimum_holdout_correspondences=options.minimum_holdout_correspondences,
        rotation_bound_deg=options.rotation_bound_deg,
        translation_bound_m=options.translation_bound_m,
        initial_rotation_step_deg=options.initial_rotation_step_deg,
        initial_translation_step_m=options.initial_translation_step_m,
        minimum_rotation_step_deg=options.minimum_rotation_step_deg,
        minimum_translation_step_m=options.minimum_translation_step_m,
        max_evaluations=options.max_evaluations,
        cauchy_scale=options.cauchy_scale,
        use_covariance=True,
        use_outlier_probability=True,
        use_reliability=True,
        acceptance_policy_id="initializer_preserving_fit_only/v0.1",
        minimum_absolute_train_objective_improvement=(
            options.minimum_absolute_train_objective_improvement
        ),
        minimum_relative_train_objective_improvement=(
            options.minimum_relative_train_objective_improvement
        ),
        maximum_train_correspondence_loss_fraction=(
            options.maximum_train_correspondence_loss_fraction
        ),
        maximum_accepted_bound_fraction=options.maximum_accepted_bound_fraction,
    )


def _validate_thresholds(values: Sequence[float]) -> list[float]:
    thresholds = [float(value) for value in values]
    if len(thresholds) < 2:
        raise ValueError("confidence calibration requires at least two thresholds")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("confidence thresholds must be finite values in [0, 1]")
    if thresholds != sorted(thresholds) or len(thresholds) != len(set(thresholds)):
        raise ValueError("confidence thresholds must be unique and ascending")
    return thresholds


def _validate_seeds(values: Sequence[int], label: str) -> list[int]:
    seeds = [int(value) for value in values]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError(f"{label} split seeds must be non-empty and unique")
    return seeds


def _required_digest(path: Path) -> str:
    digest = sha256_path(path)
    if digest is None:
        raise ValueError(f"required calibration source is unreadable: {path}")
    return digest
