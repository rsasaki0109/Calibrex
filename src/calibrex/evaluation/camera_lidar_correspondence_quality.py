"""Post-hoc quality diagnostics for probabilistic Camera--LiDAR matches."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.camera_lidar_correspondence_quality import (
    CameraLidarCorrespondenceFrameQuality,
    CameraLidarCorrespondenceGateCounts,
    CameraLidarCorrespondenceQualityArtifact,
    CameraLidarCorrespondenceQualityProvenance,
    CameraLidarCorrespondenceQualitySummary,
    CameraLidarDistributionSummary,
    CameraLidarImageCoverage,
    CameraLidarPoseObservability,
    CameraLidarQualityGrade,
    CameraLidarQualityPartition,
    CameraLidarQualityPoseRole,
)
from calibrex.core.geometry import SE3
from calibrex.core.probabilistic_correspondence import (
    ProbabilisticCorrespondenceFrame,
    ProbabilisticImageCorrespondence,
    ProbabilisticRefinementResultArtifact,
    load_probabilistic_correspondence,
    load_probabilistic_refinement_result,
)
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.result import TransformResult
from calibrex.solvers.borer_six_dof_solver import apply_local_se3_delta
from calibrex.solvers.probabilistic_camera_lidar_refiner import project_camera_point

CAMERA_LIDAR_CORRESPONDENCE_QUALITY_VERSION = (
    "calibrex.camera_lidar_correspondence_quality/v0.1"
)
CameraLidarQualityAxis = Literal["rx", "ry", "rz", "tx", "ty", "tz"]
_AXES: tuple[CameraLidarQualityAxis, ...] = ("rx", "ry", "rz", "tx", "ty", "tz")
_PARAMETER_SCALES = (1.0, 1.0, 1.0, 0.1, 0.1, 0.1)
_FINITE_DIFFERENCE_STEPS = (1.0e-3, 1.0e-3, 1.0e-3, 1.0e-4, 1.0e-4, 1.0e-4)
_INFORMATION_RANK_RELATIVE_TOLERANCE = 1.0e-8
_WEAK_AXIS_INFORMATION_FRACTION = 0.01
_ILL_CONDITIONED_THRESHOLD = 1.0e8
_GRID_COLUMNS = 4
_GRID_ROWS = 3

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class _QualityOptions:
    minimum_confidence: float
    minimum_train_correspondences: int
    minimum_holdout_correspondences: int
    use_covariance: bool
    use_outlier_probability: bool
    use_reliability: bool


@dataclass(frozen=True)
class _FrameAnalysis:
    report: CameraLidarCorrespondenceFrameQuality
    information: FloatArray


def analyze_camera_lidar_correspondence_quality(
    correspondence_path: str | Path,
    refinement_result_path: str | Path,
    *,
    pose_role: CameraLidarQualityPoseRole = "initializer",
    report_id: str | None = None,
    minimum_frame_correspondences: int = 4,
    command: list[str] | None = None,
) -> CameraLidarCorrespondenceQualityArtifact:
    """Analyze result-bound correspondences without changing pose selection."""

    if minimum_frame_correspondences < 1:
        raise ValueError("minimum_frame_correspondences must be positive")
    correspondence_file = Path(correspondence_path).resolve()
    result_file = Path(refinement_result_path).resolve()
    correspondence = load_probabilistic_correspondence(correspondence_file)
    result = load_probabilistic_refinement_result(result_file)
    correspondence_digest = _required_digest(correspondence_file)
    result_digest = _required_digest(result_file)
    if result.correspondence_artifact_id != correspondence.artifact_id:
        raise ValueError("quality inputs reference different correspondence artifacts")
    if result.provenance.correspondence_artifact_sha256 != correspondence_digest:
        raise ValueError("refinement result correspondence digest does not match input")
    partition_by_frame: dict[str, CameraLidarQualityPartition] = dict.fromkeys(
        result.train_frame_ids, "train"
    )
    partition_by_frame.update(dict.fromkeys(result.holdout_frame_ids, "holdout"))
    correspondence_frame_ids = {item.frame_id for item in correspondence.frames}
    if set(partition_by_frame) != correspondence_frame_ids:
        raise ValueError("result split frame IDs do not match correspondence frames")
    transform_result = _pose_for_role(result, pose_role)
    frames = sorted(correspondence.frames, key=lambda item: item.frame_id)
    if any(
        item.camera_frame != transform_result.parent
        or item.lidar_frame != transform_result.child
        for item in frames
    ):
        raise ValueError("quality pose frames do not match correspondence frames")
    options = _quality_options(result.options)
    frame_analyses = [
        _analyze_frame(
            frame,
            partition_by_frame[frame.frame_id],
            transform_result.as_se3(),
            options,
            minimum_frame_correspondences,
        )
        for frame in frames
    ]
    summary = _summarize(
        frame_analyses,
        options,
        minimum_frame_correspondences,
    )
    source_paths = [str(correspondence_file), str(result_file)]
    return CameraLidarCorrespondenceQualityArtifact(
        report_id=(
            report_id
            or f"{result.result_id}-{pose_role}-correspondence-quality"
        ),
        dataset_id=correspondence.dataset_id,
        split_id=correspondence.split_id,
        correspondence_artifact_id=correspondence.artifact_id,
        correspondence_artifact_sha256=correspondence_digest,
        refinement_result_id=result.result_id,
        refinement_result_sha256=result_digest,
        provider=correspondence.provider,
        pose_role=pose_role,
        transform_camera_lidar=transform_result,
        minimum_confidence=options.minimum_confidence,
        use_covariance=options.use_covariance,
        use_outlier_probability=options.use_outlier_probability,
        use_reliability=options.use_reliability,
        minimum_frame_correspondence_count=minimum_frame_correspondences,
        summary=summary,
        frames=[item.report for item in frame_analyses],
        provenance=CameraLidarCorrespondenceQualityProvenance(
            source_paths=source_paths,
            source_sha256={
                str(correspondence_file): correspondence_digest,
                str(result_file): result_digest,
            },
            generator_version=CAMERA_LIDAR_CORRESPONDENCE_QUALITY_VERSION,
            git_commit=git_commit(),
            command=command or [],
            notes=[
                "report is post-hoc and never participates in candidate selection",
                "information columns use 1 deg and 0.1 m comparison scales",
                "holdout frame diagnostics remain evaluation-only evidence",
            ],
        ),
    )


def summarize_camera_lidar_correspondence_quality_at_pose(
    frames: Sequence[ProbabilisticCorrespondenceFrame],
    partition_by_frame: Mapping[str, CameraLidarQualityPartition],
    transform_camera_lidar: SE3,
    *,
    minimum_confidence: float,
    minimum_train_correspondences: int = 24,
    minimum_holdout_correspondences: int = 8,
    minimum_frame_correspondences: int = 4,
    use_covariance: bool = True,
    use_outlier_probability: bool = True,
    use_reliability: bool = True,
) -> CameraLidarCorrespondenceQualitySummary:
    """Summarize support and observability at an explicit development pose."""

    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be in [0, 1]")
    if minimum_train_correspondences < 4:
        raise ValueError("minimum_train_correspondences must be at least four")
    if minimum_holdout_correspondences < 4:
        raise ValueError("minimum_holdout_correspondences must be at least four")
    if minimum_frame_correspondences < 1:
        raise ValueError("minimum_frame_correspondences must be positive")
    ordered = sorted(frames, key=lambda item: item.frame_id)
    frame_ids = {item.frame_id for item in ordered}
    if set(partition_by_frame) != frame_ids:
        raise ValueError("partition frame IDs must exactly match correspondence frames")
    options = _QualityOptions(
        minimum_confidence=minimum_confidence,
        minimum_train_correspondences=minimum_train_correspondences,
        minimum_holdout_correspondences=minimum_holdout_correspondences,
        use_covariance=use_covariance,
        use_outlier_probability=use_outlier_probability,
        use_reliability=use_reliability,
    )
    analyses = [
        _analyze_frame(
            frame,
            partition_by_frame[frame.frame_id],
            transform_camera_lidar,
            options,
            minimum_frame_correspondences,
        )
        for frame in ordered
    ]
    return _summarize(analyses, options, minimum_frame_correspondences)


def _analyze_frame(
    frame: ProbabilisticCorrespondenceFrame,
    partition: CameraLidarQualityPartition,
    transform: SE3,
    options: _QualityOptions,
    minimum_frame_correspondences: int,
) -> _FrameAnalysis:
    total_count = len(frame.correspondences)
    below_confidence_count = 0
    behind_camera_count = 0
    invalid_projection_count = 0
    outside_image_count = 0
    accepted_confidence: list[float] = []
    camera_depths: list[float] = []
    residuals: list[float] = []
    projected_pixels: list[tuple[float, float]] = []
    information: FloatArray = np.zeros((6, 6), dtype=float)
    factor_count = 0
    confidences = [_confidence(item, options) for item in frame.correspondences]
    ranges = [_range(item.point_lidar_m) for item in frame.correspondences]
    perturbed = _perturbed_transforms(transform)
    for item, confidence in zip(frame.correspondences, confidences, strict=True):
        if confidence < options.minimum_confidence:
            below_confidence_count += 1
            continue
        point = transform.transform_point(item.point_lidar_m)
        if point[2] <= 1.0e-9:
            behind_camera_count += 1
            continue
        u, v = project_camera_point(point, frame)
        if not math.isfinite(u) or not math.isfinite(v):
            invalid_projection_count += 1
            continue
        if not (
            0.0 <= u < frame.intrinsics.width
            and 0.0 <= v < frame.intrinsics.height
        ):
            outside_image_count += 1
            continue
        accepted_confidence.append(confidence)
        camera_depths.append(point[2])
        projected_pixels.append((u, v))
        residuals.append(
            math.hypot(u - item.image_mean_px[0], v - item.image_mean_px[1])
        )
        jacobian = _projection_jacobian(item, frame, perturbed)
        if jacobian is not None:
            covariance_inverse = _covariance_inverse(item, options.use_covariance)
            scaled_jacobian = jacobian * np.asarray(_PARAMETER_SCALES, dtype=float)
            information += (
                confidence
                * scaled_jacobian.T
                @ covariance_inverse
                @ scaled_jacobian
            )
            factor_count += 1
    accepted_count = len(accepted_confidence)
    observability = _observability(information, factor_count)
    coverage = _image_coverage(projected_pixels, frame)
    reasons: list[str] = []
    grade: CameraLidarQualityGrade = "pass"
    if accepted_count < minimum_frame_correspondences:
        reasons.append("insufficient_frame_correspondence_support")
        grade = "fail"
    if observability.rank < 6:
        reasons.append("rank_deficient_local_information")
        grade = "fail"
    if coverage.occupancy_ratio < 0.25:
        reasons.append("narrow_image_grid_coverage")
        if grade == "pass":
            grade = "warn"
    if (
        observability.condition_number is not None
        and observability.condition_number > _ILL_CONDITIONED_THRESHOLD
    ):
        reasons.append("ill_conditioned_local_information")
        if grade == "pass":
            grade = "warn"
    if observability.weak_axes:
        reasons.append("weak_axis_information")
        if grade == "pass":
            grade = "warn"
    return _FrameAnalysis(
        report=CameraLidarCorrespondenceFrameQuality(
            frame_id=frame.frame_id,
            capture_time_ns=frame.capture_time_ns,
            partition=partition,
            gates=CameraLidarCorrespondenceGateCounts(
                total_count=total_count,
                below_confidence_count=below_confidence_count,
                behind_camera_count=behind_camera_count,
                invalid_projection_count=invalid_projection_count,
                outside_image_count=outside_image_count,
                accepted_count=accepted_count,
            ),
            confidence=_distribution(confidences),
            accepted_confidence=_distribution(accepted_confidence),
            lidar_range_m=_distribution(ranges),
            accepted_camera_depth_m=_distribution(camera_depths),
            reprojection_error_px=_distribution(residuals),
            image_coverage=coverage,
            observability=observability,
            grade=grade,
            gate_reasons=reasons,
        ),
        information=information,
    )


def _summarize(
    analyses: Sequence[_FrameAnalysis],
    options: _QualityOptions,
    minimum_frame_correspondences: int,
) -> CameraLidarCorrespondenceQualitySummary:
    reports = [item.report for item in analyses]
    frame_count = len(reports)
    total_count = sum(item.gates.total_count for item in reports)
    accepted_count = sum(item.gates.accepted_count for item in reports)
    train_accepted = sum(
        item.gates.accepted_count for item in reports if item.partition == "train"
    )
    holdout_accepted = sum(
        item.gates.accepted_count for item in reports if item.partition == "holdout"
    )
    frames_meeting_minimum = sum(
        item.gates.accepted_count >= minimum_frame_correspondences for item in reports
    )
    aggregate_information = np.sum(
        np.stack([item.information for item in analyses]), axis=0
    )
    aggregate_observability = _observability(
        aggregate_information,
        sum(item.report.observability.factor_count for item in analyses),
    )
    reasons: list[str] = []
    grade: CameraLidarQualityGrade = "pass"
    if train_accepted < options.minimum_train_correspondences:
        reasons.append("insufficient_train_correspondence_support")
        grade = "fail"
    if holdout_accepted < options.minimum_holdout_correspondences:
        reasons.append("insufficient_holdout_correspondence_support")
        grade = "fail"
    if aggregate_observability.rank < 6:
        reasons.append("rank_deficient_aggregate_information")
        grade = "fail"
    if frames_meeting_minimum / frame_count < 0.95:
        reasons.append("frame_support_rate_below_0_95")
        if grade == "pass":
            grade = "warn"
    full_rank_count = sum(item.observability.rank == 6 for item in reports)
    if full_rank_count / frame_count < 0.95:
        reasons.append("full_rank_frame_rate_below_0_95")
        if grade == "pass":
            grade = "warn"
    if (
        aggregate_observability.condition_number is not None
        and aggregate_observability.condition_number > _ILL_CONDITIONED_THRESHOLD
    ):
        reasons.append("ill_conditioned_aggregate_information")
        if grade == "pass":
            grade = "warn"
    return CameraLidarCorrespondenceQualitySummary(
        frame_count=frame_count,
        train_frame_count=sum(item.partition == "train" for item in reports),
        holdout_frame_count=sum(item.partition == "holdout" for item in reports),
        total_correspondence_count=total_count,
        accepted_correspondence_count=accepted_count,
        accepted_correspondence_rate=(accepted_count / total_count if total_count else 0.0),
        train_accepted_correspondence_count=train_accepted,
        holdout_accepted_correspondence_count=holdout_accepted,
        minimum_train_correspondence_count=options.minimum_train_correspondences,
        minimum_holdout_correspondence_count=options.minimum_holdout_correspondences,
        frames_meeting_minimum_count=frames_meeting_minimum,
        frames_meeting_minimum_rate=frames_meeting_minimum / frame_count,
        full_rank_frame_count=full_rank_count,
        aggregate_observability=aggregate_observability,
        rejection_counts={
            "below_confidence": sum(
                item.gates.below_confidence_count for item in reports
            ),
            "behind_camera": sum(item.gates.behind_camera_count for item in reports),
            "invalid_projection": sum(
                item.gates.invalid_projection_count for item in reports
            ),
            "outside_image": sum(item.gates.outside_image_count for item in reports),
        },
        grade=grade,
        gate_reasons=reasons,
    )


def _pose_for_role(
    result: ProbabilisticRefinementResultArtifact,
    pose_role: CameraLidarQualityPoseRole,
) -> TransformResult:
    initial = result.initial_transform_camera_lidar
    selected = result.transform_camera_lidar
    candidate = result.candidate_transform_camera_lidar
    if pose_role == "initializer":
        return initial
    if pose_role == "candidate":
        return candidate or selected
    return selected


def _quality_options(options: dict[str, int | float | str | bool]) -> _QualityOptions:
    return _QualityOptions(
        minimum_confidence=_float_option(options, "minimum_confidence"),
        minimum_train_correspondences=_int_option(
            options, "minimum_train_correspondences"
        ),
        minimum_holdout_correspondences=_int_option(
            options, "minimum_holdout_correspondences"
        ),
        use_covariance=_bool_option(options, "use_covariance"),
        use_outlier_probability=_bool_option(options, "use_outlier_probability"),
        use_reliability=_bool_option(options, "use_reliability"),
    )


def _float_option(
    options: dict[str, int | float | str | bool], key: str
) -> float:
    value = options.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"refinement result option {key!r} is not numeric")
    return float(value)


def _int_option(options: dict[str, int | float | str | bool], key: str) -> int:
    value = options.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"refinement result option {key!r} is not an integer")
    return value


def _bool_option(options: dict[str, int | float | str | bool], key: str) -> bool:
    value = options.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"refinement result option {key!r} is not boolean")
    return value


def _confidence(item: ProbabilisticImageCorrespondence, options: _QualityOptions) -> float:
    reliability = item.reliability if options.use_reliability else 1.0
    inlier_probability = (
        1.0 - item.outlier_probability if options.use_outlier_probability else 1.0
    )
    return reliability * inlier_probability


def _perturbed_transforms(transform: SE3) -> tuple[tuple[SE3, SE3], ...]:
    pairs: list[tuple[SE3, SE3]] = []
    for axis, step in enumerate(_FINITE_DIFFERENCE_STEPS):
        positive: FloatArray = np.zeros(6, dtype=float)
        negative: FloatArray = np.zeros(6, dtype=float)
        positive[axis] = step
        negative[axis] = -step
        pairs.append(
            (
                apply_local_se3_delta(transform, positive[:3], positive[3:]),
                apply_local_se3_delta(transform, negative[:3], negative[3:]),
            )
        )
    return tuple(pairs)


def _projection_jacobian(
    item: ProbabilisticImageCorrespondence,
    frame: ProbabilisticCorrespondenceFrame,
    perturbed_transforms: Sequence[tuple[SE3, SE3]],
) -> FloatArray | None:
    jacobian: FloatArray = np.zeros((2, 6), dtype=float)
    for axis, ((positive, negative), step) in enumerate(
        zip(perturbed_transforms, _FINITE_DIFFERENCE_STEPS, strict=True)
    ):
        positive_pixel = project_camera_point(
            positive.transform_point(item.point_lidar_m), frame
        )
        negative_pixel = project_camera_point(
            negative.transform_point(item.point_lidar_m), frame
        )
        if not all(math.isfinite(value) for value in (*positive_pixel, *negative_pixel)):
            return None
        jacobian[0, axis] = (positive_pixel[0] - negative_pixel[0]) / (2.0 * step)
        jacobian[1, axis] = (positive_pixel[1] - negative_pixel[1]) / (2.0 * step)
    return jacobian


def _covariance_inverse(
    item: ProbabilisticImageCorrespondence,
    use_covariance: bool,
) -> FloatArray:
    if not use_covariance:
        return np.eye(2, dtype=float)
    a, b, c, d = item.image_covariance_px2
    determinant = a * d - b * c
    inverse: FloatArray = (
        np.asarray([[d, -b], [-c, a]], dtype=float) / determinant
    )
    return inverse


def _observability(information: FloatArray, factor_count: int) -> CameraLidarPoseObservability:
    singular_values = np.linalg.svd(information, compute_uv=False)
    maximum = float(singular_values[0]) if singular_values.size else 0.0
    tolerance = maximum * _INFORMATION_RANK_RELATIVE_TOLERANCE
    rank = int(np.sum(singular_values > tolerance)) if maximum > 0.0 else 0
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if rank == 6 and singular_values[-1] > 0.0
        else None
    )
    diagonal = np.diag(information)
    maximum_diagonal = float(np.max(diagonal)) if diagonal.size else 0.0
    fractions: dict[str, float] = {
        axis: float(max(0.0, diagonal[index]) / maximum_diagonal)
        if maximum_diagonal > 0.0
        else 0.0
        for index, axis in enumerate(_AXES)
    }
    return CameraLidarPoseObservability(
        parameter_order=list(_AXES),
        parameter_scales=list(_PARAMETER_SCALES),
        factor_count=factor_count,
        singular_values=[float(max(0.0, value)) for value in singular_values],
        rank=rank,
        condition_number=condition_number,
        axis_information_fraction=fractions,
        weak_axes=[
            axis
            for axis in _AXES
            if fractions[axis] < _WEAK_AXIS_INFORMATION_FRACTION
        ],
    )


def _image_coverage(
    pixels: Sequence[tuple[float, float]],
    frame: ProbabilisticCorrespondenceFrame,
) -> CameraLidarImageCoverage:
    cells = {
        (
            min(
                _GRID_COLUMNS - 1,
                int(u / frame.intrinsics.width * _GRID_COLUMNS),
            ),
            min(
                _GRID_ROWS - 1,
                int(v / frame.intrinsics.height * _GRID_ROWS),
            ),
        )
        for u, v in pixels
    }
    if pixels:
        values_u = [item[0] for item in pixels]
        values_v = [item[1] for item in pixels]
        bounding_area = (
            (max(values_u) - min(values_u))
            * (max(values_v) - min(values_v))
            / (frame.intrinsics.width * frame.intrinsics.height)
        )
    else:
        bounding_area = 0.0
    return CameraLidarImageCoverage(
        grid_columns=_GRID_COLUMNS,
        grid_rows=_GRID_ROWS,
        occupied_cell_count=len(cells),
        occupancy_ratio=len(cells) / (_GRID_COLUMNS * _GRID_ROWS),
        normalized_bounding_box_area=min(1.0, max(0.0, bounding_area)),
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


def _range(point: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in point))


def _required_digest(path: Path) -> str:
    digest = sha256_path(path)
    if digest is None:
        raise ValueError(f"required quality-report input is not readable: {path}")
    return digest
