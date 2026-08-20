"""Empirical SE(3) uncertainty via deterministic temporal block resampling.

The refiner is refit on seeded block subsamples of the correspondence frames.
Within each resample, correspondences are also bootstrapped with replacement
so point-level variation contributes to the empirical spread.  Whole contiguous
temporal blocks are always kept together, so neighboring measurements that
share a scene or pose are never split across the train and holdout boundary.
The resulting SE(3) spread is reported as tangent-space intervals with
observed coverage against injected truth and an overconfident interval
control.  Per-axis intervals are widened by a Sidak correction so the joint
three-axis translation and rotation boxes hit the declared coverage target.
When resampling yields a degenerate zero-width spread, coverage policy falls
back to ``warn`` instead of failing the overconfidence control.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, cast

from calibrex.core.camera_lidar_artifacts import load_camera_lidar_problem
from calibrex.core.empirical_uncertainty import (
    TANGENT_AXES,
    EmpiricalAxisInterval,
    EmpiricalBlock,
    EmpiricalResampleIteration,
    EmpiricalSe3UncertaintyArtifact,
    EmpiricalUncertaintyProvenance,
    OverconfidenceControlResult,
)
from calibrex.core.geometry import SE3, normalize_quaternion_xyzw
from calibrex.core.probabilistic_correspondence import (
    ProbabilisticCorrespondenceFrame,
    ProbabilisticRefinementEvaluationArtifact,
    ProbabilisticRefinementIterationArtifact,
    ProbabilisticRefinementProvenance,
    ProbabilisticRefinementResultArtifact,
    load_probabilistic_correspondence,
)
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.result import TransformResult
from calibrex.solvers.probabilistic_camera_lidar_refiner import (
    ProbabilisticCameraLidarRefinementOptions,
    ProbabilisticCameraLidarRefinementResult,
    ProbabilisticCameraLidarRefiner,
)

UNCERTAINTY_RUN_VERSION = "calibrex.empirical_se3_uncertainty/v0.1"
_WEAK_THRESHOLD_RATIO = 0.1
_COVERAGE_TOLERANCE = 0.05
_DEGENERATE_HALF_WIDTH = 1.0e-12
_ROTATION_AXES = TANGENT_AXES[3:]
_TRANSLATION_AXES = TANGENT_AXES[:3]
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

LiteralPolicyStatus = Literal["pass", "warn", "fail", "inconclusive"]


def run_empirical_se3_uncertainty(
    correspondence_path: str | Path,
    initialization_problem_path: str | Path,
    *,
    result_directory: str | Path,
    command: str,
    block_length: int = 5,
    target_coverage: float = 0.9,
    resample_count: int = 20,
    seed: int = 0,
    fit_block_ratio: float = 0.6,
    overconfidence_scale: float = 0.1,
    assess_coverage: bool = True,
    options: ProbabilisticCameraLidarRefinementOptions | None = None,
    result_id: str | None = None,
) -> EmpiricalSe3UncertaintyArtifact:
    """Resample temporal blocks and report empirical SE(3) coverage evidence.

    When ``assess_coverage`` is false the problem reference is ignored and the
    artifact reports the empirical spread with an inconclusive policy, matching
    public workflows that have no independent ground truth.
    """

    if block_length < 1:
        raise ValueError("block_length must be at least one frame")
    if not 0.0 < target_coverage < 1.0:
        raise ValueError("target_coverage must be in (0, 1)")
    if resample_count < 2:
        raise ValueError("resample_count must be at least two")
    if not 0.0 < fit_block_ratio < 1.0:
        raise ValueError("fit_block_ratio must be in (0, 1)")
    if not 0.0 < overconfidence_scale < 1.0:
        raise ValueError("overconfidence_scale must be in (0, 1)")

    correspondence_file = Path(correspondence_path)
    problem_file = Path(initialization_problem_path)
    correspondence = load_probabilistic_correspondence(correspondence_file)
    problem = load_camera_lidar_problem(problem_file)
    correspondence_digest = _required_digest(correspondence_file)
    problem_digest = _required_digest(problem_file)
    problem_frame_ids = {item.frame_id for item in problem.observations}
    frames = tuple(correspondence.frames)
    if len({item.camera_frame for item in frames}) != 1 or len(
        {item.lidar_frame for item in frames}
    ) != 1:
        raise ValueError("all correspondence frames must share camera/LiDAR IDs")
    camera_frame = frames[0].camera_frame
    lidar_frame = frames[0].lidar_frame
    missing = sorted({item.frame_id for item in frames} - problem_frame_ids)
    if missing:
        raise ValueError(
            "correspondence frames are absent from initialization problem: "
            + ", ".join(missing)
        )
    ordered = sorted(frames, key=lambda item: (item.capture_time_ns, item.frame_id))
    if len(ordered) < 4:
        raise ValueError("empirical uncertainty needs at least four frames")
    if block_length > len(ordered):
        raise ValueError("block_length cannot exceed the frame count")

    blocks = _form_blocks(ordered, block_length)
    if len(blocks) < 2:
        raise ValueError("empirical uncertainty needs at least two temporal blocks")

    initial_transform = problem.initial_transform_camera_lidar.as_se3()
    reference = (
        None if not assess_coverage else problem.reference_transform_camera_lidar
    )
    if reference is not None and (
        reference.parent != camera_frame or reference.child != lidar_frame
    ):
        raise ValueError("reference transform frames do not match the correspondence")
    settings = options or ProbabilisticCameraLidarRefinementOptions()

    output_root = Path(result_directory)
    output_root.mkdir(parents=True, exist_ok=True)
    splits = _sample_splits(len(blocks), resample_count, fit_block_ratio, seed)
    by_id = {item.frame_id: item for item in ordered}
    iterations: list[EmpiricalResampleIteration] = []
    resample_output_digests: dict[str, str] = {}
    estimates: list[SE3] = []
    for iteration_index, (fit_indices, holdout_indices) in enumerate(splits):
        iteration_id = f"resample-{iteration_index:04d}"
        bootstrap_rng = random.Random(seed + iteration_index + 1_000_000)
        train = _bootstrap_train_frames(
            _frames_for_blocks(by_id, blocks, fit_indices),
            bootstrap_rng,
        )
        holdout = _frames_for_blocks(by_id, blocks, holdout_indices)
        result = ProbabilisticCameraLidarRefiner().solve_partitioned(
            train,
            holdout,
            initial_transform,
            settings,
        )
        succeeded = result.status in {"converged", "at_bound"}
        output = output_root / f"{iteration_id}.yaml"
        _iteration_artifact(
            iteration_id,
            correspondence.artifact_id,
            problem.problem_id,
            correspondence.provider,
            camera_frame,
            lidar_frame,
            result,
            reference=reference,
            correspondence_sha256=correspondence_digest,
            problem_sha256=problem_digest,
            command=command.split(),
        ).save(output)
        output_digest = _required_digest(output)
        resample_output_digests[iteration_id] = output_digest
        iterations.append(
            EmpiricalResampleIteration(
                iteration_id=iteration_id,
                status="success" if succeeded else "failed",
                solver_status=result.status,
                train_block_ids=[blocks[i].block_id for i in fit_indices],
                holdout_block_ids=[blocks[i].block_id for i in holdout_indices],
                transform_camera_lidar=(
                    _transform(camera_frame, lidar_frame, result.transform_camera_lidar)
                    if succeeded
                    else None
                ),
                output_sha256=output_digest,
                reason=None if succeeded else result.reason,
            )
        )
        if succeeded:
            estimates.append(result.transform_camera_lidar)

    if len(estimates) < 2:
        raise ValueError(
            "fewer than two successful resample refits; cannot form intervals"
        )

    mean_estimate = _mean_se3(estimates)
    mean_transform = _transform(camera_frame, lidar_frame, mean_estimate)
    estimate_deltas = [_estimate_delta(estimate, mean_estimate) for estimate in estimates]
    axis_intervals = _axis_intervals(estimate_deltas, target_coverage)
    interval_halfwidth_translation_m = float(
        sum(
            item.half_width for item in axis_intervals if item.axis in _TRANSLATION_AXES
        )
        / 3.0
    )
    interval_halfwidth_rotation_deg = float(
        sum(
            item.half_width for item in axis_intervals if item.axis in _ROTATION_AXES
        )
        / 3.0
    )
    weak_directions = [str(item.axis) for item in axis_intervals if item.weak_direction]

    warnings: list[str] = []
    if any(item.status == "failed" for item in iterations):
        warnings.append(
            "some resample refits failed and were excluded from the intervals"
        )
    if len(estimates) < resample_count:
        warnings.append(f"completed {len(estimates)}/{resample_count} refits")

    policy_status: LiteralPolicyStatus = "inconclusive"
    policy_reason = (
        "no injected reference truth is declared; coverage cannot be assessed"
    )
    reference_rotation_error_deg: float | None = None
    reference_translation_error_m: float | None = None
    observed_translation: float | None = None
    observed_rotation: float | None = None
    observed_joint: float | None = None
    coverage_score: float | None = None
    control: OverconfidenceControlResult | None = None

    if reference is not None:
        reference_se3 = reference.as_se3()
        reference_rotation_error_deg = _rotation_error_deg(mean_estimate, reference_se3)
        reference_translation_error_m = _translation_error_m(mean_estimate, reference_se3)
        reference_deltas = [
            _delta_to_reference(estimate, reference_se3) for estimate in estimates
        ]
        observed_translation = _observed_coverage(
            reference_deltas, axis_intervals, _TRANSLATION_AXES
        )
        observed_rotation = _observed_coverage(
            reference_deltas, axis_intervals, _ROTATION_AXES
        )
        observed_joint = _observed_coverage(
            reference_deltas, axis_intervals, TANGENT_AXES
        )
        coverage_score = min(observed_translation, observed_rotation)
        control = _overconfidence_control(
            reference_deltas,
            axis_intervals,
            overconfidence_scale,
            target_coverage,
        )
        policy_status, policy_reason = _policy(
            coverage_score,
            control,
            target_coverage,
            axis_intervals=axis_intervals,
        )

    return EmpiricalSe3UncertaintyArtifact(
        uncertainty_id=result_id
        or f"{correspondence.artifact_id}-{problem.problem_id}-empirical-uncertainty",
        calibration_family="probabilistic_multiframe_refinement/v0.2",
        parent_frame=camera_frame,
        child_frame=lidar_frame,
        tangent_ordering=list(TANGENT_AXES),
        block_length=block_length,
        block_count=len(blocks),
        resample_count=resample_count,
        seed=seed,
        target_coverage=target_coverage,
        fit_block_ratio=fit_block_ratio,
        blocks=blocks,
        iterations=iterations,
        sample_ids=[item.iteration_id for item in iterations],
        mean_estimate_transform=mean_transform,
        reference_transform=reference,
        reference_rotation_error_deg=reference_rotation_error_deg,
        reference_translation_error_m=reference_translation_error_m,
        axis_intervals=axis_intervals,
        observed_coverage_translation=observed_translation,
        observed_coverage_rotation=observed_rotation,
        observed_coverage_joint=observed_joint,
        coverage_score=coverage_score,
        interval_halfwidth_translation_m=interval_halfwidth_translation_m,
        interval_halfwidth_rotation_deg=interval_halfwidth_rotation_deg,
        weak_directions=weak_directions,
        overconfidence_control=control,
        policy_status=policy_status,
        policy_reason=policy_reason,
        warnings=warnings,
        provenance=EmpiricalUncertaintyProvenance(
            generator=__name__,
            generator_version=UNCERTAINTY_RUN_VERSION,
            git_commit=git_commit(),
            command=command.split(),
            source_sha256={
                str(correspondence_file): correspondence_digest,
                str(problem_file): problem_digest,
            },
            resample_output_sha256=resample_output_digests,
            data_verified=(
                correspondence.dataset_license_spdx is not None
                and bool(correspondence.provenance.input_sha256)
            ),
        ),
    )


def _form_blocks(frames: Sequence[Any], block_length: int) -> list[EmpiricalBlock]:
    blocks: list[EmpiricalBlock] = []
    for start in range(0, len(frames), block_length):
        chunk = list(frames[start : start + block_length])
        blocks.append(
            EmpiricalBlock(
                block_id=f"block-{len(blocks):04d}",
                frame_ids=[item.frame_id for item in chunk],
                capture_time_start_ns=chunk[0].capture_time_ns,
                capture_time_end_ns=chunk[-1].capture_time_ns,
            )
        )
    return blocks


def _sample_splits(
    block_count: int,
    resample_count: int,
    fit_block_ratio: float,
    seed: int,
) -> list[tuple[list[int], list[int]]]:
    fit_count = max(1, min(block_count - 1, round(block_count * fit_block_ratio)))
    splits: list[tuple[list[int], list[int]]] = []
    for iteration in range(resample_count):
        rng = random.Random(seed + iteration)
        fit = sorted(rng.sample(range(block_count), fit_count))
        hold = sorted(set(range(block_count)) - set(fit))
        splits.append((fit, hold))
    return splits


def _frames_for_blocks(
    by_id: dict[str, Any],
    blocks: list[EmpiricalBlock],
    block_indices: Sequence[int],
) -> list[Any]:
    frames: list[Any] = []
    for index in block_indices:
        for frame_id in blocks[index].frame_ids:
            frames.append(by_id[frame_id])
    return frames


def _bootstrap_train_frames(
    frames: Sequence[ProbabilisticCorrespondenceFrame],
    rng: random.Random,
) -> list[ProbabilisticCorrespondenceFrame]:
    """Resample correspondences with replacement within each train frame."""

    bootstrapped: list[ProbabilisticCorrespondenceFrame] = []
    for frame in frames:
        pool = frame.correspondences
        sampled = [
            pool[rng.randrange(len(pool))].model_copy(
                update={
                    "correspondence_id": f"{frame.frame_id}-boot-{index:04d}",
                }
            )
            for index in range(len(pool))
        ]
        bootstrapped.append(frame.model_copy(update={"correspondences": sampled}))
    return bootstrapped


def _mean_se3(estimates: Sequence[SE3]) -> SE3:
    count = len(estimates)
    mean_translation = (
        sum(estimate.translation_m[0] for estimate in estimates) / count,
        sum(estimate.translation_m[1] for estimate in estimates) / count,
        sum(estimate.translation_m[2] for estimate in estimates) / count,
    )
    mean_rotation = _mean_quaternion(
        [estimate.rotation_quat_xyzw for estimate in estimates]
    )
    return SE3(mean_translation, mean_rotation)


def _mean_quaternion(
    quaternions: Sequence[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    base = quaternions[0]
    accumulated = [0.0, 0.0, 0.0, 0.0]
    for quaternion in quaternions:
        dot = sum(
            base_value * value
            for base_value, value in zip(base, quaternion, strict=True)
        )
        sign = 1.0 if dot >= 0.0 else -1.0
        for index in range(4):
            accumulated[index] += sign * quaternion[index]
    return normalize_quaternion_xyzw(accumulated)


def _estimate_delta(
    estimate: SE3,
    mean_estimate: SE3,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    relative = mean_estimate.inverse().compose(estimate)
    translation = (
        estimate.translation_m[0] - mean_estimate.translation_m[0],
        estimate.translation_m[1] - mean_estimate.translation_m[1],
        estimate.translation_m[2] - mean_estimate.translation_m[2],
    )
    rotation = _rotation_vector_deg(relative.rotation_quat_xyzw)
    return translation, rotation


def _delta_to_reference(
    estimate: SE3,
    reference: SE3,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    relative = estimate.inverse().compose(reference)
    translation = (
        reference.translation_m[0] - estimate.translation_m[0],
        reference.translation_m[1] - estimate.translation_m[1],
        reference.translation_m[2] - estimate.translation_m[2],
    )
    rotation = _rotation_vector_deg(relative.rotation_quat_xyzw)
    return translation, rotation


def _rotation_vector_deg(
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    x, y, z, w = quaternion
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    vector_norm = math.sqrt(x * x + y * y + z * z)
    if vector_norm < 1.0e-12:
        return (0.0, 0.0, 0.0)
    angle_deg = math.degrees(2.0 * math.atan2(vector_norm, w))
    return (
        x / vector_norm * angle_deg,
        y / vector_norm * angle_deg,
        z / vector_norm * angle_deg,
    )


def _axis_intervals(
    deltas: Sequence[
        tuple[tuple[float, float, float], tuple[float, float, float]]
    ],
    target_coverage: float,
) -> list[EmpiricalAxisInterval]:
    axis_target = target_coverage ** (1.0 / 3.0)
    translation_columns = _columns(deltas, _TRANSLATION_AXES)
    rotation_columns = _columns(deltas, _ROTATION_AXES)
    translation_halfwidths = [
        _central_half_width(values, axis_target)
        for values in translation_columns
    ]
    rotation_halfwidths = [
        _central_half_width(values, axis_target) for values in rotation_columns
    ]
    translation_median = _median(translation_halfwidths)
    rotation_median = _median(rotation_halfwidths)
    intervals: list[EmpiricalAxisInterval] = []
    for axis, half_width in zip(_TRANSLATION_AXES, translation_halfwidths, strict=True):
        intervals.append(
            _interval(axis, "m", half_width, _is_weak(half_width, translation_median))
        )
    for axis, half_width in zip(_ROTATION_AXES, rotation_halfwidths, strict=True):
        intervals.append(
            _interval(axis, "deg", half_width, _is_weak(half_width, rotation_median))
        )
    return intervals


def _columns(
    deltas: Sequence[
        tuple[tuple[float, float, float], tuple[float, float, float]]
    ],
    axes: tuple[str, ...],
) -> list[list[float]]:
    translation = axes[0].startswith("translation")
    return [
        [item[0 if translation else 1][index] for item in deltas]
        for index in range(3)
    ]


def _central_half_width(values: Sequence[float], target_coverage: float) -> float:
    sorted_values = sorted(values)
    lower = _percentile(sorted_values, (1.0 - target_coverage) / 2.0)
    upper = _percentile(sorted_values, (1.0 + target_coverage) / 2.0)
    return (upper - lower) / 2.0


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = quantile * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _median(values: Sequence[float]) -> float:
    return _percentile(sorted(values), 0.5)


def _is_weak(half_width: float, family_median: float) -> bool:
    if family_median <= 0.0:
        return half_width <= 0.0
    return half_width < _WEAK_THRESHOLD_RATIO * family_median


def _interval(axis: str, unit: str, half_width: float, weak: bool) -> EmpiricalAxisInterval:
    return EmpiricalAxisInterval(
        axis=cast(Any, axis),
        unit=unit,
        lower=-half_width,
        upper=half_width,
        half_width=half_width,
        weak_direction=weak,
    )


def _observed_coverage(
    reference_deltas: Sequence[
        tuple[tuple[float, float, float], tuple[float, float, float]]
    ],
    axis_intervals: Sequence[EmpiricalAxisInterval],
    axes: Sequence[str],
) -> float:
    half_widths = {str(item.axis): item.half_width for item in axis_intervals}
    inside = 0
    for translation_delta, rotation_delta in reference_deltas:
        all_inside = True
        for axis in axes:
            index = _AXIS_INDEX[axis.split("_")[1]]
            if axis.startswith("translation"):
                delta = translation_delta[index]
            else:
                delta = rotation_delta[index]
            if abs(delta) > half_widths[axis]:
                all_inside = False
                break
        if all_inside:
            inside += 1
    return inside / len(reference_deltas)


def _overconfidence_control(
    reference_deltas: Sequence[
        tuple[tuple[float, float, float], tuple[float, float, float]]
    ],
    axis_intervals: Sequence[EmpiricalAxisInterval],
    scale: float,
    target_coverage: float,
) -> OverconfidenceControlResult:
    scaled_half_widths = {
        item.axis: item.half_width * scale for item in axis_intervals
    }
    scaled_intervals = [
        item.model_copy(update={"half_width": scaled_half_widths[item.axis]})
        for item in axis_intervals
    ]
    coverage_translation = _observed_coverage(
        reference_deltas, scaled_intervals, _TRANSLATION_AXES
    )
    coverage_rotation = _observed_coverage(
        reference_deltas, scaled_intervals, _ROTATION_AXES
    )
    coverage_joint = _observed_coverage(
        reference_deltas, scaled_intervals, TANGENT_AXES
    )
    refuted = min(coverage_translation, coverage_rotation) < target_coverage
    if refuted:
        policy_status: Literal["pass", "fail"] = "pass"
        reason = (
            f"shrunken ({scale:g}x) interval under-covers the reference; "
            "overconfidence is detectable"
        )
    else:
        policy_status = "fail"
        reason = (
            f"shrunken ({scale:g}x) interval still covers the reference; "
            "overconfidence is not detectable"
        )
    return OverconfidenceControlResult(
        control_id=f"overconfident-{scale:g}x",
        scale=scale,
        coverage_translation=coverage_translation,
        coverage_rotation=coverage_rotation,
        coverage_joint=coverage_joint,
        refuted=refuted,
        policy_status=policy_status,
        reason=reason,
    )


def _is_degenerate_spread(
    axis_intervals: Sequence[EmpiricalAxisInterval],
) -> bool:
    return all(
        item.half_width <= _DEGENERATE_HALF_WIDTH for item in axis_intervals
    )


def _policy(
    coverage_score: float,
    control: OverconfidenceControlResult,
    target_coverage: float,
    *,
    axis_intervals: Sequence[EmpiricalAxisInterval] | None = None,
) -> tuple[LiteralPolicyStatus, str]:
    if coverage_score >= target_coverage - _COVERAGE_TOLERANCE:
        if control.refuted:
            return (
                "pass",
                (
                    f"observed coverage {coverage_score:.3f} meets the "
                    f"{target_coverage:.2f} target and the overconfidence "
                    "control is refuted"
                ),
            )
        if axis_intervals is not None and _is_degenerate_spread(axis_intervals):
            return (
                "warn",
                (
                    "observed coverage meets the target but resample spread is "
                    "degenerate; empirical intervals are not identifiable and "
                    "the overconfidence control is not applicable"
                ),
            )
        return (
            "fail",
            (
                "observed coverage meets the target but the overconfidence "
                "control is not refuted; reported intervals cannot be "
                "distinguished from a much tighter interval"
            ),
        )
    if coverage_score >= target_coverage - 2.0 * _COVERAGE_TOLERANCE:
        return (
            "warn",
            (
                f"observed coverage {coverage_score:.3f} is slightly below the "
                f"{target_coverage:.2f} target"
            ),
        )
    return (
        "fail",
        (
            f"observed coverage {coverage_score:.3f} is below the "
            f"{target_coverage:.2f} target"
        ),
    )


def _rotation_error_deg(left: SE3, right: SE3) -> float:
    left_quaternion = left.rotation_quat_xyzw
    right_quaternion = right.rotation_quat_xyzw
    dot = abs(
        sum(
            left_value * right_value
            for left_value, right_value in zip(
                left_quaternion, right_quaternion, strict=True
            )
        )
    )
    return math.degrees(2.0 * math.acos(min(1.0, max(-1.0, dot))))


def _translation_error_m(left: SE3, right: SE3) -> float:
    return math.sqrt(
        sum(
            (left_value - right_value) ** 2
            for left_value, right_value in zip(
                left.translation_m, right.translation_m, strict=True
            )
        )
    )


def _transform(camera_frame: str, lidar_frame: str, value: SE3) -> TransformResult:
    return TransformResult(
        parent=camera_frame,
        child=lidar_frame,
        translation_m=list(value.translation_m),
        rotation_quat_xyzw=list(value.rotation_quat_xyzw),
    )


def _iteration_artifact(
    iteration_id: str,
    correspondence_artifact_id: str,
    initialization_problem_id: str,
    provider: Any,
    camera_frame: str,
    lidar_frame: str,
    result: ProbabilisticCameraLidarRefinementResult,
    *,
    reference: TransformResult | None,
    correspondence_sha256: str,
    problem_sha256: str,
    command: list[str],
) -> ProbabilisticRefinementResultArtifact:
    initial_transform = _transform(
        camera_frame, lidar_frame, result.initial_transform_camera_lidar
    )
    output_transform = _transform(
        camera_frame, lidar_frame, result.transform_camera_lidar
    )
    if reference is not None:
        reference_se3 = reference.as_se3()
        final_rotation_error = _rotation_error_deg(
            result.transform_camera_lidar, reference_se3
        )
        final_translation_error = _translation_error_m(
            result.transform_camera_lidar, reference_se3
        )
        initial_rotation_error = _rotation_error_deg(
            result.initial_transform_camera_lidar, reference_se3
        )
        initial_translation_error = _translation_error_m(
            result.initial_transform_camera_lidar, reference_se3
        )
    else:
        initial_rotation_error = None
        initial_translation_error = None
        final_rotation_error = None
        final_translation_error = None
    return ProbabilisticRefinementResultArtifact(
        result_id=iteration_id,
        status=result.status,
        reason=result.reason,
        correspondence_artifact_id=correspondence_artifact_id,
        initialization_problem_id=initialization_problem_id,
        initialization_source="problem_initial_transform",
        provider=provider,
        initial_transform_camera_lidar=initial_transform,
        transform_camera_lidar=output_transform,
        reference_transform_camera_lidar=reference,
        initial_rotation_error_deg=initial_rotation_error,
        final_rotation_error_deg=final_rotation_error,
        initial_translation_error_m=initial_translation_error,
        final_translation_error_m=final_translation_error,
        train_frame_ids=list(result.train_frame_ids),
        holdout_frame_ids=list(result.holdout_frame_ids),
        initial_train_evaluation=_evaluation(result.initial_train_evaluation),
        final_train_evaluation=_evaluation(result.final_train_evaluation),
        initial_holdout_evaluation=_evaluation(result.initial_holdout_evaluation),
        final_holdout_evaluation=_evaluation(result.final_holdout_evaluation),
        trace=[
            ProbabilisticRefinementIterationArtifact(
                evaluation=item.evaluation,
                delta_rotation_deg_xyz=list(item.delta_rotation_deg_xyz),
                delta_translation_m_xyz=list(item.delta_translation_m_xyz),
                objective=item.objective,
                accepted=item.accepted,
            )
            for item in result.trace
        ],
        options=cast(dict[str, int | float | str | bool], asdict(result.options)),
        provenance=ProbabilisticRefinementProvenance(
            generator=__name__,
            generator_version=UNCERTAINTY_RUN_VERSION,
            git_commit=git_commit(),
            command=command,
            correspondence_artifact_sha256=correspondence_sha256,
            initialization_problem_sha256=problem_sha256,
        ),
    )


def _evaluation(value: Any) -> ProbabilisticRefinementEvaluationArtifact:
    return ProbabilisticRefinementEvaluationArtifact(**asdict(value))


def _required_digest(path: Path) -> str:
    digest = sha256_path(path)
    if digest is None:
        raise ValueError(f"required uncertainty input is not readable: {path}")
    return digest


def _text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()