"""Artifact-backed D2D-initialized probabilistic pose refinement."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from calibrex.core.camera_lidar_artifacts import (
    load_calibration_candidate_trace,
    load_camera_lidar_problem,
)
from calibrex.core.camera_lidar_confidence_calibration import (
    load_camera_lidar_confidence_calibration,
)
from calibrex.core.probabilistic_correspondence import (
    ProbabilisticRefinementAcceptanceArtifact,
    ProbabilisticRefinementEvaluationArtifact,
    ProbabilisticRefinementIterationArtifact,
    ProbabilisticRefinementProvenance,
    ProbabilisticRefinementResultArtifact,
    load_probabilistic_correspondence,
)
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.result import TransformResult
from calibrex.evaluation.camera_lidar_confidence_calibration import (
    refinement_options_from_camera_lidar_confidence_calibration,
)
from calibrex.solvers.probabilistic_camera_lidar_refiner import (
    ProbabilisticCameraLidarRefinementOptions,
    ProbabilisticCameraLidarRefinementResult,
    ProbabilisticCameraLidarRefiner,
    ProbabilisticPoseEvaluation,
)

PROBABILISTIC_CAMERA_LIDAR_RUN_VERSION = "0.3"


def run_probabilistic_camera_lidar_refinement(
    correspondence_path: str | Path,
    initialization_problem_path: str | Path,
    *,
    initialization_trace_path: str | Path | None = None,
    confidence_calibration_path: str | Path | None = None,
    result_id: str | None = None,
    options: ProbabilisticCameraLidarRefinementOptions | None = None,
    command: list[str] | None = None,
) -> ProbabilisticRefinementResultArtifact:
    """Run refinement from a problem pose or validated D2D output trace."""

    correspondence_file = Path(correspondence_path)
    problem_file = Path(initialization_problem_path)
    correspondence = load_probabilistic_correspondence(correspondence_file)
    problem = load_camera_lidar_problem(problem_file)
    if correspondence.dataset_id != problem.dataset_id:
        raise ValueError(
            "probabilistic correspondence and initialization problem dataset IDs differ"
        )
    correspondence_digest = sha256_path(correspondence_file)
    problem_digest = sha256_path(problem_file)
    if correspondence_digest is None or problem_digest is None:
        raise ValueError("probabilistic refinement inputs must be readable files")
    problem_frame_ids = {item.frame_id for item in problem.observations}
    correspondence_frame_ids = {item.frame_id for item in correspondence.frames}
    missing = sorted(correspondence_frame_ids - problem_frame_ids)
    if missing:
        raise ValueError(
            "correspondence frames are absent from initialization problem: "
            + ", ".join(missing)
        )
    frames = tuple(correspondence.frames)
    if len({item.camera_frame for item in frames}) != 1 or len(
        {item.lidar_frame for item in frames}
    ) != 1:
        raise ValueError("all correspondence frames must share camera/LiDAR IDs")
    camera_frame = frames[0].camera_frame
    lidar_frame = frames[0].lidar_frame
    for transform in (
        problem.initial_transform_camera_lidar,
        problem.reference_transform_camera_lidar,
    ):
        if transform.parent != camera_frame or transform.child != lidar_frame:
            raise ValueError(
                "camera-LiDAR problem transform frames do not match correspondence frames"
            )
    confidence_calibration_id: str | None = None
    confidence_calibration_digest: str | None = None
    confidence_calibration_file: Path | None = None
    if confidence_calibration_path is not None:
        confidence_calibration_file = Path(confidence_calibration_path).resolve()
        confidence_calibration = load_camera_lidar_confidence_calibration(
            confidence_calibration_file
        )
        if confidence_calibration.schema_version != (
            "slac.camera_lidar_confidence_calibration/v0.2"
        ):
            raise ValueError(
                "legacy support-only confidence calibrations cannot be used at runtime"
            )
        locked_calibration_options = (
            confidence_calibration.locked_refinement_options
        )
        if (
            confidence_calibration.status != "locked"
            or locked_calibration_options is None
        ):
            raise ValueError(
                "confidence calibration was rejected on development geometry"
            )
        confidence_calibration_digest = sha256_path(confidence_calibration_file)
        if confidence_calibration_digest is None:
            raise ValueError("confidence calibration lock must be a readable file")
        confidence_calibration_id = confidence_calibration.calibration_id
        if correspondence.provider != confidence_calibration.provider:
            raise ValueError(
                "correspondence provider identity differs from confidence calibration"
            )
        if problem.dataset_id == confidence_calibration.dataset_id:
            if (
                problem_digest != confidence_calibration.problem_sha256
                or correspondence_digest
                != confidence_calibration.correspondence_artifact_sha256
            ):
                raise ValueError(
                    "development inputs differ from confidence calibration sources"
                )
        elif (
            problem.dataset_id
            not in confidence_calibration.evaluation_dataset_ids_excluded
        ):
            raise ValueError(
                "target dataset is neither the development source nor a locked "
                "evaluation dataset"
            )
        split_seed = (
            options.split_seed
            if options is not None
            else locked_calibration_options.evaluation_split_seeds[0]
        )
        locked_options = (
            refinement_options_from_camera_lidar_confidence_calibration(
                confidence_calibration,
                split_seed=split_seed,
            )
        )
        if options is not None and options != locked_options:
            raise ValueError("runtime refinement options differ from confidence lock")
        settings = locked_options
    else:
        settings = options or ProbabilisticCameraLidarRefinementOptions()
    initialization_trace_id: str | None = None
    initialization_trace_status: str | None = None
    initialization_trace_hit: bool | None = None
    initialization_trace_digest: str | None = None
    initial_transform = problem.initial_transform_camera_lidar.as_se3()
    if initialization_trace_path is not None:
        trace_file = Path(initialization_trace_path)
        trace = load_calibration_candidate_trace(trace_file)
        initialization_trace_digest = sha256_path(trace_file)
        if initialization_trace_digest is None:
            raise ValueError("initialization trace must be a readable file")
        if trace.problem_sha256 != problem_digest:
            raise ValueError(
                "initialization trace problem digest does not match the problem"
            )
        expected = problem.initial_transform_camera_lidar
        output = trace.output_transform_camera_lidar
        if output.parent != expected.parent or output.child != expected.child:
            raise ValueError(
                "initialization trace transform frames do not match the problem"
            )
        initialization_trace_id = trace.trace_id
        initialization_trace_status = trace.status
        initialization_trace_hit = trace.outcome.hit
        initial_transform = output.as_se3()
    result = ProbabilisticCameraLidarRefiner().solve(
        frames,
        initial_transform,
        settings,
    )
    return _artifact(
        correspondence.artifact_id,
        problem.problem_id,
        correspondence.provider,
        camera_frame,
        lidar_frame,
        result,
        reference_transform=problem.reference_transform_camera_lidar,
        initialization_trace_id=initialization_trace_id,
        initialization_trace_status=initialization_trace_status,
        initialization_trace_hit=initialization_trace_hit,
        result_id=result_id
        or f"{correspondence.artifact_id}-{problem.problem_id}-refinement",
        correspondence_sha256=correspondence_digest,
        problem_sha256=problem_digest,
        initialization_trace_sha256=initialization_trace_digest,
        confidence_calibration_id=confidence_calibration_id,
        confidence_calibration_path=(
            str(confidence_calibration_file)
            if confidence_calibration_file is not None
            else None
        ),
        confidence_calibration_sha256=confidence_calibration_digest,
        command=command or [],
    )


def _artifact(
    correspondence_artifact_id: str,
    initialization_problem_id: str,
    provider: Any,
    camera_frame: str,
    lidar_frame: str,
    result: ProbabilisticCameraLidarRefinementResult,
    *,
    reference_transform: TransformResult,
    initialization_trace_id: str | None,
    initialization_trace_status: str | None,
    initialization_trace_hit: bool | None,
    result_id: str,
    correspondence_sha256: str,
    problem_sha256: str,
    initialization_trace_sha256: str | None,
    confidence_calibration_id: str | None,
    confidence_calibration_path: str | None,
    confidence_calibration_sha256: str | None,
    command: list[str],
) -> ProbabilisticRefinementResultArtifact:
    initial_transform = _transform(
        camera_frame,
        lidar_frame,
        result.initial_transform_camera_lidar,
    )
    output_transform = _transform(
        camera_frame,
        lidar_frame,
        result.transform_camera_lidar,
    )
    candidate_transform = _transform(
        camera_frame,
        lidar_frame,
        result.candidate_transform_camera_lidar,
    )
    return ProbabilisticRefinementResultArtifact(
        result_id=result_id,
        status=result.status,
        reason=result.reason,
        correspondence_artifact_id=correspondence_artifact_id,
        initialization_problem_id=initialization_problem_id,
        initialization_trace_id=initialization_trace_id,
        initialization_trace_status=initialization_trace_status,
        initialization_trace_hit=initialization_trace_hit,
        initialization_source=(
            "d2d_candidate_trace"
            if initialization_trace_id is not None
            else "problem_initial_transform"
        ),
        provider=provider,
        initial_transform_camera_lidar=initial_transform,
        candidate_transform_camera_lidar=candidate_transform,
        transform_camera_lidar=output_transform,
        reference_transform_camera_lidar=reference_transform,
        initial_rotation_error_deg=_rotation_error_deg(
            initial_transform,
            reference_transform,
        ),
        candidate_rotation_error_deg=_rotation_error_deg(
            candidate_transform,
            reference_transform,
        ),
        final_rotation_error_deg=_rotation_error_deg(
            output_transform,
            reference_transform,
        ),
        initial_translation_error_m=_translation_error_m(
            initial_transform,
            reference_transform,
        ),
        candidate_translation_error_m=_translation_error_m(
            candidate_transform,
            reference_transform,
        ),
        final_translation_error_m=_translation_error_m(
            output_transform,
            reference_transform,
        ),
        train_frame_ids=list(result.train_frame_ids),
        holdout_frame_ids=list(result.holdout_frame_ids),
        initial_train_evaluation=_evaluation(
            result.initial_train_evaluation
        ),
        candidate_train_evaluation=_evaluation(
            result.candidate_train_evaluation
        ),
        final_train_evaluation=_evaluation(result.final_train_evaluation),
        initial_holdout_evaluation=_evaluation(
            result.initial_holdout_evaluation
        ),
        candidate_holdout_evaluation=_evaluation(
            result.candidate_holdout_evaluation
        ),
        final_holdout_evaluation=_evaluation(
            result.final_holdout_evaluation
        ),
        acceptance=ProbabilisticRefinementAcceptanceArtifact(
            **asdict(result.acceptance)
        ),
        trace=[
            ProbabilisticRefinementIterationArtifact(
                evaluation=item.evaluation,
                delta_rotation_deg_xyz=list(
                    item.delta_rotation_deg_xyz
                ),
                delta_translation_m_xyz=list(
                    item.delta_translation_m_xyz
                ),
                objective=item.objective,
                accepted=item.accepted,
            )
            for item in result.trace
        ],
        options=cast(
            dict[str, int | float | str | bool],
            asdict(result.options),
        ),
        provenance=ProbabilisticRefinementProvenance(
            generator=__name__,
            generator_version=PROBABILISTIC_CAMERA_LIDAR_RUN_VERSION,
            git_commit=git_commit(),
            command=command,
            correspondence_artifact_sha256=correspondence_sha256,
            initialization_problem_sha256=problem_sha256,
            initialization_trace_sha256=initialization_trace_sha256,
            confidence_calibration_id=confidence_calibration_id,
            confidence_calibration_path=confidence_calibration_path,
            confidence_calibration_sha256=confidence_calibration_sha256,
        ),
    )


def _evaluation(
    value: ProbabilisticPoseEvaluation,
) -> ProbabilisticRefinementEvaluationArtifact:
    return ProbabilisticRefinementEvaluationArtifact(**asdict(value))


def _transform(
    camera_frame: str,
    lidar_frame: str,
    value: Any,
) -> TransformResult:
    return TransformResult(
        parent=camera_frame,
        child=lidar_frame,
        translation_m=list(value.translation_m),
        rotation_quat_xyzw=list(value.rotation_quat_xyzw),
    )


def _rotation_error_deg(left: TransformResult, right: TransformResult) -> float:
    left_quaternion = left.as_se3().rotation_quat_xyzw
    right_quaternion = right.as_se3().rotation_quat_xyzw
    dot = abs(
        sum(
            left_value * right_value
            for left_value, right_value in zip(
                left_quaternion,
                right_quaternion,
                strict=True,
            )
        )
    )
    return math.degrees(2.0 * math.acos(min(1.0, max(-1.0, dot))))


def _translation_error_m(
    left: TransformResult,
    right: TransformResult,
) -> float:
    return math.sqrt(
        sum(
            (left_value - right_value) ** 2
            for left_value, right_value in zip(
                left.translation_m,
                right.translation_m,
                strict=True,
            )
        )
    )
