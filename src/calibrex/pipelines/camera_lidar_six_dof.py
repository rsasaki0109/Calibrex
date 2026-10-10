"""Single D2D estimation with standard calibration results and explicit evidence."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Sequence
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Literal

from calibrex import __version__
from calibrex.core.provenance import sha256_path
from calibrex.core.result import (
    CalibrationResult,
    DegeneracyResult,
    FrameGraphSnapshot,
    MetricResult,
    QualitySummary,
    RunInfo,
    TransformEstimateProvenance,
    TransformQuality,
    TransformResult,
    build_result_provenance,
    load_result,
)
from calibrex.evaluation.borer_rotation_benchmark import load_borer_problem
from calibrex.solvers.borer_depth_to_depth_solver import (
    DepthToDepthEvaluation,
    evaluate_depth_to_depth_mi,
    resolve_depth_to_depth_options,
)
from calibrex.solvers.borer_six_dof_solver import BorerSixDofOptions, BorerSixDofSolver
from calibrex.solvers.depth_to_depth_runtime import (
    ProjectionBackend,
    create_depth_pair_projector,
    depth_pair_transfer_statistics,
    validate_cuda_cache_bytes,
)

SIX_DOF_ESTIMATE_VERSION = "calibrex.camera_lidar_six_dof_estimate/v0.1"


def run_camera_lidar_six_dof_estimate(
    problem_path: str | Path,
    *,
    options: BorerSixDofOptions | None = None,
    projection_backend: ProjectionBackend = "numpy",
    cuda_cache_bytes: int | None = None,
    initial_result_path: str | Path | None = None,
    holdout_frame_ids: Sequence[str] = (),
    command: Sequence[str],
) -> CalibrationResult:
    """Refine the declared prior once; reference values never enter optimization.

    All problem/provider file digests are verified before solving. Histogram
    ranges are resolved from training frames only and then frozen for holdout
    evaluation. Convergence records optimizer termination, not calibration
    accuracy. Insufficient inputs produce a failed result with no output pose.
    """

    started = time.perf_counter()
    validate_cuda_cache_bytes(projection_backend, cuda_cache_bytes)
    settings = options or BorerSixDofOptions()
    settings_payload = _settings_payload(settings)
    # Reject non-finite Python API inputs before loading data or starting CUDA.
    _digest(settings_payload)
    if settings.objective not in ("mutual_information", "normalized_mutual_information"):
        raise ValueError("unsupported six-DoF objective")
    path = Path(problem_path)
    loaded = load_borer_problem(path)
    problem = loaded.problem
    if settings.rotation_bound_deg > problem.rotation_bound_deg:
        raise ValueError("six-DoF rotation bound exceeds the problem rotation bound")
    if settings.translation_bound_m > problem.translation_bound_m:
        raise ValueError("six-DoF translation bound exceeds the problem translation bound")
    prior = problem.initial_transform_camera_lidar.model_copy(deep=True)
    parent, child = prior.parent, prior.child
    if not parent.strip() or not child.strip() or parent == child:
        raise ValueError("camera and LiDAR frames must be non-empty and distinct")
    key = f"T_{parent}_{child}"
    reference = problem.reference_transform_camera_lidar.model_copy(deep=True)
    _validate_transform(reference, parent, child)
    depth_by_id = {item.frame_id: item for item in loaded.depth_provider.observations}
    for binding in problem.observations:
        if depth_by_id[binding.depth_observation_frame_id].source_frame != parent:
            raise ValueError("depth observation source_frame does not match the camera frame")
    initial_digest = None
    if initial_result_path is not None:
        initial_path = Path(initial_result_path)
        initial_result = load_result(initial_path)
        if key not in initial_result.transforms:
            raise ValueError(f"initial result has no output transform {key}")
        prior = initial_result.transforms[key].model_copy(deep=True)
        initial_digest = sha256_path(initial_path)
        if initial_digest is None:
            raise ValueError("could not hash initial result")
    _validate_transform(prior, parent, child)
    prior.provenance = prior.provenance.model_copy(
        update={"role_in_comparison": "candidate"},
        deep=True,
    )
    # An inherited uncertainty/quality applies to a different estimate/run.
    prior.quality = TransformQuality()
    prior.covariance = None
    reference.provenance = reference.provenance.model_copy(
        update={"role_in_comparison": "selected_reference"},
        deep=True,
    )
    holdout_ids = set(holdout_frame_ids)
    known_ids = {item.frame_id for item in loaded.observations}
    unknown_ids = holdout_ids - known_ids
    if unknown_ids:
        raise ValueError(f"unknown holdout frame IDs: {', '.join(sorted(unknown_ids))}")
    train = tuple(item for item in loaded.observations if item.frame_id not in holdout_ids)
    holdout = tuple(item for item in loaded.observations if item.frame_id in holdout_ids)
    if not train:
        raise ValueError("at least one training frame is required")
    settings = replace(settings, d2d=resolve_depth_to_depth_options(train, settings.d2d))
    inputs_verified_at = time.perf_counter()
    projector, identity = create_depth_pair_projector(
        projection_backend,
        loaded.observations,
        cuda_cache_bytes=cuda_cache_bytes,
    )
    settings = replace(settings, projector=projector, projection_backend=identity)
    prepared_at = time.perf_counter()
    solved = BorerSixDofSolver().solve(train, prior.as_se3(), settings)
    solved_at = time.perf_counter()
    initial_holdout = final_holdout = None
    if holdout:
        initial_holdout = evaluate_depth_to_depth_mi(
            holdout,
            prior.as_se3(),
            settings.d2d,
            projector=projector,
        )
        final_holdout = evaluate_depth_to_depth_mi(
            holdout,
            solved.transform_camera_lidar,
            settings.d2d,
            projector=projector,
        )
    evaluated_at = time.perf_counter()
    config_payload = {
        "pipeline_version": SIX_DOF_ESTIMATE_VERSION,
        "options": _settings_payload(settings),
        "projection_backend": identity,
        "initial_transform": prior.as_se3().as_dict(),
        "train_frame_ids": [item.frame_id for item in train],
        "holdout_frame_ids": [item.frame_id for item in holdout],
    }
    config_digest = _digest(config_payload)
    input_digests = {
        "problem_sha256": loaded.problem_sha256,
        "depth_provider_sha256": loaded.depth_provider_sha256,
        "initial_result_sha256": initial_digest,
    }
    input_digest = _digest(input_digests)
    warnings = [
        "Optimizer convergence does not establish calibration accuracy; "
        "observability and covariance were not estimated.",
    ]
    if not holdout:
        warnings.append("No holdout frames were supplied; all objective values use training data.")
    elif final_holdout is not None and final_holdout.evaluated_frame_count != len(holdout):
        warnings.append("Some holdout frames did not meet min_visible_points.")
    if solved.final_evaluation.skipped_frame_ids:
        warnings.append("Some training frames did not meet min_visible_points.")
    status: Literal["success", "warning", "failed"] = "success"
    if solved.status == "max_evaluations":
        status = "warning"
        warnings.append(
            "Evaluation limit reached before convergence; output is a partial candidate."
        )
    elif solved.status == "insufficient_observations":
        status = "failed"
    transforms = {}
    if solved.status != "insufficient_observations":
        transforms[key] = TransformResult(
            parent=parent,
            child=child,
            translation_m=list(solved.transform_camera_lidar.translation_m),
            rotation_quat_xyzw=list(solved.transform_camera_lidar.rotation_quat_xyzw),
            estimate_id=key,
            provenance=TransformEstimateProvenance(
                producer="slac_native",
                execution_mode="offline_batch",
                role_in_comparison="output",
                evidence_level="algorithmically_refined",
                source="borer_depth_to_depth_mutual_information/v0.2",
                tool_name="calibrex",
                tool_version=__version__,
                adapter_version=SIX_DOF_ESTIMATE_VERSION,
                notes=[f"solver status: {solved.status}; {solved.stopping_reason}"],
            ),
        )
    provenance = build_result_provenance(
        producer="calibrex",
        tool_name="calibrex",
        tool_version=__version__,
        command=command,
        config_sha256=config_digest,
        input_sha256=input_digest,
        extra={
            "pipeline": SIX_DOF_ESTIMATE_VERSION,
            "problem_path": str(path),
            "input_digests": input_digests,
            "dataset_id": problem.dataset_id,
            "dataset_family": problem.dataset_family,
            "sequence_id": problem.sequence_id,
            "initial_source": (
                str(initial_result_path)
                if initial_result_path is not None
                else "problem.initial_transform_camera_lidar"
            ),
            "configuration": config_payload,
            "projection_backend": identity,
            "projection_transfer_statistics": depth_pair_transfer_statistics(projector),
            "solver_result": solved.as_dict(),
            "initial_holdout_evaluation": (
                initial_holdout.as_dict() if initial_holdout is not None else None
            ),
            "final_holdout_evaluation": (
                final_holdout.as_dict() if final_holdout is not None else None
            ),
            "timing_seconds": {
                "input_verification_and_loading": inputs_verified_at - started,
                "backend_preparation": prepared_at - inputs_verified_at,
                "solve": solved_at - prepared_at,
                "holdout_evaluation": evaluated_at - solved_at,
            },
            "pipeline_source_sha256": sha256_path(Path(__file__)),
            "data_verified": True,
            "reference_used_for_optimization": False,
        },
    )
    metrics = {}
    for name in ("mutual_information", "normalized_mutual_information", "visible_point_count"):
        metrics[f"initial_d2d_{name}"] = _metric(solved.initial_evaluation, initial_holdout, name)
        metrics[f"final_d2d_{name}"] = _metric(solved.final_evaluation, final_holdout, name)
    return CalibrationResult(
        run=RunInfo(
            id=f"{problem.problem_id}-six-dof-{config_digest[:12]}",
            slac_version=__version__,
            git_commit=provenance["git_commit"],
            config_sha256=config_digest,
            dataset_sha256=input_digest,
            status=status,
            provenance=provenance,
        ),
        frame_graph=FrameGraphSnapshot(root=parent, frames={parent: None, child: parent}),
        candidate_extrinsics={key: prior},
        reference_extrinsics={key: reference},
        transforms=transforms,
        metrics=metrics,
        degeneracy=DegeneracyResult(reason="D2D observability has not been evaluated."),
        quality=QualitySummary(
            grade="fail" if status == "failed" else "warn",
            blocking_failures=[solved.stopping_reason] if status == "failed" else [],
            warnings=warnings,
            recommendation=["Validate the output on independent observations before deployment."],
        ),
    )


def _settings_payload(settings: BorerSixDofOptions) -> dict[str, Any]:
    payload = {
        item.name: getattr(settings, item.name)
        for item in fields(settings)
        if item.name not in ("projector", "projection_backend", "d2d")
    }
    payload["d2d"] = {item.name: getattr(settings.d2d, item.name) for item in fields(settings.d2d)}
    return payload


def _digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"),
    ).hexdigest()


def _validate_transform(transform: TransformResult, parent: str, child: str) -> None:
    if (transform.parent, transform.child) != (parent, child):
        raise ValueError("initial/reference transforms must use the same camera and LiDAR frames")
    if not all(
        math.isfinite(value)
        for value in (
            *transform.translation_m,
            *transform.rotation_quat_xyzw,
        )
    ):
        raise ValueError("transform coordinates must be finite")
    transform.as_se3()  # also reject zero-norm quaternions


def _metric(
    train: DepthToDepthEvaluation,
    holdout: DepthToDepthEvaluation | None,
    name: str,
) -> MetricResult:
    return MetricResult(
        train=float(getattr(train, name)) if train.evaluated_frame_count else None,
        holdout=(
            float(getattr(holdout, name))
            if holdout is not None and holdout.evaluated_frame_count
            else None
        ),
        unit="count"
        if name == "visible_point_count"
        else "nat"
        if name == "mutual_information"
        else "ratio",
        reason="D2D objective evidence; no accuracy or acceptance threshold was applied.",
    )
