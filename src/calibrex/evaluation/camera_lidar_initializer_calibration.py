"""Freeze aggregate Camera--LiDAR PnP settings on development data only."""

from __future__ import annotations

import importlib
import math
from collections.abc import Sequence
from pathlib import Path

from calibrex.core.camera_lidar_artifacts import (
    CameraLidarCalibrationProblem,
    load_camera_lidar_problem,
)
from calibrex.core.camera_lidar_initializer_calibration import (
    CAMERA_LIDAR_AGGREGATE_FRAME_SELECTION_RULE_ID,
    CAMERA_LIDAR_AGGREGATE_PNP_METHOD,
    CameraLidarInitializerCalibrationArtifact,
    CameraLidarInitializerCalibrationCandidate,
    CameraLidarInitializerCalibrationProvenance,
    CameraLidarInitializerRecoveryGate,
    CameraLidarInitializerSeedResult,
    CameraLidarLockedInitializerOptions,
    camera_lidar_initializer_candidate_id,
    camera_lidar_initializer_gate_reasons,
)
from calibrex.core.geometry import SE3
from calibrex.core.probabilistic_correspondence import (
    ProbabilisticCorrespondenceArtifact,
    ProbabilisticPnpToolIdentity,
    load_probabilistic_correspondence,
)
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.result import TransformResult
from calibrex.solvers.opencv_probabilistic_pnp_adapter import (
    OPENCV_AGGREGATE_PNP_METHOD,
    OPENCV_LICENSE_SPDX,
    OPENCV_PNP_ADAPTER_VERSION,
    OPENCV_PNP_REFERENCE,
    OPENCV_SOURCE_URL,
    OpenCvProbabilisticPnpAdapter,
    OpenCvProbabilisticPnpOptions,
)

CAMERA_LIDAR_INITIALIZER_CALIBRATION_VERSION = (
    "calibrex.camera_lidar_initializer_calibration/v0.1"
)
DEFAULT_INITIALIZER_CONFIDENCE_THRESHOLDS: tuple[float, ...] = (
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
)
DEFAULT_INITIALIZER_RANSAC_THRESHOLDS_PX: tuple[float, ...] = (
    2.0,
    4.0,
    8.0,
    16.0,
    32.0,
)
DEFAULT_INITIALIZER_RANDOM_SEEDS: tuple[int, ...] = tuple(range(5))


def calibrate_camera_lidar_initializer(
    correspondence_path: str | Path,
    problem_path: str | Path,
    *,
    confidence_thresholds: Sequence[float] = (
        DEFAULT_INITIALIZER_CONFIDENCE_THRESHOLDS
    ),
    ransac_reprojection_thresholds_px: Sequence[float] = (
        DEFAULT_INITIALIZER_RANSAC_THRESHOLDS_PX
    ),
    random_seeds: Sequence[int] = DEFAULT_INITIALIZER_RANDOM_SEEDS,
    minimum_correspondences: int = 4,
    minimum_frame_correspondences: int = 4,
    minimum_frames: int = 4,
    ransac_confidence: float = 0.999,
    ransac_iterations: int = 10000,
    mahalanobis_inlier_threshold: float = 3.0,
    recovery_gate: CameraLidarInitializerRecoveryGate | None = None,
    evaluation_dataset_ids_excluded: Sequence[str],
    calibration_id: str | None = None,
    command: list[str] | None = None,
) -> CameraLidarInitializerCalibrationArtifact:
    """Run a complete reference-scored grid and freeze one deterministic lock."""

    correspondence_file = Path(correspondence_path).resolve()
    problem_file = Path(problem_path).resolve()
    correspondence = load_probabilistic_correspondence(correspondence_file)
    problem = load_camera_lidar_problem(problem_file)
    correspondence_digest = _required_digest(correspondence_file)
    problem_digest = _required_digest(problem_file)
    confidence_values = _validate_grid(
        confidence_thresholds,
        label="confidence",
        minimum=0.0,
        maximum=1.0,
    )
    reprojection_values = _validate_grid(
        ransac_reprojection_thresholds_px,
        label="RANSAC reprojection",
        minimum=0.0,
    )
    seeds = _validate_seeds(random_seeds)
    gate = recovery_gate or CameraLidarInitializerRecoveryGate(
        rotation_error_max_deg=0.5,
        translation_error_max_m=0.20,
        minimum_selected_frame_count=4,
        minimum_selected_correspondence_count=16,
        minimum_ransac_inlier_count=12,
        maximum_ransac_inlier_reprojection_rmse_px=8.0,
        maximum_seed_rotation_delta_deg=0.01,
        maximum_seed_translation_delta_m=0.005,
    )
    _validate_common_options(
        minimum_correspondences=minimum_correspondences,
        minimum_frame_correspondences=minimum_frame_correspondences,
        minimum_frames=minimum_frames,
        ransac_confidence=ransac_confidence,
        ransac_iterations=ransac_iterations,
        mahalanobis_inlier_threshold=mahalanobis_inlier_threshold,
    )
    excluded = [str(item) for item in evaluation_dataset_ids_excluded]
    if not excluded or len(excluded) != len(set(excluded)):
        raise ValueError("excluded evaluation dataset IDs must be non-empty and unique")
    _validate_development_inputs(correspondence, problem)
    reference = problem.reference_transform_camera_lidar.as_se3()
    adapter = OpenCvProbabilisticPnpAdapter()
    candidates: list[CameraLidarInitializerCalibrationCandidate] = []
    opencv_versions: set[str] = set()
    for minimum_confidence in confidence_values:
        for reprojection_threshold in reprojection_values:
            eligible_frame_ids = _eligible_frame_ids(
                correspondence,
                minimum_confidence=minimum_confidence,
                minimum_frame_correspondences=minimum_frame_correspondences,
            )
            seed_results: list[CameraLidarInitializerSeedResult] = []
            for random_seed in seeds:
                options = OpenCvProbabilisticPnpOptions(
                    minimum_confidence=minimum_confidence,
                    minimum_correspondences=minimum_correspondences,
                    minimum_frame_correspondences=(
                        minimum_frame_correspondences
                    ),
                    minimum_frames=minimum_frames,
                    ransac_reprojection_threshold_px=reprojection_threshold,
                    ransac_confidence=ransac_confidence,
                    ransac_iterations=ransac_iterations,
                    mahalanobis_inlier_threshold=mahalanobis_inlier_threshold,
                    random_seed=random_seed,
                )
                result = adapter.solve_aggregate(
                    correspondence,
                    artifact_sha256=correspondence_digest,
                    options=options,
                )
                if result.method != OPENCV_AGGREGATE_PNP_METHOD:
                    raise ValueError("aggregate PnP adapter method changed during sweep")
                if result.opencv_version is not None:
                    opencv_versions.add(result.opencv_version)
                if result.status == "converged":
                    if result.transform_camera_lidar is None:
                        raise ValueError("converged aggregate PnP omitted its pose")
                    if tuple(eligible_frame_ids) != result.source_frame_ids:
                        raise ValueError(
                            "aggregate PnP selected frames differ from support rule"
                        )
                    rotation_error, translation_error = _transform_errors(
                        result.transform_camera_lidar,
                        reference,
                    )
                    transform = TransformResult(
                        parent=result.camera_frame,
                        child=result.lidar_frame,
                        translation_m=list(
                            result.transform_camera_lidar.translation_m
                        ),
                        rotation_quat_xyzw=list(
                            result.transform_camera_lidar.rotation_quat_xyzw
                        ),
                    )
                else:
                    rotation_error = None
                    translation_error = None
                    transform = None
                seed_results.append(
                    CameraLidarInitializerSeedResult(
                        random_seed=random_seed,
                        status=result.status,
                        source_frame_ids=eligible_frame_ids,
                        selected_correspondence_count=(
                            result.selected_correspondence_count
                        ),
                        ransac_inlier_count=result.ransac_inlier_count,
                        ransac_inlier_reprojection_rmse_px=(
                            result.ransac_inlier_reprojection_rmse_px
                        ),
                        rotation_error_deg=rotation_error,
                        translation_error_m=translation_error,
                        transform_camera_lidar=transform,
                        reason=result.reason,
                    )
                )
            candidate = _build_candidate(
                minimum_confidence,
                reprojection_threshold,
                seed_results,
                gate,
            )
            candidates.append(candidate)
    if len(opencv_versions) > 1:
        raise ValueError("OpenCV version changed during initializer calibration")
    opencv_version = next(iter(opencv_versions), None)
    passing = [item for item in candidates if item.gate_pass]
    selected = (
        sorted(
            passing,
            key=lambda item: (
                -item.minimum_confidence,
                item.ransac_reprojection_threshold_px,
            ),
        )[0]
        if passing
        else None
    )
    if selected is not None and opencv_version is None:
        raise ValueError("passing initializer calibration lacks OpenCV identity")
    locked = (
        CameraLidarLockedInitializerOptions(
            adapter_version=OPENCV_PNP_ADAPTER_VERSION,
            opencv_version=opencv_version or "unavailable",
            minimum_confidence=selected.minimum_confidence,
            minimum_correspondences=minimum_correspondences,
            minimum_frame_correspondences=minimum_frame_correspondences,
            minimum_frames=minimum_frames,
            ransac_reprojection_threshold_px=(
                selected.ransac_reprojection_threshold_px
            ),
            ransac_confidence=ransac_confidence,
            ransac_iterations=ransac_iterations,
            mahalanobis_inlier_threshold=mahalanobis_inlier_threshold,
            evaluation_random_seeds=seeds,
        )
        if selected is not None
        else None
    )
    return CameraLidarInitializerCalibrationArtifact(
        status="locked" if selected is not None else "rejected",
        calibration_id=(
            calibration_id
            or f"{correspondence.dataset_id}-aggregate-pnp-initializer-v0.1"
        ),
        dataset_id=correspondence.dataset_id,
        sequence_id=correspondence.sequence_id or problem.sequence_id,
        split_id="development",
        problem_id=problem.problem_id,
        problem_sha256=problem_digest,
        correspondence_artifact_id=correspondence.artifact_id,
        correspondence_artifact_sha256=correspondence_digest,
        provider=correspondence.provider,
        tool=ProbabilisticPnpToolIdentity(
            name="opencv",
            version=opencv_version,
            source_url=OPENCV_SOURCE_URL,
            api_reference=OPENCV_PNP_REFERENCE,
            license_spdx=OPENCV_LICENSE_SPDX,
            execution_mode="optional_import",
        ),
        adapter_version=OPENCV_PNP_ADAPTER_VERSION,
        evaluation_dataset_ids_excluded=excluded,
        random_seeds=seeds,
        minimum_correspondences=minimum_correspondences,
        minimum_frame_correspondences=minimum_frame_correspondences,
        minimum_frames=minimum_frames,
        ransac_confidence=ransac_confidence,
        ransac_iterations=ransac_iterations,
        mahalanobis_inlier_threshold=mahalanobis_inlier_threshold,
        recovery_gate=gate,
        candidates=candidates,
        selected_candidate_id=(
            selected.candidate_id if selected is not None else None
        ),
        locked_initializer_options=locked,
        provenance=CameraLidarInitializerCalibrationProvenance(
            generator=__name__,
            generator_version=CAMERA_LIDAR_INITIALIZER_CALIBRATION_VERSION,
            git_commit=git_commit(),
            command=command or [],
            source_paths=[str(correspondence_file), str(problem_file)],
            source_sha256={
                "correspondence_artifact": correspondence_digest,
                "problem": problem_digest,
            },
            notes=[
                "Reference pose was used only to score the development grid.",
                "No initializer or reference pose was supplied to aggregate PnP.",
                "Excluded evaluation dataset contents were not read for selection.",
            ],
        ),
    )


def initializer_options_from_camera_lidar_calibration(
    calibration: CameraLidarInitializerCalibrationArtifact,
    *,
    random_seed: int,
    enforce_opencv_version: bool = True,
) -> OpenCvProbabilisticPnpOptions:
    """Recover locked options and reject adapter, tool, or seed drift."""

    if calibration.status != "locked":
        raise ValueError("initializer calibration is rejected and has no runtime lock")
    locked = calibration.locked_initializer_options
    if locked is None:
        raise ValueError("initializer calibration omitted locked runtime options")
    if calibration.method != CAMERA_LIDAR_AGGREGATE_PNP_METHOD:
        raise ValueError("initializer calibration method is unsupported")
    if locked.method != OPENCV_AGGREGATE_PNP_METHOD:
        raise ValueError("locked aggregate PnP method differs from adapter")
    if locked.adapter_version != OPENCV_PNP_ADAPTER_VERSION:
        raise ValueError("locked aggregate PnP adapter version differs from runtime")
    if locked.frame_selection_rule_id != (
        CAMERA_LIDAR_AGGREGATE_FRAME_SELECTION_RULE_ID
    ):
        raise ValueError("locked aggregate frame selection rule is unsupported")
    if random_seed not in locked.evaluation_random_seeds:
        raise ValueError("random seed is absent from the initializer evaluation lock")
    if enforce_opencv_version:
        try:
            cv2 = importlib.import_module("cv2")
        except ImportError as exc:
            raise ValueError("locked initializer requires OpenCV") from exc
        version = str(getattr(cv2, "__version__", "unknown"))
        if version != locked.opencv_version:
            raise ValueError(
                "OpenCV version differs from initializer lock: "
                f"{version} != {locked.opencv_version}"
            )
    return OpenCvProbabilisticPnpOptions(
        minimum_confidence=locked.minimum_confidence,
        minimum_correspondences=locked.minimum_correspondences,
        minimum_frame_correspondences=locked.minimum_frame_correspondences,
        minimum_frames=locked.minimum_frames,
        ransac_reprojection_threshold_px=(
            locked.ransac_reprojection_threshold_px
        ),
        ransac_confidence=locked.ransac_confidence,
        ransac_iterations=locked.ransac_iterations,
        mahalanobis_inlier_threshold=locked.mahalanobis_inlier_threshold,
        random_seed=random_seed,
    )


def _build_candidate(
    minimum_confidence: float,
    reprojection_threshold: float,
    seed_results: list[CameraLidarInitializerSeedResult],
    gate: CameraLidarInitializerRecoveryGate,
) -> CameraLidarInitializerCalibrationCandidate:
    all_converged = all(item.status == "converged" for item in seed_results)
    maxima = _candidate_maxima(seed_results) if all_converged else None
    draft = CameraLidarInitializerCalibrationCandidate(
        candidate_id=camera_lidar_initializer_candidate_id(
            minimum_confidence, reprojection_threshold
        ),
        minimum_confidence=minimum_confidence,
        ransac_reprojection_threshold_px=reprojection_threshold,
        seed_results=seed_results,
        all_seeds_converged=all_converged,
        minimum_selected_frame_count=min(
            len(item.source_frame_ids) for item in seed_results
        ),
        minimum_selected_correspondence_count=min(
            item.selected_correspondence_count for item in seed_results
        ),
        minimum_ransac_inlier_count=min(
            item.ransac_inlier_count for item in seed_results
        ),
        maximum_ransac_inlier_reprojection_rmse_px=(
            maxima[0] if maxima is not None else None
        ),
        maximum_rotation_error_deg=maxima[1] if maxima is not None else None,
        maximum_translation_error_m=maxima[2] if maxima is not None else None,
        maximum_pairwise_seed_rotation_delta_deg=(
            maxima[3] if maxima is not None else None
        ),
        maximum_pairwise_seed_translation_delta_m=(
            maxima[4] if maxima is not None else None
        ),
        gate_pass=False,
        gate_reasons=["pending_gate_evaluation"],
    )
    reasons = camera_lidar_initializer_gate_reasons(draft, gate)
    return CameraLidarInitializerCalibrationCandidate.model_validate(
        {
            **draft.model_dump(mode="python"),
            "gate_pass": not reasons,
            "gate_reasons": reasons,
        }
    )


def _candidate_maxima(
    seed_results: list[CameraLidarInitializerSeedResult],
) -> tuple[float, float, float, float, float]:
    inlier_rmse = [
        _required(item.ransac_inlier_reprojection_rmse_px)
        for item in seed_results
    ]
    rotation_errors = [_required(item.rotation_error_deg) for item in seed_results]
    translation_errors = [
        _required(item.translation_error_m) for item in seed_results
    ]
    transforms = [
        item.transform_camera_lidar.as_se3()
        for item in seed_results
        if item.transform_camera_lidar is not None
    ]
    rotation_deltas = [0.0]
    translation_deltas = [0.0]
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


def _eligible_frame_ids(
    artifact: ProbabilisticCorrespondenceArtifact,
    *,
    minimum_confidence: float,
    minimum_frame_correspondences: int,
) -> list[str]:
    return [
        frame.frame_id
        for frame in sorted(artifact.frames, key=lambda item: item.frame_id)
        if sum(
            item.reliability * (1.0 - item.outlier_probability)
            >= minimum_confidence
            for item in frame.correspondences
        )
        >= minimum_frame_correspondences
    ]


def _validate_development_inputs(
    correspondence: ProbabilisticCorrespondenceArtifact,
    problem: CameraLidarCalibrationProblem,
) -> None:
    if problem.dataset_id != correspondence.dataset_id:
        raise ValueError("problem and correspondence dataset IDs differ")
    if problem.sequence_id != correspondence.sequence_id:
        raise ValueError("problem and correspondence sequence IDs differ")
    if correspondence.split_id != "development":
        raise ValueError("initializer calibration accepts development inputs only")
    if {item.split_id for item in problem.observations} != {"development"}:
        raise ValueError("initializer problem observations must be development only")
    problem_frame_ids = sorted(item.frame_id for item in problem.observations)
    correspondence_frame_ids = sorted(
        item.frame_id for item in correspondence.frames
    )
    if problem_frame_ids != correspondence_frame_ids:
        raise ValueError("problem and correspondence frame IDs differ")
    reference = problem.reference_transform_camera_lidar
    if any(
        frame.camera_frame != reference.parent or frame.lidar_frame != reference.child
        for frame in correspondence.frames
    ):
        raise ValueError("reference transform frames differ from correspondences")


def _validate_grid(
    values: Sequence[float],
    *,
    label: str,
    minimum: float,
    maximum: float | None = None,
) -> list[float]:
    result = [float(item) for item in values]
    if len(result) < 2:
        raise ValueError(f"{label} grid requires at least two values")
    if any(
        not math.isfinite(item)
        or item <= minimum
        or (maximum is not None and item > maximum)
        for item in result
    ):
        raise ValueError(f"{label} grid contains an invalid value")
    if result != sorted(result) or len(result) != len(set(result)):
        raise ValueError(f"{label} grid must be unique and ascending")
    return result


def _validate_seeds(values: Sequence[int]) -> list[int]:
    seeds = [int(item) for item in values]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("initializer random seeds must be non-empty and unique")
    return seeds


def _validate_common_options(
    *,
    minimum_correspondences: int,
    minimum_frame_correspondences: int,
    minimum_frames: int,
    ransac_confidence: float,
    ransac_iterations: int,
    mahalanobis_inlier_threshold: float,
) -> None:
    OpenCvProbabilisticPnpOptions(
        minimum_correspondences=minimum_correspondences,
        minimum_frame_correspondences=minimum_frame_correspondences,
        minimum_frames=minimum_frames,
        ransac_confidence=ransac_confidence,
        ransac_iterations=ransac_iterations,
        mahalanobis_inlier_threshold=mahalanobis_inlier_threshold,
    )


def _transform_errors(estimate: SE3, reference: SE3) -> tuple[float, float]:
    relative = reference.compose(estimate.inverse())
    quaternion = relative.rotation_quat_xyzw
    rotation = 2.0 * math.degrees(
        math.acos(min(1.0, max(-1.0, abs(quaternion[3]))))
    )
    translation = math.sqrt(
        sum(
            (estimate.translation_m[index] - reference.translation_m[index]) ** 2
            for index in range(3)
        )
    )
    return rotation, translation


def _required(value: float | None) -> float:
    if value is None:
        raise ValueError("converged initializer metric is absent")
    return value


def _required_digest(path: Path) -> str:
    digest = sha256_path(path)
    if digest is None:
        raise ValueError(f"required initializer calibration source is unreadable: {path}")
    return digest
