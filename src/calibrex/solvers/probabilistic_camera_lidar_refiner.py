"""Multi-frame uncertainty-aware Camera--LiDAR pose refinement."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3
from calibrex.core.probabilistic_correspondence import (
    ProbabilisticCorrespondenceFrame,
    ProbabilisticImageCorrespondence,
)
from calibrex.evaluation.holdout import split_indices
from calibrex.solvers.borer_six_dof_solver import apply_local_se3_delta

FloatArray: TypeAlias = NDArray[np.float64]
ProbabilisticRefinementStatus = Literal[
    "converged",
    "max_evaluations",
    "insufficient_correspondences",
    "missing_holdout",
    "at_bound",
]
ProbabilisticRefinementSelectedSource = Literal[
    "refined_candidate",
    "initializer_rollback",
]
INITIALIZER_PRESERVING_ACCEPTANCE_POLICY_ID = (
    "initializer_preserving_fit_only/v0.1"
)


@dataclass(frozen=True)
class ProbabilisticCameraLidarRefinementOptions:
    """Bounded refinement, robustness, and prespecified ablation controls."""

    holdout_ratio: float = 0.25
    split_seed: int = 0
    minimum_confidence: float = 0.25
    minimum_train_correspondences: int = 24
    minimum_holdout_correspondences: int = 8
    rotation_bound_deg: float = 2.0
    translation_bound_m: float = 0.25
    initial_rotation_step_deg: float = 0.5
    initial_translation_step_m: float = 0.05
    minimum_rotation_step_deg: float = 0.02
    minimum_translation_step_m: float = 0.002
    max_evaluations: int = 400
    cauchy_scale: float = 3.0
    use_covariance: bool = True
    use_outlier_probability: bool = True
    use_reliability: bool = True
    acceptance_policy_id: str = INITIALIZER_PRESERVING_ACCEPTANCE_POLICY_ID
    minimum_absolute_train_objective_improvement: float = 1.0e-6
    minimum_relative_train_objective_improvement: float = 1.0e-4
    maximum_train_correspondence_loss_fraction: float = 0.05
    maximum_accepted_bound_fraction: float = 0.95

    def __post_init__(self) -> None:
        if not 0.0 < self.holdout_ratio < 1.0:
            raise ValueError("holdout_ratio must be in (0, 1)")
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be in [0, 1]")
        if self.minimum_train_correspondences < 4:
            raise ValueError("minimum_train_correspondences must be at least four")
        if self.minimum_holdout_correspondences < 4:
            raise ValueError("minimum_holdout_correspondences must be at least four")
        if self.rotation_bound_deg <= 0.0 or self.translation_bound_m <= 0.0:
            raise ValueError("pose correction bounds must be positive")
        if not (
            0.0
            < self.minimum_rotation_step_deg
            <= self.initial_rotation_step_deg
            <= self.rotation_bound_deg
        ):
            raise ValueError("rotation steps and bound are inconsistent")
        if not (
            0.0
            < self.minimum_translation_step_m
            <= self.initial_translation_step_m
            <= self.translation_bound_m
        ):
            raise ValueError("translation steps and bound are inconsistent")
        if self.max_evaluations < 13:
            raise ValueError("max_evaluations must be at least 13")
        if self.cauchy_scale <= 0.0:
            raise ValueError("cauchy_scale must be positive")
        if not self.acceptance_policy_id:
            raise ValueError("acceptance_policy_id must not be empty")
        if self.minimum_absolute_train_objective_improvement < 0.0:
            raise ValueError(
                "minimum_absolute_train_objective_improvement must be non-negative"
            )
        if self.minimum_relative_train_objective_improvement < 0.0:
            raise ValueError(
                "minimum_relative_train_objective_improvement must be non-negative"
            )
        if not 0.0 <= self.maximum_train_correspondence_loss_fraction < 1.0:
            raise ValueError(
                "maximum_train_correspondence_loss_fraction must be in [0, 1)"
            )
        if not 0.0 < self.maximum_accepted_bound_fraction <= 1.0:
            raise ValueError("maximum_accepted_bound_fraction must be in (0, 1]")


@dataclass(frozen=True)
class ProbabilisticPoseEvaluation:
    """Robust multi-frame reprojection evidence."""

    objective: float
    weighted_reprojection_rmse_px: float | None
    mean_mahalanobis_error: float | None
    valid_correspondence_count: int
    skipped_correspondence_count: int


@dataclass(frozen=True)
class ProbabilisticPoseIteration:
    """One retained local SE(3) candidate."""

    evaluation: int
    delta_rotation_deg_xyz: tuple[float, float, float]
    delta_translation_m_xyz: tuple[float, float, float]
    objective: float
    accepted: bool


@dataclass(frozen=True)
class ProbabilisticRefinementAcceptanceDecision:
    """Fit-only decision that keeps holdout evidence out of pose selection."""

    policy_id: str
    accepted: bool
    selected_source: ProbabilisticRefinementSelectedSource
    candidate_status: ProbabilisticRefinementStatus
    reasons: tuple[str, ...]
    holdout_used_for_selection: Literal[False]
    initial_train_objective: float
    candidate_train_objective: float
    absolute_train_objective_improvement: float | None
    relative_train_objective_improvement: float | None
    initial_train_correspondence_count: int
    candidate_train_correspondence_count: int
    minimum_retained_train_correspondence_count: int
    maximum_candidate_bound_fraction: float


@dataclass(frozen=True)
class ProbabilisticCameraLidarRefinementResult:
    """D2D-seeded multi-frame pose result with disjoint holdout."""

    status: ProbabilisticRefinementStatus
    reason: str
    initial_transform_camera_lidar: SE3
    candidate_transform_camera_lidar: SE3
    transform_camera_lidar: SE3
    train_frame_ids: tuple[str, ...]
    holdout_frame_ids: tuple[str, ...]
    initial_train_evaluation: ProbabilisticPoseEvaluation
    candidate_train_evaluation: ProbabilisticPoseEvaluation
    final_train_evaluation: ProbabilisticPoseEvaluation
    initial_holdout_evaluation: ProbabilisticPoseEvaluation
    candidate_holdout_evaluation: ProbabilisticPoseEvaluation
    final_holdout_evaluation: ProbabilisticPoseEvaluation
    acceptance: ProbabilisticRefinementAcceptanceDecision
    trace: tuple[ProbabilisticPoseIteration, ...]
    options: ProbabilisticCameraLidarRefinementOptions

    def as_dict(self) -> dict[str, object]:
        """Return typed metrics, trace, ablations, and method provenance."""

        return {
            "status": self.status,
            "reason": self.reason,
            "initial_transform_camera_lidar": (
                self.initial_transform_camera_lidar.as_dict()
            ),
            "candidate_transform_camera_lidar": (
                self.candidate_transform_camera_lidar.as_dict()
            ),
            "transform_camera_lidar": self.transform_camera_lidar.as_dict(),
            "train_frame_ids": list(self.train_frame_ids),
            "holdout_frame_ids": list(self.holdout_frame_ids),
            "initial_train_evaluation": asdict(
                self.initial_train_evaluation
            ),
            "candidate_train_evaluation": asdict(
                self.candidate_train_evaluation
            ),
            "final_train_evaluation": asdict(self.final_train_evaluation),
            "initial_holdout_evaluation": asdict(
                self.initial_holdout_evaluation
            ),
            "candidate_holdout_evaluation": asdict(
                self.candidate_holdout_evaluation
            ),
            "final_holdout_evaluation": asdict(
                self.final_holdout_evaluation
            ),
            "acceptance": asdict(self.acceptance),
            "trace": [asdict(item) for item in self.trace],
            "options": asdict(self.options),
            "method": "probabilistic_multiframe_refinement/v0.3",
            "initialization_contract": (
                "initial_transform_camera_lidar is supplied by the analytical "
                "D2D stage or another explicitly declared initializer"
            ),
            "implementation": (
                "independent ROS-free NumPy implementation; learned provider "
                "is isolated behind slac.probabilistic_correspondence/v0.1"
            ),
        }


class ProbabilisticCameraLidarRefiner:
    """Refine one shared extrinsic from probabilistic correspondences."""

    def solve(
        self,
        frames: Sequence[ProbabilisticCorrespondenceFrame],
        initial_transform_camera_lidar: SE3,
        options: ProbabilisticCameraLidarRefinementOptions | None = None,
    ) -> ProbabilisticCameraLidarRefinementResult:
        """Optimize training frames and retain disjoint holdout evidence."""

        settings = options or ProbabilisticCameraLidarRefinementOptions()
        identifiers = [item.frame_id for item in frames]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("probabilistic correspondence frame IDs must be unique")
        ordered = sorted(frames, key=lambda item: item.frame_id)
        train_indices, holdout_indices = split_indices(
            len(ordered), settings.holdout_ratio, settings.split_seed
        )
        train = tuple(ordered[index] for index in train_indices)
        holdout = tuple(ordered[index] for index in holdout_indices)
        return self.solve_partitioned(
            train,
            holdout,
            initial_transform_camera_lidar,
            settings,
        )

    def solve_partitioned(
        self,
        train_frames: Sequence[ProbabilisticCorrespondenceFrame],
        holdout_frames: Sequence[ProbabilisticCorrespondenceFrame],
        initial_transform_camera_lidar: SE3,
        options: ProbabilisticCameraLidarRefinementOptions | None = None,
    ) -> ProbabilisticCameraLidarRefinementResult:
        """Refine from caller-supplied disjoint train/holdout frame sets.

        The caller is responsible for assembling block-aligned frame subsets so
        neighboring measurements sharing a scene are never split across the
        train/holdout boundary.
        """

        settings = options or ProbabilisticCameraLidarRefinementOptions()
        train = tuple(train_frames)
        holdout = tuple(holdout_frames)
        identifiers = [item.frame_id for item in train] + [
            item.frame_id for item in holdout
        ]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("train and holdout frame IDs must be disjoint")
        initial_train = evaluate_probabilistic_camera_lidar_pose(
            train, initial_transform_camera_lidar, settings
        )
        initial_holdout = evaluate_probabilistic_camera_lidar_pose(
            holdout, initial_transform_camera_lidar, settings
        )
        if (
            initial_train.valid_correspondence_count
            < settings.minimum_train_correspondences
        ):
            return _terminal(
                "insufficient_correspondences",
                "too few training correspondences passed confidence/projection gates",
                initial_transform_camera_lidar,
                train,
                holdout,
                initial_train,
                initial_holdout,
                settings,
            )
        if (
            initial_holdout.valid_correspondence_count
            < settings.minimum_holdout_correspondences
        ):
            return _terminal(
                "missing_holdout",
                "too few disjoint holdout correspondences passed the gates",
                initial_transform_camera_lidar,
                train,
                holdout,
                initial_train,
                initial_holdout,
                settings,
            )
        delta: FloatArray = np.zeros(6, dtype=float)
        steps: FloatArray = np.asarray(
            [
                settings.initial_rotation_step_deg,
                settings.initial_rotation_step_deg,
                settings.initial_rotation_step_deg,
                settings.initial_translation_step_m,
                settings.initial_translation_step_m,
                settings.initial_translation_step_m,
            ],
            dtype=float,
        )
        minimum_steps: FloatArray = np.asarray(
            [
                settings.minimum_rotation_step_deg,
                settings.minimum_rotation_step_deg,
                settings.minimum_rotation_step_deg,
                settings.minimum_translation_step_m,
                settings.minimum_translation_step_m,
                settings.minimum_translation_step_m,
            ],
            dtype=float,
        )
        bounds: FloatArray = np.asarray(
            [
                settings.rotation_bound_deg,
                settings.rotation_bound_deg,
                settings.rotation_bound_deg,
                settings.translation_bound_m,
                settings.translation_bound_m,
                settings.translation_bound_m,
            ],
            dtype=float,
        )
        best = initial_train
        trace = [_trace(1, delta, best.objective, accepted=True)]
        evaluations = 1
        while np.any(steps >= minimum_steps) and evaluations < settings.max_evaluations:
            candidates: list[
                tuple[float, tuple[float, ...], FloatArray, ProbabilisticPoseEvaluation]
            ] = []
            for axis in range(6):
                if steps[axis] < minimum_steps[axis]:
                    continue
                for direction in (-1.0, 1.0):
                    candidate = delta.copy()
                    candidate[axis] = float(
                        np.clip(
                            candidate[axis] + direction * steps[axis],
                            -bounds[axis],
                            bounds[axis],
                        )
                    )
                    if np.array_equal(candidate, delta):
                        continue
                    transform = apply_local_se3_delta(
                        initial_transform_camera_lidar,
                        candidate[:3],
                        candidate[3:],
                    )
                    result = evaluate_probabilistic_camera_lidar_pose(
                        train, transform, settings
                    )
                    evaluations += 1
                    candidates.append(
                        (
                            result.objective,
                            tuple(candidate.tolist()),
                            candidate,
                            result,
                        )
                    )
                    if evaluations >= settings.max_evaluations:
                        break
                if evaluations >= settings.max_evaluations:
                    break
            if not candidates:
                break
            objective, _tie, candidate_delta, candidate_result = min(
                candidates, key=lambda item: (item[0], item[1])
            )
            accepted = objective < best.objective - 1.0e-12
            for candidate_objective, _tie, candidate, _result in candidates:
                trace.append(
                    _trace(
                        len(trace) + 1,
                        candidate,
                        candidate_objective,
                        accepted=(
                            accepted
                            and np.array_equal(candidate, candidate_delta)
                        ),
                    )
                )
            if accepted:
                delta = candidate_delta
                best = candidate_result
            else:
                steps *= 0.5
        candidate_transform = apply_local_se3_delta(
            initial_transform_camera_lidar, delta[:3], delta[3:]
        )
        candidate_holdout = evaluate_probabilistic_camera_lidar_pose(
            holdout, candidate_transform, settings
        )
        at_bound = bool(np.any(np.isclose(np.abs(delta), bounds, atol=1.0e-12)))
        converged = bool(np.all(steps < minimum_steps))
        status: ProbabilisticRefinementStatus = (
            "at_bound"
            if at_bound
            else "converged"
            if converged
            else "max_evaluations"
        )
        acceptance = _accept_candidate(
            status,
            initial_train,
            best,
            delta,
            bounds,
            settings,
        )
        selected_transform = (
            candidate_transform
            if acceptance.accepted
            else initial_transform_camera_lidar
        )
        final_train = best if acceptance.accepted else initial_train
        final_holdout = candidate_holdout if acceptance.accepted else initial_holdout
        status_reason = {
            "at_bound": "pose optimum reached a configured correction bound",
            "converged": "all pose correction steps reached their minima",
            "max_evaluations": "maximum pose evaluations reached",
        }[status]
        if not acceptance.accepted:
            status_reason += "; candidate rejected and initializer retained"
        return ProbabilisticCameraLidarRefinementResult(
            status=status,
            reason=status_reason,
            initial_transform_camera_lidar=initial_transform_camera_lidar,
            candidate_transform_camera_lidar=candidate_transform,
            transform_camera_lidar=selected_transform,
            train_frame_ids=tuple(item.frame_id for item in train),
            holdout_frame_ids=tuple(item.frame_id for item in holdout),
            initial_train_evaluation=initial_train,
            candidate_train_evaluation=best,
            final_train_evaluation=final_train,
            initial_holdout_evaluation=initial_holdout,
            candidate_holdout_evaluation=candidate_holdout,
            final_holdout_evaluation=final_holdout,
            acceptance=acceptance,
            trace=tuple(trace),
            options=settings,
        )


def evaluate_probabilistic_camera_lidar_pose(
    frames: Sequence[ProbabilisticCorrespondenceFrame],
    transform_camera_lidar: SE3,
    options: ProbabilisticCameraLidarRefinementOptions | None = None,
) -> ProbabilisticPoseEvaluation:
    """Evaluate robust covariance-weighted multi-frame reprojection."""

    settings = options or ProbabilisticCameraLidarRefinementOptions()
    pixel_squared: list[float] = []
    mahalanobis: list[float] = []
    weights: list[float] = []
    skipped = 0
    for frame in frames:
        for item in frame.correspondences:
            confidence = _confidence(item, settings)
            if confidence < settings.minimum_confidence:
                skipped += 1
                continue
            point = transform_camera_lidar.transform_point(item.point_lidar_m)
            if point[2] <= 1.0e-9:
                skipped += 1
                continue
            u, v = project_camera_point(point, frame)
            if not (
                0.0 <= u < frame.intrinsics.width
                and 0.0 <= v < frame.intrinsics.height
            ):
                skipped += 1
                continue
            residual_u = u - item.image_mean_px[0]
            residual_v = v - item.image_mean_px[1]
            pixel_squared.append(
                residual_u * residual_u + residual_v * residual_v
            )
            if settings.use_covariance:
                a, b, c, d = item.image_covariance_px2
                determinant = a * d - b * c
                squared = (
                    d * residual_u * residual_u
                    - (b + c) * residual_u * residual_v
                    + a * residual_v * residual_v
                ) / determinant
            else:
                squared = residual_u * residual_u + residual_v * residual_v
            mahalanobis.append(math.sqrt(max(0.0, squared)))
            weights.append(confidence)
    if not weights:
        return ProbabilisticPoseEvaluation(
            objective=math.inf,
            weighted_reprojection_rmse_px=None,
            mean_mahalanobis_error=None,
            valid_correspondence_count=0,
            skipped_correspondence_count=skipped,
        )
    weight_array: FloatArray = np.asarray(weights, dtype=float)
    pixel_array: FloatArray = np.asarray(pixel_squared, dtype=float)
    mahalanobis_array: FloatArray = np.asarray(mahalanobis, dtype=float)
    robust = np.log1p(
        (mahalanobis_array / settings.cauchy_scale) ** 2
    )
    return ProbabilisticPoseEvaluation(
        objective=float(np.sum(weight_array * robust) / np.sum(weight_array)),
        weighted_reprojection_rmse_px=math.sqrt(
            float(np.sum(weight_array * pixel_array))
            / float(np.sum(weight_array))
        ),
        mean_mahalanobis_error=float(
            np.sum(weight_array * mahalanobis_array) / np.sum(weight_array)
        ),
        valid_correspondence_count=len(weights),
        skipped_correspondence_count=skipped,
    )


def _confidence(
    item: ProbabilisticImageCorrespondence,
    settings: ProbabilisticCameraLidarRefinementOptions,
) -> float:
    reliability = (
        item.reliability if settings.use_reliability else 1.0
    )
    inlier_probability = (
        1.0 - item.outlier_probability
        if settings.use_outlier_probability
        else 1.0
    )
    return reliability * inlier_probability


def project_camera_point(
    point: tuple[float, float, float],
    frame: ProbabilisticCorrespondenceFrame,
) -> tuple[float, float]:
    """Project a point through the declared pinhole, MEI, or double-sphere model."""

    intrinsics = frame.intrinsics
    x_point, y_point, z_point = point
    if intrinsics.projection == "mei":
        if intrinsics.xi is None:
            raise ValueError("MEI projection requires xi")
        norm = math.sqrt(
            x_point * x_point + y_point * y_point + z_point * z_point
        )
        denominator = z_point + intrinsics.xi * norm
        if denominator <= 1.0e-12:
            return (math.inf, math.inf)
        x = x_point / denominator
        y = y_point / denominator
    elif intrinsics.projection == "double_sphere":
        if intrinsics.xi is None or intrinsics.alpha is None:
            raise ValueError("double-sphere projection requires xi and alpha")
        norm = math.sqrt(
            x_point * x_point + y_point * y_point + z_point * z_point
        )
        zeta = intrinsics.xi * norm + z_point
        second_norm = math.sqrt(x_point * x_point + y_point * y_point + zeta * zeta)
        denominator = intrinsics.alpha * second_norm + (1.0 - intrinsics.alpha) * zeta
        if denominator <= 1.0e-12:
            return (math.inf, math.inf)
        x = x_point / denominator
        y = y_point / denominator
    else:
        x = x_point / z_point
        y = y_point / z_point
    if intrinsics.distortion_model == "radial-tangential" and len(
        intrinsics.distortion
    ) in {4, 5}:
        k1, k2, p1, p2 = intrinsics.distortion[:4]
        k3 = intrinsics.distortion[4] if len(intrinsics.distortion) == 5 else 0.0
        radius2 = x * x + y * y
        radial = 1.0 + k1 * radius2 + k2 * radius2**2 + k3 * radius2**3
        distorted_x = (
            x * radial + 2.0 * p1 * x * y + p2 * (radius2 + 2.0 * x * x)
        )
        distorted_y = (
            y * radial + p1 * (radius2 + 2.0 * y * y) + 2.0 * p2 * x * y
        )
        x, y = distorted_x, distorted_y
    return (
        intrinsics.fx * x + intrinsics.cx,
        intrinsics.fy * y + intrinsics.cy,
    )


def _trace(
    evaluation: int,
    delta: FloatArray,
    objective: float,
    *,
    accepted: bool,
) -> ProbabilisticPoseIteration:
    return ProbabilisticPoseIteration(
        evaluation=evaluation,
        delta_rotation_deg_xyz=(
            float(delta[0]),
            float(delta[1]),
            float(delta[2]),
        ),
        delta_translation_m_xyz=(
            float(delta[3]),
            float(delta[4]),
            float(delta[5]),
        ),
        objective=objective,
        accepted=accepted,
    )


def _terminal(
    status: ProbabilisticRefinementStatus,
    reason: str,
    transform: SE3,
    train: Sequence[ProbabilisticCorrespondenceFrame],
    holdout: Sequence[ProbabilisticCorrespondenceFrame],
    train_evaluation: ProbabilisticPoseEvaluation,
    holdout_evaluation: ProbabilisticPoseEvaluation,
    options: ProbabilisticCameraLidarRefinementOptions,
) -> ProbabilisticCameraLidarRefinementResult:
    zero_delta: FloatArray = np.zeros(6, dtype=float)
    bounds = np.asarray(
        [
            options.rotation_bound_deg,
            options.rotation_bound_deg,
            options.rotation_bound_deg,
            options.translation_bound_m,
            options.translation_bound_m,
            options.translation_bound_m,
        ],
        dtype=float,
    )
    acceptance = _accept_candidate(
        status,
        train_evaluation,
        train_evaluation,
        zero_delta,
        bounds,
        options,
    )
    return ProbabilisticCameraLidarRefinementResult(
        status=status,
        reason=f"{reason}; candidate rejected and initializer retained",
        initial_transform_camera_lidar=transform,
        candidate_transform_camera_lidar=transform,
        transform_camera_lidar=transform,
        train_frame_ids=tuple(item.frame_id for item in train),
        holdout_frame_ids=tuple(item.frame_id for item in holdout),
        initial_train_evaluation=train_evaluation,
        candidate_train_evaluation=train_evaluation,
        final_train_evaluation=train_evaluation,
        initial_holdout_evaluation=holdout_evaluation,
        candidate_holdout_evaluation=holdout_evaluation,
        final_holdout_evaluation=holdout_evaluation,
        acceptance=acceptance,
        trace=(),
        options=options,
    )


def _accept_candidate(
    status: ProbabilisticRefinementStatus,
    initial: ProbabilisticPoseEvaluation,
    candidate: ProbabilisticPoseEvaluation,
    delta: FloatArray,
    bounds: FloatArray,
    options: ProbabilisticCameraLidarRefinementOptions,
) -> ProbabilisticRefinementAcceptanceDecision:
    """Select a candidate using training evidence only."""

    reasons: list[str] = []
    if status != "converged":
        reasons.append("solver_not_converged")
    absolute_improvement: float | None = None
    relative_improvement: float | None = None
    if math.isfinite(initial.objective) and math.isfinite(candidate.objective):
        absolute_improvement = initial.objective - candidate.objective
        relative_improvement = absolute_improvement / max(
            abs(initial.objective), 1.0e-12
        )
        if (
            absolute_improvement
            < options.minimum_absolute_train_objective_improvement
        ):
            reasons.append("insufficient_absolute_train_improvement")
        if (
            relative_improvement
            < options.minimum_relative_train_objective_improvement
        ):
            reasons.append("insufficient_relative_train_improvement")
    else:
        reasons.append("non_finite_train_objective")
    minimum_retained = math.ceil(
        initial.valid_correspondence_count
        * (1.0 - options.maximum_train_correspondence_loss_fraction)
    )
    if candidate.valid_correspondence_count < minimum_retained:
        reasons.append("train_correspondence_support_loss")
    bound_fraction = float(np.max(np.abs(delta) / bounds))
    if bound_fraction >= options.maximum_accepted_bound_fraction:
        reasons.append("candidate_near_correction_bound")
    accepted = not reasons
    return ProbabilisticRefinementAcceptanceDecision(
        policy_id=options.acceptance_policy_id,
        accepted=accepted,
        selected_source=(
            "refined_candidate" if accepted else "initializer_rollback"
        ),
        candidate_status=status,
        reasons=tuple(reasons),
        holdout_used_for_selection=False,
        initial_train_objective=initial.objective,
        candidate_train_objective=candidate.objective,
        absolute_train_objective_improvement=absolute_improvement,
        relative_train_objective_improvement=relative_improvement,
        initial_train_correspondence_count=initial.valid_correspondence_count,
        candidate_train_correspondence_count=candidate.valid_correspondence_count,
        minimum_retained_train_correspondence_count=minimum_retained,
        maximum_candidate_bound_fraction=bound_fraction,
    )
