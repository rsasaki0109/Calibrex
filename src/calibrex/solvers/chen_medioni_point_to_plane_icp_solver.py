"""Chen-Medioni discrete tangent-plane ICP specialization."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3, Vector3
from calibrex.solvers.robust_point_to_point_icp_solver import (
    IcpCandidateEvaluation,
    IcpPoint,
    RobustPointToPointIcpOptions,
    evaluate_icp_candidate,
    split_icp_source_points,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
ChenMedioniStatus = Literal[
    "converged",
    "max_iterations",
    "insufficient_correspondences",
    "degenerate_tangent_geometry",
]
RegistrationDof = Literal["x", "y", "z", "roll", "pitch", "yaw"]


@dataclass(frozen=True)
class ChenMedioniPointToPlaneOptions:
    """Target-normal and point-to-plane evidence policy."""

    normal_neighbor_count: int = 12
    minimum_normal_eigengap: float = 0.02
    rank_tolerance: float = 1.0e-7
    max_condition_number: float = 1.0e8
    known_bad_translation_m: float = 0.05
    known_bad_rotation_deg: float = 5.0
    known_bad_residual_margin_m: float = 0.005


@dataclass(frozen=True)
class ChenMedioniIteration:
    """One refreshed tangent-plane least-squares update."""

    iteration: int
    correspondence_count: int
    point_to_plane_rmse_m: float
    step_translation_m: float
    step_rotation_rad: float
    relative_objective_change: float | None


@dataclass(frozen=True)
class ChenMedioniProbe:
    """Signed six-DoF point-to-plane falsification control."""

    dof: RegistrationDof
    amount: float
    unit: Literal["m", "deg"]
    point_to_plane_rmse_m: float | None
    residual_delta_m: float | None
    detectable: bool | None


@dataclass(frozen=True)
class ChenMedioniPointToPlaneResult:
    """Registration estimate with surface and common holdout evidence."""

    status: ChenMedioniStatus
    reason: str
    common_options: RobustPointToPointIcpOptions
    options: ChenMedioniPointToPlaneOptions
    transform_target_source: SE3 | None
    train_source_ids: tuple[str, ...]
    holdout_source_ids: tuple[str, ...]
    valid_target_normal_count: int
    target_normal_fraction: float
    target_normal_eigengap_minimum: float | None
    target_normal_eigengap_median: float | None
    train_point_to_plane_rmse_m: float | None
    holdout_point_to_plane_rmse_m: float | None
    correspondence_count: int
    tangent_singular_values: tuple[float, float, float, float, float, float] | None
    tangent_rank: int
    tangent_condition_number: float | None
    common_evaluation: IcpCandidateEvaluation | None
    probes: tuple[ChenMedioniProbe, ...]
    history: tuple[ChenMedioniIteration, ...]

    def as_dict(self) -> dict[str, object]:
        """Return schema-friendly evidence and primary-paper provenance."""

        return {
            "status": self.status,
            "reason": self.reason,
            "resolved_options": {
                "common_icp": self.common_options.__dict__,
                "chen_medioni": self.options.__dict__,
            },
            "transform_target_source": (
                self.transform_target_source.as_dict()
                if self.transform_target_source is not None
                else None
            ),
            "train_source_ids": list(self.train_source_ids),
            "holdout_source_ids": list(self.holdout_source_ids),
            "valid_target_normal_count": self.valid_target_normal_count,
            "target_normal_fraction": self.target_normal_fraction,
            "target_normal_eigengap_minimum": self.target_normal_eigengap_minimum,
            "target_normal_eigengap_median": self.target_normal_eigengap_median,
            "train_point_to_plane_rmse_m": self.train_point_to_plane_rmse_m,
            "holdout_point_to_plane_rmse_m": self.holdout_point_to_plane_rmse_m,
            "correspondence_count": self.correspondence_count,
            "tangent_singular_values": self.tangent_singular_values,
            "tangent_rank": self.tangent_rank,
            "tangent_condition_number": self.tangent_condition_number,
            "common_evaluation": (
                self.common_evaluation.as_dict() if self.common_evaluation is not None else None
            ),
            "known_bad_probes": [probe.__dict__ for probe in self.probes],
            "history": [item.__dict__ for item in self.history],
            "method": "chen_medioni_discrete_tangent_plane_icp/v0.1",
            "paper": {
                "title": "Object Modeling by Registration of Multiple Range Images",
                "authors": "Yang Chen and Gerard Medioni",
                "conference_doi": "10.1109/ROBOT.1991.132043",
                "journal_doi": "10.1016/0262-8856(92)90066-C",
                "equations": [10, 11, 12],
                "primary_pdf": (
                    "https://graphics.stanford.edu/~smr/ICP/comparison/"
                    "chen-medioni-align-rob91.pdf"
                ),
            },
            "frame_convention": "p_target = R_target_source p_source + t_target_source",
            "surface_specialization": (
                "nearest discrete target sample with a PCA tangent plane; this approximates "
                "the paper's normal-line/surface intersection"
            ),
            "normal_policy": "smallest PCA eigenvector of deterministic target k-neighborhood",
            "correspondence_policy": (
                "refreshed Euclidean nearest target with optional mutual check, distance gate, "
                "valid-normal gate, and stable point-to-plane trimming"
            ),
            "evaluation_contract": "shared spatial source holdout with fresh rematching",
            "implementation": "independent NumPy implementation; no external code copied",
            "external_code_executed": False,
        }


class ChenMedioniPointToPlaneIcpSolver:
    """Register source to target using refreshed local tangent planes."""

    def solve(
        self,
        source_points: Sequence[IcpPoint],
        target_points: Sequence[IcpPoint],
        initial_transform: SE3 | None = None,
        common_options: RobustPointToPointIcpOptions | None = None,
        options: ChenMedioniPointToPlaneOptions | None = None,
    ) -> ChenMedioniPointToPlaneResult:
        shared = common_options or RobustPointToPointIcpOptions()
        solver_options = options or ChenMedioniPointToPlaneOptions()
        _validate_options(shared, solver_options)
        source = sorted(source_points, key=lambda item: item.point_id)
        target = sorted(target_points, key=lambda item: item.point_id)
        train, holdout = split_icp_source_points(source, shared)
        train_ids = tuple(item.point_id for item in train)
        holdout_ids = tuple(item.point_id for item in holdout)
        if len(train) < shared.min_correspondences or len(target) < 3:
            return _empty(
                "insufficient_correspondences",
                train_ids,
                holdout_ids,
                shared,
                solver_options,
            )

        target_array = _positions(target)
        normals, valid_normals, eigengaps = _estimate_target_normals(
            target_array, solver_options
        )
        normal_values = eigengaps[valid_normals]
        normal_count = int(np.count_nonzero(valid_normals))
        normal_fraction = normal_count / len(target)
        minimum_gap = float(np.min(normal_values)) if len(normal_values) else None
        median_gap = float(np.median(normal_values)) if len(normal_values) else None
        if normal_count < shared.min_correspondences:
            return _empty(
                "insufficient_correspondences",
                train_ids,
                holdout_ids,
                shared,
                solver_options,
                valid_normal_count=normal_count,
                normal_fraction=normal_fraction,
                minimum_gap=minimum_gap,
                median_gap=median_gap,
            )

        transform = initial_transform or SE3.identity()
        train_array = _positions(train)
        history: list[ChenMedioniIteration] = []
        status: ChenMedioniStatus = "max_iterations"
        reason = "maximum point-to-plane iterations reached"
        previous_objective: float | None = None
        for iteration in range(shared.max_iterations):
            transformed = _transform_points(train_array, transform)
            matched = _tangent_correspondences(
                transformed, target_array, normals, valid_normals, shared
            )
            moving, _planes, plane_normals, residuals, _source_indices, _target_indices = matched
            if len(moving) < shared.min_correspondences:
                return _empty(
                    "insufficient_correspondences",
                    train_ids,
                    holdout_ids,
                    shared,
                    solver_options,
                    valid_normal_count=normal_count,
                    normal_fraction=normal_fraction,
                    minimum_gap=minimum_gap,
                    median_gap=median_gap,
                    history=tuple(history),
                )
            matrix = _tangent_matrix(moving, plane_normals)
            spectrum, rank, condition = _spectrum(matrix, solver_options.rank_tolerance)
            if rank < 6 or condition > solver_options.max_condition_number:
                return _empty(
                    "degenerate_tangent_geometry",
                    train_ids,
                    holdout_ids,
                    shared,
                    solver_options,
                    valid_normal_count=normal_count,
                    normal_fraction=normal_fraction,
                    minimum_gap=minimum_gap,
                    median_gap=median_gap,
                    correspondence_count=len(moving),
                    tangent_spectrum=spectrum,
                    tangent_rank=rank,
                    tangent_condition=condition,
                    history=tuple(history),
                )
            update, _residuals, _rank, _singular = np.linalg.lstsq(
                matrix, -residuals, rcond=None
            )
            delta = _increment(update)
            transform = delta.compose(transform)
            objective = float(np.mean(np.square(residuals)))
            relative_change = (
                abs(previous_objective - objective) / max(previous_objective, 1.0e-15)
                if previous_objective is not None
                else None
            )
            translation_step = float(np.linalg.norm(update[:3]))
            rotation_step = float(np.linalg.norm(update[3:6]))
            history.append(
                ChenMedioniIteration(
                    iteration=iteration,
                    correspondence_count=len(moving),
                    point_to_plane_rmse_m=math.sqrt(objective),
                    step_translation_m=translation_step,
                    step_rotation_rad=rotation_step,
                    relative_objective_change=relative_change,
                )
            )
            previous_objective = objective
            if max(translation_step, rotation_step) <= shared.convergence_tolerance_m:
                status = "converged"
                reason = "small-motion tangent-plane update fell below tolerance"
                break

        final_transformed = _transform_points(train_array, transform)
        final = _tangent_correspondences(
            final_transformed, target_array, normals, valid_normals, shared
        )
        moving, _planes, plane_normals, residuals, _source_indices, _target_indices = final
        if len(moving) < shared.min_correspondences:
            return _empty(
                "insufficient_correspondences",
                train_ids,
                holdout_ids,
                shared,
                solver_options,
                valid_normal_count=normal_count,
                normal_fraction=normal_fraction,
                minimum_gap=minimum_gap,
                median_gap=median_gap,
                history=tuple(history),
            )
        spectrum, rank, condition = _spectrum(
            _tangent_matrix(moving, plane_normals), solver_options.rank_tolerance
        )
        if rank < 6 or condition > solver_options.max_condition_number:
            return _empty(
                "degenerate_tangent_geometry",
                train_ids,
                holdout_ids,
                shared,
                solver_options,
                valid_normal_count=normal_count,
                normal_fraction=normal_fraction,
                minimum_gap=minimum_gap,
                median_gap=median_gap,
                correspondence_count=len(moving),
                tangent_spectrum=spectrum,
                tangent_rank=rank,
                tangent_condition=condition,
                history=tuple(history),
            )
        train_rmse = _rmse(residuals)
        holdout_rmse = _point_to_plane_rmse(
            _positions(holdout), transform, target_array, normals, valid_normals, shared
        )
        common_evaluation = evaluate_icp_candidate(source, target, transform, shared)
        probes = _known_bad_probes(
            holdout,
            transform,
            holdout_rmse,
            target_array,
            normals,
            valid_normals,
            shared,
            solver_options,
        )
        return ChenMedioniPointToPlaneResult(
            status=status,
            reason=reason,
            common_options=shared,
            options=solver_options,
            transform_target_source=transform,
            train_source_ids=train_ids,
            holdout_source_ids=holdout_ids,
            valid_target_normal_count=normal_count,
            target_normal_fraction=normal_fraction,
            target_normal_eigengap_minimum=minimum_gap,
            target_normal_eigengap_median=median_gap,
            train_point_to_plane_rmse_m=train_rmse,
            holdout_point_to_plane_rmse_m=holdout_rmse,
            correspondence_count=len(moving),
            tangent_singular_values=spectrum,
            tangent_rank=rank,
            tangent_condition_number=condition,
            common_evaluation=common_evaluation,
            probes=probes,
            history=tuple(history),
        )


def _estimate_target_normals(
    target: FloatArray, options: ChenMedioniPointToPlaneOptions
) -> tuple[FloatArray, NDArray[np.bool_], FloatArray]:
    squared = np.sum((target[:, None, :] - target[None, :, :]) ** 2, axis=2)
    neighbor_count = min(options.normal_neighbor_count, len(target))
    neighborhoods = np.argsort(squared, axis=1, kind="stable")[:, :neighbor_count]
    normals = np.zeros_like(target)
    eigengaps: FloatArray = np.zeros(len(target), dtype=np.float64)
    valid: NDArray[np.bool_] = np.zeros(len(target), dtype=np.bool_)
    for index, neighbor_indices in enumerate(neighborhoods):
        points = target[neighbor_indices]
        centered = points - np.mean(points, axis=0)
        covariance = centered.T @ centered / max(len(points), 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        denominator = max(float(eigenvalues[2]), 1.0e-15)
        gap = max(float(eigenvalues[1] - eigenvalues[0]), 0.0) / denominator
        eigengaps[index] = gap
        if gap >= options.minimum_normal_eigengap:
            normals[index] = eigenvectors[:, 0]
            valid[index] = True
    return normals, valid, eigengaps


def _tangent_correspondences(
    transformed_source: FloatArray,
    target: FloatArray,
    normals: FloatArray,
    valid_normals: NDArray[np.bool_],
    options: RobustPointToPointIcpOptions,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, IntArray, IntArray]:
    squared = np.sum((transformed_source[:, None, :] - target[None, :, :]) ** 2, axis=2)
    target_indices = np.argmin(squared, axis=1)
    distances = np.sqrt(squared[np.arange(len(transformed_source)), target_indices])
    mask = (distances <= options.correspondence_distance_m) & valid_normals[target_indices]
    if options.mutual_correspondences:
        source_indices = np.argmin(squared, axis=0)
        mask &= source_indices[target_indices] == np.arange(len(transformed_source))
    selected = np.flatnonzero(mask)
    selected_targets = target_indices[selected]
    residuals = np.sum(
        normals[selected_targets] * (transformed_source[selected] - target[selected_targets]),
        axis=1,
    )
    if len(selected):
        keep = max(1, math.ceil(options.trim_fraction * len(selected)))
        order = np.argsort(np.abs(residuals), kind="stable")[:keep]
        selected = selected[order]
        selected_targets = selected_targets[order]
        residuals = residuals[order]
    return (
        transformed_source[selected],
        target[selected_targets],
        normals[selected_targets],
        residuals,
        selected.astype(np.int64),
        selected_targets.astype(np.int64),
    )


def _tangent_matrix(points: FloatArray, normals: FloatArray) -> FloatArray:
    return np.hstack((normals, np.cross(points, normals)))


def _spectrum(
    matrix: FloatArray, tolerance: float
) -> tuple[tuple[float, float, float, float, float, float], int, float]:
    singular = np.linalg.svd(matrix, compute_uv=False)
    padded = np.pad(singular, (0, max(0, 6 - len(singular))))[:6]
    values = (
        float(padded[0]),
        float(padded[1]),
        float(padded[2]),
        float(padded[3]),
        float(padded[4]),
        float(padded[5]),
    )
    threshold = tolerance * max(values[0], 1.0e-15)
    rank = int(np.count_nonzero(padded > threshold))
    condition = values[0] / values[5] if values[5] > 0.0 else math.inf
    return values, rank, condition


def _point_to_plane_rmse(
    source: FloatArray,
    transform: SE3,
    target: FloatArray,
    normals: FloatArray,
    valid_normals: NDArray[np.bool_],
    options: RobustPointToPointIcpOptions,
) -> float | None:
    if not len(source):
        return None
    matched = _tangent_correspondences(
        _transform_points(source, transform), target, normals, valid_normals, options
    )
    residuals = matched[3]
    return _rmse(residuals) if len(residuals) >= options.min_correspondences else None


def _known_bad_probes(
    holdout: Sequence[IcpPoint],
    transform: SE3,
    baseline_rmse: float | None,
    target: FloatArray,
    normals: FloatArray,
    valid_normals: NDArray[np.bool_],
    common_options: RobustPointToPointIcpOptions,
    options: ChenMedioniPointToPlaneOptions,
) -> tuple[ChenMedioniProbe, ...]:
    source = _positions(holdout)
    directions: tuple[RegistrationDof, ...] = ("x", "y", "z", "roll", "pitch", "yaw")
    axes: tuple[Vector3, Vector3, Vector3] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    probes: list[ChenMedioniProbe] = []
    for index, dof in enumerate(directions):
        translation = index < 3
        magnitude = (
            options.known_bad_translation_m
            if translation
            else math.radians(options.known_bad_rotation_deg)
        )
        for sign in (-1.0, 1.0):
            amount = sign * magnitude
            perturbed = _increment(
                np.asarray(
                    (
                        *(amount * np.asarray(axes[index % 3]) if translation else (0.0,) * 3),
                        *(amount * np.asarray(axes[index % 3]) if not translation else (0.0,) * 3),
                    ),
                    dtype=np.float64,
                )
            ).compose(transform)
            rmse = _point_to_plane_rmse(
                source, perturbed, target, normals, valid_normals, common_options
            )
            delta = (
                rmse - baseline_rmse
                if rmse is not None and baseline_rmse is not None
                else None
            )
            probes.append(
                ChenMedioniProbe(
                    dof=dof,
                    amount=(amount if translation else math.degrees(amount)),
                    unit="m" if translation else "deg",
                    point_to_plane_rmse_m=rmse,
                    residual_delta_m=delta,
                    detectable=(
                        delta >= options.known_bad_residual_margin_m
                        if delta is not None
                        else None
                    ),
                )
            )
    return tuple(probes)


def _increment(values: FloatArray) -> SE3:
    rotation = values[3:6]
    angle = float(np.linalg.norm(rotation))
    if angle > 0.0:
        scale = math.sin(0.5 * angle) / angle
        quaternion = (
            float(scale * rotation[0]),
            float(scale * rotation[1]),
            float(scale * rotation[2]),
            math.cos(0.5 * angle),
        )
    else:
        quaternion = (0.0, 0.0, 0.0, 1.0)
    return SE3((float(values[0]), float(values[1]), float(values[2])), quaternion)


def _positions(points: Sequence[IcpPoint]) -> FloatArray:
    return np.asarray([item.position_m for item in points], dtype=np.float64).reshape((-1, 3))


def _transform_points(points: FloatArray, transform: SE3) -> FloatArray:
    x, y, z, scalar = transform.rotation_quat_xyzw
    rotation: FloatArray = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * scalar), 2 * (x * z + y * scalar)],
            [2 * (x * y + z * scalar), 1 - 2 * (x * x + z * z), 2 * (y * z - x * scalar)],
            [2 * (x * z - y * scalar), 2 * (y * z + x * scalar), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return points @ rotation.T + np.asarray(transform.translation_m)


def _rmse(values: FloatArray) -> float:
    return float(math.sqrt(np.mean(np.square(values))))


def _validate_options(
    common: RobustPointToPointIcpOptions, options: ChenMedioniPointToPlaneOptions
) -> None:
    if options.normal_neighbor_count < 3:
        raise ValueError("point-to-plane normal_neighbor_count must be at least three")
    positive = (
        options.minimum_normal_eigengap,
        options.rank_tolerance,
        options.max_condition_number,
        options.known_bad_translation_m,
        options.known_bad_rotation_deg,
        options.known_bad_residual_margin_m,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in positive):
        raise ValueError("point-to-plane options must be finite and positive")
    if common.max_iterations < 1 or common.min_correspondences < 6:
        raise ValueError("common ICP options require iterations and at least six correspondences")


def _empty(
    status: ChenMedioniStatus,
    train_ids: tuple[str, ...],
    holdout_ids: tuple[str, ...],
    common_options: RobustPointToPointIcpOptions,
    options: ChenMedioniPointToPlaneOptions,
    *,
    valid_normal_count: int = 0,
    normal_fraction: float = 0.0,
    minimum_gap: float | None = None,
    median_gap: float | None = None,
    correspondence_count: int = 0,
    tangent_spectrum: tuple[float, float, float, float, float, float] | None = None,
    tangent_rank: int = 0,
    tangent_condition: float | None = None,
    history: tuple[ChenMedioniIteration, ...] = (),
) -> ChenMedioniPointToPlaneResult:
    reasons = {
        "converged": "",
        "max_iterations": "maximum point-to-plane iterations reached",
        "insufficient_correspondences": "too few valid tangent-plane correspondences",
        "degenerate_tangent_geometry": "surface normals do not constrain all six directions",
    }
    return ChenMedioniPointToPlaneResult(
        status=status,
        reason=reasons[status],
        common_options=common_options,
        options=options,
        transform_target_source=None,
        train_source_ids=train_ids,
        holdout_source_ids=holdout_ids,
        valid_target_normal_count=valid_normal_count,
        target_normal_fraction=normal_fraction,
        target_normal_eigengap_minimum=minimum_gap,
        target_normal_eigengap_median=median_gap,
        train_point_to_plane_rmse_m=None,
        holdout_point_to_plane_rmse_m=None,
        correspondence_count=correspondence_count,
        tangent_singular_values=tangent_spectrum,
        tangent_rank=tangent_rank,
        tangent_condition_number=tangent_condition,
        common_evaluation=None,
        probes=(),
        history=history,
    )
