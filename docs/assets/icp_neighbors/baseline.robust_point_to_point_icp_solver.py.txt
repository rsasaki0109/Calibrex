"""Deterministic native robust point-to-point ICP."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3, Vector3, quaternion_xyzw_from_rotation_matrix
from calibrex.evaluation.holdout import split_indices

FloatArray: TypeAlias = NDArray[np.float64]
IcpStatus = Literal[
    "converged",
    "max_iterations",
    "insufficient_correspondences",
    "degenerate_geometry",
]


@dataclass(frozen=True)
class IcpPoint:
    """A point with a stable identifier for deterministic splitting."""

    point_id: str
    position_m: Vector3


@dataclass(frozen=True)
class RobustPointToPointIcpOptions:
    max_iterations: int = 50
    convergence_tolerance_m: float = 1.0e-7
    correspondence_distance_m: float = 1.0
    trim_fraction: float = 0.8
    mutual_correspondences: bool = True
    min_correspondences: int = 6
    holdout_ratio: float = 0.2
    holdout_voxel_size_m: float = 0.5
    split_seed: int = 0
    rank_tolerance: float = 1.0e-7
    max_condition_number: float = 1.0e8
    effective_diagnostics: bool = True
    curvature_translation_step_m: float = 0.01
    curvature_rotation_step_rad: float = math.radians(0.5)
    min_translation_curvature: float = 1.0e-3
    min_rotation_curvature_m2_per_rad2: float = 1.0e-3
    multi_start_diagnostics: bool = True
    multi_start_translation_m: float = 0.05
    multi_start_rotation_rad: float = math.radians(5.0)
    symmetry_translation_separation_m: float = 0.02
    symmetry_rotation_separation_deg: float = 1.0
    symmetry_equivalent_holdout_margin_m: float = 0.005


@dataclass(frozen=True)
class IcpIteration:
    iteration: int
    correspondence_count: int
    rmse_m: float
    step_translation_m: float
    step_rotation_rad: float


@dataclass(frozen=True)
class IcpRematchingDiagnostics:
    """Black-box local response with correspondences recomputed per perturbation."""

    directions: tuple[Literal["x", "y", "z", "roll", "pitch", "yaw"], ...]
    objective_curvatures: tuple[float | None, ...]
    correspondence_jaccards: tuple[float | None, ...]
    minimum_correspondence_jaccard: float | None
    mean_correspondence_jaccard: float | None
    weak_directions: tuple[str, ...]
    translation_step_m: float
    rotation_step_rad: float


@dataclass(frozen=True)
class IcpMultiStartTrial:
    """Outcome from one deliberately perturbed ICP initialization."""

    direction: Literal["x", "y", "z", "roll", "pitch", "yaw"]
    amount: float
    unit: Literal["m", "rad"]
    status: IcpStatus
    holdout_rmse_m: float | None
    solution_translation_delta_m: float | None
    solution_rotation_delta_deg: float | None
    equivalent_holdout: bool | None
    distinct_solution: bool | None
    symmetry_ambiguous: bool | None


@dataclass(frozen=True)
class IcpCandidateEvaluation:
    """Backend-neutral evaluation of a supplied registration transform."""

    train_source_ids: tuple[str, ...]
    holdout_source_ids: tuple[str, ...]
    train_rmse_m: float | None
    holdout_rmse_m: float | None
    correspondence_count: int
    inlier_fraction: float
    rematching_diagnostics: IcpRematchingDiagnostics | None

    def as_dict(self) -> dict[str, object]:
        return {
            "train_source_ids": list(self.train_source_ids),
            "holdout_source_ids": list(self.holdout_source_ids),
            "train_rmse_m": self.train_rmse_m,
            "holdout_rmse_m": self.holdout_rmse_m,
            "correspondence_count": self.correspondence_count,
            "inlier_fraction": self.inlier_fraction,
            "rematching_diagnostics": (
                self.rematching_diagnostics.__dict__
                if self.rematching_diagnostics is not None
                else None
            ),
            "evaluation_policy": "spatial source holdout with fresh target rematching",
        }


@dataclass(frozen=True)
class RobustPointToPointIcpResult:
    status: IcpStatus
    reason: str
    transform_target_source: SE3 | None
    train_source_ids: tuple[str, ...]
    holdout_source_ids: tuple[str, ...]
    train_rmse_m: float | None
    holdout_rmse_m: float | None
    correspondence_count: int
    inlier_fraction: float
    information_singular_values: tuple[float, float, float, float, float, float] | None
    information_rank: int
    condition_number: float | None
    rematching_diagnostics: IcpRematchingDiagnostics | None
    multi_start_trials: tuple[IcpMultiStartTrial, ...]
    symmetry_ambiguous: bool
    solution_spread_translation_m: float | None
    solution_spread_rotation_deg: float | None
    history: tuple[IcpIteration, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a schema-friendly result with algorithm provenance."""

        return {
            "status": self.status,
            "reason": self.reason,
            "transform_target_source": (
                self.transform_target_source.as_dict()
                if self.transform_target_source is not None
                else None
            ),
            "train_source_ids": list(self.train_source_ids),
            "holdout_source_ids": list(self.holdout_source_ids),
            "train_rmse_m": self.train_rmse_m,
            "holdout_rmse_m": self.holdout_rmse_m,
            "correspondence_count": self.correspondence_count,
            "inlier_fraction": self.inlier_fraction,
            "information_singular_values": self.information_singular_values,
            "information_rank": self.information_rank,
            "condition_number": self.condition_number,
            "rematching_diagnostics": (
                self.rematching_diagnostics.__dict__
                if self.rematching_diagnostics is not None
                else None
            ),
            "multi_start_trials": [item.__dict__ for item in self.multi_start_trials],
            "symmetry_ambiguous": self.symmetry_ambiguous,
            "solution_spread_translation_m": self.solution_spread_translation_m,
            "solution_spread_rotation_deg": self.solution_spread_rotation_deg,
            "history": [item.__dict__ for item in self.history],
            "method": "trimmed_mutual_point_to_point_icp/v0.1",
            "frame_convention": "p_target = R_target_source p_source + t_target_source",
            "correspondence_policy": "nearest neighbor with optional mutual check and trimming",
        }


class RobustPointToPointIcpSolver:
    """Register source points to target points with refreshed correspondences."""

    def solve(
        self,
        source_points: Sequence[IcpPoint],
        target_points: Sequence[IcpPoint],
        initial_transform: SE3 | None = None,
        options: RobustPointToPointIcpOptions | None = None,
    ) -> RobustPointToPointIcpResult:
        solver_options = options or RobustPointToPointIcpOptions()
        _validate_options(solver_options)
        source = sorted(source_points, key=lambda item: item.point_id)
        target = sorted(target_points, key=lambda item: item.point_id)
        train_indices, holdout_indices = _spatial_split_indices(
            source,
            solver_options.holdout_ratio,
            solver_options.holdout_voxel_size_m,
            solver_options.split_seed,
        )
        train = [source[index] for index in train_indices]
        holdout = [source[index] for index in holdout_indices]
        train_ids = tuple(item.point_id for item in train)
        holdout_ids = tuple(item.point_id for item in holdout)
        if len(train) < solver_options.min_correspondences or not target:
            return _empty("insufficient_correspondences", train_ids, holdout_ids)

        seed_transform = initial_transform or SE3.identity()
        transform = seed_transform
        target_array = _positions(target)
        history: list[IcpIteration] = []
        status: IcpStatus = "max_iterations"
        reason = "maximum ICP iterations reached"
        correspondences: tuple[
            FloatArray, FloatArray, FloatArray, NDArray[np.int64], NDArray[np.int64]
        ] | None = None
        for iteration in range(solver_options.max_iterations):
            transformed = _transform_points(_positions(train), transform)
            correspondences = _correspondences(transformed, target_array, solver_options)
            moving, matched, distances, _source_indices, _target_indices = correspondences
            if len(moving) < solver_options.min_correspondences:
                return _empty("insufficient_correspondences", train_ids, holdout_ids)
            if _scatter_rank(moving, solver_options.rank_tolerance) < 3:
                return _empty(
                    "degenerate_geometry",
                    train_ids,
                    holdout_ids,
                    len(moving),
                    len(moving) / len(train),
                )
            spectrum, rank, condition = _information_diagnostics(moving, solver_options)
            if rank < 6 or condition > solver_options.max_condition_number:
                return _empty(
                    "degenerate_geometry",
                    train_ids,
                    holdout_ids,
                    len(moving),
                    len(moving) / len(train),
                    spectrum,
                    rank,
                    condition,
                    tuple(history),
                )
            delta = _rigid_alignment(moving, matched)
            transform = delta.compose(transform)
            translation_step = float(np.linalg.norm(delta.translation_m))
            rotation_step = _rotation_angle(delta)
            history.append(
                IcpIteration(
                    iteration,
                    len(moving),
                    float(math.sqrt(np.mean(distances * distances))),
                    translation_step,
                    rotation_step,
                )
            )
            if max(translation_step, rotation_step) <= solver_options.convergence_tolerance_m:
                status = "converged"
                reason = "transform increment fell below tolerance"
                break

        transformed = _transform_points(_positions(train), transform)
        correspondences = _correspondences(transformed, target_array, solver_options)
        moving, _matched, distances, source_indices, target_indices = correspondences
        spectrum, rank, condition = _information_diagnostics(moving, solver_options)
        holdout_rmse = _nearest_rmse(
            _transform_points(_positions(holdout), transform), target_array
        )
        rematching = (
            _rematching_diagnostics(
                _positions(train),
                target_array,
                transform,
                source_indices,
                target_indices,
                solver_options,
            )
            if solver_options.effective_diagnostics
            else None
        )
        multi_start_trials = (
            _multi_start_diagnostics(
                source,
                target,
                seed_transform,
                transform,
                holdout_rmse,
                solver_options,
            )
            if solver_options.multi_start_diagnostics
            else ()
        )
        translation_spread = [
            item.solution_translation_delta_m
            for item in multi_start_trials
            if item.solution_translation_delta_m is not None
        ]
        rotation_spread = [
            item.solution_rotation_delta_deg
            for item in multi_start_trials
            if item.solution_rotation_delta_deg is not None
        ]
        return RobustPointToPointIcpResult(
            status=status,
            reason=reason,
            transform_target_source=transform,
            train_source_ids=train_ids,
            holdout_source_ids=holdout_ids,
            train_rmse_m=float(math.sqrt(np.mean(distances * distances)))
            if len(distances)
            else None,
            holdout_rmse_m=holdout_rmse,
            correspondence_count=len(moving),
            inlier_fraction=len(moving) / len(train),
            information_singular_values=spectrum,
            information_rank=rank,
            condition_number=condition,
            rematching_diagnostics=rematching,
            multi_start_trials=multi_start_trials,
            symmetry_ambiguous=any(
                item.symmetry_ambiguous is True for item in multi_start_trials
            ),
            solution_spread_translation_m=(
                max(translation_spread) if translation_spread else None
            ),
            solution_spread_rotation_deg=(
                max(rotation_spread) if rotation_spread else None
            ),
            history=tuple(history),
        )


def evaluate_icp_candidate(
    source_points: Sequence[IcpPoint],
    target_points: Sequence[IcpPoint],
    transform_target_source: SE3,
    options: RobustPointToPointIcpOptions | None = None,
) -> IcpCandidateEvaluation:
    """Evaluate any backend transform under the native spatial holdout contract."""

    solver_options = options or RobustPointToPointIcpOptions()
    _validate_options(solver_options)
    source = sorted(source_points, key=lambda item: item.point_id)
    target = sorted(target_points, key=lambda item: item.point_id)
    train_indices, holdout_indices = _spatial_split_indices(
        source,
        solver_options.holdout_ratio,
        solver_options.holdout_voxel_size_m,
        solver_options.split_seed,
    )
    train = [source[index] for index in train_indices]
    holdout = [source[index] for index in holdout_indices]
    train_ids = tuple(item.point_id for item in train)
    holdout_ids = tuple(item.point_id for item in holdout)
    if not train or not target:
        return IcpCandidateEvaluation(train_ids, holdout_ids, None, None, 0, 0.0, None)
    train_array = _positions(train)
    target_array = _positions(target)
    transformed = _transform_points(train_array, transform_target_source)
    moving, _matched, distances, source_indices, target_indices = _correspondences(
        transformed, target_array, solver_options
    )
    holdout_rmse = _nearest_rmse(
        _transform_points(_positions(holdout), transform_target_source), target_array
    )
    rematching = (
        _rematching_diagnostics(
            train_array,
            target_array,
            transform_target_source,
            source_indices,
            target_indices,
            solver_options,
        )
        if solver_options.effective_diagnostics and len(moving)
        else None
    )
    return IcpCandidateEvaluation(
        train_source_ids=train_ids,
        holdout_source_ids=holdout_ids,
        train_rmse_m=(
            float(math.sqrt(np.mean(distances * distances))) if len(distances) else None
        ),
        holdout_rmse_m=holdout_rmse,
        correspondence_count=len(moving),
        inlier_fraction=len(moving) / len(train),
        rematching_diagnostics=rematching,
    )


def split_icp_source_points(
    source_points: Sequence[IcpPoint],
    options: RobustPointToPointIcpOptions | None = None,
) -> tuple[list[IcpPoint], list[IcpPoint]]:
    """Return the deterministic spatial train/holdout source split."""

    solver_options = options or RobustPointToPointIcpOptions()
    source = sorted(source_points, key=lambda item: item.point_id)
    train_indices, holdout_indices = _spatial_split_indices(
        source,
        solver_options.holdout_ratio,
        solver_options.holdout_voxel_size_m,
        solver_options.split_seed,
    )
    return (
        [source[index] for index in train_indices],
        [source[index] for index in holdout_indices],
    )


def _correspondences(
    transformed_source: FloatArray,
    target: FloatArray,
    options: RobustPointToPointIcpOptions,
) -> tuple[
    FloatArray, FloatArray, FloatArray, NDArray[np.int64], NDArray[np.int64]
]:
    squared = np.sum((transformed_source[:, None, :] - target[None, :, :]) ** 2, axis=2)
    target_indices = np.argmin(squared, axis=1)
    distances = np.sqrt(squared[np.arange(len(transformed_source)), target_indices])
    mask = distances <= options.correspondence_distance_m
    if options.mutual_correspondences:
        source_indices = np.argmin(squared, axis=0)
        mask &= source_indices[target_indices] == np.arange(len(transformed_source))
    selected = np.flatnonzero(mask)
    if len(selected):
        keep = max(1, math.ceil(options.trim_fraction * len(selected)))
        selected = selected[np.argsort(distances[selected], kind="stable")[:keep]]
    return (
        transformed_source[selected],
        target[target_indices[selected]],
        distances[selected],
        selected.astype(np.int64),
        target_indices[selected].astype(np.int64),
    )


def _rematching_diagnostics(
    source: FloatArray,
    target: FloatArray,
    transform: SE3,
    baseline_source_indices: NDArray[np.int64],
    baseline_target_indices: NDArray[np.int64],
    options: RobustPointToPointIcpOptions,
) -> IcpRematchingDiagnostics:
    directions: tuple[Literal["x", "y", "z", "roll", "pitch", "yaw"], ...] = (
        "x",
        "y",
        "z",
        "roll",
        "pitch",
        "yaw",
    )
    baseline_objective, baseline_pairs = _rematched_objective_and_pairs(
        source, target, transform, options
    )
    if baseline_objective is None:
        baseline_pairs = set(
            zip(
                baseline_source_indices.tolist(),
                baseline_target_indices.tolist(),
                strict=True,
            )
        )
    curvatures: list[float | None] = []
    jaccards: list[float | None] = []
    weak: list[str] = []
    axes: tuple[Vector3, Vector3, Vector3] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    for index, direction in enumerate(directions):
        is_translation = index < 3
        step = (
            options.curvature_translation_step_m
            if is_translation
            else options.curvature_rotation_step_rad
        )
        minus = _left_perturbation(axes[index % 3], -step, is_translation).compose(
            transform
        )
        plus = _left_perturbation(axes[index % 3], step, is_translation).compose(
            transform
        )
        minus_objective, minus_pairs = _rematched_objective_and_pairs(
            source, target, minus, options
        )
        plus_objective, plus_pairs = _rematched_objective_and_pairs(
            source, target, plus, options
        )
        if (
            baseline_objective is None
            or minus_objective is None
            or plus_objective is None
        ):
            curvature = None
        else:
            curvature = max(
                0.0,
                (plus_objective - 2.0 * baseline_objective + minus_objective)
                / (step * step),
            )
        curvatures.append(curvature)
        jaccards.append(
            _mean_optional(
                (
                    _set_jaccard(baseline_pairs, minus_pairs),
                    _set_jaccard(baseline_pairs, plus_pairs),
                )
            )
        )
        threshold = (
            options.min_translation_curvature
            if is_translation
            else options.min_rotation_curvature_m2_per_rad2
        )
        if curvature is None or curvature < threshold:
            weak.append(direction)
    available_jaccards = [value for value in jaccards if value is not None]
    return IcpRematchingDiagnostics(
        directions=directions,
        objective_curvatures=tuple(curvatures),
        correspondence_jaccards=tuple(jaccards),
        minimum_correspondence_jaccard=(
            min(available_jaccards) if available_jaccards else None
        ),
        mean_correspondence_jaccard=(
            float(np.mean(available_jaccards)) if available_jaccards else None
        ),
        weak_directions=tuple(weak),
        translation_step_m=options.curvature_translation_step_m,
        rotation_step_rad=options.curvature_rotation_step_rad,
    )


def _rematched_objective_and_pairs(
    source: FloatArray,
    target: FloatArray,
    transform: SE3,
    options: RobustPointToPointIcpOptions,
) -> tuple[float | None, set[tuple[int, int]]]:
    transformed = _transform_points(source, transform)
    _moving, _matched, distances, source_indices, target_indices = _correspondences(
        transformed, target, options
    )
    pairs = set(
        zip(source_indices.tolist(), target_indices.tolist(), strict=True)
    )
    if len(distances) < options.min_correspondences:
        return None, pairs
    return float(np.mean(distances * distances)), pairs


def _left_perturbation(axis: Vector3, amount: float, translation: bool) -> SE3:
    if translation:
        return SE3(
            (amount * axis[0], amount * axis[1], amount * axis[2]),
            (0.0, 0.0, 0.0, 1.0),
        )
    half = amount / 2.0
    return SE3(
        (0.0, 0.0, 0.0),
        (
            axis[0] * math.sin(half),
            axis[1] * math.sin(half),
            axis[2] * math.sin(half),
            math.cos(half),
        ),
    )


def _set_jaccard(
    left: set[tuple[int, int]], right: set[tuple[int, int]]
) -> float | None:
    union = left | right
    return len(left & right) / len(union) if union else None


def _mean_optional(values: Sequence[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    return float(np.mean(available)) if available else None


def _multi_start_diagnostics(
    source: Sequence[IcpPoint],
    target: Sequence[IcpPoint],
    seed_transform: SE3,
    baseline_transform: SE3,
    baseline_holdout_rmse_m: float | None,
    options: RobustPointToPointIcpOptions,
) -> tuple[IcpMultiStartTrial, ...]:
    directions: tuple[Literal["x", "y", "z", "roll", "pitch", "yaw"], ...] = (
        "x",
        "y",
        "z",
        "roll",
        "pitch",
        "yaw",
    )
    axes: tuple[Vector3, Vector3, Vector3] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    nested_options = replace(
        options,
        effective_diagnostics=False,
        multi_start_diagnostics=False,
    )
    trials: list[IcpMultiStartTrial] = []
    for index, direction in enumerate(directions):
        is_translation = index < 3
        amount = (
            options.multi_start_translation_m
            if is_translation
            else options.multi_start_rotation_rad
        )
        initial = _left_perturbation(
            axes[index % 3], amount, is_translation
        ).compose(seed_transform)
        result = RobustPointToPointIcpSolver().solve(
            source, target, initial_transform=initial, options=nested_options
        )
        if result.transform_target_source is None:
            translation_delta = None
            rotation_delta = None
        else:
            delta = baseline_transform.inverse().compose(result.transform_target_source)
            translation_delta = float(np.linalg.norm(delta.translation_m))
            rotation_delta = math.degrees(_rotation_angle(delta))
        equivalent = (
            abs(result.holdout_rmse_m - baseline_holdout_rmse_m)
            <= options.symmetry_equivalent_holdout_margin_m
            if result.holdout_rmse_m is not None
            and baseline_holdout_rmse_m is not None
            else None
        )
        distinct = (
            translation_delta > options.symmetry_translation_separation_m
            or rotation_delta > options.symmetry_rotation_separation_deg
            if translation_delta is not None and rotation_delta is not None
            else None
        )
        ambiguous = (
            equivalent and distinct
            if equivalent is not None and distinct is not None
            else None
        )
        trials.append(
            IcpMultiStartTrial(
                direction=direction,
                amount=amount,
                unit="m" if is_translation else "rad",
                status=result.status,
                holdout_rmse_m=result.holdout_rmse_m,
                solution_translation_delta_m=translation_delta,
                solution_rotation_delta_deg=rotation_delta,
                equivalent_holdout=equivalent,
                distinct_solution=distinct,
                symmetry_ambiguous=ambiguous,
            )
        )
    return tuple(trials)


def _rigid_alignment(source: FloatArray, target: FloatArray) -> SE3:
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    covariance = (target - target_center).T @ (source - source_center)
    left, _spectrum, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(left @ right_t)
    rotation = left @ correction @ right_t
    translation = target_center - rotation @ source_center
    return SE3(
        (float(translation[0]), float(translation[1]), float(translation[2])),
        quaternion_xyzw_from_rotation_matrix(rotation.reshape(-1)),
    )


def _information_diagnostics(
    points: FloatArray, options: RobustPointToPointIcpOptions
) -> tuple[tuple[float, float, float, float, float, float], int, float]:
    rows: list[list[float]] = []
    for x, y, z in points:
        rows.extend(
            [
                [1.0, 0.0, 0.0, 0.0, z, -y],
                [0.0, 1.0, 0.0, -z, 0.0, x],
                [0.0, 0.0, 1.0, y, -x, 0.0],
            ]
        )
    spectrum = np.linalg.svd(np.asarray(rows), compute_uv=False)
    values = tuple(float(value) for value in spectrum[:6])
    rank = int(np.count_nonzero(spectrum > options.rank_tolerance * spectrum[0]))
    condition = values[0] / values[5] if values[5] > 0.0 else math.inf
    return (values[0], values[1], values[2], values[3], values[4], values[5]), rank, condition


def _positions(points: Sequence[IcpPoint]) -> FloatArray:
    return np.asarray([item.position_m for item in points], dtype=np.float64).reshape((-1, 3))


def _spatial_split_indices(
    points: Sequence[IcpPoint], ratio: float, voxel_size_m: float, seed: int
) -> tuple[list[int], list[int]]:
    if voxel_size_m <= 0.0:
        raise ValueError("holdout_voxel_size_m must be positive")
    voxel_to_indices: dict[tuple[int, int, int], list[int]] = {}
    for index, item in enumerate(points):
        x, y, z = item.position_m
        voxel = (
            math.floor(x / voxel_size_m),
            math.floor(y / voxel_size_m),
            math.floor(z / voxel_size_m),
        )
        voxel_to_indices.setdefault(voxel, []).append(index)
    voxels = sorted(voxel_to_indices)
    train_voxels, holdout_voxels = split_indices(len(voxels), ratio, seed=seed)
    train = [
        index for voxel_index in train_voxels for index in voxel_to_indices[voxels[voxel_index]]
    ]
    holdout = [
        index for voxel_index in holdout_voxels for index in voxel_to_indices[voxels[voxel_index]]
    ]
    return train, holdout


def _scatter_rank(points: FloatArray, tolerance: float) -> int:
    centered = points - np.mean(points, axis=0)
    spectrum = np.linalg.svd(centered, compute_uv=False)
    return int(np.count_nonzero(spectrum > tolerance * spectrum[0]))


def _transform_points(points: FloatArray, transform: SE3) -> FloatArray:
    rotation = _rotation_matrix(transform)
    return points @ rotation.T + np.asarray(transform.translation_m)


def _nearest_rmse(points: FloatArray, target: FloatArray) -> float | None:
    if not len(points) or not len(target):
        return None
    squared = np.sum((points[:, None, :] - target[None, :, :]) ** 2, axis=2)
    return float(math.sqrt(np.mean(np.min(squared, axis=1))))


def _rotation_angle(transform: SE3) -> float:
    return 2.0 * math.acos(min(1.0, abs(transform.rotation_quat_xyzw[3])))


def _rotation_matrix(transform: SE3) -> FloatArray:
    x, y, z, w = transform.rotation_quat_xyzw
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _validate_options(options: RobustPointToPointIcpOptions) -> None:
    if not 0.0 < options.trim_fraction <= 1.0:
        raise ValueError("trim_fraction must be in (0, 1]")
    if options.correspondence_distance_m <= 0.0:
        raise ValueError("correspondence_distance_m must be positive")
    if options.curvature_translation_step_m <= 0.0:
        raise ValueError("curvature_translation_step_m must be positive")
    if options.curvature_rotation_step_rad <= 0.0:
        raise ValueError("curvature_rotation_step_rad must be positive")
    if options.multi_start_translation_m <= 0.0:
        raise ValueError("multi_start_translation_m must be positive")
    if options.multi_start_rotation_rad <= 0.0:
        raise ValueError("multi_start_rotation_rad must be positive")


def _empty(
    status: IcpStatus,
    train_ids: tuple[str, ...],
    holdout_ids: tuple[str, ...],
    count: int = 0,
    fraction: float = 0.0,
    spectrum: tuple[float, float, float, float, float, float] | None = None,
    rank: int = 0,
    condition: float | None = None,
    history: tuple[IcpIteration, ...] = (),
) -> RobustPointToPointIcpResult:
    reasons = {
        "insufficient_correspondences": "too few gated point correspondences",
        "degenerate_geometry": "point geometry does not constrain all six DoF",
        "converged": "",
        "max_iterations": "",
    }
    return RobustPointToPointIcpResult(
        status,
        reasons[status],
        None,
        train_ids,
        holdout_ids,
        None,
        None,
        count,
        fraction,
        spectrum,
        rank,
        condition,
        None,
        (),
        False,
        None,
        None,
        history,
    )
