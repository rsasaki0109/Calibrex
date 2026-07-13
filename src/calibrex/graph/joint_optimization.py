"""Backend-neutral residual blocks and robust joint calibration optimizer."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.evaluation.holdout import split_indices

ParameterValues: TypeAlias = Mapping[str, tuple[float, ...]]
ResidualEvaluator: TypeAlias = Callable[[ParameterValues], Sequence[float]]
JointOptimizerStatus = Literal["converged", "max_iterations", "insufficient_factors", "degenerate"]
JointFactorSplitPolicy = Literal["grouped", "train_only"]


@dataclass(frozen=True)
class JointParameterBlock:
    """One named Euclidean tangent block owned by the graph, not a backend."""

    name: str
    initial_values: tuple[float, ...]
    fixed: bool = False
    finite_difference_steps: tuple[float, ...] = ()
    known_bad_steps: tuple[float, ...] = ()

    @property
    def dimension(self) -> int:
        return len(self.initial_values)


@dataclass(frozen=True)
class JointResidualBlock:
    """A factor-local residual callback with explicit variable ownership."""

    factor_id: str
    observation_group: str
    variable_names: tuple[str, ...]
    evaluator: ResidualEvaluator
    weight: float = 1.0
    family: str = "generic"
    split_policy: JointFactorSplitPolicy = "grouped"

    def residuals(self, values: ParameterValues) -> tuple[float, ...]:
        owned = {name: values[name] for name in self.variable_names}
        return tuple(math.sqrt(self.weight) * float(value) for value in self.evaluator(owned))


@dataclass(frozen=True)
class JointOptimizerOptions:
    max_iterations: int = 50
    convergence_tolerance: float = 1.0e-8
    initial_damping: float = 1.0e-3
    huber_delta: float = 1.0
    holdout_ratio: float = 0.2
    split_seed: int = 0
    minimum_train_factors: int = 3
    rank_tolerance: float = 1.0e-8
    max_condition_number: float = 1.0e10
    default_finite_difference_step: float = 1.0e-6
    known_bad_margin: float = 1.0e-3


@dataclass(frozen=True)
class JointOptimizerIteration:
    iteration: int
    train_rmse: float
    train_robust_objective: float
    damping: float
    step_norm: float
    accepted: bool


@dataclass(frozen=True)
class JointKnownBadProbe:
    block: str
    dimension: int
    amount: float
    baseline_holdout_rmse: float | None
    perturbed_holdout_rmse: float | None
    delta: float | None
    detectable: bool | None


@dataclass(frozen=True)
class JointOptimizerResult:
    status: JointOptimizerStatus
    reason: str
    optimized_values: dict[str, tuple[float, ...]]
    train_factor_ids: tuple[str, ...]
    holdout_factor_ids: tuple[str, ...]
    train_observation_groups: tuple[str, ...]
    holdout_observation_groups: tuple[str, ...]
    train_rmse: float | None
    holdout_rmse: float | None
    information_singular_values: tuple[float, ...]
    information_rank: int
    condition_number: float | None
    weak_parameter_blocks: tuple[str, ...]
    probes: tuple[JointKnownBadProbe, ...]
    history: tuple[JointOptimizerIteration, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "optimized_values": {
                name: list(values) for name, values in sorted(self.optimized_values.items())
            },
            "train_factor_ids": list(self.train_factor_ids),
            "holdout_factor_ids": list(self.holdout_factor_ids),
            "train_observation_groups": list(self.train_observation_groups),
            "holdout_observation_groups": list(self.holdout_observation_groups),
            "train_rmse": self.train_rmse,
            "holdout_rmse": self.holdout_rmse,
            "information_singular_values": self.information_singular_values,
            "information_rank": self.information_rank,
            "condition_number": self.condition_number,
            "weak_parameter_blocks": list(self.weak_parameter_blocks),
            "known_bad_probes": [probe.__dict__ for probe in self.probes],
            "history": [item.__dict__ for item in self.history],
            "method": "backend_neutral_robust_joint_lm/v0.1",
            "information_policy": "train residual Jacobian; diagnostic, not covariance",
        }


class BackendNeutralJointOptimizer:
    """Optimize arbitrary coupled residual blocks with numeric factor Jacobians."""

    def solve(
        self,
        parameter_blocks: Sequence[JointParameterBlock],
        factors: Sequence[JointResidualBlock],
        options: JointOptimizerOptions | None = None,
    ) -> JointOptimizerResult:
        solver_options = options or JointOptimizerOptions()
        blocks = _validate_blocks(parameter_blocks)
        train, holdout, train_groups, holdout_groups = split_joint_factors(
            factors, solver_options.holdout_ratio, solver_options.split_seed
        )
        initial = {block.name: block.initial_values for block in blocks}
        if len(train) < solver_options.minimum_train_factors:
            return _empty(
                "insufficient_factors", initial, train, holdout, train_groups, holdout_groups
            )
        layout = _variable_layout(blocks)
        if not layout:
            return _empty("degenerate", initial, train, holdout, train_groups, holdout_groups)
        vector = _pack(initial, layout)
        damping = solver_options.initial_damping
        history: list[JointOptimizerIteration] = []
        status: JointOptimizerStatus = "max_iterations"
        for iteration in range(solver_options.max_iterations):
            values = _unpack(vector, initial, layout)
            residual, jacobian = _linearize(train, values, blocks, layout, solver_options)
            if residual.size == 0:
                return _empty(
                    "insufficient_factors",
                    values,
                    train,
                    holdout,
                    train_groups,
                    holdout_groups,
                )
            weights = _huber_weights(residual, solver_options.huber_delta)
            weighted_jacobian = jacobian * np.sqrt(weights)[:, None]
            weighted_residual = residual * np.sqrt(weights)
            hessian = weighted_jacobian.T @ weighted_jacobian
            gradient = weighted_jacobian.T @ weighted_residual
            try:
                step = -np.linalg.solve(hessian + damping * np.eye(hessian.shape[0]), gradient)
            except np.linalg.LinAlgError:
                return _empty(
                    "degenerate",
                    values,
                    train,
                    holdout,
                    train_groups,
                    holdout_groups,
                )
            candidate = vector + step
            current_rmse = _rmse(residual)
            current_objective = _huber_objective(residual, solver_options.huber_delta)
            candidate_values = _unpack(candidate, initial, layout)
            candidate_residual = _factor_residuals(train, candidate_values)
            candidate_rmse = _rmse(candidate_residual)
            candidate_objective = _huber_objective(
                candidate_residual, solver_options.huber_delta
            )
            accepted = candidate_objective < current_objective
            if accepted:
                vector = candidate
                damping = max(1.0e-12, damping * 0.5)
            else:
                damping = min(1.0e12, damping * 10.0)
            history.append(
                JointOptimizerIteration(
                    iteration,
                    candidate_rmse if accepted else current_rmse,
                    candidate_objective if accepted else current_objective,
                    damping,
                    float(np.linalg.norm(step)),
                    accepted,
                )
            )
            step_converged = float(np.linalg.norm(step)) <= (
                solver_options.convergence_tolerance
            )
            rejected_stationary = (
                not accepted
                and step_converged
                and float(np.linalg.norm(gradient, ord=np.inf))
                <= math.sqrt(solver_options.convergence_tolerance)
            )
            if (accepted and step_converged) or rejected_stationary:
                status = "converged"
                break
        values = _unpack(vector, initial, layout)
        residual, jacobian = _linearize(train, values, blocks, layout, solver_options)
        singular = np.linalg.svd(jacobian, compute_uv=False) if jacobian.size else np.asarray([])
        rank = int(sum(value > solver_options.rank_tolerance for value in singular))
        condition = (
            float(singular[0] / singular[-1])
            if len(singular) and singular[-1] > solver_options.rank_tolerance
            else None
        )
        weak = _weak_blocks(jacobian, layout, blocks, solver_options.rank_tolerance)
        if (
            rank < len(vector)
            or condition is None
            or condition > solver_options.max_condition_number
        ):
            status = "degenerate"
        train_rmse = _factor_rmse(train, values)
        holdout_rmse = _factor_rmse(holdout, values)
        probes = _known_bad_probes(blocks, holdout, values, holdout_rmse, solver_options)
        reason = {
            "converged": "joint LM step fell below tolerance",
            "max_iterations": "maximum joint LM iterations reached",
            "degenerate": "train Jacobian is rank deficient or ill-conditioned",
            "insufficient_factors": "not enough train factors",
        }[status]
        return JointOptimizerResult(
            status,
            reason,
            values,
            tuple(factor.factor_id for factor in train),
            tuple(factor.factor_id for factor in holdout),
            train_groups,
            holdout_groups,
            train_rmse,
            holdout_rmse,
            tuple(float(value) for value in singular),
            rank,
            condition,
            weak,
            probes,
            tuple(history),
        )


def split_joint_factors(
    factors: Sequence[JointResidualBlock], holdout_ratio: float, seed: int
) -> tuple[
    list[JointResidualBlock],
    list[JointResidualBlock],
    tuple[str, ...],
    tuple[str, ...],
]:
    """Split complete observation groups so correlated factors cannot leak."""

    grouped = [factor for factor in factors if factor.split_policy == "grouped"]
    train_only = [factor for factor in factors if factor.split_policy == "train_only"]
    groups = tuple(sorted({factor.observation_group for factor in grouped}))
    train_indices, holdout_indices = split_indices(len(groups), holdout_ratio, seed)
    if len(groups) > 1 and holdout_ratio > 0.0 and not holdout_indices:
        train_indices, holdout_indices = split_indices(len(groups), 1.0 / len(groups), seed)
    train_groups = tuple(groups[index] for index in train_indices)
    holdout_groups = tuple(groups[index] for index in holdout_indices)
    train_set, holdout_set = set(train_groups), set(holdout_groups)
    train = [factor for factor in grouped if factor.observation_group in train_set]
    train.extend(train_only)
    holdout = [factor for factor in grouped if factor.observation_group in holdout_set]
    return train, holdout, train_groups, holdout_groups


def _linearize(
    factors: Sequence[JointResidualBlock],
    values: dict[str, tuple[float, ...]],
    blocks: Sequence[JointParameterBlock],
    layout: dict[str, slice],
    options: JointOptimizerOptions,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    baseline_parts = [factor.residuals(values) for factor in factors]
    baseline = np.asarray(
        [value for residuals in baseline_parts for value in residuals],
        dtype=np.float64,
    )
    jacobian = np.zeros((len(baseline), sum(item.stop - item.start for item in layout.values())))
    row_slices: list[slice] = []
    row_cursor = 0
    for residuals in baseline_parts:
        row_slices.append(slice(row_cursor, row_cursor + len(residuals)))
        row_cursor += len(residuals)
    block_by_name = {block.name: block for block in blocks}
    for name, block_slice in layout.items():
        block = block_by_name[name]
        affected = [
            (factor, row_slice, len(baseline_parts[index]))
            for index, (factor, row_slice) in enumerate(
                zip(factors, row_slices, strict=True)
            )
            if name in factor.variable_names
        ]
        for local_index in range(block.dimension):
            step = (
                block.finite_difference_steps[local_index]
                if block.finite_difference_steps
                else options.default_finite_difference_step
            )
            plus = _perturb(values, name, local_index, step)
            minus = _perturb(values, name, local_index, -step)
            for factor, row_slice, residual_count in affected:
                plus_residuals = factor.residuals(plus)
                minus_residuals = factor.residuals(minus)
                if (
                    len(plus_residuals) != residual_count
                    or len(minus_residuals) != residual_count
                ):
                    raise ValueError("joint factor residual dimension changed during linearization")
                jacobian[row_slice, block_slice.start + local_index] = (
                    np.asarray(plus_residuals) - np.asarray(minus_residuals)
                ) / (2.0 * step)
    return baseline, jacobian


def _factor_residuals(
    factors: Sequence[JointResidualBlock], values: ParameterValues
) -> NDArray[np.float64]:
    residuals = [value for factor in factors for value in factor.residuals(values)]
    return np.asarray(residuals, dtype=np.float64)


def _factor_rmse(factors: Sequence[JointResidualBlock], values: ParameterValues) -> float | None:
    residuals = _factor_residuals(factors, values)
    return _rmse(residuals) if residuals.size else None


def _rmse(residuals: NDArray[np.float64]) -> float:
    return float(math.sqrt(float(np.mean(residuals * residuals))))


def _huber_weights(residuals: NDArray[np.float64], delta: float) -> NDArray[np.float64]:
    absolute = np.abs(residuals)
    return np.where(absolute <= delta, 1.0, delta / np.maximum(absolute, 1.0e-15))


def _huber_objective(residuals: NDArray[np.float64], delta: float) -> float:
    absolute = np.abs(residuals)
    losses = np.where(
        absolute <= delta,
        0.5 * residuals * residuals,
        delta * (absolute - 0.5 * delta),
    )
    return float(np.sum(losses))


def _validate_blocks(
    blocks: Sequence[JointParameterBlock],
) -> tuple[JointParameterBlock, ...]:
    names: set[str] = set()
    output: list[JointParameterBlock] = []
    for block in blocks:
        if not block.name or block.name in names or block.dimension < 1:
            raise ValueError("joint parameter block names must be unique and non-empty")
        if block.finite_difference_steps and len(block.finite_difference_steps) != block.dimension:
            raise ValueError("finite-difference step dimension mismatch")
        if block.known_bad_steps and len(block.known_bad_steps) != block.dimension:
            raise ValueError("known-bad step dimension mismatch")
        names.add(block.name)
        output.append(block)
    return tuple(output)


def _variable_layout(blocks: Sequence[JointParameterBlock]) -> dict[str, slice]:
    layout: dict[str, slice] = {}
    cursor = 0
    for block in blocks:
        if block.fixed:
            continue
        layout[block.name] = slice(cursor, cursor + block.dimension)
        cursor += block.dimension
    return layout


def _pack(values: ParameterValues, layout: Mapping[str, slice]) -> NDArray[np.float64]:
    vector = np.zeros(sum(item.stop - item.start for item in layout.values()))
    for name, target in layout.items():
        vector[target] = values[name]
    return vector


def _unpack(
    vector: NDArray[np.float64],
    initial: ParameterValues,
    layout: Mapping[str, slice],
) -> dict[str, tuple[float, ...]]:
    values = dict(initial)
    for name, source in layout.items():
        values[name] = tuple(float(value) for value in vector[source])
    return values


def _perturb(
    values: ParameterValues, block: str, dimension: int, amount: float
) -> dict[str, tuple[float, ...]]:
    output = dict(values)
    changed = list(output[block])
    changed[dimension] += amount
    output[block] = tuple(changed)
    return output


def _weak_blocks(
    jacobian: NDArray[np.float64],
    layout: Mapping[str, slice],
    blocks: Sequence[JointParameterBlock],
    tolerance: float,
) -> tuple[str, ...]:
    if not jacobian.size:
        return tuple(layout)
    _u, singular, vt = np.linalg.svd(jacobian, full_matrices=False)
    weak_vectors = vt[singular <= tolerance]
    if not len(weak_vectors):
        return ()
    names: list[str] = []
    for name, block_slice in layout.items():
        if any(float(np.linalg.norm(vector[block_slice])) > 0.25 for vector in weak_vectors):
            names.append(name)
    known = {block.name for block in blocks}
    return tuple(name for name in names if name in known)


def _known_bad_probes(
    blocks: Sequence[JointParameterBlock],
    holdout: Sequence[JointResidualBlock],
    values: dict[str, tuple[float, ...]],
    baseline: float | None,
    options: JointOptimizerOptions,
) -> tuple[JointKnownBadProbe, ...]:
    probes: list[JointKnownBadProbe] = []
    for block in blocks:
        if block.fixed or not block.known_bad_steps:
            continue
        for dimension, step in enumerate(block.known_bad_steps):
            for amount in (-step, step):
                perturbed = _factor_rmse(holdout, _perturb(values, block.name, dimension, amount))
                delta = (
                    perturbed - baseline if perturbed is not None and baseline is not None else None
                )
                probes.append(
                    JointKnownBadProbe(
                        block.name,
                        dimension,
                        amount,
                        baseline,
                        perturbed,
                        delta,
                        delta > options.known_bad_margin if delta is not None else None,
                    )
                )
    return tuple(probes)


def _empty(
    status: Literal["insufficient_factors", "degenerate"],
    values: ParameterValues,
    train: Sequence[JointResidualBlock],
    holdout: Sequence[JointResidualBlock],
    train_groups: tuple[str, ...],
    holdout_groups: tuple[str, ...],
) -> JointOptimizerResult:
    return JointOptimizerResult(
        status,
        "not enough train factors"
        if status == "insufficient_factors"
        else "no free observable parameters",
        dict(values),
        tuple(factor.factor_id for factor in train),
        tuple(factor.factor_id for factor in holdout),
        train_groups,
        holdout_groups,
        None,
        None,
        (),
        0,
        None,
        tuple(values),
        (),
        (),
    )
