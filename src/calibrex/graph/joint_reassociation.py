"""Leakage-safe outer-loop reassociation for backend-neutral joint factors."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, TypeAlias

from calibrex.evaluation.correspondence import (
    CorrespondenceAssignment,
    CorrespondenceStabilityEvaluation,
    evaluate_correspondence_stability,
)
from calibrex.graph.joint_optimization import (
    BackendNeutralJointOptimizer,
    JointOptimizerOptions,
    JointOptimizerResult,
    JointParameterBlock,
    JointResidualBlock,
    ParameterValues,
)

JointReassociationStatus = Literal[
    "converged",
    "max_iterations",
    "optimizer_failure",
    "invalid_update",
]


@dataclass(frozen=True)
class JointReassociationState:
    """One deterministic one-query/one-factor correspondence assignment."""

    factors: tuple[JointResidualBlock, ...]
    assignments: tuple[CorrespondenceAssignment, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "factor_ids": [factor.factor_id for factor in self.factors],
            "observation_groups": sorted(
                {factor.observation_group for factor in self.factors}
            ),
            "assignments": [item.__dict__ for item in self.assignments],
        }


JointReassociationCallback: TypeAlias = Callable[
    [ParameterValues], JointReassociationState
]


@dataclass(frozen=True)
class JointReassociationOptions:
    max_outer_iterations: int = 3
    minimum_train_pair_jaccard: float = 0.99
    minimum_train_retained_fraction: float = 0.95


@dataclass(frozen=True)
class JointReassociationIteration:
    outer_iteration: int
    data_factor_count: int
    train_stability: CorrespondenceStabilityEvaluation
    holdout_stability: CorrespondenceStabilityEvaluation
    optimizer_status: str | None
    block_max_abs_parameter_delta: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "outer_iteration": self.outer_iteration,
            "data_factor_count": self.data_factor_count,
            "train_stability": self.train_stability.as_dict(),
            "holdout_stability": self.holdout_stability.as_dict(),
            "optimizer_status": self.optimizer_status,
            "block_max_abs_parameter_delta": dict(
                sorted(self.block_max_abs_parameter_delta.items())
            ),
        }


@dataclass(frozen=True)
class JointReassociationResult:
    status: JointReassociationStatus
    reason: str
    initial_result: JointOptimizerResult
    final_result: JointOptimizerResult
    train_observation_groups: tuple[str, ...]
    holdout_observation_groups: tuple[str, ...]
    iterations: tuple[JointReassociationIteration, ...]
    terminal_train_stability: CorrespondenceStabilityEvaluation
    terminal_holdout_stability: CorrespondenceStabilityEvaluation

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "initial_optimizer_result": self.initial_result.as_dict(),
            "final_optimizer_result": self.final_result.as_dict(),
            "train_observation_groups": list(self.train_observation_groups),
            "holdout_observation_groups": list(self.holdout_observation_groups),
            "iterations": [item.as_dict() for item in self.iterations],
            "terminal_train_stability": self.terminal_train_stability.as_dict(),
            "terminal_holdout_stability": self.terminal_holdout_stability.as_dict(),
            "method": "backend_neutral_train_only_reassociation/v0.1",
            "stopping_policy": (
                "train assignment pair Jaccard and retained-query fraction only; "
                "holdout stability is diagnostic"
            ),
            "split_policy": "seeded observation groups remain invariant across rounds",
            "parameter_delta_policy": "per-block max absolute tangent delta; unit-dependent",
        }


class BackendNeutralJointReassociation:
    """Alternate deterministic reassociation and warm-start joint optimization."""

    def refine(
        self,
        parameter_blocks: Sequence[JointParameterBlock],
        initial_state: JointReassociationState,
        train_only_factors: Sequence[JointResidualBlock],
        initial_result: JointOptimizerResult,
        reassociate: JointReassociationCallback,
        optimizer_options: JointOptimizerOptions,
        options: JointReassociationOptions | None = None,
    ) -> JointReassociationResult:
        outer_options = options or JointReassociationOptions()
        _validate_options(outer_options)
        blocks = tuple(parameter_blocks)
        priors = tuple(train_only_factors)
        _validate_initial_contract(blocks, initial_state, priors, initial_result)
        train_groups = initial_result.train_observation_groups
        holdout_groups = initial_result.holdout_observation_groups
        immutable_query_groups = {
            factor.factor_id: factor.observation_group for factor in initial_state.factors
        }
        current_state = initial_state
        current_result = initial_result
        iterations: list[JointReassociationIteration] = []
        terminal_train, terminal_holdout = _split_stability(
            current_state.assignments,
            current_state.assignments,
            immutable_query_groups,
            train_groups,
            holdout_groups,
        )

        for outer_iteration in range(1, outer_options.max_outer_iterations + 1):
            try:
                candidate_state = reassociate(current_result.optimized_values)
            except (KeyError, ValueError) as exc:
                return _result(
                    "invalid_update",
                    f"reassociation callback rejected the current state: {exc}",
                    initial_result,
                    current_result,
                    train_groups,
                    holdout_groups,
                    iterations,
                    terminal_train,
                    terminal_holdout,
                )
            error = _state_error(
                candidate_state,
                immutable_query_groups,
                train_groups,
                holdout_groups,
            )
            if error is not None:
                return _result(
                    "invalid_update",
                    error,
                    initial_result,
                    current_result,
                    train_groups,
                    holdout_groups,
                    iterations,
                    terminal_train,
                    terminal_holdout,
                )
            train_stability, holdout_stability = _split_stability(
                current_state.assignments,
                candidate_state.assignments,
                immutable_query_groups,
                train_groups,
                holdout_groups,
            )
            terminal_train, terminal_holdout = train_stability, holdout_stability
            if _stable(train_stability, outer_options):
                iterations.append(
                    JointReassociationIteration(
                        outer_iteration,
                        len(candidate_state.factors),
                        train_stability,
                        holdout_stability,
                        None,
                        {},
                    )
                )
                return _result(
                    "converged",
                    "training correspondences reached the declared stability gates",
                    initial_result,
                    current_result,
                    train_groups,
                    holdout_groups,
                    iterations,
                    terminal_train,
                    terminal_holdout,
                )

            warm_blocks = tuple(
                replace(block, initial_values=current_result.optimized_values[block.name])
                for block in blocks
            )
            try:
                optimized = BackendNeutralJointOptimizer().solve(
                    warm_blocks,
                    (*candidate_state.factors, *priors),
                    optimizer_options,
                )
            except ValueError as exc:
                return _result(
                    "invalid_update",
                    f"reassociated factor graph is invalid: {exc}",
                    initial_result,
                    current_result,
                    train_groups,
                    holdout_groups,
                    iterations,
                    terminal_train,
                    terminal_holdout,
                )
            parameter_delta = _block_parameter_delta(
                current_result.optimized_values,
                optimized.optimized_values,
            )
            iterations.append(
                JointReassociationIteration(
                    outer_iteration,
                    len(candidate_state.factors),
                    train_stability,
                    holdout_stability,
                    optimized.status,
                    parameter_delta,
                )
            )
            if (
                optimized.train_observation_groups != train_groups
                or optimized.holdout_observation_groups != holdout_groups
            ):
                return _result(
                    "invalid_update",
                    "reassociation changed the seeded train/holdout observation groups",
                    initial_result,
                    current_result,
                    train_groups,
                    holdout_groups,
                    iterations,
                    terminal_train,
                    terminal_holdout,
                )
            if optimized.status != "converged":
                return _result(
                    "optimizer_failure",
                    f"inner optimizer status is {optimized.status}: {optimized.reason}",
                    initial_result,
                    optimized,
                    train_groups,
                    holdout_groups,
                    iterations,
                    terminal_train,
                    terminal_holdout,
                )
            current_state = candidate_state
            current_result = optimized

        try:
            terminal_state = reassociate(current_result.optimized_values)
        except (KeyError, ValueError) as exc:
            return _result(
                "invalid_update",
                f"terminal reassociation callback failed: {exc}",
                initial_result,
                current_result,
                train_groups,
                holdout_groups,
                iterations,
                terminal_train,
                terminal_holdout,
            )
        error = _state_error(
            terminal_state,
            immutable_query_groups,
            train_groups,
            holdout_groups,
        )
        if error is not None:
            return _result(
                "invalid_update",
                error,
                initial_result,
                current_result,
                train_groups,
                holdout_groups,
                iterations,
                terminal_train,
                terminal_holdout,
            )
        terminal_train, terminal_holdout = _split_stability(
            current_state.assignments,
            terminal_state.assignments,
            immutable_query_groups,
            train_groups,
            holdout_groups,
        )
        if _stable(terminal_train, outer_options):
            return _result(
                "converged",
                "terminal training correspondences reached the declared stability gates",
                initial_result,
                current_result,
                train_groups,
                holdout_groups,
                iterations,
                terminal_train,
                terminal_holdout,
            )
        return _result(
            "max_iterations",
            "training correspondences remained unstable at the outer iteration limit",
            initial_result,
            current_result,
            train_groups,
            holdout_groups,
            iterations,
            terminal_train,
            terminal_holdout,
        )


def _validate_options(options: JointReassociationOptions) -> None:
    if options.max_outer_iterations <= 0:
        raise ValueError("reassociation outer iteration count must be positive")
    fractions = (
        options.minimum_train_pair_jaccard,
        options.minimum_train_retained_fraction,
    )
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in fractions):
        raise ValueError("reassociation stability gates must be finite fractions")


def _validate_initial_contract(
    blocks: Sequence[JointParameterBlock],
    state: JointReassociationState,
    priors: tuple[JointResidualBlock, ...],
    result: JointOptimizerResult,
) -> None:
    error = _state_shape_error(state)
    if error is not None:
        raise ValueError(error)
    if any(factor.split_policy != "grouped" for factor in state.factors):
        raise ValueError("reassociation data factors must use grouped splitting")
    if any(factor.split_policy != "train_only" for factor in priors):
        raise ValueError("reassociation fixed factors must be train-only")
    if {block.name for block in blocks} != set(result.optimized_values):
        raise ValueError("reassociation parameter blocks must match optimized values")
    expected_factor_ids = {
        *(factor.factor_id for factor in state.factors),
        *(factor.factor_id for factor in priors),
    }
    observed_factor_ids = set(result.train_factor_ids) | set(
        result.holdout_factor_ids
    )
    if expected_factor_ids != observed_factor_ids:
        raise ValueError("initial reassociation factors must match optimizer factor IDs")
    groups = {factor.observation_group for factor in state.factors}
    expected = set(result.train_observation_groups) | set(
        result.holdout_observation_groups
    )
    if groups != expected:
        raise ValueError("initial reassociation groups must match the optimizer split")


def _state_shape_error(state: JointReassociationState) -> str | None:
    factor_ids = [factor.factor_id for factor in state.factors]
    assignment_ids = [assignment.query_id for assignment in state.assignments]
    if len(factor_ids) != len(set(factor_ids)):
        return "reassociation factor IDs must be unique"
    if len(assignment_ids) != len(set(assignment_ids)):
        return "reassociation assignment query IDs must be unique"
    if set(factor_ids) != set(assignment_ids):
        return "reassociation factors and assignment query IDs must match exactly"
    return None


def _state_error(
    state: JointReassociationState,
    immutable_query_groups: Mapping[str, str],
    train_groups: tuple[str, ...],
    holdout_groups: tuple[str, ...],
) -> str | None:
    error = _state_shape_error(state)
    if error is not None:
        return error
    if any(factor.split_policy != "grouped" for factor in state.factors):
        return "reassociated data factors must keep grouped splitting"
    current = {factor.factor_id: factor.observation_group for factor in state.factors}
    if any(
        query_id not in immutable_query_groups
        or immutable_query_groups[query_id] != observation_group
        for query_id, observation_group in current.items()
    ):
        return "reassociation added a query or changed its observation group"
    expected_groups = set(train_groups) | set(holdout_groups)
    if set(current.values()) != expected_groups:
        return "reassociation dropped a complete train or holdout observation group"
    return None


def _split_stability(
    reference: tuple[CorrespondenceAssignment, ...],
    rematched: tuple[CorrespondenceAssignment, ...],
    query_groups: Mapping[str, str],
    train_groups: tuple[str, ...],
    holdout_groups: tuple[str, ...],
) -> tuple[CorrespondenceStabilityEvaluation, CorrespondenceStabilityEvaluation]:
    train = set(train_groups)
    holdout = set(holdout_groups)

    def selected(
        assignments: tuple[CorrespondenceAssignment, ...], groups: set[str]
    ) -> tuple[CorrespondenceAssignment, ...]:
        return tuple(
            item
            for item in assignments
            if query_groups.get(item.query_id) in groups
        )

    return (
        evaluate_correspondence_stability(
            selected(reference, train), selected(rematched, train)
        ),
        evaluate_correspondence_stability(
            selected(reference, holdout), selected(rematched, holdout)
        ),
    )


def _stable(
    evaluation: CorrespondenceStabilityEvaluation,
    options: JointReassociationOptions,
) -> bool:
    return (
        evaluation.pair_jaccard is not None
        and evaluation.pair_jaccard >= options.minimum_train_pair_jaccard
        and evaluation.retained_query_fraction is not None
        and evaluation.retained_query_fraction
        >= options.minimum_train_retained_fraction
    )


def _block_parameter_delta(
    before: Mapping[str, tuple[float, ...]],
    after: Mapping[str, tuple[float, ...]],
) -> dict[str, float]:
    return {
        name: max(
            (abs(right - left) for left, right in zip(values, after[name], strict=True)),
            default=0.0,
        )
        for name, values in before.items()
    }


def _result(
    status: JointReassociationStatus,
    reason: str,
    initial: JointOptimizerResult,
    final: JointOptimizerResult,
    train_groups: tuple[str, ...],
    holdout_groups: tuple[str, ...],
    iterations: list[JointReassociationIteration],
    terminal_train: CorrespondenceStabilityEvaluation,
    terminal_holdout: CorrespondenceStabilityEvaluation,
) -> JointReassociationResult:
    return JointReassociationResult(
        status,
        reason,
        initial,
        final,
        train_groups,
        holdout_groups,
        tuple(iterations),
        terminal_train,
        terminal_holdout,
    )
