import pytest

from calibrex.evaluation.correspondence import CorrespondenceAssignment
from calibrex.graph.joint_optimization import (
    BackendNeutralJointOptimizer,
    JointOptimizerOptions,
    JointParameterBlock,
    JointResidualBlock,
)
from calibrex.graph.joint_reassociation import (
    BackendNeutralJointReassociation,
    JointReassociationOptions,
    JointReassociationState,
)


def _factor(query_id: str, group: str, target_id: str, target: float) -> JointResidualBlock:
    return JointResidualBlock(
        query_id,
        group,
        ("x",),
        lambda values: (values["x"][0] - target,),
        family=f"synthetic_target_{target_id}",
    )


def _state(target_id: str, target: float) -> JointReassociationState:
    factors = tuple(
        _factor(f"query-{index}", f"capture-{index}", target_id, target)
        for index in range(8)
    )
    return JointReassociationState(
        factors,
        tuple(
            CorrespondenceAssignment(factor.factor_id, target_id)
            for factor in factors
        ),
    )


def _initial(
    state: JointReassociationState,
) -> tuple[tuple[JointParameterBlock, ...], JointOptimizerOptions, object]:
    blocks = (JointParameterBlock("x", (0.0,), known_bad_steps=(0.1,)),)
    options = JointOptimizerOptions(
        holdout_ratio=0.25,
        split_seed=5,
        minimum_train_factors=1,
        known_bad_margin=0.01,
    )
    result = BackendNeutralJointOptimizer().solve(blocks, state.factors, options)
    return blocks, options, result


def test_joint_reassociation_recovers_truth_without_group_leakage() -> None:
    initial_state = _state("wrong", 1.0)
    blocks, optimizer_options, initial_result = _initial(initial_state)

    def reassociate(values) -> JointReassociationState:
        return _state("truth", 2.0) if values["x"][0] >= 0.5 else initial_state

    result = BackendNeutralJointReassociation().refine(
        blocks,
        initial_state,
        (),
        initial_result,
        reassociate,
        optimizer_options,
    )

    assert result.status == "converged"
    assert result.final_result.optimized_values["x"] == pytest.approx((2.0,))
    assert result.train_observation_groups == initial_result.train_observation_groups
    assert result.holdout_observation_groups == initial_result.holdout_observation_groups
    assert len(result.iterations) == 2
    assert result.iterations[0].train_stability.pair_jaccard == 0.0
    assert result.iterations[1].train_stability.pair_jaccard == 1.0
    assert result.as_dict()["method"] == "backend_neutral_train_only_reassociation/v0.1"


def test_joint_reassociation_stopping_ignores_holdout_assignment_change() -> None:
    initial_state = _state("stable", 1.0)
    blocks, optimizer_options, initial_result = _initial(initial_state)
    holdout_groups = set(initial_result.holdout_observation_groups)

    def reassociate(_values) -> JointReassociationState:
        assignments = tuple(
            CorrespondenceAssignment(
                factor.factor_id,
                "changed-holdout"
                if factor.observation_group in holdout_groups
                else "stable",
            )
            for factor in initial_state.factors
        )
        return JointReassociationState(initial_state.factors, assignments)

    result = BackendNeutralJointReassociation().refine(
        blocks,
        initial_state,
        (),
        initial_result,
        reassociate,
        optimizer_options,
    )

    assert result.status == "converged"
    assert len(result.iterations) == 1
    assert result.terminal_train_stability.pair_jaccard == 1.0
    assert result.terminal_holdout_stability.pair_jaccard == 0.0
    assert result.iterations[0].optimizer_status is None


def test_joint_reassociation_rejects_group_drop() -> None:
    initial_state = _state("initial", 1.0)
    blocks, optimizer_options, initial_result = _initial(initial_state)
    dropped_group = initial_result.train_observation_groups[0]

    def reassociate(_values) -> JointReassociationState:
        factors = tuple(
            factor
            for factor in initial_state.factors
            if factor.observation_group != dropped_group
        )
        return JointReassociationState(
            factors,
            tuple(
                CorrespondenceAssignment(factor.factor_id, "next")
                for factor in factors
            ),
        )

    result = BackendNeutralJointReassociation().refine(
        blocks,
        initial_state,
        (),
        initial_result,
        reassociate,
        optimizer_options,
    )

    assert result.status == "invalid_update"
    assert "complete train or holdout" in result.reason


def test_joint_reassociation_reports_outer_iteration_limit() -> None:
    initial_state = _state("initial", 0.0)
    blocks, optimizer_options, initial_result = _initial(initial_state)
    calls = 0

    def reassociate(_values) -> JointReassociationState:
        nonlocal calls
        calls += 1
        return _state(f"round-{calls}", float(calls))

    result = BackendNeutralJointReassociation().refine(
        blocks,
        initial_state,
        (),
        initial_result,
        reassociate,
        optimizer_options,
        JointReassociationOptions(max_outer_iterations=2),
    )

    assert result.status == "max_iterations"
    assert len(result.iterations) == 2
    assert result.terminal_train_stability.pair_jaccard == 0.0
