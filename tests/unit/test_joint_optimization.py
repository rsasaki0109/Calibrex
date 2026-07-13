import math

import pytest

from calibrex.graph.joint_optimization import (
    BackendNeutralJointOptimizer,
    JointOptimizerOptions,
    JointParameterBlock,
    JointResidualBlock,
    evaluate_joint_observability,
    split_joint_factors,
)


def _joint_problem() -> tuple[list[JointParameterBlock], list[JointResidualBlock]]:
    truth = (0.35, -0.22, 0.08, 0.17, 0.026)
    blocks = [
        JointParameterBlock("world_gauge", (0.0,), fixed=True),
        JointParameterBlock("trajectory", (0.0, 0.0), known_bad_steps=(0.05, 0.05)),
        JointParameterBlock("T_body_sensor", (0.0, 0.0), known_bad_steps=(0.05, 0.05)),
        JointParameterBlock("dt_sensor", (0.0,), known_bad_steps=(0.01,)),
    ]
    factors: list[JointResidualBlock] = []
    for index in range(30):
        t = -1.0 + 2.0 * index / 29.0
        coefficients = (
            1.0,
            t,
            t * t + 0.2,
            math.sin(1.3 * t),
            math.cos(0.7 * t),
        )
        observed = sum(a * b for a, b in zip(coefficients, truth, strict=True))

        def evaluator(
            values: dict[str, tuple[float, ...]],
            coefficients: tuple[float, ...] = coefficients,
            observed: float = observed,
        ) -> tuple[float]:
            state = (
                *values["trajectory"],
                *values["T_body_sensor"],
                *values["dt_sensor"],
            )
            predicted = sum(
                coefficient * value for coefficient, value in zip(coefficients, state, strict=True)
            )
            return (predicted - observed,)

        factors.append(
            JointResidualBlock(
                f"mixed-{index:03d}",
                f"capture-{index:03d}",
                ("trajectory", "T_body_sensor", "dt_sensor"),
                evaluator,
                family="synthetic_multisensor",
            )
        )
    return blocks, factors


def test_joint_optimizer_recovers_coupled_trajectory_extrinsic_and_time() -> None:
    blocks, factors = _joint_problem()

    result = BackendNeutralJointOptimizer().solve(
        blocks,
        factors,
        JointOptimizerOptions(
            holdout_ratio=0.25,
            split_seed=17,
            known_bad_margin=1.0e-5,
        ),
    )

    assert result.status == "converged"
    assert result.optimized_values["world_gauge"] == (0.0,)
    assert result.optimized_values["trajectory"] == pytest.approx((0.35, -0.22), abs=1e-7)
    assert result.optimized_values["T_body_sensor"] == pytest.approx((0.08, 0.17), abs=1e-7)
    assert result.optimized_values["dt_sensor"] == pytest.approx((0.026,), abs=1e-7)
    assert result.train_rmse is not None and result.train_rmse < 1e-8
    assert result.holdout_rmse is not None and result.holdout_rmse < 1e-8
    assert result.information_rank == 5
    assert result.weak_parameter_blocks == ()
    assert len(result.probes) == 10
    assert all(probe.detectable is True for probe in result.probes)
    assert set(result.train_observation_groups).isdisjoint(result.holdout_observation_groups)


def test_observation_group_split_keeps_modal_factors_together() -> None:
    block = JointParameterBlock("x", (0.0,))
    factors = [
        JointResidualBlock(
            f"{family}-{capture}",
            capture,
            ("x",),
            lambda values: (values["x"][0],),
            family=family,
        )
        for capture in ("a", "b", "c", "d")
        for family in ("camera", "lidar", "imu")
    ]

    train, holdout, train_groups, holdout_groups = split_joint_factors(factors, 0.25, 3)

    assert set(train_groups).isdisjoint(holdout_groups)
    assert {factor.observation_group for factor in train} == set(train_groups)
    assert {factor.observation_group for factor in holdout} == set(holdout_groups)
    assert block.dimension == 1


def test_train_only_prior_never_leaks_into_holdout() -> None:
    factors = [
        JointResidualBlock(
            f"measurement-{capture}",
            capture,
            ("x",),
            lambda values: (values["x"][0],),
        )
        for capture in ("a", "b", "c", "d")
    ]
    factors.append(
        JointResidualBlock(
            "pose-prior",
            "prior",
            ("x",),
            lambda values: (values["x"][0],),
            family="diagonal_prior",
            split_policy="train_only",
        )
    )

    train, holdout, _train_groups, _holdout_groups = split_joint_factors(
        factors, 0.25, 3
    )

    assert "pose-prior" in {factor.factor_id for factor in train}
    assert "pose-prior" not in {factor.factor_id for factor in holdout}


def test_joint_optimizer_reports_rank_deficient_parameter_blocks() -> None:
    blocks = [
        JointParameterBlock("extrinsic", (0.0, 0.0)),
        JointParameterBlock("time", (0.0,)),
    ]
    factors = [
        JointResidualBlock(
            f"factor-{index}",
            f"capture-{index}",
            ("extrinsic", "time"),
            lambda values, target=float(index): (
                values["extrinsic"][0] + values["extrinsic"][1] - target,
            ),
        )
        for index in range(10)
    ]

    result = BackendNeutralJointOptimizer().solve(
        blocks,
        factors,
        JointOptimizerOptions(holdout_ratio=0.2, split_seed=2),
    )

    assert result.status == "degenerate"
    assert result.information_rank == 1
    assert "extrinsic" in result.weak_parameter_blocks
    assert "time" in result.weak_parameter_blocks


def test_huber_acceptance_uses_robust_objective_not_raw_rmse() -> None:
    targets = [0.0] * 20 + [100.0]
    least_squares_mean = sum(targets) / len(targets)
    factors = [
        JointResidualBlock(
            f"sample-{index}",
            f"capture-{index}",
            ("x",),
            lambda values, target=target: (values["x"][0] - target,),
        )
        for index, target in enumerate(targets)
    ]

    result = BackendNeutralJointOptimizer().solve(
        [JointParameterBlock("x", (least_squares_mean,))],
        factors,
        JointOptimizerOptions(holdout_ratio=0.0, huber_delta=0.1),
    )

    assert result.status == "converged"
    assert result.optimized_values["x"][0] == pytest.approx(0.005, abs=1e-4)
    assert result.history[0].accepted is True
    assert result.history[0].train_rmse > math.sqrt(
        sum((least_squares_mean - target) ** 2 for target in targets) / len(targets)
    )


def test_joint_optimizer_rejects_duplicate_parameter_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        BackendNeutralJointOptimizer().solve(
            [JointParameterBlock("x", (0.0,)), JointParameterBlock("x", (1.0,))],
            [],
        )


def test_factor_callback_receives_only_declared_parameter_blocks() -> None:
    seen: list[set[str]] = []

    def evaluator(values: dict[str, tuple[float, ...]]) -> tuple[float]:
        seen.append(set(values))
        return (values["x"][0],)

    factor = JointResidualBlock("factor", "capture", ("x",), evaluator)
    assert factor.residuals({"x": (0.0,), "secret": (9.0,)}) == (0.0,)
    assert seen == [{"x"}]


def test_numeric_linearization_evaluates_only_factors_owned_by_each_block() -> None:
    calls = {"x": 0, "y": 0}

    def x_factor(values: dict[str, tuple[float, ...]]) -> tuple[float]:
        calls["x"] += 1
        return (values["x"][0] + values["x"][1] - 1.0,)

    def y_factor(values: dict[str, tuple[float, ...]]) -> tuple[float]:
        calls["y"] += 1
        return (values["y"][0] - 1.0,)

    BackendNeutralJointOptimizer().solve(
        [JointParameterBlock("x", (0.0, 0.0)), JointParameterBlock("y", (0.0,))],
        [
            JointResidualBlock("x-factor", "x-group", ("x",), x_factor),
            JointResidualBlock("y-factor", "y-group", ("y",), y_factor),
        ],
        JointOptimizerOptions(
            max_iterations=1,
            holdout_ratio=0.0,
            minimum_train_factors=1,
        ),
    )

    assert calls["x"] - calls["y"] == 4


def test_explicit_observability_evaluation_excludes_fixed_blocks() -> None:
    blocks = [
        JointParameterBlock("fixed_pose", (2.0,), fixed=True),
        JointParameterBlock("shared", (0.0, 0.0)),
    ]
    factors = [
        JointResidualBlock(
            "x",
            "capture",
            ("fixed_pose", "shared"),
            lambda values: (values["shared"][0] + values["fixed_pose"][0],),
        ),
        JointResidualBlock(
            "y",
            "capture",
            ("shared",),
            lambda values: (values["shared"][1],),
        ),
    ]

    evaluation = evaluate_joint_observability(
        blocks,
        factors,
        {"fixed_pose": (2.0,), "shared": (0.0, 0.0)},
    )

    assert evaluation.parameter_dimension == 2
    assert evaluation.residual_dimension == 2
    assert evaluation.information_rank == 2
    assert evaluation.condition_number == pytest.approx(1.0)
    assert evaluation.weak_parameter_blocks == ()
