import math

import pytest

from calibrex.graph.joint_optimization import (
    BackendNeutralJointOptimizer,
    JointOptimizerOptions,
    JointParameterBlock,
    JointResidualBlock,
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
