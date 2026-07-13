import math
from dataclasses import replace

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
    assert result.as_dict()["method"] == "backend_neutral_robust_joint_lm/v0.5"
    assert result.information_rank_threshold is not None
    assert result.weak_parameter_blocks == ()
    assert len(result.probes) == 10
    assert all(probe.detectable is True for probe in result.probes)
    assert set(result.train_observation_groups).isdisjoint(result.holdout_observation_groups)
    assert len(result.train_factor_family_diagnostics) == 1
    assert result.train_factor_family_diagnostics[0].family == "synthetic_multisensor"
    assert result.train_factor_family_diagnostics[0].robust_information_fraction == pytest.approx(
        1.0
    )
    assert len(result.holdout_factor_family_diagnostics) == 1
    assert result.holdout_factor_family_diagnostics[0].residual_rmse == pytest.approx(
        result.holdout_rmse
    )
    assert result.holdout_factor_family_diagnostics[0].robust_information_fraction is None


def test_schur_solver_matches_dense_joint_step_and_truth() -> None:
    blocks, factors = _joint_problem()
    schur_blocks = [
        replace(block, schur_role="eliminated") if block.name == "trajectory" else block
        for block in blocks
    ]
    options = JointOptimizerOptions(
        holdout_ratio=0.25,
        split_seed=17,
        known_bad_margin=1.0e-5,
    )

    dense = BackendNeutralJointOptimizer().solve(blocks, factors, options)
    dense_with_roles = BackendNeutralJointOptimizer().solve(schur_blocks, factors, options)
    schur = BackendNeutralJointOptimizer().solve(
        schur_blocks,
        factors,
        replace(options, linear_solver="schur"),
    )

    assert dense.status == schur.status == "converged"
    assert dense_with_roles.schur_eliminated_dimension == 0
    assert dense_with_roles.schur_retained_dimension == 5
    assert set(schur.optimized_values) == set(dense.optimized_values)
    for name, values in dense.optimized_values.items():
        assert schur.optimized_values[name] == pytest.approx(values, abs=1.0e-10)
    assert schur.train_factor_ids == dense.train_factor_ids
    assert schur.holdout_factor_ids == dense.holdout_factor_ids
    assert schur.train_rmse == pytest.approx(dense.train_rmse, abs=1.0e-12)
    assert schur.holdout_rmse == pytest.approx(dense.holdout_rmse, abs=1.0e-12)
    assert schur.linear_solver == "schur"
    assert schur.schur_eliminated_blocks == ("trajectory",)
    assert schur.schur_retained_blocks == ("T_body_sensor", "dt_sensor")
    assert schur.schur_eliminated_dimension == 2
    assert schur.schur_retained_dimension == 3
    assert schur.max_linear_system_residual_inf is not None
    assert schur.max_linear_system_residual_inf < 1.0e-10
    assert all(item.linear_solver == "schur" for item in schur.history)
    assert all(item.eliminated_dimension == 2 for item in schur.history)
    assert all(item.retained_dimension == 3 for item in schur.history)
    document = schur.as_dict()
    assert document["linear_solver_papers"][0]["doi"] == "10.1007/3-540-44480-7_21"
    assert document["linear_solver_papers"][1]["doi"] == "10.1145/1486525.1486527"


@pytest.mark.parametrize(
    "blocks",
    [
        [JointParameterBlock("retained", (0.0,))],
        [JointParameterBlock("eliminated", (0.0,), schur_role="eliminated")],
    ],
)
def test_schur_solver_requires_both_free_partitions(
    blocks: list[JointParameterBlock],
) -> None:
    factor = JointResidualBlock(
        "factor",
        "capture",
        (blocks[0].name,),
        lambda values: (values[blocks[0].name][0],),
    )

    with pytest.raises(ValueError, match="requires free eliminated and retained"):
        BackendNeutralJointOptimizer().solve(
            blocks,
            [factor],
            JointOptimizerOptions(
                holdout_ratio=0.0,
                minimum_train_factors=1,
                linear_solver="schur",
            ),
        )


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

    train, holdout, _train_groups, _holdout_groups = split_joint_factors(factors, 0.25, 3)

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
    assert evaluation.information_rank_threshold == pytest.approx(1.0e-8)
    assert evaluation.condition_number == pytest.approx(1.0)
    assert evaluation.weak_parameter_blocks == ()


def test_factor_family_diagnostics_report_robust_information_balance() -> None:
    blocks = [
        JointParameterBlock("camera_state", (0.0,)),
        JointParameterBlock("lidar_state", (0.0,)),
    ]
    factors = []
    for index in range(4):
        factors.extend(
            [
                JointResidualBlock(
                    f"camera-{index}",
                    f"capture-{index}",
                    ("camera_state",),
                    lambda values: (10.0 * values["camera_state"][0] + 100.0,),
                    family="camera_outlier",
                ),
                JointResidualBlock(
                    f"lidar-{index}",
                    f"capture-{index}",
                    ("lidar_state",),
                    lambda values: (values["lidar_state"][0],),
                    family="lidar_inlier",
                ),
            ]
        )

    evaluation = evaluate_joint_observability(
        blocks,
        factors,
        {"camera_state": (0.0,), "lidar_state": (0.0,)},
        JointOptimizerOptions(huber_delta=1.0),
    )

    diagnostics = {item.family: item for item in evaluation.factor_family_diagnostics}
    camera = diagnostics["camera_outlier"]
    lidar = diagnostics["lidar_inlier"]
    assert camera.factor_count == lidar.factor_count == 4
    assert camera.observation_group_count == lidar.observation_group_count == 4
    assert camera.residual_dimension == lidar.residual_dimension == 4
    assert camera.residual_rmse == pytest.approx(100.0)
    assert lidar.residual_rmse == pytest.approx(0.0)
    assert camera.mean_huber_weight == pytest.approx(0.01)
    assert lidar.mean_huber_weight == pytest.approx(1.0)
    assert camera.robust_jacobian_frobenius_norm == pytest.approx(2.0)
    assert lidar.robust_jacobian_frobenius_norm == pytest.approx(2.0)
    assert camera.robust_information_fraction == pytest.approx(0.5)
    assert lidar.robust_information_fraction == pytest.approx(0.5)
    serialized = evaluation.as_dict()["factor_family_diagnostics"]
    assert isinstance(serialized, list)
    assert "not covariance" in serialized[0]["diagnostic_kind"]


@pytest.mark.parametrize("residual_scale", [1.0, 1.0e-12, 1.0e9])
def test_joint_observability_rank_is_invariant_to_uniform_residual_scale(
    residual_scale: float,
) -> None:
    blocks = [JointParameterBlock("state", (0.0, 0.0))]
    factors = [
        JointResidualBlock(
            "x",
            "capture",
            ("state",),
            lambda values: (residual_scale * values["state"][0],),
        ),
        JointResidualBlock(
            "y",
            "capture",
            ("state",),
            lambda values: (residual_scale * values["state"][1],),
        ),
    ]

    evaluation = evaluate_joint_observability(blocks, factors, {"state": (0.0, 0.0)})

    assert evaluation.information_rank == 2
    assert evaluation.condition_number == pytest.approx(1.0)
    assert evaluation.information_rank_threshold == pytest.approx(residual_scale * 1.0e-8)
    assert evaluation.as_dict()["rank_tolerance_policy"] == ("relative_to_largest_singular_value")


def test_joint_observability_relative_threshold_rejects_scaled_near_null_direction() -> None:
    residual_scale = 1.0e9
    blocks = [JointParameterBlock("state", (0.0, 0.0))]
    factors = [
        JointResidualBlock(
            "strong",
            "capture",
            ("state",),
            lambda values: (residual_scale * values["state"][0],),
        ),
        JointResidualBlock(
            "near-null",
            "capture",
            ("state",),
            lambda values: (residual_scale * 1.0e-10 * values["state"][1],),
        ),
    ]

    evaluation = evaluate_joint_observability(blocks, factors, {"state": (0.0, 0.0)})

    assert evaluation.information_rank == 1
    assert evaluation.condition_number is None
    assert evaluation.weak_parameter_blocks == ("state",)


def test_joint_observability_attributes_structural_right_nullspace() -> None:
    blocks = [
        JointParameterBlock("extrinsic", (0.0,)),
        JointParameterBlock("time", (0.0,)),
    ]
    factor = JointResidualBlock(
        "coupled",
        "capture",
        ("extrinsic", "time"),
        lambda values: (values["extrinsic"][0] + values["time"][0],),
    )

    evaluation = evaluate_joint_observability(
        blocks,
        [factor],
        {"extrinsic": (0.0,), "time": (0.0,)},
    )

    assert evaluation.residual_dimension == 1
    assert evaluation.parameter_dimension == 2
    assert evaluation.information_rank == 1
    assert evaluation.weak_parameter_blocks == ("extrinsic", "time")


def test_joint_optimizer_rejects_unknown_factor_ownership_and_nonfinite_blocks() -> None:
    with pytest.raises(ValueError, match="unknown parameter"):
        BackendNeutralJointOptimizer().solve(
            [JointParameterBlock("x", (0.0,))],
            [
                JointResidualBlock(
                    "factor",
                    "capture",
                    ("missing",),
                    lambda values: (values["missing"][0],),
                )
            ],
        )

    with pytest.raises(ValueError, match="finite"):
        BackendNeutralJointOptimizer().solve(
            [JointParameterBlock("x", (math.nan,))],
            [],
        )


def test_joint_optimizer_rejects_nonfinite_factor_residuals_and_duplicate_ids() -> None:
    block = JointParameterBlock("x", (0.0,))
    with pytest.raises(ValueError, match="non-empty and finite"):
        BackendNeutralJointOptimizer().solve(
            [block],
            [
                JointResidualBlock(
                    "non-finite",
                    "capture",
                    ("x",),
                    lambda _values: (math.nan,),
                )
            ],
            JointOptimizerOptions(holdout_ratio=0.0, minimum_train_factors=1),
        )

    duplicate = JointResidualBlock(
        "duplicate",
        "capture",
        ("x",),
        lambda values: (values["x"][0],),
    )
    with pytest.raises(ValueError, match="factor IDs"):
        BackendNeutralJointOptimizer().solve([block], [duplicate, duplicate])


def test_joint_optimizer_rejects_invalid_options() -> None:
    with pytest.raises(ValueError, match="rank tolerance"):
        BackendNeutralJointOptimizer().solve(
            [JointParameterBlock("x", (0.0,))],
            [],
            JointOptimizerOptions(rank_tolerance=1.0),
        )


def test_joint_factor_applies_correlated_whitening_before_scalar_weight() -> None:
    factor = JointResidualBlock(
        "correlated",
        "capture",
        ("state",),
        lambda values: values["state"],
        weight=4.0,
        sqrt_information=((2.0, 1.0), (0.0, 3.0)),
    )

    assert factor.residuals({"state": (2.0, -1.0)}) == pytest.approx((6.0, -6.0))


def test_joint_optimizer_recovers_truth_with_correlated_whitening() -> None:
    truth = (0.3, -0.2)
    factors = [
        JointResidualBlock(
            f"correlated-{index}",
            f"capture-{index}",
            ("state",),
            lambda values, offset=0.01 * index: (
                values["state"][0] - truth[0] + offset,
                values["state"][1] - truth[1] - offset,
            ),
            sqrt_information=((2.0, 0.5), (0.0, 1.5)),
        )
        for index in range(6)
    ]

    result = BackendNeutralJointOptimizer().solve(
        [JointParameterBlock("state", (0.0, 0.0))],
        factors,
        JointOptimizerOptions(holdout_ratio=0.0, minimum_train_factors=1),
    )

    assert result.status == "converged"
    assert result.optimized_values["state"] == pytest.approx((0.275, -0.175), abs=1.0e-8)
    assert result.train_whitened_factor_count == 6
    assert result.as_dict()["factor_weighting_policy"] == (
        "sqrt(weight) * sqrt_information * raw_residual"
    )
    assert len(result.train_factor_whitening) == 6
    assert result.train_factor_whitening[0].sqrt_information == (
        (2.0, 0.5),
        (0.0, 1.5),
    )
    serialized = result.as_dict()["train_factor_whitening"]
    assert serialized[0]["factor_id"] == "correlated-0"
    assert serialized[0]["sqrt_information"] == [[2.0, 0.5], [0.0, 1.5]]


@pytest.mark.parametrize(
    "matrix",
    [((1.0, 0.0),), ((1.0, math.nan), (0.0, 1.0))],
)
def test_joint_optimizer_rejects_invalid_square_root_information(
    matrix: tuple[tuple[float, ...], ...],
) -> None:
    factor = JointResidualBlock(
        "invalid-information",
        "capture",
        ("state",),
        lambda values: values["state"],
        sqrt_information=matrix,
    )

    with pytest.raises(ValueError, match="square-root information"):
        BackendNeutralJointOptimizer().solve(
            [JointParameterBlock("state", (0.0, 0.0))],
            [factor],
            JointOptimizerOptions(holdout_ratio=0.0, minimum_train_factors=1),
        )
