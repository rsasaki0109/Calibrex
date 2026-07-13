import math

import numpy as np
import pytest

from calibrex.core.geometry import SE3
from calibrex.solvers.shah_robot_world_hand_eye_solver import RobotWorldHandEyePosePair
from calibrex.solvers.zhuang_roth_sudhakar_robot_world_hand_eye_solver import (
    ZhuangRothSudhakarOptions,
    ZhuangRothSudhakarSolver,
)


def _pose(
    axis: tuple[float, float, float] | np.ndarray,
    angle_deg: float,
    translation: tuple[float, float, float] | np.ndarray,
) -> SE3:
    vector = np.asarray(axis, dtype=np.float64)
    vector /= np.linalg.norm(vector)
    half = math.radians(angle_deg) / 2.0
    return SE3(
        tuple(float(value) for value in translation),
        (
            float(vector[0] * math.sin(half)),
            float(vector[1] * math.sin(half)),
            float(vector[2] * math.sin(half)),
            math.cos(half),
        ),
    )


def _problem(
    *, z_angle_deg: float = -31.0
) -> tuple[SE3, SE3, list[RobotWorldHandEyePosePair]]:
    transform_x = _pose((1.0, -2.0, 0.5), 23.0, (0.24, -0.16, 0.11))
    transform_z = _pose((-0.5, 1.0, 2.0), z_angle_deg, (-0.3, 0.5, 0.2))
    rng = np.random.default_rng(771)
    pairs = []
    for index in range(40):
        pose_a = _pose(
            rng.normal(size=3),
            float(rng.uniform(-170.0, 170.0)),
            rng.uniform(-0.5, 0.5, size=3),
        )
        pose_b = transform_z.inverse().compose(pose_a).compose(transform_x)
        pairs.append(RobotWorldHandEyePosePair(f"pose-{index:03d}", pose_a, pose_b))
    return transform_x, transform_z, pairs


def _rotation_error_deg(left: SE3, right: SE3) -> float:
    delta = left.inverse().compose(right)
    return math.degrees(
        2.0 * math.acos(min(1.0, max(-1.0, abs(delta.rotation_quat_xyzw[3]))))
    )


def test_zhuang_recovers_truth_holdout_and_controls() -> None:
    truth_x, truth_z, pairs = _problem()

    result = ZhuangRothSudhakarSolver().solve(pairs)

    assert result.status == "converged"
    assert result.transform_x is not None
    assert result.transform_z is not None
    assert result.transform_y == result.transform_z
    assert _rotation_error_deg(result.transform_x, truth_x) < 1.0e-5
    assert _rotation_error_deg(result.transform_z, truth_z) < 1.0e-5
    assert np.linalg.norm(
        np.asarray(result.transform_x.translation_m) - np.asarray(truth_x.translation_m)
    ) < 1.0e-9
    assert np.linalg.norm(
        np.asarray(result.transform_z.translation_m) - np.asarray(truth_z.translation_m)
    ) < 1.0e-9
    assert len(result.rotation_singular_values) == 6
    assert result.rotation_rank == 6
    assert result.rotation_condition_number is not None
    assert result.minimum_abs_a_scalar is not None
    assert result.recovered_abs_z_scalar is not None
    assert result.quaternion_normalization_disagreement is not None
    assert result.quaternion_normalization_disagreement < 1.0e-12
    assert result.scalar_reconstruction_rmse is not None
    assert result.scalar_reconstruction_rmse < 1.0e-12
    assert result.quaternion_sign_synchronization_fraction == 1.0
    assert result.translation_rank == 6
    assert len(result.translation_singular_values) == 6
    assert result.holdout_evaluation.rotation_closure_rmse_deg is not None
    assert result.holdout_evaluation.rotation_closure_rmse_deg < 1.0e-5
    assert result.holdout_evaluation.translation_closure_rmse_m is not None
    assert result.holdout_evaluation.translation_closure_rmse_m < 1.0e-9
    assert len(result.probes) == 24
    assert all(probe.detectable is True for probe in result.probes)
    document = result.as_dict()
    assert document["paper"]["doi"] == "10.1109/70.313105"
    assert document["external_code_executed"] is False


def test_zhuang_is_order_invariant_and_honors_weights() -> None:
    _truth_x, _truth_z, pairs = _problem()
    weighted = [
        RobotWorldHandEyePosePair(pair.pair_id, pair.pose_a, pair.pose_b, index + 1.0)
        for index, pair in enumerate(pairs)
    ]
    solver = ZhuangRothSudhakarSolver()

    first = solver.solve(weighted)
    second = solver.solve(list(reversed(weighted)))

    assert first.status == second.status == "converged"
    assert first.transform_x == second.transform_x
    assert first.transform_z == second.transform_z
    assert first.train_pair_ids == second.train_pair_ids
    assert first.holdout_pair_ids == second.holdout_pair_ids


def test_zhuang_reports_paper_a0_special_configuration() -> None:
    half_turn = _pose((1.0, 0.0, 0.0), 180.0, (0.1, 0.2, 0.3))
    pairs = [
        RobotWorldHandEyePosePair(f"half-turn-{index}", half_turn, half_turn)
        for index in range(3)
    ]

    result = ZhuangRothSudhakarSolver().solve(
        pairs, ZhuangRothSudhakarOptions(holdout_ratio=0.0)
    )

    assert result.status == "special_a_scalar_zero"
    assert result.minimum_abs_a_scalar is not None
    assert result.minimum_abs_a_scalar < 1.0e-8
    assert result.transform_x is None
    assert result.transform_z is None


def test_zhuang_reports_paper_z0_special_configuration() -> None:
    _truth_x, _truth_z, pairs = _problem(z_angle_deg=179.999999)

    result = ZhuangRothSudhakarSolver().solve(
        pairs,
        ZhuangRothSudhakarOptions(
            holdout_ratio=0.0,
            rank_tolerance=1.0e-18,
            max_rotation_condition_number=1.0e30,
        ),
    )

    assert result.status == "special_z_scalar_zero"
    assert result.recovered_abs_z_scalar is not None
    assert result.recovered_abs_z_scalar < 1.0e-8
    assert result.transform_x is None
    assert result.transform_z is None


def test_zhuang_reports_rank_degeneracy_and_validates_options() -> None:
    pose = _pose((0.0, 0.0, 1.0), 20.0, (0.1, 0.2, 0.3))
    pairs = [RobotWorldHandEyePosePair(f"duplicate-{index}", pose, pose) for index in range(6)]

    result = ZhuangRothSudhakarSolver().solve(
        pairs, ZhuangRothSudhakarOptions(holdout_ratio=0.0)
    )

    assert result.status == "degenerate_rotation"
    assert result.rotation_rank < 6
    with pytest.raises(ValueError, match="at least three"):
        ZhuangRothSudhakarSolver().solve(
            pairs, ZhuangRothSudhakarOptions(min_train_poses=2)
        )
    with pytest.raises(ValueError, match="unique"):
        ZhuangRothSudhakarSolver().solve([pairs[0], pairs[0], pairs[1]])


def test_zhuang_requires_three_train_absolute_poses() -> None:
    _truth_x, _truth_z, pairs = _problem()

    result = ZhuangRothSudhakarSolver().solve(pairs[:2])

    assert result.status == "insufficient_poses"
