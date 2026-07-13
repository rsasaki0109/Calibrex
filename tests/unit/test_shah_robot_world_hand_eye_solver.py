import math

import numpy as np

from calibrex.core.geometry import SE3
from calibrex.solvers.shah_robot_world_hand_eye_solver import (
    RobotWorldHandEyePosePair,
    ShahRobotWorldHandEyeOptions,
    ShahRobotWorldHandEyeSolver,
)


def _pose(
    axis: tuple[float, float, float],
    angle_deg: float,
    translation: tuple[float, float, float],
) -> SE3:
    norm = math.sqrt(sum(value * value for value in axis))
    unit = tuple(value / norm for value in axis)
    half = math.radians(angle_deg) / 2.0
    return SE3(
        translation,
        (
            unit[0] * math.sin(half),
            unit[1] * math.sin(half),
            unit[2] * math.sin(half),
            math.cos(half),
        ),
    )


def _problem() -> tuple[SE3, SE3, list[RobotWorldHandEyePosePair]]:
    transform_x = _pose((1.0, -2.0, 0.5), 23.0, (0.24, -0.16, 0.11))
    transform_y = _pose((-0.5, 1.0, 2.0), -31.0, (-0.3, 0.5, 0.2))
    definitions = (
        ((1.0, 0.0, 0.0), 18.0, (0.2, -0.1, 0.05)),
        ((0.0, 1.0, 0.0), -25.0, (-0.1, 0.15, 0.03)),
        ((0.0, 0.0, 1.0), 32.0, (0.04, 0.08, -0.12)),
        ((1.0, 1.0, 0.0), -40.0, (-0.05, 0.02, 0.1)),
        ((0.0, 1.0, 1.0), 48.0, (0.12, -0.04, 0.07)),
        ((1.0, 0.0, 1.0), -21.0, (-0.09, 0.03, 0.02)),
        ((1.0, -1.0, 0.5), 55.0, (0.03, 0.11, -0.06)),
        ((-0.5, 1.0, 1.0), 37.0, (-0.02, -0.07, 0.09)),
        ((1.0, 0.2, -1.0), -29.0, (0.15, 0.01, -0.04)),
        ((0.3, -1.0, 0.2), 42.0, (-0.11, 0.09, 0.08)),
        ((-1.0, 0.4, 0.7), 34.0, (0.07, -0.13, 0.01)),
        ((0.2, 0.5, 1.0), -46.0, (-0.06, 0.04, 0.14)),
    )
    pairs: list[RobotWorldHandEyePosePair] = []
    for index, (axis, angle, translation) in enumerate(definitions):
        pose_a = _pose(axis, angle, translation)
        pose_b = transform_y.inverse().compose(pose_a).compose(transform_x)
        pairs.append(RobotWorldHandEyePosePair(f"pose-{index:03d}", pose_a, pose_b))
    return transform_x, transform_y, pairs


def _rotation_error_deg(left: SE3, right: SE3) -> float:
    delta = left.inverse().compose(right)
    return math.degrees(
        2.0 * math.acos(min(1.0, max(-1.0, abs(delta.rotation_quat_xyzw[3]))))
    )


def test_shah_solver_recovers_both_transforms_with_holdout_and_probes() -> None:
    truth_x, truth_y, pairs = _problem()

    result = ShahRobotWorldHandEyeSolver().solve(pairs)

    assert result.status == "converged"
    assert result.transform_x is not None
    assert result.transform_y is not None
    assert _rotation_error_deg(result.transform_x, truth_x) < 1.0e-5
    assert _rotation_error_deg(result.transform_y, truth_y) < 1.0e-5
    assert np.linalg.norm(
        np.asarray(result.transform_x.translation_m) - np.asarray(truth_x.translation_m)
    ) < 1.0e-9
    assert np.linalg.norm(
        np.asarray(result.transform_y.translation_m) - np.asarray(truth_y.translation_m)
    ) < 1.0e-9
    assert result.rotation_dominant_multiplicity == 1
    assert result.rotation_normalized_gap is not None
    assert result.rotation_normalized_gap > 0.01
    assert result.translation_rank == 6
    assert result.holdout_evaluation.rotation_closure_rmse_deg is not None
    assert result.holdout_evaluation.rotation_closure_rmse_deg < 1.0e-5
    assert result.holdout_evaluation.translation_closure_rmse_m is not None
    assert result.holdout_evaluation.translation_closure_rmse_m < 1.0e-9
    assert len(result.probes) == 24
    assert all(probe.detectable is True for probe in result.probes)
    assert set(result.train_pair_ids).isdisjoint(result.holdout_pair_ids)
    document = result.as_dict()
    assert document["paper"]["doi"] == "10.1115/1.4024473"
    assert document["equation"] == "A_j X = Y B_j"


def test_shah_solver_is_order_invariant() -> None:
    _truth_x, _truth_y, pairs = _problem()
    solver = ShahRobotWorldHandEyeSolver()

    first = solver.solve(pairs)
    second = solver.solve(list(reversed(pairs)))

    assert first.status == second.status == "converged"
    assert first.transform_x == second.transform_x
    assert first.transform_y == second.transform_y
    assert first.train_pair_ids == second.train_pair_ids
    assert first.holdout_pair_ids == second.holdout_pair_ids


def test_shah_solver_rejects_nonunique_rotation_family() -> None:
    pose = _pose((0.0, 0.0, 1.0), 20.0, (0.1, 0.2, 0.3))
    pairs = [
        RobotWorldHandEyePosePair(f"duplicate-{index}", pose, pose)
        for index in range(6)
    ]

    result = ShahRobotWorldHandEyeSolver().solve(
        pairs, ShahRobotWorldHandEyeOptions(holdout_ratio=0.0)
    )

    assert result.status == "degenerate_rotation"
    assert result.transform_x is None
    assert result.transform_y is None
    assert len(result.rotation_singular_values) == 9
    assert result.rotation_normalized_gap is not None
    assert result.rotation_dominant_multiplicity > 1


def test_shah_solver_requires_three_train_absolute_poses() -> None:
    _truth_x, _truth_y, pairs = _problem()

    result = ShahRobotWorldHandEyeSolver().solve(pairs[:2])

    assert result.status == "insufficient_poses"
