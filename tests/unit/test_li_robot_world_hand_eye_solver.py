import math

import numpy as np

from calibrex.core.geometry import SE3
from calibrex.solvers.li_robot_world_hand_eye_solver import (
    LiRobotWorldHandEyeOptions,
    LiRobotWorldHandEyeSolver,
)
from calibrex.solvers.shah_robot_world_hand_eye_solver import RobotWorldHandEyePosePair


def _pose(
    axis: tuple[float, float, float],
    angle_deg: float,
    translation: tuple[float, float, float],
) -> SE3:
    vector = np.asarray(axis, dtype=np.float64)
    vector /= np.linalg.norm(vector)
    half = math.radians(angle_deg) / 2.0
    return SE3(
        translation,
        (
            float(vector[0] * math.sin(half)),
            float(vector[1] * math.sin(half)),
            float(vector[2] * math.sin(half)),
            math.cos(half),
        ),
    )


def _problem() -> tuple[SE3, SE3, list[RobotWorldHandEyePosePair]]:
    transform_x = _pose((1.0, -2.0, 0.5), 23.0, (0.24, -0.16, 0.11))
    transform_z = _pose((-0.5, 1.0, 2.0), -31.0, (-0.3, 0.5, 0.2))
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
    pairs = []
    for index, (axis, angle, translation) in enumerate(definitions):
        pose_a = _pose(axis, angle, translation)
        pose_b = transform_z.inverse().compose(pose_a).compose(transform_x)
        pairs.append(RobotWorldHandEyePosePair(f"pose-{index:03d}", pose_a, pose_b))
    return transform_x, transform_z, pairs


def _rotation_error_deg(left: SE3, right: SE3) -> float:
    delta = left.inverse().compose(right)
    return math.degrees(2.0 * math.acos(min(1.0, max(-1.0, abs(delta.rotation_quat_xyzw[3])))))


def test_li_simultaneous_solver_recovers_truth_holdout_and_controls() -> None:
    truth_x, truth_z, pairs = _problem()

    result = LiRobotWorldHandEyeSolver().solve(pairs)

    assert result.status == "converged"
    assert result.transform_x is not None
    assert result.transform_z is not None
    assert result.transform_y == result.transform_z
    assert _rotation_error_deg(result.transform_x, truth_x) < 1.0e-5
    assert _rotation_error_deg(result.transform_z, truth_z) < 1.0e-5
    assert (
        np.linalg.norm(
            np.asarray(result.transform_x.translation_m) - np.asarray(truth_x.translation_m)
        )
        < 1.0e-9
    )
    assert (
        np.linalg.norm(
            np.asarray(result.transform_z.translation_m) - np.asarray(truth_z.translation_m)
        )
        < 1.0e-9
    )
    assert result.linear_rank == 24
    assert len(result.linear_singular_values) == 24
    assert result.raw_linear_residual_rmse is not None
    assert result.raw_linear_residual_rmse < 1.0e-12
    assert result.rotation_x_projection_correction_frobenius is not None
    assert result.rotation_x_projection_correction_frobenius < 1.0e-12
    assert result.holdout_evaluation.rotation_closure_rmse_deg is not None
    assert result.holdout_evaluation.rotation_closure_rmse_deg < 1.0e-5
    assert result.holdout_evaluation.translation_closure_rmse_m is not None
    assert result.holdout_evaluation.translation_closure_rmse_m < 1.0e-9
    assert len(result.probes) == 24
    assert all(probe.detectable is True for probe in result.probes)
    assert set(result.train_pair_ids).isdisjoint(result.holdout_pair_ids)
    document = result.as_dict()
    assert document["paper"]["doi"] == "10.5897/IJPS.9000501"
    assert document["translation_recomputed_after_rotation_projection"] is False


def test_li_simultaneous_solver_is_order_invariant() -> None:
    _truth_x, _truth_z, pairs = _problem()
    solver = LiRobotWorldHandEyeSolver()

    first = solver.solve(pairs)
    second = solver.solve(list(reversed(pairs)))

    assert first.status == second.status == "converged"
    assert first.transform_x == second.transform_x
    assert first.transform_z == second.transform_z
    assert first.train_pair_ids == second.train_pair_ids
    assert first.holdout_pair_ids == second.holdout_pair_ids


def test_li_simultaneous_solver_rejects_rank_deficient_repeated_poses() -> None:
    pose = _pose((0.0, 0.0, 1.0), 20.0, (0.1, 0.2, 0.3))
    pairs = [RobotWorldHandEyePosePair(f"duplicate-{index}", pose, pose) for index in range(6)]

    result = LiRobotWorldHandEyeSolver().solve(pairs, LiRobotWorldHandEyeOptions(holdout_ratio=0.0))

    assert result.status == "degenerate_linear_system"
    assert result.linear_rank < 24
    assert len(result.linear_singular_values) == 24
    assert result.transform_x is None
    assert result.transform_z is None


def test_li_simultaneous_solver_requires_three_train_absolute_poses() -> None:
    _truth_x, _truth_z, pairs = _problem()

    result = LiRobotWorldHandEyeSolver().solve(pairs[:2])

    assert result.status == "insufficient_poses"
