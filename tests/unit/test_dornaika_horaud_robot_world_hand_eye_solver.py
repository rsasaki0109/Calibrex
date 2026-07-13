import math

import numpy as np

from calibrex.core.geometry import SE3
from calibrex.solvers.dornaika_horaud_robot_world_hand_eye_solver import (
    DornaikaHoraudRobotWorldHandEyeOptions,
    DornaikaHoraudRobotWorldHandEyeSolver,
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
    rng = np.random.default_rng(771)
    pairs = []
    for index in range(40):
        axis = rng.normal(size=3)
        angle = float(rng.uniform(-170.0, 170.0))
        translation = tuple(float(value) for value in rng.uniform(-0.5, 0.5, size=3))
        pose_a = _pose(tuple(float(value) for value in axis), angle, translation)
        pose_b = transform_z.inverse().compose(pose_a).compose(transform_x)
        pairs.append(RobotWorldHandEyePosePair(f"pose-{index:03d}", pose_a, pose_b))
    return transform_x, transform_z, pairs


def _rotation_error_deg(left: SE3, right: SE3) -> float:
    delta = left.inverse().compose(right)
    return math.degrees(2.0 * math.acos(min(1.0, max(-1.0, abs(delta.rotation_quat_xyzw[3])))))


def test_dornaika_horaud_recovers_truth_holdout_and_controls() -> None:
    truth_x, truth_z, pairs = _problem()

    result = DornaikaHoraudRobotWorldHandEyeSolver().solve(pairs)

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
    assert len(result.rotation_singular_values) == 4
    assert result.rotation_dominant_multiplicity == 1
    assert result.rotation_normalized_gap is not None
    assert result.rotation_normalized_gap > 0.01
    assert result.rotation_objective is not None
    assert result.rotation_objective < 1.0e-20
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
    assert document["paper"]["doi"] == "10.1109/70.704233"
    assert document["nonlinear_method_executed"] is False


def test_dornaika_horaud_is_order_invariant() -> None:
    _truth_x, _truth_z, pairs = _problem()
    solver = DornaikaHoraudRobotWorldHandEyeSolver()

    first = solver.solve(pairs)
    second = solver.solve(list(reversed(pairs)))

    assert first.status == second.status == "converged"
    assert first.transform_x == second.transform_x
    assert first.transform_z == second.transform_z
    assert first.train_pair_ids == second.train_pair_ids
    assert first.holdout_pair_ids == second.holdout_pair_ids


def test_dornaika_horaud_rejects_repeated_pose_rotation_degeneracy() -> None:
    pose = _pose((0.0, 0.0, 1.0), 20.0, (0.1, 0.2, 0.3))
    pairs = [RobotWorldHandEyePosePair(f"duplicate-{index}", pose, pose) for index in range(6)]

    result = DornaikaHoraudRobotWorldHandEyeSolver().solve(
        pairs, DornaikaHoraudRobotWorldHandEyeOptions(holdout_ratio=0.0)
    )

    assert result.status == "degenerate_rotation"
    assert result.rotation_dominant_multiplicity == 4
    assert len(result.rotation_singular_values) == 4
    assert result.transform_x is None
    assert result.transform_z is None


def test_dornaika_horaud_requires_three_train_absolute_poses() -> None:
    _truth_x, _truth_z, pairs = _problem()

    result = DornaikaHoraudRobotWorldHandEyeSolver().solve(pairs[:2])

    assert result.status == "insufficient_poses"
