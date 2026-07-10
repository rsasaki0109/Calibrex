import math

import numpy as np

from calibrex.core.geometry import SE3
from calibrex.solvers.park_martin_hand_eye_solver import (
    HandEyeMotionPair,
    ParkMartinHandEyeOptions,
    ParkMartinHandEyeSolver,
)


def _axis_transform(
    axis: tuple[float, float, float], angle_deg: float, translation: tuple[float, float, float]
) -> SE3:
    angle = math.radians(angle_deg)
    sine = math.sin(angle / 2.0)
    return SE3(
        translation,
        (axis[0] * sine, axis[1] * sine, axis[2] * sine, math.cos(angle / 2.0)),
    )


def _motions(truth: SE3) -> list[HandEyeMotionPair]:
    definitions = (
        ((1.0, 0.0, 0.0), 18.0, (0.2, -0.1, 0.05)),
        ((0.0, 1.0, 0.0), -25.0, (-0.1, 0.15, 0.03)),
        ((0.0, 0.0, 1.0), 32.0, (0.04, 0.08, -0.12)),
        ((1.0, 0.0, 0.0), -40.0, (-0.05, 0.02, 0.1)),
        ((0.0, 1.0, 0.0), 48.0, (0.12, -0.04, 0.07)),
        ((0.0, 0.0, 1.0), -21.0, (-0.09, 0.03, 0.02)),
        ((1.0, 0.0, 0.0), 55.0, (0.03, 0.11, -0.06)),
        ((0.0, 1.0, 0.0), 37.0, (-0.02, -0.07, 0.09)),
    )
    result: list[HandEyeMotionPair] = []
    for index, (axis, angle, translation) in enumerate(definitions):
        motion_a = _axis_transform(axis, angle, translation)
        motion_b = truth.inverse().compose(motion_a).compose(truth)
        result.append(HandEyeMotionPair(f"motion-{index:03d}", motion_a, motion_b))
    return result


def _rotation_error_deg(left: SE3, right: SE3) -> float:
    dot = abs(
        sum(
            a * b
            for a, b in zip(
                left.rotation_quat_xyzw,
                right.rotation_quat_xyzw,
                strict=True,
            )
        )
    )
    return math.degrees(2.0 * math.acos(min(1.0, dot)))


def test_recovers_exact_transform_and_holdout_closure() -> None:
    truth = SE3((0.28, -0.14, 0.09), (0.08, -0.11, 0.16, 0.978))
    result = ParkMartinHandEyeSolver().solve(_motions(truth))

    assert result.status == "converged"
    assert result.transform_x is not None
    assert _rotation_error_deg(result.transform_x, truth) < 1.0e-5
    assert (
        np.linalg.norm(
            np.asarray(result.transform_x.translation_m) - np.asarray(truth.translation_m)
        )
        < 1.0e-9
    )
    assert result.holdout_evaluation.rotation_closure_rmse_deg is not None
    assert result.holdout_evaluation.rotation_closure_rmse_deg < 1.0e-5
    assert result.holdout_evaluation.translation_closure_rmse_m is not None
    assert result.holdout_evaluation.translation_closure_rmse_m < 1.0e-9


def test_parallel_rotation_axes_are_rejected() -> None:
    truth = SE3((0.2, 0.1, -0.1), (0.02, 0.04, -0.06, 0.997))
    motions: list[HandEyeMotionPair] = []
    for index, angle in enumerate((10.0, 20.0, 30.0, 40.0, 50.0)):
        motion_a = _axis_transform((0.0, 0.0, 1.0), angle, (0.1, 0.0, 0.0))
        motion_b = truth.inverse().compose(motion_a).compose(truth)
        motions.append(HandEyeMotionPair(f"parallel-{index}", motion_a, motion_b))
    result = ParkMartinHandEyeSolver().solve(
        motions, options=ParkMartinHandEyeOptions(holdout_ratio=0.0)
    )

    assert result.status == "degenerate_rotation"
    assert result.rotation_axis_rank == 1


def test_pure_translation_has_insufficient_rotation_excitation() -> None:
    motions = [
        HandEyeMotionPair(
            f"translation-{index}",
            SE3((float(index), 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            SE3((float(index), 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        )
        for index in range(5)
    ]
    result = ParkMartinHandEyeSolver().solve(motions)

    assert result.status == "insufficient_motions"


def test_input_order_does_not_change_split_or_solution() -> None:
    truth = SE3((0.1, 0.05, -0.08), (0.03, -0.02, 0.04, 0.9985))
    motions = _motions(truth)
    solver = ParkMartinHandEyeSolver()
    first = solver.solve(motions)
    second = solver.solve(list(reversed(motions)))

    assert first.status == second.status == "converged"
    assert first.transform_x == second.transform_x
    assert first.train_pair_ids == second.train_pair_ids
    assert first.holdout_pair_ids == second.holdout_pair_ids
