from __future__ import annotations

import math

import pytest

from calibrex.core.geometry import SE3
from calibrex.evaluation.hand_eye_benchmark import (
    opencv_compatible_hand_eye_motions,
    split_absolute_hand_eye_poses,
)
from calibrex.solvers.park_martin_hand_eye_solver import (
    ParkMartinHandEyeOptions,
    ParkMartinHandEyeSolver,
    evaluate_hand_eye_motions,
)
from calibrex.solvers.shah_robot_world_hand_eye_solver import (
    RobotWorldHandEyePosePair,
)


def _pose(index: int) -> SE3:
    axis = (
        1.0 + (index % 3),
        -0.4 + (index % 5),
        0.7 + (index % 7),
    )
    norm = math.sqrt(sum(value * value for value in axis))
    half = math.radians(7.0 + 3.0 * index) / 2.0
    return SE3(
        (0.02 * index, -0.01 * (index % 4), 0.015 * (index % 6)),
        (
            axis[0] / norm * math.sin(half),
            axis[1] / norm * math.sin(half),
            axis[2] / norm * math.sin(half),
            math.cos(half),
        ),
    )


def _problem(count: int = 18) -> tuple[RobotWorldHandEyePosePair, ...]:
    transform_x = SE3((0.2, -0.1, 0.05), (0.1, -0.2, 0.05, 0.9733961167))
    transform_y = SE3((-0.3, 0.4, 0.1), (-0.15, 0.05, 0.1, 0.9823441352))
    return tuple(
        RobotWorldHandEyePosePair(
            pair_id=f"pose-{index:03d}",
            pose_a=(pose_a := _pose(index)),
            pose_b=transform_y.inverse().compose(pose_a).compose(transform_x),
        )
        for index in range(count)
    )


def test_absolute_split_is_disjoint_deterministic_and_digest_locked() -> None:
    poses = _problem()

    first = split_absolute_hand_eye_poses(poses, seed=17, fit_count=10, holdout_count=5)
    repeated = split_absolute_hand_eye_poses(
        tuple(reversed(poses)), seed=17, fit_count=10, holdout_count=5
    )

    assert first == repeated
    assert {pose.pair_id for pose in first.fit_poses}.isdisjoint(
        pose.pair_id for pose in first.holdout_poses
    )
    assert len(first.fit_ids_sha256) == 64
    assert len(first.holdout_ids_sha256) == 64


def test_pairwise_motions_share_opencv_four_convention_and_recover_truth() -> None:
    poses = _problem()
    split = split_absolute_hand_eye_poses(poses, seed=5, fit_count=12, holdout_count=6)
    fit = opencv_compatible_hand_eye_motions(split.fit_poses)
    holdout = opencv_compatible_hand_eye_motions(split.holdout_poses)

    result = ParkMartinHandEyeSolver().solve(
        fit,
        ParkMartinHandEyeOptions(holdout_ratio=0.0),
    )

    assert len(fit) == 66
    assert len(holdout) == 15
    assert result.status == "converged"
    assert result.transform_x is not None
    evaluation = evaluate_hand_eye_motions(holdout, result.transform_x)
    assert evaluation.rotation_closure_rmse_deg == pytest.approx(0.0, abs=1.0e-5)
    assert evaluation.translation_closure_rmse_m == pytest.approx(0.0, abs=1.0e-8)


def test_absolute_split_rejects_source_pose_leakage_preconditions() -> None:
    poses = _problem()

    with pytest.raises(ValueError, match="population"):
        split_absolute_hand_eye_poses(poses, seed=0, fit_count=15, holdout_count=5)
    with pytest.raises(ValueError, match="unique"):
        split_absolute_hand_eye_poses((*poses, poses[0]), seed=0, fit_count=10, holdout_count=5)
