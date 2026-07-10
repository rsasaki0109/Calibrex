import math

import pytest

from calibrex.solvers.radar_trajectory_yaw_solver import (
    RadarTrajectoryVelocityPair,
    RadarTrajectoryYawSolver,
    RadarTrajectoryYawSolverOptions,
)


def _pairs(yaw_deg: float, count: int = 24) -> list[RadarTrajectoryVelocityPair]:
    yaw = math.radians(yaw_deg)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    headings = (-80.0, -55.0, -30.0, -5.0, 20.0, 45.0, 70.0, 95.0)
    pairs = []
    for index in range(count):
        heading = math.radians(headings[index % len(headings)])
        speed = 5.0 + index % 7
        rx, ry = speed * math.cos(heading), speed * math.sin(heading)
        pairs.append(
            RadarTrajectoryVelocityPair(
                frame_id=f"frame-{index:03d}",
                velocity_radar_mps=(rx, ry, 0.0),
                trajectory_velocity_at_radar_origin_ego_mps=(
                    cosine * rx - sine * ry,
                    sine * rx + cosine * ry,
                    0.0,
                ),
            )
        )
    return pairs


@pytest.mark.parametrize("truth_deg", [23.0, 179.5, -179.5])
def test_radar_trajectory_yaw_solver_recovers_truth_and_holdout(truth_deg: float) -> None:
    result = RadarTrajectoryYawSolver().solve(
        _pairs(truth_deg),
        RadarTrajectoryYawSolverOptions(holdout_ratio=0.25, split_seed=7),
    )

    assert result.status == "converged"
    assert result.yaw_rad is not None
    error = math.atan2(
        math.sin(result.yaw_rad - math.radians(truth_deg)),
        math.cos(result.yaw_rad - math.radians(truth_deg)),
    )
    assert abs(error) < 1.0e-9
    assert result.holdout_rmse_mps is not None and result.holdout_rmse_mps < 1.0e-9
    assert len(result.train_frame_ids) == 18
    assert len(result.holdout_frame_ids) == 6
    assert set(result.train_frame_ids).isdisjoint(result.holdout_frame_ids)
    assert all(probe.detectable is True for probe in result.probes)


def test_radar_trajectory_yaw_solver_is_order_invariant_and_robust() -> None:
    pairs = _pairs(30.0, count=30)
    for index in range(0, 30, 7):
        pair = pairs[index]
        pairs[index] = RadarTrajectoryVelocityPair(
            frame_id=pair.frame_id,
            velocity_radar_mps=pair.velocity_radar_mps,
            trajectory_velocity_at_radar_origin_ego_mps=(-20.0, 15.0, 0.0),
        )
    options = RadarTrajectoryYawSolverOptions(
        holdout_ratio=0.2,
        split_seed=3,
        huber_delta_mps=0.25,
    )

    forward = RadarTrajectoryYawSolver().solve(pairs, options)
    reverse = RadarTrajectoryYawSolver().solve(list(reversed(pairs)), options)

    assert forward.yaw_rad is not None
    assert reverse.yaw_rad == pytest.approx(forward.yaw_rad)
    assert reverse.train_frame_ids == forward.train_frame_ids
    assert reverse.holdout_frame_ids == forward.holdout_frame_ids
    assert abs(forward.yaw_rad - math.radians(30.0)) < math.radians(1.0)


def test_radar_trajectory_yaw_solver_rejects_single_heading_coverage() -> None:
    pairs = [
        RadarTrajectoryVelocityPair(
            frame_id=f"frame-{index}",
            velocity_radar_mps=(5.0 + index, 0.0, 0.0),
            trajectory_velocity_at_radar_origin_ego_mps=(0.0, 5.0 + index, 0.0),
        )
        for index in range(10)
    ]

    result = RadarTrajectoryYawSolver().solve(pairs)

    assert result.status == "degenerate_motion"
    assert result.yaw_rad is None
    assert result.direction_diversity == pytest.approx(0.0)
