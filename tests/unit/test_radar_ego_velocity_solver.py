import math

from calibrex.solvers import RadarEgoVelocitySolver as PublicSolver
from calibrex.solvers.radar_ego_velocity_solver import (
    RadarDopplerObservation,
    RadarEgoVelocitySolver,
)


def _observations(
    velocity: tuple[float, float, float],
    *,
    outlier: bool = False,
) -> list[RadarDopplerObservation]:
    directions = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (-1.0, 0.5, 0.3),
        (0.4, -0.8, 1.2),
    ]
    observations = []
    for index, direction in enumerate(directions):
        norm = math.sqrt(sum(value * value for value in direction))
        unit = tuple(value / norm for value in direction)
        doppler = -sum(left * right for left, right in zip(unit, velocity, strict=True))
        if outlier and index == len(directions) - 1:
            doppler += 20.0
        observations.append(RadarDopplerObservation(direction, doppler))
    return observations


def test_radar_ego_velocity_solver_recovers_velocity_with_outlier() -> None:
    assert PublicSolver is RadarEgoVelocitySolver
    truth = (8.0, -1.5, 0.7)

    result = RadarEgoVelocitySolver().solve(_observations(truth, outlier=True))

    assert result.status == "converged"
    assert result.velocity_radar_mps is not None
    assert result.rank == 3
    assert result.inlier_count == 7
    assert result.rmse_mps is not None and result.rmse_mps < 0.2
    error = math.sqrt(
        sum(
            (left - right) ** 2
            for left, right in zip(result.velocity_radar_mps, truth, strict=True)
        )
    )
    assert error < 0.25


def test_radar_ego_velocity_solver_rejects_degenerate_los_geometry() -> None:
    observations = [
        RadarDopplerObservation((1.0, 0.0, 0.0), -5.0)
        for _ in range(8)
    ]

    result = RadarEgoVelocitySolver().solve(observations)

    assert result.status == "degenerate_geometry"
    assert result.velocity_radar_mps is None
    assert result.rank == 1


def test_radar_ego_velocity_solver_requires_minimum_returns() -> None:
    result = RadarEgoVelocitySolver().solve(_observations((1.0, 2.0, 3.0))[:4])

    assert result.status == "insufficient_returns"
    assert result.velocity_radar_mps is None
