"""Tests for deterministic solid-state residual outlier filtering."""

from __future__ import annotations

from calibrex.core.geometry import SE3
from calibrex.graph.lidar_point_to_plane import LidarPointToPlaneObservation
from calibrex.solvers.robust_lidar_observations import (
    filter_lidar_observations_mad,
)


def _plane_observations() -> list[LidarPointToPlaneObservation]:
    observations = [
        LidarPointToPlaneObservation(
            point_lidar_m=(float(index), 0.0, 0.002 * (index % 2)),
            plane_point_world_m=(float(index), 0.0, 0.0),
            plane_normal_world=(0.0, 0.0, 1.0),
        )
        for index in range(12)
    ]
    observations.extend(
        [
            LidarPointToPlaneObservation(
                point_lidar_m=(12.0, 0.0, 0.8),
                plane_point_world_m=(12.0, 0.0, 0.0),
                plane_normal_world=(0.0, 0.0, 1.0),
            ),
            LidarPointToPlaneObservation(
                point_lidar_m=(13.0, 0.0, -0.7),
                plane_point_world_m=(13.0, 0.0, 0.0),
                plane_normal_world=(0.0, 0.0, 1.0),
            ),
        ]
    )
    return observations


def test_mad_filter_rejects_transient_large_residuals() -> None:
    result = filter_lidar_observations_mad(
        _plane_observations(),
        variable="T_base_lidar",
        t_ego_lidar=SE3.identity(),
        sensor="lidar",
        minimum_threshold_m=0.01,
        minimum_inliers=6,
    )

    assert result.total_count == 14
    assert result.rejected_count == 2
    assert result.retained_count == 12
    assert result.threshold_m is not None
    assert result.threshold_m < 0.8
    assert result.fallback_used is False
