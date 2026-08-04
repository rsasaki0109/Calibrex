from __future__ import annotations

import math

from calibrex.core.geometry import SE3
from calibrex.data.livox import LivoxPointRecord
from calibrex.data.odometry_track import OdometryPoseSample, OdometryTrack
from calibrex.solvers.continuous_time_lidar_pair_solver import (
    ContinuousTimeLidarPairOptions,
    ContinuousTimeLidarPairProblem,
    ContinuousTimeLidarPairSolver,
    _select_evenly_spaced_indices,
)
from calibrex.solvers.fixed_trajectory_se3_solver import FixedTrajectorySe3SolverOptions


def _yaw_quaternion(degrees: float) -> tuple[float, float, float, float]:
    half = math.radians(degrees) * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _world_plane_records() -> list[LivoxPointRecord]:
    records: list[LivoxPointRecord] = []
    for x in range(0, 9):
        for y in range(-4, 5):
            records.append(
                LivoxPointRecord(
                    point=(0.5 * x, 0.5 * y, 0.0, 0.0),
                    normal_xyz=(0.0, 0.0, 1.0),
                )
            )
    for y in range(-4, 5):
        for z in range(0, 7):
            records.append(
                LivoxPointRecord(
                    point=(4.0, 0.5 * y, 0.5 * z, 0.0),
                    normal_xyz=(1.0, 0.0, 0.0),
                )
            )
    for x in range(0, 9):
        for z in range(0, 7):
            records.append(
                LivoxPointRecord(
                    point=(0.5 * x, -2.0, 0.5 * z, 0.0),
                    normal_xyz=(0.0, 1.0, 0.0),
                )
            )
    return records


def test_continuous_time_profile_reoptimizes_extrinsic_and_clock() -> None:
    source_records = _world_plane_records()
    true_extrinsic = SE3((0.12, -0.05, 0.04), _yaw_quaternion(8.0))
    odometry_samples = [
        OdometryPoseSample(
            timestamp_ns=1_000_000_000 + index * 100_000_000,
            pose=SE3(
                (0.12 * index, 0.0, 0.0),
                _yaw_quaternion(4.0 * index),
            ),
        )
        for index in range(12)
    ]
    track = OdometryTrack(odometry_samples)
    true_offset_s = 0.03
    target_points: list[tuple[float, float, float]] = []
    target_timestamps_ns: list[int] = []
    selected_world_points = [record.point[:3] for record in source_records[::7]]
    for sample in odometry_samples[1:-1]:
        for point_world in selected_world_points[:12]:
            point_source = sample.pose.inverse().transform_point(point_world)
            target_points.append(true_extrinsic.inverse().transform_point(point_source))
            target_timestamps_ns.append(sample.timestamp_ns - round(true_offset_s * 1e9))

    initial_extrinsic = SE3((0.18, -0.02, 0.08), _yaw_quaternion(3.0))
    split = int(len(target_points) * 0.75)
    problem = ContinuousTimeLidarPairProblem(
        source_records=source_records,
        target_points=target_points,
        target_capture_timestamps_ns=target_timestamps_ns,
        odometry_track=track,
        t_base_source=SE3.identity(),
        initial_t_source_target=initial_extrinsic,
        variable="T_base_lidar_target",
        sensor="lidar_target",
        voxel_size_m=0.5,
        correspondence_gate_m=0.8,
        train_indices=tuple(range(split)),
        holdout_indices=tuple(range(split, len(target_points))),
    )
    result = ContinuousTimeLidarPairSolver().solve(
        problem,
        ContinuousTimeLidarPairOptions(
            max_abs_time_offset_sec=0.1,
            initial_time_step_sec=0.02,
                minimum_time_step_sec=0.002,
                max_iterations=6,
                outlier_policy="none",
                solver=FixedTrajectorySe3SolverOptions(max_iterations=12),
        ),
    )

    assert result.status in {"converged", "max_iterations"}
    assert result.observability is not None
    assert result.train_correspondence_count >= 6
    assert result.final_train_rmse_m is not None
    assert result.initial_train_rmse_m is not None
    assert result.final_train_rmse_m <= result.initial_train_rmse_m + 1.0e-9
    assert abs(result.estimated_time_offset_sec - true_offset_s) <= 0.025
    assert result.iterations
    assert all(
        len(iteration.candidate_train_rmse_m) == len(iteration.candidate_offsets_sec)
        for iteration in result.iterations
    )
    assert any(
        rmse is not None
        for iteration in result.iterations
        for rmse in iteration.candidate_train_rmse_m
    )


def test_adaptive_plane_map_can_disable_uniform_fallback() -> None:
    source_records = _world_plane_records()
    odometry_samples = [
        OdometryPoseSample(
            timestamp_ns=2_000_000_000 + index * 100_000_000,
            pose=SE3((0.08 * index, 0.0, 0.0), _yaw_quaternion(2.0 * index)),
        )
        for index in range(8)
    ]
    track = OdometryTrack(odometry_samples)
    selected_points = [record.point[:3] for record in source_records[::9]]
    target_points = selected_points * len(odometry_samples[1:-1])
    target_timestamps_ns = [
        sample.timestamp_ns
        for sample in odometry_samples[1:-1]
        for _ in range(len(selected_points))
    ]
    split = len(target_points) // 2
    problem = ContinuousTimeLidarPairProblem(
        source_records=source_records,
        target_points=target_points,
        target_capture_timestamps_ns=target_timestamps_ns,
        odometry_track=track,
        t_base_source=SE3.identity(),
        initial_t_source_target=SE3.identity(),
        variable="T_base_lidar_target",
        sensor="lidar_target",
        voxel_size_m=0.5,
        correspondence_gate_m=0.8,
        train_indices=tuple(range(split)),
        holdout_indices=tuple(range(split, len(target_points))),
    )
    result = ContinuousTimeLidarPairSolver().solve(
        problem,
        ContinuousTimeLidarPairOptions(
            voxel_strategy="adaptive",
            adaptive_use_uniform_fallback=False,
            adaptive_min_points_per_voxel=3,
            max_iterations=2,
            outlier_policy="none",
            solver=FixedTrajectorySe3SolverOptions(max_iterations=4),
        ),
    )

    assert result.train_correspondence_count >= 6


def test_continuous_time_seeded_sampling_is_deterministic_and_changes_population() -> None:
    evenly_spaced = _select_evenly_spaced_indices(range(20), 8)
    seeded_first = _select_evenly_spaced_indices(range(20), 8, seed=17)
    seeded_second = _select_evenly_spaced_indices(range(20), 8, seed=17)

    assert seeded_first == seeded_second
    assert seeded_first[0] == 0
    assert seeded_first[-1] == 19
    assert seeded_first != evenly_spaced
