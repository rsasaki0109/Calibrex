"""The first-run speed-ups must not change a single bit of the results.

Each test compares an optimized path with a straightforward reference written the
way the code was before the optimization.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from calibrex.data.ros_cdr import decode_ros2_imu
from calibrex.data.ros_cdr_writer import encode_imu
from calibrex.data.ros_messages import filter_nonfinite_pointcloud_rows
from calibrex.solvers.imu_lidar_rotation_solver import GyroSeries
from calibrex.solvers.scan_to_scan_odometry import (
    ScanOdometryOptions,
    TargetNormals,
    estimate_normals,
    preprocess_scan,
    register_point_to_plane,
)


def _room(rng: np.random.Generator, count: int = 4000) -> np.ndarray:
    """Points on three perpendicular planes plus clutter (some neighbourhoods are not planar)."""

    walls = [
        np.column_stack([rng.uniform(-8, 8, count), rng.uniform(-8, 8, count), np.zeros(count)]),
        np.column_stack([rng.uniform(-8, 8, count), np.full(count, 6.0), rng.uniform(0, 3, count)]),
        np.column_stack([np.full(count, -7.0), rng.uniform(-8, 8, count), rng.uniform(0, 3, count)]),
        rng.uniform(-8, 8, (count // 4, 3)),
    ]
    return np.vstack(walls) + rng.normal(0.0, 0.01, (3 * count + count // 4, 3))


def test_lazy_normals_are_bit_identical_to_the_eager_ones() -> None:
    rng = np.random.default_rng(1)
    points = _room(rng)
    options = ScanOdometryOptions()
    tree = cKDTree(points)
    normals, planar = estimate_normals(points, tree, options)
    lazy = TargetNormals(points, tree, options)
    wanted = rng.choice(len(points), size=1500, replace=False)
    lazy.ensure(wanted[:700])
    lazy.ensure(wanted)  # partly present already
    lazy.ensure(np.array([], dtype=np.intp))
    assert np.array_equal(lazy.normals[wanted], normals[wanted])
    assert np.array_equal(lazy.planar[wanted], planar[wanted])
    assert planar.any() and not planar.all()


def test_registration_is_identical_with_lazy_eager_and_default_normals() -> None:
    rng = np.random.default_rng(2)
    target = _room(rng)
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec([0.01, -0.02, 0.03]).as_matrix()
    transform[:3, 3] = [0.15, -0.1, 0.05]
    source = (target[::2] - transform[:3, 3]) @ transform[:3, :3]
    source = source + rng.normal(0.0, 0.005, source.shape)
    options = ScanOdometryOptions()
    tree = cKDTree(target)
    eager = estimate_normals(target, tree, options)
    results = [
        register_point_to_plane(source, target, np.eye(4), options),
        register_point_to_plane(source, target, np.eye(4), options, target_tree=tree),
        register_point_to_plane(
            source, target, np.eye(4), options, target_tree=tree, target_normals=eager
        ),
        register_point_to_plane(
            source,
            target,
            np.eye(4),
            options,
            target_tree=tree,
            target_normals=TargetNormals(target, tree, options),
        ),
    ]
    first = results[0]
    assert first.converged
    for other in results[1:]:
        assert np.array_equal(first.transform, other.transform)
        assert (first.iterations, first.correspondences) == (other.iterations, other.correspondences)
        assert first.rmse_m == other.rmse_m
        assert first.hessian_min_eigenvalue == other.hessian_min_eigenvalue


def test_preprocess_scan_range_crop_matches_linalg_norm() -> None:
    rng = np.random.default_rng(3)
    cloud = rng.normal(size=(200_000, 3)) * 25.0
    options = ScanOdometryOptions()
    ranges = np.linalg.norm(cloud, axis=1)
    keep = cloud[(ranges >= options.min_range_m) & (ranges <= options.max_range_m)]
    expected = preprocess_scan(keep, ScanOdometryOptions(min_range_m=0.0, max_range_m=1.0e9))
    assert np.array_equal(
        preprocess_scan(cloud, options),
        expected,
    )


def test_nonfinite_filter_matches_isfinite_all() -> None:
    rng = np.random.default_rng(4)
    xyz = rng.normal(size=(1000, 3))
    xyz[3, 0] = np.nan
    xyz[10, 2] = np.inf
    xyz[11, 1] = -np.inf
    intensity = rng.normal(size=1000)
    filtered, kept_intensity, _, dropped = filter_nonfinite_pointcloud_rows(np, xyz, intensity, None)
    mask = np.isfinite(xyz).all(axis=1)
    assert dropped == 3
    assert np.array_equal(filtered, xyz[mask])
    assert np.array_equal(kept_intensity, intensity[mask])


@pytest.mark.parametrize("frame_id", ["imu", "livox_frame", "a"])
def test_imu_block_decode_reads_every_field(frame_id: str) -> None:
    payload = encode_imu(
        frame_id=frame_id,
        timestamp_ns=1_234_567_890_123,
        angular_velocity=(0.1, -0.2, 0.3),
        linear_acceleration=(9.5, 0.25, -1.0),
        orientation_xyzw=(0.1, 0.2, 0.3, 0.9),
        orientation_known=True,
    )
    message = decode_ros2_imu("/imu", 1, payload)
    assert message.frame_id == frame_id
    assert message.timestamp_ns == 1_234_567_890_123
    assert message.orientation_xyzw == (0.1, 0.2, 0.3, 0.9)
    assert message.angular_velocity == (0.1, -0.2, 0.3)
    assert message.linear_acceleration == (9.5, 0.25, -1.0)
    assert len(message.orientation_covariance) == 9
    assert len(message.angular_velocity_covariance) == 9
    assert len(message.linear_acceleration_covariance) == 9
    with pytest.raises(Exception, match="truncated"):
        decode_ros2_imu("/imu", 1, payload[:-8])


def test_gyro_tables_are_shared_by_content_not_identity() -> None:
    rng = np.random.default_rng(5)
    times = np.cumsum(rng.uniform(0.004, 0.006, 500))
    gyro = rng.normal(0.0, 0.5, (500, 3))
    first = GyroSeries(times, gyro)
    again = GyroSeries(times.copy(), gyro.copy())
    other = GyroSeries(times, gyro * 1.0001)
    query = np.linspace(times[5], times[-5], 40)
    expected = first.orientation_and_bias_sum(query)
    for got, want in zip(again.orientation_and_bias_sum(query), expected, strict=True):
        assert np.array_equal(got, want)
    assert not np.array_equal(other.orientation_and_bias_sum(query)[0], expected[0])
    assert np.array_equal(again.integral_at(query), first.integral_at(query))
