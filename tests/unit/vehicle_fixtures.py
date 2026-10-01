"""A simulated ground vehicle written as a bag, for the vehicle pairs of ``calibrex check``.

The vehicle (``base_link``) drives a bicycle-model path with hills and suspension
motion; an INS (``imu_link``) and a LiDAR (``velo_link``) are mounted with known
rotations. The bag carries the INS as ``nav_msgs/Odometry`` (pose and body-frame
twist), wheel speed and yaw rate as ``TwistStamped``, and header-only IMU and
point-cloud messages (the LiDAR odometry is injected by the tests, so no scans
are needed). ``/tf_static`` holds the true mounts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from calibrex.check.vehicle_inputs import LidarOdometryTrack
from calibrex.data.ros_cdr_writer import (
    encode_header_only,
    encode_odometry,
    encode_tf_message,
    encode_twist_stamped,
)
from calibrex.data.rosbag2_writer import Rosbag2Writer
from calibrex.solvers.scan_to_scan_odometry import ScanOdometryOptions

FloatArray = NDArray[np.float64]
BASE_NS = 1_700_000_000_000_000_000
DT = 0.1


def rotation_matrix(roll_pitch_yaw_deg: tuple[float, float, float]) -> FloatArray:
    return np.asarray(
        Rotation.from_euler("xyz", roll_pitch_yaw_deg, degrees=True).as_matrix(), dtype=np.float64
    )


R_BASE_IMU = rotation_matrix((0.2, -0.4, 0.3))
R_BASE_VELO = rotation_matrix((-0.8, 0.5, -0.3))
LEVER_VELO = np.array([1.2, 0.0, 1.5])


def _transform(rotation: FloatArray, translation: FloatArray | None = None) -> FloatArray:
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    if translation is not None:
        matrix[:3, 3] = translation
    return matrix


T_BASE_IMU = _transform(R_BASE_IMU)
T_BASE_VELO = _transform(R_BASE_VELO, LEVER_VELO)


@dataclass(frozen=True)
class Simulation:
    times_s: FloatArray
    vehicle_poses: list[FloatArray]  # T_world_base
    lidar_poses: list[FloatArray]  # T_first_scan of the LiDAR (what scan odometry reports)


def simulate(duration_s: float = 120.0, *, seed: int = 3, turning: bool = True) -> Simulation:
    rng = np.random.default_rng(seed)
    times = np.arange(0.0, duration_s, DT)
    heading = 0.0
    position = np.zeros(3)
    vehicle_poses = []
    for index, time in enumerate(times):
        speed = 8.0 + 2.0 * np.sin(0.05 * time) + 1.5 * np.sin(0.7 * time)
        yaw_rate = (0.15 * np.sin(0.2 * time) + 0.05 * np.sin(1.1 * time)) if turning else 0.0
        if index:
            new_heading = heading + yaw_rate * DT
            middle = 0.5 * (heading + new_heading)
            position = position + speed * DT * np.array([np.cos(middle), np.sin(middle), 0.0])
            heading = new_heading
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler(
            "ZYX", [heading, 0.002 * np.sin(3.0 * time), 0.003 * np.sin(2.3 * time)]
        ).as_matrix()
        pose[:3, 3] = position
        vehicle_poses.append(pose)
    lidar_world = []
    for pose in vehicle_poses:
        noise = np.eye(4)
        noise[:3, :3] = Rotation.from_rotvec(rng.normal(0.0, 0.0003, 3)).as_matrix()
        noise[:3, 3] = rng.normal(0.0, 0.005, 3)
        lidar_world.append(pose @ T_BASE_VELO @ noise)
    first = np.linalg.inv(lidar_world[0])
    return Simulation(times, vehicle_poses, [first @ pose for pose in lidar_world])


def lidar_track(simulation: Simulation, topic: str = "/velodyne_points") -> LidarOdometryTrack:
    return LidarOdometryTrack(
        topic=topic,
        times_s=tuple(float(BASE_NS * 1e-9 + t) for t in simulation.times_s),
        poses=tuple(simulation.lidar_poses),
        registrations=(),
        options=ScanOdometryOptions(),
    )


def _body_twist(
    poses: list[FloatArray], index: int
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Velocity and angular rate in the body frame between samples ``index`` and ``index + 1``."""

    last = min(index, len(poses) - 2)
    here, there = poses[last], poses[last + 1]
    velocity = here[:3, :3].T @ (there[:3, 3] - here[:3, 3]) / DT
    rate = Rotation.from_matrix(here[:3, :3].T @ there[:3, :3]).as_rotvec() / DT
    return (
        (float(velocity[0]), float(velocity[1]), float(velocity[2])),
        (float(rate[0]), float(rate[1]), float(rate[2])),
    )


def write_vehicle_bag(
    path: Path,
    simulation: Simulation,
    *,
    ins_topic: str = "/oxts/odometry",
    wheel_topic: str = "/wheel/twist",
    ins_frame: str = "imu_link",
    zero_ins_twist: bool = False,
    tf_edges: list[tuple[str, str, FloatArray]] | None = None,
) -> Path:
    imu_world = [pose @ T_BASE_IMU for pose in simulation.vehicle_poses]
    edges = tf_edges or [
        ("base_link", "imu_link", T_BASE_IMU),
        ("base_link", "velo_link", T_BASE_VELO),
    ]
    transforms = [
        (
            parent,
            child,
            (float(m[0, 3]), float(m[1, 3]), float(m[2, 3])),
            tuple(float(v) for v in Rotation.from_matrix(m[:3, :3]).as_quat()),
        )
        for parent, child, m in edges
    ]
    with Rosbag2Writer(path) as writer:
        writer.add_topic("/tf_static", "tf2_msgs/msg/TFMessage", latched=True)
        writer.write("/tf_static", BASE_NS, encode_tf_message(transforms))  # type: ignore[arg-type]
        writer.add_topic("/velodyne_points", "sensor_msgs/msg/PointCloud2")
        writer.write("/velodyne_points", BASE_NS, encode_header_only("velo_link"))
        writer.add_topic("/oxts/imu", "sensor_msgs/msg/Imu")
        writer.write("/oxts/imu", BASE_NS, encode_header_only("imu_link"))
        writer.add_topic(ins_topic, "nav_msgs/msg/Odometry")
        writer.add_topic(wheel_topic, "geometry_msgs/msg/TwistStamped")
        for index, time in enumerate(simulation.times_s):
            stamp = BASE_NS + round(time * 1e9)
            pose = imu_world[index]
            velocity, rate = _body_twist(imu_world, index)
            if zero_ins_twist:
                velocity, rate = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
            quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat()
            writer.write(
                ins_topic,
                stamp,
                encode_odometry(
                    frame_id="odom",
                    child_frame_id=ins_frame,
                    timestamp_ns=stamp,
                    position=(float(pose[0, 3]), float(pose[1, 3]), float(pose[2, 3])),
                    orientation_xyzw=(
                        float(quaternion[0]),
                        float(quaternion[1]),
                        float(quaternion[2]),
                        float(quaternion[3]),
                    ),
                    linear_velocity=velocity,
                    angular_velocity=rate,
                ),
            )
            wheel_velocity, wheel_rate = _body_twist(simulation.vehicle_poses, index)
            writer.write(
                wheel_topic,
                stamp,
                encode_twist_stamped(
                    frame_id="base_link",
                    timestamp_ns=stamp,
                    linear=(wheel_velocity[0], 0.0, 0.0),
                    angular=(0.0, 0.0, wheel_rate[2]),
                ),
            )
    return path
