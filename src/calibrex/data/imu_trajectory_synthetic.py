"""A synthetic hand-held IMU + sensor-trajectory recording with a known extrinsic.

Used by the browser demo and the tests.  The sensor sways and wobbles on every
axis, as a hand-held device does, so every rotation axis and every lever-arm
axis is observable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

FloatArray: TypeAlias = NDArray[np.float64]
GRAVITY_MPS2 = np.array([0.0, 0.0, -9.81])


@dataclass(frozen=True)
class SyntheticImuTrajectory:
    """Text inputs plus the truth they were generated from."""

    imu_csv: str
    trajectory_tum: str
    rotation_lidar_imu: FloatArray
    translation_lidar_imu_m: FloatArray
    time_offset_s: float
    gyro_bias_rps: FloatArray


def _attitude(local: FloatArray) -> Rotation:
    roll = 0.35 * np.sin(1.1 * local) + 0.12 * np.sin(7.0 * local)
    pitch = 0.3 * np.sin(0.9 * local + 1.0) + 0.1 * np.sin(8.5 * local + 0.5)
    yaw = 0.8 * np.sin(0.3 * local) + 0.4 * np.sin(1.3 * local) + 0.1 * np.sin(6.0 * local)
    return Rotation.from_euler("xyz", np.column_stack([roll, pitch, yaw]))


def _position(local: FloatArray) -> FloatArray:
    return np.column_stack(
        [0.6 * np.sin(0.4 * local), 0.5 * np.cos(0.5 * local), 0.15 * np.sin(0.7 * local)]
    )


def synthetic_imu_trajectory(
    *,
    duration_s: float = 120.0,
    seed: int = 0,
    epoch_s: float = 1.7e9,
    rotation_rpy_deg: tuple[float, float, float] = (0.8, -1.5, 2.0),
    translation_m: tuple[float, float, float] = (0.011, 0.023, -0.044),
    time_offset_s: float = 0.012,
    gyro_bias_rps: tuple[float, float, float] = (0.004, -0.002, 0.003),
    accel_bias_mps2: tuple[float, float, float] = (0.03, -0.02, 0.05),
    imu_rate_hz: float = 200.0,
    pose_rate_hz: float = 10.0,
) -> SyntheticImuTrajectory:
    """Generate 200 Hz IMU samples and 10 Hz sensor poses (``t_imu = t_sensor + dt``)."""

    rng = np.random.default_rng(seed)
    rotation = Rotation.from_euler("xyz", rotation_rpy_deg, degrees=True).as_matrix()
    lever = np.asarray(translation_m, dtype=np.float64)
    step = 1e-3
    imu_times = epoch_s + time_offset_s + np.arange(0.0, duration_s, 1.0 / imu_rate_hz)
    local = imu_times - time_offset_s - epoch_s

    def imu_pose(offset: float) -> tuple[Rotation, FloatArray]:
        sensor = _attitude(local + offset)
        origin = _position(local + offset) + sensor.apply(lever)
        return sensor * Rotation.from_matrix(rotation), origin

    before, back = imu_pose(-step)
    now, here = imu_pose(0.0)
    after, ahead = imu_pose(step)
    gyro = (before.inv() * after).as_rotvec() / (2.0 * step)
    acceleration = (ahead - 2.0 * here + back) / step**2
    accel = now.inv().apply(acceleration - GRAVITY_MPS2) + np.asarray(accel_bias_mps2)
    gyro = gyro + np.asarray(gyro_bias_rps) + rng.normal(0.0, 0.002, gyro.shape)
    accel = accel + rng.normal(0.0, 0.02, accel.shape)
    imu_rows = np.column_stack([imu_times, gyro, accel])
    imu_csv = "t,gx,gy,gz,ax,ay,az\n" + "\n".join(
        f"{row[0]:.9f}," + ",".join(f"{value:.6f}" for value in row[1:]) for row in imu_rows
    )

    pose_local = 0.5 + np.arange(0.0, duration_s - 1.0, 1.0 / pose_rate_hz)
    noise = Rotation.from_rotvec(rng.normal(0.0, 0.0003, (len(pose_local), 3)))
    attitude = _attitude(pose_local) * noise
    positions = _position(pose_local) + rng.normal(0.0, 0.002, (len(pose_local), 3))
    quaternions = attitude.as_quat()
    trajectory = "# t x y z qx qy qz qw (T_world_sensor)\n" + "\n".join(
        f"{epoch_s + t:.6f} "
        + " ".join(f"{value:.6f}" for value in position)
        + " "
        + " ".join(f"{value:.8f}" for value in quaternion)
        for t, position, quaternion in zip(pose_local, positions, quaternions, strict=True)
    )
    return SyntheticImuTrajectory(
        imu_csv=imu_csv,
        trajectory_tum=trajectory,
        rotation_lidar_imu=rotation,
        translation_lidar_imu_m=lever,
        time_offset_s=time_offset_s,
        gyro_bias_rps=np.asarray(gyro_bias_rps, dtype=np.float64),
    )
