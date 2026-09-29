"""Convert a Livox ROS 2 bag into the ROS 1 bag LI-Init reads.

Point clouds are read through Calibrex's Livox stream profiles (so point
times are normalized to offsets after the header stamp) and written as
``livox_ros_driver/CustomMsg`` with ``offset_time`` in nanoseconds; IMU
messages are copied, converting acceleration to the unit LI-Init is told to
expect through ``mean_acc_norm``.

This helper needs the ``rosbags`` package (Apache-2.0) in addition to
Calibrex; it is not a Calibrex dependency.  Usage::

    python convert_livox_ros2_to_ros1.py BAG_DIR PROFILE OUTPUT.bag [--max-seconds S]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

from calibrex.data.livox_ros2 import (
    LIVOX_PROFILES,
    STANDARD_GRAVITY_MPS2,
    iter_livox_points,
    load_livox_imu,
)

CUSTOM_POINT = """uint32 offset_time
float32 x
float32 y
float32 z
uint8 reflectivity
uint8 tag
uint8 line
"""
CUSTOM_MSG = """std_msgs/Header header
uint64 timebase
uint32 point_num
uint8 lidar_id
uint8[3] rsvd
livox_ros_driver/CustomPoint[] points
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bag", type=Path)
    parser.add_argument("profile", choices=sorted(LIVOX_PROFILES))
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument(
        "--imu-acceleration-unit",
        choices=("mps2", "g"),
        default="mps2",
        help="unit written to the ROS 1 bag (set LI-Init mean_acc_norm to match)",
    )
    args = parser.parse_args()

    store = get_typestore(Stores.ROS1_NOETIC)
    types = {}
    types.update(get_types_from_msg(CUSTOM_POINT, "livox_ros_driver/msg/CustomPoint"))
    types.update(get_types_from_msg(CUSTOM_MSG, "livox_ros_driver/msg/CustomMsg"))
    store.register(types)
    header_type = store.types["std_msgs/msg/Header"]
    time_type = store.types["builtin_interfaces/msg/Time"]
    custom_msg = store.types["livox_ros_driver/msg/CustomMsg"]
    custom_point = store.types["livox_ros_driver/msg/CustomPoint"]
    imu_type = store.types["sensor_msgs/msg/Imu"]
    vector = store.types["geometry_msgs/msg/Vector3"]
    quaternion = store.types["geometry_msgs/msg/Quaternion"]

    profile = LIVOX_PROFILES[args.profile]
    imu = load_livox_imu(args.bag, profile)
    start = float(imu.times_s[0])
    end = start + args.max_seconds if args.max_seconds else float("inf")
    scale = 1.0 / STANDARD_GRAVITY_MPS2 if args.imu_acceleration_unit == "g" else 1.0

    def stamp(seconds: float) -> object:
        whole = int(seconds)
        return time_type(sec=whole, nanosec=round((seconds - whole) * 1e9))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Messages are written in time order as they stream in; nothing but the
    # (small) IMU series is held in memory.
    count = 0
    imu_written = 0
    zero_covariance = np.zeros(9, dtype=np.float64)
    with Writer(args.output) as writer:
        lidar = writer.add_connection("/livox/lidar", custom_msg.__msgtype__, typestore=store)
        imu_connection = writer.add_connection("/livox/imu", imu_type.__msgtype__, typestore=store)

        def write_imu_until(limit_s: float) -> None:
            nonlocal imu_written
            while imu_written < len(imu.times_s) and imu.times_s[imu_written] <= limit_s:
                time_s = float(imu.times_s[imu_written])
                gyro = imu.gyro_rps[imu_written]
                accel = imu.accel_mps2[imu_written] * scale
                message = imu_type(
                    header=header_type(seq=imu_written, stamp=stamp(time_s), frame_id="livox_imu"),
                    orientation=quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                    orientation_covariance=zero_covariance,
                    angular_velocity=vector(x=float(gyro[0]), y=float(gyro[1]), z=float(gyro[2])),
                    angular_velocity_covariance=zero_covariance,
                    linear_acceleration=vector(
                        x=float(accel[0]), y=float(accel[1]), z=float(accel[2])
                    ),
                    linear_acceleration_covariance=zero_covariance,
                )
                writer.write(
                    imu_connection,
                    round(time_s * 1e9),
                    store.serialize_ros1(message, imu_connection.msgtype),
                )
                imu_written += 1

        for time_s, xyz, offsets in iter_livox_points(args.bag, profile):
            if time_s > end:
                break
            if offsets is None:
                raise SystemExit("the profile provides no per-point times; LI-Init needs them")
            write_imu_until(time_s)
            order = np.argsort(offsets, kind="stable")
            points = [
                custom_point(
                    offset_time=int(max(offsets[index], 0.0) * 1e9),
                    x=float(xyz[index, 0]),
                    y=float(xyz[index, 1]),
                    z=float(xyz[index, 2]),
                    reflectivity=0,
                    tag=0,
                    line=0,
                )
                for index in order
            ]
            message = custom_msg(
                header=header_type(seq=count, stamp=stamp(time_s), frame_id="livox_frame"),
                timebase=round(time_s * 1e9),
                point_num=len(points),
                lidar_id=0,
                rsvd=np.zeros(3, dtype=np.uint8),
                points=points,
            )
            writer.write(
                lidar, round(time_s * 1e9), store.serialize_ros1(message, lidar.msgtype)
            )
            count += 1
        write_imu_until(end if end != float("inf") else float(imu.times_s[-1]))
    print(f"wrote {count} scans and {imu_written} IMU samples to {args.output}")


if __name__ == "__main__":
    main()
