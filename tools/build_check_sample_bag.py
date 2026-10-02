"""Write the tiny synthetic rosbag2 that the browser check page offers as a sample.

The bag has a ``/tf_static`` rig (``base_link`` with an IMU, two LiDARs, a camera and a
GNSS antenna), an IMU, two PointCloud2 LiDARs, a mono camera with its CameraInfo and a
NavSatFix. It is deterministic (no random numbers, fixed stamps) and a few hundred KB
at most; the sensor data is synthetic noise-free geometry, not a recording, so it is
good for a plan (what could be checked) and not for running the estimators::

    python tools/build_check_sample_bag.py            # docs/app/samples/check_sample/
    python tools/build_check_sample_bag.py --output /tmp/check_sample

The writer is ``calibrex.data.rosbag2_writer``; the sqlite file header carries the
sqlite library version, so compare the bag by its messages (not by bytes) across
machines.
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

from calibrex.data.ros_cdr_writer import (
    POINT_FIELD_FLOAT32,
    Transform,
    encode_camera_info,
    encode_image,
    encode_imu,
    encode_navsatfix,
    encode_pointcloud2,
    encode_tf_message,
)
from calibrex.data.rosbag2_writer import Rosbag2Writer

DEFAULT_OUTPUT = Path("docs/app/samples/check_sample")
START_NS = 1_700_000_000_000_000_000
DURATION_S = 2.0
IMU_HZ = 100
LIDAR_HZ = 5
CAMERA_HZ = 5
GNSS_HZ = 5

IDENTITY = (0.0, 0.0, 0.0, 1.0)
YAW_180 = (0.0, 0.0, 1.0, 0.0)
OPTICAL = (-0.5, 0.5, -0.5, 0.5)  # camera body (x forward) -> optical (z forward)

RIG: list[Transform] = [
    ("base_link", "imu_link", (0.0, 0.0, 0.12), IDENTITY),
    ("base_link", "lidar_front", (1.1, 0.0, 0.55), IDENTITY),
    ("base_link", "lidar_rear", (-1.1, 0.0, 0.55), YAW_180),
    ("base_link", "camera_optical", (0.9, 0.05, 0.85), OPTICAL),
    ("base_link", "gnss_link", (0.0, 0.0, 1.3), IDENTITY),
]
POINT_FIELDS = (
    ("x", 0, POINT_FIELD_FLOAT32, 1),
    ("y", 4, POINT_FIELD_FLOAT32, 1),
    ("z", 8, POINT_FIELD_FLOAT32, 1),
    ("intensity", 12, POINT_FIELD_FLOAT32, 1),
)
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 24


def _cloud(frame_id: str, stamp_ns: int, phase: float) -> bytes:
    """A 16-ring, 24-azimuth range-image cloud of a cylindrical room (xyz + intensity)."""

    points = bytearray()
    for ring in range(16):
        elevation = math.radians(-15.0 + 2.0 * ring)
        for column in range(24):
            azimuth = math.radians(15.0 * column) + phase
            radius = 6.0 + 0.5 * math.sin(3.0 * azimuth)
            x = radius * math.cos(elevation) * math.cos(azimuth)
            y = radius * math.cos(elevation) * math.sin(azimuth)
            z = radius * math.sin(elevation)
            points += struct.pack("<ffff", x, y, z, 0.5 + 0.5 * math.sin(azimuth + elevation))
    return encode_pointcloud2(
        frame_id=frame_id,
        timestamp_ns=stamp_ns,
        fields=POINT_FIELDS,
        point_step=16,
        data=bytes(points),
    )


def _image(stamp_ns: int, index: int) -> bytes:
    pixels = bytes(
        (16 * x + 5 * y + 9 * index) % 256 for y in range(IMAGE_HEIGHT) for x in range(IMAGE_WIDTH)
    )
    return encode_image(
        frame_id="camera_optical",
        timestamp_ns=stamp_ns,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
        encoding="mono8",
        step=IMAGE_WIDTH,
        data=pixels,
    )


def build_sample_bag(directory: str | Path) -> Path:
    """Write the sample bag to ``directory`` (replacing a previous sample) and return it."""

    directory = Path(directory)
    with Rosbag2Writer(
        directory,
        custom_data={"generator": "tools/build_check_sample_bag.py", "synthetic": "true"},
        overwrite=True,
    ) as writer:
        writer.add_topic("/tf_static", "tf2_msgs/msg/TFMessage", latched=True)
        writer.add_topic("/imu/data", "sensor_msgs/msg/Imu")
        writer.add_topic("/lidar_front/points", "sensor_msgs/msg/PointCloud2")
        writer.add_topic("/lidar_rear/points", "sensor_msgs/msg/PointCloud2")
        writer.add_topic("/camera/image_raw", "sensor_msgs/msg/Image")
        writer.add_topic("/camera/camera_info", "sensor_msgs/msg/CameraInfo")
        writer.add_topic("/gnss/fix", "sensor_msgs/msg/NavSatFix")
        writer.write("/tf_static", START_NS, encode_tf_message(RIG, secs=START_NS // 1_000_000_000))
        for index in range(int(DURATION_S * IMU_HZ)):
            stamp = START_NS + index * 1_000_000_000 // IMU_HZ
            t = index / IMU_HZ
            writer.write(
                "/imu/data",
                stamp,
                encode_imu(
                    frame_id="imu_link",
                    timestamp_ns=stamp,
                    angular_velocity=(0.2 * math.sin(2.0 * t), 0.1 * math.cos(3.0 * t), 0.3),
                    linear_acceleration=(0.5 * math.sin(t), 0.0, 9.81),
                ),
            )
        for index in range(int(DURATION_S * LIDAR_HZ)):
            stamp = START_NS + index * 1_000_000_000 // LIDAR_HZ
            writer.write("/lidar_front/points", stamp, _cloud("lidar_front", stamp, 0.02 * index))
            writer.write(
                "/lidar_rear/points", stamp, _cloud("lidar_rear", stamp, 0.02 * index + math.pi)
            )
        for index in range(int(DURATION_S * CAMERA_HZ)):
            stamp = START_NS + index * 1_000_000_000 // CAMERA_HZ
            writer.write("/camera/image_raw", stamp, _image(stamp, index))
            writer.write(
                "/camera/camera_info",
                stamp,
                encode_camera_info(
                    frame_id="camera_optical",
                    timestamp_ns=stamp,
                    height=IMAGE_HEIGHT,
                    width=IMAGE_WIDTH,
                    k=(30.0, 0.0, 16.0, 0.0, 30.0, 12.0, 0.0, 0.0, 1.0),
                    d=(0.0, 0.0, 0.0, 0.0, 0.0),
                ),
            )
        for index in range(int(DURATION_S * GNSS_HZ)):
            stamp = START_NS + index * 1_000_000_000 // GNSS_HZ
            writer.write(
                "/gnss/fix",
                stamp,
                encode_navsatfix(
                    frame_id="gnss_link",
                    secs=stamp // 1_000_000_000,
                    nsecs=stamp % 1_000_000_000,
                    latitude=35.6586 + 1e-6 * index,
                    longitude=139.7454,
                    altitude=40.0,
                ),
            )
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="bag directory")
    args = parser.parse_args()
    bag = build_sample_bag(args.output)
    size = sum(path.stat().st_size for path in bag.iterdir())
    print(f"wrote {bag} ({size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
