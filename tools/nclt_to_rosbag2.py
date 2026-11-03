#!/usr/bin/env python3
"""Convert a window of an NCLT session to a rosbag2 for ``calibrex check``/``estimate``.

An evaluation harness, not part of the Calibrex package. The University of Michigan North
Campus Long-Term dataset (NCLT; Carlevaris-Bianco, Ushani and Eustice, IJRR 2016; Open
Database License) records a Segway with a Velodyne HDL-32E, wheel encoders, a FOG and an
IMU. This tool streams one session's archives from NCLT's S3 bucket and writes, for the
window ``[start_s, start_s + duration_s)`` after the first Velodyne packet:

* ``/velodyne_points`` (``PointCloud2``, frame ``velodyne``): the raw ``velodyne_hits.bin``
  points in the sensor frame (never NCLT's ``velodyne_sync``, which is motion compensated
  and in the body frame), one scan per turn of the head (about 7.4 Hz), every
  ``decimate``-th hit kept, each with its packet's time after the scan stamp (``time``);
* ``/wheel_odom`` (``nav_msgs/Odometry``, ``odom`` -> ``base_link``): NCLT's 100 Hz
  odometry (an EKF on the wheel encoders, the FOG heading and the IMU's roll and pitch),
  as a pose and the body-frame forward speed and yaw rate;
* ``/tf_static``: ``base_link`` -> ``velodyne`` from NCLT's Table 4 (``x_body,vel``), the
  dataset's own calibration (itself estimated from odometry and LiDAR scan matching).

NCLT's body frame is x forward, y right, z down on the wheel axle; ``base_link`` here is
REP-103 (x forward, y left, z up), the body turned 180 deg about x::

    python tools/nclt_to_rosbag2.py --session 2012-01-08 --start-s 60 --duration-s 540 \\
        --decimate 4 --output BAG --work-dir WORK
"""

from __future__ import annotations

import argparse
import io
import json
import math
import struct
import subprocess
import sys
import tarfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import IO

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calibrex.data.ros_cdr_writer import (
    POINT_FIELD_FLOAT32,
    encode_odometry,
    encode_pointcloud2,
    encode_tf_message,
)
from calibrex.data.rosbag2_writer import Rosbag2Writer

S3 = "https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu"
X_BODY_VEL = (0.002, -0.004, -0.957, 0.807, 0.166, -90.703)
"""NCLT Table 4: the Velodyne in the body frame (x, y, z in m; roll, pitch, heading in deg)."""
MAGIC = struct.pack("<HHHH", 44444, 44444, 44444, 44444)
FIELDS = (
    ("x", 0, POINT_FIELD_FLOAT32, 1),
    ("y", 4, POINT_FIELD_FLOAT32, 1),
    ("z", 8, POINT_FIELD_FLOAT32, 1),
    ("time", 12, POINT_FIELD_FLOAT32, 1),
)
LIDAR_TOPIC, WHEEL_TOPIC = "/velodyne_points", "/wheel_odom"
FloatArray = NDArray[np.float64]


def ssc_to_matrix(ssc: tuple[float, ...]) -> FloatArray:
    """NCLT's 6-DOF ``[x, y, z, roll, pitch, heading]`` (deg) as ``T`` (Rz Ry Rx, then t)."""

    from scipy.spatial.transform import Rotation

    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("ZYX", [ssc[5], ssc[4], ssc[3]], degrees=True).as_matrix()
    matrix[:3, 3] = ssc[:3]
    return matrix


BODY_TO_BASE = np.diag([1.0, -1.0, -1.0, 1.0])
"""``T_base_body``: REP-103 base_link from NCLT's x-forward, y-right, z-down body frame."""


def reference_base_velodyne() -> FloatArray:
    """``T_base_link_velodyne`` from NCLT's ``x_body,vel``."""

    return BODY_TO_BASE @ ssc_to_matrix(X_BODY_VEL)


# --------------------------------------------------------------------------- odometry


def odometry_twist(rows: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """``(utime, position, quaternion_xyzw, (forward speed, yaw rate))`` in REP-103 frames.

    ``rows`` are ``odometry_mu_100hz.csv`` lines: utime and the body's x, y, z, roll,
    pitch, heading in NCLT's local frame (x north-ish, y east-ish, z down). Velocities are
    central differences, rotated into the body frame; the yaw rate is about base_link's z.
    """

    from scipy.spatial.transform import Rotation

    utime = rows[:, 0]
    seconds = (utime - utime[0]) * 1.0e-6
    rotations = Rotation.from_euler("ZYX", rows[:, [6, 5, 4]])
    velocity_local = np.gradient(rows[:, 1:4], seconds, axis=0)
    velocity_body = rotations.inv().apply(velocity_local)
    # body angular velocity from consecutive orientations: R_k^T R_{k+1} = exp(w dt)
    relative = rotations[:-1].inv() * rotations[1:]
    omega = relative.as_rotvec() / np.diff(seconds)[:, None]
    omega = np.vstack([omega, omega[-1:]])
    twist = np.column_stack([velocity_body[:, 0], -omega[:, 2]])  # z down -> z up
    flip = Rotation.from_matrix(BODY_TO_BASE[:3, :3])
    pose_rotation = flip * rotations * flip  # base_link in a z-up odom frame
    position = rows[:, 1:4] * np.array([1.0, -1.0, -1.0])
    return utime, position, pose_rotation.as_quat(), twist


def read_odometry(sensor_archive: Path, session: str) -> FloatArray:
    with tarfile.open(sensor_archive, "r:gz") as archive:
        member = archive.getmember(f"{session}/odometry_mu_100hz.csv")
        handle = archive.extractfile(member)
        assert handle is not None
        rows = np.loadtxt(io.TextIOWrapper(handle), delimiter=",", dtype=np.float64)
    return rows[np.argsort(rows[:, 0])]


# ----------------------------------------------------------------------------- lidar


def packets(stream: IO[bytes]) -> Iterator[tuple[int, NDArray[np.uint16]]]:
    """``(utime, hits)`` of each ``velodyne_hits.bin`` packet; hits as ``(n, 4)`` uint16."""

    while True:
        header = stream.read(24)
        if len(header) < 24:
            return
        if header[:8] != MAGIC:
            raise ValueError("velodyne_hits.bin: packet magic not found")
        count = struct.unpack_from("<I", header, 8)[0]
        utime = struct.unpack_from("<Q", header, 12)[0]
        raw = stream.read(8 * count)
        if len(raw) < 8 * count:
            return
        hits = np.frombuffer(raw, dtype="<u2").reshape(count, 4)
        yield utime, hits


def hits_to_points(hits: NDArray[np.uint16]) -> FloatArray:
    """Sensor-frame xyz in metres; NCLT writes out-of-range hits as (0, 0, 0)."""

    xyz = hits[:, :3].astype(np.float64) * 0.005 - 100.0
    keep = np.any(hits[:, :3] != 20000, axis=1)  # 20000 encodes 0 m
    return xyz[keep]


def revolutions(
    items: Iterable[tuple[int, FloatArray]],
) -> Iterator[list[tuple[int, FloatArray]]]:
    """Group ``(utime, points)`` packets into whole turns of the head, as a Velodyne driver does.

    A turn ends once the head's azimuth (the circular mean over a packet's points) has swept
    360 deg since the turn began; the last, partial turn is dropped. NCLT's HDL-32E turns at
    about 7.4 Hz, not 10 Hz: scans of a fixed 100 ms would each hold three quarters of a turn,
    and a scan-to-scan match would see the same direction one turn (about 135 ms) apart,
    not 100 ms, overstating the LiDAR's speed by about a third.
    """

    current: list[tuple[int, FloatArray]] = []
    swept = 0.0
    previous: float | None = None
    for utime, points in items:
        if len(points):
            angles = np.arctan2(points[:, 1], points[:, 0])
            azimuth = math.atan2(float(np.sum(np.sin(angles))), float(np.sum(np.cos(angles))))
            if previous is not None:
                swept += (azimuth - previous + math.pi) % (2.0 * math.pi) - math.pi
            previous = azimuth
            if abs(swept) >= 2.0 * math.pi:
                yield current
                current = []
                swept -= math.copysign(2.0 * math.pi, swept)
        current.append((utime, points))


def velodyne_stream(session: str) -> tuple[subprocess.Popen[bytes], tarfile.TarFile]:
    url = f"{S3}/velodyne_data/{session}_vel.tar.gz"
    process = subprocess.Popen(["curl", "-sSf", url], stdout=subprocess.PIPE)
    assert process.stdout is not None
    return process, tarfile.open(fileobj=process.stdout, mode="r|gz")


def convert(
    session: str,
    output: Path,
    work: Path,
    *,
    start_s: float,
    duration_s: float,
    decimate: int,
) -> dict[str, object]:
    work.mkdir(parents=True, exist_ok=True)
    sensor = work / f"{session}_sen.tar.gz"
    if not sensor.exists():
        subprocess.run(
            ["curl", "-sSf", "-o", str(sensor), f"{S3}/sensor_data/{session}_sen.tar.gz"],
            check=True,
        )
    odometry = read_odometry(sensor, session)
    utime, position, quaternion, twist = odometry_twist(odometry)
    reference = reference_base_velodyne()
    from scipy.spatial.transform import Rotation

    tf = encode_tf_message(
        [
            (
                "base_link",
                "velodyne",
                tuple(float(v) for v in reference[:3, 3]),
                tuple(float(v) for v in Rotation.from_matrix(reference[:3, :3]).as_quat()),
            )
        ]  # type: ignore[list-item]
    )
    process, archive = velodyne_stream(session)
    scans = points_written = 0
    first: int | None = None
    window: tuple[int, int] | None = None
    try:
        with Rosbag2Writer(output) as writer:
            writer.add_topic("/tf_static", "tf2_msgs/msg/TFMessage", latched=True)
            writer.add_topic(LIDAR_TOPIC, "sensor_msgs/msg/PointCloud2")
            writer.add_topic(WHEEL_TOPIC, "nav_msgs/msg/Odometry")
            for member in archive:
                if not member.name.endswith("velodyne_hits.bin"):
                    continue
                stream = archive.extractfile(member)
                assert stream is not None

                def inside_window(stream: IO[bytes]) -> Iterator[tuple[int, FloatArray]]:
                    nonlocal first, window
                    for packet_utime, hits in packets(stream):
                        if first is None:
                            first = packet_utime
                            lo = first + round(start_s * 1e6)
                            window = (lo, lo + round(duration_s * 1e6))
                            writer.write("/tf_static", window[0] * 1000, tf)
                        assert window is not None
                        if packet_utime >= window[1]:
                            return
                        if packet_utime >= window[0]:
                            yield packet_utime, hits_to_points(hits)

                for turn in revolutions(inside_window(stream)):
                    points_written += _write_scan(writer, turn[0][0], turn, decimate)
                    scans += 1
                break
            assert window is not None
            inside = (utime >= window[0]) & (utime < window[1])
            for k in np.flatnonzero(inside):
                stamp = int(utime[k]) * 1000
                writer.write(
                    WHEEL_TOPIC,
                    stamp,
                    encode_odometry(
                        frame_id="odom",
                        child_frame_id="base_link",
                        timestamp_ns=stamp,
                        position=tuple(float(v) for v in position[k]),  # type: ignore[arg-type]
                        orientation_xyzw=tuple(float(v) for v in quaternion[k]),  # type: ignore[arg-type]
                        linear_velocity=(float(twist[k, 0]), 0.0, 0.0),
                        angular_velocity=(0.0, 0.0, float(twist[k, 1])),
                    ),
                )
    finally:
        process.kill()
        process.wait()
    return {
        "session": session,
        "start_s": start_s,
        "duration_s": duration_s,
        "decimate": decimate,
        "scans": scans,
        "points": points_written,
        "wheel_messages": int(np.count_nonzero(inside)),
        "reference_base_link_velodyne": reference.tolist(),
        "x_body_vel": list(X_BODY_VEL),
    }


def _write_scan(
    writer: Rosbag2Writer,
    scan_start: int,
    packet_points: list[tuple[int, FloatArray]],
    decimate: int,
) -> int:
    xyz = np.concatenate([points for _t, points in packet_points])
    offsets = np.concatenate(
        [np.full(len(points), (t - scan_start) * 1e-6) for t, points in packet_points]
    )
    xyz, offsets = xyz[::decimate], offsets[::decimate]
    packed = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("t", "<f4")])
    packed["x"], packed["y"], packed["z"], packed["t"] = xyz[:, 0], xyz[:, 1], xyz[:, 2], offsets
    stamp = scan_start * 1000
    writer.write(
        LIDAR_TOPIC,
        stamp,
        encode_pointcloud2(
            frame_id="velodyne",
            timestamp_ns=stamp,
            fields=FIELDS,
            point_step=16,
            data=packed.tobytes(),
        ),
    )
    return len(xyz)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", required=True, help="e.g. 2012-01-08")
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--decimate", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = convert(
        args.session,
        args.output,
        args.work_dir,
        start_s=args.start_s,
        duration_s=args.duration_s,
        decimate=max(1, args.decimate),
    )
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
