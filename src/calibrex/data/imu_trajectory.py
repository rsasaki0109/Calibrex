"""Plain-text IMU samples and sensor trajectories, for calibration without bags.

Two inputs, both as text so they can come from a file or a browser upload:

* an IMU CSV with columns ``t, gx, gy, gz, ax, ay, az`` (seconds, rad/s,
  m/s^2 or g); commas or whitespace separate the columns, and a non-numeric
  first line is treated as a header; and
* a trajectory of the sensor whose extrinsic is wanted (for example a LiDAR
  odometry or SLAM output) in TUM format, ``t x y z qx qy qz qw`` per line,
  ``#`` starting a comment.  Poses are ``T_world_sensor``.

Both must use the same clock up to the constant offset that is calibrated.
"""

from __future__ import annotations

import math
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from calibrex.data.livox_ros2 import STANDARD_GRAVITY_MPS2, ImuSamples
from calibrex.solvers.gnss_lever_arm_solver import OdometryWindow

FloatArray: TypeAlias = NDArray[np.float64]


def _numeric_rows(text: str, columns: int, what: str) -> FloatArray:
    rows: list[list[float]] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.replace(",", " ").split()
        try:
            values = [float(field) for field in fields]
        except ValueError:
            if not rows:
                continue  # header
            raise ValueError(f"{what} line {number}: non-numeric value") from None
        if len(values) != columns:
            raise ValueError(f"{what} line {number}: expected {columns} columns, got {len(values)}")
        rows.append(values)
    if len(rows) < 2:
        raise ValueError(f"{what} has fewer than two samples")
    array = np.asarray(rows, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{what} contains non-finite values")
    return array


def parse_imu_csv(text: str, *, acceleration_unit: Literal["mps2", "g"] = "mps2") -> ImuSamples:
    """Parse ``t, gx, gy, gz, ax, ay, az`` rows into time-sorted, de-duplicated samples."""

    rows = _numeric_rows(text, 7, "IMU CSV")
    rows = rows[np.argsort(rows[:, 0], kind="stable")]
    keep = np.r_[True, np.diff(rows[:, 0]) > 0.0]
    rows = rows[keep]
    accel = rows[:, 4:7] * (STANDARD_GRAVITY_MPS2 if acceleration_unit == "g" else 1.0)
    return ImuSamples(times_s=rows[:, 0], gyro_rps=rows[:, 1:4], accel_mps2=accel)


def parse_tum_trajectory(text: str) -> tuple[FloatArray, FloatArray]:
    """Parse a TUM trajectory into times and ``(N, 4, 4)`` poses ``T_world_sensor``."""

    rows = _numeric_rows(text, 8, "trajectory")
    rows = rows[np.argsort(rows[:, 0], kind="stable")]
    rows = rows[np.r_[True, np.diff(rows[:, 0]) > 0.0]]
    quaternions = rows[:, 4:8]
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms < 1e-9):
        raise ValueError("trajectory contains a zero quaternion")
    poses = np.repeat(np.eye(4)[None], len(rows), axis=0)
    poses[:, :3, :3] = Rotation.from_quat(quaternions / norms[:, None]).as_matrix()
    poses[:, :3, 3] = rows[:, 1:4]
    return rows[:, 0], poses


def split_trajectory_windows(
    times_s: FloatArray,
    poses: FloatArray,
    *,
    window_duration_s: float = 10.0,
    max_gap_s: float = 0.5,
    min_window_poses: int = 20,
    prefix: str = "trajectory/",
) -> list[OdometryWindow]:
    """Cut a trajectory into windows of ``window_duration_s`` without gaps.

    A gap longer than ``max_gap_s`` starts a new block, so no window spans a
    tracking loss; windows with fewer than ``min_window_poses`` poses are
    dropped.
    """

    windows: list[OdometryWindow] = []
    breaks = np.flatnonzero(np.diff(times_s) > max_gap_s) + 1
    for block, indices in enumerate(np.split(np.arange(len(times_s)), breaks)):
        if len(indices) == 0:
            continue
        start = times_s[indices[0]]
        slot = np.floor((times_s[indices] - start) / window_duration_s).astype(np.int64)
        for label in np.unique(slot):
            members = indices[slot == label]
            if len(members) >= min_window_poses:
                windows.append(
                    OdometryWindow(
                        f"{prefix}b{block}w{int(label)}",
                        block,
                        times_s[members],
                        poses[members],
                    )
                )
    return windows


def trajectory_rate_hz(times_s: FloatArray) -> float:
    """Median pose rate, for reporting."""

    steps = np.diff(times_s)
    return float(1.0 / np.median(steps)) if len(steps) else math.nan
