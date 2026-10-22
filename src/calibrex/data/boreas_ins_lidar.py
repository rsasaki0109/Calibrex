"""ROS-independent Boreas inputs for INS--LiDAR trajectory hand-eye calibration.

Use the Applanix-frame ``gps_post_process.csv``, never the sensor pose CSVs.
The reference extrinsic is read separately, for comparison after a fit only.
See the official pyboreas DATA_REFERENCE and utils.get_transform conventions:
``T_enu_applanix`` has rotation ``Rx(-roll) Ry(-pitch) Rz(-heading)``.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from calibrex.solvers.trajectory_hand_eye_solver import ReferenceTrajectory

FloatArray: TypeAlias = NDArray[np.float64]
_COLUMNS = ("GPSTime", "easting", "northing", "altitude", "roll", "pitch", "heading")


def read_boreas_ins_trajectory(
    path: str | Path,
    *,
    start_s: float | None = None,
    end_s: float | None = None,
    max_gap_s: float = 1.0,
) -> ReferenceTrajectory:
    """Read Applanix poses without loading or applying any reference extrinsic.

    ``GPSTime`` is in seconds, unlike the microseconds in scan filenames.
    Bounds select reference samples inclusively; callers must include an
    interpolation/clock-search margin around their sensor timestamps.
    """
    if not math.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("max_gap_s must be finite and positive")
    if any(value is not None and not math.isfinite(value) for value in (start_s, end_s)):
        raise ValueError("trajectory bounds must be finite")
    if start_s is not None and end_s is not None and start_s > end_s:
        raise ValueError("start_s must not exceed end_s")
    rows: list[list[float]] = []
    previous: float | None = None
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        if len(header) != len(set(header)) or not set(_COLUMNS).issubset(header):
            raise ValueError("expected Applanix gps_post_process.csv columns, not sensor poses")
        for line, row in enumerate(reader, start=2):
            try:
                values = [float(row[name]) for name in _COLUMNS]
            except (ValueError, TypeError) as error:
                raise ValueError(f"invalid Applanix row {line}") from error
            if None in row or any(not math.isfinite(value) for value in values):
                raise ValueError(f"non-finite or malformed Applanix row {line}")
            time_s = values[0]
            if previous is not None and time_s <= previous:
                raise ValueError("Applanix timestamps must be strictly increasing")
            previous = time_s
            if (start_s is None or time_s >= start_s) and (end_s is None or time_s <= end_s):
                rows.append(values)
    if len(rows) < 2:
        raise ValueError("at least two Applanix samples required inside the bounds")
    data = np.asarray(rows, dtype=np.float64)
    poses = np.tile(np.eye(4), (len(rows), 1, 1))
    poses[:, :3, 3] = data[:, 1:4]
    # SciPy's intrinsic XYZ implements the ordered passive rotations specified
    # by the dataset, including their signs; ordinary active xyz is different.
    poses[:, :3, :3] = Rotation.from_euler("XYZ", -data[:, 4:7]).as_matrix()
    return ReferenceTrajectory(data[:, 0], poses, max_gap_s=max_gap_s)


def boreas_scan_time_s(path: str | Path) -> float:
    """Convert an original Boreas scan's 16-digit UTC microseconds to seconds."""
    stem = Path(path).stem
    if re.fullmatch(r"\d{16}", stem) is None:
        raise ValueError("expected a 16-digit original Boreas scan timestamp")
    return int(stem) * 1.0e-6


def read_boreas_lidar_xyz(path: str | Path) -> FloatArray:
    """Read original Velodyne XYZ from float32 ``[x,y,z,intensity,ring,time]``.

    Per-point offsets are not used by the rigid-scan hand-eye baseline. Aeva
    ten-field files are a different format and must not enter this adapter.
    Range/nonfinite filtering remains the odometry solver's responsibility.
    """
    source = Path(path)
    if source.stat().st_size % 24:
        raise ValueError("Boreas Velodyne payload must contain six float32 fields per point")
    return np.fromfile(source, dtype="<f4").reshape(-1, 6)[:, :3].astype(np.float64)


def read_boreas_reference_extrinsic(path: str | Path) -> FloatArray:
    """Read ``T_applanix_lidar`` for post-fit reference comparison, not fitting."""
    matrix = np.loadtxt(path, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("reference extrinsic must be a finite 4x4 matrix")
    rotation = matrix[:3, :3]
    if (
        not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0)
    ):
        raise ValueError("reference extrinsic must be an SE(3) transform")
    return matrix
