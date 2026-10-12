"""Optional exact compiled LiDAR deskew on captured native odometry."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numba import njit
from numpy.typing import NDArray

from calibrex.core.geometry import SE3
from calibrex.core.time import apply_time_offset_ns
from calibrex.data.odometry_track import OdometryTrack
from calibrex.solvers import numba_continuous_time_adapter as a

FloatArray: TypeAlias = NDArray[np.float64]
_MAX_DESKEW_ROW_BYTES = 64 * 1024**2
_MIN_DESKEW_ROWS = 4096


@njit(cache=True, nogil=True, fastmath=False)  # type: ignore[untyped-decorator]
def _direct_world_rows(poses: FloatArray, points: FloatArray) -> FloatArray:
    world = np.empty((len(points), 3), dtype=np.float64)
    for row in range(len(points)):
        q = (poses[row, 3], poses[row, 4], poses[row, 5], poses[row, 6])
        p = a._rotated(q, (points[row, 0], points[row, 1], points[row, 2]))
        world[row] = (p[0]+poses[row, 0], p[1]+poses[row, 1], p[2]+poses[row, 2])
    return world


@dataclass(frozen=True)
class _DeskewRows:
    points: FloatArray
    poses: FloatArray
    timestamps: list[int]
    extrapolation_s: FloatArray
    usable: bool


class _PreparedDeskew:
    def __init__(self, context: a.NumbaContinuousTimeQueryContext) -> None:
        self.context = context
        # Replay visits each capture once. Do not retain its row payloads.
        context._pose_cache_limit_bytes = 0
        self.rows = self.calls = 0

    @classmethod
    def capture(cls, track: OdometryTrack) -> _PreparedDeskew | None:
        context = a.NumbaContinuousTimeQueryContext.capture(track)
        return None if context is None else cls(context)

    def batch(
        self, xyz: Any, offsets: Any, reference_timestamp_ns: int,
        selection: range, track: OdometryTrack, base: SE3, *,
        injection_s: float = 0.0, tolerance_s: float = np.inf,
    ) -> _DeskewRows | None:
        if (len(selection) < _MIN_DESKEW_ROWS or len(selection)*192 > _MAX_DESKEW_ROW_BYTES
            or type(xyz) is not np.ndarray
            or type(offsets) is not np.ndarray or xyz.ndim != 2 or xyz.shape[1] < 3
            or offsets.ndim != 1 or len(offsets) != len(xyz)
            or xyz.dtype not in (np.float32,np.float64)
            or offsets.dtype not in (np.float32,np.float64)
            or type(reference_timestamp_ns) is not int or not np.isfinite(injection_s)):
            return None
        section = slice(selection.start, selection.stop, selection.step)
        points = np.array(xyz[section,:3],dtype=np.float64,order='C',copy=True)
        times = offsets[section]
        if (not np.isfinite(points).all() or not np.isfinite(times).all()
            or np.any(np.abs(points) >= 1e100)):
            return None
        timestamps = [reference_timestamp_ns+round(float(t)*1_000_000_000) for t in times]
        if abs(injection_s) >= 1e-12:
            timestamps = [apply_time_offset_ns(t,injection_s) for t in timestamps]
        result = self.context._evaluate_rows(
            track,timestamps,points,base,SE3.identity(),tolerance_s,
        )
        if result is None:
            return None
        poses,_unused_world,usable = result
        queries = np.asarray(timestamps,dtype=np.int64)
        first,last = int(self.context.times[0]),int(self.context.times[-1])
        extrapolation = np.zeros(len(queries),dtype=np.float64)
        before,after = queries < first,queries > last
        extrapolation[before] = (first-queries[before]).astype(np.float64)/1_000_000_000
        extrapolation[after] = (queries[after]-last).astype(np.float64)/1_000_000_000
        self.calls += 1
        visited = len(queries) if usable else int(np.flatnonzero(extrapolation > tolerance_s)[0])+1
        self.rows += visited
        return _DeskewRows(points,poses,timestamps,extrapolation,usable)

    def identity(self) -> str:
        from importlib.metadata import version

        return (f'calibrex.numba_lidar_deskew/v0.1;numba={version("numba")};'
                'fastmath=false;cache=true;point_pose_cpu=true;'
                f'source_sha256={hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}')
