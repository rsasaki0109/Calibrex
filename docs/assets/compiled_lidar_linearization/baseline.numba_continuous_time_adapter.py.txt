"""Optional exact-order CPU kernel for continuous-time pose/world coordinates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib.metadata import version
from itertools import pairwise
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numba import njit
from numpy.typing import NDArray

from calibrex.core.geometry import SE3, Vector3
from calibrex.data.odometry_track import OdometryTrack, _PreparedInterval

if TYPE_CHECKING:
    from calibrex.graph.lidar_point_to_plane import _ObservationArrays

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
Quaternion: TypeAlias = tuple[float, float, float, float]


def numba_continuous_time_identity() -> str:
    """Identify the optional runtime and its arithmetic/compilation policy."""
    return f"calibrex.numba_continuous_time/v0.1;numba={version('numba')};fastmath=false;cache=true"


@njit(cache=True, nogil=True, fastmath=False, inline="always")  # type: ignore[untyped-decorator]
def _normalized(q: Quaternion) -> Quaternion:
    x, y, z, w = q
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    return x / norm, y / norm, z / norm, w / norm


@njit(cache=True, nogil=True, fastmath=False, inline="always")  # type: ignore[untyped-decorator]
def _product(a: Quaternion, b: Quaternion) -> Quaternion:
    x, y, z, w = a
    rx, ry, rz, rw = b
    return (
        w * rx + x * rw + y * rz - z * ry,
        w * ry - x * rz + y * rw + z * rx,
        w * rz + x * ry - y * rx + z * rw,
        w * rw - x * rx - y * ry - z * rz,
    )


@njit(cache=True, nogil=True, fastmath=False, inline="always")  # type: ignore[untyped-decorator]
def _rotated(q: Quaternion, p: Vector3) -> Vector3:
    q = _normalized(q)
    x, y, z, w = _normalized(q)
    result = _product(_product(q, (p[0], p[1], p[2], 0.0)), (-x, -y, -z, w))
    return result[0], result[1], result[2]


@njit(cache=True, nogil=True, fastmath=False, inline="always")  # type: ignore[untyped-decorator]
def _composed(
    t: Vector3,
    q: Quaternion,
    rt: Vector3,
    rq: Quaternion,
) -> tuple[Vector3, Quaternion]:
    translated = _rotated(q, rt)
    return (
        (t[0] + translated[0], t[1] + translated[1], t[2] + translated[2]),
        _normalized(_normalized(_product(q, rq))),
    )


@njit(cache=True, nogil=True, fastmath=False)  # type: ignore[untyped-decorator]
def _pose_world_rows(
    queries: IntArray,
    times: IntArray,
    samples: FloatArray,
    q0: FloatArray,
    q1: FloatArray,
    theta: FloatArray,
    sine: FloatArray,
    base: FloatArray,
    extrinsic: FloatArray,
    points: FloatArray,
    tolerance: float,
) -> tuple[FloatArray, FloatArray, NDArray[np.bool_], FloatArray, int, int]:
    rows = np.empty((len(queries), 7), dtype=np.float64)
    world = np.empty((len(queries), 3), dtype=np.float64)
    clamps = np.zeros(len(queries), dtype=np.bool_)
    extrapolation = np.zeros(len(queries), dtype=np.float64)
    calculated = 0
    bt = (base[0], base[1], base[2])
    bq = (base[3], base[4], base[5], base[6])
    et = (extrinsic[0], extrinsic[1], extrinsic[2])
    eq = (extrinsic[3], extrinsic[4], extrinsic[5], extrinsic[6])
    for row in range(len(queries)):
        timestamp = queries[row]
        if timestamp <= times[0] or timestamp >= times[-1]:
            index = 0 if timestamp <= times[0] else len(times) - 1
            clamps[row] = timestamp < times[0] or timestamp > times[-1]
            extrapolation[row] = max(
                0.0, float(times[0] - timestamp) / 1e9, float(timestamp - times[-1]) / 1e9
            )
            t = (samples[index, 0], samples[index, 1], samples[index, 2])
            q = (samples[index, 3], samples[index, 4], samples[index, 5], samples[index, 6])
        else:
            lo, hi = 0, len(times)
            while lo < hi:
                mid = (lo + hi) // 2
                if timestamp < times[mid]:
                    hi = mid
                else:
                    lo = mid + 1
            left, right = lo - 1, lo
            alpha = float(timestamp - times[left]) / float(times[right] - times[left])
            t = (
                samples[left, 0] + (samples[right, 0] - samples[left, 0]) * alpha,
                samples[left, 1] + (samples[right, 1] - samples[left, 1]) * alpha,
                samples[left, 2] + (samples[right, 2] - samples[left, 2]) * alpha,
            )
            if theta[left] < 0.0:
                q = _normalized(
                    (
                        q0[left, 0] + alpha * (q1[left, 0] - q0[left, 0]),
                        q0[left, 1] + alpha * (q1[left, 1] - q0[left, 1]),
                        q0[left, 2] + alpha * (q1[left, 2] - q0[left, 2]),
                        q0[left, 3] + alpha * (q1[left, 3] - q0[left, 3]),
                    )
                )
            else:
                wl = math.sin((1.0 - alpha) * theta[left]) / sine[left]
                wr = math.sin(alpha * theta[left]) / sine[left]
                q = _normalized(
                    (
                        wl * q0[left, 0] + wr * q1[left, 0],
                        wl * q0[left, 1] + wr * q1[left, 1],
                        wl * q0[left, 2] + wr * q1[left, 2],
                        wl * q0[left, 3] + wr * q1[left, 3],
                    )
                )
            q = _normalized(q)
            calculated += 1
        if extrapolation[row] > tolerance:
            return rows, world, clamps, extrapolation, calculated, row + 1
        t, q = _composed(t, q, bt, bq)
        rows[row] = (t[0], t[1], t[2], q[0], q[1], q[2], q[3])
        wt, wq = _composed(t, q, et, eq)
        p = _rotated(wq, (points[row, 0], points[row, 1], points[row, 2]))
        world[row] = (p[0] + wt[0], p[1] + wt[1], p[2] + wt[2])
    return rows, world, clamps, extrapolation, calculated, len(queries)


@njit(cache=True, nogil=True, fastmath=False)  # type: ignore[untyped-decorator]
def _residual_rows(
    points: FloatArray,
    planes: FloatArray,
    normals: FloatArray,
    poses: FloatArray,
    weights: FloatArray,
    transform: FloatArray,
) -> FloatArray:
    result = np.empty(len(points), dtype=np.float64)
    t = (transform[0], transform[1], transform[2])
    q = (transform[3], transform[4], transform[5], transform[6])
    for row in range(len(points)):
        pt = (poses[row, 0], poses[row, 1], poses[row, 2])
        pq = (poses[row, 3], poses[row, 4], poses[row, 5], poses[row, 6])
        wt, wq = _composed(pt, pq, t, q)
        rotated = _rotated(wq, (points[row, 0], points[row, 1], points[row, 2]))
        dx = rotated[0] + wt[0] - planes[row, 0]
        dy = rotated[1] + wt[1] - planes[row, 1]
        dz = rotated[2] + wt[2] - planes[row, 2]
        result[row] = weights[row] * (
            normals[row, 0] * dx + normals[row, 1] * dy + normals[row, 2] * dz
        )
    return result


def numba_rig_residuals(arrays: _ObservationArrays, transform: SE3) -> list[float]:
    """Fuse the existing ordered residual formula without changing differences."""
    values = np.asarray((*transform.translation_m, *transform.rotation_quat_xyzw), dtype=np.float64)
    return list(
        _residual_rows(
            arrays.points,
            arrays.plane_points,
            arrays.normals,
            arrays.poses,
            arrays.sqrt_weights,
            values,
        ).tolist()
    )


@dataclass
class NumbaContinuousTimeQueryContext:
    """Capture a supported immutable trajectory once per solve."""

    times: IntArray
    samples: FloatArray
    q0: FloatArray
    q1: FloatArray
    theta: FloatArray
    sine: FloatArray
    sample_snapshot: object
    timestamp_snapshot: object
    used: bool = False

    @classmethod
    def capture(cls, track: OdometryTrack) -> NumbaContinuousTimeQueryContext | None:
        """Return None for trajectories requiring the original scalar semantics."""
        if (
            type(track) is not OdometryTrack
            or "interpolate" in track.__dict__
            or len(track.samples) < 2
        ):
            return None
        if any(
            type(sample.pose) is not SE3 or type(sample.timestamp_ns) is not int
            for sample in track.samples
        ):
            return None
        raw_times = track._timestamps_ns
        if (
            raw_times != tuple(sample.timestamp_ns for sample in track.samples)
            or any(abs(timestamp) >= 2**63 for timestamp in raw_times)
            or raw_times[-1] - raw_times[0] > 2**53
            or any(a >= b for a, b in pairwise(raw_times))
        ):
            return None
        times = np.asarray(raw_times, dtype=np.int64)
        samples = np.asarray(
            [
                (*sample.pose.translation_m, *sample.pose.rotation_quat_xyzw)
                for sample in track.samples
            ],
            dtype=np.float64,
        )
        if not np.isfinite(samples).all():
            return None
        prepared = [_PreparedInterval.capture(a.pose, b.pose) for a, b in pairwise(track.samples)]
        q0 = np.asarray([interval.q0 for interval in prepared], dtype=np.float64)
        q1 = np.asarray([interval.q1 for interval in prepared], dtype=np.float64)
        theta = np.asarray(
            [-1.0 if interval.theta is None else interval.theta for interval in prepared],
            dtype=np.float64,
        )
        sine = np.asarray([interval.sin_theta for interval in prepared], dtype=np.float64)
        for array in (times, samples, q0, q1, theta, sine):
            array.flags.writeable = False
        return cls(times, samples, q0, q1, theta, sine, track.samples, track._timestamps_ns)

    def evaluate(
        self,
        track: OdometryTrack,
        timestamps: list[int],
        points: list[Vector3],
        base: SE3,
        extrinsic: SE3,
        tolerance: float,
    ) -> tuple[list[SE3], FloatArray, bool] | None:
        """Preserve scalar query counters and stop at the first extrapolation failure."""
        if (
            track.samples is not self.sample_snapshot
            or track._timestamps_ns is not self.timestamp_snapshot
            or "interpolate" in track.__dict__
            or type(base) is not SE3
            or type(extrinsic) is not SE3
            or any(
                type(timestamp) is not int
                or abs(timestamp) >= 2**63
                or abs(timestamp - int(self.times[0])) > 2**53
                for timestamp in timestamps
            )
        ):
            return None
        base_values = np.asarray((*base.translation_m, *base.rotation_quat_xyzw), dtype=np.float64)
        transform = np.asarray((*extrinsic.translation_m, *extrinsic.rotation_quat_xyzw))
        coordinates = np.asarray(points, dtype=np.float64).reshape((-1, 3))
        if not all(np.isfinite(array).all() for array in (base_values, transform, coordinates)):
            return None
        rows, world, clamps, extrapolation, _calculated, visited = _pose_world_rows(
            np.asarray(timestamps, dtype=np.int64),
            self.times,
            self.samples,
            self.q0,
            self.q1,
            self.theta,
            self.sine,
            base_values,
            transform,
            coordinates,
            tolerance,
        )
        self.used = True
        with track._cache_lock:
            track.interpolation_count += visited
            track.clamp_count += int(np.count_nonzero(clamps[:visited]))
            for value in extrapolation[:visited]:
                track._record_extrapolation(float(value))
        usable = visited == len(timestamps) and not np.any(extrapolation[:visited] > tolerance)
        # These rows have already performed both public-constructor normalizations.
        poses = []
        if usable:
            for row in rows:
                pose = object.__new__(SE3)
                object.__setattr__(pose, "translation_m", tuple(row[:3].tolist()))
                object.__setattr__(pose, "rotation_quat_xyzw", tuple(row[3:].tolist()))
                poses.append(pose)
        return poses, world, usable
