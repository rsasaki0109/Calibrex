"""Optional exact-order CPU kernel for continuous-time pose/world coordinates."""

from __future__ import annotations

import math
import sys
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib.metadata import version
from itertools import pairwise
from typing import TYPE_CHECKING, TypeAlias, cast

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads
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
    return f"calibrex.numba_continuous_time/v0.5;numba={version('numba')};fastmath=false;cache=true"


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


@njit(cache=True, nogil=True, fastmath=False)  # type: ignore[untyped-decorator]
def _world_rows(poses: FloatArray, extrinsic: FloatArray, points: FloatArray) -> FloatArray:
    world = np.empty((len(poses), 3), dtype=np.float64)
    et = (extrinsic[0], extrinsic[1], extrinsic[2])
    eq = (extrinsic[3], extrinsic[4], extrinsic[5], extrinsic[6])
    for row in range(len(poses)):
        t = (poses[row, 0], poses[row, 1], poses[row, 2])
        q = (poses[row, 3], poses[row, 4], poses[row, 5], poses[row, 6])
        wt, wq = _composed(t, q, et, eq)
        p = _rotated(wq, (points[row, 0], points[row, 1], points[row, 2]))
        world[row] = (p[0] + wt[0], p[1] + wt[1], p[2] + wt[2])
    return world


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


@njit(cache=True, nogil=True, fastmath=False)  # type: ignore[untyped-decorator]
def _jacobian_rows(
    points: FloatArray,
    planes: FloatArray,
    normals: FloatArray,
    poses: FloatArray,
    weights: FloatArray,
    transforms: FloatArray,
    steps: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    residuals = _residual_rows(points, planes, normals, poses, weights, transforms[0])
    jacobian = np.empty((len(points), 6), dtype=np.float64)
    for column in range(6):
        plus = _residual_rows(points, planes, normals, poses, weights, transforms[1 + 2 * column])
        minus = _residual_rows(points, planes, normals, poses, weights, transforms[2 + 2 * column])
        for row in range(len(points)):
            jacobian[row, column] = (plus[row] - minus[row]) / (2.0 * steps[column])
    return residuals, jacobian


@njit(cache=True, nogil=True, fastmath=False)  # type: ignore[untyped-decorator]
def _normal_rows(
    rows: FloatArray,
    residuals: FloatArray,
    huber: bool,
    delta: float,
) -> tuple[FloatArray, FloatArray]:
    hessian = np.zeros((6, 6), dtype=np.float64)
    gradient = np.zeros(6, dtype=np.float64)
    for index in range(len(rows)):
        absolute = abs(residuals[index])
        weight = delta / absolute if huber and absolute > delta and absolute != 0.0 else 1.0
        for left in range(6):
            weighted = weight * rows[index, left]
            gradient[left] += weighted * residuals[index]
            for right in range(6):
                hessian[left, right] += weighted * rows[index, right]
    return hessian, gradient


@njit(cache=True, nogil=True, fastmath=False, inline="never")  # type: ignore[untyped-decorator]
def _one_residual(
    point: FloatArray, plane: FloatArray, normal: FloatArray, pose: FloatArray,
    weight: float, transform: FloatArray,
) -> float:
    wt, wq = _composed(
        (pose[0], pose[1], pose[2]), (pose[3], pose[4], pose[5], pose[6]),
        (transform[0], transform[1], transform[2]),
        (transform[3], transform[4], transform[5], transform[6]),
    )
    rotated = _rotated(wq, (point[0], point[1], point[2]))
    dx, dy, dz = (rotated[0]+wt[0]-plane[0], rotated[1]+wt[1]-plane[1],
                  rotated[2]+wt[2]-plane[2])
    return float(weight * (normal[0]*dx + normal[1]*dy + normal[2]*dz))


@njit(cache=True, nogil=True, fastmath=False, parallel=True)  # type: ignore[untyped-decorator]
def _parallel_jacobian_rows(
    points: FloatArray, planes: FloatArray, normals: FloatArray, poses: FloatArray,
    weights: FloatArray, transforms: FloatArray, steps: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    residuals = np.empty(len(points), dtype=np.float64)
    jacobian = np.empty((len(points), 6), dtype=np.float64)
    for row in prange(len(points)):
        point, plane, normal, pose, weight = (
            points[row], planes[row], normals[row], poses[row], weights[row],
        )
        residuals[row] = _one_residual(point, plane, normal, pose, weight, transforms[0])
        for column in range(6):
            plus = _one_residual(point, plane, normal, pose, weight, transforms[1+2*column])
            minus = _one_residual(point, plane, normal, pose, weight, transforms[2+2*column])
            jacobian[row, column] = (plus-minus)/(2.0*steps[column])
    return residuals, jacobian


def numba_rig_linearization(
    arrays: _ObservationArrays,
    transforms: Sequence[SE3],
    steps: Sequence[float],
) -> tuple[FloatArray, FloatArray]:
    """Keep all thirteen finite-difference evaluations in owned numeric arrays."""
    values = np.asarray(
        [(*t.translation_m, *t.rotation_quat_xyzw) for t in transforms], dtype=np.float64
    )
    arguments = (arrays.points, arrays.plane_points, arrays.normals, arrays.poses,
                 arrays.sqrt_weights, values, np.asarray(steps, dtype=np.float64))
    if len(arrays.points) >= 4096:
        if any(step == 0.0 for step in steps):
            raise ZeroDivisionError("division by zero")
        previous = get_num_threads()
        try:
            # The thread mask is local to this calling thread. Bound work and
            # restore its caller's mask even when compilation/evaluation fails.
            set_num_threads(min(previous, 4))
            return cast(tuple[FloatArray, FloatArray], _parallel_jacobian_rows(*arguments))
        finally:
            set_num_threads(previous)
    return cast(
        tuple[FloatArray, FloatArray],
        _jacobian_rows(*arguments),
    )


def numba_rig_normal_equations(
    rows: FloatArray,
    residuals: FloatArray,
    huber: bool,
    delta: float,
) -> tuple[list[list[float]], list[float]]:
    """Accumulate every row/product in the original robust weighted order."""
    hessian, gradient = _normal_rows(rows, residuals, huber, delta)
    return hessian.tolist(), gradient.tolist()


_PoseKey: TypeAlias = tuple[tuple[int, ...], bytes]
_NATIVE_RECORD_EXTRAPOLATION = OdometryTrack._record_extrapolation


@dataclass(frozen=True)
class _PreparedPoseBatch:
    poses: tuple[SE3, ...]
    rows: FloatArray
    clamps: int
    max_extrapolation_s: float
    size_bytes: int


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
    _pose_cache_limit_bytes: int = 128 * 1024 * 1024
    _pose_cache_bytes: int = 0
    _pose_cache: OrderedDict[_PoseKey, _PreparedPoseBatch] = field(default_factory=OrderedDict)

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
        result = self._evaluate(track, timestamps, points, base, extrinsic, tolerance, False)
        return None if result is None else (result[0], result[2], result[3])

    def _evaluate_rows(
        self, track: OdometryTrack, timestamps: list[int], points: FloatArray,
        base: SE3, extrinsic: SE3, tolerance: float,
    ) -> tuple[FloatArray, FloatArray, bool] | None:
        result = self._evaluate(track, timestamps, points, base, extrinsic, tolerance, True)
        return None if result is None else (result[1], result[2], result[3])

    def _evaluate(
        self, track: OdometryTrack, timestamps: list[int], points: list[Vector3] | FloatArray,
        base: SE3, extrinsic: SE3, tolerance: float, rows_only: bool,
    ) -> tuple[list[SE3], FloatArray, FloatArray, bool] | None:
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
        key = (tuple(timestamps), base_values.tobytes())
        can_reuse = (
            self._pose_cache_limit_bytes > 0
            and "_record_extrapolation" not in track.__dict__
            and OdometryTrack._record_extrapolation is _NATIVE_RECORD_EXTRAPOLATION
        )
        with track._cache_lock:
            cached = self._pose_cache.get(key) if can_reuse else None
            if cached is not None and cached.max_extrapolation_s <= tolerance:
                self._pose_cache.move_to_end(key)
                track.interpolation_count += len(timestamps)
                track.clamp_count += cached.clamps
                track._record_extrapolation(cached.max_extrapolation_s)
            else:
                cached = None
        if cached is not None:
            # Only trajectory/base geometry is retained. Changed points and
            # extrinsics always get the same ordered world-coordinate formula.
            world = _world_rows(cached.rows, transform, coordinates)
            self.used = True
            poses = [] if rows_only else (
                list(cached.poses) if cached.poses else _materialize_pose_rows(cached.rows)
            )
            return poses, cached.rows, world, True
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
            if ("_record_extrapolation" not in track.__dict__
                and OdometryTrack._record_extrapolation is _NATIVE_RECORD_EXTRAPOLATION):
                if visited:
                    track._record_extrapolation(float(np.max(extrapolation[:visited])))
            else:
                for value in extrapolation[:visited]:
                    track._record_extrapolation(float(value))
        usable = visited == len(timestamps) and not np.any(extrapolation[:visited] > tolerance)
        # These rows have already performed both public-constructor normalizations.
        poses = []
        if usable:
            if not rows_only:
                poses = _materialize_pose_rows(rows)
            if can_reuse and len(rows):
                pose_tuple = tuple(poses)
                # Charge all retained payloads, even shared Python floats/ints,
                # plus a conservative allowance for the LRU node and entry.
                size = sys.getsizeof(rows) + sys.getsizeof(pose_tuple) + sys.getsizeof(key)
                size += sys.getsizeof(key[0]) + sys.getsizeof(key[1]) + 512
                size += len(key[0]) * sys.getsizeof(2**63 - 1)
                if poses:
                    prototype = poses[0]
                    pose_size = (
                        sys.getsizeof(prototype)
                        + max(sys.getsizeof(pose.__dict__) for pose in poses)
                        + sys.getsizeof(prototype.translation_m)
                        + sys.getsizeof(prototype.rotation_quat_xyzw) + 7 * sys.getsizeof(0.0)
                    )
                    size += len(poses) * pose_size
                if size <= self._pose_cache_limit_bytes:
                    rows.flags.writeable = False
                    batch = _PreparedPoseBatch(
                        pose_tuple, rows, int(np.count_nonzero(clamps)),
                        float(np.max(extrapolation)), size,
                    )
                    with track._cache_lock:
                        previous = self._pose_cache.pop(key, None)
                        if previous is not None:
                            self._pose_cache_bytes -= previous.size_bytes
                        while self._pose_cache and self._pose_cache_bytes + size > (
                            self._pose_cache_limit_bytes
                        ):
                            evicted = self._pose_cache.popitem(last=False)[1]
                            self._pose_cache_bytes -= evicted.size_bytes
                        self._pose_cache[key] = batch
                        self._pose_cache_bytes += size
        rows.flags.writeable = False
        return poses, rows, world, usable


def _materialize_pose_rows(rows: FloatArray) -> list[SE3]:
    poses = []
    for start in range(0, len(rows), 4096):
        for row in rows[start : start + 4096].tolist():
            pose = object.__new__(SE3)
            object.__setattr__(pose, "translation_m", tuple(row[:3]))
            object.__setattr__(pose, "rotation_quat_xyzw", tuple(row[3:]))
            poses.append(pose)
    return poses
