"""Vectorized scan-to-scan point-to-plane LiDAR odometry.

Trajectory hand-eye calibration needs relative LiDAR motions that are
independent of the reference sensor.  This module estimates them from the
point clouds alone: each scan is voxel-downsampled, target normals come from a
local PCA over a KD-tree neighbourhood, and a robust (Huber) Gauss--Newton
solve minimizes point-to-plane distances.  The initial guess is a constant
velocity extrapolation of the previous LiDAR motion, never an INS/GNSS prior,
so the resulting motions cannot inherit the extrinsic under test.

Scans are treated as rigid snapshots; motion distortion within a sweep is not
compensated and is reported as a limitation by callers.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

FloatArray: TypeAlias = NDArray[np.float64]
RotationModel: TypeAlias = Callable[[float, FloatArray], FloatArray]


@dataclass(frozen=True)
class ScanOdometryOptions:
    """Tuning of the point-to-plane registration."""

    voxel_size_m: float = 0.5
    min_range_m: float = 3.0
    max_range_m: float = 80.0
    normal_neighbours: int = 10
    max_planarity_ratio: float = 0.2
    max_correspondence_m: float = 1.5
    huber_delta_m: float = 0.1
    max_iterations: int = 30
    convergence_translation_m: float = 1.0e-4
    convergence_rotation_rad: float = 1.0e-5
    min_correspondences: int = 200
    local_map_scans: int = 1
    deskew_iterations: int = 1


@dataclass(frozen=True)
class ScanRegistration:
    """Relative motion ``T_target_source`` and its registration quality."""

    transform: FloatArray
    converged: bool
    iterations: int
    correspondences: int
    rmse_m: float
    hessian_min_eigenvalue: float


@dataclass(frozen=True)
class ScanOdometryResult:
    """Poses ``T_first_scan`` for every scan and per-step registrations."""

    poses: tuple[FloatArray, ...]
    registrations: tuple[ScanRegistration, ...]

    @property
    def all_converged(self) -> bool:
        """Return whether every step converged with enough correspondences."""

        return all(item.converged for item in self.registrations)


def preprocess_scan(points: FloatArray, options: ScanOdometryOptions) -> FloatArray:
    """Crop by range and voxel-downsample an ``(N, 3+)`` point array."""

    xyz = np.asarray(points, dtype=np.float64)[:, :3]
    # Same sum order and rounding as ``np.linalg.norm(xyz, axis=1)``, about 5x faster.
    ranges = np.sqrt(xyz[:, 0] * xyz[:, 0] + xyz[:, 1] * xyz[:, 1] + xyz[:, 2] * xyz[:, 2])
    xyz = xyz[(ranges >= options.min_range_m) & (ranges <= options.max_range_m)]
    if xyz.size == 0:
        return xyz.reshape(0, 3)
    keys = np.floor(xyz / options.voxel_size_m).astype(np.int64)
    flat = _flatten_voxel_keys(keys)
    if flat is None:
        _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        sums = np.zeros((counts.size, 3))
        np.add.at(sums, inverse.reshape(-1), xyz)
        return sums / counts[:, None]
    # One int64 per voxel keeps the lexicographic voxel order of the row-wise
    # ``unique``, and ``bincount`` adds the points of a voxel in input order
    # exactly like ``add.at``, so the result is bit-identical but much faster.
    _, inverse, counts = np.unique(flat, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    sums = np.empty((counts.size, 3))
    for axis in range(3):
        sums[:, axis] = np.bincount(inverse, weights=xyz[:, axis], minlength=counts.size)
    return sums / counts[:, None]


def _flatten_voxel_keys(keys: NDArray[np.int64]) -> NDArray[np.int64] | None:
    """Pack ``(N, 3)`` voxel indices into one order-preserving int64 each, or ``None``."""

    # Per-column extrema: the same integers as ``keys.min(axis=0)`` / ``max(axis=0)``,
    # without the slow strided reduction.
    lows = np.array([keys[:, axis].min() for axis in range(3)])
    spans = np.array([keys[:, axis].max() for axis in range(3)]) - lows + 1
    if int(spans[0]) * int(spans[1]) * int(spans[2]) >= 2**62:
        return None
    shifted = keys - lows
    return np.asarray((shifted[:, 0] * spans[1] + shifted[:, 1]) * spans[2] + shifted[:, 2])


def estimate_normals(
    points: FloatArray,
    tree: cKDTree,
    options: ScanOdometryOptions,
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Return unit normals and a mask of locally planar points."""

    k = min(options.normal_neighbours, len(points))
    _, indices = tree.query(points, k=k)
    return _normals_from_neighbours(points, indices, k, options)


def _normals_from_neighbours(
    points: FloatArray,
    indices: NDArray[np.intp],
    k: int,
    options: ScanOdometryOptions,
) -> tuple[FloatArray, NDArray[np.bool_]]:
    neighbours = points[indices]
    # ``neighbours.mean(axis=1, keepdims=True)`` adds the k neighbours in order and divides
    # by k; the explicit loop does exactly that without the strided-reduction overhead.
    total = neighbours[:, 0].copy()
    for column in range(1, neighbours.shape[1]):
        total += neighbours[:, column]
    centered = neighbours - (total / neighbours.shape[1])[:, None, :]
    covariance = np.einsum("nki,nkj->nij", centered, centered) / max(k, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0]
    planar = eigenvalues[:, 0] <= options.max_planarity_ratio * np.maximum(eigenvalues[:, 1], 1e-12)
    return normals, planar


class TargetNormals:
    """Normals of a target cloud, computed only for the points a registration hits.

    Registration reads the normal and planarity of the nearest target point of
    each source point, which is a fraction of a (local-map) target.  Every
    normal is a function of that point's own neighbourhood, and the batched
    ``eigh`` and ``einsum`` treat each point independently, so a lazily
    computed normal is bit-identical to the one :func:`estimate_normals` returns
    for the whole cloud.
    """

    def __init__(self, points: FloatArray, tree: cKDTree, options: ScanOdometryOptions) -> None:
        self._points = points
        self._tree = tree
        self._options = options
        self._k = min(options.normal_neighbours, len(points))
        self.normals: FloatArray = np.zeros((len(points), 3))
        self.planar: NDArray[np.bool_] = np.zeros(len(points), dtype=bool)
        self._done: NDArray[np.bool_] = np.zeros(len(points), dtype=bool)

    @classmethod
    def precomputed(cls, normals: FloatArray, planar: NDArray[np.bool_]) -> TargetNormals:
        """Wrap normals that were computed for every point."""

        holder = cls.__new__(cls)
        holder.normals = normals
        holder.planar = planar
        holder._done = np.ones(len(planar), dtype=bool)
        return holder

    def ensure(self, indices: NDArray[np.intp]) -> None:
        """Compute the normals of the target points ``indices`` that are still missing."""

        missing = indices[~self._done[indices]]
        if missing.size == 0:
            return
        missing = np.unique(missing)
        _, neighbours = self._tree.query(self._points[missing], k=self._k)
        normals, planar = _normals_from_neighbours(self._points, neighbours, self._k, self._options)
        self.normals[missing] = normals
        self.planar[missing] = planar
        self._done[missing] = True


def register_point_to_plane(
    source: FloatArray,
    target: FloatArray,
    initial: FloatArray,
    options: ScanOdometryOptions | None = None,
    *,
    target_tree: cKDTree | None = None,
    target_normals: tuple[FloatArray, NDArray[np.bool_]] | TargetNormals | None = None,
) -> ScanRegistration:
    """Estimate ``T_target_source`` from preprocessed point arrays."""

    opts = options or ScanOdometryOptions()
    tree = target_tree if target_tree is not None else cKDTree(target)
    if target_normals is None:
        lazy = TargetNormals(target, tree, opts)
    elif isinstance(target_normals, TargetNormals):
        lazy = target_normals
    else:
        lazy = TargetNormals.precomputed(*target_normals)
    transform = np.array(initial, dtype=np.float64)
    converged = False
    count = 0
    rmse = math.inf
    last_hessian: FloatArray | None = None
    last_count = 0
    iteration = 0
    for iteration in range(1, opts.max_iterations + 1):  # noqa: B007
        moved = source @ transform[:3, :3].T + transform[:3, 3]
        distances, indices = tree.query(moved, k=1, distance_upper_bound=opts.max_correspondence_m)
        valid = np.isfinite(distances)
        hit = indices[valid]
        lazy.ensure(hit)
        valid[valid] &= lazy.planar[hit]
        count = int(valid.sum())
        if count < opts.min_correspondences:
            break
        matched = indices[valid]
        p = moved[valid]
        q = target[matched]
        n = lazy.normals[matched]
        residual = np.einsum("ij,ij->i", p - q, n)
        # Same products and differences as ``np.cross`` / ``np.hstack``, without
        # their temporaries.
        jacobian = np.empty((len(p), 6))
        np.multiply(p[:, 1], n[:, 2], out=jacobian[:, 0])
        jacobian[:, 0] -= p[:, 2] * n[:, 1]
        np.multiply(p[:, 2], n[:, 0], out=jacobian[:, 1])
        jacobian[:, 1] -= p[:, 0] * n[:, 2]
        np.multiply(p[:, 0], n[:, 1], out=jacobian[:, 2])
        jacobian[:, 2] -= p[:, 1] * n[:, 0]
        jacobian[:, 3:] = n
        weights = np.where(
            np.abs(residual) <= opts.huber_delta_m,
            1.0,
            opts.huber_delta_m / np.maximum(np.abs(residual), 1e-12),
        )
        weighted = jacobian * weights[:, None]
        hessian = weighted.T @ jacobian
        gradient = weighted.T @ residual
        rmse = float(math.sqrt(float(np.mean(residual * residual))))
        try:
            delta = -np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        # Only the last solved iteration's smallest eigenvalue is reported.
        last_hessian = hessian
        last_count = count
        transform = _exp_se3(delta) @ transform
        if _norm3(delta[3:]) < opts.convergence_translation_m and (
            _norm3(delta[:3]) < opts.convergence_rotation_rad
        ):
            converged = True
            break
    min_eigenvalue = (
        0.0 if last_hessian is None else float(np.linalg.eigvalsh(last_hessian)[0]) / last_count
    )
    return ScanRegistration(
        transform=transform,
        converged=converged and count >= opts.min_correspondences,
        iterations=iteration,
        correspondences=count,
        rmse_m=rmse,
        hessian_min_eigenvalue=min_eigenvalue,
    )


def run_scan_to_scan_odometry(
    scans: Sequence[FloatArray],
    options: ScanOdometryOptions | None = None,
    *,
    times_s: Sequence[float] | None = None,
) -> ScanOdometryResult:
    """Chain scan-to-scan registrations into poses ``T_first_scan``.

    Each step registers scan ``i`` onto scan ``i - 1`` starting from the
    previous step's motion (constant velocity), scaled by the ratio of the
    time gaps when ``times_s`` is given, so no external motion prior enters
    the odometry.
    """

    return odometry_from_loader(lambda index: scans[index], len(scans), options, times_s=times_s)


def odometry_from_loader(
    load_scan: Callable[[int], FloatArray],
    count: int,
    options: ScanOdometryOptions | None = None,
    *,
    times_s: Sequence[float] | None = None,
) -> ScanOdometryResult:
    """Run scan-to-map odometry over a sliding window of recent scans.

    With ``local_map_scans == 1`` each scan is registered to the previous scan
    only.  Larger values register it to the union of the last scans, which
    densifies sparse or non-repetitive scanners such as Livox units.
    """

    if times_s is not None and len(times_s) != count:
        raise ValueError("times_s must have one entry per scan")
    odometry = IncrementalScanOdometry(options)
    for index in range(count):
        odometry.add(load_scan(index), None if times_s is None else float(times_s[index]))
    return odometry.result()


class IncrementalScanOdometry:
    """Scan-to-map odometry that accepts one scan at a time.

    Streaming consumers (for example a multi-gigabyte bag read once) can start
    a new instance whenever the stream should be segmented.
    """

    def __init__(
        self,
        options: ScanOdometryOptions | None = None,
        *,
        rotation_model: RotationModel | None = None,
    ) -> None:
        """Create odometry; ``rotation_model`` supplies in-sweep rotations for deskewing.

        ``rotation_model(scan_time_s, offsets_s)`` returns the sensor rotation
        from the scan time to each point's capture time (``(N, 3, 3)``, LiDAR
        frame), typically from a gyro.  Without it, deskewing extrapolates the
        LiDAR's own previous motion at constant velocity.
        """

        self.rotation_model = rotation_model
        self.options = options or ScanOdometryOptions()
        if self.options.local_map_scans < 1:
            raise ValueError("local_map_scans must be at least 1")
        self.poses: list[FloatArray] = []
        self.registrations: list[ScanRegistration] = []
        self._recent: deque[FloatArray] = deque(maxlen=self.options.local_map_scans)
        self._previous_motion = np.eye(4)
        self._previous_time: float | None = None
        self._previous_gap: float | None = None

    def add(
        self,
        scan: FloatArray,
        time_s: float | None = None,
        point_offsets_s: FloatArray | None = None,
    ) -> ScanRegistration | None:
        """Register ``scan`` and return its registration (``None`` for the first).

        ``point_offsets_s`` are per-point capture times relative to ``time_s``.
        When given, the sweep is motion-compensated to ``time_s`` under a
        constant-velocity model, then re-registered ``deskew_iterations`` times
        with the refined motion.
        """

        if not self.poses:
            self.poses.append(np.eye(4))
            self._recent.append(preprocess_scan(scan, self.options))
            self._previous_time = time_s
            return None
        initial = self._previous_motion
        gap: float | None = None
        if time_s is not None and self._previous_time is not None:
            gap = time_s - self._previous_time
            if self._previous_gap is not None and self._previous_gap > 0.0 and gap > 0.0:
                initial = scale_motion(self._previous_motion, gap / self._previous_gap)
            self._previous_gap = gap
        self._previous_time = time_s
        target = _local_map(self._recent, self.poses[-1], self.options)
        tree = cKDTree(target)
        normals = TargetNormals(target, tree, self.options)
        deskew = point_offsets_s is not None and gap is not None and gap > 0.0
        rounds = 1 + (self.options.deskew_iterations if deskew else 0)
        motion = initial
        registration: ScanRegistration | None = None
        source = scan
        processed = scan  # replaced in the first round
        measured: FloatArray | None = None  # the model does not depend on the motion estimate
        for _ in range(rounds):
            if deskew:
                assert point_offsets_s is not None and gap is not None
                if self.rotation_model is not None and time_s is not None:
                    if measured is None:
                        measured = self.rotation_model(time_s, point_offsets_s)
                    source = deskew_points_with_rotations(
                        scan, point_offsets_s, measured, motion, gap
                    )
                else:
                    source = deskew_points(scan, point_offsets_s, motion, gap)
            processed = preprocess_scan(source, self.options)
            registration = register_point_to_plane(
                processed,
                target,
                motion,
                self.options,
                target_tree=tree,
                target_normals=normals,
            )
            motion = registration.transform
        assert registration is not None
        self.registrations.append(registration)
        self._previous_motion = registration.transform
        pose = self.poses[-1] @ registration.transform
        self.poses.append(pose)
        # ``processed`` is the last round's source, preprocessed once already.
        self._recent.append(processed @ pose[:3, :3].T + pose[:3, 3])
        return registration

    def result(self) -> ScanOdometryResult:
        """Return poses ``T_first_scan`` and registrations so far."""

        return ScanOdometryResult(poses=tuple(self.poses), registrations=tuple(self.registrations))


def deskew_points(
    points: FloatArray,
    offsets_s: FloatArray,
    motion: FloatArray,
    period_s: float,
) -> FloatArray:
    """Move points captured ``offsets_s`` after the sweep time back to it.

    ``motion`` is the sensor motion over ``period_s`` expressed in the sweep
    frame (constant velocity); a point captured at offset ``tau`` is mapped
    through the interpolated pose ``motion ** (tau / period_s)``.
    """

    from scipy.spatial.transform import Rotation

    xyz = np.asarray(points, dtype=np.float64)[:, :3]
    fractions = np.asarray(offsets_s, dtype=np.float64) / period_s
    rotation_vector = Rotation.from_matrix(motion[:3, :3]).as_rotvec()
    rotations = Rotation.from_rotvec(np.outer(fractions, rotation_vector))
    return np.asarray(rotations.apply(xyz) + np.outer(fractions, motion[:3, 3]), dtype=np.float64)


def deskew_points_with_rotations(
    points: FloatArray,
    offsets_s: FloatArray,
    rotations: FloatArray,
    motion: FloatArray,
    period_s: float,
) -> FloatArray:
    """Deskew with measured in-sweep rotations and constant-velocity translation."""

    xyz = np.asarray(points, dtype=np.float64)[:, :3]
    fractions = np.asarray(offsets_s, dtype=np.float64) / period_s
    return np.asarray(
        np.einsum("nij,nj->ni", rotations, xyz) + np.outer(fractions, motion[:3, 3]),
        dtype=np.float64,
    )


def scale_motion(motion: FloatArray, factor: float) -> FloatArray:
    """Approximately scale a small rigid motion (rotation vector and translation)."""

    from scipy.spatial.transform import Rotation

    scaled = np.eye(4)
    rotation_vector = Rotation.from_matrix(motion[:3, :3]).as_rotvec() * factor
    scaled[:3, :3] = Rotation.from_rotvec(rotation_vector).as_matrix()
    scaled[:3, 3] = motion[:3, 3] * factor
    return scaled


def _local_map(
    recent: deque[FloatArray], reference_pose: FloatArray, options: ScanOdometryOptions
) -> FloatArray:
    """Express the recent scans (stored in the first-scan frame) in ``reference_pose``."""

    if len(recent) == 1 and options.local_map_scans == 1:
        points = recent[0]
    else:
        points = preprocess_scan(np.vstack(list(recent)), _unbounded(options))
    inverse = np.linalg.inv(reference_pose)
    return points @ inverse[:3, :3].T + inverse[:3, 3]


def _unbounded(options: ScanOdometryOptions) -> ScanOdometryOptions:
    return replace(options, min_range_m=0.0, max_range_m=math.inf)


def _norm3(vector: FloatArray) -> float:
    """Euclidean norm of a small contiguous vector (``np.linalg.norm`` without its overhead)."""

    return math.sqrt(float(vector.dot(vector)))


def _exp_se3(delta: FloatArray) -> FloatArray:
    """Left-perturbation exponential with ``delta = (rotation, translation)``."""

    rotation_vector = delta[:3]
    angle = _norm3(rotation_vector)
    skew = np.array(
        [
            [0.0, -rotation_vector[2], rotation_vector[1]],
            [rotation_vector[2], 0.0, -rotation_vector[0]],
            [-rotation_vector[1], rotation_vector[0], 0.0],
        ]
    )
    if angle < 1.0e-12:
        rotation = np.eye(3) + skew
        left_jacobian = np.eye(3) + 0.5 * skew
    else:
        a = math.sin(angle) / angle
        b = (1.0 - math.cos(angle)) / (angle * angle)
        c = (angle - math.sin(angle)) / (angle**3)
        rotation = np.eye(3) + a * skew + b * (skew @ skew)
        left_jacobian = np.eye(3) + b * skew + c * (skew @ skew)
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = left_jacobian @ delta[3:]
    return result
