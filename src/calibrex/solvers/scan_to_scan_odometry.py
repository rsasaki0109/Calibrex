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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

FloatArray: TypeAlias = NDArray[np.float64]


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
    ranges = np.linalg.norm(xyz, axis=1)
    xyz = xyz[(ranges >= options.min_range_m) & (ranges <= options.max_range_m)]
    if xyz.size == 0:
        return xyz.reshape(0, 3)
    keys = np.floor(xyz / options.voxel_size_m).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((counts.size, 3))
    np.add.at(sums, inverse.reshape(-1), xyz)
    return sums / counts[:, None]


def estimate_normals(
    points: FloatArray,
    tree: cKDTree,
    options: ScanOdometryOptions,
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Return unit normals and a mask of locally planar points."""

    k = min(options.normal_neighbours, len(points))
    _, indices = tree.query(points, k=k)
    neighbours = points[indices]
    centered = neighbours - neighbours.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered) / max(k, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0]
    planar = eigenvalues[:, 0] <= options.max_planarity_ratio * np.maximum(eigenvalues[:, 1], 1e-12)
    return normals, planar


def register_point_to_plane(
    source: FloatArray,
    target: FloatArray,
    initial: FloatArray,
    options: ScanOdometryOptions | None = None,
    *,
    target_tree: cKDTree | None = None,
    target_normals: tuple[FloatArray, NDArray[np.bool_]] | None = None,
) -> ScanRegistration:
    """Estimate ``T_target_source`` from preprocessed point arrays."""

    opts = options or ScanOdometryOptions()
    tree = target_tree if target_tree is not None else cKDTree(target)
    normals, planar = target_normals or estimate_normals(target, tree, opts)
    transform = np.array(initial, dtype=np.float64)
    converged = False
    count = 0
    rmse = math.inf
    min_eigenvalue = 0.0
    iteration = 0
    for iteration in range(1, opts.max_iterations + 1):  # noqa: B007
        moved = source @ transform[:3, :3].T + transform[:3, 3]
        distances, indices = tree.query(moved, k=1, distance_upper_bound=opts.max_correspondence_m)
        valid = np.isfinite(distances)
        valid[valid] &= planar[indices[valid]]
        count = int(valid.sum())
        if count < opts.min_correspondences:
            break
        p = moved[valid]
        q = target[indices[valid]]
        n = normals[indices[valid]]
        residual = np.einsum("ij,ij->i", p - q, n)
        jacobian = np.hstack([np.cross(p, n), n])
        weights = np.where(
            np.abs(residual) <= opts.huber_delta_m,
            1.0,
            opts.huber_delta_m / np.maximum(np.abs(residual), 1e-12),
        )
        hessian = (jacobian * weights[:, None]).T @ jacobian
        gradient = (jacobian * weights[:, None]).T @ residual
        rmse = float(math.sqrt(float(np.mean(residual * residual))))
        try:
            delta = -np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        min_eigenvalue = float(np.linalg.eigvalsh(hessian)[0]) / count
        transform = _exp_se3(delta) @ transform
        if (
            float(np.linalg.norm(delta[3:])) < opts.convergence_translation_m
            and float(np.linalg.norm(delta[:3])) < opts.convergence_rotation_rad
        ):
            converged = True
            break
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
    """Run scan-to-scan odometry while holding only two preprocessed scans."""

    opts = options or ScanOdometryOptions()
    if times_s is not None and len(times_s) != count:
        raise ValueError("times_s must have one entry per scan")
    poses = [np.eye(4)]
    registrations: list[ScanRegistration] = []
    if count == 0:
        return ScanOdometryResult(poses=(), registrations=())
    target = preprocess_scan(load_scan(0), opts)
    previous_motion = np.eye(4)
    previous_gap: float | None = None
    for index in range(1, count):
        source = preprocess_scan(load_scan(index), opts)
        initial = previous_motion
        if times_s is not None:
            gap = float(times_s[index]) - float(times_s[index - 1])
            if previous_gap is not None and previous_gap > 0.0 and gap > 0.0:
                initial = scale_motion(previous_motion, gap / previous_gap)
            previous_gap = gap
        tree = cKDTree(target)
        registration = register_point_to_plane(
            source,
            target,
            initial,
            opts,
            target_tree=tree,
            target_normals=estimate_normals(target, tree, opts),
        )
        registrations.append(registration)
        previous_motion = registration.transform
        poses.append(poses[-1] @ registration.transform)
        target = source
    return ScanOdometryResult(poses=tuple(poses), registrations=tuple(registrations))


def scale_motion(motion: FloatArray, factor: float) -> FloatArray:
    """Approximately scale a small rigid motion (rotation vector and translation)."""

    from scipy.spatial.transform import Rotation

    scaled = np.eye(4)
    rotation_vector = Rotation.from_matrix(motion[:3, :3]).as_rotvec() * factor
    scaled[:3, :3] = Rotation.from_rotvec(rotation_vector).as_matrix()
    scaled[:3, 3] = motion[:3, 3] * factor
    return scaled


def _exp_se3(delta: FloatArray) -> FloatArray:
    """Left-perturbation exponential with ``delta = (rotation, translation)``."""

    rotation_vector = delta[:3]
    angle = float(np.linalg.norm(rotation_vector))
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
