"""LiDAR-LiDAR extrinsic by registering one LiDAR's scans to the other's local maps.

A reference LiDAR's odometry places its scans in one frame; around each sample
time a local map is built from its nearby scans.  A scan of the target LiDAR
taken at that time must then land on the map through ``P X``, where ``P`` is
the reference pose ``T_odom_reference`` and ``X = T_reference_target`` is the
extrinsic.  All samples are solved jointly for one ``X`` by Gauss-Newton on
point-to-plane residuals with Huber weights, re-associating correspondences
between iterations.

Unlike motion-based hand-eye calibration, which needs each LiDAR's relative
motion to be accurate and fails when one odometry is weak (a 16-beam LiDAR
turned on its side, for example), every sample constrains all six degrees of
freedom directly from the scene, and the LiDARs need only a small overlap in
what they see over a few seconds.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

FloatArray: TypeAlias = NDArray[np.float64]
EXTRINSIC_DOFS: tuple[str, ...] = ("roll", "pitch", "yaw", "x", "y", "z")


@dataclass(frozen=True)
class MapSampleOptions:
    """Correspondence gating, robust loss, and iterations."""

    max_correspondence_m: float = 0.5
    normal_neighbours: int = 10
    max_planarity_ratio: float = 0.2
    huber_m: float = 0.05
    iterations: int = 15
    convergence_rotation_rad: float = 1.0e-6
    convergence_translation_m: float = 1.0e-5


@dataclass
class MapSample:
    """One target scan and the reference map around its time."""

    time_s: float
    block: int
    reference_pose: FloatArray  # T_odom_reference at the sample time
    target_points: FloatArray  # (N, 3) in the target LiDAR frame
    map_points: FloatArray  # (M, 3) in the odometry frame
    _tree: cKDTree | None = None
    _normals: FloatArray | None = None
    _planar: NDArray[np.bool_] | None = None

    def prepare(self, options: MapSampleOptions) -> None:
        """Build the map's kd-tree and planar normals once."""

        if self._tree is not None:
            return
        self._tree = cKDTree(self.map_points)
        k = min(options.normal_neighbours, len(self.map_points))
        _, indices = self._tree.query(self.map_points, k=k)
        neighbours = self.map_points[indices]
        centered = neighbours - neighbours.mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", centered, centered) / max(k, 1)
        values, vectors = np.linalg.eigh(covariance)
        self._normals = vectors[:, :, 0]
        self._planar = values[:, 0] <= options.max_planarity_ratio * values[:, 1]

    def residuals(
        self, extrinsic: FloatArray, options: MapSampleOptions
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Point-to-plane residuals, their normals, and the transformed points."""

        self.prepare(options)
        assert self._tree is not None and self._normals is not None and self._planar is not None
        transform = self.reference_pose @ extrinsic
        world = self.target_points @ transform[:3, :3].T + transform[:3, 3]
        distances, indices = self._tree.query(
            world, distance_upper_bound=options.max_correspondence_m
        )
        valid = np.isfinite(distances)
        valid[valid] &= self._planar[indices[valid]]
        normals = self._normals[indices[valid]]
        targets = self.map_points[indices[valid]]
        residual = np.einsum("ij,ij->i", world[valid] - targets, normals)
        return residual, normals, self.target_points[valid]


@dataclass(frozen=True)
class MapExtrinsicResult:
    """Joint estimate of ``X = T_reference_target`` with its analytic covariance."""

    status: Literal["converged", "max_iterations", "insufficient_samples"]
    transform: FloatArray | None
    covariance: FloatArray | None  # 6x6 over (left rotation about reference axes, translation)
    sigma_m: float | None
    samples: int
    correspondences: int
    iterations: int


def solve_map_extrinsic(
    samples: Sequence[MapSample],
    initial: FloatArray,
    options: MapSampleOptions | None = None,
) -> MapExtrinsicResult:
    """Gauss-Newton with Huber weights over every sample's correspondences.

    The update is a left perturbation of ``X``: a rotation ``w`` about the
    reference LiDAR's axes and a translation ``v`` in its frame,
    ``X <- (Exp(w), v) X``.
    """

    opts = options or MapSampleOptions()
    if len(samples) < 3:
        return MapExtrinsicResult("insufficient_samples", None, None, None, len(samples), 0, 0)
    extrinsic = np.asarray(initial, dtype=np.float64).copy()
    hessian = np.eye(6)
    sigma = opts.huber_m
    count = 0
    status: Literal["converged", "max_iterations", "insufficient_samples"] = "max_iterations"
    iteration = 0
    for iteration in range(1, opts.iterations + 1):
        hessian = np.zeros((6, 6))
        gradient = np.zeros(6)
        stacked: list[FloatArray] = []
        count = 0
        for sample in samples:
            residual, normals, points = sample.residuals(extrinsic, opts)
            if len(residual) == 0:
                continue
            jacobian = _jacobian(sample.reference_pose, extrinsic, normals, points)
            weight = np.minimum(1.0, opts.huber_m / np.maximum(np.abs(residual), 1e-12))
            hessian += jacobian.T @ (jacobian * weight[:, None])
            gradient += jacobian.T @ (weight * residual)
            stacked.append(residual)
            count += len(residual)
        if count < 30:
            return MapExtrinsicResult(
                "insufficient_samples", None, None, None, len(samples), count, iteration
            )
        step = -np.linalg.solve(hessian + 1e-9 * np.eye(6), gradient)
        update = np.eye(4)
        update[:3, :3] = Rotation.from_rotvec(step[:3]).as_matrix()
        update[:3, 3] = step[3:]
        # Left perturbation about the reference frame: rotate X, then translate.
        extrinsic = np.block(
            [
                [
                    update[:3, :3] @ extrinsic[:3, :3],
                    (update[:3, :3] @ extrinsic[:3, 3] + step[3:])[:, None],
                ],
                [np.zeros((1, 3)), np.ones((1, 1))],
            ]
        )
        residuals = np.concatenate(stacked)
        sigma = max(float(1.4826 * np.median(np.abs(residuals))), 1e-6)
        if (
            np.linalg.norm(step[:3]) < opts.convergence_rotation_rad
            and np.linalg.norm(step[3:]) < opts.convergence_translation_m
        ):
            status = "converged"
            break
    covariance = np.linalg.inv(hessian + 1e-12 * np.eye(6)) * sigma**2
    return MapExtrinsicResult(status, extrinsic, covariance, sigma, len(samples), count, iteration)


def chi_square(
    samples: Sequence[MapSample],
    extrinsic: FloatArray,
    sigma_m: float,
    options: MapSampleOptions | None = None,
) -> tuple[float, float, int]:
    """Clipped chi-square, median absolute residual, and correspondences for fixed ``X``.

    Points whose correspondence falls outside the gate contribute the clip
    value, so a wrong extrinsic cannot lower its score by losing points.
    """

    opts = options or MapSampleOptions()
    total = 0.0
    residual_parts: list[FloatArray] = []
    count = 0
    for sample in samples:
        residual, _, _ = sample.residuals(extrinsic, opts)
        lost = len(sample.target_points) - len(residual)
        total += float(np.sum(np.minimum((residual / sigma_m) ** 2, 25.0))) + 25.0 * lost
        residual_parts.append(residual)
        count += len(residual)
    stacked = np.concatenate(residual_parts) if residual_parts else np.zeros(0)
    median = float(np.median(np.abs(stacked))) if len(stacked) else math.nan
    return total, median, count


def perturb_extrinsic(extrinsic: FloatArray, dof: str, amount: float) -> FloatArray:
    """Left-perturb one DoF: a rotation (rad) about a reference axis, or a translation (m)."""

    index = EXTRINSIC_DOFS.index(dof)
    moved = extrinsic.copy()
    if index < 3:
        axis = np.zeros(3)
        axis[index] = amount
        rotation = Rotation.from_rotvec(axis).as_matrix()
        moved[:3, :3] = rotation @ extrinsic[:3, :3]
        moved[:3, 3] = rotation @ extrinsic[:3, 3]
    else:
        moved[index - 3, 3] += amount
    return moved


def _jacobian(
    pose: FloatArray, extrinsic: FloatArray, normals: FloatArray, points: FloatArray
) -> FloatArray:
    """d(residual)/d(w, v) for the left perturbation of ``X``."""

    # q = P (Exp(w) X p + v) with p_ref = X p the point in the reference frame.
    in_reference = points @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    normals_in_reference = normals @ pose[:3, :3]
    rotation_part = np.cross(in_reference, normals_in_reference)
    return np.column_stack([rotation_part, normals_in_reference])
