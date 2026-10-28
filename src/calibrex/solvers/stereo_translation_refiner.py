"""Experimental translation-only projective ICP against metric stereo surfaces.

Rotation stays fixed. No reference extrinsic or ground-truth pose enters the fit.
Depth is optical-axis z in metres; points are in the LiDAR frame. This module
is ROS-independent and is not registered as a production solver default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class StereoSurfaceObservation:
    frame_id: str
    depth_m: np.ndarray
    lidar_points: np.ndarray
    intrinsics: tuple[float, float, float, float]
    baseline_m: float

    def __post_init__(self) -> None:
        if self.depth_m.ndim != 2 or min(self.depth_m.shape) < 3:
            raise ValueError("depth must be a two-dimensional image")
        if self.lidar_points.ndim != 2 or self.lidar_points.shape[1] != 3:
            raise ValueError("points must be Nx3")
        if not np.isfinite(self.lidar_points).all():
            raise ValueError("points must be finite")
        if (
            not np.isfinite(self.intrinsics).all()
            or min(self.intrinsics[:2]) <= 0
            or not np.isfinite(self.baseline_m)
            or self.baseline_m <= 0
        ):
            raise ValueError("finite camera intrinsics and positive stereo baseline required")


@dataclass(frozen=True)
class StereoTranslationOptions:
    sampling: Literal["uniform", "point_priority"] = "uniform"
    max_iterations: int = 15
    max_points_per_frame: int = 2000
    min_points_per_frame: int = 64
    translation_bound_m: float = 0.5
    convergence_step_m: float = 1e-4
    max_depth_jump_fraction: float = 0.05
    disparity_sigma_px: float = 0.5
    lidar_sigma_m: float = 0.02
    depth_gate_floor_m: float = 1.0
    max_condition_number: float = 1000.0

    def __post_init__(self) -> None:
        if self.sampling not in ("uniform", "point_priority"):
            raise ValueError("unknown correspondence sampling rule")
        if not 3 <= self.min_points_per_frame <= self.max_points_per_frame:
            raise ValueError("inconsistent correspondence limits")
        if self.max_iterations < 1 or any(
            not np.isfinite(v) or v <= 0
            for v in [
                self.translation_bound_m,
                self.convergence_step_m,
                self.max_depth_jump_fraction,
                self.disparity_sigma_px,
                self.lidar_sigma_m,
                self.depth_gate_floor_m,
                self.max_condition_number,
            ]
        ):
            raise ValueError("positive finite refinement settings required")


@dataclass(frozen=True)
class StereoTranslationResult:
    status: str
    accepted: bool
    output_T_camera_lidar: list[list[float]]
    candidate_T_camera_lidar: list[list[float]]
    trace: list[dict[str, Any]]
    options: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Surface:
    observation: StereoSurfaceObservation
    vertices: np.ndarray
    normals: np.ndarray
    sigma_depth: np.ndarray
    sigma_plane: np.ndarray
    valid: np.ndarray


def _surface(obs: StereoSurfaceObservation, options: StereoTranslationOptions) -> _Surface:
    depth = np.asarray(obs.depth_m, dtype=float)
    v, u = np.indices(depth.shape)
    fx, fy, cx, cy = obs.intrinsics
    rays = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(depth)], axis=-1)
    vertices = rays * depth[..., None]
    normals = np.zeros_like(vertices)
    valid = np.zeros_like(depth, dtype=bool)
    center = depth[1:-1, 1:-1]
    neighbors = np.stack(
        [center, depth[1:-1, :-2], depth[1:-1, 2:], depth[:-2, 1:-1], depth[2:, 1:-1]]
    )
    mask = np.isfinite(neighbors).all(axis=0) & (neighbors > 0).all(axis=0)
    with np.errstate(invalid="ignore"):
        mask &= (
            np.max(np.abs(neighbors - center), axis=0) <= options.max_depth_jump_fraction * center
        )
        cross = np.cross(
            vertices[1:-1, 2:] - vertices[1:-1, :-2], vertices[2:, 1:-1] - vertices[:-2, 1:-1]
        )
        length = np.linalg.norm(cross, axis=-1)
    mask &= np.isfinite(length) & (length > 1e-12)
    interior = normals[1:-1, 1:-1]
    interior[mask] = cross[mask] / length[mask, None]
    valid[1:-1, 1:-1] = mask
    sigma_depth = options.disparity_sigma_px * depth**2 / (fx * obs.baseline_m)
    sensitivity = np.abs(np.sum(normals * rays, axis=-1))
    sigma_plane = np.sqrt(options.lidar_sigma_m**2 + (sensitivity * sigma_depth) ** 2)
    return _Surface(obs, vertices, normals, sigma_depth, sigma_plane, valid)


def _sample_indices(indices: np.ndarray, options: StereoTranslationOptions) -> np.ndarray:
    """Choose eligible points with either legacy spacing or fixed SplitMix64 priority.

    Priority depends only on the immutable point index within a frame. Sorting
    selected positions preserves the matcher's pixel order for linear algebra.
    """
    if len(indices) <= options.max_points_per_frame:
        return np.arange(len(indices))
    if options.sampling == "uniform":
        return np.linspace(0, len(indices) - 1, options.max_points_per_frame, dtype=int)
    priority = indices.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    priority = (priority ^ (priority >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
    priority = (priority ^ (priority >> 27)) * np.uint64(0x94D049BB133111EB)
    priority ^= priority >> 31
    selected = np.argsort(priority, kind="stable")[: options.max_points_per_frame]
    return np.sort(selected)


def _correspondences(
    surface: _Surface,
    rotation: np.ndarray,
    translation: np.ndarray,
    options: StereoTranslationOptions,
) -> tuple[np.ndarray, np.ndarray]:
    points = surface.observation.lidar_points @ rotation.T
    camera = points + translation
    h, w = surface.valid.shape
    fx, fy, cx, cy = surface.observation.intrinsics
    indices = np.flatnonzero(camera[:, 2] > 0.1)
    xyz = camera[indices]
    u = np.rint(fx * xyz[:, 0] / xyz[:, 2] + cx).astype(int)
    v = np.rint(fy * xyz[:, 1] / xyz[:, 2] + cy).astype(int)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    indices, u, v = indices[inside], u[inside], v[inside]
    # Keep the nearest point at each pixel before matching stereo surfaces.
    by_depth = np.argsort(camera[indices, 2], kind="stable")
    _, first = np.unique((v * w + u)[by_depth], return_index=True)
    selected = by_depth[first]
    indices, u, v = indices[selected], u[selected], v[selected]
    valid = surface.valid[v, u]
    indices, u, v = indices[valid], u[valid], v[valid]
    gate = np.maximum(options.depth_gate_floor_m, 3 * surface.sigma_depth[v, u])
    valid = np.abs(camera[indices, 2] - surface.vertices[v, u, 2]) <= gate
    indices, u, v = indices[valid], u[valid], v[valid]
    if len(indices) > options.max_points_per_frame:
        selected = _sample_indices(indices, options)
        indices, u, v = indices[selected], u[selected], v[selected]
    normals = surface.normals[v, u]
    sigma = surface.sigma_plane[v, u]
    design = normals / sigma[:, None]
    target = np.sum(normals * (surface.vertices[v, u] - points[indices]), axis=1) / sigma
    return design, target


def refine_stereo_translation(
    observations: list[StereoSurfaceObservation],
    initial_T_camera_lidar: np.ndarray,
    options: StereoTranslationOptions | None = None,
) -> StereoTranslationResult:
    """Fit translation with robust point-to-plane residuals and explicit rank checks.

    Correspondences and normals use stereo geometry only. An unsuccessful fit
    publishes the input transform unchanged, while retaining its raw candidate.
    """
    settings = options or StereoTranslationOptions()
    initial = np.asarray(initial_T_camera_lidar, dtype=float)
    if (
        not observations
        or initial.shape != (4, 4)
        or not np.isfinite(initial).all()
        or not np.allclose(initial[3], [0, 0, 0, 1])
        or not np.allclose(initial[:3, :3].T @ initial[:3, :3], np.eye(3), atol=1e-6)
        or abs(np.linalg.det(initial[:3, :3]) - 1) > 1e-6
    ):
        raise ValueError("observations and a finite rigid transform required")
    if len({o.frame_id for o in observations}) != len(observations):
        raise ValueError("frame IDs must be distinct")
    surfaces = [_surface(obs, settings) for obs in observations]
    translation = initial[:3, 3].copy()
    lower = translation - settings.translation_bound_m
    upper = translation + settings.translation_bound_m
    trace: list[dict[str, Any]] = []
    status = "max_iterations"
    for iteration in range(settings.max_iterations):
        pairs = [_correspondences(s, initial[:3, :3], translation, settings) for s in surfaces]
        counts = [len(a) for a, _ in pairs]
        if min(counts) < settings.min_points_per_frame:
            status = "insufficient_support"
            trace.append({"iteration": iteration, "counts": counts, "status": status})
            break
        design = np.concatenate([a for a, _ in pairs])
        target = np.concatenate([b for _, b in pairs])
        residual = design @ translation - target
        # Soft-L1 influence weights; rank is assessed on the robustly weighted geometry.
        weights = (1 + residual**2) ** (-0.25)
        singular = np.linalg.svd(design * weights[:, None], compute_uv=False)
        condition = float(singular[0] / singular[-1]) if singular[-1] > 1e-12 else None
        entry: dict[str, Any] = {
            "iteration": iteration,
            "counts": counts,
            "singular_values": singular.tolist(),
            "condition_number": condition,
        }
        trace.append(entry)
        if condition is None or condition > settings.max_condition_number:
            status = "rank_deficient"
            break
        fit = least_squares(
            lambda t, design=design, target=target: design @ t - target,
            translation,
            jac=lambda t, design=design: design.copy(),
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=50,
        )
        step = float(np.linalg.norm(fit.x - translation))
        entry.update(
            step_m=step,
            inner_success=bool(fit.success),
            translation_m=fit.x.tolist(),
            robust_cost=float(fit.cost),
        )
        translation = fit.x
        if not fit.success:
            status = "inner_not_converged"
            break
        if np.any(
            np.minimum(translation - lower, upper - translation) < settings.convergence_step_m
        ):
            status = "at_bound"
            break
        if step < settings.convergence_step_m:
            status = "converged"
            break
    candidate = initial.copy()
    candidate[:3, 3] = translation
    if status == "converged":
        final_pairs = [
            _correspondences(s, initial[:3, :3], translation, settings) for s in surfaces
        ]
        final_counts = [len(a) for a, _ in final_pairs]
        if min(final_counts) < settings.min_points_per_frame:
            status = "insufficient_support"
        else:
            final_a = np.concatenate([a for a, _ in final_pairs])
            final_b = np.concatenate([b for _, b in final_pairs])
            final_weights = (1 + (final_a @ translation - final_b) ** 2) ** (-0.25)
            singular = np.linalg.svd(final_a * final_weights[:, None], compute_uv=False)
            condition = float(singular[0] / singular[-1]) if singular[-1] > 1e-12 else None
            if condition is None or condition > settings.max_condition_number:
                status = "rank_deficient"
        trace.append({"final_counts": final_counts, "status": status})
    accepted = status == "converged"
    return StereoTranslationResult(
        status,
        accepted,
        (candidate if accepted else initial).tolist(),
        candidate.tolist(),
        trace,
        asdict(settings),
    )
