"""Experimental six-DoF alignment using pixel-scaled inverse-depth residuals.

Only metric stereo depth, pinhole intrinsics and LiDAR points enter this solver.
No reference calibration is used. Local conditioning is not accuracy validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from calibrex.solvers.stereo_support_field import build_support_field, support_residuals
from calibrex.solvers.stereo_translation_refiner import StereoSurfaceObservation


@dataclass(frozen=True)
class DisparityPoseOptions:
    support_mode: Literal["hard", "tapered"] = "hard"
    points_per_frame: int = 3000
    min_points_per_frame: int = 128
    max_nfev: int = 150
    rotation_bound_deg: float = 3.0
    translation_bound_m: float = 0.5
    sigma_px: float = 0.5
    initial_gate_px: float = 8.0
    neighborhood_range_px: float = 2.0
    min_retained_fraction: float = 0.9
    max_condition_number: float = 1e6

    def __post_init__(self) -> None:
        if self.support_mode not in ("hard", "tapered"):
            raise ValueError("unknown support mode")
        if not 6 <= self.min_points_per_frame <= self.points_per_frame or self.max_nfev < 1:
            raise ValueError("positive budgets and sufficient correspondence limits required")
        if (
            any(
                not np.isfinite(v) or v <= 0
                for v in [
                    self.rotation_bound_deg,
                    self.translation_bound_m,
                    self.sigma_px,
                    self.initial_gate_px,
                    self.neighborhood_range_px,
                    self.max_condition_number,
                ]
            )
            or not 0 < self.min_retained_fraction <= 1
        ):
            raise ValueError("finite positive settings and retention in (0,1] required")


@dataclass(frozen=True)
class DisparityPoseResult:
    status: str
    accepted: bool
    output_T_camera_lidar: list[list[float]]
    candidate_T_camera_lidar: list[list[float]]
    options: dict[str, Any]
    diagnostics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def sample_disparity(
    points_camera: np.ndarray, observation: StereoSurfaceObservation, max_range_px: float
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinearly sample pixel-scaled inverse depth; require four valid neighbors."""
    fx, fy, cx, cy = observation.intrinsics
    h, w = observation.depth_m.shape
    z = points_camera[:, 2]
    safe_z = np.maximum(z, 0.1)
    u = fx * points_camera[:, 0] / safe_z + cx
    v = fy * points_camera[:, 1] / safe_z + cy
    valid = (z > 0.1) & (u >= 0) & (v >= 0) & (u < w - 1) & (v < h - 1)
    x = np.floor(np.clip(u, 0, w - 2)).astype(int)
    y = np.floor(np.clip(v, 0, h - 2)).astype(int)
    depth = observation.depth_m
    neighbors = np.stack([depth[y, x], depth[y, x + 1], depth[y + 1, x], depth[y + 1, x + 1]])
    good = np.isfinite(neighbors) & (neighbors > 0)
    disparity = fx * observation.baseline_m / np.where(good, neighbors, 1.0)
    valid &= good.all(axis=0) & (np.ptp(disparity, axis=0) <= max_range_px)
    a, b = np.clip(u - x, 0, 1), np.clip(v - y, 0, 1)
    sampled = (
        (1 - a) * (1 - b) * disparity[0]
        + a * (1 - b) * disparity[1]
        + (1 - a) * b * disparity[2]
        + a * b * disparity[3]
    )
    return sampled - fx * observation.baseline_m / safe_z, valid


def solve_disparity_pose(
    observations: list[StereoSurfaceObservation],
    initial_T_camera_lidar: np.ndarray,
    options: DisparityPoseOptions | None = None,
) -> DisparityPoseResult:
    """Fit a fixed point population with soft-L1 residuals in half-pixel units."""
    settings = options or DisparityPoseOptions()
    initial = np.asarray(initial_T_camera_lidar, dtype=float)
    if (
        not observations
        or len({o.frame_id for o in observations}) != len(observations)
        or initial.shape != (4, 4)
        or not np.isfinite(initial).all()
        or not np.allclose(initial[3], [0, 0, 0, 1])
        or not np.allclose(initial[:3, :3].T @ initial[:3, :3], np.eye(3), atol=1e-6)
        or abs(np.linalg.det(initial[:3, :3]) - 1) > 1e-6
    ):
        raise ValueError("distinct observations and a rigid finite initializer required")
    scales = np.array(
        [np.deg2rad(settings.rotation_bound_deg)] * 3 + [settings.translation_bound_m] * 3
    )
    populations = []
    for obs in observations:
        camera = obs.lidar_points @ initial[:3, :3].T + initial[:3, 3]
        residual, valid = sample_disparity(camera, obs, settings.neighborhood_range_px)
        ids = np.flatnonzero(valid & (np.abs(residual) <= settings.initial_gate_px))
        # Nearest visible LiDAR point per initial pixel, before deterministic sampling.
        fx, fy, cx, cy = obs.intrinsics
        xyz = camera[ids]
        pixels = np.floor(fy * xyz[:, 1] / xyz[:, 2] + cy).astype(int) * obs.depth_m.shape[
            1
        ] + np.floor(fx * xyz[:, 0] / xyz[:, 2] + cx).astype(int)
        order = np.argsort(xyz[:, 2], kind="stable")
        _, first = np.unique(pixels[order], return_index=True)
        ids = ids[order[first]]
        if len(ids) > settings.points_per_frame:
            priority = ids.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
            priority = (priority ^ (priority >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
            priority = (priority ^ (priority >> 27)) * np.uint64(0x94D049BB133111EB)
            priority ^= priority >> 31
            ids = ids[np.argsort(priority, kind="stable")[: settings.points_per_frame]]
        populations.append(obs.lidar_points[ids])
    counts = [len(p) for p in populations]
    diagnostics: dict[str, Any] = {"initial_counts": counts}

    def finish(status: str, candidate: np.ndarray) -> DisparityPoseResult:
        accepted = status == "converged"
        return DisparityPoseResult(
            status,
            accepted,
            (candidate if accepted else initial).tolist(),
            candidate.tolist(),
            asdict(settings),
            diagnostics,
        )

    if min(counts) < settings.min_points_per_frame:
        return finish("insufficient_support", initial)

    fields = (
        [build_support_field(o, settings.neighborhood_range_px) for o in observations]
        if settings.support_mode == "tapered"
        else []
    )

    def transform(x: np.ndarray) -> np.ndarray:
        matrix = initial.copy()
        matrix[:3, :3] = Rotation.from_rotvec(x[:3] * scales[:3]).as_matrix() @ initial[:3, :3]
        matrix[:3, 3] = initial[:3, 3] + x[3:] * scales[3:]
        return matrix

    def evaluate(x: np.ndarray) -> tuple[np.ndarray, list[int]]:
        matrix = transform(x)
        residuals = []
        penalties = []
        retained = []
        for frame_index, (obs, points) in enumerate(zip(observations, populations, strict=True)):
            residual, valid = sample_disparity(
                points @ matrix[:3, :3].T + matrix[:3, 3], obs, settings.neighborhood_range_px
            )
            if settings.support_mode == "tapered":
                data, penalty = support_residuals(
                    points @ matrix[:3, :3].T + matrix[:3, 3],
                    obs,
                    fields[frame_index],
                    settings.sigma_px,
                    settings.initial_gate_px / settings.sigma_px,
                )
                residuals.append(data)
                penalties.append(penalty)
            else:
                # Missing data cannot silently shrink the objective population.
                residuals.append(
                    np.where(
                        valid,
                        residual / settings.sigma_px,
                        settings.initial_gate_px / settings.sigma_px,
                    )
                )
            retained.append(int(valid.sum()))
        return np.concatenate(residuals + penalties), retained

    fit = least_squares(
        lambda x: evaluate(x)[0],
        np.zeros(6),
        bounds=(-1.0, 1.0),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=settings.max_nfev,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        diff_step=1e-4,
    )
    final_residual, retained = evaluate(fit.x)
    singular = np.linalg.svd(fit.jac[: sum(counts)], compute_uv=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 1e-12 else None
    diagnostics.update(
        support_mode=settings.support_mode,
        nfev=int(fit.nfev),
        termination=int(fit.status),
        cost=float(fit.cost),
        initial_cost=float(np.sum(np.sqrt(1 + evaluate(np.zeros(6))[0] ** 2) - 1)),
        retained_counts=retained,
        scaled_parameters=fit.x.tolist(),
        weighted_jacobian_singular_values=singular.tolist(),
        condition_number=condition,
        median_absolute_residual_px=float(np.median(np.abs(final_residual)) * settings.sigma_px),
    )
    status = "converged"
    if not fit.success:
        status = "max_evaluations"
    elif np.max(np.abs(fit.x)) >= 0.999:
        status = "at_bound"
    elif any(
        n < settings.min_points_per_frame or n / c < settings.min_retained_fraction
        for n, c in zip(retained, counts, strict=True)
    ):
        status = "lost_support"
    elif condition is None or condition > settings.max_condition_number:
        status = "rank_deficient"
    return finish(status, transform(fit.x))
