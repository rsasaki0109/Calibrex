"""Camera-LiDAR translation along the optical axis from a rectified stereo pair.

Image edges see a camera-LiDAR translation only through parallax, and in a street scene
not even that along the optical axis: moving the candidate along it keeps a road point on
the road and a side-wall point on its wall, so the shifted cloud still lines up. A
surface that faces the camera does not have that freedom. Its depth is fixed by the
stereo pair, independently of the LiDAR.

This module projects each LiDAR point on a camera-facing surface into both rectified
images with the candidate, block-matches the left patch along the right row
(zero-normalised cross-correlation, sub-pixel parabola) and models the disparity residual
as::

    measured - predicted = delta + offset * base / Z**2

``delta`` absorbs a constant disparity offset of the stereo rig (KITTI's reads about
-0.6 px); ``offset`` is how much farther than the stereo depth the candidate places the
points, i.e. the candidate's translation error along the optical axis (candidate minus
estimate). It is separable from ``delta`` because it fades with depth. The
uncertainty is a block jackknife over frames, and a control re-estimates with the
candidate moved by ``control_m``: an axis whose estimate does not follow it is not
observable on the recording.

Only the stereo pair's own calibration (its rectified projections) is used, never a
camera-LiDAR reference.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from scipy import ndimage
from scipy.spatial import cKDTree

from calibrex.core.result import StrictModel

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class RectifiedStereo:
    """A rectified stereo pair: both 3x4 projections from the frame the points are given in."""

    left: FloatArray
    right: FloatArray

    @property
    def base_px_m(self) -> float:
        """Focal length times baseline: disparity (px) times depth (m)."""

        return float(self.left[0, 3] - self.right[0, 3])


@dataclass(frozen=True)
class StereoFrame:
    """One rectified image pair (grayscale, 0-255) and the LiDAR scan taken with it."""

    left: FloatArray
    right: FloatArray
    points: FloatArray


@dataclass(frozen=True)
class StereoDepthOptions:
    """Matching, surface selection and evidence settings."""

    patch_half_px: int = 4
    search_px: int = 10
    min_zncc: float = 0.9
    min_texture: float = 6.0
    min_depth_m: float = 3.0
    max_depth_m: float = 60.0
    facing_cos: float = 0.9
    occlusion_ratio: float = 1.05
    normal_neighbours: int = 12
    blocks: int = 6
    control_m: float = 0.10
    min_points: int = 2000
    max_std_m: float = 0.05
    min_response: float = 0.5
    max_response: float = 1.5

    def __post_init__(self) -> None:
        if self.blocks < 3 or self.patch_half_px < 1 or self.search_px < 2:
            raise ValueError("need three blocks, a patch and a disparity search")
        if not 0.0 < self.facing_cos < 1.0 or self.control_m <= 0.0:
            raise ValueError("facing_cos must be in (0, 1) and control_m positive")


class StereoDepthEstimate(StrictModel):
    """The candidate's translation error along the optical axis, from stereo depth."""

    method: str = "stereo_depth/v0.1: disparity residual on camera-facing LiDAR surfaces"
    offset_m: float | None = Field(
        default=None, description="candidate minus estimate along the optical axis"
    )
    std_m: float | None = Field(default=None, description="block-jackknife standard deviation")
    disparity_offset_px: float | None = None
    control_m: float
    control_response: float | None = Field(
        default=None,
        description="change of the estimate when the candidate moves by control_m, over "
        "control_m (1 when the axis is fully observed)",
    )
    points: int = Field(ge=0)
    frames: int = Field(ge=0)
    observable: bool
    reason: str


def frame_residuals(
    frame: StereoFrame,
    stereo: RectifiedStereo,
    transform: FloatArray,
    options: StereoDepthOptions,
) -> tuple[FloatArray, FloatArray]:
    """Depth (m) and disparity residual (measured minus predicted, px) of matched points.

    ``transform`` maps LiDAR points into the frame of ``stereo``'s projections, whose
    third axis is the optical axis.
    """

    points = np.asarray(frame.points, dtype=np.float64) @ transform[:3, :3].T + transform[:3, 3]
    depth = points[:, 2]
    near = (depth > options.min_depth_m) & (depth < options.max_depth_m)
    points = points[near & (np.abs(points[:, 0]) < 1.5 * depth)]
    empty = np.zeros(0)
    if len(points) <= options.normal_neighbours:
        return empty, empty
    homogeneous = np.c_[points, np.ones(len(points))]
    left = homogeneous @ stereo.left.T
    right = homogeneous @ stereo.right.T
    u_left = left[:, 0] / left[:, 2]
    v_left = left[:, 1] / left[:, 2]
    predicted = u_left - right[:, 0] / right[:, 2]

    half, search = options.patch_half_px, options.search_px
    height, width = frame.left.shape
    column = np.round(u_left).astype(int)
    row = np.round(v_left).astype(int)
    shift = np.round(predicted).astype(int)
    inside = (
        (row >= half)
        & (row < height - half)
        & (column >= half)
        & (column < width - half)
        & (column - shift - search - half >= 0)
        & (column - shift + search + half < width)
    )
    # A return with a nearer one inside its patch is hidden from the camera (the LiDAR
    # sees from elsewhere) or straddles a depth edge: neither patch shows its surface.
    on_image = (row >= 0) & (row < height) & (column >= 0) & (column < width)
    nearest = np.full((height, width), np.inf)
    np.minimum.at(nearest, (row[on_image], column[on_image]), points[on_image, 2])
    nearest = ndimage.minimum_filter(nearest, size=2 * half + 1)
    inside &= on_image
    inside[inside] &= (
        points[inside, 2] <= options.occlusion_ratio * nearest[row[inside], column[inside]]
    )
    inside[inside] = _facing(points, options)[inside]
    column, row, shift = column[inside], row[inside], shift[inside]
    depth, predicted = points[inside, 2], predicted[inside]
    if len(depth) == 0:
        return empty, empty

    offsets = np.arange(-half, half + 1)
    patch_rows, patch_cols = (axis.ravel() for axis in np.meshgrid(offsets, offsets, indexing="ij"))
    left_patch = frame.left[row[:, None] + patch_rows, column[:, None] + patch_cols]
    left_patch = left_patch - left_patch.mean(axis=1, keepdims=True)
    left_norm = np.sqrt((left_patch**2).sum(axis=1))
    candidates = shift[:, None] + np.arange(-search, search + 1)
    right_cols = column[:, None, None] - candidates[:, :, None] + patch_cols[None, None, :]
    right_patch = frame.right[row[:, None, None] + patch_rows[None, None, :], right_cols]
    right_patch = right_patch - right_patch.mean(axis=2, keepdims=True)
    right_norm = np.sqrt((right_patch**2).sum(axis=2))
    score = (right_patch * left_patch[:, None, :]).sum(axis=2) / np.maximum(
        right_norm * left_norm[:, None], 1.0e-9
    )
    index = np.arange(len(score))
    best = score.argmax(axis=1)
    peak = np.clip(best, 1, 2 * search - 1)
    before, at, after = score[index, peak - 1], score[index, peak], score[index, peak + 1]
    curvature = before - 2.0 * at + after
    sub_pixel = np.divide(
        0.5 * (before - after),
        curvature,
        out=np.zeros_like(curvature),
        where=np.abs(curvature) > 1.0e-9,
    )
    measured = candidates[index, peak] + sub_pixel
    keep = (
        (best > 0)
        & (best < 2 * search)
        & (score[index, best] > options.min_zncc)
        & (left_norm / np.sqrt(len(patch_rows)) > options.min_texture)
        & (measured > 1.0)
    )
    return depth[keep], (measured - predicted)[keep]


def fit_depth_offset(
    depth: FloatArray, residual: FloatArray, base_px_m: float
) -> tuple[float, float]:
    """``(delta_px, offset_m)`` of ``residual = delta + offset * base / depth**2``.

    Least squares, three passes trimming residuals beyond 3 robust standard deviations.
    """

    sensitivity = base_px_m / depth**2
    keep = np.ones(len(depth), dtype=bool)
    coefficients = np.zeros(2)
    for _ in range(3):
        design = np.c_[np.ones(int(keep.sum())), sensitivity[keep]]
        coefficients, *_ = np.linalg.lstsq(design, residual[keep], rcond=None)
        error = residual - coefficients[0] - coefficients[1] * sensitivity
        spread = 1.4826 * float(np.median(np.abs(error[keep])))
        keep = np.abs(error) <= 3.0 * max(spread, 1.0e-6)
    return float(coefficients[0]), float(coefficients[1])


def estimate_depth_offset(
    frames: Sequence[StereoFrame],
    stereo: RectifiedStereo,
    transform: FloatArray,
    options: StereoDepthOptions | None = None,
) -> StereoDepthEstimate:
    """The candidate's error along the optical axis, its jackknife std and its control."""

    opts = options or StereoDepthOptions()
    base = stereo.base_px_m
    if len(frames) < opts.blocks:
        return StereoDepthEstimate(
            control_m=opts.control_m,
            points=0,
            frames=len(frames),
            observable=False,
            reason=f"{len(frames)} stereo frame(s), fewer than the {opts.blocks} blocks",
        )
    per_frame = [frame_residuals(frame, stereo, transform, opts) for frame in frames]
    points = sum(len(depth) for depth, _ in per_frame)
    if points < opts.min_points:
        return StereoDepthEstimate(
            control_m=opts.control_m,
            points=points,
            frames=len(frames),
            observable=False,
            reason=f"{points} matched points on camera-facing surfaces, fewer than "
            f"{opts.min_points}",
        )
    delta, offset = _fit(per_frame, base)
    blocks = np.array_split(np.arange(len(frames)), opts.blocks)
    fits = np.asarray(
        [
            _fit([item for i, item in enumerate(per_frame) if i not in set(block.tolist())], base)
            for block in blocks
        ]
    )
    count = len(blocks)
    std = float(np.sqrt((count - 1) / count * ((fits[:, 1] - fits[:, 1].mean()) ** 2).sum()))
    moved = transform.copy()
    moved[2, 3] += opts.control_m
    _, moved_offset = _fit([frame_residuals(f, stereo, moved, opts) for f in frames], base)
    response = (moved_offset - offset) / opts.control_m
    problems = []
    if std > opts.max_std_m:
        problems.append(f"jackknife std {std:.3f} m exceeds {opts.max_std_m:g} m")
    if not opts.min_response <= response <= opts.max_response:
        problems.append(
            f"moving the candidate {opts.control_m:g} m along the optical axis moved the "
            f"estimate by {response * opts.control_m:+.3f} m (response {response:.2f}, "
            f"outside {opts.min_response:g} to {opts.max_response:g})"
        )
    return StereoDepthEstimate(
        offset_m=offset,
        std_m=std,
        disparity_offset_px=delta,
        control_m=opts.control_m,
        control_response=response,
        points=points,
        frames=len(frames),
        observable=not problems,
        reason="; ".join(problems)
        or f"{points} matched points on camera-facing surfaces; the {opts.control_m:g} m "
        f"control moved the estimate by {response * opts.control_m:+.3f} m",
    )


def _fit(per_frame: Sequence[tuple[FloatArray, FloatArray]], base: float) -> tuple[float, float]:
    depth = np.concatenate([item[0] for item in per_frame])
    residual = np.concatenate([item[1] for item in per_frame])
    return fit_depth_offset(depth, residual, base)


def _facing(points: FloatArray, options: StereoDepthOptions) -> NDArray[np.bool_]:
    """Points whose local surface normal is within ``facing_cos`` of the optical axis."""

    _, neighbours = cKDTree(points).query(points, k=options.normal_neighbours)
    local = points[neighbours] - points[neighbours].mean(axis=1, keepdims=True)
    _, vectors = np.linalg.eigh(np.einsum("nki,nkj->nij", local, local))
    return np.asarray(np.abs(vectors[:, 2, 0]) > options.facing_cos)
