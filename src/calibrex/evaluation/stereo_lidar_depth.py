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
    third axis is the optical axis. Only points on camera-facing surfaces are used.
    """

    points = _points_in_view(frame, transform, options)
    empty = np.zeros(0)
    if len(points) <= options.normal_neighbours:
        return empty, empty
    index, residual = _match(frame, stereo, points, _facing(points, options), options)
    return points[index, 2], residual


def _points_in_view(
    frame: StereoFrame, transform: FloatArray, options: StereoDepthOptions
) -> FloatArray:
    points = np.asarray(frame.points, dtype=np.float64) @ transform[:3, :3].T + transform[:3, 3]
    depth = points[:, 2]
    near = (depth > options.min_depth_m) & (depth < options.max_depth_m)
    return np.asarray(points[near & (np.abs(points[:, 0]) < 1.5 * depth)])


def _match(
    frame: StereoFrame,
    stereo: RectifiedStereo,
    points: FloatArray,
    usable: NDArray[np.bool_],
    options: StereoDepthOptions,
) -> tuple[NDArray[np.intp], FloatArray]:
    """Indices of the ``usable`` points matched between the images, and their residual.

    The left patch around each point is searched along the right row near the disparity
    the candidate predicts (zero-normalised cross-correlation, sub-pixel parabola).
    """

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
    inside &= usable
    index = np.flatnonzero(inside)
    if len(index) == 0:
        return index, np.zeros(0)
    column, row, shift, predicted = column[index], row[index], shift[index], predicted[index]

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
    rows = np.arange(len(score))
    best = score.argmax(axis=1)
    peak = np.clip(best, 1, 2 * search - 1)
    before, at, after = score[rows, peak - 1], score[rows, peak], score[rows, peak + 1]
    curvature = before - 2.0 * at + after
    sub_pixel = np.divide(
        0.5 * (before - after),
        curvature,
        out=np.zeros_like(curvature),
        where=np.abs(curvature) > 1.0e-9,
    )
    measured = candidates[rows, peak] + sub_pixel
    keep = (
        (best > 0)
        & (best < 2 * search)
        & (score[rows, best] > options.min_zncc)
        & (left_norm / np.sqrt(len(patch_rows)) > options.min_texture)
        & (measured > 1.0)
    )
    return index[keep], (measured - predicted)[keep]


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


# ---------------------------------------------------------------- the joint 6-DoF fit

STEREO_AXES: tuple[str, ...] = ("x", "y", "z", "roll", "pitch", "yaw")


@dataclass(frozen=True)
class StereoAlignmentOptions:
    """Surface selection, iterations and evidence of the joint fit."""

    matching: StereoDepthOptions = StereoDepthOptions()
    planarity: float = 0.05
    """Smallest over middle eigenvalue of a point's neighbourhood: below it is a surface."""
    min_spread: float = 0.05
    """Middle over largest eigenvalue: above it the neighbourhood is not one scan line."""
    min_view_cos: float = 0.3
    """``|normal . ray|`` above it: the surface is not seen edge-on."""
    iterations: int = 2
    max_step_m: float = 0.2
    max_step_deg: float = 2.0
    """Largest correction one iteration applies per axis: an axis the data barely
    constrain (y against a roll of the rig on the road) must not throw the transform out
    of view before its own std and control say it is not trusted."""
    rig_terms: bool = True
    """Fit the rig's own disparity scale and inter-camera roll beside its offset."""
    translation_control_m: float = 0.10
    max_translation_std_m: float = 0.05
    max_rotation_std_deg: float = 0.5


class StereoAxisEstimate(StrictModel):
    """One axis of the joint fit: the candidate's error on it and whether it is trusted."""

    name: str
    unit: str
    offset: float = Field(description="candidate minus estimate (m or deg)")
    std: float = Field(ge=0.0, description="block-jackknife standard deviation")
    control_response: float | None = Field(
        default=None,
        description="translation axes: change of the estimate when the candidate moves by "
        "the control, over the control (1 when the axis is fully observed)",
    )
    observable: bool
    reason: str


class StereoAlignmentEstimate(StrictModel):
    """The candidate's six-DoF error from stereo depth on LiDAR surfaces of any facing."""

    method: str = (
        "stereo_alignment/v0.1: disparity residual on planar LiDAR surfaces, translation and "
        "rotation fitted jointly"
    )
    axes: list[StereoAxisEstimate] = Field(default_factory=list)
    disparity_offset_px: float | None = None
    disparity_scale: float | None = Field(
        default=None, description="the rig's relative disparity scale error (rig terms)"
    )
    rig_roll_px: float | None = Field(
        default=None,
        description="disparity error per unit of normalised image row: a roll between the "
        "two cameras (rig terms)",
    )
    points: int = Field(ge=0)
    frames: int = Field(ge=0)
    reason: str


RIG_COLUMNS = 2


def surface_rows(
    frame: StereoFrame,
    stereo: RectifiedStereo,
    transform: FloatArray,
    options: StereoAlignmentOptions,
) -> tuple[FloatArray, FloatArray]:
    """Per matched point: the regressors of the candidate's error and the rig, and the residual.

    A point ``p`` on a surface with normal ``n`` that the candidate puts ``d`` away from
    where it is projects onto a pixel whose stereo depth belongs to the surface, so the
    disparity residual is ``base * (n . d) / (Z * (n . p))``. With ``d = dt + w x p`` for
    a translation error ``dt`` and a small rotation error ``w`` (rad, about the camera
    axes, applied on the left) the regressors are ``base * n / (Z (n . p))`` and
    ``base * (p x n) / (Z (n . p))``. Two more columns model the rig itself (zero when
    ``rig_terms`` is off): a disparity scale error (the predicted disparity) and a roll
    between the cameras (the normalised image row). On KITTI's road both look like a
    LiDAR height error, and on one-sided streets like a lateral one.
    """

    matching = options.matching
    points = _points_in_view(frame, transform, matching)
    if len(points) <= matching.normal_neighbours:
        return np.zeros((0, 6 + RIG_COLUMNS)), np.zeros(0)
    _, neighbours = cKDTree(points).query(points, k=matching.normal_neighbours)
    local = points[neighbours] - points[neighbours].mean(axis=1, keepdims=True)
    values, vectors = np.linalg.eigh(np.einsum("nki,nkj->nij", local, local))
    normals = vectors[:, :, 0]
    facing = np.abs((normals * points).sum(axis=1)) / np.linalg.norm(points, axis=1)
    usable = (
        (values[:, 0] < options.planarity * values[:, 1])
        & (values[:, 1] > options.min_spread * values[:, 2])
        & (facing > options.min_view_cos)
    )
    index, residual = _match(frame, stereo, points, usable, matching)
    p, n = points[index], normals[index]
    scale = stereo.base_px_m / (p[:, 2:3] * (n * p).sum(axis=1, keepdims=True))
    rig = np.zeros((len(index), RIG_COLUMNS))
    if options.rig_terms and len(index):
        homogeneous = np.c_[p, np.ones(len(p))]
        left = homogeneous @ stereo.left.T
        right = homogeneous @ stereo.right.T
        rig[:, 0] = left[:, 0] / left[:, 2] - right[:, 0] / right[:, 2]
        rig[:, 1] = (left[:, 1] / left[:, 2] - stereo.left[1, 2]) / stereo.left[1, 1]
    return np.c_[n * scale, np.cross(p, n) * scale, rig], residual


def fit_surface_offsets(rows: FloatArray, residual: FloatArray) -> FloatArray:
    """``[delta_px, dx, dy, dz, wx, wy, wz, scale, roll_px]`` of
    ``residual = delta + rows @ coefficients``.

    Least squares, three passes trimming residuals beyond 3 robust standard deviations.
    A rig column that is all zero (rig terms off) gets a zero coefficient.
    """

    keep = np.ones(len(residual), dtype=bool)
    coefficients = np.zeros(1 + rows.shape[1])
    for _ in range(3):
        design = np.c_[np.ones(int(keep.sum())), rows[keep]]
        coefficients, *_ = np.linalg.lstsq(design, residual[keep], rcond=None)
        error = residual - coefficients[0] - rows @ coefficients[1:]
        spread = 1.4826 * float(np.median(np.abs(error[keep])))
        keep = np.abs(error) <= 3.0 * max(spread, 1.0e-6)
    return np.asarray(coefficients, dtype=np.float64)


def estimate_stereo_alignment(
    frames: Sequence[StereoFrame],
    stereo: RectifiedStereo,
    transform: FloatArray,
    options: StereoAlignmentOptions | None = None,
) -> StereoAlignmentEstimate:
    """The candidate's error on all six axes, each with its jackknife std.

    The fit is re-linearised ``iterations`` times at the corrected transform. A
    translation axis is trusted when its std is within ``max_translation_std_m`` and
    moving the candidate along it by ``translation_control_m`` moves the estimate by
    half to one and a half of that; a rotation axis when its std is within
    ``max_rotation_std_deg`` (rotation has no control here).
    """

    opts = options or StereoAlignmentOptions()
    matching = opts.matching
    if len(frames) < matching.blocks:
        return StereoAlignmentEstimate(
            points=0,
            frames=len(frames),
            reason=f"{len(frames)} stereo frame(s), fewer than the {matching.blocks} blocks",
        )
    current = np.asarray(transform, dtype=np.float64).copy()
    for _ in range(opts.iterations):
        per_frame = [surface_rows(frame, stereo, current, opts) for frame in frames]
        current = _corrected(current, _bounded(fit_surface_offsets(*_stack(per_frame)), opts))
    per_frame = [surface_rows(frame, stereo, current, opts) for frame in frames]
    points = sum(len(residual) for _, residual in per_frame)
    if points < matching.min_points:
        return StereoAlignmentEstimate(
            points=points,
            frames=len(frames),
            reason=f"{points} matched points on planar surfaces, fewer than {matching.min_points}",
        )
    final = fit_surface_offsets(*_stack(per_frame))
    estimate = _corrected(current, final)
    blocks = np.array_split(np.arange(len(frames)), matching.blocks)
    fits = np.asarray(
        [
            fit_surface_offsets(
                *_stack([item for i, item in enumerate(per_frame) if i not in set(block.tolist())])
            )[1:]
            for block in blocks
        ]
    )
    count = len(blocks)
    std = np.sqrt((count - 1) / count * ((fits - fits.mean(axis=0)) ** 2).sum(axis=0))
    error = _error(transform, estimate)
    axes: list[StereoAxisEstimate] = []
    for axis, name in enumerate(STEREO_AXES):
        if axis < 3:
            moved = current.copy()
            moved[axis, 3] += opts.translation_control_m
            shifted = fit_surface_offsets(
                *_stack([surface_rows(frame, stereo, moved, opts) for frame in frames])
            )
            response = (shifted[1 + axis] - final[1 + axis]) / opts.translation_control_m
            problems = []
            if std[axis] > opts.max_translation_std_m:
                problems.append(
                    f"jackknife std {std[axis]:.3f} m exceeds {opts.max_translation_std_m:g} m"
                )
            if not matching.min_response <= response <= matching.max_response:
                problems.append(
                    f"moving the candidate {opts.translation_control_m:g} m along {name} moved "
                    f"the estimate by {response * opts.translation_control_m:+.3f} m (response "
                    f"{response:.2f}, outside {matching.min_response:g} to "
                    f"{matching.max_response:g})"
                )
            axes.append(
                StereoAxisEstimate(
                    name=name,
                    unit="m",
                    offset=float(error[axis]),
                    std=float(std[axis]),
                    control_response=float(response),
                    observable=not problems,
                    reason="; ".join(problems)
                    or f"the {opts.translation_control_m:g} m control moved the estimate by "
                    f"{response * opts.translation_control_m:+.3f} m",
                )
            )
        else:
            spread = float(np.degrees(std[axis]))
            trusted = spread <= opts.max_rotation_std_deg
            axes.append(
                StereoAxisEstimate(
                    name=name,
                    unit="deg",
                    offset=float(error[axis]),
                    std=spread,
                    observable=trusted,
                    reason=f"jackknife std {spread:.2f} deg"
                    + ("" if trusted else f", over {opts.max_rotation_std_deg:g} deg"),
                )
            )
    return StereoAlignmentEstimate(
        axes=axes,
        disparity_offset_px=float(final[0]),
        disparity_scale=float(final[7]) if opts.rig_terms else None,
        rig_roll_px=float(final[8]) if opts.rig_terms else None,
        points=points,
        frames=len(frames),
        reason=f"{points} matched points on planar surfaces, {opts.iterations} iteration(s)",
    )


def _stack(per_frame: Sequence[tuple[FloatArray, FloatArray]]) -> tuple[FloatArray, FloatArray]:
    rows = [item[0] for item in per_frame if len(item[1])]
    residual = [item[1] for item in per_frame if len(item[1])]
    if not rows:
        return np.zeros((0, 6 + RIG_COLUMNS)), np.zeros(0)
    return np.concatenate(rows), np.concatenate(residual)


def _bounded(coefficients: FloatArray, options: StereoAlignmentOptions) -> FloatArray:
    bounded = coefficients.copy()
    bounded[1:4] = np.clip(bounded[1:4], -options.max_step_m, options.max_step_m)
    limit = np.radians(options.max_step_deg)
    bounded[4:7] = np.clip(bounded[4:7], -limit, limit)
    return bounded


def _corrected(transform: FloatArray, coefficients: FloatArray) -> FloatArray:
    """The transform with the fitted error removed: ``p_true = Exp(-w) p - dt``."""

    from scipy.spatial.transform import Rotation

    undo = Rotation.from_rotvec(-coefficients[4:7]).as_matrix()
    result = transform.copy()
    result[:3, :3] = undo @ transform[:3, :3]
    result[:3, 3] = undo @ transform[:3, 3] - coefficients[1:4]
    return result


def _error(candidate: FloatArray, estimate: FloatArray) -> FloatArray:
    """Candidate minus estimate: translation (m) and left rotation about the camera axes (deg)."""

    from scipy.spatial.transform import Rotation

    rotation = Rotation.from_matrix(candidate[:3, :3] @ estimate[:3, :3].T).as_rotvec()
    return np.r_[candidate[:3, 3] - estimate[:3, 3], np.degrees(rotation)]
