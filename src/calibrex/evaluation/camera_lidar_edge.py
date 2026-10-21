"""Targetless camera-LiDAR extrinsic by edge alignment, with held-out controls.

The objective is the one of Levinson and Thrun (RSS 2013): LiDAR points on the
*near side* of a depth discontinuity should project onto image edges.  The image
edge response is the gradient magnitude spilled outward with an inverse-distance
falloff (their Eq. 1), and the LiDAR weights are the square roots of the
near-side range jumps along each beam (their Eq. 2).  This module is an
independent, vectorised implementation of that objective used as an *offline,
pooled* estimator rather than their online tracker:

* beams are recovered from the elevation of each point, so unorganised clouds
  without a ``ring`` field work;
* frames are pooled (a spacing of seconds, so the views differ) and a
  deterministic coordinate search maximises the pooled objective from the
  candidate;
* the standard deviation of every DoF is a leave-one-block-out jackknife over
  contiguous blocks of frames;
* every DoF has a known-bad control: shifted both ways, the estimate must beat
  the shifted transform on held-out blocks; a DoF whose control is missed is
  reported as unobservable.

The camera frame is the optical frame (``z`` forward).  Scans are treated as rigid
snapshots: neither the platform's motion nor the LiDAR's rolling sweep is
compensated, which limits how small a rotation error can be resolved.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Generator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias, TypedDict, TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.spatial.transform import Rotation

from calibrex import __version__
from calibrex.core.camera_lidar_edge import (
    DEFAULT_MIN_CONTROL_T,
    CameraLidarEdgeArtifact,
    CameraLidarEdgeControl,
    CameraLidarEdgeDofName,
    CameraLidarEdgeDofRecord,
    CameraLidarEdgeMotion,
    CameraLidarEdgeObjective,
    CameraLidarEdgeProvenance,
    CameraLidarEdgeSamples,
    CameraLidarEdgeTransform,
)
from calibrex.core.progress import emit_stage, emit_tick
from calibrex.core.provenance import git_commit
from calibrex.data.livox_ros2 import bag_input_digest
from calibrex.data.ros2_camera_lidar import (
    CameraLidarFrame,
    PointTimeEncoding,
    load_camera_lidar_frames,
)
from calibrex.evaluation.camera_lidar_motion import LidarMotion, move_frames_to_image_time
from calibrex.evaluation.edge_preparation_cache import EdgePreparationCache

FloatArray: TypeAlias = NDArray[np.float64]
Float32Array: TypeAlias = NDArray[np.float32]
DistortionModel = Literal["none", "radtan", "equidistant"]
DOF_NAMES: tuple[CameraLidarEdgeDofName, ...] = ("roll", "pitch", "yaw", "x", "y", "z")
LIMITATIONS: tuple[str, ...] = (
    "The camera frame is assumed to be the optical frame (x right, y down, z forward) and the "
    "intrinsics are taken as correct; an intrinsics error reads as an extrinsic error.",
    "Image edges and LiDAR depth discontinuities constrain rotation well and translation "
    "weakly (translation is a parallax effect that scales with 1/depth): on the Hilti 2022 and "
    "KITTI recordings the translation estimate differed from the reference by up to 11 cm with "
    "a reported std of 1-2 cm. Translation is therefore not estimated by default; its "
    "held-out control is still recorded.",
    "The jackknife standard deviation does not include systematic effects (lens model, "
    "timing, scene composition); the verdict floors cover those.",
    "Edge alignment needs structured scenes with depth discontinuities and image edges; "
    "textureless or very distant scenes yield an inconclusive result.",
)
RIGID_SCAN_LIMITATION = (
    "Scans are treated as rigid snapshots: platform motion and the LiDAR's rolling sweep are "
    "not compensated, and the image is paired with the scan at the time its visible points "
    "were measured only when the cloud has a per-point time field."
)


def motion_limitation(motion: CameraLidarEdgeMotion) -> str:
    """What moving the points to the image time with ``motion`` does and does not cover."""

    parts = [
        f"LiDAR points were moved to the image time with the rig motion from "
        f"{motion.source_topic} (constant velocity between each point's and the image's time"
    ]
    parts.append(
        "; each point by its own measurement time)"
        if motion.per_point_time
        else "; the cloud has no per-point time, so a scan is moved as a whole and the "
        "rolling sweep is not compensated)"
    )
    if motion.kind == "gyro":
        parts.append(". The source is a gyro: rotation only, the rig's translation is not")
    if motion.frames_without_motion:
        parts.append(
            f". {motion.frames_without_motion} frame(s) had no motion sample nearby and were "
            "used as recorded"
        )
    return "".join(parts) + "."


@dataclass(frozen=True)
class EdgeCamera:
    """Pinhole intrinsics with optional radial-tangential or equidistant distortion."""

    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: DistortionModel = "none"
    distortion: tuple[float, ...] = ()

    def project(self, points: FloatArray | Float32Array) -> tuple[FloatArray, FloatArray, Any]:
        """Pixel coordinates ``(u, v)`` of camera-frame points and a mask of valid ones."""

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        valid = z > 1.0e-3
        safe = np.where(valid, z, 1.0)
        x = x / safe
        y = y / safe
        coefficients = self.distortion
        if self.distortion_model == "equidistant" and len(coefficients) >= 4:
            radius = np.sqrt(x * x + y * y)
            theta = np.arctan(radius)
            k1, k2, k3, k4 = coefficients[:4]
            t2 = theta * theta
            distorted = theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
            scale = np.divide(distorted, radius, out=np.ones_like(radius), where=radius > 1e-12)
            x = x * scale
            y = y * scale
        elif self.distortion_model == "radtan" and coefficients:
            r2 = x * x + y * y
            k1, k2 = coefficients[0], coefficients[1]
            p1 = coefficients[2] if len(coefficients) > 2 else 0.0
            p2 = coefficients[3] if len(coefficients) > 3 else 0.0
            k3 = coefficients[4] if len(coefficients) > 4 else 0.0
            radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
            if len(coefficients) >= 8:
                denominator = 1.0 + r2 * (
                    coefficients[5] + r2 * (coefficients[6] + r2 * coefficients[7])
                )
                radial = radial / denominator
            xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
            x, y = xd, yd
        return self.fx * x + self.cx, self.fy * y + self.cy, valid

    def as_dict(self) -> dict[str, Any]:
        """Schema-safe record of the intrinsics."""

        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "distortion_model": self.distortion_model,
            "distortion": list(self.distortion),
        }


@dataclass(frozen=True)
class EdgeAlignmentOptions:
    """Frame selection, objective, search and evidence settings."""

    spacing_s: float = 1.0
    max_frames: int = 48
    max_seconds: float | None = None
    # A recording too short for ``max_frames`` scans ``spacing_s`` apart (KITTI 0005 is
    # 15 s) spreads ``max_frames`` scans over its length instead, at least this far apart.
    # Recordings long enough for the regular selection are unaffected.  ``None`` turns
    # it off.
    short_recording_spacing_s: float | None = 0.25
    # ...but only when the view changes: consecutive denser images must differ by this much
    # (median mean absolute difference, fraction of full scale; static AIST Livox recordings
    # read at most 0.011, moving KITTI drives at least 0.14).
    short_recording_min_view_change: float = 0.05
    max_gap_s: float = 0.06
    min_frames: int = 8
    blocks: int = 6
    min_jump_m: float = 0.3
    jump_exponent: float = 0.5
    max_edge_points_per_frame: int = 4000
    min_edge_points_per_frame: int = 150
    image_max_width: int = 960
    spill_gamma: float = 0.98
    spill_radius_px: int = 40
    # The edge response minus its local mean (a Gaussian of this many pixels of the
    # downscaled image): without it the objective rewards moving points into textured
    # regions, and on KITTI highway and city drives it drifted to a wrong rotation several
    # degrees away whose objective was higher than at the calibration.  ``None`` keeps the
    # raw response.
    edge_background_sigma_px: float | None = 20.0
    # A spinning LiDAR at a standing rig repeats one scan: a frame whose image differs from
    # the last kept frame's by less than this (mean absolute difference of every fourth
    # pixel, fraction of full scale) adds no view, only weight, and makes the blocks of the
    # jackknife and the controls copies of each other.  Such frames are dropped; a solid-
    # state LiDAR keeps them (its pattern fills in).  KITTI 0057 stands for 29 s at a
    # crossing (changes 0.002-0.034, from traffic) and drives off (0.06-0.18).  ``None``
    # keeps every frame.
    repeated_view_change: float | None = 0.05
    image_margin_px: float = 80.0
    rotation_bound_deg: float = 12.0
    translation_bound_m: float = 0.2
    rotation_levels_deg: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1, 0.05)
    translation_levels_m: tuple[float, ...] = (0.02, 0.01, 0.005, 0.0025, 0.001)
    jackknife_first_level: int = 1
    max_moves_per_level: int = 12
    rotation_control_deg: float = 2.0
    translation_control_m: float = 0.05
    # A DoF is observed only when its held-out control reaches this t statistic.  On the 23
    # KITTI drives of the first camera-lidar audit, every axis judged more than 0.75 deg off
    # KITTI's calibration had its control at t 2.09 to 2.33 (2.0 was the bound then).
    min_control_t: float = DEFAULT_MIN_CONTROL_T
    observable_rotation_std_deg: float = 0.5
    observable_translation_std_m: float = 0.03
    min_ring_fraction: float = 0.5
    min_gain: float = 2.0e-5
    estimate_translation: bool = False
    # Wide-basin start: the objective is profiled along each rotation axis over
    # +-``start_scan_deg`` (the other axes at the candidate) and the search also runs from
    # each axis's maximum alone and from all of them together.  A candidate a few degrees
    # off often sits next to a local maximum the coordinate search cannot leave (KITTI
    # 0005 pitch +5 deg stops at +2).  Such a far optimum replaces the current one only
    # when it observes more rotation DoFs, at least ``start_scan_min_observed`` of them,
    # and has none at the search bound (see ``_replaces_current``).  ``None`` turns the
    # scan off.
    start_scan_deg: float | None = 8.0
    start_scan_step_deg: float = 0.5
    start_scan_min_observed: int = 2
    # Accept clouds without beam structure (solid-state, non-repetitive scanning): their
    # depth discontinuities are found among angular neighbours instead of along beams.
    unstructured_lidar: bool = True
    # Optional extra margin for adopting a profile-start optimum. Every rotation axis
    # it would report must meet this control t statistic, including axes already
    # observed at the current optimum. Unobserved axes do not veto it. None retains
    # the v2 adoption rule; 3.5 is a development proposal, not a validated default.
    start_scan_min_control_t: float | None = None
    preparation_workers: int = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.preparation_workers, bool)
            or not isinstance(self.preparation_workers, int)
            or not 1 <= self.preparation_workers <= 32
        ):
            raise ValueError("edge preparation workers must be an integer between 1 and 32")
        if len(self.rotation_levels_deg) != len(self.translation_levels_m):
            raise ValueError("rotation and translation search levels must match")
        if self.blocks < 3 or self.min_frames < self.blocks:
            raise ValueError("need at least three blocks and as many frames as blocks")
        if self.short_recording_spacing_s is not None and self.short_recording_spacing_s <= 0.0:
            raise ValueError("short_recording_spacing_s must be positive")
        if self.start_scan_deg is not None and not (
            0.0 < self.start_scan_step_deg <= self.start_scan_deg <= self.rotation_bound_deg
        ):
            raise ValueError("the start scan needs 0 < step <= range <= the rotation bound")
        if not 0 <= self.start_scan_min_observed <= 3:
            raise ValueError("start_scan_min_observed counts rotation DoFs: 0 to 3")
        if self.start_scan_min_control_t is not None and not (
            math.isfinite(self.start_scan_min_control_t)
            and self.start_scan_min_control_t >= self.min_control_t
        ):
            raise ValueError("start_scan_min_control_t must be finite and at least min_control_t")

    @property
    def effective_spacing_s(self) -> float:
        """Seconds between scans: spread ``max_frames`` over ``max_seconds`` when it is set."""

        if self.max_seconds is None:
            return self.spacing_s
        return max(self.spacing_s, self.max_seconds / self.max_frames)

    def as_dict(self) -> dict[str, Any]:
        """Schema-safe record of the options."""

        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in asdict(self).items()
        }


# --------------------------------------------------------------------------- inputs


def image_edge_response(
    gray: NDArray[np.uint8],
    *,
    max_width: int,
    gamma: float,
    radius: int,
    background_sigma: float | None = None,
) -> tuple[Float32Array, int] | None:
    """Edge image with inverse-distance spill (Levinson-Thrun Eq. 1) and its downscale factor.

    ``background_sigma`` (pixels of the downscaled image) subtracts the response's local
    mean, a Gaussian of that width: a point then scores for lying on an edge rather than
    in a textured region, so the objective does not drift towards the image's busiest
    part. ``None`` when the image carries no edges (flat or saturated).
    """

    factor = max(1, math.ceil(gray.shape[1] / max_width))
    image = gray.astype(np.float32)
    if factor > 1:
        height = (image.shape[0] // factor) * factor
        width = (image.shape[1] // factor) * factor
        image = image[:height, :width].reshape(height // factor, factor, width // factor, factor)
        image = image.mean(axis=(1, 3))
    image = ndimage.gaussian_filter(image, 1.0)
    magnitude = np.hypot(ndimage.sobel(image, axis=1), ndimage.sobel(image, axis=0))
    scale = float(np.percentile(magnitude, 99.0))
    if scale < 4.0:  # under about one gray level of gradient: nothing to align
        return None
    edge = np.minimum(magnitude / scale, 1.0).astype(np.float32)
    # The spill is computed on 2x2 max-pooled edges (half the radius, gamma squared per step):
    # it is a smooth, slowly varying field and this is four times cheaper per step.
    pooled = edge[: edge.shape[0] // 2 * 2, : edge.shape[1] // 2 * 2]
    pooled = pooled.reshape(pooled.shape[0] // 2, 2, pooled.shape[1] // 2, 2).max(axis=(1, 3))
    step = np.float32(gamma * gamma)
    spread_small = _edge_spill(pooled, step, max(1, radius // 2))
    spread = np.repeat(np.repeat(spread_small, 2, axis=0), 2, axis=1)
    padded = np.pad(
        spread,
        ((0, edge.shape[0] - spread.shape[0]), (0, edge.shape[1] - spread.shape[1])),
        mode="edge",
    )
    spread = np.maximum(padded, edge)
    response = edge / 3.0 + spread * (2.0 / 3.0)
    if background_sigma is not None:
        response = response - ndimage.gaussian_filter(response, background_sigma)
    return response.astype(np.float32), factor


def _edge_spill(pooled: Float32Array, step: np.float32, iterations: int) -> Float32Array:
    """The same rounded max-filter frontier with logarithmically many filters.

    Write F for one 3x3 maximum followed by one float32 multiplication. Both
    operations commute with maximum, and square maximum filters compose by
    adding radii (including at reflected boundaries). If S_a = max(F^0..F^a),
    then max(S_a, F^b(S_a)) covers every exponent 0..a+b whenever b <= a+1.
    Grow by 1, 2, 4, ... and the remainder. Each skipped filter is replaced by
    a wider one, but every float32 multiplication is retained, in order: using
    step**b would change rounding. Default twenty iterations need five filters.
    """

    spread = pooled.copy()
    if not np.isfinite(step) or not np.all(np.isfinite(pooled)):
        front = pooled.copy()
        for _ in range(iterations):
            front = step * ndimage.maximum_filter(front, size=3)
            np.maximum(spread, front, out=spread)
        return spread
    done = 0
    while done < iterations:
        width = min(done + 1, iterations - done)
        front = ndimage.maximum_filter(spread, size=2 * width + 1)
        for _ in range(width):
            np.multiply(step, front, out=front)
        np.maximum(spread, front, out=spread)
        done += width
    return spread


def _ring_centers(elevation_deg: FloatArray) -> FloatArray:
    low = float(elevation_deg.min())
    high = float(elevation_deg.max())
    if high - low < 1.0:
        return np.empty(0)
    counts, edges = np.histogram(elevation_deg, bins=np.arange(low, high + 0.05, 0.05))
    smooth = ndimage.gaussian_filter1d(counts.astype(np.float64), 1.0)
    positive = smooth[smooth > 0.0]
    if positive.size == 0:
        return np.empty(0)
    floor = 0.2 * float(np.median(positive))
    is_peak = (smooth > np.roll(smooth, 1)) & (smooth >= np.roll(smooth, -1)) & (smooth > floor)
    centers = 0.5 * (edges[:-1] + edges[1:])[is_peak]
    merged: list[float] = []
    for value in centers:
        if not merged or value - merged[-1] > 0.12:
            merged.append(float(value))
    return np.asarray(merged, dtype=np.float64)


def _nearest_ring_indices(elevation: FloatArray, centers: FloatArray) -> NDArray[np.intp]:
    """Preserve dense argmin's first-index ties with two sorted neighbors.

    Recovered beam centers are separated by more than 0.12 degrees. The much
    weaker spacing/range guard below also keeps rounded-distance ties from
    extending beyond the two adjacent centers. Unusual inputs retain argmin.
    """
    if (
        not np.all(np.isfinite(elevation))
        or not np.all(np.isfinite(centers))
        or np.max(np.abs(elevation), initial=0.0) > 180.0
        or np.max(np.abs(centers), initial=0.0) > 180.0
        or not np.all(np.diff(centers) > 1e-9)
    ):
        return np.abs(elevation[:, None] - centers[None, :]).argmin(axis=1)
    insertion = np.searchsorted(centers, elevation, side="left")
    left = np.maximum(insertion - 1, 0)
    right = np.minimum(insertion, len(centers) - 1)
    left_distance = np.abs(elevation - centers[left])
    right_distance = np.abs(elevation - centers[right])
    return np.where(left_distance <= right_distance, left, right)


def lidar_depth_edges(
    points: Float32Array,
    *,
    min_jump_m: float,
    exponent: float,
    max_points: int,
    min_ring_fraction: float,
    unstructured: bool = False,
) -> tuple[Float32Array, Float32Array, int, float]:
    """Near-side depth discontinuities along each beam (Levinson-Thrun Eq. 2).

    Beams are recovered from point elevation.  Returns ``(points, weights, ring count,
    fraction of points within 0.08 deg of a beam)``.  A cloud without beam structure (a
    solid-state, non-repetitive LiDAR) gives no edge points, unless ``unstructured`` is
    set: its discontinuities are then found among angular neighbours instead (ring count
    0), see :func:`_unstructured_depth_edges`.
    """

    xyz = np.asarray(points, dtype=np.float64)
    horizontal = np.hypot(xyz[:, 0], xyz[:, 1])
    ranges = np.sqrt(horizontal * horizontal + xyz[:, 2] * xyz[:, 2])
    keep = ranges > 0.5
    xyz, horizontal, ranges = xyz[keep], horizontal[keep], ranges[keep]
    empty = np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.float32)
    if xyz.shape[0] < 1000:
        return (*empty, 0, 0.0)
    elevation = np.degrees(np.arctan2(xyz[:, 2], horizontal))
    centers = _ring_centers(elevation)
    if centers.size < 8 or centers.size > 200:
        if not unstructured:
            return (*empty, int(centers.size), 0.0)
        edge_points, weights = _unstructured_depth_edges(
            xyz, ranges, min_jump_m=min_jump_m, exponent=exponent, max_points=max_points
        )
        return edge_points, weights, 0, 0.0
    ring = _nearest_ring_indices(elevation, centers)
    on_beam = float(np.mean(np.abs(elevation - centers[ring]) < 0.08))
    if on_beam < min_ring_fraction:
        if not unstructured:
            return (*empty, int(centers.size), on_beam)
        edge_points, weights = _unstructured_depth_edges(
            xyz, ranges, min_jump_m=min_jump_m, exponent=exponent, max_points=max_points
        )
        return edge_points, weights, 0, on_beam
    azimuth = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    order = np.lexsort((azimuth, ring))
    ring_s, azimuth_s, range_s = ring[order], azimuth[order], ranges[order]
    same = ring_s[1:] == ring_s[:-1]
    step = np.diff(azimuth_s)
    usable = same & (step > 0.0)
    if not np.any(usable):
        return (*empty, int(centers.size), on_beam)
    max_gap = float(np.clip(4.0 * np.median(step[usable]), 0.2, 2.0))
    linked = same & (step < max_gap)
    previous = np.concatenate(([False], linked))
    following = np.concatenate((linked, [False]))
    previous_range = np.concatenate(([range_s[0]], range_s[:-1]))
    next_range = np.concatenate((range_s[1:], [range_s[-1]]))
    jump = np.maximum.reduce(
        [
            np.where(previous, previous_range - range_s, 0.0),
            np.where(following, next_range - range_s, 0.0),
            np.zeros_like(range_s),
        ]
    )
    selected = jump >= min_jump_m
    edge_points = xyz[order][selected].astype(np.float32)
    weights = np.power(jump[selected], exponent).astype(np.float32)
    if edge_points.shape[0] > max_points:
        top = np.argsort(-weights, kind="stable")[:max_points]
        top.sort()
        edge_points, weights = edge_points[top], weights[top]
    return edge_points, weights, int(centers.size), on_beam


def _unstructured_depth_edges(
    xyz: FloatArray,
    ranges: FloatArray,
    *,
    min_jump_m: float,
    exponent: float,
    max_points: int,
    neighbours: int = 8,
    relative_jump: float = 0.1,
) -> tuple[Float32Array, Float32Array]:
    """Near-side depth discontinuities among the angular neighbours of each return.

    For a cloud without beams the jump of a return is the largest amount by which one of
    its ``neighbours`` nearest returns in direction (unit vectors, so no azimuth wrap) lies
    further away; neighbours further than four times the median neighbour spacing are
    ignored, so a sparse region does not pair returns across a gap.  Neighbours are
    further apart than consecutive returns of a beam, so a surface seen at a grazing
    angle changes range between them too: a jump must also reach ``relative_jump`` of
    the range.
    """

    from scipy.spatial import cKDTree

    directions = xyz / ranges[:, None]
    distance, index = cKDTree(directions).query(directions, k=neighbours + 1)
    distance, index = distance[:, 1:], index[:, 1:]
    spacing = float(np.median(distance[:, 0]))
    linked = distance <= 4.0 * spacing
    jump = np.where(linked, ranges[index] - ranges[:, None], 0.0).max(axis=1)
    selected = jump >= np.maximum(min_jump_m, relative_jump * ranges)
    edge_points = xyz[selected].astype(np.float32)
    weights = np.power(jump[selected], exponent).astype(np.float32)
    if edge_points.shape[0] > max_points:
        top = np.argsort(-weights, kind="stable")[:max_points]
        top.sort()
        edge_points, weights = edge_points[top], weights[top]
    return edge_points, weights


@dataclass(frozen=True)
class PreparedFrame:
    """A frame reduced to what the objective needs."""

    time_s: float
    gap_s: float
    edge: Float32Array
    scale: int
    points: Float32Array
    weights: Float32Array
    rings: int


# REP 103: a camera body frame is x forward, y left, z up; the optical frame that
# sensor_msgs/Image uses is z forward, x right, y down.  p_optical = R p_body.
REP103_OPTICAL_FROM_BODY: FloatArray = np.asarray(
    [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=np.float64
)
_BODY_FRAME_DIAGNOSIS_FRAMES = 16


def _body_frame_diagnosis(
    frames: Sequence[CameraLidarFrame],
    camera: EdgeCamera,
    candidate: FloatArray,
    options: EdgeAlignmentOptions,
    dropped: Sequence[str],
    camera_frame: str,
) -> list[str]:
    """Name an image published in the camera body frame when most frames see no edges.

    An image in the camera's REP 103 body frame (x forward) instead of its optical frame
    (z forward) puts the LiDAR outside the camera's view under any candidate. The frames
    are read once more with the optical rotation applied, counting the LiDAR edge points
    that would fall in the image; nothing is changed or assumed, the reason only says
    that this reading would make the frames usable.
    """

    out_of_view = sum("fall in the image under the candidate" in reason for reason in dropped)
    if not frames or out_of_view == 0 or out_of_view * 2 < len(dropped):
        return []
    sample = list(frames)[:_BODY_FRAME_DIAGNOSIS_FRAMES]
    edges = [
        lidar_depth_edges(
            frame.points,
            min_jump_m=options.min_jump_m,
            exponent=options.jump_exponent,
            max_points=options.max_edge_points_per_frame,
            min_ring_fraction=options.min_ring_fraction,
            unstructured=options.unstructured_lidar,
        )[0]
        for frame in sample
    ]

    def usable(transform: FloatArray) -> int:
        count = 0
        for frame, points in zip(sample, edges, strict=True):
            if points.shape[0] == 0:
                continue
            u, v, valid = camera.project(
                points.astype(np.float64) @ transform[:3, :3].T + transform[:3, 3]
            )
            height, width = frame.image.shape
            margin = options.image_margin_px
            inside = (
                valid
                & (u > -margin)
                & (v > -margin)
                & (u < width - 1 + margin)
                & (v < height - 1 + margin)
            )
            count += int(inside.sum()) >= options.min_edge_points_per_frame
        return count

    body = np.eye(4)
    body[:3, :3] = REP103_OPTICAL_FROM_BODY
    count = usable(body @ candidate)
    if count * 2 < len(sample) or count <= usable(candidate):
        return []
    return [
        f"the camera frame '{camera_frame}' looks like a REP 103 body frame (x forward, "
        "z up) rather than the optical frame (z forward, x right, y down) that "
        f"sensor_msgs/Image requires: read as a body frame, {count} of {len(sample)} "
        "frames put LiDAR edges in the image. Publish the image in the optical frame, or "
        "add the optical frame as a child of the body frame (rotation_quat_xyzw "
        "[0.5, -0.5, 0.5, -0.5]) in a slac.check_frames file passed with --tf and map the "
        "image topic to it with --frame-map TOPIC=FRAME"
    ]


_EdgeInputs: TypeAlias = tuple[Float32Array, int, Float32Array, Float32Array, int, float] | None


class _PreparationKeywords(TypedDict, total=False):
    preparation_cache: EdgePreparationCache


def _preparation_keywords(cache: EdgePreparationCache | None) -> _PreparationKeywords:
    # Existing input recorders forward solver keywords into schema-bound snapshots.
    # An unused runtime cache must not add a field to those recorded keyword sets.
    return {} if cache is None else {"preparation_cache": cache}


def _depth_edge_inputs(
    frame: CameraLidarFrame,
    options: EdgeAlignmentOptions,
    preparation_cache: EdgePreparationCache | None,
) -> tuple[Float32Array, Float32Array, int, float]:
    def compute() -> tuple[Float32Array, Float32Array, int, float]:
        return lidar_depth_edges(
            frame.points,
            min_jump_m=options.min_jump_m,
            exponent=options.jump_exponent,
            max_points=options.max_edge_points_per_frame,
            min_ring_fraction=options.min_ring_fraction,
            unstructured=options.unstructured_lidar,
        )

    if preparation_cache is None:
        return compute()
    return preparation_cache._get_or_compute(
        "lidar-depth-edges/v1",
        frame.points,
        (
            options.min_jump_m,
            options.jump_exponent,
            options.max_edge_points_per_frame,
            options.min_ring_fraction,
            options.unstructured_lidar,
        ),
        compute,
    )


def _prepare_edge_inputs(
    frame: CameraLidarFrame,
    options: EdgeAlignmentOptions,
    preparation_cache: EdgePreparationCache | None = None,
) -> _EdgeInputs:
    def compute() -> tuple[Float32Array, int] | None:
        return image_edge_response(
            frame.image,
            max_width=options.image_max_width,
            gamma=options.spill_gamma,
            radius=options.spill_radius_px,
            background_sigma=options.edge_background_sigma_px,
        )

    response = (
        compute()
        if preparation_cache is None
        else preparation_cache._get_or_compute(
            "image-edge-response/v1",
            frame.image,
            (
                options.image_max_width,
                options.spill_gamma,
                options.spill_radius_px,
                options.edge_background_sigma_px,
            ),
            compute,
        )
    )
    if response is None:
        return None
    points, weights, rings, on_beam = _depth_edge_inputs(frame, options, preparation_cache)
    edge, scale = response
    return edge, scale, points, weights, rings, on_beam


def _ordered_edge_inputs(
    frames: Sequence[CameraLidarFrame],
    options: EdgeAlignmentOptions,
    preparation_cache: EdgePreparationCache | None = None,
) -> Generator[_EdgeInputs, None, None]:
    def prepare(frame: CameraLidarFrame, selected: EdgeAlignmentOptions) -> _EdgeInputs:
        if preparation_cache is None:
            return _prepare_edge_inputs(frame, selected)
        return _prepare_edge_inputs(frame, selected, preparation_cache)

    workers = min(options.preparation_workers, len(frames))
    if workers <= 1:
        for frame in frames:
            yield prepare(frame, options)
        return
    from collections import deque
    from concurrent.futures import Future, ThreadPoolExecutor
    from itertools import islice

    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="calibrex-offline-edge")
    remaining = iter(frames)
    pending: deque[Future[_EdgeInputs]] = deque()
    try:
        pending.extend(
            executor.submit(prepare, frame, options)
            for frame in islice(remaining, workers)
        )
        while pending:
            result = pending.popleft().result()
            yield result
            # Limit retained futures/results to the worker count. Do not queue
            # the complete recording while earlier frames are being consumed.
            next_frame = next(remaining, None)
            if next_frame is not None:
                pending.append(executor.submit(prepare, next_frame, options))
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def prepare_frames(
    frames: Sequence[CameraLidarFrame],
    camera: EdgeCamera,
    candidate: FloatArray,
    options: EdgeAlignmentOptions,
    *,
    preparation_cache: EdgePreparationCache | None = None,
) -> tuple[list[PreparedFrame], list[str]]:
    """Edge images and in-view edge points of every usable frame, and why others were dropped."""

    prepared: list[PreparedFrame] = []
    dropped: list[str] = []
    emit_stage(f"camera-lidar: preparing {len(frames)} frame(s)")
    rotation = candidate[:3, :3]
    translation = candidate[:3, 3]
    last_view: Float32Array | None = None
    pending = _ordered_edge_inputs(frames, options, preparation_cache)
    try:
        for index, (frame, inputs) in enumerate(zip(frames, pending, strict=True)):
            emit_tick(index + 1, len(frames))
            if inputs is None:
                dropped.append(f"t={frame.time_s:.1f}s: the image has no edges")
                continue
            edge, scale, points, weights, rings, on_beam = inputs
            if points.shape[0] == 0:
                dropped.append(
                    f"t={frame.time_s:.1f}s: no beam structure in the cloud ({rings} elevation "
                    f"clusters, {on_beam:.0%} of points on a beam)"
                )
                continue
            camera_points = points.astype(np.float64) @ rotation.T + translation
            u, v, valid = camera.project(camera_points)
            margin = options.image_margin_px
            height, width = frame.image.shape
            inside = (
                valid
                & (u > -margin)
                & (v > -margin)
                & (u < width - 1 + margin)
                & (v < height - 1 + margin)
            )
            if int(inside.sum()) < options.min_edge_points_per_frame:
                dropped.append(
                    f"t={frame.time_s:.1f}s: only {int(inside.sum())} LiDAR edge points "
                    "fall in the image under the candidate"
                )
                continue
            view = frame.image[::4, ::4].astype(np.float32)
            if options.repeated_view_change is not None and rings > 0 and last_view is not None:
                change = float(np.mean(np.abs(view - last_view))) / 255.0
                if view.shape == last_view.shape and change < options.repeated_view_change:
                    dropped.append(
                        f"t={frame.time_s:.1f}s: repeats the last kept view (image change "
                        f"{change:.3f} < {options.repeated_view_change:g}; a spinning LiDAR at a "
                        "standing rig adds no view)"
                    )
                    continue
            last_view = view
            prepared.append(
                PreparedFrame(
                    time_s=frame.time_s,
                    gap_s=frame.gap_s,
                    edge=edge,
                    scale=scale,
                    points=points[inside],
                    weights=weights[inside],
                    rings=rings,
                )
            )
    finally:
        pending.close()
    return prepared, dropped


# ------------------------------------------------------------------------ objective


def _rotation_from_vector(vector_rad: FloatArray) -> FloatArray:
    return np.asarray(Rotation.from_rotvec(vector_rad).as_matrix(), dtype=np.float64)


class EdgeObjective:
    """Pooled edge-alignment score of a set of prepared frames.

    ``score(theta)`` is the weighted mean edge response at the projections of the
    frames' LiDAR edge points under ``[Exp(theta_rot) R_start | t_start + theta_t]``, where
    ``theta`` holds a rotation vector in degrees (a correction about the camera axes) and a
    translation in metres.  Points that leave the image count as zero, so the denominator is
    fixed.
    """

    def __init__(
        self, frames: Sequence[PreparedFrame], camera: EdgeCamera, start: FloatArray
    ) -> None:
        if not frames:
            raise ValueError("an objective needs at least one frame")
        self._source_frames = tuple(frames)
        self._source_start = start.copy()
        self.camera = camera
        rotation, translation = start[:3, :3], start[:3, 3]
        self._points = np.concatenate(
            [frame.points.astype(np.float64) @ rotation.T for frame in frames]
        ).astype(np.float32)
        self._start_translation = translation.astype(np.float32)
        self._weights = np.concatenate([frame.weights for frame in frames]).astype(np.float32)
        counts = [len(frame.points) for frame in frames]
        sizes = np.asarray([frame.edge.size for frame in frames], dtype=np.int64)
        offsets = np.concatenate(([0], np.cumsum(sizes)))[:-1]

        def per_point(values: Sequence[float] | NDArray[Any]) -> Float32Array:
            return np.repeat(np.asarray(values, dtype=np.float32), counts)

        self._shift = per_point([0.5 * (frame.scale - 1.0) for frame in frames])
        self._inverse_scale = per_point([1.0 / frame.scale for frame in frames])
        self._u_limit = per_point([frame.edge.shape[1] - 1 for frame in frames])
        self._v_limit = per_point([frame.edge.shape[0] - 1 for frame in frames])
        self._stride = np.repeat(
            np.asarray([frame.edge.shape[1] for frame in frames], dtype=np.int64), counts
        )
        self._offset = np.repeat(offsets, counts)
        self._flat = np.concatenate([frame.edge.ravel() for frame in frames])
        self._total_weight = float(self._weights.sum(dtype=np.float64))

    def score(self, theta: FloatArray) -> float:
        """The pooled objective at ``theta`` (degrees, metres)."""

        rotation = _rotation_from_vector(np.radians(theta[:3])).astype(np.float32)
        moved = self._points @ rotation.T
        moved += self._start_translation + theta[3:].astype(np.float32)
        u, v, valid = self.camera.project(moved)
        u -= self._shift
        u *= self._inverse_scale
        v -= self._shift
        v *= self._inverse_scale
        inside = valid & (u >= 0.0) & (v >= 0.0) & (u < self._u_limit) & (v < self._v_limit)
        if not np.any(inside):
            return 0.0
        u, v = u[inside], v[inside]
        left = u.astype(np.int64)
        top = v.astype(np.int64)
        du = u - left
        dv = v - top
        stride = self._stride[inside]
        base = self._offset[inside] + top * stride + left
        flat = self._flat
        response = (1.0 - du) * ((1.0 - dv) * flat[base] + dv * flat[base + stride]) + du * (
            (1.0 - dv) * flat[base + 1] + dv * flat[base + stride + 1]
        )
        return float(response @ self._weights[inside]) / self._total_weight


@dataclass(frozen=True)
class _EdgeScoreRequest:
    """One scalar or an independent, ordered set of maximum candidates.

    The synchronous driver evaluates every pose in its original order. An async
    driver can batch the candidates, but must return the selected index and its
    confirmed score. Consecutive line-search moves remain separate requests.
    """

    objective: EdgeObjective
    theta: FloatArray
    select_maximum: bool = False

    def evaluate(self) -> float | tuple[int, float]:
        values = [self.objective.score(pose) for pose in self.theta]
        if self.select_maximum:
            index = int(np.argmax(values))
            return index, values[index]
        return float(values[0])


_EdgeReturn = TypeVar("_EdgeReturn")
_EdgeReply: TypeAlias = float | tuple[int, float]
_EdgeWorkflow: TypeAlias = Generator[_EdgeScoreRequest, _EdgeReply, _EdgeReturn]


def _drive_edge_workflow(
    workflow: _EdgeWorkflow[_EdgeReturn],
    *,
    observer: Callable[[_EdgeScoreRequest, _EdgeReply], None] | None = None,
) -> _EdgeReturn:
    try:
        request = next(workflow)
        while True:
            reply = request.evaluate()
            if observer is not None:
                observer(request, reply)
            request = workflow.send(reply)
    except StopIteration as done:
        return done.value  # type: ignore[no-any-return]
    finally:
        workflow.close()


def _edge_score(objective: EdgeObjective, theta: FloatArray) -> _EdgeWorkflow[float]:
    reply = yield _EdgeScoreRequest(objective, theta[np.newaxis, :])
    assert isinstance(reply, float)
    return reply


def _edge_maximum(objective: EdgeObjective, theta: FloatArray) -> _EdgeWorkflow[tuple[int, float]]:
    reply = yield _EdgeScoreRequest(objective, theta, select_maximum=True)
    assert isinstance(reply, tuple)
    return reply


def search(
    objective: EdgeObjective,
    start: FloatArray,
    options: EdgeAlignmentOptions,
    *,
    axes: Sequence[int],
    first_level: int = 0,
) -> tuple[FloatArray, float]:
    """Coordinate search over ``axes``, coarse to fine, inside the bounds."""

    return _drive_edge_workflow(
        _search_workflow(objective, start, options, axes=axes, first_level=first_level)
    )


def _search_workflow(
    objective: EdgeObjective,
    start: FloatArray,
    options: EdgeAlignmentOptions,
    *,
    axes: Sequence[int],
    first_level: int = 0,
) -> _EdgeWorkflow[tuple[FloatArray, float]]:
    theta = start.astype(np.float64).copy()
    best = yield from _edge_score(objective, theta)
    bounds = np.array([options.rotation_bound_deg] * 3 + [options.translation_bound_m] * 3)
    levels = list(zip(options.rotation_levels_deg, options.translation_levels_m, strict=True))
    for rotation_step, translation_step in levels[first_level:]:
        steps = np.array([rotation_step] * 3 + [translation_step] * 3)
        for _move in range(options.max_moves_per_level):
            # Steepest coordinate move: coarse steps on weakly constrained axes must not run
            # ahead of the strongly constrained ones, or the search drifts along valleys.
            trials: list[FloatArray] = []
            previous = theta
            for axis in axes:
                for sign in (-1.0, 1.0):
                    trial = theta.copy()
                    trial[axis] += sign * steps[axis]
                    if abs(trial[axis]) > bounds[axis]:
                        continue
                    trials.append(trial)
            if not trials:
                break
            index, value = yield from _edge_maximum(objective, np.asarray(trials))
            if value <= best + options.min_gain:
                break
            best, theta = value, trials[index]
            # Keep going in the direction that worked while it keeps improving.
            direction = theta - previous
            while True:
                trial = theta + direction
                if np.any(np.abs(trial) > bounds):
                    break
                value = yield from _edge_score(objective, trial)
                if value <= best + options.min_gain:
                    break
                best, theta = value, trial
    return theta, best


def fit_extrinsic(
    objective: EdgeObjective,
    start: FloatArray,
    options: EdgeAlignmentOptions,
    *,
    first_level: int = 0,
) -> tuple[FloatArray, float]:
    """Rotation first (translation held at the start), then translation at that rotation.

    Translation trades against rotation along flat valleys of the objective, so the
    rotation is not allowed to move with it: the rotation DoFs answer "is the rotation
    right for the candidate's translation", the translation DoFs "is the translation
    right for the estimated rotation".
    """

    return _drive_edge_workflow(_fit_workflow(objective, start, options, first_level=first_level))


def _fit_workflow(
    objective: EdgeObjective,
    start: FloatArray,
    options: EdgeAlignmentOptions,
    *,
    first_level: int = 0,
) -> _EdgeWorkflow[tuple[FloatArray, float]]:
    theta, _ = yield from _search_workflow(
        objective, start, options, axes=(0, 1, 2), first_level=first_level
    )
    if not options.estimate_translation:
        return theta, (yield from _edge_score(objective, theta))
    return (
        yield from _search_workflow(
            objective, theta, options, axes=(3, 4, 5), first_level=first_level
        )
    )


# ---------------------------------------------------------------------- the estimate


def _transform(start: FloatArray, theta: FloatArray) -> FloatArray:
    """``[Exp(theta_rot) R_start | t_start + theta_t]``: the rotation correction is applied on
    the left (about the camera axes) and does not move the translation."""

    result = start.copy()
    result[:3, :3] = _rotation_from_vector(np.radians(theta[:3])) @ start[:3, :3]
    result[:3, 3] = start[:3, 3] + theta[3:]
    return np.asarray(result, dtype=np.float64)


def _transform_record(matrix: FloatArray, parent: str, child: str) -> CameraLidarEdgeTransform:
    return CameraLidarEdgeTransform(
        parent_frame=parent,
        child_frame=child,
        translation_m=[float(value) for value in matrix[:3, 3]],
        rotation_quat_xyzw=[
            float(value) for value in Rotation.from_matrix(matrix[:3, :3]).as_quat()
        ],
    )


def _block_slices(count: int, blocks: int) -> list[range]:
    edges = np.linspace(0, count, blocks + 1).round().astype(int)
    return [range(int(edges[i]), int(edges[i + 1])) for i in range(blocks)]


def estimate_from_frames(
    frames: Sequence[CameraLidarFrame],
    camera: EdgeCamera,
    candidate: FloatArray,
    *,
    options: EdgeAlignmentOptions | None = None,
    provenance: CameraLidarEdgeProvenance,
    parent_frame: str = "camera",
    child_frame: str = "lidar",
    reference_label: str | None = None,
    start_note: str | None = None,
    motion_compensation: CameraLidarEdgeMotion | None = None,
    preparation_cache: EdgePreparationCache | None = None,
) -> CameraLidarEdgeArtifact:
    """Estimate ``T_camera_lidar`` from synchronized frames, starting at ``candidate``.

    ``motion_compensation`` records how the frames' points were moved to the image time
    (see :mod:`calibrex.evaluation.camera_lidar_motion`); ``None`` means not at all.
    A caller-owned ``preparation_cache`` can reuse image/depth edges when checking
    another candidate; selection, search and evidence still run for that candidate.
    """

    return _drive_edge_workflow(
        _estimate_workflow(
            frames,
            camera,
            candidate,
            options=options,
            provenance=provenance,
            parent_frame=parent_frame,
            child_frame=child_frame,
            reference_label=reference_label,
            start_note=start_note,
            motion_compensation=motion_compensation,
            preparation_cache=preparation_cache,
        )
    )


def _estimate_workflow(
    frames: Sequence[CameraLidarFrame],
    camera: EdgeCamera,
    candidate: FloatArray,
    *,
    options: EdgeAlignmentOptions | None = None,
    provenance: CameraLidarEdgeProvenance,
    parent_frame: str = "camera",
    child_frame: str = "lidar",
    reference_label: str | None = None,
    start_note: str | None = None,
    motion_compensation: CameraLidarEdgeMotion | None = None,
    preparation_cache: EdgePreparationCache | None = None,
) -> _EdgeWorkflow[CameraLidarEdgeArtifact]:
    opts = options or EdgeAlignmentOptions()
    prepared, dropped = prepare_frames(
        frames, camera, candidate, opts, **_preparation_keywords(preparation_cache)
    )
    limitations = list(LIMITATIONS)
    limitations.insert(
        1,
        motion_limitation(motion_compensation)
        if motion_compensation is not None
        else RIGID_SCAN_LIMITATION,
    )
    if start_note:
        limitations.append(start_note)
    samples = CameraLidarEdgeSamples(
        frames=len(prepared),
        edge_points=int(sum(len(frame.points) for frame in prepared)),
        blocks=opts.blocks,
        median_abs_time_gap_s=float(np.median([abs(f.gap_s) for f in prepared]))
        if prepared
        else None,
        max_abs_time_gap_s=float(max(abs(f.gap_s) for f in prepared)) if prepared else None,
        first_frame_time_s=prepared[0].time_s if prepared else None,
        last_frame_time_s=prepared[-1].time_s if prepared else None,
        median_rings=int(np.median([f.rings for f in prepared])) if prepared else None,
    )
    start_record = _transform_record(candidate, parent_frame, child_frame)
    if len(prepared) < opts.min_frames:
        unsolved_reasons = [
            f"only {len(prepared)} usable frame(s) of {len(frames)} synchronized; "
            f"at least {opts.min_frames} are needed",
            *_body_frame_diagnosis(frames, camera, candidate, opts, dropped, parent_frame),
            *dropped[:5],
        ]
        zero = Rotation.from_matrix(candidate[:3, :3]).as_rotvec()
        unsolved_dofs = [
            CameraLidarEdgeDofRecord(
                name=name,
                unit="deg" if index < 3 else "m",
                value=float(math.degrees(zero[index]))
                if index < 3
                else float(candidate[index - 3, 3]),
                delta_from_start=0.0,
                std_jackknife=0.0,
                std_reported=0.0,
                status="unobservable",
            )
            for index, name in enumerate(DOF_NAMES)
        ]
        return CameraLidarEdgeArtifact(
            solver_status="insufficient_samples",
            policy_status="inconclusive",
            policy_reasons=unsolved_reasons,
            calibrated_dofs=[],
            start_transform=start_record,
            dofs=unsolved_dofs,
            samples=samples,
            motion_compensation=motion_compensation,
            jackknife_fits=0,
            camera=camera.as_dict(),
            options=opts.as_dict(),
            reference=reference_label,
            limitations=limitations,
            provenance=provenance,
        )

    objective = EdgeObjective(prepared, camera, candidate)
    zero_theta = np.zeros(6)
    start_score = yield from _edge_score(objective, zero_theta)
    emit_stage("camera-lidar: maximising the edge-alignment objective")
    theta, score = yield from _fit_workflow(objective, zero_theta, opts)
    blocks = _block_slices(len(prepared), opts.blocks)
    block_objectives = [
        EdgeObjective([prepared[i] for i in block], camera, candidate) for block in blocks
    ]
    std_floor = 0.5 * np.array(
        [opts.rotation_levels_deg[-1]] * 3 + [opts.translation_levels_m[-1]] * 3
    )
    fit_array, jackknife, controls = yield from _evidence_workflow(
        prepared, camera, candidate, theta, blocks, block_objectives, opts
    )
    start_reason: str | None = None
    observed = _observed_rotation_axes(theta, jackknife, controls, std_floor, opts)
    candidate_observed = len(observed)
    tried = [theta[:3].copy()]
    for scan_start in _profile_starts((yield from _profile_workflow(objective, opts))):
        emit_stage(
            "camera-lidar: searching from the profile maxima "
            f"({', '.join(f'{v:+.1f}' for v in scan_start[:3])} deg)"
        )
        widened, widened_score = yield from _fit_workflow(objective, scan_start, opts)
        if any(
            np.all(np.abs(widened[:3] - seen) <= opts.rotation_levels_deg[-1]) for seen in tried
        ):
            continue  # an optimum already weighed
        tried.append(widened[:3].copy())
        widened_evidence = yield from _evidence_workflow(
            prepared, camera, candidate, widened, blocks, block_objectives, opts
        )
        widened_observed = _observed_rotation_axes(
            widened, widened_evidence[1], widened_evidence[2], std_floor, opts
        )
        if _replaces_current(observed, widened_observed, widened, widened_evidence[2], opts):
            start_reason = (
                "the estimate comes from the search started at the profile maxima "
                f"({', '.join(f'{v:+.1f}' for v in scan_start[:3])} deg): it observes "
                f"{len(widened_observed)} rotation DoF(s) against {candidate_observed} from the "
                "candidate"
            )
            theta, score, observed = widened, widened_score, widened_observed
            fit_array, jackknife, controls = widened_evidence
    estimate = _transform(candidate, theta)

    reference_vector = Rotation.from_matrix(estimate[:3, :3]).as_rotvec()
    dofs: list[CameraLidarEdgeDofRecord] = []
    for index, name in enumerate(DOF_NAMES):
        rotation_axis = index < 3
        unit: Literal["deg", "m"] = "deg" if rotation_axis else "m"
        value = (
            float(math.degrees(reference_vector[index]))
            if rotation_axis
            else float(estimate[index - 3, 3])
        )
        delta = (
            float(theta[index])
            if rotation_axis
            else float(estimate[index - 3, 3] - candidate[index - 3, 3])
        )
        control = controls[index]
        std_bound = (
            opts.observable_rotation_std_deg if rotation_axis else opts.observable_translation_std_m
        )
        bound = opts.rotation_bound_deg if rotation_axis else opts.translation_bound_m
        at_bound = bool(abs(theta[index]) >= bound - 1e-9)
        std = float(max(jackknife[index], std_floor[index]))
        status: Literal["estimated", "unobservable"] = (
            "estimated"
            if control.detected
            and std <= std_bound
            and not at_bound
            and (rotation_axis or opts.estimate_translation)
            else "unobservable"
        )
        dofs.append(
            CameraLidarEdgeDofRecord(
                name=name,
                unit=unit,
                value=value,
                delta_from_start=delta,
                std_jackknife=std,
                std_reported=std,
                status=status,
                at_search_bound=at_bound,
                known_bad_control=control,
            )
        )
    candidate_vector = Rotation.from_matrix(candidate[:3, :3]).as_rotvec()
    reference_rotation = Rotation.from_matrix(candidate[:3, :3])
    error_vector = (Rotation.from_matrix(estimate[:3, :3]) * reference_rotation.inv()).as_rotvec()
    for index, record in enumerate(dofs):
        if index < 3:
            dofs[index] = record.model_copy(
                update={
                    "reference_value": float(math.degrees(candidate_vector[index])),
                    "error_to_reference": float(math.degrees(error_vector[index])),
                }
            )
        else:
            dofs[index] = record.model_copy(
                update={
                    "reference_value": float(candidate[index - 3, 3]),
                    "error_to_reference": float(estimate[index - 3, 3] - candidate[index - 3, 3]),
                }
            )

    estimated = [d.name for d in dofs if d.status == "estimated"]
    rotation_estimated = [name for name in estimated if name in {"roll", "pitch", "yaw"}]
    reasons: list[str] = []
    for record in dofs:
        if record.status == "estimated":
            continue
        held_out = record.known_bad_control
        if record.name in {"x", "y", "z"} and not opts.estimate_translation:
            why = "translation is not estimated (parallax resolves it too weakly; see limitations)"
        elif record.at_search_bound:
            why = "reached the search bound"
        elif held_out is not None and held_out.detected:
            why = f"jackknife std {record.std_reported:.3g} {record.unit} exceeds its bound"
        else:
            why = "its known-bad control was not detected on held-out frames"
        reasons.append(f"{record.name}: {why}")
    if len(rotation_estimated) == 3:
        policy: Literal["pass", "warn", "inconclusive"] = "pass"
        reasons.insert(0, "all three rotation DoFs are observed with detected held-out controls")
    elif rotation_estimated:
        policy = "warn"
        reasons.insert(0, f"only {', '.join(rotation_estimated)} of the rotation DoFs are observed")
    else:
        policy = "inconclusive"
        reasons.insert(0, "no rotation DoF is observed with a detected held-out control")
    if start_reason is not None:
        reasons.append(start_reason)
    if dropped:
        reasons.append(f"{len(dropped)} frame(s) dropped: {dropped[0]}")
    return CameraLidarEdgeArtifact(
        solver_status="converged",
        policy_status=policy,
        policy_reasons=reasons,
        calibrated_dofs=[d.name for d in dofs if d.status == "estimated"],
        transform=_transform_record(estimate, parent_frame, child_frame),
        start_transform=start_record,
        dofs=dofs,
        objective=CameraLidarEdgeObjective(
            start=float(start_score), estimate=float(score), gain=float(score - start_score)
        ),
        samples=samples,
        motion_compensation=motion_compensation,
        jackknife_fits=len(fit_array),
        camera=camera.as_dict(),
        options=opts.as_dict(),
        reference=reference_label,
        limitations=limitations,
        provenance=provenance,
    )


def _evidence(
    prepared: Sequence[PreparedFrame],
    camera: EdgeCamera,
    candidate: FloatArray,
    theta: FloatArray,
    blocks: Sequence[range],
    block_objectives: Sequence[EdgeObjective],
    options: EdgeAlignmentOptions,
) -> tuple[FloatArray, FloatArray, list[CameraLidarEdgeControl]]:
    """Leave-one-block-out fits around ``theta``, their jackknife std and the controls."""

    return _drive_edge_workflow(
        _evidence_workflow(prepared, camera, candidate, theta, blocks, block_objectives, options)
    )


def _evidence_workflow(
    prepared: Sequence[PreparedFrame],
    camera: EdgeCamera,
    candidate: FloatArray,
    theta: FloatArray,
    blocks: Sequence[range],
    block_objectives: Sequence[EdgeObjective],
    options: EdgeAlignmentOptions,
) -> _EdgeWorkflow[tuple[FloatArray, FloatArray, list[CameraLidarEdgeControl]]]:
    fits: list[FloatArray] = []
    for index, block in enumerate(blocks):
        emit_stage(f"camera-lidar: jackknife fit {index + 1}/{len(blocks)}")
        held = set(block)
        kept = [prepared[i] for i in range(len(prepared)) if i not in held]
        fold = EdgeObjective(kept, camera, candidate)
        fold_theta, _ = yield from _fit_workflow(
            fold, theta, options, first_level=options.jackknife_first_level
        )
        fits.append(fold_theta)
    fit_array = np.asarray(fits)
    # Jackknife of the estimate's own components: rotation about the camera axes (the
    # left correction) and the translation of the composed transform.
    components = np.empty_like(fit_array)
    for row, fit in enumerate(fit_array):
        components[row, :3] = fit[:3]
        components[row, 3:] = _transform(candidate, fit)[:3, 3]
    count = len(blocks)
    jackknife = np.sqrt(
        (count - 1) / count * np.sum((components - components.mean(axis=0)) ** 2, axis=0)
    )
    return (
        fit_array,
        jackknife,
        (yield from _controls_workflow(block_objectives, fit_array, options)),
    )


def _observed_rotation_axes(
    theta: FloatArray,
    jackknife: FloatArray,
    controls: Sequence[CameraLidarEdgeControl],
    std_floor: FloatArray,
    options: EdgeAlignmentOptions,
) -> tuple[int, ...]:
    """Rotation DoFs that would be reported as estimated (the rule of the DoF records)."""

    return tuple(
        axis
        for axis in range(3)
        if controls[axis].detected
        and max(float(jackknife[axis]), float(std_floor[axis]))
        <= options.observable_rotation_std_deg
        and abs(float(theta[axis])) < options.rotation_bound_deg - 1e-9
    )


def _profile_start(objective: EdgeObjective, options: EdgeAlignmentOptions) -> FloatArray | None:
    """Per rotation axis, the maximum of the objective profile through the candidate."""

    return _drive_edge_workflow(_profile_workflow(objective, options))


def _profile_workflow(
    objective: EdgeObjective, options: EdgeAlignmentOptions
) -> _EdgeWorkflow[FloatArray | None]:
    if options.start_scan_deg is None:
        return None
    emit_stage("camera-lidar: profiling the objective along each rotation axis")
    offsets = np.arange(
        -options.start_scan_deg,
        options.start_scan_deg + 0.5 * options.start_scan_step_deg,
        options.start_scan_step_deg,
    )
    start = np.zeros(6)
    for axis in range(3):
        poses = np.zeros((len(offsets), 6))
        poses[:, axis] = offsets
        index, _score = yield from _edge_maximum(objective, poses)
        start[axis] = float(offsets[index])
    return start


def _replaces_current(
    observed: Sequence[int],
    widened_observed: Sequence[int],
    widened: FloatArray,
    controls: Sequence[CameraLidarEdgeControl],
    options: EdgeAlignmentOptions,
) -> bool:
    """Whether a far optimum (from a profile-maxima start) replaces the current one.

    A far optimum can be a ridge of the scene rather than the calibration (a corridor),
    and the objective alone does not tell them apart.  It replaces the current optimum only
    when more rotation DoFs pass their held-out controls and std bound there, at least
    ``start_scan_min_observed`` of them, and none of its rotation DoFs is at the search
    bound.  On KITTI the earlier rule (more DoFs observed, or as many with a higher
    objective) let wrong optima in: on 0029 one that tied the candidate's count won on the
    objective with yaw 9.6 deg off; on 0056 one with yaw at the search bound replaced a
    candidate optimum that observed nothing.  With those two excluded, one observing a
    single DoF still judged roll 6.6 deg off in a turned run of 0056.

    An optional stricter control margin applies to every reported rotation axis of
    the far optimum. This is an adoption gate, not a change to the ordinary DoF
    observability rule: rejecting the far optimum retains the current fit and its
    evidence. It must not remove axes from the rejected fit and report that fit.
    """

    if np.any(np.abs(widened[:3]) >= options.rotation_bound_deg - 1e-9):
        return False
    if not (
        len(widened_observed) > len(observed)
        and len(widened_observed) >= options.start_scan_min_observed
    ):
        return False
    margin = options.start_scan_min_control_t
    return margin is None or all(
        (t := controls[axis].min_t_statistic) is not None and math.isfinite(t) and t >= margin
        for axis in widened_observed
    )


def _profile_starts(scan_start: FloatArray | None) -> list[FloatArray]:
    """Starts from the profile maxima: each axis's maximum on its own, then all of them.

    An axis's profile is taken with the other axes at the candidate, so its maximum is a
    start for that axis alone: on KITTI a camera turned 3 deg about one axis was found
    from that axis's maximum, while the search from all the maxima together (and from the
    candidate) ran along a ridge to a wrong optimum with fewer axes observed.
    """

    if scan_start is None:
        return []
    starts: list[FloatArray] = []
    for axis in range(3):
        if scan_start[axis] != 0.0:
            single = np.zeros(6)
            single[axis] = scan_start[axis]
            starts.append(single)
    if len(starts) > 1:
        starts.append(scan_start.copy())
    return starts


def _controls(
    block_objectives: Sequence[EdgeObjective], fits: FloatArray, options: EdgeAlignmentOptions
) -> list[CameraLidarEdgeControl]:
    """Per DoF: on held-out blocks, the fit of the other blocks must beat shifted fits.

    Block ``k`` is scored at ``theta_k`` (the fit without block ``k``) and at ``theta_k``
    shifted by the control amount on one DoF, once in each direction.  For each
    direction the per-block score drops are tested with a one-sided t statistic over the
    blocks; the control is detected when both directions reach ``min_control_t``.
    """

    return _drive_edge_workflow(_controls_workflow(block_objectives, fits, options))


def _controls_workflow(
    block_objectives: Sequence[EdgeObjective], fits: FloatArray, options: EdgeAlignmentOptions
) -> _EdgeWorkflow[list[CameraLidarEdgeControl]]:
    base = []
    for objective, fit in zip(block_objectives, fits, strict=True):
        base.append((yield from _edge_score(objective, fit)))
    count = len(block_objectives)
    result: list[CameraLidarEdgeControl] = []
    for axis in range(6):
        amount = options.rotation_control_deg if axis < 3 else options.translation_control_m
        drops = np.empty((2, count))
        for block, (objective, fit) in enumerate(zip(block_objectives, fits, strict=True)):
            for row, sign in enumerate((-1.0, 1.0)):
                shifted = fit.copy()
                shifted[axis] += sign * amount
                drops[row, block] = base[block] - (yield from _edge_score(objective, shifted))
        means = drops.mean(axis=1)
        errors = drops.std(axis=1, ddof=1) / math.sqrt(count)
        statistics = np.where(errors > 0.0, means / np.where(errors > 0.0, errors, 1.0), 0.0)
        worst = int(np.argmin(statistics))
        result.append(
            CameraLidarEdgeControl(
                amount=amount,
                unit="deg" if axis < 3 else "m",
                held_out_blocks=count,
                detected_blocks=int(np.sum(drops.min(axis=0) > 0.0)),
                mean_objective_drop=float(means[worst]),
                min_t_statistic=float(statistics[worst]),
                detected=bool(statistics.min() >= options.min_control_t and means.min() > 0.0),
            )
        )
    return result


def default_provenance(
    *,
    bag: str | Path,
    image_topic: str,
    lidar_topic: str,
    dataset_family: str,
    dataset_license: str,
    command: Sequence[str],
    input_sha256: str | None = None,
    input_digest_scope: str | None = None,
) -> CameraLidarEdgeProvenance:
    """Provenance record of a run on one bag (a caller's own bag digest may be given)."""

    digest, scope = bag_input_digest([Path(bag)])
    digest = input_sha256 or digest
    scope = input_digest_scope or scope
    return CameraLidarEdgeProvenance(
        generator="calibrex.evaluation.camera_lidar_edge",
        generator_version=__version__,
        git_commit=git_commit(),
        command=list(command),
        dataset_family=dataset_family,
        sequence_ids=[Path(bag).name],
        image_topic=image_topic,
        lidar_topic=lidar_topic,
        input_sha256=digest,
        input_digest_scope=scope,
        dataset_license=dataset_license,
    )


def visible_time_rule(
    camera: EdgeCamera, candidate: FloatArray
) -> Callable[[Float32Array, FloatArray | None], float]:
    """Reference-time rule: the median per-point time of the points the camera sees."""

    width = 2.0 * camera.cx
    height = 2.0 * camera.cy

    def rule(points: Float32Array, times: FloatArray | None) -> float:
        if times is None or times.shape[0] != points.shape[0]:
            return 0.0
        moved = points.astype(np.float64) @ candidate[:3, :3].T + candidate[:3, 3]
        u, v, valid = camera.project(moved)
        inside = valid & (u >= 0.0) & (v >= 0.0) & (u < width) & (v < height)
        if int(inside.sum()) < 50:
            return 0.0
        return float(np.median(times[inside]))

    return rule


@dataclass(frozen=True)
class CameraLidarRunInputs:
    """What :func:`run_ros2_camera_lidar_edge` needs besides the bag."""

    image_topic: str
    lidar_topic: str
    camera: EdgeCamera
    candidate: FloatArray
    point_time: tuple[str, PointTimeEncoding] | None = None
    dataset_family: str = "calibrex-check"
    dataset_license: str = "user-provided"
    parent_frame: str = "camera"
    child_frame: str = "lidar"
    reference: str | None = None
    command: Sequence[str] = field(default_factory=tuple)
    input_sha256: str | None = None
    input_digest_scope: str | None = None
    motion: LidarMotion | None = None
    """The rig's motion in the LiDAR frame; when given, points are moved to the image time."""


def run_ros2_camera_lidar_edge(
    bag: str | Path,
    inputs: CameraLidarRunInputs,
    options: EdgeAlignmentOptions | None = None,
    *,
    preparation_cache: EdgePreparationCache | None = None,
) -> CameraLidarEdgeArtifact:
    """Read synchronized frames from a bag and estimate ``T_camera_lidar`` from the candidate."""

    opts = options or EdgeAlignmentOptions()
    emit_stage(f"camera-lidar: pairing {inputs.image_topic} with {inputs.lidar_topic}")
    selection: dict[str, object] = {}
    frames = load_camera_lidar_frames(
        bag,
        inputs.image_topic,
        inputs.lidar_topic,
        spacing_s=opts.effective_spacing_s,
        short_recording_spacing_s=opts.short_recording_spacing_s,
        short_recording_min_view_change=opts.short_recording_min_view_change,
        report=selection,
        max_frames=opts.max_frames,
        max_seconds=opts.max_seconds,
        max_gap_s=opts.max_gap_s,
        point_time=inputs.point_time,
        reference_time=visible_time_rule(inputs.camera, inputs.candidate)
        if inputs.point_time is not None
        else None,
    )
    note = (
        "image and scan are paired at the time the camera's visible points were measured "
        f"(per-point time field '{inputs.point_time[0]}')"
        if inputs.point_time is not None
        else "the cloud has no per-point time: image and scan are paired at the header stamp"
    )
    change = selection.get("median_view_change")
    change_text = f"{change:.3f}" if isinstance(change, float) else "unknown"
    if selection.get("selection") == "short_recording":
        note += (
            f"; the recording is shorter than {opts.max_frames} scans "
            f"{opts.effective_spacing_s:g} s apart, so up to {opts.max_frames} scans at least "
            f"{opts.short_recording_spacing_s:g} s apart were spread over it (median view "
            f"change {change_text})"
        )
    elif selection.get("selection") == "regular_static_view":
        note += (
            f"; the recording is short but its view hardly changes (median view change "
            f"{change_text}), so scans stay "
            f"{opts.effective_spacing_s:g} s apart: denser frames would repeat it"
        )
    motion_summary: CameraLidarEdgeMotion | None = None
    if inputs.motion is not None:
        frames, motion_summary = move_frames_to_image_time(frames, inputs.motion)
    return estimate_from_frames(
        frames,
        inputs.camera,
        np.asarray(inputs.candidate, dtype=np.float64),
        options=opts,
        provenance=default_provenance(
            bag=bag,
            image_topic=inputs.image_topic,
            lidar_topic=inputs.lidar_topic,
            dataset_family=inputs.dataset_family,
            dataset_license=inputs.dataset_license,
            command=inputs.command,
            input_sha256=inputs.input_sha256,
            input_digest_scope=inputs.input_digest_scope,
        ),
        parent_frame=inputs.parent_frame,
        child_frame=inputs.child_frame,
        reference_label=inputs.reference,
        start_note=note,
        motion_compensation=motion_summary,
        **_preparation_keywords(preparation_cache),
    )
