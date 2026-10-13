"""Rectify an equidistant (Kannala-Brandt) fisheye stereo pair for stereo depth.

:mod:`calibrex.evaluation.stereo_lidar_depth` needs a rectified pair: two pinhole cameras
with the same orientation, side by side along x. Kalibr calibrates fisheye pairs (the
Hilti 2022 alphasense, for example) with the ``equidistant`` model,

    theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8),

where ``theta`` is a ray's angle from the optical axis. This module projects and
unprojects that model (numpy only), builds the rectifying rotations from the pair's
extrinsic (x along the baseline, z the mean optical axis), and resamples each image onto
a shared pinhole camera. The result carries the rectified pinhole projections in the
rectified left frame and the rotation from the left camera into it, so LiDAR points in
the left camera's frame can be judged with the stereo-depth estimators unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeAlias

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from scipy import ndimage

from calibrex.evaluation.stereo_lidar_depth import RectifiedStereo

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class EquidistantCamera:
    """A fisheye camera with the equidistant (Kannala-Brandt) distortion model."""

    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, float, float, float]
    width: int
    height: int

    def project(self, points: FloatArray) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
        """Pixels ``(u, v)`` of camera-frame points, and which are in front of the camera."""

        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        radius = np.hypot(x, y)
        theta = np.arctan2(radius, z)
        k1, k2, k3, k4 = self.distortion
        t2 = theta * theta
        distorted = theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
        scale = np.divide(distorted, radius, out=np.ones_like(radius), where=radius > 1.0e-12)
        return self.fx * x * scale + self.cx, self.fy * y * scale + self.cy, z > 0.0

    def unproject(self, u: FloatArray, v: FloatArray) -> FloatArray:
        """Unit rays (camera frame) of pixels, inverting the distortion by Newton steps."""

        mx = (np.asarray(u, dtype=np.float64) - self.cx) / self.fx
        my = (np.asarray(v, dtype=np.float64) - self.cy) / self.fy
        distorted = np.hypot(mx, my)
        k1, k2, k3, k4 = self.distortion
        theta = distorted.copy()
        for _ in range(10):
            t2 = theta * theta
            value = theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - distorted
            slope = 1.0 + t2 * (3.0 * k1 + t2 * (5.0 * k2 + t2 * (7.0 * k3 + t2 * 9.0 * k4)))
            theta = theta - value / slope
        scale = np.divide(
            np.sin(theta), distorted, out=np.ones_like(distorted), where=distorted > 1.0e-12
        )
        return np.stack([mx * scale, my * scale, np.cos(theta)], axis=-1)


@dataclass(frozen=True)
class RadtanCamera:
    """A pinhole camera with radial-tangential distortion (``k1 k2 p1 p2 [k3]``)."""

    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, ...]
    width: int
    height: int

    def project(self, points: FloatArray) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
        """Pixels ``(u, v)`` of camera-frame points, and which are in front of the camera."""

        z = points[:, 2]
        safe = np.where(np.abs(z) > 1.0e-12, z, 1.0e-12)
        x, y = points[:, 0] / safe, points[:, 1] / safe
        k1, k2, p1, p2, k3 = (tuple(self.distortion) + (0.0,) * 5)[:5]
        r2 = x * x + y * y
        radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        return self.fx * xd + self.cx, self.fy * yd + self.cy, z > 0.0


class StereoCamera(Protocol):
    """What rectification needs of a camera: its focal lengths, size and projection."""

    @property
    def fx(self) -> float: ...

    @property
    def fy(self) -> float: ...

    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...

    def project(self, points: FloatArray) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]: ...


@dataclass(frozen=True)
class FisheyeStereoRectification:
    """A rectified fisheye pair: shared pinhole camera, rotations and resampling maps."""

    left: StereoCamera
    right: StereoCamera
    rectified_from_left: FloatArray
    """3x3 rotation from the left camera's frame into the rectified frame."""
    rectified_from_right: FloatArray
    baseline_m: float
    fx: float
    cx: float
    cy: float
    width: int
    height: int
    left_map: FloatArray
    """``(2, height, width)``: raw left ``(row, column)`` of each rectified pixel."""
    right_map: FloatArray

    @property
    def stereo(self) -> RectifiedStereo:
        """The rectified projections, from the rectified left frame."""

        intrinsics = np.array([[self.fx, 0.0, self.cx], [0.0, self.fx, self.cy], [0.0, 0.0, 1.0]])
        return RectifiedStereo(
            left=intrinsics @ np.c_[np.eye(3), np.zeros(3)],
            right=intrinsics @ np.c_[np.eye(3), [-self.baseline_m, 0.0, 0.0]],
        )

    def rectify_left(self, image: FloatArray) -> FloatArray:
        """The left image resampled onto the rectified pinhole camera (0 outside it)."""

        return _resample(image, self.left_map)

    def rectify_right(self, image: FloatArray) -> FloatArray:
        """The right image resampled onto the rectified pinhole camera (0 outside it)."""

        return _resample(image, self.right_map)


def rectify_fisheye_pair(
    left: StereoCamera,
    right: StereoCamera,
    right_from_left: FloatArray,
    *,
    focal_scale: float = 1.0,
) -> FisheyeStereoRectification:
    """Rectify a fisheye pair given ``T_right_left`` (left-frame points into the right frame).

    The rectified frame's x axis runs along the baseline (from the left camera to the
    right one), its z axis is the mean of the two optical axes made orthogonal to it,
    and both rectified cameras share that orientation and one pinhole camera: the left
    camera's size, its mean focal length times ``focal_scale``, centred.
    """

    rotation = np.asarray(right_from_left, dtype=np.float64)[:3, :3]
    translation = np.asarray(right_from_left, dtype=np.float64)[:3, 3]
    right_centre = -rotation.T @ translation  # the right camera's centre in the left frame
    baseline = float(np.linalg.norm(right_centre))
    if baseline < 1.0e-6:
        raise ValueError("the stereo pair has no baseline")
    axis_x = right_centre / baseline
    optical = np.array([0.0, 0.0, 1.0]) + rotation.T @ np.array([0.0, 0.0, 1.0])
    axis_y = np.cross(optical, axis_x)
    axis_y /= np.linalg.norm(axis_y)
    axis_z = np.cross(axis_x, axis_y)
    rectified_from_left = np.stack([axis_x, axis_y, axis_z])
    rectified_from_right = rectified_from_left @ rotation.T

    fx = 0.5 * (left.fx + left.fy) * focal_scale
    cx, cy = 0.5 * (left.width - 1), 0.5 * (left.height - 1)
    rows, cols = np.mgrid[0 : left.height, 0 : left.width].astype(np.float64)
    rays = np.stack([(cols - cx) / fx, (rows - cy) / fx, np.ones_like(cols)], axis=-1)
    maps = []
    for camera, to_rectified in ((left, rectified_from_left), (right, rectified_from_right)):
        in_camera = rays.reshape(-1, 3) @ to_rectified  # = to_rectified.T @ ray, per ray
        u, v, front = camera.project(in_camera)
        u = np.where(front, u, -1.0e6)
        v = np.where(front, v, -1.0e6)
        maps.append(np.stack([v.reshape(rows.shape), u.reshape(rows.shape)]))
    return FisheyeStereoRectification(
        left=left,
        right=right,
        rectified_from_left=rectified_from_left,
        rectified_from_right=rectified_from_right,
        baseline_m=baseline,
        fx=fx,
        cx=cx,
        cy=cy,
        width=left.width,
        height=left.height,
        left_map=maps[0],
        right_map=maps[1],
    )


def _resample(image: FloatArray, coordinates: FloatArray) -> FloatArray:
    return np.asarray(
        ndimage.map_coordinates(
            np.asarray(image, dtype=np.float64), coordinates, order=1, mode="constant", cval=0.0
        )
    )


def read_kalibr_fisheye_pair(
    camchain: object, left: str = "cam0", right: str = "cam1"
) -> tuple[EquidistantCamera, EquidistantCamera, FloatArray]:
    """The two cameras of a Kalibr camchain and ``T_right_left`` (the right's ``T_cn_cnm1``).

    ``camchain`` is a path or the parsed mapping. Both cameras must use the pinhole
    projection with the ``equidistant`` distortion, and ``right`` must follow ``left``
    in the chain.
    """

    from pathlib import Path

    from calibrex.core.io import read_mapping

    chain = (
        read_mapping(Path(camchain)) if isinstance(camchain, (str, Path)) else dict(camchain)  # type: ignore[call-overload]
    )
    cameras = []
    for name in (left, right):
        entry = chain[name]
        if entry.get("distortion_model") != "equidistant":
            msg = f"{name}: distortion model {entry.get('distortion_model')!r}, not equidistant"
            raise ValueError(msg)
        fx, fy, cx, cy = (float(value) for value in entry["intrinsics"])
        k1, k2, k3, k4 = (float(value) for value in entry["distortion_coeffs"])
        width, height = (int(value) for value in entry["resolution"])
        cameras.append(EquidistantCamera(fx, fy, cx, cy, (k1, k2, k3, k4), width, height))
    if "T_cn_cnm1" not in chain[right]:
        msg = f"{right} has no T_cn_cnm1 to {left}"
        raise ValueError(msg)
    return cameras[0], cameras[1], np.asarray(chain[right]["T_cn_cnm1"], dtype=np.float64)


VERTICAL_OFFSET_COLUMNS: tuple[str, ...] = (
    "pair",
    "row_px",
    "column_px",
    "disparity_px",
    "offset_px",
    "vertical_texture",
)
"""The columns of :func:`vertical_offset_samples`."""


def vertical_offset_samples(
    pairs: list[tuple[FloatArray, FloatArray]],
    *,
    patches_per_pair: int = 300,
    half_px: int = 6,
    rows_px: int = 4,
    max_disparity_px: int = 60,
    min_zncc: float = 0.9,
    min_gradient: float = 25.0,
    seed: int = 0,
) -> FloatArray:
    """Every matched patch's vertical disparity (px, right minus left row) and where it is.

    A rectified pair puts a point on the same row in both images; a residual rotation of
    the rig's calibration (a tenth of a degree is a pixel at these focal lengths) shifts
    the right image's rows. Textured left patches are matched in the right image over
    ``max_disparity_px`` columns and ``rows_px`` rows each way (zero-normalised
    cross-correlation, sub-pixel parabola in rows). Uses the images only, never a
    LiDAR or a reference.

    Returns an ``(N, 6)`` array with :data:`VERTICAL_OFFSET_COLUMNS`: the pair's index,
    the patch's left row and column, its (whole-pixel) horizontal disparity, its vertical
    offset, and 1 where the patch's texture runs more across rows than along them (its
    rows are then well constrained), else 0.
    """

    rng = np.random.default_rng(seed)
    samples: list[tuple[float, float, float, float, float, float]] = []
    for index, (left, right) in enumerate(pairs):
        gy, gx = np.gradient(np.asarray(left, dtype=np.float64))
        margin = half_px + rows_px + 1
        strong = np.argwhere(np.hypot(gx, gy) > min_gradient)
        height, width = left.shape
        strong = strong[
            (strong[:, 0] >= margin)
            & (strong[:, 0] < height - margin)
            & (strong[:, 1] >= margin + max_disparity_px)
            & (strong[:, 1] < width - margin)
        ]
        if len(strong) == 0:
            continue
        picks = strong[rng.choice(len(strong), min(patches_per_pair, len(strong)), replace=False)]
        size = 2 * half_px + 1
        for row, col in picks:
            rows = slice(row - half_px, row + half_px + 1)
            cols = slice(col - half_px, col + half_px + 1)
            patch = left[rows, cols]
            patch = patch - patch.mean()
            norm = np.sqrt((patch**2).sum())
            if norm < 1.0e-6:
                continue
            # every window of the right image within rows_px rows and max_disparity_px
            # columns to the left: windows[i, j] sits rows_px - i rows up and
            # max_disparity_px - j columns left
            strip = right[
                row - rows_px - half_px : row + rows_px + half_px + 1,
                col - max_disparity_px - half_px : col + half_px + 1,
            ]
            windows = sliding_window_view(strip, (size, size))
            centred = windows - windows.mean(axis=(2, 3), keepdims=True)
            energy = np.sqrt((centred**2).sum(axis=(2, 3)))
            scores = (centred * patch).sum(axis=(2, 3)) / np.maximum(norm * energy, 1.0e-9)
            scores = scores[:, ::-1]  # column j now holds disparity j
            i, dx = np.unravel_index(int(np.argmax(scores)), scores.shape)
            if scores[i, dx] < min_zncc or i in (0, 2 * rows_px):
                continue
            before, at, after = scores[i - 1, dx], scores[i, dx], scores[i + 1, dx]
            curvature = before - 2.0 * at + after
            step = 0.5 * (before - after) / curvature if abs(curvature) > 1.0e-9 else 0.0
            vertical = float(np.abs(gy[rows, cols]).mean() > np.abs(gx[rows, cols]).mean())
            samples.append(
                (
                    float(index),
                    float(row),
                    float(col),
                    float(dx),
                    i - rows_px + float(step),
                    vertical,
                )
            )
    return np.asarray(samples, dtype=np.float64).reshape(-1, len(VERTICAL_OFFSET_COLUMNS))


def estimate_vertical_offset(
    pairs: list[tuple[FloatArray, FloatArray]],
    *,
    patches_per_pair: int = 300,
    half_px: int = 6,
    rows_px: int = 4,
    max_disparity_px: int = 60,
    min_zncc: float = 0.9,
    min_gradient: float = 25.0,
    seed: int = 0,
) -> float | None:
    """Median vertical disparity (px, right minus left row) of rectified image pairs.

    The median of :func:`vertical_offset_samples` over every matched patch; ``None``
    when nothing matched.
    """

    samples = vertical_offset_samples(
        pairs,
        patches_per_pair=patches_per_pair,
        half_px=half_px,
        rows_px=rows_px,
        max_disparity_px=max_disparity_px,
        min_zncc=min_zncc,
        min_gradient=min_gradient,
        seed=seed,
    )
    return float(np.median(samples[:, 4])) if len(samples) else None


def with_right_row_shift(
    rectification: FisheyeStereoRectification, offset_px: float
) -> FisheyeStereoRectification:
    """The rectification with the right image's rows moved by ``offset_px``.

    ``offset_px`` is what :func:`estimate_vertical_offset` measured (right minus left
    row): the corrected right image takes each row from ``offset_px`` rows further down,
    so a point lands on the same row in both images again.
    """

    from dataclasses import replace

    rows = ndimage.shift(rectification.right_map[0], (-offset_px, 0.0), order=1, mode="nearest")
    cols = ndimage.shift(rectification.right_map[1], (-offset_px, 0.0), order=1, mode="nearest")
    return replace(rectification, right_map=np.stack([rows, cols]))
