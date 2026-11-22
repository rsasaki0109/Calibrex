"""Camera intrinsics without a target, from a camera turned in place.

When a camera only rotates, where a scene point appears in one frame fixes where it appears
in every other frame through the intrinsics and the rotation alone: no depth enters. So a
video of a camera turned about its lens, about at least two axes, determines the focal
lengths, the principal point and the lens distortion (OpenCV pinhole / radtan: k1, k2, p1,
p2) without a checkerboard.

The steps:

1. Features are tracked with pyramidal Lucas-Kanade; a track is kept only while tracking
   it back lands within ``forward_backward_px`` of where it started.
2. Keyframes are taken every time the features have moved ``keyframe_motion_px`` (median).
3. The focal length and k1 are found together on a grid: with them, the undistorted
   correspondences between keyframes a few apart must be related by a rotation
   (``K^-1 H K`` orthonormal). Searching the focal length alone is biased by tens of percent
   because barrel distortion bends the image motion the way a shorter lens does.
4. A least-squares fit refines the eight intrinsics and every keyframe's rotation from the
   transfer error: a feature seen in keyframe a is undistorted, rotated into keyframe b and
   projected with the distortion; it should land where it was tracked. The scene points are
   not unknowns, which keeps the fit small and quick to converge.

**The camera must turn about its lens.** A hand-held camera also translates as it turns,
and the parallax of that translation is taken up by the intrinsics without raising the
residual: on a simulated camera the focal length came out too long by about 0.8 times the
ratio of the lens's offset from the rotation centre to the scene distance (5 cm at 4 m:
+0.9 %; 15 cm at 10 m: +1.1 %), while rotating about the lens gave it within 0.1 %. Turn a
camera on a tripod or pan-tilt head, or face a scene tens of metres away.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares, minimize
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from calibrex.evaluation.board_camera_lidar import _opencv
from calibrex.evaluation.intrinsics_guide import IntrinsicsEstimate

FloatArray: TypeAlias = NDArray[np.float64]
INTRINSIC_NAMES = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2")


@dataclass(frozen=True)
class SelfCalibrationOptions:
    """Tracking, keyframing and fitting settings."""

    max_features: int = 400
    redetect_below: int = 250
    min_feature_distance_px: float = 12.0
    forward_backward_px: float = 0.3
    keyframe_motion_px: float = 15.0
    """A keyframe is taken when the tracked features moved this much (median) since the last."""
    min_track_keyframes: int = 3
    max_tracks: int = 2500
    pair_gap: int = 4
    """Transfer residuals pair a track's observations in keyframes up to this many apart."""
    init_gap: int = 4
    focal_grid: tuple[float, float, int] = (0.3, 2.5, 45)
    """Focal-length grid as (low, high, count) fractions of the image width."""
    k1_grid: tuple[float, float, int] = (-0.5, 0.3, 17)
    min_axis_span_deg: float = 20.0
    """Rotation about at least two camera axes must span this much."""
    max_nfev: int = 150


@dataclass(frozen=True)
class SelfCalibration:
    """The estimate and how well the recording supports it."""

    estimate: IntrinsicsEstimate
    rotation_span_deg: tuple[float, float, float]
    """How far the camera turned about its x (tilt), y (pan) and z (roll) axes."""
    keyframes: int
    tracks: int
    pairs: int
    warnings: tuple[str, ...]
    keyframe_times_s: tuple[float, ...] = ()
    rotations: FloatArray | None = None
    """Keyframe orientations, camera to a world frame fixed at the first keyframe: (N, 3, 3)."""


@dataclass
class _Tracks:
    width: int
    height: int
    keyframe_times: list[float] = field(default_factory=list)
    observations: list[dict[int, tuple[float, float]]] = field(default_factory=list)
    """Per keyframe: track id -> pixel."""


def track_keyframes(
    frames: Iterable[tuple[float, NDArray[np.uint8]]], options: SelfCalibrationOptions
) -> _Tracks:
    """Track features through grey frames and keep the keyframes' observations."""

    cv2 = _opencv()
    lk = {"winSize": (21, 21), "maxLevel": 3}
    previous: NDArray[np.uint8] | None = None
    points = np.empty((0, 1, 2), np.float32)
    ids = np.empty(0, np.int64)
    next_id = 0
    tracks: _Tracks | None = None
    anchor: dict[int, tuple[float, float]] = {}
    for time_s, raw in frames:
        image = np.ascontiguousarray(raw, dtype=np.uint8)
        height, width = image.shape[:2]
        if tracks is None:
            tracks = _Tracks(width, height)
        if previous is not None and len(points):
            moved, status, _ = cv2.calcOpticalFlowPyrLK(previous, image, points, None, **lk)
            back, back_status, _ = cv2.calcOpticalFlowPyrLK(image, previous, moved, None, **lk)
            drift = np.linalg.norm((back - points).reshape(-1, 2), axis=1)
            xy = moved.reshape(-1, 2)
            keep = (
                (status.ravel() == 1)
                & (back_status.ravel() == 1)
                & (drift < options.forward_backward_px)
                & (xy[:, 0] > 2)
                & (xy[:, 1] > 2)
                & (xy[:, 0] < width - 3)
                & (xy[:, 1] < height - 3)
            )
            points, ids = moved[keep], ids[keep]
        current = {
            int(i): (float(x), float(y))
            for i, (x, y) in zip(ids, points.reshape(-1, 2), strict=True)
        }
        shared = [i for i in current if i in anchor]
        motion = (
            float(np.median([math.dist(current[i], anchor[i]) for i in shared]))
            if shared
            else math.inf
        )
        if not tracks.keyframe_times or motion >= options.keyframe_motion_px or len(shared) < 30:
            tracks.keyframe_times.append(float(time_s))
            tracks.observations.append(current)
            anchor = current
        if len(points) < options.redetect_below:
            mask = np.full((height, width), 255, np.uint8)
            for x, y in points.reshape(-1, 2):
                cv2.circle(mask, (int(x), int(y)), int(options.min_feature_distance_px), 0, -1)
            found = cv2.goodFeaturesToTrack(
                image,
                options.max_features - len(points),
                0.01,
                options.min_feature_distance_px,
                mask=mask,
            )
            if found is not None:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
                found = cv2.cornerSubPix(
                    image, found.astype(np.float32), (5, 5), (-1, -1), criteria
                )
                points = np.concatenate([points, found.reshape(-1, 1, 2)]).astype(np.float32)
                ids = np.concatenate([ids, np.arange(next_id, next_id + len(found))])
                next_id += len(found)
        previous = image
    if tracks is None:
        raise ValueError("no frames")
    return tracks


# ----------------------------------------------------------------------- model


def undistort_points(params: FloatArray, pixels: FloatArray) -> FloatArray:
    """Normalized rays ``(x, y, 1)`` of pixels under pinhole / radtan ``params``."""

    fx, fy, cx, cy, k1, k2, p1, p2 = params
    xd, yd = (pixels[:, 0] - cx) / fx, (pixels[:, 1] - cy) / fy
    x, y = xd.copy(), yd.copy()
    for _ in range(8):
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 * r2
        x = (xd - (2 * p1 * x * y + p2 * (r2 + 2 * x * x))) / radial
        y = (yd - (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)) / radial
    return np.column_stack([x, y, np.ones_like(x)])


def project_points(params: FloatArray, rays: FloatArray) -> FloatArray:
    """Pixels of camera-frame directions under pinhole / radtan ``params``."""

    fx, fy, cx, cy, k1, k2, p1, p2 = params
    x, y = rays[:, 0] / rays[:, 2], rays[:, 1] / rays[:, 2]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.column_stack([fx * xd + cx, fy * yd + cy])


def _orthonormality(params: FloatArray, pairs: list[tuple[FloatArray, FloatArray]]) -> float:
    cv2 = _opencv()
    cost = 0.0
    for before, after in pairs:
        a, b = undistort_points(params, before), undistort_points(params, after)
        homography, _ = cv2.findHomography(a[:, :2], b[:, :2], 0)
        if homography is None:
            return math.inf
        det = np.linalg.det(homography)
        if not math.isfinite(det) or det <= 0:
            return math.inf
        m = homography / np.cbrt(det)
        cost += float(np.sum((m @ m.T - np.eye(3)) ** 2))
    return cost


def _common(tracks: _Tracks, a: int, b: int) -> tuple[FloatArray, FloatArray]:
    shared = [i for i in tracks.observations[b] if i in tracks.observations[a]]
    return (
        np.array([tracks.observations[a][i] for i in shared], dtype=np.float64).reshape(-1, 2),
        np.array([tracks.observations[b][i] for i in shared], dtype=np.float64).reshape(-1, 2),
    )


def initial_focal_k1(tracks: _Tracks, options: SelfCalibrationOptions) -> tuple[float, float]:
    """Focal length and k1 from a grid, refined by Nelder-Mead (principal point at the centre)."""

    n = len(tracks.observations)
    pairs = []
    for a in range(0, max(n - options.init_gap, 0), max(1, options.init_gap // 2)):
        before, after = _common(tracks, a, a + options.init_gap)
        if len(before) >= 30:
            pairs.append((before, after))
    if len(pairs) < 3:
        raise ValueError(
            "too few keyframe pairs share features: turn the camera more slowly, over a "
            "textured scene"
        )
    width, height = tracks.width, tracks.height

    def cost(focal: float, k1: float) -> float:
        params = np.array([focal, focal, width / 2, height / 2, k1, 0.0, 0.0, 0.0])
        return _orthonormality(params, pairs)

    f_low, f_high, f_count = options.focal_grid
    k_low, k_high, k_count = options.k1_grid
    best = min(
        (cost(f, k), f, k)
        for f in np.linspace(f_low * width, f_high * width, f_count)
        for k in np.linspace(k_low, k_high, k_count)
    )
    refined = minimize(
        lambda z: cost(float(z[0]), float(z[1])),
        [best[1], best[2]],
        method="Nelder-Mead",
        options={"xatol": 0.2, "fatol": 1e-12, "maxiter": 300},
    )
    return float(refined.x[0]), float(refined.x[1])


def _initial_rotations(tracks: _Tracks, params: FloatArray) -> FloatArray:
    """Keyframe orientations (camera to world, first = identity) chained from rotations."""

    cv2 = _opencv()
    rotations = [np.eye(3)]
    for k in range(1, len(tracks.observations)):
        before, after = _common(tracks, k - 1, k)
        if len(before) < 8:
            rotations.append(rotations[-1])
            continue
        a, b = undistort_points(params, before), undistort_points(params, after)
        homography, _ = cv2.findHomography(a[:, :2], b[:, :2], cv2.RANSAC, 2.0 / params[0])
        u, _, vt = np.linalg.svd(homography if homography is not None else np.eye(3))
        relative = u @ vt
        if np.linalg.det(relative) < 0:
            relative = -relative
        # b ~ R_b^T R_a a, so R_b = R_a relative^T
        rotations.append(rotations[-1] @ relative.T)
    return np.asarray(rotations)


@dataclass(frozen=True)
class _Pairs:
    a: NDArray[np.int64]
    b: NDArray[np.int64]
    before: FloatArray
    after: FloatArray
    reuse: float
    tracks: int


def _transfer_pairs(tracks: _Tracks, options: SelfCalibrationOptions) -> _Pairs:
    counts: dict[int, int] = {}
    for observation in tracks.observations:
        for i in observation:
            counts[i] = counts.get(i, 0) + 1
    usable = sorted(i for i, c in counts.items() if c >= options.min_track_keyframes)
    if len(usable) > options.max_tracks:
        usable = [
            usable[i] for i in np.linspace(0, len(usable) - 1, options.max_tracks).astype(int)
        ]
    keep = set(usable)
    a_index, b_index, before, after = [], [], [], []
    uses: dict[tuple[int, int], int] = {}
    n = len(tracks.observations)
    for a in range(n):
        for b in range(a + 1, min(n, a + options.pair_gap + 1)):
            for i, xy in tracks.observations[b].items():
                if i in keep and i in tracks.observations[a]:
                    a_index.append(a)
                    b_index.append(b)
                    before.append(tracks.observations[a][i])
                    after.append(xy)
                    uses[(i, a)] = uses.get((i, a), 0) + 1
                    uses[(i, b)] = uses.get((i, b), 0) + 1
    if len(a_index) < 100:
        raise ValueError("too few tracked features shared between keyframes")
    return _Pairs(
        np.asarray(a_index),
        np.asarray(b_index),
        np.asarray(before, dtype=np.float64),
        np.asarray(after, dtype=np.float64),
        float(np.mean(list(uses.values()))),
        len(keep),
    )


def _rotation_span_deg(rotations: FloatArray) -> tuple[float, float, float]:
    mean = Rotation.from_matrix(rotations).mean()
    relative = (mean.inv() * Rotation.from_matrix(rotations)).as_rotvec()
    span = np.degrees(relative.max(axis=0) - relative.min(axis=0))
    return float(span[0]), float(span[1]), float(span[2])


def self_calibrate(
    frames: Iterable[tuple[float, NDArray[np.uint8]]],
    options: SelfCalibrationOptions | None = None,
) -> SelfCalibration:
    """Intrinsics of a camera turned in place, from ``(time, grey image)`` frames."""

    opts = options or SelfCalibrationOptions()
    tracks = track_keyframes(frames, opts)
    if len(tracks.observations) < 6:
        raise ValueError(
            f"only {len(tracks.observations)} keyframes: turn the camera further (each "
            f"keyframe needs {opts.keyframe_motion_px:g} px of image motion)"
        )
    focal, k1 = initial_focal_k1(tracks, opts)
    width, height = tracks.width, tracks.height
    start = np.array([focal, focal, width / 2, height / 2, k1, 0.0, 0.0, 0.0])
    rotations = _initial_rotations(tracks, start)
    pairs = _transfer_pairs(tracks, opts)
    n_kf = len(tracks.observations)

    def unpack(x: FloatArray) -> tuple[FloatArray, FloatArray]:
        rotvecs = np.vstack([np.zeros(3), x[8:].reshape(-1, 3)])
        return x[:8], Rotation.from_rotvec(rotvecs).as_matrix()

    def residual(x: FloatArray) -> FloatArray:
        params, rs = unpack(x)
        world = np.einsum("nij,nj->ni", rs[pairs.a], undistort_points(params, pairs.before))
        camera = np.einsum("nji,nj->ni", rs[pairs.b], world)
        return np.asarray((project_points(params, camera) - pairs.after).ravel(), np.float64)

    x0 = np.concatenate([start, Rotation.from_matrix(rotations[1:]).as_rotvec().ravel()])
    m = len(pairs.a)
    sparsity = lil_matrix((2 * m, len(x0)), dtype=int)
    sparsity[:, :8] = 1
    for j in range(m):
        for k in (pairs.a[j], pairs.b[j]):
            if k > 0:
                sparsity[2 * j : 2 * j + 2, 8 + 3 * (k - 1) : 8 + 3 * k] = 1
    scale = np.concatenate(
        [[10.0, 10.0, 5.0, 5.0, 0.01, 0.01, 0.001, 0.001], np.full(len(x0) - 8, 0.003)]
    )
    solution = least_squares(
        residual,
        x0,
        jac_sparsity=sparsity,
        method="trf",
        loss="soft_l1",
        f_scale=2.0,
        x_scale=scale,
        max_nfev=opts.max_nfev,
    )
    params, rs = unpack(solution.x)
    errors = solution.fun.reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum(errors**2, axis=1))))
    std = _intrinsic_std(solution.jac, errors, len(x0), pairs.reuse)
    span = _rotation_span_deg(rs)
    warnings = _warnings(span, opts, rms)
    estimate = IntrinsicsEstimate(
        width=width,
        height=height,
        fx=float(params[0]),
        fy=float(params[1]),
        cx=float(params[2]),
        cy=float(params[3]),
        distortion=(float(params[4]), float(params[5]), float(params[6]), float(params[7])),
        std=dict(zip(INTRINSIC_NAMES, std, strict=True)),
        rms_px=rms,
        views=n_kf,
    )
    return SelfCalibration(
        estimate,
        span,
        n_kf,
        pairs.tracks,
        m,
        tuple(warnings),
        tuple(tracks.keyframe_times),
        rs,
    )


def _intrinsic_std(jacobian: Any, errors: FloatArray, n_params: int, reuse: float) -> list[float]:
    """Stds of the intrinsics from the Jacobian, the residual RMS per coordinate, and the
    observation reuse (each tracked point enters several pairs, so the residuals are not
    independent: the variance is scaled by how many pairs use a point on average)."""

    dense = jacobian.toarray() if hasattr(jacobian, "toarray") else np.asarray(jacobian)
    normal = dense.T @ dense
    sigma2 = float(np.mean(errors**2)) * max(1.0, reuse)
    try:
        covariance = np.linalg.pinv(normal) * sigma2
    except np.linalg.LinAlgError:
        return [math.nan] * 8
    return [float(math.sqrt(max(covariance[i, i], 0.0))) for i in range(8)]


def _warnings(
    span: tuple[float, float, float], options: SelfCalibrationOptions, rms: float
) -> list[str]:
    warnings = []
    names = ("tilt (about x)", "pan (about y)", "roll (about z)")
    wide = [
        name for name, value in zip(names, span, strict=True) if value >= options.min_axis_span_deg
    ]
    if len(wide) < 2:
        warnings.append(
            "the camera turned less than "
            f"{options.min_axis_span_deg:g} deg about two axes (tilt {span[0]:.0f}, pan "
            f"{span[1]:.0f}, roll {span[2]:.0f} deg): the focal length is poorly determined; "
            "pan and tilt the camera further"
        )
    if rms > 1.0:
        warnings.append(
            f"the transfer RMS is {rms:.2f} px: the camera may have translated, or the scene "
            "moved; turn it about its lens and face a static scene"
        )
    warnings.append(
        "the camera must turn about its lens: a lens offset L from the rotation centre at "
        "scene distance D biases the focal length by about +0.8 L/D (5 cm at 4 m: +1 %), "
        "which the residuals do not show"
    )
    return warnings
