"""Camera rotations from image sequences (a "visual gyro") for camera-IMU calibration.

Features are tracked with pyramidal Lucas-Kanade between consecutive (strided)
frames, undistorted to normalized coordinates with the camera model, and the
relative rotation is estimated:

* with the essential matrix (5-point RANSAC and cheirality) whenever it keeps
  enough inliers after the cheirality check; otherwise
* with a pure-rotation model (bearing-vector Kabsch inside RANSAC).

The essential matrix is preferred even when a pure rotation fits the
features within a pixel.  Walking translation over one frame pair produces a
nearly uniform image flow that a pure rotation absorbs as a rotation of about
t / depth, roughly 1 deg for 10 cm at 5 m.  On the Hilti 2022 development
recording, a configuration that used pure rotation for about 90 % of the
pairs was 0.3-0.5 deg further from Kalibr than this one (about 55 %
essential).  For a truly pure rotation the essential matrix is
degenerate: its per-pair rotation error was typically 0.07 deg in the tests,
but occasionally several degrees.  An essential-matrix rotation is therefore
rejected in favour of the pure-rotation fit when the two disagree by more than
``degenerate_disagreement_deg`` while the pure-rotation fit leaves sub-pixel
residuals (``degenerate_parallax_px``): walking translation shifts the
apparent rotation by about a degree, a degenerate solution by more.

A failed frame pair ends the current track segment instead of assuming zero
motion.  Orientations are chained within a segment and cut into windows,
which the IMU-LiDAR rotation evaluation consumes unchanged: it only uses the
orientations.  OpenCV (``pip install calibrex[opencv]``) is needed only here.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.data.imu_trajectory import split_trajectory_windows
from calibrex.solvers.gnss_lever_arm_solver import OdometryWindow

FloatArray: TypeAlias = NDArray[np.float64]


def _opencv() -> Any:
    try:
        return importlib.import_module("cv2")
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError("camera rotations need OpenCV: pip install 'calibrex[opencv]'") from exc


@dataclass(frozen=True)
class CameraModel:
    """Pinhole intrinsics with equidistant (fisheye), radtan, atan, or no distortion.

    ``atan`` is the field-of-view model of Devernay and Faugeras with a distortion centre,
    as Kontiki's ``AtanCamera`` (the LiU GoPro datasets): ``distortion = (gamma, wc_x,
    wc_y)``. A pixel ``p`` maps to ``n = K^-1 p``; with ``L = n - wc`` and ``r = |L|`` the
    undistorted point is ``wc + tan(gamma r) / gamma * L / r``.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: Literal["equidistant", "radtan", "atan", "none"] = "none"
    distortion: tuple[float, ...] = ()
    readout_s: float = 0.0
    """Rolling-shutter readout, first row to last (0: global shutter). From a camchain's
    ``readout_s``, or Kalibr's per-row ``line_delay`` times the image height."""

    @classmethod
    def from_kalibr(cls, camera: Mapping[str, Any]) -> CameraModel:
        """Build from one camera entry of a Kalibr camchain."""

        if camera.get("camera_model", "pinhole") != "pinhole":
            raise ValueError(f"unsupported camera model {camera.get('camera_model')!r}")
        model = str(camera.get("distortion_model", "none"))
        if model not in {"equidistant", "radtan", "atan", "none"}:
            raise ValueError(f"unsupported distortion model {model!r}")
        fx, fy, cx, cy = (float(value) for value in camera["intrinsics"])
        coeffs = tuple(float(value) for value in camera.get("distortion_coeffs", ()))
        if model == "atan" and (len(coeffs) != 3 or not 0.0 < coeffs[0] < math.pi):
            raise ValueError("atan distortion needs distortion_coeffs [gamma, wc_x, wc_y]")
        readout = float(camera.get("readout_s", 0.0) or 0.0)
        if not readout and camera.get("line_delay"):
            rows = camera.get("resolution", (0, 0))[1]
            if not rows:
                raise ValueError("line_delay needs the camera resolution [width, height]")
            readout = float(camera["line_delay"]) * float(rows)
        if not 0.0 <= readout < 1.0:
            raise ValueError(f"rolling-shutter readout {readout!r} s is not in [0, 1)")
        return cls(fx, fy, cx, cy, model, coeffs, readout)  # type: ignore[arg-type]

    def normalize(self, pixels: FloatArray) -> FloatArray:
        """Undistorted normalized image coordinates ``(N, 2)`` of pixel coordinates."""

        points = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
        if len(points) == 0:
            return np.zeros((0, 2))
        matrix = np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])
        if self.distortion_model == "none":
            return np.column_stack(
                [(points[:, 0, 0] - self.cx) / self.fx, (points[:, 0, 1] - self.cy) / self.fy]
            )
        if self.distortion_model == "atan":
            gamma, wcx, wcy = self.distortion
            centre = np.array([wcx, wcy])
            normalized = np.column_stack(
                [(points[:, 0, 0] - self.cx) / self.fx, (points[:, 0, 1] - self.cy) / self.fy]
            )
            offset = normalized - centre
            radius = np.linalg.norm(offset, axis=1)
            scale = np.ones_like(radius)
            moved = radius > 1e-12
            scale[moved] = np.tan(gamma * radius[moved]) / (gamma * radius[moved])
            return np.asarray(centre + offset * scale[:, None], dtype=np.float64)
        cv2 = _opencv()
        coeffs: FloatArray = np.asarray(self.distortion, dtype=np.float64)
        if self.distortion_model == "equidistant":
            result = cv2.fisheye.undistortPoints(points, matrix, coeffs)
        else:
            # cv2.undistortPoints stops after 5 fixed-point iterations: with a strong
            # barrel distortion (KITTI's cameras, k1 = -0.37) that leaves up to 4 px of
            # error in the image corners. Iterate to convergence instead.
            criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12)
            result = cv2.undistortPointsIter(points, matrix, coeffs, None, None, criteria)
        return np.asarray(result, dtype=np.float64).reshape(-1, 2)


@dataclass(frozen=True)
class VisualRotationOptions:
    """Tracking and robust-estimation settings."""

    # Contrast-limited histogram equalization before detection and tracking.
    # On the low-texture Hilti exp07 corridor it cut failed frame pairs from
    # 577 to 16 of 1321 (with the lower feature quality below).
    equalize_contrast: bool = True
    clahe_clip_limit: float = 3.0
    clahe_tiles: int = 8
    max_features: int = 600
    feature_quality: float = 0.001
    min_feature_distance_px: float = 10.0
    lk_window_px: int = 21
    lk_levels: int = 3
    redetect_below: int = 150
    min_inliers: int = 30
    ransac_threshold_px: float = 1.0
    ransac_iterations: int = 200
    essential_inlier_fraction: float = 0.0
    degenerate_disagreement_deg: float = 2.0
    degenerate_parallax_px: float = 0.5
    window_duration_s: float = 10.0
    min_window_frames: int = 20
    seed: int = 0


@dataclass
class VisualRotationStats:
    """How the frame pairs were resolved."""

    pairs: int = 0
    essential: int = 0
    pure_rotation: int = 0
    failed: int = 0
    segments: int = 0


@dataclass(frozen=True)
class CameraRotationTrack:
    """Chained camera orientations ``R_world_camera`` per segment."""

    times_s: FloatArray
    orientations: FloatArray  # (N, 3, 3)
    segment: NDArray[np.int64]
    stats: VisualRotationStats = field(default_factory=VisualRotationStats)

    def windows(self, options: VisualRotationOptions | None = None) -> list[OdometryWindow]:
        """Cut every segment into windows of ``window_duration_s``."""

        opts = options or VisualRotationOptions()
        poses = np.repeat(np.eye(4)[None], len(self.times_s), axis=0)
        poses[:, :3, :3] = self.orientations
        windows: list[OdometryWindow] = []
        for label in np.unique(self.segment):
            members = self.segment == label
            windows += split_trajectory_windows(
                self.times_s[members],
                poses[members],
                window_duration_s=opts.window_duration_s,
                min_window_poses=opts.min_window_frames,
                prefix=f"segment{int(label)}/",
            )
        return windows


def _bearings(points: FloatArray) -> FloatArray:
    rays = np.column_stack([points, np.ones(len(points))])
    return np.asarray(rays / np.linalg.norm(rays, axis=1, keepdims=True), dtype=np.float64)


def _kabsch(source: FloatArray, target: FloatArray) -> FloatArray:
    """Rotation ``R`` minimising ``sum |target - R source|^2``."""

    left, _, right = np.linalg.svd(target.T @ source)
    fix = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(left @ right)))])
    return np.asarray(left @ fix @ right, dtype=np.float64)


def pure_rotation_ransac(
    previous: FloatArray,
    current: FloatArray,
    threshold_rad: float,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[FloatArray, NDArray[np.bool_]] | None:
    """RANSAC over two-bearing Kabsch fits, then a refit on the inliers.

    ``current ~ R previous``: ``R`` maps bearings of the previous frame into
    the current frame.
    """

    count = len(previous)
    if count < 3:
        return None
    # every hypothesis at once: the same draws as one fit per iteration, and the first
    # with the most inliers wins as it would in a loop
    samples = np.array([rng.choice(count, 2, replace=False) for _ in range(max(iterations, 1))])
    left, _, right = np.linalg.svd(np.swapaxes(current[samples], 1, 2) @ previous[samples])
    left[:, :, 2] *= np.sign(np.linalg.det(left @ right))[:, None]
    rotations = left @ right
    angles = np.linalg.norm(np.cross(current, previous @ np.swapaxes(rotations, 1, 2)), axis=2)
    hits = angles < threshold_rad
    best = hits[int(np.argmax(hits.sum(axis=1)))]
    if best.sum() < 3:
        return None
    rotation = _kabsch(previous[best], current[best])
    angles = np.linalg.norm(np.cross(current, previous @ rotation.T), axis=1)
    inliers = angles < threshold_rad
    return _kabsch(previous[inliers], current[inliers]), inliers


def _degenerate(
    essential_rotation: FloatArray,
    previous: FloatArray,
    current: FloatArray,
    focal_px: float,
    threshold: float,
    options: VisualRotationOptions,
    rng: np.random.Generator,
) -> bool:
    """Whether a pure rotation fits sub-pixel yet the essential rotation is far from it."""

    pure = pure_rotation_ransac(
        _bearings(previous), _bearings(current), threshold, options.ransac_iterations, rng
    )
    if pure is None or pure[1].sum() < options.min_inliers:
        return False
    rotation, inliers = pure
    disagreement = np.degrees(np.linalg.norm(_rotvec(essential_rotation.T @ rotation)))
    if disagreement <= options.degenerate_disagreement_deg:
        return False
    a, b = _bearings(previous[inliers]), _bearings(current[inliers])
    parallax = focal_px * float(np.median(np.linalg.norm(np.cross(b, a @ rotation.T), axis=1)))
    return parallax < options.degenerate_parallax_px


def _rotvec(matrix: FloatArray) -> FloatArray:
    from scipy.spatial.transform import Rotation

    return np.asarray(Rotation.from_matrix(matrix).as_rotvec(), dtype=np.float64)


def relative_rotation(
    previous: FloatArray,
    current: FloatArray,
    focal_px: float,
    options: VisualRotationOptions,
    rng: np.random.Generator,
) -> tuple[FloatArray, str, NDArray[np.bool_]] | None:
    """Rotation mapping previous-frame to current-frame directions, and its model.

    ``previous`` and ``current`` are normalized image coordinates of the same
    tracked features.
    """

    if len(previous) < options.min_inliers:
        return None
    threshold = options.ransac_threshold_px / focal_px
    cv2 = _opencv()
    essential, mask = cv2.findEssentialMat(
        previous, current, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=threshold
    )
    if essential is not None and essential.shape == (3, 3):
        count, rotation, _, pose_mask = cv2.recoverPose(
            essential, previous, current, np.eye(3), mask=mask
        )
        enough = max(options.min_inliers, options.essential_inlier_fraction * len(previous))
        rotation = np.asarray(rotation, dtype=np.float64)
        if count >= enough and not _degenerate(
            rotation, previous, current, focal_px, threshold, options, rng
        ):
            return (
                rotation,
                "essential",
                np.asarray(pose_mask).ravel() > 0,
            )
    pure = pure_rotation_ransac(
        _bearings(previous), _bearings(current), threshold, options.ransac_iterations, rng
    )
    if pure is None or pure[1].sum() < options.min_inliers:
        return None
    return pure[0], "pure_rotation", pure[1]


ROLLING_SHUTTER_PASSES = 2
"""Re-estimations of a frame pair's rotation with its rows moved to mid-frame time."""


def _unrolled_rotation(
    result: tuple[FloatArray, str, NDArray[np.bool_]],
    previous_px: Any,
    current_px: Any,
    previous: FloatArray,
    current: FloatArray,
    rows: int,
    readout_s: float,
    interval_s: float,
    focal: float,
    options: VisualRotationOptions,
    rng: np.random.Generator,
) -> tuple[FloatArray, str, NDArray[np.bool_]]:
    """The pair's rotation with every feature moved from its row's time to mid-frame.

    Row ``y`` of a rolling-shutter frame is exposed ``(y / rows - 1/2) * readout_s`` after
    mid-frame. With the camera turning at ``omega`` (camera frame) over the pair, a bearing
    ``b`` seen then is ``Exp(-omega delta) b_mid``, so ``b_mid = Exp(omega delta) b``.
    ``omega`` comes from the pair's own rotation (``R^T = Exp(omega dt)``), so no gyro
    enters; the estimate is refined ``ROLLING_SHUTTER_PASSES`` times, and a refit that
    fails keeps the fit before it.
    """

    from scipy.spatial.transform import Rotation

    def unroll(points: FloatArray, pixels: Any, omega: FloatArray) -> FloatArray:
        delta = (np.asarray(pixels, dtype=np.float64).reshape(-1, 2)[:, 1] / rows - 0.5) * readout_s
        rays = _bearings(points)
        moved = Rotation.from_rotvec(omega[None, :] * delta[:, None]).apply(rays)
        return np.asarray(moved[:, :2] / moved[:, 2:3], dtype=np.float64)

    for _ in range(ROLLING_SHUTTER_PASSES):
        omega = Rotation.from_matrix(result[0].T).as_rotvec() / interval_s
        refined = relative_rotation(
            unroll(previous, previous_px, omega),
            unroll(current, current_px, omega),
            focal,
            options,
            rng,
        )
        if refined is None:  # keep the last fit rather than break the tracked segment
            return result
        result = refined
    return result


def track_camera_rotations(
    frames: Iterable[tuple[float, NDArray[np.uint8]]],
    camera: CameraModel,
    options: VisualRotationOptions | None = None,
) -> CameraRotationTrack:
    """Chain frame-to-frame rotations into orientations, one segment per unbroken run."""

    opts = options or VisualRotationOptions()
    cv2 = _opencv()
    rng = np.random.default_rng(opts.seed)
    stats = VisualRotationStats()
    times: list[float] = []
    orientations: list[FloatArray] = []
    segments: list[int] = []
    previous_image: NDArray[np.uint8] | None = None
    features: Any = None
    orientation = np.eye(3)
    segment = 0

    clahe = (
        cv2.createCLAHE(
            clipLimit=opts.clahe_clip_limit, tileGridSize=(opts.clahe_tiles, opts.clahe_tiles)
        )
        if opts.equalize_contrast
        else None
    )

    def detect(image: NDArray[np.uint8]) -> Any:
        return cv2.goodFeaturesToTrack(
            image, opts.max_features, opts.feature_quality, opts.min_feature_distance_px
        )

    for frame_time_s, raw in frames:
        # a rolling-shutter frame is stamped at its first row; its rotation belongs to mid-frame
        time_s = frame_time_s + 0.5 * camera.readout_s
        image = raw if clahe is None else clahe.apply(raw)
        if previous_image is None:
            previous_image, features = image, detect(image)
            times.append(time_s)
            orientations.append(orientation.copy())
            segments.append(segment)
            continue
        stats.pairs += 1
        result = None
        tracked = None
        if features is not None and len(features) >= opts.min_inliers:
            tracked, status, _ = cv2.calcOpticalFlowPyrLK(
                previous_image,
                image,
                features,
                None,
                winSize=(opts.lk_window_px, opts.lk_window_px),
                maxLevel=opts.lk_levels,
            )
            good = status.ravel() == 1
            features, tracked = features[good], tracked[good]
            before = camera.normalize(features.reshape(-1, 2))
            after = camera.normalize(tracked.reshape(-1, 2))
            focal = 0.5 * (camera.fx + camera.fy)
            result = relative_rotation(before, after, focal, opts, rng)
            if result is not None and camera.readout_s > 0.0 and time_s > times[-1]:
                result = _unrolled_rotation(
                    result, features, tracked, before, after, raw.shape[0],
                    camera.readout_s, time_s - times[-1], focal, opts, rng,
                )  # fmt: skip
        if result is None:
            stats.failed += 1
            segment += 1
            orientation = np.eye(3)
            features = detect(image)
        else:
            rotation, model, inliers = result
            if model == "essential":
                stats.essential += 1
            else:
                stats.pure_rotation += 1
            # x_current = R x_previous, so R_world_current = R_world_previous R^T.
            orientation = orientation @ rotation.T
            assert tracked is not None
            features = tracked[inliers].reshape(-1, 1, 2)
            if len(features) < opts.redetect_below:
                features = detect(image)
        times.append(time_s)
        orientations.append(orientation.copy())
        segments.append(segment)
        previous_image = image
    stats.segments = segment + 1 if times else 0
    return CameraRotationTrack(
        times_s=np.asarray(times, dtype=np.float64),
        orientations=np.asarray(orientations, dtype=np.float64).reshape(-1, 3, 3),
        segment=np.asarray(segments, dtype=np.int64),
        stats=stats,
    )
