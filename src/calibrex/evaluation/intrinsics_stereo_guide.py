"""Guided stereo calibration with a checkerboard: both intrinsics and the baseline.

:class:`StereoGuide` runs an :class:`~calibrex.evaluation.intrinsics_guide.IntrinsicsGuide`
for each camera (coverage, tilts and precision, as for one camera) and, in addition, keeps
the moments when both cameras see the board, still, in a pose unlike the kept ones. With
both intrinsics known, OpenCV's ``stereoCalibrate`` (intrinsics fixed) gives the right
camera's pose in the left frame from those views: ``X_right = R X_left + T``, Kalibr's
``T_cn_cnm1``, in metres since the board's square size is.

Because only still frames are used, two cameras that are not hardware-synchronized work:
the board does not move between their exposures.

The baseline's std is the spread of the per-view baselines (each view's board poses, from
PnP with the intrinsics) over the square root of their number. It is done when both
cameras are done and at least ``min_stereo_views`` views give a baseline std under
``baseline_rel_std`` of its length.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.evaluation.board_camera_lidar import Checkerboard, _opencv
from calibrex.evaluation.intrinsics_guide import (
    Advice,
    GuideFrame,
    GuideOptions,
    IntrinsicsEstimate,
    IntrinsicsGuide,
    camchain_entry,
    draw_guide,
)

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class StereoExtrinsics:
    """The right camera's pose in the left frame: ``X_right = rotation X_left + translation``."""

    rotation: FloatArray
    translation_m: FloatArray
    translation_std_m: FloatArray
    """Per axis: the views' jackknife spread and both cameras' intrinsics' stds, combined."""
    baseline_std_m: float
    rotation_std_deg: float
    rms_px: float
    views: int

    @property
    def baseline_m(self) -> float:
        return float(np.linalg.norm(self.translation_m))

    def transform(self) -> FloatArray:
        out = np.eye(4)
        out[:3, :3] = self.rotation
        out[:3, 3] = self.translation_m
        return out


@dataclass(frozen=True)
class StereoFrame:
    left: GuideFrame
    right: GuideFrame
    captured: bool
    advice: Advice


@dataclass
class StereoGuide:
    """Two cameras' guides plus the views both cameras share."""

    board: Checkerboard
    left_size: tuple[int, int]
    right_size: tuple[int, int]
    options: GuideOptions = field(default_factory=GuideOptions)
    min_stereo_views: int = 10
    max_stereo_views: int = 40
    baseline_rel_std: float = 0.01
    left: IntrinsicsGuide = field(init=False)
    right: IntrinsicsGuide = field(init=False)
    views: list[tuple[FloatArray, FloatArray]] = field(default_factory=list)
    extrinsics: StereoExtrinsics | None = None

    def __post_init__(self) -> None:
        self.left = IntrinsicsGuide(self.board, *self.left_size, self.options)
        self.right = IntrinsicsGuide(self.board, *self.right_size, self.options)

    def update(self, gray_left: NDArray[np.uint8], gray_right: NDArray[np.uint8]) -> StereoFrame:
        left = self.left.update(gray_left)
        right = self.right.update(gray_right)
        captured = False
        if (
            left.corners is not None
            and right.corners is not None
            and left.still
            and right.still
            and len(self.views) < self.max_stereo_views
            and self._new_pose(left.corners)
        ):
            self.views.append((left.corners, _same_order(left.corners, right.corners)))
            captured = True
            self._solve()
        return StereoFrame(left, right, captured, self._advice(left, right))

    @property
    def done(self) -> bool:
        return self.left.done and self.right.done and self._extrinsics_done()

    def _extrinsics_done(self) -> bool:
        e = self.extrinsics
        return (
            e is not None
            and e.views >= self.min_stereo_views
            and e.baseline_std_m <= self.baseline_rel_std * e.baseline_m
        )

    def _new_pose(self, corners: FloatArray) -> bool:
        width, height = self.left_size
        diagonal = math.hypot(width, height)
        center = corners.mean(axis=0)
        spread = float(np.linalg.norm(corners.max(axis=0) - corners.min(axis=0)))
        for kept, _ in self.views:
            moved = float(np.linalg.norm(kept.mean(axis=0) - center)) / diagonal
            scaled = abs(float(np.linalg.norm(kept.max(axis=0) - kept.min(axis=0))) - spread)
            if moved < self.options.new_pose_fraction and scaled < 0.1 * spread:
                return False
        return True

    def _solve(self) -> None:
        left, right = self.left.estimate, self.right.estimate
        if left is None or right is None or len(self.views) < 3:
            return
        self.extrinsics = stereo_extrinsics(self.views, self.board, left, right)

    def _advice(self, left: GuideFrame, right: GuideFrame) -> Advice:
        if self.done:
            return Advice("done", "Done: press s to save, q to quit")
        if not self.left.done:
            return _prefixed("Left", left.advice)
        if not self.right.done:
            return _prefixed("Right", right.advice)
        if left.corners is None or right.corners is None:
            return Advice("find", "Show the whole board to both cameras")
        if not (left.still and right.still):
            return Advice("hold", "Hold the board still")
        return Advice("vary", "Both cameras: move the board closer, farther and across")


def _same_order(left: FloatArray, right: FloatArray) -> FloatArray:
    """The right corners in the left corners' order.

    A board with an even number of squares one way and an odd number the other looks the
    same turned half a turn, so each camera may number its corners from either end. The
    cameras of a rig are rolled by much less than 90 deg relative to each other, so the
    first-to-last corner runs the same way in both images.
    """

    if float(np.dot(left[-1] - left[0], right[-1] - right[0])) < 0.0:
        return right[::-1].copy()
    return right


def _prefixed(camera: str, advice: Advice) -> Advice:
    return Advice(advice.kind, f"{camera}: {advice.text}", advice.target_cell, advice.tilt)


def _matrix(estimate: IntrinsicsEstimate) -> FloatArray:
    return estimate.matrix


def _distortion(estimate: IntrinsicsEstimate) -> FloatArray:
    return np.array([*estimate.distortion, 0.0])


def stereo_extrinsics(
    views: list[tuple[FloatArray, FloatArray]],
    board: Checkerboard,
    left: IntrinsicsEstimate,
    right: IntrinsicsEstimate,
) -> StereoExtrinsics:
    """``stereoCalibrate`` with fixed intrinsics, and its std.

    The std combines the views' jackknife spread with both cameras' intrinsics moved by
    their stds one at a time: an intrinsics error shifts the baseline (a principal point
    1 px off at 0.7 m moves it by about 0.7 m / f), and the views alone cannot show it.
    """

    rotation, translation, rms = _stereo_calibrate(views, board, left, right)
    length = float(np.linalg.norm(translation))
    n = len(views)
    if n < 4:
        return StereoExtrinsics(
            rotation, translation, np.full(3, math.inf), math.inf, math.inf, rms, n
        )
    # Each estimate's departure from the full one: with each view left out (jackknife,
    # scaled by (n - 1) / n), and with each intrinsic moved by its std (linear propagation).
    departures: list[tuple[FloatArray, FloatArray, float, float]] = []
    jackknife = (n - 1) / n
    for i in range(n):
        r, t, _ = _stereo_calibrate(views[:i] + views[i + 1 :], board, left, right)
        departures.append((r, t, jackknife, 1.0))
    for which, estimate in (("left", left), ("right", right)):
        for name, std in estimate.std.items():
            if not math.isfinite(std) or std <= 0.0:
                continue
            moved = _perturbed(estimate, name, std)
            pair = (moved, right) if which == "left" else (left, moved)
            r, t, _ = _stereo_calibrate(views, board, *pair)
            departures.append((r, t, 1.0, 0.0))
    cv2 = _opencv()
    ts = np.array([t for _, t, _, _ in departures])
    weights = np.array([w for _, _, w, _ in departures])
    # The jackknife spreads about their own mean, the propagated ones about the estimate.
    jk = np.array([j for *_, j in departures]) > 0
    centre = np.where(jk[:, None], ts[jk].mean(axis=0), translation)
    t_std = np.sqrt(np.sum(weights[:, None] * (ts - centre) ** 2, axis=0))
    lengths = np.linalg.norm(ts, axis=1)
    length_centre = np.where(jk, lengths[jk].mean(), length)
    length_std = float(np.sqrt(np.sum(weights * (lengths - length_centre) ** 2)))
    angles = np.array([cv2.Rodrigues(r @ rotation.T)[0].ravel() for r, *_ in departures])
    angle_centre = np.where(jk[:, None], angles[jk].mean(axis=0), 0.0)
    r_std = float(
        np.degrees(np.sqrt(np.sum(weights[:, None] * (angles - angle_centre) ** 2) / 3.0))
    )
    return StereoExtrinsics(rotation, translation, t_std, length_std, r_std, rms, n)


def _perturbed(estimate: IntrinsicsEstimate, name: str, delta: float) -> IntrinsicsEstimate:
    distortion_names = ("k1", "k2", "p1", "p2")
    if name in distortion_names:
        k1, k2, p1, p2 = (
            value + (delta if index == distortion_names.index(name) else 0.0)
            for index, value in enumerate(estimate.distortion)
        )
        return replace(estimate, distortion=(k1, k2, p1, p2))
    return replace(estimate, **{name: getattr(estimate, name) + delta})


def _stereo_calibrate(
    views: list[tuple[FloatArray, FloatArray]],
    board: Checkerboard,
    left: IntrinsicsEstimate,
    right: IntrinsicsEstimate,
) -> tuple[FloatArray, FloatArray, float]:
    cv2 = _opencv()
    objects = [board.object_points().astype(np.float32)] * len(views)
    image_left = [v[0].reshape(-1, 1, 2).astype(np.float32) for v in views]
    image_right = [v[1].reshape(-1, 1, 2).astype(np.float32) for v in views]
    rms, *_, rotation, translation, _, _ = cv2.stereoCalibrate(
        objects,
        image_left,
        image_right,
        _matrix(left),
        _distortion(left),
        _matrix(right),
        _distortion(right),
        (left.width, left.height),
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    return (
        np.asarray(rotation, np.float64),
        np.asarray(translation, np.float64).ravel(),
        float(rms),
    )


def stereo_camchain(guide: StereoGuide) -> dict[str, Any]:
    """``cam0`` and ``cam1`` (with ``T_cn_cnm1``) of a Kalibr camchain."""

    if guide.left.estimate is None or guide.right.estimate is None:
        raise ValueError("the cameras are not calibrated yet")
    chain: dict[str, Any] = {
        "cam0": camchain_entry(guide.left.estimate),
        "cam1": camchain_entry(guide.right.estimate),
    }
    if guide.extrinsics is not None:
        chain["cam1"]["T_cn_cnm1"] = guide.extrinsics.transform().tolist()
    return chain


def stereo_status(guide: StereoGuide) -> str:
    e = guide.extrinsics
    if e is None:
        return f"stereo views {len(guide.views)}/{guide.min_stereo_views}"
    std = e.baseline_std_m
    return (
        f"stereo views {len(guide.views)}/{guide.min_stereo_views}  baseline "
        f"{e.baseline_m:.4f}+-{std:.4f} m  rms {e.rms_px:.2f} px"
    )


def draw_stereo(
    left_image: NDArray[np.uint8],
    right_image: NDArray[np.uint8],
    guide: StereoGuide,
    frame: StereoFrame,
) -> NDArray[np.uint8]:
    """Both annotated frames side by side; the stereo advice on the camera it concerns."""

    status = (stereo_status(guide),)
    left_advice = frame.advice if not frame.advice.text.startswith("Right:") else frame.left.advice
    right_advice = frame.advice if not frame.advice.text.startswith("Left:") else frame.right.advice
    left = draw_guide(left_image, guide.left, frame.left, left_advice, status)
    right = draw_guide(right_image, guide.right, frame.right, right_advice, status)
    if left.shape[0] != right.shape[0]:
        cv2 = _opencv()
        scale = left.shape[0] / right.shape[0]
        right = cv2.resize(right, (round(right.shape[1] * scale), left.shape[0]))
    return np.hstack([left, right])


@dataclass(frozen=True)
class StereoRun:
    guide: StereoGuide | None
    done: bool
    saved: bool
    frames: int


def run_stereo_guide(
    pairs: Iterator[tuple[NDArray[np.uint8], NDArray[np.uint8]]],
    board: Checkerboard,
    *,
    options: GuideOptions | None = None,
    show: bool = True,
    record: Path | None = None,
    save: Callable[[StereoGuide], None] | None = None,
    stop_when_done: bool = False,
) -> StereoRun:
    """The stereo counterpart of
    :func:`~calibrex.evaluation.intrinsics_guide.run_guided_calibration`."""

    cv2 = _opencv()
    guide: StereoGuide | None = None
    writer: Any = None
    saved = False
    count = 0
    window = "calibrex stereo intrinsics"
    try:
        for image_left, image_right in pairs:
            count += 1
            gray_left = (
                image_left if image_left.ndim == 2 else cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
            )
            gray_right = (
                image_right
                if image_right.ndim == 2
                else cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)
            )
            if guide is None:
                guide = StereoGuide(
                    board,
                    (gray_left.shape[1], gray_left.shape[0]),
                    (gray_right.shape[1], gray_right.shape[0]),
                    options or GuideOptions(),
                )
            frame = guide.update(gray_left, gray_right)
            canvas = draw_stereo(image_left, image_right, guide, frame)
            if record is not None:
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(record),
                        cv2.VideoWriter_fourcc(*"MJPG"),
                        15.0,
                        (canvas.shape[1], canvas.shape[0]),
                    )
                writer.write(canvas)
            if show:
                try:
                    cv2.imshow(window, canvas)
                except cv2.error as exc:
                    raise RuntimeError(
                        "this OpenCV has no window support (opencv-python-headless): install "
                        "opencv-python, or run with --no-window --record VIDEO"
                    ) from exc
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if (
                    key == ord("s")
                    and save is not None
                    and guide.left.estimate
                    and guide.right.estimate
                ):
                    save(guide)
                    saved = True
            elif stop_when_done and guide.done:
                break
    finally:
        if writer is not None:
            writer.release()
        if show:
            with contextlib.suppress(cv2.error):  # a window that never opened
                cv2.destroyWindow(window)
    if (
        not show
        and guide is not None
        and save is not None
        and guide.left.estimate
        and guide.right.estimate
    ):
        save(guide)
        saved = True
    return StereoRun(guide, guide.done if guide is not None else False, saved, count)
