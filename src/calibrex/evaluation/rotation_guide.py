"""Live guidance for turning a camera in place (``calibrex intrinsics rotate --camera``).

:mod:`calibrex.evaluation.rotation_self_calibration` needs the camera turned about two
axes at least, smoothly, over a textured static scene. :class:`RotationGuide` watches the
live frames and says what to do next: point at more texture, turn more slowly, or pan,
tilt or roll further. It tracks features frame to frame and, whenever they have moved
``keyframe_motion_px``, turns the homography from the last keyframe into a rotation with a
guessed focal length (``focal_guess`` times the image width). The spans about each axis are
therefore approximate (a focal guess 20 % off scales pan and tilt by 20 %; roll is exact),
which is why the target, 25 deg, is above the 20 deg the calibration warns below.

Translation cannot be judged live (the lens distortion bends the image motion as much as a
few centimetres of translation does), so the guide only reminds you to turn about the lens.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from calibrex.evaluation.board_camera_lidar import _opencv
from calibrex.evaluation.intrinsics_guide import _GREEN, _GREY, _WHITE, _YELLOW, _text, _wrapped

FloatArray: TypeAlias = NDArray[np.float64]
GrayFrame: TypeAlias = tuple[float, NDArray[np.uint8]]

AXES = ("pan", "tilt", "roll")
"""Turns about the camera's y, x and z axes, in the order the guide asks for them."""
_AXIS_INDEX = {"tilt": 0, "pan": 1, "roll": 2}
_HOW = {
    "pan": "Pan: turn the camera left and right about its lens",
    "tilt": "Tilt: turn the camera up and down about its lens",
    "roll": "Roll: turn the camera about its viewing direction",
}
HINT = "Turn about the lens, not your arm; keep the scene still and far"


@dataclass(frozen=True)
class RotationGuideOptions:
    target_span_deg: float = 25.0
    """Each axis's span to reach (the guide's spans are approximate)."""
    focal_guess: float = 0.8
    """Focal length guess as a fraction of the image width (about 64 deg across)."""
    min_features: int = 60
    max_features: int = 300
    keyframe_motion_px: float = 15.0
    fast_fraction: float = 0.06
    """Faster than this fraction of the image width per frame is too fast to track well."""
    min_keyframes: int = 20


@dataclass(frozen=True)
class RotationAdvice:
    kind: str
    """``texture``, ``slow``, ``turn`` or ``done``."""
    text: str
    axis: str | None = None


@dataclass(frozen=True)
class RotationFrame:
    points: FloatArray
    keyframe: bool
    flow_px: float
    span_deg: dict[str, float]
    advice: RotationAdvice


@dataclass
class RotationGuide:
    width: int
    height: int
    options: RotationGuideOptions = field(default_factory=RotationGuideOptions)
    orientations: list[FloatArray] = field(default_factory=list)
    """Camera-to-world rotation at each keyframe (the first keyframe is the world)."""
    span_deg: dict[str, float] = field(default_factory=lambda: dict.fromkeys(AXES, 0.0))
    _previous: NDArray[np.uint8] | None = None
    _points: NDArray[np.float32] = field(default_factory=lambda: np.empty((0, 1, 2), np.float32))
    _ids: NDArray[np.int64] = field(default_factory=lambda: np.empty(0, np.int64))
    _next_id: int = 0
    _anchor: dict[int, tuple[float, float]] = field(default_factory=dict)
    _low: FloatArray = field(default_factory=lambda: np.zeros(3))
    _high: FloatArray = field(default_factory=lambda: np.zeros(3))

    @property
    def focal_px(self) -> float:
        return self.options.focal_guess * self.width

    @property
    def done(self) -> bool:
        return len(self.orientations) >= self.options.min_keyframes and all(
            self.span_deg[axis] >= self.options.target_span_deg for axis in AXES
        )

    def update(self, gray: NDArray[np.uint8]) -> RotationFrame:
        cv2 = _opencv()
        image = np.ascontiguousarray(gray, dtype=np.uint8)
        flow = 0.0
        lost = False
        if self._previous is not None and len(self._points):
            lk = {"winSize": (21, 21), "maxLevel": 3}
            moved, status, _ = cv2.calcOpticalFlowPyrLK(
                self._previous, image, self._points, None, **lk
            )
            back, back_status, _ = cv2.calcOpticalFlowPyrLK(
                image, self._previous, moved, None, **lk
            )
            drift = np.linalg.norm((back - self._points).reshape(-1, 2), axis=1)
            xy = moved.reshape(-1, 2)
            keep = (
                (status.ravel() == 1)
                & (back_status.ravel() == 1)
                & (drift < 0.5)
                & (xy[:, 0] >= 0)
                & (xy[:, 1] >= 0)
                & (xy[:, 0] < self.width)
                & (xy[:, 1] < self.height)
            )
            if keep.any():
                step = (moved - self._points).reshape(-1, 2)[keep]
                flow = float(np.median(np.linalg.norm(step, axis=1)))
            lost = keep.sum() < 0.5 * len(keep)
            self._points, self._ids = moved[keep], self._ids[keep]
        keyframe = self._keyframe()
        if len(self._points) < self.options.max_features * 0.7:
            self._detect(image)
        self._previous = image
        too_fast = lost or flow > self.options.fast_fraction * self.width
        return RotationFrame(
            self._points.reshape(-1, 2).astype(np.float64),
            keyframe,
            flow,
            dict(self.span_deg),
            self._advice(too_fast),
        )

    def _keyframe(self) -> bool:
        current = {
            int(i): (float(x), float(y))
            for i, (x, y) in zip(self._ids, self._points.reshape(-1, 2), strict=True)
        }
        shared = [i for i in current if i in self._anchor]
        if not self.orientations:
            if len(current) >= self.options.min_features:
                self.orientations.append(np.eye(3))
                self._anchor = current
                return True
            return False
        if len(shared) < 20:
            # The chain broke (too fast, or a blank view): start again from here.
            self._anchor = current
            return False
        motion = float(np.median([math.dist(current[i], self._anchor[i]) for i in shared]))
        if motion < 2.0:
            return False
        rotation = self._rotation(
            np.array([self._anchor[i] for i in shared]), np.array([current[i] for i in shared])
        )
        if rotation is None:
            if motion >= self.options.keyframe_motion_px:
                self._anchor = current
            return False
        # x_current = R x_anchor in camera rays, so camera-to-world C_c = C_a R^T. Every
        # frame's orientation widens the spans (turning back between keyframes would miss
        # the extremes); keyframes chain the orientations.
        orientation = self.orientations[-1] @ rotation.T
        self._widen(orientation)
        if motion < self.options.keyframe_motion_px:
            return False
        self.orientations.append(orientation)
        self._anchor = current
        return True

    def _rotation(self, before: FloatArray, after: FloatArray) -> FloatArray | None:
        cv2 = _opencv()
        homography, inliers = cv2.findHomography(before, after, cv2.RANSAC, 2.0)
        if homography is None or int(inliers.sum()) < 12:
            return None
        k = np.array(
            [[self.focal_px, 0.0, self.width / 2], [0.0, self.focal_px, self.height / 2], [0, 0, 1]]
        )
        m = np.linalg.solve(k, homography @ k)
        m *= np.sign(np.linalg.det(m))
        u, _, vt = np.linalg.svd(m)
        rotation: FloatArray = u @ vt
        return rotation

    def _widen(self, orientation: FloatArray) -> None:
        """Turns about the camera axes since the first keyframe: the spans are their ranges."""

        rotvec = np.degrees(Rotation.from_matrix(orientation).as_rotvec())
        self._low = np.minimum(self._low, rotvec)
        self._high = np.maximum(self._high, rotvec)
        for axis, index in _AXIS_INDEX.items():
            self.span_deg[axis] = float(self._high[index] - self._low[index])

    def _detect(self, image: NDArray[np.uint8]) -> None:
        cv2 = _opencv()
        mask = np.full(image.shape[:2], 255, np.uint8)
        for x, y in self._points.reshape(-1, 2):
            cv2.circle(mask, (int(x), int(y)), 10, 0, -1)
        found = cv2.goodFeaturesToTrack(
            image, self.options.max_features - len(self._points), 0.01, 10, mask=mask
        )
        if found is None:
            return
        found = found.astype(np.float32).reshape(-1, 1, 2)
        self._points = np.concatenate([self._points, found])
        self._ids = np.concatenate(
            [self._ids, np.arange(self._next_id, self._next_id + len(found))]
        )
        self._next_id += len(found)

    def _advice(self, too_fast: bool) -> RotationAdvice:
        if self.done:
            return RotationAdvice("done", "Done: press s to calibrate, q to quit")
        if len(self._points) < self.options.min_features:
            return RotationAdvice("texture", "Point the camera at a static scene with more texture")
        if too_fast:
            return RotationAdvice("slow", "Slower: turn smoothly so the features can be followed")
        target = self.options.target_span_deg
        axis = next((a for a in AXES if self.span_deg[a] < target), None)
        if axis is None:
            return RotationAdvice("turn", "Keep turning: a few more keyframes")
        return RotationAdvice(
            "turn", f"{_HOW[axis]} ({self.span_deg[axis]:.0f}/{target:.0f} deg)", axis
        )


def draw_rotation_guide(
    image: NDArray[np.uint8], guide: RotationGuide, frame: RotationFrame
) -> NDArray[np.uint8]:
    """The frame with the tracked features, the advice and a bar per axis."""

    cv2 = _opencv()
    canvas = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    height, width = canvas.shape[:2]
    for x, y in frame.points:
        cv2.circle(canvas, (round(x), round(y)), 2, _GREEN, -1)
    if frame.keyframe:
        cv2.rectangle(canvas, (2, 2), (width - 3, height - 3), _WHITE, 3)
    scale = max(0.5, width / 1280.0)
    color = _GREEN if frame.advice.kind == "done" else _YELLOW
    for index, line in enumerate(_wrapped(frame.advice.text, scale * 0.9, width - 24)):
        _text(canvas, line, (12, int((34 + 32 * index) * scale)), scale * 0.9, color)
    target = guide.options.target_span_deg
    bar_width = int(min(width * 0.45, 360 * scale))
    for index, axis in enumerate(AXES):
        y = height - int((96 - 26 * index) * scale)
        x0 = int(12 * scale) + int(70 * scale)
        filled = min(1.0, frame.span_deg[axis] / target)
        reached = filled >= 1.0
        _text(canvas, axis, (12, y + int(8 * scale)), scale * 0.6, _WHITE)
        cv2.rectangle(
            canvas, (x0, y - int(8 * scale)), (x0 + bar_width, y + int(8 * scale)), _GREY, 1
        )
        cv2.rectangle(
            canvas,
            (x0, y - int(8 * scale)),
            (x0 + int(filled * bar_width), y + int(8 * scale)),
            _GREEN if reached else _YELLOW,
            -1,
        )
        if axis == frame.advice.axis:
            cv2.rectangle(
                canvas,
                (x0 - 3, y - int(11 * scale)),
                (x0 + bar_width + 3, y + int(11 * scale)),
                _YELLOW,
                2,
            )
    _text(
        canvas,
        f"keyframes {len(guide.orientations)}  features {len(frame.points)}  {HINT}",
        (12, height - int(12 * scale)),
        scale * 0.5,
        _WHITE,
    )
    return canvas


@dataclass(frozen=True)
class RotationRun:
    recorded: list[GrayFrame]
    """Every ``stride``-th grey frame, with its time, for the self-calibration."""
    done: bool
    accepted: bool
    """Whether to calibrate: done or ``s`` pressed (not ``q``)."""


def run_rotation_guide(
    frames: Iterator[tuple[float, NDArray[np.uint8]]],
    *,
    options: RotationGuideOptions | None = None,
    show: bool = True,
    record: Path | None = None,
    stride: int = 1,
    side_by_side: bool = False,
    stop_when_done: bool = True,
    on_status: Callable[[RotationGuide, RotationFrame], None] | None = None,
) -> RotationRun:
    """Guide the turning on ``(time, BGR or grey image)`` frames and keep them.

    With ``side_by_side`` the guide watches the left half (both cameras turn together) and
    keeps the whole frames. With a window, ``s`` accepts what was recorded and ``q`` quits;
    ``stop_when_done`` stops as soon as the guide is done.
    """

    cv2 = _opencv()
    guide: RotationGuide | None = None
    writer: Any = None
    recorded: list[GrayFrame] = []
    accepted = False
    window = "calibrex rotate"
    count = 0
    try:
        for time_s, image in frames:
            gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if count % stride == 0:
                recorded.append((float(time_s), gray))
            count += 1
            view = gray[:, : gray.shape[1] // 2] if side_by_side else gray
            shown = image[:, : image.shape[1] // 2] if side_by_side else image
            if guide is None:
                guide = RotationGuide(
                    view.shape[1], view.shape[0], options or RotationGuideOptions()
                )
            frame = guide.update(view)
            if on_status is not None:
                on_status(guide, frame)
            canvas = draw_rotation_guide(shown, guide, frame)
            if side_by_side:
                right = image[:, image.shape[1] // 2 : 2 * (image.shape[1] // 2)]
                right = right if right.ndim == 3 else cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
                canvas = np.hstack([canvas, right])
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
                        "opencv-python, or run with --no-window"
                    ) from exc
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    return RotationRun(recorded, guide.done, False)
                if key == ord("s"):
                    accepted = True
                    break
            if stop_when_done and guide.done:
                accepted = True
                break
    finally:
        if writer is not None:
            writer.release()
        if show:
            with contextlib.suppress(cv2.error):  # a window that never opened
                cv2.destroyWindow(window)
    done = guide.done if guide is not None else False
    # The frames ran out (end of a video, or the time limit): calibrate on what there is.
    return RotationRun(recorded, done, accepted or not show or done)
