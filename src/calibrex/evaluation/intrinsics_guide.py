"""Guided camera-intrinsics calibration with a checkerboard: what to do next, frame by frame.

A person holds a checkerboard in front of the camera (or moves the camera in front of a
fixed board). :class:`IntrinsicsGuide` looks at every frame and keeps the ones that add
something: a part of the image the board has not covered yet, a tilt it has not shown,
or, once both are complete, a pose unlike the ones kept. Only still frames are kept, so
motion blur and rolling-shutter skew stay out. After every kept frame the intrinsics are
re-estimated with OpenCV's ``calibrateCameraExtended``, which also gives each
parameter's standard deviation.

:meth:`IntrinsicsGuide.update` returns an :class:`Advice` saying what to do next, in this
order: find the board, hold it still, move it to the uncovered part of the image nearest
to it, tilt it the way it has not been tilted, then vary position, distance and tilt until
the focal lengths and the principal point are known to the bounds of
:class:`GuideOptions`. :func:`draw_guide` paints the advice over the frame;
:func:`run_guided_calibration` is the live loop behind ``calibrex intrinsics guide``.

The model is OpenCV's pinhole with radial-tangential distortion (k1, k2, p1, p2; k3 is
fixed at 0), which Kalibr calls ``pinhole`` / ``radtan``: :func:`camchain_entry` writes it
as a Kalibr camchain entry that Calibrex's camera pairs read.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.evaluation.board_camera_lidar import Checkerboard, _opencv

FloatArray: TypeAlias = NDArray[np.float64]
Tilt = Literal["left", "right", "up", "down"]
TILTS: tuple[Tilt, ...] = ("left", "right", "up", "down")
AdviceKind = Literal["find", "hold", "move", "tilt", "vary", "done"]


@dataclass(frozen=True)
class GuideOptions:
    """When a frame is kept and when the calibration is good enough."""

    grid: tuple[int, int] = (5, 4)
    """The image is cut into ``columns x rows`` cells; the board must visit each one."""
    still_px: float = 1.0
    """A frame is kept only when the corners moved at most this much (median, pixels)
    since the previous frame."""
    tilt_deg: float = 25.0
    """A view counts as tilted left, right, up or down beyond this angle."""
    new_pose_deg: float = 8.0
    """Once the cells and tilts are complete, a frame is kept when its board differs from
    every kept one by this rotation, or by ``new_pose_fraction`` of the image diagonal."""
    new_pose_fraction: float = 0.12
    min_views: int = 6
    """Views before the first calibration; it also waits for two tilt directions, since
    views of a board facing the camera leave the focal length undetermined."""
    min_views_done: int = 15
    """Views before the calibration can be done, however small its stds."""
    max_views: int = 50
    focal_rel_std: float = 0.003
    """Done when the focal lengths' std is below this fraction of their value..."""
    principal_std_px: float = 2.0
    """...the principal point's std is below this many pixels..."""
    max_rms_px: float = 1.0
    """...and the reprojection RMS is below this many pixels."""


@dataclass(frozen=True)
class IntrinsicsEstimate:
    """OpenCV pinhole + radial-tangential intrinsics with their standard deviations."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, float, float, float]
    """``(k1, k2, p1, p2)``."""
    std: dict[str, float]
    """Standard deviations of ``fx fy cx cy k1 k2 p1 p2``."""
    rms_px: float
    views: int

    @property
    def matrix(self) -> FloatArray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class Advice:
    """What to do next."""

    kind: AdviceKind
    text: str
    target_cell: tuple[int, int] | None = None
    """``(column, row)`` of the image cell to move the board to."""
    tilt: Tilt | None = None


@dataclass(frozen=True)
class GuideFrame:
    """The guide's view of one frame."""

    corners: FloatArray | None
    """``(N, 2)`` inner corners in pixels, or ``None`` when the board was not found."""
    still: bool
    captured: bool
    advice: Advice
    estimate: IntrinsicsEstimate | None


@dataclass
class _View:
    corners: FloatArray
    rotation: FloatArray
    center: FloatArray


@dataclass
class IntrinsicsGuide:
    """Keeps the frames that add to the calibration and says what to do next."""

    board: Checkerboard
    width: int
    height: int
    options: GuideOptions = field(default_factory=GuideOptions)
    views: list[_View] = field(default_factory=list)
    coverage: NDArray[np.int64] = field(init=False)
    tilts: set[Tilt] = field(default_factory=set)
    estimate: IntrinsicsEstimate | None = None
    _previous: FloatArray | None = None

    def __post_init__(self) -> None:
        columns, rows = self.options.grid
        self.coverage = np.zeros((rows, columns), dtype=np.int64)

    # ------------------------------------------------------------------ frames

    def update(self, gray: NDArray[np.uint8]) -> GuideFrame:
        """Detect the board in one grey frame, keep the frame if it adds, and advise."""

        corners = find_corners(gray, self.board)
        return self.update_corners(corners)

    def update_corners(self, corners: FloatArray | None) -> GuideFrame:
        """:meth:`update` for corners already found (``None``: no board in the frame)."""

        previous, self._previous = self._previous, corners
        if corners is None:
            return GuideFrame(None, False, False, self._advice(None), self.estimate)
        still = previous is not None and bool(
            np.median(np.linalg.norm(corners - previous, axis=1)) <= self.options.still_px
        )
        captured = False
        if still and len(self.views) < self.options.max_views:
            view = self._view(corners)
            if self._adds(view):
                self._keep(view)
                captured = True
        advice = self._advice(corners) if still or captured else self._moving_advice(corners)
        return GuideFrame(corners, still, captured, advice, self.estimate)

    @property
    def done(self) -> bool:
        return self._remaining() == []

    # ------------------------------------------------------------------ views

    def _camera_guess(self) -> FloatArray:
        if self.estimate is not None:
            return self.estimate.matrix
        focal = 0.9 * max(self.width, self.height)
        return np.array([[focal, 0.0, self.width / 2], [0.0, focal, self.height / 2], [0, 0, 1.0]])

    def _view(self, corners: FloatArray) -> _View:
        cv2 = _opencv()
        distortion = (
            np.array([*self.estimate.distortion, 0.0]) if self.estimate is not None else None
        )
        ok, rvec, _ = cv2.solvePnP(
            self.board.object_points(),
            corners.reshape(-1, 1, 2),
            self._camera_guess(),
            distortion,
            flags=cv2.SOLVEPNP_IPPE,
        )
        rotation = cv2.Rodrigues(rvec)[0] if ok else np.eye(3)
        return _View(corners, np.asarray(rotation, np.float64), corners.mean(axis=0))

    def _cells(self, corners: FloatArray) -> set[tuple[int, int]]:
        columns, rows = self.options.grid
        cells = set()
        for x, y in corners:
            column = min(max(int(x / self.width * columns), 0), columns - 1)
            row = min(max(int(y / self.height * rows), 0), rows - 1)
            cells.add((column, row))
        return cells

    def _adds(self, view: _View) -> bool:
        if any(self.coverage[row, column] == 0 for column, row in self._cells(view.corners)):
            return True
        if any(tilt not in self.tilts for tilt in view_tilts(view.rotation, self.options)):
            return True
        if self._requirements_met():
            return False
        if self.tilts == set(TILTS) and not self.coverage.all():
            return False
        diagonal = math.hypot(self.width, self.height)
        for kept in self.views:
            angle = _angle_deg(kept.rotation.T @ view.rotation)
            shift = float(np.linalg.norm(kept.center - view.center)) / diagonal
            if angle < self.options.new_pose_deg and shift < self.options.new_pose_fraction:
                return False
        return bool(self.coverage.all()) and self.tilts == set(TILTS)

    def _keep(self, view: _View) -> None:
        self.views.append(view)
        for column, row in self._cells(view.corners):
            self.coverage[row, column] += 1
        self.tilts.update(view_tilts(view.rotation, self.options))
        if len(self.views) >= self.options.min_views and len(self.tilts) >= 2:
            self.estimate = calibrate(
                [kept.corners for kept in self.views], self.board, self.width, self.height
            )

    # ------------------------------------------------------------------ advice

    def _requirements_met(self) -> bool:
        estimate = self.estimate
        if estimate is None:
            return False
        o = self.options
        return (
            len(self.views) >= o.min_views_done
            and estimate.std["fx"] <= o.focal_rel_std * estimate.fx
            and estimate.std["fy"] <= o.focal_rel_std * estimate.fy
            and estimate.std["cx"] <= o.principal_std_px
            and estimate.std["cy"] <= o.principal_std_px
            and estimate.rms_px <= o.max_rms_px
        )

    def _remaining(self) -> list[str]:
        remaining = []
        if not self.coverage.all():
            remaining.append("coverage")
        if self.tilts != set(TILTS):
            remaining.append("tilts")
        if not self._requirements_met():
            remaining.append("precision")
        return remaining

    def _moving_advice(self, corners: FloatArray) -> Advice:
        if self.done:
            return Advice("done", "Done: press s to save, q to quit")
        return Advice("hold", "Hold the board still")

    def _advice(self, corners: FloatArray | None) -> Advice:
        if self.done:
            return Advice("done", "Done: press s to save, q to quit")
        if corners is None:
            return Advice("find", "Show the whole board to the camera")
        if not self.coverage.all():
            target = self._nearest_uncovered(corners.mean(axis=0))
            return Advice("move", "Move the board to the highlighted area", target_cell=target)
        missing = [tilt for tilt in TILTS if tilt not in self.tilts]
        if missing:
            tilt = missing[0]
            edge = {"left": "left", "right": "right", "up": "top", "down": "bottom"}[tilt]
            return Advice("tilt", f"Tilt ~30 deg: turn the board's {edge} edge away", tilt=tilt)
        estimate = self.estimate
        if estimate is not None and estimate.rms_px > self.options.max_rms_px:
            return Advice("vary", "Reprojection error is high: check the board is flat")
        if estimate is not None and max(estimate.std["cx"], estimate.std["cy"]) > (
            self.options.principal_std_px
        ):
            return Advice("vary", "Tilt the board near the image corners and edges")
        return Advice("vary", "Vary distance and tilt: bring the board closer and farther")

    def _nearest_uncovered(self, center: FloatArray) -> tuple[int, int]:
        columns, rows = self.options.grid
        best, best_distance = (0, 0), math.inf
        for row in range(rows):
            for column in range(columns):
                if self.coverage[row, column]:
                    continue
                cell = np.array(
                    [(column + 0.5) * self.width / columns, (row + 0.5) * self.height / rows]
                )
                distance = float(np.linalg.norm(cell - center))
                if distance < best_distance:
                    best, best_distance = (column, row), distance
        return best


DETECT_MAX_SIDE_PX = 640
"""Larger frames are searched at this size and the corners refined at full resolution,
so a 1280 x 960 frame takes about as long as a 640 x 480 one."""


def find_corners(gray: NDArray[np.uint8], board: Checkerboard) -> FloatArray | None:
    """The board's inner corners in pixels (columns fastest), or ``None``."""

    cv2 = _opencv()
    image = np.ascontiguousarray(gray, dtype=np.uint8)
    pattern = (board.columns, board.rows)
    scale = min(1.0, DETECT_MAX_SIDE_PX / max(image.shape))
    small = (
        image
        if scale == 1.0
        else cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    )
    found, corners = cv2.findChessboardCornersSB(small, pattern, flags=cv2.CALIB_CB_ACCURACY)
    if not found:
        return None
    corners = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
    if scale < 1.0:
        # Pixel centres: x_full + 0.5 = (x_small + 0.5) / scale.
        corners = (corners + 0.5) / scale - 0.5
        spacing = np.median(np.linalg.norm(np.diff(corners.reshape(-1, 2), axis=0), axis=1))
        half = max(2, min(11, int(spacing / 4)))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)
        corners = cv2.cornerSubPix(image, corners, (half, half), (-1, -1), criteria)
    return np.asarray(corners, dtype=np.float64).reshape(-1, 2)


def view_tilts(rotation: FloatArray, options: GuideOptions) -> set[Tilt]:
    """Which way a board pose (``p_camera = rotation p_board + t``) is tilted.

    The board's normal is turned toward the camera; "left" means the board's left edge
    is farther from the camera than its right edge, so its normal points to the left.
    """

    normal = rotation[:, 2]
    if normal[2] > 0.0:
        normal = -normal
    sideways = math.degrees(math.atan2(normal[0], -normal[2]))
    vertical = math.degrees(math.atan2(normal[1], -normal[2]))
    tilts: set[Tilt] = set()
    if sideways <= -options.tilt_deg:
        tilts.add("left")
    if sideways >= options.tilt_deg:
        tilts.add("right")
    if vertical <= -options.tilt_deg:
        tilts.add("up")
    if vertical >= options.tilt_deg:
        tilts.add("down")
    return tilts


def _angle_deg(rotation: FloatArray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(rotation)) - 1.0) / 2.0))))


def calibrate(
    corners: list[FloatArray], board: Checkerboard, width: int, height: int
) -> IntrinsicsEstimate:
    """OpenCV calibration (k3 fixed at 0) of the kept views, with parameter stds."""

    cv2 = _opencv()
    objects = [board.object_points().astype(np.float32)] * len(corners)
    images = [view.reshape(-1, 1, 2).astype(np.float32) for view in corners]
    rms, matrix, distortion, _, _, std_intrinsics, _, _ = cv2.calibrateCameraExtended(
        objects, images, (width, height), None, None, flags=cv2.CALIB_FIX_K3
    )
    std = np.asarray(std_intrinsics, dtype=np.float64).ravel()
    names = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2")
    coefficients = np.asarray(distortion, dtype=np.float64).ravel()
    return IntrinsicsEstimate(
        width=width,
        height=height,
        fx=float(matrix[0, 0]),
        fy=float(matrix[1, 1]),
        cx=float(matrix[0, 2]),
        cy=float(matrix[1, 2]),
        distortion=(
            float(coefficients[0]),
            float(coefficients[1]),
            float(coefficients[2]),
            float(coefficients[3]),
        ),
        std={name: float(std[index]) for index, name in enumerate(names)},
        rms_px=float(rms),
        views=len(corners),
    )


def camchain_entry(estimate: IntrinsicsEstimate) -> dict[str, Any]:
    """The estimate as one camera of a Kalibr camchain (``pinhole`` / ``radtan``)."""

    return {
        "camera_model": "pinhole",
        "intrinsics": [estimate.fx, estimate.fy, estimate.cx, estimate.cy],
        "distortion_model": "radtan",
        "distortion_coeffs": list(estimate.distortion),
        "resolution": [estimate.width, estimate.height],
    }


# ---------------------------------------------------------------------- drawing

_GREEN = (80, 200, 80)
_YELLOW = (0, 220, 255)
_RED = (60, 60, 230)
_WHITE = (255, 255, 255)
_GREY = (150, 150, 150)


def draw_guide(
    image: NDArray[np.uint8], guide: IntrinsicsGuide, frame: GuideFrame
) -> NDArray[np.uint8]:
    """The frame (BGR or grey) with the coverage, the target, the advice and the estimate."""

    cv2 = _opencv()
    canvas = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    height, width = canvas.shape[:2]
    columns, rows = guide.options.grid
    tint = canvas.copy()
    for row in range(rows):
        for column in range(columns):
            top_left = (int(column * width / columns), int(row * height / rows))
            bottom_right = (int((column + 1) * width / columns), int((row + 1) * height / rows))
            if guide.coverage[row, column]:
                cv2.rectangle(tint, top_left, bottom_right, _GREEN, -1)
            cv2.rectangle(canvas, top_left, bottom_right, _GREY, 1)
    canvas = cv2.addWeighted(tint, 0.18, canvas, 0.82, 0.0)
    advice = frame.advice
    if frame.corners is not None:
        color = _GREEN if frame.still else _RED
        for x, y in frame.corners:
            cv2.circle(canvas, (round(x), round(y)), 3, color, -1)
        if frame.captured:
            cv2.polylines(canvas, [_outline(frame.corners, guide.board)], True, _WHITE, 3)
    if advice.target_cell is not None:
        column, row = advice.target_cell
        top_left = (int(column * width / columns), int(row * height / rows))
        bottom_right = (int((column + 1) * width / columns), int((row + 1) * height / rows))
        cv2.rectangle(canvas, top_left, bottom_right, _YELLOW, 3)
        if frame.corners is not None:
            start = frame.corners.mean(axis=0)
            end = ((top_left[0] + bottom_right[0]) / 2, (top_left[1] + bottom_right[1]) / 2)
            cv2.arrowedLine(
                canvas,
                (int(start[0]), int(start[1])),
                (int(end[0]), int(end[1])),
                _YELLOW,
                3,
                tipLength=0.08,
            )
    if advice.tilt is not None and frame.corners is not None:
        _draw_tilt(canvas, frame.corners.mean(axis=0), advice.tilt)
    scale = max(0.5, width / 1280.0)
    color = _YELLOW if advice.kind != "done" else _GREEN
    for index, line in enumerate(_wrapped(advice.text, scale * 0.9, width - 24)):
        _text(canvas, line, (12, int((34 + 32 * index) * scale)), scale * 0.9, color)
    for index, line in enumerate(status_lines(guide)):
        _text(
            canvas, line, (12, height - int((12 + 26 * (3 - index)) * scale)), scale * 0.6, _WHITE
        )
    return canvas


def status_lines(guide: IntrinsicsGuide) -> list[str]:
    """Views, coverage, tilts and the estimate with its stds, as short lines."""

    tilts = " ".join(f"{tilt}{'+' if tilt in guide.tilts else '-'}" for tilt in TILTS)
    covered = int(np.count_nonzero(guide.coverage))
    lines = [f"views {len(guide.views)}  cells {covered}/{guide.coverage.size}  tilts {tilts}"]
    estimate = guide.estimate
    if estimate is None:
        lines.append(f"calibrating after {guide.options.min_views} views and 2 tilts")
        lines.append("")
    else:
        s = estimate.std
        lines.append(
            f"fx {estimate.fx:.1f}+-{s['fx']:.1f}  fy {estimate.fy:.1f}+-{s['fy']:.1f}  "
            f"rms {estimate.rms_px:.2f} px"
        )
        lines.append(
            f"cx {estimate.cx:.1f}+-{s['cx']:.1f}  cy {estimate.cy:.1f}+-{s['cy']:.1f}  "
            f"k1 {estimate.distortion[0]:+.3f} k2 {estimate.distortion[1]:+.3f}"
        )
    return lines


def _outline(corners: FloatArray, board: Checkerboard) -> NDArray[np.int32]:
    grid = corners.reshape(board.rows, board.columns, 2)
    ring = [grid[0, 0], grid[0, -1], grid[-1, -1], grid[-1, 0]]
    return np.array(ring, dtype=np.int32).reshape(-1, 1, 2)


def _draw_tilt(canvas: NDArray[np.uint8], center: FloatArray, tilt: Tilt) -> None:
    cv2 = _opencv()
    dx, dy = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}[tilt]
    length = 0.12 * max(canvas.shape[:2])
    start = (int(center[0] - dx * length / 2), int(center[1] - dy * length / 2))
    end = (int(center[0] + dx * length / 2), int(center[1] + dy * length / 2))
    cv2.arrowedLine(canvas, start, end, _YELLOW, 4, tipLength=0.3)
    cv2.ellipse(
        canvas,
        (int(center[0]), int(center[1])),
        (int(length / 2), int(length / 6)) if dy == 0 else (int(length / 6), int(length / 2)),
        0,
        0,
        360,
        _YELLOW,
        2,
    )


def _wrapped(text: str, scale: float, max_width: int) -> list[str]:
    """``text`` cut into lines no wider than ``max_width`` pixels at ``scale``."""

    cv2 = _opencv()
    thickness = max(1, round(2 * scale))
    lines: list[str] = []
    for word in text.split():
        candidate = f"{lines[-1]} {word}" if lines else word
        width = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
        if lines and width <= max_width:
            lines[-1] = candidate
        else:
            lines.append(word)
    return lines


def _text(
    canvas: NDArray[np.uint8],
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
) -> None:
    cv2 = _opencv()
    thickness = max(1, round(2 * scale))
    cv2.putText(
        canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA
    )
    cv2.putText(
        canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA
    )


# ---------------------------------------------------------------------- live loop


def frames_from(
    source: int | str, resolution: tuple[int, int] | None = None
) -> Iterator[NDArray[np.uint8]]:
    """BGR frames from a camera index or a video file, until it ends.

    ``resolution`` asks a camera for ``(width, height)``; the camera may pick another.
    """

    cv2 = _opencv()
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise OSError(f"cannot open camera or video {source!r}")
    if resolution is not None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                return
            yield image
    finally:
        capture.release()


@dataclass(frozen=True)
class GuideRun:
    """How a guided calibration ended."""

    estimate: IntrinsicsEstimate | None
    done: bool
    saved: bool
    frames: int


def run_guided_calibration(
    frames: Iterator[NDArray[np.uint8]],
    board: Checkerboard,
    *,
    options: GuideOptions | None = None,
    show: bool = True,
    record: Path | None = None,
    save: Callable[[IntrinsicsEstimate], None] | None = None,
    stop_when_done: bool = False,
) -> GuideRun:
    """Guide through ``frames``: draw the advice, keep the useful ones, save on ``s``.

    With ``show`` the frames are shown in a window (``q`` quits, ``s`` saves); without it
    the loop runs to the end of ``frames`` (or until done with ``stop_when_done``) and
    saves what it has. ``record`` writes the annotated frames to a video file.
    """

    cv2 = _opencv()
    guide: IntrinsicsGuide | None = None
    writer: Any = None
    saved = False
    count = 0
    window = "calibrex intrinsics"
    try:
        for image in frames:
            count += 1
            gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if guide is None:
                guide = IntrinsicsGuide(
                    board, gray.shape[1], gray.shape[0], options or GuideOptions()
                )
            frame = guide.update(gray)
            canvas = draw_guide(image, guide, frame)
            if record is not None:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    writer = cv2.VideoWriter(
                        str(record), fourcc, 15.0, (canvas.shape[1], canvas.shape[0])
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
                if key == ord("s") and guide.estimate is not None and save is not None:
                    save(guide.estimate)
                    saved = True
            elif stop_when_done and guide.done:
                break
    finally:
        if writer is not None:
            writer.release()
        if show:
            with contextlib.suppress(cv2.error):
                cv2.destroyWindow(window)
    estimate = guide.estimate if guide is not None else None
    if not show and estimate is not None and save is not None:
        save(estimate)
        saved = True
    return GuideRun(estimate, guide.done if guide is not None else False, saved, count)
