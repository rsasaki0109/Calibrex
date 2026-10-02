"""Render the ``calibrex check`` yaw-sweep and rig README GIFs from real check summaries.

Inputs (all real, nothing mocked):

* ``docs/assets/calibrex_check_sweep/step_*.json``: one summary per injected yaw angle,
  produced by ``tools/check_yaw_sweep_run.py`` running the real ``calibrex check`` on the
  pooled KITTI dev bag;
* a KITTI raw drive (``image_02`` + ``velodyne_points``) and the date-level calibration
  files, only for the camera overlay of the yaw-sweep GIF and the camera frame of the rig GIF.

Perturbation chain.  ``calibrex check --tf`` replaces the ``velo_link`` edge
``T_base<-velo`` of the deployed ``/tf_static`` by ``R' = Rz(delta) R`` (rotation about the
parent ``imu_link = base_link`` z axis, translation unchanged).  The camera is fixed in the
rig, so the overlay projects a Velodyne point ``p`` as::

    p_cam = T_cam<-velo(vendor) * inv(T_base<-velo(vendor)) * T_base<-velo(delta) * p
    pixel = P_rect_02 * R_rect_00 * p_cam

With ``delta = 0`` this is exactly the KITTI calibration.

Usage (needs numpy and Pillow)::

    python tools/check_sweep_gifs.py yaw-sweep --drive-dir DRIVE --image-index 40
    python tools/check_sweep_gifs.py rig --drive-dir DRIVE
    python tools/check_sweep_gifs.py manifest --drive-dir DRIVE
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from PIL import Image as PILImage

REPO = Path(__file__).resolve().parents[1]
SWEEP_DIR = REPO / "docs/assets/calibrex_check_sweep"
YAW_GIF = Path("docs/assets/calibrex-check-yaw-sweep.gif")
RIG_GIF = Path("docs/assets/calibrex-check-rig.gif")
GIF_MANIFEST = SWEEP_DIR / "gifs.json"
GIF_MANIFEST_SCHEMA = "calibrex.check_sweep_gifs/v0"

WIDTH = 960
HEIGHT = 540
FPS = 10
HOLD_FRAMES = 4  # extra frames held at the +/- extremes
MAX_BYTES = 1_500_000
PAIR_ORDER = ("lidar-vehicle", "lidar-wheel_odometry", "ins-lidar", "imu-vehicle")
ROTATION_EXAGGERATION = 8
DEFAULT_IMAGE_INDEX = 100
POINT_STRIDE = 2  # draw every 2nd Velodyne return: keeps the GIF small
GROUND_Z_VELO = -1.55  # Velodyne is ~1.73 m above ground: drop ground returns

Color = tuple[int, int, int]
BG: Color = (22, 27, 34)
CARD: Color = (33, 40, 51)
BORDER: Color = (62, 72, 88)
TEXT: Color = (236, 240, 246)
MUTED: Color = (150, 162, 180)
STATUS_COLOR: dict[str, Color] = {
    "pass": (46, 160, 90),
    "warn": (224, 160, 30),
    "fail": (220, 60, 60),
    "inconclusive": (128, 136, 150),
}
STATUS_LABEL = {"pass": "PASS", "warn": "WARN", "fail": "FAIL", "inconclusive": "INCONCLUSIVE"}
AXIS_COLOR: tuple[Color, Color, Color] = ((235, 70, 70), (80, 200, 100), (80, 130, 245))


# --------------------------------------------------------------------------- summaries


def load_steps(directory: Path = SWEEP_DIR) -> list[dict[str, Any]]:
    """Load the per-step check summaries, sorted by injected angle."""

    steps = [json.loads(path.read_text(encoding="utf-8")) for path in directory.glob("step_*.json")]
    steps.sort(key=lambda step: step["yaw_injected_deg"])
    return steps


def loop_order(count: int) -> list[int]:
    """Indices going up the sweep then back down, without repeating the end points."""

    return [*range(count), *range(count - 2, 0, -1)]


def frame_durations_ms(count: int, fps: int = FPS, hold: int = HOLD_FRAMES) -> list[int]:
    """Per-frame durations for a loop of ``count`` sweep frames; the extremes are held."""

    base = round(1000 / fps)
    durations = [base] * count
    up = (count + 2) // 2  # frames in the ascending half
    durations[0] += hold * base
    durations[up - 1] += hold * base
    return durations


def pair_summary(step: dict[str, Any], pair: str) -> dict[str, Any] | None:
    """The summary of one pair at one step, or ``None`` when the pair was not run."""

    return step["pairs"].get(pair)


def pair_yaw_axis(step: dict[str, Any], pair: str) -> dict[str, Any] | None:
    """The yaw axis of a pair at a step, or ``None`` when the pair does not judge yaw."""

    record = pair_summary(step, pair)
    return None if record is None else record["axes"].get("yaw")


# --------------------------------------------------------------------------- geometry


def rotation_z(angle_deg: float) -> np.ndarray:
    """Rotation matrix about z."""

    a = math.radians(angle_deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])


def quat_to_matrix(q: Sequence[float]) -> np.ndarray:
    """Rotation matrix of a unit quaternion ``(x, y, z, w)``."""

    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def se3(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    """4x4 homogeneous transform."""

    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = translation
    return out


def invert(transform: np.ndarray) -> np.ndarray:
    """Exact inverse of a 4x4 transform (KITTI rotations are only 7-digit orthonormal)."""

    return np.linalg.inv(transform)


def perturbed_velo_tf(vendor: np.ndarray, yaw_deg: float) -> np.ndarray:
    """``T_base<-velo`` after ``--tf``: ``R' = Rz(yaw) R`` about the parent z, same translation."""

    out = vendor.copy()
    out[:3, :3] = rotation_z(yaw_deg) @ vendor[:3, :3]
    return out


def vendor_velo_tf(step: dict[str, Any]) -> np.ndarray:
    """``T_base<-velo`` of a step's summary (the tf that step handed to ``calibrex check``)."""

    tf = step["velo_link_tf"]
    return se3(quat_to_matrix(tf["rotation_quat_xyzw"]), tf["translation_m"])


def vendor_from_steps(steps: Sequence[dict[str, Any]]) -> np.ndarray:
    """The unperturbed ``T_base<-velo``: the summary closest to zero, with its yaw undone."""

    step = min(steps, key=lambda s: abs(s["yaw_injected_deg"]))
    tf = vendor_velo_tf(step)
    return perturbed_velo_tf(tf, -step["yaw_injected_deg"])


# --------------------------------------------------------------------------- KITTI input


def _kv_floats(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, _, rest = line.partition(":")
        try:
            values[key.strip()] = np.array([float(v) for v in rest.split()])
        except ValueError:
            continue
    return values


class KittiCalib:
    """The KITTI date-level calibration needed for the camera overlay."""

    def __init__(self, date_dir: Path) -> None:
        velo_cam = _kv_floats(date_dir / "calib_velo_to_cam.txt")
        cam_cam = _kv_floats(date_dir / "calib_cam_to_cam.txt")
        self.t_cam_velo = se3(velo_cam["R"].reshape(3, 3), velo_cam["T"])
        rect = np.eye(4)
        rect[:3, :3] = cam_cam["R_rect_00"].reshape(3, 3)
        self.r_rect = rect
        self.p_rect2 = cam_cam["P_rect_02"].reshape(3, 4)
        self.size = (int(cam_cam["S_rect_02"][0]), int(cam_cam["S_rect_02"][1]))
        imu_velo = _kv_floats(date_dir / "calib_imu_to_velo.txt")
        self.t_velo_imu = se3(imu_velo["R"].reshape(3, 3), imu_velo["T"])
        self.files = [
            date_dir / "calib_velo_to_cam.txt",
            date_dir / "calib_cam_to_cam.txt",
            date_dir / "calib_imu_to_velo.txt",
        ]

    def project(
        self, points_velo: np.ndarray, t_base_velo: np.ndarray, t_base_velo_vendor: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pixels (N, 2) and depths (N,) of Velodyne points under a (perturbed) ``T_base<-velo``."""

        camera_from_base = self.t_cam_velo @ invert(t_base_velo_vendor)
        homogeneous = np.hstack([points_velo, np.ones((len(points_velo), 1))])
        cam = (self.r_rect @ camera_from_base @ t_base_velo @ homogeneous.T).T
        depth = cam[:, 2]
        pixels = (self.p_rect2 @ cam.T).T
        scale = np.where(pixels[:, 2] > 1e-6, pixels[:, 2], 1e-6)
        return pixels[:, :2] / scale[:, None], depth


def read_velodyne(path: Path) -> np.ndarray:
    """Read a KITTI ``.bin`` scan as (N, 4) float32 (x, y, z, reflectance)."""

    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def sha256_file(path: Path) -> str:
    """SHA-256 of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def drive_inputs(drive_dir: Path, image_index: int) -> dict[str, Path]:
    """The raw files one overlay frame is built from."""

    name = f"{image_index:010d}"
    return {
        "image": drive_dir / "image_02/data" / f"{name}.png",
        "velodyne": drive_dir / "velodyne_points/data" / f"{name}.bin",
    }


# --------------------------------------------------------------------------- drawing


def _pil() -> Any:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("Pillow is required to render the GIFs (pip install pillow)") from exc
    return Image, ImageDraw, ImageFont


def load_font(size: int, bold: bool = False) -> Any:
    """DejaVu Sans via Pillow, falling back to Pillow's default font."""

    _, _, image_font = _pil()
    names = ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf") if bold else ("DejaVuSans.ttf",)
    for name in names:
        try:
            return image_font.truetype(name, size)
        except OSError:
            continue
    return image_font.load_default()


DEPTH_ANCHORS = np.array(
    [
        [0.00, 255, 70, 60],
        [0.20, 255, 170, 40],
        [0.40, 240, 235, 70],
        [0.60, 70, 220, 120],
        [0.80, 60, 190, 240],
        [1.00, 130, 110, 255],
    ]
)


def depth_colors(depth: np.ndarray, near: float = 5.0, far: float = 30.0) -> np.ndarray:
    """Near-to-far colour ramp (red, amber, yellow, green, cyan, violet), (N, 3) uint8."""

    t = np.clip((depth - near) / (far - near), 0.0, 1.0)
    return np.stack(
        [np.interp(t, DEPTH_ANCHORS[:, 0], DEPTH_ANCHORS[:, c]) for c in (1, 2, 3)], axis=1
    ).astype(np.uint8)


def text_width(draw: Any, text: str, font: Any) -> int:
    """Pixel width of ``text``."""

    return int(draw.textlength(text, font=font))


def draw_pill(
    draw: Any, x: int, y: int, label: str, color: Color, font: Any, height: int = 22
) -> int:
    """Rounded status pill with its left edge at ``x``; returns the right edge."""

    width = text_width(draw, label, font) + 18
    draw.rounded_rectangle((x, y, x + width, y + height), radius=height // 2, fill=color)
    draw.text((x + 9, y + height // 2), label, font=font, fill=(255, 255, 255), anchor="lm")
    return x + width


def draw_pill_right(
    draw: Any, right: int, y: int, label: str, color: Color, font: Any, height: int = 22
) -> int:
    """Rounded status pill with its right edge at ``right``; returns the left edge."""

    width = text_width(draw, label, font) + 18
    draw_pill(draw, right - width, y, label, color, font, height)
    return right - width


def pair_pill(record: dict[str, Any] | None) -> tuple[str, Color, bool]:
    """Label, colour and partial marker of a pair record."""

    if record is None:
        return "SKIPPED", STATUS_COLOR["inconclusive"], False
    status = record["status"]
    partial = record.get("coverage") == "partial" and bool(record["axes"])
    return STATUS_LABEL.get(status, status.upper()), STATUS_COLOR.get(status, MUTED), partial


def draw_header(draw: Any, title: str, subtitle: str) -> None:
    """Title strip."""

    draw.text((16, 14), title, font=load_font(20, True), fill=TEXT)
    draw.text((WIDTH - 16, 22), subtitle, font=load_font(12), fill=MUTED, anchor="rm")


def draw_verdict_panel(
    draw: Any, step: dict[str, Any], box: tuple[int, int, int, int], *, show_pairs: bool = True
) -> None:
    """Injected angle, overall verdict and the per-pair verdict pills."""

    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=10, fill=CARD, outline=BORDER)
    angle = step["yaw_injected_deg"]
    draw.text((x0 + 14, y0 + 8), "injected yaw on velo_link", font=load_font(12), fill=MUTED)
    draw.text((x0 + 14, y0 + 22), f"{angle:+.2f}°", font=load_font(42, True), fill=TEXT)
    verdict = step["overall_verdict"]
    draw.text((x0 + 14, y0 + 86), "calibrex check overall", font=load_font(12), fill=MUTED)
    draw_pill_right(
        draw,
        x1 - 14,
        y0 + 80,
        STATUS_LABEL.get(verdict, verdict.upper()),
        STATUS_COLOR.get(verdict, MUTED),
        load_font(14, True),
        height=26,
    )
    if not show_pairs:
        return
    small = load_font(11)
    name_font = load_font(13, True)
    pill_font = load_font(11, True)
    y = y0 + 126
    for pair in PAIR_ORDER:
        record = pair_summary(step, pair)
        label, color, partial = pair_pill(record)
        draw.text((x0 + 14, y), pair, font=name_font, fill=TEXT)
        left = draw_pill_right(draw, x1 - 14, y - 2, label, color, pill_font, height=20)
        if partial:
            draw.text((left - 6, y + 8), "partial", font=small, fill=MUTED, anchor="rm")
        axes = {} if record is None else record["axes"]
        if "yaw" in axes:
            axis = axes["yaw"]
            err, tol = abs(axis["candidate_error_deg"]), axis["tolerance_deg"]
            note = f"yaw |δ| {err:.2f}° / tol {tol:.2f}°"
        elif axes:
            note = "judged: " + ", ".join(
                f"{n} {abs(a['candidate_error_deg']):.2f}°" for n, a in axes.items()
            )
            note += "; yaw not judged"
        elif record is not None:
            note = "no axis constrained by the data"
        else:
            note = "not run"
        draw.text((x0 + 14, y + 20), note, font=small, fill=MUTED)
        y += 47
    _ = y1


def draw_ratio_bar(draw: Any, step: dict[str, Any], box: tuple[int, int, int, int]) -> None:
    """``|error| / tolerance`` bar for the lidar-vehicle yaw axis, with the tolerance ticks."""

    x0, y0, x1, _y1 = box
    axis = pair_yaw_axis(step, "lidar-vehicle")
    font = load_font(12)
    draw.text(
        (x0, y0),
        "lidar-vehicle  yaw  |error| / tolerance   (pass ≤ 1\u00d7, fail > 2\u00d7)",
        font=load_font(12, True),
        fill=TEXT,
    )
    bar_y0, bar_y1 = y0 + 22, y0 + 44
    scale_max = 7.0
    width = x1 - x0

    def px(value: float) -> int:
        return x0 + round(width * min(value, scale_max) / scale_max)

    draw.rectangle((x0, bar_y0, x1, bar_y1), fill=(44, 52, 66))
    for lo, hi, key in ((0.0, 1.0, "pass"), (1.0, 2.0, "warn"), (2.0, scale_max, "fail")):
        zone = tuple(int(c * 0.35 + 44 * 0.65) for c in STATUS_COLOR[key])
        draw.rectangle((px(lo), bar_y0, px(hi), bar_y1), fill=zone)
    if axis is not None:
        ratio = abs(axis["candidate_error_deg"]) / axis["tolerance_deg"]
        status = axis["status"]
        draw.rectangle(
            (x0, bar_y0 + 5, px(ratio), bar_y1 - 5), fill=STATUS_COLOR.get(status, MUTED)
        )
        err, tol = abs(axis["candidate_error_deg"]), axis["tolerance_deg"]
        value = f"{ratio:.2f}\u00d7 tol  ({err:.2f}° / {tol:.2f}°)"
        draw.text((x1, y0), value, font=font, fill=TEXT, anchor="ra")
    for tick in (1.0, 2.0):
        draw.line((px(tick), bar_y0 - 4, px(tick), bar_y1 + 4), fill=TEXT, width=2)
        draw.text((px(tick), bar_y1 + 6), f"{tick:g}\u00d7 tol", font=font, fill=MUTED, anchor="ma")
    draw.text((x0, bar_y1 + 6), "0", font=font, fill=MUTED, anchor="la")
    draw.text((x1, bar_y1 + 6), f"{scale_max:g}\u00d7", font=font, fill=MUTED, anchor="ra")


def draw_sweep_track(
    draw: Any,
    steps: Sequence[dict[str, Any]],
    current: int,
    box: tuple[int, int, int, int],
    *,
    label: bool = True,
) -> None:
    """A timeline of every injected angle coloured by its overall verdict, with a marker."""

    x0, y0, x1, y1 = box
    angles = [s["yaw_injected_deg"] for s in steps]
    lo, hi = min(angles), max(angles)

    def px(angle: float) -> int:
        return x0 + round((x1 - x0) * (angle - lo) / (hi - lo))

    mid = (y0 + y1) // 2
    draw.line((x0, mid, x1, mid), fill=BORDER, width=2)
    for step in steps:
        x = px(step["yaw_injected_deg"])
        color = STATUS_COLOR.get(step["overall_verdict"], MUTED)
        draw.ellipse((x - 5, mid - 5, x + 5, mid + 5), fill=color)
    cx = px(steps[current]["yaw_injected_deg"])
    draw.polygon([(cx, mid - 9), (cx - 7, mid - 20), (cx + 7, mid - 20)], fill=TEXT)
    font = load_font(11)
    for tick in (lo, 0.0, hi):
        draw.text((px(tick), y1 + 2), f"{tick:+.0f}°", font=font, fill=MUTED, anchor="ma")
    if label:
        draw.text((x0 - 8, mid), "overall verdict at each step", font=font, fill=MUTED, anchor="rm")


TRANSPARENT_INDEX = 255


def quantize_frames(frames: list[PILImage.Image]) -> list[PILImage.Image]:
    """Quantize all frames to one shared adaptive palette (no dithering) for small GIFs."""

    image_mod, _, _ = _pil()
    sample = frames[:: max(1, len(frames) // 8)]
    mosaic = image_mod.new("RGB", (WIDTH, HEIGHT * len(sample)))
    for i, frame in enumerate(sample):
        mosaic.paste(frame, (0, i * HEIGHT))
    palette_image = mosaic.quantize(colors=255, method=image_mod.Quantize.MEDIANCUT, dither=0)
    palette = (palette_image.getpalette() or [])[: 3 * TRANSPARENT_INDEX]
    palette += [0] * (768 - len(palette))
    palette_image.putpalette(palette)
    return [frame.quantize(palette=palette_image, dither=0) for frame in frames]


def delta_frames(quantized: list[PILImage.Image]) -> list[PILImage.Image]:
    """Make pixels unchanged since the previous frame transparent (frame 0 stays opaque)."""

    image_mod, _, _ = _pil()
    out = [quantized[0]]
    previous = np.asarray(quantized[0])
    for frame in quantized[1:]:
        current = np.asarray(frame)
        delta = np.where(current == previous, TRANSPARENT_INDEX, current).astype(np.uint8)
        image = image_mod.fromarray(delta, mode="P")
        image.putpalette(frame.getpalette() or [])
        out.append(image)
        previous = current
    return out


def save_gif(frames: list[PILImage.Image], durations: list[int], output: Path) -> None:
    """Write a looping GIF whose frames only carry the pixels that changed."""

    output.parent.mkdir(parents=True, exist_ok=True)
    quantized = quantize_frames(frames)
    deltas = delta_frames(quantized)
    deltas[0].save(
        output,
        save_all=True,
        append_images=deltas[1:],
        duration=durations,
        loop=0,
        optimize=False,
        disposal=1,
        transparency=TRANSPARENT_INDEX,
    )


# --------------------------------------------------------------------------- yaw sweep GIF


def render_overlay(
    base_gray: Any,
    points_velo: np.ndarray,
    calib: KittiCalib,
    t_base_velo: np.ndarray,
    t_vendor: np.ndarray,
    crop: tuple[int, int, int, int],
    size: tuple[int, int],
) -> PILImage.Image:
    """Camera image (cropped, dimmed) with depth-coloured Velodyne points projected."""

    image_mod, _, _ = _pil()
    uv, depth = calib.project(points_velo, t_base_velo, t_vendor)
    cx0, cy0, cx1, cy1 = crop
    scale = size[0] / (cx1 - cx0)
    keep = (
        (points_velo[:, 2] > GROUND_Z_VELO)
        & (depth > 3.0)
        & (depth < 45.0)
        & (uv[:, 0] >= cx0)
        & (uv[:, 0] < cx1)
        & (uv[:, 1] >= cy0)
        & (uv[:, 1] < cy1)
    )
    uv, depth = uv[keep], depth[keep]
    order = np.argsort(-depth)  # far first so near points stay on top
    uv, depth = uv[order], depth[order]
    px = np.round((uv[:, 0] - cx0) * scale).astype(int)
    py = np.round((uv[:, 1] - cy0) * scale).astype(int)
    canvas = np.array(base_gray.resize(size, image_mod.Resampling.LANCZOS).convert("RGB"))
    colors = depth_colors(depth)
    for dx in (0, 1):
        for dy in (0, 1):
            xs = np.clip(px + dx, 0, size[0] - 1)
            ys = np.clip(py + dy, 0, size[1] - 1)
            canvas[ys, xs] = colors
    return image_mod.fromarray(canvas)


def load_dimmed_image(path: Path) -> Any:
    """The camera image as a dimmed grayscale picture so the coloured points stand out."""

    image_mod, _, _ = _pil()
    gray = image_mod.open(path).convert("L")
    arr = np.asarray(gray, dtype=np.float32) * 0.55 + 14.0
    return image_mod.fromarray(arr.clip(0, 255).astype(np.uint8))


def build_yaw_sweep_frames(
    steps: Sequence[dict[str, Any]], drive_dir: Path, image_index: int
) -> tuple[list[PILImage.Image], list[int]]:
    """All frames of the yaw-sweep GIF and their durations."""

    image_mod, image_draw, _ = _pil()
    inputs = drive_inputs(drive_dir, image_index)
    calib = KittiCalib(drive_dir.parent)
    points = read_velodyne(inputs["velodyne"])[::POINT_STRIDE, :3].astype(np.float64)
    gray = load_dimmed_image(inputs["image"])
    t_vendor = vendor_from_steps(steps)
    crop = (330, 40, 890, 340)  # 560 x 300 px of the 1242 x 375 rectified image
    panel = (16, 56, 616, 56 + round(600 * 300 / 560))  # 600 x 321
    size = (panel[2] - panel[0], panel[3] - panel[1])
    order = loop_order(len(steps))
    frames = []
    for index in order:
        step = steps[index]
        t_delta = perturbed_velo_tf(t_vendor, step["yaw_injected_deg"])
        overlay = render_overlay(gray, points, calib, t_delta, t_vendor, crop, size)
        frame = image_mod.new("RGB", (WIDTH, HEIGHT), BG)
        draw = image_draw.Draw(frame)
        draw_header(
            draw,
            "calibrex check: yaw injected into the deployed tf",
            "KITTI 2011_09_26 · pooled dev drives",
        )
        frame.paste(overlay, (panel[0], panel[1]))
        draw.rectangle(panel, outline=BORDER)
        cap = load_font(11)
        draw.text(
            (panel[0], panel[3] + 6),
            f"image_02 + Velodyne scan (drive 0005, frame {image_index}); colour = depth 5-30 m, "
            "ground returns hidden",
            font=cap,
            fill=MUTED,
        )
        draw.text(
            (panel[0], panel[3] + 22),
            "points projected with the KITTI calib after turning velo_link by Rz(δ) about "
            "base_link z",
            font=cap,
            fill=MUTED,
        )
        draw_verdict_panel(draw, step, (632, 56, 944, panel[3]))
        draw_ratio_bar(draw, step, (16, 424, 944, 470))
        draw_sweep_track(draw, steps, index, (230, 498, 930, 516))
        frames.append(frame)
    return frames, frame_durations_ms(len(frames))


# --------------------------------------------------------------------------- rig GIF


def view_basis(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Rows (right, up, toward-viewer) of an orthographic z-up view."""

    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    forward = np.array(
        [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), -math.sin(el)]
    )  # camera looks along this direction
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return np.stack([right, up, -forward])


def rig_frames_spec(
    steps: Sequence[dict[str, Any]], calib: KittiCalib | None
) -> dict[str, np.ndarray]:
    """Rig nodes (4x4 in base_link) at the vendor tf: base_link=imu_link, velo_link, cam_00."""

    t_vendor = vendor_from_steps(steps)
    nodes = {"base_link": np.eye(4), "velo_link": t_vendor}
    if calib is not None:
        nodes["cam_00"] = t_vendor @ invert(calib.t_cam_velo)
    return nodes


def build_rig_frames(
    steps: Sequence[dict[str, Any]], drive_dir: Path | None
) -> tuple[list[PILImage.Image], list[int]]:
    """All frames of the rig GIF and their durations."""

    image_mod, image_draw, _ = _pil()
    calib = KittiCalib(drive_dir.parent) if drive_dir is not None else None
    nodes = rig_frames_spec(steps, calib)
    t_vendor = nodes["velo_link"]
    order = loop_order(len(steps))
    durations = frame_durations_ms(len(order))
    frames = []
    view = (16, 56, 636, 524)
    center = np.array([0.4, -0.08, 0.42])
    scale = 235.0
    axis_len = 0.36
    layer_size = (view[2] - view[0], view[3] - view[1])
    for i, index in enumerate(order):
        step = steps[index]
        angle = step["yaw_injected_deg"]
        azimuth = -40.0 + 12.0 * math.sin(2 * math.pi * i / len(order))
        basis = view_basis(azimuth, 38.0)
        cx, cy = layer_size[0] / 2, layer_size[1] / 2 + 34

        def project(point: np.ndarray, basis: np.ndarray = basis, cx: float = cx, cy: float = cy):
            v = basis @ (np.asarray(point) - center)
            return cx + scale * v[0], cy - scale * v[1], v[2]

        record = pair_summary(step, "lidar-vehicle")
        label, color, partial = pair_pill(record)
        status = record["status"] if record else "inconclusive"
        frame = image_mod.new("RGB", (WIDTH, HEIGHT), BG)
        frame_draw = image_draw.Draw(frame)
        draw_header(
            frame_draw,
            "calibrex check: the rig frame tree under a yaw error",
            "KITTI 2011_09_26 · /tf_static",
        )
        layer = image_mod.new("RGB", layer_size, CARD)  # 3D drawn on its own layer: clipped
        draw = image_draw.Draw(layer)
        _draw_ground_grid(draw, project)
        # edges
        base_xy = project(nodes["base_link"][:3, 3])[:2]
        velo_xy = project(t_vendor[:3, 3])[:2]
        draw.line((*base_xy, *velo_xy), fill=color, width=5)
        foot = project(np.array([t_vendor[0, 3], t_vendor[1, 3], -0.1]))[:2]
        _dashed(draw, velo_xy, foot, BORDER)
        if "cam_00" in nodes:
            cam_xy = project(nodes["cam_00"][:3, 3])[:2]
            _dashed(draw, base_xy, cam_xy, MUTED)
        # deployed (vendor) velo axes as thin reference, drawn rotation of the swept one
        drawn = np.eye(4)
        drawn[:3, :3] = rotation_z(ROTATION_EXAGGERATION * angle) @ t_vendor[:3, :3]
        drawn[:3, 3] = t_vendor[:3, 3]
        _draw_axes(draw, project, t_vendor, axis_len, thin=True, dim=True)
        _draw_axes(draw, project, nodes["base_link"], axis_len * 1.25)
        _draw_axes(draw, project, drawn, axis_len)
        if "cam_00" in nodes:
            _draw_axes(draw, project, nodes["cam_00"], axis_len * 0.7, dim=True)
        _draw_node(draw, project(nodes["base_link"][:3, 3]), TEXT, "base_link = imu_link", (16, 22))
        _draw_node(draw, project(drawn[:3, 3]), color, "velo_link", (16, -18), bold=True)
        if "cam_00" in nodes:
            _draw_node(
                draw, project(nodes["cam_00"][:3, 3]), MUTED, "cam_00 (calib file)", (-14, 20)
            )
        draw.text(
            (12, 10),
            f"rotation drawn \u00d7{ROTATION_EXAGGERATION} (true injected yaw {angle:+.2f}°)",
            font=load_font(14, True),
            fill=(255, 210, 120),
        )
        draw.text(
            (12, 32),
            "thin gray axes = deployed velo_link; edge colour = lidar-vehicle verdict",
            font=load_font(11),
            fill=MUTED,
        )
        draw.text(
            (12, layer_size[1] - 22),
            "cam_00 is from calib_velo_to_cam.txt (not in /tf_static, not checked)",
            font=load_font(11),
            fill=MUTED,
        )
        frame.paste(layer, (view[0], view[1]))
        frame_draw.rounded_rectangle(view, radius=10, outline=BORDER)
        # right panel
        _draw_rig_panel(frame_draw, step, steps, index, label, color, partial, status)
        frames.append(frame)
    return frames, durations


def _draw_rig_panel(
    draw: Any,
    step: dict[str, Any],
    steps: Sequence[dict[str, Any]],
    index: int,
    label: str,
    color: Color,
    partial: bool,
    status: str,
) -> None:
    x0, y0, x1, y1 = 652, 56, 944, 524
    draw.rounded_rectangle((x0, y0, x1, y1), radius=10, fill=CARD, outline=BORDER)
    draw.text((x0 + 14, y0 + 10), "injected yaw on velo_link", font=load_font(12), fill=MUTED)
    draw.text(
        (x0 + 14, y0 + 26),
        f"{step['yaw_injected_deg']:+.2f}°",
        font=load_font(46, True),
        fill=TEXT,
    )
    draw.text((x0 + 14, y0 + 92), "lidar-vehicle verdict", font=load_font(12), fill=MUTED)
    right = draw_pill(draw, x0 + 14, y0 + 110, label, color, load_font(14, True), height=28)
    if partial:
        draw.text((right + 10, y0 + 124), "partial", font=load_font(12), fill=MUTED, anchor="lm")
    axis = pair_yaw_axis(step, "lidar-vehicle")
    if axis is not None:
        ratio = abs(axis["candidate_error_deg"]) / axis["tolerance_deg"]
        draw.text(
            (x0 + 14, y0 + 150),
            f"yaw |δ| {abs(axis['candidate_error_deg']):.2f}° / tol "
            f"{axis['tolerance_deg']:.2f}° = {ratio:.2f}\u00d7",
            font=load_font(12),
            fill=TEXT,
        )
    verdict = step["overall_verdict"]
    draw.text((x0 + 14, y0 + 184), "overall (4 pairs)", font=load_font(12), fill=MUTED)
    draw_pill(
        draw,
        x0 + 14,
        y0 + 202,
        STATUS_LABEL.get(verdict, verdict.upper()),
        STATUS_COLOR.get(verdict, MUTED),
        load_font(13, True),
        height=26,
    )
    draw.text((x0 + 14, y0 + 250), "frame tree (/tf_static)", font=load_font(12), fill=MUTED)
    mono = load_font(12)
    rows = [("base_link", TEXT), ("  imu_link  (identity)", TEXT), ("    velo_link", color)]
    for k, (text, fill) in enumerate(rows):
        draw.text((x0 + 14, y0 + 272 + 20 * k), text, font=mono, fill=fill)
    tf = step["velo_link_tf"]["translation_m"]
    draw.text(
        (x0 + 14, y0 + 338),
        f"velo_link t = ({tf[0]:.3f}, {tf[1]:.3f}, {tf[2]:.3f}) m",
        font=load_font(11),
        fill=MUTED,
    )
    draw.text(
        (x0 + 14, y0 + 356),
        "translation unchanged by the injection",
        font=load_font(11),
        fill=MUTED,
    )
    draw.text(
        (x0 + 14, y0 + 396), "sweep (overall verdict per step)", font=load_font(11), fill=MUTED
    )
    draw_sweep_track(draw, steps, index, (x0 + 24, y0 + 424, x1 - 24, y0 + 446), label=False)
    _ = status


def _dashed(draw: Any, a: tuple[float, float], b: tuple[float, float], fill: Color) -> None:
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    count = max(1, int(length // 8))
    for k in range(count):
        t0, t1 = k / count, (k + 0.5) / count
        draw.line(
            (
                a[0] + (b[0] - a[0]) * t0,
                a[1] + (b[1] - a[1]) * t0,
                a[0] + (b[0] - a[0]) * t1,
                a[1] + (b[1] - a[1]) * t1,
            ),
            fill=fill,
            width=2,
        )


def _draw_ground_grid(draw: Any, project: Any) -> None:
    for gx in np.arange(-0.5, 1.51, 0.5):
        a, b = project(np.array([gx, -1.0, -0.1])), project(np.array([gx, 1.0, -0.1]))
        draw.line((a[0], a[1], b[0], b[1]), fill=BORDER, width=1)
    for gy in np.arange(-1.0, 1.01, 0.5):
        a, b = project(np.array([-0.5, gy, -0.1])), project(np.array([1.5, gy, -0.1]))
        draw.line((a[0], a[1], b[0], b[1]), fill=BORDER, width=1)


def _draw_axes(
    draw: Any,
    project: Any,
    transform: np.ndarray,
    length: float,
    *,
    thin: bool = False,
    dim: bool = False,
) -> None:
    origin = transform[:3, 3]
    o = project(origin)
    ends = []
    for k in range(3):
        tip = project(origin + transform[:3, k] * length)
        base = AXIS_COLOR[k]
        color = tuple(int(c * 0.55 + 70 * 0.45) for c in base) if dim else base
        if thin:
            color = (110, 118, 132)
        draw.line((o[0], o[1], tip[0], tip[1]), fill=color, width=1 if thin else 3)
        ends.append((tip, color, "xyz"[k]))
    if not thin:
        font = load_font(11, True)
        for tip, color, name in ends:
            draw.text((tip[0] + 4, tip[1] - 6), name, font=font, fill=color)


def _draw_node(
    draw: Any,
    xyz: tuple[float, float, float],
    color: Color,
    label: str,
    offset: tuple[int, int],
    bold: bool = False,
) -> None:
    x, y, _ = xyz
    draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color, outline=(255, 255, 255), width=2)
    font = load_font(13, bold)
    anchor = "lm" if offset[0] >= 0 else "rm"
    draw.text((x + offset[0], y + offset[1]), label, font=font, fill=TEXT, anchor=anchor)


# --------------------------------------------------------------------------- manifest / CLI


def git_commit() -> str:
    """The repository's current commit hash (or ``unknown``)."""

    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def calibrex_version() -> str:
    """The calibrex version declared in ``pyproject.toml``."""

    for line in (REPO / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        if line.startswith("version"):
            return line.split("=", 1)[1].strip().strip('"')
    return "unknown"


def build_manifest(
    drive_dir: Path | None,
    image_index: int,
    outputs: Sequence[Path],
    steps: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Provenance of the two GIFs: input digests, versions, commands and license note."""

    sweep_manifest = json.loads((SWEEP_DIR / "manifest.json").read_text(encoding="utf-8"))
    kitti_inputs: dict[str, str] = {}
    if drive_dir is not None:
        files = [
            *drive_inputs(drive_dir, image_index).values(),
            *KittiCalib(drive_dir.parent).files,
        ]
        kitti_inputs = {path.name: sha256_file(path) for path in files}
    gifs = {}
    for output in outputs:
        path = REPO / output
        if path.exists():
            gifs[output.as_posix()] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    return {
        "schema": GIF_MANIFEST_SCHEMA,
        "generator": "tools/check_sweep_gifs.py",
        "calibrex_version": calibrex_version(),
        "git_commit": git_commit(),
        "check_generator_version": sweep_manifest["generator_version"],
        "check_git_commit": sweep_manifest["git_commit"],
        "pooled_bag_input_sha256": sweep_manifest["bag_input_sha256"],
        "drives": sweep_manifest["drives"],
        "perturbation": sweep_manifest["perturbation"],
        "steps": {
            "count": len(steps),
            "angles_deg": [s["yaw_injected_deg"] for s in steps],
            "summary_sha256": {
                f"step_{s['yaw_injected_deg']:+.2f}.json": sha256_file(
                    SWEEP_DIR / f"step_{s['yaw_injected_deg']:+.2f}.json"
                )
                for s in steps
            },
        },
        "check_command": steps[0]["command"],
        "check_command_note": "shown for the first step; --tf and --output differ per step",
        "gif_commands": [
            "python tools/check_sweep_gifs.py yaw-sweep --drive-dir <KITTI drive 0005> "
            f"--image-index {image_index}",
            "python tools/check_sweep_gifs.py rig --drive-dir <KITTI drive 0005>",
        ],
        "overlay": {
            "drive": "2011_09_26_drive_0005_sync",
            "image_index": image_index,
            "projection": "pixel = P_rect_02 R_rect_00 T_cam<-velo inv(T_base<-velo) "
            "T_base<-velo(delta) p, T_base<-velo(delta) = [Rz(delta) R | t]",
            "kitti_input_sha256": kitti_inputs,
        },
        "rig_rotation_exaggeration": ROTATION_EXAGGERATION,
        "gifs": gifs,
        "dataset_license": "KITTI raw data, CC BY-NC-SA 3.0: non-commercial use only; "
        "the GIFs contain a KITTI camera image and must stay non-commercial",
        "animation": {"width": WIDTH, "height": HEIGHT, "fps": FPS},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("command", choices=["yaw-sweep", "rig", "manifest"])
    parser.add_argument("--drive-dir", type=Path, help="KITTI raw drive directory")
    parser.add_argument("--image-index", type=int, default=DEFAULT_IMAGE_INDEX)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    steps = load_steps()
    if not steps:
        parser.error(f"no step_*.json summaries in {SWEEP_DIR}")
    if args.command == "manifest":
        manifest = build_manifest(args.drive_dir, args.image_index, [YAW_GIF, RIG_GIF], steps)
        GIF_MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", "utf-8")
        return 0
    if args.command == "yaw-sweep":
        if args.drive_dir is None:
            parser.error("--drive-dir is required")
        frames, durations = build_yaw_sweep_frames(steps, args.drive_dir, args.image_index)
        output = args.output or REPO / YAW_GIF
    else:
        frames, durations = build_rig_frames(steps, args.drive_dir)
        output = args.output or REPO / RIG_GIF
    save_gif(frames, durations, output)
    size = output.stat().st_size
    print(f"wrote {output} ({len(frames)} frames, {size} bytes)")
    if size > MAX_BYTES:
        print(f"warning: larger than {MAX_BYTES} bytes", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
