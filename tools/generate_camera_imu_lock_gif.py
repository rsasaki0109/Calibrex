"""Render docs/assets/camera-imu-lock.gif from the committed Hilti exp21 rate series.

The animation starts from a deliberately wrong camera-IMU hypothesis, then moves
rotation (slerp), time offset and gyro bias (lerp) to calibrex's estimate so the
gyro curves, rotated into the camera frame, lock onto the camera-derived rates.
Everything is deterministic from ``docs/assets/camera_imu_lock/series.npz``
(produced by ``tools/extract_camera_imu_lock_series.py``); only numpy + Pillow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

FloatArray = NDArray[np.float64]

WIDTH, HEIGHT = 960, 540
SS = 2  # supersampling factor for anti-aliasing
FRAME_MS = 80
SERIES_PATH = Path("docs/assets/camera_imu_lock/series.npz")
OUTPUT_PATH = Path("docs/assets/camera-imu-lock.gif")

# Timeline in frames at 12.5 fps (80 ms): hold, solve, lock, compare, crossfade.
N_HOLD, N_SOLVE, N_LOCK, N_COMPARE, N_FADE = 12, 40, 10, 16, 6
N_FRAMES = N_HOLD + N_SOLVE + N_LOCK + N_COMPARE + N_FADE

START_ROTATION_DEG = 16.0
START_ROTATION_AXIS = (0.6, 0.5, 0.62)
START_TIME_OFFSET_ERROR_S = 0.060
VIEW_T0_S, VIEW_T1_S = 1.0, 7.0

BG = (15, 23, 42)
PANEL = (23, 33, 58)
GRID = (51, 65, 95)
TEXT = (226, 232, 240)
MUTED = (148, 163, 184)
GYRO = (251, 146, 60)
CAMERA = (56, 189, 248)
RED = (248, 113, 113)
AMBER = (251, 191, 36)
GREEN = (74, 222, 128)

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(str(FONT_DIR / name), size * SS)
    except OSError:
        return ImageFont.load_default()


def quat_to_matrix(q: FloatArray) -> FloatArray:
    x, y, z, w = (q / np.linalg.norm(q)).tolist()
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def exp_so3(rotvec: FloatArray) -> FloatArray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.eye(3)
    k = rotvec / angle
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * kx + (1 - math.cos(angle)) * kx @ kx


def rotation_angle_deg(matrix: FloatArray) -> float:
    cos = (np.trace(matrix) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, float(cos)))))


@dataclass(frozen=True)
class Series:
    cam_start_s: FloatArray
    cam_end_s: FloatArray
    cam_rate_dps: FloatArray
    imu_t_s: FloatArray
    gyro_dps: FloatArray
    gyro_cum_deg: FloatArray  # cumulative integral of the raw gyro, deg
    r_est: FloatArray
    dt_est_s: float
    bias_est_dps: FloatArray
    r_kalibr: FloatArray
    dt_kalibr_s: float


def load_series(path: Path) -> Series:
    data = np.load(path)
    gyro = np.asarray(data["gyro_rps"], dtype=np.float64) * (180.0 / math.pi)
    t = np.asarray(data["imu_t_s"], dtype=np.float64)
    cum = np.vstack(
        [np.zeros(3), np.cumsum(0.5 * (gyro[1:] + gyro[:-1]) * np.diff(t)[:, None], axis=0)]
    )
    return Series(
        cam_start_s=np.asarray(data["cam_start_s"], dtype=np.float64),
        cam_end_s=np.asarray(data["cam_end_s"], dtype=np.float64),
        cam_rate_dps=np.asarray(data["cam_rate_rps"], dtype=np.float64) * (180.0 / math.pi),
        imu_t_s=t,
        gyro_dps=gyro,
        gyro_cum_deg=cum,
        r_est=quat_to_matrix(np.asarray(data["est_quat_xyzw"], dtype=np.float64)),
        dt_est_s=float(data["est_time_offset_s"]),
        bias_est_dps=np.asarray(data["est_gyro_bias_rps"], dtype=np.float64) * (180.0 / math.pi),
        r_kalibr=np.asarray(data["kalibr_T_cam_imu"], dtype=np.float64)[:3, :3],
        dt_kalibr_s=float(data["kalibr_time_offset_s"]),
    )


@dataclass(frozen=True)
class Params:
    rotation: FloatArray
    time_offset_s: float
    bias_dps: FloatArray


def interpolate(series: Series, s: float) -> Params:
    """Parameters at progress s: 0 is the wrong hypothesis, 1 the calibrex estimate."""

    axis = np.asarray(START_ROTATION_AXIS, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    start_rotation = series.r_est @ exp_so3(axis * math.radians(START_ROTATION_DEG))
    # slerp from the hypothesis to the estimate along the geodesic
    delta = start_rotation.T @ series.r_est
    angle = rotation_angle_deg(delta)
    if angle < 1e-9:
        rotation = series.r_est
    else:
        rv = _log_so3(delta)
        rotation = start_rotation @ exp_so3(rv * s)
    return Params(
        rotation=rotation,
        time_offset_s=(1 - s) * (series.dt_est_s + START_TIME_OFFSET_ERROR_S) + s * series.dt_est_s,
        bias_dps=s * series.bias_est_dps,
    )


def _log_so3(matrix: FloatArray) -> FloatArray:
    angle = math.radians(rotation_angle_deg(matrix))
    if angle < 1e-12:
        return np.zeros(3)
    axis = np.array(
        [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]]
    )
    return axis / (2.0 * math.sin(angle)) * angle


def gyro_curve(series: Series, params: Params) -> tuple[FloatArray, FloatArray]:
    """Gyro in the camera frame, drawn on the camera clock (t_cam = t_imu - dt)."""

    rotated = (series.gyro_dps - params.bias_dps) @ params.rotation.T
    return series.imu_t_s - params.time_offset_s, rotated


def interval_rmse_dps(series: Series, params: Params) -> float:
    """RMS of the 3D rate error over the displayed camera intervals."""

    keep = (series.cam_start_s >= VIEW_T0_S) & (series.cam_end_s <= VIEW_T1_S)
    start = series.cam_start_s[keep] + params.time_offset_s
    end = series.cam_end_s[keep] + params.time_offset_s
    cum = np.column_stack(
        [np.interp(np.r_[start, end], series.imu_t_s, series.gyro_cum_deg[:, k]) for k in range(3)]
    )
    n = len(start)
    mean_imu = (cum[n:] - cum[:n]) / (end - start)[:, None] - params.bias_dps
    predicted = mean_imu @ params.rotation.T
    err = predicted - series.cam_rate_dps[keep]
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))


class Canvas:
    def __init__(self) -> None:
        self.image = Image.new("RGB", (WIDTH * SS, HEIGHT * SS), BG)
        self.draw = ImageDraw.Draw(self.image)

    def text(
        self,
        xy: tuple[float, float],
        content: str,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        fill: tuple[int, int, int] = TEXT,
        anchor: str = "la",
    ) -> None:
        self.draw.text((xy[0] * SS, xy[1] * SS), content, font=font, fill=fill, anchor=anchor)

    def line(
        self,
        points: list[tuple[float, float]],
        fill: tuple[int, int, int],
        width: float = 1.0,
    ) -> None:
        self.draw.line([(x * SS, y * SS) for x, y in points], fill=fill, width=round(width * SS))

    def rect(
        self,
        box: tuple[float, float, float, float],
        fill: tuple[int, int, int],
        radius: float = 0.0,
    ) -> None:
        scaled = (box[0] * SS, box[1] * SS, box[2] * SS, box[3] * SS)
        if radius:
            self.draw.rounded_rectangle(scaled, radius=radius * SS, fill=fill)
        else:
            self.draw.rectangle(scaled, fill=fill)

    def finish(self) -> Image.Image:
        return self.image.resize((WIDTH, HEIGHT), Image.LANCZOS)


def _mix(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return (
        round(a[0] + (b[0] - a[0]) * t),
        round(a[1] + (b[1] - a[1]) * t),
        round(a[2] + (b[2] - a[2]) * t),
    )


def _smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


PLOT_X0, PLOT_X1 = 70.0, 650.0
PLOT_TOPS = (74.0, 224.0, 374.0)
PLOT_H = 130.0


def _axis_limits(series: Series) -> tuple[float, float]:
    peak = float(np.max(np.abs(series.cam_rate_dps)))
    limit = math.ceil(peak / 20.0) * 20.0
    return -limit, limit


def render_frame(
    series: Series,
    s: float,
    phase: str,
    compare: float,
    rmse_history: list[float],
) -> Image.Image:
    params = interpolate(series, s)
    canvas = Canvas()
    title, label, small = _font(21, True), _font(13), _font(11)
    canvas.text((24, 10), "Camera-IMU: the curves lock together", title)
    canvas.text(
        (24, 38),
        "Hilti 2022 exp21 (real data) - camera rate from tracked features vs gyro in camera frame",
        small,
        MUTED,
    )
    lo, hi = _axis_limits(series)
    x_scale = (PLOT_X1 - PLOT_X0) / (VIEW_T1_S - VIEW_T0_S)

    def px(t: float) -> float:
        return PLOT_X0 + (t - VIEW_T0_S) * x_scale

    times, rotated = gyro_curve(series, params)
    in_view = (times >= VIEW_T0_S - 0.2) & (times <= VIEW_T1_S + 0.2)
    for axis_index, top in enumerate(PLOT_TOPS):
        canvas.rect((PLOT_X0 - 6, top, PLOT_X1 + 6, top + PLOT_H), PANEL, radius=6)

        def py(v: float, top: float = top) -> float:
            return top + PLOT_H * (hi - v) / (hi - lo)

        for value in (lo, lo / 2, 0.0, hi / 2, hi):
            canvas.line(
                [(PLOT_X0, py(value)), (PLOT_X1, py(value))],
                GRID if value else MUTED,
                0.8 if value else 1.0,
            )
            canvas.text((PLOT_X0 - 10, py(value)), f"{value:.0f}", small, MUTED, "rm")
        for second in range(int(VIEW_T0_S), int(VIEW_T1_S) + 1):
            canvas.line([(px(second), top + PLOT_H - 4), (px(second), top + PLOT_H)], MUTED, 1.0)
        canvas.text((PLOT_X0 + 4, top + 4), "xyz"[axis_index] + " rate (deg/s)", small, MUTED)
        for k in range(len(series.cam_start_s)):
            a, b = series.cam_start_s[k], series.cam_end_s[k]
            if b < VIEW_T0_S or a > VIEW_T1_S:
                continue
            y = py(float(np.clip(series.cam_rate_dps[k, axis_index], lo, hi)))
            canvas.line([(px(a), y), (px(b), y)], CAMERA, 3.0)
        curve = [
            (px(float(t)), py(float(np.clip(v, lo, hi))))
            for t, v in zip(times[in_view], rotated[in_view, axis_index], strict=True)
        ]
        canvas.line(curve, GYRO, 1.6)
    canvas.text((PLOT_X1, PLOT_TOPS[2] + PLOT_H + 5), "time (s), ticks at 1 s", small, MUTED, "ra")
    # legend
    canvas.line([(PLOT_X0, 62), (PLOT_X0 + 24, 62)], GYRO, 2.0)
    canvas.text((PLOT_X0 + 30, 62), "gyro, rotated by R, shifted by dt", small, TEXT, "lm")
    canvas.line([(PLOT_X0 + 250, 62), (PLOT_X0 + 274, 62)], CAMERA, 3.0)
    canvas.text((PLOT_X0 + 280, 62), "camera rate (tracked features)", small, TEXT, "lm")

    # status panel
    x0, x1 = 682.0, 940.0
    canvas.rect((x0, 74.0, x1, 504.0), PANEL, radius=8)
    status = {
        "hold": ("WRONG HYPOTHESIS", RED),
        "solve": ("CALIBRATING", _mix(RED, AMBER, _smoothstep(s * 2))),
        "lock": ("CALIBREX ESTIMATE", GREEN),
        "compare": ("CALIBREX ESTIMATE", GREEN),
    }[phase]
    if phase == "solve" and s > 0.5:
        status = ("CALIBRATING", _mix(AMBER, GREEN, _smoothstep((s - 0.5) * 2)))
    canvas.rect((x0 + 14, 88.0, x1 - 14, 114.0), _mix(PANEL, status[1], 0.25), radius=5)
    canvas.text(((x0 + x1) / 2, 101.0), status[0], _font(13, True), status[1], "mm")

    rotation_error = rotation_angle_deg(series.r_est.T @ params.rotation)
    rmse = interval_rmse_dps(series, params)
    rows = [
        ("rotation offset to estimate", f"{rotation_error:5.1f} deg"),
        ("time offset dt", f"{params.time_offset_s * 1e3:+6.1f} ms"),
        ("gyro bias |b|", f"{float(np.linalg.norm(params.bias_dps)):5.2f} deg/s"),
    ]
    y = 128.0
    for name, value in rows:
        canvas.text((x0 + 16, y), name, small, MUTED)
        canvas.text((x1 - 16, y + 17), value, _font(17, True), TEXT, "ra")
        y += 50
    canvas.text((x0 + 16, y), "rate RMSE (camera vs gyro)", small, MUTED)
    canvas.text((x1 - 16, y + 17), f"{rmse:5.2f} deg/s", _font(19, True), status[1], "ra")

    # RMSE sparkline
    sy0, sy1 = 335.0, 395.0
    canvas.rect((x0 + 16, sy0, x1 - 16, sy1), BG, radius=4)
    if len(rmse_history) > 1:
        top_value = max(rmse_history[0], 1e-6)
        pts = [
            (
                x0 + 20 + (x1 - x0 - 40) * i / (N_HOLD + N_SOLVE + N_LOCK),
                sy1 - 5 - (sy1 - sy0 - 10) * min(v / top_value, 1.0),
            )
            for i, v in enumerate(rmse_history)
        ]
        canvas.line(pts, status[1], 1.8)
    canvas.text((x0 + 20, sy1 + 3), "RMSE history", _font(10), MUTED)

    if compare > 0.0:
        fade = _smoothstep(compare)
        c = _mix(PANEL, (255, 255, 255), 0.0)
        canvas.rect((x0 + 8, 408.0, x1 - 8, 498.0), c, radius=6)
        rot_diff = rotation_angle_deg(series.r_est.T @ series.r_kalibr)
        dt_diff_ms = (series.dt_est_s - series.dt_kalibr_s) * 1e3
        col = _mix(PANEL, TEXT, fade)
        canvas.text((x0 + 16, 414), "vs Kalibr (target-based)", _font(12, True), col)
        canvas.text((x0 + 16, 434), f"rotation diff   {rot_diff:4.2f} deg", label, col)
        canvas.text((x0 + 16, 454), f"dt calibrex   {series.dt_est_s * 1e3:+5.2f} ms", label, col)
        canvas.text(
            (x0 + 16, 472), f"dt Kalibr     {series.dt_kalibr_s * 1e3:+5.2f} ms", label, col
        )
        canvas.text((x1 - 16, 454), f"({dt_diff_ms:+.2f})", label, col, "ra")
    else:
        canvas.text(
            (x0 + 16, 420),
            "gyro bias starts at 0;",
            small,
            MUTED,
        )
        canvas.text((x0 + 16, 436), "R slerps, dt and b lerp", small, MUTED)
        canvas.text((x0 + 16, 452), "to calibrex's estimate", small, MUTED)
    canvas.text(
        (24, 522),
        "calibrex camera-imu rotation, development recording exp21 only. "
        "Data: Hilti SLAM Challenge 2022 terms.",
        _font(10),
        MUTED,
    )
    return canvas.finish()


def build_frames(series: Series) -> list[Image.Image]:
    plan: list[tuple[float, str, float]] = []
    plan += [(0.0, "hold", 0.0)] * N_HOLD
    plan += [(_smoothstep((i + 1) / N_SOLVE), "solve", 0.0) for i in range(N_SOLVE)]
    plan += [(1.0, "lock", 0.0)] * N_LOCK
    plan += [(1.0, "compare", min(1.0, (i + 1) / 4)) for i in range(N_COMPARE)]
    history: list[float] = []
    frames: list[Image.Image] = []
    for s, phase, compare in plan:
        history.append(interval_rmse_dps(series, interpolate(series, s)))
        frames.append(render_frame(series, s, phase, compare, list(history)))
    first = render_frame(series, 0.0, "hold", 0.0, [history[0]])
    last = frames[-1]
    for i in range(N_FADE):  # crossfade back to the first frame for a seamless loop
        frames.append(Image.blend(last, first, (i + 1) / (N_FADE + 1)))
    return frames


def save_gif(frames: list[Image.Image], path: Path, colors: int = 64) -> None:
    sample = frames[:: max(1, len(frames) // 12)]
    sheet = Image.new("RGB", (WIDTH, HEIGHT * len(sample)))
    for i, frame in enumerate(sample):
        sheet.paste(frame, (0, HEIGHT * i))
    palette = sheet.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.NONE)
    quantized = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(
        path,
        save_all=True,
        append_images=quantized[1:],
        duration=FRAME_MS,
        loop=0,
        optimize=True,
        disposal=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=Path, default=SERIES_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    series = load_series(args.series)
    frames = build_frames(series)
    save_gif(frames, args.output)
    if args.preview_dir:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
        last = N_HOLD + N_SOLVE + N_LOCK + N_COMPARE - 2
        for name, index in (("start", 2), ("mid", N_HOLD + N_SOLVE // 2), ("lock", last)):
            frames[index].save(args.preview_dir / f"camera-imu-lock-{name}.png")
    provenance = args.series.with_name("series.provenance.json")
    if provenance.exists() and args.output == OUTPUT_PATH:
        manifest = json.loads(provenance.read_text())
        manifest["gif"] = {
            "path": str(OUTPUT_PATH),
            "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            "size_bytes": args.output.stat().st_size,
            "frames": len(frames),
            "width": WIDTH,
            "height": HEIGHT,
            "frame_ms": FRAME_MS,
        }
        provenance.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    first = interval_rmse_dps(series, interpolate(series, 0.0))
    last = interval_rmse_dps(series, interpolate(series, 1.0))
    size = args.output.stat().st_size
    print(f"frames={len(frames)} rmse {first:.2f} -> {last:.2f} deg/s size={size}")


if __name__ == "__main__":
    main()
