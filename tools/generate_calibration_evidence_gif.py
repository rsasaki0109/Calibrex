#!/usr/bin/env python3
"""Generate the README calibration evidence GIF.

The asset is intentionally illustrative. It shows a candidate LiDAR extrinsic
moving toward a reference map while evidence panels update; it is not a
benchmark result or an accuracy claim.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import tempfile
from itertools import pairwise
from pathlib import Path

WIDTH = 960
HEIGHT = 540
FPS = 12
FRAME_COUNT = 56
SCALE = 36.0
ORIGIN = (318, 300)

Color = tuple[int, int, int]

BG = (14, 19, 32)
PANEL = (24, 31, 47)
PANEL_ALT = (17, 24, 39)
GRID = (46, 57, 78)
MUTED = (107, 124, 150)
TEXT = (229, 231, 235)
REFERENCE = (52, 211, 153)
CANDIDATE = (251, 113, 133)
OPTIMIZED = (34, 211, 238)
WARNING = (245, 158, 11)
GOOD = (34, 197, 94)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/calibration-evidence-demo.gif"),
    )
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required to generate the GIF")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="calibrex_gif_") as tmp_name:
        tmp = Path(tmp_name)
        for index in range(FRAME_COUNT):
            progress = smoothstep(index / (FRAME_COUNT - 1))
            image = bytearray(bytes(BG) * (WIDTH * HEIGHT))
            draw_frame(image, progress)
            write_ppm(tmp / f"frame_{index:03d}.ppm", image)
        palette = tmp / "palette.png"
        text_filter = build_text_filter()
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-framerate",
                str(FPS),
                "-thread_queue_size",
                "64",
                "-i",
                str(tmp / "frame_%03d.ppm"),
                "-vf",
                f"{text_filter},palettegen=max_colors=128",
                "-frames:v",
                "1",
                "-update",
                "true",
                str(palette),
            ],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-framerate",
                str(FPS),
                "-thread_queue_size",
                "64",
                "-i",
                str(tmp / "frame_%03d.ppm"),
                "-i",
                str(palette),
                "-lavfi",
                f"[0:v]{text_filter}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
                "-loop",
                "0",
                str(args.output),
            ],
            check=True,
        )
    return 0


def draw_frame(image: bytearray, progress: float) -> None:
    fill_rect(image, 0, 0, WIDTH, HEIGHT, BG)
    draw_grid(image)
    draw_rig_scene(image, progress)
    draw_evidence_panel(image, progress)
    draw_timeline(image, progress)


def draw_grid(image: bytearray) -> None:
    for x in range(40, 620, 36):
        line(image, x, 82, x, 428, GRID, alpha=0.45)
    for y in range(88, 430, 36):
        line(image, 40, y, 620, y, GRID, alpha=0.45)
    rect(image, 32, 72, 604, 370, (63, 78, 105), alpha=0.8)


def draw_rig_scene(image: bytearray, progress: float) -> None:
    reference_points = map_points()
    initial_theta = math.radians(13.0)
    theta = initial_theta * (1.0 - progress)
    tx = 1.25 * (1.0 - progress)
    ty = -0.70 * (1.0 - progress)
    point_color = mix(CANDIDATE, OPTIMIZED, progress)

    for x, y in reference_points:
        px, py = project(x, y)
        circle(image, px, py, 2, REFERENCE, alpha=0.60)

    for x, y in reference_points:
        xr, yr = transform2(x, y, tx, ty, theta)
        px, py = project(xr, yr)
        circle(image, px, py, 3, point_color, alpha=0.78)

    draw_vehicle(image, progress)
    draw_axes(image, 0.0, 0.0, 0.0, REFERENCE, alpha=0.80)
    draw_axes(image, tx, ty, theta, point_color, alpha=0.95)

    arc_x, arc_y = project(0.0, 0.0)
    arc_radius = int(72 * (1.0 - 0.35 * progress))
    for step in range(18):
        a0 = initial_theta * (step / 18.0) * (1.0 - progress)
        a1 = initial_theta * ((step + 1) / 18.0) * (1.0 - progress)
        x0 = int(arc_x + math.cos(a0) * arc_radius)
        y0 = int(arc_y - math.sin(a0) * arc_radius)
        x1 = int(arc_x + math.cos(a1) * arc_radius)
        y1 = int(arc_y - math.sin(a1) * arc_radius)
        line(image, x0, y0, x1, y1, WARNING, alpha=0.5)


def draw_vehicle(image: bytearray, progress: float) -> None:
    cx, cy = project(0.0, 0.0)
    fill_rect(image, cx - 54, cy - 26, 108, 52, (31, 41, 55), alpha=0.95)
    rect(image, cx - 54, cy - 26, 108, 52, (148, 163, 184), alpha=0.9)
    fill_rect(image, cx + 18, cy - 15, 26, 30, (55, 65, 81), alpha=0.95)
    circle(image, cx, cy, 7, mix(CANDIDATE, OPTIMIZED, progress), alpha=1.0)


def draw_axes(
    image: bytearray,
    tx: float,
    ty: float,
    theta: float,
    color: Color,
    *,
    alpha: float,
) -> None:
    ox, oy = project(tx, ty)
    x1, y1 = project(tx + math.cos(theta) * 1.5, ty + math.sin(theta) * 1.5)
    x2, y2 = project(tx - math.sin(theta) * 1.0, ty + math.cos(theta) * 1.0)
    thick_line(image, ox, oy, x1, y1, color, thickness=3, alpha=alpha)
    thick_line(image, ox, oy, x2, y2, (96, 165, 250), thickness=3, alpha=alpha)
    circle(image, ox, oy, 5, color, alpha=alpha)


def draw_evidence_panel(image: bytearray, progress: float) -> None:
    fill_rect(image, 650, 76, 276, 356, PANEL, alpha=0.98)
    rect(image, 650, 76, 276, 356, (71, 85, 105), alpha=0.85)

    # Holdout residual curve.
    chart_x, chart_y, chart_w, chart_h = 674, 142, 222, 96
    fill_rect(image, chart_x, chart_y, chart_w, chart_h, PANEL_ALT, alpha=1.0)
    rect(image, chart_x, chart_y, chart_w, chart_h, (51, 65, 85), alpha=0.9)
    values = [0.084 - 0.063 * smoothstep(i / 42.0) for i in range(43)]
    visible = max(2, int(2 + progress * (len(values) - 2)))
    draw_curve(image, values[:visible], chart_x, chart_y, chart_w, chart_h, OPTIMIZED)
    current = values[visible - 1]
    dot_x = chart_x + int((visible - 1) / (len(values) - 1) * chart_w)
    dot_y = chart_y + int((0.09 - current) / 0.075 * chart_h)
    circle(image, dot_x, dot_y, 5, OPTIMIZED, alpha=1.0)

    # Train and holdout bars.
    bar_x, bar_y = 674, 270
    train = 0.70 - 0.46 * progress
    holdout = 0.88 - 0.62 * progress
    metric_bar(image, bar_x, bar_y, train, OPTIMIZED)
    metric_bar(image, bar_x, bar_y + 32, holdout, GOOD if progress > 0.72 else WARNING)

    # Weak DoF cells.
    cells = ["x", "y", "z", "r", "p", "y"]
    for index, _label in enumerate(cells):
        x = 674 + index * 35
        cell_color = GOOD if index in {2, 3, 4} or progress > 0.74 else WARNING
        fill_rect(image, x, 360, 24, 18, cell_color, alpha=0.86)
        rect(image, x, 360, 24, 18, (229, 231, 235), alpha=0.30)


def draw_timeline(image: bytearray, progress: float) -> None:
    y = 474
    x0, w = 92, 776
    fill_rect(image, x0, y, w, 10, (31, 41, 55), alpha=1.0)
    fill_rect(image, x0, y, int(w * progress), 10, OPTIMIZED, alpha=0.95)
    for step in [0.0, 0.33, 0.66, 1.0]:
        x = x0 + int(w * step)
        circle(image, x, y + 5, 9, OPTIMIZED if progress >= step else MUTED, alpha=1.0)


def map_points() -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for i in range(-10, 11):
        x = i * 0.45
        points.append((x, -2.7))
        points.append((x, 2.7))
        if i % 2 == 0:
            points.append((x, 0.0))
    for j in range(-7, 8):
        y = j * 0.38
        points.append((4.1, y))
    for i in range(28):
        x = -4.6 + (i % 7) * 1.25
        y = -1.8 + (i // 7) * 1.05 + 0.12 * math.sin(i)
        points.append((x, y))
    return points


def draw_curve(
    image: bytearray,
    values: list[float],
    x: int,
    y: int,
    width: int,
    height: int,
    color: Color,
) -> None:
    if len(values) < 2:
        return
    points: list[tuple[int, int]] = []
    for index, value in enumerate(values):
        px = x + int(index / 42.0 * width)
        py = y + int((0.09 - value) / 0.075 * height)
        points.append((px, py))
    for left, right in pairwise(points):
        thick_line(image, left[0], left[1], right[0], right[1], color, thickness=3, alpha=0.95)


def metric_bar(image: bytearray, x: int, y: int, value: float, color: Color) -> None:
    fill_rect(image, x, y, 220, 12, (31, 41, 55), alpha=1.0)
    fill_rect(image, x, y, int(220 * value), 12, color, alpha=0.95)


def project(x: float, y: float) -> tuple[int, int]:
    return int(ORIGIN[0] + x * SCALE), int(ORIGIN[1] - y * SCALE)


def transform2(x: float, y: float, tx: float, ty: float, theta: float) -> tuple[float, float]:
    c = math.cos(theta)
    s = math.sin(theta)
    return c * x - s * y + tx, s * x + c * y + ty


def smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def mix(left: Color, right: Color, ratio: float) -> Color:
    return tuple(
        int(left[index] + (right[index] - left[index]) * ratio)
        for index in range(3)
    )


def fill_rect(
    image: bytearray,
    x: int,
    y: int,
    width: int,
    height: int,
    color: Color,
    *,
    alpha: float = 1.0,
) -> None:
    for yy in range(max(0, y), min(HEIGHT, y + height)):
        for xx in range(max(0, x), min(WIDTH, x + width)):
            pixel(image, xx, yy, color, alpha)


def rect(
    image: bytearray,
    x: int,
    y: int,
    width: int,
    height: int,
    color: Color,
    *,
    alpha: float = 1.0,
) -> None:
    line(image, x, y, x + width, y, color, alpha=alpha)
    line(image, x + width, y, x + width, y + height, color, alpha=alpha)
    line(image, x + width, y + height, x, y + height, color, alpha=alpha)
    line(image, x, y + height, x, y, color, alpha=alpha)


def thick_line(
    image: bytearray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: Color,
    *,
    thickness: int,
    alpha: float,
) -> None:
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for index in range(steps + 1):
        x = int(x0 + (x1 - x0) * index / steps)
        y = int(y0 + (y1 - y0) * index / steps)
        circle(image, x, y, thickness, color, alpha=alpha)


def line(
    image: bytearray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: Color,
    *,
    alpha: float = 1.0,
) -> None:
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for index in range(steps + 1):
        x = int(x0 + (x1 - x0) * index / steps)
        y = int(y0 + (y1 - y0) * index / steps)
        pixel(image, x, y, color, alpha)


def circle(
    image: bytearray,
    cx: int,
    cy: int,
    radius: int,
    color: Color,
    *,
    alpha: float = 1.0,
) -> None:
    r2 = radius * radius
    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r2:
                pixel(image, x, y, color, alpha)


def pixel(image: bytearray, x: int, y: int, color: Color, alpha: float) -> None:
    if x < 0 or y < 0 or x >= WIDTH or y >= HEIGHT:
        return
    index = (y * WIDTH + x) * 3
    if alpha >= 1.0:
        image[index] = color[0]
        image[index + 1] = color[1]
        image[index + 2] = color[2]
        return
    inv = 1.0 - alpha
    image[index] = int(image[index] * inv + color[0] * alpha)
    image[index + 1] = int(image[index + 1] * inv + color[1] * alpha)
    image[index + 2] = int(image[index + 2] * inv + color[2] * alpha)


def write_ppm(path: Path, image: bytearray) -> None:
    with path.open("wb") as handle:
        handle.write(f"P6\n{WIDTH} {HEIGHT}\n255\n".encode("ascii"))
        handle.write(image)


def build_text_filter() -> str:
    font_file = subprocess.check_output(
        ["fc-match", "-f", "%{file}", "Noto Sans"],
        text=True,
    ).strip()
    labels = [
        ("Calibration evidence demo", 48, 26, 28, "E5E7EB"),
        ("candidate -> refine -> evaluate", 52, 438, 20, "CBD5E1"),
        ("reference map", 68, 94, 18, "34D399"),
        ("candidate / optimized scan", 68, 118, 18, "22D3EE"),
        ("Evidence report", 674, 96, 24, "E5E7EB"),
        ("Holdout residual", 674, 124, 17, "CBD5E1"),
        ("Train / holdout", 674, 246, 17, "CBD5E1"),
        ("Weak DoF proxy", 674, 334, 17, "CBD5E1"),
        ("calibrate", 78, 494, 17, "CBD5E1"),
        ("evaluate", 392, 494, 17, "CBD5E1"),
        ("visualize", 748, 494, 17, "CBD5E1"),
    ]
    return ",".join(drawtext(font_file, *label) for label in labels)


def drawtext(
    font_file: str,
    text: str,
    x: int,
    y: int,
    size: int,
    color: str,
) -> str:
    escaped_text = text.replace("\\", "\\\\").replace(":", "\\:")
    escaped_font = font_file.replace("\\", "\\\\").replace(":", "\\:")
    return (
        "drawtext="
        f"fontfile='{escaped_font}':"
        f"text='{escaped_text}':"
        f"x={x}:y={y}:fontsize={size}:fontcolor=0x{color}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
