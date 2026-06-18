#!/usr/bin/env python3
"""Generate the README calibration evidence GIF from public RGB-D data.

The visual uses real frames from the public TUM RGB-D fr1/xyz sequence. The
animated alignment offset and residual curve are an evidence-viewer proxy, not a
benchmark accuracy claim.
"""

from __future__ import annotations

import argparse
import bisect
import shutil
import struct
import subprocess
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path

WIDTH = 960
HEIGHT = 540
FPS = 12
FRAME_COUNT = 36
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

RGB_PANEL = (32, 92, 568, 426)
RIGHT_PANEL = (628, 92, 300, 426)
DEPTH_PANEL = (648, 126, 260, 195)
CHART = (652, 394, 240, 48)

Color = tuple[int, int, int]

BG = (12, 17, 29)
PANEL = (20, 29, 45)
PANEL_ALT = (14, 21, 35)
GRID = (54, 68, 91)
TEXT_DIM = (148, 163, 184)
REFERENCE = (52, 211, 153)
CANDIDATE = (251, 113, 133)
OPTIMIZED = (34, 211, 238)
WARNING = (245, 158, 11)
GOOD = (34, 197, 94)


@dataclass(frozen=True)
class TumFramePair:
    rgb_timestamp: float
    rgb_path: Path
    depth_timestamp: float
    depth_path: Path


@dataclass(frozen=True)
class RgbImage:
    width: int
    height: int
    data: bytearray


@dataclass(frozen=True)
class DepthImage:
    width: int
    height: int
    values: list[int]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/public/rgbd_dataset_freiburg1_xyz"),
        help="Path to the extracted TUM RGB-D fr1/xyz sequence.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/calibration-evidence-demo.gif"),
    )
    parser.add_argument("--frames", type=int, default=FRAME_COUNT)
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required to generate the GIF")
    ensure_tum_dataset(args.dataset_root)

    pairs = select_pairs(pair_tum_frames(args.dataset_root), args.frames)
    if len(pairs) < 2:
        raise SystemExit("Need at least two RGB-D pairs to generate the GIF")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="calibrex_public_gif_") as tmp_name:
        tmp = Path(tmp_name)
        residual_history: list[float] = []
        for index, pair in enumerate(pairs):
            progress = smoothstep(index / (len(pairs) - 1))
            rgb = read_rgb_png(args.dataset_root / pair.rgb_path)
            depth = read_depth_png(args.dataset_root / pair.depth_path)
            residual_history.append(residual_proxy(progress))
            image = bytearray(bytes(BG) * (WIDTH * HEIGHT))
            draw_frame(
                image=image,
                rgb=rgb,
                depth=depth,
                pair=pair,
                progress=progress,
                residual_history=residual_history,
            )
            write_ppm(tmp / f"frame_{index:03d}.ppm", image)
        encode_gif(tmp, args.output)
    return 0


def ensure_tum_dataset(root: Path) -> None:
    if (root / "rgb.txt").exists() and (root / "depth.txt").exists():
        return
    raise SystemExit(
        "TUM RGB-D fr1/xyz data is required.\n"
        "Download and extract it to data/public/rgbd_dataset_freiburg1_xyz:\n"
        "  https://cvg.cit.tum.de/rgbd/dataset/freiburg1/"
        "rgbd_dataset_freiburg1_xyz.tgz"
    )


def read_tum_listing(path: Path) -> list[tuple[float, Path]]:
    entries: list[tuple[float, Path]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        timestamp, relative_path = line.split()[:2]
        entries.append((float(timestamp), Path(relative_path)))
    return entries


def pair_tum_frames(root: Path) -> list[TumFramePair]:
    rgb_entries = read_tum_listing(root / "rgb.txt")
    depth_entries = read_tum_listing(root / "depth.txt")
    depth_times = [timestamp for timestamp, _path in depth_entries]
    pairs: list[TumFramePair] = []
    for rgb_timestamp, rgb_path in rgb_entries:
        insert_at = bisect.bisect_left(depth_times, rgb_timestamp)
        candidates = [
            index
            for index in (insert_at - 1, insert_at)
            if 0 <= index < len(depth_entries)
        ]
        if not candidates:
            continue
        best_index = min(
            candidates,
            key=lambda index: abs(depth_entries[index][0] - rgb_timestamp),
        )
        depth_timestamp, depth_path = depth_entries[best_index]
        pairs.append(
            TumFramePair(
                rgb_timestamp=rgb_timestamp,
                rgb_path=rgb_path,
                depth_timestamp=depth_timestamp,
                depth_path=depth_path,
            )
        )
    return pairs


def select_pairs(pairs: list[TumFramePair], frame_count: int) -> list[TumFramePair]:
    if len(pairs) <= frame_count:
        return pairs
    start = len(pairs) // 8
    stop = len(pairs) * 7 // 8
    span = max(1, stop - start)
    selected: list[TumFramePair] = []
    last_index = -1
    for index in range(frame_count):
        candidate = start + round(index * span / (frame_count - 1))
        candidate = min(max(candidate, last_index + 1), len(pairs) - 1)
        selected.append(pairs[candidate])
        last_index = candidate
    return selected


def read_rgb_png(path: Path) -> RgbImage:
    width, height, bit_depth, color_type, rows = decode_png(path)
    if bit_depth != 8 or color_type != 2:
        raise ValueError(f"{path} must be 8-bit RGB PNG")
    data = unfilter_png(rows, width=width, height=height, bytes_per_pixel=3)
    return RgbImage(width=width, height=height, data=data)


def read_depth_png(path: Path) -> DepthImage:
    width, height, bit_depth, color_type, rows = decode_png(path)
    if bit_depth != 16 or color_type != 0:
        raise ValueError(f"{path} must be 16-bit grayscale PNG")
    data = unfilter_png(rows, width=width, height=height, bytes_per_pixel=2)
    values = [
        (data[index] << 8) | data[index + 1]
        for index in range(0, len(data), 2)
    ]
    return DepthImage(width=width, height=height, values=values)


def decode_png(path: Path) -> tuple[int, int, int, int, bytes]:
    with path.open("rb") as handle:
        if handle.read(len(PNG_SIGNATURE)) != PNG_SIGNATURE:
            raise ValueError(f"{path} is not a PNG file")
        width = height = bit_depth = color_type = interlace = -1
        idat_parts: list[bytes] = []
        while True:
            raw_length = handle.read(4)
            if not raw_length:
                break
            length = struct.unpack(">I", raw_length)[0]
            chunk_type = handle.read(4)
            chunk_data = handle.read(length)
            handle.read(4)
            if chunk_type == b"IHDR":
                (
                    width,
                    height,
                    bit_depth,
                    color_type,
                    _compression,
                    _filter_method,
                    interlace,
                ) = struct.unpack(">IIBBBBB", chunk_data)
            elif chunk_type == b"IDAT":
                idat_parts.append(chunk_data)
            elif chunk_type == b"IEND":
                break
    if interlace != 0:
        raise ValueError(f"{path} uses unsupported interlaced PNG encoding")
    return width, height, bit_depth, color_type, zlib.decompress(b"".join(idat_parts))


def unfilter_png(
    data: bytes,
    *,
    width: int,
    height: int,
    bytes_per_pixel: int,
) -> bytearray:
    row_size = width * bytes_per_pixel
    output = bytearray(row_size * height)
    previous = bytearray(row_size)
    input_offset = 0
    output_offset = 0
    for _row in range(height):
        filter_type = data[input_offset]
        input_offset += 1
        scanline = data[input_offset : input_offset + row_size]
        input_offset += row_size
        reconstructed = bytearray(row_size)
        for index, value in enumerate(scanline):
            left = reconstructed[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
            up = previous[index]
            up_left = previous[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
            reconstructed[index] = (
                value + png_filter_prediction(filter_type, left, up, up_left)
            ) & 0xFF
        output[output_offset : output_offset + row_size] = reconstructed
        previous = reconstructed
        output_offset += row_size
    return output


def png_filter_prediction(filter_type: int, left: int, up: int, up_left: int) -> int:
    if filter_type == 0:
        return 0
    if filter_type == 1:
        return left
    if filter_type == 2:
        return up
    if filter_type == 3:
        return (left + up) // 2
    if filter_type == 4:
        return paeth(left, up, up_left)
    raise ValueError(f"Unsupported PNG filter type: {filter_type}")


def paeth(left: int, up: int, up_left: int) -> int:
    estimate = left + up - up_left
    distance_left = abs(estimate - left)
    distance_up = abs(estimate - up)
    distance_up_left = abs(estimate - up_left)
    if distance_left <= distance_up and distance_left <= distance_up_left:
        return left
    if distance_up <= distance_up_left:
        return up
    return up_left


def draw_frame(
    *,
    image: bytearray,
    rgb: RgbImage,
    depth: DepthImage,
    pair: TumFramePair,
    progress: float,
    residual_history: list[float],
) -> None:
    fill_rect(image, 0, 0, WIDTH, HEIGHT, BG)
    fill_rect(image, 26, 86, 580, 438, PANEL, alpha=1.0)
    fill_rect(
        image,
        RIGHT_PANEL[0],
        RIGHT_PANEL[1],
        RIGHT_PANEL[2],
        RIGHT_PANEL[3],
        PANEL,
        alpha=1.0,
    )
    rect(image, 26, 86, 580, 438, GRID, alpha=0.85)
    rect(image, RIGHT_PANEL[0], RIGHT_PANEL[1], RIGHT_PANEL[2], RIGHT_PANEL[3], GRID, alpha=0.85)

    draw_scaled_rgb(image, rgb, *RGB_PANEL)
    draw_depth_overlay(image, depth, progress)
    fill_rect(
        image,
        RGB_PANEL[0],
        RGB_PANEL[1] + RGB_PANEL[3] - 34,
        RGB_PANEL[2],
        34,
        BG,
        alpha=0.72,
    )

    draw_depth_map(image, depth, *DEPTH_PANEL)
    draw_evidence_panel(image, depth, pair, progress, residual_history)
    draw_timeline(image, progress)


def draw_scaled_rgb(
    image: bytearray,
    rgb: RgbImage,
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    for yy in range(height):
        source_y = min(rgb.height - 1, yy * rgb.height // height)
        for xx in range(width):
            source_x = min(rgb.width - 1, xx * rgb.width // width)
            source_index = (source_y * rgb.width + source_x) * 3
            target_index = ((y + yy) * WIDTH + x + xx) * 3
            image[target_index] = rgb.data[source_index]
            image[target_index + 1] = rgb.data[source_index + 1]
            image[target_index + 2] = rgb.data[source_index + 2]


def draw_depth_overlay(image: bytearray, depth: DepthImage, progress: float) -> None:
    edge_points = detect_depth_edges(depth)
    offset_x = round(26 * (1.0 - progress))
    offset_y = round(-15 * (1.0 - progress))
    candidate_color = mix(CANDIDATE, OPTIMIZED, progress)
    for x, y in edge_points:
        panel_x, panel_y = depth_to_rgb_panel(depth, x, y)
        pixel(image, panel_x, panel_y, REFERENCE, alpha=0.20)
    for x, y in edge_points:
        panel_x, panel_y = depth_to_rgb_panel(depth, x, y)
        circle(image, panel_x + offset_x, panel_y + offset_y, 1, candidate_color, alpha=0.85)


def detect_depth_edges(depth: DepthImage, max_points: int = 850) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int, int]] = []
    stride = 5
    radius = 3
    for y in range(radius, depth.height - radius, stride):
        for x in range(radius, depth.width - radius, stride):
            center = depth.values[y * depth.width + x]
            if center == 0:
                continue
            neighbors = [
                depth.values[y * depth.width + x - radius],
                depth.values[y * depth.width + x + radius],
                depth.values[(y - radius) * depth.width + x],
                depth.values[(y + radius) * depth.width + x],
            ]
            valid_neighbors = [value for value in neighbors if value > 0]
            if len(valid_neighbors) < 2:
                continue
            gradient = max(valid_neighbors) - min(valid_neighbors)
            if gradient > 70:
                candidates.append((gradient, x, y))
    candidates.sort(reverse=True)
    return [(x, y) for _gradient, x, y in candidates[:max_points]]


def depth_to_rgb_panel(depth: DepthImage, x: int, y: int) -> tuple[int, int]:
    panel_x = RGB_PANEL[0] + x * RGB_PANEL[2] // depth.width
    panel_y = RGB_PANEL[1] + y * RGB_PANEL[3] // depth.height
    return panel_x, panel_y


def draw_depth_map(
    image: bytearray,
    depth: DepthImage,
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    low, high = depth_visual_range(depth)
    for yy in range(height):
        source_y = min(depth.height - 1, yy * depth.height // height)
        for xx in range(width):
            source_x = min(depth.width - 1, xx * depth.width // width)
            value = depth.values[source_y * depth.width + source_x]
            color = depth_color(value, low, high)
            pixel(image, x + xx, y + yy, color, alpha=1.0)
    rect(image, x, y, width, height, GRID, alpha=0.9)


def depth_visual_range(depth: DepthImage) -> tuple[int, int]:
    samples = [
        value
        for index, value in enumerate(depth.values)
        if value > 0 and index % 64 == 0
    ]
    if not samples:
        return 1, 2
    samples.sort()
    low = samples[int(len(samples) * 0.05)]
    high = samples[int(len(samples) * 0.95)]
    if high <= low:
        high = low + 1
    return low, high


def depth_color(value: int, low: int, high: int) -> Color:
    if value <= 0:
        return (8, 13, 23)
    ratio = min(1.0, max(0.0, (value - low) / (high - low)))
    if ratio < 0.5:
        return mix((252, 211, 77), (34, 211, 238), ratio * 2.0)
    return mix((34, 211, 238), (99, 102, 241), (ratio - 0.5) * 2.0)


def draw_evidence_panel(
    image: bytearray,
    depth: DepthImage,
    pair: TumFramePair,
    progress: float,
    residual_history: list[float],
) -> None:
    chart_x, chart_y, chart_width, chart_height = CHART
    fill_rect(image, chart_x, chart_y, chart_width, chart_height, PANEL_ALT, alpha=1.0)
    rect(image, chart_x, chart_y, chart_width, chart_height, GRID, alpha=0.95)
    for line_index in range(1, 4):
        y = chart_y + line_index * chart_height // 4
        line(image, chart_x, y, chart_x + chart_width, y, GRID, alpha=0.45)
    draw_curve(image, residual_history, CHART, OPTIMIZED)

    coverage = valid_depth_coverage(depth)
    sync_quality = max(0.0, 1.0 - abs(pair.rgb_timestamp - pair.depth_timestamp) / 0.025)
    residual_score = max(0.0, min(1.0, 1.0 - residual_history[-1] / 0.12))
    metric_bar(image, 652, 456, coverage, GOOD)
    metric_bar(image, 652, 476, sync_quality, OPTIMIZED)
    metric_bar(image, 652, 496, residual_score, GOOD if progress > 0.70 else WARNING)

    draw_dof_cells(image, 652, 350, progress)


def draw_curve(
    image: bytearray,
    values: list[float],
    chart: tuple[int, int, int, int],
    color: Color,
) -> None:
    if len(values) < 2:
        return
    x, y, width, height = chart
    min_value = 0.018
    max_value = 0.116
    points: list[tuple[int, int]] = []
    for index, value in enumerate(values):
        px = x + round(index * width / max(1, len(values) - 1))
        ratio = (value - min_value) / (max_value - min_value)
        py = y + height - round(max(0.0, min(1.0, ratio)) * height)
        points.append((px, py))
    for index in range(1, len(points)):
        left = points[index - 1]
        right = points[index]
        thick_line(image, left[0], left[1], right[0], right[1], color, thickness=2, alpha=0.95)
    circle(image, points[-1][0], points[-1][1], 4, color, alpha=1.0)


def metric_bar(image: bytearray, x: int, y: int, value: float, color: Color) -> None:
    fill_rect(image, x, y, 240, 8, (31, 41, 55), alpha=1.0)
    fill_rect(image, x, y, round(240 * max(0.0, min(1.0, value))), 8, color, alpha=0.95)


def draw_dof_cells(image: bytearray, x: int, y: int, progress: float) -> None:
    for index in range(6):
        cell_x = x + index * 39
        color = GOOD if index in {0, 1, 3, 4} or progress > 0.76 else WARNING
        fill_rect(image, cell_x, y, 28, 18, color, alpha=0.88)
        rect(image, cell_x, y, 28, 18, (229, 231, 235), alpha=0.24)


def draw_timeline(image: bytearray, progress: float) -> None:
    x0, y, width = 82, 516, 786
    fill_rect(image, x0, y, width, 8, (31, 41, 55), alpha=1.0)
    fill_rect(image, x0, y, round(width * progress), 8, OPTIMIZED, alpha=0.95)
    for marker in [0.0, 0.33, 0.66, 1.0]:
        x = x0 + round(width * marker)
        circle(image, x, y + 4, 8, OPTIMIZED if progress >= marker else TEXT_DIM, alpha=1.0)


def valid_depth_coverage(depth: DepthImage) -> float:
    valid = sum(1 for value in depth.values if value > 0)
    return valid / len(depth.values)


def residual_proxy(progress: float) -> float:
    remaining_offset = 1.0 - progress
    return 0.023 + 0.086 * remaining_offset * remaining_offset


def smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def mix(left: Color, right: Color, ratio: float) -> Color:
    return tuple(
        round(left[index] + (right[index] - left[index]) * ratio)
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
        x = round(x0 + (x1 - x0) * index / steps)
        y = round(y0 + (y1 - y0) * index / steps)
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
        x = round(x0 + (x1 - x0) * index / steps)
        y = round(y0 + (y1 - y0) * index / steps)
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
    inverse = 1.0 - alpha
    image[index] = round(image[index] * inverse + color[0] * alpha)
    image[index + 1] = round(image[index + 1] * inverse + color[1] * alpha)
    image[index + 2] = round(image[index + 2] * inverse + color[2] * alpha)


def write_ppm(path: Path, image: bytearray) -> None:
    with path.open("wb") as handle:
        handle.write(f"P6\n{WIDTH} {HEIGHT}\n255\n".encode("ascii"))
        handle.write(image)


def encode_gif(frame_dir: Path, output: Path) -> None:
    palette = frame_dir / "palette.png"
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
            str(frame_dir / "frame_%03d.ppm"),
            "-vf",
            f"{text_filter},palettegen=max_colors=96",
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
            str(frame_dir / "frame_%03d.ppm"),
            "-i",
            str(palette),
            "-lavfi",
            f"[0:v]{text_filter}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
            "-loop",
            "0",
            str(output),
        ],
        check=True,
    )


def build_text_filter() -> str:
    font_file = subprocess.check_output(
        ["fc-match", "-f", "%{file}", "Noto Sans"],
        text=True,
    ).strip()
    labels = [
        ("Public-data calibration evidence", 34, 22, 26, "E5E7EB"),
        ("TUM RGB-D fr1/xyz frames; RGB/depth alignment proxy", 34, 55, 16, "CBD5E1"),
        ("RGB frame with depth-edge overlay", 50, 104, 17, "E5E7EB"),
        ("green: reference depth edges   cyan/pink: candidate offset", 50, 486, 14, "CBD5E1"),
        ("Depth map", 652, 104, 17, "E5E7EB"),
        ("Local DoF visibility proxy", 652, 326, 16, "CBD5E1"),
        ("Evidence sidecar", 652, 374, 17, "E5E7EB"),
        ("residual proxy", 652, 444, 14, "CBD5E1"),
        ("depth coverage", 652, 456, 13, "CBD5E1"),
        ("timestamp pair", 652, 476, 13, "CBD5E1"),
        ("holdout proxy", 652, 496, 13, "CBD5E1"),
        ("public data", 62, 520, 14, "CBD5E1"),
        ("candidate", 325, 520, 14, "CBD5E1"),
        ("evaluate", 588, 520, 14, "CBD5E1"),
        ("report", 842, 520, 14, "CBD5E1"),
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
    escaped_text = (
        text.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace(":", "\\:")
        .replace(",", "\\,")
    )
    escaped_font = font_file.replace("\\", "\\\\").replace(":", "\\:")
    return (
        "drawtext="
        f"fontfile='{escaped_font}':"
        f"text='{escaped_text}':"
        f"x={x}:y={y}:fontsize={size}:fontcolor=0x{color}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
