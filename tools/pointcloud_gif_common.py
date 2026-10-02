"""Shared numpy + Pillow helpers for the point-cloud README GIFs (no matplotlib, no GPU).

Points are projected with a hand-written perspective camera and splatted with a
painter's algorithm (far to near), so occlusion is correct and the output is
deterministic.  Text uses DejaVu Sans, as the other README GIF tools do.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

FloatArray: TypeAlias = NDArray[np.float64]

WIDTH = 960
HEIGHT = 540
FPS = 10
BG = (13, 17, 23)
PANEL = (22, 27, 34)
RULE = (48, 54, 61)
TEXT = (230, 237, 243)
MUTED = (139, 148, 158)
CYAN = (70, 215, 255)
ORANGE = (255, 150, 40)
GOOD = (86, 211, 100)
BAD = (255, 123, 114)

_FONT_DIRS = (
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/usr/share/fonts/dejavu"),
    Path("/Library/Fonts"),
    Path("C:/Windows/Fonts"),
)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """DejaVu Sans at ``size`` px, or Pillow's default font when it is not installed."""

    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for directory in _FONT_DIRS:
        candidate = directory / name
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


@dataclass(frozen=True)
class Camera:
    """Look-at perspective camera; ``focal_px`` is in pixels of the viewport."""

    eye: FloatArray
    target: FloatArray
    focal_px: float
    width: int
    height: int
    up_hint: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def basis(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        forward = self.target - self.eye
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, np.asarray(self.up_hint, dtype=np.float64))
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        return forward, right, up

    def project(
        self, points: FloatArray
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], FloatArray]:
        """Pixel columns, rows and depths of ``points``; depth <= 0.3 m is dropped by callers."""

        forward, right, up = self.basis()
        rel = np.asarray(points, dtype=np.float64) - self.eye
        depth = rel @ forward
        safe = np.maximum(depth, 1.0e-6)
        u = self.focal_px * (rel @ right) / safe + self.width / 2.0
        v = self.height / 2.0 - self.focal_px * (rel @ up) / safe
        return np.rint(u).astype(np.int64), np.rint(v).astype(np.int64), depth


def orbit_camera(
    target: Sequence[float],
    azimuth_deg: float,
    elevation_deg: float,
    distance_m: float,
    focal_px: float,
    width: int,
    height: int,
) -> Camera:
    """Camera on a sphere around ``target`` (z up); azimuth 0 looks along -x."""

    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    centre = np.asarray(target, dtype=np.float64)
    offset = distance_m * np.array(
        [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
    )
    return Camera(centre + offset, centre, focal_px, width, height)


def splat(
    canvas: NDArray[np.uint8],
    camera: Camera,
    clouds: Sequence[tuple[FloatArray, Any, int]],
    *,
    near_m: float = 0.3,
    fade: tuple[float, float] = (1.0, 0.45),
    origin: tuple[int, int] = (0, 0),
) -> None:
    """Draw ``(points, rgb or (N, 3) colours, size_px)`` clouds with correct occlusion.

    The viewport is ``camera.width x camera.height`` placed at ``origin`` (x, y)
    of the canvas.  Colours fade from ``fade[0]`` (near) to ``fade[1]`` (far).
    """

    parts_u: list[NDArray[np.int64]] = []
    parts_v: list[NDArray[np.int64]] = []
    parts_z: list[FloatArray] = []
    parts_c: list[NDArray[np.float64]] = []
    parts_s: list[NDArray[np.int64]] = []
    for points, rgb, size in clouds:
        if len(points) == 0:
            continue
        u, v, z = camera.project(points)
        keep = (z > near_m) & (u >= 0) & (u < camera.width - 2) & (v >= 0) & (v < camera.height - 2)
        parts_u.append(u[keep])
        parts_v.append(v[keep])
        parts_z.append(z[keep])
        rgb_array = np.asarray(rgb, dtype=np.float64)
        if rgb_array.ndim == 2:  # one colour per point
            parts_c.append(rgb_array[keep])
        else:
            parts_c.append(np.tile(rgb_array, (int(keep.sum()), 1)))
        parts_s.append(np.full(int(keep.sum()), size, dtype=np.int64))
    if not parts_z:
        return
    u = np.concatenate(parts_u)
    v = np.concatenate(parts_v)
    z = np.concatenate(parts_z)
    colour = np.concatenate(parts_c)
    size = np.concatenate(parts_s)
    z_near, z_far = np.percentile(z, [2.0, 98.0])
    t = np.clip((z - z_near) / max(z_far - z_near, 1.0e-6), 0.0, 1.0)
    shade = fade[0] + (fade[1] - fade[0]) * t
    colour = np.clip(colour * shade[:, None], 0, 255).astype(np.uint8)
    order = np.argsort(-z, kind="stable")
    u, v, size, colour = u[order], v[order], size[order], colour[order]
    ox, oy = origin
    # Fancy assignment keeps the last duplicate index, so the nearest point wins.
    for dy in range(int(size.max())):
        for dx in range(int(size.max())):
            sel = (size > dy) & (size > dx)
            canvas[oy + v[sel] + dy, ox + u[sel] + dx] = colour[sel]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rel_path(path: Path, root: Path) -> str:
    """``path`` relative to ``root`` when inside it, else as given."""

    try:
        return str(Path(path).resolve().relative_to(root))
    except ValueError:
        return str(path)


def git_head(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    size: int,
    fill: tuple[int, int, int] = TEXT,
    bold: bool = False,
    anchor: str = "la",
) -> None:
    draw.text(xy, value, font=font(size, bold), fill=fill, anchor=anchor)


def smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def save_gif(frames: Sequence[Image.Image], path: str | Path, fps: int = FPS) -> None:
    """Write a looping GIF with one shared 256-colour palette (stable colours, no flicker)."""

    step = max(1, len(frames) // 12)
    sample = frames[::step][:12]
    mosaic = Image.new("RGB", (frames[0].width, frames[0].height * len(sample)))
    for index, frame in enumerate(sample):
        mosaic.paste(frame, (0, index * frames[0].height))
    palette_image = mosaic.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=0)
    quantised = [
        frame.quantize(palette=palette_image, dither=Image.Dither.NONE) for frame in frames
    ]
    quantised[0].save(
        path,
        save_all=True,
        append_images=quantised[1:],
        duration=round(1000 / fps),
        loop=0,
        optimize=True,
        disposal=1,
    )


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
