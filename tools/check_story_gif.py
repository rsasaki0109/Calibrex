"""Render the "catch a bad calibration, then see the fix" README GIF from real check runs.

Three acts, one pair (lidar-vehicle) shown prominently:

1. the deployed tf is wrong (``velo_link`` yaw +3 deg): ``calibrex check`` says FAIL;
2. the tf slides (slerp) to the transform calibrex estimated from the data;
3. that estimate is deployed and ``calibrex check`` is re-run: PASS.

Inputs: the two real check summaries and ``run_manifest.json`` under
``docs/assets/calibrex_check_story/`` (written by ``tools/check_story_run.py``) and, for
the camera overlay only, KITTI drive 0005 (``image_02`` + ``velodyne_points``) and the
date-level calibration files.  The overlay uses the projection of
``tools/check_sweep_gifs.py``: the camera is fixed in the rig, so a Velodyne point is
projected through ``T_cam<-velo(vendor) inv(T_base<-velo(vendor)) T_base<-velo(tf) p``.

Usage (needs numpy and Pillow)::

    python tools/check_story_gif.py render --drive-dir DRIVE
    python tools/check_story_gif.py manifest --drive-dir DRIVE
    python tools/check_story_gif.py preview --drive-dir DRIVE --out-dir /tmp/previews
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_sweep_gifs as sweep

if TYPE_CHECKING:
    from PIL import Image as PILImage

REPO = Path(__file__).resolve().parents[1]
STORY_DIR = REPO / "docs/assets/calibrex_check_story"
ACT1 = STORY_DIR / "act1_deployed_wrong.json"
ACT3 = STORY_DIR / "act3_recheck_estimate.json"
RUN_MANIFEST = STORY_DIR / "run_manifest.json"
PROVENANCE = STORY_DIR / "provenance.json"
GIF = Path("docs/assets/calibrex-check-story.gif")
PROVENANCE_SCHEMA = "calibrex.check_story_gif/v0"
PAIR = "lidar-vehicle"

IMAGE_INDEX = 60
CROP = (500, 54, 1100, 375)  # 600 x 321 px of the 1242 x 375 rectified image, shown 1:1
ZOOM_REGION = (846, 255, 1002, 333)  # 156 x 78 px, shown at 2x
DEPTH_RANGE = (5.0, 25.0)

ACT1_MS = 3600
ACT2_FRAMES = 14
ACT2_FRAME_MS = 100
ACT2_HOLD_MS = 300
ACT3_MS = 4200

BG = sweep.BG
CARD = sweep.CARD
BORDER = sweep.BORDER
TEXT = sweep.TEXT
MUTED = sweep.MUTED
DIM: tuple[int, int, int] = (110, 120, 136)
STATUS_COLOR = sweep.STATUS_COLOR

PANEL = (16, 60, 616, 381)
CARD_BOX = (632, 60, 944, 214)
INSET_BOX = (632, 226, 944, 382)


# --------------------------------------------------------------------------- summaries


def load_json(path: Path) -> dict[str, Any]:
    """Read one JSON file."""

    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def lidar_vehicle(summary: dict[str, Any]) -> dict[str, Any]:
    """The lidar-vehicle pair record of a story summary."""

    record = summary["pairs"][PAIR]
    assert isinstance(record, dict)
    return record


def velo_tf(summary: dict[str, Any]) -> np.ndarray:
    """``T_base<-velo`` the summary's check run was given."""

    return sweep.vendor_velo_tf(summary)


def vendor_tf(act1: dict[str, Any]) -> np.ndarray:
    """The unperturbed (vendor) ``T_base<-velo``: act 1's tf with its injected yaw undone."""

    return sweep.perturbed_velo_tf(velo_tf(act1), -act1["yaw_injected_deg"])


def vendor_to_estimate_yaw_deg(act1: dict[str, Any]) -> float:
    """Yaw distance between the vendor tf and calibrex's estimate (act-1 error minus injected)."""

    axis = lidar_vehicle(act1)["axes"]["yaw"]
    return abs(abs(axis["candidate_error_deg"]) - abs(act1["yaw_injected_deg"]))


def slerp_rotation(r0: np.ndarray, r1: np.ndarray, t: float) -> np.ndarray:
    """Spherical interpolation between two rotation matrices (no SciPy needed)."""

    relative = r0.T @ r1
    angle = float(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))
    if angle < 1e-9:
        return r0.copy()
    axis = np.array(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ]
    )
    axis = axis / np.linalg.norm(axis)
    theta = angle * t
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    step = np.eye(3) + np.sin(theta) * skew + (1 - np.cos(theta)) * (skew @ skew)
    result: np.ndarray = r0 @ step
    return result


def slerp_tf(t0: np.ndarray, t1: np.ndarray, t: float) -> np.ndarray:
    """Interpolate two rigid transforms: slerp the rotation, lerp the translation."""

    out = np.eye(4)
    out[:3, :3] = slerp_rotation(t0[:3, :3], t1[:3, :3], t)
    out[:3, 3] = (1 - t) * t0[:3, 3] + t * t1[:3, 3]
    return out


def ease(t: float) -> float:
    """Smoothstep easing."""

    return t * t * (3.0 - 2.0 * t)


def yaw_between_deg(t_a: np.ndarray, t_b: np.ndarray) -> float:
    """Signed z-axis angle of ``R_a R_b^T`` in degrees (about the base z axis)."""

    delta = t_a[:3, :3] @ t_b[:3, :3].T
    return float(np.degrees(np.arctan2(delta[1, 0], delta[0, 0])))


# --------------------------------------------------------------------------- drawing


def overlay(
    gray: Any,
    points: np.ndarray,
    calib: sweep.KittiCalib,
    tf: np.ndarray,
    vendor: np.ndarray,
    crop: tuple[int, int, int, int],
    scale: float,
) -> PILImage.Image:
    """Image crop (dimmed) with depth-coloured, ground-free LiDAR dots at ``tf``."""

    image_mod, _, _ = sweep._pil()
    uv, depth = calib.project(points, tf, vendor)
    x0, y0, x1, y1 = crop
    size = (round((x1 - x0) * scale), round((y1 - y0) * scale))
    keep = (
        (points[:, 2] > sweep.GROUND_Z_VELO)
        & (depth > DEPTH_RANGE[0])
        & (depth < DEPTH_RANGE[1])
        & (uv[:, 0] >= x0)
        & (uv[:, 0] < x1)
        & (uv[:, 1] >= y0)
        & (uv[:, 1] < y1)
    )
    uv, depth = uv[keep], depth[keep]
    order = np.argsort(-depth)
    uv, depth = uv[order], depth[order]
    px = np.round((uv[:, 0] - x0) * scale).astype(int)
    py = np.round((uv[:, 1] - y0) * scale).astype(int)
    canvas = np.array(gray.crop(crop).resize(size, image_mod.Resampling.LANCZOS).convert("RGB"))
    colors = sweep.depth_colors(depth, *DEPTH_RANGE)
    radius = 1 if scale < 1.5 else 2
    for dx in range(radius + 1):
        for dy in range(radius + 1):
            xs = np.clip(px + dx, 0, size[0] - 1)
            ys = np.clip(py + dy, 0, size[1] - 1)
            canvas[ys, xs] = colors
    return image_mod.fromarray(canvas)


def draw_steps_row(draw: Any, active: int, y: int) -> None:
    """Three step chips; the active one is lit."""

    labels = ("1  Deployed calibration", "2  Calibrex estimate", "3  Re-check")
    font = sweep.load_font(13, True)
    x = 16
    chip_w = (sweep.WIDTH - 32 - 2 * 14) // 3
    for index, label in enumerate(labels, start=1):
        lit = index == active
        fill = (52, 64, 86) if lit else (28, 34, 44)
        outline = (120, 170, 255) if lit else BORDER
        draw.rounded_rectangle((x, y, x + chip_w, y + 30), radius=8, fill=fill, outline=outline)
        draw.text(
            (x + chip_w // 2, y + 15),
            label,
            font=font,
            fill=TEXT if lit else DIM,
            anchor="mm",
        )
        x += chip_w + 14


def draw_yaw_bar(
    draw: Any,
    box: tuple[int, int, int, int],
    error_deg: float,
    tolerance_deg: float,
    color: tuple[int, int, int],
    label: str,
) -> None:
    """``|yaw error|`` against the tolerance: green zone up to the tolerance, red beyond."""

    x0, y0, x1, y1 = box
    scale_max = 4.0 * tolerance_deg
    width = x1 - x0

    def px(value: float) -> int:
        return x0 + round(width * min(value, scale_max) / scale_max)

    draw.rectangle(box, fill=(44, 52, 66))
    draw.rectangle((x0, y0, px(tolerance_deg), y1), fill=(34, 84, 56))
    draw.rectangle((px(tolerance_deg), y0, x1, y1), fill=(92, 42, 46))
    fill_right = max(px(error_deg), x0 + 3)
    draw.rectangle((x0, y0 + 5, fill_right, y1 - 5), fill=color)
    draw.line((px(tolerance_deg), y0 - 5, px(tolerance_deg), y1 + 5), fill=TEXT, width=2)
    small = sweep.load_font(11)
    draw.text(
        (px(tolerance_deg), y1 + 7),
        f"tolerance {tolerance_deg:.2f}°",
        font=small,
        fill=MUTED,
        anchor="ma",
    )
    draw.text((x0, y1 + 7), "0°", font=small, fill=MUTED, anchor="la")
    draw.text((x1, y1 + 7), f"{scale_max:.1f}°", font=small, fill=MUTED, anchor="ra")
    draw.text((x0, y0 - 8), label, font=sweep.load_font(12, True), fill=TEXT, anchor="ls")


def draw_card(
    draw: Any,
    record: dict[str, Any] | None,
    *,
    interpolated_yaw: float | None = None,
    tolerance_deg: float = 0.5,
) -> None:
    """lidar-vehicle verdict card: the pill and the yaw bar (verdict from a real run)."""

    x0, y0, x1, y1 = CARD_BOX
    draw.rounded_rectangle(CARD_BOX, radius=10, fill=CARD, outline=BORDER)
    draw.text((x0 + 14, y0 + 10), "calibrex check", font=sweep.load_font(12), fill=MUTED)
    draw.text((x0 + 14, y0 + 26), "lidar-vehicle", font=sweep.load_font(20, True), fill=TEXT)
    if record is not None:
        status = record["status"]
        label = sweep.STATUS_LABEL.get(status, status.upper())
        color = STATUS_COLOR.get(status, MUTED)
        sweep.draw_pill_right(
            draw, x1 - 14, y0 + 14, label, color, sweep.load_font(30, True), height=48
        )
        yaw = record["axes"]["yaw"]
        pitch = record["axes"]["pitch"]
        error = abs(yaw["candidate_error_deg"])
        tolerance = yaw["tolerance_deg"]
        draw_yaw_bar(
            draw,
            (x0 + 14, y0 + 88, x1 - 14, y0 + 106),
            error,
            tolerance,
            STATUS_COLOR[yaw["status"]],
            f"yaw  |δ| = {error:.2f}°",
        )
        note = (
            f"pitch also judged: |δ| {abs(pitch['candidate_error_deg']):.2f}° ({pitch['status']})"
        )
        draw.text((x0 + 14, y1 - 8), note, font=sweep.load_font(11), fill=MUTED, anchor="ls")
    else:
        assert interpolated_yaw is not None
        sweep.draw_pill_right(
            draw, x1 - 14, y0 + 14, "· · ·", (74, 84, 100), sweep.load_font(30, True), height=48
        )
        draw_yaw_bar(
            draw,
            (x0 + 14, y0 + 88, x1 - 14, y0 + 106),
            interpolated_yaw,
            tolerance_deg,
            (150, 162, 180),
            f"yaw offset from estimate = {interpolated_yaw:.2f}°",
        )
        draw.text(
            (x0 + 14, y1 - 8),
            "interpolated tf, not a check result",
            font=sweep.load_font(11),
            fill=MUTED,
            anchor="ls",
        )


def compose_frame(
    *,
    step: int,
    title: str,
    subtitle: str,
    panel_image: PILImage.Image,
    inset_image: PILImage.Image,
    record: dict[str, Any] | None,
    interpolated_yaw: float | None,
    footnotes: Sequence[tuple[str, tuple[int, int, int]]],
) -> PILImage.Image:
    """One 960x540 frame."""

    image_mod, image_draw, _ = sweep._pil()
    frame = image_mod.new("RGB", (sweep.WIDTH, sweep.HEIGHT), BG)
    draw = image_draw.Draw(frame)
    draw.text((16, 8), title, font=sweep.load_font(28, True), fill=TEXT)
    draw.text((16, 56), subtitle, font=sweep.load_font(13), fill=MUTED, anchor="ls")
    draw.text(
        (sweep.WIDTH - 16, 18),
        "KITTI 2011_09_26 · real calibrex check runs",
        font=sweep.load_font(12),
        fill=MUTED,
        anchor="rm",
    )
    frame.paste(panel_image, (PANEL[0], PANEL[1]))
    draw.rectangle(PANEL, outline=BORDER)
    zx0, zy0, zx1, zy1 = ZOOM_REGION
    draw.rectangle(
        (
            PANEL[0] + zx0 - CROP[0],
            PANEL[1] + zy0 - CROP[1],
            PANEL[0] + zx1 - CROP[0],
            PANEL[1] + zy1 - CROP[1],
        ),
        outline=(255, 255, 255),
        width=1,
    )
    draw_card(draw, record, interpolated_yaw=interpolated_yaw)
    frame.paste(inset_image, (INSET_BOX[0], INSET_BOX[1]))
    draw.rectangle(INSET_BOX, outline=(255, 255, 255))
    draw.text(
        (INSET_BOX[0] + 8, INSET_BOX[1] + 6),
        "zoom \u00d72 on the bollards",
        font=sweep.load_font(11, True),
        fill=TEXT,
    )
    draw.text(
        (PANEL[0], PANEL[3] + 6),
        "KITTI camera image + LiDAR points (colour = distance 5-25 m, ground points hidden)",
        font=sweep.load_font(11),
        fill=MUTED,
    )
    draw_steps_row(draw, step, 412)
    y = 458
    for text, color in footnotes:
        draw.text((16, y), text, font=sweep.load_font(13), fill=color)
        y += 22
    return frame


def build(
    drive_dir: Path, image_index: int = IMAGE_INDEX
) -> tuple[list[PILImage.Image], list[int], dict[str, PILImage.Image]]:
    """All frames, their durations, and one preview frame per act."""

    act1 = load_json(ACT1)
    act3 = load_json(ACT3)
    calib = sweep.KittiCalib(drive_dir.parent)
    inputs = sweep.drive_inputs(drive_dir, image_index)
    points = sweep.read_velodyne(inputs["velodyne"])[:, :3].astype(np.float64)
    gray = sweep.load_dimmed_image(inputs["image"])
    vendor = vendor_tf(act1)
    tf1, tf3 = velo_tf(act1), velo_tf(act3)
    rec1, rec3 = lidar_vehicle(act1), lidar_vehicle(act3)
    yaw_start = abs(rec1["axes"]["yaw"]["candidate_error_deg"])
    yaw_end = abs(rec3["axes"]["yaw"]["candidate_error_deg"])
    off_vendor = vendor_to_estimate_yaw_deg(act1)
    inj = act1["yaw_injected_deg"]

    def render(tf: np.ndarray) -> tuple[PILImage.Image, PILImage.Image]:
        main = overlay(gray, points, calib, tf, vendor, CROP, 1.0)
        inset = overlay(gray, points, calib, tf, vendor, ZOOM_REGION, 2.0)
        return main, inset

    frames: list[PILImage.Image] = []
    durations: list[int] = []
    previews: dict[str, PILImage.Image] = {}

    main, inset = render(tf1)
    f1 = compose_frame(
        step=1,
        title=f"1  Deployed calibration: yaw {inj:+.0f}°",
        subtitle="the LiDAR-to-vehicle transform is rotated; the points miss the image edges",
        panel_image=main,
        inset_image=inset,
        record=rec1,
        interpolated_yaw=None,
        footnotes=[
            (
                f"calibrex check: yaw is {yaw_start:.2f}° from what the data says, "
                f"{yaw_start / rec1['axes']['yaw']['tolerance_deg']:.1f}\u00d7 tolerance  →  FAIL",
                TEXT,
            ),
            (
                "deployed tf = the vendor tf with a deliberate yaw error injected into velo_link",
                MUTED,
            ),
        ],
    )
    frames.append(f1)
    durations.append(ACT1_MS)
    previews["act1"] = f1

    for i in range(ACT2_FRAMES):
        t = ease((i + 1) / ACT2_FRAMES)
        tf = slerp_tf(tf1, tf3, t)
        main, inset = render(tf)
        offset = abs(yaw_between_deg(tf, tf3))
        frame = compose_frame(
            step=2,
            title="2  Calibrex estimate from the data",
            subtitle="moving to the transform estimated from the data",
            panel_image=main,
            inset_image=inset,
            record=None,
            interpolated_yaw=offset,
            footnotes=[
                (
                    "the estimate comes from LiDAR odometry of the drive (no target, no markers)",
                    TEXT,
                ),
                ("the points slide onto the edges", MUTED),
            ],
        )
        frames.append(frame)
        durations.append(ACT2_FRAME_MS + (ACT2_HOLD_MS if i == ACT2_FRAMES - 1 else 0))
        if i == ACT2_FRAMES // 2:
            previews["act2"] = frame

    main, inset = render(tf3)
    f3 = compose_frame(
        step=3,
        title="3  Re-check: PASS",
        subtitle="the estimate is now the deployed tf, and calibrex check is run again",
        panel_image=main,
        inset_image=inset,
        record=rec3,
        interpolated_yaw=None,
        footnotes=[
            (
                f"yaw |δ| is now {yaw_end:.2f}°; pitch and yaw judged: a partial PASS",
                TEXT,
            ),
            (
                f"for reference, the vendor KITTI tf is yaw {off_vendor:.2f}° from this estimate; "
                "roll is not observable from driving",
                MUTED,
            ),
            (
                "imu-vehicle: inconclusive on KITTI (no axis constrained); other pairs not shown",
                DIM,
            ),
        ],
    )
    frames.append(f3)
    durations.append(ACT3_MS)
    previews["act3"] = f3
    previews["inset"] = inset
    return frames, durations, previews


# --------------------------------------------------------------------------- provenance


def git_commit() -> str:
    """The repository HEAD (or ``unknown``)."""

    return sweep.git_commit()


def build_provenance(drive_dir: Path, image_index: int, frames: int) -> dict[str, Any]:
    """Provenance of the GIF: digests of every input, versions, commands and license note."""

    run = load_json(RUN_MANIFEST)
    act1 = load_json(ACT1)
    act3 = load_json(ACT3)
    calib = sweep.KittiCalib(drive_dir.parent)
    kitti = {
        path.name: sweep.sha256_file(path)
        for path in [*sweep.drive_inputs(drive_dir, image_index).values(), *calib.files]
    }
    gif = REPO / GIF
    return {
        "schema": PROVENANCE_SCHEMA,
        "output": GIF.as_posix(),
        "generator": "tools/check_story_gif.py",
        "run_tool": "tools/check_story_run.py",
        "calibrex_version": sweep.calibrex_version(),
        "git_commit": git_commit(),
        "check_generator_version": act1["generator_version"],
        "check_git_commit": act1["git_commit"],
        "pooled_bag_input_sha256": run["bag_input_sha256"],
        "drives": run["drives"],
        "story": {
            "act1": "deployed velo_link yaw "
            f"{act1['yaw_injected_deg']:+g} deg (R' = Rz(yaw) R about base_link z): calibrex check "
            f"lidar-vehicle {lidar_vehicle(act1)['status']}",
            "act2": "slerp of velo_link rotation (lerp of translation) from the act-1 tf to the "
            "estimate; interpolated, not a check result",
            "act3": "estimate deployed as --tf, calibrex check re-run: lidar-vehicle "
            f"{lidar_vehicle(act3)['status']} (partial: pitch, yaw)",
            "estimate_derivation": run["estimate_derivation"],
            "estimate_rotation_quat_xyzw": run["estimate_rotation_quat_xyzw"],
            "vendor_to_estimate_yaw_deg": round(vendor_to_estimate_yaw_deg(act1), 4),
            "overall_verdicts_not_shown": {
                "act1": act1["overall_verdict"],
                "act3": act3["overall_verdict"],
            },
        },
        "summaries_sha256": {
            path.name: sweep.sha256_file(path) for path in (ACT1, ACT3, RUN_MANIFEST)
        },
        "check_commands": {"act1": act1["command"], "act3": act3["command"]},
        "check_command_note": "--tf and --output differ per act; frames files are in the work dir",
        "gif_commands": [
            "python tools/check_story_run.py --bag <pooled KITTI dev bag> --work-dir <dir>",
            "python tools/check_story_gif.py render --drive-dir <KITTI drive 0005> "
            f"--image-index {image_index}",
        ],
        "overlay": {
            "drive": "2011_09_26_drive_0005_sync (a dev drive)",
            "image_index": image_index,
            "crop_px": list(CROP),
            "zoom_region_px": list(ZOOM_REGION),
            "points": "ground returns hidden (velo z < "
            f"{sweep.GROUND_Z_VELO} m), distance {DEPTH_RANGE[0]:g}-{DEPTH_RANGE[1]:g} m kept",
            "kitti_input_sha256": kitti,
        },
        "gif": {
            "sha256": sweep.sha256_file(gif) if gif.exists() else None,
            "size_bytes": gif.stat().st_size if gif.exists() else None,
            "frames": frames,
            "width": sweep.WIDTH,
            "height": sweep.HEIGHT,
        },
        "dataset_license": "KITTI raw data, CC BY-NC-SA 3.0: non-commercial use only; the GIF "
        "contains a KITTI camera image and must stay non-commercial",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("command", choices=["render", "manifest", "preview"])
    parser.add_argument("--drive-dir", type=Path, required=True)
    parser.add_argument("--image-index", type=int, default=IMAGE_INDEX)
    parser.add_argument("--out-dir", type=Path, help="preview PNG directory")
    args = parser.parse_args(argv)

    if args.command == "manifest":
        count = load_json(PROVENANCE)["gif"]["frames"]
        PROVENANCE.write_text(
            json.dumps(
                build_provenance(args.drive_dir, args.image_index, count), indent=1, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )
        return 0
    frames, durations, previews = build(args.drive_dir, args.image_index)
    if args.command == "preview":
        out = args.out_dir or Path(".")
        out.mkdir(parents=True, exist_ok=True)
        for name, image in previews.items():
            image.save(out / f"story-{name}.png")
        return 0
    output = REPO / GIF
    sweep.save_gif(frames, durations, output)
    size = output.stat().st_size
    print(f"wrote {output} ({len(frames)} frames, {size} bytes, {sum(durations) / 1000:.1f} s)")
    PROVENANCE.write_text(
        json.dumps(
            build_provenance(args.drive_dir, args.image_index, len(frames)),
            indent=1,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if size > sweep.MAX_BYTES:
        print(f"warning: larger than {sweep.MAX_BYTES} bytes", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
