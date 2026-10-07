"""Render the camera-LiDAR check README GIF from two real ``calibrex check`` runs.

``calibrex check --pairs camera-lidar`` judges the camera-LiDAR rotation of a bag from its
images and point clouds, without a target. This GIF shows two real runs on KITTI development
drive 0005 (converted with ``calibrex convert kitti-raw --camera image_02``):

1. the vendor calibration: ``pass`` on the judged axes;
2. the vendor calibration turned +3 deg about the camera y axis (``T_cam<-lidar`` left-multiplied
   by R): ``fail``.

Frames between the two ends slide the overlay between the two candidates (a pure projection,
not check results; marked "interpolated" in the picture). Only rotation is judged: translation
is not, and the picture says so.

Usage (``render`` needs numpy and Pillow)::

    calibrex convert kitti-raw DRIVE --camera image_02 --output BAG
    python tools/camera_lidar_check_gif.py run --bag BAG --work-dir /tmp/clcheck
    python tools/camera_lidar_check_gif.py render --drive-dir DRIVE
    python tools/camera_lidar_check_gif.py manifest --drive-dir DRIVE
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
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
OUT_DIR = REPO / "docs/assets/calibrex_check_camera_lidar"
GIF = Path("docs/assets/calibrex-check-camera-lidar.gif")
PROVENANCE = OUT_DIR / "provenance.json"
RUN_SCHEMA = "calibrex.check_camera_lidar_run/v0"
PROVENANCE_SCHEMA = "calibrex.check_camera_lidar_gif/v0"
PAIR = "camera-lidar"
CAMERA_FRAME = "cam2_optical"
PERTURB_DEG = 3.0
PERTURB_AXIS = "y"  # camera optical y: "pitch" in the check's roll / pitch / yaw naming
IMAGE_INDEX = 60
CROP = (150, 40, 1100, 330)  # 950 x 290 px of the 1242 x 375 rectified image
SCALE = 928 / 950
DEPTH_RANGE = (5.0, 30.0)
SLIDE_FRAMES = 12
FRAME_MS = 70
HOLD_MS = 2800
CLI_SNIPPET = "import sys; from calibrex.cli.main import main; sys.exit(main())"
PATH = re.compile(r"(?<![\w.|])/(?:[^\s'\"<>/]+/)*[^\s'\"<>/]+")

PANEL = (16, 56, 944, 346)
CARD_L = (16, 360, 336, 526)
CARD_M = (352, 360, 640, 526)
CARD_R = (656, 360, 944, 526)


# ------------------------------------------------------------------------------- run


def rotation_about(axis: str, degrees: float) -> np.ndarray:
    """Rotation matrix about a coordinate axis."""

    from scipy.spatial.transform import Rotation

    matrix: np.ndarray = Rotation.from_euler(axis, degrees, degrees=True).as_matrix()
    return matrix


def write_frames(bag: Path, degrees: float, output: Path) -> None:
    """Write a ``slac.check_frames`` file with the camera tf turned ``degrees`` about camera y."""

    import yaml
    from scipy.spatial.transform import Rotation

    from calibrex.check.tf_sources import load_bag_tf_static

    source = load_bag_tf_static(bag)
    assert source is not None
    edge = next(e for e in source.edges if e.child == CAMERA_FRAME)
    rotation = Rotation.from_quat(edge.transform.rotation_quat_xyzw) * Rotation.from_euler(
        PERTURB_AXIS, -degrees, degrees=True
    )  # T_imu<-cam' = T_imu<-cam R^T  <=>  T_cam<-imu' = R T_cam<-imu
    payload = {
        "schema_version": "slac.check_frames/v0.1",
        "frames": [
            {
                "name": edge.child,
                "parent": edge.parent,
                "translation_m": [float(v) for v in edge.transform.translation_m],
                "rotation_quat_xyzw": [float(v) for v in rotation.as_quat()],
            }
        ],
    }
    output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def summarize(artifact: dict[str, Any], degrees: float, command: Sequence[str]) -> dict[str, Any]:
    """Reduce a ``calibrex check`` artifact to the small summary the GIF renders from."""

    record = next(p for p in artifact["pairs"] if p["pair"] == PAIR)
    provenance = artifact["provenance"]
    return {
        "schema": RUN_SCHEMA,
        "perturbation_deg": degrees,
        "perturbation_axis": f"camera optical {PERTURB_AXIS}",
        "overall_verdict": artifact["overall_verdict"],
        "status": record["status"],
        "coverage": record.get("coverage"),
        "axes": {
            a["name"]: {
                "candidate_error_deg": round(a["candidate_error"], 4),
                "tolerance_deg": round(a["tolerance"], 4),
                "ratio": round(a["ratio"], 4),
                "status": a["status"],
                "estimate_std_deg": round(a["estimate_std"], 4),
            }
            for a in record["axes"]
        },
        "unchecked": [a["name"] for a in record.get("unchecked_axes", [])],
        "notes": record.get("notes", []),
        "command": [PATH.sub(".../x", c) if c.startswith("/") else c for c in command],
        "created_at": provenance["created_at"],
        "generator_version": provenance["generator_version"],
        "git_commit": provenance.get("git_commit"),
        "bag_input_sha256": artifact["bag"]["input_sha256"],
    }


def run(bag: Path, work_dir: Path) -> None:
    """Run ``calibrex check --pairs camera-lidar`` at 0 and +3 deg and store the summaries."""

    work_dir.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    for name, degrees in (("reference", 0.0), ("perturbed", PERTURB_DEG)):
        frames = work_dir / f"frames_{name}.yaml"
        write_frames(bag, degrees, frames)
        output = work_dir / f"{name}.json"
        command = [
            "calibrex",
            "check",
            str(bag),
            "--pairs",
            PAIR,
            "--cache-dir",
            str(work_dir / "cache"),
            "--tf",
            str(frames),
            "--output",
            str(output),
            "--fail-on",
            "never",
        ]
        subprocess.run([sys.executable, "-c", CLI_SNIPPET, *command[1:]], check=True, env=env)
        artifact = json.loads(output.read_text(encoding="utf-8"))
        summary = summarize(artifact, degrees, command)
        (OUT_DIR / f"{name}.json").write_text(
            json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"{name}: {summary['status']} {summary['axes']}")


# ----------------------------------------------------------------------------- render


class Scene:
    """One KITTI frame: dimmed image, Velodyne points and the date-level calibration."""

    def __init__(self, drive_dir: Path, image_index: int) -> None:
        inputs = sweep.drive_inputs(drive_dir, image_index)
        self.inputs = inputs
        self.calib = sweep.KittiCalib(drive_dir.parent)
        self.gray = sweep.load_dimmed_image(inputs["image"])
        points = sweep.read_velodyne(inputs["velodyne"])
        self.points = points[points[:, 2] > sweep.GROUND_Z_VELO]
        homogeneous = np.hstack([self.points[:, :3], np.ones((len(self.points), 1))])
        self.cam_vendor = (self.calib.r_rect @ self.calib.t_cam_velo @ homogeneous.T)[:3].T

    def overlay(self, degrees: float) -> PILImage.Image:
        """Image crop with the points projected through the vendor tf turned ``degrees``."""

        image_mod, _, _ = sweep._pil()
        cam = self.cam_vendor @ rotation_about(PERTURB_AXIS, degrees).T
        depth = cam[:, 2]
        pixels = (self.calib.p_rect2 @ np.vstack([cam.T, np.ones(len(cam))])).T
        scale = np.where(pixels[:, 2] > 1e-6, pixels[:, 2], 1e-6)
        uv = pixels[:, :2] / scale[:, None]
        x0, y0, x1, y1 = CROP
        keep = (
            (depth > DEPTH_RANGE[0])
            & (depth < DEPTH_RANGE[1])
            & (uv[:, 0] >= x0)
            & (uv[:, 0] < x1)
            & (uv[:, 1] >= y0)
            & (uv[:, 1] < y1)
        )
        uv, depth = uv[keep], depth[keep]
        order = np.argsort(-depth)
        uv, depth = uv[order], depth[order]
        size = (PANEL[2] - PANEL[0], PANEL[3] - PANEL[1])
        px = np.round((uv[:, 0] - x0) * SCALE).astype(int)
        py = np.round((uv[:, 1] - y0) * SCALE).astype(int)
        canvas = np.array(
            self.gray.crop(CROP).resize(size, image_mod.Resampling.LANCZOS).convert("RGB")
        )
        colors = sweep.depth_colors(depth, *DEPTH_RANGE)
        for dx in (0, 1):
            for dy in (0, 1):
                canvas[np.clip(py + dy, 0, size[1] - 1), np.clip(px + dx, 0, size[0] - 1)] = colors
        return image_mod.fromarray(canvas)


def load_summary(name: str) -> dict[str, Any]:
    """A stored run summary."""

    data = json.loads((OUT_DIR / f"{name}.json").read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def draw_card(
    draw: Any,
    box: tuple[int, int, int, int],
    heading: str,
    summary: dict[str, Any] | None,
    degrees: float,
) -> None:
    """The verdict card of one real run (or a neutral card while the overlay slides)."""

    x0, y0, x1, _y1 = box
    draw.rounded_rectangle(box, radius=10, fill=sweep.CARD, outline=sweep.BORDER)
    draw.text((x0 + 14, y0 + 10), heading, font=sweep.load_font(12), fill=sweep.MUTED)
    draw.text(
        (x0 + 14, y0 + 26), f"{degrees:+.1f}°", font=sweep.load_font(40, True), fill=sweep.TEXT
    )
    if summary is None:
        draw.text(
            (x0 + 14, y0 + 84),
            "interpolated overlay,",
            font=sweep.load_font(12),
            fill=sweep.MUTED,
        )
        draw.text(
            (x0 + 14, y0 + 100), "not a check result", font=sweep.load_font(12), fill=sweep.MUTED
        )
        return
    status = summary["status"]
    sweep.draw_pill_right(
        draw,
        x1 - 14,
        y0 + 22,
        sweep.STATUS_LABEL.get(status, status.upper()),
        sweep.STATUS_COLOR.get(status, sweep.MUTED),
        sweep.load_font(16, True),
        height=30,
    )
    draw.text(
        (x1 - 14, y0 + 62),
        "calibrex check --pairs camera-lidar",
        font=sweep.load_font(10),
        fill=sweep.MUTED,
        anchor="ra",
    )
    small = sweep.load_font(12)
    y = y0 + 84
    for name, axis in summary["axes"].items():
        err, tol = abs(axis["candidate_error_deg"]), axis["tolerance_deg"]
        color = sweep.STATUS_COLOR.get(axis["status"], sweep.MUTED)
        draw.text((x0 + 14, y), f"{name}", font=sweep.load_font(13, True), fill=sweep.TEXT)
        draw.text((x0 + 66, y), f"|δ| {err:.2f}° / tol {tol:.2f}°", font=small, fill=color)
        y += 20
    if summary["unchecked"]:
        draw.text(
            (x0 + 14, y + 2),
            "not judged: " + ", ".join(summary["unchecked"]),
            font=small,
            fill=sweep.MUTED,
        )


def draw_notes(draw: Any, box: tuple[int, int, int, int], x_offset: int = 0) -> None:
    """Static honesty note: what this pair judges."""

    x0, y0, _x1, _y1 = box
    draw.rounded_rectangle(box, radius=10, fill=sweep.CARD, outline=sweep.BORDER)
    font = sweep.load_font(12)
    lines = [
        ("Rotation only.", True),
        ("x y z are never judged:", False),
        ("the edge objective is flat in", False),
        ("translation (up to 11 cm off).", False),
        ("The overlay shows projected", False),
        ("LiDAR returns, coloured by depth.", False),
    ]
    y = y0 + 12
    for text, bold in lines:
        draw.text(
            (x0 + 14 + x_offset, y),
            text,
            font=sweep.load_font(13, True) if bold else font,
            fill=sweep.TEXT if bold else sweep.MUTED,
        )
        y += 22


def build_frames(scene: Scene) -> tuple[list[PILImage.Image], list[int]]:
    """Hold the reference, slide to +3 deg, hold the failing candidate, slide back."""

    image_mod, image_draw, _ = sweep._pil()
    reference, perturbed = load_summary("reference"), load_summary("perturbed")

    def frame(degrees: float, summary: dict[str, Any] | None, heading: str) -> PILImage.Image:
        canvas = image_mod.new("RGB", (sweep.WIDTH, sweep.HEIGHT), sweep.BG)
        draw = image_draw.Draw(canvas)
        sweep.draw_header(
            draw,
            "Camera ↔ LiDAR: rotation judged with no target",
            "KITTI drive 0005 · real calibrex check runs",
        )
        canvas.paste(scene.overlay(degrees), (PANEL[0], PANEL[1]))
        draw.rectangle(PANEL, outline=sweep.BORDER)
        draw_card(draw, CARD_L, heading, summary, degrees)
        draw.rounded_rectangle(CARD_M, radius=10, fill=sweep.CARD, outline=sweep.BORDER)
        draw.text(
            (CARD_M[0] + 14, CARD_M[1] + 10),
            "what the check sees",
            font=sweep.load_font(12),
            fill=sweep.MUTED,
        )
        for i, text in enumerate(
            (
                "LiDAR depth edges are scored",
                "against image edges, pooled",
                "over 16 frames, 1 s apart.",
            )
        ):
            draw.text(
                (CARD_M[0] + 14, CARD_M[1] + 34 + 20 * i),
                text,
                font=sweep.load_font(12),
                fill=sweep.TEXT,
            )
        draw.text(
            (CARD_M[0] + 14, CARD_M[1] + 104),
            "Turned about the camera y axis",
            font=sweep.load_font(12),
            fill=sweep.MUTED,
        )
        draw.text(
            (CARD_M[0] + 14, CARD_M[1] + 124),
            "(the check's pitch).",
            font=sweep.load_font(12),
            fill=sweep.MUTED,
        )
        draw_notes(draw, CARD_R)
        return canvas

    frames: list[PILImage.Image] = []
    durations: list[int] = []
    frames.append(frame(0.0, reference, "deployed: vendor calibration"))
    durations.append(HOLD_MS)
    for i in range(1, SLIDE_FRAMES):
        t = i / SLIDE_FRAMES
        eased = t * t * (3.0 - 2.0 * t)
        frames.append(frame(PERTURB_DEG * eased, None, "sliding the candidate"))
        durations.append(FRAME_MS)
    frames.append(frame(PERTURB_DEG, perturbed, "deployed: turned +3° (known bad)"))
    durations.append(HOLD_MS)
    for i in range(1, SLIDE_FRAMES):
        t = 1.0 - i / SLIDE_FRAMES
        eased = t * t * (3.0 - 2.0 * t)
        frames.append(frame(PERTURB_DEG * eased, None, "sliding the candidate"))
        durations.append(FRAME_MS)
    return frames, durations


def render(drive_dir: Path, output: Path | None) -> Path:
    """Render and save the GIF."""

    scene = Scene(drive_dir, IMAGE_INDEX)
    frames, durations = build_frames(scene)
    target = output or REPO / GIF
    sweep.save_gif(frames, durations, target)
    print(f"wrote {target} ({len(frames)} frames, {target.stat().st_size} bytes)")
    return target


def manifest(drive_dir: Path) -> None:
    """Write the provenance JSON of the GIF."""

    reference, perturbed = load_summary("reference"), load_summary("perturbed")
    files = [
        *sweep.drive_inputs(drive_dir, IMAGE_INDEX).values(),
        *sweep.KittiCalib(drive_dir.parent).files,
        drive_dir.parent / "calib_cam_to_cam.txt",
    ]
    gif = REPO / GIF
    payload = {
        "schema": PROVENANCE_SCHEMA,
        "generator": "tools/camera_lidar_check_gif.py",
        "output": GIF.as_posix(),
        "calibrex_version": sweep.calibrex_version(),
        "git_commit": sweep.git_commit(),
        "drive": "2011_09_26_drive_0005_sync (development drive; no drive 0027-0059)",
        "bag": "calibrex convert kitti-raw DRIVE --camera image_02",
        "bag_input_sha256": reference["bag_input_sha256"],
        "check_runs": {
            "reference": {k: reference[k] for k in ("status", "axes", "unchecked", "command")},
            "perturbed": {k: perturbed[k] for k in ("status", "axes", "unchecked", "command")},
        },
        "run_summary_sha256": {
            f"{name}.json": sweep.sha256_file(OUT_DIR / f"{name}.json")
            for name in ("reference", "perturbed")
        },
        "perturbation": f"T_cam<-lidar' = R_{PERTURB_AXIS}({PERTURB_DEG} deg) T_cam<-lidar "
        "(camera optical axes)",
        "overlay": {
            "image_index": IMAGE_INDEX,
            "crop_px": list(CROP),
            "projection": "pixel = P_rect_02 R(delta) R_rect_00 T_cam<-velo p; frames between "
            "the two check runs only slide R(delta)",
            "kitti_input_sha256": {p.name: sweep.sha256_file(p) for p in files},
        },
        "gif_commands": [
            "calibrex convert kitti-raw <KITTI drive 0005> --camera image_02 --output <bag>",
            "python tools/camera_lidar_check_gif.py run --bag <bag> --work-dir <dir>",
            "python tools/camera_lidar_check_gif.py render --drive-dir <KITTI drive 0005>",
        ],
        "gif": {
            "sha256": sweep.sha256_file(gif),
            "size_bytes": gif.stat().st_size,
            "width": sweep.WIDTH,
            "height": sweep.HEIGHT,
        },
        "dataset_license": "KITTI raw data, CC BY-NC-SA 3.0: non-commercial use only; the GIF "
        "contains a KITTI camera image and must stay non-commercial",
    }
    PROVENANCE.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {PROVENANCE}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="run the two real check runs")
    run_p.add_argument("--bag", type=Path, required=True)
    run_p.add_argument("--work-dir", type=Path, required=True)
    render_p = sub.add_parser("render", help="render the GIF from the stored summaries")
    render_p.add_argument("--drive-dir", type=Path, required=True)
    render_p.add_argument("--output", type=Path)
    manifest_p = sub.add_parser("manifest", help="write provenance.json")
    manifest_p.add_argument("--drive-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args.bag, args.work_dir)
    elif args.command == "render":
        render(args.drive_dir, args.output)
    else:
        manifest(args.drive_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
