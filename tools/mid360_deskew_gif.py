"""README GIF: one hand-held MID360 sweep during fast rotation, raw vs gyro-deskewed.

Data: RTK-SLAM ``construction_seq1`` (Livox MID360 + built-in IMU, CC BY 4.0).  The scan
is the one with the highest mean gyro rate over its 0.1 s sweep.  Deskewing uses
calibrex's own ``gyro_rotation_model`` (``calibrex.solvers.imu_lidar_rotation_solver``)
with the rotation, time offset and gyro bias that ``calibrex imu-lidar livox``
estimated for this sequence, applied through ``deskew_points_with_rotations`` (rotation
only: translation within 0.1 s is ignored).  No data is simulated.

``extract``  needs the bag and calibrex; it stores ~10k points and the IMU window as ``.npz``.
``render``   needs only numpy, scipy, Pillow and calibrex's version string.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pointcloud_gif_common as pc

ROOT = Path(__file__).resolve().parents[1]
FloatArray = NDArray[np.float64]
DEFAULT_BAG = Path("/media/sasaki/aiueo2/datasets/rtk_slam/ros2/construction_seq1")
DEFAULT_CALIB = Path("/media/sasaki/aiueo2/datasets/rtk_slam/rtk_slam_eval/calib/calib.yaml")
DEFAULT_ARTIFACT = ROOT / "docs/assets/mid360_imu_lidar_rotation_rtk_slam_construction_seq1.yaml"
DEFAULT_NPZ = ROOT / "docs/assets/mid360-deskew/construction_seq1_sweep.npz"
DEFAULT_GIF = ROOT / "docs/assets/mid360-deskew.gif"
DEFAULT_MANIFEST = ROOT / "docs/assets/mid360-deskew/manifest.json"
SWEEP_S = 0.1
MIN_RANGE_M = 0.5
MAX_RANGE_M = 25.0
IMU_WINDOW_S = 0.6
# Phases (frames): raw, morph to deskewed, deskewed, morph back.  Loop is seamless.
HOLD_RAW, MORPH_IN, HOLD_DESKEW, MORPH_OUT = 10, 12, 14, 12
FRAMES = HOLD_RAW + MORPH_IN + HOLD_DESKEW + MORPH_OUT
CAMERA_AZIMUTH = 180.0
CAMERA_SWING_DEG = 28.0
CAMERA_ELEVATION = 8.0
CAMERA_DISTANCE = 10.0
FOCAL_PX = 640.0
VIEW_W = 700
THICKNESS_RANGE_M = 15.0


def read_estimate(artifact: Path) -> tuple[FloatArray, float, FloatArray]:
    import yaml
    from scipy.spatial.transform import Rotation

    data = yaml.safe_load(artifact.read_text(encoding="utf-8"))
    rotation = Rotation.from_quat(data["rotation_quat_xyzw"]).as_matrix()
    return rotation, float(data["time_offset_s"]), np.asarray(data["gyro_bias_rps"], dtype=float)


def extract(bag: Path, artifact: Path, calib: Path, out: Path) -> None:
    """Find the fastest sweep, deskew it with calibrex's gyro model, store the subset."""

    import json

    from calibrex import __version__
    from calibrex.data.livox_ros2 import (
        LIVOX_PROFILES,
        bag_input_digest,
        iter_livox_points,
        load_livox_imu,
    )
    from calibrex.solvers.imu_lidar_rotation_solver import GyroSeries, gyro_rotation_model
    from calibrex.solvers.scan_to_scan_odometry import deskew_points_with_rotations

    profile = LIVOX_PROFILES["rtk-slam"]
    imu = load_livox_imu(bag, profile)
    gyro = GyroSeries(imu.times_s, imu.gyro_rps)
    # Peak of the 0.1 s mean rate, from the IMU alone; then the scan whose sweep matches best.
    window = round(SWEEP_S * 200)
    rate = np.degrees(np.linalg.norm(imu.gyro_rps, axis=1))
    mean_rate = np.convolve(rate, np.ones(window) / window, mode="valid")
    peak_s = float(imu.times_s[int(np.argmax(mean_rate))])
    best: tuple[float, float, FloatArray, FloatArray] | None = None
    stop_after = peak_s + 1.0 - float(imu.times_s[0])
    for time_s, points, offsets in iter_livox_points(bag, profile, max_seconds=stop_after):
        if abs(time_s - peak_s) > 1.0 or offsets is None:
            continue
        inside = (imu.times_s >= time_s) & (imu.times_s <= time_s + SWEEP_S)
        sweep_rate = float(np.degrees(np.linalg.norm(imu.gyro_rps[inside].mean(axis=0))))
        if best is None or sweep_rate > best[1]:
            best = (time_s, sweep_rate, points, offsets)
    assert best is not None
    scan_s, sweep_rate, points, offsets = best
    rotation, offset_s, bias = read_estimate(artifact)
    keep = np.linalg.norm(points, axis=1)
    keep = (keep >= MIN_RANGE_M) & (keep <= MAX_RANGE_M) & np.isfinite(points).all(axis=1)
    points, offsets = points[keep], offsets[keep]
    model = gyro_rotation_model(gyro, rotation, bias, offset_s)
    rotations = model(scan_s, offsets)
    deskewed = deskew_points_with_rotations(points, offsets, rotations, np.eye(4), SWEEP_S)
    from scipy.spatial.transform import Rotation

    sweep_angle = float(
        np.degrees(Rotation.from_matrix(rotations[int(np.argmax(offsets))]).magnitude())
    )
    around = (imu.times_s >= scan_s - IMU_WINDOW_S / 2) & (imu.times_s <= scan_s + IMU_WINDOW_S / 2)
    digest, scope = bag_input_digest([bag])
    meta: dict[str, Any] = {
        "bag": bag.name,
        "bag_digest_sha256": digest,
        "bag_digest_scope": scope,
        "artifact": artifact.name,
        "artifact_sha256": pc.file_sha256(artifact),
        "calib_yaml_sha256": pc.file_sha256(calib) if calib.exists() else None,
        "calibrex_version_extract": __version__,
        "calibrex_commit_extract": pc.git_head(ROOT),
        "scan_header_s": scan_s,
        "scan_seconds_from_imu_start": scan_s - float(imu.times_s[0]),
        "sweep_mean_rate_deg_s": sweep_rate,
        "sweep_rotation_deg": sweep_angle,
        "sweep_duration_s": float(offsets.max()),
        "time_offset_s": offset_s,
        "gyro_bias_rps": [float(x) for x in bias],
        "range_filter_m": [MIN_RANGE_M, MAX_RANGE_M],
        "profile": "rtk-slam",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        raw=points.astype(np.float32),
        deskewed=deskewed.astype(np.float32),
        offsets_s=offsets.astype(np.float32),
        imu_t_s=(imu.times_s[around] - scan_s).astype(np.float64),
        imu_rate_deg_s=rate[around].astype(np.float32),
        meta=np.asarray(json.dumps(meta, sort_keys=True)),
    )
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB): {meta}")


def thickness_cm(points: FloatArray) -> float:
    """Median local surface thickness (sqrt of the smallest covariance eigenvalue of 12-NN)."""

    from scipy.spatial import cKDTree

    near = points[np.linalg.norm(points, axis=1) < THICKNESS_RANGE_M]
    _, index = cKDTree(near).query(near, k=12)
    around = near[index]
    centred = around - around.mean(axis=1, keepdims=True)
    values = np.linalg.eigvalsh(np.einsum("nki,nkj->nij", centred, centred) / 12.0)
    return float(np.median(np.sqrt(np.clip(values[:, 0], 0.0, None))) * 100.0)


def height_colours(points: FloatArray) -> FloatArray:
    anchors = np.array([[60, 110, 255], [60, 200, 255], [120, 255, 200], [255, 245, 150]], float)
    position = np.clip((points[:, 2] + 2.0) / 5.0, 0.0, 1.0) * (len(anchors) - 1)
    low = np.floor(position).astype(int).clip(0, len(anchors) - 2)
    frac = (position - low)[:, None]
    return np.asarray(anchors[low] * (1.0 - frac) + anchors[low + 1] * frac)


def schedule() -> list[tuple[float, str]]:
    """Per-frame ``(deskew amount 0..1, label)``."""

    out: list[tuple[float, str]] = [(0.0, "raw sweep")] * HOLD_RAW
    out += [(pc.smoothstep((k + 1) / MORPH_IN), "gyro deskew") for k in range(MORPH_IN)]
    out += [(1.0, "gyro-deskewed")] * HOLD_DESKEW
    out += [(1.0 - pc.smoothstep((k + 1) / MORPH_OUT), "undo deskew") for k in range(MORPH_OUT)]
    return out


def render(npz: Path, gif: Path, preview_dir: Path | None, manifest: Path | None) -> dict[str, Any]:
    import json

    from PIL import Image, ImageDraw

    from calibrex import __version__

    data = np.load(npz)
    meta: dict[str, Any] = json.loads(str(data["meta"]))
    raw = data["raw"].astype(np.float64)
    deskewed = data["deskewed"].astype(np.float64)
    colours = height_colours(deskewed)
    plan = schedule()
    thickness = [thickness_cm((1.0 - a) * raw + a * deskewed) for a, _ in plan]
    centre = np.median(deskewed[np.linalg.norm(deskewed, axis=1) < 12.0], axis=0)
    view_h = pc.HEIGHT - 44
    frames: list[Image.Image] = []
    for number, (amount, label) in enumerate(plan):
        azimuth = CAMERA_AZIMUTH + CAMERA_SWING_DEG * np.sin(2.0 * np.pi * number / len(plan))
        camera = pc.orbit_camera(
            centre, azimuth, CAMERA_ELEVATION, CAMERA_DISTANCE, FOCAL_PX, VIEW_W, view_h
        )
        canvas = np.empty((pc.HEIGHT, pc.WIDTH, 3), dtype=np.uint8)
        canvas[:] = pc.BG
        canvas[44:, :VIEW_W] = (10, 13, 18)
        pc.splat(
            canvas,
            camera,
            [((1.0 - amount) * raw + amount * deskewed, colours, 2)],
            origin=(0, 44),
        )
        image = Image.fromarray(canvas)
        draw_chrome(ImageDraw.Draw(image), data, meta, label, amount, thickness, number)
        frames.append(image)
    gif.parent.mkdir(parents=True, exist_ok=True)
    pc.save_gif(frames, gif)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        for tag, index in (
            ("raw", HOLD_RAW - 1),
            ("mid", HOLD_RAW + MORPH_IN // 2),
            ("deskewed", HOLD_RAW + MORPH_IN + HOLD_DESKEW - 1),
        ):
            frames[index].save(preview_dir / f"mid360-deskew-{tag}.png")
    info = {
        "frames": len(frames),
        "points": len(raw),
        "thickness_raw_cm": thickness[0],
        "thickness_deskewed_cm": thickness[HOLD_RAW + MORPH_IN],
        "meta": meta,
    }
    if manifest is not None:
        pc.write_json(manifest, manifest_payload(info, npz, gif, __version__))
    return info


def draw_chrome(
    draw: Any,
    data: Any,
    meta: dict[str, Any],
    label: str,
    amount: float,
    thickness: list[float],
    number: int,
) -> None:
    draw.rectangle([0, 0, pc.WIDTH, 43], fill=pc.PANEL)
    draw.line([0, 43, pc.WIDTH, 43], fill=pc.RULE)
    pc.text(
        draw,
        (16, 22),
        "Hand-held MID360: a 0.1 s sweep, bent by rotation, straightened by the gyro",
        16,
        bold=True,
        anchor="lm",
    )
    x0 = VIEW_W + 14
    draw.rectangle([x0 - 8, 44, pc.WIDTH, pc.HEIGHT], fill=pc.PANEL)
    draw.line([x0 - 8, 44, x0 - 8, pc.HEIGHT], fill=pc.RULE)
    deskewed = amount >= 0.999
    colour = pc.GOOD if deskewed else (pc.BAD if amount <= 0.001 else pc.TEXT)
    pc.text(draw, (x0, 60), label, 19, colour, bold=True)
    pc.text(draw, (x0, 96), "angular rate over the sweep", 11, pc.MUTED)
    pc.text(draw, (x0, 112), f"{meta['sweep_mean_rate_deg_s']:.0f} deg/s", 26, pc.TEXT, bold=True)
    pc.text(draw, (x0, 154), "sweep", 11, pc.MUTED)
    pc.text(
        draw,
        (x0, 170),
        f"{meta['sweep_duration_s'] * 1000:.0f} ms, {meta['sweep_rotation_deg']:.0f} deg turned",
        15, pc.TEXT, bold=True,
    )  # fmt: skip
    pc.text(draw, (x0, 206), "local surface thickness (12-NN, <15 m)", 11, pc.MUTED)
    pc.text(draw, (x0, 222), f"{thickness[number]:.2f} cm", 26, colour, bold=True)
    # IMU trace with the sweep shaded
    left, right, top, bottom = x0, 944, 292, 392
    t = data["imu_t_s"]
    rate = data["imu_rate_deg_s"]
    peak = float(max(rate.max(), 1.0)) * 1.1
    px = lambda s: left + (right - left) * (s - t[0]) / (t[-1] - t[0])  # noqa: E731
    draw.rectangle([left, top, right, bottom], outline=pc.RULE)
    draw.rectangle([px(0.0), top + 1, px(SWEEP_S), bottom - 1], fill=(40, 52, 70))
    draw.line(
        [(px(s), bottom - (bottom - top) * float(r) / peak) for s, r in zip(t, rate, strict=True)],
        fill=pc.ORANGE, width=2,
    )  # fmt: skip
    pc.text(draw, (left, top - 14), "/livox/imu |gyro| (deg/s), sweep shaded", 11, pc.MUTED)
    pc.text(draw, (left, bottom + 6), f"{t[0]:+.1f} s", 10, pc.MUTED)
    pc.text(draw, (right, bottom + 6), f"{t[-1]:+.1f} s", 10, pc.MUTED, anchor="ra")
    for index, line in enumerate(
        (
            "RTK-SLAM construction_seq1 (CC BY 4.0)",
            f"t = {meta['scan_seconds_from_imu_start']:.1f} s; fastest sweep of the run.",
            "Deskew: calibrex gyro_rotation_model with its",
            "own imu-lidar estimate (rotation, bias, offset).",
            "Rotation only; heights colour the points.",
        )
    ):
        pc.text(draw, (x0, 428 + 14 * index), line, 10, pc.MUTED)


def manifest_payload(info: dict[str, Any], npz: Path, gif: Path, version: str) -> dict[str, Any]:
    meta = info["meta"]

    def rel(path: Path) -> str:
        return pc.rel_path(path, ROOT)

    return {
        "schema": "calibrex.readme_pointcloud_gif/v1",
        "output": rel(gif),
        "sha256": pc.file_sha256(gif),
        "size_bytes": gif.stat().st_size,
        "animation": {
            "width": pc.WIDTH,
            "height": pc.HEIGHT,
            "fps": pc.FPS,
            "frames": info["frames"],
        },
        "generator": "tools/mid360_deskew_gif.py",
        "calibrex_version": version,
        "calibrex_commit": pc.git_head(ROOT),
        "commands": [
            "python tools/mid360_deskew_gif.py extract --bag <construction_seq1>",
            "python tools/mid360_deskew_gif.py render",
        ],
        "dataset": {
            "name": "RTK-SLAM",
            "sequence": "construction_seq1 (spent for claims; used here for visuals only)",
            "license": "CC BY 4.0 (https://huggingface.co/datasets/Willyzw/rtk-slam-dataset)",
            "bag_digest_sha256": meta["bag_digest_sha256"],
            "bag_digest_scope": meta["bag_digest_scope"],
            "topics": ["/livox/points", "/livox/imu"],
        },
        "inputs": {
            "intermediate_npz": {"path": rel(npz), "sha256": pc.file_sha256(npz)},
            "imu_lidar_estimate_artifact": {
                "path": meta["artifact"],
                "sha256": meta["artifact_sha256"],
            },
            "calib_yaml_sha256": meta["calib_yaml_sha256"],
            "scan_header_s": meta["scan_header_s"],
            "extract_commit": meta["calibrex_commit_extract"],
        },
        "deskew": {
            "model": "calibrex.solvers.imu_lidar_rotation_solver.gyro_rotation_model",
            "applied_with": "calibrex.solvers.scan_to_scan_odometry.deskew_points_with_rotations",
            "translation": "none (rotation-only)",
            "time_offset_s": meta["time_offset_s"],
            "gyro_bias_rps": meta["gyro_bias_rps"],
        },
        "metrics": {
            "sweep_mean_rate_deg_s": meta["sweep_mean_rate_deg_s"],
            "sweep_rotation_deg": meta["sweep_rotation_deg"],
            "thickness_raw_cm": info["thickness_raw_cm"],
            "thickness_deskewed_cm": info["thickness_deskewed_cm"],
        },
        "caveats": [
            "A single sweep: sparse (about 10k points within 25 m) and with no ground truth; the "
            "thickness figure is a local-planarity proxy, not an accuracy measure.",
            "Rotation-only deskew: the translation during the 0.1 s sweep is ignored.",
            "The deskew uses the imu-lidar estimate for this same sequence (not held out); the "
            "GIF illustrates the correction, it is not evidence of calibration accuracy.",
            "The morph between raw and deskewed positions is a linear blend for display.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MID360 gyro-deskew README GIF")
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("extract")
    ex.add_argument("--bag", type=Path, default=DEFAULT_BAG)
    ex.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    ex.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    ex.add_argument("--output", type=Path, default=DEFAULT_NPZ)
    rd = sub.add_parser("render")
    rd.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    rd.add_argument("--output", type=Path, default=DEFAULT_GIF)
    rd.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    rd.add_argument("--preview-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.command == "extract":
        extract(args.bag, args.artifact, args.calib, args.output)
    else:
        print(render(args.npz, args.output, args.preview_dir, args.manifest))


if __name__ == "__main__":
    main()
