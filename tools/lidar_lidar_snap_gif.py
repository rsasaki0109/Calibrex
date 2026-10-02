"""README GIF: a perturbed LiDAR-LiDAR extrinsic snaps onto calibrex's estimate (NTU VIRAL tnp_01).

Two stages, so the GIF is reproducible without the 20 GB bag:

``extract``  (needs the bag and calibrex) runs calibrex's own reference odometry and
             map code over a short slow-moving window of the development recording
             ``tnp_01`` and stores a small preprocessed subset as ``.npz``.
``render``   (numpy, scipy, Pillow, calibrex) reads that subset and draws the GIF.

Cyan is LiDAR 1 (the reference map: scans placed by calibrex's odometry).  Orange is
LiDAR 2, transformed by an extrinsic that moves along SE(3) (slerp + lerp) from a
deliberately perturbed start to the estimate in ``docs/assets/ntu_viral_lidar_lidar/
tnp_01.yaml``.  The live residual is the median absolute point-to-plane distance from
calibrex's ``MapSample.residuals`` for the shown transform.
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
DEFAULT_BAG = Path("/media/sasaki/aiueo2/datasets/ntu_viral_release/tnp_01_rosbag2")
DEFAULT_ARTIFACT = ROOT / "docs/assets/ntu_viral_lidar_lidar/tnp_01.yaml"
DEFAULT_NPZ = ROOT / "docs/assets/lidar-lidar-snap/tnp_01_window.npz"
DEFAULT_GIF = ROOT / "docs/assets/lidar-lidar-snap.gif"
DEFAULT_MANIFEST = ROOT / "docs/assets/lidar-lidar-snap/manifest.json"
REFERENCE_TOPIC = "/os1_cloud_node1/points"
TARGET_TOPIC = "/os1_cloud_node2/points"
# Scan indices (10 Hz) of the window: odometry starts at FIRST_SCAN; every
# SAMPLE_STRIDE-th scan from SAMPLE_FIRST to SAMPLE_LAST is a target sample, each
# registered to the 11 reference scans around it, like calibrex's own samples.
FIRST_SCAN = 570
LAST_SCAN = 650
SAMPLE_FIRST = 580
SAMPLE_LAST = 640
SAMPLE_STRIDE = 10
MAP_HALF_WINDOW = 5
# Deliberately wrong start: left-perturbation about the reference LiDAR's axes
# (like calibrex's known-bad controls), about 5.6 deg and 0.31 m in total.
START_ROTVEC_DEG = (3.5, -3.0, 3.5)
START_TRANSLATION_M = (0.22, -0.18, 0.12)
FRAMES = 60
# Phases (frames): hold the start, snap, hold the estimate, relax back.
HOLD_START = 8
SNAP = 20
HOLD_END = 14
RELAX = FRAMES - HOLD_START - SNAP - HOLD_END
DESIGN_NOTE = "design value differs from the estimate by ~5 cm in y and z (docs/benchmarks)"


def read_transform(artifact: Path) -> FloatArray:
    import yaml
    from scipy.spatial.transform import Rotation

    data = yaml.safe_load(artifact.read_text(encoding="utf-8"))["transform"]
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(data["rotation_quat_xyzw"]).as_matrix()
    transform[:3, 3] = data["translation_m"]
    return transform


def design_transform() -> FloatArray:
    """``inv(T_body_horz) @ T_body_vert`` from the NTU VIRAL ``T_Body2Lidar`` design files."""

    horz = np.eye(4)
    horz[:3, 3] = (-0.050, 0.000, 0.055)
    vert = np.array(
        [
            [-1.0, 0.0, 0.0, -0.550],
            [0.0, 0.0, 1.0, 0.030],
            [0.0, 1.0, 0.0, 0.050],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return np.asarray(np.linalg.inv(horz) @ vert)


def perturbed_start(estimate: FloatArray) -> FloatArray:
    from scipy.spatial.transform import Rotation

    rotation = Rotation.from_rotvec(np.radians(START_ROTVEC_DEG)).as_matrix()
    start = estimate.copy()
    start[:3, :3] = rotation @ estimate[:3, :3]
    start[:3, 3] = rotation @ estimate[:3, 3] + np.asarray(START_TRANSLATION_M)
    return start


def interpolate(a: FloatArray, b: FloatArray, alpha: float) -> FloatArray:
    """SE(3) interpolation: spherical interpolation of rotation, linear of translation."""

    from scipy.spatial.transform import Rotation, Slerp

    slerp = Slerp([0.0, 1.0], Rotation.from_matrix([a[:3, :3], b[:3, :3]]))
    out = np.eye(4)
    out[:3, :3] = slerp([alpha]).as_matrix()[0]
    out[:3, 3] = (1.0 - alpha) * a[:3, 3] + alpha * b[:3, 3]
    return out


def offset_to(estimate: FloatArray, transform: FloatArray) -> tuple[float, float]:
    """Rotation angle (deg) and translation distance (m) of ``transform`` from ``estimate``."""

    from scipy.spatial.transform import Rotation

    delta = Rotation.from_matrix(estimate[:3, :3].T @ transform[:3, :3]).magnitude()
    return float(np.degrees(delta)), float(np.linalg.norm(transform[:3, 3] - estimate[:3, 3]))


def extract(bag: Path, artifact: Path, out: Path) -> None:
    """Run calibrex's reference odometry and keep a small subset of the window."""

    import json

    from calibrex import __version__
    from calibrex.data.livox_ros2 import LivoxStreamProfile, bag_input_digest, iter_livox_points
    from calibrex.evaluation.lidar_lidar_map import LidarLidarMapOptions
    from calibrex.solvers.scan_to_scan_odometry import IncrementalScanOdometry, preprocess_scan

    options = LidarLidarMapOptions()

    def profile(topic: str) -> LivoxStreamProfile:
        return LivoxStreamProfile("lidar", topic, "", "t", "offset_s", "mps2")

    odometry = IncrementalScanOdometry(options.odometry)
    scans: dict[int, FloatArray] = {}
    times: dict[int, float] = {}
    stream = iter_livox_points(bag, profile(REFERENCE_TOPIC), max_seconds=LAST_SCAN / 10.0 + 5.0)
    for index, (time_s, points, offsets) in enumerate(stream):
        if index < FIRST_SCAN:
            continue
        if index > LAST_SCAN:
            break
        finite = np.isfinite(points).all(axis=1)
        odometry.add(points[finite], time_s, offsets[finite] if offsets is not None else None)
        pose = odometry.poses[-1]
        local = preprocess_scan(points[finite], options.preprocess)
        scans[index] = local @ pose[:3, :3].T + pose[:3, 3]
        times[index] = time_s
    poses = {FIRST_SCAN + k: pose for k, pose in enumerate(odometry.poses)}
    targets: dict[int, tuple[float, FloatArray]] = {}
    for index, (time_s, points, _) in enumerate(
        iter_livox_points(bag, profile(TARGET_TOPIC), max_seconds=LAST_SCAN / 10.0 + 5.0)
    ):
        if index > LAST_SCAN:
            break
        if index in range(SAMPLE_FIRST, SAMPLE_LAST + 1, SAMPLE_STRIDE):
            kept = preprocess_scan(points[np.isfinite(points).all(axis=1)], options.preprocess)
            targets[index] = (time_s, kept)
    sample_ids = sorted(targets)
    unbounded = type(options.preprocess)(
        voxel_size_m=options.preprocess.voxel_size_m, min_range_m=0.0, max_range_m=1.0e9
    )
    window = range(SAMPLE_FIRST - MAP_HALF_WINDOW, SAMPLE_LAST + MAP_HALF_WINDOW + 1)
    map_points = preprocess_scan(np.vstack([scans[i] for i in window]), unbounded)
    digest, scope = bag_input_digest([bag])
    meta: dict[str, Any] = {
        "bag": bag.name,
        "bag_digest_sha256": digest,
        "bag_digest_scope": scope,
        "artifact": artifact.name,
        "artifact_sha256": pc.file_sha256(artifact),
        "calibrex_version_extract": __version__,
        "calibrex_commit_extract": pc.git_head(ROOT),
        "reference_topic": REFERENCE_TOPIC,
        "target_topic": TARGET_TOPIC,
        "window_scans": [FIRST_SCAN, LAST_SCAN],
        "sample_scans": sample_ids,
        "voxel_size_m": options.preprocess.voxel_size_m,
        "reference_time_first_s": times[min(times)],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        map_points=map_points.astype(np.float32),
        sample_poses=np.stack([poses[i] for i in sample_ids]),
        sample_times_s=np.asarray([targets[i][0] for i in sample_ids]),
        target_counts=np.asarray([len(targets[i][1]) for i in sample_ids]),
        target_points=np.vstack([targets[i][1] for i in sample_ids]).astype(np.float32),
        meta=np.asarray(json.dumps(meta, sort_keys=True)),
    )
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB); map {len(map_points)} points")


class Window:
    """The stored subset, with calibrex ``MapSample`` objects sharing one map index."""

    def __init__(self, npz: Path) -> None:
        import json

        from calibrex.solvers.lidar_lidar_map_solver import MapSample, MapSampleOptions

        data = np.load(npz)
        self.meta: dict[str, Any] = json.loads(str(data["meta"]))
        self.map_points = data["map_points"].astype(np.float64)
        self.poses = data["sample_poses"]
        counts = data["target_counts"]
        parts = np.split(data["target_points"].astype(np.float64), np.cumsum(counts)[:-1])
        self.targets = parts
        self.options = MapSampleOptions()
        self.samples: list[MapSample] = []
        for index, points in enumerate(parts):
            sample = MapSample(
                float(data["sample_times_s"][index]), 0, self.poses[index], points, self.map_points
            )
            if self.samples:
                first = self.samples[0]
                sample._tree, sample._normals, sample._planar = (
                    first._tree,
                    first._normals,
                    first._planar,
                )
            sample.prepare(self.options)
            self.samples.append(sample)

    def median_residual_m(self, extrinsic: FloatArray) -> float:
        parts = [np.abs(s.residuals(extrinsic, self.options)[0]) for s in self.samples]
        merged = np.concatenate(parts)
        return float(np.median(merged)) if len(merged) else float("nan")

    def target_world(self, extrinsic: FloatArray) -> FloatArray:
        out = []
        for pose, points in zip(self.poses, self.targets, strict=True):
            transform = pose @ extrinsic
            out.append(points @ transform[:3, :3].T + transform[:3, 3])
        return np.vstack(out)

    def map_ground(self) -> NDArray[np.bool_]:
        """Display mask: map points on near-horizontal surfaces (from the map's own normals)."""

        normals = self.samples[0]._normals
        assert normals is not None
        return np.asarray(np.abs(normals[:, 2]) > GROUND_NORMAL_Z)

    def target_ground(self, extrinsic: FloatArray) -> NDArray[np.bool_]:
        """Display mask for LiDAR 2, from its own 10-neighbour normals (no map involved)."""

        from scipy.spatial import cKDTree

        world = self.target_world(extrinsic)
        _, index = cKDTree(world).query(world, k=10)
        around = world[index]
        centred = around - around.mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", centred, centred)
        normal_z = np.abs(np.linalg.eigh(covariance)[1][:, 2, 0])
        return np.asarray(normal_z > GROUND_NORMAL_Z)


def schedule() -> list[tuple[float, str]]:
    """Per-frame ``(alpha from start to estimate, phase label)`` for a seamless loop."""

    out: list[tuple[float, str]] = []
    out += [(0.0, "start: perturbed")] * HOLD_START
    out += [(pc.smoothstep((k + 1) / SNAP), "calibrating") for k in range(SNAP)]
    out += [(1.0, "calibrex estimate")] * HOLD_END
    out += [(1.0 - pc.smoothstep((k + 1) / RELAX), "relaxing") for k in range(RELAX)]
    return out


def render(
    npz: Path, artifact: Path, gif: Path, preview_dir: Path | None, manifest: Path | None = None
) -> dict[str, Any]:
    from PIL import Image, ImageDraw

    window = Window(npz)
    estimate = read_transform(artifact)
    start = perturbed_start(estimate)
    design = design_transform()
    plan = schedule()
    residuals = [window.median_residual_m(interpolate(start, estimate, alpha)) for alpha, _ in plan]
    design_residual = window.median_residual_m(design)
    start_residual = residuals[0]
    end_residual = residuals[HOLD_START + SNAP]
    show_map = ~window.map_ground()
    at_estimate = window.target_world(estimate)
    show_target = ~window.target_ground(estimate) & (
        np.linalg.norm(at_estimate[:, :2], axis=1) < 40
    )
    centre = np.median(at_estimate[show_target], axis=0)
    view_w, view_h = 640, pc.HEIGHT - 44
    frames: list[Image.Image] = []
    for number, (alpha, label) in enumerate(plan):
        transform = interpolate(start, estimate, alpha)
        camera = pc.orbit_camera(
            centre, CAMERA_AZIMUTH, CAMERA_ELEVATION, CAMERA_DISTANCE, 560.0, view_w, view_h
        )
        canvas = np.empty((pc.HEIGHT, pc.WIDTH, 3), dtype=np.uint8)
        canvas[:] = pc.BG
        canvas[44:, :view_w] = (10, 13, 18)
        pc.splat(
            canvas,
            camera,
            [
                (window.map_points[show_map], pc.CYAN, 2),
                (window.target_world(transform)[show_target], pc.ORANGE, 2),
            ],
            origin=(0, 44),
        )
        image = Image.fromarray(canvas)
        draw = ImageDraw.Draw(image)
        draw_chrome(
            draw, window, label, residuals, number, transform, estimate,
            design_residual, start_residual, end_residual,
        )  # fmt: skip
        frames.append(image)
    gif.parent.mkdir(parents=True, exist_ok=True)
    pc.save_gif(frames, gif)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        for tag, index in (
            ("start", HOLD_START - 1),
            ("mid", HOLD_START + SNAP // 2),
            ("snapped", HOLD_START + SNAP + HOLD_END - 1),
        ):
            frames[index].save(preview_dir / f"lidar-lidar-snap-{tag}.png")
    from calibrex import __version__

    info = {
        "frames": len(frames),
        "start_residual_m": start_residual,
        "estimate_residual_m": end_residual,
        "design_residual_m": design_residual,
        "start_offset": offset_to(estimate, start),
        "design_offset": offset_to(estimate, design),
        "meta": window.meta,
        "map_points": len(window.map_points),
        "target_points": int(sum(len(t) for t in window.targets)),
    }
    if manifest is not None:
        pc.write_json(manifest, manifest_payload(info, npz, artifact, gif, __version__))
    return info


def manifest_payload(
    info: dict[str, Any], npz: Path, artifact: Path, gif: Path, version: str
) -> dict[str, Any]:
    """Provenance record: inputs with digests, versions, commands, licences, caveats."""

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
        "generator": "tools/lidar_lidar_snap_gif.py",
        "calibrex_version": version,
        "calibrex_commit": pc.git_head(ROOT),
        "commands": [
            "python tools/lidar_lidar_snap_gif.py extract --bag <tnp_01_rosbag2>",
            "python tools/lidar_lidar_snap_gif.py render",
        ],
        "dataset": {
            "name": "NTU VIRAL",
            "recording": "tnp_01 (development recording)",
            "license": "CC BY-NC-SA 4.0",
            "bag_digest_sha256": meta["bag_digest_sha256"],
            "bag_digest_scope": meta["bag_digest_scope"],
            "topics": [meta["reference_topic"], meta["target_topic"]],
        },
        "inputs": {
            "intermediate_npz": {"path": rel(npz), "sha256": pc.file_sha256(npz)},
            "extrinsic_artifact": {
                "path": rel(artifact),
                "sha256": pc.file_sha256(artifact),
                "note": "calibrex lidar-lidar ros2 on the full recording (see its provenance)",
            },
            "window_scans_10hz": meta["window_scans"],
            "sample_scans": meta["sample_scans"],
            "extract_commit": meta["calibrex_commit_extract"],
        },
        "animation_parameters": {
            "start": "estimate left-perturbed by rotvec deg "
            f"{list(START_ROTVEC_DEG)} and {list(START_TRANSLATION_M)} m",
            "interpolation": "SE(3): slerp rotation, lerp translation, smoothstep easing",
            "residual": "median |point-to-plane| from calibrex MapSample.residuals, 7 samples",
        },
        "metrics": {
            "start_residual_m": info["start_residual_m"],
            "estimate_residual_m": info["estimate_residual_m"],
            "design_residual_m": info["design_residual_m"],
            "start_offset_deg_m": info["start_offset"],
            "design_offset_deg_m": info["design_offset"],
        },
        "caveats": [
            "Real data, but the 7 target scans use one shared 0.2 m-voxel map of 71 reference "
            "scans, not calibrex's per-sample 11-scan maps.",
            "Ground returns are hidden for display only; the residual uses all points.",
            "The start offset is synthetic and labelled as such; the estimate is from the full "
            "recording, not from this window.",
            "The design value is a rounded datasheet number, not a measurement.",
        ],
    }


# Fixed camera (a still camera keeps the GIF small: only the orange cloud changes).
CAMERA_AZIMUTH = 120.0
CAMERA_ELEVATION = 22.0
CAMERA_DISTANCE = 15.0
GROUND_NORMAL_Z = 0.8


def draw_chrome(
    draw: Any,
    window: Window,
    label: str,
    residuals: list[float],
    number: int,
    transform: FloatArray,
    estimate: FloatArray,
    design_residual: float,
    start_residual: float,
    end_residual: float,
) -> None:
    """Title bar, legend, live numbers, and the residual trace."""

    draw.rectangle([0, 0, pc.WIDTH, 43], fill=pc.PANEL)
    draw.line([0, 43, pc.WIDTH, 43], fill=pc.RULE)
    pc.text(
        draw,
        (16, 22),
        "Two LiDARs, one rigid body: the extrinsic snaps into focus",
        17,
        bold=True,
        anchor="lm",
    )
    pc.text(draw, (944, 22), "NTU VIRAL tnp_01 / 2 x Ouster OS1-16", 12, pc.MUTED, anchor="rm")
    x0 = 656
    draw.rectangle([x0 - 8, 44, pc.WIDTH, pc.HEIGHT], fill=pc.PANEL)
    draw.line([x0 - 8, 44, x0 - 8, pc.HEIGHT], fill=pc.RULE)
    # legend
    draw.ellipse([x0, 62, x0 + 10, 72], fill=pc.CYAN)
    pc.text(draw, (x0 + 18, 67), "LiDAR 1: reference map (odometry)", 12, anchor="lm")
    draw.ellipse([x0, 84, x0 + 10, 94], fill=pc.ORANGE)
    pc.text(draw, (x0 + 18, 89), "LiDAR 2: 7 scans through X", 12, anchor="lm")
    # state
    snapped = label == "calibrex estimate"
    colour = pc.GOOD if snapped else (pc.BAD if label.startswith("start") else pc.TEXT)
    pc.text(draw, (x0, 122), label, 19, colour, bold=True)
    rot_deg, trans_m = offset_to(estimate, transform)
    now = residuals[number]
    pc.text(draw, (x0, 160), "median point-to-plane residual", 11, pc.MUTED)
    pc.text(draw, (x0, 176), f"{now * 100:5.1f} cm", 30, colour, bold=True)
    pc.text(draw, (x0, 224), "offset from calibrex estimate", 11, pc.MUTED)
    pc.text(
        draw, (x0, 240), f"{rot_deg:4.1f} deg   {trans_m * 100:4.1f} cm", 18, pc.TEXT, bold=True
    )
    # trace
    left, right, top, bottom = x0, 944, 300, 430
    draw.rectangle([left, top, right, bottom], outline=pc.RULE)
    peak = max(residuals) * 1.1
    pts = [
        (left + (right - left) * k / (len(residuals) - 1), bottom - (bottom - top) * r / peak)
        for k, r in enumerate(residuals)
    ]
    draw.line(pts, fill=pc.RULE, width=2)
    draw.line(pts[: number + 1], fill=pc.ORANGE, width=2)
    dx, dy = pts[number]
    draw.ellipse([dx - 4, dy - 4, dx + 4, dy + 4], fill=colour)
    design_y = bottom - (bottom - top) * design_residual / peak
    draw.line([left, design_y, right, design_y], fill=pc.MUTED, width=1)
    pc.text(
        draw,
        (right - 4, design_y - 2),
        f"design value {design_residual * 100:.1f} cm",
        10,
        pc.MUTED,
        anchor="rb",
    )
    pc.text(draw, (left, top - 14), "residual over the animation", 11, pc.MUTED)
    pc.text(draw, (left, bottom + 8), "same 7 target scans, same map; only X changes", 10, pc.MUTED)
    pc.text(
        draw, (left, 458), "Ground returns hidden for display; residual uses all.", 10, pc.MUTED
    )
    pc.text(draw, (left, 472), "Estimate: calibrex lidar-lidar ros2 on full tnp_01.", 10, pc.MUTED)
    pc.text(draw, (left, 486), "Residual: calibrex MapSample, 0.2 m voxels.", 10, pc.MUTED)
    pc.text(draw, (left, 506), "NTU VIRAL data: CC BY-NC-SA 4.0. Dev recording;", 10, pc.MUTED)
    pc.text(draw, (left, 520), "residual floor ~5 cm (16 beams, odometry map).", 10, pc.MUTED)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("extract")
    ex.add_argument("--bag", type=Path, default=DEFAULT_BAG)
    ex.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    ex.add_argument("--output", type=Path, default=DEFAULT_NPZ)
    rd = sub.add_parser("render")
    rd.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    rd.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    rd.add_argument("--output", type=Path, default=DEFAULT_GIF)
    rd.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    rd.add_argument("--preview-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.command == "extract":
        extract(args.bag, args.artifact, args.output)
    else:
        info = render(args.npz, args.artifact, args.output, args.preview_dir, args.manifest)
        print(info)


if __name__ == "__main__":
    main()
