#!/usr/bin/env python3
"""Generate the README fixed 3D LiDAR-to-LiDAR calibration evidence GIF.

The default visual uses real public Livox Horizon-Horizon PCD frames from the
official Livox automatic calibration example. It does not redistribute the
upstream archives and does not claim benchmark accuracy. The animation is an
evidence-viewer proxy for how Calibrex compares a fixed 3D LiDAR-to-LiDAR
candidate against a selected reference.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import shutil
import struct
import subprocess
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

WIDTH = 960
HEIGHT = 540
FPS = 12
FRAME_COUNT = 36

A2D2_SENSOR_CONFIG_URL = (
    "https://aev-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/"
    "cams_lidars.json"
)
A2D2_LIDAR_SAMPLE_URL = (
    "https://aev-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/"
    "camera_lidar-20180810150607_lidar_frontleft.tar"
)
A2D2_LIDAR_SAMPLE_NAME = "20180810150607_lidar_front_left_000000060.npz"
A2D2_LIDAR_SAMPLE_START = 1536
A2D2_LIDAR_SAMPLE_SIZE = 2_977_425
LIVOX_BASE_PCD_URL = (
    "https://terra-1-g.djicdn.com/65c028cd298f4669a7f0e40e50ba1131/"
    "Showcase/Base_LiDAR_Frames.tar.gz"
)
LIVOX_TARGET_PCD_URL = (
    "https://terra-1-g.djicdn.com/65c028cd298f4669a7f0e40e50ba1131/"
    "Showcase/Target-LiDAR-Frames.tar.gz"
)
LIVOX_BASE_SAMPLE_NAME = "base_horizon_100432.pcd"
LIVOX_TARGET_SAMPLE_NAME = "target_horizon_100538.pcd"

SCENE_PANEL = (28, 86, 594, 426)
RIGHT_PANEL = (650, 92, 278, 414)
CHART = (676, 266, 216, 68)

Color = tuple[int, int, int]
Point3 = tuple[float, float, float]
Point2 = tuple[int, int]

BG = (10, 15, 27)
PANEL = (18, 27, 43)
PANEL_ALT = (13, 20, 34)
GRID = (55, 68, 90)
TEXT_DIM = (148, 163, 184)
REFERENCE = (52, 211, 153)
CANDIDATE = (251, 113, 133)
OPTIMIZED = (34, 211, 238)
WARNING = (245, 158, 11)
GOOD = (34, 197, 94)
VEHICLE = (42, 52, 68)
ROAD = (27, 38, 55)


@dataclass(frozen=True)
class LidarPose:
    name: str
    origin: Point3


@dataclass(frozen=True)
class LidarCloudPair:
    source_points: list[Point3]
    target_points: list[Point3]
    source_label: str
    target_label: str
    source_pose_name: str
    target_pose_name: str
    source_total: int
    target_total: int
    source_path: Path
    subtitle: str
    scene_caption: str
    legend: str
    provenance: str


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["livox-horizon-horizon", "a2d2"],
        default="livox-horizon-horizon",
        help="Public data source used to generate the evidence animation.",
    )
    parser.add_argument(
        "--sensor-config",
        type=Path,
        help="Optional local A2D2 cams_lidars.json. If omitted, the public URL is tried.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Directory for the extracted public data sample.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/calibration-evidence-demo.gif"),
    )
    parser.add_argument("--frames", type=int, default=FRAME_COUNT)
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="Use the built-in fixed multi-LiDAR fallback instead of fetching A2D2 metadata.",
    )
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required to generate the GIF")

    if args.source == "livox-horizon-horizon":
        data_dir = args.data_dir or Path("data/public/livox_horizon_horizon_pair")
        ensure_livox_horizon_pair(data_dir, allow_network=not args.no_network)
        cloud_pair = load_livox_horizon_cloud_pair(data_dir)
        lidars, metadata_source = livox_horizon_setup(), "Livox official PCD sample"
    else:
        data_dir = args.data_dir or Path("data/public/a2d2_lidar_pair")
        ensure_a2d2_lidar_sample(data_dir, allow_network=not args.no_network)
        cloud_pair = load_a2d2_lidar_cloud_pair(data_dir / A2D2_LIDAR_SAMPLE_NAME)
        lidars, metadata_source = load_a2d2_lidar_setup(
            args.sensor_config,
            allow_network=not args.no_network,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="calibrex_lidar_lidar_gif_") as tmp_name:
        tmp = Path(tmp_name)
        residual_history: list[float] = []
        for index in range(args.frames):
            progress = smoothstep(index / max(1, args.frames - 1))
            residual_history.append(residual_proxy(progress))
            image = bytearray(bytes(BG) * (WIDTH * HEIGHT))
            draw_frame(
                image=image,
                lidars=lidars,
                cloud_pair=cloud_pair,
                progress=progress,
                residual_history=residual_history,
                metadata_source=metadata_source,
            )
            write_ppm(tmp / f"frame_{index:03d}.ppm", image)
        encode_gif(tmp, args.output, cloud_pair)
    return 0


def ensure_livox_horizon_pair(data_dir: Path, *, allow_network: bool) -> None:
    """Ensure two real Livox Horizon PCD frames are present locally."""

    base_path = data_dir / LIVOX_BASE_SAMPLE_NAME
    target_path = data_dir / LIVOX_TARGET_SAMPLE_NAME
    if base_path.exists() and target_path.exists():
        return
    if not allow_network:
        raise SystemExit(
            f"{base_path} and {target_path} are required. Run without --no-network "
            "or fetch the Livox Horizon-Horizon sample first."
        )
    data_dir.mkdir(parents=True, exist_ok=True)
    extract_first_pcd_from_tar_gz(LIVOX_BASE_PCD_URL, base_path)
    extract_first_pcd_from_tar_gz(LIVOX_TARGET_PCD_URL, target_path)


def extract_first_pcd_from_tar_gz(url: str, output: Path) -> None:
    """Stream a public tar.gz and write its first PCD member."""

    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=90) as response, tarfile.open(
        fileobj=response,
        mode="r|gz",
    ) as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".pcd"):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            output.write_bytes(handle.read())
            return
    raise SystemExit(f"{url} did not contain a PCD file")


def ensure_a2d2_lidar_sample(data_dir: Path, *, allow_network: bool) -> None:
    """Ensure a small real A2D2 LiDAR NPZ sample is present locally."""

    sample_path = data_dir / A2D2_LIDAR_SAMPLE_NAME
    if sample_path.exists() and sample_path.stat().st_size == A2D2_LIDAR_SAMPLE_SIZE:
        return
    if not allow_network:
        raise SystemExit(
            f"{sample_path} is required. Run without --no-network or fetch the A2D2 "
            "sample with tools/download_public_dataset.py a2d2_lidar_pair_sample."
        )
    data_dir.mkdir(parents=True, exist_ok=True)
    start = A2D2_LIDAR_SAMPLE_START
    end = start + A2D2_LIDAR_SAMPLE_SIZE - 1
    request = Request(A2D2_LIDAR_SAMPLE_URL, headers={"Range": f"bytes={start}-{end}"})
    with urlopen(request, timeout=90) as response:
        data = response.read()
    if len(data) != A2D2_LIDAR_SAMPLE_SIZE:
        raise SystemExit(
            f"downloaded {len(data)} bytes from A2D2, expected {A2D2_LIDAR_SAMPLE_SIZE}"
        )
    sample_path.write_bytes(data)


def load_livox_horizon_cloud_pair(data_dir: Path) -> LidarCloudPair:
    """Load two solid-state Livox Horizon PCD frames from public example data."""

    base_path = data_dir / LIVOX_BASE_SAMPLE_NAME
    target_path = data_dir / LIVOX_TARGET_SAMPLE_NAME
    source_points, source_total = read_binary_pcd_xyz(base_path, stride=8)
    target_points, target_total = read_binary_pcd_xyz(target_path, stride=7)
    if not source_points or not target_points:
        raise SystemExit(f"{data_dir} does not contain usable Livox Horizon PCD points")
    return LidarCloudPair(
        source_points=source_points[:2400],
        target_points=target_points[:2400],
        source_label="base Horizon points",
        target_label="target Horizon points",
        source_pose_name="base_horizon",
        target_pose_name="target_horizon",
        source_total=source_total,
        target_total=target_total,
        source_path=base_path,
        subtitle="Livox official Horizon-Horizon public PCD sample",
        scene_caption="Real solid-state Livox points: reference vs candidate extrinsic",
        legend="green/cyan: base/target Horizon returns   pink: perturbed candidate",
        provenance="provenance: Livox official Horizon-Horizon PCD",
    )


def read_binary_pcd_xyz(path: Path, *, stride: int) -> tuple[list[Point3], int]:
    """Read x/y/z from a binary float32 PCD file with no third-party deps."""

    data = path.read_bytes()
    header_end = data.index(b"\n", data.index(b"DATA binary")) + 1
    header = data[:header_end].decode("ascii", errors="strict")
    fields: list[str] = []
    sizes: list[int] = []
    types: list[str] = []
    counts: list[int] = []
    point_count = 0
    for line_text in header.splitlines():
        parts = line_text.split()
        if not parts:
            continue
        key = parts[0]
        if key == "FIELDS":
            fields = parts[1:]
        elif key == "SIZE":
            sizes = [int(value) for value in parts[1:]]
        elif key == "TYPE":
            types = parts[1:]
        elif key == "COUNT":
            counts = [int(value) for value in parts[1:]]
        elif key == "POINTS":
            point_count = int(parts[1])
    if not fields or not sizes or not types:
        raise SystemExit(f"{path} has incomplete PCD metadata")
    if not counts:
        counts = [1] * len(fields)
    if fields[:3] != ["x", "y", "z"]:
        raise SystemExit(f"{path} must store x/y/z as the first fields")
    if sizes[:3] != [4, 4, 4] or types[:3] != ["F", "F", "F"]:
        raise SystemExit(f"{path} must store x/y/z as float32 fields")
    point_step = sum(size * count for size, count in zip(sizes, counts, strict=True))
    if point_step <= 0:
        raise SystemExit(f"{path} has invalid PCD point step")
    available = (len(data) - header_end) // point_step
    point_count = min(point_count or available, available)
    points: list[Point3] = []
    total = 0
    for index in range(point_count):
        x, y, z = struct.unpack_from("<fff", data, header_end + index * point_step)
        if not (1.0 <= x <= 62.0 and -22.0 <= y <= 22.0 and -10.0 <= z <= 6.0):
            continue
        total += 1
        if total % stride == 0:
            points.append((float(x), float(y), float(z)))
    return points, total


def load_a2d2_lidar_cloud_pair(path: Path) -> LidarCloudPair:
    """Load two physical LiDAR point sets from a real A2D2 NPZ file."""

    with zipfile.ZipFile(path) as archive:
        points_header, points_raw = read_npy_array(archive, "pcloud_points.npy")
        _ids_header, lidar_ids_raw = read_npy_array(archive, "pcloud_attr.lidar_id.npy")
        _valid_header, valid_raw = read_npy_array(archive, "pcloud_attr.valid.npy")

    shape = points_header["shape"]
    if shape[1] != 3:
        raise SystemExit(f"{path} pcloud_points must be Nx3")

    source_points: list[Point3] = []
    target_points: list[Point3] = []
    source_total = 0
    target_total = 0
    for index, (lidar_id, valid) in enumerate(zip(lidar_ids_raw, valid_raw, strict=True)):
        if not valid:
            continue
        x = float(points_raw[index * 3])
        y = float(points_raw[index * 3 + 1])
        z = float(points_raw[index * 3 + 2])
        if not (2.0 <= x <= 55.0 and -22.0 <= y <= 22.0 and -3.2 <= z <= 7.5):
            continue
        point = (x, y, z)
        if lidar_id == 0:
            source_total += 1
            if source_total % 5 == 0:
                source_points.append(point)
        elif lidar_id == 1:
            target_total += 1
            if target_total % 4 == 0:
                target_points.append(point)

    if not source_points or not target_points:
        raise SystemExit(f"{path} does not contain usable lidar_id 0/1 point sets")

    return LidarCloudPair(
        source_points=source_points[:1800],
        target_points=target_points[:1800],
        source_label="lidar_id 0 points",
        target_label="lidar_id 1 points",
        source_pose_name="front_left",
        target_pose_name="front_right",
        source_total=source_total,
        target_total=target_total,
        source_path=path,
        subtitle="A2D2 public NPZ point cloud split by physical lidar_id",
        scene_caption="Real A2D2 LiDAR points: reference vs candidate extrinsic",
        legend="green: lidar_id 0/1 real returns   cyan/pink: perturbed candidate",
        provenance="provenance: A2D2 real NPZ range sample",
    )


def read_npy_array(
    archive: zipfile.ZipFile,
    name: str,
) -> tuple[dict[str, object], tuple[object, ...]]:
    """Read simple C-order numeric NPY arrays from an NPZ archive."""

    data = archive.read(name)
    if data[:6] != b"\x93NUMPY":
        raise SystemExit(f"{name} is not an NPY array")
    offset = 6
    version = data[offset : offset + 2]
    offset += 2
    if version == b"\x01\x00":
        header_length = struct.unpack("<H", data[offset : offset + 2])[0]
        offset += 2
    else:
        header_length = struct.unpack("<I", data[offset : offset + 4])[0]
        offset += 4
    header = ast.literal_eval(data[offset : offset + header_length].decode("latin1").strip())
    offset += header_length
    if header.get("fortran_order") is not False:
        raise SystemExit(f"{name} uses unsupported Fortran order")
    shape = header.get("shape")
    if not isinstance(shape, tuple):
        raise SystemExit(f"{name} has invalid shape metadata")
    count = 1
    for dimension in shape:
        count *= int(dimension)
    dtype = header.get("descr")
    type_map = {"<f8": "d", "<i8": "q", "|b1": "?"}
    if dtype not in type_map:
        raise SystemExit(f"{name} uses unsupported dtype {dtype}")
    values = struct.unpack_from("<" + type_map[dtype] * count, data, offset)
    return header, values


def load_a2d2_lidar_setup(
    sensor_config: Path | None,
    *,
    allow_network: bool,
) -> tuple[list[LidarPose], str]:
    """Load A2D2 LiDAR origins or return a deterministic fixed-rig fallback."""

    if sensor_config is not None:
        payload = json.loads(sensor_config.read_text(encoding="utf-8"))
        return parse_lidars(payload), f"A2D2 metadata: {sensor_config}"

    if allow_network:
        try:
            with urlopen(A2D2_SENSOR_CONFIG_URL, timeout=12) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return parse_lidars(payload), "A2D2 public cams_lidars.json"
        except (OSError, URLError, json.JSONDecodeError):
            pass

    return fallback_lidars(), "fixed multi-LiDAR fallback"


def livox_horizon_setup() -> list[LidarPose]:
    """Return a compact fixed rig for the Livox Horizon-Horizon example."""

    return [
        LidarPose("base_horizon", (1.58, -0.18, 1.18)),
        LidarPose("target_horizon", (1.66, 0.32, 1.12)),
    ]


def parse_lidars(payload: dict[str, object]) -> list[LidarPose]:
    raw_lidars = payload.get("lidars")
    if not isinstance(raw_lidars, dict):
        return fallback_lidars()

    lidars: list[LidarPose] = []
    for name, raw_lidar in raw_lidars.items():
        if not isinstance(raw_lidar, dict):
            continue
        raw_view = raw_lidar.get("view")
        if not isinstance(raw_view, dict):
            continue
        raw_origin = raw_view.get("origin")
        if not isinstance(raw_origin, list) or len(raw_origin) != 3:
            continue
        try:
            origin = tuple(float(value) for value in raw_origin)
        except (TypeError, ValueError):
            continue
        lidars.append(LidarPose(str(name), origin))  # type: ignore[arg-type]
    return sorted(lidars, key=lambda lidar: lidar.name) or fallback_lidars()


def fallback_lidars() -> list[LidarPose]:
    """Return fixed vehicle 3D LiDAR poses matching the A2D2 setup shape."""

    return [
        LidarPose("front_center", (1.72, 0.00, 1.12)),
        LidarPose("front_left", (1.71, 0.63, 1.12)),
        LidarPose("front_right", (1.71, -0.65, 1.12)),
        LidarPose("rear_left", (-0.43, 0.59, 1.15)),
        LidarPose("rear_right", (-0.43, -0.61, 1.12)),
    ]


def draw_frame(
    *,
    image: bytearray,
    lidars: list[LidarPose],
    cloud_pair: LidarCloudPair,
    progress: float,
    residual_history: list[float],
    metadata_source: str,
) -> None:
    fill_rect(image, 0, 0, WIDTH, HEIGHT, BG)
    fill_rect(image, SCENE_PANEL[0], SCENE_PANEL[1], SCENE_PANEL[2], SCENE_PANEL[3], PANEL)
    fill_rect(image, RIGHT_PANEL[0], RIGHT_PANEL[1], RIGHT_PANEL[2], RIGHT_PANEL[3], PANEL)
    rect(image, SCENE_PANEL[0], SCENE_PANEL[1], SCENE_PANEL[2], SCENE_PANEL[3], GRID, alpha=0.85)
    rect(
        image,
        RIGHT_PANEL[0],
        RIGHT_PANEL[1],
        RIGHT_PANEL[2],
        RIGHT_PANEL[3],
        GRID,
        alpha=0.85,
    )

    draw_calibration_scene(image, lidars, cloud_pair, progress)
    draw_evidence_panel(image, cloud_pair, progress, residual_history, metadata_source)
    draw_timeline(image, progress)


def draw_calibration_scene(
    image: bytearray,
    lidars: list[LidarPose],
    cloud_pair: LidarCloudPair,
    progress: float,
) -> None:
    draw_road_grid(image)
    draw_real_lidar_clouds(image, cloud_pair, progress)
    draw_vehicle_box(image)

    source = lidar_by_name(lidars, cloud_pair.source_pose_name)
    target = lidar_by_name(lidars, cloud_pair.target_pose_name)
    current_target = animated_candidate_pose(target, progress)

    for lidar in lidars:
        color = REFERENCE if lidar.name != target.name else mix(CANDIDATE, OPTIMIZED, progress)
        draw_lidar_sensor(
            image,
            lidar.origin,
            color,
            strong=lidar.name in {source.name, target.name},
        )
    draw_lidar_sensor(
        image,
        current_target.origin,
        mix(CANDIDATE, OPTIMIZED, progress),
        strong=True,
    )

    draw_transform_arrow(image, source.origin, target.origin, REFERENCE, alpha=0.60)
    draw_transform_arrow(
        image,
        source.origin,
        current_target.origin,
        mix(CANDIDATE, OPTIMIZED, progress),
    )
    draw_delta_vector(image, target.origin, current_target.origin, progress)


def lidar_by_name(lidars: list[LidarPose], name: str) -> LidarPose:
    for lidar in lidars:
        if lidar.name == name:
            return lidar
    return lidars[0]


def animated_candidate_pose(reference: LidarPose, progress: float) -> LidarPose:
    remaining = 1.0 - progress
    offset = (
        0.22 * remaining,
        -0.30 * remaining,
        0.13 * remaining,
    )
    return LidarPose(
        f"{reference.name}_candidate",
        (
            reference.origin[0] + offset[0],
            reference.origin[1] + offset[1],
            reference.origin[2] + offset[2],
        ),
    )


def draw_road_grid(image: bytearray) -> None:
    fill_rect(image, 48, 372, 540, 116, ROAD, alpha=0.84)
    for y in [-4.0, -2.0, 0.0, 2.0, 4.0]:
        left = project((-3.0, y, -0.05))
        right = project((7.0, y, -0.05))
        line(image, left[0], left[1], right[0], right[1], GRID, alpha=0.38)
    for x in [-2.0, 0.0, 2.0, 4.0, 6.0]:
        bottom = project((x, -4.5, -0.05))
        top = project((x, 4.5, -0.05))
        line(image, bottom[0], bottom[1], top[0], top[1], GRID, alpha=0.38)


def draw_real_lidar_clouds(
    image: bytearray,
    cloud_pair: LidarCloudPair,
    progress: float,
) -> None:
    remaining = 1.0 - progress
    color = mix(CANDIDATE, OPTIMIZED, progress)
    for index, point in enumerate(cloud_pair.source_points):
        px, py = project_lidar_point(point)
        circle(image, px, py, 1, REFERENCE if index % 3 else (110, 231, 183), alpha=0.58)
    for index, point in enumerate(cloud_pair.target_points):
        px, py = project_lidar_point(point)
        circle(image, px, py, 1, REFERENCE, alpha=0.23)
        if index % 2:
            continue
        shifted = (
            point[0] + 1.8 * remaining,
            point[1] - 1.2 * remaining,
            point[2] + 0.45 * remaining,
        )
        x, y = project_lidar_point(shifted)
        circle(image, x, y, 1, color, alpha=0.72)


def scene_points() -> list[Point3]:
    points: list[Point3] = []
    for x_index in range(34):
        x = -2.2 + x_index * 0.26
        for lane_y in (-1.8, 1.8):
            wave = math.sin(x * 1.7) * 0.08
            points.append((x, lane_y + wave, 0.0))
    for side_y in (-3.2, 3.2):
        for x_index in range(24):
            x = -1.4 + x_index * 0.32
            for z_index in range(4):
                z = 0.35 + z_index * 0.44
                points.append((x, side_y, z))
    for angle_index in range(60):
        angle = angle_index * math.tau / 60
        radius = 1.35 + 0.16 * math.sin(angle * 3.0)
        points.append((3.8 + radius * math.cos(angle), radius * math.sin(angle), 0.18))
    return points


def draw_vehicle_box(image: bytearray) -> None:
    bottom = [(-1.0, -1.25, 0.0), (4.0, -1.25, 0.0), (4.0, 1.25, 0.0), (-1.0, 1.25, 0.0)]
    roof = [(-0.6, -1.0, 1.35), (2.6, -1.0, 1.35), (2.6, 1.0, 1.35), (-0.6, 1.0, 1.35)]
    bottom_points = [project(point) for point in bottom]
    roof_points = [project(point) for point in roof]
    fill_polygon(image, bottom_points, (30, 41, 58), alpha=0.86)
    fill_polygon(image, roof_points, VEHICLE, alpha=0.92)
    for index in range(4):
        next_index = (index + 1) % 4
        line_between(image, bottom_points[index], bottom_points[next_index], (100, 116, 139), 0.70)
        line_between(image, roof_points[index], roof_points[next_index], (185, 196, 214), 0.82)
        line_between(image, bottom_points[index], roof_points[index], (100, 116, 139), 0.48)


def draw_lidar_sensor(image: bytearray, origin: Point3, color: Color, *, strong: bool) -> None:
    x, y = project(origin)
    radius = 6 if strong else 4
    circle(image, x, y, radius + 3, (5, 9, 16), alpha=0.56)
    circle(image, x, y, radius, color, alpha=1.0)
    endpoints = (
        (origin[0] + 0.55, origin[1], origin[2]),
        (origin[0], origin[1] + 0.42, origin[2]),
    )
    for endpoint in endpoints:
        ex, ey = project(endpoint)
        line(image, x, y, ex, ey, color, alpha=0.68)


def draw_transform_arrow(
    image: bytearray,
    source: Point3,
    target: Point3,
    color: Color,
    *,
    alpha: float = 0.95,
) -> None:
    sx, sy = project((source[0], source[1], source[2] + 0.10))
    tx, ty = project((target[0], target[1], target[2] + 0.10))
    thick_line(image, sx, sy, tx, ty, color, thickness=2, alpha=alpha)
    circle(image, tx, ty, 5, color, alpha=alpha)


def draw_delta_vector(
    image: bytearray,
    reference: Point3,
    candidate: Point3,
    progress: float,
) -> None:
    if progress > 0.97:
        return
    rx, ry = project((reference[0], reference[1], reference[2] + 0.28))
    cx, cy = project((candidate[0], candidate[1], candidate[2] + 0.28))
    line(image, rx, ry, cx, cy, WARNING, alpha=0.88)
    circle(image, rx, ry, 3, REFERENCE, alpha=0.95)
    circle(image, cx, cy, 3, WARNING, alpha=0.95)


def draw_evidence_panel(
    image: bytearray,
    cloud_pair: LidarCloudPair,
    progress: float,
    residual_history: list[float],
    metadata_source: str,
) -> None:
    fill_rect(image, 674, 132, 220, 76, PANEL_ALT)
    rect(image, 674, 132, 220, 76, GRID, alpha=0.92)
    metric_bar(
        image,
        674,
        156,
        1.0 - residual_proxy(progress) / 0.105,
        mix(WARNING, GOOD, progress),
    )
    metric_bar(image, 674, 180, min(1.0, 0.40 + progress * 0.58), OPTIMIZED)

    chart_x, chart_y, chart_width, chart_height = CHART
    fill_rect(image, chart_x, chart_y, chart_width, chart_height, PANEL_ALT)
    rect(image, chart_x, chart_y, chart_width, chart_height, GRID, alpha=0.95)
    for line_index in range(1, 4):
        y = chart_y + line_index * chart_height // 4
        line(image, chart_x, y, chart_x + chart_width, y, GRID, alpha=0.45)
    draw_curve(image, residual_history, CHART, OPTIMIZED)

    draw_dof_cells(image, 674, 378, progress)
    draw_source_badge(image, metadata_source)
    draw_sample_count_ticks(image, cloud_pair)


def draw_source_badge(image: bytearray, metadata_source: str) -> None:
    color = GOOD if metadata_source.startswith(("A2D2", "Livox")) else WARNING
    fill_rect(image, 674, 464, 220, 16, PANEL_ALT)
    metric_bar(image, 674, 464, 1.0, color)


def draw_sample_count_ticks(image: bytearray, cloud_pair: LidarCloudPair) -> None:
    source_ratio = min(1.0, cloud_pair.source_total / 12_000)
    target_ratio = min(1.0, cloud_pair.target_total / 10_000)
    fill_rect(image, 674, 486, 220, 8, (31, 41, 55), alpha=1.0)
    fill_rect(image, 674, 486, round(220 * source_ratio), 8, REFERENCE, alpha=0.85)
    fill_rect(image, 674, 496, 220, 8, (31, 41, 55), alpha=1.0)
    fill_rect(image, 674, 496, round(220 * target_ratio), 8, OPTIMIZED, alpha=0.85)


def draw_curve(
    image: bytearray,
    values: list[float],
    chart: tuple[int, int, int, int],
    color: Color,
) -> None:
    if len(values) < 2:
        return
    x, y, width, height = chart
    min_value = 0.020
    max_value = 0.110
    points: list[Point2] = []
    for index, value in enumerate(values):
        px = x + round(index * width / max(1, len(values) - 1))
        ratio = (value - min_value) / (max_value - min_value)
        py = y + height - round(max(0.0, min(1.0, ratio)) * height)
        points.append((px, py))
    for index in range(1, len(points)):
        line_between(image, points[index - 1], points[index], color, 0.95, thickness=2)
    circle(image, points[-1][0], points[-1][1], 4, color, alpha=1.0)


def metric_bar(image: bytearray, x: int, y: int, value: float, color: Color) -> None:
    fill_rect(image, x, y, 220, 8, (31, 41, 55), alpha=1.0)
    fill_rect(image, x, y, round(220 * max(0.0, min(1.0, value))), 8, color, alpha=0.95)


def draw_dof_cells(image: bytearray, x: int, y: int, progress: float) -> None:
    for index in range(6):
        cell_x = x + index * 35
        good = index in {0, 1, 3, 4} or progress > 0.82
        color = GOOD if good else WARNING
        fill_rect(image, cell_x, y, 24, 22, color, alpha=0.90)
        rect(image, cell_x, y, 24, 22, (229, 231, 235), alpha=0.24)


def draw_timeline(image: bytearray, progress: float) -> None:
    x0, y, width = 82, 516, 786
    fill_rect(image, x0, y, width, 8, (31, 41, 55), alpha=1.0)
    fill_rect(image, x0, y, round(width * progress), 8, OPTIMIZED, alpha=0.95)
    for marker in [0.0, 0.33, 0.66, 1.0]:
        x = x0 + round(width * marker)
        circle(image, x, y + 4, 8, OPTIMIZED if progress >= marker else TEXT_DIM, alpha=1.0)


def residual_proxy(progress: float) -> float:
    remaining = 1.0 - progress
    return 0.024 + 0.080 * remaining * remaining


def project(point: Point3) -> Point2:
    x, y, z = point
    return (
        round(318 + x * 61 - y * 72),
        round(410 + x * 17 + y * 21 - z * 92),
    )


def project_lidar_point(point: Point3) -> Point2:
    x, y, z = point
    return (
        round(292 + x * 2.6 + y * 7.0),
        round(430 - x * 2.0 - z * 15.0),
    )


def smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def mix(left: Color, right: Color, ratio: float) -> Color:
    return tuple(
        round(left[index] + (right[index] - left[index]) * max(0.0, min(1.0, ratio)))
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


def fill_polygon(image: bytearray, points: list[Point2], color: Color, *, alpha: float) -> None:
    if not points:
        return
    min_y = max(0, min(y for _x, y in points))
    max_y = min(HEIGHT - 1, max(y for _x, y in points))
    for y in range(min_y, max_y + 1):
        intersections: list[int] = []
        for index, start in enumerate(points):
            end = points[(index + 1) % len(points)]
            if (start[1] <= y < end[1]) or (end[1] <= y < start[1]):
                ratio = (y - start[1]) / (end[1] - start[1])
                intersections.append(round(start[0] + ratio * (end[0] - start[0])))
        intersections.sort()
        for index in range(0, len(intersections), 2):
            if index + 1 >= len(intersections):
                continue
            for x in range(max(0, intersections[index]), min(WIDTH, intersections[index + 1])):
                pixel(image, x, y, color, alpha)


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


def line_between(
    image: bytearray,
    left: Point2,
    right: Point2,
    color: Color,
    alpha: float,
    *,
    thickness: int = 1,
) -> None:
    if thickness <= 1:
        line(image, left[0], left[1], right[0], right[1], color, alpha=alpha)
    else:
        thick_line(
            image,
            left[0],
            left[1],
            right[0],
            right[1],
            color,
            thickness=thickness,
            alpha=alpha,
        )


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


def encode_gif(frame_dir: Path, output: Path, cloud_pair: LidarCloudPair) -> None:
    palette = frame_dir / "palette.png"
    text_filter = build_text_filter(cloud_pair)
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


def build_text_filter(cloud_pair: LidarCloudPair) -> str:
    font_file = subprocess.check_output(
        ["fc-match", "-f", "%{file}", "Noto Sans"],
        text=True,
    ).strip()
    labels = [
        ("Fixed 3D LiDAR to fixed 3D LiDAR", 34, 22, 26, "E5E7EB"),
        (cloud_pair.subtitle, 34, 55, 16, "CBD5E1"),
        (cloud_pair.scene_caption, 50, 104, 17, "E5E7EB"),
        (
            cloud_pair.legend,
            50,
            486,
            14,
            "CBD5E1",
        ),
        ("Evidence", 674, 104, 17, "E5E7EB"),
        ("holdout point-to-plane", 674, 138, 13, "CBD5E1"),
        ("overlap consistency", 674, 162, 13, "CBD5E1"),
        ("residual history", 676, 242, 16, "E5E7EB"),
        ("DoF visibility", 674, 354, 16, "E5E7EB"),
        ("x  y  z  r  p  yaw", 674, 406, 13, "CBD5E1"),
        (cloud_pair.provenance, 674, 444, 13, "CBD5E1"),
        (cloud_pair.source_label, 674, 486, 12, "CBD5E1"),
        (cloud_pair.target_label, 674, 496, 12, "CBD5E1"),
        ("public setup", 62, 520, 14, "CBD5E1"),
        ("candidate", 325, 520, 14, "CBD5E1"),
        ("optimize", 588, 520, 14, "CBD5E1"),
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
