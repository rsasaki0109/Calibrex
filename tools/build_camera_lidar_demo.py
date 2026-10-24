"""Prepare a real KITTI camera/LiDAR projection playground with byte-bound inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets/camera-lidar-live"
OUTPUT = ROOT / "docs/app/samples"
FRAMES = (60, 90)


def digest(path: Path) -> str:
    """SHA-256 of a source or display asset."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calibration() -> tuple[np.ndarray, np.ndarray]:
    """Map Velodyne into rectified camera 2, including P_rect_02's translation."""
    values: dict[str, np.ndarray] = {}
    for name in ("calib_cam_to_cam.txt", "calib_velo_to_cam.txt"):
        for line in (ASSETS / "source" / name).read_text().splitlines():
            key, _, rest = line.partition(":")
            try:
                values[key] = np.array([float(v) for v in rest.split()])
            except ValueError:
                continue
    projection = values["P_rect_02"].reshape(3, 4)
    intrinsic = projection[:, :3]
    transform = np.eye(4)
    rect = values["R_rect_00"].reshape(3, 3)
    transform[:3, :3] = rect @ values["R"].reshape(3, 3)
    transform[:3, 3] = rect @ values["T"] + np.linalg.solve(intrinsic, projection[:, 3])
    return intrinsic, transform


def camera_points(points: np.ndarray) -> np.ndarray:
    """Apply KITTI's supplied camera-2 transform to real Velodyne returns."""
    _, transform = calibration()
    return points[:, :3] @ transform[:3, :3].T + transform[:3, 3]


def project(points: np.ndarray, offsets: list[float]) -> np.ndarray:
    """Pixels and camera Z after optical-axis XYZ rotation and XYZ translation."""
    intrinsic, _ = calibration()
    rotation = Rotation.from_euler("xyz", offsets[:3], degrees=True).as_matrix()
    moved = points @ rotation.T + np.asarray(offsets[3:])
    uvw = moved @ intrinsic.T
    return np.c_[uvw[:, :2] / uvw[:, 2, None], moved[:, 2]]


def prepare(source: Path) -> None:
    """Preserve original photographs/calibration and deterministic real-return subsets."""
    target = ASSETS / "source"
    target.mkdir(parents=True, exist_ok=True)
    receipt = json.loads((source / "download-receipt.json").read_text())
    for name, entry in receipt.items():
        if digest(source / name) != entry["sha256"]:
            raise ValueError(f"Input hash mismatch: {name}")
        if not name.endswith(".bin"):
            shutil.copyfile(source / name, target / name)
    intrinsic, _ = calibration()
    for frame in FRAMES:
        name = f"{frame:010d}"
        raw = np.fromfile(source / f"{name}.bin", dtype="<f4").reshape(-1, 4)
        cam = camera_points(raw)
        uvw = cam @ intrinsic.T
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = uvw[:, :2] / uvw[:, 2, None]
        keep = (cam[:, 2] > 3) & (cam[:, 2] < 60) & (raw[:, 2] > -1.6)
        keep &= (uv[:, 0] > -220) & (uv[:, 0] < 1462)
        keep &= (uv[:, 1] > -160) & (uv[:, 1] < 535)
        indices = np.flatnonzero(keep)
        indices = indices[np.linspace(0, len(indices)-1, min(6000, len(indices))).astype(int)]
        np.savez_compressed(target / f"points_{frame:04d}.npz",
                            velodyne=raw[indices], source_indices=indices)
    (ASSETS / "download-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


def build_frame(frame: int) -> dict[str, Any]:
    """Build one reproducible browser asset plus cross-language projection references."""
    source = ASSETS / "source"
    points = camera_points(np.load(source / f"points_{frame:04d}.npz")["velodyne"])
    points = np.round(points, 6)
    intrinsic, _ = calibration()
    with Image.open(source / f"{frame:010d}.png") as image:
        width, height = image.size
    offsets = [[0., 0., 0., 0., 0., 0.], [1., 3., -2., .1, -.05, .2]]
    return {
        "schema": "calibrex.camera_lidar_projection_demo/v1",
        "capture_kind": "real", "frame": frame, "image_size": [width, height],
        "intrinsics": [float(intrinsic[0, 0]), float(intrinsic[1, 1]),
                       float(intrinsic[0, 2]), float(intrinsic[1, 2])],
        "points": points.tolist(),
        "reference_projections": [
            {"offsets": offset, "uvz": project(points[:8], offset).tolist()}
            for offset in offsets
        ],
    }


def build() -> None:
    """Write lazy frame assets and original/derived source hashes."""
    sources = [*sorted((ASSETS / "source").iterdir()), ASSETS / "download-receipt.json",
               Path(__file__)]
    index: dict[str, Any] = {
        "schema": "calibrex.camera_lidar_projection_index/v1",
        "capture_kind": "real", "dataset": "KITTI raw / 2011_09_26_drive_0005_sync",
        "license": "CC BY-NC-SA 3.0", "frames": {},
        "limitation": "Manual projection experiment, not an optimizer or a new check verdict.",
        "provenance": {str(p.relative_to(ROOT)): digest(p) for p in sources},
    }
    for frame in FRAMES:
        data = build_frame(frame)
        with Image.open(ASSETS / "source" / f"{frame:010d}.png") as image:
            image.convert("RGB").save(ASSETS / f"frame_{frame:04d}.jpg", quality=90)
        path = OUTPUT / f"camera_lidar_{frame:04d}.json"
        path.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
        index["frames"][str(frame)] = {"url": path.name, "sha256": digest(path),
                                       "image_sha256": digest(ASSETS / f"frame_{frame:04d}.jpg")}
    (OUTPUT / "camera_lidar_demo.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps(index["frames"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-from", type=Path)
    args = parser.parse_args()
    if args.prepare_from:
        prepare(args.prepare_from)
    build()
