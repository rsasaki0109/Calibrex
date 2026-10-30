"""Initializer-only geometry boundary; no model, reference problem or external loader."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class FileRef(Strict):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Request(Strict):
    schema_version: Literal["calibrex.cofii2p_input/v0.1"]
    frame_id: str
    image: FileRef
    cloud: FileRef
    initializer: FileRef
    camera: FileRef
    source_crop_top_rows: int = Field(ge=0)


class Initializer(Strict):
    schema_version: Literal["calibrex.explicit_initial_transform/v0.1"]
    transform_camera_lidar: list[list[float]]
    provenance: str = Field(min_length=1)


class Camera(Strict):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    intrinsic_matrix: list[list[float]]


class Prepared(Strict):
    scope: str
    frame_id: str
    source_commit: str
    driver_sha256: str
    input_sha256: dict[str, str]
    artifact: FileRef
    array_shapes: dict[str, list[int]]
    source_to_input_pixel: list[list[float]]
    initial_camera_lidar: list[list[float]]
    preprocessing: dict


def rigid(value: list[list[float]]) -> np.ndarray:
    m = np.asarray(value, dtype=float)
    if (
        m.shape != (4, 4)
        or not np.isfinite(m).all()
        or not np.allclose(m[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
        or not np.allclose(m[:3, :3].T @ m[:3, :3], np.eye(3), atol=1e-6, rtol=0)
        or not np.isclose(np.linalg.det(m[:3, :3]), 1.0, atol=1e-6, rtol=0)
    ):
        raise ValueError("proper finite rigid4x4 transform required")
    return m


def compose_correction(correction: list[list[float]], initial: list[list[float]]) -> np.ndarray:
    return rigid(correction) @ rigid(initial)


def pixel_map(
    width: int, height: int, crop_top: int
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    if crop_top >= height:
        raise ValueError("crop removes image")
    resized = (round(width * 0.5), round((height - crop_top) * 0.5))
    if resized[0] < 512 or resized[1] < 160:
        raise ValueError("source too small for512x160 crop")
    crop = ((resized[0] - 512) // 2, (resized[1] - 160) // 2)
    sx, sy = resized[0] / width, resized[1] / (height - crop_top)
    affine = np.array(
        [
            [sx, 0, (sx - 1) / 2 - crop[0]],
            [0, sy, (sy - 1) / 2 - sy * crop_top - crop[1]],
            [0, 0, 1.0],
        ]
    )
    return affine, resized, crop


def map_input_pixels_to_source(pixels: np.ndarray, affine: np.ndarray) -> np.ndarray:
    p = np.asarray(pixels, float)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
        raise ValueError("finite Nx2 pixels required")
    return (np.column_stack([p, np.ones(len(p))]) @ np.linalg.inv(affine).T)[:, :2]


def prepare(request_path: Path, output: Path) -> Prepared:
    if output.exists():
        raise ValueError("refuse overwrite")
    request = Request.model_validate_json(request_path.read_text())
    refs = [request.image, request.cloud, request.initializer, request.camera]
    hashes = {}
    for ref in refs:
        actual = hashlib.sha256(Path(ref.path).read_bytes()).hexdigest()
        if actual != ref.sha256:
            raise ValueError("input digest mismatch " + ref.path)
        hashes[ref.path] = actual
    hashes[str(request_path)] = hashlib.sha256(request_path.read_bytes()).hexdigest()
    initial = Initializer.model_validate_json(Path(request.initializer.path).read_text())
    transform = rigid(initial.transform_camera_lidar)
    camera = Camera.model_validate_json(Path(request.camera.path).read_text())
    k = np.asarray(camera.intrinsic_matrix, float)
    if (
        k.shape != (3, 3)
        or not np.isfinite(k).all()
        or k[0, 0] <= 0
        or k[1, 1] <= 0
        or not np.allclose(k[2], [0, 0, 1])
    ):
        raise ValueError("invalid pinhole intrinsics")
    image = cv2.imread(request.image.path, cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (camera.height, camera.width):
        raise ValueError("image dimensions mismatch")
    a, size, crop = pixel_map(camera.width, camera.height, request.source_crop_top_rows)
    image = cv2.resize(image[request.source_crop_top_rows :], size, interpolation=cv2.INTER_LINEAR)
    image = image[crop[1] : crop[1] + 160, crop[0] : crop[0] + 512]
    cloud = np.load(request.cloud.path, allow_pickle=False)
    if set(cloud.files) != {"point_lidar_m", "intensity", "normal_lidar", "point_id"}:
        raise ValueError("explicit four-array cloud contract required")
    xyz = np.asarray(cloud["point_lidar_m"], float)
    normal = np.asarray(cloud["normal_lidar"], float)
    intensity = np.asarray(cloud["intensity"], float)
    ids = np.asarray(cloud["point_id"])
    n = len(xyz)
    if (
        xyz.shape != (n, 3)
        or normal.shape != (n, 3)
        or intensity.shape != (n, 1)
        or ids.shape != (n,)
        or ids.dtype.kind not in "iu"
        or len(np.unique(ids)) != n
        or n < 1
    ):
        raise ValueError("invalid cloud shapes/IDs")
    if not all(np.isfinite(x).all() for x in [xyz, normal, intensity]) or not np.allclose(
        np.linalg.norm(normal, axis=1), 1, atol=1e-4
    ):
        raise ValueError("finite points/features and unit normals required")
    arrays = {
        "point_lidar_m": xyz,
        "point_initial_camera_m": xyz @ transform[:3, :3].T + transform[:3, 3],
        "normal_initial_camera": normal @ transform[:3, :3].T,
        "intensity": intensity,
        "point_id": ids,
        "image_bgr_chw": np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32) / 255.0,
        "input_intrinsics": a @ k,
    }
    output.mkdir(parents=True)
    artifact = output / "input.npz"
    np.savez_compressed(artifact, **arrays)
    report = Prepared(
        scope="geometry/preprocessing boundary only; no CoFiI2P inference or pyramid",
        frame_id=request.frame_id,
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256=hashes,
        artifact=FileRef(
            path=str(artifact), sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()
        ),
        array_shapes={key: list(value.shape) for key, value in arrays.items()},
        source_to_input_pixel=a.tolist(),
        initial_camera_lidar=transform.tolist(),
        preprocessing={
            "resize": "half dimensions rounded; OpenCV pixel-center affine",
            "crop": "center512x160",
            "source_crop_top_rows": request.source_crop_top_rows,
            "channel_order": "BGR (explicit OpenCV convention; checkpoint parity pending)",
            "point_sampling": "none; original rows/IDs preserved",
        },
    )
    (output / "manifest.json").write_text(report.model_dump_json(indent=2) + "\n")
    (output / "manifest.schema.json").write_text(
        json.dumps(Prepared.model_json_schema(), indent=2) + "\n"
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("request", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    prepare(a.request, a.output)
