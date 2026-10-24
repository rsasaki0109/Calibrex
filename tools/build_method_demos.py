"""Build method-gallery receipts and synthetic fisheye stereo/depth demonstrations."""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from generate_camera_imu_lock_gif import interpolate, interval_rmse_dps, load_series
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial.transform import Rotation

from calibrex.evaluation.fisheye_stereo import (
    EquidistantCamera,
    rectify_fisheye_pair,
    vertical_offset_samples,
)
from calibrex.evaluation.stereo_camera_models import DoubleSphereCamera, MeiCamera

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/app/samples"


def sha(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def png(values: Any) -> str:
    out = io.BytesIO()
    Image.fromarray(np.asarray(values, dtype=np.uint8)).save(out, format="PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()


def cast(rays: Any, origin: Any, texture: Any) -> tuple[Any, Any]:
    shape = rays.shape[:-1]
    rays = rays.reshape(-1, 3)
    distance = np.full(len(rays), np.inf)
    with np.errstate(invalid="ignore", divide="ignore"):
        for depth, xmin, xmax, ymin, ymax in [
            (12.0, -30.0, 30.0, -30.0, 30.0),
            (6.0, 0.2, 2.0, -1.0, 1.0),
            (3.0, -1.5, -0.2, -0.8, 0.8),
        ]:
            t = (depth - origin[2]) / rays[:, 2]
            p = origin + rays * t[:, None]
            valid = (
                (t > 0)
                & (p[:, 0] >= xmin)
                & (p[:, 0] <= xmax)
                & (p[:, 1] >= ymin)
                & (p[:, 1] <= ymax)
            )
            distance = np.where(valid & (t < distance), t, distance)
        points = origin + rays * distance[:, None]
    valid = np.isfinite(points).all(axis=1)
    coords = np.nan_to_num(points[:, :2], nan=0, posinf=0, neginf=0) * 70 + 1024
    image = ndimage.map_coordinates(texture, [coords[:, 1], coords[:, 0]], order=1, mode="wrap")
    checker = (
        np.floor(np.nan_to_num(points[:, 0], nan=0, posinf=0, neginf=0) * 4)
        + np.floor(np.nan_to_num(points[:, 1], nan=0, posinf=0, neginf=0) * 4)
    ) % 2
    image = 0.75 * image + 0.25 * (40 + 180 * checker)
    image[~valid] = 0
    return image.reshape(shape), points.reshape((*shape, 3))


def stereo_demo() -> dict[str, Any]:
    width, height = 320, 240
    cameras = {
        "kb4": EquidistantCamera(180, 180, 159.5, 119.5, (-0.02, 0.001, 0, 0), width, height),
        "mei": MeiCamera(
            300, 300, 159.5, 119.5, 0.8, (-0.01, 0.001, 0.0002, -0.0001), width, height
        ),
        "ds": DoubleSphereCamera(260, 260, 159.5, 119.5, 0.5, 0.6, width, height),
    }
    rng = np.random.default_rng(73)
    texture = ndimage.gaussian_filter(rng.normal(size=(2048, 2048)), 1.5)
    texture = np.clip(127 + texture * 230, 0, 255)
    tf = np.eye(4)
    tf[:3, :3] = Rotation.from_euler("xyz", [3, 10, 0], degrees=True).as_matrix()
    centre = np.array([0.3, 0, 0])
    tf[:3, 3] = -tf[:3, :3] @ centre
    rows, cols = np.mgrid[:height, :width].astype(float)
    models = {}
    for name, camera in cameras.items():
        left, _ = cast(camera.unproject(cols, rows), np.zeros(3), texture)
        right, _ = cast(camera.unproject(cols, rows) @ tf[:3, :3], centre, texture)
        rect = rectify_fisheye_pair(camera, camera, tf, focal_scale=180 / camera.fx)
        left_rect, right_rect = rect.rectify_left(left), rect.rectify_right(right)
        samples = vertical_offset_samples(
            [(left_rect, right_rect)],
            patches_per_pair=500,
            half_px=4,
            max_disparity_px=35,
            min_disparity_px=1,
            rows_px=3,
            min_gradient=2,
            min_zncc=0.85,
        )
        rays = np.stack(
            [(cols - rect.cx) / rect.fx, (rows - rect.cy) / rect.fx, np.ones_like(cols)], axis=-1
        )
        _, truth = cast(rays @ rect.rectified_from_left, np.zeros(3), texture)
        truth_depth = (truth @ rect.rectified_from_left.T)[..., 2]
        depth = rect.stereo.base_px_m / samples[:, 3]
        iy, ix = samples[:, 1].astype(int), samples[:, 2].astype(int)
        target = truth_depth[iy, ix]
        relative = np.abs(depth - target) / target
        valid = np.isfinite(relative)
        sparse = Image.fromarray(
            np.repeat((left_rect * 0.35).astype(np.uint8)[..., None], 3, axis=2)
        )
        draw = ImageDraw.Draw(sparse)
        for x, y, z in zip(ix, iy, depth, strict=True):
            a = float(np.clip((z - 3) / 9, 0, 1))
            draw.ellipse(
                (x - 2, y - 2, x + 2, y + 2),
                fill=(int(255 * (1 - a)), int(130 + 100 * a), int(80 + 175 * a)),
            )
        pair = np.repeat(np.hstack([left_rect, right_rect])[..., None], 3, axis=2).astype(np.uint8)
        pair[20::30, :, :] = [60, 215, 180]
        models[name] = {
            "raw": png(np.hstack([left, right])),
            "rectified": png(pair),
            "depth": png(np.asarray(sparse)),
            "matches": len(samples),
            "vertical_median_px": float(np.median(np.abs(samples[:, 4]))),
            "median_relative_depth_error": float(np.median(relative[valid])),
            "baseline_m": rect.baseline_m,
            "focal_px": rect.fx,
            "image_size": [width, height],
            "correspondences": [
                {
                    "left": [float(s[2]), float(s[1])],
                    "right": [float(s[2] - s[3]), float(s[1] + s[4])],
                    "disparity_px": float(s[3]),
                    "depth_m": float(z),
                }
                for s, z in zip(samples, depth, strict=True)
            ],
        }
    return {
        "schema": "calibrex.fisheye_stereo_demo/v1",
        "models": models,
        "scene": {"depth_planes_m": [3, 6, 12], "seed": 73, "right_from_left": tf.tolist()},
        "method": (
            "Native rectification plus ZNCC patch matches; integer horizontal disparity, "
            "Z=fB/d. No truth depths enter matching."
        ),
        "limitation": (
            "Synthetic geometry test, not real-camera accuracy. Sparse depth, "
            "integer-disparity quantization and occlusion outliers are retained."
        ),
        "provenance": {
            "generator": "tools/build_method_demos.py",
            "sources": {
                p: sha(p)
                for p in [
                    "tools/build_method_demos.py",
                    "src/calibrex/evaluation/stereo_camera_models.py",
                    "src/calibrex/evaluation/fisheye_stereo.py",
                ]
            },
        },
    }


def gallery_receipt() -> dict[str, Any]:
    series = load_series(ROOT / "docs/assets/camera_imu_lock/series.npz")
    return {
        "schema": "calibrex.method_demo_metrics/v1",
        "camera_imu": {
            "before": interval_rmse_dps(series, interpolate(series, 0)),
            "after": interval_rmse_dps(series, interpolate(series, 1)),
        },
        "sources": {
            p: sha(p)
            for p in [
                "tools/build_method_demos.py",
                "tools/generate_camera_imu_lock_gif.py",
                "docs/assets/camera_imu_lock/series.npz",
                "docs/assets/camera_imu_lock/series.provenance.json",
            ]
        },
    }


if __name__ == "__main__":
    stereo = stereo_demo()
    for code, model in stereo["models"].items():
        path = OUTPUT / f"fisheye_stereo_{code}.json"
        path.write_text(json.dumps(model, separators=(",", ":"), allow_nan=False) + "\n")
        stereo["models"][code] = {
            k: v for k, v in model.items()
            if k not in ["raw", "rectified", "depth", "correspondences"]
        }
        stereo["models"][code].update(
            {"url": path.name, "sha256": sha(str(path.relative_to(ROOT)))}
        )
    for name, data in [
        ("fisheye_stereo_demo.json", stereo),
        ("method_demo_metrics.json", gallery_receipt()),
    ]:
        path = OUTPUT / name
        path.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
        print(name, path.stat().st_size)
    print(stereo["models"])
