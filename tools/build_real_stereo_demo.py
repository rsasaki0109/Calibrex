"""Build the browser's real fisheye demo from committed OpenCV stereo photographs.

Rectification and image-only ZNCC matching use the native Calibrex implementation.
No rendered scene, ground-truth depth, or board corner labels enter matching.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from build_method_demos import png
from PIL import Image, ImageDraw

from calibrex.evaluation.fisheye_stereo import (
    EquidistantCamera,
    rectify_fisheye_pair,
    vertical_offset_samples,
)
from calibrex.evaluation.stereo_camera_models import DoubleSphereCamera, MeiCamera

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets/fisheye-stereo-real"
OUTPUT = ROOT / "docs/app/samples"
MATCH_OPTIONS = {
    "patches_per_pair": 900,
    "half_px": 5,
    "max_disparity_px": 96,
    "min_disparity_px": 3,
    "rows_px": 3,
    "min_gradient": 3,
    "min_zncc": 0.95,
}


def digest(path: Path) -> str:
    """Bind a committed source or generated receipt to its bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_frame(frame: str, model: str = "kb4") -> dict[str, Any]:
    """Rectify and match one real pair using the upstream stereo calibration."""
    calibration = json.loads((ASSETS / "calibration.json").read_text())
    fitted = (
        None if model == "kb4" else json.loads((ASSETS / f"calibration_{model}.json").read_text())
    )
    sources = json.loads((ASSETS / "sources.json").read_text())
    if model not in ("kb4", "mei", "ds"):
        raise ValueError("Unknown camera model")
    if frame not in sources["frames"]:
        raise ValueError("Unknown real stereo frame")
    width, height = (v // 2 for v in calibration["image_size"])
    cameras = []
    images = []
    for side in ("left", "right"):
        path = ASSETS / "source" / f"{side}_{frame}.jpg"
        if digest(path) != sources["files"][f"source/{path.name}"]["sha256"]:
            raise ValueError(f"Source hash mismatch: {path.name}")
        entry = calibration[side]
        # Pillow's resize maps pixel centres: u_small = (u_full + 0.5)/2 - 0.5.
        fx, fy, cx, cy = entry["intrinsics"]
        if fitted is None:
            cameras.append(
                EquidistantCamera(
                    fx / 2,
                    fy / 2,
                    (cx + 0.5) / 2 - 0.5,
                    (cy + 0.5) / 2 - 0.5,
                    tuple(entry["distortion"]),
                    width,
                    height,
                )
            )
        else:
            p = fitted["parameters_" + side]
            intrinsic = [p[0] / 2, p[1] / 2, (p[2] + 0.5) / 2 - 0.5, (p[3] + 0.5) / 2 - 0.5]
            cameras.append(
                MeiCamera(*intrinsic, p[4], tuple(p[5:]), width, height)
                if model == "mei"
                else DoubleSphereCamera(*intrinsic, p[4], p[5], width, height)
            )
        with Image.open(path) as image:
            images.append(
                np.asarray(
                    image.convert("RGB").resize(
                        (width, height),
                        Image.Resampling.LANCZOS,
                    )
                )
            )
    focal = sum(calibration["left"]["intrinsics"][:2]) / 4 * 0.9
    transform = calibration["right_from_left"] if fitted is None else fitted["right_from_left"]
    rect = rectify_fisheye_pair(
        cameras[0],
        cameras[1],
        np.asarray(transform),
        focal_scale=focal / (0.5 * (cameras[0].fx + cameras[0].fy)),
    )
    # Preserve photographed colour for display; matching sees luminance only.
    rectified = [
        np.stack([resample(image[..., channel]) for channel in range(3)], axis=-1)
        for image, resample in zip(images, (rect.rectify_left, rect.rectify_right), strict=True)
    ]
    # Equivalent BLAS/SciPy paths can put an integral intensity just below or above
    # that integer. Truncating directly changes the grayscale gradient candidate
    # pool, then every seeded random pick. Snap sub-1e-8 intensity roundoff before
    # the shared display/matching conversion; keep the native geometric maps intact.
    rectified = [np.uint8(np.clip(np.round(image, 8), 0, 255)) for image in rectified]
    gray = [
        np.asarray(Image.fromarray(np.uint8(image)).convert("L"), dtype=float)
        for image in rectified
    ]
    samples = vertical_offset_samples([(gray[0], gray[1])], **MATCH_OPTIONS)
    if len(samples) == 0:
        raise ValueError(f"No accepted matches: {frame}")
    depth = rect.stereo.base_px_m / samples[:, 3]
    sparse = Image.fromarray(np.uint8(rectified[0] * 0.45))
    draw = ImageDraw.Draw(sparse)
    for sample, z in zip(samples, depth, strict=True):
        a = float(np.clip((z - 0.3) / 2.7, 0, 1))
        x, y = sample[2], sample[1]
        draw.ellipse(
            (x - 3, y - 3, x + 3, y + 3),
            fill=(int(255 * (1 - a)), int(130 + 100 * a), int(80 + 175 * a)),
        )
    pair = np.uint8(np.hstack(rectified))
    pair[20::40] = [60, 215, 180]
    return {
        "capture_kind": "real",
        "frame": frame,
        "camera_model": model,
        "raw": png(np.hstack(images)),
        "rectified": png(pair),
        "depth": png(np.asarray(sparse)),
        "matches": len(samples),
        "vertical_median_px": float(np.median(np.abs(samples[:, 4]))),
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
        "depth_truth": None,
        "fit_excluded_reprojection_px": None
        if fitted is None
        else fitted["holdout_right_reprojection_rmse_px"],
    }


def build() -> None:
    """Write lazy-loaded frames and a byte-bound provenance index."""
    sources = json.loads((ASSETS / "sources.json").read_text())
    index: dict[str, Any] = {
        "schema": "calibrex.real_fisheye_stereo_demo/v1",
        "capture_kind": "real",
        "dataset": sources["dataset"],
        "camera_model": "kb4",
        "frames": {},
        "models": {model: list(sources["frames"]) for model in ("kb4", "mei", "ds")},
        "model_frames": {},
        "matching": MATCH_OPTIONS,
        "method": "Native rectification and image-only ZNCC. No truth depths enter matching.",
        "limitations": (
            "No independent depth truth or depth-accuracy score. Integer disparity; "
            "repetitive board patterns, occlusion and weak texture can produce outliers. "
            "No uniqueness or left-right consistency filter. Colour clipped to 0.3-3.0 m. "
            "This is a saved result, not live calibration or a held-out benchmark."
        ),
        "provenance": {
            "generator": "tools/build_real_stereo_demo.py",
            "sources": {},
        },
    }
    paths = [
        Path(__file__),
        ROOT / "tools/build_method_demos.py",
        ROOT / "src/calibrex/evaluation/fisheye_stereo.py",
        ROOT / "src/calibrex/evaluation/stereo_camera_models.py",
        ASSETS / "calibration.json",
        ASSETS / "sources.json",
        ASSETS / "calibration_mei.json",
        ASSETS / "calibration_ds.json",
        ROOT / "tools/fit_real_fisheye_models.py",
        *(ASSETS / path for path in sources["files"]),
    ]
    index["provenance"]["sources"] = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    for frame in sources["frames"]:
        data = build_frame(frame)
        path = OUTPUT / f"fisheye_real_{frame}.json"
        path.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
        index["frames"][frame] = {
            "url": path.name,
            "sha256": digest(path),
            "matches": data["matches"],
            "vertical_median_px": data["vertical_median_px"],
        }
    for model in ("mei", "ds"):
        index["model_frames"][model] = {}
        for frame in sources["frames"]:
            data = build_frame(frame, model)
            path = OUTPUT / f"fisheye_real_{model}_{frame}.json"
            path.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
            index["model_frames"][model][frame] = {
                "url": path.name,
                "sha256": digest(path),
                "matches": data["matches"],
                "vertical_median_px": data["vertical_median_px"],
            }
    (OUTPUT / "fisheye_real_demo.json").write_text(
        json.dumps(index, separators=(",", ":"), allow_nan=False) + "\n",
    )
    print(json.dumps(index["frames"], indent=2))


if __name__ == "__main__":
    build()
