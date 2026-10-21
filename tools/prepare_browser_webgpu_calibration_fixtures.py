"""Raw synthetic sensor fixtures for full-estimator development; no downloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def prepare(output: Path, *, include_stationary_control: bool = False) -> None:
    """Render distinct synchronized scenes and preserve their generator lineage."""

    if output.exists():
        raise ValueError("output already exists; preserve earlier fixtures")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    # Reuse the repository's established independent ray-cast test scene. This
    # is a developer tool run from the checkout, not an installed-library API.
    from tests.unit.camera_lidar_scene import (
        CX,
        CY,
        FOCAL,
        cast_points,
        make_scene,
        perturbed,
        render_image,
        true_transform,
    )

    output.mkdir(parents=True)
    sources = [Path(__file__).resolve(), root / "tests/unit/camera_lidar_scene.py"]
    lineage = {
        "schema_version": "calibrex.browser_webgpu_calibration_fixture/v0.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "generator": "tools.prepare_browser_webgpu_calibration_fixtures",
        "sources": {
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "accuracy_claim": False,
        "origin": {
            "kind": "synthetic_ray_cast_rectangles",
            "seeds": list(range(48)),
            "description": "Distinct rendered layouts; no repeated frames or accuracy claim.",
        },
    }
    truth = true_transform()
    raw = [(render_image(scene := make_scene(i)), cast_points(scene, truth)) for i in range(48)]
    for count, translation in [(2, False), (12, False), (12, True), (48, False)]:
        record = {
            **lineage,
            "frames": count,
            "estimate_translation": translation,
            "origin": {**lineage["origin"], "seeds": list(range(count))},
        }
        payload: dict[str, Any] = {
            "frames": count,
            "intrinsics": np.array([FOCAL, FOCAL, CX, CY]),
            "candidate": perturbed(truth, (0.8, -1.2, 0.0)),
            "estimate_translation": translation,
            "provenance_json": json.dumps(record, sort_keys=True),
        }
        for i, (image, points) in enumerate(raw[:count]):
            payload[f"image_{i}"] = image
            payload[f"points_{i}"] = points
        name = f"synthetic-raw-{count}-{'six-dof' if translation else 'rotation'}.npz"
        np.savez_compressed(output / name, **payload)
    if include_stationary_control:
        record = {
            **lineage,
            "frames": 48,
            "estimate_translation": False,
            "origin": {
                "kind": "synthetic_stationary_scan_noise_control",
                "seeds": list(range(8)),
                "noise_seed": 10429,
                "scans_per_layout": 6,
                "description": (
                    "Eight rendered layouts with six distinct image/cloud observations each. "
                    "Low gray/range noise within a layout exercises repeated-view rejection; "
                    "48 distinct raw inputs, no real-data accuracy claim."
                ),
            },
        }
        payload = {
            "frames": 48,
            "intrinsics": np.array([FOCAL, FOCAL, CX, CY]),
            "candidate": perturbed(truth, (0.8, -1.2, 0.0)),
            "estimate_translation": False,
            "provenance_json": json.dumps(record, sort_keys=True),
        }
        rng = np.random.default_rng(10429)
        image_hashes, point_hashes = set(), set()
        for layout in range(8):
            for scan in range(6):
                index = 6 * layout + scan
                image, points = raw[layout][0].copy(), raw[layout][1].copy()
                if scan:
                    image = np.clip(
                        image.astype(np.int16)
                        + rng.integers(-1, 2, size=image.shape, dtype=np.int16),
                        0,
                        255,
                    ).astype(np.uint8)
                    points = np.asarray(points * np.float32(1.0 + scan * 0.0001), dtype=np.float32)
                image_hashes.add(hashlib.sha256(image.tobytes()).hexdigest())
                point_hashes.add(hashlib.sha256(points.tobytes()).hexdigest())
                payload[f"image_{index}"] = image
                payload[f"points_{index}"] = points
        assert len(image_hashes) == len(point_hashes) == 48
        np.savez_compressed(output / "synthetic-raw-48-stationary-control.npz", **payload)
    for source in sources:
        (output / (source.name + ".txt")).write_bytes(source.read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stationary-control", action="store_true")
    args = parser.parse_args()
    prepare(args.out, include_stationary_control=args.stationary_control)
