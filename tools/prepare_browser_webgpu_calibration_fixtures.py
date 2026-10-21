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


def prepare(output: Path) -> None:
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
    for source in sources:
        (output / (source.name + ".txt")).write_bytes(source.read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    prepare(parser.parse_args().out)
