"""Export the committed NTU VIRAL subset for the browser's before/after sample.

This packages two saved transforms, not optimizer iterations. The full-data
residuals come from the existing digest-bound README manifest. Display points
are filtered and downsampled, so the browser never recomputes these metrics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from lidar_lidar_snap_gif import (
    DEFAULT_ARTIFACT,
    DEFAULT_MANIFEST,
    DEFAULT_NPZ,
    Window,
    perturbed_start,
    read_transform,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/app/samples/alignment_demo.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sampled(points: Any, limit: int) -> list[float]:
    kept = points[np.linspace(0, len(points) - 1, min(limit, len(points)), dtype=int)]
    return np.round(kept, 3).reshape(-1).tolist()


def build() -> dict[str, Any]:
    manifest = json.loads(DEFAULT_MANIFEST.read_text())
    for path, key in [(DEFAULT_NPZ, "intermediate_npz"), (DEFAULT_ARTIFACT, "extrinsic_artifact")]:
        if digest(path) != manifest["inputs"][key]["sha256"]:
            raise ValueError(f"Source digest mismatch: {path}")
    window = Window(DEFAULT_NPZ)
    estimate = read_transform(DEFAULT_ARTIFACT)
    before = window.target_world(perturbed_start(estimate))
    after = window.target_world(estimate)
    keep = ~window.target_ground(estimate) & (np.linalg.norm(after[:, :2], axis=1) < 40)
    ref = window.map_points[~window.map_ground()]
    return {
        "schema": "calibrex.browser_alignment_demo/v1",
        "dataset": manifest["dataset"],
        "metrics": manifest["metrics"],
        "camera": {
            "target": np.median(after[keep], axis=0).tolist(),
            "azimuth_deg": 120,
            "elevation_deg": 22,
            "distance_m": 15,
        },
        "reference": sampled(ref, 18000),
        # Identical sample indices on both sides preserve the same point correspondences.
        "before": sampled(before[keep], 12000),
        "after": sampled(after[keep], 12000),
        "provenance": {
            "generator": "tools/build_alignment_demo.py",
            "generator_sha256": digest(Path(__file__)),
            "sources": [
                {"path": str(p.relative_to(ROOT)), "sha256": digest(p)}
                for p in [DEFAULT_NPZ, DEFAULT_ARTIFACT, DEFAULT_MANIFEST]
            ],
            "display": (
                "Ground hidden; deterministic point subset rounded to 1 mm. "
                "Metrics use the full saved subset, not display points."
            ),
            "comparison": (
                "Deliberately perturbed start versus a saved Calibrex estimate from the full "
                "recording. This is not a live solve or optimizer history."
            ),
            "caveats": manifest["caveats"],
        },
    }


if __name__ == "__main__":
    payload = build()
    OUTPUT.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"Wrote {OUTPUT.relative_to(ROOT)} ({OUTPUT.stat().st_size:,} bytes)")
    print({k: len(payload[k]) // 3 for k in ["reference", "before", "after"]})
