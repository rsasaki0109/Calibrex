"""Exact-plane Monte Carlo coverage grid, followed by curved surface diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tools.evaluate_normal_tangent import run
from tools.known_geometry_joint import CAMERA
from tools.normal_tangent_uncertainty import cone_radius
from tools.normal_uncertainty import cone_radius as old_radius


def validate(plan_path: Path, output: Path) -> None:
    if output.exists() or output.with_suffix(".planes.json").exists():
        raise ValueError("refuse overwrite")
    plan = json.loads(plan_path.read_text())
    for p, sha in plan["source_sha256"].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha:
            raise ValueError("changed " + p)
    rng = np.random.default_rng(20261031)
    rows = []
    for a, b in [(0.0, 0.0), (0.3, -0.1), (0.8, 0.4)]:
        for d in [6.0, 12.0, 24.0]:
            for u, v in [(CAMERA.cx, CAMERA.cy), (40.0, 40.0), (280.0, 150.0)]:
                truth = np.array([a, b, d])
                samples = truth + rng.normal(size=(20000, 3)) * np.sqrt(
                    [0.25 / 50, 0.25 / 50, 0.25 / 25]
                )
                mapping = np.array(
                    [[CAMERA.fx, 0, 0], [0, CAMERA.fy, 0], [-(u - CAMERA.cx), -(v - CAMERA.cy), 1]]
                )
                n = samples @ mapping.T
                t = truth @ mapping.T
                angle = np.degrees(
                    np.arccos(
                        np.clip(
                            np.abs(n @ t) / (np.linalg.norm(n, axis=1) * np.linalg.norm(t)), 0, 1
                        )
                    )
                )
                radius = cone_radius(samples, np.full(20000, u), np.full(20000, v))
                old = old_radius(samples, np.full(20000, u), np.full(20000, v))
                rows.append(
                    {
                        "coefficients": truth.tolist(),
                        "pixel": [u, v],
                        "coverage": float(np.mean(angle <= radius)),
                        "old_coverage": float(np.mean(angle <= old)),
                        "median_radius_deg": float(np.median(radius)),
                        "old_median_radius_deg": float(np.median(old)),
                        "fraction_radius_le_15": float(np.mean(radius <= 15)),
                    }
                )
    output.with_suffix(".planes.json").write_text(
        json.dumps(
            {
                "seed": 20261031,
                "samples_per_case": 20000,
                "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                "coverage_screen_passed": all(r["coverage"] >= 0.94 for r in rows),
                "cases": rows,
            },
            indent=2,
        )
        + "\n"
    )
    # Curved cases are diagnostic even if the plane coverage gate fails; no adoption.
    run(plan_path, output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("plan", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    validate(a.plan, a.output)
