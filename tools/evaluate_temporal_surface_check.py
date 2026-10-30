"""Controlled projection validation followed by all-center temporal surface diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from tools.known_geometry_joint import CAMERA, Report
from tools.map_surface_errors import summarize
from tools.occlusion_geometry import SCENES, raycast
from tools.temporal_surface_check import STATUS, classify


def controls() -> list[dict]:
    v, u = np.mgrid[10:182:4, 10:310:4]
    rays = np.stack(
        [(u - CAMERA.cx) / CAMERA.fx, (v - CAMERA.cy) / CAMERA.fy, np.ones_like(u)], axis=-1
    ).reshape(-1, 3)
    vv, uu = np.indices((192, 320))
    full_rays = np.stack(
        [(uu - CAMERA.cx) / CAMERA.fx, (vv - CAMERA.cy) / CAMERA.fy, np.ones_like(uu)], axis=-1
    )
    rows = []
    for scene, spheres in SCENES.items():
        points = rays * raycast(np.zeros(3), rays, spheres)[:, None]
        for axis in range(3):
            for sign in [-1, 1]:
                origin = np.zeros(3)
                origin[axis] = sign * 0.1
                transform = np.eye(4)
                transform[:3, 3] = -origin
                target = raycast(origin, full_rays, spheres)
                status = classify(points, target, transform, [240, 240, 159.5, 95.5], 120)
                visible = np.isclose(
                    raycast(origin, points - origin, spheres), 1, atol=1e-7, rtol=0
                )
                rate = float(np.mean(status[visible] == 3))
                rows.append(
                    {
                        "scene": scene,
                        "axis": axis,
                        "sign": sign,
                        "visible": int(visible.sum()),
                        "visible_consistency_fraction": rate,
                        "passed": rate >= 0.95,
                        "categories": {
                            name: int(np.sum(status == code)) for code, name in STATUS.items()
                        },
                    }
                )
    return rows


def run(plan_path: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("refuse overwrite")
    plan = json.loads(plan_path.read_text())
    for group in ["source_sha256", "input_sha256"]:
        for p, sha in plan[group].items():
            if hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha:
                raise ValueError("changed " + p)
    validation = controls()
    output.with_suffix(".controls.json").write_text(json.dumps(validation, indent=2) + "\n")
    if not all(x["passed"] for x in validation):
        raise ValueError("projection control gate failed")
    report = Report(
        scope="18 center surfaces with36 temporal checks; no extrinsic fits",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    )
    for f in plan["centers"]:
        maps = np.load(f["map_path"], allow_pickle=False)
        mask = maps["mask"]
        v, u = np.nonzero(mask)
        z = maps["depth"][mask]
        fx, fy, cx, cy = f["intrinsics"]
        points = np.column_stack([((u + 10) - cx) * z / fx, ((v + 10) - cy) * z / fy, z])
        temporal = np.ones(len(points), dtype=bool)
        neighbors = []
        for neighbor in f["neighbors"]:
            if not neighbor["motion_supported"]:
                status = np.full(len(points), 6, dtype=np.uint8)
            else:
                status = classify(
                    points,
                    np.load(neighbor["depth"], allow_pickle=False),
                    np.array(neighbor["transform"]),
                    f["intrinsics"],
                    f["fb"],
                )
            temporal &= status == 3
            neighbors.append(
                {
                    "index": neighbor["index"],
                    "motion_supported": neighbor["motion_supported"],
                    "counts": {name: int(np.sum(status == code)) for code, name in STATUS.items()},
                }
            )
        selected = np.zeros_like(mask)
        selected[mask] = temporal
        populations = []
        for name, pop in [
            ("baseline", mask),
            ("temporal_both", selected),
            ("not_temporally_verified", mask & ~selected),
        ]:
            yy, xx = np.nonzero(pop)
            cells = len(
                set(
                    zip(
                        (yy * 4 // pop.shape[0]).tolist(),
                        (xx * 4 // pop.shape[1]).tolist(),
                        strict=True,
                    )
                )
            )
            populations.append(
                {
                    "population": name,
                    **summarize(pop, maps["supported"], maps["distance"], maps["angle"]),
                    "domain_retention": float(pop.mean()),
                    "cells": cells,
                }
            )
        report.cases.append(
            {
                "drive": f["drive"],
                "center": f["center"],
                "neighbors": neighbors,
                "populations": populations,
            }
        )
        output.write_text(report.model_dump_json(indent=2) + "\n")
    output.with_suffix(".schema.json").write_text(
        json.dumps(Report.model_json_schema(), indent=2) + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("plan", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    run(a.plan, a.output)
