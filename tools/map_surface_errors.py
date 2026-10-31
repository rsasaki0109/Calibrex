"""Fixed offline spatial attribution of real surface-quality failures."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree
from tools.camera_surface import StereoGeometry
from tools.camera_surface_prediction import gated_estimate
from tools.known_geometry_joint import Report

from calibrex.solvers.chen_medioni_point_to_plane_icp_solver import (
    ChenMedioniPointToPlaneOptions,
    _estimate_target_normals,
)


def boundary_band(depth: np.ndarray, fb: float) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    d = np.divide(fb, depth, out=np.zeros_like(depth), where=valid)
    edge = ~valid
    for axis in [0, 1]:
        jumps = np.abs(np.diff(d, axis=axis)) > 1.0
        left = [slice(None), slice(None)]
        right = left.copy()
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        edge[tuple(left)] |= jumps
        edge[tuple(right)] |= jumps
    return maximum_filter(edge, size=21)[10:-10, 10:-10].astype(bool)


def summarize(
    mask: np.ndarray, supported: np.ndarray, distance: np.ndarray, angle: np.ndarray
) -> dict:
    n = int(mask.sum())
    pairs = mask & supported
    return {
        "selected": n,
        "normal_pairs": int(pairs.sum()),
        "distance_failure_fraction": float(np.mean(distance[mask] > 1)) if n else None,
        "normal_failure_fraction": float(np.mean(angle[pairs] > 15)) if pairs.any() else None,
    }


def run(plan_path: Path, workspace: Path, output: Path) -> None:
    if output.exists() or workspace.exists():
        raise ValueError("refuse overwrite")
    plan = json.loads(plan_path.read_text())
    for group in ["source_sha256", "input_sha256"]:
        for p, sha in plan[group].items():
            if hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha:
                raise ValueError("changed " + p)
    workspace.mkdir(parents=True)
    report = Report(
        scope="18 offline spatial surface error maps; no fits or new masks",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    )
    for f in plan["frames"]:
        g = StereoGeometry(**f["geometry"])
        depth = np.load(f["depth"], allow_pickle=False)
        e = gated_estimate(depth, g)
        mask = e["accepted"]
        v, u = np.nonzero(mask)
        z = e["depth"][mask]
        target = np.column_stack([((u + 10) - g.cx) * z / g.fx, ((v + 10) - g.cy) * z / g.fy, z])
        a = np.load(f["path"])
        t = np.array(f["reference"])
        source = a["source_lidar"] @ t[:3, :3].T + t[:3, 3]
        normals, valid, _ = _estimate_target_normals(source, ChenMedioniPointToPlaneOptions())
        dist, index = cKDTree(source).query(target)
        supported = np.zeros_like(mask)
        supported[mask] = (dist <= 1) & valid[index]
        distance = np.full(mask.shape, np.nan)
        distance[mask] = dist
        angle = np.full(mask.shape, np.nan)
        all_angle = np.degrees(
            np.arccos(np.clip(np.abs(np.sum(e["normal"][mask] * normals[index], axis=1)), 0, 1))
        )
        angle[mask] = all_angle
        angle[~supported] = np.nan
        boundary = boundary_band(depth, g.fb)
        regions = []
        for label, region in [
            ("all", mask),
            ("boundary_band", mask & boundary),
            ("interior", mask & ~boundary),
        ]:
            regions.append({"region": label, **summarize(region, supported, distance, angle)})
        cells = []
        yy, xx = np.indices(mask.shape)
        for row in range(4):
            for col in range(4):
                cell = mask & (yy * 4 // mask.shape[0] == row) & (xx * 4 // mask.shape[1] == col)
                cells.append(
                    {"row": row, "column": col, **summarize(cell, supported, distance, angle)}
                )
        artifact = workspace / (f["drive"] + "-" + f["frame"] + ".npz")
        np.savez_compressed(
            artifact,
            mask=mask,
            supported=supported,
            distance=distance,
            angle=angle,
            boundary=boundary,
            depth=e["depth"],
        )
        report.cases.append(
            {
                "drive": f["drive"],
                "frame": f["frame"],
                "regions": regions,
                "cells": cells,
                "map_path": str(artifact),
                "map_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "image": f["image"],
            }
        )
        output.write_text(report.model_dump_json(indent=2) + "\n")
    output.with_suffix(".schema.json").write_text(
        json.dumps(Report.model_json_schema(), indent=2) + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["plan", "workspace", "output"]:
        p.add_argument(name, type=Path)
    a = p.parse_args()
    run(a.plan, a.workspace, a.output)
