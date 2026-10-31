"""Fixed photometric cost probes at frozen stereo disparities; no matching changes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from tools.known_geometry_joint import Report
from tools.stereo_rerectification import compute_depth

from calibrex.data.kitti import read_calibration_file

OFFSETS = np.array([-8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0])


def patch_costs(
    left: np.ndarray, right: np.ndarray, u: np.ndarray, v: np.ndarray, d: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dy, dx = np.mgrid[-2:3, -2:3]
    dx = dx.ravel()
    dy = dy.ravel()
    x = (u[:, None] + dx).astype(np.float32)
    y = (v[:, None] + dy).astype(np.float32)
    left_patches = cv2.remap(left, x, y, cv2.INTER_LINEAR).astype(float)
    texture = left_patches.std(axis=1)
    left_patches = left_patches - left_patches.mean(axis=1, keepdims=True)
    valid = (
        (u >= 2)
        & (u < left.shape[1] - 3)
        & (v >= 2)
        & (v < left.shape[0] - 3)
        & np.isfinite(d)
        & (d > 8)
    )
    costs = []
    for offset in OFFSETS:
        xr = x - (d + offset)[:, None].astype(np.float32)
        valid &= (xr.min(axis=1) >= 0) & (xr.max(axis=1) < right.shape[1] - 1)
        r = cv2.remap(right, xr, y, cv2.INTER_LINEAR).astype(float)
        r = r - r.mean(axis=1, keepdims=True)
        costs.append(np.mean(np.abs(left_patches - r), axis=1))
    return np.stack(costs, axis=1), texture, valid


def stats(cost: np.ndarray, texture: np.ndarray, mask: np.ndarray) -> dict:
    if not mask.any():
        return {
            "count": 0,
            "texture_quantiles": None,
            "base_cost_quantiles": None,
            "ambiguous_fraction": None,
            "alternative_better_fraction": None,
        }
    c = cost[mask]
    t = texture[mask]
    base = c[:, 4]
    alternative = c[:, np.abs(OFFSETS) >= 2].min(axis=1)
    return {
        "count": int(mask.sum()),
        "texture_quantiles": np.quantile(t, [0.1, 0.5, 0.9]).tolist(),
        "base_cost_quantiles": np.quantile(base, [0.1, 0.5, 0.9]).tolist(),
        "ambiguous_fraction": float(np.mean(alternative <= base + 0.5)),
        "alternative_better_fraction": float(np.mean(alternative < base - 1)),
        "low_texture_fraction": float(np.mean(t <= 5)),
    }


def run(plan_path: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("refuse overwrite")
    plan = json.loads(plan_path.read_text())
    cv2.setNumThreads(1)
    for group in ["source_sha256", "input_sha256"]:
        for p, sha in plan[group].items():
            if hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha:
                raise ValueError("changed " + p)
    report = Report(
        scope="18 fixed stereo appearance diagnoses; no depth or mask changes",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    )
    for f in plan["frames"]:
        images = [cv2.imread(f[k], cv2.IMREAD_GRAYSCALE) for k in ["left", "right"]]
        cam = read_calibration_file(Path(f["calibration"]))
        pl, pr = [np.array(cam["P_rect_" + k]).reshape(3, 4) for k in ["02", "03"]]
        restored, g = compute_depth(*images, pl, pr, f["settings"])
        np.testing.assert_array_equal(restored, np.load(f["depth"], allow_pickle=False))
        rectified = []
        supports = []
        for im, p, r, new in zip(
            images, [pl, pr], [g["r1"], g["r2"]], [g["p1"], g["p2"]], strict=True
        ):
            xx, yy = cv2.initUndistortRectifyMap(
                p[:, :3],
                np.zeros(5),
                np.array(r),
                np.array(new)[:, :3],
                (im.shape[1], im.shape[0]),
                cv2.CV_32FC1,
            )
            rectified.append(cv2.remap(im, xx, yy, cv2.INTER_LINEAR))
            supports.append((xx >= 0) & (xx < im.shape[1] - 1) & (yy >= 0) & (yy < im.shape[0] - 1))
        forward = cv2.StereoSGBM_create(**f["settings"]).compute(*rectified).astype(float) / 16
        maps = np.load(f["map_path"], allow_pickle=False)
        mask = maps["mask"].copy()
        grid = np.zeros_like(mask)
        grid[::4, ::4] = True
        mask &= grid
        vv, uu = np.nonzero(mask)
        v = vv + 10
        u = uu + 10
        rays = (
            np.column_stack([u, v, np.ones(len(u))])
            @ np.linalg.inv(pl[:, :3]).T
            @ np.array(g["r1"]).T
        )
        proj = rays @ np.array(g["p1"])[:, :3].T
        ur = np.rint(proj[:, 0] / proj[:, 2]).astype(int)
        vr = np.rint(proj[:, 1] / proj[:, 2]).astype(int)
        inside = (ur >= 2) & (ur < forward.shape[1] - 3) & (vr >= 2) & (vr < forward.shape[0] - 3)
        ur = np.clip(ur, 0, forward.shape[1] - 1)
        vr = np.clip(vr, 0, forward.shape[0] - 1)
        d = forward[vr, ur]
        cost, texture, valid = patch_costs(*rectified, ur, vr, d)
        # Exclude remap-invalid patches for every probed alternative, with erosion margin.
        support_left = cv2.erode(supports[0].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(
            bool
        )
        support_right = cv2.erode(supports[1].astype(np.uint8), np.ones((7, 7), np.uint8)).astype(
            bool
        )
        valid &= inside & support_left[vr, ur]
        for offset in OFFSETS:
            xr = np.rint(ur - d - offset).astype(int)
            valid &= (xr >= 0) & (xr < forward.shape[1])
            valid &= support_right[vr, np.clip(xr, 0, forward.shape[1] - 1)]
        angle = maps["angle"][mask]
        supported = maps["supported"][mask]
        depth = restored[v, u]
        rows = []
        for label, pop in [
            ("all", np.ones(len(u), bool)),
            ("normal_agrees", supported & (angle <= 15)),
            ("normal_disagrees", supported & (angle > 15)),
            ("distance_fails", maps["distance"][mask] > 1),
        ]:
            for lower, upper in [(0, 80), (0, 10), (10, 20), (20, 40), (40, 80)]:
                rows.append(
                    {
                        "population": label,
                        "depth_band_m": [lower, upper],
                        **stats(cost, texture, valid & pop & (depth > lower) & (depth <= upper)),
                    }
                )
        report.cases.append(
            {
                "drive": f["drive"],
                "frame": f["frame"],
                "sampled": len(u),
                "valid_cost_probes": int(valid.sum()),
                "groups": rows,
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
