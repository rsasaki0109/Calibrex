"""Controlled metric motion validation and all36 short-baseline stereoVO pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from tools.known_geometry_joint import Report
from tools.stereo_metric_motion import metric_pnp, sample_points
from tools.stereo_rerectification import compute_depth

from calibrex.data.kitti import read_calibration_file


def controlled() -> list[dict]:
    k = np.array([[720.0, 0, 620.0], [0, 720.0, 180.0], [0, 0, 1.0]])
    rows = []
    for axis in range(3):
        for sign in [-1, 1]:
            for seed in [0, 730, 731]:
                rng = np.random.default_rng(100 * axis + (sign + 1) + seed)
                truth = np.eye(4)
                r = np.zeros(3)
                r[axis] = sign
                truth[:3, :3] = Rotation.from_euler("xyz", r, degrees=True).as_matrix()
                truth[axis, 3] = sign * 0.1
                points = np.column_stack(
                    [rng.uniform(-4, 4, 600), rng.uniform(-2, 2, 600), rng.uniform(5, 30, 600)]
                )
                target = points @ truth[:3, :3].T + truth[:3, 3]
                pixels = target[:, :2] / target[:, 2, None] * 720 + [620, 180]
                measured = points.copy()
                if seed:
                    disparity = 390 / points[:, 2] + rng.normal(0, 0.5, 600)
                    measured = points * (390 / disparity / points[:, 2])[:, None]
                    pixels += rng.normal(0, 0.5, pixels.shape)
                    pixels[:60] = rng.uniform([0, 0], [1240, 360], (60, 2))
                result = metric_pnp(measured, pixels, k)
                fit = np.array(result["transform"]) if result["transform"] else None
                rot = (
                    float(
                        Rotation.from_matrix(fit[:3, :3] @ truth[:3, :3].T).magnitude()
                        * 180
                        / np.pi
                    )
                    if fit is not None
                    else None
                )
                trans = (
                    float(np.linalg.norm(fit[:3, 3] - truth[:3, 3])) if fit is not None else None
                )
                rows.append(
                    {
                        "axis": axis,
                        "sign": sign,
                        "seed": seed,
                        "result": result,
                        "rotation_error_deg": rot,
                        "translation_error_m": trans,
                        "passed": rot is not None and rot < 0.2 and trans < 0.05,
                    }
                )
    return rows


def run(plan_path: Path, workspace: Path, output: Path) -> None:
    if output.exists() or workspace.exists():
        raise ValueError("refuse overwrite")
    plan = json.loads(plan_path.read_text())
    for group in ["source_sha256", "input_sha256"]:
        for p, sha in plan[group].items():
            if hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha:
                raise ValueError("changed " + p)
    workspace.mkdir(parents=True)
    cv2.setNumThreads(1)
    controls = controlled()
    output.with_suffix(".controls.json").write_text(json.dumps(controls, indent=2) + "\n")
    if not all(x["passed"] for x in controls):
        raise ValueError("controlled motion gate failed; real run blocked")
    report = Report(
        scope="36 camera-only stereo metric motion estimates; no LiDAR reference",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    )
    cache = {}
    for pair in plan["pairs"]:
        intrinsic = pair["intrinsics"]
        fx, fy, cx, cy = intrinsic
        k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        data = []
        for f in pair["frames"]:
            key = f["left"]
            if key not in cache:
                left, right = [cv2.imread(f[x], cv2.IMREAD_GRAYSCALE) for x in ["left", "right"]]
                cam = read_calibration_file(Path(pair["calibration"]))
                pl, pr = [np.array(cam["P_rect_" + s]).reshape(3, 4) for s in ["02", "03"]]
                depth, geometry = compute_depth(left, right, pl, pr, pair["settings"])
                if f["expected_depth"] is not None:
                    np.testing.assert_array_equal(
                        depth, np.load(f["expected_depth"], allow_pickle=False)
                    )
                path = workspace / (pair["drive"] + f"-{f['index']:010d}.npy")
                np.save(path, depth)
                cache[key] = (
                    cv2.equalizeHist(left),
                    depth,
                    {
                        "path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "stereo_geometry": geometry,
                    },
                )
            data.append(cache[key])
        orb = cv2.ORB_create(nfeatures=2000)
        features = [orb.detectAndCompute(d[0], None) for d in data]
        descriptors = [x[1] for x in features]
        fits = []
        matches = (
            []
            if any(x is None for x in descriptors)
            else sorted(
                cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(*descriptors),
                key=lambda x: (x.queryIdx, x.trainIdx),
            )
        )
        pixels = [
            np.array(
                [features[i][0][m.queryIdx if i == 0 else m.trainIdx].pt for m in matches], float
            ).reshape(-1, 2)
            for i in range(2)
        ]
        for i, j in [(0, 1), (1, 0)]:
            xyz, valid = sample_points(data[i][1], pixels[i], k)
            fit = metric_pnp(xyz, pixels[j][valid], k)
            ids = np.array(fit.get("inlier_ids", []), dtype=int)
            cells = []
            for pix in [pixels[i][valid][ids], pixels[j][valid][ids]]:
                h, w = data[i][0].shape
                u = np.clip((pix[:, 0] * 4 / w).astype(int), 0, 3)
                v = np.clip((pix[:, 1] * 4 / h).astype(int), 0, 3)
                cells.append(len(set(zip(u.tolist(), v.tolist(), strict=True))))
            fit["cells"] = cells
            fits.append(fit)
        cycle = None
        if all(f["transform"] for f in fits):
            matrix = np.array(fits[1]["transform"]) @ np.array(fits[0]["transform"])
            cycle = {
                "rotation_deg": float(
                    Rotation.from_matrix(matrix[:3, :3]).magnitude() * 180 / np.pi
                ),
                "translation_m": float(np.linalg.norm(matrix[:3, 3])),
            }
        passed = (
            all(
                f["inliers"] >= 30
                and f["inliers"] / max(1, f["count"]) >= 0.5
                and min(f["cells"]) >= 6
                for f in fits
            )
            and cycle is not None
            and cycle["rotation_deg"] <= 0.5
            and cycle["translation_m"] <= 0.1
        )
        report.cases.append(
            {
                "drive": pair["drive"],
                "indices": [f["index"] for f in pair["frames"]],
                "depth_artifacts": [d[2] for d in data],
                "matches": len(matches),
                "directions": fits,
                "cycle": cycle,
                "internal_consistency_passed": passed,
                "metric_motion_ground_truth_available": False,
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
