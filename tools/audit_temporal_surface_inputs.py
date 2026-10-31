"""Audit independent temporal evidence without camera/LiDAR reference geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from tools.known_geometry_joint import Report

from calibrex.data.kitti import read_timestamps


def image_pair(first: Path, second: Path, intrinsics: list[float]) -> dict:
    import cv2

    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    images = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in [first, second]]
    if any(x is None for x in images):
        raise ValueError("image decode failure")
    orb = cv2.ORB_create(nfeatures=2000)
    features = [orb.detectAndCompute(x, None) for x in images]
    points = [x[0] for x in features]
    descriptors = [x[1] for x in features]
    if any(x is None for x in descriptors):
        return {
            "keypoints": [len(x) for x in points],
            "matches": 0,
            "essential_inliers": 0,
            "cheirality_inliers": 0,
            "cells": [0, 0],
        }
    matches = sorted(
        cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(*descriptors),
        key=lambda m: (m.queryIdx, m.trainIdx),
    )
    p = np.array([points[0][m.queryIdx].pt for m in matches], float)
    q = np.array([points[1][m.trainIdx].pt for m in matches], float)
    fx, fy, cx, cy = intrinsics
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    record = {
        "keypoints": [len(x) for x in points],
        "matches": len(matches),
        "essential_inliers": 0,
        "cheirality_inliers": 0,
        "cells": [0, 0],
    }
    if len(matches) < 8:
        return record
    e, mask = cv2.findEssentialMat(p, q, k, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if e is None or e.shape != (3, 3):
        return record
    record["essential_inliers"] = int((mask > 0).sum())
    count, r, t, selected = cv2.recoverPose(e, p, q, k, mask=mask.copy())
    record.update(
        cheirality_inliers=int(count), rotation=r.tolist(), translation_direction=t.ravel().tolist()
    )
    good = selected.ravel() > 0
    cells = []
    for xy, im in zip([p, q], images, strict=True):
        ix = np.clip((xy[good, 0] * 4 / im.shape[1]).astype(int), 0, 3)
        iy = np.clip((xy[good, 1] * 4 / im.shape[0]).astype(int), 0, 3)
        cells.append(len(set(zip(ix.tolist(), iy.tolist(), strict=True))))
    record["cells"] = cells
    return record


def run(plan_path: Path, output: Path) -> None:
    import cv2

    if output.exists():
        raise ValueError("refuse overwrite")
    plan = json.loads(plan_path.read_text())
    for group in ["source_sha256", "input_sha256"]:
        for p, sha in plan[group].items():
            if hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha:
                raise ValueError("changed " + p)
    report = Report(
        scope="12 adjacent image pair readiness audit; no temporal depth filtering",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    )
    for drive in plan["drives"]:
        timestamps = read_timestamps(Path(drive["camera_timestamps"]))
        imu = read_timestamps(Path(drive["imu_timestamps"]))
        for a, b in zip(drive["frames"][:-1], drive["frames"][1:], strict=True):
            result = image_pair(Path(a["image"]), Path(b["image"]), drive["intrinsics"])
            result["directional_support_screen"] = (
                result["cheirality_inliers"] >= 30 and min(result["cells"]) >= 6
            )
            report.cases.append(
                {
                    "drive": drive["drive"],
                    "first": a["frame"],
                    "second": b["frame"],
                    "interval_s": (
                        timestamps[b["index"]].timestamp_ns - timestamps[a["index"]].timestamp_ns
                    )
                    * 1e-9,
                    "camera_imu_offsets_ms": [
                        (timestamps[x["index"]].timestamp_ns - imu[x["index"]].timestamp_ns) * 1e-6
                        for x in [a, b]
                    ],
                    "oxts_packets_present": [Path(x["oxts"]).exists() for x in [a, b]],
                    "opencv_version": cv2.__version__,
                    "independent_metric_camera_motion_available": False,
                    "metric_motion_blocker": (
                        "available IMU-to-camera chain passes through target "
                        "LiDAR-camera extrinsic; monocular translation scale unknown"
                    ),
                    "image_support": result,
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
