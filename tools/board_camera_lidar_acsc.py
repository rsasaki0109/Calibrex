#!/usr/bin/env python3
"""Run ``calibrex.evaluation.board_camera_lidar`` on the ACSC sample (development data).

The sample (``HViktorTsoi/ACSC``, GPL-3.0 repository; data not redistributed) holds
``images/*.png``, ``pcds/*.npy`` (N x 4: x, y, z, intensity) and the camera's
``intrinsic`` and ``distortion``. Writes a JSON record of the per-axis result and the
per-pose residuals for ``docs/benchmarks/board_camera_lidar.md``.

    python tools/board_camera_lidar_acsc.py ACSC/sample --output acsc_board.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from calibrex import __version__
from calibrex.evaluation.board_camera_lidar import (
    BoardPose,
    Checkerboard,
    calibrate_board,
    detect_board,
    plane_residuals,
)
from calibrex.evaluation.visual_rotation import CameraModel

# LiDAR x forward, y left, z up into the optical frame x right, y down, z forward
AXIS_PRIOR = np.array(
    [[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
)


def _numbers(path: Path) -> list[float]:
    # the sample's files run their rows together without separators at line ends
    return [float(value) for value in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", path.read_text())]


def _camera(sample: Path) -> CameraModel:
    k = _numbers(sample / "intrinsic")
    distortion = _numbers(sample / "distortion")
    # [[fx, skew, cx], [0, fy, cy], [0, 0, 1]]; the skew is dropped
    return CameraModel(k[0], k[4], k[2], k[5], "radtan", tuple(distortion[:5]))


def main(argv: list[str] | None = None) -> int:
    import cv2

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sample", type=Path, help="the ACSC sample directory")
    parser.add_argument("--board", default="7x5x0.08", help="inner corners and square size")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    camera = _camera(args.sample)
    board = Checkerboard.parse(args.board)
    poses: list[BoardPose] = []
    missing: list[str] = []
    for image in sorted((args.sample / "images").glob("*.png")):
        view = detect_board(cv2.imread(str(image), cv2.IMREAD_GRAYSCALE), board, camera)
        if view is None:
            missing.append(image.stem)
            continue
        cloud = np.load(args.sample / "pcds" / f"{image.stem}.npy")[:, :3]
        poses.append(BoardPose(image.stem, view, cloud[np.linalg.norm(cloud, axis=1) > 0.5]))
    result = calibrate_board(poses, board, AXIS_PRIOR)
    record: dict[str, object] = {
        "calibrex_version": __version__,
        "board": args.board,
        "camera": camera.__dict__,
        "prior": AXIS_PRIOR.tolist(),
        "poses_without_board": missing,
        "status": result.status,
        "reason": result.reason,
        "poses_used": list(result.poses_used),
        "poses_rejected": [list(item) for item in result.poses_rejected],
        "normal_singular_values": result.normal_singular_values,
        "unobserved_directions": [list(d) for d in result.unobserved_directions],
        "axes": [axis.__dict__ for axis in result.axes],
    }
    transform = result.transform_camera_lidar
    if transform is not None:
        record["T_camera_lidar"] = transform.tolist()
        record["rotation_from_prior_deg"] = (
            Rotation.from_matrix(transform[:3, :3] @ AXIS_PRIOR[:3, :3].T)
            .as_rotvec(degrees=True)
            .tolist()
        )
        residuals = []
        for pose in poses:
            found = plane_residuals(pose, board, transform)
            if found is None:
                continue
            lidar, normal_deg, offset_m = found
            residuals.append(
                {
                    "pose": pose.pose_id,
                    "distance_m": float(np.linalg.norm(pose.view.translation)),
                    "normal_deg": normal_deg,
                    "offset_m": offset_m,
                    "lidar_inliers": lidar.inliers,
                    "lidar_rms_m": lidar.rms_m,
                    "reprojection_px": pose.view.reprojection_px,
                }
            )
        record["per_pose"] = residuals
    args.output.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("status", "axes")}, indent=1))
    return 0 if result.status == "converged" else 1


if __name__ == "__main__":
    raise SystemExit(main())
