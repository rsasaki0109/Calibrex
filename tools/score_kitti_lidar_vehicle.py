"""Score LiDAR-to-vehicle rotations on held-out blocks of KITTI raw drives.

Two steps, because the LiDAR odometry is the slow part::

    # 1. LiDAR odometry motions and OXTS body-frame motions per drive (cached).
    python tools/score_kitti_lidar_vehicle.py collect DRIVE... --cache CACHE.pkl

    # 2. Held-out scores and per-drive reference agreement.
    python tools/score_kitti_lidar_vehicle.py score CACHE.pkl OUT_DIR

Scoring, for the pooled drives with every third 10-second block held out:

* ``calibrex_native``: rotation and lever fitted on the train blocks;
* ``kitti_vendor``: the rotation of KITTI's ``calib_imu_to_velo`` (the OXTS
  frame taken as the vehicle frame), with only the lever fitted on the train
  blocks.

The per-block metric is the mean clipped squared normalized non-holonomic
residual of the held-out LiDAR motions (lower is better).  Separately, per
drive, Calibrex fitted on that drive's LiDAR motions is compared with the
vehicle frame fitted on that drive's OXTS body-frame motions, composed with
``calib_imu_to_velo``; the metric is the pitch/yaw difference norm (deg).
"""

from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from calibrex.data.kitti_ins_lidar import load_kitti_ins_lidar_drive
from calibrex.evaluation.ins_lidar_hand_eye import _velodyne_loader
from calibrex.evaluation.vehicle_frame import oxts_motions
from calibrex.solvers.scan_to_scan_odometry import ScanOdometryOptions, odometry_from_loader
from calibrex.solvers.vehicle_frame_solver import (
    VehicleFrameOptions,
    motions_from_poses,
    solve_vehicle_frame,
    vehicle_residuals,
)

BLOCK_S = 10.0
OPTIONS = VehicleFrameOptions()


def collect(drives: list[str], cache: str) -> None:
    data = {}
    for path in drives:
        drive = load_kitti_ins_lidar_drive(path)
        frames = drive.velodyne_frames
        times = [frame.time_s for frame in frames]
        odometry = odometry_from_loader(
            _velodyne_loader(frames), len(frames), ScanOdometryOptions(), times_s=times
        )
        data[Path(path).name] = {
            "lidar": motions_from_poses(times, list(odometry.poses), block_duration_s=BLOCK_S),
            "oxts": oxts_motions(path, block_duration_s=BLOCK_S),
            "vendor": drive.vendor_t_imu_lidar,
            "input_sha256": drive.input_sha256,
        }
        print(Path(path).name, len(frames), flush=True)
    with open(cache, "wb") as stream:
        pickle.dump(data, stream)


def _block_score(motions, rotation, lever) -> float:
    residuals = vehicle_residuals(motions, rotation, lever, OPTIONS)
    return float(np.mean(np.minimum(residuals**2, 25.0))) if len(residuals) else math.nan


def _lever_only(motions, rotation) -> float:
    def residuals(lever):
        return vehicle_residuals(motions, rotation, float(lever[0]), OPTIONS)

    return float(least_squares(residuals, [0.0], loss="huber", f_scale=1.5).x[0])


def score(cache: str, output: str) -> None:
    with open(cache, "rb") as stream:
        data = pickle.load(stream)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    names = sorted(data)
    vendor = data[names[0]]["vendor"][:3, :3]
    tagged = []
    for name in names:
        for item in data[name]["lidar"]:
            tagged.append((f"{name}/b{item.block}", item))
    keys = sorted({key for key, _ in tagged})
    holdout_keys = {key for index, key in enumerate(keys) if index % 3 == 1}
    train = [item for key, item in tagged if key not in holdout_keys]
    fit = solve_vehicle_frame(train, OPTIONS)
    assert fit.rotation is not None and fit.lever_m is not None
    vendor_lever = _lever_only(train, vendor)
    scores: dict[str, dict[str, float]] = {"calibrex_native": {}, "kitti_vendor": {}}
    for key in sorted(holdout_keys):
        block = [item for tag, item in tagged if tag == key]
        scores["calibrex_native"][key] = _block_score(block, fit.rotation, fit.lever_m)
        scores["kitti_vendor"][key] = _block_score(block, vendor, vendor_lever)
    agreement = {}
    for name in names:
        own = solve_vehicle_frame(data[name]["lidar"], OPTIONS, initial=fit.rotation)
        oxts = solve_vehicle_frame(data[name]["oxts"], OPTIONS)
        if own.rotation is None or oxts.rotation is None:
            agreement[name] = math.nan
            continue
        reference = oxts.rotation @ vendor
        difference = np.degrees(Rotation.from_matrix(own.rotation @ reference.T).as_rotvec())
        agreement[name] = float(np.hypot(difference[1], difference[2]))
    euler = Rotation.from_matrix(fit.rotation).as_euler("xyz", degrees=True)
    result = {
        "calibrex_rotation_rpy_deg": euler.tolist(),
        "calibrex_lever_m": fit.lever_m,
        "vendor_lever_m": vendor_lever,
        "holdout_block_scores": scores,
        "pitch_yaw_agreement_deg": agreement,
        "drives": names,
        "input_sha256": {name: data[name]["input_sha256"] for name in names},
    }
    (out / "scores.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    calibrex = np.array(list(scores["calibrex_native"].values()))
    vendor_scores = np.array(list(scores["kitti_vendor"].values()))
    print(
        "held-out blocks",
        len(calibrex),
        "mean score calibrex",
        round(float(np.nanmean(calibrex)), 3),
        "vendor",
        round(float(np.nanmean(vendor_scores)), 3),
        "calibrex better in",
        int(np.sum(calibrex < vendor_scores)),
    )
    print("per-drive pitch/yaw agreement (deg)", {k: round(v, 3) for k, v in agreement.items()})


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        cache_index = sys.argv.index("--cache")
        collect(sys.argv[2:cache_index], sys.argv[cache_index + 1])
    else:
        score(sys.argv[2], sys.argv[3])
