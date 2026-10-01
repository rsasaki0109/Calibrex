"""Diagnose the camera-IMU jackknife std: per-window rotation scale scatter.

On the Hilti 2022 development recordings the camera-IMU reported std is set by
the window jackknife (0.08-0.30 deg) rather than by the analytic std
(0.05-0.11 deg), so every verdict is ``inconclusive`` against the 0.1 deg
bound.  This tool decomposes that gap.

It tracks one camera of one bag once, fits the rotation on the train windows,
and then, per window, compares the rotation angle the camera tracked with the
angle the gyro predicts through the fitted rotation::

    ratio = sum(obs * exp) / sum(exp^2)

over the window's intervals.  A ratio near 1 everywhere means the two agree in
scale; a per-window scatter shows that each window carries its own scale bias,
which the jackknife converts into the reported std.

Usage::

    python tools/diagnose_camera_imu_std.py BAG CAMCHAIN CAMERA IMU_TOPIC
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from calibrex.core.io import read_mapping
from calibrex.data.imu_trajectory import split_trajectory_windows
from calibrex.data.ros2_camera_imu import load_ros2_imu
from calibrex.evaluation.camera_imu_rotation import CameraImuRunOptions, camera_rotation_windows
from calibrex.evaluation.imu_lidar_rotation import ImuLidarRunOptions, rate_intervals
from calibrex.evaluation.visual_rotation import CameraModel, VisualRotationOptions
from calibrex.solvers.gnss_lever_arm_solver import OdometryWindow
from calibrex.solvers.imu_lidar_rotation_solver import GyroSeries, ImuLidarRotationSolver


def _duration_windows(track, duration_s: float) -> list[OdometryWindow]:
    poses = np.repeat(np.eye(4)[None], len(track.times_s), axis=0)
    poses[:, :3, :3] = track.orientations
    out: list[OdometryWindow] = []
    for label in np.unique(track.segment):
        members = track.segment == label
        out += split_trajectory_windows(
            track.times_s[members],
            poses[members],
            window_duration_s=duration_s,
            min_window_poses=VisualRotationOptions().min_window_frames,
            prefix=f"segment{int(label)}/",
        )
    return [OdometryWindow(f"bag/{w.window_id}", w.block, w.times_s, w.poses) for w in out]


def main() -> None:
    bag, camchain, camera_key, imu_topic = sys.argv[1:5]
    entry = read_mapping(Path(camchain))[camera_key]
    camera = CameraModel.from_kalibr(entry)
    options = CameraImuRunOptions(frame_stride=4)
    _, tracks = camera_rotation_windows([bag], entry["rostopic"], camera, options)
    track = tracks[0]
    print(
        f"tracking: pairs {track.stats.pairs} essential {track.stats.essential} "
        f"pure {track.stats.pure_rotation} failed {track.stats.failed} "
        f"segments {track.stats.segments}"
    )
    imu = load_ros2_imu(bag, imu_topic)
    gyro = GyroSeries(imu.times_s, imu.gyro_rps)

    for duration in (5.0, 10.0, 20.0):
        windows = _duration_windows(track, duration)
        if len(windows) < 6:
            continue
        opts = ImuLidarRunOptions()
        intervals = rate_intervals(windows, gyro, opts.solver)
        positions = np.arange(len(windows))
        train_ids = positions[positions % opts.holdout_every != 1]
        train = intervals.subset(np.isin(intervals.window_ids, train_ids))
        solver = ImuLidarRotationSolver()
        fit = solver.solve(gyro, train, opts.solver)
        if fit.rotation is None:
            print(f"duration {duration}: fit failed")
            continue
        R, bias, dt0 = fit.rotation, fit.gyro_bias_rps, fit.time_offset_s
        groups = min(opts.jackknife_groups, len(train_ids))
        errors = []
        for members in np.array_split(train_ids, groups):
            sub = solver.solve(gyro, train.subset(~np.isin(train.window_ids, members)), opts.solver)
            assert sub.rotation is not None
            errors.append(np.degrees(Rotation.from_matrix(sub.rotation @ R.T).as_rotvec()))
        jackknife = np.sqrt(
            (groups - 1)
            / groups
            * np.sum((np.array(errors) - np.mean(errors, axis=0)) ** 2, axis=0)
        )
        analytic = np.degrees([fit.dof(name).std for name in ("roll", "pitch", "yaw")])
        ratios = []
        for wid in np.unique(intervals.window_ids):
            observed, expected = [], []
            for index in np.where(intervals.window_ids == wid)[0]:
                start, end = intervals.start_s[index], intervals.end_s[index]
                mask = (gyro.times_s >= start - dt0) & (gyro.times_s <= end - dt0)
                if mask.sum() < 2:
                    continue
                times = gyro.times_s[mask] - dt0
                steps = np.diff(times)
                rates = 0.5 * (gyro.gyro_rps[mask][:-1] + gyro.gyro_rps[mask][1:]) - bias
                vector = np.sum(rates * steps[:, None], axis=0)
                expected.append(float(np.linalg.norm(R @ vector)))
                observed.append(
                    float(
                        np.linalg.norm(
                            Rotation.from_matrix(intervals.lidar_delta[index]).as_rotvec()
                        )
                    )
                )
            observed, expected = np.asarray(observed), np.asarray(expected)
            if len(observed) > 5 and np.median(np.degrees(expected)) > 0.3:
                ratios.append(float(np.sum(observed * expected) / np.sum(expected * expected)))
        ratios = np.asarray(ratios)
        print(
            f"duration {duration:4.0f}s windows {len(windows):3d} "
            f"jackknife(xyz deg) {np.round(jackknife, 4).tolist()} "
            f"analytic {np.round(analytic, 4).tolist()}"
        )
        print(
            f"    moving-window scale ratio min/median/max "
            f"{ratios.min():.4f}/{np.median(ratios):.4f}/{ratios.max():.4f} (n={len(ratios)})"
        )


if __name__ == "__main__":
    main()
