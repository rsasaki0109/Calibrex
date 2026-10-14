#!/usr/bin/env python3
"""Is the camera-IMU offset Hilti's cameras share a property of the rig or of the motion?

An evaluation harness, not part of the Calibrex package. On ``exp21`` every camera's
``camera-imu`` estimate differs from Kalibr's ``T_cam_imu`` by about the same rotation
about the IMU axes, and the check's attribution names the IMU
(``docs/benchmarks/hilti_camera_imu.md``). A rotation of the gyro frame against Kalibr's
IMU frame would be the same in every part of a recording and in every recording. A bias
of the estimator would move with the motion. For every camera this tool tracks the
images once, as the check does, and fits the gyro to the tracked rotations on

* ``all`` moving windows;
* the ``first`` and ``second`` half in time: different motion;
* the ``even`` and ``odd`` windows: the same motion, interleaved.

``--window-s`` cuts the tracks into shorter windows than the check's, for more of them.

Each fit gives the estimate's difference to Kalibr about the IMU axes,
``w = rotvec(C^T E)`` (``E = C exp(w)``, ``C`` and ``E`` Kalibr's and the estimated
``R_cam_imu``), with a block jackknife std over windows. The cameras' ``w`` are then
pooled per split (inverse-variance, per IMU axis) into a rig-level offset with a
chi-square of the cameras' spread about it (the cameras share the gyro and the motion,
so the pooled std is optimistic). It also reports, on all windows:

* the rate-model fit of a general linear map ``w_cam = A w_gyro``: the rotation part of
  ``A`` (polar decomposition) against a rotation-only fit of the same rates, and the
  symmetric part (gyro scale and non-orthogonality about the IMU axes). A symmetric
  part that is the same for every camera, and a rotation-only fit that moves towards
  the polar rotation's, points at the gyro;
* the RMS gyro rate about each IMU axis per split: the motion each fit saw.

::

    python tools/hilti_camera_imu_offset.py exp21_head.bag \\
        --camchain calib_3_cam0-1-camchain-imucam.yaml \\
        --camchain calib_3_cam2-camchain-imucam.yaml \\
        --camchain calib_3_cam3-camchain-imucam.yaml \\
        --camchain calib_3_cam4-camchain-imucam.yaml \\
        --output exp21_camera_imu_offset.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from calibrex.check.tf_sources import camchain_entries
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.data.imu_trajectory import split_trajectory_windows
from calibrex.data.ros2_camera_imu import load_ros2_imu
from calibrex.evaluation.camera_imu_rotation import CameraImuRunOptions, camera_rotation_windows
from calibrex.evaluation.imu_lidar_rotation import window_rotation_deg
from calibrex.evaluation.visual_rotation import CameraModel
from calibrex.solvers.gnss_lever_arm_solver import OdometryWindow
from calibrex.solvers.imu_lidar_rotation_solver import (
    GyroSeries,
    ImuLidarRotationSolver,
    RateIntervals,
    RotationOptions,
    rate_intervals,
)

FloatArray = NDArray[np.float64]
SPLITS = ("all", "first", "second", "even", "odd")
STD_FLOOR_DEG = 0.05


def shorter_windows(track: object, duration_s: float, min_frames: int) -> list[OdometryWindow]:
    """The track cut into windows of ``duration_s`` instead of the check's."""

    times, orientations, segment = (
        track.times_s,  # type: ignore[attr-defined]
        track.orientations,  # type: ignore[attr-defined]
        track.segment,  # type: ignore[attr-defined]
    )
    poses = np.repeat(np.eye(4)[None], len(times), axis=0)
    poses[:, :3, :3] = orientations
    windows: list[OdometryWindow] = []
    for label in np.unique(segment):
        members = segment == label
        windows += split_trajectory_windows(
            times[members],
            poses[members],
            window_duration_s=duration_s,
            min_window_poses=min_frames,
            prefix=f"segment{int(label)}/",
        )
    return windows


def split_windows(windows: list[OdometryWindow]) -> dict[str, list[OdometryWindow]]:
    ordered = sorted(windows, key=lambda window: float(window.times_s[0]))
    half = len(ordered) // 2
    return {
        "all": ordered,
        "first": ordered[:half],
        "second": ordered[half:],
        "even": ordered[0::2],
        "odd": ordered[1::2],
    }


def offset_deg(candidate: FloatArray, estimate: FloatArray) -> FloatArray:
    """``w`` about the IMU axes with ``estimate = candidate exp(w)``."""

    return np.degrees(Rotation.from_matrix(candidate.T @ estimate).as_rotvec())


def jackknife_std(values: FloatArray) -> FloatArray:
    count = len(values)
    return np.sqrt((count - 1) / count * ((values - values.mean(axis=0)) ** 2).sum(axis=0))


def fit_split(
    windows: list[OdometryWindow],
    gyro: GyroSeries,
    candidate: FloatArray,
    solver_options: RotationOptions,
    blocks: int,
) -> dict[str, object]:
    solver = ImuLidarRotationSolver()
    intervals = rate_intervals(windows, gyro, solver_options)
    fit = solver.solve(gyro, intervals, solver_options)
    if fit.rotation is None:
        return {"windows": len(windows), "status": fit.status}
    ids = np.arange(len(windows))
    leave_out = []
    for group in np.array_split(ids, min(blocks, len(windows))):
        rows = ~np.isin(intervals.window_ids, group)
        part = solver.solve(gyro, intervals.subset(rows), solver_options)
        if part.rotation is not None:
            leave_out.append(offset_deg(candidate, part.rotation))
    std = jackknife_std(np.array(leave_out)) if len(leave_out) >= 3 else np.full(3, np.nan)
    mean_gyro = _mean_gyro(gyro, intervals, fit.time_offset_s) - fit.gyro_bias_rps
    return {
        "windows": len(windows),
        "intervals": len(intervals),
        "offset_deg": [round(float(v), 4) for v in offset_deg(candidate, fit.rotation)],
        "std_deg": [round(float(v), 4) for v in std],
        "time_offset_ms": round(1.0e3 * fit.time_offset_s, 3),
        "rms_gyro_rate_dps": [
            round(float(v), 2) for v in np.degrees(np.sqrt((mean_gyro**2).mean(axis=0)))
        ],
        "_fit": (fit.rotation, fit.gyro_bias_rps, fit.time_offset_s, intervals),
    }


def _mean_gyro(gyro: GyroSeries, intervals: RateIntervals, offset_s: float) -> FloatArray:
    duration = (intervals.end_s - intervals.start_s)[:, None]
    return np.asarray(
        (
            gyro.integral_at(intervals.end_s + offset_s)
            - gyro.integral_at(intervals.start_s + offset_s)
        )
        / duration,
        dtype=np.float64,
    )


def _robust_lstsq(design: FloatArray, values: FloatArray) -> FloatArray:
    """Rows of ``values ~ design @ X``, refitted three times without 3-sigma outliers."""

    keep = np.ones(len(values), dtype=bool)
    solution = np.zeros((design.shape[1], values.shape[1]))
    for _ in range(4):
        solution, *_ = np.linalg.lstsq(design[keep], values[keep], rcond=None)
        error = np.linalg.norm(values - design @ solution, axis=1)
        spread = 1.4826 * float(np.median(error[keep]))
        keep = error <= 3.0 * max(spread, 1.0e-9)
    return solution


def linear_map(
    candidate: FloatArray,
    gyro: GyroSeries,
    fitted: tuple[FloatArray, FloatArray, float, RateIntervals],
    blocks: int,
) -> dict[str, object]:
    """Rate-model fits of ``w_cam = A w_gyro`` and of ``w_cam = R w_gyro``, about IMU axes."""

    _rotation, bias, offset_s, intervals = fitted
    rates_gyro = _mean_gyro(gyro, intervals, offset_s) - bias
    rates_cam = intervals.lidar_rate_rps

    def solve(rows: NDArray[np.bool_]) -> tuple[FloatArray, FloatArray, FloatArray]:
        a = _robust_lstsq(rates_gyro[rows], rates_cam[rows]).T  # w_cam = A w_gyro
        u, _, vt = np.linalg.svd(a)
        polar = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
        symmetric = polar.T @ a  # A = polar @ symmetric, about the gyro axes
        u, _, vt = np.linalg.svd(rates_cam[rows].T @ rates_gyro[rows])
        rotation = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
        return (
            offset_deg(candidate, polar),
            offset_deg(candidate, rotation),
            100.0 * (0.5 * (symmetric + symmetric.T) - np.eye(3)),
        )

    every = np.ones(len(intervals), dtype=bool)
    polar, rotation, symmetric = solve(every)
    parts = [
        solve(~np.isin(intervals.window_ids, group))
        for group in np.array_split(np.unique(intervals.window_ids), blocks)
    ]
    upper = np.triu_indices(3)
    return {
        "polar_rotation_offset_deg": [round(float(v), 4) for v in polar],
        "polar_rotation_std_deg": [
            round(float(v), 4) for v in jackknife_std(np.array([p[0] for p in parts]))
        ],
        "rotation_only_offset_deg": [round(float(v), 4) for v in rotation],
        "rotation_only_std_deg": [
            round(float(v), 4) for v in jackknife_std(np.array([p[1] for p in parts]))
        ],
        # xx, xy, xz, yy, yz, zz of (S - I) in percent, S the symmetric part about IMU axes
        "symmetric_minus_identity_pct": [round(float(v), 3) for v in symmetric[upper]],
        "symmetric_std_pct": [
            round(float(v), 3) for v in jackknife_std(np.array([p[2][upper] for p in parts]))
        ],
    }


def pool(per_camera: dict[str, dict[str, object]], split: str) -> dict[str, object]:
    """Inverse-variance mean of the cameras' offsets per IMU axis, with the cameras' chi2.

    The cameras share the gyro and the motion, so their errors are not independent and
    the pooled std is optimistic; the chi2 says whether they agree.
    """

    offsets, stds = [], []
    for entry in per_camera.values():
        fit = entry["splits"][split]  # type: ignore[index]
        if "offset_deg" in fit and not np.isnan(fit["std_deg"]).any():
            offsets.append(fit["offset_deg"])
            stds.append(np.maximum(fit["std_deg"], STD_FLOOR_DEG))
    if len(offsets) < 2:
        return {"cameras": len(offsets)}
    value, sigma = np.array(offsets), np.array(stds)
    weight = 1.0 / sigma**2
    mean = (weight * value).sum(axis=0) / weight.sum(axis=0)
    chi2 = (((value - mean) / sigma) ** 2).sum(axis=0)
    return {
        "cameras": len(offsets),
        "offset_deg": [round(float(v), 4) for v in mean],
        "std_deg": [round(float(v), 4) for v in 1.0 / np.sqrt(weight.sum(axis=0))],
        "camera_chi2": [round(float(v), 2) for v in chi2],
        "camera_dof": len(offsets) - 1,
    }


def differences(rig: dict[str, dict[str, object]]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for a, b in (("first", "second"), ("even", "odd")):
        if "offset_deg" not in rig[a] or "offset_deg" not in rig[b]:
            continue
        va, vb = np.array(rig[a]["offset_deg"]), np.array(rig[b]["offset_deg"])
        sa, sb = np.array(rig[a]["std_deg"]), np.array(rig[b]["std_deg"])
        out[f"{a}-{b}_deg"] = [round(float(v), 4) for v in va - vb]
        out[f"{a}-{b}_z"] = [round(float(v), 2) for v in (va - vb) / np.hypot(sa, sb)]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("bag", type=Path)
    parser.add_argument("--camchain", type=Path, action="append", required=True)
    parser.add_argument("--imu-topic", default="/alphasense/imu")
    parser.add_argument("--cameras", default="0,1,2,3,4")
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument(
        "--window-s", type=float, default=None, help="cut the tracks into windows this long"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    commit = git_commit()
    entries = {
        entry["rostopic"]: entry for path in args.camchain for _, _, entry in camchain_entries(path)
    }
    started = time.monotonic()
    imu = load_ros2_imu(args.bag, args.imu_topic)
    gyro = GyroSeries(imu.times_s, imu.gyro_rps)
    options = CameraImuRunOptions()  # the check's tracking and evaluation settings
    solver_options = options.evaluation.solver
    per_camera: dict[str, dict[str, object]] = {}
    for index in (int(item) for item in args.cameras.split(",")):
        topic = f"/alphasense/cam{index}/image_raw"
        entry = entries[topic]
        candidate = np.asarray(entry["T_cam_imu"], dtype=np.float64)[:3, :3]
        windows, tracks = camera_rotation_windows(
            [args.bag], topic, CameraModel.from_kalibr(entry), options
        )
        if args.window_s is not None:
            windows = shorter_windows(tracks[0], args.window_s, options.visual.min_window_frames)
        moving = [
            window
            for window in windows
            if window_rotation_deg(window) >= options.evaluation.min_window_rotation_deg
        ]
        splits = {
            name: fit_split(subset, gyro, candidate, solver_options, args.blocks)
            for name, subset in split_windows(moving).items()
        }
        fitted = splits["all"].pop("_fit", None)
        for fit in splits.values():
            fit.pop("_fit", None)
        per_camera[f"cam{index}"] = {
            "windows": len(windows),
            "moving_windows": len(moving),
            "splits": splits,
            "linear_map": (
                linear_map(candidate, gyro, fitted, args.blocks) if fitted is not None else None
            ),
        }
        print(f"cam{index}", json.dumps(per_camera[f"cam{index}"]["splits"]["all"]), flush=True)
    rig = {split: pool(per_camera, split) for split in SPLITS}
    output = {
        "convention": (
            "offset_deg is w about the IMU axes with E = C exp(w): C Kalibr's R_cam_imu, "
            "E the estimate"
        ),
        "cameras": per_camera,
        "rig": rig,
        "rig_differences": differences(rig),
        "provenance": {
            "git_commit": commit,
            "tool_sha256": sha256_path(Path(__file__)),
            "bag": {"name": args.bag.name, "size_bytes": args.bag.stat().st_size},
            "camchain_sha256": {path.name: sha256_path(path) for path in args.camchain},
            "imu_topic": args.imu_topic,
            "window_s": args.window_s,
            "std_floor_deg": STD_FLOOR_DEG,
            "elapsed_s": round(time.monotonic() - started, 1),
        },
    }
    print("rig", json.dumps(rig), json.dumps(output["rig_differences"]), flush=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
