"""Export measured Hilti camera/gyro interval data for an interactive clock demo."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from generate_camera_imu_lock_gif import Params, exp_so3, interval_rmse_dps, load_series

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/assets/camera_imu_lock"
OUTPUT = ROOT / "docs/app/samples/camera_imu_timing.json"


def motion_windows(data: dict[str, Any], count: int = 20) -> list[dict[str, Any]]:
    """Select disjoint equal-count windows using only measured camera-rate RMS."""
    rates = np.array(data["camera_rate_dps"])
    candidates = [
        (i, float(np.sqrt(np.mean(np.sum(rates[i : i + count] ** 2, axis=1)))))
        for i in range(len(rates) - count + 1)
    ]
    pairs = [
        (high[1] - low[1], high, low)
        for high in candidates
        for low in candidates
        if abs(high[0] - low[0]) >= count
    ]
    if not pairs:
        raise ValueError("Need two disjoint measured windows")
    gap, high, low = max(pairs, key=lambda p: p[0])
    if gap <= 1e-12:
        raise ValueError("No measured angular-rate RMS contrast")
    windows = []
    for code, indices in [
        ("all", list(range(len(rates)))),
        ("higher", list(range(high[0], high[0] + count))),
        ("lower", list(range(low[0], low[0] + count))),
    ]:
        windows.append(
            {
                "id": code,
                "indices": indices,
                "camera_rms_dps": float(np.sqrt(np.mean(np.sum(rates[indices] ** 2, axis=1)))),
                "start_s": data["camera_start_s"][indices[0]],
                "end_s": data["camera_end_s"][indices[-1]],
            }
        )
    return windows


def build_data() -> dict[str, Any]:
    """Keep one common camera interval set valid across all injected offsets."""
    series = load_series(SOURCE / "series.npz")
    keep = (series.cam_start_s >= 1) & (series.cam_end_s <= 7)
    start, end = series.cam_start_s[keep], series.cam_end_s[keep]
    if (start + series.dt_est_s - 0.05).min() < series.imu_t_s[0] or (
        end + series.dt_est_s + 0.05
    ).max() > series.imu_t_s[-1]:
        raise ValueError("The full timing range needs extrapolation")
    # Sampled trapezoidal cumulative integral in the camera frame. Linear
    # interpolation of these cumulative samples matches the existing GIF metric.
    cumulative = series.gyro_cum_deg @ series.r_est.T
    cumulative -= (series.imu_t_s - series.imu_t_s[0])[:, None] * (
        series.bias_est_dps @ series.r_est.T
    )
    references = []
    for ms in [-50, -10, 0, 10, 50]:
        shifted_start = start + series.dt_est_s + ms / 1000
        shifted_end = end + series.dt_est_s + ms / 1000
        predicted = np.column_stack(
            [
                (
                    np.interp(shifted_end, series.imu_t_s, cumulative[:, k])
                    - np.interp(shifted_start, series.imu_t_s, cumulative[:, k])
                )
                / (end - start)
                for k in range(3)
            ]
        )
        references.append(
            {
                "shift_ms": ms,
                "predicted_dps": predicted.tolist(),
                "rmse_dps": interval_rmse_dps(
                    series,
                    Params(series.r_est, series.dt_est_s + ms / 1000, series.bias_est_dps),
                ),
            }
        )
    sources = [
        SOURCE / "series.npz",
        SOURCE / "series.provenance.json",
        Path(__file__),
        ROOT / "tools/generate_camera_imu_lock_gif.py",
    ]
    rotation_references = []
    for axis in range(3):
        for degrees in [-5, 2.5, 5]:
            vector = np.zeros(3)
            vector[axis] = np.deg2rad(degrees)
            rotation = exp_so3(vector) @ series.r_est
            for ms in [-50, 0, 50]:
                rotation_references.append(
                    {
                        "axis": axis,
                        "degrees": degrees,
                        "shift_ms": ms,
                        "rmse_dps": interval_rmse_dps(
                            series,
                            Params(rotation, series.dt_est_s + ms / 1000, series.bias_est_dps),
                        ),
                    }
                )
    data = {
        "schema": "calibrex.camera_imu_timing_demo/v1",
        "capture_kind": "real",
        "dataset": "Hilti 2022 exp21 / cam0 and alphasense IMU",
        "source_provenance": "../assets/camera_imu_lock/series.provenance.json",
        "license": "Hilti SLAM Challenge 2022 terms",
        "camera_start_s": start.tolist(),
        "camera_end_s": end.tolist(),
        "camera_rate_dps": series.cam_rate_dps[keep].tolist(),
        "imu_time_s": series.imu_t_s.tolist(),
        "gyro_cumulative_camera_deg": cumulative.tolist(),
        "saved_offset_s": series.dt_est_s,
        "fixed_rotation_camera_from_imu": series.r_est.tolist(),
        "fixed_gyro_bias_imu_dps": series.bias_est_dps.tolist(),
        "shift_range_ms": [-50, 50],
        "references": references,
        "rotation_range_deg": [-5, 5],
        "rotation_step_deg": 0.25,
        "rotation_references": rotation_references,
        "method": (
            "t_imu = t_camera + saved_offset + injected_shift. "
            "R_candidate = R_camera_axis(injected_angle) @ R_saved; fixed IMU-frame gyro bias. "
            "Interval means use linear interpolation of sampled trapezoidal cumulative "
            "gyro integrals. "
            "The same camera intervals are used at every shift; no extrapolation."
        ),
        "limitations": (
            "Displayed development window only. Zero injection is the saved estimate, "
            "not clock truth. "
            "Camera tracking noise, timing, rotation and bias can all contribute to residuals. "
            "No optimizer, independent accuracy score or new calibration verdict."
        ),
        "provenance": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
    }
    data["motion_windows"] = motion_windows(data)
    data["window_selection"] = (
        "Two disjoint contiguous 20-observation windows maximizing the difference in "
        "measured camera angular-rate RMS; ties follow ascending candidate start order. "
        "No gyro residual, saved calibration or heatmap minimum enters selection. "
        "RMS amplitude is not a test of observability or calibration accuracy."
    )
    references = []
    for window in data["motion_windows"]:
        ids = window["indices"]
        for axis in [0, 2]:
            for angle, shift in [(-2.5, -25), (0, 0), (2.5, 25)]:
                vector = np.zeros(3)
                vector[axis] = np.deg2rad(angle)
                rotation = exp_so3(vector) @ series.r_est
                a = start[ids] + series.dt_est_s + shift / 1000
                b = end[ids] + series.dt_est_s + shift / 1000
                means = np.column_stack(
                    [
                        (
                            np.interp(b, series.imu_t_s, series.gyro_cum_deg[:, k])
                            - np.interp(a, series.imu_t_s, series.gyro_cum_deg[:, k])
                        )
                        / (b - a)
                        for k in range(3)
                    ]
                )
                error = (means - series.bias_est_dps) @ rotation.T - series.cam_rate_dps[keep][ids]
                references.append(
                    {
                        "window": window["id"],
                        "axis": axis,
                        "degrees": angle,
                        "shift_ms": shift,
                        "rmse_dps": float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
                    }
                )
    data["window_references"] = references
    return data


if __name__ == "__main__":
    data = build_data()
    OUTPUT.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
    print(OUTPUT.name, OUTPUT.stat().st_size, len(data["camera_start_s"]), "camera intervals")
