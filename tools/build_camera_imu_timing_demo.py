"""Export measured Hilti camera/gyro interval data for an interactive clock demo."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from generate_camera_imu_lock_gif import Params, interval_rmse_dps, load_series

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/assets/camera_imu_lock"
OUTPUT = ROOT / "docs/app/samples/camera_imu_timing.json"


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
    return {
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
        "method": (
            "t_imu = t_camera + saved_offset + injected_shift; fixed saved rotation and bias. "
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


if __name__ == "__main__":
    data = build_data()
    OUTPUT.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
    print(OUTPUT.name, OUTPUT.stat().st_size, len(data["camera_start_s"]), "camera intervals")
