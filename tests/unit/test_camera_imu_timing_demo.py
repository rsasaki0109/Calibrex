"""The timing experiment preserves measured Hilti intervals and existing metrics."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from build_camera_imu_timing_demo import OUTPUT, SOURCE, build_data, motion_windows  # noqa: E402
from generate_camera_imu_lock_gif import Params, interval_rmse_dps, load_series  # noqa: E402


def test_measured_timing_asset_and_source_identity() -> None:
    saved = json.loads(OUTPUT.read_text())
    generated = build_data()
    assert saved["provenance"] == generated["provenance"]
    for key in [
        "camera_start_s",
        "camera_end_s",
        "camera_rate_dps",
        "imu_time_s",
        "gyro_cumulative_camera_deg",
    ]:
        np.testing.assert_allclose(saved[key], generated[key], atol=1e-9, rtol=0)
    for actual, expected in zip(saved["references"], generated["references"], strict=True):
        assert actual["rmse_dps"] == pytest.approx(expected["rmse_dps"], abs=1e-9)
        np.testing.assert_allclose(
            actual["predicted_dps"], expected["predicted_dps"], atol=1e-9, rtol=0
        )
    assert saved["capture_kind"] == "real"
    assert len(saved["camera_start_s"]) == 59
    for path, value in saved["provenance"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == value


def test_rotation_references_use_camera_frame_left_multiplication() -> None:
    saved = json.loads(OUTPUT.read_text())
    series = load_series(SOURCE / "series.npz")
    for ref in saved["rotation_references"]:
        rotation = Rotation.from_euler("xyz"[ref["axis"]], ref["degrees"], degrees=True).as_matrix()
        value = interval_rmse_dps(
            series,
            Params(
                rotation @ series.r_est,
                series.dt_est_s + ref["shift_ms"] / 1000,
                series.bias_est_dps,
            ),
        )
        assert ref["rmse_dps"] == pytest.approx(value, abs=1e-9)


def test_motion_windows_are_selected_by_measured_amplitude_not_calibration_residuals() -> None:
    data = json.loads(OUTPUT.read_text())
    windows = motion_windows(data)
    high, low = windows[1:]
    assert len(high["indices"]) == len(low["indices"]) == 20
    assert not set(high["indices"]) & set(low["indices"])
    assert high["camera_rms_dps"] > low["camera_rms_dps"]
    rates = np.array(data["camera_rate_dps"])
    rms = [
        float(np.sqrt(np.mean(np.sum(rates[i : i + 20] ** 2, axis=1))))
        for i in range(len(rates) - 19)
    ]
    best = max(a - b for i, a in enumerate(rms) for j, b in enumerate(rms) if abs(i - j) >= 20)
    assert high["camera_rms_dps"] - low["camera_rms_dps"] == pytest.approx(best)
    for actual, saved in zip(windows, data["motion_windows"], strict=True):
        assert actual["indices"] == saved["indices"]
        assert actual["camera_rms_dps"] == pytest.approx(saved["camera_rms_dps"])


def test_window_reference_metrics_reconstruct_original_imu_integration() -> None:
    data = json.loads(OUTPUT.read_text())
    series = load_series(SOURCE / "series.npz")
    starts, ends = np.array(data["camera_start_s"]), np.array(data["camera_end_s"])
    camera_rates = np.array(data["camera_rate_dps"])
    for ref in data["window_references"]:
        ids = next(w["indices"] for w in data["motion_windows"] if w["id"] == ref["window"])
        a, b = (
            starts[ids] + series.dt_est_s + ref["shift_ms"] / 1000,
            ends[ids] + series.dt_est_s + ref["shift_ms"] / 1000,
        )
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
        rotation = Rotation.from_euler("xyz"[ref["axis"]], ref["degrees"], degrees=True).as_matrix()
        predicted = (means - series.bias_est_dps) @ (rotation @ series.r_est).T
        value = np.sqrt(np.mean(np.sum((predicted - camera_rates[ids]) ** 2, axis=1)))
        assert ref["rmse_dps"] == pytest.approx(value, abs=1e-9)


@pytest.mark.parametrize("shift_ms", [-50, -10, 0, 10, 50])
def test_timing_profile_uses_fixed_rotation_bias_and_complete_intervals(shift_ms: int) -> None:
    saved = json.loads(OUTPUT.read_text())
    series = load_series(SOURCE / "series.npz")
    assert saved["saved_offset_s"] == series.dt_est_s
    np.testing.assert_allclose(saved["fixed_rotation_camera_from_imu"], series.r_est)
    np.testing.assert_allclose(saved["fixed_gyro_bias_imu_dps"], series.bias_est_dps)
    start = np.array(saved["camera_start_s"]) + series.dt_est_s + shift_ms / 1000
    end = np.array(saved["camera_end_s"]) + series.dt_est_s + shift_ms / 1000
    assert start.min() >= series.imu_t_s[0]
    assert end.max() <= series.imu_t_s[-1]
    # Reconstruct in the original IMU frame, then rotate and subtract the saved
    # bias. The browser asset integrates in the camera frame instead.
    means = np.column_stack(
        [
            (
                np.interp(end, series.imu_t_s, series.gyro_cum_deg[:, k])
                - np.interp(start, series.imu_t_s, series.gyro_cum_deg[:, k])
            )
            / (end - start)
            for k in range(3)
        ]
    )
    expected = (means - series.bias_est_dps) @ series.r_est.T
    reference = next(r for r in saved["references"] if r["shift_ms"] == shift_ms)
    np.testing.assert_allclose(reference["predicted_dps"], expected, atol=1e-9, rtol=0)
    assert reference["rmse_dps"] == pytest.approx(
        interval_rmse_dps(
            series,
            Params(series.r_est, series.dt_est_s + shift_ms / 1000, series.bias_est_dps),
        ),
        abs=1e-10,
    )
