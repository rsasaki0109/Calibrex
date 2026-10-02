from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

pytest.importorskip("PIL")
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "generate_camera_imu_lock_gif.py"
SERIES = ROOT / "docs" / "assets" / "camera_imu_lock" / "series.npz"
PROVENANCE = ROOT / "docs" / "assets" / "camera_imu_lock" / "series.provenance.json"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("generate_camera_imu_lock_gif", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_series_is_small_and_provenance_complete() -> None:
    assert SERIES.stat().st_size < 1_000_000
    manifest = json.loads(PROVENANCE.read_text())
    assert manifest["dataset"]["sequence"] == "exp21_ros2"
    assert manifest["dataset"]["license"] == "Hilti SLAM Challenge 2022 terms"
    hashes = manifest["inputs_sha256"]
    assert all(len(hashes[key]) == 64 for key in ("camchain", "estimate_artifact"))
    assert manifest["calibrex_version"] and manifest["commands"]


def test_estimate_locks_curves_and_matches_kalibr() -> None:
    tool = _load()
    series = tool.load_series(SERIES)
    start = tool.interval_rmse_dps(series, tool.interpolate(series, 0.0))
    end = tool.interval_rmse_dps(series, tool.interpolate(series, 1.0))
    assert end < 0.6 * start
    rot_diff = tool.rotation_angle_deg(series.r_est.T @ series.r_kalibr)
    assert rot_diff == pytest.approx(0.45, abs=0.1)
    assert (series.dt_est_s - series.dt_kalibr_s) * 1e3 == pytest.approx(-0.21, abs=0.02)
    assert np.isfinite(series.cam_rate_dps).all()


def test_generator_writes_expected_gif(tmp_path: Path) -> None:
    tool = _load()
    output = tmp_path / "lock.gif"
    series = tool.load_series(SERIES)
    frames = tool.build_frames(series)
    tool.save_gif(frames, output)
    assert len(frames) == tool.N_FRAMES
    with Image.open(output) as gif:
        assert gif.size == (960, 540)
        # Pillow merges identical consecutive (hold) frames, summing their durations.
        assert 60 <= gif.n_frames <= tool.N_FRAMES
        total_ms = 0
        for index in range(gif.n_frames):
            gif.seek(index)
            total_ms += int(gif.info["duration"])
        assert total_ms == tool.N_FRAMES * tool.FRAME_MS
        assert gif.info["loop"] == 0
    assert output.stat().st_size <= 1_500_000
