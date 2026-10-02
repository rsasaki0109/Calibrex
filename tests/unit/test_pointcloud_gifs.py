"""The point-cloud README GIFs regenerate from their committed intermediates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("PIL")
pytest.importorskip("scipy")

ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import lidar_lidar_snap_gif  # noqa: E402
import mid360_deskew_gif  # noqa: E402
from PIL import Image  # noqa: E402

MAX_BYTES = int(1.5 * 1024 * 1024)


def _check_gif(path: Path, frames: int) -> None:
    with Image.open(path) as gif:
        assert gif.size == (960, 540)
        assert gif.n_frames == frames
        assert gif.info["loop"] == 0
        assert gif.info["duration"] == 100
    assert path.stat().st_size <= MAX_BYTES


def test_schedules_loop_seamlessly() -> None:
    snap = lidar_lidar_snap_gif.schedule()
    assert len(snap) == lidar_lidar_snap_gif.FRAMES
    assert snap[0][0] == 0.0
    assert snap[-1][0] < 0.1  # eases back to the start frame
    deskew = mid360_deskew_gif.schedule()
    assert len(deskew) == mid360_deskew_gif.FRAMES
    assert deskew[0][0] == 0.0
    assert deskew[-1][0] < 0.1


def test_se3_interpolation_endpoints() -> None:
    estimate = lidar_lidar_snap_gif.read_transform(lidar_lidar_snap_gif.DEFAULT_ARTIFACT)
    start = lidar_lidar_snap_gif.perturbed_start(estimate)
    at_start = lidar_lidar_snap_gif.interpolate(start, estimate, 0.0)
    at_end = lidar_lidar_snap_gif.interpolate(start, estimate, 1.0)
    assert abs(at_start - start).max() < 1.0e-9
    assert abs(at_end - estimate).max() < 1.0e-9
    rotation_deg, translation_m = lidar_lidar_snap_gif.offset_to(estimate, start)
    assert 4.0 < rotation_deg < 8.0
    assert 0.2 < translation_m < 0.5


@pytest.mark.skipif(
    not lidar_lidar_snap_gif.DEFAULT_NPZ.exists(), reason="committed intermediate is absent"
)
def test_lidar_lidar_snap_gif_renders(tmp_path: Path) -> None:
    gif = tmp_path / "snap.gif"
    manifest = tmp_path / "manifest.json"
    info = lidar_lidar_snap_gif.render(
        lidar_lidar_snap_gif.DEFAULT_NPZ,
        lidar_lidar_snap_gif.DEFAULT_ARTIFACT,
        gif,
        None,
        manifest,
    )
    _check_gif(gif, lidar_lidar_snap_gif.FRAMES)
    assert info["estimate_residual_m"] < info["start_residual_m"]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["dataset"]["license"] == "CC BY-NC-SA 4.0"
    assert len(payload["sha256"]) == 64


@pytest.mark.skipif(
    not mid360_deskew_gif.DEFAULT_NPZ.exists(), reason="committed intermediate is absent"
)
def test_mid360_deskew_gif_renders(tmp_path: Path) -> None:
    gif = tmp_path / "deskew.gif"
    manifest = tmp_path / "manifest.json"
    info = mid360_deskew_gif.render(mid360_deskew_gif.DEFAULT_NPZ, gif, None, manifest)
    _check_gif(gif, mid360_deskew_gif.FRAMES)
    assert info["thickness_deskewed_cm"] < info["thickness_raw_cm"]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["dataset"]["license"].startswith("CC BY 4.0")
