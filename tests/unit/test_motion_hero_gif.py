"""Unit tests for motion hero GIF rendering helpers."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

requires_text_bake_tools = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("fc-match") is None,
    reason="text baking requires ffmpeg and fontconfig",
)

ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import motion_hero_gif  # noqa: E402


def test_compute_motion_bounds_uses_trajectory_margin_only() -> None:
    trajectory = ((0.0, 0.0, 0.0), (2.0, 1.0, 0.5))
    bounds = motion_hero_gif.compute_motion_bounds(trajectory, margin_m=4.0)
    assert bounds.x_min == pytest.approx(-4.0)
    assert bounds.x_max == pytest.approx(6.0)
    assert bounds.y_min == pytest.approx(-4.0)
    assert bounds.y_max == pytest.approx(5.0)


def test_build_motion_hero_frame_states_decouples_map_and_hud_progress() -> None:
    states = motion_hero_gif.build_motion_hero_frame_states(
        batch_count=108,
        accepted_batch_count=0,
        holdout_rmses=(0.3,) * 108,
        gate_statuses=("pass",) * 108,
        batch_transforms=(motion_hero_gif.SE3.identity(),) * 108,
        scan_count=420,
        frames=50,
    )
    assert states[0].scan_index == 0
    assert states[0].batch_index == 0
    assert states[-1].scan_index == 419
    assert states[-1].batch_index == 107
    mid = states[28]
    assert mid.scan_index > 200
    assert mid.batch_index > 50


def test_verify_motion_text_rendered_rejects_blank_title_band() -> None:
    width = motion_hero_gif.MOTION_HERO_WIDTH
    height = motion_hero_gif.MOTION_HERO_HEIGHT
    pixels = bytearray(motion_hero_gif.BG * width * height)
    frame_state = motion_hero_gif.MotionHeroFrameState(
        scan_index=0,
        batch_index=0,
        holdout_rmse_history=(0.25,),
        gate_statuses=("pass",),
        accepted_batch_count=1,
        batch_count=2,
        current_holdout_rmse_m=0.25,
        batch_transform=motion_hero_gif.SE3.identity(),
    )
    with pytest.raises(RuntimeError, match="text not rendered"):
        motion_hero_gif.verify_motion_text_rendered(pixels, width, height, frame_state)


@requires_text_bake_tools
def test_bake_motion_frame_text_renders_title_band(tmp_path: Path) -> None:
    width = motion_hero_gif.MOTION_HERO_WIDTH
    height = motion_hero_gif.MOTION_HERO_HEIGHT
    ppm_path = tmp_path / "frame.ppm"
    with ppm_path.open("wb") as handle:
        handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        handle.write(bytearray(motion_hero_gif.BG * width * height))
    frame_state = motion_hero_gif.MotionHeroFrameState(
        scan_index=0,
        batch_index=0,
        holdout_rmse_history=(0.25,),
        gate_statuses=("pass",),
        accepted_batch_count=1,
        batch_count=2,
        current_holdout_rmse_m=0.25,
        batch_transform=motion_hero_gif.SE3.identity(),
    )
    motion_hero_gif.bake_motion_frame_text(ppm_path, frame_state, width=width, height=height)
    pixels, read_width, read_height = motion_hero_gif.read_ppm_pixels(ppm_path)
    assert (read_width, read_height) == (width, height)
    motion_hero_gif.verify_motion_text_rendered(pixels, width, height, frame_state)


def test_resolve_motion_font_raises_when_fc_match_returns_missing_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_check_output(*_args: object, **_kwargs: object) -> str:
        return "/tmp/missing-motion-hero-font.ttf"

    monkeypatch.setattr(subprocess, "check_output", _fake_check_output)
    with pytest.raises(FileNotFoundError, match="font file missing"):
        motion_hero_gif.resolve_motion_font()


def test_gate_strip_uses_minimum_cell_size() -> None:
    assert motion_hero_gif.MOTION_GATE_CELL_PX >= 8
    assert motion_hero_gif.MOTION_GATE_GAP_PX >= 2
