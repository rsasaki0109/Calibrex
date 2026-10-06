"""Tests for the README workflow GIF generator and its committed transcript."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import workflow_gif as wf  # noqa: E402


def test_transcript_tells_estimate_check_drift() -> None:
    transcript = wf.load_transcript()
    steps = [act["step"] for act in transcript["acts"]]
    assert steps == ["estimate", "check", "drift"]
    commands = [act["command"] for act in transcript["acts"]]
    assert commands[0].startswith("calibrex estimate ")
    assert "--tf est/frames.yaml" in commands[1] and commands[1].startswith("calibrex check ")
    assert commands[2].startswith("calibrex drift ")
    assert transcript["output"] == wf.GIF.relative_to(ROOT).as_posix()
    for act in transcript["acts"]:
        assert len(act["full_output_sha256"]) == 64
        assert all("/media/" not in line and "/home/" not in line for line in act["lines"])


def test_transcript_lines_fit_the_terminal_panel() -> None:
    transcript = wf.load_transcript()
    mono = wf.load_mono(wf.FONT_SIZE)
    width = wf.PANEL[2] - wf.PANEL[0] - 28
    for act in transcript["acts"]:
        for line in act["lines"]:
            assert mono.getlength(line) <= width, line
        assert len(act["lines"]) * wf.LINE_H < wf.PANEL[3] - wf.PANEL[1] - 70


def test_transcript_outputs_keep_the_honest_parts() -> None:
    acts = {act["step"]: "\n".join(act["lines"]) for act in wf.load_transcript()["acts"]}
    assert "NOT MEASURED" in acts["estimate"]
    assert "partial" in acts["check"] and "unchecked" in acts["check"]
    assert "deviating bag(s)" in acts["drift"]


def test_gif_renders_within_budget(tmp_path: Path) -> None:
    out = tmp_path / "workflow.gif"
    assert wf.render(out) == 0
    assert 0 < out.stat().st_size <= wf.MAX_BYTES
