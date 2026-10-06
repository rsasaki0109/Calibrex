"""Render the README "estimate, check, drift" workflow GIF from real, abridged command outputs.

The three acts are terminal recordings of ``calibrex estimate``, ``calibrex check --tf`` and
``calibrex drift`` on the Koide hard-localization ``indoor_easy_01`` / ``indoor_easy_02``
recordings (plus ``indoor_easy_01`` with its IMU rotated 2 degrees by
``tools/rotate_imu_in_bag.py``).  Every output line comes from
``docs/assets/calibrex_workflow/transcript.json``, which holds the verbatim lines of the real runs
(omitted lines are marked ``...``) and the SHA-256 of each full output.  Nothing is simulated;
the GIF only types the command and reveals the recorded lines.

Usage (needs numpy and Pillow)::

    python tools/workflow_gif.py render
    python tools/workflow_gif.py preview --out-dir /tmp/previews
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_sweep_gifs as sweep

REPO = Path(__file__).resolve().parents[1]
TRANSCRIPT = REPO / "docs/assets/calibrex_workflow/transcript.json"
GIF = REPO / "docs/assets/calibrex-workflow.gif"
MAX_BYTES = 1_500_000
SCHEMA = "calibrex.workflow_gif_transcript/v0"

BG = sweep.BG
PANEL_BG: sweep.Color = (13, 17, 23)
BORDER = sweep.BORDER
TEXT = sweep.TEXT
MUTED = sweep.MUTED
DIM: sweep.Color = (110, 120, 136)
PROMPT: sweep.Color = (88, 166, 255)
GREEN = sweep.STATUS_COLOR["pass"]
AMBER = sweep.STATUS_COLOR["warn"]
RED = sweep.STATUS_COLOR["fail"]

STEP_LABELS = ("1  estimate", "2  check", "3  drift")
LINE_H = 20
FONT_SIZE = 14
PANEL = (16, 64, sweep.WIDTH - 16, 488)
TYPE_FRAMES = 6
TYPE_MS = 90
REVEAL_MS = 170
REVEAL_CHUNK = 3
HOLD_MS = (3200, 4200, 4800)
FIRST_PAUSE_MS = 500

# (pattern, colour): the first match wins at each position.
TOKENS: tuple[tuple[str, sweep.Color], ...] = (
    (r"NOT MEASURED|unobservable|WARNING", AMBER),
    (r"\bfail\b|\bdrift\b|deviating bag\(s\): \S+", RED),
    (r"\bpass\b|\bstable\b|\bobserved\b|\bestimated\b", GREEN),
    (r"\.\.\.$", DIM),
)


def load_transcript(path: Path = TRANSCRIPT) -> dict[str, Any]:
    """Read and sanity-check the transcript."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise SystemExit(f"{path} is not a {SCHEMA} transcript")
    return data


def load_mono(size: int) -> Any:
    """DejaVu Sans Mono via Pillow, falling back to Pillow's default font."""

    _, _, image_font = sweep._pil()
    try:
        return image_font.truetype("DejaVuSansMono.ttf", size)
    except OSError:
        return image_font.load_default()


def colour_runs(line: str) -> list[tuple[str, sweep.Color]]:
    """Split a line into (text, colour) runs for the status words."""

    pattern = re.compile("|".join(f"({p})" for p, _ in TOKENS))
    runs: list[tuple[str, sweep.Color]] = []
    pos = 0
    for match in pattern.finditer(line):
        if match.start() > pos:
            runs.append((line[pos : match.start()], TEXT))
        index = next(i for i, g in enumerate(match.groups()) if g is not None)
        runs.append((match.group(0), TOKENS[index][1]))
        pos = match.end()
    if pos < len(line):
        runs.append((line[pos:], TEXT))
    return runs


def draw_chips(draw: Any, active: int) -> None:
    """Three step chips; the active one is lit."""

    font = sweep.load_font(13, True)
    x = 16
    y = 16
    chip_w = (sweep.WIDTH - 32 - 2 * 14) // 3
    for index, label in enumerate(STEP_LABELS, start=1):
        lit = index == active
        fill = (52, 64, 86) if lit else (28, 34, 44)
        outline = (120, 170, 255) if lit else BORDER
        draw.rounded_rectangle((x, y, x + chip_w, y + 32), radius=8, fill=fill, outline=outline)
        draw.text(
            (x + chip_w // 2, y + 16), label, font=font, fill=TEXT if lit else DIM, anchor="mm"
        )
        x += chip_w + 14


def draw_frame(act_index: int, act: dict[str, Any], typed: int, shown: int, caption: bool) -> Any:
    """One frame: the command typed to ``typed`` characters and ``shown`` output lines."""

    image_mod, draw_mod, _ = sweep._pil()
    image = image_mod.new("RGB", (sweep.WIDTH, sweep.HEIGHT), BG)
    draw = draw_mod.Draw(image)
    draw_chips(draw, act_index + 1)
    draw.rounded_rectangle(PANEL, radius=10, fill=PANEL_BG, outline=BORDER)
    mono = load_mono(FONT_SIZE)
    char_w = mono.getlength("M")
    x0 = PANEL[0] + 14
    y = PANEL[1] + 12
    command = act["command"]
    draw.text((x0, y), "$ ", font=mono, fill=PROMPT)
    command_shown = command[:typed]
    # wrap long commands at the panel width
    max_chars = int((PANEL[2] - PANEL[0] - 28) / char_w) - 2
    parts = [
        command_shown[i : i + max_chars] for i in range(0, max(len(command_shown), 1), max_chars)
    ]
    for part in parts:
        draw.text((x0 + 2 * char_w, y), part, font=mono, fill=TEXT)
        y += LINE_H
    y += 4
    for line in act["lines"][:shown]:
        x = x0
        for text, colour in colour_runs(line):
            draw.text((x, y), text, font=mono, fill=colour)
            x += mono.getlength(text)
        y += LINE_H
    if caption:
        font = sweep.load_font(14)
        draw.text(
            (sweep.WIDTH // 2, 512),
            act["caption"],
            font=font,
            fill=MUTED,
            anchor="mm",
        )
    return image


def build_frames(transcript: dict[str, Any]) -> tuple[list[Any], list[int]]:
    """All frames and their durations in milliseconds."""

    frames: list[Any] = []
    durations: list[int] = []
    for index, act in enumerate(transcript["acts"]):
        command = act["command"]
        steps = [round(len(command) * (k + 1) / TYPE_FRAMES) for k in range(TYPE_FRAMES)]
        frames.append(draw_frame(index, act, 0, 0, True))
        durations.append(FIRST_PAUSE_MS)
        for typed in steps:
            frames.append(draw_frame(index, act, typed, 0, True))
            durations.append(TYPE_MS)
        total = len(act["lines"])
        shown = 0
        while shown < total:
            shown = min(total, shown + REVEAL_CHUNK)
            frames.append(draw_frame(index, act, len(command), shown, True))
            durations.append(REVEAL_MS)
        durations[-1] = HOLD_MS[index % len(HOLD_MS)]
    return frames, durations


def render(output: Path) -> int:
    """Render the GIF and enforce the size budget."""

    frames, durations = build_frames(load_transcript())
    sweep.save_gif(frames, durations, output)
    size = output.stat().st_size
    print(f"{output}: {len(frames)} frames, {size} bytes")
    if size > MAX_BYTES:
        raise SystemExit(f"GIF is {size} bytes, over the {MAX_BYTES} budget")
    return 0


def preview(out_dir: Path) -> int:
    """Write the last frame of every act as a PNG for inspection."""

    out_dir.mkdir(parents=True, exist_ok=True)
    transcript = load_transcript()
    for index, act in enumerate(transcript["acts"]):
        path = out_dir / f"act{index + 1}_{act['step']}.png"
        draw_frame(index, act, len(act["command"]), len(act["lines"]), True).save(path)
        print(path)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    render_p = sub.add_parser("render")
    render_p.add_argument("--output", type=Path, default=GIF)
    preview_p = sub.add_parser("preview")
    preview_p.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "render":
        return render(args.output)
    return preview(args.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
