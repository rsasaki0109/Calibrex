# ruff: noqa: E501
"""Render ``docs/assets/readme-check-hero.svg`` from the committed ``calibrex check`` demo.

Every number and verdict in the SVG is read from
``docs/assets/calibrex_check_demo/summary.json``; the sources are bound into the SVG
``<metadata>`` by SHA-256. Output is deterministic (no timestamps), so a unit test can
require the committed file to equal a fresh render::

    python tools/generate_readme_check_hero.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from html import escape
from pathlib import Path
from typing import Any

HERO_VERSION = "1"
ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = Path("docs/assets/calibrex_check_demo")
SOURCES = ("summary.json", "summary.md", "terminal_summary.txt")
OUTPUT = Path("docs/assets/readme-check-hero.svg")
PAIR_ORDER = ("lidar-vehicle", "lidar-wheel_odometry", "ins-lidar", "imu-vehicle")
SCALE = 3.0  # bar spans 0..SCALE x tolerance

WIDTH = 960
COL_W = 424
COL_X = (28, 508)
HEIGHT = 576

STYLE = """
    :root { --win:#f6f8fa; --bar:#e4e8ec; --edge:#d0d7de; --fg:#1f2328; --mute:#57606a;
      --track:#d8dee4; --tick:#57606a; --pass:#1a7f37; --warn:#9a6700; --fail:#cf222e;
      --inc:#6e7781; --pillfg:#ffffff; --prompt:#0969da; }
    @media (prefers-color-scheme: dark) {
      :root { --win:#0d1117; --bar:#161b22; --edge:#30363d; --fg:#e6edf3; --mute:#8b949e;
        --track:#21262d; --tick:#8b949e; --pass:#2ea043; --warn:#d29922; --fail:#f85149;
        --inc:#6e7681; --pillfg:#0d1117; --prompt:#58a6ff; }
    }
    text { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "DejaVu Sans Mono", monospace;
      fill: var(--fg); }
    .mute { fill: var(--mute); }
    .h { font-size: 15px; font-weight: 700; }
    .t { font-size: 14px; }
    .s { font-size: 13px; }
    .pill-text { font-size: 13px; font-weight: 700; fill: var(--pillfg); text-anchor: middle; }
    .pass { fill: var(--pass); } .warn { fill: var(--warn); }
    .fail { fill: var(--fail); } .inconclusive { fill: var(--inc); }
    .prompt { fill: var(--prompt); font-weight: 700; }
""".strip("\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fmt(value: float) -> str:
    return f"{value:.3g}"


def _tol(value: float) -> str:
    return f"{value:g}"


def _short_command(command: list[str]) -> str:
    """The command with paths shortened and the demo-only flags (topic mapping, pair list) dropped."""
    out: list[str] = []
    skip = False
    for token in command:
        if skip:
            skip = False
            continue
        if token in ("--topic-kind", "--pairs"):
            skip = True
            continue
        out.append(token.rsplit("/", 1)[-1] if token.startswith(".../") else token)
    return " ".join(out)


def _worst_axis(pair: dict[str, Any]) -> tuple[str, float, float, str] | None:
    """(axis, |delta|, tolerance, status) of the axis closest to or beyond its tolerance."""
    best: tuple[str, float, float, str] | None = None
    for name, axis in pair["axes"].items():
        err = abs(float(axis["candidate_error_deg"]))
        tol = float(axis["tolerance_deg"])
        if best is None or err / tol > best[1] / best[2]:
            best = (name, err, tol, str(axis["status"]))
    return best


def _pill(x: float, y: float, status: str) -> str:
    label = status.upper() if status != "inconclusive" else "INCONCLUSIVE"
    width = 128 if status == "inconclusive" else 62
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="22" rx="11" class="{status}"/>'
        f'<text x="{x + width / 2}" y="{y + 16}" class="pill-text">{label}</text>'
    )


def _column(x: int, variant: dict[str, Any], title: str, command: str) -> list[str]:
    parts = [f'<g transform="translate({x} 168)">']
    parts.append(f'<text x="0" y="0" class="h">{escape(title)}</text>')
    parts.append(f'<text x="0" y="22" class="s mute">{escape(command)}</text>')
    row_y = 46
    for name in PAIR_ORDER:
        pair = variant["pairs"][name]
        status = str(pair["status"])
        y = row_y
        parts.append(f'<text x="0" y="{y + 16}" class="t">{escape(name)}</text>')
        parts.append(_pill(214, y, status))
        if pair["coverage"] == "partial" and status != "inconclusive":
            parts.append(f'<text x="284" y="{y + 16}" class="s mute">&#9680; partial</text>')
        worst = _worst_axis(pair)
        bar_y = y + 32
        parts.append(f'<rect x="0" y="{bar_y}" width="260" height="8" rx="4" fill="var(--track)"/>')
        if worst is None:
            parts.append(f'<text x="270" y="{bar_y + 9}" class="s mute">no axis judged</text>')
        else:
            axis, err, tol, axis_status = worst
            fill = min(err / tol / SCALE, 1.0) * 260
            parts.append(
                f'<rect x="0" y="{bar_y}" width="{fill:.1f}" height="8" rx="4" class="{axis_status}"/>'
            )
            parts.append(
                f'<rect x="{260 / SCALE:.1f}" y="{bar_y - 4}" width="2" height="16" fill="var(--tick)"/>'
            )
            parts.append(
                f'<text x="270" y="{bar_y + 9}" class="s">{axis} {_fmt(err)}/{_tol(tol)}&#176;</text>'
            )
        row_y += 62
    overall = str(variant["overall_verdict"])
    parts.append(f'<rect x="0" y="{row_y - 2}" width="{COL_W}" height="1" fill="var(--edge)"/>')
    parts.append(f'<text x="0" y="{row_y + 28}" class="h">overall:</text>')
    parts.append(_pill(84, row_y + 11, overall))
    parts.append("</g>")
    return parts


def render_hero(root: Path = ROOT) -> str:
    demo = root / DEMO_DIR
    summary = json.loads((demo / "summary.json").read_text(encoding="utf-8"))
    vendor, injected = summary["variants"]["vendor"], summary["variants"]["yaw1"]
    inj_deg = float(injected["yaw_injected_deg"])
    yaw = injected["pairs"]["lidar-vehicle"]["axes"]["yaw"]
    drives = ", ".join(d.split("_drive_")[1].split("_")[0] for d in summary["drives"])
    sources = {(DEMO_DIR / name).as_posix(): _sha256(demo / name) for name in SOURCES}
    provenance = json.dumps(
        {
            "artifact": "calibrex.readme-check-hero",
            "calibrex_version": summary["generator_version"],
            "demo_git_commit": summary["git_commit"],
            "demo_schema": summary["schema"],
            "generator": "tools/generate_readme_check_hero.py",
            "generator_version": HERO_VERSION,
            "sources": sources,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    frame = summary["frame"]
    title = "calibrex check catches a 1 degree yaw error in a deployed tf"
    desc = (
        f"On pooled KITTI development drives, the vendor tf is overall "
        f"{vendor['overall_verdict']}; with velo_link yaw +{_tol(inj_deg)} deg the overall verdict is "
        f"{injected['overall_verdict']} (yaw {_fmt(abs(yaw['candidate_error_deg']))} deg against a "
        f"{_tol(yaw['tolerance_deg'])} deg tolerance)."
    )
    cmd_a = _short_command(vendor["command"])
    cmd_b = _short_command(injected["command"])
    flag = cmd_b.replace(cmd_a, "").strip()
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title description">',
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="description">{escape(desc)}</desc>',
        f'<metadata id="calibrex-provenance">{escape(provenance, quote=True)}</metadata>',
        f"<style>\n{STYLE}\n</style>",
        f'<rect x="1" y="1" width="{WIDTH - 2}" height="{HEIGHT - 2}" rx="14" '
        'fill="var(--win)" stroke="var(--edge)" stroke-width="2"/>',
        f'<path d="M1 15a14 14 0 0 1 14-14h{WIDTH - 30}a14 14 0 0 1 14 14v26H1z" fill="var(--bar)"/>',
        '<circle cx="28" cy="22" r="6" fill="#ff5f57"/><circle cx="48" cy="22" r="6" fill="#febc2e"/>'
        '<circle cx="68" cy="22" r="6" fill="#28c840"/>',
        f'<text x="{WIDTH / 2}" y="27" class="s mute" text-anchor="middle">'
        f"calibrex check &#183; KITTI development drives {escape(drives)} (pooled)</text>",
        f'<text x="28" y="78" class="t"><tspan class="prompt">$</tspan> {escape(cmd_a)}</text>',
        f'<text x="28" y="104" class="t"><tspan class="prompt">$</tspan> {escape(cmd_a)} '
        f'<tspan class="prompt">{escape(flag)}</tspan></text>',
        f'<text x="28" y="130" class="s mute">verdict per sensor pair; bar = worst judged axis, '
        f"tick = tolerance, full width = {_tol(SCALE)}x tolerance</text>",
    ]
    out += _column(COL_X[0], vendor, "deployed tf: vendor calibration", "no override")
    out += _column(
        COL_X[1],
        injected,
        f"deployed tf: {summary['frame']} yaw +{_tol(inj_deg)}\u00b0",
        escape(flag),
    )
    out.append(f'<rect x="{COL_X[1] - 20}" y="158" width="1" height="330" fill="var(--edge)"/>')
    out.append(
        f'<text x="28" y="{HEIGHT - 52}" class="t">'
        f"A {_tol(inj_deg)}&#176; yaw error in the deployed tf on {escape(frame)} is caught: "
        f"yaw {_fmt(abs(yaw['candidate_error_deg']))}&#176; vs {_tol(yaw['tolerance_deg'])}&#176; tolerance.</text>"
    )
    out.append(
        f'<text x="28" y="{HEIGHT - 26}" class="s mute">'
        "partial = some axes not judged; development data, not a held-out claim</text>"
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / OUTPUT)
    args = parser.parse_args()
    args.output.write_text(render_hero(), encoding="utf-8")


if __name__ == "__main__":
    main()
