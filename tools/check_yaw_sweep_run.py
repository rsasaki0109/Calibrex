"""Run ``calibrex check`` for a sweep of yaw errors injected into the deployed tf.

For each injected angle the real CLI is run on the pooled KITTI dev bag with a
``--tf`` file written by ``tools/make_check_frames_perturbation.py`` (``velo_link``
turned by the angle about its parent's z axis, ``R' = Rz(yaw) R``). The estimator
cache (``--cache-dir``) makes every run after the first a few seconds. A small
per-step summary (verdict, per-pair status, coverage and per-axis errors) is written to
``--out-dir`` (``step_<angle>.json``) together with ``manifest.json`` (provenance), so
``tools/check_sweep_gifs.py`` can render the GIFs without re-running check::

    python tools/check_yaw_sweep_run.py --bag BAG --work-dir /tmp/sweep \\
        --out-dir docs/assets/calibrex_check_sweep
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SWEEP_SCHEMA = "calibrex.check_yaw_sweep_step/v0"
MANIFEST_SCHEMA = "calibrex.check_yaw_sweep_manifest/v0"
DEFAULT_PAIRS = "lidar-vehicle,lidar-wheel_odometry,ins-lidar,imu-vehicle"
CLI_SNIPPET = "import sys; from calibrex.cli.main import main; sys.exit(main())"
PATH = re.compile(r"(?<![\w.|])/(?:[^\s'\"<>/]+/)*[^\s'\"<>/]+")


def default_angles(limit: float = 3.0, step: float = 0.25) -> list[float]:
    """Angles from ``-limit`` to ``+limit`` in ``step`` increments (a half cycle)."""

    count = round(2 * limit / step)
    return [round(-limit + i * step, 4) for i in range(count + 1)]


def step_name(angle: float) -> str:
    """File stem for one step, e.g. ``step_-0.25`` / ``step_+3.00``."""

    return f"step_{angle:+.2f}"


def redact(text: str) -> str:
    """Shorten absolute paths to ``.../name`` so committed files carry no local paths."""

    return PATH.sub(lambda match: ".../" + match.group(0).rsplit("/", 1)[-1], text)


def sha256_file(path: Path) -> str:
    """SHA-256 of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(artifact: dict[str, Any], angle: float, command: list[str]) -> dict[str, Any]:
    """Reduce a ``calibrex check`` artifact JSON to the small per-step summary."""

    pairs: dict[str, Any] = {}
    for pair in artifact["pairs"]:
        if pair["status"] == "skipped":
            continue
        pairs[pair["pair"]] = {
            "status": pair["status"],
            "coverage": pair.get("coverage"),
            "axes": {
                axis["name"]: {
                    "candidate_error_deg": round(axis["candidate_error"], 4),
                    "tolerance_deg": round(axis["tolerance"], 4),
                    "ratio": round(axis["ratio"], 4),
                    "status": axis["status"],
                }
                for axis in pair["axes"]
            },
            "unchecked": [axis["name"] for axis in pair.get("unchecked_axes", [])],
        }
    provenance = artifact["provenance"]
    edge = next(
        e["transform"]
        for e in artifact["frame_tree"]["edges"]
        if e["transform"]["child_frame"] == "velo_link"
    )
    return {
        "schema": SWEEP_SCHEMA,
        "yaw_injected_deg": angle,
        "frame": "velo_link",
        "overall_verdict": artifact["overall_verdict"],
        "pairs": pairs,
        "velo_link_tf": {
            "parent_frame": edge["parent_frame"],
            "translation_m": edge["translation_m"],
            "rotation_quat_xyzw": edge["rotation_quat_xyzw"],
        },
        "command": [item if "=" in item else redact(item) for item in command],
        "created_at": provenance["created_at"],
        "generator_version": provenance["generator_version"],
        "git_commit": provenance.get("git_commit"),
        "bag_input_sha256": artifact["bag"]["input_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--bag", type=Path, required=True, help="pooled converted rosbag2")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, help="estimator cache (default WORK/cache)")
    parser.add_argument("--out-dir", type=Path, default=REPO / "docs/assets/calibrex_check_sweep")
    parser.add_argument("--limit", type=float, default=3.0)
    parser.add_argument("--step", type=float, default=0.25)
    parser.add_argument("--pairs", default=DEFAULT_PAIRS)
    parser.add_argument("--vehicle-frame", default="base_link")
    args = parser.parse_args(argv)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir or args.work_dir / "cache"
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    angles = default_angles(args.limit, args.step)
    angles.sort(key=abs)  # zero first: it runs the estimators once, the rest hit the cache
    summaries: list[dict[str, Any]] = []
    for angle in angles:
        frames = args.work_dir / f"frames_{step_name(angle)}.yaml"
        subprocess.run(
            [
                sys.executable,
                str(REPO / "tools/make_check_frames_perturbation.py"),
                str(args.bag),
                "velo_link",
                f"{angle:g}",
                "--output",
                str(frames),
            ],
            check=True,
            env=env,
        )
        output = args.work_dir / f"{step_name(angle)}.json"
        command = [
            "calibrex",
            "check",
            str(args.bag),
            "--vehicle-frame",
            args.vehicle_frame,
            "--topic-kind",
            "/oxts/twist=wheel",
            "--pairs",
            args.pairs,
            "--cache-dir",
            str(cache),
            "--tf",
            str(frames),
            "--output",
            str(output),
            "--fail-on",
            "never",
        ]
        started = time.monotonic()
        subprocess.run(
            [sys.executable, "-c", CLI_SNIPPET, *command[1:]],
            check=True,
            env=env,
        )
        artifact = json.loads(output.read_text(encoding="utf-8"))
        summary = summarize(artifact, angle, command)
        (args.out_dir / f"{step_name(angle)}.json").write_text(
            json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        summaries.append(summary)
        print(f"{angle:+.2f}: {summary['overall_verdict']} ({time.monotonic() - started:.0f} s)")
    first = summaries[0]
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "frame": "velo_link",
        "perturbation": "R' = Rz(yaw) * R about the parent (imu_link=base_link) z axis; "
        "translation unchanged (tools/make_check_frames_perturbation.py)",
        "angles_deg": sorted(s["yaw_injected_deg"] for s in summaries),
        "pairs": args.pairs.split(","),
        "bag_input_sha256": first["bag_input_sha256"],
        "generator_version": first["generator_version"],
        "git_commit": first["git_commit"],
        "drives": ["0005", "0009", "0014", "0015", "0022"],
        "dataset": "KITTI raw 2011_09_26 (CC BY-NC-SA 3.0, non-commercial use only)",
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
