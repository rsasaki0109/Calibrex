"""Run the two real ``calibrex check`` runs behind the "check story" GIF.

Act 1 deploys a wrong tf (``velo_link`` turned by ``--yaw-deg`` about the parent z axis)
and checks it. Act 3 restores the calibration file (the KITTI vendor tf from the bag's
``/tf_static``, i.e. the same perturbation with a 0 deg yaw) and checks again. Calibrex
checks the calibrations here; it is not shown estimating or fixing anything. Summaries
are written to ``--out-dir`` so ``tools/check_story_gif.py`` renders without re-running
check::

    python tools/check_story_run.py --bag BAG --work-dir /tmp/story \\
        --out-dir docs/assets/calibrex_check_story
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_yaw_sweep_run import CLI_SNIPPET, sha256_file, summarize

REPO = Path(__file__).resolve().parents[1]
STORY_SCHEMA = "calibrex.check_story_summary/v0"
MANIFEST_SCHEMA = "calibrex.check_story_manifest/v0"
PAIRS = "lidar-vehicle,lidar-wheel_odometry,ins-lidar,imu-vehicle"
FRAME = "velo_link"


def run_check(
    bag: Path, cache: Path, frames: Path, output: Path, env: dict[str, str]
) -> tuple[dict[str, Any], list[str]]:
    """Run the real ``calibrex check`` CLI and return its artifact and command."""

    command = [
        "calibrex",
        "check",
        str(bag),
        "--vehicle-frame",
        "base_link",
        "--topic-kind",
        "/oxts/twist=wheel",
        "--pairs",
        PAIRS,
        "--cache-dir",
        str(cache),
        "--tf",
        str(frames),
        "--output",
        str(output),
        "--fail-on",
        "never",
    ]
    subprocess.run([sys.executable, "-c", CLI_SNIPPET, *command[1:]], check=True, env=env)
    return json.loads(output.read_text(encoding="utf-8")), command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=REPO / "docs/assets/calibrex_check_story")
    parser.add_argument("--yaw-deg", type=float, default=3.0)
    args = parser.parse_args(argv)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir or args.work_dir / "cache"
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))

    frames = {}
    for act, yaw in (("act1", args.yaw_deg), ("act3", 0.0)):
        frames[act] = args.work_dir / f"frames_{act}.yaml"
        subprocess.run(
            [
                sys.executable,
                str(REPO / "tools/make_check_frames_perturbation.py"),
                str(args.bag),
                FRAME,
                f"{yaw:g}",
                "--output",
                str(frames[act]),
            ],
            check=True,
            env=env,
        )
    art1, cmd1 = run_check(args.bag, cache, frames["act1"], args.work_dir / "act1.json", env)
    art3, cmd3 = run_check(args.bag, cache, frames["act3"], args.work_dir / "act3.json", env)

    outputs = {
        "act1_deployed_wrong.json": summarize(art1, args.yaw_deg, cmd1),
        "act3_recheck_vendor.json": summarize(art3, 0.0, cmd3),
    }
    for name, summary in outputs.items():
        summary["schema"] = STORY_SCHEMA
        (args.out_dir / name).write_text(
            json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(name, summary["overall_verdict"], summary["pairs"]["lidar-vehicle"]["status"])
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "frame": FRAME,
        "injected_yaw_deg": args.yaw_deg,
        "act3_deployed_tf": "the bag's /tf_static (KITTI vendor calibration), yaw 0 deg",
        "bag_input_sha256": art1["bag"]["input_sha256"],
        "summary_sha256": {n: sha256_file(args.out_dir / n) for n in outputs},
        "dataset": "KITTI raw 2011_09_26 (CC BY-NC-SA 3.0, non-commercial use only)",
        "drives": ["0005", "0009", "0014", "0015", "0022"],
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
