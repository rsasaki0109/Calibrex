"""Run the fixed fourteen-input boundary ablation in isolated worker processes."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tools import kitti_camera_lidar_baseline_experiment as experiment
from tools import kitti_camera_lidar_v3_replay as replay
from tools.kitti_camera_lidar_initialization_compare import ComparisonReport, sha


def _worker(task: tuple[Path, Path, str, str, str | None, bool, bool, bool]) -> dict:
    captures, output, drive, name, injection, nearest, geometry, restart = task
    experiment.DRIVES = (drive,)
    replay.RUNS = {name: injection}
    return experiment.run(
        captures,
        output,
        boundary_tangent=not (nearest or geometry or restart),
        nearest_edge=nearest,
        match_geometry=geometry,
        geometry_restart=restart,
    ).model_dump(mode="json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nearest-edge", action="store_true")
    parser.add_argument("--match-geometry", action="store_true")
    parser.add_argument("--geometry-restart", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = [
        (
            args.captures,
            args.output / "shards" / drive / name,
            drive,
            name,
            injection,
            args.nearest_edge,
            args.match_geometry,
            args.geometry_restart,
        )
        for drive in experiment.DRIVES
        for name, injection in replay.RUNS.items()
    ]
    with ProcessPoolExecutor(max_workers=4) as pool:
        reports = list(pool.map(_worker, tasks))
    report = ComparisonReport(
        status="complete" if all(r["status"] == "complete" for r in reports) else "blocked",
        provenance={
            **reports[0]["provenance"],
            "batch_generator_sha256": sha(Path(__file__)),
            "execution": "four isolated processes, all fourteen fixed inputs",
        },
        inputs=[row for r in reports for row in r["inputs"]],
    )
    # Preserve shard receipts and place arm reports at the usual publication paths.
    import shutil

    for _, shard, drive, name, _, _, _, _ in tasks:
        shutil.copytree(shard / drive / name, args.output / drive / name)
    (args.output / "report.json").write_text(report.model_dump_json(indent=2))
    (args.output / "report.schema.json").write_text(
        json.dumps(report.model_json_schema(), indent=2)
    )
    return 0 if report.status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
