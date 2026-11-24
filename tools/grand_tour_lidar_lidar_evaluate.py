#!/usr/bin/env python3
"""Run the pre-registered GrandTour LiDAR-LiDAR evaluation once per mission.

An evaluation harness, not part of the Calibrex package. Run it with ``zarr<3``:

    uv run --with "zarr<3" python tools/grand_tour_lidar_lidar_evaluate.py \\
        --preregistration docs/benchmarks/grand_tour_lidar_lidar_preregistration.yaml \\
        --work WORK --output docs/assets/grand_tour_lidar_lidar/evaluation

Per evaluation mission of the pre-registration, it:

1. downloads ``hesai_points.tar``, ``livox_points.tar`` and ``tf.tar`` from the pinned
   Hugging Face revision and checks each SHA-256 against the pin;
2. converts them to a dense rosbag2 and an L1-like sparse one
   (``tools/grand_tour_zarr_to_rosbag2.py``);
3. runs ``calibrex lidar-lidar ros2`` for every method of the pre-registration;
4. runs ``tools/grand_tour_same_time_alignment.py`` on each estimate.

Every command, exit code and attempt is written to ``runs.json``. The rerun policy is the
pre-registration's: a step that fails is repeated with the same command, up to
``max_attempts`` in all. A method that still has no artifact fails that mission.
Converted bags are deleted after each mission to bound disk use; the artifacts are kept.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
DATASET_URL = "https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset/resolve"
TARS = ("hesai_points.tar", "livox_points.tar", "tf.tar")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_step(
    log: list[dict[str, Any]], name: str, command: list[str], max_attempts: int, **kwargs: Any
) -> int:
    """Run ``command`` up to ``max_attempts`` times; record every attempt."""

    code = -1
    for attempt in range(1, max_attempts + 1):
        started = time.time()
        completed = subprocess.run(command, capture_output=True, text=True, **kwargs)
        code = completed.returncode
        log.append(
            {
                "step": name,
                "attempt": attempt,
                "command": command,
                "exit_code": code,
                "seconds": round(time.time() - started, 1),
                "stdout_tail": completed.stdout[-2000:],
                "stderr_tail": completed.stderr[-2000:],
            }
        )
        if code in (0, 1):  # calibrex exits 1 for a failing verdict: a result, not an error
            return code
    return code


def method_command(
    method: dict[str, Any], bag: Path, transforms: tuple[Path, Path], output: Path
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "calibrex.cli.main",
        "lidar-lidar",
        "ros2",
        str(bag),
        "--reference-topic",
        "/hesai_points",
        "--target-topic",
        "/livox_points",
        "--point-time-field",
        "time",
        "--point-time-encoding",
        "absolute_s",
        "--target-point-time-field",
        "time",
        "--target-point-time-encoding",
        "absolute_ns",
        "--body-transforms",
        str(transforms[0]),
        str(transforms[1]),
        "--registration-query-workers",
        "4",
        "--dataset-family",
        "grand_tour",
        "--dataset-license",
        "MIT",
        "--json",
        "--output",
        str(output),
    ]
    return command + [str(item) for item in method["flags"]]


def evaluate_mission(
    mission: dict[str, Any],
    prereg: dict[str, Any],
    work: Path,
    output: Path,
    log: list[dict[str, Any]],
) -> None:
    name = mission["mission"]
    attempts = int(prereg["rerun_policy"]["max_attempts"])
    revision = prereg["dataset"]["revision"]
    folder = work / name
    data = folder / "data"
    data.mkdir(parents=True, exist_ok=True)
    for tar in TARS:
        path = folder / tar
        if not path.exists() or sha256_file(path) != mission["sha256"][tar]:
            url = f"{DATASET_URL}/{revision}/{name}/data/{tar}"
            run_step(
                log,
                f"{name}:download:{tar}",
                ["curl", "-sSL", "--retry", "5", "-o", str(path), url],
                attempts,
            )
        digest = sha256_file(path) if path.exists() else None
        log.append(
            {
                "step": f"{name}:verify:{tar}",
                "sha256": digest,
                "pinned": mission["sha256"][tar],
                "ok": digest == mission["sha256"][tar],
            }
        )
        if digest != mission["sha256"][tar]:
            raise RuntimeError(f"{name}/{tar} does not match its pinned SHA-256")
        with tarfile.open(path) as archive:
            archive.extractall(data, filter="data")
    (data / ".zgroup").write_text('{"zarr_format": 2}', encoding="utf-8")
    converter = REPO / "tools" / "grand_tour_zarr_to_rosbag2.py"
    bags = {"dense": folder / "dense" / "bag", "sparse": folder / "sparse" / "bag"}
    sparse = prereg["emulated_sparse_target"]
    for kind, bag in bags.items():
        if bag.exists():
            shutil.rmtree(bag)
        bag.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(converter),
            "--mission-data",
            str(data),
            "--groups",
            "hesai_points",
            "livox_points",
            "--output",
            str(bag),
        ]
        if kind == "sparse":
            command += [
                "--sparse",
                "livox_points",
                str(sparse["max_points"]),
                str(sparse["max_range_m"]),
                "--seed",
                str(sparse["seed"]),
            ]
        run_step(log, f"{name}:convert:{kind}", command, attempts)
        shutil.copy(bag.parent / "conversion.json", output / f"{name}_{kind}_conversion.json")
    transforms = (
        bags["dense"].parent / "T_box_hesai_points.yaml",
        bags["dense"].parent / "T_box_livox_points.yaml",
    )
    aligner = REPO / "tools" / "grand_tour_same_time_alignment.py"
    for method_id, method in prereg["methods"].items():
        if method.get("kind") != "calibrex":
            continue
        artifact = output / f"{name}_{method_id}.yaml"
        run_step(
            log,
            f"{name}:{method_id}",
            method_command(method, bags[method["bag"]], transforms, artifact),
            attempts,
            cwd=REPO,
        )
        if artifact.exists():
            frames = same_time_frames(data, prereg["same_time_check"]["fractions"])
            run_step(
                log,
                f"{name}:{method_id}:same_time",
                [
                    sys.executable,
                    str(aligner),
                    "--mission-data",
                    str(data),
                    "--reference",
                    "hesai_points",
                    "--target",
                    "livox_points",
                    "--artifact",
                    str(artifact),
                    "--tf-reference",
                    "hesai_lidar",
                    "--tf-target",
                    "livox_lidar",
                    "--frames",
                    *map(str, frames),
                    "--output",
                    str(output / f"{name}_{method_id}_same_time.json"),
                ],
                attempts,
            )
    for bag in bags.values():
        shutil.rmtree(bag.parent, ignore_errors=True)
    shutil.rmtree(data, ignore_errors=True)


def same_time_frames(data: Path, fractions: list[float]) -> list[int]:
    """Target scan indices at the pre-registered fractions of the Livox recording."""

    import zarr

    count = zarr.open_group(str(data), mode="r")["livox_points"]["timestamp"].shape[0]
    return [min(count - 1, int(fraction * count)) for fraction in fractions]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--missions", nargs="*", help="restrict to these missions (development)")
    args = parser.parse_args(argv)
    prereg = yaml.safe_load(args.preregistration.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO
    ).stdout.strip()
    log: list[dict[str, Any]] = []
    record = {
        "preregistration": str(args.preregistration),
        "preregistration_sha256": sha256_file(args.preregistration),
        "git_head": head,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "steps": log,
    }
    for mission in prereg["recordings"]["evaluation"]:
        if args.missions and mission["mission"] not in args.missions:
            continue
        try:
            evaluate_mission(mission, prereg, args.work, args.output, log)
        except Exception as exc:  # recorded; the scorer fails this mission's methods
            log.append({"step": f"{mission['mission']}:error", "error": repr(exc)})
        (args.output / "runs.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    record["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (args.output / "runs.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
