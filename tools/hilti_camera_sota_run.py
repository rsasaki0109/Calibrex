#!/usr/bin/env python3
"""Run ``calibrex check`` on one Hilti 2022 recording for the camera SOTA audits.

An evaluation harness, not part of the Calibrex package. It is the run recipe of the
pre-registered camera-camera and camera-imu v2 audits
(``docs/benchmarks/hilti_camera_sota_preregistration.yaml``):

1. hash the bag (SHA-256 of its bytes, before any message is read), or take the hash
   that ``tools/hilti_download.py`` computed while downloading it;
2. run the check on the bag with Hilti's own calibration as the candidate (the
   *baseline*): the four Kalibr camchains and ``lidar_calibration.yaml``;
3. for every ``--inject CAM:AXIS:DEG``, run it again with that camera's candidate
   pose turned by ``DEG`` about its own ``AXIS`` (``T_imu_cam`` right-multiplied by
   the turn), so the camera turns against the IMU and every other sensor. The check's
   attribution reports such a camera as turned by ``+DEG`` about that axis;
4. write ``manifest.json``: the bag's name, size and hash, the calibration files'
   hashes, every command with its exit code and wall time, and the environment.

Every run's check artifact and evidence directory stay in ``--output-dir``. The check
runs with ``--fail-on never``, so a non-zero exit code is an error, not a verdict. A run
is never repeated by this tool; a failed command is recorded with its exit code.

::

    python tools/hilti_camera_sota_run.py exp05_head.bag --recording exp05 \\
        --calibration-dir hilti_2022 --pairs camera-camera \\
        --inject cam3:y:1.0 --inject cam1:y:1.0 --output-dir out/exp05
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

CAMCHAINS = (
    "calib_3_cam0-1-camchain-imucam.yaml",
    "calib_3_cam2-camchain-imucam.yaml",
    "calib_3_cam3-camchain-imucam.yaml",
    "calib_3_cam4-camchain-imucam.yaml",
)
LIDAR_CALIBRATION = "lidar_calibration.yaml"
AXES = {"x": 0, "y": 1, "z": 2}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def turned_camchain(source: Path, target: Path, camera: str, axis: str, degrees: float) -> bool:
    """Write ``source`` to ``target`` with ``camera``'s ``T_cam_imu`` turned; ``False``
    when the file does not hold that camera (it is then copied unchanged)."""

    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    topic = f"/alphasense/{camera}/image_raw"
    found = False
    for entry in payload.values():
        if isinstance(entry, dict) and entry.get("rostopic") == topic:
            # the camera's pose turns by +degrees about its own axis:
            # T_imu_cam' = T_imu_cam Rot, so T_cam_imu' = Rot^T T_cam_imu
            turn = np.eye(4)
            turn[:3, :3] = Rotation.from_rotvec(
                np.radians(-degrees) * np.eye(3)[AXES[axis]]
            ).as_matrix()
            entry["T_cam_imu"] = (turn @ np.asarray(entry["T_cam_imu"], dtype=float)).tolist()
            found = True
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return found


def run_check(
    bag: Path, calibration: Path, pairs: str, output: Path, cache_dir: Path | None
) -> dict[str, object]:
    command = [
        str(Path(sys.executable).with_name("calibrex")),
        "check",
        str(bag),
        "--pairs",
        pairs,
        *[
            item
            for name in (*CAMCHAINS, LIDAR_CALIBRATION)
            for item in ("--tf", str(calibration / name))
        ],
        "--fail-on",
        "never",
        "--evidence-dir",
        str(output.with_suffix("")) + "_evidence",
        "--output",
        str(output),
    ]
    command += ["--cache-dir", str(cache_dir)] if cache_dir is not None else ["--no-cache"]
    started = time.monotonic()
    with output.with_suffix(".log").open("w", encoding="utf-8") as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
    return {
        "command": command,
        "exit_code": code,
        "wall_s": round(time.monotonic() - started, 1),
        "artifact": output.name if output.exists() else None,
    }


def git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("bag", type=Path)
    parser.add_argument("--recording", required=True, help="e.g. exp05")
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--pairs", required=True, help="passed to calibrex check --pairs")
    parser.add_argument(
        "--inject", action="append", default=[], help="CAM:AXIS:DEG, e.g. cam3:y:1.0"
    )
    parser.add_argument("--cache-dir", type=Path, default=None, help="default: --no-cache")
    parser.add_argument(
        "--download-record",
        type=Path,
        default=None,
        help="the bag's .download.json from tools/hilti_download.py: its SHA-256, computed "
        "while downloading, is used instead of reading the bag again",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    manifest: dict[str, object] = {
        "recording": args.recording,
        "bag": {"name": args.bag.name, "size_bytes": args.bag.stat().st_size},
        "pairs": args.pairs,
        "calibration_sha256": {
            name: sha256_file(args.calibration_dir / name)
            for name in (*CAMCHAINS, LIDAR_CALIBRATION)
        },
        "git_commit": git_commit(root),
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "runs": {},
    }
    started = time.monotonic()
    bag_record = manifest["bag"]
    assert isinstance(bag_record, dict)
    if args.download_record is not None:
        download = json.loads(args.download_record.read_text(encoding="utf-8"))
        if download["bytes"][1] != args.bag.stat().st_size:
            parser.error("the download record's byte range does not match the bag's size")
        bag_record["sha256"] = download["sha256"]
        bag_record["sha256_source"] = "computed while downloading (tools/hilti_download.py)"
        bag_record["download"] = download
    else:
        bag_record["sha256"] = sha256_file(args.bag)
        bag_record["sha256_source"] = "computed from the file"
    bag_record["hash_wall_s"] = round(time.monotonic() - started, 1)
    # the baseline and every injection run at once: they read the same bag, which the
    # page cache then holds, and each check is mostly single-threaded
    jobs: dict[str, tuple[Path, dict[str, object]]] = {"baseline": (args.calibration_dir, {})}
    for spec in args.inject:
        camera, axis, degrees = spec.split(":")
        name = f"inject_{camera}_{axis}{float(degrees):+g}"
        calibration = args.output_dir / f"{name}_calibration"
        calibration.mkdir(exist_ok=True)
        hits = [
            turned_camchain(
                args.calibration_dir / chain, calibration / chain, camera, axis, float(degrees)
            )
            for chain in CAMCHAINS
        ]
        if sum(hits) != 1:
            parser.error(f"{camera} is in {sum(hits)} camchains, expected exactly one")
        shutil.copyfile(args.calibration_dir / LIDAR_CALIBRATION, calibration / LIDAR_CALIBRATION)
        jobs[name] = (calibration, {"camera": camera, "axis": axis, "degrees": float(degrees)})
    with ThreadPoolExecutor(len(jobs)) as pool:
        futures = {
            name: pool.submit(
                run_check,
                args.bag,
                calibration,
                args.pairs,
                args.output_dir / f"{name}.json",
                args.cache_dir,
            )
            for name, (calibration, _) in jobs.items()
        }
    runs: dict[str, object] = {
        name: {**details, **futures[name].result()} for name, (_, details) in jobs.items()
    }
    manifest["runs"] = runs
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    # --fail-on never: a non-zero exit is an error, not a verdict
    return 0 if all(run["exit_code"] == 0 for run in runs.values()) else 1  # type: ignore[index]


if __name__ == "__main__":
    sys.exit(main())
