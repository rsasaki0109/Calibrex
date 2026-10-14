#!/usr/bin/env python3
"""Run ``calibrex check`` camera-lidar on one KITTI raw drive for the camera-lidar SOTA audit.

An evaluation harness, not part of the Calibrex package. For one unzipped
``2011_09_26_drive_XXXX_sync`` directory it:

1. converts the drive with ``calibrex convert kitti-raw DRIVE --camera image_02``, which
   writes the rectified ``image_02`` and the vendor calibration (``calib_velo_to_cam``,
   ``calib_cam_to_cam``, ``calib_imu_to_velo``) as ``/tf_static``: the vendor calibration
   is the candidate;
2. runs ``calibrex check BAG --pairs camera-lidar`` on it (the *baseline*);
3. for every ``--inject AXIS:DEG``, runs it again with the camera's candidate pose turned
   by ``DEG`` about its own ``AXIS`` (x, y or z of ``cam2_optical``):
   ``T_imu_cam' = T_imu_cam Rot(AXIS, -DEG)``, so ``T_camera_lidar' = Rot(AXIS, DEG)
   T_camera_lidar`` and the check reads ``+DEG`` on roll, pitch or yaw;
4. writes ``manifest.json``: the drive, the calibration files' hashes, the download
   record, every command with its exit code and wall time, and the environment.

The checks run with ``--fail-on never`` and ``--no-cache``, so a non-zero exit code is an
error, not a verdict. A run is never repeated by this tool. The converted bag is deleted
afterwards unless ``--keep-bag``::

    python tools/kitti_camera_lidar_sota_run.py RAW/2011_09_26/2011_09_26_drive_0005_sync \\
        --recording 0005 --inject x:3 --inject y:3 --inject z:3 --bag-dir WORK/bag \\
        --output-dir RUNS/0005
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

CAMERA = "image_02"
CAMERA_FRAME = "cam2_optical"
CALIBRATION_FILES = ("calib_cam_to_cam.txt", "calib_imu_to_velo.txt", "calib_velo_to_cam.txt")
AXES = {"x": 0, "y": 1, "z": 2}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def calibrex() -> str:
    return str(Path(sys.executable).with_name("calibrex"))


def turned_frames(bag: Path, axis: str, degrees: float, target: Path) -> None:
    """Write a ``slac.check_frames`` file with the camera turned by ``degrees`` about ``axis``."""

    from calibrex.check.tf_sources import load_bag_tf_static

    source = load_bag_tf_static(bag)
    edge = next(item for item in source.edges if item.child == CAMERA_FRAME)  # type: ignore[union-attr]
    turn = Rotation.from_rotvec(np.radians(-degrees) * np.eye(3)[AXES[axis]])
    rotation = Rotation.from_quat(edge.transform.rotation_quat_xyzw) * turn
    payload = {
        "schema_version": "slac.check_frames/v0.1",
        "frames": [
            {
                "name": edge.child,
                "parent": edge.parent,
                "translation_m": [float(value) for value in edge.transform.translation_m],
                "rotation_quat_xyzw": [float(value) for value in rotation.as_quat()],
            }
        ],
    }
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def run(command: list[str], log: Path) -> dict[str, object]:
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        code = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    return {
        "command": command,
        "exit_code": code.returncode,
        "wall_s": round(time.monotonic() - started, 1),
    }


def run_check(bag: Path, output: Path, frames: Path | None) -> dict[str, object]:
    command = [calibrex(), "check", str(bag), "--pairs", "camera-lidar", "--no-cache"]
    command += ["--tf", str(frames)] if frames is not None else []
    command += ["--fail-on", "never", "--evidence-dir", str(output.with_suffix("")) + "_evidence"]
    command += ["--output", str(output)]
    result = run(command, output.with_suffix(".log"))
    result["artifact"] = output.name if output.exists() else None
    return result


def git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("drive", type=Path, help="an unzipped *_sync drive directory")
    parser.add_argument("--recording", required=True, help="e.g. 0005")
    parser.add_argument("--calib-dir", type=Path, default=None, help="default: the drive's parent")
    parser.add_argument("--inject", action="append", default=[], help="AXIS:DEG, e.g. y:3")
    parser.add_argument("--bag-dir", type=Path, required=True, help="where the bag is written")
    parser.add_argument("--keep-bag", action="store_true")
    parser.add_argument(
        "--download-record", type=Path, default=None, help="the drive archive's download record"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    calib_dir = args.calib_dir or args.drive.parent
    root = Path(__file__).resolve().parents[1]
    manifest: dict[str, object] = {
        "recording": args.recording,
        "drive": args.drive.name,
        "calibration_sha256": {name: sha256_file(calib_dir / name) for name in CALIBRATION_FILES},
        "download": (
            json.loads(args.download_record.read_text(encoding="utf-8"))
            if args.download_record is not None
            else None
        ),
        "git_commit": git_commit(root),
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    convert = run(
        [
            calibrex(),
            "convert",
            "kitti-raw",
            str(args.drive),
            "--calib-dir",
            str(calib_dir),
            "--camera",
            CAMERA,
            "--output",
            str(args.bag_dir),
            "--overwrite",
        ],
        args.output_dir / "convert.log",
    )
    manifest["convert"] = convert
    conversion = args.bag_dir / "calibrex_conversion.json"
    if convert["exit_code"] == 0 and conversion.is_file():
        shutil.copyfile(conversion, args.output_dir / "conversion.json")
    runs: dict[str, object] = {}
    if convert["exit_code"] == 0:
        jobs: dict[str, tuple[Path | None, dict[str, object]]] = {"baseline": (None, {})}
        for spec in args.inject:
            axis, degrees = spec.split(":")
            name = f"inject_{axis}{float(degrees):+g}"
            frames = args.output_dir / f"{name}_frames.yaml"
            turned_frames(args.bag_dir, axis, float(degrees), frames)
            jobs[name] = (frames, {"axis": axis, "degrees": float(degrees)})
        # the checks read the same bag, which the page cache then holds
        with ThreadPoolExecutor(len(jobs)) as pool:
            futures = {
                name: pool.submit(run_check, args.bag_dir, args.output_dir / f"{name}.json", frames)
                for name, (frames, _) in jobs.items()
            }
        runs = {name: {**details, **futures[name].result()} for name, (_, details) in jobs.items()}
    manifest["runs"] = runs
    if not args.keep_bag:
        shutil.rmtree(args.bag_dir, ignore_errors=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    codes = [convert["exit_code"], *(run["exit_code"] for run in runs.values())]  # type: ignore[index]
    return 0 if all(code == 0 for code in codes) else 1


if __name__ == "__main__":
    sys.exit(main())
