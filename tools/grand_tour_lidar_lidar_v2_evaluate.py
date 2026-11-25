#!/usr/bin/env python3
"""Run the frozen GrandTour v2 profile from pinned Zarr archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import yaml
from tools import grand_tour_lidar_lidar_evaluate as v1
from tools.grand_tour_zarr_to_rosbag2 import thin_scan, transform_matrix

from calibrex.cli.main import main as cli
from calibrex.data.livox_ros2 import LivoxScan, LivoxStreamProfile
from calibrex.evaluation import lidar_lidar_map

MODES = {
    "calibrex_dense": {
        "kind": "calibrex",
        "flags": ["--map-half-window-scans", 0, "--window-sensitivity", 5],
    },
    "calibrex_sparse": {
        "kind": "calibrex",
        "flags": ["--preset", "unitree-go2-l1"],
    },
}
TARS = ("hesai_points.tar", "livox_points.tar", "tf.tar")
SPARSE = {"max_points": 2000, "max_range_m": 20.0, "seed": 0}
FRACTIONS = [0.1, 0.3, 0.5, 0.7, 0.9]
PREREG = Path("docs/benchmarks/grand_tour_lidar_lidar_v2_preregistration.yaml")


def scans(
    data: Path, profile: LivoxStreamProfile, sparse: bool, max_seconds: float | None
) -> Iterator[LivoxScan]:
    """Replay the converter's scan selection and decoder's timestamp convention."""

    import zarr

    name = profile.point_topic.removeprefix("/")
    group = zarr.open_group(str(data), mode="r")[name]
    stamps = group["timestamp"][:]
    valid = group["valid"][:, 0]
    chunk = min(group["points"].chunks[0], group["time"].chunks[0])
    rng = np.random.default_rng(SPARSE["seed"])
    for start in range(0, len(stamps), chunk):
        stop = min(start + chunk, len(stamps))
        points, times = group["points"][start:stop], group["time"][start:stop]
        for row, index in enumerate(range(start, stop)):
            count = int(valid[index])
            xyz = points[row, :count].astype(np.float64)
            raw = times[row, :count]
            stamp_ns = round(float(stamps[index]) * 1e9)
            stamp = stamp_ns * 1e-9
            if max_seconds is not None and stamp - stamps[0] > max_seconds:
                return
            if sparse and name == "livox_points":
                xyz, raw = thin_scan(xyz, raw, SPARSE["max_points"], SPARSE["max_range_m"], rng)
            keep = np.isfinite(xyz).all(axis=1) & np.any(xyz != 0, axis=1) & np.isfinite(raw)
            offsets = None
            if profile.point_time_field is not None:
                offsets = (
                    (raw[keep] - float(stamp_ns)) * 1e-9
                    if profile.point_time_encoding == "absolute_ns"
                    else raw[keep] - stamp
                )
            yield stamp, xyz[keep], offsets


def fit(data: Path, method: str, output: Path, input_digest: str) -> int:
    import zarr

    root = zarr.open_group(str(data), mode="r")
    transforms = (data / "T_box_hesai_points.yaml", data / "T_box_livox_points.yaml")
    for path, name in zip(transforms, ("hesai_points", "livox_points"), strict=True):
        transform = transform_matrix(root[name].attrs["transform"])
        path.write_text("T: " + json.dumps(transform.tolist()) + "\n", encoding="utf-8")

    def replay(
        bag: Path, profile: LivoxStreamProfile, *, max_seconds: float | None = None
    ) -> Iterator[LivoxScan]:
        return scans(data, profile, method == "calibrex_sparse", max_seconds)

    command = v1.method_command(MODES[method], data.parent, transforms, output)
    with (
        patch.object(lidar_lidar_map, "iter_livox_points", replay),
        patch.object(
            lidar_lidar_map, "bag_input_digest", return_value=(input_digest, "pinned_zarr_archives")
        ),
    ):
        return cli(command[3:])


def run_mission(
    mission: dict[str, Any], revision: str, work: Path, output: Path, log: list[dict[str, Any]]
) -> None:
    name = mission["mission"]
    data = work / name / "data"
    data.mkdir(parents=True, exist_ok=False)
    try:
        for tar in TARS:
            path = data.parent / tar
            url = f"{v1.DATASET_URL}/{revision}/{name}/data/{tar}?download=true"
            exit_code = v1.run_step(
                log,
                f"{name}:download:{tar}",
                ["curl", "-fsSL", "--retry", "2", "-o", str(path), url],
                1,
            )
            actual = v1.sha256_file(path) if path.exists() else None
            expected = mission["sha256"][tar]
            log.append({"step": f"{name}:verify:{tar}", "sha256": actual, "pinned": expected})
            if exit_code != 0 or actual != expected:
                raise ValueError(f"archive verification failed for {name}/{tar}")
            with tarfile.open(path) as archive:
                archive.extractall(data, filter="data")
            path.unlink()
        (data / ".zgroup").write_text('{"zarr_format": 2}', encoding="utf-8")
        digest = hashlib.sha256(json.dumps(mission["sha256"], sort_keys=True).encode()).hexdigest()
        frames = v1.same_time_frames(data, FRACTIONS)
        for method in MODES:
            artifact = output / f"{name}_{method}.yaml"
            v1.run_step(
                log,
                f"{name}:{method}",
                [
                    sys.executable,
                    "-m",
                    "tools.grand_tour_lidar_lidar_v2_evaluate",
                    "fit",
                    str(data),
                    method,
                    str(artifact),
                    digest,
                ],
                1,
            )
            if artifact.exists() and yaml.safe_load(artifact.read_text())["transform"] is not None:
                v1.run_step(
                    log,
                    f"{name}:{method}:same_time",
                    [
                        sys.executable,
                        str(Path(__file__).with_name("grand_tour_same_time_alignment.py")),
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
                        str(output / f"{name}_{method}_same_time.json"),
                    ],
                    1,
                )
    finally:
        shutil.rmtree(data.parent)


def main() -> int:
    if len(sys.argv) == 6 and sys.argv[1] == "fit":
        return fit(Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]), sys.argv[5])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    protocol = yaml.safe_load(PREREG.read_text(encoding="utf-8"))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            protocol["frozen_code_commit"],
            "HEAD",
            "--",
            "src",
            "tools",
            "uv.lock",
            "pyproject.toml",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "src",
            "tools",
            "uv.lock",
            "pyproject.toml",
            str(PREREG),
        ],
        check=True,
    )
    subprocess.run(["git", "cat-file", "-e", f"HEAD:{PREREG}"], check=True)
    args.output.mkdir(parents=True, exist_ok=False)
    steps: list[dict[str, Any]] = []
    record = {
        "preregistration": str(PREREG),
        "preregistration_sha256": v1.sha256_file(PREREG),
        "git_head": head,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "steps": steps,
    }
    run_log = args.output / "runs.json"
    for mission in protocol["recordings"]["evaluation"]:
        print(f"{mission['mission']} started", flush=True)
        try:
            run_mission(mission, protocol["dataset"]["revision"], args.work, args.output, steps)
        except (OSError, ValueError, tarfile.TarError) as exc:
            steps.append({"step": f"{mission['mission']}:error", "error": repr(exc)})
        run_log.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"{mission['mission']} finished", flush=True)
    record["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    run_log.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
