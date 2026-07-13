#!/usr/bin/env python3
"""Download small public datasets used by Calibrex examples.

This helper intentionally supports only datasets with direct public download
URLs. Datasets that require login, click-through terms, or API credentials must
be downloaded through their official tools and then pointed at by a slac
manifest.
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path
from urllib.request import Request, urlopen, urlretrieve

import yaml

from calibrex.data.downloads import download_livox_horizon_horizon_pcd_sample
from calibrex.solvers.native_planar_board_solver import ACFR_VLP_SOURCE_URL

A2D2_LIDAR_SAMPLE_URL = (
    "https://aev-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/"
    "camera_lidar-20180810150607_lidar_frontleft.tar"
)
A2D2_LIDAR_SAMPLE_NAME = "20180810150607_lidar_front_left_000000060.npz"
A2D2_LIDAR_SAMPLE_START = 1536
A2D2_LIDAR_SAMPLE_SIZE = 2_977_425


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        choices=[
            "tum_rgbd_freiburg1_xyz",
            "a2d2_sensor_setup",
            "a2d2_lidar_pair_sample",
            "livox_horizon_horizon_pcd_sample",
            "acfr_vlp_plane_poses",
        ],
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("examples/public_datasets/catalog.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/public"))
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args()

    if args.dataset == "a2d2_lidar_pair_sample":
        download_a2d2_lidar_pair_sample(args.output_dir)
        return 0
    if args.dataset == "livox_horizon_horizon_pcd_sample":
        downloaded = download_livox_horizon_horizon_pcd_sample(args.output_dir)
        for path in downloaded.files:
            print(f"ready {path}")
        return 0
    if args.dataset == "acfr_vlp_plane_poses":
        download_acfr_vlp_plane_poses(args.output_dir)
        return 0

    catalog = yaml.safe_load(args.catalog.read_text(encoding="utf-8"))
    entry = catalog["datasets"][args.dataset]
    download_url = entry.get("download_url")
    if not download_url:
        raise SystemExit(f"{args.dataset} does not have a direct public download URL")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / Path(download_url).name
    print(f"downloading {download_url}")
    urlretrieve(download_url, archive)
    print(f"wrote {archive}")

    if not args.no_extract and tarfile.is_tarfile(archive):
        print(f"extracting {archive}")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(args.output_dir)
    return 0


def download_a2d2_lidar_pair_sample(output_dir: Path) -> None:
    """Download one real A2D2 LiDAR NPZ sample from inside the public tar."""

    target_dir = output_dir / "a2d2_lidar_pair"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / A2D2_LIDAR_SAMPLE_NAME
    start = A2D2_LIDAR_SAMPLE_START
    end = start + A2D2_LIDAR_SAMPLE_SIZE - 1
    print(f"downloading A2D2 LiDAR sample bytes={start}-{end}")
    request = Request(A2D2_LIDAR_SAMPLE_URL, headers={"Range": f"bytes={start}-{end}"})
    with urlopen(request, timeout=90) as response:
        data = response.read()
    if len(data) != A2D2_LIDAR_SAMPLE_SIZE:
        raise SystemExit(f"expected {A2D2_LIDAR_SAMPLE_SIZE} bytes, got {len(data)}")
    target.write_bytes(data)
    print(f"wrote {target}")


def download_acfr_vlp_plane_poses(output_dir: Path) -> None:
    """Download the pinned Apache-2.0 ACFR VLP extracted-pose example."""

    import hashlib

    target_dir = output_dir / "acfr_vlp_plane_poses"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "poses.csv"
    request = Request(ACFR_VLP_SOURCE_URL, headers={"User-Agent": "Calibrex"})
    with urlopen(request, timeout=90) as response:
        data = response.read()
    expected = "024bc6ed9009652761e9c0df49b106d325a10c88d41a80e9b36ea55fd567e110"
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise SystemExit(f"ACFR poses.csv digest mismatch: expected {expected}, got {actual}")
    target.write_bytes(data)
    print(f"wrote {target}")

if __name__ == "__main__":
    raise SystemExit(main())
