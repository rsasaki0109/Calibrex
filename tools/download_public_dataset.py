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
from calibrex.data.ethz_hand_eye import (
    ETHZ_ROBOT_ARM_REAL_ARCHIVE,
    ETHZ_ROBOT_ARM_REAL_SHA256,
    ETHZ_ROBOT_ARM_REAL_URL,
)
from calibrex.solvers.native_planar_board_solver import ACFR_VLP_SOURCE_URL

A2D2_LIDAR_SAMPLE_URL = (
    "https://aev-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/"
    "camera_lidar-20180810150607_lidar_frontleft.tar"
)
A2D2_LIDAR_SAMPLE_NAME = "20180810150607_lidar_front_left_000000060.npz"
A2D2_LIDAR_SAMPLE_START = 1536
A2D2_LIDAR_SAMPLE_SIZE = 2_977_425
A2D2_CAMERA_FRONTLEFT_URL = (
    "https://aev-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/"
    "camera_lidar-20180810150607_camera_frontleft.tar"
)
A2D2_CALIBRATION_URL = (
    "https://aev-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/cams_lidars.json"
)
A2D2_PANDEY_RANGES = (
    (
        A2D2_CAMERA_FRONTLEFT_URL,
        "20180810150607_camera_frontleft_000000060.png",
        1536,
        3_008_998,
        "b07c5bc0c985a0fa9604c81b03649033855dde6f7587a4dc33e0d6216e958aca",
    ),
    (
        A2D2_CAMERA_FRONTLEFT_URL,
        "20180810150607_camera_frontleft_000000061.png",
        3_014_144,
        3_010_263,
        "a1389f0ed253dc0fe84d175034edfe6159e3f220813776c601a7455e01f00616",
    ),
    (
        A2D2_LIDAR_SAMPLE_URL,
        "20180810150607_lidar_frontleft_000000060.npz",
        1536,
        2_977_425,
        "1605ad835324e73d31998c94493c646307716404662d3e446a15d3f3f0616c63",
    ),
    (
        A2D2_LIDAR_SAMPLE_URL,
        "20180810150607_lidar_frontleft_000000061.npz",
        2_979_840,
        2_946_473,
        "fee0b3ef3f347e422ee38c01fe41b2722b1148a076062b9c4a0df9a42dd24f91",
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        choices=[
            "tum_rgbd_freiburg1_xyz",
            "a2d2_sensor_setup",
            "a2d2_lidar_pair_sample",
            "a2d2_pandey_mutual_information",
            "livox_horizon_horizon_pcd_sample",
            "acfr_vlp_plane_poses",
            "ethz_hand_eye_robot_arm_real",
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
    if args.dataset == "a2d2_pandey_mutual_information":
        download_a2d2_pandey_mutual_information(args.output_dir)
        return 0
    if args.dataset == "livox_horizon_horizon_pcd_sample":
        downloaded = download_livox_horizon_horizon_pcd_sample(args.output_dir)
        for path in downloaded.files:
            print(f"ready {path}")
        return 0
    if args.dataset == "acfr_vlp_plane_poses":
        download_acfr_vlp_plane_poses(args.output_dir)
        return 0
    if args.dataset == "ethz_hand_eye_robot_arm_real":
        download_ethz_hand_eye_robot_arm_real(args.output_dir)
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


def download_a2d2_pandey_mutual_information(output_dir: Path) -> None:
    """Range-fetch two synchronized real A2D2 camera/LiDAR pairs."""

    import hashlib

    target_dir = output_dir / "a2d2_pandey_mutual_information"
    target_dir.mkdir(parents=True, exist_ok=True)
    for url, name, start, size, expected_sha256 in A2D2_PANDEY_RANGES:
        target = target_dir / name
        end = start + size - 1
        print(f"downloading A2D2 member {name} bytes={start}-{end}")
        request = Request(url, headers={"Range": f"bytes={start}-{end}"})
        with urlopen(request, timeout=90) as response:
            data = response.read()
        if len(data) != size:
            raise SystemExit(f"expected {size} bytes for {name}, got {len(data)}")
        actual_sha256 = hashlib.sha256(data).hexdigest()
        if actual_sha256 != expected_sha256:
            raise SystemExit(
                f"A2D2 digest mismatch for {name}: expected {expected_sha256}, "
                f"got {actual_sha256}"
            )
        target.write_bytes(data)
        print(f"wrote {target}")
    calibration_target = target_dir / "cams_lidars.json"
    with urlopen(A2D2_CALIBRATION_URL, timeout=90) as response:
        calibration_data = response.read()
    expected_calibration_sha256 = (
        "ffec04167050b9c0397121720b8f0bad2cacee83d03c1b7e864619394629c8d2"
    )
    if hashlib.sha256(calibration_data).hexdigest() != expected_calibration_sha256:
        raise SystemExit("A2D2 cams_lidars.json digest mismatch")
    calibration_target.write_bytes(calibration_data)
    print(f"wrote {calibration_target}")


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


def download_ethz_hand_eye_robot_arm_real(output_dir: Path) -> None:
    """Download the pinned ETHZ ASL real robot-arm pose-stream archive."""

    import hashlib

    target_dir = output_dir / "ethz_hand_eye_robot_arm_real"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / ETHZ_ROBOT_ARM_REAL_ARCHIVE
    request = Request(ETHZ_ROBOT_ARM_REAL_URL, headers={"User-Agent": "Calibrex"})
    with urlopen(request, timeout=90) as response:
        data = response.read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != ETHZ_ROBOT_ARM_REAL_SHA256:
        raise SystemExit(
            "ETHZ hand-eye archive digest mismatch: "
            f"expected {ETHZ_ROBOT_ARM_REAL_SHA256}, got {actual}"
        )
    target.write_bytes(data)
    print(f"wrote {target}")

if __name__ == "__main__":
    raise SystemExit(main())
