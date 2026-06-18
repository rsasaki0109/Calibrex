#!/usr/bin/env python3
"""Download small public datasets used by Calibrex examples.

This helper intentionally supports only datasets with direct public download
URLs. Datasets that require login, click-through terms, or API credentials must
be downloaded through their official tools and then pointed at by a Calibrex
manifest.
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["tum_rgbd_freiburg1_xyz", "a2d2_sensor_setup"])
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("examples/public_datasets/catalog.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/public"))
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args()

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


if __name__ == "__main__":
    raise SystemExit(main())
