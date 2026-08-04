#!/usr/bin/env python3
"""Download small public datasets used by Calibrex examples.

This helper supports direct public download URLs and the public SharePoint
cookie/Range flow used by TIERS Indoor02. Datasets that require login,
click-through terms, or API credentials must be downloaded through their
official tools and then pointed at by a slac manifest.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import tarfile
import threading
import time
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
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

INDOOR02_SHARE_URL = (
    "https://utufi.sharepoint.com/:u:/s/msteams_0ed7e9/"
    "EYXGcc1Z-y1FpnDwQ1geIoEBovXvLfoxZwt36J1_t2PugA?e=UwHhXd"
)
INDOOR02_SOURCE_PATH = (
    "/sites/msteams_0ed7e9/Shared Documents/Datasets/TIERS_Multi_Lidar_Dataset/"
    "indoor02_sauna_normal_2022-02-21-19-05-17.bag"
)
INDOOR02_SIZE_BYTES = 17_974_826_130
INDOOR02_SHA256_FIRST_MIB = (
    "164c2f8dcfd42463ad4981ba25a97668f015ced776db2781fc1e4211fd8fd574"
)
AGROB_MODULAR_E_URL = (
    "https://zenodo.org/records/8083431/files/"
    "2023-06-13T170852Z.zip?download=1"
)
AGROB_MODULAR_E_ARCHIVE = "2023-06-13T170852Z.zip"
AGROB_MODULAR_E_SIZE_BYTES = 2_548_542_803
AGROB_MODULAR_E_MD5 = "fa0cea9ea4aafd5e7a0d75676e8e064d"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        choices=[
            "tum_rgbd_freiburg1_xyz",
            "a2d2_sensor_setup",
            "a2d2_lidar_pair_sample",
            "a2d2_pandey_mutual_information",
            "tiers_lidars_dataset_indoor02",
            "livox_horizon_horizon_pcd_sample",
            "acfr_vlp_plane_poses",
            "ethz_hand_eye_robot_arm_real",
            "agrob_modular_e",
        ],
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("examples/public_datasets/catalog.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/public"))
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument(
        "--agrob-workers",
        type=int,
        default=1,
        help="parallel HTTP Range workers for agrob_modular_e (default: 1)",
    )
    parser.add_argument(
        "--agrob-chunk-mib",
        type=int,
        default=32,
        help="AgRob Range chunk size in MiB (default: 32)",
    )
    parser.add_argument(
        "--agrob-chunk-kib",
        type=int,
        default=None,
        help="override AgRob Range chunk size in KiB (useful for slow endpoints)",
    )
    args = parser.parse_args()

    if args.dataset == "a2d2_lidar_pair_sample":
        download_a2d2_lidar_pair_sample(args.output_dir)
        return 0
    if args.dataset == "a2d2_pandey_mutual_information":
        download_a2d2_pandey_mutual_information(args.output_dir)
        return 0
    if args.dataset == "tiers_lidars_dataset_indoor02":
        download_tiers_lidars_dataset_indoor02(args.output_dir)
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
    if args.dataset == "agrob_modular_e":
        chunk_size = (
            args.agrob_chunk_kib * 1024
            if args.agrob_chunk_kib is not None
            else args.agrob_chunk_mib * 1024 * 1024
        )
        download_agrob_modular_e(
            args.output_dir,
            extract=not args.no_extract,
            workers=args.agrob_workers,
            chunk_size=chunk_size,
        )
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


def download_tiers_lidars_dataset_indoor02(output_dir: Path) -> None:
    """Resume a public SharePoint Range download of the TIERS Indoor02 bag.

    The share link first redirects to an HTML file view and sets a public
    ``FedAuth`` cookie.  The SharePoint ``download.aspx`` endpoint is then
    addressed directly with sequential byte ranges.  Sequential append makes
    interruption recovery safe without preallocating a misleading full-size
    sparse file.
    """

    import hashlib
    from http.cookiejar import CookieJar
    from urllib.parse import quote
    from urllib.request import HTTPCookieProcessor, build_opener

    target_dir = output_dir / "tiers_lidars_dataset"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "indoor02.bag"

    if target.exists() and target.stat().st_size == INDOOR02_SIZE_BYTES:
        with target.open("rb") as stream:
            digest = hashlib.sha256(stream.read(1024 * 1024)).hexdigest()
        if digest != INDOOR02_SHA256_FIRST_MIB:
            raise SystemExit(f"Indoor02 first-MiB digest mismatch: {target}")
        print(f"ready {target} (size and first-MiB SHA-256 verified)")
        return
    if target.exists() and target.stat().st_size > INDOOR02_SIZE_BYTES:
        raise SystemExit(f"Indoor02 target is larger than expected: {target}")

    cookie_jar = CookieJar()
    opener = build_opener(HTTPCookieProcessor(cookie_jar))
    landing_request = Request(INDOOR02_SHARE_URL, headers={"User-Agent": "Calibrex"})
    with opener.open(landing_request, timeout=90) as landing:
        # Opening the share link is intentional: it establishes the public
        # guest cookie used by the direct download endpoint below.
        _ = landing.read(1024)

    site_root = "https://utufi.sharepoint.com/sites/msteams_0ed7e9"
    download_url = (
        f"{site_root}/_layouts/15/download.aspx?SourceUrl="
        f"{quote(INDOOR02_SOURCE_PATH, safe='')}"
    )
    chunk_size = 32 * 1024 * 1024
    start = target.stat().st_size if target.exists() else 0
    print(f"downloading {INDOOR02_SHARE_URL}")
    while start < INDOOR02_SIZE_BYTES:
        end = min(start + chunk_size, INDOOR02_SIZE_BYTES) - 1
        request = Request(
            download_url,
            headers={
                "Accept": "*/*",
                "Range": f"bytes={start}-{end}",
                "User-Agent": "Calibrex",
            },
        )
        with opener.open(request, timeout=180) as response:
            content_range = response.headers.get("Content-Range", "")
            expected_range = f"bytes {start}-{end}/{INDOOR02_SIZE_BYTES}"
            if response.status != 206 or content_range != expected_range:
                raise SystemExit(
                    "Indoor02 SharePoint Range request failed: "
                    f"status={response.status}, content-range={content_range!r}"
                )
            received = 0
            with target.open("ab") as stream:
                while True:
                    piece = response.read(1024 * 1024)
                    if not piece:
                        break
                    stream.write(piece)
                    received += len(piece)
        expected_length = end - start + 1
        if received != expected_length:
            raise SystemExit(
                f"Indoor02 Range length mismatch: expected {expected_length}, "
                f"got {received}"
            )
        start = end + 1
        print(f"downloaded {start}/{INDOOR02_SIZE_BYTES} bytes")

    with target.open("rb") as stream:
        digest = hashlib.sha256(stream.read(1024 * 1024)).hexdigest()
    if digest != INDOOR02_SHA256_FIRST_MIB:
        raise SystemExit(
            "Indoor02 first-MiB digest mismatch: "
            f"expected {INDOOR02_SHA256_FIRST_MIB}, got {digest}"
        )
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


def download_agrob_modular_e(
    output_dir: Path,
    *,
    extract: bool = True,
    workers: int = 1,
    chunk_size: int = 32 * 1024 * 1024,
) -> None:
    """Resume and checksum the public AgRob ROS 2 Livox/RS-LiDAR archive.

    Zenodo accepts byte ranges but may throttle or interrupt long responses.
    The sequential mode completes each range in a sidecar chunk before
    appending it; the parallel mode uses a separate preallocated sidecar and
    an atomic range ledger. The archive is promoted to its final name only
    after the published MD5 matches; extraction is optional and path-traversal
    safe.
    """

    target_dir = output_dir / "agrob_scorpion"
    target_dir.mkdir(parents=True, exist_ok=True)
    archive = target_dir / AGROB_MODULAR_E_ARCHIVE
    if workers < 1 or workers > 256:
        raise ValueError("AgRob workers must be between 1 and 256")
    if chunk_size < 64 * 1024 or chunk_size > 512 * 1024 * 1024:
        raise ValueError("AgRob chunk_size must be between 64 KiB and 512 MiB")
    partial = archive.with_suffix(archive.suffix + ".part")
    if archive.exists():
        if archive.stat().st_size != AGROB_MODULAR_E_SIZE_BYTES:
            raise SystemExit(f"AgRob archive exists with unexpected size: {archive}")
        digest = _md5_file(archive)
        if digest != AGROB_MODULAR_E_MD5:
            raise SystemExit(
                f"AgRob archive MD5 mismatch: expected {AGROB_MODULAR_E_MD5}, "
                f"got {digest}"
            )
    else:
        if workers == 1:
            if partial.exists() and partial.stat().st_size > AGROB_MODULAR_E_SIZE_BYTES:
                raise SystemExit(f"AgRob partial archive is larger than expected: {partial}")
            chunk = partial.with_name(partial.name + ".chunk")
            start = partial.stat().st_size if partial.exists() else 0
            while start < AGROB_MODULAR_E_SIZE_BYTES:
                end = min(start + chunk_size, AGROB_MODULAR_E_SIZE_BYTES) - 1
                _download_agrob_range(chunk, start, end)
                with chunk.open("rb") as source, partial.open("ab") as destination:
                    while piece := source.read(1024 * 1024):
                        destination.write(piece)
                received = chunk.stat().st_size
                if received != end - start + 1:
                    raise SystemExit(
                        f"AgRob chunk append mismatch: expected {end - start + 1}, "
                        f"got {received}"
                    )
                chunk.unlink()
                start = end + 1
                print(f"downloaded {start}/{AGROB_MODULAR_E_SIZE_BYTES} bytes")
        else:
            parallel_suffix = f".parallel-{chunk_size // 1024}k"
            parallel_partial = target_dir / (
                AGROB_MODULAR_E_ARCHIVE + parallel_suffix + ".part"
            )
            parallel_state = target_dir / (
                AGROB_MODULAR_E_ARCHIVE + parallel_suffix + ".state.json"
            )
            _download_agrob_ranges_parallel(
                parallel_partial,
                parallel_state,
                workers=workers,
                chunk_size=chunk_size,
            )
            parallel_partial.replace(partial)
            if parallel_state.exists():
                parallel_state.unlink()
        digest = _md5_file(partial)
        if digest != AGROB_MODULAR_E_MD5:
            raise SystemExit(
                f"AgRob archive MD5 mismatch: expected {AGROB_MODULAR_E_MD5}, "
                f"got {digest}; retaining {partial}"
            )
        partial.replace(archive)
        print(f"verified {archive}")

    if extract:
        extracted = target_dir / "2023-06-13T170852Z"
        _extract_zip_safely(archive, extracted)
        print(f"extracted {extracted}")


def _download_agrob_range(chunk: Path, start: int, end: int) -> None:
    """Fetch one exact AgRob byte range with bounded retry/backoff."""

    expected_size = end - start + 1
    for attempt in range(1, 13):
        try:
            request = Request(
                AGROB_MODULAR_E_URL,
                headers={
                    "Accept": "*/*",
                    "Range": f"bytes={start}-{end}",
                    "User-Agent": "Calibrex-public-dataset-downloader",
                },
            )
            with urlopen(request, timeout=180) as response:
                content_range = response.headers.get("Content-Range", "")
                expected_range = f"bytes {start}-{end}/{AGROB_MODULAR_E_SIZE_BYTES}"
                if response.status != 206 or content_range != expected_range:
                    raise RuntimeError(
                        "AgRob Range request failed: "
                        f"status={response.status}, content-range={content_range!r}"
                    )
                with chunk.open("wb") as destination:
                    while piece := response.read(1024 * 1024):
                        destination.write(piece)
            if chunk.stat().st_size != expected_size:
                raise RuntimeError(
                    f"AgRob Range length mismatch: expected {expected_size}, "
                    f"got {chunk.stat().st_size}"
                )
            return
        except (HTTPError, OSError, RuntimeError, URLError) as exc:
            if attempt == 12:
                raise SystemExit(
                    f"AgRob Range {start}-{end} failed after {attempt} attempts: {exc}"
                ) from exc
            retry_after = getattr(getattr(exc, "headers", None), "get", lambda *_: None)(
                "Retry-After"
            )
            try:
                delay = max(5, min(120, int(retry_after))) if retry_after else 10
            except ValueError:
                delay = 10
            print(
                f"AgRob Range {start}-{end} attempt {attempt} failed; "
                f"retrying in {delay}s: {exc}"
            )
            time.sleep(delay)


def _download_agrob_ranges_parallel(
    partial: Path,
    state_path: Path,
    *,
    workers: int,
    chunk_size: int,
) -> None:
    """Resume exact AgRob ranges concurrently with an atomic progress ledger.

    Zenodo's public endpoint can be slow per connection while still accepting
    many independent ranges.  The preallocated file is never promoted to the
    public archive name until every range is complete and the full MD5 matches.
    """

    chunk_count = (AGROB_MODULAR_E_SIZE_BYTES + chunk_size - 1) // chunk_size
    completed: set[int]
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SystemExit(f"invalid AgRob parallel state: {state_path}") from exc
        expected_state = {
            "schema_version": "slac.agrob_parallel_download/v0.1",
            "size_bytes": AGROB_MODULAR_E_SIZE_BYTES,
            "chunk_size": chunk_size,
            "chunk_count": chunk_count,
            "md5": AGROB_MODULAR_E_MD5,
        }
        if any(state.get(key) != value for key, value in expected_state.items()):
            raise SystemExit(
                "AgRob parallel state does not match this archive/chunk layout; "
                f"remove or resume {state_path} explicitly"
            )
        raw_completed = state.get("completed_chunks", [])
        if not isinstance(raw_completed, list) or not all(
            isinstance(index, int) for index in raw_completed
        ):
            raise SystemExit(f"invalid completed chunk list: {state_path}")
        completed = set(raw_completed)
        if any(index < 0 or index >= chunk_count for index in completed):
            raise SystemExit(f"completed chunk index out of range: {state_path}")
        if not partial.exists() or partial.stat().st_size != AGROB_MODULAR_E_SIZE_BYTES:
            raise SystemExit(
                "AgRob parallel state exists but the preallocated partial file is "
                f"missing or has the wrong size: {partial}"
            )
    else:
        if partial.exists():
            raise SystemExit(
                f"AgRob parallel partial exists without a state ledger: {partial}"
            )
        with partial.open("wb") as stream:
            stream.seek(AGROB_MODULAR_E_SIZE_BYTES - 1)
            stream.write(b"\0")
        completed = set()
        _write_agrob_parallel_state(
            state_path,
            completed,
            chunk_size=chunk_size,
            chunk_count=chunk_count,
        )

    pending = [index for index in range(chunk_count) if index not in completed]
    if not pending:
        print(f"all {chunk_count} AgRob ranges are present; verifying")
        return

    state_lock = threading.Lock()
    started = time.monotonic()
    initial_completed = len(completed)

    def fetch(index: int) -> int:
        start = index * chunk_size
        end = min(start + chunk_size, AGROB_MODULAR_E_SIZE_BYTES) - 1
        expected_size = end - start + 1
        for attempt in range(1, 13):
            try:
                request = Request(
                    AGROB_MODULAR_E_URL,
                    headers={
                        "Accept": "*/*",
                        "Range": f"bytes={start}-{end}",
                        "User-Agent": "Calibrex-public-dataset-downloader",
                    },
                )
                with urlopen(request, timeout=300) as response:
                    content_range = response.headers.get("Content-Range", "")
                    expected_range = (
                        f"bytes {start}-{end}/{AGROB_MODULAR_E_SIZE_BYTES}"
                    )
                    if response.status != 206 or content_range != expected_range:
                        raise RuntimeError(
                            "AgRob parallel Range request failed: "
                            f"status={response.status}, content-range={content_range!r}"
                        )
                    # The server supplies Content-Length for the exact Range.
                    # Read that bounded amount once instead of waiting for an
                    # additional EOF read; some public proxies leave a
                    # completed response in CLOSE_WAIT for a long time.
                    payload = response.read(expected_size)
                    with partial.open("r+b") as destination:
                        destination.seek(start)
                        destination.write(payload)
                    received = len(payload)
                if received != expected_size:
                    raise RuntimeError(
                        f"AgRob parallel Range length mismatch: expected {expected_size}, "
                        f"got {received}"
                    )
                return index
            except (HTTPError, OSError, RuntimeError, URLError) as exc:
                if attempt == 12:
                    raise RuntimeError(
                        f"AgRob parallel Range {start}-{end} failed after "
                        f"{attempt} attempts: {exc}"
                    ) from exc
                retry_after = getattr(
                    getattr(exc, "headers", None), "get", lambda *_: None
                )("Retry-After")
                try:
                    delay = max(5, min(120, int(retry_after))) if retry_after else 10
                except ValueError:
                    delay = 10
                time.sleep(delay)

    print(
        f"downloading AgRob with {workers} Range workers: "
        f"{len(pending)} ranges of up to {chunk_size} bytes"
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch, index) for index in pending]
        for future in concurrent.futures.as_completed(futures):
            index = future.result()
            with state_lock:
                completed.add(index)
                _write_agrob_parallel_state(
                    state_path,
                    completed,
                    chunk_size=chunk_size,
                    chunk_count=chunk_count,
                )
                finished = len(completed) - initial_completed
                elapsed = max(time.monotonic() - started, 1.0)
                print(
                    f"AgRob ranges {len(completed)}/{chunk_count} "
                    f"({finished / elapsed:.2f}/s)"
                )


def _write_agrob_parallel_state(
    state_path: Path,
    completed: set[int],
    *,
    chunk_size: int,
    chunk_count: int,
) -> None:
    """Atomically persist the completed AgRob range indices."""

    payload = {
        "schema_version": "slac.agrob_parallel_download/v0.1",
        "size_bytes": AGROB_MODULAR_E_SIZE_BYTES,
        "chunk_size": chunk_size,
        "chunk_count": chunk_count,
        "md5": AGROB_MODULAR_E_MD5,
        "completed_chunks": sorted(completed),
    }
    temporary = state_path.with_name(state_path.name + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    temporary.replace(state_path)


def _md5_file(path: Path) -> str:
    """Return an MD5 digest without loading a public archive into memory."""

    digest = hashlib.md5()
    with path.open("rb") as stream:
        while piece := stream.read(8 * 1024 * 1024):
            digest.update(piece)
    return digest.hexdigest()


def _extract_zip_safely(archive: Path, destination: Path) -> None:
    """Extract a ZIP archive while rejecting paths outside ``destination``."""

    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if root != target and root not in target.parents:
                raise SystemExit(f"unsafe AgRob archive member: {member.filename}")
        bundle.extractall(destination)

if __name__ == "__main__":
    raise SystemExit(main())
