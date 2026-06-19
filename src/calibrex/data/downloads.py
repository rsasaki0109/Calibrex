"""Small public dataset download helpers used by demos and tools."""

from __future__ import annotations

import tarfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

LIVOX_BASE_PCD_URL = (
    "https://terra-1-g.djicdn.com/65c028cd298f4669a7f0e40e50ba1131/"
    "Showcase/Base_LiDAR_Frames.tar.gz"
)
LIVOX_TARGET_PCD_URL = (
    "https://terra-1-g.djicdn.com/65c028cd298f4669a7f0e40e50ba1131/"
    "Showcase/Target-LiDAR-Frames.tar.gz"
)
LIVOX_PAIR_DIRNAME = "livox_horizon_horizon_pair"
LIVOX_BASE_PCD_NAME = "base_horizon_100432.pcd"
LIVOX_TARGET_PCD_NAME = "target_horizon_100538.pcd"


@dataclass(frozen=True)
class DownloadedDataset:
    """Local materialization of a small public dataset."""

    dataset_id: str
    path: Path
    files: tuple[Path, ...]
    source_urls: tuple[str, ...]
    downloaded: bool


def download_livox_horizon_horizon_pcd_sample(
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> DownloadedDataset:
    """Stream one real base/target PCD frame from Livox public tarballs."""

    output_path = Path(output_dir)
    target_dir = output_path / LIVOX_PAIR_DIRNAME
    target_dir.mkdir(parents=True, exist_ok=True)
    base_pcd = target_dir / LIVOX_BASE_PCD_NAME
    target_pcd = target_dir / LIVOX_TARGET_PCD_NAME
    downloaded = False
    if overwrite or not base_pcd.exists():
        _extract_first_pcd_from_tar_gz(LIVOX_BASE_PCD_URL, base_pcd)
        downloaded = True
    if overwrite or not target_pcd.exists():
        _extract_first_pcd_from_tar_gz(LIVOX_TARGET_PCD_URL, target_pcd)
        downloaded = True
    return DownloadedDataset(
        dataset_id="livox_horizon_horizon_pcd_sample",
        path=target_dir,
        files=(base_pcd, target_pcd),
        source_urls=(LIVOX_BASE_PCD_URL, LIVOX_TARGET_PCD_URL),
        downloaded=downloaded,
    )


def livox_horizon_horizon_pcd_sample_path(output_dir: str | Path) -> Path:
    """Return the expected local path for the Livox Horizon-Horizon PCD sample."""

    return Path(output_dir) / LIVOX_PAIR_DIRNAME


def _extract_first_pcd_from_tar_gz(url: str, output: Path) -> None:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=90) as response, tarfile.open(
        fileobj=response,
        mode="r|gz",
    ) as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".pcd"):
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            output.write_bytes(handle.read())
            return
    msg = f"{url} did not contain a PCD file"
    raise RuntimeError(msg)
