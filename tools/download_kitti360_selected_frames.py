#!/usr/bin/env python3
"""Range-download a provenance-locked KITTI-360 development subset."""

from __future__ import annotations

import argparse
import binascii
import io
import shutil
import sys
import time
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile, ZipInfo

from calibrex.core.provenance import git_commit, sha256_path
from calibrex.data.kitti import read_timestamps
from calibrex.data.kitti_camera_lidar_problem import uniform_kitti_frame_ids
from calibrex.data.remote_archive_selection import (
    RemoteArchiveRole,
    RemoteArchiveSelectedMember,
    RemoteArchiveSelectionArtifact,
    RemoteArchiveSelectionProvenance,
    RemoteArchiveSource,
)

DOWNLOADER_VERSION = "calibrex.download_kitti360_selected_frames/v0.5"
_S3_ROOT = "https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360"
_USER_AGENT = "Calibrex-KITTI360-Range-Downloader/0.1"


class HTTPRangeReader(io.RawIOBase):
    """Seekable, block-cached HTTP byte-range reader."""

    def __init__(
        self,
        url: str,
        *,
        block_size_bytes: int = 8 * 1024 * 1024,
        cache_block_count: int = 8,
        timeout_seconds: float = 60.0,
        retry_count: int = 4,
        read_cache_directories: Sequence[str | Path] = (),
        write_cache_directory: str | Path | None = None,
        prefix_cache_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        if block_size_bytes < 64 * 1024:
            raise ValueError("block_size_bytes must be at least 64 KiB")
        if cache_block_count < 1:
            raise ValueError("cache_block_count must be positive")
        if timeout_seconds <= 0.0 or retry_count < 1:
            raise ValueError("timeout and retry count must be positive")
        self.url = url
        self.block_size_bytes = block_size_bytes
        self.cache_block_count = cache_block_count
        self.timeout_seconds = timeout_seconds
        self.retry_count = retry_count
        self.read_cache_directories: tuple[Path, ...] = tuple(
            Path(path).resolve() for path in read_cache_directories
        )
        self.write_cache_directory: Path | None = (
            Path(write_cache_directory).resolve()
            if write_cache_directory is not None
            else None
        )
        self.prefix_cache_path: Path | None = (
            Path(prefix_cache_path).resolve()
            if prefix_cache_path is not None
            else None
        )
        if self.write_cache_directory is not None:
            self.write_cache_directory.mkdir(parents=True, exist_ok=True)
        self.position = 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        self._bypass_cached_blocks: set[int] = set()
        headers = self._head()
        content_length = headers.get("Content-Length")
        if content_length is None:
            raise ValueError(f"remote archive has no Content-Length: {url}")
        self.size_bytes = int(content_length)
        self.etag = headers.get("ETag")
        self.last_modified = headers.get("Last-Modified")
        self.accept_ranges = headers.get("Accept-Ranges", "").lower() == "bytes"
        if not self.accept_ranges:
            raise ValueError(f"remote archive does not advertise byte ranges: {url}")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            target = offset
        elif whence == io.SEEK_CUR:
            target = self.position + offset
        elif whence == io.SEEK_END:
            target = self.size_bytes + offset
        else:
            raise ValueError(f"unsupported seek mode: {whence}")
        if target < 0:
            raise ValueError("cannot seek before the start of a remote archive")
        self.position = min(target, self.size_bytes)
        return self.position

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.size_bytes:
            return b""
        requested = self.size_bytes - self.position if size < 0 else size
        requested = min(requested, self.size_bytes - self.position)
        if requested <= 0:
            return b""
        start = self.position
        end = start + requested
        chunks: list[bytes] = []
        cursor = start
        while cursor < end:
            block_index = cursor // self.block_size_bytes
            block = self._block(block_index)
            block_start = block_index * self.block_size_bytes
            local_start = cursor - block_start
            local_end = min(len(block), end - block_start)
            chunks.append(block[local_start:local_end])
            cursor = block_start + local_end
        payload = b"".join(chunks)
        if len(payload) != requested:
            raise OSError(
                f"remote range read length mismatch: expected {requested}, got {len(payload)}"
            )
        self.position = end
        return payload

    def _head(self) -> dict[str, str]:
        request = Request(
            self.url,
            method="HEAD",
            headers={"User-Agent": _USER_AGENT},
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            return dict(response.headers.items())

    def _block(self, block_index: int) -> bytes:
        cached = self.cache.pop(block_index, None)
        if cached is not None:
            self.cache[block_index] = cached
            return cached
        start = block_index * self.block_size_bytes
        end = min(self.size_bytes, start + self.block_size_bytes) - 1
        expected = end - start + 1
        payload = self._cached_block(block_index, start, expected)
        if payload is None:
            payload = self._fetch_range(start, end)
            self._write_cached_block(block_index, payload)
            if self.write_cache_directory is not None:
                self._bypass_cached_blocks.discard(block_index)
        self.cache[block_index] = payload
        while len(self.cache) > self.cache_block_count:
            self.cache.popitem(last=False)
        return payload

    def _cached_block(
        self,
        block_index: int,
        start: int,
        expected: int,
    ) -> bytes | None:
        if block_index in self._bypass_cached_blocks:
            return None
        cache_name = f"part-{block_index:04d}.bin"
        directories: tuple[Path, ...] = self.read_cache_directories
        if self.write_cache_directory is not None:
            directories = (self.write_cache_directory, *directories)
        for directory in directories:
            path = directory / cache_name
            if path.is_file() and path.stat().st_size == expected:
                return path.read_bytes()
        prefix = self.prefix_cache_path
        if (
            prefix is not None
            and prefix.is_file()
            and start + expected <= prefix.stat().st_size
        ):
            with prefix.open("rb") as handle:
                handle.seek(start)
                payload: bytes = handle.read(expected)
            if len(payload) == expected:
                return payload
        return None

    def invalidate_cached_range(self, start: int, end: int) -> None:
        """Bypass local cache blocks after a ZIP integrity failure.

        ``end`` is exclusive. The next read of every affected block is served
        by an authenticated HTTP range response and, when configured, written
        to the higher-priority write cache.
        """

        if start < 0 or end <= start or end > self.size_bytes:
            raise ValueError("invalid cache invalidation range")
        first_block = start // self.block_size_bytes
        last_block = (end - 1) // self.block_size_bytes
        for block_index in range(first_block, last_block + 1):
            self.cache.pop(block_index, None)
            self._bypass_cached_blocks.add(block_index)

    def _write_cached_block(self, block_index: int, payload: bytes) -> None:
        directory = self.write_cache_directory
        if directory is None:
            return
        path = directory / f"part-{block_index:04d}.bin"
        partial = path.with_name(f".{path.name}.calibrex-partial")
        if path.exists():
            if path.read_bytes() != payload:
                raise OSError(f"range cache block differs from fetched bytes: {path}")
            return
        partial.unlink(missing_ok=True)
        with partial.open("xb") as handle:
            handle.write(payload)
        partial.replace(path)

    def _fetch_range(self, start: int, end: int) -> bytes:
        expected = end - start + 1
        for attempt in range(1, self.retry_count + 1):
            request = Request(
                self.url,
                headers={
                    "Range": f"bytes={start}-{end}",
                    "User-Agent": _USER_AGENT,
                },
            )
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    status = getattr(response, "status", response.getcode())
                    content_range = response.headers.get("Content-Range", "")
                    payload = bytes(response.read())
                if status != 206 or not content_range.startswith(f"bytes {start}-{end}/"):
                    raise OSError(
                        f"server returned invalid range response {status}: {content_range}"
                    )
                if len(payload) != expected:
                    raise OSError(
                        f"range {start}-{end} expected {expected} bytes, got {len(payload)}"
                    )
                return payload
            except (HTTPError, URLError, OSError) as exc:
                if attempt == self.retry_count:
                    raise OSError(
                        f"range {start}-{end} failed after {attempt} attempts: {exc}"
                    ) from exc
                time.sleep(min(8.0, float(2 ** (attempt - 1))))
        raise AssertionError("unreachable range retry state")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize endpoint-inclusive KITTI-360 fisheye/Velodyne files "
            "without downloading whole sequence archives"
        )
    )
    parser.add_argument("sequence_id")
    parser.add_argument("--image-timestamps", type=Path, required=True)
    parser.add_argument("--velodyne-timestamps", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--camera-stream", choices=["image_02", "image_03"], default="image_03")
    parser.add_argument("--frame-count", type=int, default=25)
    parser.add_argument(
        "--center-frame-ids",
        help="comma-separated frame IDs whose inclusive windows are materialized",
    )
    parser.add_argument("--neighbor-radius", type=int, default=0)
    parser.add_argument("--image-archive-url")
    parser.add_argument("--velodyne-archive-url")
    parser.add_argument("--block-size-mib", type=int, default=8)
    parser.add_argument("--cache-block-count", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.frame_count < 2:
        raise ValueError("frame-count must be at least two")
    if args.neighbor_radius < 0:
        raise ValueError("neighbor-radius must be non-negative")
    center_frame_ids = _parse_center_frame_ids(args.center_frame_ids)
    image_timestamps_path = args.image_timestamps.resolve()
    velodyne_timestamps_path = args.velodyne_timestamps.resolve()
    image_timestamps = read_timestamps(image_timestamps_path)
    velodyne_timestamps = read_timestamps(velodyne_timestamps_path)
    camera_suffix = args.camera_stream[-2:]
    image_url = args.image_archive_url or (
        f"{_S3_ROOT}/data_2d_raw/{args.sequence_id}_image_{camera_suffix}.zip"
    )
    velodyne_url = args.velodyne_archive_url or (
        f"{_S3_ROOT}/data_3d_raw/{args.sequence_id}_velodyne.zip"
    )
    output_root = args.output_root.resolve()
    image_directory = output_root / args.camera_stream / "data_rgb"
    velodyne_directory = output_root / "velodyne_points" / "data"
    image_directory.mkdir(parents=True, exist_ok=True)
    velodyne_directory.mkdir(parents=True, exist_ok=True)
    _copy_verified(
        image_timestamps_path,
        output_root / args.camera_stream / "timestamps.txt",
    )
    _copy_verified(
        velodyne_timestamps_path,
        output_root / "velodyne_points" / "timestamps.txt",
    )
    block_size_bytes = args.block_size_mib * 1024 * 1024
    image_reader = HTTPRangeReader(
        image_url,
        block_size_bytes=block_size_bytes,
        cache_block_count=args.cache_block_count,
    )
    velodyne_reader = HTTPRangeReader(
        velodyne_url,
        block_size_bytes=block_size_bytes,
        cache_block_count=args.cache_block_count,
    )
    members: list[RemoteArchiveSelectedMember] = []
    selection_policy: Literal[
        "endpoint_inclusive_same_numeric_frame_id/v0.1",
        "centered_same_numeric_frame_windows/v0.2",
    ]
    with ZipFile(image_reader) as image_archive, ZipFile(velodyne_reader) as lidar_archive:
        image_infos = image_archive.infolist()
        lidar_infos = lidar_archive.infolist()
        image_index = _index_members(
            image_infos,
            args.sequence_id,
            f"{args.camera_stream}/data_rgb",
            ".png",
        )
        lidar_index = _index_members(
            lidar_infos,
            args.sequence_id,
            "velodyne_points/data",
            ".bin",
        )
        if center_frame_ids:
            frame_ids = _select_centered_common_frame_ids(
                image_index,
                lidar_index,
                image_timestamp_count=len(image_timestamps),
                velodyne_timestamp_count=len(velodyne_timestamps),
                center_frame_ids=center_frame_ids,
                neighbor_radius=args.neighbor_radius,
            )
            selection_policy = "centered_same_numeric_frame_windows/v0.2"
        else:
            frame_ids = _select_common_frame_ids(
                image_index,
                lidar_index,
                image_timestamp_count=len(image_timestamps),
                velodyne_timestamp_count=len(velodyne_timestamps),
                selected_count=args.frame_count,
            )
            selection_policy = "endpoint_inclusive_same_numeric_frame_id/v0.1"
        for position, frame_id in enumerate(frame_ids, start=1):
            image_info = image_index[frame_id]
            lidar_info = lidar_index[frame_id]
            members.append(
                _materialize_member(
                    image_archive,
                    image_info,
                    image_directory / f"{frame_id}.png",
                    role="image",
                    frame_id=frame_id,
                )
            )
            members.append(
                _materialize_member(
                    lidar_archive,
                    lidar_info,
                    velodyne_directory / f"{frame_id}.bin",
                    role="pointcloud",
                    frame_id=frame_id,
                )
            )
            print(
                f"[{position:02d}/{len(frame_ids):02d}] materialized frame {frame_id}",
                flush=True,
            )
    timestamp_digests = {
        str(image_timestamps_path): _required_digest(image_timestamps_path),
        str(velodyne_timestamps_path): _required_digest(velodyne_timestamps_path),
    }
    artifact = RemoteArchiveSelectionArtifact(
        artifact_id=f"kitti360-{args.sequence_id}-{args.camera_stream}-{len(frame_ids)}",
        dataset_family="KITTI-360",
        dataset_id=f"kitti360_{args.sequence_id.removesuffix('_sync')}",
        sequence_id=args.sequence_id,
        dataset_license_spdx="CC-BY-NC-SA-3.0",
        selection_policy=selection_policy,
        requested_frame_count=len(frame_ids),
        frame_ids=frame_ids,
        archives=[
            _archive_source("image", image_reader),
            _archive_source("pointcloud", velodyne_reader),
        ],
        members=members,
        output_root=str(output_root),
        provenance=RemoteArchiveSelectionProvenance(
            generator="tools.download_kitti360_selected_frames",
            generator_version=DOWNLOADER_VERSION,
            git_commit=git_commit(),
            command=sys.argv,
            input_sha256=timestamp_digests,
        ),
    )
    manifest_output = args.manifest_output.resolve()
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    artifact.save(manifest_output)
    print(
        f"saved {len(members)} selected files and manifest {manifest_output}",
        flush=True,
    )
    return 0


def _index_members(
    infos: list[ZipInfo],
    sequence_id: str,
    relative_directory: str,
    extension: str,
) -> dict[str, ZipInfo]:
    marker = f"{sequence_id}/{relative_directory.strip('/')}/"
    indexed: dict[str, ZipInfo] = {}
    for info in infos:
        normalized = info.filename.replace("\\", "/")
        if marker not in normalized or not normalized.endswith(extension):
            continue
        filename = normalized.rsplit("/", 1)[-1]
        frame_id = filename.removesuffix(extension)
        if len(frame_id) != 10 or not frame_id.isdigit():
            continue
        if frame_id in indexed:
            raise ValueError(f"duplicate ZIP member for frame {frame_id}: {marker}")
        indexed[frame_id] = info
    if not indexed:
        raise ValueError(f"ZIP archive contains no members under {marker}")
    return indexed


def _select_common_frame_ids(
    image_index: dict[str, ZipInfo],
    lidar_index: dict[str, ZipInfo],
    *,
    image_timestamp_count: int,
    velodyne_timestamp_count: int,
    selected_count: int,
) -> list[str]:
    timestamp_limit = min(image_timestamp_count, velodyne_timestamp_count)
    common_frame_ids = sorted(
        (
            frame_id
            for frame_id in image_index.keys() & lidar_index.keys()
            if int(frame_id) < timestamp_limit
        ),
        key=int,
    )
    if len(common_frame_ids) < selected_count:
        raise ValueError(
            "image/LiDAR ZIP member intersection with timestamp coverage has "
            f"{len(common_frame_ids)} frames, cannot select {selected_count}"
        )
    selected_positions = uniform_kitti_frame_ids(
        len(common_frame_ids),
        selected_count,
    )
    return [common_frame_ids[int(position)] for position in selected_positions]


def _parse_center_frame_ids(raw: str | None) -> list[str]:
    if raw is None:
        return []
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if len(values) < 2:
        raise ValueError("center-frame-ids requires at least two frame IDs")
    normalized: list[str] = []
    for value in values:
        if not value.isdigit() or int(value) < 0:
            raise ValueError(f"invalid center frame ID: {value!r}")
        normalized.append(f"{int(value):010d}")
    if len(normalized) != len(set(normalized)):
        raise ValueError("center frame IDs must be unique")
    return normalized


def _select_centered_common_frame_ids(
    image_index: dict[str, ZipInfo],
    lidar_index: dict[str, ZipInfo],
    *,
    image_timestamp_count: int,
    velodyne_timestamp_count: int,
    center_frame_ids: list[str],
    neighbor_radius: int,
) -> list[str]:
    timestamp_limit = min(image_timestamp_count, velodyne_timestamp_count)
    available = {
        frame_id
        for frame_id in image_index.keys() & lidar_index.keys()
        if int(frame_id) < timestamp_limit
    }
    if not available:
        raise ValueError("image/LiDAR/timestamp intersection is empty")
    missing_centers = sorted(set(center_frame_ids) - available, key=int)
    if missing_centers:
        raise ValueError(
            "center frames are absent from the image/LiDAR/timestamp intersection: "
            + ", ".join(missing_centers[:10])
        )
    minimum_available = min(int(frame_id) for frame_id in available)
    maximum_available = max(int(frame_id) for frame_id in available)
    requested = {
        f"{int(center):010d}"
        for center_frame_id in center_frame_ids
        for center in range(
            int(center_frame_id) - neighbor_radius,
            int(center_frame_id) + neighbor_radius + 1,
        )
        if minimum_available <= center <= maximum_available
    }
    missing = sorted(requested - available, key=int)
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            "centered frame window is absent from the image/LiDAR/timestamp "
            f"intersection: {preview}"
        )
    return sorted(requested, key=int)


def _materialize_member(
    archive: ZipFile,
    info: ZipInfo,
    output_path: Path,
    *,
    role: RemoteArchiveRole,
    frame_id: str,
) -> RemoteArchiveSelectedMember:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(f".{output_path.name}.calibrex-partial")
    if output_path.exists():
        if not _matches_zip_info(output_path, info):
            raise ValueError(
                f"existing selected file differs from archive member: {output_path}"
            )
        partial_path.unlink(missing_ok=True)
    else:
        if partial_path.exists() and not _matches_zip_info(partial_path, info):
            partial_path.unlink()
        if not partial_path.exists():
            _copy_zip_member_with_cache_recovery(
                archive,
                info,
                partial_path,
                copy_length=1024 * 1024,
            )
        if not _matches_zip_info(partial_path, info):
            raise OSError(f"materialized ZIP member failed CRC/size check: {info.filename}")
        partial_path.replace(output_path)
    return RemoteArchiveSelectedMember(
        role=role,
        frame_id=frame_id,
        member_path=info.filename,
        crc32=f"{info.CRC:08x}",
        compressed_size_bytes=info.compress_size,
        uncompressed_size_bytes=info.file_size,
        local_path=str(output_path.resolve()),
        local_sha256=_required_digest(output_path),
        local_size_bytes=output_path.stat().st_size,
    )


def _copy_zip_member_with_cache_recovery(
    archive: ZipFile,
    info: ZipInfo,
    output_path: Path,
    *,
    copy_length: int = 1024 * 1024,
) -> None:
    """Extract one member, repairing stale local range-cache blocks once."""

    for attempt in range(2):
        output_path.unlink(missing_ok=True)
        try:
            with archive.open(info, "r") as source, output_path.open("xb") as target:
                shutil.copyfileobj(source, target, length=copy_length)
            return
        except BadZipFile:
            output_path.unlink(missing_ok=True)
            reader = archive.fp
            if attempt > 0 or not isinstance(reader, HTTPRangeReader):
                raise
            encoded_name_size = len(info.filename.encode("utf-8"))
            member_end = min(
                reader.size_bytes,
                info.header_offset
                + 30
                + encoded_name_size
                + len(info.extra)
                + info.compress_size
                + 24,
            )
            reader.invalidate_cached_range(info.header_offset, member_end)


def _archive_source(
    role: RemoteArchiveRole, reader: HTTPRangeReader
) -> RemoteArchiveSource:
    return RemoteArchiveSource(
        role=role,
        url=reader.url,
        size_bytes=reader.size_bytes,
        etag=reader.etag,
        last_modified=reader.last_modified,
        accept_ranges=reader.accept_ranges,
    )


def _copy_verified(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_digest = _required_digest(source)
    if destination.exists():
        if _required_digest(destination) != source_digest:
            raise ValueError(f"existing timestamp copy differs from source: {destination}")
        return
    shutil.copy2(source, destination)
    if _required_digest(destination) != source_digest:
        raise OSError(f"timestamp copy digest mismatch: {destination}")


def _crc32_path(path: Path) -> int:
    checksum = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum = binascii.crc32(chunk, checksum)
    return checksum & 0xFFFFFFFF


def _matches_zip_info(path: Path, info: ZipInfo) -> bool:
    return path.stat().st_size == info.file_size and _crc32_path(path) == info.CRC


def _required_digest(path: Path) -> str:
    digest = sha256_path(path)
    if digest is None:
        raise ValueError(f"required file is not readable: {path}")
    return digest


if __name__ == "__main__":
    raise SystemExit(main())
