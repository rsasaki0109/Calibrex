#!/usr/bin/env python3
"""Range-download KITTI raw center images and LiDAR/OXTS windows."""

from __future__ import annotations

import argparse
import binascii
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal
from zipfile import ZipFile, ZipInfo

from tools.download_kitti360_selected_frames import (
    HTTPRangeReader,
    _copy_zip_member_with_cache_recovery,
    _materialize_member,
)

from calibrex.core.provenance import git_commit, sha256_path
from calibrex.data.kitti import read_timestamps
from calibrex.data.remote_archive_selection import (
    REMOTE_ARCHIVE_SELECTION_SCHEMA_VERSION_V0_3,
    RemoteArchiveSelectedMember,
    RemoteArchiveSelectedMetadataMember,
    RemoteArchiveSelectionArtifact,
    RemoteArchiveSelectionProvenance,
    RemoteArchiveSource,
)

DOWNLOADER_VERSION = "calibrex.download_kitti_raw_lidar_windows/v0.1"
_S3_ROOT = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
MetadataRole = Literal[
    "image_timestamps",
    "pointcloud_timestamps",
    "pose_timestamps",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize KITTI raw center images and centered LiDAR/OXTS "
            "windows from one official sync archive"
        )
    )
    parser.add_argument("sequence_id")
    parser.add_argument("--center-frame-ids", required=True)
    parser.add_argument("--neighbor-radius", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--camera-stream", choices=["image_02", "image_03"], default="image_02")
    parser.add_argument("--archive-url")
    parser.add_argument("--block-size-mib", type=int, default=8)
    parser.add_argument("--cache-block-count", type=int, default=16)
    parser.add_argument(
        "--read-cache-directory",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--write-cache-directory", type=Path)
    parser.add_argument("--prefix-cache-file", type=Path)
    parser.add_argument("--artifact-id")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.neighbor_radius < 0:
        raise ValueError("neighbor-radius must be non-negative")
    centers = _parse_frame_ids(args.center_frame_ids)
    sequence_id = str(args.sequence_id)
    if not sequence_id.endswith("_sync"):
        raise ValueError("KITTI raw sequence_id must end in _sync")
    date_id = sequence_id[:10]
    drive_id = sequence_id.removesuffix("_sync")
    archive_url = args.archive_url or (
        f"{_S3_ROOT}/{drive_id}/{sequence_id}.zip"
    )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reader = HTTPRangeReader(
        archive_url,
        block_size_bytes=args.block_size_mib * 1024 * 1024,
        cache_block_count=args.cache_block_count,
        read_cache_directories=args.read_cache_directory,
        write_cache_directory=args.write_cache_directory,
        prefix_cache_path=args.prefix_cache_file,
    )
    root_marker = f"{date_id}/{sequence_id}"
    with ZipFile(reader) as archive:
        image_index = _index_frames(
            archive.infolist(), root_marker, f"{args.camera_stream}/data", ".png"
        )
        lidar_index = _index_frames(
            archive.infolist(), root_marker, "velodyne_points/data", ".bin"
        )
        pose_index = _index_frames(
            archive.infolist(), root_marker, "oxts/data", ".txt"
        )
        timestamp_specs: tuple[tuple[MetadataRole, str, Path], ...] = (
            (
                "image_timestamps",
                f"{root_marker}/{args.camera_stream}/timestamps.txt",
                output_root / args.camera_stream / "timestamps.txt",
            ),
            (
                "pointcloud_timestamps",
                f"{root_marker}/velodyne_points/timestamps.txt",
                output_root / "velodyne_points" / "timestamps.txt",
            ),
            (
                "pose_timestamps",
                f"{root_marker}/oxts/timestamps.txt",
                output_root / "oxts" / "timestamps.txt",
            ),
        )
        metadata_members = [
            _materialize_metadata(archive, role, member_path, local_path)
            for role, member_path, local_path in timestamp_specs
        ]
        timestamp_limit = min(
            len(read_timestamps(Path(item.local_path)))
            for item in metadata_members
        )
        window_ids = _select_window_frame_ids(
            lidar_index,
            pose_index,
            centers,
            neighbor_radius=args.neighbor_radius,
            timestamp_limit=timestamp_limit,
        )
        pose_ids = _select_motion_pose_frame_ids(
            pose_index,
            window_ids,
            timestamp_limit=timestamp_limit,
        )
        missing_images = sorted(set(centers) - set(image_index), key=int)
        if missing_images:
            raise ValueError(
                "center images are absent from archive: "
                + ", ".join(missing_images[:10])
            )
        members: list[RemoteArchiveSelectedMember] = []
        for position, frame_id in enumerate(centers, start=1):
            members.append(
                _materialize_member(
                    archive,
                    image_index[frame_id],
                    output_root / args.camera_stream / "data" / f"{frame_id}.png",
                    role="image",
                    frame_id=frame_id,
                )
            )
            print(
                f"[image {position:03d}/{len(centers):03d}] {frame_id}",
                flush=True,
            )
        for position, frame_id in enumerate(window_ids, start=1):
            members.append(
                _materialize_member(
                    archive,
                    lidar_index[frame_id],
                    output_root / "velodyne_points" / "data" / f"{frame_id}.bin",
                    role="pointcloud",
                    frame_id=frame_id,
                )
            )
            print(
                f"[pointcloud {position:03d}/{len(window_ids):03d}] {frame_id}",
                flush=True,
            )
        for position, frame_id in enumerate(pose_ids, start=1):
            members.append(
                _materialize_member(
                    archive,
                    pose_index[frame_id],
                    output_root / "oxts" / "data" / f"{frame_id}.txt",
                    role="pose",
                    frame_id=frame_id,
                )
            )
            print(
                f"[pose {position:03d}/{len(pose_ids):03d}] {frame_id}",
                flush=True,
            )
    selected_ids = sorted(set(centers) | set(window_ids) | set(pose_ids), key=int)
    artifact = RemoteArchiveSelectionArtifact(
        schema_version=REMOTE_ARCHIVE_SELECTION_SCHEMA_VERSION_V0_3,
        artifact_id=(
            args.artifact_id
            or f"kitti-raw-{sequence_id}-{args.camera_stream}-r{args.neighbor_radius}"
        ),
        dataset_family="KITTI raw",
        dataset_id=f"kitti_raw_{drive_id}",
        sequence_id=sequence_id,
        dataset_license_spdx="LicenseRef-KITTI",
        selection_policy=(
            "center_images_with_centered_pointcloud_pose_windows/v0.3"
        ),
        requested_frame_count=len(selected_ids),
        frame_ids=selected_ids,
        center_frame_ids=centers,
        stream_frame_ids={
            "image": centers,
            "pointcloud": window_ids,
            "pose": pose_ids,
        },
        archives=[
            RemoteArchiveSource(
                role="multi_stream",
                url=reader.url,
                size_bytes=reader.size_bytes,
                etag=reader.etag,
                last_modified=reader.last_modified,
                accept_ranges=reader.accept_ranges,
            )
        ],
        members=members,
        metadata_members=metadata_members,
        output_root=str(output_root),
        provenance=RemoteArchiveSelectionProvenance(
            generator="tools.download_kitti_raw_lidar_windows",
            generator_version=DOWNLOADER_VERSION,
            git_commit=git_commit(),
            command=sys.argv,
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


def _parse_frame_ids(raw: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    normalized = [f"{int(value):010d}" for value in values if value.isdigit()]
    if len(normalized) != len(values) or len(normalized) < 2:
        raise ValueError("center-frame-ids must contain at least two numeric IDs")
    if normalized != sorted(set(normalized), key=int):
        raise ValueError("center frame IDs must be unique and ordered")
    return normalized


def _index_frames(
    infos: Sequence[ZipInfo],
    root_marker: str,
    relative_directory: str,
    extension: str,
) -> dict[str, ZipInfo]:
    prefix = f"{root_marker}/{relative_directory.strip('/')}/"
    result: dict[str, ZipInfo] = {}
    for info in infos:
        normalized = info.filename.replace("\\", "/")
        if not normalized.startswith(prefix) or not normalized.endswith(extension):
            continue
        frame_id = normalized.rsplit("/", 1)[-1].removesuffix(extension)
        if len(frame_id) != 10 or not frame_id.isdigit():
            continue
        if frame_id in result:
            raise ValueError(f"duplicate archive frame: {frame_id}")
        result[frame_id] = info
    if not result:
        raise ValueError(f"archive contains no frames under {prefix}")
    return result


def _select_window_frame_ids(
    lidar_index: dict[str, ZipInfo],
    pose_index: dict[str, ZipInfo],
    center_frame_ids: Sequence[str],
    *,
    neighbor_radius: int,
    timestamp_limit: int,
) -> list[str]:
    available = {
        frame_id
        for frame_id in lidar_index.keys() & pose_index.keys()
        if int(frame_id) < timestamp_limit
    }
    missing_centers = sorted(set(center_frame_ids) - available, key=int)
    if missing_centers:
        raise ValueError(
            "center LiDAR/OXTS frames are absent: " + ", ".join(missing_centers)
        )
    minimum = min(int(value) for value in available)
    maximum = max(int(value) for value in available)
    requested = {
        f"{frame_id:010d}"
        for center in center_frame_ids
        for frame_id in range(
            max(minimum, int(center) - neighbor_radius),
            min(maximum, int(center) + neighbor_radius) + 1,
        )
    }
    missing = sorted(requested - available, key=int)
    if missing:
        raise ValueError(
            "centered LiDAR/OXTS window has missing frames: "
            + ", ".join(missing[:10])
        )
    return sorted(requested, key=int)


def _select_motion_pose_frame_ids(
    pose_index: dict[str, ZipInfo],
    pointcloud_frame_ids: Sequence[str],
    *,
    timestamp_limit: int,
) -> list[str]:
    requested = {
        frame_id
        for source in pointcloud_frame_ids
        for frame_id in (source, _motion_adjacent_frame_id(source))
    }
    available = {
        frame_id for frame_id in pose_index if int(frame_id) < timestamp_limit
    }
    missing = sorted(requested - available, key=int)
    if missing:
        raise ValueError(
            "motion-compensation OXTS frames are absent: "
            + ", ".join(missing[:10])
        )
    return sorted(requested, key=int)


def _motion_adjacent_frame_id(frame_id: str) -> str:
    value = int(frame_id)
    adjacent = value + 1 if value in (0, 1) else value - 1
    return f"{adjacent:010d}"


def _materialize_metadata(
    archive: ZipFile,
    role: MetadataRole,
    member_path: str,
    output_path: Path,
) -> RemoteArchiveSelectedMetadataMember:
    info = archive.getinfo(member_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(f".{output_path.name}.calibrex-partial")
    if output_path.exists():
        if not _matches_zip_info(output_path, info):
            raise ValueError(f"existing metadata differs from archive: {output_path}")
        partial.unlink(missing_ok=True)
    else:
        partial.unlink(missing_ok=True)
        _copy_zip_member_with_cache_recovery(archive, info, partial)
        if not _matches_zip_info(partial, info):
            raise OSError(f"metadata failed ZIP CRC/size check: {member_path}")
        partial.replace(output_path)
    digest = sha256_path(output_path)
    if digest is None:
        raise OSError(f"metadata is unreadable after extraction: {output_path}")
    return RemoteArchiveSelectedMetadataMember(
        role=role,
        member_path=info.filename,
        crc32=f"{info.CRC:08x}",
        compressed_size_bytes=info.compress_size,
        uncompressed_size_bytes=info.file_size,
        local_path=str(output_path.resolve()),
        local_sha256=digest,
        local_size_bytes=output_path.stat().st_size,
    )


def _matches_zip_info(path: Path, info: ZipInfo) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == info.file_size
        and _crc32_path(path) == info.CRC
    )


def _crc32_path(path: Path) -> int:
    checksum = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum = binascii.crc32(chunk, checksum)
    return checksum & 0xFFFFFFFF


if __name__ == "__main__":
    raise SystemExit(main())
