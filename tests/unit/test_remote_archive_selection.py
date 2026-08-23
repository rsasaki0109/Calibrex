from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest
import tools.download_kitti360_selected_frames as downloader
import tools.download_kitti_raw_lidar_windows as raw_downloader

from calibrex.core.validation import validate_file
from calibrex.data.remote_archive_selection import (
    REMOTE_ARCHIVE_SELECTION_SCHEMA_VERSION_V0_3,
    RemoteArchiveSelectedMember,
    RemoteArchiveSelectedMetadataMember,
    RemoteArchiveSelectionArtifact,
    RemoteArchiveSelectionProvenance,
    RemoteArchiveSource,
    verify_remote_archive_selection_files,
)


def test_remote_archive_selection_is_schema_valid(tmp_path: Path) -> None:
    frame_ids = ["0000000000", "0000000010"]
    members = [
        RemoteArchiveSelectedMember(
            role=role,
            frame_id=frame_id,
            member_path=f"archive/{role}/{frame_id}.{extension}",
            crc32="1234abcd",
            compressed_size_bytes=8,
            uncompressed_size_bytes=10,
            local_path=str(tmp_path / role / f"{frame_id}.{extension}"),
            local_sha256="a" * 64,
            local_size_bytes=10,
        )
        for role, extension in (("image", "png"), ("pointcloud", "bin"))
        for frame_id in frame_ids
    ]
    artifact = RemoteArchiveSelectionArtifact(
        artifact_id="kitti360-drive-0002-dev",
        dataset_family="KITTI-360",
        dataset_id="kitti360_2013_05_28_drive_0002",
        sequence_id="2013_05_28_drive_0002_sync",
        dataset_license_spdx="CC-BY-NC-SA-3.0",
        requested_frame_count=2,
        frame_ids=frame_ids,
        archives=[
            RemoteArchiveSource(
                role="image",
                url="https://example.test/images.zip",
                size_bytes=100,
                etag='"image"',
                accept_ranges=True,
            ),
            RemoteArchiveSource(
                role="pointcloud",
                url="https://example.test/lidar.zip",
                size_bytes=100,
                etag='"lidar"',
                accept_ranges=True,
            ),
        ],
        members=members,
        output_root=str(tmp_path),
        provenance=RemoteArchiveSelectionProvenance(
            generator="pytest",
            generator_version="1",
            input_sha256={"timestamps": "b" * 64},
        ),
    )
    output = tmp_path / "selection.yaml"

    artifact.save(output)

    assert validate_file(output, "remote-archive-selection").valid
    with pytest.raises(ValueError):
        RemoteArchiveSelectionArtifact.model_validate(
            artifact.model_dump(exclude={"members"})
            | {"members": artifact.model_dump()["members"][:-1]}
        )


def test_remote_archive_selection_verifies_local_file_integrity(
    tmp_path: Path,
) -> None:
    frame_ids = ["0000000000", "0000000010"]
    members: list[RemoteArchiveSelectedMember] = []
    for role, extension in (("image", "png"), ("pointcloud", "bin")):
        for frame_id in frame_ids:
            relative_path = Path(role) / f"{frame_id}.{extension}"
            path = tmp_path / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = f"{role}:{frame_id}".encode()
            path.write_bytes(payload)
            members.append(
                RemoteArchiveSelectedMember(
                    role=role,
                    frame_id=frame_id,
                    member_path=f"archive/{relative_path.as_posix()}",
                    crc32="1234abcd",
                    compressed_size_bytes=len(payload),
                    uncompressed_size_bytes=len(payload),
                    local_path=str(relative_path),
                    local_sha256=hashlib.sha256(payload).hexdigest(),
                    local_size_bytes=len(payload),
                )
            )
    artifact = RemoteArchiveSelectionArtifact(
        artifact_id="integrity-fixture",
        dataset_family="KITTI-360",
        dataset_id="kitti360_fixture",
        sequence_id="fixture",
        dataset_license_spdx="CC-BY-NC-SA-3.0",
        requested_frame_count=2,
        frame_ids=frame_ids,
        archives=[
            RemoteArchiveSource(
                role="image",
                url="https://example.test/images.zip",
                size_bytes=100,
                accept_ranges=True,
            ),
            RemoteArchiveSource(
                role="pointcloud",
                url="https://example.test/lidar.zip",
                size_bytes=100,
                accept_ranges=True,
            ),
        ],
        members=members,
        output_root=str(tmp_path),
        provenance=RemoteArchiveSelectionProvenance(
            generator="pytest",
            generator_version="1",
        ),
    )

    assert verify_remote_archive_selection_files(artifact, base_path=tmp_path) == []

    tampered = tmp_path / "image" / "0000000000.png"
    tampered.write_bytes(b"x" * tampered.stat().st_size)
    missing = tmp_path / "pointcloud" / "0000000010.bin"
    missing.unlink()

    issues = verify_remote_archive_selection_files(artifact, base_path=tmp_path)
    assert issues == [
        "digest_mismatch:member:image:0000000000:image\\0000000000.png",
        "missing:member:pointcloud:0000000010:pointcloud\\0000000010.bin",
    ]


def test_http_range_reader_supports_zipfile_random_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_buffer = io.BytesIO()
    with ZipFile(archive_buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("sequence/image_03/data_rgb/0000000000.png", b"image-data")
        archive.writestr("sequence/velodyne_points/data/0000000000.bin", b"lidar-data")
    payload = archive_buffer.getvalue()

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        del timeout
        if request.get_method() == "HEAD":
            return _FakeResponse(
                b"",
                status=200,
                headers={
                    "Content-Length": str(len(payload)),
                    "Accept-Ranges": "bytes",
                    "ETag": '"fixture"',
                },
            )
        raw_range = request.get_header("Range")
        assert raw_range is not None
        start_text, end_text = raw_range.removeprefix("bytes=").split("-", 1)
        start = int(start_text)
        end = int(end_text)
        return _FakeResponse(
            payload[start : end + 1],
            status=206,
            headers={"Content-Range": f"bytes {start}-{end}/{len(payload)}"},
        )

    monkeypatch.setattr(downloader, "urlopen", fake_urlopen)
    reader = downloader.HTTPRangeReader(
        "https://example.test/archive.zip",
        block_size_bytes=64 * 1024,
    )

    with ZipFile(reader) as archive:
        assert (
            archive.read("sequence/image_03/data_rgb/0000000000.png")
            == b"image-data"
        )
        assert (
            archive.read("sequence/velodyne_points/data/0000000000.bin")
            == b"lidar-data"
        )


def test_remote_archive_multistream_window_selection_is_schema_valid(
    tmp_path: Path,
) -> None:
    centers = ["0000000000", "0000000002"]
    frame_ids = ["0000000000", "0000000001", "0000000002"]
    members = [
        RemoteArchiveSelectedMember(
            role=role,
            frame_id=frame_id,
            member_path=f"archive/{role}/{frame_id}.{extension}",
            crc32="1234abcd",
            compressed_size_bytes=8,
            uncompressed_size_bytes=10,
            local_path=str(tmp_path / role / f"{frame_id}.{extension}"),
            local_sha256="a" * 64,
            local_size_bytes=10,
        )
        for role, extension, selected in (
            ("image", "png", centers),
            ("pointcloud", "bin", frame_ids),
            ("pose", "txt", frame_ids),
        )
        for frame_id in selected
    ]
    metadata = [
        RemoteArchiveSelectedMetadataMember(
            role=role,
            member_path=f"archive/{role}.txt",
            crc32="1234abcd",
            compressed_size_bytes=8,
            uncompressed_size_bytes=10,
            local_path=str(tmp_path / f"{role}.txt"),
            local_sha256="b" * 64,
            local_size_bytes=10,
        )
        for role in (
            "image_timestamps",
            "pointcloud_timestamps",
            "pose_timestamps",
        )
    ]
    artifact = RemoteArchiveSelectionArtifact(
        schema_version=REMOTE_ARCHIVE_SELECTION_SCHEMA_VERSION_V0_3,
        artifact_id="kitti-raw-window",
        dataset_family="KITTI raw",
        dataset_id="kitti_raw_2011_09_30_drive_0018",
        sequence_id="2011_09_30_drive_0018_sync",
        dataset_license_spdx="LicenseRef-KITTI",
        selection_policy=(
            "center_images_with_centered_pointcloud_pose_windows/v0.3"
        ),
        requested_frame_count=3,
        frame_ids=frame_ids,
        center_frame_ids=centers,
        stream_frame_ids={
            "image": centers,
            "pointcloud": frame_ids,
            "pose": frame_ids,
        },
        archives=[
            RemoteArchiveSource(
                role="multi_stream",
                url="https://example.test/raw.zip",
                size_bytes=100,
                etag='"raw"',
                accept_ranges=True,
            )
        ],
        members=members,
        metadata_members=metadata,
        output_root=str(tmp_path),
        provenance=RemoteArchiveSelectionProvenance(
            generator="pytest",
            generator_version="1",
        ),
    )
    output = tmp_path / "raw-selection.yaml"

    artifact.save(output)

    assert validate_file(output, "auto").valid


def test_kitti_raw_window_selection_clips_sequence_boundary() -> None:
    lidar = {
        f"{frame_id:010d}": ZipInfo(f"lidar/{frame_id:010d}.bin")
        for frame_id in range(7)
    }
    poses = {
        f"{frame_id:010d}": ZipInfo(f"pose/{frame_id:010d}.txt")
        for frame_id in range(7)
    }

    selected = raw_downloader._select_window_frame_ids(
        lidar,
        poses,
        ["0000000000", "0000000005"],
        neighbor_radius=1,
        timestamp_limit=7,
    )

    assert selected == [
        "0000000000",
        "0000000001",
        "0000000004",
        "0000000005",
        "0000000006",
    ]


def test_common_frame_selection_uses_archive_intersection_and_real_ids() -> None:
    image_index = {
        f"{frame_id:010d}": ZipInfo(f"sequence/image/{frame_id:010d}.png")
        for frame_id in range(4_613, 4_620)
    }
    lidar_index = {
        f"{frame_id:010d}": ZipInfo(f"sequence/lidar/{frame_id:010d}.bin")
        for frame_id in range(4_610, 4_618)
    }

    selected = downloader._select_common_frame_ids(
        image_index,
        lidar_index,
        image_timestamp_count=19_240,
        velodyne_timestamp_count=19_240,
        selected_count=3,
    )

    assert selected == ["0000004613", "0000004615", "0000004617"]


def test_centered_frame_selection_materializes_inclusive_windows() -> None:
    image_index = {
        f"{frame_id:010d}": ZipInfo(f"sequence/image/{frame_id:010d}.png")
        for frame_id in range(4_610, 4_621)
    }
    lidar_index = {
        f"{frame_id:010d}": ZipInfo(f"sequence/lidar/{frame_id:010d}.bin")
        for frame_id in range(4_611, 4_620)
    }

    selected = downloader._select_centered_common_frame_ids(
        image_index,
        lidar_index,
        image_timestamp_count=19_240,
        velodyne_timestamp_count=19_240,
        center_frame_ids=downloader._parse_center_frame_ids("4613,4617"),
        neighbor_radius=1,
    )

    assert selected == [
        "0000004612",
        "0000004613",
        "0000004614",
        "0000004616",
        "0000004617",
        "0000004618",
    ]

    clipped = downloader._select_centered_common_frame_ids(
        image_index,
        lidar_index,
        image_timestamp_count=19_240,
        velodyne_timestamp_count=19_240,
        center_frame_ids=downloader._parse_center_frame_ids("4611,4619"),
        neighbor_radius=2,
    )
    assert clipped == [
        "0000004611",
        "0000004612",
        "0000004613",
        "0000004617",
        "0000004618",
        "0000004619",
    ]


def test_materialize_member_replaces_stale_partial_atomically(tmp_path: Path) -> None:
    archive_buffer = io.BytesIO()
    member_name = "sequence/image_03/data_rgb/0000004613.png"
    with ZipFile(archive_buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(member_name, b"complete-image-data")
    archive_buffer.seek(0)
    output = tmp_path / "nested" / "0000004613.png"
    partial = output.parent / ".0000004613.png.calibrex-partial"
    output.parent.mkdir(parents=True)
    partial.write_bytes(b"truncated")

    with ZipFile(archive_buffer) as archive:
        artifact_member = downloader._materialize_member(
            archive,
            archive.getinfo(member_name),
            output,
            role="image",
            frame_id="0000004613",
        )

    assert output.read_bytes() == b"complete-image-data"
    assert not partial.exists()
    assert artifact_member.local_size_bytes == len(b"complete-image-data")

    partial.write_bytes(b"stale-after-success")
    with ZipFile(archive_buffer) as archive:
        downloader._materialize_member(
            archive,
            archive.getinfo(member_name),
            output,
            role="image",
            frame_id="0000004613",
        )
    assert not partial.exists()


def test_materialize_member_creates_missing_parent_directory(tmp_path: Path) -> None:
    archive_buffer = io.BytesIO()
    member_name = "sequence/image_02/data/0000000000.png"
    with ZipFile(archive_buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(member_name, b"image")
    archive_buffer.seek(0)
    output = tmp_path / "missing" / "parent" / "0000000000.png"

    with ZipFile(archive_buffer) as archive:
        downloader._materialize_member(
            archive,
            archive.getinfo(member_name),
            output,
            role="image",
            frame_id="0000000000",
        )

    assert output.read_bytes() == b"image"


def test_materialize_member_repairs_stale_range_cache_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_buffer = io.BytesIO()
    member_name = "sequence/image_02/data/0000000000.png"
    with ZipFile(archive_buffer, "w") as archive:
        archive.writestr("padding-before.bin", b"a" * 70_000)
        archive.writestr(member_name, b"verified-image")
        archive.writestr("padding-after.bin", b"b" * 70_000)
    payload = archive_buffer.getvalue()
    block_size = 64 * 1024
    with ZipFile(io.BytesIO(payload)) as archive:
        header_offset = archive.getinfo(member_name).header_offset
    block_index = header_offset // block_size
    block_start = block_index * block_size
    block_end = min(len(payload), block_start + block_size)
    stale_block = bytearray(payload[block_start:block_end])
    local_offset = header_offset - block_start
    stale_block[local_offset : local_offset + 4] = b"FAIL"
    stale_cache = tmp_path / "stale-cache"
    stale_cache.mkdir()
    (stale_cache / f"part-{block_index:04d}.bin").write_bytes(stale_block)

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        del timeout
        if request.get_method() == "HEAD":
            return _FakeResponse(
                b"",
                status=200,
                headers={
                    "Content-Length": str(len(payload)),
                    "Accept-Ranges": "bytes",
                    "ETag": '"repair-fixture"',
                },
            )
        raw_range = request.get_header("Range")
        assert raw_range is not None
        start_text, end_text = raw_range.removeprefix("bytes=").split("-", 1)
        start = int(start_text)
        end = int(end_text)
        return _FakeResponse(
            payload[start : end + 1],
            status=206,
            headers={"Content-Range": f"bytes {start}-{end}/{len(payload)}"},
        )

    monkeypatch.setattr(downloader, "urlopen", fake_urlopen)
    repaired_cache = tmp_path / "repaired-cache"
    reader = downloader.HTTPRangeReader(
        "https://example.test/archive.zip",
        block_size_bytes=block_size,
        read_cache_directories=[stale_cache],
        write_cache_directory=repaired_cache,
    )
    output = tmp_path / "0000000000.png"

    with ZipFile(reader) as archive:
        downloader._materialize_member(
            archive,
            archive.getinfo(member_name),
            output,
            role="image",
            frame_id="0000000000",
        )

    assert output.read_bytes() == b"verified-image"
    repaired = repaired_cache / f"part-{block_index:04d}.bin"
    assert repaired.read_bytes() == payload[block_start:block_end]


class _FakeResponse:
    def __init__(
        self,
        payload: bytes,
        *,
        status: int,
        headers: dict[str, str],
    ) -> None:
        self.payload = payload
        self.status = status
        self.headers = headers

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload

    def getcode(self) -> int:
        return self.status
