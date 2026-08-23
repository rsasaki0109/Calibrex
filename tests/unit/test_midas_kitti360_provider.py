from __future__ import annotations

from pathlib import Path

import pytest
import tools.run_midas_kitti360_provider as provider

from calibrex.data.remote_archive_selection import (
    RemoteArchiveSelectedMember,
    RemoteArchiveSelectionArtifact,
    RemoteArchiveSelectionProvenance,
    RemoteArchiveSource,
)


def test_midas_selection_manifest_locks_real_frame_ids(tmp_path: Path) -> None:
    sequence = tmp_path / "2013_05_28_drive_0002_sync"
    frame_ids = ["0000004613", "0000018997"]
    artifact = _selection_artifact(sequence, frame_ids)
    manifest = tmp_path / "selection.yaml"
    artifact.save(manifest)

    selected, loaded = provider._frame_ids_from_selection(
        manifest,
        sequence=sequence,
        camera_stream="image_03",
        frame_count=2,
        timestamp_count=19_240,
    )

    assert selected == frame_ids
    assert loaded.selection_policy == "endpoint_inclusive_same_numeric_frame_id/v0.1"


def test_midas_selection_manifest_rejects_wrong_camera_stream(tmp_path: Path) -> None:
    sequence = tmp_path / "2013_05_28_drive_0002_sync"
    artifact = _selection_artifact(sequence, ["0000004613", "0000018997"])
    manifest = tmp_path / "selection.yaml"
    artifact.save(manifest)

    with pytest.raises(ValueError, match="camera stream"):
        provider._frame_ids_from_selection(
            manifest,
            sequence=sequence,
            camera_stream="image_02",
            frame_count=2,
            timestamp_count=19_240,
        )


def _selection_artifact(
    sequence: Path,
    frame_ids: list[str],
) -> RemoteArchiveSelectionArtifact:
    members = [
        RemoteArchiveSelectedMember(
            role=role,
            frame_id=frame_id,
            member_path=f"archive/{relative_path}",
            crc32="1234abcd",
            compressed_size_bytes=8,
            uncompressed_size_bytes=10,
            local_path=str(sequence / relative_path),
            local_sha256="a" * 64,
            local_size_bytes=10,
        )
        for frame_id in frame_ids
        for role, relative_path in (
            ("image", f"image_03/data_rgb/{frame_id}.png"),
            ("pointcloud", f"velodyne_points/data/{frame_id}.bin"),
        )
    ]
    return RemoteArchiveSelectionArtifact(
        artifact_id="kitti360-drive-0002-dev",
        dataset_family="KITTI-360",
        dataset_id="kitti360_2013_05_28_drive_0002",
        sequence_id=sequence.name,
        dataset_license_spdx="CC-BY-NC-SA-3.0",
        requested_frame_count=len(frame_ids),
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
        output_root=str(sequence),
        provenance=RemoteArchiveSelectionProvenance(
            generator="pytest",
            generator_version="1",
        ),
    )
