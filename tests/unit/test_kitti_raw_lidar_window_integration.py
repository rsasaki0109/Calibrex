from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from tools.run_superglue_correspondence_provider import (
    _verify_lidar_integration_binding,
)

from calibrex.core.provenance import sha256_path
from calibrex.core.validation import validate_file
from calibrex.data.kitti_raw_lidar_window_integration import (
    integrate_kitti_raw_lidar_windows,
    load_kitti_raw_lidar_window_integration,
    verify_kitti_raw_lidar_window_integration_files,
)
from calibrex.data.remote_archive_selection import (
    REMOTE_ARCHIVE_SELECTION_SCHEMA_VERSION_V0_3,
    RemoteArchiveSelectedMember,
    RemoteArchiveSelectedMetadataMember,
    RemoteArchiveSelectionArtifact,
    RemoteArchiveSelectionProvenance,
    RemoteArchiveSource,
)


def _required_digest(path: Path) -> str:
    digest = sha256_path(path)
    assert digest is not None
    return digest


def _selected_member(
    role: str,
    frame_id: str,
    path: Path,
) -> RemoteArchiveSelectedMember:
    return RemoteArchiveSelectedMember(
        role=role,  # type: ignore[arg-type]
        frame_id=frame_id,
        member_path=f"archive/{role}/{path.name}",
        crc32="00000000",
        compressed_size_bytes=path.stat().st_size,
        uncompressed_size_bytes=path.stat().st_size,
        local_path=str(path.resolve()),
        local_sha256=_required_digest(path),
        local_size_bytes=path.stat().st_size,
    )


def _write_fixture(root: Path) -> tuple[Path, Path]:
    sequence = root / "2011_09_30" / "2011_09_30_drive_0018_sync"
    image_directory = sequence / "image_02" / "data"
    lidar_directory = sequence / "velodyne_points" / "data"
    pose_directory = sequence / "oxts" / "data"
    image_directory.mkdir(parents=True)
    lidar_directory.mkdir(parents=True)
    pose_directory.mkdir(parents=True)
    centers = ["0000000010", "0000000013"]
    pointcloud_ids = [f"{value:010d}" for value in range(9, 15)]
    pose_ids = [f"{value:010d}" for value in range(8, 15)]
    members: list[RemoteArchiveSelectedMember] = []
    for frame_id in centers:
        image = image_directory / f"{frame_id}.png"
        image.write_bytes(f"image-{frame_id}".encode())
        members.append(_selected_member("image", frame_id, image))
    for frame_id in pointcloud_ids:
        scan = lidar_directory / f"{frame_id}.bin"
        np.asarray([[2.0, 0.0, 0.0, float(int(frame_id))]], dtype="<f4").tofile(scan)
        members.append(_selected_member("pointcloud", frame_id, scan))
    latitude_deg = 49.0
    longitude_origin_deg = 8.0
    metres_per_longitude_degree = (
        6378137.0 * math.cos(math.radians(latitude_deg)) * math.pi / 180.0
    )
    for frame_id in pose_ids:
        pose = pose_directory / f"{frame_id}.txt"
        x_metres = int(frame_id) - 8
        longitude_deg = longitude_origin_deg + x_metres / metres_per_longitude_degree
        values = [
            latitude_deg,
            longitude_deg,
            100.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        pose.write_text(" ".join(str(value) for value in values), encoding="utf-8")
        members.append(_selected_member("pose", frame_id, pose))
    metadata: list[RemoteArchiveSelectedMetadataMember] = []
    for role, path in (
        ("image_timestamps", sequence / "image_02" / "timestamps.txt"),
        ("pointcloud_timestamps", sequence / "velodyne_points" / "timestamps.txt"),
        ("pose_timestamps", sequence / "oxts" / "timestamps.txt"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("2011-09-30 00:00:00.000000000\n", encoding="utf-8")
        metadata.append(
            RemoteArchiveSelectedMetadataMember(
                role=role,  # type: ignore[arg-type]
                member_path=f"archive/{role}.txt",
                crc32="00000000",
                compressed_size_bytes=path.stat().st_size,
                uncompressed_size_bytes=path.stat().st_size,
                local_path=str(path.resolve()),
                local_sha256=_required_digest(path),
                local_size_bytes=path.stat().st_size,
            )
        )
    all_ids = sorted(set(centers) | set(pointcloud_ids) | set(pose_ids), key=int)
    selection = RemoteArchiveSelectionArtifact(
        schema_version=REMOTE_ARCHIVE_SELECTION_SCHEMA_VERSION_V0_3,
        artifact_id="kitti-raw-integration-fixture",
        dataset_family="KITTI raw",
        dataset_id="kitti_raw_2011_09_30_drive_0018",
        sequence_id="2011_09_30_drive_0018_sync",
        dataset_license_spdx="LicenseRef-KITTI",
        selection_policy="center_images_with_centered_pointcloud_pose_windows/v0.3",
        requested_frame_count=len(all_ids),
        frame_ids=all_ids,
        center_frame_ids=centers,
        stream_frame_ids={
            "image": centers,
            "pointcloud": pointcloud_ids,
            "pose": pose_ids,
        },
        archives=[
            RemoteArchiveSource(
                role="multi_stream",
                url="https://example.test/raw.zip",
                size_bytes=1,
                accept_ranges=True,
            )
        ],
        members=members,
        metadata_members=metadata,
        output_root=str(sequence.resolve()),
        provenance=RemoteArchiveSelectionProvenance(
            generator="pytest",
            generator_version="fixture",
        ),
    )
    selection_path = root / "selection.yaml"
    selection.save(selection_path)
    calibration = root / "calib_imu_to_velo.txt"
    calibration.write_text(
        "R: 1 0 0 0 1 0 0 0 1\nT: 0 0 0\n",
        encoding="utf-8",
    )
    return selection_path, calibration


def test_integrates_raw_window_with_motion_pose_neighbors(tmp_path: Path) -> None:
    selection, calibration = _write_fixture(tmp_path)
    output_directory = tmp_path / "integrated"

    artifact = integrate_kitti_raw_lidar_windows(
        selection,
        calibration,
        [10, 13],
        neighbor_radius_frames=1,
        output_directory=output_directory,
        minimum_range_m=1.0,
        voxel_resolution_m=0.0,
        command=("pytest",),
    )
    manifest = tmp_path / "integration.yaml"
    artifact.save(manifest)

    first = np.fromfile(output_directory / "0000000010.bin", dtype="<f4").reshape(-1, 4)
    assert first[:, 0] == pytest.approx([1.0, 2.0, 3.0], abs=2.0e-6)
    assert [pose.frame_id for pose in artifact.resolved_poses] == [
        f"{value:010d}" for value in range(8, 15)
    ]
    assert artifact.parameters.pose_origin_frame_id == "0000000008"
    assert artifact.windows[0].source_point_count == 3
    assert artifact.parameters.voxel_reducer == "none"
    assert verify_kitti_raw_lidar_window_integration_files(artifact) == []
    assert validate_file(manifest, "auto").valid
    assert (
        load_kitti_raw_lidar_window_integration(manifest).output_set_sha256
        == artifact.output_set_sha256
    )
    _verify_lidar_integration_binding(
        manifest,
        lidar_directory=output_directory,
        frame_ids=["0000000010", "0000000013"],
        sequence_id="2011_09_30_drive_0018_sync",
        dataset_family="kitti_raw",
    )


def test_raw_integration_verifier_reports_tampered_timestamp(tmp_path: Path) -> None:
    selection, calibration = _write_fixture(tmp_path)
    artifact = integrate_kitti_raw_lidar_windows(
        selection,
        calibration,
        [10, 13],
        neighbor_radius_frames=1,
        output_directory=tmp_path / "integrated",
        voxel_resolution_m=0.0,
    )
    timestamp = Path(artifact.timestamp_files[0].path)
    timestamp.write_text("tampered", encoding="utf-8")

    issues = verify_kitti_raw_lidar_window_integration_files(artifact)

    assert f"size_mismatch:{timestamp}" in issues
    assert f"digest_mismatch:{timestamp}" in issues
