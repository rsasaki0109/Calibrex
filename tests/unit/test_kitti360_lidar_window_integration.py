from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from calibrex.core.provenance import sha256_path
from calibrex.core.validation import validate_file
from calibrex.data.kitti360_lidar_window_integration import (
    _resolve_pose_frames,
    integrate_kitti360_lidar_windows,
    load_kitti360_lidar_window_integration,
    verify_kitti360_lidar_window_integration_files,
)
from calibrex.data.remote_archive_selection import (
    RemoteArchiveSelectedMember,
    RemoteArchiveSelectionArtifact,
    RemoteArchiveSelectionProvenance,
    RemoteArchiveSource,
)


def _matrix_row(frame_id: int, translation_x: float) -> str:
    matrix = np.eye(4, dtype=float)
    matrix[0, 3] = translation_x
    values = " ".join(str(value) for value in matrix[:3, :].reshape(-1))
    return f"{frame_id} {values}\n"


def _required_digest(path: Path) -> str:
    digest = sha256_path(path)
    assert digest is not None
    return digest


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    sequence = root / "sequence"
    image_directory = sequence / "image_03" / "data_rgb"
    lidar_directory = sequence / "velodyne_points" / "data"
    image_directory.mkdir(parents=True)
    lidar_directory.mkdir(parents=True)
    frame_ids = ["0000000009", "0000000010", "0000000011"]
    members: list[RemoteArchiveSelectedMember] = []
    for frame_id in frame_ids:
        image_path = image_directory / f"{frame_id}.png"
        lidar_path = lidar_directory / f"{frame_id}.bin"
        image_path.write_bytes(f"image-{frame_id}".encode())
        np.asarray([[2.0, 0.0, 0.0, float(int(frame_id))]], dtype=np.float32).tofile(lidar_path)
        for role, path in (("image", image_path), ("pointcloud", lidar_path)):
            members.append(
                RemoteArchiveSelectedMember(
                    role=role,
                    frame_id=frame_id,
                    member_path=f"archive/{path.name}",
                    crc32="00000000",
                    compressed_size_bytes=path.stat().st_size,
                    uncompressed_size_bytes=path.stat().st_size,
                    local_path=str(path.resolve()),
                    local_sha256=_required_digest(path),
                    local_size_bytes=path.stat().st_size,
                )
            )
    selection = RemoteArchiveSelectionArtifact(
        artifact_id="kitti360-window-fixture",
        dataset_family="KITTI-360",
        dataset_id="kitti360_fixture",
        sequence_id="2013_05_28_drive_0002_sync",
        dataset_license_spdx="CC-BY-NC-SA-3.0",
        selection_policy="centered_same_numeric_frame_windows/v0.2",
        requested_frame_count=len(frame_ids),
        frame_ids=frame_ids,
        archives=[
            RemoteArchiveSource(
                role="image",
                url="https://example.test/image.zip",
                size_bytes=1,
                accept_ranges=True,
            ),
            RemoteArchiveSource(
                role="pointcloud",
                url="https://example.test/lidar.zip",
                size_bytes=1,
                accept_ranges=True,
            ),
        ],
        members=members,
        output_root=str(sequence.resolve()),
        provenance=RemoteArchiveSelectionProvenance(
            generator="pytest",
            generator_version="fixture",
        ),
    )
    selection_path = root / "selection.yaml"
    selection.save(selection_path)

    poses_path = root / "poses.txt"
    poses_path.write_text(_matrix_row(8, 8.0) + _matrix_row(12, 12.0), encoding="utf-8")
    calibration = root / "calibration"
    calibration.mkdir()
    identity = np.eye(4, dtype=float)
    values = " ".join(str(value) for value in identity[:3, :].reshape(-1))
    (calibration / "calib_cam_to_velo.txt").write_text(values, encoding="utf-8")
    (calibration / "calib_cam_to_pose.txt").write_text(f"image_00: {values}\n", encoding="utf-8")
    return selection_path, poses_path, calibration


def test_integrates_motion_compensated_window_with_interpolated_poses(
    tmp_path: Path,
) -> None:
    selection, poses, calibration = _write_fixture(tmp_path)
    output_directory = tmp_path / "integrated"

    artifact = integrate_kitti360_lidar_windows(
        selection,
        poses,
        calibration,
        [10],
        neighbor_radius_frames=1,
        output_directory=output_directory,
        minimum_range_m=1.0,
        voxel_resolution_m=0.0,
        command=("pytest",),
    )
    manifest_path = tmp_path / "integration.yaml"
    artifact.save(manifest_path)

    points = np.fromfile(output_directory / "0000000010.bin", dtype="<f4").reshape(-1, 4)
    assert points[:, 0] == pytest.approx([1.0, 2.0, 3.0])
    assert points[:, 1:3] == pytest.approx(np.zeros((3, 2)))
    assert points[:, 3] == pytest.approx([9.0, 10.0, 11.0])
    assert artifact.windows[0].source_point_count == 3
    assert artifact.windows[0].output.point_count == 3
    assert artifact.parameters.voxel_reducer == "none"
    assert [pose.method for pose in artifact.resolved_poses] == [
        "official",
        "bracketed_se3_interpolation",
        "bracketed_se3_interpolation",
        "bracketed_se3_interpolation",
    ]
    assert verify_kitti360_lidar_window_integration_files(artifact) == []
    assert validate_file(manifest_path, "auto").valid
    assert (
        load_kitti360_lidar_window_integration(manifest_path).output_set_sha256
        == artifact.output_set_sha256
    )


def test_verifier_reports_tampered_integrated_output(tmp_path: Path) -> None:
    selection, poses, calibration = _write_fixture(tmp_path)
    artifact = integrate_kitti360_lidar_windows(
        selection,
        poses,
        calibration,
        [10],
        neighbor_radius_frames=1,
        output_directory=tmp_path / "integrated",
        minimum_range_m=0.0,
        voxel_resolution_m=0.002,
    )
    output = Path(artifact.windows[0].output.path)
    output.write_bytes(output.read_bytes() + b"tamper")

    issues = verify_kitti360_lidar_window_integration_files(artifact)

    assert f"size_mismatch:{output}" in issues
    assert f"digest_mismatch:{output}" in issues


def test_pose_resolution_allows_only_bounded_endpoint_extrapolation() -> None:
    first = np.eye(4, dtype=np.float64)
    first[0, 3] = 1.0
    second = np.eye(4, dtype=np.float64)
    second[0, 3] = 9.0

    poses, records = _resolve_pose_frames(
        {1: first, 9: second},
        {0, 1},
        maximum_extrapolation_frames=1,
    )

    assert poses[0][0, 3] == pytest.approx(0.0)
    assert records[0].method == "bounded_endpoint_se3_extrapolation"
    assert records[0].left_official_frame_id == "0000000001"
    assert records[0].right_official_frame_id == "0000000009"
    assert records[0].interpolation_alpha == pytest.approx(-0.125)
    with pytest.raises(ValueError, match="allowed 0"):
        _resolve_pose_frames(
            {1: first, 9: second},
            {0},
            maximum_extrapolation_frames=0,
        )
