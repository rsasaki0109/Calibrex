from __future__ import annotations

from pathlib import Path

import numpy as np

from calibrex.cli.main import main
from calibrex.core.camera_lidar_correspondence_export import (
    CameraLidarCorrespondenceExportFrame,
    CameraLidarCorrespondenceExportManifest,
    build_probabilistic_correspondence_artifact,
)
from calibrex.core.geometry import SE3
from calibrex.core.probabilistic_correspondence import (
    CorrespondenceProviderIdentity,
    ProbabilisticCorrespondenceFrame,
    ProbabilisticCorrespondenceProvenance,
    ProbabilisticImageCorrespondence,
)
from calibrex.core.provenance import sha256_path
from calibrex.core.validation import validate_file
from calibrex.data.depth import DepthCameraIntrinsics, DepthFileReference
from calibrex.solvers.probabilistic_camera_lidar_refiner import (
    evaluate_probabilistic_camera_lidar_pose,
)


def test_kitti_family_npz_export_becomes_schema_valid_correspondence(
    tmp_path: Path,
) -> None:
    export_paths = []
    for index in range(2):
        export_path = tmp_path / f"frame-{index}.npz"
        np.savez(
            export_path,
            point_lidar_m=np.asarray(
                [
                    [-1.0, -0.5, 8.0],
                    [1.0, -0.5, 8.0],
                    [-1.0, 0.5, 8.0],
                    [1.0, 0.5, 8.0],
                ],
                dtype=float,
            ),
            image_mean_px=np.asarray(
                [[560.0, 322.5], [720.0, 322.5], [560.0, 397.5], [720.0, 397.5]],
                dtype=float,
            ),
            image_covariance_px2=np.tile(np.eye(2, dtype=float).reshape(1, 4), (4, 1)),
            outlier_probability=np.full(4, 0.05),
            reliability=np.full(4, 0.95),
            correspondence_id=np.asarray(["a", "b", "c", "d"]),
        )
        export_paths.append(export_path)

    intrinsics = DepthCameraIntrinsics(
        width=1280,
        height=720,
        fx=640.0,
        fy=640.0,
        cx=640.0,
        cy=360.0,
    )
    frames = []
    for index, export_path in enumerate(export_paths):
        digest = sha256_path(export_path)
        assert digest is not None
        frames.append(
            CameraLidarCorrespondenceExportFrame(
                frame_id=f"frame-{index}",
                capture_time_ns=index,
                camera_frame="camera",
                lidar_frame="lidar",
                intrinsics=intrinsics,
                export=DepthFileReference(
                    path=export_path.name,
                    sha256=digest,
                    size_bytes=export_path.stat().st_size,
                    encoding="npz_float64_correspondence_export",
                ),
            )
        )
    manifest = CameraLidarCorrespondenceExportManifest(
        artifact_id="kitti-raw-correspondence",
        dataset_id="kitti-raw-0018",
        dataset_family="kitti_raw",
        sequence_id="2011_09_30_drive_0018_sync",
        split_id="evaluation",
        dataset_license_spdx="KITTI-terms-of-use",
        provider=CorrespondenceProviderIdentity(
            provider="fixture-provider",
            model="fixture-correspondence",
            version="1",
            source_repository="https://example.invalid/provider",
            source_commit="a" * 40,
            license_spdx="MIT",
            checkpoint_redistribution="not redistributed",
        ),
        frames=frames,
        provenance=ProbabilisticCorrespondenceProvenance(
            generator="pytest",
            generator_version="1",
            input_sha256={"provider_input": "b" * 64},
        ),
    )
    manifest_path = tmp_path / "export.yaml"
    output_path = tmp_path / "correspondence.yaml"
    manifest.save(manifest_path)

    artifact = build_probabilistic_correspondence_artifact(
        manifest_path,
        command=["pytest", "adapter"],
    )
    artifact.save(output_path)

    assert artifact.dataset_id == "kitti-raw-0018"
    assert artifact.dataset_family == "kitti_raw"
    assert artifact.sequence_id == "2011_09_30_drive_0018_sync"
    assert len(artifact.frames) == 2
    assert sum(len(frame.correspondences) for frame in artifact.frames) == 8
    assert artifact.provenance.input_sha256["export_manifest"] == sha256_path(
        manifest_path
    )
    assert validate_file(
        manifest_path, "camera-lidar-correspondence-export"
    ).valid
    assert validate_file(output_path, "probabilistic-correspondence").valid

    cli_output = tmp_path / "cli-correspondence.yaml"
    assert (
        main(
            [
                "camera-lidar",
                "build-probabilistic-correspondence",
                str(manifest_path),
                "--output",
                str(cli_output),
                "--json",
            ]
        )
        == 0
    )
    assert validate_file(cli_output, "probabilistic-correspondence").valid


def test_kitti360_mei_projection_is_supported_by_pose_evaluation() -> None:
    intrinsics = DepthCameraIntrinsics(
        width=1400,
        height=1400,
        fx=700.0,
        fy=700.0,
        cx=700.0,
        cy=700.0,
        projection="mei",
        xi=1.0,
        distortion_model="radial-tangential",
        distortion=[0.0, 0.0, 0.0, 0.0],
    )
    points = [
        (-1.0, -0.5, 8.0),
        (1.0, -0.5, 8.0),
        (-1.0, 0.5, 8.0),
        (1.0, 0.5, 8.0),
    ]
    correspondences = []
    for index, (x, y, z) in enumerate(points):
        norm = (x * x + y * y + z * z) ** 0.5
        denominator = z + norm
        correspondences.append(
            ProbabilisticImageCorrespondence(
                correspondence_id=str(index),
                point_lidar_m=[x, y, z],
                image_mean_px=[
                    intrinsics.fx * x / denominator + intrinsics.cx,
                    intrinsics.fy * y / denominator + intrinsics.cy,
                ],
                image_covariance_px2=[1.0, 0.0, 0.0, 1.0],
                outlier_probability=0.0,
                reliability=1.0,
            )
        )
    frame = ProbabilisticCorrespondenceFrame(
        frame_id="kitti360-mei-frame",
        capture_time_ns=0,
        camera_frame="camera",
        lidar_frame="lidar",
        intrinsics=intrinsics,
        correspondences=correspondences,
    )
    evaluation = evaluate_probabilistic_camera_lidar_pose(
        [frame],
        transform_camera_lidar=SE3.identity(),
    )
    assert evaluation.valid_correspondence_count == 4
    assert evaluation.weighted_reprojection_rmse_px == 0.0
