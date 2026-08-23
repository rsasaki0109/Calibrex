from __future__ import annotations

import math
from pathlib import Path

import pytest

from calibrex.cli.main import main
from calibrex.core.geometry import SE3
from calibrex.core.probabilistic_correspondence import (
    CorrespondenceProviderIdentity,
    ProbabilisticCorrespondenceArtifact,
    ProbabilisticCorrespondenceFrame,
    ProbabilisticCorrespondenceProvenance,
    ProbabilisticImageCorrespondence,
)
from calibrex.core.validation import validate_file
from calibrex.data.depth import DepthCameraIntrinsics
from calibrex.solvers.opencv_probabilistic_pnp_adapter import (
    OpenCvProbabilisticPnpAdapter,
    OpenCvProbabilisticPnpOptions,
)

pytest.importorskip("cv2")


def test_opencv_probabilistic_pnp_recovers_synthetic_pose(
    tmp_path: Path,
) -> None:
    artifact = _artifact()
    path = tmp_path / "correspondence.yaml"
    artifact.save(path)

    result = OpenCvProbabilisticPnpAdapter().solve_artifact(path, "000000")

    assert result.status == "converged"
    assert result.transform_camera_lidar is not None
    assert result.ransac_inlier_count == 8
    assert result.probabilistic_inlier_count == 8
    assert result.weighted_reprojection_rmse_px is not None
    assert result.weighted_reprojection_rmse_px < 1.0e-5
    assert result.ransac_inlier_reprojection_rmse_px is not None
    assert result.ransac_inlier_reprojection_rmse_px < 1.0e-5
    assert result.artifact_sha256
    assert result.as_dict()["tool"]["license_spdx"] == "Apache-2.0"
    output = tmp_path / "pnp-result.yaml"
    result.to_artifact(
        result_id="synthetic-pnp-result",
        options=OpenCvProbabilisticPnpOptions(),
        command=["calibrex", "camera-lidar", "refine-probabilistic-pnp"],
    ).save(output)
    payload = output.read_text(encoding="utf-8")
    assert "slac.probabilistic_pnp_result/v0.2" in payload
    assert "correspondence_artifact_sha256" in payload

    seeded = OpenCvProbabilisticPnpAdapter().solve_artifact(
        path,
        "000000",
        initial_transform_camera_lidar=SE3.identity(),
        initialization_artifact_sha256="c" * 64,
    )
    assert seeded.status == "converged"
    seeded_artifact = seeded.to_artifact(
        result_id="seeded",
        options=OpenCvProbabilisticPnpOptions(),
    )
    assert seeded_artifact.initial_transform_camera_lidar is not None
    assert (
        seeded_artifact.provenance.initialization_artifact_sha256
        == "c" * 64
    )


def test_probabilistic_pnp_cli_writes_valid_result(tmp_path: Path) -> None:
    path = tmp_path / "correspondence.yaml"
    output = tmp_path / "pnp-result.yaml"
    _artifact().save(path)

    exit_code = main(
        [
            "camera-lidar",
            "refine-probabilistic-pnp",
            str(path),
            "000000",
            "--output",
            str(output),
            "--json",
        ]
    )

    assert exit_code == 0
    report = validate_file(output, "probabilistic-pnp-result")
    assert report.valid


def test_opencv_probabilistic_pnp_recovers_synthetic_mei_pose(
    tmp_path: Path,
) -> None:
    truth = SE3(
        translation_m=(0.2, -0.1, 0.3),
        rotation_quat_xyzw=(
            0.0,
            math.sin(math.radians(4.0) / 2.0),
            0.0,
            math.cos(math.radians(4.0) / 2.0),
        ),
    )
    artifact = _mei_artifact(truth)
    path = tmp_path / "mei-correspondence.yaml"
    artifact.save(path)

    result = OpenCvProbabilisticPnpAdapter().solve_artifact(path, "000000")

    assert result.status == "converged"
    assert result.transform_camera_lidar is not None
    assert result.selected_correspondence_count == 12
    assert result.ransac_inlier_count == 12
    assert result.probabilistic_inlier_count == 12
    assert result.weighted_reprojection_rmse_px is not None
    assert result.weighted_reprojection_rmse_px < 1.0e-3
    assert result.transform_camera_lidar.translation_m == pytest.approx(
        truth.translation_m,
        abs=1.0e-5,
    )
    quaternion_dot = abs(
        sum(
            left * right
            for left, right in zip(
                result.transform_camera_lidar.rotation_quat_xyzw,
                truth.rotation_quat_xyzw,
                strict=True,
            )
        )
    )
    assert math.degrees(2.0 * math.acos(min(1.0, quaternion_dot))) < 1.0e-4
    assert "declared MEI model" in result.reason
    output = result.to_artifact(
        result_id="synthetic-mei-pnp-result",
        options=OpenCvProbabilisticPnpOptions(),
    )
    assert output.method == "opencv_probabilistic_pnp_ransac/v0.2"
    assert output.provenance.generator_version == "0.3"


def test_opencv_probabilistic_aggregate_pnp_records_selected_frames(
    tmp_path: Path,
) -> None:
    artifact = _artifact()
    first = artifact.frames[0]
    second = first.model_copy(
        update={"frame_id": "000001", "capture_time_ns": 1}
    )
    artifact = artifact.model_copy(update={"frames": [first, second]})
    path = tmp_path / "aggregate-correspondence.yaml"
    artifact.save(path)
    options = OpenCvProbabilisticPnpOptions(
        minimum_confidence=0.25,
        minimum_frame_correspondences=4,
        minimum_frames=2,
    )

    result = OpenCvProbabilisticPnpAdapter().solve_artifact_aggregate(
        path,
        options,
    )

    assert result.status == "converged"
    assert result.transform_camera_lidar is not None
    assert result.source_frame_ids == ("000000", "000001")
    assert result.selected_correspondence_count == 16
    assert result.ransac_inlier_count == 16
    assert result.method == "opencv_probabilistic_aggregate_pnp_ransac/v0.3"
    output = tmp_path / "aggregate-pnp.yaml"
    result.to_artifact(
        result_id="synthetic-aggregate-pnp",
        options=options,
    ).save(output)
    assert validate_file(output, "auto").schema_version.endswith("/v0.2")


def test_probabilistic_aggregate_pnp_cli_writes_valid_result(
    tmp_path: Path,
) -> None:
    artifact = _artifact()
    first = artifact.frames[0]
    artifact = artifact.model_copy(
        update={
            "frames": [
                first,
                first.model_copy(
                    update={"frame_id": "000001", "capture_time_ns": 1}
                ),
            ]
        }
    )
    path = tmp_path / "aggregate-correspondence.yaml"
    output = tmp_path / "aggregate-pnp.yaml"
    artifact.save(path)

    exit_code = main(
        [
            "camera-lidar",
            "refine-probabilistic-pnp-aggregate",
            str(path),
            "--output",
            str(output),
            "--minimum-frame-correspondences",
            "4",
            "--minimum-frames",
            "2",
            "--json",
        ]
    )

    assert exit_code == 0
    report = validate_file(output, "probabilistic-pnp-result")
    assert report.valid
    payload = output.read_text(encoding="utf-8")
    assert "000000" in payload
    assert "000001" in payload


def _artifact() -> ProbabilisticCorrespondenceArtifact:
    points = [
        (-1.0, -1.0, 4.0),
        (1.0, -1.0, 4.0),
        (-1.0, 1.0, 4.0),
        (1.0, 1.0, 4.0),
        (-2.0, -1.0, 8.0),
        (2.0, -1.0, 8.0),
        (-2.0, 1.0, 8.0),
        (2.0, 1.0, 8.0),
    ]
    correspondences = []
    for index, (x, y, z) in enumerate(points):
        x_camera = x + 0.2
        y_camera = y - 0.1
        z_camera = z + 0.3
        correspondences.append(
            ProbabilisticImageCorrespondence(
                correspondence_id=str(index),
                point_lidar_m=[x, y, z],
                image_mean_px=[
                    500.0 * x_camera / z_camera + 320.0,
                    500.0 * y_camera / z_camera + 240.0,
                ],
                image_covariance_px2=[0.25, 0.0, 0.0, 0.25],
                outlier_probability=0.01,
                reliability=0.99,
            )
        )
    return ProbabilisticCorrespondenceArtifact(
        artifact_id="synthetic-pnp",
        dataset_id="synthetic",
        split_id="test",
        provider=CorrespondenceProviderIdentity(
            provider="fixture",
            model="exact",
            version="1",
            source_repository="https://example.test/provider",
            source_commit="a" * 40,
            license_spdx="Apache-2.0",
            checkpoint_redistribution="not applicable",
        ),
        frames=[
            ProbabilisticCorrespondenceFrame(
                frame_id="000000",
                capture_time_ns=0,
                camera_frame="camera",
                lidar_frame="lidar",
                intrinsics=DepthCameraIntrinsics(
                    width=640,
                    height=480,
                    fx=500.0,
                    fy=500.0,
                    cx=320.0,
                    cy=240.0,
                ),
                correspondences=correspondences,
            )
        ],
        provenance=ProbabilisticCorrespondenceProvenance(
            generator="pytest",
            generator_version="1",
            input_sha256={"fixture": "b" * 64},
        ),
    )


def _mei_artifact(truth: SE3) -> ProbabilisticCorrespondenceArtifact:
    intrinsics = DepthCameraIntrinsics(
        width=1400,
        height=1400,
        fx=1485.4388981875156,
        fy=1484.9477411748708,
        cx=698.8831678403096,
        cy=698.1454188772306,
        projection="mei",
        xi=2.553513913248276,
        distortion_model="radial-tangential",
        distortion=[
            0.049370396274089505,
            4.506845547864531,
            0.0013477698472982495,
            -0.0007034048261505528,
        ],
    )
    points = [
        (-1.5, -1.0, 4.0),
        (1.5, -1.0, 4.0),
        (-1.5, 1.0, 4.0),
        (1.5, 1.0, 4.0),
        (-2.0, -0.5, 6.0),
        (2.0, -0.5, 6.0),
        (-2.0, 0.5, 6.0),
        (2.0, 0.5, 6.0),
        (-1.0, -1.5, 8.0),
        (1.0, -1.5, 8.0),
        (-1.0, 1.5, 8.0),
        (1.0, 1.5, 8.0),
    ]
    correspondences = [
        ProbabilisticImageCorrespondence(
            correspondence_id=str(index),
            point_lidar_m=list(point),
            image_mean_px=list(_project_mei(truth.transform_point(point), intrinsics)),
            image_covariance_px2=[0.25, 0.0, 0.0, 0.25],
            outlier_probability=0.01,
            reliability=0.99,
        )
        for index, point in enumerate(points)
    ]
    return ProbabilisticCorrespondenceArtifact(
        artifact_id="synthetic-mei-pnp",
        dataset_id="synthetic",
        split_id="test",
        provider=CorrespondenceProviderIdentity(
            provider="fixture",
            model="exact-mei",
            version="1",
            source_repository="https://example.test/provider",
            source_commit="a" * 40,
            license_spdx="Apache-2.0",
            checkpoint_redistribution="not applicable",
        ),
        frames=[
            ProbabilisticCorrespondenceFrame(
                frame_id="000000",
                capture_time_ns=0,
                camera_frame="camera",
                lidar_frame="lidar",
                intrinsics=intrinsics,
                correspondences=correspondences,
            )
        ],
        provenance=ProbabilisticCorrespondenceProvenance(
            generator="pytest",
            generator_version="1",
            input_sha256={"fixture": "b" * 64},
        ),
    )


def _project_mei(
    point_camera: tuple[float, float, float],
    intrinsics: DepthCameraIntrinsics,
) -> tuple[float, float]:
    x_point, y_point, z_point = point_camera
    assert intrinsics.xi is not None
    norm = math.sqrt(
        x_point * x_point + y_point * y_point + z_point * z_point
    )
    normalized_x = x_point / (z_point + intrinsics.xi * norm)
    normalized_y = y_point / (z_point + intrinsics.xi * norm)
    k1, k2, p1, p2 = intrinsics.distortion
    radius_squared = normalized_x * normalized_x + normalized_y * normalized_y
    radial = 1.0 + k1 * radius_squared + k2 * radius_squared**2
    distorted_x = (
        normalized_x * radial
        + 2.0 * p1 * normalized_x * normalized_y
        + p2 * (radius_squared + 2.0 * normalized_x * normalized_x)
    )
    distorted_y = (
        normalized_y * radial
        + p1 * (radius_squared + 2.0 * normalized_y * normalized_y)
        + 2.0 * p2 * normalized_x * normalized_y
    )
    return (
        intrinsics.fx * distorted_x + intrinsics.cx,
        intrinsics.fy * distorted_y + intrinsics.cy,
    )
