from __future__ import annotations

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
    assert result.artifact_sha256
    assert result.as_dict()["tool"]["license_spdx"] == "Apache-2.0"
    output = tmp_path / "pnp-result.yaml"
    result.to_artifact(
        result_id="synthetic-pnp-result",
        options=OpenCvProbabilisticPnpOptions(),
        command=["calibrex", "camera-lidar", "refine-probabilistic-pnp"],
    ).save(output)
    payload = output.read_text(encoding="utf-8")
    assert "slac.probabilistic_pnp_result/v0.1" in payload
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
