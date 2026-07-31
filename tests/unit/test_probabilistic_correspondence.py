from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from calibrex.core.probabilistic_correspondence import (
    CorrespondenceProviderIdentity,
    CorrespondenceTrainingDeclaration,
    ProbabilisticCorrespondenceArtifact,
    ProbabilisticCorrespondenceFrame,
    ProbabilisticCorrespondenceProvenance,
    ProbabilisticImageCorrespondence,
    load_probabilistic_correspondence,
)
from calibrex.data.depth import DepthCameraIntrinsics


def test_probabilistic_correspondence_round_trip(tmp_path: Path) -> None:
    artifact = _artifact()
    output = tmp_path / "correspondence.yaml"
    artifact.save(output)

    loaded = load_probabilistic_correspondence(output)

    assert loaded == artifact
    assert loaded.provider.training is not None
    assert loaded.provider.training.evaluation_overlap == "none"


def test_probabilistic_correspondence_rejects_non_positive_covariance() -> None:
    with pytest.raises(ValidationError, match="positive definite"):
        ProbabilisticImageCorrespondence(
            correspondence_id="bad",
            point_lidar_m=[0.0, 0.0, 5.0],
            image_mean_px=[320.0, 240.0],
            image_covariance_px2=[1.0, 2.0, 2.0, 1.0],
            outlier_probability=0.1,
            reliability=0.9,
        )


def _artifact() -> ProbabilisticCorrespondenceArtifact:
    points = [
        (-1.0, -1.0, 5.0),
        (1.0, -1.0, 5.0),
        (-1.0, 1.0, 5.0),
        (1.0, 1.0, 5.0),
    ]
    return ProbabilisticCorrespondenceArtifact(
        artifact_id="synthetic-correspondence",
        dataset_id="synthetic",
        split_id="test",
        provider=CorrespondenceProviderIdentity(
            provider="fixture",
            model="known-correspondence",
            version="1",
            source_repository="https://example.test/provider",
            source_commit="a" * 40,
            license_spdx="Apache-2.0",
            checkpoint_redistribution="not applicable",
            training=CorrespondenceTrainingDeclaration(
                training_datasets=["synthetic-train"],
                evaluation_overlap="none",
                test_data_used_for_online_refinement=False,
                evidence="disjoint generated seeds",
            ),
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
                    fx=400.0,
                    fy=400.0,
                    cx=320.0,
                    cy=240.0,
                ),
                correspondences=[
                    ProbabilisticImageCorrespondence(
                        correspondence_id=str(index),
                        point_lidar_m=list(point),
                        image_mean_px=[
                            400.0 * point[0] / point[2] + 320.0,
                            400.0 * point[1] / point[2] + 240.0,
                        ],
                        image_covariance_px2=[1.0, 0.0, 0.0, 1.0],
                        outlier_probability=0.0,
                        reliability=1.0,
                    )
                    for index, point in enumerate(points)
                ],
            )
        ],
        provenance=ProbabilisticCorrespondenceProvenance(
            generator="pytest",
            generator_version="1",
            input_sha256={"fixture": "b" * 64},
        ),
    )
