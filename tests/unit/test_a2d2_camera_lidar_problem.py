from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from calibrex.data.a2d2_camera_lidar_problem import (
    build_a2d2_camera_lidar_problem,
)
from calibrex.data.depth import (
    DepthCameraIntrinsics,
    DepthImageTransform,
    DepthMapObservation,
    DepthProviderArtifact,
    DepthProviderEnvironment,
    DepthProviderIdentity,
    DepthProviderProvenance,
    depth_file_reference,
)
from calibrex.evaluation.borer_rotation_benchmark import load_borer_problem


def test_a2d2_builder_backprojects_and_digest_pins_points(
    tmp_path: Path,
) -> None:
    provider_path = _provider(tmp_path)
    for frame_id in ("000000060", "000000061"):
        _a2d2_npz(tmp_path, frame_id)
    output_directory = tmp_path / "generated-lidar"
    manifest_path = tmp_path / "lidar-manifest.yaml"

    problem = build_a2d2_camera_lidar_problem(
        tmp_path,
        provider_path,
        output_lidar_directory=output_directory,
        generated_manifest_path=manifest_path,
        command=["pytest"],
    )
    problem_path = tmp_path / "problem.yaml"
    problem.save(problem_path)
    loaded = load_borer_problem(problem_path)

    assert loaded.problem.dataset_family == "A2D2"
    assert len(loaded.observations) == 2
    assert loaded.observations[0].lidar_points.shape == (100, 3)
    assert (
        loaded.reference_transform_camera_lidar
        == loaded.problem.initial_transform_camera_lidar.as_se3()
    )
    assert manifest_path.is_file()

    np.save(
        output_directory / "000000060.npy",
        np.zeros((100, 3)),
        allow_pickle=False,
    )
    with pytest.raises(ValueError, match="LiDAR digest mismatch"):
        load_borer_problem(problem_path)


def _provider(root: Path) -> Path:
    observations = []
    intrinsics = DepthCameraIntrinsics(
        width=30,
        height=20,
        fx=20.0,
        fy=20.0,
        cx=15.0,
        cy=10.0,
    )
    for index, frame_id in enumerate(("000000060", "000000061")):
        image = root / f"image-{frame_id}.png"
        image.write_bytes(b"fixture-image")
        depth = root / f"depth-{frame_id}.npy"
        np.save(depth, np.ones((20, 30), dtype=np.float32), allow_pickle=False)
        observations.append(
            DepthMapObservation(
                frame_id=frame_id,
                capture_time_ns=index,
                source_frame="camera_frontleft",
                image=depth_file_reference(image, encoding="png_rgb8"),
                depth=depth_file_reference(depth, encoding="npy_float32"),
                intrinsics=intrinsics,
                image_transform=DepthImageTransform(
                    source_width=30,
                    source_height=20,
                    output_width=30,
                    output_height=20,
                    scale_x=1.0,
                    scale_y=1.0,
                    interpolation="none",
                ),
                scale_convention="relative_depth",
                valid_fraction=1.0,
            )
        )
    artifact = DepthProviderArtifact(
        artifact_id="a2d2-fixture",
        provider=DepthProviderIdentity(
            provider="fixture",
            model="fixture",
            version="1",
            source_repository="https://example.test",
            source_commit="a" * 40,
            license_spdx="Apache-2.0",
            checkpoint_redistribution="not applicable",
        ),
        environment=DepthProviderEnvironment(execution_mode="imported"),
        scale_convention="relative_depth",
        observations=observations,
        provenance=DepthProviderProvenance(
            generator="pytest",
            generator_version="1",
            input_manifest_sha256="b" * 64,
            output_manifest_sha256="c" * 64,
        ),
    )
    path = root / "provider.yaml"
    artifact.save(path)
    return path


def _a2d2_npz(root: Path, frame_id: str) -> None:
    rows = np.tile(np.arange(10, dtype=float), 10)
    cols = np.repeat(np.arange(10, 20, dtype=float), 10)
    np.savez(
        root / f"fixture_lidar_frontleft_{frame_id}.npz",
        **{
            "pcloud_attr.valid": np.ones(100, dtype=bool),
            "pcloud_attr.row": rows,
            "pcloud_attr.col": cols,
            "pcloud_attr.depth": np.full(100, 5.0),
        },
    )
