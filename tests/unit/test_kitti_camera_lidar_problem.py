from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from calibrex.cli.main import main
from calibrex.core.camera_lidar_artifacts import load_camera_lidar_problem
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
from calibrex.data.kitti_camera_lidar_problem import (
    build_kitti_raw_camera_lidar_problem,
    read_rectified_velodyne_to_camera_transform,
)
from calibrex.evaluation.borer_rotation_benchmark import load_borer_problem

_DIGEST = "0" * 64


def _write_fixture(root: Path) -> tuple[Path, Path]:
    sequence = root / "2011_09_30" / "2011_09_30_drive_0018_sync"
    image_dir = sequence / "image_02" / "data"
    lidar_dir = sequence / "velodyne_points" / "data"
    image_dir.mkdir(parents=True)
    lidar_dir.mkdir(parents=True)
    (sequence.parent / "calib_velo_to_cam.txt").write_text(
        "R: 1 0 0 0 1 0 0 0 1\nT: 1 2 3\n",
        encoding="utf-8",
    )
    (sequence.parent / "calib_cam_to_cam.txt").write_text(
        "R_rect_00: 1 0 0 0 1 0 0 0 1\n"
        "P_rect_02: 100 0 2 -50 0 100 1 0 0 0 1 0\n",
        encoding="utf-8",
    )
    (sequence / "velodyne_points" / "timestamps.txt").write_text(
        "2011-09-30 12:00:00.000000000\n"
        "2011-09-30 12:00:00.100000000\n",
        encoding="utf-8",
    )
    observations = []
    for index in range(2):
        frame_id = f"{index:010d}"
        image = image_dir / f"{frame_id}.png"
        depth = root / "depth" / f"{frame_id}.npy"
        depth.parent.mkdir(exist_ok=True)
        image.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
        np.save(depth, np.full((2, 4), 5.0, dtype=np.float32))
        np.asarray([[1.0, 0.0, 5.0, 0.5]], dtype=np.float32).tofile(
            lidar_dir / f"{frame_id}.bin"
        )
        observations.append(
            DepthMapObservation(
                frame_id=frame_id,
                capture_time_ns=index * 100_000_000,
                source_frame="camera_2",
                image=depth_file_reference(image, encoding="png"),
                depth=depth_file_reference(depth, encoding="npy_float32"),
                intrinsics=DepthCameraIntrinsics(
                    width=4,
                    height=2,
                    fx=100.0,
                    fy=100.0,
                    cx=2.0,
                    cy=1.0,
                ),
                image_transform=DepthImageTransform(
                    source_width=4,
                    source_height=2,
                    output_width=4,
                    output_height=2,
                    scale_x=1.0,
                    scale_y=1.0,
                    interpolation="bilinear",
                ),
                scale_convention="metric_z",
                scale_to_m=1.0,
                valid_fraction=1.0,
            )
        )
    provider = DepthProviderArtifact(
        artifact_id="featdepth-fixture",
        provider=DepthProviderIdentity(
            provider="FeatDepth",
            model="FeatDepth KITTI raw",
            version="fixture",
            source_repository="https://github.com/sconlyshootery/FeatDepth",
            source_commit="550420b3fb51a027549716b74c6fbce41651d3a5",
            license_spdx="MIT",
            checkpoint_redistribution="not redistributed",
        ),
        environment=DepthProviderEnvironment(execution_mode="subprocess"),
        scale_convention="metric_z",
        observations=observations,
        provenance=DepthProviderProvenance(
            generator="test",
            generator_version="fixture",
            input_manifest_sha256=_DIGEST,
            output_manifest_sha256=_DIGEST,
        ),
    )
    provider_path = root / "featdepth.json"
    provider.save(provider_path)
    return sequence, provider_path


def test_rectified_transform_includes_selected_camera_origin_offset(
    tmp_path: Path,
) -> None:
    sequence, _ = _write_fixture(tmp_path)

    transform = read_rectified_velodyne_to_camera_transform(sequence.parent)

    assert transform.translation_m == pytest.approx((0.5, 2.0, 3.0))
    assert transform.rotation_quat_xyzw == pytest.approx((0.0, 0.0, 0.0, 1.0))


def test_build_kitti_problem_binds_and_digest_locks_inputs(tmp_path: Path) -> None:
    sequence, provider_path = _write_fixture(tmp_path)

    problem = build_kitti_raw_camera_lidar_problem(sequence, provider_path)

    assert problem.sequence_id == "2011_09_30_drive_0018_sync"
    assert problem.dataset_id == "kitti_raw_2011_09_30_drive_0018"
    assert [item.frame_id for item in problem.observations] == [
        "0000000000",
        "0000000001",
    ]
    assert problem.reference_transform_camera_lidar.parent == "camera_2"
    assert problem.reference_transform_camera_lidar.translation_m == pytest.approx(
        [0.5, 2.0, 3.0]
    )
    assert problem.provenance.source_sha256 != _DIGEST


def test_build_kitti_problem_cli_writes_valid_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sequence, provider_path = _write_fixture(tmp_path)
    output = tmp_path / "problem.yaml"

    status = main(
        [
            "camera-lidar",
            "build-kitti-problem",
            str(sequence),
            str(provider_path),
            "--output",
            str(output),
            "--json",
        ]
    )

    assert status == 0
    assert load_camera_lidar_problem(output).problem_id.endswith("-image_02-d2d")
    assert len(load_borer_problem(output).observations) == 2
    assert '"frame_count": 2' in capsys.readouterr().out


def test_builder_rejects_provider_image_from_another_stream(tmp_path: Path) -> None:
    sequence, provider_path = _write_fixture(tmp_path)
    provider = DepthProviderArtifact.model_validate_json(
        provider_path.read_text(encoding="utf-8")
    )
    provider.observations[0].image.path = str(tmp_path / "unrelated.png")
    provider.save(provider_path)

    with pytest.raises(
        ValueError,
        match=r"verification failed|requested KITTI stream",
    ):
        build_kitti_raw_camera_lidar_problem(sequence, provider_path)
