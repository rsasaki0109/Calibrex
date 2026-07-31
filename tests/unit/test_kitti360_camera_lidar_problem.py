from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from calibrex.cli.main import main
from calibrex.core.camera_lidar_artifacts import load_camera_lidar_problem
from calibrex.core.io import write_mapping
from calibrex.core.provenance import sha256_path
from calibrex.data.depth import (
    DepthImageTransform,
    DepthMapObservation,
    DepthProviderArtifact,
    DepthProviderEnvironment,
    DepthProviderIdentity,
    DepthProviderProvenance,
    depth_file_reference,
)
from calibrex.data.kitti360_camera_lidar_problem import (
    build_kitti360_camera_lidar_problem,
    motion_compensate_kitti360_scan,
    read_kitti360_fisheye_intrinsics,
    read_kitti360_poses,
    read_kitti360_velodyne_to_camera_transform,
)
from calibrex.data.manifest import DatasetManifest, StreamManifest
from calibrex.evaluation.borer_rotation_benchmark import load_borer_problem

_DIGEST = "0" * 64


def _paths_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item.resolve())):
        file_digest = sha256_path(path)
        assert file_digest is not None
        digest.update(str(path.resolve()).encode())
        digest.update(b"\0")
        digest.update(file_digest.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _matrix_line(name: str, matrix: np.ndarray) -> str:
    values = " ".join(str(value) for value in matrix[:3, :].reshape(-1))
    return f"{name}: {values}\n"


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    sequence = root / "2013_05_28_drive_0000_sync"
    calibration = root / "calibration"
    image_dir = sequence / "image_03" / "data_rgb"
    lidar_dir = sequence / "velodyne_points" / "data"
    image_dir.mkdir(parents=True)
    lidar_dir.mkdir(parents=True)
    calibration.mkdir()
    (calibration / "image_03.yaml").write_text(
        "%YAML:1.0\n"
        "model_type: MEI\n"
        "image_width: 8\nimage_height: 6\n"
        "mirror_parameters:\n  xi: 1.5\n"
        "distortion_parameters:\n"
        "  k1: 0.1\n  k2: 0.2\n  p1: 0.01\n  p2: -0.01\n"
        "projection_parameters:\n"
        "  gamma1: 4.0\n  gamma2: 4.1\n  u0: 3.5\n  v0: 2.5\n",
        encoding="utf-8",
    )
    camera0_to_velodyne = np.eye(4)
    camera0_to_velodyne[0, 3] = 1.0
    (calibration / "calib_cam_to_velo.txt").write_text(
        " ".join(
            str(value) for value in camera0_to_velodyne[:3, :].reshape(-1)
        ),
        encoding="utf-8",
    )
    pose_camera0 = np.eye(4)
    pose_camera3 = np.eye(4)
    pose_camera3[1, 3] = 2.0
    (calibration / "calib_cam_to_pose.txt").write_text(
        _matrix_line("image_00", pose_camera0)
        + _matrix_line("image_03", pose_camera3),
        encoding="utf-8",
    )
    (sequence / "image_03" / "timestamps.txt").write_text(
        "2013-05-28 00:00:00.000000000\n", encoding="utf-8"
    )
    (sequence / "velodyne_points" / "timestamps.txt").write_text(
        "2013-05-28 00:00:00.001000000\n", encoding="utf-8"
    )

    image = image_dir / "0000000000.png"
    depth = root / "depth" / "0000000000.npy"
    depth.parent.mkdir()
    image.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    np.save(depth, np.full((6, 8), 5.0, dtype=np.float32))
    np.asarray([[1.0, 0.0, 5.0, 0.5]], dtype=np.float32).tofile(
        lidar_dir / "0000000000.bin"
    )
    intrinsics = read_kitti360_fisheye_intrinsics(calibration)
    provider = DepthProviderArtifact(
        artifact_id="midas-kitti360-fixture",
        provider=DepthProviderIdentity(
            provider="MiDaS",
            model="dpt_beit_large_512",
            version="v3.1",
            source_repository="https://github.com/isl-org/MiDaS",
            source_commit="1645b7e1675301fdfac03640738fe5a6531e17d6",
            license_spdx="MIT",
            checkpoint_redistribution="not redistributed",
        ),
        environment=DepthProviderEnvironment(execution_mode="subprocess"),
        scale_convention="relative_depth",
        observations=[
            DepthMapObservation(
                frame_id="0000000000",
                capture_time_ns=0,
                source_frame="camera_3",
                image=depth_file_reference(image, encoding="png_rgb8"),
                depth=depth_file_reference(depth, encoding="npy_float32"),
                intrinsics=intrinsics,
                image_transform=DepthImageTransform(
                    source_width=8,
                    source_height=6,
                    output_width=8,
                    output_height=6,
                    scale_x=1.0,
                    scale_y=1.0,
                    interpolation="bicubic",
                ),
                scale_convention="relative_depth",
                valid_fraction=1.0,
            )
        ],
        provenance=DepthProviderProvenance(
            generator="test",
            generator_version="fixture",
            input_manifest_sha256=_DIGEST,
            output_manifest_sha256=_DIGEST,
        ),
    )
    provider_path = root / "midas.json"
    provider.save(provider_path)
    return sequence, calibration, provider_path


def test_reads_mei_intrinsics_and_official_transform_chain(tmp_path: Path) -> None:
    _, calibration, _ = _write_fixture(tmp_path)

    intrinsics = read_kitti360_fisheye_intrinsics(calibration)
    transform = read_kitti360_velodyne_to_camera_transform(calibration)

    assert intrinsics.projection == "mei"
    assert intrinsics.distortion == pytest.approx([0.1, 0.2, 0.01, -0.01])
    assert transform.translation_m == pytest.approx((-1.0, -2.0, 0.0))


def test_build_and_cli_write_schema_valid_kitti360_problem(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sequence, calibration, provider_path = _write_fixture(tmp_path)
    direct = build_kitti360_camera_lidar_problem(
        sequence, provider_path, calibration_root=calibration
    )

    assert direct.dataset_family == "KITTI-360"
    assert "not motion compensated" in direct.time_convention
    assert direct.reference_transform_camera_lidar.parent == "camera_3"

    output = tmp_path / "problem.yaml"
    status = main(
        [
            "camera-lidar",
            "build-kitti360-problem",
            str(sequence),
            str(provider_path),
            "--calibration-root",
            str(calibration),
            "--output",
            str(output),
            "--json",
        ]
    )

    assert status == 0
    assert load_camera_lidar_problem(output).dataset_family == "KITTI-360"
    loaded = load_borer_problem(output)
    assert loaded.observations[0].camera.projection == "mei"
    assert loaded.observations[0].camera.distortion == pytest.approx(
        (0.1, 0.2, 0.01, -0.01)
    )
    assert '"frame_count": 1' in capsys.readouterr().out


def test_official_motion_compensation_scales_pose_delta_by_azimuth(
    tmp_path: Path,
) -> None:
    pose_path = tmp_path / "poses.txt"
    current = np.eye(4)
    previous = np.eye(4)
    previous[0, 3] = 1.0
    pose_path.write_text(
        _matrix_line("2", previous).replace(":", "")
        + _matrix_line("3", current).replace(":", ""),
        encoding="utf-8",
    )
    points = np.asarray(
        [[-1.0, 0.0, 0.0, 0.7], [1.0, 0.0, 0.0, 0.8]],
        dtype=np.float32,
    )

    compensated = motion_compensate_kitti360_scan(
        points,
        frame_id=3,
        poses=read_kitti360_poses(pose_path),
        velodyne_to_pose=np.eye(4),
    )

    assert compensated[0, :3] == pytest.approx([-0.5, 0.0, 0.0])
    assert compensated[1, :3] == pytest.approx([1.0, 0.0, 0.0])
    assert compensated[:, 3] == pytest.approx([0.7, 0.8])


def test_generated_lidar_directory_requires_matching_valid_manifest(
    tmp_path: Path,
) -> None:
    sequence, calibration, provider_path = _write_fixture(tmp_path)
    generated = tmp_path / "compensated"
    generated.mkdir()
    source = sequence / "velodyne_points" / "data" / "0000000000.bin"
    target = generated / source.name
    target.write_bytes(source.read_bytes())

    with pytest.raises(ValueError, match="requires lidar_manifest_path"):
        build_kitti360_camera_lidar_problem(
            sequence,
            provider_path,
            calibration_root=calibration,
            lidar_directory=generated,
        )

    manifest_path = generated / "manifest.yaml"
    manifest = DatasetManifest(
        name="compensated-fixture",
        streams={
            "velodyne_points": StreamManifest(
                kind="pointcloud",
                count=1,
                path=str(generated),
            )
        },
        provenance={
            "generator": "test",
            "output_sha256": _paths_digest([target]),
        },
    )
    write_mapping(manifest_path, manifest.model_dump(mode="json"))
    problem = build_kitti360_camera_lidar_problem(
        sequence,
        provider_path,
        calibration_root=calibration,
        lidar_directory=generated,
        lidar_manifest_path=manifest_path,
    )

    assert "official motion compensation adapter" in problem.time_convention
    assert problem.observations[0].lidar.path == str(target)

    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="output_sha256"):
        build_kitti360_camera_lidar_problem(
            sequence,
            provider_path,
            calibration_root=calibration,
            lidar_directory=generated,
            lidar_manifest_path=manifest_path,
        )
