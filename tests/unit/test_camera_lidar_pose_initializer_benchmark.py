from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest
from tools import run_i2pnet_pose_benchmark as benchmark

from calibrex.core.camera_lidar_artifacts import (
    CameraLidarArtifactProvenance,
    CameraLidarHitDefinition,
    CameraLidarIsolationDeclaration,
    CameraLidarPerturbation,
)
from calibrex.core.camera_lidar_correspondence_export import (
    CameraLidarCorrespondenceExportFrame,
    CameraLidarCorrespondenceExportManifest,
)
from calibrex.core.camera_lidar_pose_initializer_benchmark import (
    CameraLidarPoseInitializerBenchmarkProtocol,
    CameraLidarPoseInitializerProvider,
    load_camera_lidar_pose_initializer_protocol,
)
from calibrex.core.geometry import SE3
from calibrex.core.io import write_mapping
from calibrex.core.probabilistic_correspondence import (
    CorrespondenceProviderIdentity,
    ProbabilisticCorrespondenceProvenance,
)
from calibrex.core.provenance import sha256_path
from calibrex.core.result import TransformEstimateProvenance, TransformResult
from calibrex.core.validation import validate_file
from calibrex.data.depth import DepthCameraIntrinsics, DepthFileReference

SHA = sha256(b"pose-initializer-protocol").hexdigest()


def _protocol(
    *,
    partition: str = "development",
    provider_selection_split: str = "development",
) -> CameraLidarPoseInitializerBenchmarkProtocol:
    return CameraLidarPoseInitializerBenchmarkProtocol.model_validate(
        {
            "protocol_id": "i2pnet-dev-v0.1",
            "problem_id": "kitti-raw-0018",
            "problem_sha256": SHA,
            "dataset_id": "kitti_raw_0018",
            "dataset_family": "kitti_raw",
            "dataset_license_spdx": "LicenseRef-KITTI",
            "sequence_id": "2011_09_30_drive_0018_sync",
            "camera_stream": "image_02",
            "partition": partition,
            "frame_sampling": "three endpoint-excluding uniform frames",
            "frame_ids": ["1", "2", "3"],
            "perturbation_count": 1,
            "perturbations": [
                CameraLidarPerturbation(
                    trial_id="pitch-pos10-tz-pos1",
                    rotation_deg_xyz=[0.0, 10.0, 0.0],
                    translation_m_xyz=[1.0, 0.0, 1.0],
                ).model_dump(mode="json")
            ],
            "hit": CameraLidarHitDefinition(
                rotation_error_max_deg=0.5,
                translation_error_max_m=0.2,
            ).model_dump(mode="json"),
            "provider": CameraLidarPoseInitializerProvider(
                provider_id="i2pnet_kitti_large",
                model="I2PNet/KITTI-large",
                source_repository="https://github.com/IRMVLab/I2PNet",
                source_commit="a" * 40,
                license_spdx="MIT",
                checkpoint_sha256="b" * 64,
                adapter_version="adapter/v0.2",
                adapter_sha256="c" * 64,
                neighbor_backend="torch",
                training_datasets=["KITTI Odometry"],
                evaluation_overlap="unknown",
            ).model_dump(mode="json"),
            "isolation": CameraLidarIsolationDeclaration(
                provider_selection_split=provider_selection_split,
                threshold_selection_split="development",
                evaluation_split=partition,
                test_data_used_for_selection=False,
                evidence="development-only provider audit",
            ).model_dump(mode="json"),
            "failure_policy": "retain every nonzero provider exit as a failed trial",
            "metric_definitions": {
                "rotation_error_deg": "quaternion geodesic error",
                "translation_error_m": "Euclidean translation error",
                "hit": "strict joint threshold",
            },
            "provenance": CameraLidarArtifactProvenance(
                generator="pytest",
                generator_version="v0.1",
                source_sha256=SHA,
            ).model_dump(mode="json"),
        }
    )


def test_protocol_round_trip_and_auto_validation(tmp_path: Path) -> None:
    path = tmp_path / "protocol.yaml"
    _protocol().save(path)

    loaded = load_camera_lidar_pose_initializer_protocol(path)
    report = validate_file(path)

    assert loaded.protocol_id == "i2pnet-dev-v0.1"
    assert report.kind == "camera-lidar-pose-initializer-protocol"


def test_protocol_rejects_duplicate_trials_and_evaluation_selection() -> None:
    payload = _protocol().model_dump(mode="json")
    payload["perturbation_count"] = 2
    payload["perturbations"] *= 2
    with pytest.raises(ValueError, match="trial IDs must be unique"):
        CameraLidarPoseInitializerBenchmarkProtocol.model_validate(payload)

    with pytest.raises(ValueError, match="evaluation-data selection"):
        _protocol(partition="evaluation", provider_selection_split="evaluation")


def test_left_camera_frame_perturbation_matches_training_convention() -> None:
    reference = SE3(
        (0.05624655421119172, -0.07481401634273016, -0.32779358344330395),
        (
            -0.4991459120684823,
            0.5040847568861339,
            -0.4968172956264801,
            -0.49992448540430684,
        ),
    )

    initial = benchmark._left_camera_frame_perturbation(
        reference,
        [0.0, 10.0, 0.0],
        [1.0, 0.0, 1.0],
    )

    assert initial.translation_m == pytest.approx(
        [0.9984712842515605, -0.07481401634273016, 0.6674192259985624]
    )
    assert initial.rotation_quat_xyzw == pytest.approx(
        [
            -0.5405469915869995,
            0.4585952723556799,
            -0.451423323117593,
            -0.5419560032001984,
        ]
    )


def test_strict_hit_excludes_equal_thresholds() -> None:
    definition = CameraLidarHitDefinition(
        rotation_error_max_deg=0.5,
        translation_error_max_m=0.2,
    )

    assert benchmark._hit(0.49, 0.19, definition)
    assert not benchmark._hit(0.5, 0.19, definition)
    assert not benchmark._hit(0.49, 0.2, definition)


def test_provider_output_rejects_backend_outside_frozen_protocol(tmp_path: Path) -> None:
    protocol = _protocol()
    initial_digest = "d" * 64
    initial = TransformResult(
        parent="camera",
        child="lidar",
        translation_m=[0.0, 0.0, 0.0],
        rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
    )
    intrinsics = DepthCameraIntrinsics(
        width=16,
        height=8,
        fx=10.0,
        fy=10.0,
        cx=8.0,
        cy=4.0,
    )
    frames = [
        CameraLidarCorrespondenceExportFrame(
            frame_id=frame_id,
            capture_time_ns=index,
            camera_frame="camera",
            lidar_frame="lidar",
            intrinsics=intrinsics,
            export=DepthFileReference(
                path=f"{frame_id}.npz",
                sha256="e" * 64,
                size_bytes=1,
                encoding="npz",
            ),
        )
        for index, frame_id in enumerate(protocol.frame_ids)
    ]
    manifest = CameraLidarCorrespondenceExportManifest(
        artifact_id="fixture",
        dataset_id=protocol.dataset_id,
        dataset_family=protocol.dataset_family,
        sequence_id=protocol.sequence_id,
        split_id=protocol.partition,
        dataset_license_spdx=protocol.dataset_license_spdx,
        provider=CorrespondenceProviderIdentity(
            provider="I2PNet diagnostic cost-volume adapter",
            model=protocol.provider.model,
            version="fixture",
            source_repository=protocol.provider.source_repository,
            source_commit=protocol.provider.source_commit,
            license_spdx=protocol.provider.license_spdx,
            checkpoint=DepthFileReference(
                path="checkpoint.pt",
                sha256=protocol.provider.checkpoint_sha256,
                size_bytes=1,
                encoding="pytorch",
            ),
            checkpoint_redistribution="not redistributed",
        ),
        frames=frames,
        provenance=ProbabilisticCorrespondenceProvenance(
            generator="pytest",
            generator_version=protocol.provider.adapter_version,
            command=[
                "provider.py",
                "--neighbor-backend",
                "cuda_extension",
                "--camera-stream",
                protocol.camera_stream,
            ],
            input_sha256={
                "adapter_source": protocol.provider.adapter_sha256,
                "initial_transform": initial_digest,
                "i2pnet_checkpoint": protocol.provider.checkpoint_sha256,
            },
        ),
    )
    manifest_path = tmp_path / "manifest.yaml"
    manifest.save(manifest_path)
    manifest_digest = sha256_path(manifest_path)
    assert manifest_digest is not None
    output = TransformResult(
        parent="camera",
        child="lidar",
        translation_m=[0.0, 0.0, 0.0],
        rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
        provenance=TransformEstimateProvenance(
            role_in_comparison="output",
            notes=[
                f"initial transform sha256={initial_digest}",
                f"source sha256 correspondence_manifest={manifest_digest}",
                f"source sha256 adapter_source={protocol.provider.adapter_sha256}",
                f"source sha256 i2pnet_checkpoint={protocol.provider.checkpoint_sha256}",
            ],
        ),
    )
    pose_path = tmp_path / "pose.yaml"
    write_mapping(pose_path, output.model_dump(mode="json", exclude_none=True))

    with pytest.raises(ValueError, match="neighbor backend"):
        benchmark._verified_provider_output(
            pose_path,
            manifest_path=manifest_path,
            initial=initial,
            initial_digest=initial_digest,
            protocol=protocol,
        )
