from pathlib import Path

import pytest
from pydantic import ValidationError

from calibrex.data.depth import (
    DepthCameraIntrinsics,
    DepthImageTransform,
    DepthMapObservation,
    DepthProviderArtifact,
    DepthProviderEnvironment,
    DepthProviderIdentity,
    DepthProviderTrainingDeclaration,
    default_depth_provider_provenance,
    depth_file_reference,
    load_depth_provider,
    verify_depth_provider_files,
)


def _artifact(tmp_path: Path) -> DepthProviderArtifact:
    image_path = tmp_path / "000000.png"
    depth_path = tmp_path / "000000.npy"
    image_path.write_bytes(b"fixture-image")
    depth_path.write_bytes(b"fixture-depth")
    observation = DepthMapObservation(
        frame_id="000000",
        capture_time_ns=1_000,
        source_frame="camera0",
        image=depth_file_reference(image_path, encoding="png_rgb8"),
        depth=depth_file_reference(depth_path, encoding="npy_float32"),
        intrinsics=DepthCameraIntrinsics(
            width=4,
            height=3,
            fx=2.0,
            fy=2.0,
            cx=1.5,
            cy=1.0,
        ),
        image_transform=DepthImageTransform(
            source_width=4,
            source_height=3,
            output_width=4,
            output_height=3,
            scale_x=1.0,
            scale_y=1.0,
            interpolation="none",
        ),
        scale_convention="relative_depth",
        valid_fraction=1.0,
    )
    return DepthProviderArtifact(
        artifact_id="fixture-depth",
        provider=DepthProviderIdentity(
            provider="fixture",
            model="fixture-depth",
            version="1.0",
            source_repository="https://example.test/depth",
            source_commit="0123456789abcdef",
            license_spdx="MIT",
            checkpoint_redistribution="not redistributed",
            training=DepthProviderTrainingDeclaration(
                training_dataset="fixture train",
                training_split="train",
                evaluation_overlap="none",
                test_data_used_for_online_refinement=False,
                evidence="disjoint fixture IDs",
            ),
        ),
        environment=DepthProviderEnvironment(
            execution_mode="imported",
            python_version="3.12",
        ),
        scale_convention="relative_depth",
        observations=[observation],
        provenance=default_depth_provider_provenance(
            generator_version="calibrex.depth_provider/v0.1",
            input_manifest_sha256="1" * 64,
            output_manifest_sha256="2" * 64,
        ),
    )


def test_depth_provider_round_trip_and_digest_verification(tmp_path: Path) -> None:
    path = tmp_path / "provider.yaml"
    artifact = _artifact(tmp_path)

    artifact.save(path)

    loaded = load_depth_provider(path)
    assert loaded == artifact
    assert loaded.provider.training is not None
    assert loaded.provider.training.evaluation_overlap == "none"
    assert verify_depth_provider_files(loaded) == []


def test_mei_intrinsics_require_explicit_distortion() -> None:
    with pytest.raises(ValidationError, match="radial-tangential"):
        DepthCameraIntrinsics(
            width=1400,
            height=1400,
            fx=1485.0,
            fy=1485.0,
            cx=699.0,
            cy=698.0,
            projection="mei",
            xi=2.55,
        )

    intrinsics = DepthCameraIntrinsics(
        width=1400,
        height=1400,
        fx=1485.0,
        fy=1485.0,
        cx=699.0,
        cy=698.0,
        projection="mei",
        xi=2.55,
        distortion_model="radial-tangential",
        distortion=[0.05, 4.5, 0.001, -0.001],
    )

    assert intrinsics.projection == "mei"
    assert len(intrinsics.distortion) == 4


def test_depth_provider_detects_changed_output(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    Path(artifact.observations[0].depth.path).write_bytes(b"changed")

    issues = verify_depth_provider_files(artifact)

    assert len(issues) == 1
    assert issues[0].startswith("digest_mismatch:")


def test_metric_depth_requires_scale(tmp_path: Path) -> None:
    payload = _artifact(tmp_path).observations[0].model_dump(mode="json")
    payload["scale_convention"] = "metric_z"

    with pytest.raises(ValidationError, match="scale_to_m"):
        DepthMapObservation.model_validate(payload)


def test_provider_rejects_mixed_scale_conventions(tmp_path: Path) -> None:
    payload = _artifact(tmp_path).model_dump(mode="json")
    payload["scale_convention"] = "metric_z"

    with pytest.raises(ValidationError, match="scale_convention differs"):
        DepthProviderArtifact.model_validate(payload)
