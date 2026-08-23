"""Provider-export contract for real KITTI and KITTI-360 correspondences.

External depth or correspondence providers write one digest-pinned ``.npz``
export per capture frame.  This module only validates that export and turns it
into the provider-neutral probabilistic correspondence artifact.  No provider
runtime, ROS package, or copyleft implementation is imported here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, model_validator

from calibrex.core.camera_lidar_artifacts import Sha256
from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.probabilistic_correspondence import (
    CorrespondenceProviderIdentity,
    ProbabilisticCorrespondenceArtifact,
    ProbabilisticCorrespondenceFrame,
    ProbabilisticCorrespondenceProvenance,
    ProbabilisticImageCorrespondence,
)
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.result import StrictModel
from calibrex.data.depth import DepthCameraIntrinsics, DepthFileReference

CAMERA_LIDAR_CORRESPONDENCE_EXPORT_SCHEMA_VERSION: Literal[
    "slac.camera_lidar_correspondence_export/v0.1"
] = "slac.camera_lidar_correspondence_export/v0.1"
CAMERA_LIDAR_CORRESPONDENCE_ADAPTER_VERSION = (
    "calibrex.camera_lidar_correspondence_export/v0.1"
)


class CameraLidarCorrespondenceExportFrame(StrictModel):
    """One provider-generated NPZ export and its sensor-frame metadata."""

    frame_id: str
    capture_time_ns: int
    camera_frame: str
    lidar_frame: str
    intrinsics: DepthCameraIntrinsics
    export: DepthFileReference


class CameraLidarCorrespondenceExportManifest(StrictModel):
    """Schema-valid provider boundary for KITTI-family correspondence exports."""

    schema_version: Literal[
        "slac.camera_lidar_correspondence_export/v0.1"
    ] = CAMERA_LIDAR_CORRESPONDENCE_EXPORT_SCHEMA_VERSION
    artifact_id: str
    dataset_id: str
    dataset_family: Literal["kitti_raw", "kitti_360"]
    sequence_id: str
    split_id: str
    dataset_license_spdx: str = Field(min_length=1)
    provider: CorrespondenceProviderIdentity
    frames: list[CameraLidarCorrespondenceExportFrame] = Field(min_length=1)
    provenance: ProbabilisticCorrespondenceProvenance
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_frame_ids(self) -> CameraLidarCorrespondenceExportManifest:
        """Require frame identity and a complete provider license boundary."""

        identifiers = [item.frame_id for item in self.frames]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("correspondence export frame IDs must be unique")
        pairs = {(item.camera_frame, item.lidar_frame) for item in self.frames}
        if len(pairs) != 1:
            raise ValueError("correspondence export frames must share sensor frame IDs")
        for field_name in (
            "provider",
            "model",
            "version",
            "source_repository",
            "source_commit",
            "license_spdx",
            "checkpoint_redistribution",
        ):
            value = getattr(self.provider, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"correspondence provider {field_name} must be declared"
                )
        return self

    def save(self, path: str | Path) -> None:
        """Save the provider export manifest as YAML or JSON."""

        write_mapping(Path(path), self.model_dump(mode="json", exclude_none=True))


def camera_lidar_correspondence_export_json_schema() -> dict[str, Any]:
    """Return the JSON Schema for provider correspondence export manifests."""

    return CameraLidarCorrespondenceExportManifest.model_json_schema()


def load_camera_lidar_correspondence_export(
    path: str | Path,
) -> CameraLidarCorrespondenceExportManifest:
    """Load and validate a provider correspondence export manifest."""

    return CameraLidarCorrespondenceExportManifest.model_validate(
        read_mapping(Path(path))
    )


def build_probabilistic_correspondence_artifact(
    manifest_path: str | Path,
    *,
    command: list[str] | None = None,
) -> ProbabilisticCorrespondenceArtifact:
    """Convert verified NPZ frame exports into a provider-neutral artifact."""

    manifest_file = Path(manifest_path).resolve()
    manifest = load_camera_lidar_correspondence_export(manifest_file)
    manifest_digest = _required_digest(manifest_file)
    frames: list[ProbabilisticCorrespondenceFrame] = []
    input_digests: dict[str, str] = {"export_manifest": manifest_digest}
    for frame in manifest.frames:
        export_path = _resolve_reference(manifest_file.parent, frame.export)
        observed_digest = _required_digest(export_path)
        if observed_digest != frame.export.sha256:
            raise ValueError(
                f"correspondence export digest mismatch for {frame.frame_id}: "
                f"expected {frame.export.sha256}, observed {observed_digest}"
            )
        if export_path.stat().st_size != frame.export.size_bytes:
            raise ValueError(
                f"correspondence export size mismatch for {frame.frame_id}: "
                f"expected {frame.export.size_bytes}, observed {export_path.stat().st_size}"
            )
        input_digests[f"frame:{frame.frame_id}"] = observed_digest
        frames.append(
            _load_frame_export(
                export_path,
                frame=frame,
            )
        )
    return ProbabilisticCorrespondenceArtifact(
        artifact_id=manifest.artifact_id,
        dataset_id=manifest.dataset_id,
        split_id=manifest.split_id,
        dataset_family=manifest.dataset_family,
        sequence_id=manifest.sequence_id,
        dataset_license_spdx=manifest.dataset_license_spdx,
        provider=manifest.provider,
        frames=frames,
        provenance=ProbabilisticCorrespondenceProvenance(
            generator="calibrex.core.camera_lidar_correspondence_export",
            generator_version=CAMERA_LIDAR_CORRESPONDENCE_ADAPTER_VERSION,
            git_commit=git_commit(),
            command=command or [],
            input_sha256=input_digests,
        ),
        warnings=list(manifest.warnings),
    )


def _load_frame_export(
    path: Path,
    *,
    frame: CameraLidarCorrespondenceExportFrame,
) -> ProbabilisticCorrespondenceFrame:
    try:
        archive = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load correspondence export {path}: {exc}") from exc
    with archive:
        required = {"point_lidar_m", "image_mean_px"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(
                f"correspondence export {path} is missing arrays: {', '.join(missing)}"
            )
        points = np.asarray(archive["point_lidar_m"], dtype=float)
        means = np.asarray(archive["image_mean_px"], dtype=float)
        count = _validate_array_shape(points, (None, 3), "point_lidar_m", path)
        _validate_array_shape(means, (count, 2), "image_mean_px", path)
        covariance = np.asarray(
            archive["image_covariance_px2"]
            if "image_covariance_px2" in archive.files
            else np.tile(np.eye(2, dtype=float).reshape(1, 4), (count, 1)),
            dtype=float,
        )
        _validate_array_shape(covariance, (count, 4), "image_covariance_px2", path)
        outlier = np.asarray(
            archive["outlier_probability"]
            if "outlier_probability" in archive.files
            else np.zeros(count, dtype=float),
            dtype=float,
        )
        reliability = np.asarray(
            archive["reliability"]
            if "reliability" in archive.files
            else np.ones(count, dtype=float),
            dtype=float,
        )
        _validate_array_shape(outlier, (count,), "outlier_probability", path)
        _validate_array_shape(reliability, (count,), "reliability", path)
        identifiers = (
            [str(value) for value in archive["correspondence_id"]]
            if "correspondence_id" in archive.files
            else [str(index) for index in range(count)]
        )
        if len(identifiers) != count or len(set(identifiers)) != count:
            raise ValueError(f"correspondence IDs are invalid in {path}")
        if not all(
            np.isfinite(values).all()
            for values in (points, means, covariance, outlier, reliability)
        ):
            raise ValueError(f"correspondence export contains non-finite values: {path}")
        correspondences = [
            ProbabilisticImageCorrespondence(
                correspondence_id=identifiers[index],
                point_lidar_m=points[index].tolist(),
                image_mean_px=means[index].tolist(),
                image_covariance_px2=covariance[index].tolist(),
                outlier_probability=float(outlier[index]),
                reliability=float(reliability[index]),
            )
            for index in range(count)
        ]
    return ProbabilisticCorrespondenceFrame(
        frame_id=frame.frame_id,
        capture_time_ns=frame.capture_time_ns,
        camera_frame=frame.camera_frame,
        lidar_frame=frame.lidar_frame,
        intrinsics=frame.intrinsics,
        correspondences=correspondences,
    )


def _validate_array_shape(
    values: np.ndarray[Any, Any],
    expected: tuple[int | None, ...],
    name: str,
    path: Path,
) -> int:
    if values.ndim != len(expected) or any(
        size is not None and actual != size
        for actual, size in zip(values.shape, expected, strict=True)
    ):
        raise ValueError(
            f"{name} in {path} must have shape "
            + "x".join("N" if size is None else str(size) for size in expected)
        )
    return int(values.shape[0])


def _resolve_reference(base: Path, reference: DepthFileReference) -> Path:
    path = Path(reference.path)
    return path if path.is_absolute() else (base / path).resolve()


def _required_digest(path: Path) -> Sha256:
    digest = sha256_path(path)
    if digest is None or not path.is_file():
        raise ValueError(f"correspondence export file is not readable: {path}")
    return digest
