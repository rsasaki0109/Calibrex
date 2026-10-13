"""Solver-neutral depth-map and provider artifact contracts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.result import StrictModel

DEPTH_PROVIDER_SCHEMA_VERSION: Literal[
    "slac.depth_provider/v0.1"
] = "slac.depth_provider/v0.1"
DepthScaleConvention = Literal["metric_z", "metric_range", "relative_depth", "disparity"]
DepthEncoding = Literal["npy_float32", "npy_float64", "png_uint16", "exr_float32"]


class DepthFileReference(StrictModel):
    """Content-addressed depth-provider input or output file."""

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    encoding: str


class DepthCameraIntrinsics(StrictModel):
    """Camera intrinsics aligned with the depth grid."""

    width: int = Field(gt=1)
    height: int = Field(gt=1)
    fx: float = Field(gt=0.0)
    fy: float = Field(gt=0.0)
    cx: float
    cy: float
    projection: Literal["pinhole", "double_sphere", "mei"] = "pinhole"
    xi: float | None = None
    alpha: float | None = Field(default=None, gt=0.0, lt=1.0)
    distortion_model: str = "none"
    distortion: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_projection_parameters(self) -> DepthCameraIntrinsics:
        """Require the parameters needed by the selected projection."""

        if self.projection == "double_sphere" and (
            self.xi is None or self.alpha is None
        ):
            raise ValueError("double_sphere intrinsics require xi and alpha")
        if self.projection == "mei":
            if self.xi is None:
                raise ValueError("mei intrinsics require xi")
            if self.distortion_model != "radial-tangential":
                raise ValueError(
                    "mei intrinsics require radial-tangential distortion"
                )
            if len(self.distortion) != 4:
                raise ValueError(
                    "mei intrinsics require distortion [k1, k2, p1, p2]"
                )
        return self


class DepthImageTransform(StrictModel):
    """Declared mapping from the source image grid to the depth grid."""

    source_width: int = Field(gt=1)
    source_height: int = Field(gt=1)
    output_width: int = Field(gt=1)
    output_height: int = Field(gt=1)
    scale_x: float = Field(gt=0.0)
    scale_y: float = Field(gt=0.0)
    crop_left_px: float = 0.0
    crop_top_px: float = 0.0
    interpolation: str


class DepthMapObservation(StrictModel):
    """One synchronized, digest-pinned camera depth observation."""

    frame_id: str
    capture_time_ns: int
    source_frame: str
    image: DepthFileReference
    depth: DepthFileReference
    uncertainty: DepthFileReference | None = None
    intrinsics: DepthCameraIntrinsics
    image_transform: DepthImageTransform
    scale_convention: DepthScaleConvention
    scale_to_m: float | None = Field(default=None, gt=0.0)
    invalid_values: list[float] = Field(default_factory=list)
    valid_fraction: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_metric_scale(self) -> DepthMapObservation:
        """Require an explicit metric scale for metric depth encodings."""

        if self.scale_convention.startswith("metric_") and self.scale_to_m is None:
            raise ValueError("metric depth observations require scale_to_m")
        return self


class DepthProviderTrainingDeclaration(StrictModel):
    """Declared relationship between provider training data and evaluation."""

    training_dataset: str
    training_split: str | None = None
    evaluation_overlap: Literal["none", "possible", "confirmed", "unknown"]
    test_data_used_for_online_refinement: bool
    evidence: str


class DepthProviderIdentity(StrictModel):
    """Model/provider identity and redistribution boundary."""

    provider: str
    model: str
    version: str
    source_repository: str
    source_commit: str
    license_spdx: str
    checkpoint: DepthFileReference | None = None
    checkpoint_redistribution: str
    training: DepthProviderTrainingDeclaration | None = None


class DepthProviderEnvironment(StrictModel):
    """Pinned external environment used to produce depth maps."""

    execution_mode: Literal["subprocess", "container", "imported"]
    command: list[str] = Field(default_factory=list)
    container_digest: str | None = None
    python_version: str | None = None
    dependencies: dict[str, str] = Field(default_factory=dict)
    device: str | None = None


class DepthPreprocessingStep(StrictModel):
    """Ordered provider preprocessing/postprocessing operation."""

    name: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class DepthProviderProvenance(StrictModel):
    """Provenance of the generated provider artifact."""

    generator: str
    generator_version: str
    git_commit: str | None = None
    input_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DepthProviderArtifact(StrictModel):
    """Schema-valid manifest for externally generated or imported depth maps."""

    schema_version: Literal["slac.depth_provider/v0.1"] = DEPTH_PROVIDER_SCHEMA_VERSION
    artifact_id: str
    provider: DepthProviderIdentity
    environment: DepthProviderEnvironment
    scale_convention: DepthScaleConvention
    preprocessing: list[DepthPreprocessingStep] = Field(default_factory=list)
    observations: list[DepthMapObservation] = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)
    provenance: DepthProviderProvenance

    @model_validator(mode="after")
    def check_scale_consistency(self) -> DepthProviderArtifact:
        """Keep one scale convention per provider artifact."""

        inconsistent = [
            item.frame_id
            for item in self.observations
            if item.scale_convention != self.scale_convention
        ]
        if inconsistent:
            raise ValueError(
                "observation scale_convention differs from artifact: "
                + ", ".join(inconsistent)
            )
        return self

    def save(self, path: str | Path) -> None:
        """Save the provider artifact as YAML or JSON."""

        write_mapping(Path(path), self.model_dump(mode="json", exclude_none=True))


def depth_provider_json_schema() -> dict[str, Any]:
    """Return the JSON schema for depth-provider artifacts."""

    return DepthProviderArtifact.model_json_schema()


def load_depth_provider(path: str | Path) -> DepthProviderArtifact:
    """Load and validate a depth-provider artifact."""

    return DepthProviderArtifact.model_validate(read_mapping(Path(path)))


def depth_file_reference(
    path: str | Path,
    *,
    encoding: str,
) -> DepthFileReference:
    """Build a digest reference for an existing regular file."""

    file_path = Path(path)
    digest = sha256_path(file_path)
    if not file_path.is_file() or digest is None:
        raise ValueError(f"depth-provider file is not readable: {file_path}")
    return DepthFileReference(
        path=str(file_path),
        sha256=digest,
        size_bytes=file_path.stat().st_size,
        encoding=encoding,
    )


def verify_depth_provider_files(
    artifact: DepthProviderArtifact,
    *,
    base_path: str | Path | None = None,
) -> list[str]:
    """Return deterministic missing/digest mismatch issues for referenced files."""

    root = Path(base_path) if base_path is not None else None
    references = []
    if artifact.provider.checkpoint is not None:
        references.append(artifact.provider.checkpoint)
    for observation in artifact.observations:
        references.extend((observation.image, observation.depth))
        if observation.uncertainty is not None:
            references.append(observation.uncertainty)
    issues = []
    for reference in references:
        path = Path(reference.path)
        if root is not None and not path.is_absolute():
            path = root / path
        observed = sha256_path(path)
        if observed is None:
            issues.append(f"missing:{reference.path}")
        elif observed != reference.sha256:
            issues.append(
                f"digest_mismatch:{reference.path}:"
                f"expected={reference.sha256}:observed={observed}"
            )
    return issues


def default_depth_provider_provenance(
    *,
    generator_version: str,
    input_manifest_sha256: str,
    output_manifest_sha256: str,
) -> DepthProviderProvenance:
    """Build standard provenance for a depth-provider artifact."""

    return DepthProviderProvenance(
        generator="calibrex.depth_provider",
        generator_version=generator_version,
        git_commit=git_commit(),
        input_manifest_sha256=input_manifest_sha256,
        output_manifest_sha256=output_manifest_sha256,
    )
