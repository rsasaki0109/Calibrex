"""Immutable, commercial-safe handoff lock for the optional Koide adapter.

The lock records only facts that are either published by the upstream project
or are Calibrex execution policy.  It deliberately does not copy Koide,
SuperGlue, ROS, or model weights into the package.  A published final image
digest is pinned; the mutable base image named by the upstream Dockerfile is
left explicitly unresolved and is required from an operator if a rebuild is
attempted.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.result import StrictModel

KOIDE_EXECUTION_LOCK_SCHEMA_VERSION: Literal[
    "slac.koide_execution_lock/v0.1"
] = "slac.koide_execution_lock/v0.1"
KOIDE_LOCK_SOURCE_REPOSITORY = "https://github.com/koide3/direct_visual_lidar_calibration"
KOIDE_LOCK_IMAGE_REPOSITORY = "koide3/direct_visual_lidar_calibration"
KOIDE_LOCK_HUMBLE_IMAGE = (
    "koide3/direct_visual_lidar_calibration@"
    "sha256:f7363cc3deadc2419f3527a7b9ec7bdabaa3031fd8e1b9c6cc46666ae1301bbb"
)
KOIDE_LOCK_SOURCE_COMMIT = "02a0dc039f5509708f384be4ff3228e0ae09352d"

_IMAGE_RE = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")


class KoideLockSource(StrictModel):
    """Pinned upstream source identity and official references."""

    repository: str = KOIDE_LOCK_SOURCE_REPOSITORY
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    license_spdx: Literal["MIT"] = "MIT"
    documentation_urls: list[str] = Field(min_length=1)


class KoideLockImage(StrictModel):
    """Published image pin and explicit rebuild limitation."""

    repository: str = KOIDE_LOCK_IMAGE_REPOSITORY
    tag: Literal["humble", "jazzy", "noetic"] = "humble"
    digest: str = Field(pattern=r"^[^@\s]+@sha256:[0-9a-f]{64}$")
    # The upstream Dockerfile uses this mutable tag.  Do not invent a digest
    # for it: a reproducible rebuild requires the operator to resolve and
    # record the base digest before building.
    base_image: str = "koide3/gtsam_docker:humble"
    base_image_digest: str | None = Field(default=None, pattern=r"^[^@\s]+@sha256:[0-9a-f]{64}$")
    base_image_digest_required_for_rebuild: Literal[True] = True

    @model_validator(mode="after")
    def validate_image_repository(self) -> KoideLockImage:
        if not self.digest.startswith(self.repository + "@"):
            raise ValueError("Koide image digest must match the pinned repository")
        return self


class KoideLockContainerPolicy(StrictModel):
    """Calibrex isolation policy applied around the official image."""

    engine: Literal["docker", "podman"] = "docker"
    network_mode: Literal["none"] = "none"
    input_mount_readonly: Literal[True] = True
    output_mount_writable: Literal[True] = True
    rootfs_readonly: Literal[True] = True
    source_image: str = KOIDE_LOCK_HUMBLE_IMAGE

    @model_validator(mode="after")
    def validate_source_image(self) -> KoideLockContainerPolicy:
        if not _IMAGE_RE.fullmatch(self.source_image):
            raise ValueError("source_image must be an immutable repository@sha256 reference")
        return self


KoideLockStageName = Literal["preprocess", "initial_guess", "calibrate"]


class KoideLockStage(StrictModel):
    """One argv-only command copied from the upstream official example."""

    name: KoideLockStageName
    argv: list[str] = Field(min_length=1)
    official_documentation_url: str
    requires_manual_gui: bool = False

    @model_validator(mode="after")
    def reject_shell_and_superglue(self) -> KoideLockStage:
        if any("\x00" in item for item in self.argv):
            raise ValueError("Koide lock stage argv must not contain NUL bytes")
        if any("superglue" in item.lower() for item in self.argv):
            raise ValueError("commercial Koide lock stages must exclude SuperGlue")
        return self


class KoideLockDatasetContract(StrictModel):
    """Capture requirements stated by the upstream data-collection guide."""

    capture_format: Literal["ros2-bag"] = "ros2-bag"
    required_message_types: list[str] = Field(
        default_factory=lambda: [
            "sensor_msgs/msg/Image",
            "sensor_msgs/msg/PointCloud2",
            "sensor_msgs/msg/CameraInfo",
        ],
        min_length=2,
    )
    camera_intrinsics_precalibrated: Literal[True] = True
    rigid_mount_required: Literal[True] = True
    minimum_pairings: int = Field(default=1, ge=1)
    operator_capture_note: str = Field(min_length=1)


class KoideExecutionLock(StrictModel):
    """Digest-bound handoff contract; this is not evidence that Koide ran."""

    schema_version: Literal["slac.koide_execution_lock/v0.1"] = (
        KOIDE_EXECUTION_LOCK_SCHEMA_VERSION
    )
    lock_id: str = Field(min_length=1)
    profile: Literal["commercial"] = "commercial"
    source: KoideLockSource
    image: KoideLockImage
    container: KoideLockContainerPolicy
    initial_guess_mode: Literal["manual", "precomputed"] = "manual"
    superglue_policy: Literal["excluded"] = "excluded"
    stages: list[KoideLockStage] = Field(min_length=3, max_length=3)
    dataset_contract: KoideLockDatasetContract
    operator_steps: list[str] = Field(min_length=1)
    lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_workflow(self) -> KoideExecutionLock:
        names = [stage.name for stage in self.stages]
        if names != ["preprocess", "initial_guess", "calibrate"]:
            raise ValueError("Koide lock stages must be preprocess, initial_guess, calibrate")
        manual = self.stages[1]
        if "initial_guess_manual" not in " ".join(manual.argv):
            raise ValueError("commercial Koide lock must use the manual initial-guess command")
        if self.container.source_image != self.image.digest:
            raise ValueError("container.source_image must equal image.digest")
        return self

    def with_lock_digest(self) -> KoideExecutionLock:
        """Return a copy with the canonical digest over lock content."""

        payload = self.model_dump(mode="json", exclude_none=False)
        payload.pop("lock_sha256", None)
        digest = _canonical_sha256(payload)
        return self.model_copy(update={"lock_sha256": digest})

    def verify_lock_digest(self) -> None:
        """Raise when the checked-in lock content has changed."""

        expected = self.with_lock_digest().lock_sha256
        if self.lock_sha256 != expected:
            raise ValueError(
                "Koide execution lock self-digest mismatch: "
                f"declared={self.lock_sha256}, expected={expected}"
            )

    def save(self, path: str | Path) -> None:
        """Write a digest-complete YAML or JSON lock."""

        write_mapping(
            Path(path),
            self.with_lock_digest().model_dump(mode="json", exclude_none=False),
        )


def koide_execution_lock_json_schema() -> dict[str, Any]:
    """Return the generated JSON Schema for the public lock."""

    return KoideExecutionLock.model_json_schema()


def load_koide_execution_lock(path: str | Path) -> KoideExecutionLock:
    """Load and verify a schema-valid immutable lock."""

    lock = KoideExecutionLock.model_validate(read_mapping(Path(path)))
    lock.verify_lock_digest()
    return lock


def _canonical_sha256(payload: object) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "KOIDE_EXECUTION_LOCK_SCHEMA_VERSION",
    "KOIDE_LOCK_HUMBLE_IMAGE",
    "KOIDE_LOCK_SOURCE_COMMIT",
    "KOIDE_LOCK_SOURCE_REPOSITORY",
    "KoideExecutionLock",
    "KoideLockContainerPolicy",
    "KoideLockDatasetContract",
    "KoideLockImage",
    "KoideLockSource",
    "KoideLockStage",
    "koide_execution_lock_json_schema",
    "load_koide_execution_lock",
]
