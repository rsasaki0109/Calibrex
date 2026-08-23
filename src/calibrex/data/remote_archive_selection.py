"""Schema-valid provenance for selected members of remote dataset archives."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.provenance import sha256_path
from calibrex.core.result import StrictModel

REMOTE_ARCHIVE_SELECTION_SCHEMA_VERSION: Literal[
    "slac.remote_archive_selection/v0.2"
] = "slac.remote_archive_selection/v0.2"
REMOTE_ARCHIVE_SELECTION_SCHEMA_VERSION_V0_3: Literal[
    "slac.remote_archive_selection/v0.3"
] = "slac.remote_archive_selection/v0.3"

RemoteArchiveRole = Literal["image", "pointcloud", "pose"]
RemoteArchiveSourceRole = Literal[
    "image", "pointcloud", "pose", "multi_stream"
]


class RemoteArchiveSource(StrictModel):
    """HTTP identity for one range-addressed upstream archive."""

    role: RemoteArchiveSourceRole
    url: str
    size_bytes: int = Field(gt=0)
    etag: str | None = None
    last_modified: str | None = None
    accept_ranges: bool


class RemoteArchiveSelectedMember(StrictModel):
    """One verified archive member materialized as a local dataset file."""

    role: RemoteArchiveRole
    frame_id: str = Field(pattern=r"^[0-9]{10}$")
    member_path: str
    crc32: str = Field(pattern=r"^[0-9a-f]{8}$")
    compressed_size_bytes: int = Field(ge=0)
    uncompressed_size_bytes: int = Field(ge=0)
    local_path: str
    local_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    local_size_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def check_size(self) -> RemoteArchiveSelectedMember:
        """Require the materialized file to match the ZIP member size."""

        if self.local_size_bytes != self.uncompressed_size_bytes:
            raise ValueError("local file size differs from ZIP member size")
        return self


class RemoteArchiveSelectedMetadataMember(StrictModel):
    """One stream-level metadata member materialized from an archive."""

    role: Literal[
        "image_timestamps",
        "pointcloud_timestamps",
        "pose_timestamps",
    ]
    member_path: str
    crc32: str = Field(pattern=r"^[0-9a-f]{8}$")
    compressed_size_bytes: int = Field(ge=0)
    uncompressed_size_bytes: int = Field(ge=0)
    local_path: str
    local_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    local_size_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def check_size(self) -> RemoteArchiveSelectedMetadataMember:
        if self.local_size_bytes != self.uncompressed_size_bytes:
            raise ValueError("local metadata size differs from ZIP member size")
        return self


class RemoteArchiveSelectionProvenance(StrictModel):
    """Downloader identity, command, and local metadata input digests."""

    generator: str
    generator_version: str
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    input_sha256: dict[str, str] = Field(default_factory=dict)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @field_validator("input_sha256")
    @classmethod
    def check_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        invalid = [
            key
            for key, digest in value.items()
            if len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ]
        if invalid:
            raise ValueError(f"input_sha256 contains invalid SHA-256 values: {invalid}")
        return value


class RemoteArchiveSelectionArtifact(StrictModel):
    """Endpoint-inclusive image/point-cloud subset with immutable lineage."""

    schema_version: Literal[
        "slac.remote_archive_selection/v0.1",
        "slac.remote_archive_selection/v0.2",
        "slac.remote_archive_selection/v0.3",
    ] = REMOTE_ARCHIVE_SELECTION_SCHEMA_VERSION
    artifact_id: str
    dataset_family: str
    dataset_id: str
    sequence_id: str
    dataset_license_spdx: str
    selection_policy: Literal[
        "endpoint_inclusive_same_numeric_frame_id/v0.1",
        "centered_same_numeric_frame_windows/v0.2",
        "center_images_with_centered_pointcloud_pose_windows/v0.3",
    ] = "endpoint_inclusive_same_numeric_frame_id/v0.1"
    requested_frame_count: int = Field(ge=2)
    frame_ids: list[str] = Field(min_length=2)
    center_frame_ids: list[str] | None = Field(default=None, min_length=2)
    stream_frame_ids: dict[RemoteArchiveRole, list[str]] | None = None
    archives: list[RemoteArchiveSource] = Field(min_length=1, max_length=3)
    members: list[RemoteArchiveSelectedMember] = Field(min_length=4)
    metadata_members: list[RemoteArchiveSelectedMetadataMember] = Field(
        default_factory=list
    )
    output_root: str
    provenance: RemoteArchiveSelectionProvenance

    @model_validator(mode="after")
    def check_selection(self) -> RemoteArchiveSelectionArtifact:
        """Require one image and point cloud for every selected frame."""

        if len(self.frame_ids) != self.requested_frame_count:
            raise ValueError("selected frame count differs from requested count")
        if len(self.frame_ids) != len(set(self.frame_ids)):
            raise ValueError("selected frame IDs must be unique")
        if self.frame_ids != sorted(self.frame_ids, key=int):
            raise ValueError("selected frame IDs must be numerically ordered")
        if self.schema_version.endswith("/v0.3"):
            self._check_multistream_window_selection()
            return self
        if (
            self.center_frame_ids is not None
            or self.stream_frame_ids is not None
            or self.metadata_members
        ):
            raise ValueError("remote selection v0.1/v0.2 cannot use v0.3 fields")
        if {item.role for item in self.archives} != {"image", "pointcloud"}:
            raise ValueError("selection requires image and pointcloud archives")
        member_keys = [(item.role, item.frame_id) for item in self.members]
        if len(member_keys) != len(set(member_keys)):
            raise ValueError("selected archive member role/frame pairs must be unique")
        expected = {
            (role, frame_id)
            for role in ("image", "pointcloud")
            for frame_id in self.frame_ids
        }
        if set(member_keys) != expected:
            raise ValueError("selected members do not cover every frame and stream")
        return self

    def _check_multistream_window_selection(self) -> None:
        if self.selection_policy != (
            "center_images_with_centered_pointcloud_pose_windows/v0.3"
        ):
            raise ValueError("remote selection v0.3 requires multistream policy")
        if self.center_frame_ids is None or self.stream_frame_ids is None:
            raise ValueError("remote selection v0.3 requires stream frame coverage")
        if self.center_frame_ids != sorted(set(self.center_frame_ids), key=int):
            raise ValueError("center frame IDs must be unique and ordered")
        if set(self.stream_frame_ids) != {"image", "pointcloud", "pose"}:
            raise ValueError("remote selection v0.3 requires three named streams")
        for role, frame_ids in self.stream_frame_ids.items():
            if frame_ids != sorted(set(frame_ids), key=int):
                raise ValueError(f"{role} stream frame IDs must be unique and ordered")
        image_ids = self.stream_frame_ids["image"]
        pointcloud_ids = self.stream_frame_ids["pointcloud"]
        pose_ids = self.stream_frame_ids["pose"]
        if image_ids != self.center_frame_ids:
            raise ValueError("raw image coverage must equal center frames")
        if set(self.center_frame_ids) - set(pointcloud_ids):
            raise ValueError("center frames must have point clouds")
        required_pose_ids = {
            frame_id
            for pointcloud_id in pointcloud_ids
            for frame_id in (
                pointcloud_id,
                _raw_motion_adjacent_frame_id(pointcloud_id),
            )
        }
        if set(pose_ids) != required_pose_ids:
            raise ValueError("raw pose coverage must include each motion neighbor exactly")
        expected_frame_ids = sorted(
            set(image_ids) | set(pointcloud_ids) | set(pose_ids),
            key=int,
        )
        if self.frame_ids != expected_frame_ids:
            raise ValueError("remote selection frame IDs must equal the stream union")
        expected_streams = {
            "image": image_ids,
            "pointcloud": pointcloud_ids,
            "pose": pose_ids,
        }
        if [item.role for item in self.archives] != ["multi_stream"]:
            raise ValueError("raw multistream selection requires one source archive")
        member_keys = [(item.role, item.frame_id) for item in self.members]
        if len(member_keys) != len(set(member_keys)):
            raise ValueError("selected archive member role/frame pairs must be unique")
        expected_members = {
            (role, frame_id)
            for role, frame_ids in expected_streams.items()
            for frame_id in frame_ids
        }
        if set(member_keys) != expected_members:
            raise ValueError("remote selection v0.3 member coverage is incomplete")
        metadata_roles = [item.role for item in self.metadata_members]
        if metadata_roles != [
            "image_timestamps",
            "pointcloud_timestamps",
            "pose_timestamps",
        ]:
            raise ValueError("remote selection v0.3 timestamp coverage is incomplete")

    def save(self, path: str | Path) -> None:
        """Save this archive selection as schema-valid YAML or JSON."""

        write_mapping(Path(path), self.model_dump(mode="json", exclude_none=True))


def load_remote_archive_selection(path: str | Path) -> RemoteArchiveSelectionArtifact:
    """Load and validate a remote archive selection artifact."""

    return RemoteArchiveSelectionArtifact.model_validate(read_mapping(Path(path)))


def verify_remote_archive_selection_files(
    artifact: RemoteArchiveSelectionArtifact,
    *,
    base_path: str | Path | None = None,
) -> list[str]:
    """Return deterministic local size and SHA-256 integrity issues."""

    root = Path(base_path) if base_path is not None else None
    references = [
        (
            f"member:{item.role}:{item.frame_id}",
            item.local_path,
            item.local_size_bytes,
            item.local_sha256,
        )
        for item in artifact.members
    ]
    references.extend(
        (
            f"metadata:{item.role}",
            item.local_path,
            item.local_size_bytes,
            item.local_sha256,
        )
        for item in artifact.metadata_members
    )
    issues: list[str] = []
    for label, declared_path, size_bytes, expected_digest in sorted(references):
        path = Path(declared_path)
        if root is not None and not path.is_absolute():
            path = root / path
        digest = sha256_path(path)
        if digest is None or not path.is_file():
            issues.append(f"missing:{label}:{declared_path}")
            continue
        if path.stat().st_size != size_bytes:
            issues.append(f"size_mismatch:{label}:{declared_path}")
        if digest != expected_digest:
            issues.append(f"digest_mismatch:{label}:{declared_path}")
    return issues


def remote_archive_selection_json_schema() -> dict[str, Any]:
    """Return the standalone JSON Schema for archive selections."""

    return RemoteArchiveSelectionArtifact.model_json_schema()


def _raw_motion_adjacent_frame_id(frame_id: str) -> str:
    value = int(frame_id)
    adjacent = value + 1 if value in (0, 1) else value - 1
    return f"{adjacent:010d}"
