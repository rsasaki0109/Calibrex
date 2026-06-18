"""Evidence artifact lineage helpers for calibration evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

ArtifactRole = Literal["train", "holdout", "query"]
LeakageStatus = Literal["pass", "fail"]


@dataclass(frozen=True)
class DatasetSliceArtifact:
    """Frame-level dataset slice used by an evaluation artifact."""

    artifact_id: str
    role: ArtifactRole
    frame_ids: tuple[str, ...]
    selection_policy: str

    def as_dict(self) -> dict[str, object]:
        """Return a JSON/YAML-friendly representation."""

        return {
            "artifact_id": self.artifact_id,
            "kind": "dataset_slice",
            "role": self.role,
            "frame_ids": list(self.frame_ids),
            "selection_policy": self.selection_policy,
        }


@dataclass(frozen=True)
class MapArtifact:
    """Map artifact with explicit frame lineage."""

    artifact_id: str
    map_type: str
    source_slice_id: str
    frame_ids: tuple[str, ...]
    dependency_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        """Return a JSON/YAML-friendly representation."""

        return {
            "artifact_id": self.artifact_id,
            "kind": "map",
            "map_type": self.map_type,
            "source_slice_id": self.source_slice_id,
            "frame_ids": list(self.frame_ids),
            "dependency_ids": list(self.dependency_ids),
        }


@dataclass(frozen=True)
class CorrespondenceArtifact:
    """Correspondence artifact with map and query lineage."""

    artifact_id: str
    correspondence_type: str
    source_map_id: str
    query_slice_id: str
    query_frame_ids: tuple[str, ...]
    dependency_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        """Return a JSON/YAML-friendly representation."""

        return {
            "artifact_id": self.artifact_id,
            "kind": "correspondence",
            "correspondence_type": self.correspondence_type,
            "source_map_id": self.source_map_id,
            "query_slice_id": self.query_slice_id,
            "query_frame_ids": list(self.query_frame_ids),
            "dependency_ids": list(self.dependency_ids),
        }


@dataclass(frozen=True)
class LeakageValidation:
    """Frame-level leakage validation for map/query evaluation artifacts."""

    status: LeakageStatus
    issue_count: int
    overlapping_frame_ids: tuple[str, ...]
    checked_dependency_ids: tuple[str, ...]
    reason: str

    def as_dict(self) -> dict[str, object]:
        """Return a JSON/YAML-friendly representation."""

        return {
            "status": self.status,
            "issue_count": self.issue_count,
            "overlapping_frame_ids": list(self.overlapping_frame_ids),
            "checked_dependency_ids": list(self.checked_dependency_ids),
            "reason": self.reason,
        }


def stable_artifact_id(prefix: str, components: Iterable[str]) -> str:
    """Build a deterministic short artifact id from textual components."""

    digest = sha256()
    for component in components:
        digest.update(component.encode("utf-8"))
        digest.update(b"\0")
    return f"{prefix}_{digest.hexdigest()[:12]}"


def validate_no_frame_overlap(
    *,
    map_frame_ids: Iterable[str],
    query_frame_ids: Iterable[str],
    checked_dependency_ids: Iterable[str] = (),
) -> LeakageValidation:
    """Validate that map-building frames and query/evaluation frames do not overlap."""

    overlap = tuple(sorted(set(map_frame_ids) & set(query_frame_ids)))
    dependency_ids = tuple(checked_dependency_ids)
    if overlap:
        return LeakageValidation(
            status="fail",
            issue_count=len(overlap),
            overlapping_frame_ids=overlap,
            checked_dependency_ids=dependency_ids,
            reason="map and query artifacts share frame ids",
        )
    return LeakageValidation(
        status="pass",
        issue_count=0,
        overlapping_frame_ids=(),
        checked_dependency_ids=dependency_ids,
        reason="map and query artifacts use disjoint frame ids",
    )
