"""Dataset manifest schema and loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from calibrex.core.io import read_mapping

DATASET_MANIFEST_SCHEMA_VERSION: Literal["slac.dataset_manifest/v0.1"] = (
    "slac.dataset_manifest/v0.1"
)

StreamKind = Literal[
    "image",
    "depth_image",
    "rgbd",
    "pointcloud",
    "imu",
    "radar",
    "metadata",
    "pose_graph",
    "trajectory",
    "other",
]


class StrictModel(BaseModel):
    """Base model with stable, explicit fields."""

    model_config = ConfigDict(extra="forbid")


class StreamManifest(StrictModel):
    """One timestamped stream in a dataset manifest."""

    kind: StreamKind
    count: int | None = Field(default=None, ge=0)
    path: str | None = None
    topic: str | None = None
    sensor: str | None = None
    frame_id: str | None = None
    timestamp_field: str = "timestamp_ns"
    fields: list[str] = Field(default_factory=list)


class DatasetManifest(StrictModel):
    """Portable dataset manifest independent of ROS message classes."""

    schema_version: Literal["slac.dataset_manifest/v0.1"] = DATASET_MANIFEST_SCHEMA_VERSION
    name: str
    description: str | None = None
    time_base: str = "sensor_time_ns"
    streams: dict[str, StreamManifest]
    provenance: dict[str, str] = Field(default_factory=dict)

    @field_validator("streams")
    @classmethod
    def require_streams(cls, value: dict[str, StreamManifest]) -> dict[str, StreamManifest]:
        if not value:
            msg = "dataset manifest requires at least one stream"
            raise ValueError(msg)
        return value


def manifest_json_schema() -> dict[str, Any]:
    """Return the JSON schema for dataset manifests."""

    return DatasetManifest.model_json_schema()


def find_manifest(path: str | Path) -> Path | None:
    """Return the manifest path for a dataset path when present."""

    dataset_path = Path(path)
    if dataset_path.is_file() and dataset_path.name in {"manifest.yaml", "manifest.yml"}:
        return dataset_path
    if dataset_path.is_dir():
        for name in ("manifest.yaml", "manifest.yml"):
            candidate = dataset_path / name
            if candidate.exists():
                return candidate
    return None


def load_manifest(path: str | Path) -> DatasetManifest:
    """Load a dataset manifest file or a directory containing one."""

    manifest_path = find_manifest(path)
    if manifest_path is None:
        msg = f"dataset manifest not found at {path}"
        raise FileNotFoundError(msg)
    return DatasetManifest.model_validate(read_mapping(manifest_path))
