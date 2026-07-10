"""Filesystem dataset adapter."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from calibrex.data.base import StreamSummary, TimestampedRecord
from calibrex.data.manifest import DatasetManifest, find_manifest, load_manifest


class FilesystemDataset:
    """Dataset reader for directory-based examples and exported logs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.manifest = _load_optional_manifest(self.path)

    def streams(self) -> list[StreamSummary]:
        """Return manifest streams when available, otherwise suffix-based groups."""

        if self.manifest is not None:
            return [
                StreamSummary(
                    name=name,
                    kind=stream.kind,
                    message_count=stream.count,
                    topic=stream.topic,
                    sensor=stream.sensor,
                )
                for name, stream in sorted(self.manifest.streams.items())
            ]
        return _discover_suffix_streams(self.path)

    def records(self, stream: str) -> Iterable[TimestampedRecord]:
        """Yield records for a manifest stream.

        The alpha loader supports path globs with timestamp extraction deferred
        to future stream-specific adapters.
        """

        if self.manifest is None or stream not in self.manifest.streams:
            return []
        stream_manifest = self.manifest.streams[stream]
        if stream_manifest.path is None:
            return []
        matches = sorted(self.path.glob(stream_manifest.path))
        return [
            TimestampedRecord(stream=stream, timestamp_ns=0, payload_path=str(match))
            for match in matches
        ]


def _load_optional_manifest(path: Path) -> DatasetManifest | None:
    if find_manifest(path) is None:
        return None
    return load_manifest(path)


def _discover_suffix_streams(path: Path) -> list[StreamSummary]:
    if not path.exists():
        return []
    suffix_groups: dict[str, int] = {}
    for file_path in (entry for entry in path.rglob("*") if entry.is_file()):
        suffix = file_path.suffix.lower() or "<none>"
        suffix_groups[suffix] = suffix_groups.get(suffix, 0) + 1
    return [
        StreamSummary(name=suffix, kind=_kind_from_suffix(suffix), message_count=count)
        for suffix, count in sorted(suffix_groups.items())
    ]


def _kind_from_suffix(suffix: str) -> str:
    image_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    pointcloud_suffixes = {".pcd", ".ply", ".las", ".laz"}
    table_suffixes = {".csv", ".json", ".yaml", ".yml"}
    if suffix in image_suffixes:
        return "image"
    if suffix in pointcloud_suffixes:
        return "pointcloud"
    if suffix in table_suffixes:
        return "metadata"
    return "file"
