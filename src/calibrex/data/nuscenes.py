"""nuScenes metadata reader for public autonomous-driving datasets."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from calibrex.data.base import StreamSummary, TimestampedRecord


@dataclass(frozen=True)
class NuScenesMetadataStats:
    """Lightweight nuScenes table diagnostics."""

    status: str
    version: str | None
    scene_count: int = 0
    sample_count: int = 0
    sample_data_count: int = 0
    sensor_count: int = 0
    calibrated_sensor_count: int = 0
    ego_pose_count: int = 0
    keyframe_count: int = 0
    file_count: int = 0
    missing_file_count: int = 0
    modality_counts: dict[str, int] = field(default_factory=dict)
    channel_counts: dict[str, int] = field(default_factory=dict)
    calibrated_sensors: dict[str, dict[str, object]] = field(default_factory=dict)
    timestamp_start_ns: int | None = None
    timestamp_end_ns: int | None = None
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON/YAML-friendly representation."""

        return {
            "status": self.status,
            "version": self.version,
            "scene_count": self.scene_count,
            "sample_count": self.sample_count,
            "sample_data_count": self.sample_data_count,
            "sensor_count": self.sensor_count,
            "calibrated_sensor_count": self.calibrated_sensor_count,
            "ego_pose_count": self.ego_pose_count,
            "keyframe_count": self.keyframe_count,
            "file_count": self.file_count,
            "missing_file_count": self.missing_file_count,
            "modality_counts": dict(sorted(self.modality_counts.items())),
            "channel_counts": dict(sorted(self.channel_counts.items())),
            "calibrated_sensors": dict(sorted(self.calibrated_sensors.items())),
            "timestamp_start_ns": self.timestamp_start_ns,
            "timestamp_end_ns": self.timestamp_end_ns,
            "reason": self.reason,
        }


class NuScenesDataset:
    """Reader for extracted nuScenes metadata tables."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.version_path = find_nuscenes_version_path(self.path)
        self.data_root = _data_root(self.path, self.version_path)

    def streams(self) -> list[StreamSummary]:
        """Return nuScenes streams grouped by sensor channel."""

        if self.version_path is None:
            return []
        channel_counts = _channel_counts(self.version_path)
        sensors = _sensor_by_channel(self.version_path)
        streams: list[StreamSummary] = []
        for channel, count in sorted(channel_counts.items()):
            sensor = sensors.get(channel, {})
            modality = _string(sensor.get("modality")) or "metadata"
            streams.append(
                StreamSummary(
                    name=channel,
                    kind=_kind_for_modality(modality),
                    message_count=count,
                    topic=channel,
                    sensor=channel.lower(),
                )
            )
        return streams

    def records(self, stream: str) -> Iterable[TimestampedRecord]:
        """Yield normalized sample_data records for one nuScenes channel."""

        if self.version_path is None:
            return []
        rows = _sample_data_with_sensor(self.version_path)
        records: list[TimestampedRecord] = []
        for row in rows:
            if row.channel != stream:
                continue
            records.append(
                TimestampedRecord(
                    stream=stream,
                    timestamp_ns=row.timestamp_ns,
                    payload_path=str(self.data_root / row.filename) if row.filename else None,
                    metadata={
                        "token": row.token,
                        "sample_token": row.sample_token,
                        "calibrated_sensor_token": row.calibrated_sensor_token,
                        "ego_pose_token": row.ego_pose_token,
                        "is_key_frame": row.is_key_frame,
                        "modality": row.modality,
                    },
                )
            )
        return records


@dataclass(frozen=True)
class _SampleDataRow:
    token: str
    channel: str
    modality: str
    timestamp_ns: int
    filename: str | None
    sample_token: str | None
    calibrated_sensor_token: str | None
    ego_pose_token: str | None
    is_key_frame: bool


def find_nuscenes_version_path(path: str | Path) -> Path | None:
    """Find a nuScenes version directory containing JSON metadata tables."""

    root = Path(path)
    if (root / "sample_data.json").exists() and (root / "sensor.json").exists():
        return root
    candidates = sorted(root.glob("v1.0-*"))
    for candidate in candidates:
        if (candidate / "sample_data.json").exists() and (candidate / "sensor.json").exists():
            return candidate
    return None


def summarize_nuscenes_metadata(root: str | Path) -> NuScenesMetadataStats:
    """Summarize nuScenes metadata without importing the nuScenes SDK."""

    requested_path = Path(root)
    version_path = find_nuscenes_version_path(requested_path)
    if version_path is None:
        return NuScenesMetadataStats(
            status="unavailable",
            version=None,
            reason="nuScenes metadata tables were not found",
        )

    data_root = _data_root(requested_path, version_path)
    sample_data_rows = _sample_data_with_sensor(version_path)
    timestamps = [row.timestamp_ns for row in sample_data_rows]
    modality_counts = Counter(row.modality for row in sample_data_rows)
    channel_counts = Counter(row.channel for row in sample_data_rows)
    keyframe_count = sum(1 for row in sample_data_rows if row.is_key_frame)
    file_count = 0
    missing_file_count = 0
    for row in sample_data_rows:
        if not row.filename:
            continue
        if (data_root / row.filename).exists():
            file_count += 1
        else:
            missing_file_count += 1

    return NuScenesMetadataStats(
        status="scored",
        version=version_path.name,
        scene_count=len(_load_table(version_path, "scene")),
        sample_count=len(_load_table(version_path, "sample")),
        sample_data_count=len(sample_data_rows),
        sensor_count=len(_load_table(version_path, "sensor")),
        calibrated_sensor_count=len(_load_table(version_path, "calibrated_sensor")),
        ego_pose_count=len(_load_table(version_path, "ego_pose")),
        keyframe_count=keyframe_count,
        file_count=file_count,
        missing_file_count=missing_file_count,
        modality_counts=dict(modality_counts),
        channel_counts=dict(channel_counts),
        calibrated_sensors=_calibrated_sensor_summary(version_path),
        timestamp_start_ns=min(timestamps) if timestamps else None,
        timestamp_end_ns=max(timestamps) if timestamps else None,
    )


def read_nuscenes_reference_extrinsics(root: str | Path) -> dict[str, dict[str, object]]:
    """Read nuScenes calibrated_sensor transforms as Calibrex reference extrinsics."""

    version_path = find_nuscenes_version_path(root)
    if version_path is None:
        return {}
    references: dict[str, dict[str, object]] = {}
    for channel, values in _calibrated_sensor_summary(version_path).items():
        translation = values.get("translation_m")
        rotation = values.get("rotation_quat_xyzw")
        if not isinstance(translation, list) or not isinstance(rotation, list):
            continue
        child = channel.lower()
        references[f"T_ego_{child}"] = {
            "convention": "T_parent_child",
            "parent": "ego",
            "child": child,
            "translation_m": translation,
            "rotation_quat_xyzw": rotation,
            "quality": {"grade": "pass"},
            "source": "nuScenes calibrated_sensor",
            "channel": channel,
            "modality": values.get("modality"),
        }
    return references


def _sample_data_with_sensor(version_path: Path) -> list[_SampleDataRow]:
    sensors_by_token = _index_by_token(_load_table(version_path, "sensor"))
    calibrated_by_token = _index_by_token(_load_table(version_path, "calibrated_sensor"))
    rows: list[_SampleDataRow] = []
    for row in _load_table(version_path, "sample_data"):
        calibrated = calibrated_by_token.get(_string(row.get("calibrated_sensor_token")) or "")
        sensor = (
            sensors_by_token.get(_string(calibrated.get("sensor_token")) or "")
            if calibrated is not None
            else None
        )
        channel = _string(sensor.get("channel")) if sensor is not None else None
        modality = _string(sensor.get("modality")) if sensor is not None else None
        rows.append(
            _SampleDataRow(
                token=_string(row.get("token")) or "",
                channel=channel or "unknown",
                modality=modality or "unknown",
                timestamp_ns=_timestamp_us_to_ns(row.get("timestamp")),
                filename=_string(row.get("filename")),
                sample_token=_string(row.get("sample_token")),
                calibrated_sensor_token=_string(row.get("calibrated_sensor_token")),
                ego_pose_token=_string(row.get("ego_pose_token")),
                is_key_frame=bool(row.get("is_key_frame")),
            )
        )
    return rows


def _calibrated_sensor_summary(version_path: Path) -> dict[str, dict[str, object]]:
    sensors_by_token = _index_by_token(_load_table(version_path, "sensor"))
    summary: dict[str, dict[str, object]] = {}
    for row in _load_table(version_path, "calibrated_sensor"):
        sensor = sensors_by_token.get(_string(row.get("sensor_token")) or "")
        if sensor is None:
            continue
        channel = _string(sensor.get("channel"))
        if channel is None:
            continue
        translation = _numeric_list(row.get("translation"), expected=3)
        rotation_wxyz = _numeric_list(row.get("rotation"), expected=4)
        camera_intrinsic = _matrix_or_none(row.get("camera_intrinsic"))
        summary[channel] = {
            "modality": _string(sensor.get("modality")) or "unknown",
            "translation_m": translation,
            "rotation_quat_xyzw": _wxyz_to_xyzw(rotation_wxyz),
            "camera_intrinsic": camera_intrinsic,
        }
    return summary


def _channel_counts(version_path: Path) -> dict[str, int]:
    counts = Counter(row.channel for row in _sample_data_with_sensor(version_path))
    return dict(counts)


def _sensor_by_channel(version_path: Path) -> dict[str, Mapping[str, object]]:
    output: dict[str, Mapping[str, object]] = {}
    for row in _load_table(version_path, "sensor"):
        channel = _string(row.get("channel"))
        if channel is not None:
            output[channel] = row
    return output


def _load_table(version_path: Path, name: str) -> list[dict[str, object]]:
    path = version_path / f"{name}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict)]


def _index_by_token(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for row in rows:
        token = _string(row.get("token"))
        if token is not None:
            output[token] = row
    return output


def _data_root(requested_path: Path, version_path: Path | None) -> Path:
    if version_path is None:
        return requested_path
    if version_path.parent != requested_path and version_path.name.startswith("v1.0-"):
        return version_path.parent
    if version_path.name.startswith("v1.0-"):
        return requested_path
    return version_path


def _kind_for_modality(modality: str) -> str:
    if modality == "camera":
        return "image"
    if modality == "lidar":
        return "pointcloud"
    if modality == "radar":
        return "radar"
    return modality


def _timestamp_us_to_ns(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int | float):
        return int(value) * 1000
    return 0


def _string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _numeric_list(value: object, *, expected: int) -> list[float] | None:
    if not isinstance(value, list | tuple) or len(value) != expected:
        return None
    output: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            return None
        output.append(float(item))
    return output


def _matrix_or_none(value: object) -> list[list[float]] | None:
    if not isinstance(value, list):
        return None
    matrix: list[list[float]] = []
    for row in value:
        values = _numeric_list(row, expected=3)
        if values is None:
            return None
        matrix.append(values)
    return matrix


def _wxyz_to_xyzw(value: list[float] | None) -> list[float] | None:
    if value is None:
        return None
    return [value[1], value[2], value[3], value[0]]
