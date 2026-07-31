"""Provenance-locked inputs for the KITTI raw Camera-LiDAR benchmark."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator

from calibrex import __version__
from calibrex.core.exceptions import DatasetError
from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.result import StrictModel

KITTI_BENCHMARK_INPUT_SCHEMA_VERSION: Literal["slac.kitti_benchmark_input/v0.1"] = (
    "slac.kitti_benchmark_input/v0.1"
)
KITTI_RAW_0005_DATASET_ID = "kitti_raw_2011_09_26_drive_0005"
KITTI_RAW_0005_SEQUENCE_ID = "2011_09_26_drive_0005_sync"
KITTI_RAW_OFFICIAL_URL = "https://www.cvlibs.net/datasets/kitti/raw_data.php"

# Twenty pairs span the complete 154-frame drive without selecting frames after
# inspecting metric values. The selection is part of the benchmark contract.
KITTI_RAW_0005_BENCHMARK_FRAME_IDS = tuple(f"{index:010d}" for index in range(0, 154, 8))

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
KITTIBenchmarkInputRole = Literal["camera", "lidar", "timestamp", "calibration"]


class KITTIBenchmarkInputFile(StrictModel):
    """One digest-locked file used by the KITTI benchmark."""

    path: str
    role: KITTIBenchmarkInputRole
    frame_id: str | None = None
    sha256: str = Field(pattern=_SHA256_PATTERN)
    size_bytes: int = Field(ge=0)


class KITTIBenchmarkInputProvenance(StrictModel):
    """Producer and source identity for a generated input lock."""

    generator: str = "calibrex.data.kitti_benchmark"
    generator_version: str = __version__
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_commit: str | None = None
    command: str
    official_url: str = KITTI_RAW_OFFICIAL_URL
    redistribution: str = "raw data not redistributed; obtain it from the official KITTI source"


class KITTIBenchmarkInputManifest(StrictModel):
    """Schema-valid inventory of every raw input selected for the benchmark."""

    schema_version: Literal["slac.kitti_benchmark_input/v0.1"] = (
        KITTI_BENCHMARK_INPUT_SCHEMA_VERSION
    )
    dataset_id: str = KITTI_RAW_0005_DATASET_ID
    sequence_id: str = KITTI_RAW_0005_SEQUENCE_ID
    source_sequence_path: str
    selection_policy: str = "20 prespecified pairs at frame indices 0:154:8"
    frame_ids: list[str] = Field(min_length=1)
    camera_stream: str = "image_02"
    lidar_stream: str = "velodyne_points"
    timestamp_line_counts: dict[str, int]
    calibration_files: list[str] = Field(min_length=2)
    files: list[KITTIBenchmarkInputFile] = Field(min_length=1)
    input_sha256: str = Field(pattern=_SHA256_PATTERN)
    provenance: KITTIBenchmarkInputProvenance

    @field_validator("frame_ids")
    @classmethod
    def require_unique_frame_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            msg = "KITTI benchmark frame_ids must be unique"
            raise ValueError(msg)
        if any(len(item) != 10 or not item.isdigit() for item in value):
            msg = "KITTI benchmark frame_ids must be ten-digit decimal strings"
            raise ValueError(msg)
        return value

    def save(self, path: str | Path) -> None:
        """Write the input manifest as JSON or YAML."""

        write_mapping(Path(path), self.model_dump(mode="json", exclude_none=True))


def kitti_benchmark_input_json_schema() -> dict[str, Any]:
    """Return the JSON schema for KITTI benchmark input manifests."""

    return KITTIBenchmarkInputManifest.model_json_schema()


def load_kitti_benchmark_input(path: str | Path) -> KITTIBenchmarkInputManifest:
    """Load and validate a KITTI benchmark input manifest."""

    return KITTIBenchmarkInputManifest.model_validate(read_mapping(Path(path)))


def verify_kitti_benchmark_input(
    manifest: KITTIBenchmarkInputManifest,
    *,
    sequence_path: str | Path | None = None,
) -> None:
    """Raise when a locked KITTI input file no longer matches its digest."""

    sequence = Path(sequence_path or manifest.source_sequence_path)
    if sequence.name != manifest.sequence_id:
        raise DatasetError(
            f"KITTI input sequence mismatch: expected {manifest.sequence_id}, got {sequence.name}"
        )
    common_root = sequence.parent
    issues: list[str] = []
    for item in manifest.files:
        path = common_root / item.path
        if not path.is_file():
            issues.append(f"missing:{item.path}")
            continue
        if path.stat().st_size != item.size_bytes:
            issues.append(f"size:{item.path}")
            continue
        if sha256_path(path) != item.sha256:
            issues.append(f"sha256:{item.path}")
    if issues:
        preview = ", ".join(issues[:8])
        remainder = len(issues) - min(len(issues), 8)
        suffix = f" (and {remainder} more)" if remainder else ""
        raise DatasetError(f"KITTI benchmark input verification failed: {preview}{suffix}")


def build_kitti_raw_0005_benchmark_input(
    sequence_path: str | Path,
    *,
    frame_ids: tuple[str, ...] = KITTI_RAW_0005_BENCHMARK_FRAME_IDS,
    command: str,
) -> KITTIBenchmarkInputManifest:
    """Inspect and digest-lock the prespecified KITTI raw benchmark inputs."""

    sequence = Path(sequence_path)
    if not sequence.is_dir():
        raise DatasetError(f"KITTI raw sequence directory does not exist: {sequence}")
    if sequence.name != KITTI_RAW_0005_SEQUENCE_ID:
        raise DatasetError(
            "KITTI benchmark requires sequence directory "
            f"{KITTI_RAW_0005_SEQUENCE_ID!r}, got {sequence.name!r}"
        )

    common_root = sequence.parent
    selected_paths: list[tuple[Path, KITTIBenchmarkInputRole, str | None]] = []
    for frame_id in frame_ids:
        selected_paths.extend(
            [
                (sequence / "image_02" / "data" / f"{frame_id}.png", "camera", frame_id),
                (
                    sequence / "velodyne_points" / "data" / f"{frame_id}.bin",
                    "lidar",
                    frame_id,
                ),
            ]
        )
    selected_paths.extend(
        [
            (sequence / "image_02" / "timestamps.txt", "timestamp", None),
            (sequence / "velodyne_points" / "timestamps.txt", "timestamp", None),
            (common_root / "calib_cam_to_cam.txt", "calibration", None),
            (common_root / "calib_velo_to_cam.txt", "calibration", None),
        ]
    )

    missing = [path for path, _, _ in selected_paths if not path.is_file()]
    if missing:
        preview = ", ".join(str(path.relative_to(common_root)) for path in missing[:8])
        remainder = len(missing) - min(len(missing), 8)
        suffix = f" (and {remainder} more)" if remainder else ""
        raise DatasetError(f"KITTI benchmark inputs are missing: {preview}{suffix}")

    empty = [path for path, _, _ in selected_paths if path.stat().st_size == 0]
    if empty:
        preview = ", ".join(str(path.relative_to(common_root)) for path in empty[:8])
        raise DatasetError(f"KITTI benchmark inputs are empty: {preview}")

    timestamp_line_counts = {
        "image_02": _timestamp_line_count(sequence / "image_02" / "timestamps.txt"),
        "velodyne_points": _timestamp_line_count(
            sequence / "velodyne_points" / "timestamps.txt"
        ),
    }
    required_timestamp_lines = max(int(frame_id) for frame_id in frame_ids) + 1
    short_streams = {
        stream: count
        for stream, count in timestamp_line_counts.items()
        if count < required_timestamp_lines
    }
    if short_streams:
        detail = ", ".join(f"{stream}={count}" for stream, count in sorted(short_streams.items()))
        raise DatasetError(
            "KITTI benchmark timestamp files do not cover selected frame "
            f"{frame_ids[-1]}: {detail}; require at least {required_timestamp_lines} lines"
        )

    files = [
        _input_file(path, role=role, frame_id=frame_id, common_root=common_root)
        for path, role, frame_id in selected_paths
    ]
    return KITTIBenchmarkInputManifest(
        source_sequence_path=str(sequence),
        frame_ids=list(frame_ids),
        timestamp_line_counts=timestamp_line_counts,
        calibration_files=["calib_cam_to_cam.txt", "calib_velo_to_cam.txt"],
        files=files,
        input_sha256=_input_set_sha256(files),
        provenance=KITTIBenchmarkInputProvenance(
            git_commit=git_commit(),
            command=command,
        ),
    )


def _input_file(
    path: Path,
    *,
    role: KITTIBenchmarkInputRole,
    frame_id: str | None,
    common_root: Path,
) -> KITTIBenchmarkInputFile:
    digest = sha256_path(path)
    if digest is None:
        raise DatasetError(f"could not hash KITTI benchmark input: {path}")
    return KITTIBenchmarkInputFile(
        path=path.relative_to(common_root).as_posix(),
        role=role,
        frame_id=frame_id,
        sha256=digest,
        size_bytes=path.stat().st_size,
    )


def _input_set_sha256(files: list[KITTIBenchmarkInputFile]) -> str:
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda value: value.path):
        digest.update(item.path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.sha256.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(item.size_bytes).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _timestamp_line_count(path: Path) -> int:
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except UnicodeDecodeError as exc:
        raise DatasetError(f"KITTI timestamp file is not valid UTF-8: {path}") from exc
