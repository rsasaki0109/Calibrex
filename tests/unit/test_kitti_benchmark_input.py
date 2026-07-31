from __future__ import annotations

from pathlib import Path

import pytest

from calibrex.core.exceptions import DatasetError
from calibrex.data.kitti_benchmark import (
    KITTI_RAW_0005_BENCHMARK_FRAME_IDS,
    KITTI_RAW_0005_SEQUENCE_ID,
    build_kitti_raw_0005_benchmark_input,
    load_kitti_benchmark_input,
    verify_kitti_benchmark_input,
)


def _write_benchmark_inputs(root: Path) -> Path:
    sequence = root / KITTI_RAW_0005_SEQUENCE_ID
    camera_dir = sequence / "image_02" / "data"
    lidar_dir = sequence / "velodyne_points" / "data"
    camera_dir.mkdir(parents=True)
    lidar_dir.mkdir(parents=True)
    for frame_id in KITTI_RAW_0005_BENCHMARK_FRAME_IDS:
        (camera_dir / f"{frame_id}.png").write_bytes(f"image-{frame_id}".encode())
        (lidar_dir / f"{frame_id}.bin").write_bytes(f"lidar-{frame_id}".encode())
    camera_timestamps = "".join(
        f"2011-09-26 13:00:{index % 60:02d}.000000000\n" for index in range(154)
    )
    lidar_timestamps = "".join(
        f"2011-09-26 13:00:{index % 60:02d}.005000000\n" for index in range(154)
    )
    (sequence / "image_02" / "timestamps.txt").write_text(camera_timestamps, encoding="utf-8")
    (sequence / "velodyne_points" / "timestamps.txt").write_text(
        lidar_timestamps,
        encoding="utf-8",
    )
    (root / "calib_cam_to_cam.txt").write_text("P_rect_02: 1 0 0 0\n", encoding="utf-8")
    (root / "calib_velo_to_cam.txt").write_text(
        "R: 1 0 0 0 1 0 0 0 1\nT: 0 0 0\n",
        encoding="utf-8",
    )
    return sequence


def test_build_kitti_benchmark_input_locks_selected_files(tmp_path: Path) -> None:
    sequence = _write_benchmark_inputs(tmp_path)
    manifest = build_kitti_raw_0005_benchmark_input(
        sequence,
        command="calibrex kitti lock-benchmark-input ...",
    )

    assert manifest.schema_version == "slac.kitti_benchmark_input/v0.1"
    assert tuple(manifest.frame_ids) == KITTI_RAW_0005_BENCHMARK_FRAME_IDS
    assert len(manifest.files) == (2 * len(KITTI_RAW_0005_BENCHMARK_FRAME_IDS)) + 4
    assert {item.role for item in manifest.files} == {
        "camera",
        "lidar",
        "timestamp",
        "calibration",
    }
    assert len(manifest.input_sha256) == 64
    assert manifest.timestamp_line_counts == {"image_02": 154, "velodyne_points": 154}
    assert manifest.provenance.official_url.startswith("https://www.cvlibs.net/")
    assert "not redistributed" in manifest.provenance.redistribution

    output = tmp_path / "locked-input.yaml"
    manifest.save(output)
    assert load_kitti_benchmark_input(output) == manifest
    verify_kitti_benchmark_input(manifest)


def test_kitti_benchmark_input_digest_changes_with_selected_file(tmp_path: Path) -> None:
    sequence = _write_benchmark_inputs(tmp_path)
    first = build_kitti_raw_0005_benchmark_input(sequence, command="first")
    frame_id = KITTI_RAW_0005_BENCHMARK_FRAME_IDS[0]
    (sequence / "image_02" / "data" / f"{frame_id}.png").write_bytes(b"changed")
    second = build_kitti_raw_0005_benchmark_input(sequence, command="second")

    assert first.input_sha256 != second.input_sha256
    with pytest.raises(DatasetError, match="verification failed"):
        verify_kitti_benchmark_input(first)


def test_kitti_benchmark_input_reports_missing_files(tmp_path: Path) -> None:
    sequence = tmp_path / KITTI_RAW_0005_SEQUENCE_ID
    sequence.mkdir()

    with pytest.raises(DatasetError, match="KITTI benchmark inputs are missing"):
        build_kitti_raw_0005_benchmark_input(sequence, command="missing")


def test_kitti_benchmark_input_rejects_short_timestamps(tmp_path: Path) -> None:
    sequence = _write_benchmark_inputs(tmp_path)
    (sequence / "image_02" / "timestamps.txt").write_text("one line\n", encoding="utf-8")

    with pytest.raises(DatasetError, match="do not cover selected frame"):
        build_kitti_raw_0005_benchmark_input(sequence, command="short")


def test_kitti_benchmark_input_rejects_wrong_sequence(tmp_path: Path) -> None:
    sequence = tmp_path / "different_sequence"
    sequence.mkdir()

    with pytest.raises(DatasetError, match=KITTI_RAW_0005_SEQUENCE_ID):
        build_kitti_raw_0005_benchmark_input(sequence, command="wrong")
