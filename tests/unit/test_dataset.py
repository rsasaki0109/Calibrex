import json
import struct
import zipfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from calibrex.core.config import DatasetConfig
from calibrex.core.geometry import SE3
from calibrex.core.time import (
    TimestampNormalizer,
    apply_time_offset_ns,
    nearest_timestamp_pairs,
)
from calibrex.data.a2d2 import A2D2LidarDataset, summarize_a2d2_lidar_npz
from calibrex.data.inspect import inspect_dataset
from calibrex.data.kitti import (
    KITTIRawDataset,
    read_calibration_file,
    read_kitti_initial_transforms,
)
from calibrex.data.livox import (
    LivoxPCDDataset,
    read_livox_binary_pcd,
    summarize_livox_pair_point_to_plane,
    summarize_livox_pcd,
)
from calibrex.data.manifest import load_manifest
from calibrex.data.nuscenes import NuScenesDataset, read_nuscenes_reference_extrinsics
from calibrex.data.tum_rgbd import (
    TUMRGBDDataset,
    associate_rgb_depth,
    read_image_index,
    write_associations,
)


def test_load_dataset_manifest() -> None:
    manifest = load_manifest("examples/synthetic_camera_lidar_imu")
    assert manifest.schema_version == "slac.dataset_manifest/v0.1"
    assert manifest.streams["lidar0"].fields == ["x", "y", "z", "intensity"]


def test_inspect_filesystem_uses_manifest() -> None:
    inspection = inspect_dataset(
        DatasetConfig(type="filesystem", path="examples/synthetic_camera_lidar_imu")
    )
    assert inspection.manifest == "examples/synthetic_camera_lidar_imu/manifest.yaml"
    assert {stream.name for stream in inspection.streams} == {"camera0", "imu0", "lidar0"}
    assert inspection.as_dict()["manifest"] == "examples/synthetic_camera_lidar_imu/manifest.yaml"


def test_timestamp_normalization_and_offset() -> None:
    assert TimestampNormalizer("sensor_time_sec").to_nanoseconds(1.25) == 1_250_000_000
    assert TimestampNormalizer("sensor_time_ms").to_nanoseconds(12) == 12_000_000
    assert apply_time_offset_ns(10_000, 0.001) == 1_010_000


def test_nearest_timestamp_pairs() -> None:
    pairs = nearest_timestamp_pairs(
        reference_timestamps_ns=[100, 200, 300],
        candidate_timestamps_ns=[95, 210, 500],
        tolerance_ns=15,
    )
    assert pairs == [(100, 95), (200, 210)]


def test_missing_filesystem_dataset_inspection(tmp_path: Path) -> None:
    inspection = inspect_dataset(DatasetConfig(type="filesystem", path=str(tmp_path / "missing")))
    assert not inspection.exists
    assert inspection.warnings == ["dataset path does not exist"]


def test_tum_rgbd_reader_parses_indices_and_associations(tmp_path: Path) -> None:
    (tmp_path / "rgb").mkdir()
    (tmp_path / "depth").mkdir()
    (tmp_path / "rgb.txt").write_text("1.000000 rgb/0001.png\n1.100000 rgb/0002.png\n")
    (tmp_path / "depth.txt").write_text("1.010000 depth/0001.png\n")
    (tmp_path / "groundtruth.txt").write_text("1.000000 0 0 0 0 0 0 1\n")
    rgb = read_image_index(tmp_path / "rgb.txt")
    depth = read_image_index(tmp_path / "depth.txt")
    associations = associate_rgb_depth(rgb, depth, max_difference_sec=0.02)
    write_associations(tmp_path / "associations.txt", associations)

    dataset = TUMRGBDDataset(tmp_path)
    streams = {stream.name: stream.message_count for stream in dataset.streams()}
    assert streams["rgb"] == 2
    assert streams["depth"] == 1
    assert streams["rgbd_associations"] == 1
    assert len(list(dataset.records("groundtruth"))) == 1


def test_kitti_raw_reader_counts_fixed_lidar_streams(tmp_path: Path) -> None:
    sequence = tmp_path / "2011_09_26_drive_0005_sync"
    for directory in [
        sequence / "image_02" / "data",
        sequence / "image_03" / "data",
        sequence / "velodyne_points" / "data",
        sequence / "oxts" / "data",
    ]:
        directory.mkdir(parents=True)
    (sequence / "image_02" / "data" / "0000000000.png").write_text("")
    (sequence / "image_03" / "data" / "0000000000.png").write_text("")
    (sequence / "velodyne_points" / "data" / "0000000000.bin").write_bytes(b"\x00" * 16)
    (sequence / "oxts" / "data" / "0000000000.txt").write_text("0 0 0\n")
    timestamp = "2011-09-26 13:02:25.123456789\n"
    (sequence / "image_02" / "timestamps.txt").write_text(timestamp)
    (sequence / "image_03" / "timestamps.txt").write_text(timestamp)
    (sequence / "velodyne_points" / "timestamps.txt").write_text(timestamp)
    (sequence / "oxts" / "timestamps.txt").write_text(timestamp)
    (tmp_path / "calib_velo_to_cam.txt").write_text("R: 1 0 0 0 1 0 0 0 1\nT: 0 0 0\n")

    dataset = KITTIRawDataset(sequence)
    streams = {stream.name: stream.message_count for stream in dataset.streams()}
    assert streams["velodyne_points"] == 1
    assert streams["calibration"] == 1
    assert len(list(dataset.records("velodyne_points"))) == 1
    assert read_calibration_file(tmp_path / "calib_velo_to_cam.txt")["T"] == [0.0, 0.0, 0.0]
    transforms = read_kitti_initial_transforms(tmp_path)
    assert "T_camera0_lidar0" in transforms
    assert transforms["T_camera0_lidar0"].translation_m == (0.0, 0.0, 0.0)


def test_a2d2_lidar_npz_reader_summarizes_physical_lidars(tmp_path: Path) -> None:
    sample = tmp_path / "20180810150607_lidar_front_left_000000060.npz"
    _write_a2d2_lidar_npz(sample)

    dataset = A2D2LidarDataset(tmp_path)
    streams = {stream.name: stream for stream in dataset.streams()}
    assert streams["a2d2_lidar_npz"].kind == "pointcloud"
    assert streams["a2d2_lidar_npz"].message_count == 1
    records = list(dataset.records("a2d2_lidar_npz"))
    assert records[0].payload_path == str(sample)

    stats = summarize_a2d2_lidar_npz(tmp_path)
    assert stats.status == "scored"
    assert stats.sample_count == 1
    assert stats.total_point_count == 4
    assert stats.total_valid_count == 3
    assert stats.physical_lidar_ids == (0, 1)
    physical = {lidar.lidar_id: lidar for lidar in stats.samples[0].physical_lidars}
    assert physical[0].valid_count == 2
    assert physical[0].bounds_min_m == (1.0, 2.0, 0.0)
    assert physical[0].bounds_max_m == (2.0, 3.0, 1.0)
    assert physical[1].point_count == 2
    assert physical[1].valid_count == 1
    assert physical[1].bounds_min_m == (-1.0, 4.0, 2.0)

    inspection = inspect_dataset(DatasetConfig(type="a2d2_lidar", path=str(tmp_path)))
    assert inspection.dataset_type == "a2d2_lidar"
    assert not inspection.warnings
    diagnostics = inspection.diagnostics["a2d2_lidar"]
    assert isinstance(diagnostics, dict)
    assert diagnostics["physical_lidar_ids"] == [0, 1]


def test_livox_pcd_reader_summarizes_solid_state_pair(tmp_path: Path) -> None:
    base = tmp_path / "base_horizon_100432.pcd"
    target = tmp_path / "target_horizon_100538.pcd"
    _write_livox_binary_pcd(
        base,
        [
            (0.1, 0.1, 0.0, 10.0),
            (0.4, 0.2, 0.0, 20.0),
            (1.2, 0.2, 0.1, 30.0),
        ],
    )
    _write_livox_binary_pcd(
        target,
        [
            (0.2, 0.1, 0.0, 11.0),
            (1.1, 0.3, 0.1, 21.0),
            (2.5, 0.0, 0.0, 31.0),
        ],
    )

    dataset = LivoxPCDDataset(tmp_path)
    streams = {stream.name: stream for stream in dataset.streams()}
    assert streams["livox_pcd"].kind == "pointcloud"
    assert streams["livox_pcd"].message_count == 2
    records = list(dataset.records("livox_pcd"))
    assert records[0].payload_path == str(base)

    points = read_livox_binary_pcd(base)
    assert tuple(round(value, 3) for value in points[0]) == (0.1, 0.1, 0.0, 10.0)

    stats = summarize_livox_pcd(tmp_path, pair_voxel_size_m=1.0)
    assert stats.status == "scored"
    assert stats.sample_count == 2
    assert stats.sampled_point_count == 6
    assert tuple(round(value, 3) for value in stats.bounds_min_m or ()) == (0.1, 0.0, 0.0)
    assert tuple(round(value, 3) for value in stats.bounds_max_m or ()) == (2.5, 0.3, 0.1)
    assert stats.pair_source_voxel_count == 2
    assert stats.pair_target_voxel_count == 3
    assert stats.pair_shared_voxel_count == 2
    assert stats.pair_unmatched_source_voxel_count == 0
    assert stats.pair_unmatched_target_voxel_count == 1
    assert stats.pair_source_voxel_recall_in_target == 1.0
    assert stats.pair_target_voxel_recall_in_source == 2.0 / 3.0
    assert stats.pair_shared_voxel_centroid_rmse_m is not None

    shifted_stats = summarize_livox_pcd(
        tmp_path,
        pair_voxel_size_m=1.0,
        target_transform=SE3((10.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    )
    assert shifted_stats.pair_target_transform_applied
    assert shifted_stats.pair_transform_convention is not None
    assert shifted_stats.pair_shared_voxel_count == 0
    assert shifted_stats.pair_source_voxel_recall_in_target == 0.0

    inspection = inspect_dataset(DatasetConfig(type="livox_pcd", path=str(tmp_path)))
    assert inspection.dataset_type == "livox_pcd"
    assert not inspection.warnings
    diagnostics = inspection.diagnostics["livox_pcd"]
    assert isinstance(diagnostics, dict)
    assert diagnostics["pair_shared_voxel_count"] == 2
    assert diagnostics["pair_source_voxel_recall_in_target"] == 1.0


def test_livox_pair_point_to_plane_uses_source_normals(tmp_path: Path) -> None:
    base = tmp_path / "base_horizon_100432.pcd"
    target = tmp_path / "target_horizon_100538.pcd"
    normal = (0.0, 0.0, 1.0)
    _write_livox_binary_pcd_with_normals(
        base,
        [
            (0.1, 0.1, 0.0, 10.0, *normal),
            (0.4, 0.1, 0.0, 20.0, *normal),
            (0.1, 0.4, 0.0, 30.0, *normal),
        ],
    )
    _write_livox_binary_pcd_with_normals(
        target,
        [
            (0.1, 0.1, 0.1, 11.0, *normal),
            (0.4, 0.1, 0.1, 21.0, *normal),
            (0.1, 0.4, 0.1, 31.0, *normal),
        ],
    )

    stats = summarize_livox_pair_point_to_plane(
        tmp_path,
        voxel_size_m=1.0,
        correspondence_gate_m=1.0,
        inlier_threshold_m=0.2,
    )

    assert stats.status == "scored"
    assert stats.split_policy == "single_pair_source_map_target_query"
    assert stats.train_frame_ids == ("base_horizon_100432",)
    assert stats.holdout_frame_ids == ("target_horizon_100538",)
    assert not stats.independent_holdout
    assert stats.map_voxel_count == 1
    assert stats.matched_point_count == 3
    assert stats.unmatched_point_count == 0
    assert stats.unmatched_fraction == 0.0
    assert round(stats.median_abs_m or 0.0, 6) == 0.1
    assert round(stats.p90_abs_m or 0.0, 6) == 0.1
    assert round(stats.rmse_m or 0.0, 6) == 0.1
    assert stats.inlier_fraction == 1.0


def test_nuscenes_reader_inspects_sensor_metadata(tmp_path: Path) -> None:
    version = tmp_path / "v1.0-mini"
    version.mkdir()
    (tmp_path / "samples" / "LIDAR_TOP").mkdir(parents=True)
    (tmp_path / "samples" / "CAM_FRONT").mkdir(parents=True)
    (tmp_path / "samples" / "RADAR_FRONT").mkdir(parents=True)
    (tmp_path / "samples" / "LIDAR_TOP" / "000.bin").write_bytes(b"lidar")
    (tmp_path / "samples" / "CAM_FRONT" / "000.jpg").write_bytes(b"camera")
    (tmp_path / "samples" / "RADAR_FRONT" / "000.pcd").write_bytes(b"radar")
    _write_json(
        version / "sensor.json",
        [
            {"token": "sensor_lidar", "channel": "LIDAR_TOP", "modality": "lidar"},
            {"token": "sensor_camera", "channel": "CAM_FRONT", "modality": "camera"},
            {"token": "sensor_radar", "channel": "RADAR_FRONT", "modality": "radar"},
        ],
    )
    _write_json(
        version / "calibrated_sensor.json",
        [
            {
                "token": "cal_lidar",
                "sensor_token": "sensor_lidar",
                "translation": [1.0, 0.0, 2.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
            },
            {
                "token": "cal_camera",
                "sensor_token": "sensor_camera",
                "translation": [1.5, 0.0, 1.8],
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "camera_intrinsic": [[1000.0, 0.0, 800.0], [0.0, 1000.0, 450.0], [0.0, 0.0, 1.0]],
            },
            {
                "token": "cal_radar",
                "sensor_token": "sensor_radar",
                "translation": [2.0, 0.0, 0.5],
                "rotation": [1.0, 0.0, 0.0, 0.0],
            },
        ],
    )
    _write_json(version / "ego_pose.json", [{"token": "ego0"}])
    _write_json(version / "scene.json", [{"token": "scene0", "name": "scene-0001"}])
    _write_json(version / "sample.json", [{"token": "sample0", "scene_token": "scene0"}])
    _write_json(
        version / "sample_data.json",
        [
            {
                "token": "sd_lidar",
                "sample_token": "sample0",
                "ego_pose_token": "ego0",
                "calibrated_sensor_token": "cal_lidar",
                "filename": "samples/LIDAR_TOP/000.bin",
                "timestamp": 1_000_000,
                "is_key_frame": True,
            },
            {
                "token": "sd_camera",
                "sample_token": "sample0",
                "ego_pose_token": "ego0",
                "calibrated_sensor_token": "cal_camera",
                "filename": "samples/CAM_FRONT/000.jpg",
                "timestamp": 1_000_000,
                "is_key_frame": True,
            },
            {
                "token": "sd_radar",
                "sample_token": "sample0",
                "ego_pose_token": "ego0",
                "calibrated_sensor_token": "cal_radar",
                "filename": "samples/RADAR_FRONT/000.pcd",
                "timestamp": 1_000_000,
                "is_key_frame": True,
            },
        ],
    )

    dataset = NuScenesDataset(tmp_path)
    streams = {stream.name: stream for stream in dataset.streams()}
    assert streams["LIDAR_TOP"].kind == "pointcloud"
    assert streams["CAM_FRONT"].kind == "image"
    assert streams["RADAR_FRONT"].kind == "radar"
    lidar_records = list(dataset.records("LIDAR_TOP"))
    assert lidar_records[0].timestamp_ns == 1_000_000_000
    assert lidar_records[0].metadata["modality"] == "lidar"

    inspection = inspect_dataset(DatasetConfig(type="nuscenes", path=str(tmp_path)))
    assert inspection.dataset_type == "nuscenes"
    assert not inspection.warnings
    diagnostics = inspection.diagnostics["nuscenes"]
    assert isinstance(diagnostics, dict)
    assert diagnostics["scene_count"] == 1
    assert diagnostics["sample_data_count"] == 3
    assert diagnostics["modality_counts"] == {"camera": 1, "lidar": 1, "radar": 1}
    calibrated = diagnostics["calibrated_sensors"]
    assert isinstance(calibrated, dict)
    assert calibrated["LIDAR_TOP"]["rotation_quat_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    reference_extrinsics = read_nuscenes_reference_extrinsics(tmp_path)
    assert reference_extrinsics["T_ego_lidar_top"]["parent"] == "ego"
    assert reference_extrinsics["T_ego_lidar_top"]["child"] == "lidar_top"
    assert reference_extrinsics["T_ego_lidar_top"]["translation_m"] == [1.0, 0.0, 2.0]
    assert reference_extrinsics["T_ego_lidar_top"]["rotation_quat_xyzw"] == [
        0.0,
        0.0,
        0.0,
        1.0,
    ]


def test_dataset_config_sample_limit_defaults_to_none() -> None:
    dataset = DatasetConfig(type="a2d2_lidar", path="unused")
    assert dataset.sample_limit is None


def test_dataset_config_rejects_non_positive_sample_limit() -> None:
    for invalid in (0, -1):
        with pytest.raises(ValidationError):
            DatasetConfig(type="a2d2_lidar", path="unused", sample_limit=invalid)


def test_a2d2_lidar_inspection_defaults_to_three_samples(tmp_path: Path) -> None:
    for index in range(5):
        _write_a2d2_lidar_npz(tmp_path / f"20180810150607_lidar_front_left_{index:06d}.npz")

    inspection = inspect_dataset(DatasetConfig(type="a2d2_lidar", path=str(tmp_path)))
    diagnostics = inspection.diagnostics["a2d2_lidar"]
    assert isinstance(diagnostics, dict)
    assert diagnostics["sample_count"] == 5
    assert diagnostics["sampled_file_count"] == 3


def test_a2d2_lidar_inspection_honors_custom_sample_limit(tmp_path: Path) -> None:
    for index in range(5):
        _write_a2d2_lidar_npz(tmp_path / f"20180810150607_lidar_front_left_{index:06d}.npz")

    inspection = inspect_dataset(
        DatasetConfig(type="a2d2_lidar", path=str(tmp_path), sample_limit=2)
    )
    diagnostics = inspection.diagnostics["a2d2_lidar"]
    assert isinstance(diagnostics, dict)
    assert diagnostics["sample_count"] == 5
    assert diagnostics["sampled_file_count"] == 2


def test_livox_pcd_inspection_honors_custom_sample_limit(tmp_path: Path) -> None:
    for index in range(6):
        _write_livox_binary_pcd(
            tmp_path / f"frame_horizon_{index:06d}.pcd",
            [(0.1 * index, 0.1, 0.0, 10.0), (0.2 * index, 0.2, 0.0, 20.0)],
        )

    default_inspection = inspect_dataset(DatasetConfig(type="livox_pcd", path=str(tmp_path)))
    default_diagnostics = default_inspection.diagnostics["livox_pcd"]
    assert isinstance(default_diagnostics, dict)
    assert default_diagnostics["sample_count"] == 6
    assert default_diagnostics["sampled_file_count"] == 4

    custom_inspection = inspect_dataset(
        DatasetConfig(type="livox_pcd", path=str(tmp_path), sample_limit=2)
    )
    custom_diagnostics = custom_inspection.diagnostics["livox_pcd"]
    assert isinstance(custom_diagnostics, dict)
    assert custom_diagnostics["sample_count"] == 6
    assert custom_diagnostics["sampled_file_count"] == 2


def test_kitti_raw_inspection_honors_custom_sample_limit(tmp_path: Path) -> None:
    sequence = tmp_path / "2011_09_26_drive_0005_sync"
    velodyne_dir = sequence / "velodyne_points" / "data"
    velodyne_dir.mkdir(parents=True)
    for index in range(5):
        (velodyne_dir / f"{index:010d}.bin").write_bytes(
            struct.pack("<4f", 1.0, 2.0, 3.0, 0.5)
        )

    default_inspection = inspect_dataset(DatasetConfig(type="kitti_raw", path=str(sequence)))
    default_velodyne = default_inspection.diagnostics["velodyne_points"]
    assert isinstance(default_velodyne, dict)
    assert default_velodyne["frame_count"] == 5
    assert default_velodyne["sampled_frame_count"] == 3

    custom_inspection = inspect_dataset(
        DatasetConfig(type="kitti_raw", path=str(sequence), sample_limit=1)
    )
    custom_velodyne = custom_inspection.diagnostics["velodyne_points"]
    assert isinstance(custom_velodyne, dict)
    assert custom_velodyne["frame_count"] == 5
    assert custom_velodyne["sampled_frame_count"] == 1


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_a2d2_lidar_npz(path: Path) -> None:
    points = [
        1.0,
        2.0,
        0.0,
        2.0,
        3.0,
        1.0,
        -1.0,
        4.0,
        2.0,
        4.0,
        -2.0,
        1.5,
    ]
    lidar_ids = [0, 0, 1, 1]
    valid = [True, True, True, False]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        _write_npy(archive, "pcloud_points.npy", "<f8", (4, 3), points)
        _write_npy(archive, "pcloud_attr.lidar_id.npy", "<i8", (4,), lidar_ids)
        _write_npy(archive, "pcloud_attr.valid.npy", "|b1", (4,), valid)


def _write_npy(
    archive: zipfile.ZipFile,
    name: str,
    descr: str,
    shape: tuple[int, ...],
    values: list[object],
) -> None:
    header = {
        "descr": descr,
        "fortran_order": False,
        "shape": shape,
    }
    header_bytes = (repr(header) + " " * 64).encode("latin1")
    padding = 16 - ((10 + len(header_bytes) + 1) % 16)
    header_bytes = header_bytes + b" " * padding + b"\n"
    if descr == "<f8":
        body = struct.pack("<" + "d" * len(values), *(float(value) for value in values))
    elif descr == "<i8":
        body = struct.pack("<" + "q" * len(values), *(int(value) for value in values))
    elif descr == "|b1":
        body = bytes(1 if bool(value) else 0 for value in values)
    else:
        raise ValueError(descr)
    archive.writestr(
        name,
        b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header_bytes)) + header_bytes + body,
    )


def _write_livox_binary_pcd(path: Path, points: list[tuple[float, float, float, float]]) -> None:
    header = "\n".join(
        [
            "# .PCD v0.7 - Point Cloud Data file format",
            "VERSION 0.7",
            "FIELDS x y z intensity",
            "SIZE 4 4 4 4",
            "TYPE F F F F",
            "COUNT 1 1 1 1",
            f"WIDTH {len(points)}",
            "HEIGHT 1",
            "VIEWPOINT 0 0 0 1 0 0 0",
            f"POINTS {len(points)}",
            "DATA binary",
            "",
        ]
    ).encode("ascii")
    body = b"".join(struct.pack("<ffff", *point) for point in points)
    path.write_bytes(header + body)


def _write_livox_binary_pcd_with_normals(
    path: Path,
    points: list[tuple[float, float, float, float, float, float, float]],
) -> None:
    header = "\n".join(
        [
            "# .PCD v0.7 - Point Cloud Data file format",
            "VERSION 0.7",
            "FIELDS x y z intensity normal_x normal_y normal_z curvature",
            "SIZE 4 4 4 4 4 4 4 4",
            "TYPE F F F F F F F F",
            "COUNT 1 1 1 1 1 1 1 1",
            f"WIDTH {len(points)}",
            "HEIGHT 1",
            "VIEWPOINT 0 0 0 1 0 0 0",
            f"POINTS {len(points)}",
            "DATA binary",
            "",
        ]
    ).encode("ascii")
    body = b"".join(
        struct.pack("<ffffffff", x, y, z, intensity, nx, ny, nz, 0.0)
        for x, y, z, intensity, nx, ny, nz in points
    )
    path.write_bytes(header + body)
