import struct
import zlib
from pathlib import Path

from calibrex.core.config import DatasetConfig, load_config
from calibrex.core.frames import FrameGraph
from calibrex.core.geometry import SE3
from calibrex.data.inspect import inspect_dataset
from calibrex.data.kitti import (
    find_camera_lidar_pairs,
    project_velodyne_to_camera,
    read_oxts_packet,
    read_velodyne_bin,
    score_lidar_camera_depth_edge_alignment,
    score_lidar_camera_edge_alignment,
    summarize_lidar_world_map_consistency,
    summarize_oxts_motion,
    summarize_timestamp_alignment,
    summarize_velodyne_points,
)
from calibrex.data.public_datasets import load_public_dataset_catalog
from calibrex.graph.problem import build_problem


def _write_velodyne_points(path: Path, points: list[tuple[float, float, float, float]]) -> None:
    values = [value for point in points for value in point]
    path.write_bytes(struct.pack("<" + ("ffff" * len(points)), *values))


def _write_oxts_packet(path: Path, *, yaw_rad: float, vn: float, ve: float) -> None:
    fields = [
        49.0,
        8.0,
        110.0,
        0.0,
        0.0,
        yaw_rad,
        vn,
        ve,
        0.0,
        0.0,
        0.0,
        0.2,
        0.0,
        0.0,
        0.0,
    ]
    path.write_text(" ".join(str(value) for value in fields), encoding="utf-8")


def _write_png(path: Path, *, width: int, height: int) -> None:
    raw_rows = b"".join(
        b"\x00"
        + b"".join(
            b"\x00\x00\x00" if x < width // 2 else b"\xff\xff\xff"
            for x in range(width)
        )
        for _ in range(height)
    )
    compressed = zlib.compress(raw_rows)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", compressed)
        + _png_chunk(b"IEND", b"")
    )


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def test_public_dataset_catalog_lists_public_examples() -> None:
    catalog = load_public_dataset_catalog()
    assert "tum_rgbd_freiburg1_xyz" in catalog.datasets
    assert "kitti_raw_2011_09_26_drive_0005" in catalog.datasets
    assert "nuscenes_mini" in catalog.datasets
    assert "a2d2_sensor_setup" in catalog.datasets
    assert "a2d2_lidar_pair_sample" in catalog.datasets
    assert "tiers_livox_lidars_cali" in catalog.datasets
    assert catalog.datasets["tum_rgbd_freiburg1_xyz"].calibrex_config is not None
    assert catalog.datasets["nuscenes_mini"].calibrex_config is not None
    assert catalog.datasets["a2d2_sensor_setup"].family == "a2d2"
    assert catalog.datasets["a2d2_lidar_pair_sample"].family == "a2d2"
    assert catalog.datasets["tiers_livox_lidars_cali"].family == "tiers_lidars"


def test_tum_rgbd_public_config_compiles() -> None:
    config = load_config("examples/public_datasets/tum_rgbd_freiburg1_xyz/config.yaml")
    inspection = inspect_dataset(config.dataset)
    problem = build_problem(config, FrameGraph.from_config(config), inspection)
    assert inspection.manifest == "examples/public_datasets/tum_rgbd_freiburg1_xyz/manifest.yaml"
    assert problem.pipeline == "rgbd_open3d_slac"
    assert "open3d_control_grid" in {variable.name for variable in problem.variables}


def test_kitti_public_config_compiles_lidar_camera_factor() -> None:
    config = load_config("examples/public_datasets/kitti_raw_2011_09_26_drive_0005/config.yaml")
    problem = build_problem(config, FrameGraph.from_config(config), inspect_dataset(config.dataset))
    assert config.evaluation.kitti.max_projection_pairs == 20
    assert config.evaluation.kitti.projection_sample_points == 800
    assert config.evaluation.kitti.perturbation_rotation_deg == [0.5, 1.0]
    assert config.evaluation.kitti.perturbation_translation_m == [0.05, 0.10]
    assert "lidar_camera_mutual_information" in {factor.name for factor in problem.factors}
    assert "lidar_rig_point_to_plane" in {factor.name for factor in problem.factors}
    assert "koide_lidar_camera" in {factor.name for factor in problem.factors}
    assert "fixed_lidar_mount_prior" in {factor.name for factor in problem.factors}
    factor_variables = {variable for factor in problem.factors for variable in factor.variables}
    assert "T_base_link_camera0" in factor_variables
    assert "T_base_link_lidar0" in factor_variables
    assert "lidar_plane_map" in {variable.name for variable in problem.variables}


def test_nuscenes_public_config_compiles_sensor_graph() -> None:
    config = load_config("examples/public_datasets/nuscenes_mini/config.yaml")
    inspection = inspect_dataset(config.dataset)
    problem = build_problem(config, FrameGraph.from_config(config), inspection)
    assert config.dataset.type == "nuscenes"
    assert inspection.dataset_type == "nuscenes"
    assert "lidar_top" in config.sensors
    assert "radar_front" in config.sensors
    assert "fixed_lidar_mount_prior" in {factor.name for factor in problem.factors}


def test_kitti_velodyne_reader_and_inspection_stats(tmp_path: Path) -> None:
    sequence = tmp_path / "2011_09_26_drive_0005_sync"
    velodyne_dir = sequence / "velodyne_points" / "data"
    velodyne_dir.mkdir(parents=True)
    point_file = velodyne_dir / "0000000000.bin"
    _write_velodyne_points(
        point_file,
        [
            (1.0, 2.0, 3.0, 0.5),
            (-1.0, 0.0, 4.0, 0.25),
        ],
    )

    points = read_velodyne_bin(point_file)
    assert points == [(1.0, 2.0, 3.0, 0.5), (-1.0, 0.0, 4.0, 0.25)]

    stats = summarize_velodyne_points(sequence)
    assert stats.frame_count == 1
    assert stats.sampled_point_count == 2
    assert stats.bounds_min_m == (-1.0, 0.0, 3.0)
    assert stats.bounds_max_m == (1.0, 2.0, 4.0)
    assert stats.intensity_mean == 0.375

    inspection = inspect_dataset(DatasetConfig(type="kitti_raw", path=str(sequence)))
    velodyne_diagnostics = inspection.diagnostics["velodyne_points"]
    assert isinstance(velodyne_diagnostics, dict)
    assert velodyne_diagnostics["sampled_point_count"] == 2


def test_kitti_velodyne_planarity_stats(tmp_path: Path) -> None:
    sequence = tmp_path / "2011_09_26_drive_0005_sync"
    velodyne_dir = sequence / "velodyne_points" / "data"
    velodyne_dir.mkdir(parents=True)
    point_file = velodyne_dir / "0000000000.bin"
    plane_points = [
        (float(x), float(y), 0.0, 1.0)
        for x in range(3)
        for y in range(3)
    ]
    _write_velodyne_points(point_file, plane_points)

    stats = summarize_velodyne_points(sequence)
    assert stats.planarity_voxel_count == 1
    assert stats.local_planarity_mean == 1.0
    assert stats.roughness_mean_m == 0.0
    assert stats.map_sharpness_score == 1.0
    assert stats.point_to_plane_rmse_train_m == 0.0
    assert stats.point_to_plane_rmse_holdout_m == 0.0


def test_kitti_lidar_world_map_consistency_from_oxts(tmp_path: Path) -> None:
    sequence = tmp_path / "2011_09_26_drive_0005_sync"
    velodyne_dir = sequence / "velodyne_points" / "data"
    oxts_dir = sequence / "oxts" / "data"
    velodyne_dir.mkdir(parents=True)
    oxts_dir.mkdir(parents=True)
    plane_points = [
        (float(x), float(y), 0.0, 1.0)
        for x in range(3)
        for y in range(3)
    ]
    _write_velodyne_points(velodyne_dir / "0000000000.bin", plane_points)
    _write_velodyne_points(velodyne_dir / "0000000001.bin", plane_points)
    _write_oxts_packet(oxts_dir / "0000000000.txt", yaw_rad=0.0, vn=2.0, ve=0.0)
    _write_oxts_packet(oxts_dir / "0000000001.txt", yaw_rad=0.0, vn=2.0, ve=0.0)

    stats = summarize_lidar_world_map_consistency(sequence)
    assert stats.status == "scored"
    assert stats.train_frame_count == 1
    assert stats.holdout_frame_count == 1
    assert stats.split_policy == "temporal_tail_holdout"
    assert stats.train_frame_ids == ("0000000000",)
    assert stats.holdout_frame_ids == ("0000000001",)
    assert stats.dataset_slices[0]["role"] == "train"
    assert stats.dataset_slices[1]["role"] == "holdout"
    assert stats.map_artifact["map_type"] == "voxel_plane_map"
    assert stats.correspondence_artifact["correspondence_type"] == "point_to_voxel_plane"
    assert stats.leakage_validation["status"] == "pass"
    assert stats.leakage_validation["issue_count"] == 0
    assert stats.stability["status"] == "limited"
    assert stats.stability["window_count"] == 1
    assert stats.stability["scored_window_count"] == 1
    assert stats.stability["holdout_rmse_spread_m"] == 0.0
    assert stats.train_voxel_count == 1
    assert stats.holdout_residual_count == 9
    assert stats.point_to_plane_rmse_train_m == 0.0
    assert stats.point_to_plane_rmse_holdout_m == 0.0


def test_kitti_oxts_reader_and_motion_stats(tmp_path: Path) -> None:
    sequence = tmp_path / "2011_09_26_drive_0005_sync"
    oxts_dir = sequence / "oxts" / "data"
    oxts_dir.mkdir(parents=True)
    _write_oxts_packet(oxts_dir / "0000000000.txt", yaw_rad=0.0, vn=2.0, ve=0.0)
    _write_oxts_packet(oxts_dir / "0000000001.txt", yaw_rad=0.5, vn=4.0, ve=3.0)
    (sequence / "oxts" / "timestamps.txt").write_text(
        "2011-09-26 13:00:00.000000000\n2011-09-26 13:00:02.000000000\n",
        encoding="utf-8",
    )

    packet = read_oxts_packet(oxts_dir / "0000000000.txt")
    assert packet.horizontal_speed_mps == 2.0

    stats = summarize_oxts_motion(sequence)
    assert stats.packet_count == 2
    assert stats.timestamp_count == 2
    assert stats.duration_sec == 2.0
    assert stats.mean_speed_mps == 3.5
    assert stats.max_speed_mps == 5.0
    assert round(stats.yaw_excitation_deg or 0.0, 3) == 28.648


def test_kitti_timestamp_alignment_stats(tmp_path: Path) -> None:
    sequence = tmp_path / "2011_09_26_drive_0005_sync"
    image_dir = sequence / "image_02" / "data"
    velodyne_dir = sequence / "velodyne_points" / "data"
    image_dir.mkdir(parents=True)
    velodyne_dir.mkdir(parents=True)
    (sequence / "oxts").mkdir(parents=True)
    (image_dir / "0000000000.png").write_bytes(b"")
    (image_dir / "0000000001.png").write_bytes(b"")
    _write_velodyne_points(
        velodyne_dir / "0000000000.bin",
        [(1.0, 0.0, 0.0, 0.5), (2.0, 0.0, 0.0, 0.5)],
    )
    _write_velodyne_points(
        velodyne_dir / "0000000001.bin",
        [(3.0, 0.0, 0.0, 0.5)],
    )
    (sequence / "image_02" / "timestamps.txt").write_text(
        "2011-09-26 13:00:00.000000000\n2011-09-26 13:00:01.000000000\n",
        encoding="utf-8",
    )
    (sequence / "velodyne_points" / "timestamps.txt").write_text(
        "2011-09-26 13:00:00.010000000\n2011-09-26 13:00:01.020000000\n",
        encoding="utf-8",
    )
    (sequence / "oxts" / "timestamps.txt").write_text(
        "2011-09-26 13:00:00.000000000\n2011-09-26 13:00:01.050000000\n",
        encoding="utf-8",
    )

    stats = summarize_timestamp_alignment(sequence)
    assert stats.camera_lidar_pair_count == 2
    assert stats.camera_lidar_mean_abs_dt_ms == 15.0
    assert stats.camera_lidar_max_abs_dt_ms == 20.0
    assert stats.lidar_oxts_pair_count == 2
    assert stats.lidar_oxts_max_abs_dt_ms == 30.0

    pairs = find_camera_lidar_pairs(sequence)
    assert len(pairs) == 2
    assert pairs[0].camera_index == 0
    assert pairs[0].lidar_index == 0
    assert pairs[0].delta_ms == 10.0
    assert pairs[0].lidar_point_count == 2
    assert pairs[1].delta_ms == 20.0


def test_kitti_lidar_points_project_into_camera_image(tmp_path: Path) -> None:
    sequence = tmp_path / "2011_09_26_drive_0005_sync"
    image_dir = sequence / "image_02" / "data"
    velodyne_dir = sequence / "velodyne_points" / "data"
    image_dir.mkdir(parents=True)
    velodyne_dir.mkdir(parents=True)
    camera_file = image_dir / "0000000000.png"
    lidar_file = velodyne_dir / "0000000000.bin"
    _write_png(camera_file, width=20, height=20)
    _write_velodyne_points(
        lidar_file,
        [
            (0.0, 0.0, 5.0, 1.0),
            (0.0, 0.0, 8.0, 1.0),
            (1.0, 1.0, 5.0, 0.5),
            (10.0, 0.0, 5.0, 0.5),
        ],
    )
    (tmp_path / "calib_velo_to_cam.txt").write_text(
        "R: 1 0 0 0 1 0 0 0 1\nT: 0 0 0\n",
        encoding="utf-8",
    )
    (tmp_path / "calib_cam_to_cam.txt").write_text(
        """
S_rect_02: 20 20
R_rect_00: 1 0 0 0 1 0 0 0 1
P_rect_02: 10 0 10 0 0 10 10 0 0 0 1 0
""".strip(),
        encoding="utf-8",
    )

    projection = project_velodyne_to_camera(
        sequence,
        camera_path=camera_file,
        lidar_path=lidar_file,
    )
    assert projection.status == "projected"
    assert projection.image_size_px == (20, 20)
    assert projection.sampled_point_count == 4
    assert projection.projected_point_count == 3
    assert projection.points[0].u_px == 10.0
    assert projection.points[0].v_px == 10.0

    edge_alignment = score_lidar_camera_edge_alignment(projection)
    assert edge_alignment.status == "scored"
    assert edge_alignment.scored_point_count == 3
    assert edge_alignment.edge_point_count == 2
    assert edge_alignment.edge_fraction == 2.0 / 3.0

    depth_edge_alignment = score_lidar_camera_depth_edge_alignment(projection)
    assert depth_edge_alignment.status == "scored"
    assert depth_edge_alignment.discontinuity_point_count == 3
    assert depth_edge_alignment.edge_point_count == 2
    assert depth_edge_alignment.edge_fraction == 2.0 / 3.0

    shifted_projection = project_velodyne_to_camera(
        sequence,
        camera_path=camera_file,
        lidar_path=lidar_file,
        t_camera_lidar_override=SE3((100.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    )
    assert shifted_projection.status == "empty"
    assert shifted_projection.projected_point_count == 0
