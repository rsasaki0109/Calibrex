import json
import struct
import zlib
from pathlib import Path

import pytest

from calibrex.cli.main import main
from calibrex.core.report_artifacts import (
    ReportDegeneracyArtifact,
    ReportMetricsArtifact,
    ReportObservabilityArtifact,
    ReportSummaryArtifact,
)
from calibrex.core.result import load_result


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


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_minimal_nuscenes_fixture(root: Path) -> None:
    version = root / "v1.0-mini"
    version.mkdir(parents=True)
    for channel in ["LIDAR_TOP", "CAM_FRONT", "RADAR_FRONT"]:
        (root / "samples" / channel).mkdir(parents=True)
    (root / "samples" / "LIDAR_TOP" / "000.bin").write_bytes(b"lidar")
    (root / "samples" / "CAM_FRONT" / "000.jpg").write_bytes(b"camera")
    (root / "samples" / "RADAR_FRONT" / "000.pcd").write_bytes(b"radar")
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
                "camera_intrinsic": [
                    [1000.0, 0.0, 800.0],
                    [0.0, 1000.0, 450.0],
                    [0.0, 0.0, 1.0],
                ],
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


def test_calibrate_dry_run() -> None:
    assert main(["calibrate", "examples/configs/minimal.yaml", "--dry-run", "--json"]) == 0


def test_validate_command_detects_schema_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate", "examples/configs/minimal.yaml", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)

    assert payload == {
        "kind": "config",
        "path": "examples/configs/minimal.yaml",
        "schema_version": "calibrex.config/v0.1",
        "valid": True,
    }


def test_calibrate_json_includes_report_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "calibrate",
                "examples/configs/minimal.yaml",
                "--output-dir",
                str(tmp_path),
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)

    assert payload["report"] == str(tmp_path / "report.html")
    assert payload["report_artifacts"] == {
        "html_report": str(tmp_path / "report.html"),
        "summary": str(tmp_path / "summary.json"),
        "metrics": str(tmp_path / "metrics.json"),
        "observability": str(tmp_path / "observability.json"),
        "degeneracy": str(tmp_path / "degeneracy.json"),
    }
    for path in payload["report_artifacts"].values():
        assert Path(path).exists()


def test_calibrate_evaluate_visualize_export(tmp_path: Path) -> None:
    assert (
        main(
            [
                "calibrate",
                "examples/configs/minimal.yaml",
                "--output-dir",
                str(tmp_path),
                "--json",
            ]
        )
        == 0
    )
    result = tmp_path / "result.yaml"
    assert result.exists()
    assert (tmp_path / "report.html").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "observability.json").exists()
    assert (tmp_path / "degeneracy.json").exists()
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    ReportSummaryArtifact.model_validate(summary)
    assert summary["schema_version"] == "calibrex.report.summary/v0.1"
    assert summary["run"]["id"]
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    ReportMetricsArtifact.model_validate(metrics)
    assert metrics["schema_version"] == "calibrex.report.metrics/v0.1"
    assert "schema_validation" in metrics["metrics"]
    assert "common" in metrics["metric_families"]
    assert "schema_validation" in metrics["metric_families"]["common"]["metrics"]
    assert metrics["metric_families"]["common"]["metric_count"] >= 1
    observability = json.loads((tmp_path / "observability.json").read_text(encoding="utf-8"))
    ReportObservabilityArtifact.model_validate(observability)
    degeneracy = json.loads((tmp_path / "degeneracy.json").read_text(encoding="utf-8"))
    ReportDegeneracyArtifact.model_validate(degeneracy)
    assert main(["validate", str(result)]) == 0
    assert main(["validate", str(tmp_path / "summary.json")]) == 0
    assert main(["validate", str(tmp_path / "metrics.json")]) == 0
    assert main(["validate", str(tmp_path / "observability.json")]) == 0
    assert main(["validate", str(tmp_path / "degeneracy.json")]) == 0
    assert (
        main(["validate", str(tmp_path / "summary.json"), "--kind", "report-summary"])
        == 0
    )
    evaluated_dir = tmp_path / "evaluated"
    assert (
        main(
            [
                "evaluate",
                str(result),
                "--output-dir",
                str(evaluated_dir),
                "--export-html",
                "--json",
            ]
        )
        == 0
    )
    assert (evaluated_dir / "result.yaml").exists()
    assert (evaluated_dir / "report.html").exists()
    assert (evaluated_dir / "summary.json").exists()
    assert (evaluated_dir / "metrics.json").exists()
    assert (evaluated_dir / "observability.json").exists()
    assert (evaluated_dir / "degeneracy.json").exists()
    visualized_dir = tmp_path / "visualized"
    assert (
        main(
            [
                "visualize",
                str(result),
                "--output-dir",
                str(visualized_dir),
                "--export-html",
                "--json",
            ]
        )
        == 0
    )
    assert (visualized_dir / "report.html").exists()
    assert (visualized_dir / "summary.json").exists()
    assert (visualized_dir / "metrics.json").exists()
    assert (visualized_dir / "observability.json").exists()
    assert (visualized_dir / "degeneracy.json").exists()
    assert (
        main(["export", str(result), "--format", "ros-tf", "--output", str(tmp_path / "tf.yaml")])
        == 0
    )
    assert (tmp_path / "tf.yaml").exists()


def test_compare_command_writes_machine_readable_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    left_result = Path("examples/precomputed/result.yaml")
    right = load_result(left_result)
    right.run.id = "candidate_variant"
    right.metrics["lidar_point_to_plane_rmse_m"].holdout = 0.031
    right.transforms["T_base_lidar0"].translation_m = [0.05, 0.0, 0.4]
    right_result = tmp_path / "candidate_result.yaml"
    right.save(right_result)
    output = tmp_path / "comparison.json"

    assert (
        main(
            [
                "compare",
                str(left_result),
                str(right_result),
                "--output",
                str(output),
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert payload == written
    assert payload["schema_version"] == "calibrex.comparison/v0.1"
    assert payload["left"]["run_id"] == "precomputed_example"
    assert payload["right"]["run_id"] == "candidate_variant"
    assert payload["metrics"]["lidar_point_to_plane_rmse_m"]["winner"] == "left"
    assert payload["summary"]["max_translation_delta_m"] == 0.05
    assert payload["transform_groups"]["transforms"]["comparison_count"] == 2


def test_visualize_reference_result_writes_3d_rig_overlay(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    online = load_result("examples/precomputed/result.yaml")
    reference = load_result("examples/precomputed/result.yaml")
    reference.run.id = "reference_rig"
    reference.transforms["T_base_lidar0"].translation_m = [0.02, 0.0, 0.4]
    online_path = tmp_path / "online.yaml"
    reference_path = tmp_path / "reference.yaml"
    online.save(online_path)
    reference.save(reference_path)
    output_dir = tmp_path / "visualized"

    assert (
        main(
            [
                "visualize",
                str(online_path),
                "--reference-result",
                str(reference_path),
                "--output-dir",
                str(output_dir),
                "--export-html",
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    rig_3d_path = Path(payload["rig_3d_viewer"])
    assert rig_3d_path.exists()
    rig_3d_html = rig_3d_path.read_text(encoding="utf-8")
    assert "3D Calibration Rig View" in rig_3d_html
    assert "Reference" in rig_3d_html
    assert "Online / Estimated" in rig_3d_html
    assert "T_base_lidar0" in rig_3d_html
    report_html = (output_dir / "report.html").read_text(encoding="utf-8")
    assert "rig_3d_viewer" in report_html


def test_schema_commands(tmp_path: Path) -> None:
    config_schema = tmp_path / "config.schema.json"
    result_schema = tmp_path / "result.schema.json"
    manifest_schema = tmp_path / "dataset_manifest.schema.json"
    report_summary_schema = tmp_path / "report_summary.schema.json"
    report_metrics_schema = tmp_path / "report_metrics.schema.json"
    report_observability_schema = tmp_path / "report_observability.schema.json"
    report_degeneracy_schema = tmp_path / "report_degeneracy.schema.json"
    assert main(["schema", "config", "--output", str(config_schema)]) == 0
    assert main(["schema", "result", "--output", str(result_schema)]) == 0
    assert main(["schema", "dataset-manifest", "--output", str(manifest_schema)]) == 0
    assert main(["schema", "report-summary", "--output", str(report_summary_schema)]) == 0
    assert main(["schema", "report-metrics", "--output", str(report_metrics_schema)]) == 0
    assert (
        main(["schema", "report-observability", "--output", str(report_observability_schema)])
        == 0
    )
    assert main(["schema", "report-degeneracy", "--output", str(report_degeneracy_schema)]) == 0
    assert config_schema.exists()
    assert result_schema.exists()
    assert manifest_schema.exists()
    assert report_summary_schema.exists()
    assert report_metrics_schema.exists()
    assert report_observability_schema.exists()
    assert report_degeneracy_schema.exists()
    summary_schema = json.loads(report_summary_schema.read_text(encoding="utf-8"))
    metrics_schema = json.loads(report_metrics_schema.read_text(encoding="utf-8"))
    assert summary_schema["properties"]["schema_version"]["const"] == (
        "calibrex.report.summary/v0.1"
    )
    assert metrics_schema["properties"]["schema_version"]["const"] == (
        "calibrex.report.metrics/v0.1"
    )


def test_metrics_command() -> None:
    assert main(["metrics", "--json"]) == 0


def test_public_datasets_commands() -> None:
    assert main(["public-datasets", "list", "--json"]) == 0
    assert main(["public-datasets", "show", "tum_rgbd_freiburg1_xyz", "--json"]) == 0
    assert main(["public-datasets", "show", "a2d2_lidar_pair_sample", "--json"]) == 0
    assert main(["public-datasets", "show", "tiers_livox_lidars_cali", "--json"]) == 0
    assert (
        main(
            [
                "inspect",
                "examples/public_datasets/a2d2_lidar_pair_sample",
                "--type",
                "a2d2-lidar",
                "--json",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "inspect",
                "examples/public_datasets/kitti_raw_2011_09_26_drive_0005",
                "--type",
                "kitti-raw",
                "--json",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "calibrate",
                "examples/public_datasets/kitti_raw_2011_09_26_drive_0005/config.yaml",
                "--dry-run",
                "--json",
            ]
        )
        == 0
    )


def test_nuscenes_calibrate_imports_reference_extrinsics(tmp_path: Path) -> None:
    dataset = tmp_path / "nuscenes"
    _write_minimal_nuscenes_fixture(dataset)
    output_dir = tmp_path / "outputs"
    config = tmp_path / "nuscenes_config.yaml"
    config.write_text(
        f"""
schema_version: calibrex.config/v0.1
project:
  name: nuscenes_reference_fixture
  output_dir: {output_dir}
  domain: autonomous_driving
dataset:
  type: nuscenes
  path: {dataset}
sensors:
  lidar_top:
    type: lidar
    model: nuscenes-top-lidar
    topic: LIDAR_TOP
    fields: [x, y, z, intensity]
  cam_front:
    type: camera
    model: pinhole
    topic: CAM_FRONT
  radar_front:
    type: radar
    model: nuscenes-front-radar
    topic: RADAR_FRONT
frames:
  ego:
    root: true
  lidar_top:
    parent: ego
    transform:
      estimate: false
  cam_front:
    parent: ego
    transform:
      estimate: false
  radar_front:
    parent: ego
    transform:
      estimate: false
pipeline:
  type: multi_sensor_slac
solver:
  backend: scipy
outputs:
  result: result.yaml
  report: report.html
""".strip(),
        encoding="utf-8",
    )

    assert main(["calibrate", str(config), "--json"]) == 0

    result = load_result(output_dir / "result.yaml")
    assert result.metrics["nuscenes_reference_extrinsic_count"].value == 3.0
    assert result.metrics["extrinsic_reference_comparison_count"].value == 3.0
    assert result.metrics["extrinsic_reference_translation_delta_max_m"].value is not None
    assert result.metrics["extrinsic_reference_translation_delta_max_m"].grade == "fail"
    assert result.metrics["extrinsic_reference_rotation_delta_max_deg"].value == 0.0
    assert result.candidate_extrinsics["T_ego_lidar_top"].translation_m == [0.0, 0.0, 0.0]
    assert result.reference_extrinsics["T_ego_lidar_top"].parent == "ego"
    assert result.reference_extrinsics["T_ego_lidar_top"].child == "lidar_top"
    assert result.reference_extrinsics["T_ego_lidar_top"].translation_m == [1.0, 0.0, 2.0]
    lidar_reference_provenance = result.reference_extrinsics["T_ego_lidar_top"].provenance
    assert lidar_reference_provenance.producer == "dataset_provider"
    assert lidar_reference_provenance.execution_mode == "dataset_reference"
    assert lidar_reference_provenance.role_in_comparison == "selected_reference"
    assert lidar_reference_provenance.evidence_level == "dataset_provided"
    assert result.reference_extrinsics["T_ego_cam_front"].translation_m == [1.5, 0.0, 1.8]
    assert result.reference_extrinsics["T_ego_radar_front"].translation_m == [2.0, 0.0, 0.5]
    assert result.run.provenance["dataset_initialization"]["status"] == "loaded"
    report_html = (output_dir / "report.html").read_text(encoding="utf-8")
    assert "Candidate Extrinsics" in report_html
    assert "Reference Extrinsics" in report_html
    assert "T_ego_lidar_top" in report_html

    candidate_path = tmp_path / "candidate_extrinsics.yaml"
    candidate_path.write_text(
        """
candidate_extrinsics:
  T_ego_lidar_top:
    convention: T_parent_child
    parent: ego
    child: lidar_top
    translation_m: [1.0, 0.0, 2.0]
    rotation_quat_xyzw: [0.0, 0.0, 0.0, 1.0]
  T_ego_cam_front:
    convention: T_parent_child
    parent: ego
    child: cam_front
    translation_m: [1.5, 0.0, 1.8]
    rotation_quat_xyzw: [0.0, 0.0, 0.0, 1.0]
  T_ego_radar_front:
    convention: T_parent_child
    parent: ego
    child: radar_front
    translation_m: [2.0, 0.0, 0.5]
    rotation_quat_xyzw: [0.0, 0.0, 0.0, 1.0]
""".strip(),
        encoding="utf-8",
    )
    candidate_output_dir = tmp_path / "candidate_outputs"

    assert (
        main(
            [
                "calibrate",
                str(config),
                "--dry-run",
                "--candidate-extrinsics",
                str(candidate_path),
                "--json",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "calibrate",
                str(config),
                "--output-dir",
                str(candidate_output_dir),
                "--candidate-extrinsics",
                str(candidate_path),
                "--json",
            ]
        )
        == 0
    )

    candidate_result = load_result(candidate_output_dir / "result.yaml")
    assert candidate_result.metrics["candidate_extrinsic_import_count"].value == 3.0
    assert candidate_result.metrics["extrinsic_reference_translation_delta_max_m"].value == 0.0
    assert candidate_result.metrics["extrinsic_reference_translation_delta_max_m"].grade == "pass"
    assert candidate_result.metrics["extrinsic_reference_rotation_delta_max_deg"].value == 0.0
    assert candidate_result.run.provenance["external_candidate_extrinsics"]["imported_count"] == 3
    assert candidate_result.candidate_extrinsics["T_ego_lidar_top"].translation_m == [
        1.0,
        0.0,
        2.0,
    ]
    imported_candidate_provenance = candidate_result.candidate_extrinsics[
        "T_ego_lidar_top"
    ].provenance
    assert imported_candidate_provenance.execution_mode == "imported"
    assert imported_candidate_provenance.role_in_comparison == "candidate"
    assert imported_candidate_provenance.evidence_level == (
        "imported_without_documented_derivation"
    )
    assert imported_candidate_provenance.source_path == str(candidate_path)


def test_kitti_inspect_human_output_includes_lidar_diagnostics(
    tmp_path: Path,
    capsys,
) -> None:
    sequence = tmp_path / "2011_09_26_drive_0005_sync"
    velodyne_dir = sequence / "velodyne_points" / "data"
    velodyne_dir.mkdir(parents=True)
    _write_velodyne_points(
        velodyne_dir / "0000000000.bin",
        [(float(x), float(y), 0.0, 1.0) for x in range(3) for y in range(3)],
    )

    assert main(["inspect", str(sequence), "--type", "kitti-raw"]) == 0
    output = capsys.readouterr().out
    assert "Dataset" in output
    assert "Velodyne" not in output
    assert "velodyne_points:" in output
    assert "LiDAR Quality" in output
    assert "lidar_point_to_plane_rmse_m" in output
    assert "recommendations:" in output


def test_kitti_import_calib_command(tmp_path: Path) -> None:
    (tmp_path / "calib_velo_to_cam.txt").write_text("R: 1 0 0 0 1 0 0 0 1\nT: 1 2 3\n")
    output = tmp_path / "transforms.yaml"
    assert main(["kitti", "import-calib", str(tmp_path), "--output", str(output)]) == 0
    assert output.exists()
    assert main(["kitti", "import-calib", str(tmp_path), "--json"]) == 0


def test_kitti_fixed_lidar_calibrate_result_includes_public_dataset_diagnostics(
    tmp_path: Path,
) -> None:
    sequence = tmp_path / "2011_09_26_drive_0005_sync"
    for directory in [
        sequence / "image_02" / "data",
        sequence / "velodyne_points" / "data",
        sequence / "oxts" / "data",
    ]:
        directory.mkdir(parents=True)

    kitti_points = [
        *((float(x), float(y), 5.0, 1.0) for x in range(3) for y in range(3)),
        *((0.0, float(y), 8.0, 1.0) for y in range(3)),
    ]
    _write_png(sequence / "image_02" / "data" / "0000000000.png", width=20, height=20)
    _write_png(sequence / "image_02" / "data" / "0000000001.png", width=20, height=20)
    _write_velodyne_points(sequence / "velodyne_points" / "data" / "0000000000.bin", kitti_points)
    _write_velodyne_points(sequence / "velodyne_points" / "data" / "0000000001.bin", kitti_points)
    _write_oxts_packet(sequence / "oxts" / "data" / "0000000000.txt", yaw_rad=0.0, vn=2.0, ve=0.0)
    _write_oxts_packet(sequence / "oxts" / "data" / "0000000001.txt", yaw_rad=0.2, vn=4.0, ve=1.0)
    (sequence / "image_02" / "timestamps.txt").write_text(
        "2011-09-26 13:00:00.000000000\n2011-09-26 13:00:01.000000000\n",
        encoding="utf-8",
    )
    (sequence / "velodyne_points" / "timestamps.txt").write_text(
        "2011-09-26 13:00:00.005000000\n2011-09-26 13:00:01.005000000\n",
        encoding="utf-8",
    )
    (sequence / "oxts" / "timestamps.txt").write_text(
        "2011-09-26 13:00:00.000000000\n2011-09-26 13:00:02.000000000\n",
        encoding="utf-8",
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
    external_result = tmp_path / "koide_result.yaml"
    external_result.write_text(
        """
transforms:
  T_camera0_lidar0:
    convention: T_parent_child
    parent: camera0
    child: lidar0
    translation_m: [4.0, 5.0, 6.0]
    rotation_quat_xyzw: [0.0, 0.0, 0.0, 1.0]
""".strip(),
        encoding="utf-8",
    )

    config = tmp_path / "config.yaml"
    output_dir = tmp_path / "outputs"
    config.write_text(
        f"""
schema_version: calibrex.config/v0.1
project:
  name: kitti_fixed_lidar_fixture
  output_dir: {output_dir}
  domain: autonomous_driving
dataset:
  type: kitti_raw
  path: {sequence}
sensors:
  camera0:
    type: camera
    model: pinhole
    topic: image_02
  lidar0:
    type: lidar
    model: velodyne_hdl64e
    topic: velodyne_points
    fields: [x, y, z, intensity]
  imu0:
    type: imu
    model: oxts
    topic: oxts
frames:
  base_link:
    root: true
  camera0:
    parent: base_link
    transform:
      estimate: true
      initial:
        translation: [0.0, 0.0, 0.0]
        rotation_quat_xyzw: [0.0, 0.0, 0.0, 1.0]
  lidar0:
    parent: base_link
    transform:
      estimate: true
      initial:
        translation: [0.0, 0.0, 0.0]
        rotation_quat_xyzw: [0.0, 0.0, 0.0, 1.0]
  imu0:
    parent: base_link
    transform:
      estimate: false
pipeline:
  type: multi_sensor_slac
  factors:
    lidar_point_to_surfel:
      enabled: true
    lidar_camera_mutual_information:
      enabled: true
    koide_lidar_camera:
      enabled: true
      options:
        result_path: {external_result}
    fixed_lidar_mount_prior:
      enabled: true
solver:
  backend: scipy
evaluation:
  kitti:
    max_projection_pairs: 2
    projection_sample_points: 800
    perturbation_rotation_deg: [1.0]
    perturbation_translation_m: [0.10]
  metrics:
    - lidar_point_to_plane_rmse_m
    - vehicle_yaw_excitation_deg
    - camera_lidar_timestamp_alignment_ms
    - koide_lidar_camera_result_available
outputs:
  result: result.yaml
  report: report.html
""".strip(),
        encoding="utf-8",
    )

    assert main(["calibrate", str(config), "--json"]) == 0

    result = load_result(output_dir / "result.yaml")
    assert result.transforms["T_base_link_lidar0"].translation_m == [4.0, 5.0, 6.0]
    output_provenance = result.transforms["T_base_link_lidar0"].provenance
    assert output_provenance.producer == "external_tool"
    assert output_provenance.execution_mode == "imported"
    assert output_provenance.role_in_comparison == "output"
    assert output_provenance.evidence_level == "algorithmically_refined"
    assert output_provenance.tool_name == "koide_lidar_camera"
    assert "lidar_point_to_plane_rmse_m" in result.metrics
    assert "lidar_world_map_point_to_plane_rmse_m" in result.metrics
    assert "lidar_world_map_point_to_plane_p95_holdout_m" in result.metrics
    assert result.metrics["lidar_world_map_leakage_issue_count"].value == 0.0
    assert result.metrics["lidar_world_map_leakage_issue_count"].grade == "pass"
    assert result.metrics["lidar_world_map_stability_window_count"].value == 1.0
    assert result.metrics["lidar_world_map_stability_scored_window_count"].value == 1.0
    assert result.metrics["lidar_world_map_stability_holdout_rmse_spread_m"].value == 0.0
    assert result.metrics["lidar_world_map_train_voxel_count"].value == 1.0
    assert result.metrics["lidar_world_map_holdout_residual_count"].value is not None
    assert result.metrics["lidar_world_map_perturbation_case_count"].value == 12.0
    assert result.metrics["lidar_world_map_perturbation_detectable_fraction"].value is not None
    assert result.metrics["lidar_world_map_perturbation_train_rmse_delta_mean_m"].value is not None
    world_map_holdout_delta = result.metrics[
        "lidar_world_map_perturbation_holdout_rmse_delta_mean_m"
    ]
    assert world_map_holdout_delta.value is not None
    assert result.metrics["lidar_world_map_perturbation_holdout_rmse_delta_max_m"].value is not None
    assert result.metrics["lidar_world_map_perturbation_p95_holdout_delta_mean_m"].value is not None
    assert result.metrics["lidar_world_map_weak_dof_count"].value is not None
    assert result.metrics["lidar_world_map_min_dof_sensitivity_m"].value is not None
    assert result.metrics["lidar_world_map_sensitivity_roll_m"].value is not None
    assert result.metrics["lidar_world_map_sensitivity_z_m"].value is not None
    assert any(direction.endswith("_lidar0") for direction in result.observability.weak_directions)
    assert "weak LiDAR world-map DoF" in (result.degeneracy.reason or "")
    assert result.metrics["lidar_perturbation_case_count"].value == 12.0
    assert result.metrics["lidar_perturbation_detectable_fraction"].value is not None
    assert result.metrics["lidar_perturbation_train_rmse_delta_mean_m"].value is not None
    assert result.metrics["lidar_perturbation_holdout_rmse_delta_mean_m"].value is not None
    assert result.metrics["lidar_perturbation_holdout_rmse_delta_max_m"].value is not None
    assert "vehicle_yaw_excitation_deg" in result.metrics
    assert "camera_lidar_timestamp_alignment_ms" in result.metrics
    assert "lidar_camera_transform_pairs" in result.metrics
    assert "lidar_camera_overlay_readiness" in result.metrics
    assert "lidar_camera_overlay_score" in result.metrics
    assert result.metrics["lidar_camera_projection_frame_count"].value == 2.0
    assert result.metrics["lidar_camera_projected_points"].train == 12.0
    assert result.metrics["lidar_camera_projected_points"].holdout == 12.0
    assert result.metrics["lidar_camera_projection_ratio"].train == 1.0
    assert result.metrics["lidar_camera_projection_ratio"].holdout == 1.0
    assert result.metrics["lidar_camera_projection_depth_median_m"].train == 5.0
    assert result.metrics["lidar_camera_projection_depth_median_m"].holdout == 5.0
    assert result.metrics["lidar_camera_projection_depth_span_m"].train == 3.0
    assert result.metrics["lidar_camera_projection_depth_span_m"].holdout == 3.0
    assert result.metrics["lidar_camera_projection_horizontal_coverage"].train == 0.2
    assert result.metrics["lidar_camera_projection_horizontal_coverage"].holdout == 0.2
    assert result.metrics["lidar_camera_projection_vertical_coverage"].train == 0.2
    assert result.metrics["lidar_camera_projection_vertical_coverage"].holdout == 0.2
    assert (result.metrics["lidar_camera_edge_alignment_score"].train or 0.0) > 0.4
    assert (result.metrics["lidar_camera_edge_alignment_score"].holdout or 0.0) > 0.4
    assert (result.metrics["lidar_camera_edge_gradient_mean"].train or 0.0) > 100.0
    assert (result.metrics["lidar_camera_edge_gradient_mean"].holdout or 0.0) > 100.0
    assert result.metrics["lidar_camera_depth_discontinuity_points"].train == 12.0
    assert result.metrics["lidar_camera_depth_discontinuity_points"].holdout == 12.0
    assert (result.metrics["lidar_camera_depth_edge_alignment_score"].train or 0.0) >= 0.5
    assert (result.metrics["lidar_camera_depth_edge_alignment_score"].holdout or 0.0) >= 0.5
    assert (result.metrics["lidar_camera_depth_edge_gradient_mean"].train or 0.0) > 100.0
    assert (result.metrics["lidar_camera_depth_edge_gradient_mean"].holdout or 0.0) > 100.0
    assert result.metrics["lidar_camera_perturbation_case_count"].value == 24.0
    assert result.metrics["lidar_camera_perturbation_detectable_fraction"].value is not None
    assert result.metrics["lidar_camera_perturbation_edge_delta_mean"].value is not None
    assert result.metrics["lidar_camera_perturbation_depth_edge_delta_mean"].value is not None
    assert result.metrics["lidar_camera_perturbation_projection_ratio_delta_mean"].value is not None
    assert "lidar_camera_mutual_information_score" in result.metrics
    assert result.metrics["koide_lidar_camera_result_available"].grade == "pass"
    assert result.run.provenance["dataset_initialization"]["applied_to"] == "T_base_link_lidar0"
    assert result.run.provenance["solver_adapter"] == "koide_lidar_camera"
    assert result.run.provenance["solver_adapter_applied_transforms"] == ["T_base_link_lidar0"]
    assert result.artifacts.camera_lidar_overlay is not None
    assert result.artifacts.rig_3d_viewer is not None
    overlay_path = Path(result.artifacts.camera_lidar_overlay)
    rig_3d_path = Path(result.artifacts.rig_3d_viewer)
    assert overlay_path.exists()
    assert rig_3d_path.exists()
    overlay_html = overlay_path.read_text(encoding="utf-8")
    rig_3d_html = rig_3d_path.read_text(encoding="utf-8")
    assert "Camera-LiDAR Overlay" in overlay_html
    assert "Projected Overlay" in overlay_html
    assert "Projected 12 of 12" in overlay_html
    assert "Frame Pairs" in overlay_html
    assert "0000000000.bin" in overlay_html
    assert "3D Calibration Rig View" in rig_3d_html
    assert "Online / Estimated" in rig_3d_html
    report_html = (output_dir / "report.html").read_text(encoding="utf-8")
    assert "camera_lidar_overlay" in report_html
    assert "rig_3d_viewer" in report_html
    assert "Observability" in report_html
    assert "LiDAR World-Map Diagnostics" in report_html
    assert "DoF Sensitivity" in report_html
    assert "lidar_world_map_sensitivity_roll_m" in report_html
    assert "lidar_world_map_weak_dof_count" in report_html
    world_map_metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "lidar_world_map_point_to_plane_rmse_m" in world_map_metrics["metrics"]
    observability = json.loads((output_dir / "observability.json").read_text(encoding="utf-8"))
    assert "weak_directions" in observability
    degeneracy = json.loads((output_dir / "degeneracy.json").read_text(encoding="utf-8"))
    assert degeneracy["degeneracy"]["grade"] in {"pass", "warn", "fail"}
    inspection = result.run.provenance["dataset_inspection"]
    assert inspection["diagnostics"]["velodyne_points"]["sampled_point_count"] == 24
    world_map_lineage = inspection["diagnostics"]["lidar_world_map_consistency"]
    assert world_map_lineage["split_policy"] == "temporal_tail_holdout"
    assert world_map_lineage["train_frame_ids"] == ["0000000000"]
    assert world_map_lineage["holdout_frame_ids"] == ["0000000001"]
    assert world_map_lineage["map_artifact"]["kind"] == "map"
    assert world_map_lineage["correspondence_artifact"]["kind"] == "correspondence"
    assert world_map_lineage["leakage_validation"]["status"] == "pass"
    assert world_map_lineage["stability"]["status"] == "limited"
    assert world_map_lineage["stability"]["window_count"] == 1
    assert inspection["diagnostics"]["oxts_motion"]["packet_count"] == 2
    assert inspection["diagnostics"]["timestamp_alignment"]["camera_lidar_pair_count"] == 2
    assert inspection["diagnostics"]["camera_lidar_pairs"][0]["delta_ms"] == 5.0
    assert inspection["diagnostics"]["camera_lidar_pairs"][0]["lidar_point_count"] == 12
    assert inspection["diagnostics"]["camera_lidar_pairs"][1]["delta_ms"] == 5.0
    assert inspection["diagnostics"]["camera_lidar_pairs"][1]["lidar_point_count"] == 12
    assert result.quality.recommendation


def test_compile_command(tmp_path: Path) -> None:
    output = tmp_path / "problem.yaml"
    assert main(["compile", "examples/configs/minimal.yaml", "--output", str(output)]) == 0
    assert output.exists()
    assert main(["compile", "examples/configs/minimal.yaml", "--json"]) == 0


def test_open3d_slac_example_calibrates(tmp_path: Path) -> None:
    assert (
        main(
            [
                "calibrate",
                "examples/rgbd_open3d_slac/config.yaml",
                "--output-dir",
                str(tmp_path),
                "--json",
            ]
        )
        == 0
    )
    assert (tmp_path / "result.yaml").exists()
    assert (tmp_path / "report.html").exists()


def test_public_tum_rgbd_config_calibrates(tmp_path: Path) -> None:
    assert (
        main(
            [
                "calibrate",
                "examples/public_datasets/tum_rgbd_freiburg1_xyz/config.yaml",
                "--output-dir",
                str(tmp_path),
                "--json",
            ]
        )
        == 0
    )
    assert (tmp_path / "result.yaml").exists()
