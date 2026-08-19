from pathlib import Path

import jsonschema

from calibrex.diagnostics import (
    build_doctor_artifact,
    doctor_json_schema,
    infer_dataset_type,
)


def test_infer_dataset_type_from_suffix_and_declared_config(tmp_path: Path) -> None:
    assert infer_dataset_type(tmp_path / "recording.mcap") == "mcap"
    assert infer_dataset_type(tmp_path / "recording.bag") == "rosbag1"

    dataset = tmp_path / "declared"
    dataset.mkdir()
    (dataset / "config.yaml").write_text(
        """
schema_version: slac.config/v0.1
dataset:
  type: kitti_raw
  path: .
""".strip(),
        encoding="utf-8",
    )
    assert infer_dataset_type(dataset) == "kitti_raw"


def test_doctor_environment_artifact_is_schema_valid() -> None:
    artifact = build_doctor_artifact(
        calibrex_version="test",
        command=["calibrex", "doctor", "--json"],
    )

    assert artifact.status == "pass"
    assert artifact.dataset is None
    assert artifact.provenance.producer == "calibrex"
    assert artifact.provenance.dataset_digest_status == "not_applicable"
    jsonschema.validate(artifact.model_dump(mode="json"), doctor_json_schema())


def test_doctor_missing_dataset_fails_with_provenance(tmp_path: Path) -> None:
    missing = tmp_path / "missing.mcap"
    artifact = build_doctor_artifact(
        calibrex_version="test",
        command=["calibrex", "doctor", str(missing)],
        path=missing,
    )

    assert artifact.status == "fail"
    assert artifact.dataset is not None
    assert artifact.dataset.dataset_type == "mcap"
    assert artifact.dataset.exists is False
    assert artifact.provenance.dataset_type_source == "inferred"
    assert artifact.provenance.dataset_digest_status == "not_computed"
    jsonschema.validate(artifact.model_dump(mode="json"), doctor_json_schema())


def test_doctor_rosbag2_suggests_template(tmp_path: Path) -> None:
    bag_dir = tmp_path / "my_bag"
    bag_dir.mkdir()
    (bag_dir / "metadata.yaml").write_text(
        "rosbag2_bagfile_information:\n  version: 4\n", encoding="utf-8"
    )

    artifact = build_doctor_artifact(
        calibrex_version="test",
        command=["calibrex", "doctor", str(bag_dir)],
        path=bag_dir,
    )

    lidar_workflows = [w for w in artifact.workflows if w.workflow_id == "lidar-lidar-evidence"]
    assert lidar_workflows, "expected a lidar-lidar-evidence workflow suggestion"
    w = lidar_workflows[0]
    assert w.next_command is not None
    assert "velodyne_vlp16_pair_rosbag2" in w.next_command
    assert "calibrex calibrate" in w.next_command
    jsonschema.validate(artifact.model_dump(mode="json"), doctor_json_schema())


def test_doctor_rosbag1_suggests_template(tmp_path: Path) -> None:
    bag_file = tmp_path / "recording.bag"
    bag_file.write_bytes(b"#ROSBAG V2.0\n")

    artifact = build_doctor_artifact(
        calibrex_version="test",
        command=["calibrex", "doctor", str(bag_file)],
        path=bag_file,
    )

    lidar_workflows = [w for w in artifact.workflows if w.workflow_id == "lidar-lidar-evidence"]
    assert lidar_workflows, "expected a lidar-lidar-evidence workflow suggestion"
    w = lidar_workflows[0]
    assert w.next_command is not None
    assert "velodyne_vlp16_pair_rosbag1" in w.next_command
    assert "calibrex calibrate" in w.next_command
    jsonschema.validate(artifact.model_dump(mode="json"), doctor_json_schema())
