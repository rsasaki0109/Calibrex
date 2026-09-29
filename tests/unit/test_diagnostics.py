from pathlib import Path

import jsonschema

from calibrex.core.environment_readiness import (
    PythonPathIsolation,
    build_environment_readiness_artifact,
    environment_readiness_json_schema,
    infer_dataset_type,
    load_environment_readiness,
    python_path_isolation,
)
from calibrex.diagnostics import build_doctor_artifact, doctor_json_schema


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


def test_environment_readiness_artifact_is_schema_valid() -> None:
    artifact = build_environment_readiness_artifact(
        calibrex_version="test",
        command=["calibrex", "doctor", "--json"],
    )

    assert artifact.status == "pass"
    assert artifact.schema_version == "slac.environment_readiness/v0.1"
    assert artifact.dataset is None
    assert artifact.provenance.producer == "calibrex"
    assert artifact.provenance.dataset_digest_status == "not_applicable"
    assert artifact.environment.dependencies["pydantic"].available is True
    jsonschema.validate(
        artifact.model_dump(mode="json"),
        environment_readiness_json_schema(),
    )


def test_environment_readiness_round_trip(tmp_path: Path) -> None:
    artifact = build_environment_readiness_artifact(
        calibrex_version="test",
        command=["calibrex", "doctor", "--json"],
    )
    output = tmp_path / "readiness.yaml"
    artifact.save(output)
    loaded = load_environment_readiness(output)
    assert loaded.schema_version == artifact.schema_version
    assert loaded.environment.python_version == artifact.environment.python_version


def test_doctor_builder_alias_matches_environment_readiness() -> None:
    artifact = build_doctor_artifact(
        calibrex_version="test",
        command=["calibrex", "doctor", "--json"],
    )
    jsonschema.validate(artifact.model_dump(mode="json"), doctor_json_schema())


def test_doctor_missing_dataset_fails_with_provenance(tmp_path: Path) -> None:
    missing = tmp_path / "missing.mcap"
    artifact = build_environment_readiness_artifact(
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
    jsonschema.validate(
        artifact.model_dump(mode="json"),
        environment_readiness_json_schema(),
    )


def test_doctor_rosbag2_suggests_template(tmp_path: Path) -> None:
    bag_dir = tmp_path / "my_bag"
    bag_dir.mkdir()
    (bag_dir / "metadata.yaml").write_text(
        "rosbag2_bagfile_information:\n  version: 4\n", encoding="utf-8"
    )

    artifact = build_environment_readiness_artifact(
        calibrex_version="test",
        command=["calibrex", "doctor", str(bag_dir)],
        path=bag_dir,
    )

    lidar_suggestions = [
        item
        for item in artifact.workflow_suggestions
        if item.workflow_id == "lidar-lidar-evidence"
    ]
    assert lidar_suggestions, "expected a lidar-lidar-evidence workflow suggestion"
    suggestion = lidar_suggestions[0]
    assert suggestion.next_command is not None
    assert suggestion.template_path == (
        "examples/sensor_templates/velodyne_vlp16_pair_rosbag2"
    )
    assert "velodyne_vlp16_pair_rosbag2" in suggestion.next_command
    assert "calibrex init --template" in suggestion.next_command
    jsonschema.validate(
        artifact.model_dump(mode="json"),
        environment_readiness_json_schema(),
    )


def test_doctor_rosbag1_suggests_template(tmp_path: Path) -> None:
    bag_file = tmp_path / "recording.bag"
    bag_file.write_bytes(b"#ROSBAG V2.0\n")

    artifact = build_environment_readiness_artifact(
        calibrex_version="test",
        command=["calibrex", "doctor", str(bag_file)],
        path=bag_file,
    )

    lidar_suggestions = [
        item
        for item in artifact.workflow_suggestions
        if item.workflow_id == "lidar-lidar-evidence"
    ]
    assert lidar_suggestions, "expected a lidar-lidar-evidence workflow suggestion"
    suggestion = lidar_suggestions[0]
    assert suggestion.next_command is not None
    assert suggestion.template_path == (
        "examples/sensor_templates/velodyne_vlp16_pair_rosbag1"
    )
    assert "velodyne_vlp16_pair_rosbag1" in suggestion.next_command
    assert "calibrex init --template" in suggestion.next_command
    jsonschema.validate(
        artifact.model_dump(mode="json"),
        environment_readiness_json_schema(),
    )


def _isolation(**overrides: object) -> PythonPathIsolation:
    options: dict[str, object] = {
        "sys_path": ["/venv/lib/python3.12/site-packages"],
        "environ": {},
        "prefixes": ["/venv"],
        "loaded_modules": ["json", "calibrex"],
        "module_origin": lambda name: f"/venv/lib/python3.12/site-packages/{name}/__init__.py",
    }
    options.update(overrides)
    return python_path_isolation(**options)  # type: ignore[arg-type]


def test_python_path_isolation_is_clean_inside_the_environment() -> None:
    isolation = _isolation()

    assert isolation.status == "isolated"
    assert isolation.environment_prefix.endswith("venv")
    assert not isolation.recommendations


def test_ros_paths_are_reported_without_shadowing() -> None:
    ros_site = "/opt/ros/jazzy/lib/python3.12/site-packages"
    isolation = _isolation(
        sys_path=[ros_site, "/venv/lib/python3.12/site-packages"],
        environ={"PYTHONPATH": ros_site},
    )

    assert isolation.status == "isolated"
    assert isolation.ros_path_entries == [ros_site]
    assert isolation.external_path_entries == [ros_site]
    assert "env -u PYTHONPATH" in isolation.recommendations[0]


def test_dependency_resolved_from_ros_is_shadowed() -> None:
    def origin(name: str) -> str:
        if name == "numpy":
            return "/opt/ros/jazzy/lib/python3.12/site-packages/numpy/__init__.py"
        return f"/venv/lib/python3.12/site-packages/{name}/__init__.py"

    isolation = _isolation(module_origin=origin)

    assert isolation.status == "shadowed"
    assert list(isolation.shadowed_modules) == ["numpy"]


def test_colcon_workspace_is_detected_through_ament_prefix_path() -> None:
    workspace = "/home/user/ws/install/my_pkg/lib/python3.12/site-packages"
    isolation = _isolation(
        sys_path=[workspace],
        environ={"AMENT_PREFIX_PATH": "/home/user/ws/install/my_pkg"},
    )

    assert isolation.ros_path_entries == [workspace]


def test_loaded_ros_modules_break_the_ros_independence_claim() -> None:
    isolation = _isolation(loaded_modules=["rclpy", "rclpy.node", "launchpad_unrelated"])

    assert isolation.status == "shadowed"
    assert isolation.loaded_ros_modules == ["rclpy", "rclpy.node"]


def test_shadowed_environment_downgrades_readiness_to_warn(monkeypatch: object) -> None:
    import calibrex.core.environment_readiness as readiness

    shadowed = _isolation(loaded_modules=["rclpy"])
    monkeypatch.setattr(readiness, "python_path_isolation", lambda: shadowed)  # type: ignore[attr-defined]

    artifact = build_environment_readiness_artifact(calibrex_version="0.0.0", command=[])

    assert artifact.status == "warn"
    assert artifact.environment.core_ros_independent is False
