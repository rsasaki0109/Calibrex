import json
from pathlib import Path

import jsonschema
import yaml


def test_config_schema_validates_minimal_example() -> None:
    schema = json.loads(Path("schemas/config.schema.json").read_text(encoding="utf-8"))
    config = yaml.safe_load(Path("examples/configs/minimal.yaml").read_text(encoding="utf-8"))
    jsonschema.validate(config, schema)


def test_config_schema_validates_kitti_example() -> None:
    schema = json.loads(Path("schemas/config.schema.json").read_text(encoding="utf-8"))
    config = yaml.safe_load(
        Path("examples/public_datasets/kitti_raw_2011_09_26_drive_0005/config.yaml").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.validate(config, schema)


def test_config_schema_validates_nuscenes_example() -> None:
    schema = json.loads(Path("schemas/config.schema.json").read_text(encoding="utf-8"))
    config = yaml.safe_load(Path("examples/public_datasets/nuscenes_mini/config.yaml").read_text(
        encoding="utf-8"
    ))
    jsonschema.validate(config, schema)


def test_result_schema_validates_precomputed_example() -> None:
    schema = json.loads(Path("schemas/result.schema.json").read_text(encoding="utf-8"))
    result = yaml.safe_load(Path("examples/precomputed/result.yaml").read_text(encoding="utf-8"))
    jsonschema.validate(result, schema)


def test_dataset_manifest_schema_validates_synthetic_example() -> None:
    schema = json.loads(Path("schemas/dataset_manifest.schema.json").read_text(encoding="utf-8"))
    manifest = yaml.safe_load(
        Path("examples/synthetic_camera_lidar_imu/manifest.yaml").read_text(encoding="utf-8")
    )
    jsonschema.validate(manifest, schema)


def test_dataset_manifest_schema_validates_rgbd_open3d_example() -> None:
    schema = json.loads(Path("schemas/dataset_manifest.schema.json").read_text(encoding="utf-8"))
    manifest = yaml.safe_load(
        Path("examples/rgbd_open3d_slac/manifest.yaml").read_text(encoding="utf-8")
    )
    jsonschema.validate(manifest, schema)


def test_dataset_manifest_schema_validates_public_dataset_examples() -> None:
    schema = json.loads(Path("schemas/dataset_manifest.schema.json").read_text(encoding="utf-8"))
    for path in [
        "examples/public_datasets/tum_rgbd_freiburg1_xyz/manifest.yaml",
        "examples/public_datasets/kitti_raw_2011_09_26_drive_0005/manifest.yaml",
        "examples/public_datasets/nuscenes_mini/manifest.yaml",
        "examples/public_datasets/a2d2_sensor_setup/manifest.yaml",
        "examples/public_datasets/a2d2_lidar_pair_sample/manifest.yaml",
    ]:
        manifest = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        jsonschema.validate(manifest, schema)
