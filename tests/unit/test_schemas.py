import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from calibrex.core.assessment import assessment_json_schema
from calibrex.core.config import config_json_schema
from calibrex.core.evidence_bundle import (
    evidence_bundle_json_schema,
    evidence_bundle_verification_json_schema,
    verify_evidence_bundle,
)
from calibrex.core.evidence_contract import policy_json_schema, protocol_json_schema
from calibrex.core.report_artifacts import report_artifact_json_schema
from calibrex.core.result import load_result, result_json_schema
from calibrex.data.manifest import manifest_json_schema
from calibrex.evaluation.compare import compare_results, comparison_json_schema
from calibrex.visualization.report import write_report_artifacts


def test_static_schema_files_match_generated_schemas() -> None:
    generators: dict[str, Callable[[], dict[str, Any]]] = {
        "config.schema.json": config_json_schema,
        "result.schema.json": result_json_schema,
        "comparison.schema.json": comparison_json_schema,
        "assessment.schema.json": assessment_json_schema,
        "policy.schema.json": policy_json_schema,
        "protocol.schema.json": protocol_json_schema,
        "dataset_manifest.schema.json": manifest_json_schema,
        "evidence_bundle.schema.json": evidence_bundle_json_schema,
        "evidence_bundle_verification.schema.json": evidence_bundle_verification_json_schema,
        "report_summary.schema.json": lambda: report_artifact_json_schema("report-summary"),
        "report_metrics.schema.json": lambda: report_artifact_json_schema("report-metrics"),
        "report_observability.schema.json": lambda: report_artifact_json_schema(
            "report-observability"
        ),
        "report_degeneracy.schema.json": lambda: report_artifact_json_schema(
            "report-degeneracy"
        ),
        "report_evidence.schema.json": lambda: report_artifact_json_schema("report-evidence"),
    }
    for filename, generate_schema in generators.items():
        static_schema = json.loads((Path("schemas") / filename).read_text(encoding="utf-8"))
        assert static_schema == generate_schema()


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


def test_config_schema_validates_livox_pcd_example() -> None:
    schema = json.loads(Path("schemas/config.schema.json").read_text(encoding="utf-8"))
    config = yaml.safe_load(
        Path("examples/public_datasets/livox_horizon_horizon_pcd_sample/config.yaml").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.validate(config, schema)


def test_result_schema_validates_precomputed_example() -> None:
    schema = json.loads(Path("schemas/result.schema.json").read_text(encoding="utf-8"))
    result = yaml.safe_load(Path("examples/precomputed/result.yaml").read_text(encoding="utf-8"))
    jsonschema.validate(result, schema)


def test_result_schema_validates_livox_cached_evidence_example() -> None:
    schema = json.loads(Path("schemas/result.schema.json").read_text(encoding="utf-8"))
    result = yaml.safe_load(
        Path(
            "examples/public_datasets/livox_horizon_horizon_pcd_sample/cached_evidence_result.yaml"
        ).read_text(encoding="utf-8")
    )
    jsonschema.validate(result, schema)


def test_comparison_schema_validates_generated_comparison() -> None:
    schema = json.loads(Path("schemas/comparison.schema.json").read_text(encoding="utf-8"))
    result = load_result("examples/precomputed/result.yaml")
    comparison = compare_results(result, result).model_dump(mode="json")
    jsonschema.validate(comparison, schema)


def test_report_sidecar_schemas_validate_generated_sidecars(tmp_path: Path) -> None:
    result = load_result(
        "examples/public_datasets/livox_horizon_horizon_pcd_sample/cached_evidence_result.yaml"
    )
    write_report_artifacts(result, tmp_path)
    for sidecar_name, schema_name in {
        "summary.json": "report_summary.schema.json",
        "metrics.json": "report_metrics.schema.json",
        "observability.json": "report_observability.schema.json",
        "degeneracy.json": "report_degeneracy.schema.json",
        "evidence.json": "report_evidence.schema.json",
        "assessment.json": "assessment.schema.json",
        "policy.json": "policy.schema.json",
        "protocol.json": "protocol.schema.json",
        "bundle.json": "evidence_bundle.schema.json",
        "verification.json": "evidence_bundle_verification.schema.json",
    }.items():
        schema = json.loads((Path("schemas") / schema_name).read_text(encoding="utf-8"))
        sidecar = json.loads((tmp_path / sidecar_name).read_text(encoding="utf-8"))
        jsonschema.validate(sidecar, schema)
    verification_schema = json.loads(
        Path("schemas/evidence_bundle_verification.schema.json").read_text(encoding="utf-8")
    )
    verification = verify_evidence_bundle(tmp_path / "bundle.json").model_dump(mode="json")
    jsonschema.validate(verification, verification_schema)


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
        "examples/public_datasets/livox_horizon_horizon_pcd_sample/manifest.yaml",
        "examples/public_datasets/tiers_livox_lidars_cali/manifest.yaml",
    ]:
        manifest = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        jsonschema.validate(manifest, schema)
