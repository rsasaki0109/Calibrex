import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from calibrex.calibration_ci import calibration_ci_json_schema
from calibrex.core.assessment import assessment_json_schema
from calibrex.core.benchmark import (
    benchmark_definition_json_schema,
    benchmark_json_schema,
)
from calibrex.core.calibration_lifecycle import calibration_lifecycle_json_schema
from calibrex.core.camera_lidar_artifacts import (
    bullseye_plot_json_schema,
    calibration_candidate_trace_json_schema,
    camera_lidar_benchmark_protocol_json_schema,
    camera_lidar_problem_json_schema,
)
from calibrex.core.camera_lidar_confidence_calibration import (
    camera_lidar_confidence_calibration_json_schema,
)
from calibrex.core.camera_lidar_correspondence_export import (
    camera_lidar_correspondence_export_json_schema,
)
from calibrex.core.camera_lidar_correspondence_quality import (
    camera_lidar_correspondence_quality_json_schema,
)
from calibrex.core.camera_lidar_failure_analysis import (
    camera_lidar_failure_analysis_json_schema,
)
from calibrex.core.camera_lidar_initializer_calibration import (
    camera_lidar_initializer_calibration_json_schema,
)
from calibrex.core.camera_lidar_pose_initializer_benchmark import (
    camera_lidar_pose_initializer_protocol_json_schema,
)
from calibrex.core.camera_lidar_pose_initializer_failure_analysis import (
    camera_lidar_pose_initializer_failure_analysis_json_schema,
)
from calibrex.core.camera_lidar_provider_support_comparison import (
    camera_lidar_provider_support_comparison_json_schema,
)
from calibrex.core.camera_lidar_sota_audit import (
    camera_lidar_sota_audit_protocol_json_schema,
    camera_lidar_sota_audit_result_json_schema,
)
from calibrex.core.capture_manifest import (
    capture_manifest_json_schema,
    capture_manifest_verification_json_schema,
)
from calibrex.core.capture_readiness import capture_readiness_json_schema
from calibrex.core.config import config_json_schema
from calibrex.core.continuous_time_camera_lidar_artifacts import (
    continuous_time_camera_lidar_problem_json_schema,
    continuous_time_camera_lidar_result_json_schema,
)
from calibrex.core.continuous_time_contract import (
    continuous_time_trajectory_json_schema,
)
from calibrex.core.continuous_time_fit_artifacts import (
    continuous_time_fit_json_schema,
    continuous_time_measurements_json_schema,
)
from calibrex.core.continuous_time_imu_accel_bias import (
    continuous_time_imu_accel_bias_json_schema,
)
from calibrex.core.continuous_time_imu_clock_offset import (
    continuous_time_imu_clock_offset_json_schema,
)
from calibrex.core.continuous_time_imu_intrinsics import (
    continuous_time_imu_intrinsics_json_schema,
)
from calibrex.core.continuous_time_imu_lever_arm import (
    continuous_time_imu_lever_arm_json_schema,
)
from calibrex.core.continuous_time_imu_preintegration import (
    continuous_time_imu_preintegration_json_schema,
)
from calibrex.core.continuous_time_lidar_ablation import (
    CONTINUOUS_TIME_LIDAR_ABLATION_SCHEMA_VERSION,
    ContinuousTimeLidarAblationManifest,
    continuous_time_lidar_ablation_json_schema,
)
from calibrex.core.continuous_time_lidar_artifacts import (
    continuous_time_lidar_pair_json_schema,
)
from calibrex.core.continuous_time_lidar_point_to_plane import (
    continuous_time_lidar_point_to_plane_json_schema,
)
from calibrex.core.continuous_time_lidar_train_diagnostics import (
    continuous_time_lidar_train_diagnostics_json_schema,
)
from calibrex.core.continuous_time_sliding_window import (
    continuous_time_sliding_window_json_schema,
)
from calibrex.core.dynamic_window import dynamic_window_consistency_json_schema
from calibrex.core.empirical_uncertainty import empirical_se3_uncertainty_json_schema
from calibrex.core.environment_readiness import environment_readiness_json_schema
from calibrex.core.evidence_bundle import (
    evidence_bundle_json_schema,
    evidence_bundle_verification_json_schema,
    verify_evidence_bundle,
)
from calibrex.core.evidence_contract import policy_json_schema, protocol_json_schema
from calibrex.core.external_run import external_run_json_schema
from calibrex.core.koide_handoff import koide_execution_lock_json_schema
from calibrex.core.koide_readiness import koide_readiness_json_schema
from calibrex.core.koide_runner import koide_runner_json_schema
from calibrex.core.livox_time_ablation import (
    LIVOX_TIME_ABLATION_SCHEMA_VERSION,
    LivoxTimeAblationManifest,
    livox_time_ablation_json_schema,
)
from calibrex.core.mcap_integrity import mcap_integrity_json_schema
from calibrex.core.online_timeline import online_timeline_json_schema
from calibrex.core.probabilistic_correspondence import (
    probabilistic_correspondence_json_schema,
    probabilistic_pnp_result_json_schema,
    probabilistic_refinement_result_json_schema,
)
from calibrex.core.report_artifacts import report_artifact_json_schema
from calibrex.core.result import load_result, result_json_schema
from calibrex.core.solid_state import solid_state_context_json_schema
from calibrex.core.solid_state_cross_dataset_benchmark import (
    solid_state_cross_dataset_benchmark_config_json_schema,
    solid_state_cross_dataset_benchmark_json_schema,
)
from calibrex.core.solid_state_failure_analysis import (
    solid_state_failure_analysis_json_schema,
)
from calibrex.core.solid_state_metrology_evaluation import (
    solid_state_metrology_evaluation_json_schema,
)
from calibrex.core.solid_state_synthetic_benchmark import (
    solid_state_synthetic_benchmark_json_schema,
)
from calibrex.core.trajectory import trajectory_json_schema
from calibrex.core.trajectory_window_drift import trajectory_window_drift_json_schema
from calibrex.core.transform_artifacts import transform_artifact_json_schema
from calibrex.data.depth import depth_provider_json_schema
from calibrex.data.kitti360_lidar_window_integration import (
    kitti360_lidar_window_integration_json_schema,
)
from calibrex.data.kitti_benchmark import kitti_benchmark_input_json_schema
from calibrex.data.kitti_raw_lidar_window_integration import (
    kitti_raw_lidar_window_integration_json_schema,
)
from calibrex.data.manifest import manifest_json_schema
from calibrex.data.remote_archive_selection import remote_archive_selection_json_schema
from calibrex.diagnostics import doctor_json_schema
from calibrex.evaluation.compare import compare_results, comparison_json_schema
from calibrex.evaluation.kitti_falsification_benchmark import (
    kitti_falsification_json_schema,
)
from calibrex.evaluation.koide_pilot import koide_pilot_json_schema
from calibrex.evaluation.report_compare import (
    compare_reports,
    report_comparison_json_schema,
)
from calibrex.export.autoware_promotion import autoware_promotion_json_schema
from calibrex.export.autoware_smoke import autoware_smoke_json_schema
from calibrex.visualization.report import write_report_artifacts


def test_static_schema_files_match_generated_schemas() -> None:
    generators: dict[str, Callable[[], dict[str, Any]]] = {
        "config.schema.json": config_json_schema,
        "result.schema.json": result_json_schema,
        "comparison.schema.json": comparison_json_schema,
        "report_comparison.schema.json": report_comparison_json_schema,
        "dynamic_window_consistency.schema.json": dynamic_window_consistency_json_schema,
        "assessment.schema.json": assessment_json_schema,
        "benchmark.schema.json": benchmark_json_schema,
        "benchmark_definition.schema.json": benchmark_definition_json_schema,
        "policy.schema.json": policy_json_schema,
        "protocol.schema.json": protocol_json_schema,
        "transforms.schema.json": transform_artifact_json_schema,
        "dataset_manifest.schema.json": manifest_json_schema,
        "remote_archive_selection.schema.json": remote_archive_selection_json_schema,
        "kitti360_lidar_window_integration.schema.json": (
            kitti360_lidar_window_integration_json_schema
        ),
        "kitti_raw_lidar_window_integration.schema.json": (
            kitti_raw_lidar_window_integration_json_schema
        ),
        "doctor.schema.json": doctor_json_schema,
        "environment_readiness.schema.json": environment_readiness_json_schema,
        "calibration_ci.schema.json": calibration_ci_json_schema,
        "calibration_lifecycle.schema.json": calibration_lifecycle_json_schema,
        "external_run.schema.json": external_run_json_schema,
        "kitti_falsification.schema.json": kitti_falsification_json_schema,
        "kitti_benchmark_input.schema.json": kitti_benchmark_input_json_schema,
        "depth_provider.schema.json": depth_provider_json_schema,
        "continuous_time_camera_lidar_problem.schema.json": (
            continuous_time_camera_lidar_problem_json_schema
        ),
        "continuous_time_camera_lidar_result.schema.json": (
            continuous_time_camera_lidar_result_json_schema
        ),
        "continuous_time_trajectory.schema.json": (
            continuous_time_trajectory_json_schema
        ),
        "continuous_time_trajectory_measurements.schema.json": (
            continuous_time_measurements_json_schema
        ),
        "continuous_time_trajectory_fit.schema.json": (
            continuous_time_fit_json_schema
        ),
        "probabilistic_correspondence.schema.json": (
            probabilistic_correspondence_json_schema
        ),
        "probabilistic_pnp_result.schema.json": probabilistic_pnp_result_json_schema,
        "probabilistic_refinement_result.schema.json": (
            probabilistic_refinement_result_json_schema
        ),
        "empirical_se3_uncertainty.schema.json": (
            empirical_se3_uncertainty_json_schema
        ),
        "camera_lidar_problem.schema.json": camera_lidar_problem_json_schema,
        "camera_lidar_correspondence_export.schema.json": (
            camera_lidar_correspondence_export_json_schema
        ),
        "camera_lidar_confidence_calibration.schema.json": (
            camera_lidar_confidence_calibration_json_schema
        ),
        "camera_lidar_initializer_calibration.schema.json": (
            camera_lidar_initializer_calibration_json_schema
        ),
        "camera_lidar_provider_support_comparison.schema.json": (
            camera_lidar_provider_support_comparison_json_schema
        ),
        "camera_lidar_pose_initializer_protocol.schema.json": (
            camera_lidar_pose_initializer_protocol_json_schema
        ),
        "camera_lidar_pose_initializer_failure_analysis.schema.json": (
            camera_lidar_pose_initializer_failure_analysis_json_schema
        ),
        "camera_lidar_correspondence_quality.schema.json": (
            camera_lidar_correspondence_quality_json_schema
        ),
        "camera_lidar_failure_analysis.schema.json": (
            camera_lidar_failure_analysis_json_schema
        ),
        "camera_lidar_sota_audit_protocol.schema.json": (
            camera_lidar_sota_audit_protocol_json_schema
        ),
        "camera_lidar_sota_audit_result.schema.json": (
            camera_lidar_sota_audit_result_json_schema
        ),
        "camera_lidar_benchmark_protocol.schema.json": (
            camera_lidar_benchmark_protocol_json_schema
        ),
        "calibration_candidate_trace.schema.json": (
            calibration_candidate_trace_json_schema
        ),
        "bullseye_plot.schema.json": bullseye_plot_json_schema,
        "evidence_bundle.schema.json": evidence_bundle_json_schema,
        "evidence_bundle_verification.schema.json": evidence_bundle_verification_json_schema,
        "online_timeline.schema.json": online_timeline_json_schema,
        "trajectory.schema.json": trajectory_json_schema,
        "trajectory_window_drift.schema.json": trajectory_window_drift_json_schema,
        "capture_readiness.schema.json": capture_readiness_json_schema,
        "capture_manifest.schema.json": capture_manifest_json_schema,
        "capture_manifest_verification.schema.json": capture_manifest_verification_json_schema,
        "mcap_integrity.schema.json": mcap_integrity_json_schema,
        "koide_readiness.schema.json": koide_readiness_json_schema,
        "koide_execution_lock.schema.json": koide_execution_lock_json_schema,
        "koide_pilot.schema.json": koide_pilot_json_schema,
        "koide_runner_config.schema.json": koide_runner_json_schema,
        "continuous_time_lidar_pair_result.schema.json": (
            continuous_time_lidar_pair_json_schema
        ),
        "continuous_time_lidar_point_to_plane.schema.json": (
            continuous_time_lidar_point_to_plane_json_schema
        ),
        "continuous_time_imu_preintegration.schema.json": (
            continuous_time_imu_preintegration_json_schema
        ),
        "continuous_time_imu_lever_arm.schema.json": (
            continuous_time_imu_lever_arm_json_schema
        ),
        "continuous_time_imu_clock_offset.schema.json": (
            continuous_time_imu_clock_offset_json_schema
        ),
        "continuous_time_imu_accel_bias.schema.json": (
            continuous_time_imu_accel_bias_json_schema
        ),
        "continuous_time_imu_intrinsics.schema.json": (
            continuous_time_imu_intrinsics_json_schema
        ),
        "continuous_time_sliding_window.schema.json": (
            continuous_time_sliding_window_json_schema
        ),
        "continuous_time_lidar_train_diagnostics.schema.json": (
            continuous_time_lidar_train_diagnostics_json_schema
        ),
        "continuous_time_lidar_ablation.schema.json": (
            continuous_time_lidar_ablation_json_schema
        ),
        "solid_state_cross_dataset_benchmark_config.schema.json": (
            solid_state_cross_dataset_benchmark_config_json_schema
        ),
        "solid_state_cross_dataset_benchmark.schema.json": (
            solid_state_cross_dataset_benchmark_json_schema
        ),
        "solid_state_failure_analysis.schema.json": solid_state_failure_analysis_json_schema,
        "solid_state_synthetic_benchmark.schema.json": (
            solid_state_synthetic_benchmark_json_schema
        ),
        "solid_state_metrology_evaluation.schema.json": (
            solid_state_metrology_evaluation_json_schema
        ),
        "solid_state_context.schema.json": solid_state_context_json_schema,
        "livox_time_ablation.schema.json": livox_time_ablation_json_schema,
        "autoware_promotion.schema.json": autoware_promotion_json_schema,
        "autoware_smoke.schema.json": autoware_smoke_json_schema,
        "report_summary.schema.json": lambda: report_artifact_json_schema("report-summary"),
        "report_metrics.schema.json": lambda: report_artifact_json_schema("report-metrics"),
        "report_observability.schema.json": lambda: report_artifact_json_schema(
            "report-observability"
        ),
        "report_degeneracy.schema.json": lambda: report_artifact_json_schema("report-degeneracy"),
        "report_evidence.schema.json": lambda: report_artifact_json_schema("report-evidence"),
    }
    for filename, generate_schema in generators.items():
        static_schema = json.loads((Path("schemas") / filename).read_text(encoding="utf-8"))
        assert static_schema == generate_schema()


def test_livox_time_ablation_schema_validates_manifest() -> None:
    schema = json.loads(
        Path("schemas/livox_time_ablation.schema.json").read_text(encoding="utf-8")
    )
    manifest = LivoxTimeAblationManifest.model_validate(
        {
            "schema_version": LIVOX_TIME_ABLATION_SCHEMA_VERSION,
            "tool": "tools/run_livox_time_ablation.py",
            "tool_version": "0.1.0",
            "base_config_path": "config.yaml",
            "base_config_sha256": "a" * 64,
            "policy": {
                "deskew_off_means": "observe offsets without consuming them",
            },
            "variants": [
                {
                    "id": "deskew_on_offset_p0ms",
                    "config_path": "deskew_on_offset_p0ms/config.yaml",
                    "config_sha256": "b" * 64,
                    "use_point_time_offsets": True,
                    "inject_time_offset_s": 0.0,
                    "run_status": "completed",
                    "final_rolling_rmse_m": 0.2,
                    "quality_grade": "pass",
                    "trajectory_gate_status": "pass",
                    "deskew_applied": True,
                    "point_time_mapping_status": "stable",
                    "point_time_mapping_residual_max_abs_s": 0.001,
                }
            ],
        }
    ).model_dump(mode="json", exclude_none=True)
    jsonschema.validate(manifest, schema)


def test_continuous_time_lidar_ablation_schema_validates_manifest() -> None:
    schema = json.loads(
        Path("schemas/continuous_time_lidar_ablation.schema.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = ContinuousTimeLidarAblationManifest.model_validate(
        {
            "schema_version": CONTINUOUS_TIME_LIDAR_ABLATION_SCHEMA_VERSION,
            "tool": "tools/run_continuous_time_lidar_ablation.py",
            "tool_version": "0.1.0",
            "base_config_path": "config.yaml",
            "base_config_sha256": "a" * 64,
            "policy": {
                "same_capture_windows": True,
                "same_temporal_holdout": True,
                "same_solver_budget": True,
                "baseline_definition": "uniform voxel planes with no MAD rejection",
            },
            "variants": [
                {
                    "id": "adaptive_mad",
                    "config_path": "adaptive_mad/config.yaml",
                    "config_sha256": "b" * 64,
                    "result_path": "adaptive_mad/continuous_time_lidar_pair.yaml",
                    "result_sha256": "c" * 64,
                    "voxel_strategy": "adaptive",
                    "outlier_policy": "mad",
                    "status": "converged",
                    "estimated_time_offset_sec": -0.02,
                    "final_train_rmse_m": 0.34,
                    "final_holdout_rmse_m": 0.38,
                    "observability_rank": 6,
                    "outlier_rejected_count": 144,
                    "quality_grade": "pass",
                }
            ],
        }
    ).model_dump(mode="json", exclude_none=True)
    jsonschema.validate(manifest, schema)


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
    config = yaml.safe_load(
        Path("examples/public_datasets/nuscenes_mini/config.yaml").read_text(encoding="utf-8")
    )
    jsonschema.validate(config, schema)


def test_config_schema_validates_livox_pcd_example() -> None:
    schema = json.loads(Path("schemas/config.schema.json").read_text(encoding="utf-8"))
    config = yaml.safe_load(
        Path("examples/public_datasets/livox_horizon_horizon_pcd_sample/config.yaml").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.validate(config, schema)


def test_config_schema_validates_a2d2_native_point_to_plane_example() -> None:
    schema = json.loads(Path("schemas/config.schema.json").read_text(encoding="utf-8"))
    config = yaml.safe_load(
        Path(
            "examples/public_datasets/a2d2_lidar_pair_sample/native_point_to_plane_config.yaml"
        ).read_text(encoding="utf-8")
    )
    jsonschema.validate(config, schema)


def test_config_schema_validates_kitti_lidar_camera_evidence_example() -> None:
    schema = json.loads(Path("schemas/config.schema.json").read_text(encoding="utf-8"))
    config = yaml.safe_load(
        Path("examples/public_datasets/kitti_lidar_camera_evidence/config.yaml").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.validate(config, schema)


def test_config_schema_validates_tiers_camera_lidar_capture_time_example() -> None:
    schema = json.loads(Path("schemas/config.schema.json").read_text(encoding="utf-8"))
    config = yaml.safe_load(
        Path(
            "examples/public_datasets/tiers_livox_lidars_cali/camera_lidar_capture_time_config.yaml"
        ).read_text(encoding="utf-8")
    )
    jsonschema.validate(config, schema)


def test_config_schema_validates_tiers_livox_pair_example() -> None:
    schema = json.loads(Path("schemas/config.schema.json").read_text(encoding="utf-8"))
    config = yaml.safe_load(
        Path("examples/public_datasets/tiers_livox_lidars_cali/config.yaml").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.validate(config, schema)


def test_config_schema_validates_tiers_indoor02_motion_variants() -> None:
    schema = json.loads(Path("schemas/config.schema.json").read_text(encoding="utf-8"))
    for path in [
        "examples/public_datasets/tiers_lidars_dataset_indoor02/online_motion_pose_burst_config.yaml",
        "examples/public_datasets/tiers_lidars_dataset_indoor02/online_motion_baseframe_config.yaml",
    ]:
        config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
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


def test_result_schema_validates_kitti_lidar_camera_cached_evidence_example() -> None:
    schema = json.loads(Path("schemas/result.schema.json").read_text(encoding="utf-8"))
    result = yaml.safe_load(
        Path(
            "examples/public_datasets/kitti_lidar_camera_evidence/cached_evidence_result.yaml"
        ).read_text(encoding="utf-8")
    )
    jsonschema.validate(result, schema)


def test_comparison_schema_validates_generated_comparison() -> None:
    schema = json.loads(Path("schemas/comparison.schema.json").read_text(encoding="utf-8"))
    result = load_result("examples/precomputed/result.yaml")
    comparison = compare_results(result, result).model_dump(mode="json")
    jsonschema.validate(comparison, schema)


def test_report_comparison_schema_validates_generated_report_comparison() -> None:
    schema = json.loads(Path("schemas/report_comparison.schema.json").read_text(encoding="utf-8"))
    reference = load_result("examples/precomputed/result.yaml")
    candidate = load_result("examples/precomputed/result.yaml")
    candidate.run.id = "candidate_variant"
    native = load_result(
        "examples/public_datasets/livox_horizon_horizon_pcd_sample/cached_evidence_result.yaml"
    )
    report = compare_reports(
        [("reference", reference), ("candidate", candidate), ("native", native)],
        reference_label="reference",
    ).model_dump(mode="json")
    jsonschema.validate(report, schema)


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
        "transforms.json": "transforms.schema.json",
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
        "examples/public_datasets/tiers_lidars_dataset_indoor02/manifest.yaml",
    ]:
        manifest = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        jsonschema.validate(manifest, schema)
