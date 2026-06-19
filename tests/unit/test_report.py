from calibrex.core.result import (
    ArtifactSet,
    CalibrationResult,
    DegeneracyResult,
    FrameGraphSnapshot,
    MetricResult,
    ObservabilityResult,
    QualitySummary,
    RunInfo,
    TransformResult,
)
from calibrex.visualization.report import render_html_report
from calibrex.visualization.rig3d import render_rig_3d_artifact


def test_report_renders_candidate_reference_delta_table() -> None:
    result = CalibrationResult(
        run=RunInfo(id="report-unit", calibrex_version="0.1.0"),
        frame_graph=FrameGraphSnapshot(
            root="ego",
            frames={"ego": None, "lidar_top": "ego"},
        ),
        candidate_extrinsics={
            "T_ego_lidar_top": TransformResult(
                parent="ego",
                child="lidar_top",
                translation_m=[1.0, 0.0, 2.0],
                rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            )
        },
        reference_extrinsics={
            "T_ego_lidar_top_ref": TransformResult(
                parent="ego",
                child="lidar_top",
                translation_m=[1.02, 0.04, 2.0],
                rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            )
        },
    )

    html = render_html_report(result)

    assert "Candidate vs Reference Extrinsics" in html
    assert "T_ego_lidar_top" in html
    assert "T_ego_lidar_top_ref" in html
    assert "ego -> lidar_top" in html
    assert "0.0447214" in html
    assert "PASS" in html


def test_report_renders_calibration_scoreboard() -> None:
    result = CalibrationResult(
        run=RunInfo(id="report-unit", calibrex_version="0.1.0"),
        frame_graph=FrameGraphSnapshot(
            root="ego",
            frames={"ego": None, "lidar_top": "ego"},
        ),
        candidate_extrinsics={
            "T_ego_lidar_top": TransformResult(
                parent="ego",
                child="lidar_top",
                translation_m=[1.0, 0.0, 2.0],
                rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            )
        },
        reference_extrinsics={
            "T_ego_lidar_top_ref": TransformResult(
                parent="ego",
                child="lidar_top",
                translation_m=[1.01, 0.0, 2.0],
                rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            )
        },
        metrics={
            "lidar_world_map_point_to_plane_rmse_m": MetricResult(
                holdout=0.03,
                grade="pass",
                unit="m",
            ),
            "lidar_world_map_weak_dof_count": MetricResult(
                value=2.0,
                grade="warn",
            ),
        },
        observability=ObservabilityResult(
            rank=4,
            weak_directions=["yaw_lidar0", "z_lidar0"],
            grade="warn",
        ),
        degeneracy=DegeneracyResult(
            grade="warn",
            reason="limited vertical excitation",
        ),
        quality=QualitySummary(
            grade="warn",
            warnings=["limited vertical excitation"],
        ),
        artifacts=ArtifactSet(html_report="report.html", camera_lidar_overlay="overlay.png"),
    )

    html = render_html_report(result)

    assert "Calibration Scoreboard" in html
    assert "PASS 1 / WARN 1 / FAIL 0" in html
    assert "1 / 1" in html
    assert "yaw_lidar0, z_lidar0" in html
    assert "limited vertical excitation" in html
    assert "linked report outputs" in html
    assert "summary.json" in html
    assert "metrics.json" in html
    assert "evidence.json" in html


def test_report_renders_lidar_pair_evidence_section() -> None:
    result = CalibrationResult(
        run=RunInfo(
            id="livox-report-unit",
            calibrex_version="0.1.0",
            provenance={
                "livox_pair_evidence": {
                    "candidate_transform": "T_base_horizon_target_horizon",
                    "target_transform_applied": True,
                    "transform_convention": (
                        "T_source_target maps target PCD points into source PCD frame"
                    ),
                    "holdout_geometry": {
                        "status": "scored",
                        "split_policy": "single_pair_source_map_target_query",
                        "independent_holdout": False,
                        "map_voxel_count": 679,
                        "matched_point_count": 16884,
                        "unmatched_fraction": 0.2865714527169779,
                        "voxel_size_m": 1.0,
                        "correspondence_gate_m": 1.5,
                        "inlier_threshold_m": 0.25,
                        "plane_normal_source": "pcd_normal_fields_or_local_fallback",
                        "reason": "single source/target PCD pair",
                    },
                    "known_bad_perturbation": (
                        "left-multiplied source-frame SE(3) controls"
                    ),
                    "known_bad_case_count": 24,
                },
                "evidence_cases": [
                    {
                        "family": "lidar_pair",
                        "case_id": "pitch_deg:+1deg",
                        "check": "Known-Bad Controls",
                        "status": "pass",
                        "dof": "pitch_deg",
                        "amount": 1.0,
                        "unit": "deg",
                        "convention": "left-multiplied source-frame SE(3) perturbation",
                        "metric_values": {
                            "lidar_pair_holdout_point_to_plane_p90_abs_m": 0.84,
                            "lidar_pair_holdout_point_to_plane_unmatched_fraction": 0.29,
                        },
                        "delta_values": {
                            "source_recall_delta": 0.007,
                            "centroid_rmse_delta_m": -0.008,
                            "point_to_plane_p90_delta_m": 0.034,
                            "point_to_plane_rmse_delta_m": 0.004,
                        },
                    }
                ]
            },
        ),
        frame_graph=FrameGraphSnapshot(
            root="base_horizon",
            frames={"base_horizon": None, "target_horizon": "base_horizon"},
        ),
        metrics={
            "lidar_pair_source_voxel_recall_in_target": MetricResult(
                value=0.25,
                grade="pass",
            ),
            "lidar_pair_shared_voxel_centroid_rmse_m": MetricResult(
                value=0.538,
                unit="m",
                grade="pass",
            ),
            "lidar_pair_holdout_point_to_plane_p90_abs_m": MetricResult(
                value=0.22,
                unit="m",
                grade="pass",
            ),
            "lidar_pair_known_bad_detectable_fraction": MetricResult(
                value=0.75,
                grade="pass",
            ),
        },
    )

    html = render_html_report(result)

    assert "LiDAR Pair Evidence" in html
    assert "Evidence Protocol" in html
    assert "livox_pair_single_pair_holdout_point_to_plane/v0.1" in html
    assert "single_pair_source_map_target_query" in html
    assert "matched_point_count=16884" in html
    assert "Metrics Origin" in html
    assert "Evidence Summary" in html
    assert "Candidate Support" in html
    assert "Holdout Geometry" in html
    assert "Known-Bad Controls" in html
    assert "Known-Bad Case Details" in html
    assert "Decision Boundary" in html
    assert "Supported by this evidence protocol" in html
    assert "pitch_deg:+1deg" in html
    assert "0.034" in html
    assert "lidar_pair_source_voxel_recall_in_target" in html
    assert "lidar_pair_shared_voxel_centroid_rmse_m" in html
    assert "lidar_pair_holdout_point_to_plane_p90_abs_m" in html
    assert "lidar_pair_known_bad_detectable_fraction" in html
    assert "absolute ground truth" in html


def test_report_marks_cached_evidence_artifacts() -> None:
    result = CalibrationResult(
        run=RunInfo(
            id="cached-report-unit",
            calibrex_version="0.1.0",
            provenance={
                "metrics_origin": "cached",
                "data_verified": False,
                "computed_at": "2026-06-18T10:53:34Z",
            },
        ),
        frame_graph=FrameGraphSnapshot(root="base", frames={"base": None}),
    )

    html = render_html_report(result)

    assert "CACHED EVIDENCE" in html
    assert "RAW DATA NOT READ OR RECOMPUTED" in html
    assert "metrics_origin" in html
    assert "data_verified" in html


def test_report_marks_large_candidate_reference_delta_as_warn() -> None:
    result = CalibrationResult(
        run=RunInfo(id="report-unit", calibrex_version="0.1.0"),
        frame_graph=FrameGraphSnapshot(
            root="ego",
            frames={"ego": None, "lidar_top": "ego"},
        ),
        candidate_extrinsics={
            "T_ego_lidar_top": TransformResult(
                parent="ego",
                child="lidar_top",
                translation_m=[0.0, 0.0, 0.0],
                rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            )
        },
        reference_extrinsics={
            "T_ego_lidar_top_ref": TransformResult(
                parent="ego",
                child="lidar_top",
                translation_m=[1.0, 0.0, 0.0],
                rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            )
        },
    )

    html = render_html_report(result)

    assert "1" in html
    assert "WARN" in html


def test_report_explains_unmatched_candidate_reference_edges() -> None:
    result = CalibrationResult(
        run=RunInfo(id="report-unit", calibrex_version="0.1.0"),
        frame_graph=FrameGraphSnapshot(
            root="ego",
            frames={"ego": None, "lidar_top": "ego", "camera_front": "ego"},
        ),
        candidate_extrinsics={
            "T_ego_lidar_top": TransformResult(
                parent="ego",
                child="lidar_top",
                translation_m=[0.0, 0.0, 0.0],
                rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            )
        },
        reference_extrinsics={
            "T_ego_camera_front": TransformResult(
                parent="ego",
                child="camera_front",
                translation_m=[0.0, 0.0, 0.0],
                rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            )
        },
    )

    html = render_html_report(result)

    assert "No candidate extrinsics matched reference parent/child pairs" in html


def test_rig_3d_artifact_renders_reference_online_and_candidate_layers() -> None:
    result = CalibrationResult(
        run=RunInfo(id="rig-3d-unit", calibrex_version="0.1.0"),
        frame_graph=FrameGraphSnapshot(
            root="ego",
            frames={"ego": None, "lidar_top": "ego"},
        ),
        transforms={
            "T_ego_lidar_top": TransformResult(
                parent="ego",
                child="lidar_top",
                translation_m=[1.05, 0.0, 2.0],
                rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            )
        },
        candidate_extrinsics={
            "T_ego_lidar_top_candidate": TransformResult(
                parent="ego",
                child="lidar_top",
                translation_m=[0.95, 0.0, 2.0],
                rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            )
        },
        reference_extrinsics={
            "T_ego_lidar_top_reference": TransformResult(
                parent="ego",
                child="lidar_top",
                translation_m=[1.0, 0.0, 2.0],
                rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            )
        },
    )

    html = render_rig_3d_artifact(result)

    assert "3D Calibration Rig View" in html
    assert "https://cdn.jsdelivr.net/npm/three" in html
    assert "Online / Estimated" in html
    assert "Reference" in html
    assert "Candidate" in html
    assert "T_ego_lidar_top_reference" in html
    assert "T_ego_lidar_top_candidate" in html
    assert "0.05" in html
