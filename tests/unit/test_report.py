from slac.core.result import (
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
from slac.visualization.report import render_html_report
from slac.visualization.rig3d import render_rig_3d_artifact


def test_report_renders_candidate_reference_delta_table() -> None:
    result = CalibrationResult(
        run=RunInfo(id="report-unit", slac_version="0.1.0"),
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
        run=RunInfo(id="report-unit", slac_version="0.1.0"),
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
    assert "assessment.json" in html
    assert "protocol.json" in html
    assert "policy.json" in html
    assert "transforms.json" in html
    assert "bundle.json" in html
    assert "verification.json" in html


def test_report_renders_lidar_pair_evidence_section() -> None:
    result = CalibrationResult(
        run=RunInfo(
            id="livox-report-unit",
            slac_version="0.1.0",
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
                        "support_population_id": (
                            "livox_pair_support:train=base_horizon_100432:"
                            "holdout=target_horizon_100538:eligible_points=23666:"
                            "voxel_m=1:gate_m=1.5"
                        ),
                        "support_definition": (
                            "eligible population is every finite target PCD point"
                        ),
                        "eligible_point_count": 23666,
                        "considered_point_count": 23666,
                        "accepted_correspondence_count": 16884,
                        "support_ratio": 0.7134285472830221,
                        "exclusion_counts": {"no_plane_within_gate": 6782},
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
                            "lidar_pair_holdout_point_to_plane_support_ratio": 0.71,
                            "lidar_pair_holdout_point_to_plane_unmatched_fraction": 0.29,
                        },
                        "delta_values": {
                            "source_recall_delta": 0.007,
                            "centroid_rmse_delta_m": -0.008,
                            "point_to_plane_p90_delta_m": 0.034,
                            "point_to_plane_rmse_delta_m": 0.004,
                            "support_ratio_delta": -0.003,
                        },
                    }
                ],
                "raw_input_files": [
                    {
                        "path": "data/public/livox_horizon_horizon_pair/base_horizon_100432.pcd",
                        "role": "source",
                        "sha256": "0123456789abcdef0123456789abcdef",
                        "size_bytes": 4096,
                        "source_url": "https://example.invalid/base.tar.gz",
                    },
                    {
                        "path": (
                            "data/public/livox_horizon_horizon_pair/"
                            "target_horizon_100538.pcd"
                        ),
                        "role": "target",
                        "sha256": "abcdef0123456789abcdef0123456789",
                        "size_bytes": 8192,
                        "source_url": "https://example.invalid/target.tar.gz",
                    },
                ],
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
    assert "Raw Input Files" in html
    assert "base_horizon_100432.pcd" in html
    assert "0123456789abcdef..." in html
    assert "Falsification Assessment" in html
    assert "INCONCLUSIVE" in html
    assert "holdout_independence" in html
    assert "livox_pair_single_pair_holdout_point_to_plane/v0.1" in html
    assert "single_pair_source_map_target_query" in html
    assert "matched_point_count=16884" in html
    assert "support_ratio=0.7134285472830221" in html
    assert "support_population_id=livox_pair_support:" in html
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
            slac_version="0.1.0",
            provenance={
                "metrics_origin": "cached",
                "data_verified": False,
                "computed_at": "2026-06-18T10:53:34Z",
            },
        ),
        frame_graph=FrameGraphSnapshot(root="base", frames={"base": None}),
    )

    html = render_html_report(result)

    assert "EVIDENCE INPUTS NOT VERIFIED AS RAW RECOMPUTATION" in html
    assert "metrics_origin" in html
    assert "data_verified" in html


def test_report_marks_large_candidate_reference_delta_as_warn() -> None:
    result = CalibrationResult(
        run=RunInfo(id="report-unit", slac_version="0.1.0"),
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
        run=RunInfo(id="report-unit", slac_version="0.1.0"),
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
        run=RunInfo(id="rig-3d-unit", slac_version="0.1.0"),
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


def _synthetic_online_timeline() -> tuple[CalibrationResult, object]:
    from slac.core.online_timeline import (
        OnlineBatchSnapshot,
        OnlineCalibrationTimelineArtifact,
    )
    from slac.core.report_artifacts import ReportRunInfo

    batches = [
        OnlineBatchSnapshot(
            batch_index=0,
            frame_count=1,
            point_count=500,
            train_point_count=400,
            holdout_point_count=100,
            correspondence_count=120,
            holdout_correspondence_count=30,
            estimate=TransformResult(
                parent="base_horizon",
                child="livox_avia",
                translation_m=[0.1, 0.0, 0.2],
                rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            ),
            estimate_accepted=True,
            batch_holdout_rmse_m=0.08,
            rolling_rmse_m=0.08,
            observability=ObservabilityResult(rank=4, condition_number=12.0),
            batch_observability=ObservabilityResult(rank=4, condition_number=10.0),
            gate_status="inconclusive",
            gate_reason="insufficient observability; retained for accumulation",
            retained_for_accumulation=True,
        ),
        OnlineBatchSnapshot(
            batch_index=1,
            frame_count=1,
            point_count=500,
            train_point_count=400,
            holdout_point_count=100,
            correspondence_count=140,
            holdout_correspondence_count=35,
            estimate=TransformResult(
                parent="base_horizon",
                child="livox_avia",
                translation_m=[0.11, 0.01, 0.21],
                rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            ),
            estimate_accepted=True,
            batch_holdout_rmse_m=0.12,
            rolling_rmse_m=0.10,
            observability=ObservabilityResult(rank=6, condition_number=9.5),
            batch_observability=ObservabilityResult(rank=5, condition_number=11.0),
            gate_status="pass",
            gate_reason="holdout evidence supports this batch's update",
            retained_for_accumulation=False,
        ),
        OnlineBatchSnapshot(
            batch_index=2,
            frame_count=1,
            point_count=500,
            train_point_count=400,
            holdout_point_count=100,
            correspondence_count=130,
            holdout_correspondence_count=32,
            estimate=TransformResult(
                parent="base_horizon",
                child="livox_avia",
                translation_m=[0.12, 0.02, 0.22],
                rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            ),
            estimate_accepted=False,
            batch_holdout_rmse_m=0.55,
            rolling_rmse_m=0.10,
            observability=ObservabilityResult(rank=6, condition_number=8.0),
            batch_observability=ObservabilityResult(rank=6, condition_number=8.0),
            gate_status="fail",
            gate_reason="holdout RMSE 0.5500 m exceeds the 0.4000 m gate",
            retained_for_accumulation=False,
        ),
    ]
    timeline = OnlineCalibrationTimelineArtifact(
        run=ReportRunInfo(
            id="online-report-unit",
            status="warning",
            domain="robotics",
            slac_version="0.1.0",
            created_at="2026-07-03T00:00:00Z",
        ),
        variable="T_base_horizon_livox_avia",
        source_sensor="livox_horizon",
        target_sensor="livox_avia",
        batch_size=500,
        rolling_window=2000,
        holdout_ratio=0.2,
        seed=0,
        accumulation_batches=3,
        batches=batches,
        final_gate_status="fail",
        accepted_batch_count=1,
        rejected_batch_count=1,
        inconclusive_batch_count=1,
    )
    result = CalibrationResult(
        run=RunInfo(
            id="online-report-unit",
            slac_version="0.1.0",
            provenance={
                "online_gate_min_rank": 6,
                "online_gate_max_holdout_rmse_m": 0.4,
                "online_gate_max_rolling_regression_m": 0.15,
            },
        ),
        frame_graph=FrameGraphSnapshot(root="base_horizon", frames={"base_horizon": None}),
        transforms={
            "T_base_horizon_livox_avia": TransformResult(
                parent="base_horizon",
                child="livox_avia",
                translation_m=[0.11, 0.01, 0.21],
                rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            )
        },
    )
    return result, timeline


def test_report_renders_online_timeline_section() -> None:
    result, timeline = _synthetic_online_timeline()

    html = render_html_report(result, timeline=timeline)

    assert "Online Timeline" in html
    assert "Gate Verdicts" in html
    assert "Holdout and Rolling RMSE" in html
    assert "Observability" in html
    assert "Session Summary" in html
    assert html.count('class="gate-cell gate-pass"') == 1
    assert html.count('class="gate-cell gate-fail"') == 1
    assert html.count('class="gate-cell gate-inconclusive"') == 1
    assert html.count('class="rank-lifted"') == 1
    assert "1 /" in html and "/ 1 /" in html
    assert "Retained for accumulation" in html
    assert "min_rank=6" in html
    assert "max_holdout_rmse_m=0.4" in html


def test_report_online_timeline_rmse_chart_has_expected_polylines() -> None:
    result, timeline = _synthetic_online_timeline()

    html = render_html_report(result, timeline=timeline)

    import re

    holdout_match = re.search(
        r'<polyline[^>]*stroke="#2563eb"[^>]*points="([^"]+)"',
        html,
    )
    rolling_match = re.search(
        r'<polyline[^>]*stroke="#059669"[^>]*points="([^"]+)"',
        html,
    )
    assert holdout_match is not None
    assert rolling_match is not None
    assert len(holdout_match.group(1).split()) == 3
    assert len(rolling_match.group(1).split()) == 3
    assert re.search(r'stroke="#d97706"', html) is not None
    assert re.search(r'stroke="#dc2626"', html) is not None
    assert "RMSE (m)" in html
    assert "Batch index" in html


def test_report_omits_online_timeline_section_for_offline_results() -> None:
    result = CalibrationResult(
        run=RunInfo(id="offline-report-unit", slac_version="0.1.0"),
        frame_graph=FrameGraphSnapshot(root="ego", frames={"ego": None}),
    )

    html = render_html_report(result)

    assert "Online Timeline" not in html
    assert 'class="gate-strip"' not in html


def test_write_report_artifacts_passes_timeline_to_html(tmp_path) -> None:
    from slac.visualization.report import write_report_artifacts

    result, timeline = _synthetic_online_timeline()

    write_report_artifacts(result, tmp_path, timeline=timeline)

    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "Online Timeline" in html
