from calibrex.core.result import (
    CalibrationResult,
    FrameGraphSnapshot,
    RunInfo,
    TransformResult,
)
from calibrex.visualization.report import render_html_report


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
