import math
from pathlib import Path

import numpy as np

from calibrex.core.config import CalibrationConfig
from calibrex.core.frames import FrameGraph
from calibrex.core.geometry import SE3, rotate_vector_xyzw
from calibrex.data.inspect import DatasetInspection
from calibrex.solvers.acfr_board_edges import extract_acfr_vlp_board_edges
from calibrex.solvers.native_planar_board_solver import (
    NativePlanarBoardSolver,
    _read_acfr_vlp_rows,
    read_acfr_vlp_horn_point_observations,
    read_acfr_vlp_plane_observations,
    read_acfr_vlp_point_plane_observations,
)


def _write_acfr_poses(
    path: Path,
    transform: SE3,
    count: int = 20,
    *,
    corners: bool = True,
    lidar_scale: float = 1.0,
) -> list[int]:
    """Write ACFR-format captures; return the LiDAR loop's start corner per capture.

    Camera corners are written clockwise about the camera normal and LiDAR corners
    counter-clockwise about the LiDAR normal, as in ACFR's public file. The LiDAR
    loop starts at a random corner, so some captures take the board's symmetric
    labeling. ``lidar_scale`` stretches the LiDAR board's long side.
    """

    rng = np.random.default_rng(21)
    lines: list[str] = []
    starts: list[int] = []
    for index in range(count):
        lidar_normal = rng.normal(size=3)
        lidar_normal /= np.linalg.norm(lidar_normal)
        distance_m = float(rng.uniform(1.0, 4.0))
        lidar_center = -distance_m * lidar_normal
        camera_normal = np.asarray(
            rotate_vector_xyzw(transform.rotation_quat_xyzw, tuple(lidar_normal))
        )
        camera_center = np.asarray(transform.transform_point(lidar_center))
        u = np.cross(lidar_normal, rng.normal(size=3))
        u /= np.linalg.norm(u)
        v = np.cross(lidar_normal, u)
        signs = ((1, -1), (1, 1), (-1, 1), (-1, -1))
        loop = [lidar_center + a * 0.305 * u + b * 0.425 * v for a, b in signs]
        camera_corners = [np.asarray(transform.transform_point(tuple(c))) for c in loop][::-1]
        start = int(rng.integers(0, 4))
        starts.append(start)
        lidar_corners = [
            lidar_center + a * 0.305 * u + b * 0.425 * lidar_scale * v
            for a, b in (signs[(k + start) % 4] for k in range(4))
        ]
        if not corners:
            camera_corners = lidar_corners = [np.zeros(3)] * 4
        rows = [
            camera_center * 1000.0,
            camera_normal,
            *(corner * 1000.0 for corner in camera_corners),
            lidar_center * 1000.0,
            lidar_normal,
            *(corner * 1000.0 for corner in lidar_corners),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.asarray([distance_m, 0.0, 0.0]),
            np.ones(3),
            np.asarray([index + 1.0, 0.0, 0.0]),
        ]
        lines.extend(",".join(f"{value:.12g}" for value in row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return starts


def _config(dataset_path: Path) -> CalibrationConfig:
    return CalibrationConfig.model_validate(
        {
            "dataset": {"type": "filesystem", "path": str(dataset_path)},
            "sensors": {
                "camera0": {"type": "camera"},
                "lidar0": {"type": "lidar"},
            },
            "frames": {
                "camera0": {"root": True},
                "lidar0": {
                    "parent": "camera0",
                    "transform": {"estimate": True},
                },
            },
            "pipeline": {
                "factors": {
                    "planar_board_plane_correspondence": {
                        "enabled": True,
                        "options": {"normal_sign_policy": "preserve"},
                    }
                }
            },
            "solver": {"backend": "native_planar_board", "seed": 4},
        }
    )


def test_native_adapter_recovers_plane_transform_and_falsifies_controls(
    tmp_path: Path,
) -> None:
    half = math.radians(73.0) / 2.0
    truth = SE3((0.12, -0.08, 0.21), (0.0, math.sin(half), 0.0, math.cos(half)))
    _write_acfr_poses(tmp_path / "poses.csv", truth)
    config = _config(tmp_path)

    result = NativePlanarBoardSolver().solve(
        config,
        FrameGraph.from_config(config),
        DatasetInspection("filesystem", str(tmp_path), True),
    )

    assert result.status == "pass"
    assert result.observability is not None
    assert result.observability.rank == 3
    assert result.metrics["planar_board_normal_rmse_deg"].holdout is not None
    assert result.metrics["planar_board_normal_rmse_deg"].holdout < 1.0e-5
    assert result.metrics["planar_board_known_bad_detectable_fraction"].value == 1.0
    assert result.metrics["point_plane_center_rmse_m"].holdout is not None
    assert result.metrics["point_plane_center_rmse_m"].holdout < 1.0e-5
    assert result.metrics["point_plane_known_bad_detectable_fraction"].value == 1.0
    assert result.metrics["point_plane_joint_rank"].value == 6.0
    assert result.metrics["horn_point_rmse_m"].holdout is not None
    assert result.metrics["horn_point_rmse_m"].holdout < 1.0e-5
    assert result.metrics["horn_point_known_bad_detectable_fraction"].value == 1.0
    assert result.metrics["horn_point_joint_rank"].value == 6.0
    assert result.metrics["horn_point_quaternion_normalized_eigengap"].value is not None
    point_plane = result.provenance["native_point_plane_baseline"]
    assert point_plane["result"]["method"] == "verma_center_normal_procrustes_irls/v0.1"
    assert (
        point_plane["result"]["train_frame_ids"]
        == result.provenance["native_planar_board"]["result"]["train_frame_ids"]
    )
    horn = result.provenance["native_horn_point_baseline"]
    assert horn["result"]["method"] == "horn_unit_quaternion_point_alignment_irls/v0.1"
    assert horn["evidence_pass"] is True
    assert horn["common_capture_split"]["matches_plane_only"] is True
    assert horn["common_capture_split"]["matches_point_plane"] is True
    assert result.metrics["line_plane_matched_capture_fraction"].value == 1.0
    assert result.metrics["line_plane_boundary_distance_rmse_m"].holdout is not None
    assert result.metrics["line_plane_boundary_distance_rmse_m"].holdout < 1.0e-8
    assert result.metrics["line_plane_known_bad_detectable_fraction"].value == 1.0
    line_plane = result.provenance["native_line_plane_baseline"]
    assert line_plane["evidence_pass"] is True
    assert line_plane["common_capture_split"]["matches_plane_only"] is True
    assert line_plane["result"]["method"] == "zhou_line_plane_closed_form/v0.1"
    assert len(line_plane["result"]["known_bad_probes"]) == 12
    line_estimate = line_plane["result"]["transform_camera_lidar"]["translation_m"]
    assert np.linalg.norm(np.asarray(line_estimate) - truth.translation_m) < 1.0e-8
    estimated = result.transforms["T_camera0_lidar0"]
    assert np.linalg.norm(np.asarray(estimated.translation_m) - truth.translation_m) < 1.0e-8
    assert result.provenance["metrics_origin"] == "recomputed"
    assert result.provenance["data_verified"] is True


def test_acfr_reader_rejects_incomplete_capture(tmp_path: Path) -> None:
    path = tmp_path / "poses.csv"
    path.write_text("1,2,3\n", encoding="utf-8")

    try:
        read_acfr_vlp_plane_observations(path)
    except ValueError as exc:
        assert "complete 19-row captures" in str(exc)
    else:
        raise AssertionError("incomplete ACFR capture was accepted")

    try:
        read_acfr_vlp_point_plane_observations(path)
    except ValueError as exc:
        assert "complete 19-row captures" in str(exc)
    else:
        raise AssertionError("incomplete ACFR point+plane capture was accepted")

    try:
        read_acfr_vlp_horn_point_observations(path)
    except ValueError as exc:
        assert "complete 19-row captures" in str(exc)
    else:
        raise AssertionError("incomplete ACFR Horn point capture was accepted")


def _truth() -> SE3:
    half = math.radians(73.0) / 2.0
    return SE3((0.12, -0.08, 0.21), (0.0, math.sin(half), 0.0, math.cos(half)))


def _rotated(transform: SE3, degrees: float) -> SE3:
    half = math.radians(degrees) / 2.0
    return SE3((0.0, 0.0, 0.0), (math.sin(half), 0.0, 0.0, math.cos(half))).compose(transform)


def test_edge_extractor_orders_corners_and_resolves_board_symmetry(tmp_path: Path) -> None:
    truth = _truth()
    starts = _write_acfr_poses(tmp_path / "poses.csv", truth)
    rows = _read_acfr_vlp_rows(tmp_path / "poses.csv")

    # A prior 20° off still resolves every capture; the symmetric labeling is
    # chosen exactly where the LiDAR loop starts at the opposite corner.
    extraction = extract_acfr_vlp_board_edges(rows, _rotated(truth, 20.0))

    assert len(extraction.observations) == 20
    assert extraction.rejected_frame_ids == ()
    assert {item.frame_id: item.lidar_shift for item in extraction.captures} == {
        f"acfr-vlp-{index + 1:03d}": (4 - start) % 4 for index, start in enumerate(starts)
    }
    assert extraction.symmetric_frame_ids == tuple(
        f"acfr-vlp-{index + 1:03d}" for index, start in enumerate(starts) if start == 2
    )
    for item in extraction.observations:
        for boundary in item.boundaries:
            camera = np.asarray(boundary.camera_line.direction)
            lidar = np.asarray(
                rotate_vector_xyzw(truth.rotation_quat_xyzw, boundary.lidar_line.direction)
            )
            assert camera @ lidar > 1.0 - 1.0e-9
    assert extraction.as_dict()["source_rows"] == {
        "camera_corners": [2, 3, 4, 5],
        "lidar_corners": [8, 9, 10, 11],
        "rows_per_capture": 19,
    }


def test_edge_extractor_reports_degenerate_and_mismatched_boards(tmp_path: Path) -> None:
    truth = _truth()
    _write_acfr_poses(tmp_path / "zero.csv", truth, count=4, corners=False)
    zero = extract_acfr_vlp_board_edges(_read_acfr_vlp_rows(tmp_path / "zero.csv"), truth)
    assert zero.observations == ()
    assert {item.status for item in zero.captures} == {"degenerate_corners"}

    # A LiDAR board 25 % longer than the camera board is not the same target.
    _write_acfr_poses(tmp_path / "long.csv", truth, count=4, lidar_scale=1.25)
    long = extract_acfr_vlp_board_edges(_read_acfr_vlp_rows(tmp_path / "long.csv"), truth)
    assert long.observations == ()
    assert {item.status for item in long.captures} == {"edge_length_mismatch"}


def test_backend_reports_line_plane_row_when_corners_are_missing(tmp_path: Path) -> None:
    _write_acfr_poses(tmp_path / "poses.csv", _truth(), corners=False)
    config = _config(tmp_path)

    result = NativePlanarBoardSolver().solve(
        config,
        FrameGraph.from_config(config),
        DatasetInspection("filesystem", str(tmp_path), True),
    )

    # The plane rows are untouched; the line+plane row fails honestly.
    assert result.status == "pass"
    assert result.metrics["line_plane_matched_capture_fraction"].value == 0.0
    line_plane = result.provenance["native_line_plane_baseline"]
    assert line_plane["evidence_pass"] is False
    assert line_plane["result"]["status"] == "insufficient_constraints"
    assert "line+plane independent baseline did not pass all default gates" in result.warnings
