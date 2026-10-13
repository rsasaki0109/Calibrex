"""Planar-board line+plane on ACFR's public VLP-16 captures (opt-in).

Needs ``poses.csv`` from ``tools/download_public_dataset.py acfr_vlp_plane_poses``
under ``data/public`` or at ``CALIBREX_ACFR_VLP_POSES``. The gates are the
configured defaults (0.03 m, 3°, 80 % of 12 controls); none is tuned here.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from calibrex.cli.main import main
from calibrex.core.provenance import sha256_path
from calibrex.solvers.acfr_board_edges import extract_acfr_vlp_board_edges
from calibrex.solvers.native_planar_board_solver import (
    _read_acfr_vlp_rows,
    read_acfr_vlp_plane_observations,
)
from calibrex.solvers.planar_board_lidar_camera_solver import (
    PlanarBoardLidarCameraSolver,
    PlanarBoardSolverOptions,
)
from calibrex.solvers.planar_board_line_plane_solver import (
    LinePlaneBoardObservation,
    PlanarBoardLinePlaneSolver,
)

POSES = Path(
    os.environ.get("CALIBREX_ACFR_VLP_POSES", "data/public/acfr_vlp_plane_poses/poses.csv")
)
POSES_SHA256 = "024bc6ed9009652761e9c0df49b106d325a10c88d41a80e9b36ea55fd567e110"
CONFIG = Path("examples/public_datasets/acfr_vlp_plane_poses/config.yaml")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not POSES.is_file(),
        reason="download acfr_vlp_plane_poses or set CALIBREX_ACFR_VLP_POSES",
    ),
]


def test_cli_line_plane_row_passes_its_gates_on_the_shared_split(tmp_path: Path) -> None:
    assert sha256_path(POSES) == POSES_SHA256
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["dataset"]["path"] = str(POSES.parent.resolve())
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "out"

    assert main(["calibrate", str(config_path), "--output-dir", str(output)]) == 0
    result = yaml.safe_load((output / "result.yaml").read_text(encoding="utf-8"))
    metrics = result["metrics"]
    assert metrics["line_plane_matched_capture_fraction"]["value"] == 1.0
    assert metrics["line_plane_boundary_distance_rmse_m"]["holdout"] <= 0.03
    assert metrics["line_plane_boundary_direction_rmse_deg"]["holdout"] <= 3.0
    assert metrics["line_plane_known_bad_detectable_fraction"]["value"] >= 0.8
    assert main(["validate", str(output / "result.yaml")]) == 0


def test_unresolved_board_symmetry_fails_the_boundary_gate() -> None:
    plane = PlanarBoardLidarCameraSolver().solve(
        read_acfr_vlp_plane_observations(POSES),
        options=PlanarBoardSolverOptions(normal_sign_policy="preserve"),
    )
    assert plane.transform_camera_lidar is not None
    extraction = extract_acfr_vlp_board_edges(
        _read_acfr_vlp_rows(POSES), plane.transform_camera_lidar
    )
    assert extraction.symmetric_frame_ids

    def source_order(item: LinePlaneBoardObservation) -> LinePlaneBoardObservation:
        if item.frame_id not in extraction.symmetric_frame_ids:
            return item
        edges = item.boundaries
        return replace(
            item,
            boundaries=tuple(
                replace(edges[k], lidar_line=edges[(k + 2) % 4].lidar_line) for k in range(4)
            ),
        )

    resolved = PlanarBoardLinePlaneSolver().solve(extraction.observations)
    naive = PlanarBoardLinePlaneSolver().solve(
        [source_order(item) for item in extraction.observations]
    )
    assert resolved.holdout_evaluation.boundary_distance_rmse_m is not None
    assert naive.holdout_evaluation.boundary_distance_rmse_m is not None
    assert resolved.holdout_evaluation.boundary_distance_rmse_m <= 0.03
    assert naive.holdout_evaluation.boundary_distance_rmse_m > 0.03
