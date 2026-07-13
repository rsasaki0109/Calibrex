import math
from pathlib import Path

import numpy as np

from calibrex.core.config import CalibrationConfig
from calibrex.core.frames import FrameGraph
from calibrex.core.geometry import SE3, rotate_vector_xyzw
from calibrex.data.inspect import DatasetInspection
from calibrex.solvers.native_planar_board_solver import (
    NativePlanarBoardSolver,
    read_acfr_vlp_plane_observations,
)


def _write_acfr_poses(path: Path, transform: SE3, count: int = 20) -> None:
    rng = np.random.default_rng(21)
    lines: list[str] = []
    for index in range(count):
        lidar_normal = rng.normal(size=3)
        lidar_normal /= np.linalg.norm(lidar_normal)
        distance_m = float(rng.uniform(1.0, 4.0))
        lidar_center = -distance_m * lidar_normal
        camera_normal = np.asarray(
            rotate_vector_xyzw(transform.rotation_quat_xyzw, tuple(lidar_normal))
        )
        camera_center = np.asarray(transform.transform_point(lidar_center))
        rows = [
            camera_center * 1000.0,
            camera_normal,
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            lidar_center * 1000.0,
            lidar_normal,
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
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
