from pathlib import Path

import numpy as np
import pytest

from calibrex.data.nuscenes_radar import NuScenesRadarPCD
from calibrex.evaluation.radar_spatiotemporal import (
    estimate_planar_radar_ego_velocity,
    summarize_nuscenes_radar_spatiotemporal,
)
from calibrex.solvers.radar_spatiotemporal_lever_arm_solver import (
    RadarSpatiotemporalLeverArmOptions,
)


def _planar_static_pcd(velocity: tuple[float, float]) -> NuScenesRadarPCD:
    directions = np.asarray(
        [
            (1.0, 0.0),
            (0.0, 1.0),
            (-1.0, 0.0),
            (0.0, -1.0),
            (0.707, 0.707),
            (-0.707, 0.707),
            (-0.707, -0.707),
            (0.707, -0.707),
        ]
    )
    positions = directions * 10.0
    radial = -(directions @ np.asarray(velocity))
    vx = radial * directions[:, 0]
    vy = radial * directions[:, 1]
    count = len(directions)
    return NuScenesRadarPCD(
        fields={
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": np.zeros(count),
            "vx": vx,
            "vy": vy,
            "vx_comp": np.zeros(count),
            "vy_comp": np.zeros(count),
            "dyn_prop": np.ones(count),
        },
        point_count=count,
    )


def test_planar_scan_velocity_is_recomputed_from_raw_doppler() -> None:
    velocity = estimate_planar_radar_ego_velocity(_planar_static_pcd((7.2, -1.4)))

    assert velocity == pytest.approx((7.2, -1.4, 0.0), abs=2.0e-3)


def test_planar_scan_velocity_rejects_single_los_geometry() -> None:
    pcd = _planar_static_pcd((5.0, 0.0))
    pcd.fields["y"] = np.zeros(pcd.point_count)
    pcd.fields["vy"] = np.zeros(pcd.point_count)

    assert estimate_planar_radar_ego_velocity(pcd) is None


def test_nuscenes_adapter_reports_missing_public_download(tmp_path: Path) -> None:
    evidence = summarize_nuscenes_radar_spatiotemporal(
        dataset_path=tmp_path,
        channel="RADAR_FRONT",
        rotation_ego_radar_xyzw=(0.0, 0.0, 0.0, 1.0),
        options=RadarSpatiotemporalLeverArmOptions(),
    )

    assert evidence.status == "unavailable"
    assert evidence.result is None
    assert evidence.raw_input_files == ()
