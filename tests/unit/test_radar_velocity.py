"""Tests for nuScenes radar PCD reading and velocity-consistency metrics."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import pytest

from slac.core.geometry import SE3
from slac.data.nuscenes_radar import read_nuscenes_radar_pcd, write_nuscenes_radar_pcd
from slac.evaluation.radar import (
    RadarVelocityConsistencyStats,
    score_radar_velocity_frame,
    summarize_nuscenes_radar_velocity_consistency,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _synthetic_static_radar_fields(
    *,
    ego_speed_mps: float,
    azimuths_deg: tuple[float, ...] = (0.0, 15.0, -20.0, 30.0, -35.0, 10.0),
    t_ego_radar: SE3 | None = None,
) -> dict[str, list[float]]:
    extrinsic = t_ego_radar or SE3.identity()
    inv_rotation = extrinsic.rotation_quat_xyzw
    from slac.core.geometry import quaternion_conjugate_xyzw, rotate_vector_xyzw

    v_radar = rotate_vector_xyzw(
        quaternion_conjugate_xyzw(inv_rotation),
        (ego_speed_mps, 0.0, 0.0),
    )
    vx_value = -v_radar[0]
    vy_value = -v_radar[1]
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    vxs: list[float] = []
    vys: list[float] = []
    dyn_props: list[int] = []
    vx_comps: list[float] = []
    vy_comps: list[float] = []
    for azimuth_deg in azimuths_deg:
        azimuth = math.radians(azimuth_deg)
        x = 20.0 * math.cos(azimuth)
        y = 20.0 * math.sin(azimuth)
        z = 0.5
        xs.append(x)
        ys.append(y)
        zs.append(z)
        vxs.append(vx_value)
        vys.append(vy_value)
        dyn_props.append(1)
        vx_comps.append(0.0)
        vy_comps.append(0.0)
    return {
        "x": xs,
        "y": ys,
        "z": zs,
        "vx": vxs,
        "vy": vys,
        "dyn_prop": dyn_props,
        "vx_comp": vx_comps,
        "vy_comp": vy_comps,
        "invalid_state": [0] * len(xs),
        "ambig_state": [3] * len(xs),
    }


def _write_nuscenes_radar_fixture(
    root: Path,
    *,
    ego_speed_mps: float,
    radar_fields: dict[str, list[float]] | None = None,
    second_pose_translation: tuple[float, float, float] | None = None,
) -> None:
    version = root / "v1.0-mini"
    version.mkdir(parents=True)
    radar_dir = root / "samples" / "RADAR_FRONT"
    radar_dir.mkdir(parents=True)
    radar_path = radar_dir / "000.pcd"
    fields = radar_fields or _synthetic_static_radar_fields(ego_speed_mps=ego_speed_mps)
    write_nuscenes_radar_pcd(radar_path, fields)

    _write_json(
        version / "sensor.json",
        [{"token": "sensor_radar", "channel": "RADAR_FRONT", "modality": "radar"}],
    )
    _write_json(
        version / "calibrated_sensor.json",
        [
            {
                "token": "cal_radar",
                "sensor_token": "sensor_radar",
                "translation": [2.0, 0.0, 0.5],
                "rotation": [1.0, 0.0, 0.0, 0.0],
            }
        ],
    )
    _write_json(version / "scene.json", [{"token": "scene0", "name": "scene-0001"}])
    _write_json(version / "sample.json", [{"token": "sample0", "scene_token": "scene0"}])
    translation_1 = (0.0, 0.0, 0.0)
    dt_s = 0.1
    translation_2 = second_pose_translation or (
        ego_speed_mps * dt_s,
        0.0,
        0.0,
    )
    _write_json(
        version / "ego_pose.json",
        [
            {
                "token": "ego0",
                "translation": list(translation_1),
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "timestamp": 1_000_000,
            },
            {
                "token": "ego1",
                "translation": list(translation_2),
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "timestamp": 1_100_000,
            },
        ],
    )
    _write_json(
        version / "sample_data.json",
        [
            {
                "token": "sd_radar0",
                "sample_token": "sample0",
                "ego_pose_token": "ego0",
                "calibrated_sensor_token": "cal_radar",
                "filename": "samples/RADAR_FRONT/000.pcd",
                "timestamp": 1_000_000,
                "is_key_frame": True,
            },
            {
                "token": "sd_radar1",
                "sample_token": "sample0",
                "ego_pose_token": "ego1",
                "calibrated_sensor_token": "cal_radar",
                "filename": "samples/RADAR_FRONT/000.pcd",
                "timestamp": 1_100_000,
                "is_key_frame": True,
            },
        ],
    )


def test_read_nuscenes_radar_pcd_from_constructed_bytes(tmp_path: Path) -> None:
    fields = _synthetic_static_radar_fields(ego_speed_mps=8.0)
    path = tmp_path / "radar.pcd"
    write_nuscenes_radar_pcd(path, fields)

    decoded = read_nuscenes_radar_pcd(path)
    assert decoded.point_count == len(fields["x"])
    assert decoded.array("x").shape[0] == len(fields["x"])
    assert float(decoded.array("vx")[0]) == pytest.approx(fields["vx"][0])
    assert float(decoded.array("dyn_prop")[2]) == 1.0


def test_score_radar_velocity_frame_correct_extrinsic_near_zero(tmp_path: Path) -> None:
    fields = _synthetic_static_radar_fields(ego_speed_mps=10.0)
    path = tmp_path / "radar.pcd"
    write_nuscenes_radar_pcd(path, fields)
    pcd = read_nuscenes_radar_pcd(path)

    stats = score_radar_velocity_frame(
        pcd=pcd,
        ego_velocity_mps=(10.0, 0.0, 0.0),
        t_ego_radar=SE3.identity(),
    )
    assert stats is not None
    assert stats.inlier_count >= 5
    assert max(stats.abs_residuals_mps) < 0.05


def test_score_radar_velocity_frame_yaw_perturbation_worse(tmp_path: Path) -> None:
    fields = _synthetic_static_radar_fields(ego_speed_mps=10.0)
    path = tmp_path / "radar.pcd"
    write_nuscenes_radar_pcd(path, fields)
    pcd = read_nuscenes_radar_pcd(path)

    baseline = score_radar_velocity_frame(
        pcd=pcd,
        ego_velocity_mps=(10.0, 0.0, 0.0),
        t_ego_radar=SE3.identity(),
    )
    assert baseline is not None
    yaw_rad = math.radians(5.0)
    half = yaw_rad / 2.0
    perturbed_extrinsic = SE3((0.0, 0.0, 0.0), (0.0, 0.0, math.sin(half), math.cos(half)))
    perturbed = score_radar_velocity_frame(
        pcd=pcd,
        ego_velocity_mps=(10.0, 0.0, 0.0),
        t_ego_radar=perturbed_extrinsic,
    )
    assert perturbed is not None
    assert statistics.median(perturbed.abs_residuals_mps) > statistics.median(
        baseline.abs_residuals_mps
    ) + 0.1


def test_summarize_radar_velocity_degenerate_cases(tmp_path: Path) -> None:
    def identity_lookup(_sensor: str) -> SE3:
        return SE3.identity()

    _write_nuscenes_radar_fixture(tmp_path / "no_motion", ego_speed_mps=0.0)
    stats, _ = summarize_nuscenes_radar_velocity_consistency(
        dataset_path=tmp_path / "no_motion",
        radar_sensors={"radar_front": "RADAR_FRONT"},
        extrinsics_lookup=identity_lookup,
    )
    assert stats.status == "inconclusive"
    assert stats.median_abs_residual_mps is None

    empty_fields = _synthetic_static_radar_fields(ego_speed_mps=10.0)
    empty_fields["x"] = []
    empty_fields["y"] = []
    empty_fields["z"] = []
    empty_fields["vx"] = []
    empty_fields["vy"] = []
    empty_fields["dyn_prop"] = []
    empty_fields["vx_comp"] = []
    empty_fields["vy_comp"] = []
    empty_fields["invalid_state"] = []
    empty_fields["ambig_state"] = []
    _write_nuscenes_radar_fixture(
        tmp_path / "no_returns",
        ego_speed_mps=10.0,
        radar_fields=empty_fields,
    )
    stats_empty, _ = summarize_nuscenes_radar_velocity_consistency(
        dataset_path=tmp_path / "no_returns",
        radar_sensors={"radar_front": "RADAR_FRONT"},
        extrinsics_lookup=identity_lookup,
    )
    assert stats_empty.status == "inconclusive"


def test_summarize_radar_velocity_end_to_end_with_perturbation_probe(tmp_path: Path) -> None:
    _write_nuscenes_radar_fixture(tmp_path, ego_speed_mps=12.0)
    stats, provenance = summarize_nuscenes_radar_velocity_consistency(
        dataset_path=tmp_path,
        radar_sensors={"radar_front": "RADAR_FRONT"},
        extrinsics_lookup=lambda _sensor: SE3.identity(),
    )
    assert isinstance(stats, RadarVelocityConsistencyStats)
    assert stats.status == "scored"
    assert stats.median_abs_residual_mps is not None
    assert stats.median_abs_residual_mps < 0.05
    perturbation = provenance.get("perturbation_probe")
    assert isinstance(perturbation, dict)
    assert perturbation.get("detectable") is True
