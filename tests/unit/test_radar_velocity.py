"""Tests for nuScenes radar PCD reading and velocity-consistency metrics."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import pytest

from calibrex.core.geometry import SE3
from calibrex.data.nuscenes_radar import read_nuscenes_radar_pcd, write_nuscenes_radar_pcd
from calibrex.evaluation import radar as radar_module
from calibrex.evaluation.radar import (
    RadarStaticReturnCounts,
    RadarStaticReturnDiagnostics,
    RadarVelocityConsistencyStats,
    RadarVelocityObservation,
    RadarVelocityPolicy,
    RadarYawObservabilityStats,
    aggregate_radar_sensor_assessments,
    assess_radar_velocity_policy,
    compute_radar_velocity_residuals,
    score_radar_velocity_frame,
    split_radar_velocity_residuals,
    summarize_nuscenes_radar_velocity_consistency,
    summarize_radar_static_returns,
    summarize_radar_yaw_observability,
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
    from calibrex.core.geometry import quaternion_conjugate_xyzw, rotate_vector_xyzw

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
    eligible_frame_count: int = 1,
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
    dt_s = 0.1
    pose_count = eligible_frame_count + 1
    ego_poses = []
    sample_data = []
    for index in range(pose_count):
        translation = (ego_speed_mps * dt_s * index, 0.0, 0.0)
        if index == 1 and second_pose_translation is not None:
            translation = second_pose_translation
        timestamp = 1_000_000 + 100_000 * index
        ego_poses.append(
            {
                "token": f"ego{index}",
                "translation": list(translation),
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "timestamp": timestamp,
            }
        )
        sample_data.append(
            {
                "token": f"sd_radar{index}",
                "sample_token": "sample0",
                "ego_pose_token": f"ego{index}",
                "calibrated_sensor_token": "cal_radar",
                "filename": "samples/RADAR_FRONT/000.pcd",
                "timestamp": timestamp,
                "is_key_frame": True,
            }
        )
    _write_json(
        version / "ego_pose.json",
        ego_poses,
    )
    _write_json(version / "sample_data.json", sample_data)


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


def test_pure_radar_velocity_residuals_score_candidate_rotation() -> None:
    observations = tuple(
        RadarVelocityObservation(
            frame_id="frame-0",
            line_of_sight_radar=(math.cos(angle), math.sin(angle), 0.0),
            measured_radial_velocity_mps=-10.0 * math.cos(angle),
            ego_velocity_mps=(10.0, 0.0, 0.0),
        )
        for angle in (0.0, 0.3, -0.5)
    )

    baseline = compute_radar_velocity_residuals(
        observations,
        t_ego_radar=SE3.identity(),
    )
    yaw = math.radians(10.0)
    perturbed = compute_radar_velocity_residuals(
        observations,
        t_ego_radar=SE3(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)),
        ),
    )

    assert max(abs(item.residual_mps) for item in baseline) < 1e-9
    assert statistics.median(abs(item.residual_mps) for item in perturbed) > 0.1


def test_radar_velocity_split_is_deterministic_and_frame_disjoint() -> None:
    observations = tuple(
        RadarVelocityObservation(
            frame_id=f"frame-{frame_index}",
            line_of_sight_radar=(1.0, 0.0, 0.0),
            measured_radial_velocity_mps=-5.0,
            ego_velocity_mps=(5.0, 0.0, 0.0),
        )
        for frame_index in range(10)
        for _return_index in range(3)
    )
    residuals = compute_radar_velocity_residuals(
        observations,
        t_ego_radar=SE3.identity(),
    )

    first = split_radar_velocity_residuals(residuals, holdout_ratio=0.2, seed=7)
    second = split_radar_velocity_residuals(reversed(residuals), holdout_ratio=0.2, seed=7)

    assert first.train_frame_ids == second.train_frame_ids
    assert first.holdout_frame_ids == second.holdout_frame_ids
    assert len(first.holdout_frame_ids) == 2
    assert set(first.train_frame_ids).isdisjoint(first.holdout_frame_ids)
    assert {item.frame_id for item in first.train} == set(first.train_frame_ids)
    assert {item.frame_id for item in first.holdout} == set(first.holdout_frame_ids)


def test_pure_radar_velocity_residuals_reject_zero_line_of_sight() -> None:
    observation = RadarVelocityObservation(
        frame_id="frame-0",
        line_of_sight_radar=(0.0, 0.0, 0.0),
        measured_radial_velocity_mps=0.0,
        ego_velocity_mps=(1.0, 0.0, 0.0),
    )

    with pytest.raises(ValueError, match="line of sight"):
        compute_radar_velocity_residuals(
            (observation,),
            t_ego_radar=SE3.identity(),
        )


def test_static_return_diagnostics_include_zero_static_frames() -> None:
    diagnostics = summarize_radar_static_returns(
        (
            RadarStaticReturnCounts("frame-0", 10, 8, 6),
            RadarStaticReturnCounts("frame-1", 10, 8, 0),
        )
    )

    assert diagnostics.status == "supported"
    assert diagnostics.frame_count == 2
    assert diagnostics.static_return_count == 6
    assert diagnostics.static_fraction == pytest.approx(0.375)


def test_yaw_observability_distinguishes_supported_and_degenerate_geometry() -> None:
    supported_observations = tuple(
        RadarVelocityObservation(
            frame_id=f"frame-{frame_index}",
            line_of_sight_radar=(math.cos(angle), math.sin(angle), 0.0),
            measured_radial_velocity_mps=0.0,
            ego_velocity_mps=(10.0, 0.0, 0.0),
        )
        for frame_index in range(2)
        for angle in (-0.7, -0.3, 0.3, 0.7)
    )
    supported = summarize_radar_yaw_observability(
        supported_observations,
        t_ego_radar=SE3.identity(),
        frame_ids=("frame-1",),
        min_observation_count=4,
    )
    assert supported.status == "supported"
    assert supported.frame_count == 1
    assert supported.observation_count == 4
    assert supported.yaw_sensitivity_rms_mps_per_rad is not None
    assert supported.yaw_sensitivity_rms_mps_per_rad > 2.0

    degenerate_observations = tuple(
        RadarVelocityObservation(
            frame_id="frame-0",
            line_of_sight_radar=(0.0, 0.0, 1.0),
            measured_radial_velocity_mps=0.0,
            ego_velocity_mps=(10.0, 0.0, 0.0),
        )
        for _ in range(5)
    )
    degenerate = summarize_radar_yaw_observability(
        degenerate_observations,
        t_ego_radar=SE3.identity(),
    )
    assert degenerate.status == "inconclusive"
    assert degenerate.yaw_sensitivity_rms_mps_per_rad == 0.0


def test_summarize_radar_velocity_degenerate_cases(tmp_path: Path) -> None:
    def identity_lookup(_sensor: str) -> SE3:
        return SE3.identity()

    _write_nuscenes_radar_fixture(tmp_path / "no_motion", ego_speed_mps=0.0)
    stats, no_motion_provenance = summarize_nuscenes_radar_velocity_consistency(
        dataset_path=tmp_path / "no_motion",
        radar_sensors={"radar_front": "RADAR_FRONT"},
        extrinsics_lookup=identity_lookup,
    )
    assert stats.status == "inconclusive"
    assert stats.median_abs_residual_mps is None
    no_motion_diagnostics = no_motion_provenance["input_diagnostics"]
    assert no_motion_diagnostics["attempted_transition_count"] == 1
    assert no_motion_diagnostics["accepted_frame_count"] == 0
    assert no_motion_diagnostics["counts_by_code"] == {"insufficient_ego_motion": 1}

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
    stats_empty, empty_provenance = summarize_nuscenes_radar_velocity_consistency(
        dataset_path=tmp_path / "no_returns",
        radar_sensors={"radar_front": "RADAR_FRONT"},
        extrinsics_lookup=identity_lookup,
    )
    assert stats_empty.status == "inconclusive"
    empty_support = empty_provenance["static_return_diagnostics"]
    assert empty_support["status"] == "inconclusive"
    assert empty_support["static_return_count"] == 0
    assert empty_support["static_fraction"] is None
    empty_diagnostics = empty_provenance["input_diagnostics"]
    assert empty_diagnostics["counts_by_code"] == {"no_eligible_static_returns": 1}


@pytest.mark.parametrize("failure", ["missing", "corrupt"])
def test_radar_input_failures_are_provenanced_without_absolute_paths(
    tmp_path: Path,
    failure: str,
) -> None:
    _write_nuscenes_radar_fixture(tmp_path, ego_speed_mps=10.0)
    radar_path = tmp_path / "samples" / "RADAR_FRONT" / "000.pcd"
    expected_code = "payload_not_found"
    if failure == "missing":
        radar_path.unlink()
    else:
        radar_path.write_bytes(b"not a radar pcd")
        expected_code = "payload_read_error"

    stats, provenance = summarize_nuscenes_radar_velocity_consistency(
        dataset_path=tmp_path,
        radar_sensors={"radar_front": "RADAR_FRONT"},
        extrinsics_lookup=lambda _sensor: SE3.identity(),
    )

    assert stats.status == "inconclusive"
    diagnostics = provenance["input_diagnostics"]
    assert diagnostics["attempted_transition_count"] == 1
    assert diagnostics["accepted_frame_count"] == 0
    assert diagnostics["counts_by_code"] == {expected_code: 1}
    event = diagnostics["events"][0]
    assert event["payload_ref"] == "samples/RADAR_FRONT/000.pcd"
    assert str(tmp_path) not in str(event)


def test_summarize_radar_velocity_end_to_end_with_perturbation_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_nuscenes_radar_fixture(tmp_path, ego_speed_mps=12.0)
    original_read = radar_module.read_nuscenes_radar_pcd
    read_paths: list[Path] = []

    def counted_read(path: str | Path) -> object:
        read_paths.append(Path(path))
        return original_read(path)

    monkeypatch.setattr(radar_module, "read_nuscenes_radar_pcd", counted_read)
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
    assert len(read_paths) == 1
    split = provenance.get("split")
    assert isinstance(split, dict)
    assert split["policy"] == "seeded_frame_holdout/v0.1"
    assert perturbation["comparison_support"] == "all_fallback_no_holdout"
    assert perturbation["frame_ids"] == split["train_frame_ids"]
    summaries = provenance.get("residual_summary")
    assert isinstance(summaries, dict)
    assert summaries["train"]["count"] == 6
    assert summaries["train"]["median_abs_residual_mps"] < 0.05
    assert summaries["holdout"] == {
        "count": 0,
        "median_abs_residual_mps": None,
        "rmse_mps": None,
    }
    challenge = provenance.get("known_bad_rotation_probes")
    assert isinstance(challenge, dict)
    assert challenge["challenge_id"] == "radar_lidar_mandatory_yaw_controls/v0.1"
    assert challenge["mandatory_amounts_deg"] == [-10.0, -5.0, 5.0, 10.0]
    assert challenge["required_count"] == 4
    assert challenge["scored_count"] == 4
    assert challenge["supported_count"] == 0
    assert challenge["detectable_count"] == 0
    assert challenge["detectable_fraction"] is None
    probes = challenge["probes"]
    assert [probe["amount_deg"] for probe in probes] == [-10.0, -5.0, 5.0, 10.0]
    assert all(probe["frame_ids"] == split["train_frame_ids"] for probe in probes)
    assert all(probe["supported"] is False for probe in probes)
    assert all(probe["detectable"] is None for probe in probes)
    assert all(probe["diagnostic_detectable"] is True for probe in probes)


def test_mandatory_radar_probes_share_holdout_and_are_detectable(tmp_path: Path) -> None:
    _write_nuscenes_radar_fixture(
        tmp_path,
        ego_speed_mps=12.0,
        eligible_frame_count=10,
    )

    stats, provenance = summarize_nuscenes_radar_velocity_consistency(
        dataset_path=tmp_path,
        radar_sensors={"radar_front": "RADAR_FRONT"},
        extrinsics_lookup=lambda _sensor: SE3.identity(),
        holdout_ratio=0.2,
        split_seed=7,
    )

    assert stats.status == "scored"
    split = provenance["split"]
    assert len(split["train_frame_ids"]) == 8
    assert len(split["holdout_frame_ids"]) == 2
    summaries = provenance["residual_summary"]
    assert summaries["train"]["count"] == 48
    assert summaries["holdout"]["count"] == 12
    assert summaries["train"]["median_abs_residual_mps"] < 0.05
    assert summaries["holdout"]["median_abs_residual_mps"] < 0.05
    challenge = provenance["known_bad_rotation_probes"]
    assert challenge["required_count"] == 4
    assert challenge["supported_count"] == 4
    assert challenge["detectable_count"] == 4
    assert challenge["detectable_fraction"] == 1.0
    assert all(probe["supported"] is True for probe in challenge["probes"])
    assert all(probe["detectable"] is True for probe in challenge["probes"])
    assert all(
        probe["frame_ids"] == split["holdout_frame_ids"]
        for probe in challenge["probes"]
    )
    static_diagnostics = provenance["static_return_diagnostics"]
    assert static_diagnostics["static_return_count"] == 60
    assert static_diagnostics["static_fraction"] == 1.0
    observability = provenance["holdout_yaw_observability"]["radar_front"]
    assert observability["status"] == "supported"
    assert observability["frame_count"] == 2
    assert observability["observation_count"] == 12
    assert observability["translation_observable"] is False


def test_radar_velocity_policy_pass_fail_and_inconclusive_precedence() -> None:
    static = RadarStaticReturnDiagnostics(
        frame_count=5,
        total_return_count=100,
        range_eligible_count=80,
        static_return_count=60,
        static_fraction=0.75,
        status="supported",
        reason=None,
    )
    observable = RadarYawObservabilityStats(
        frame_count=5,
        observation_count=60,
        yaw_sensitivity_rms_mps_per_rad=2.0,
        yaw_sensitivity_median_abs_mps_per_rad=1.5,
        status="supported",
        reason=None,
    )
    policy = RadarVelocityPolicy(
        max_holdout_median_abs_residual_mps=0.5,
        max_holdout_rmse_mps=0.8,
    )
    common = {
        "holdout_static": static,
        "observability_by_sensor": {"radar_front": observable},
        "mandatory_required_count": 4,
        "mandatory_supported_count": 4,
        "mandatory_detectable_fraction": 1.0,
        "policy": policy,
    }

    passed = assess_radar_velocity_policy(
        holdout_summary={"median_abs_residual_mps": 0.4, "rmse_mps": 0.7},
        **common,
    )
    failed = assess_radar_velocity_policy(
        holdout_summary={"median_abs_residual_mps": 0.6, "rmse_mps": 0.7},
        **common,
    )
    inconclusive = assess_radar_velocity_policy(
        holdout_summary={"median_abs_residual_mps": 0.6, "rmse_mps": 0.9},
        **{**common, "mandatory_supported_count": 3},
    )

    assert passed.status == "pass"
    assert failed.status == "fail"
    assert inconclusive.status == "inconclusive"
    assert inconclusive.rules[-1].status == "fail"


def test_radar_velocity_policy_requires_declared_residual_thresholds() -> None:
    assessment = assess_radar_velocity_policy(
        holdout_summary={"median_abs_residual_mps": 0.0, "rmse_mps": 0.0},
        holdout_static=RadarStaticReturnDiagnostics(
            frame_count=5,
            total_return_count=50,
            range_eligible_count=50,
            static_return_count=50,
            static_fraction=1.0,
            status="supported",
            reason=None,
        ),
        observability_by_sensor={
            "radar_front": RadarYawObservabilityStats(
                frame_count=5,
                observation_count=50,
                yaw_sensitivity_rms_mps_per_rad=2.0,
                yaw_sensitivity_median_abs_mps_per_rad=1.0,
                status="supported",
                reason=None,
            )
        },
        mandatory_required_count=4,
        mandatory_supported_count=4,
        mandatory_detectable_fraction=1.0,
        policy=RadarVelocityPolicy(),
    )

    assert assessment.status == "inconclusive"
    assert assessment.rules[-1].rule_id == "holdout_velocity_consistency"


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        (("pass", "pass"), "pass"),
        (("pass", "inconclusive"), "inconclusive"),
        (("pass", "fail"), "fail"),
        (("fail", "inconclusive"), "fail"),
        ((), "inconclusive"),
    ],
)
def test_multi_radar_policy_aggregation(
    statuses: tuple[str, ...],
    expected: str,
) -> None:
    evaluations = {
        f"radar_{index}": {"assessment": {"status": status}}
        for index, status in enumerate(statuses)
    }

    aggregate = aggregate_radar_sensor_assessments(evaluations)

    assert aggregate["status"] == expected
    assert aggregate["required_sensors"] == sorted(evaluations)


def test_multi_radar_same_timestamps_get_sensor_disjoint_evaluations(tmp_path: Path) -> None:
    _write_nuscenes_radar_fixture(
        tmp_path,
        ego_speed_mps=12.0,
        eligible_frame_count=10,
    )
    policy = RadarVelocityPolicy(
        min_holdout_frames=2,
        min_holdout_static_returns=12,
        min_static_fraction=0.5,
        min_yaw_sensitivity_rms_mps_per_rad=0.57,
        max_holdout_median_abs_residual_mps=0.5,
        max_holdout_rmse_mps=0.8,
    )

    _stats, provenance = summarize_nuscenes_radar_velocity_consistency(
        dataset_path=tmp_path,
        radar_sensors={
            "radar_front": "RADAR_FRONT",
            "radar_duplicate": "RADAR_FRONT",
        },
        extrinsics_lookup=lambda _sensor: SE3.identity(),
        holdout_ratio=0.2,
        split_seed=7,
        policy=policy,
    )

    sensors = provenance["sensor_evaluations"]
    assert sensors["radar_front"]["assessment"]["status"] == "pass"
    assert sensors["radar_duplicate"]["assessment"]["status"] == "pass"
    front_ids = set(sensors["radar_front"]["split"]["holdout_frame_ids"])
    duplicate_ids = set(sensors["radar_duplicate"]["split"]["holdout_frame_ids"])
    assert len(front_ids) == 2
    assert len(duplicate_ids) == 2
    assert front_ids.isdisjoint(duplicate_ids)
    assert all(frame_id.startswith("radar_front:") for frame_id in front_ids)
    assert all(frame_id.startswith("radar_duplicate:") for frame_id in duplicate_ids)
    assert provenance["assessment"]["status"] == "pass"
