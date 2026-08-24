"""Focused v0.1 radar replacement/reverification contract tests."""

from __future__ import annotations

from pathlib import Path

from calibrex.core.io import write_mapping
from calibrex.core.provenance import sha256_path
from calibrex.core.radar_service import (
    DigestRef,
    MetricObservation,
    RadarSensorIdentity,
    RadarServicePlan,
    build_radar_service_plan,
    evaluate_radar_service,
    verify_radar_service,
)


def _ref(path: Path, role: str) -> DigestRef:
    digest = sha256_path(path)
    assert digest is not None
    return DigestRef(role=role, path=path.as_posix(), sha256=digest)


def _plan(
    root: Path,
    *,
    replay_status: str = "PASS",
    replay_decision: str = "ADOPT",
    known_bad_gate: str = "PASS",
    observability_gate: str = "PASS",
    missing_role: str | None = None,
    range_bias: float = 0.03,
    time_offset: float = 0.001,
    doppler_sign_convention: str = "positive_away",
    evidence_doppler_sign_convention: str | None = None,
    changed_components: list[str] | None = None,
    mutable_components: list[str] | None = None,
) -> RadarServicePlan:
    root.mkdir(parents=True, exist_ok=True)
    replay_path = root / "replay.json"
    write_mapping(
        replay_path,
        {
            "status": replay_status,
            "decision": replay_decision,
            "gates": {
                "holdout": "PASS",
                "known_bad": known_bad_gate,
                "observability": observability_gate,
            },
        },
    )
    protocol_id = "radar-protocol-synthetic-v01"
    capture_id = "radar-capture-synthetic-v01"
    evidence: dict[str, DigestRef] = {}
    for role in ("protocol", "capture", "holdout", "known_bad", "observability"):
        if role == missing_role:
            continue
        evidence_path = root / f"{role}.json"
        write_mapping(
            evidence_path,
            {
                "role": role,
                "protocol_id": protocol_id,
                "capture_id": capture_id,
                "doppler_sign_convention": (
                    evidence_doppler_sign_convention or doppler_sign_convention
                ),
                "frame_convention": "radar_sensor_frame_x_forward_y_left_z_up",
            },
        )
        evidence[role] = _ref(evidence_path, role)
    return RadarServicePlan(
        plan_id="radar-service-test",
        vehicle_id="vehicle-test",
        sensor_kit_id="kit-test",
        old_radar=RadarSensorIdentity(
            sensor_id="radar-old",
            serial="RADAR-OLD",
            model="synthetic-radar",
            firmware="1.0",
            mount="front-bumper",
            frame="radar_old",
            sensor_type="radar",
        ),
        new_radar=RadarSensorIdentity(
            sensor_id="radar-new",
            serial="RADAR-NEW",
            model="synthetic-radar",
            firmware="1.0",
            mount="front-bumper",
            frame="radar_new",
            sensor_type="radar",
        ),
        candidate_replay=_ref(replay_path, "candidate_replay"),
        evidence=evidence,
        metric_observations=[
            MetricObservation(
                metric_id="range_bias_m", value=range_bias, unit="m", split="holdout"
            ),
            MetricObservation(
                metric_id="azimuth_bias_deg", value=0.4, unit="deg", split="holdout"
            ),
            MetricObservation(
                metric_id="doppler_scale_error_ratio", value=0.01, unit="ratio", split="holdout"
            ),
            MetricObservation(
                metric_id="time_offset_s", value=time_offset, unit="s", split="holdout"
            ),
            MetricObservation(
                metric_id="radar_lidar_association_rmse_m",
                value=0.08,
                unit="m",
                split="holdout",
            ),
            MetricObservation(
                metric_id="fov_overlap_ratio", value=0.9, unit="ratio", split="holdout"
            ),
            MetricObservation(
                metric_id="known_bad_delta", value=0.25, unit="delta", split="known_bad"
            ),
            MetricObservation(
                metric_id="observability_rank", value=8, unit="rank", split="holdout"
            ),
        ],
        mutable_components=mutable_components or ["radar_extrinsics", "radar_time_offset"],
        observed_changed_components=changed_components or [],
        doppler_sign_convention=doppler_sign_convention,
        frame_convention="radar_sensor_frame_x_forward_y_left_z_up",
        max_abs_range_bias_m=0.1,
        max_abs_azimuth_bias_deg=1.0,
        max_doppler_scale_error_ratio=0.05,
        max_time_offset_abs_s=0.01,
        max_radar_lidar_association_rmse_m=0.2,
        min_fov_overlap_ratio=0.8,
        min_known_bad_delta=0.1,
        min_observability_rank=5,
        protocol_id=protocol_id,
        capture_id=capture_id,
    )


def test_different_output_directories_ready_and_verifiable(tmp_path: Path) -> None:
    plan = _plan(tmp_path / "definition")
    plan_path = tmp_path / "plan-output" / "plan.yaml"
    evaluation_path = tmp_path / "evaluation-output" / "evaluation.json"

    built = build_radar_service_plan(plan, output=plan_path)
    assert built.artifact_sha256 != "0" * 64
    result = evaluate_radar_service(plan_path, output=evaluation_path)

    assert result.status == "READY"
    assert result.gates == {
        "candidate_replay": "PASS",
        "holdout": "PASS",
        "known_bad": "PASS",
        "observability": "PASS",
    }
    assert result.doppler_sign_convention == "positive_away"
    assert verify_radar_service(plan_path).valid
    assert verify_radar_service(evaluation_path, plan=plan_path).valid


def test_tampered_input_holds_and_plan_verification_fails(tmp_path: Path) -> None:
    source_root = tmp_path / "definition"
    plan_path = tmp_path / "out" / "plan.yaml"
    build_radar_service_plan(_plan(source_root), output=plan_path)
    (source_root / "capture.json").write_text('{"capture_id": "tampered"}\n', encoding="utf-8")

    result = evaluate_radar_service(plan_path)
    assert result.status == "HOLD"
    assert any("digest mismatch" in reason for reason in result.reasons)
    assert not verify_radar_service(plan_path).valid


def test_missing_required_evidence_is_held(tmp_path: Path) -> None:
    result = evaluate_radar_service(
        build_radar_service_plan(_plan(tmp_path / "definition", missing_role="observability"))
    )
    assert result.status == "HOLD"
    assert result.missing_evidence_roles == ["observability"]


def test_bad_known_bad_or_observability_replay_gate_is_held(tmp_path: Path) -> None:
    known_bad = evaluate_radar_service(
        build_radar_service_plan(_plan(tmp_path / "known-bad", known_bad_gate="FAIL"))
    )
    assert known_bad.status == "HOLD"
    assert known_bad.gates["known_bad"] == "FAIL"

    observability = evaluate_radar_service(
        build_radar_service_plan(_plan(tmp_path / "observability", observability_gate="FAIL"))
    )
    assert observability.status == "HOLD"
    assert observability.gates["observability"] == "FAIL"


def test_range_and_time_budgets_are_fail_closed(tmp_path: Path) -> None:
    range_result = evaluate_radar_service(
        build_radar_service_plan(_plan(tmp_path / "range", range_bias=0.2))
    )
    assert range_result.status == "HOLD"
    assert any("range bias" in reason for reason in range_result.reasons)

    time_result = evaluate_radar_service(
        build_radar_service_plan(_plan(tmp_path / "time", time_offset=0.02))
    )
    assert time_result.status == "HOLD"
    assert any("time offset" in reason for reason in time_result.reasons)


def test_doppler_convention_mismatch_is_held(tmp_path: Path) -> None:
    result = evaluate_radar_service(
        build_radar_service_plan(
            _plan(
                tmp_path / "convention",
                evidence_doppler_sign_convention="positive_toward",
            )
        )
    )
    assert result.status == "HOLD"
    assert any("doppler_sign_convention mismatch" in reason for reason in result.reasons)


def test_unallowlisted_component_is_held(tmp_path: Path) -> None:
    result = evaluate_radar_service(
        build_radar_service_plan(
            _plan(
                tmp_path / "allowlist",
                changed_components=["radar_frame_id"],
                mutable_components=["radar_extrinsics"],
            )
        )
    )
    assert result.status == "HOLD"
    assert result.unallowlisted_changed_components == ["radar_frame_id"]
