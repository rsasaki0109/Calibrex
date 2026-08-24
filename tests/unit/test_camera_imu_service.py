"""Focused v0.1 camera--IMU service contract tests."""

from __future__ import annotations

from pathlib import Path

from calibrex.core.camera_imu_service import (
    CameraImuServicePlan,
    DigestRef,
    MetricObservation,
    SensorIdentity,
    build_camera_imu_service_plan,
    evaluate_camera_imu_service,
    verify_camera_imu_service,
)
from calibrex.core.io import write_mapping
from calibrex.core.provenance import sha256_path


def _ref(path: Path, role: str) -> DigestRef:
    digest = sha256_path(path)
    assert digest is not None
    return DigestRef(role=role, path=path.as_posix(), sha256=digest)


def _plan(
    root: Path,
    *,
    replay_status: str = "PASS",
    replay_decision: str = "ADOPT",
    holdout_gate: str = "PASS",
    missing_role: str | None = None,
    clock_offset: float = 0.001,
    changed_components: list[str] | None = None,
    mutable_components: list[str] | None = None,
) -> CameraImuServicePlan:
    root.mkdir(parents=True, exist_ok=True)
    replay_path = root / "replay.json"
    write_mapping(
        replay_path,
        {
            "status": replay_status,
            "decision": replay_decision,
            "gates": {
                "holdout": holdout_gate,
                "known_bad": "PASS",
                "observability": "PASS",
            },
        },
    )
    evidence: dict[str, DigestRef] = {}
    for role in ("protocol", "capture", "holdout", "known_bad", "observability"):
        if role == missing_role:
            continue
        evidence_path = root / f"{role}.json"
        write_mapping(
            evidence_path,
            {
                "role": role,
                "protocol_id": "protocol-synthetic-v01",
                "capture_id": "capture-synthetic-v01",
            },
        )
        evidence[role] = _ref(evidence_path, role)
    return CameraImuServicePlan(
        plan_id="camera-imu-service-test",
        vehicle_id="vehicle-test",
        old_camera=SensorIdentity(
            sensor_id="camera-old",
            serial="CAM-OLD",
            model="synthetic-camera",
            firmware="1.0",
            mount="front",
            frame="camera_old",
            sensor_type="camera",
        ),
        new_camera=SensorIdentity(
            sensor_id="camera-new",
            serial="CAM-NEW",
            model="synthetic-camera",
            firmware="1.0",
            mount="front",
            frame="camera_new",
            sensor_type="camera",
        ),
        old_imu=SensorIdentity(
            sensor_id="imu-old",
            serial="IMU-OLD",
            model="synthetic-imu",
            firmware="1.0",
            mount="body",
            frame="imu_old",
            sensor_type="imu",
        ),
        new_imu=SensorIdentity(
            sensor_id="imu-new",
            serial="IMU-NEW",
            model="synthetic-imu",
            firmware="1.0",
            mount="body",
            frame="imu_new",
            sensor_type="imu",
        ),
        candidate_replay=_ref(replay_path, "candidate_replay"),
        evidence=evidence,
        metric_observations=[
            MetricObservation(
                metric_id="clock_offset_s",
                value=clock_offset,
                unit="s",
                split="train",
            ),
            MetricObservation(
                metric_id="rotation_holdout_rmse_rad",
                value=0.01,
                unit="rad",
                split="holdout",
            ),
            MetricObservation(
                metric_id="lever_arm_holdout_rmse_m_s2",
                value=0.02,
                unit="m/s^2",
                split="holdout",
            ),
            MetricObservation(
                metric_id="bias_norm",
                value=0.05,
                unit="m/s^2",
                split="holdout",
            ),
            MetricObservation(
                metric_id="known_bad_delta",
                value=0.2,
                unit="delta",
                split="known_bad",
            ),
            MetricObservation(
                metric_id="observability_rank",
                value=8,
                unit="rank",
                split="holdout",
            ),
        ],
        mutable_components=mutable_components or ["camera_extrinsics", "imu_extrinsics"],
        observed_changed_components=changed_components or [],
        max_clock_offset_abs_s=0.01,
        max_rotation_holdout_rmse_rad=0.05,
        max_lever_arm_holdout_rmse_m_s2=0.1,
        max_bias_norm=0.2,
        min_known_bad_delta=0.1,
        min_observability_rank=5,
    )


def test_different_output_directory_is_ready_and_verifiable(tmp_path: Path) -> None:
    plan = _plan(tmp_path / "definition")
    plan_path = tmp_path / "plan-output" / "plan.yaml"
    evaluation_path = tmp_path / "evaluation-output" / "evaluation.json"

    built = build_camera_imu_service_plan(plan, output=plan_path)
    assert built.artifact_sha256 != "0" * 64
    result = evaluate_camera_imu_service(plan_path, output=evaluation_path)

    assert result.status == "READY"
    assert result.gates == {
        "candidate_replay": "PASS",
        "holdout": "PASS",
        "known_bad": "PASS",
        "observability": "PASS",
    }
    assert verify_camera_imu_service(plan_path).valid
    assert verify_camera_imu_service(evaluation_path, plan=plan_path).valid


def test_tampered_input_is_held_and_plan_verification_fails(tmp_path: Path) -> None:
    source_root = tmp_path / "definition"
    plan_path = tmp_path / "out" / "plan.yaml"
    plan = _plan(source_root)
    build_camera_imu_service_plan(plan, output=plan_path)
    (source_root / "capture.json").write_text("{\"capture_id\": \"tampered\"}\n", encoding="utf-8")

    result = evaluate_camera_imu_service(plan_path)
    assert result.status == "HOLD"
    assert any("digest mismatch" in reason for reason in result.reasons)
    assert not verify_camera_imu_service(plan_path).valid


def test_missing_required_evidence_is_held(tmp_path: Path) -> None:
    plan = _plan(tmp_path / "definition", missing_role="observability")
    result = evaluate_camera_imu_service(build_camera_imu_service_plan(plan))
    assert result.status == "HOLD"
    assert result.missing_evidence_roles == ["observability"]


def test_bad_replay_gate_is_held(tmp_path: Path) -> None:
    plan = _plan(tmp_path / "definition", holdout_gate="FAIL")
    result = evaluate_camera_imu_service(build_camera_imu_service_plan(plan))
    assert result.status == "HOLD"
    assert result.gates["holdout"] == "FAIL"


def test_clock_budget_is_fail_closed(tmp_path: Path) -> None:
    plan = _plan(tmp_path / "definition", clock_offset=0.02)
    result = evaluate_camera_imu_service(build_camera_imu_service_plan(plan))
    assert result.status == "HOLD"
    assert any("clock offset" in reason for reason in result.reasons)


def test_unallowlisted_component_is_held(tmp_path: Path) -> None:
    plan = _plan(
        tmp_path / "definition",
        changed_components=["imu_bias"],
        mutable_components=["camera_extrinsics"],
    )
    result = evaluate_camera_imu_service(build_camera_imu_service_plan(plan))
    assert result.status == "HOLD"
    assert result.unallowlisted_changed_components == ["imu_bias"]
