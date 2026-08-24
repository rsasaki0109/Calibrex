"""Focused v0.1 multi-LiDAR service contract tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from calibrex.core.io import write_mapping
from calibrex.core.multi_lidar_service import (
    MultiLidarArtifactRef,
    MultiLidarEdgeArtifactRef,
    MultiLidarSensorIdentity,
    MultiLidarServicePlan,
    build_multi_lidar_service_plan,
    evaluate_multi_lidar_service,
    verify_multi_lidar_service,
)
from calibrex.core.provenance import sha256_path
from calibrex.core.result import TransformResult


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _plan(
    *,
    baseline: list[MultiLidarEdgeArtifactRef],
    candidate: list[MultiLidarEdgeArtifactRef],
    replay: dict[str, object] | None = None,
    allowlist: list[str] | None = None,
) -> MultiLidarServicePlan:
    replay_payload = replay or {
        "status": "PASS",
        "decision": "ADOPT",
        "gates": dict.fromkeys(("holdout", "known_bad", "observability"), "PASS"),
    }
    replay_ref = MultiLidarArtifactRef(
        role="candidate_replay",
        sha256=_digest(replay_payload),
        payload=replay_payload,
    )
    return MultiLidarServicePlan(
        plan_id="test-service",
        vehicle_id="vehicle-1",
        old_sensor=MultiLidarSensorIdentity(
            sensor_id="old", serial="old-1", model="lidar", firmware="1", mount="old"
        ),
        new_sensor=MultiLidarSensorIdentity(
            sensor_id="new", serial="new-1", model="lidar", firmware="1", mount="new"
        ),
        baseline_edges=baseline,
        candidate_edges=candidate,
        mutable_incident_edge_ids=allowlist or [],
        candidate_replay=replay_ref,
        graph_root="base_link",
    )


def _edge(edge_id: str, child: str, value: int) -> MultiLidarEdgeArtifactRef:
    payload = {"edge_id": edge_id, "value": value}
    return MultiLidarEdgeArtifactRef(
        role="edge",
        sha256=_digest(payload),
        payload=payload,
        edge_id=edge_id,
        parent_frame="base_link",
        child_frame=child,
        transform=TransformResult(
            parent="base_link",
            child=child,
            translation_m=[0.0, 0.0, 0.0],
            rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
        ),
    )


def test_happy_path_is_ready_and_self_verified() -> None:
    baseline = [_edge("incident-old", "lidar_old", 1)]
    candidate = [_edge("incident-new", "lidar_new", 2)]
    plan = build_multi_lidar_service_plan(
        _plan(baseline=baseline, candidate=candidate, allowlist=["incident-old", "incident-new"])
    )
    evaluation = evaluate_multi_lidar_service(plan)
    assert evaluation.status == "READY"
    assert evaluation.gates == {
        "candidate_replay": "PASS",
        "holdout": "PASS",
        "known_bad": "PASS",
        "observability": "PASS",
    }
    evaluation.verify_artifact_digest()


def test_tampered_file_is_held(tmp_path: Path) -> None:
    source = tmp_path / "edge.json"
    write_mapping(source, {"edge_id": "incident", "value": 1})
    edge = MultiLidarEdgeArtifactRef(
        role="edge",
        path=source.name,
        sha256=sha256_path(source),
        edge_id="incident",
        parent_frame="base_link",
        child_frame="lidar_old",
    )
    build_multi_lidar_service_plan(
        _plan(baseline=[edge], candidate=[edge], allowlist=[]),
        output=tmp_path / "plan.yaml",
    )
    source.write_text('{"edge_id": "incident", "value": 99}\n', encoding="utf-8")
    evaluation = evaluate_multi_lidar_service(tmp_path / "plan.yaml", candidate_replay=None)
    assert evaluation.status == "HOLD"
    assert any("digest mismatch" in reason for reason in evaluation.reasons)
    assert not verify_multi_lidar_service(tmp_path / "plan.yaml").valid


def test_unrelated_edge_drift_is_not_allowlisted() -> None:
    baseline = [_edge("incident", "lidar_old", 1), _edge("unrelated", "lidar_aux", 1)]
    candidate = [_edge("incident", "lidar_new", 2), _edge("unrelated", "lidar_aux", 9)]
    plan = build_multi_lidar_service_plan(
        _plan(baseline=baseline, candidate=candidate, allowlist=["incident"])
    )
    evaluation = evaluate_multi_lidar_service(plan)
    assert evaluation.status == "HOLD"
    assert evaluation.unallowlisted_changed_edge_ids == ["unrelated"]


def test_bad_replay_gate_is_not_ready() -> None:
    baseline = [_edge("incident-old", "lidar_old", 1)]
    candidate = [_edge("incident-new", "lidar_new", 2)]
    plan = build_multi_lidar_service_plan(
        _plan(
            baseline=baseline,
            candidate=candidate,
            allowlist=["incident-old", "incident-new"],
            replay={
                "status": "PASS",
                "decision": "ADOPT",
                "gates": {"holdout": "FAIL", "known_bad": "PASS", "observability": "PASS"},
            },
        )
    )
    evaluation = evaluate_multi_lidar_service(plan)
    assert evaluation.status == "HOLD"
    assert evaluation.gates["holdout"] == "FAIL"


def test_disconnected_graph_and_cycle_closure_fail() -> None:
    disconnected = build_multi_lidar_service_plan(
        _plan(
            baseline=[_edge("incident-old", "lidar_old", 1)],
            candidate=[
                _edge("incident-new", "lidar_new", 2),
                MultiLidarEdgeArtifactRef(
                    role="edge",
                    sha256="a" * 64,
                    edge_id="orphan",
                    parent_frame="orphan_root",
                    child_frame="orphan_lidar",
                ),
            ],
            allowlist=["incident-old", "incident-new", "orphan"],
        )
    )
    result = evaluate_multi_lidar_service(disconnected)
    assert result.status == "HOLD"
    assert result.graph.connected is False

    cycle = _edge("cycle", "base_link", 3)
    cycle = cycle.model_copy(
        update={
            "transform": TransformResult(
                parent="base_link",
                child="base_link",
                translation_m=[1.0, 0.0, 0.0],
                rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            )
        }
    )
    cycle_plan = build_multi_lidar_service_plan(
        _plan(
            baseline=[_edge("incident-old", "lidar_old", 1)],
            candidate=[cycle],
            allowlist=["incident-old", "cycle"],
        )
    )
    cycle_result = evaluate_multi_lidar_service(cycle_plan)
    assert cycle_result.status == "HOLD"
    assert cycle_result.graph.cycle_count == 1
    assert any("closure" in reason for reason in cycle_result.reasons)


def test_separate_plan_output_directory_remains_ready(tmp_path: Path) -> None:
    """A plan emitted outside its definition directory keeps source refs usable."""

    definition = Path("examples/multi_lidar_service/synthetic/plan.yaml")
    plan_path = tmp_path / "plan" / "plan.yaml"
    evaluation_path = tmp_path / "evaluation" / "evaluation.yaml"
    build_multi_lidar_service_plan(definition, output=plan_path)
    evaluation = evaluate_multi_lidar_service(plan_path, output=evaluation_path)
    assert evaluation.status == "READY"
    verification = verify_multi_lidar_service(evaluation_path, plan=plan_path)
    assert verification.valid is True
