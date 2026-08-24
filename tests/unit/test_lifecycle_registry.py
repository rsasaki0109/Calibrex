"""Focused trust-contract tests for the filesystem lifecycle registry."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import patch

import jsonschema
import pytest

from calibrex.cli.main import main
from calibrex.core.capture_manifest import SensorIdentity, inspect_capture
from calibrex.core.lifecycle_registry import (
    ZERO_SHA256,
    LifecycleEvent,
    LifecycleProvenance,
    LifecycleRegistryConcurrencyError,
    LifecycleRegistryError,
    LifecycleRegistryVerificationError,
    _event_digest,
    evaluate_lifecycle,
    init_registry,
    install_sensor,
    lifecycle_evaluation_json_schema,
    lifecycle_event_json_schema,
    lifecycle_registry_json_schema,
    load_registry,
    promote_lifecycle,
    register_calibration_edge,
    register_sensor,
    remove_sensor,
    rollback_lifecycle,
)


def _registry(tmp_path: Path) -> Path:
    root = tmp_path / "registry"
    init_registry(root, registry_id="fleet", operator="test", timestamp="2026-01-01T00:00:00Z")
    register_sensor(
        root,
        sensor_id="lidar-1",
        vehicle_id="car-1",
        sensor_kit_id="kit-1",
        serial="serial-1",
        model="vlp16",
        firmware="1.0",
        mount="roof-front",
        install=True,
        operator="test",
        timestamp="2026-01-01T00:00:01Z",
    )
    return root


def _capture(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "capture.bin"
    source.write_bytes(b"capture")
    artifact = inspect_capture(
        source,
        source_format="files",
        capture_id="capture-1",
        session_id="session-1",
        vehicle_id="car-1",
        sensor_kit_id="kit-1",
        sensors=[
            SensorIdentity(
                sensor_id="lidar-1",
                sensor_type="lidar",
                serial="serial-1",
                model="vlp16",
                firmware="1.0",
                mount="roof-front",
                frame_id="lidar",
            )
        ],
    )
    output = tmp_path / "capture.yaml"
    artifact.model_copy(update={"status": "ready"}).with_artifact_digest().save(output)
    return output


def _event(registry_root: Path, *, reason: str = "test event") -> LifecycleEvent:
    """Build a content-fixed event for chain and concurrency tests."""

    registry_id = load_registry(registry_root).manifest().registry_id
    return LifecycleEvent(
        registry_id=registry_id,
        sequence=0,
        event_type="register",
        status="REGISTERED",
        provenance=LifecycleProvenance(
            operator="test",
            reason=reason,
            timestamp="2026-01-01T00:00:10Z",
            git_commit="fixed-test-commit",
        ),
    )


def _write_inputs(tmp_path: Path, *, status: str = "PASS") -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    result = tmp_path / "candidate.json"
    evidence = tmp_path / "evidence.json"
    assessment = tmp_path / "assessment.json"
    promotion = tmp_path / "promotion.json"
    smoke = tmp_path / "smoke.json"
    # Use the complete native v0.1 result contract.  The registry deliberately
    # fail-closes recognized native results instead of accepting a version
    # string with no required run/frame provenance.
    result.write_text(
        json.dumps(
            {
                "schema_version": "slac.result/v0.1",
                "run": {"id": "candidate-run", "slac_version": "test"},
                "frame_graph": {"root": "base_link", "frames": {}},
            }
        ),
        encoding="utf-8",
    )
    evidence.write_text(json.dumps({"status": status}), encoding="utf-8")
    assessment.write_text(json.dumps({"status": status}), encoding="utf-8")
    promotion.write_text(
        json.dumps({"status": status, "decision": "ADOPT" if status == "PASS" else "DO_NOT_ADOPT"}),
        encoding="utf-8",
    )
    smoke.write_text(
        json.dumps({"status": status, "admission_label": "production-admissible"}),
        encoding="utf-8",
    )
    return {
        "result": result,
        "evidence": evidence,
        "assessment": assessment,
        "promotion": promotion,
        "smoke": smoke,
    }


def test_init_append_verify_and_tamper(tmp_path: Path) -> None:
    root = _registry(tmp_path)
    register_calibration_edge(
        root,
        edge_id="edge-1",
        vehicle_id="car-1",
        sensor_kit_id="kit-1",
        parent_frame="base_link",
        child_frame="lidar",
        operator="test",
        timestamp="2026-01-01T00:00:02Z",
    )
    assert load_registry(root).verify().valid
    event_path = root / "events.jsonl"
    event_path.write_text(
        event_path.read_text(encoding="utf-8").replace("REGISTERED", "REMOVED"), encoding="utf-8"
    )
    report = load_registry(root).verify()
    assert not report.valid
    assert any("digest mismatch" in error for error in report.errors)


def test_stale_compare_and_swap_is_blocked(tmp_path: Path) -> None:
    root = _registry(tmp_path)
    registry = load_registry(root)
    event = registry.events()[0]
    with pytest.raises(LifecycleRegistryConcurrencyError):
        registry.append_event(event, expected_sequence=-1)


def test_duplicate_registration_and_stale_projection_update_are_rejected(
    tmp_path: Path,
) -> None:
    root = _registry(tmp_path)
    registry = load_registry(root)
    original = registry.events()[0]
    with pytest.raises(LifecycleRegistryVerificationError, match="already registered"):
        registry.append_event(original)
    assert len(load_registry(root).events()) == 1

    register_calibration_edge(
        root,
        edge_id="edge-1",
        vehicle_id="car-1",
        sensor_kit_id="kit-1",
        parent_frame="base_link",
        child_frame="lidar",
        operator="test",
    )
    registry = load_registry(root)
    edge = registry.state().edges["edge-1"]
    stale = LifecycleEvent(
        registry_id="fleet",
        sequence=0,
        event_type="promote",
        status="PROMOTED",
        edge_ids=["edge-1"],
        provenance=LifecycleProvenance(
            operator="test",
            reason="stale projection test",
            timestamp="2026-01-01T00:00:03Z",
            git_commit="fixed-test-commit",
        ),
        payload={
            "base_edge_sequences": {"edge-1": edge.updated_sequence - 1},
            "edge_updates": {"edge-1": edge.model_dump(mode="json", exclude_none=False)},
        },
    )
    with pytest.raises(LifecycleRegistryVerificationError, match="stale edge update"):
        registry.append_event(stale)


def test_weak_evidence_does_not_change_incumbent(tmp_path: Path) -> None:
    root = _registry(tmp_path)
    initial = tmp_path / "incumbent.json"
    initial.write_text(
        json.dumps({"schema_version": "slac.result/v0.1", "value": "old"}), encoding="utf-8"
    )
    register_calibration_edge(
        root,
        edge_id="edge-1",
        vehicle_id="car-1",
        sensor_kit_id="kit-1",
        parent_frame="base_link",
        child_frame="lidar",
        incumbent_result=initial,
        operator="test",
        timestamp="2026-01-01T00:00:02Z",
    )
    inputs = _write_inputs(tmp_path, status="INCONCLUSIVE")
    capture = _capture(tmp_path)
    evaluation = evaluate_lifecycle(
        root,
        edge_id="edge-1",
        capture_manifest=capture,
        candidate_result=inputs["result"],
        evidence=inputs["evidence"],
        assessment=inputs["assessment"],
        promotion=inputs["promotion"],
        smoke=inputs["smoke"],
        operator="test",
        timestamp="2026-01-01T00:00:03Z",
    )
    assert evaluation.admission in {"HOLD", "DO_NOT_ADOPT"}
    edge = load_registry(root).state().edges["edge-1"]
    assert edge.incumbent_result is not None
    assert edge.incumbent_result.sha256 != inputs["result"].read_bytes().hex()
    assert edge.last_admission != "ADOPT"


def test_candidate_promote_partial_edge_and_rollback(tmp_path: Path) -> None:
    root = _registry(tmp_path)
    initial = tmp_path / "incumbent.json"
    initial.write_text(
        json.dumps({"schema_version": "slac.result/v0.1", "value": "old"}), encoding="utf-8"
    )
    for edge_id in ("edge-1", "edge-2"):
        register_calibration_edge(
            root,
            edge_id=edge_id,
            vehicle_id="car-1",
            sensor_kit_id="kit-1",
            parent_frame="base_link",
            child_frame=edge_id,
            incumbent_result=initial,
            operator="test",
            timestamp="2026-01-01T00:00:02Z",
        )
    inputs = _write_inputs(tmp_path)
    capture = _capture(tmp_path)
    evaluate_lifecycle(
        root,
        edge_id="edge-1",
        capture_manifest=capture,
        candidate_result=inputs["result"],
        evidence=inputs["evidence"],
        assessment=inputs["assessment"],
        promotion=inputs["promotion"],
        smoke=inputs["smoke"],
        operator="test",
        timestamp="2026-01-01T00:00:03Z",
    )
    promote_lifecycle(root, edge_id="edge-1", operator="test", timestamp="2026-01-01T00:00:04Z")
    state = load_registry(root).state()
    assert state.edges["edge-1"].incumbent_result is not None
    assert state.edges["edge-2"].incumbent_result is not None
    rollback_lifecycle(root, edge_id="edge-1", operator="test", timestamp="2026-01-01T00:00:05Z")
    state = load_registry(root).state()
    assert state.edges["edge-1"].incumbent_result is not None
    assert (
        state.edges["edge-1"].incumbent_result.sha256
        == state.edges["edge-2"].incumbent_result.sha256
    )
    assert load_registry(root).verify().valid


def test_replaced_sensor_identity_is_blocked(tmp_path: Path) -> None:
    root = _registry(tmp_path)
    with pytest.raises(LifecycleRegistryError, match="identity mismatch"):
        install_sensor(
            root,
            sensor_id="lidar-1",
            serial="other-serial",
            model="vlp16",
            firmware="1.0",
            mount="roof-front",
        )


def test_duplicate_sequence_and_fork_are_detected(tmp_path: Path) -> None:
    duplicate_root = _registry(tmp_path / "duplicate")
    duplicate_path = duplicate_root / "events.jsonl"
    lines = duplicate_path.read_text(encoding="utf-8").splitlines()
    duplicate_path.write_text("\n".join([*lines, lines[0]]) + "\n", encoding="utf-8")
    duplicate_report = load_registry(duplicate_root).verify()
    assert not duplicate_report.valid
    assert any("duplicate event sequence" in error for error in duplicate_report.errors)

    fork_root = _registry(tmp_path / "fork")
    load_registry(fork_root).append_event(_event(fork_root))
    fork_path = fork_root / "events.jsonl"
    fork_lines = fork_path.read_text(encoding="utf-8").splitlines()
    fork_payload = json.loads(fork_lines[1])
    fork_payload["previous_event_sha256"] = ZERO_SHA256
    fork_event = LifecycleEvent.model_validate(fork_payload).model_copy(
        update={"event_sha256": ZERO_SHA256}
    )
    fork_payload["event_sha256"] = _event_digest(fork_event)
    fork_lines[1] = json.dumps(fork_payload, sort_keys=True, separators=(",", ":"))
    fork_path.write_text("\n".join(fork_lines) + "\n", encoding="utf-8")
    fork_report = load_registry(fork_root).verify()
    assert not fork_report.valid
    assert any("previous digest" in error for error in fork_report.errors)


def test_stale_head_and_checkpoint_are_detected(tmp_path: Path) -> None:
    root = _registry(tmp_path)
    head_path = root / "head.json"
    head = json.loads(head_path.read_text(encoding="utf-8"))
    head["sequence"] = -1
    head_path.write_text(json.dumps(head), encoding="utf-8")
    report = load_registry(root).verify()
    assert not report.valid
    assert any("stale registry head sequence" in error for error in report.errors)


def test_missing_and_tampered_source_artifacts_fail_closed(tmp_path: Path) -> None:
    root = _registry(tmp_path)
    incumbent = tmp_path / "incumbent.json"
    incumbent.write_text(json.dumps({"schema_version": "slac.result/v0.1"}), encoding="utf-8")
    register_calibration_edge(
        root,
        edge_id="edge-1",
        vehicle_id="car-1",
        sensor_kit_id="kit-1",
        parent_frame="base_link",
        child_frame="lidar",
        incumbent_result=incumbent,
        operator="test",
    )
    incumbent.unlink()
    missing = load_registry(root).verify()
    assert not missing.valid
    assert any("does not exist" in error for error in missing.errors)
    incumbent.write_text(
        json.dumps({"schema_version": "slac.result/v0.1", "changed": True}), encoding="utf-8"
    )
    tampered = load_registry(root).verify()
    assert not tampered.valid
    assert any("digest mismatch" in error for error in tampered.errors)


def test_cross_vehicle_and_sensor_kit_boundaries_are_blocked(tmp_path: Path) -> None:
    root = _registry(tmp_path)
    register_sensor(
        root,
        sensor_id="lidar-2",
        vehicle_id="car-2",
        sensor_kit_id="kit-2",
        serial="serial-2",
        model="vlp16",
        firmware="1.0",
        mount="roof-front",
        operator="test",
    )
    with pytest.raises(LifecycleRegistryError, match="cross-vehicle"):
        register_sensor(
            root,
            sensor_id="lidar-3",
            vehicle_id="car-1",
            sensor_kit_id="kit-2",
            serial="serial-3",
            model="vlp16",
            firmware="1.0",
            mount="roof-front",
            operator="test",
        )
    with pytest.raises(LifecycleRegistryError, match="cross-vehicle"):
        register_calibration_edge(
            root,
            edge_id="edge-cross",
            vehicle_id="car-1",
            sensor_kit_id="kit-2",
            parent_frame="base_link",
            child_frame="lidar",
        )


def test_remove_and_remount_preserve_physical_sensor_identity(tmp_path: Path) -> None:
    root = _registry(tmp_path)
    remove_sensor(root, sensor_id="lidar-1", operator="test", timestamp="2026-01-01T00:00:02Z")
    removed = load_registry(root).state().sensors["lidar-1"]
    assert removed.state == "removed"
    assert removed.serial == "serial-1"
    install_sensor(
        root,
        sensor_id="lidar-1",
        serial="serial-1",
        model="vlp16",
        firmware="1.0",
        mount="roof-front",
        operator="test",
        timestamp="2026-01-01T00:00:03Z",
    )
    remounted = load_registry(root).state().sensors["lidar-1"]
    assert remounted.state == "installed"
    assert remounted.remount_count == 1
    assert remounted.install_count == 2
    assert [event.event_type for event in load_registry(root).events()][-2:] == [
        "remove",
        "install",
    ]
    assert load_registry(root).verify().valid


def test_lock_contention_and_concurrent_append_attempts_are_serialized(tmp_path: Path) -> None:
    root = _registry(tmp_path)
    registry = load_registry(root)
    with (
        registry._lock(timeout_s=0.1),
        pytest.raises(LifecycleRegistryConcurrencyError),
        registry._lock(timeout_s=0.02),
    ):
        pass

    results: list[LifecycleEvent] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(4)

    def append_worker(index: int) -> None:
        try:
            barrier.wait(timeout=2.0)
            results.append(load_registry(root).append_event(_event(root, reason=f"worker-{index}")))
        except BaseException as exc:  # pragma: no cover - failure is asserted below
            errors.append(exc)

    threads = [threading.Thread(target=append_worker, args=(index,)) for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)
    assert not errors
    assert sorted(event.sequence for event in results) == [1, 2, 3, 4]
    assert load_registry(root).verify().valid


def test_source_path_escape_and_symlink_are_rejected(tmp_path: Path) -> None:
    root = _registry(tmp_path / "registry")
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"schema_version": "slac.result/v0.1"}), encoding="utf-8")
    with pytest.raises(LifecycleRegistryError, match="escapes registry root"):
        register_calibration_edge(
            root,
            edge_id="edge-escape",
            vehicle_id="car-1",
            sensor_kit_id="kit-1",
            parent_frame="base_link",
            child_frame="lidar",
            incumbent_result="../outside.json",
        )
    link = root / "linked.json"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable on this filesystem")
    with pytest.raises(LifecycleRegistryError, match="escapes registry root"):
        register_calibration_edge(
            root,
            edge_id="edge-symlink",
            vehicle_id="car-1",
            sensor_kit_id="kit-1",
            parent_frame="base_link",
            child_frame="lidar",
            incumbent_result=link,
        )


def test_projection_failure_leaves_recoverable_pending_transaction(tmp_path: Path) -> None:
    root = _registry(tmp_path)
    registry = load_registry(root)
    with patch.object(
        registry, "_write_head", side_effect=OSError("injected head failure")
    ), pytest.raises(OSError, match="injected head failure"):
        registry.append_event(_event(root))
    assert (root / ".registry.pending.json").is_file()
    failed = registry.verify()
    assert not failed.valid
    assert any("incomplete append" in error for error in failed.errors)
    recovered = registry.recover()
    assert recovered.valid
    assert not (root / ".registry.pending.json").exists()
    assert len(registry.events()) == 2
    assert registry.verify().valid


def test_event_digest_is_deterministic_and_artifacts_validate_against_schemas(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    init_registry(first, registry_id="deterministic", timestamp="2026-01-01T00:00:00Z")
    init_registry(second, registry_id="deterministic", timestamp="2026-01-01T00:00:00Z")
    event_first = load_registry(first).append_event(_event(first))
    event_second = load_registry(second).append_event(_event(second))
    assert event_first.event_sha256 == event_second.event_sha256
    jsonschema.validate(
        event_first.model_dump(mode="json", exclude_none=False), lifecycle_event_json_schema()
    )
    jsonschema.validate(
        load_registry(first).manifest().model_dump(mode="json"), lifecycle_registry_json_schema()
    )

    root = _registry(tmp_path / "evaluation")
    register_calibration_edge(
        root,
        edge_id="edge-1",
        vehicle_id="car-1",
        sensor_kit_id="kit-1",
        parent_frame="base_link",
        child_frame="lidar",
        operator="test",
    )
    inputs = _write_inputs(tmp_path / "evaluation-inputs")
    capture = _capture(tmp_path / "evaluation-capture")
    evaluation = evaluate_lifecycle(
        root,
        edge_id="edge-1",
        capture_manifest=capture,
        candidate_result=inputs["result"],
        evidence=inputs["evidence"],
        assessment=inputs["assessment"],
        promotion=inputs["promotion"],
        smoke=inputs["smoke"],
        operator="test",
        timestamp="2026-01-01T00:00:03Z",
    )
    evaluation.verify_artifact_digest()
    jsonschema.validate(
        evaluation.model_dump(mode="json", exclude_none=False), lifecycle_evaluation_json_schema()
    )


def test_cli_lifecycle_init_status_verify_evaluate_promote_rollback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "cli-registry"
    assert main(
        [
            "lifecycle",
            "init",
            "--registry-root",
            str(root),
            "--registry-id",
            "cli-registry",
            "--timestamp",
            "2026-01-01T00:00:00Z",
            "--json",
        ]
    ) == 0
    assert main(
        [
            "lifecycle",
            "register-sensor",
            "--registry-root",
            str(root),
            "--sensor-id",
            "lidar-1",
            "--vehicle-id",
            "car-1",
            "--sensor-kit-id",
            "kit-1",
            "--serial",
            "serial-1",
            "--model",
            "vlp16",
            "--firmware",
            "1.0",
            "--mount",
            "roof-front",
            "--install",
            "--timestamp",
            "2026-01-01T00:00:01Z",
            "--json",
        ]
    ) == 0
    incumbent = tmp_path / "cli-incumbent.json"
    incumbent.write_text(json.dumps({"schema_version": "slac.result/v0.1"}), encoding="utf-8")
    assert main(
        [
            "lifecycle",
            "register-edge",
            "--registry-root",
            str(root),
            "--edge-id",
            "edge-1",
            "--vehicle-id",
            "car-1",
            "--sensor-kit-id",
            "kit-1",
            "--parent-frame",
            "base_link",
            "--child-frame",
            "lidar",
            "--incumbent-result",
            str(incumbent),
            "--timestamp",
            "2026-01-01T00:00:02Z",
            "--json",
        ]
    ) == 0
    inputs = _write_inputs(tmp_path / "cli-inputs")
    capture = _capture(tmp_path / "cli-capture")
    assert main(
        [
            "lifecycle",
            "capture",
            "--registry-root",
            str(root),
            "--capture-manifest",
            str(capture),
            "--timestamp",
            "2026-01-01T00:00:03Z",
            "--json",
        ]
    ) == 0
    evaluation_path = tmp_path / "cli-evaluation.json"
    assert main(
        [
            "lifecycle",
            "evaluate",
            "--registry-root",
            str(root),
            "--edge-id",
            "edge-1",
            "--capture-manifest",
            str(capture),
            "--candidate-result",
            str(inputs["result"]),
            "--evidence",
            str(inputs["evidence"]),
            "--assessment",
            str(inputs["assessment"]),
            "--promotion",
            str(inputs["promotion"]),
            "--smoke",
            str(inputs["smoke"]),
            "--output",
            str(evaluation_path),
            "--timestamp",
            "2026-01-01T00:00:04Z",
            "--json",
        ]
    ) == 0
    assert main(
        [
            "lifecycle",
            "promote",
            "--registry-root",
            str(root),
            "--edge-id",
            "edge-1",
            "--evaluation",
            str(evaluation_path),
            "--timestamp",
            "2026-01-01T00:00:05Z",
            "--json",
        ]
    ) == 0
    assert main(
        [
            "lifecycle",
            "rollback",
            "--registry-root",
            str(root),
            "--edge-id",
            "edge-1",
            "--timestamp",
            "2026-01-01T00:00:06Z",
            "--json",
        ]
    ) == 0
    assert main(["lifecycle", "verify", "--registry-root", str(root), "--json"]) == 0
    assert main(["lifecycle", "status", "--registry-root", str(root), "--json"]) == 0
    assert load_registry(root).verify().valid
    assert load_registry(root).state().edges["edge-1"].last_admission == "ROLLBACK"
    assert capsys.readouterr().out
