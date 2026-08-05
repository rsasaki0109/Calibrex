"""Tests for physical solid-state LiDAR ground-truth evaluation."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import jsonschema

from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.provenance import sha256_path
from calibrex.core.result import (
    EstimateEvidenceLevel,
    TransformEstimateProvenance,
    TransformResult,
)
from calibrex.core.solid_state_metrology_evaluation import (
    SolidStateMetrologyDownstreamMetric,
    SolidStateMetrologyEstimate,
    SolidStateMetrologyEvaluationArtifact,
    SolidStateMetrologyEvaluationMetrics,
    SolidStateMetrologyEvaluationProtocol,
    SolidStateMetrologyEvaluationProvenance,
    SolidStateMetrologyEvaluationThresholds,
    SolidStateMetrologyReference,
    SolidStateMetrologySession,
    evaluate_solid_state_metrology,
    verify_solid_state_metrology_evidence,
)


def _transform(
    translation: tuple[float, float, float],
    *,
    yaw_deg: float = 0.0,
    evidence_level: EstimateEvidenceLevel = "algorithmically_refined",
) -> TransformResult:
    half = math.radians(yaw_deg) * 0.5
    return TransformResult(
        parent="source",
        child="target",
        translation_m=list(translation),
        rotation_quat_xyzw=[0.0, 0.0, math.sin(half), math.cos(half)],
        estimate_id="test-transform",
        provenance=TransformEstimateProvenance(
            producer=(
                "human" if evidence_level == "independently_measured" else "slac_native"
            ),
            execution_mode=(
                "manual" if evidence_level == "independently_measured" else "offline_batch"
            ),
            role_in_comparison="selected_reference"
            if evidence_level == "independently_measured"
            else "output",
            evidence_level=evidence_level,
            source="tests/unit/test_solid_state_metrology_evaluation.py",
        ),
    )


def _artifact(
    estimate_translation: tuple[float, float, float] = (0.12, -0.05, 0.04),
) -> SolidStateMetrologyEvaluationArtifact:
    reference_transform = _transform(
        (0.12, -0.05, 0.04),
        yaw_deg=8.0,
        evidence_level="independently_measured",
    )
    def session_reference(session_id: str) -> SolidStateMetrologyReference:
        path = f"metrology/{session_id}-reference.csv"
        return SolidStateMetrologyReference(
            extrinsic_method="surveyed_rig",
            clock_method="hardware_trigger",
            independent_of_solver=True,
            transform=reference_transform,
            time_offset_sec=0.03,
            rotation_uncertainty_deg=0.01,
            translation_uncertainty_m=0.0005,
            time_uncertainty_sec=0.00001,
            source_paths=[path],
            source_sha256={path: "f" * 64},
        )

    sessions = [
        SolidStateMetrologySession(
            id="session-0",
            remount_id="mount-a",
            reference=session_reference("session-0"),
        ),
        SolidStateMetrologySession(
            id="session-1",
            remount_id="mount-a",
            reference=session_reference("session-1"),
        ),
        SolidStateMetrologySession(
            id="session-2",
            remount_id="mount-b",
            reference=session_reference("session-2"),
        ),
    ]
    estimates = [
        SolidStateMetrologyEstimate(
            id=f"estimate-{index}",
            session_id=session.id,
            transform=_transform(estimate_translation, yaw_deg=8.0),
            time_offset_sec=0.03,
            source_path=f"results/{session.id}.yaml",
            source_sha256=(chr(97 + index) * 64),
        )
        for index, session in enumerate(sessions)
    ]
    return SolidStateMetrologyEvaluationArtifact(
        evaluation_id="test-physical-evaluation",
        protocol=SolidStateMetrologyEvaluationProtocol(
            name="test_protocol",
            source_sensor="lidar_source",
            target_sensor="lidar_target",
        ),
        reference=SolidStateMetrologyReference(
            extrinsic_method="surveyed_rig",
            clock_method="hardware_trigger",
            independent_of_solver=True,
            transform=reference_transform,
            time_offset_sec=0.03,
            rotation_uncertainty_deg=0.01,
            translation_uncertainty_m=0.0005,
            time_uncertainty_sec=0.00001,
            source_paths=["metrology/reference.csv"],
            source_sha256={"metrology/reference.csv": "d" * 64},
        ),
        sessions=sessions,
        estimates=estimates,
        downstream_metric=SolidStateMetrologyDownstreamMetric(
            name="held_out_depth_edge_rmse",
            value=0.02,
            baseline_value=0.08,
            unit="m",
            held_out=True,
            independent_of_solver=True,
            max_value=0.05,
        ),
        thresholds=SolidStateMetrologyEvaluationThresholds(),
        metrics=SolidStateMetrologyEvaluationMetrics(),
        provenance=SolidStateMetrologyEvaluationProvenance(
            generator="test",
            generator_version="0",
            source_sha256={"test": "e" * 64},
        ),
    )


def _artifact_with_materialized_sources(
    tmp_path: Path,
) -> SolidStateMetrologyEvaluationArtifact:
    """Create a measured-looking packet whose declared sources really exist."""

    artifact = _artifact()

    def materialize(relative_path: str, payload: bytes) -> str:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        digest = sha256_path(path)
        assert digest is not None
        return digest

    reference_path = "metrology/reference.csv"
    reference_digest = materialize(reference_path, b"independent-reference\n")
    sessions = []
    for session in artifact.sessions:
        assert session.reference is not None
        assert len(session.reference.source_paths) == 1
        session_reference_path = session.reference.source_paths[0]
        session_reference_digest = materialize(
            session_reference_path,
            f"independent-reference:{session.id}".encode(),
        )
        session_reference = session.reference.model_copy(
            update={
                "source_sha256": {
                    session_reference_path: session_reference_digest
                }
            }
        )
        capture_path = f"captures/{session.id}.bin"
        capture_digest = materialize(capture_path, session.id.encode("utf-8"))
        sessions.append(
            session.model_copy(
                update={
                    "reference": session_reference,
                    "capture_path": capture_path,
                    "capture_sha256": capture_digest,
                }
            )
        )

    estimates = []
    for estimate in artifact.estimates:
        assert estimate.source_path is not None
        estimate_digest = materialize(
            estimate.source_path,
            estimate.id.encode("utf-8"),
        )
        estimates.append(
            estimate.model_copy(update={"source_sha256": estimate_digest})
        )

    return artifact.model_copy(
        update={
            "reference": artifact.reference.model_copy(
                update={"source_sha256": {reference_path: reference_digest}}
            ),
            "sessions": sessions,
            "estimates": estimates,
        }
    )


def _verified_artifact(
    tmp_path: Path,
    *,
    estimate_translation: tuple[float, float, float] = (0.12, -0.05, 0.04),
) -> SolidStateMetrologyEvaluationArtifact:
    materialized = _artifact_with_materialized_sources(tmp_path)
    artifact = materialized.model_copy(
        update={
            "estimates": [
                estimate.model_copy(
                    update={
                        "transform": _transform(
                            estimate_translation,
                            yaw_deg=8.0,
                        )
                    }
                )
                for estimate in materialized.estimates
            ]
        }
    )
    return artifact.model_copy(
        update={
            "evidence_integrity": verify_solid_state_metrology_evidence(
                artifact,
                base_dir=tmp_path,
            )
        }
    )


def test_physical_reference_and_repeat_remounts_pass(tmp_path: Path) -> None:
    evaluated = evaluate_solid_state_metrology(_verified_artifact(tmp_path))

    assert evaluated.status == "pass"
    assert evaluated.decision == "pass"
    assert evaluated.metrics.passing_run_count == 3
    assert evaluated.metrics.remount_count == 2
    assert evaluated.metrics.downstream_metric_passed is True
    assert evaluated.reasons == []


def test_physical_accuracy_failure_is_not_inconclusive(tmp_path: Path) -> None:
    evaluated = evaluate_solid_state_metrology(
        _verified_artifact(
            tmp_path,
            estimate_translation=(0.2, -0.05, 0.04),
        )
    )

    assert evaluated.status == "fail"
    assert evaluated.metrics.passing_run_count == 0
    assert "translation_error_exceeds_threshold" in evaluated.reasons


def test_missing_session_reference_blocks_physical_pass(tmp_path: Path) -> None:
    artifact = _verified_artifact(tmp_path)
    sessions = [
        artifact.sessions[0].model_copy(update={"reference": None}),
        *artifact.sessions[1:],
    ]
    artifact = artifact.model_copy(update={"sessions": sessions})
    artifact = artifact.model_copy(
        update={
            "evidence_integrity": verify_solid_state_metrology_evidence(
                artifact,
                base_dir=tmp_path,
            )
        }
    )

    evaluated = evaluate_solid_state_metrology(artifact)

    assert evaluated.status == "inconclusive"
    assert evaluated.decision == "review"
    assert "missing_session_reference" in evaluated.reasons
    assert evaluated.metrics.evaluated_run_count == 2


def test_session_reference_drives_that_session_error(tmp_path: Path) -> None:
    artifact = _verified_artifact(tmp_path)
    assert artifact.sessions[2].reference is not None
    shifted_reference = artifact.sessions[2].reference.model_copy(
        update={
            "transform": _transform(
                (0.15, -0.05, 0.04),
                yaw_deg=8.0,
                evidence_level="independently_measured",
            )
        }
    )
    sessions = [
        *artifact.sessions[:2],
        artifact.sessions[2].model_copy(update={"reference": shifted_reference}),
    ]
    artifact = artifact.model_copy(update={"sessions": sessions})
    artifact = artifact.model_copy(
        update={
            "evidence_integrity": verify_solid_state_metrology_evidence(
                artifact,
                base_dir=tmp_path,
            )
        }
    )

    evaluated = evaluate_solid_state_metrology(artifact)

    assert evaluated.status == "fail"
    assert evaluated.metrics.passing_run_count == 2
    session_metric = next(
        metric
        for metric in evaluated.metrics.run_metrics
        if metric.session_id == "session-2"
    )
    assert session_metric.translation_error_m is not None
    assert abs(session_metric.translation_error_m - 0.03) < 1.0e-12
    assert "translation_error_exceeds_threshold" in session_metric.reasons


def test_missing_physical_evidence_stays_planned() -> None:
    artifact = _artifact()
    artifact = artifact.model_copy(update={"estimates": []})

    evaluated = evaluate_solid_state_metrology(artifact)

    assert evaluated.status == "planned"
    assert evaluated.decision == "collect"
    assert "missing_downstream_metric" not in evaluated.reasons
    assert "missing_estimates" in evaluated.reasons


def test_checked_template_is_schema_valid() -> None:
    asset_path = Path("docs/assets/solid-state-metrology-evaluation-v01.yaml")
    schema_path = Path("schemas/solid_state_metrology_evaluation.schema.json")
    payload = read_mapping(asset_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    jsonschema.validate(payload, schema)
    artifact = SolidStateMetrologyEvaluationArtifact.model_validate(payload)
    assert artifact.status == "planned"
    assert artifact.metrics.run_count == 0


def test_cli_prepares_four_capture_two_remount_collection_plan(
    tmp_path: Path,
) -> None:
    module_path = Path("tools/run_solid_state_metrology_evaluation.py").resolve()
    spec = importlib.util.spec_from_file_location("solid_state_metrology_tool", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    output_path = tmp_path / "collection-plan.yaml"
    markdown_path = tmp_path / "collection-plan.md"
    assert (
        module.main(
            [
                "--prepare-collection-plan",
                "--output",
                str(output_path),
                "--markdown-output",
                str(markdown_path),
            ]
        )
        == 0
    )
    result = SolidStateMetrologyEvaluationArtifact.model_validate(
        read_mapping(output_path)
    )
    schema = json.loads(
        Path("schemas/solid_state_metrology_evaluation.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.validate(read_mapping(output_path), schema)
    assert result.status == "inconclusive"
    assert result.protocol.minimum_usable_sessions == 4
    assert result.protocol.minimum_remounts == 2
    assert result.metrics.run_count == 4
    assert result.metrics.remount_count == 2
    assert [session.remount_id for session in result.sessions] == [
        "remount-1",
        "remount-2",
        "remount-1",
        "remount-2",
    ]
    assert all(session.reference is not None for session in result.sessions)
    assert all(session.capture_path is not None for session in result.sessions)
    assert all(estimate.source_path is not None for estimate in result.estimates)
    assert "Per-session references: `0/4`" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_cli_re_evaluates_a_measured_packet_and_enforces_pass(tmp_path: Path) -> None:
    module_path = Path("tools/run_solid_state_metrology_evaluation.py").resolve()
    spec = importlib.util.spec_from_file_location("solid_state_metrology_tool", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    input_path = tmp_path / "input.yaml"
    output_path = tmp_path / "output.yaml"
    write_mapping(
        input_path,
        _artifact_with_materialized_sources(tmp_path).model_dump(mode="json"),
    )

    assert (
        module.main(
            [
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--enforce",
            ]
        )
        == 0
    )
    result = SolidStateMetrologyEvaluationArtifact.model_validate(
        read_mapping(output_path)
    )
    assert result.status == "pass"
    assert result.metrics.passing_run_count == 3
    assert result.evidence_integrity.checked is True
    assert result.evidence_integrity.passed is True
    assert all(
        check.status == "verified"
        for check in result.evidence_integrity.source_checks
    )


def test_cli_integrity_gate_rejects_tampered_source(tmp_path: Path) -> None:
    module_path = Path("tools/run_solid_state_metrology_evaluation.py").resolve()
    spec = importlib.util.spec_from_file_location("solid_state_metrology_tool", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    input_path = tmp_path / "input.yaml"
    output_path = tmp_path / "output.yaml"
    write_mapping(
        input_path,
        _artifact_with_materialized_sources(tmp_path).model_dump(mode="json"),
    )
    (tmp_path / "captures/session-1.bin").write_bytes(b"tampered")

    assert (
        module.main(
            [
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--enforce",
            ]
        )
        == 1
    )
    result = SolidStateMetrologyEvaluationArtifact.model_validate(
        read_mapping(output_path)
    )
    assert result.status == "inconclusive"
    assert "evidence_integrity_failed" in result.reasons
    assert "evidence_source_digest_mismatch" in result.reasons
    assert any(
        check.owner_id == "session-1" and check.status == "mismatch"
        for check in result.evidence_integrity.source_checks
    )


def test_integrity_gate_detects_duplicate_ids_and_unknown_session_link(
    tmp_path: Path,
) -> None:
    artifact = _artifact()
    sessions = [
        artifact.sessions[0].model_copy(update={"id": "duplicate-session"}),
        artifact.sessions[1].model_copy(update={"id": "duplicate-session"}),
        artifact.sessions[2],
    ]
    estimates = [
        artifact.estimates[0].model_copy(update={"session_id": "missing-session"}),
        artifact.estimates[1],
        artifact.estimates[2],
    ]
    integrity = verify_solid_state_metrology_evidence(
        artifact.model_copy(update={"sessions": sessions, "estimates": estimates}),
        base_dir=tmp_path,
    )

    assert integrity.passed is False
    assert "duplicate_session_id" in integrity.issues
    assert "estimate_session_link_missing" in integrity.issues
    assert "evidence_source_missing" in integrity.issues
