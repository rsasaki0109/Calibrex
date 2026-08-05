"""Tests for physical solid-state LiDAR ground-truth evaluation."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import jsonschema

from calibrex.core.io import read_mapping, write_mapping
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
    sessions = [
        SolidStateMetrologySession(id="session-0", remount_id="mount-a"),
        SolidStateMetrologySession(id="session-1", remount_id="mount-a"),
        SolidStateMetrologySession(id="session-2", remount_id="mount-b"),
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


def test_physical_reference_and_repeat_remounts_pass() -> None:
    evaluated = evaluate_solid_state_metrology(_artifact())

    assert evaluated.status == "pass"
    assert evaluated.decision == "pass"
    assert evaluated.metrics.passing_run_count == 3
    assert evaluated.metrics.remount_count == 2
    assert evaluated.metrics.downstream_metric_passed is True
    assert evaluated.reasons == []


def test_physical_accuracy_failure_is_not_inconclusive() -> None:
    evaluated = evaluate_solid_state_metrology(
        _artifact(estimate_translation=(0.2, -0.05, 0.04))
    )

    assert evaluated.status == "fail"
    assert evaluated.metrics.passing_run_count == 0
    assert "translation_error_exceeds_threshold" in evaluated.reasons


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


def test_cli_re_evaluates_a_measured_packet_and_enforces_pass(tmp_path: Path) -> None:
    module_path = Path("tools/run_solid_state_metrology_evaluation.py").resolve()
    spec = importlib.util.spec_from_file_location("solid_state_metrology_tool", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    input_path = tmp_path / "input.yaml"
    output_path = tmp_path / "output.yaml"
    write_mapping(input_path, _artifact().model_dump(mode="json"))

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
