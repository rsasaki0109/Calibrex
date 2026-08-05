#!/usr/bin/env python3
"""Evaluate solid-state LiDAR estimates against independent physical truth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from calibrex import __version__
from calibrex.core.io import read_mapping, write_mapping, write_text
from calibrex.core.provenance import sha256_path
from calibrex.core.solid_state_metrology_evaluation import (
    SOLID_STATE_METROLOGY_EVALUATION_SCHEMA_VERSION,
    SolidStateMetrologyDownstreamMetric,
    SolidStateMetrologyEstimate,
    SolidStateMetrologyEvaluationArtifact,
    SolidStateMetrologyEvaluationProtocol,
    SolidStateMetrologyEvaluationProvenance,
    SolidStateMetrologyEvaluationThresholds,
    SolidStateMetrologyReference,
    SolidStateMetrologySession,
    evaluate_solid_state_metrology,
    verify_solid_state_metrology_evidence,
)

TOOL_PATH = "tools/run_solid_state_metrology_evaluation.py"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        help="schema-valid artifact edited with measured reference and estimates",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--enforce", action="store_true")
    parser.add_argument(
        "--prepare-collection-plan",
        action="store_true",
        help="generate a four-capture/two-remount packet to fill with measurements",
    )
    parser.add_argument(
        "--plan-sessions",
        type=int,
        default=4,
        help="number of capture sessions in the generated plan (default: 4)",
    )
    parser.add_argument(
        "--plan-remounts",
        type=int,
        default=2,
        help="number of mechanical remounts in the generated plan (default: 2)",
    )
    parser.add_argument(
        "--verify-sources",
        action="store_true",
        help="recompute declared physical-source digests and relationship checks",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def _template() -> SolidStateMetrologyEvaluationArtifact:
    """Return an explicitly incomplete physical-evaluation template."""

    return SolidStateMetrologyEvaluationArtifact(
        evaluation_id="solid-state-metrology-v0.1-template",
        protocol=SolidStateMetrologyEvaluationProtocol(
            name="solid_state_physical_ground_truth",
            source_sensor="lidar_source",
            target_sensor="lidar_target",
            notes=[
                "replace the reference and estimates with measured artifacts",
                "provide an independent spatial/temporal reference for every usable session",
                "repeat across at least two remounts before claiming PASS",
            ],
        ),
        reference=SolidStateMetrologyReference(
            notes=[
                "planned template: no physical reference is supplied",
                "reference provenance must be independent of the solver",
            ]
        ),
        thresholds=SolidStateMetrologyEvaluationThresholds(),
        provenance=SolidStateMetrologyEvaluationProvenance(
            generator=TOOL_PATH,
            generator_version=__version__,
            source_sha256={"template": "0" * 64},
            notes=[
                "template is not a calibration result",
                "do not convert missing measurements into a PASS",
            ],
        ),
    )


def _collection_plan_template(
    *,
    session_count: int,
    remount_count: int,
) -> SolidStateMetrologyEvaluationArtifact:
    """Return a schema-valid, empty physical collection packet."""

    remount_ids = [f"remount-{index + 1}" for index in range(remount_count)]
    sessions: list[SolidStateMetrologySession] = []
    estimates: list[SolidStateMetrologyEstimate] = []
    for index in range(session_count):
        session_id = f"session-{index:02d}"
        remount_id = remount_ids[index % remount_count]
        reference_path = f"metrology/{session_id}-reference.yaml"
        sessions.append(
            SolidStateMetrologySession(
                id=session_id,
                remount_id=remount_id,
                reference=SolidStateMetrologyReference(
                    extrinsic_method="surveyed_rig",
                    clock_method="hardware_trigger",
                    source_paths=[reference_path],
                    notes=[
                        "fill with the independent transform/time measurement for this session",
                        "do not derive this reference from the solver estimate",
                    ],
                ),
                capture_path=f"captures/{session_id}.mcap",
                notes=[
                    "replace the capture path with the finalized bag/MCAP source",
                    "record the capture SHA-256 after the file is finalized",
                ],
            )
        )
        estimates.append(
            SolidStateMetrologyEstimate(
                id=f"estimate-{index:02d}",
                session_id=session_id,
                source_path=f"estimates/{session_id}.yaml",
                notes=[
                    "fill with the solver output for this capture",
                    "record the output SHA-256 after the estimate is finalized",
                ],
            )
        )
    return SolidStateMetrologyEvaluationArtifact(
        evaluation_id="solid-state-metrology-v0.1-collection-plan",
        protocol=SolidStateMetrologyEvaluationProtocol(
            name="solid_state_physical_ground_truth_collection",
            source_sensor="lidar_source",
            target_sensor="lidar_target",
            minimum_usable_sessions=session_count,
            minimum_remounts=remount_count,
            require_per_session_reference=True,
            notes=[
                "collection plan only: replace every placeholder with measured evidence",
                "recommended layout is two captures per remount",
            ],
        ),
        reference=SolidStateMetrologyReference(
            extrinsic_method="surveyed_rig",
            clock_method="hardware_trigger",
            source_paths=["metrology/reference-protocol.yaml"],
            notes=[
                "fill with the common reference-method and frame/time convention record",
                "session.reference holds the measured value used for each run",
            ],
        ),
        sessions=sessions,
        estimates=estimates,
        downstream_metric=SolidStateMetrologyDownstreamMetric(
            name="held_out_downstream_metric",
            unit="declare",
            notes=[
                "fill with a held-out physical-use metric and predeclared threshold",
                "the metric must not be consumed by the solver",
            ],
        ),
        thresholds=SolidStateMetrologyEvaluationThresholds(),
        provenance=SolidStateMetrologyEvaluationProvenance(
            generator=TOOL_PATH,
            generator_version=__version__,
            source_sha256={"collection-plan-template": "0" * 64},
            notes=[
                "collection plan is not a calibration result",
                "no physical PASS is generated until measured evidence is supplied",
            ],
        ),
    )


def _provenance(
    *,
    source_digest: str,
    input_path: Path | None,
    input_digest: str | None,
    command: list[str],
) -> SolidStateMetrologyEvaluationProvenance:
    source_sha256 = {"generator": source_digest}
    notes = [
        "physical reference values must be measured independently of the solver",
        "no physical PASS is generated when reference or repeat-session evidence is missing",
    ]
    if input_path is not None and input_digest is not None:
        source_sha256[str(input_path)] = input_digest
        notes.append("input artifact digest binds the declared reference and estimates")
    return SolidStateMetrologyEvaluationProvenance(
        generator=TOOL_PATH,
        generator_version=__version__,
        source_sha256=source_sha256,
        command=command,
        notes=notes,
    )


def _render_markdown(artifact: SolidStateMetrologyEvaluationArtifact) -> str:
    """Render a compact physical-evaluation report."""

    per_session_reference_count = sum(
        session.reference is not None
        and session.reference.transform is not None
        and session.reference.time_offset_sec is not None
        for session in artifact.sessions
    )
    lines = [
        "# Solid-state LiDAR physical ground-truth evaluation",
        "",
        f"- Schema: `{artifact.schema_version}`",
        f"- Evaluation: `{artifact.evaluation_id}`",
        f"- Status: **{artifact.status.upper()}**",
        f"- Decision: `{artifact.decision}`",
        "",
        "This report is PASS-capable only when independent spatial/temporal "
        "references for every usable session, repeat remount sessions, and the "
        "declared downstream holdout metric are all present.",
        "",
        "## Reference",
        "",
        f"- Extrinsic method: `{artifact.reference.extrinsic_method}`",
        f"- Clock method: `{artifact.reference.clock_method}`",
        f"- Independent of solver: `{artifact.reference.independent_of_solver}`",
        f"- Per-session references: `{per_session_reference_count}/"
        f"{artifact.metrics.usable_session_count}`",
        "",
        "## Runs",
        "",
        "| Estimate | Session | Remount | Rotation error (deg) | "
        "Translation error (m) | Clock error (ms) | Gate |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    if artifact.metrics.run_metrics:
        for metric in artifact.metrics.run_metrics:
            rotation = (
                "—"
                if metric.rotation_error_deg is None
                else f"{metric.rotation_error_deg:.6f}"
            )
            translation = (
                "—"
                if metric.translation_error_m is None
                else f"{metric.translation_error_m:.6f}"
            )
            clock = (
                "—"
                if metric.time_offset_error_sec is None
                else f"{metric.time_offset_error_sec * 1000.0:.3f}"
            )
            lines.append(
                f"| {metric.estimate_id} | {metric.session_id} | "
                f"{metric.remount_id or '—'} | {rotation} | {translation} | "
                f"{clock} | {'PASS' if metric.gate_passed else 'FAIL'} |"
            )
    else:
        lines.append("| — | — | — | — | — | — | NOT RUN |")
    verified_source_count = sum(
        check.status == "verified"
        for check in artifact.evidence_integrity.source_checks
    )
    lines.extend(
        [
            "",
            "## Evidence integrity",
            "",
            f"- Checked: `{artifact.evidence_integrity.checked}`",
            f"- Passed: `{artifact.evidence_integrity.passed}`",
            f"- Sources verified: `{verified_source_count}/"
            f"{len(artifact.evidence_integrity.source_checks)}`",
        ]
    )
    if artifact.evidence_integrity.source_checks:
        lines.extend(
            [
                "",
                "| Role | Owner | Path | Status |",
                "|---|---|---|---|",
            ]
        )
        lines.extend(
            f"| {check.role} | {check.owner_id} | {check.path} | {check.status} |"
            for check in artifact.evidence_integrity.source_checks
        )
    if artifact.evidence_integrity.issues:
        lines.extend(
            [
                "",
                "Integrity issues:",
                *(
                    f"- {issue}"
                    for issue in artifact.evidence_integrity.issues
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Usable sessions: `{artifact.metrics.usable_session_count}`",
            f"- Remounts: `{artifact.metrics.remount_count}`",
            f"- Passing runs: `{artifact.metrics.passing_run_count}/"
            f"{artifact.metrics.evaluated_run_count}`",
            f"- Downstream metric gate: `{artifact.metrics.downstream_metric_passed}`",
        ]
    )
    if artifact.reasons:
        lines.extend(["", "## Reasons", ""])
        lines.extend(f"- {reason}" for reason in artifact.reasons)
    lines.extend(
        [
            "",
            "The synthetic benchmark remains a separate solver-mechanics gate; this "
            "artifact is the path for physical metrology evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Load or create a template, evaluate it, and write schema-valid evidence."""

    args = _parser().parse_args(argv)
    if args.verify_sources and args.input is None:
        _parser().error("--verify-sources requires --input")
    if args.prepare_collection_plan and args.input is not None:
        _parser().error("--prepare-collection-plan cannot be combined with --input")
    if args.plan_sessions < 1:
        _parser().error("--plan-sessions must be >= 1")
    if args.plan_remounts < 1:
        _parser().error("--plan-remounts must be >= 1")
    if args.plan_sessions < args.plan_remounts:
        _parser().error("--plan-sessions must be >= --plan-remounts")
    command = [TOOL_PATH, *(argv or sys.argv[1:])]
    source_digest = sha256_path(Path(__file__).resolve())
    if source_digest is None:
        raise RuntimeError(f"could not hash generator: {Path(__file__).resolve()}")

    input_digest: str | None = None
    if args.input is None:
        artifact = (
            _collection_plan_template(
                session_count=args.plan_sessions,
                remount_count=args.plan_remounts,
            )
            if args.prepare_collection_plan
            else _template()
        )
    else:
        input_digest = sha256_path(args.input)
        if input_digest is None:
            raise RuntimeError(f"could not hash input artifact: {args.input}")
        artifact = SolidStateMetrologyEvaluationArtifact.model_validate(
            read_mapping(args.input)
        )
    artifact = artifact.model_copy(
        update={
            "provenance": _provenance(
                source_digest=source_digest,
                input_path=args.input,
                input_digest=input_digest,
                command=command,
            )
        }
    )
    if args.input is not None and (args.verify_sources or args.enforce):
        artifact = artifact.model_copy(
            update={
                "evidence_integrity": verify_solid_state_metrology_evidence(
                    artifact,
                    base_dir=args.input.resolve().parent,
                )
            }
        )
    evaluated = evaluate_solid_state_metrology(artifact)
    write_mapping(args.output, evaluated.model_dump(mode="json", exclude_none=True))
    if args.markdown_output is not None:
        write_text(args.markdown_output, _render_markdown(evaluated))

    payload = {
        "schema_version": SOLID_STATE_METROLOGY_EVALUATION_SCHEMA_VERSION,
        "output": str(args.output),
        "status": evaluated.status,
        "decision": evaluated.decision,
        "run_count": evaluated.metrics.run_count,
        "passing_run_count": evaluated.metrics.passing_run_count,
        "evidence_integrity_checked": evaluated.evidence_integrity.checked,
        "evidence_integrity_passed": evaluated.evidence_integrity.passed,
        "per_session_reference_count": sum(
            session.reference is not None
            and session.reference.transform is not None
            and session.reference.time_offset_sec is not None
            for session in evaluated.sessions
        ),
        "evidence_source_check_count": len(
            evaluated.evidence_integrity.source_checks
        ),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"solid-state metrology evaluation: {evaluated.status} "
            f"runs={evaluated.metrics.passing_run_count}/"
            f"{evaluated.metrics.evaluated_run_count} "
            f"decision={evaluated.decision}"
        )
    if args.enforce and evaluated.status != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
