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
    SolidStateMetrologyEvaluationArtifact,
    SolidStateMetrologyEvaluationProtocol,
    SolidStateMetrologyEvaluationProvenance,
    SolidStateMetrologyEvaluationThresholds,
    SolidStateMetrologyReference,
    evaluate_solid_state_metrology,
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

    lines = [
        "# Solid-state LiDAR physical ground-truth evaluation",
        "",
        f"- Schema: `{artifact.schema_version}`",
        f"- Evaluation: `{artifact.evaluation_id}`",
        f"- Status: **{artifact.status.upper()}**",
        f"- Decision: `{artifact.decision}`",
        "",
        "This report is PASS-capable only when independent spatial/temporal "
        "references, repeat remount sessions, and the declared downstream holdout "
        "metric are all present.",
        "",
        "## Reference",
        "",
        f"- Extrinsic method: `{artifact.reference.extrinsic_method}`",
        f"- Clock method: `{artifact.reference.clock_method}`",
        f"- Independent of solver: `{artifact.reference.independent_of_solver}`",
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
    command = [TOOL_PATH, *(argv or sys.argv[1:])]
    source_digest = sha256_path(Path(__file__).resolve())
    if source_digest is None:
        raise RuntimeError(f"could not hash generator: {Path(__file__).resolve()}")

    input_digest: str | None = None
    if args.input is None:
        artifact = _template()
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
