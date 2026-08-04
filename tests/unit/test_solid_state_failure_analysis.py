from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from calibrex.core.solid_state_failure_analysis import (
    SOLID_STATE_FAILURE_ANALYSIS_SCHEMA_VERSION,
    SolidStateFailureAnalysisFinding,
    SolidStateFailureAnalysisManifest,
    SolidStateFailureAnalysisProvenance,
    SolidStateFailureAnalysisReplicate,
    SolidStateFailureAnalysisSummary,
    SolidStateFailureAnalysisVariant,
)
from calibrex.core.solid_state_failure_analysis_report import (
    render_solid_state_failure_analysis_html,
    render_solid_state_failure_analysis_markdown,
)


def _variant(
    variant_id: str,
    *,
    strategy: str,
    policy: str,
    train: float,
    holdout: float,
    rejected: int,
) -> SolidStateFailureAnalysisVariant:
    return SolidStateFailureAnalysisVariant(
        id=variant_id,
        voxel_strategy=strategy,  # type: ignore[arg-type]
        outlier_policy=policy,  # type: ignore[arg-type]
        result_path=f"{variant_id}.yaml",
        optimization_status="converged",
        final_train_rmse_m=train,
        final_holdout_rmse_m=holdout,
        train_correspondence_count=100,
        holdout_correspondence_count=50,
        observability_rank=6,
        condition_number=4.0,
        outlier_rejected_count=rejected,
        outlier_rejection_rate=rejected / (150 + rejected),
        train_holdout_gap_m=holdout - train,
    )


def _analysis() -> SolidStateFailureAnalysisManifest:
    adaptive = _variant(
        "adaptive_mad",
        strategy="adaptive",
        policy="mad",
        train=0.20,
        holdout=0.40,
        rejected=10,
    )
    uniform = _variant(
        "uniform_none",
        strategy="uniform",
        policy="none",
        train=0.21,
        holdout=0.30,
        rejected=0,
    )
    replicate = SolidStateFailureAnalysisReplicate(
        id="example__late__seed_0",
        split_id="late_holdout",
        seed=0,
        status="scored",
        winner="uniform",
        holdout_improvement_percent=-33.333,
        adaptive=adaptive,
        uniform=uniform,
        adaptive_holdout_minus_uniform_m=0.10,
        adaptive_train_minus_uniform_m=-0.01,
        adaptive_generalization_gap_minus_uniform_m=0.11,
        diagnosis="uniform_wins_with_adaptive_train_advantage",
        signals=["train_holdout_mismatch"],
    )
    return SolidStateFailureAnalysisManifest(
        tool="tools/run_solid_state_failure_analysis.py",
        tool_version="0.4.0",
        benchmark_manifest_path="benchmark.yaml",
        benchmark_manifest_sha256="a" * 64,
        dataset_id="example",
        dataset_name="Example",
        analysis_scope="fixture",
        summary=SolidStateFailureAnalysisSummary(
            replicate_count=1,
            scored_replicate_count=1,
            adaptive_wins=0,
            uniform_wins=1,
            ties=0,
            adaptive_win_rate=0.0,
            mean_holdout_improvement_percent=-33.333,
            mean_adaptive_train_minus_uniform_m=-0.01,
            mean_adaptive_generalization_gap_minus_uniform_m=0.11,
            mean_adaptive_outlier_rejection_rate=10 / 160,
            adaptive_rank_six_count=1,
            uniform_rank_six_count=1,
            adaptive_converged_count=1,
            uniform_converged_count=1,
            diagnosis_counts={"uniform_wins_with_adaptive_train_advantage": 1},
        ),
        replicates=[replicate],
        findings=[
            SolidStateFailureAnalysisFinding(
                id="fixture_finding",
                severity="candidate_cause",
                confidence="low",
                title="Fixture signal",
                evidence="fixture evidence",
                supporting_replicates=[replicate.id],
                caveat="fixture caveat",
            )
        ],
        conclusion="fixture conclusion",
        provenance=SolidStateFailureAnalysisProvenance(
            source_paths=["benchmark.yaml"],
            source_sha256={"benchmark.yaml": "a" * 64},
            tool_version="0.4.0",
        ),
    )


def test_failure_analysis_schema_and_reports() -> None:
    analysis = _analysis()
    schema = json.loads(
        Path("schemas/solid_state_failure_analysis.schema.json").read_text(encoding="utf-8")
    )
    payload = analysis.model_dump(mode="json", exclude_none=True)
    jsonschema.validate(payload, schema)
    assert analysis.schema_version == SOLID_STATE_FAILURE_ANALYSIS_SCHEMA_VERSION
    assert "train_holdout_mismatch" in render_solid_state_failure_analysis_markdown(analysis)
    assert "fixture conclusion" in render_solid_state_failure_analysis_html(analysis)
