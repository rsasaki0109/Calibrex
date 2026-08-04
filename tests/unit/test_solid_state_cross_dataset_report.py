from __future__ import annotations

from pathlib import Path

from calibrex.core.solid_state_cross_dataset_benchmark import (
    SolidStateCrossDatasetBenchmarkAggregate,
    SolidStateCrossDatasetBenchmarkConfidenceInterval,
    SolidStateCrossDatasetBenchmarkDataset,
    SolidStateCrossDatasetBenchmarkManifest,
    SolidStateCrossDatasetBenchmarkProtocol,
    SolidStateCrossDatasetBenchmarkProvenance,
    SolidStateCrossDatasetBenchmarkReplicate,
    SolidStateCrossDatasetBenchmarkVariant,
)
from calibrex.core.solid_state_cross_dataset_report import (
    render_solid_state_cross_dataset_benchmark_html,
    render_solid_state_cross_dataset_benchmark_markdown,
    write_solid_state_cross_dataset_benchmark_reports,
)


def _manifest() -> SolidStateCrossDatasetBenchmarkManifest:
    adaptive = SolidStateCrossDatasetBenchmarkVariant(
        id="adaptive_mad",
        voxel_strategy="adaptive",
        outlier_policy="mad",
        optimization_status="converged",
        final_holdout_rmse_m=0.1,
        holdout_correspondence_count=20,
        observability_rank=6,
    )
    uniform = SolidStateCrossDatasetBenchmarkVariant(
        id="uniform_none",
        voxel_strategy="uniform",
        outlier_policy="none",
        optimization_status="converged",
        final_holdout_rmse_m=0.2,
        holdout_correspondence_count=20,
        observability_rank=6,
    )
    replicate = SolidStateCrossDatasetBenchmarkReplicate(
        id="example__late__seed_17",
        split_id="late",
        seed=17,
        ablation_manifest_path="ablation.yaml",
        status="scored",
        variants=[adaptive, uniform],
        holdout_improvement_percent=50.0,
        winner="adaptive",
        comparison_note="paired holdout",
    )
    interval = SolidStateCrossDatasetBenchmarkConfidenceInterval(
        confidence_level=0.95,
        lower=40.0,
        upper=60.0,
        statistic="mean_holdout_improvement_percent",
    )
    dataset = SolidStateCrossDatasetBenchmarkDataset(
        id="example",
        name="Example",
        family="fixture",
        dataset_manifest_path="manifest.yaml",
        config_path="config.yaml",
        ablation_manifest_path="ablation.yaml",
        reference_mode="trajectory_only",
        status="scored",
        variants=[adaptive, uniform],
        adaptive_variant_id="adaptive_mad",
        uniform_variant_id="uniform_none",
        holdout_improvement_percent=50.0,
        winner="adaptive",
        comparison_note="one replicate",
        replicates=[replicate],
        replicate_count=1,
        scored_replicate_count=1,
        adaptive_win_rate=1.0,
        holdout_improvement_mean_percent=50.0,
        holdout_improvement_median_percent=50.0,
        holdout_improvement_ci=interval,
    )
    protocol = SolidStateCrossDatasetBenchmarkProtocol(
        name="fixture",
        solver="continuous_time_lidar_pair",
        variants=["uniform_none", "adaptive_mad"],
        bounded_replay_note="fixture",
    )
    aggregate = SolidStateCrossDatasetBenchmarkAggregate(
        dataset_count=1,
        scored_dataset_count=1,
        inconclusive_dataset_count=0,
        adaptive_wins=1,
        uniform_wins=0,
        ties=0,
        mean_holdout_improvement_percent=50.0,
        median_holdout_improvement_percent=50.0,
        total_replicate_count=1,
        scored_replicate_count=1,
        adaptive_win_rate=1.0,
        holdout_improvement_ci=interval,
        conclusion="fixture conclusion",
    )
    return SolidStateCrossDatasetBenchmarkManifest(
        tool_version="0.4.0",
        spec_path="spec.yaml",
        spec_sha256="a" * 64,
        protocol=protocol,
        datasets=[dataset],
        aggregate=aggregate,
        provenance=SolidStateCrossDatasetBenchmarkProvenance(
            source_paths=["spec.yaml"],
            source_sha256={"spec.yaml": "a" * 64},
            tool_version="0.4.0",
        ),
    )


def test_solid_state_benchmark_reports_include_replicates_ci_and_provenance(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    markdown = render_solid_state_cross_dataset_benchmark_markdown(manifest)
    html = render_solid_state_cross_dataset_benchmark_html(manifest)
    assert "example" in markdown
    assert "late" in markdown
    assert "[40.00, 60.00]%" in markdown
    assert "fixture conclusion" in html
    assert "example__late__seed_17" not in html

    paths = write_solid_state_cross_dataset_benchmark_reports(
        manifest,
        markdown_path=tmp_path / "report.md",
        html_path=tmp_path / "report.html",
    )
    assert paths == (tmp_path / "report.md", tmp_path / "report.html")
    assert (tmp_path / "report.md").read_text(encoding="utf-8") == markdown
    assert (tmp_path / "report.html").read_text(encoding="utf-8") == html
