from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from calibrex.core.benchmark import (
    aggregate_benchmark_definition,
    load_benchmark,
    load_benchmark_definition,
    render_benchmark_markdown,
    render_benchmark_table_markdown,
)

ROOT = Path(__file__).resolve().parents[2]
ASSET = ROOT / "docs" / "assets" / "ethz-robot-world-hand-eye-benchmark.json"
DEFINITION = ROOT / "docs" / "assets" / "ethz-robot-world-hand-eye-benchmark.definition.json"
MARKDOWN = ROOT / "docs" / "assets" / "ethz-robot-world-hand-eye-benchmark.md"


def test_ethz_benchmark_definition_and_asset_are_schema_valid() -> None:
    benchmark = json.loads(ASSET.read_text(encoding="utf-8"))
    definition = json.loads(DEFINITION.read_text(encoding="utf-8"))
    benchmark_schema = json.loads(
        (ROOT / "schemas" / "benchmark.schema.json").read_text(encoding="utf-8")
    )
    definition_schema = json.loads(
        (ROOT / "schemas" / "benchmark_definition.schema.json").read_text(encoding="utf-8")
    )

    jsonschema.validate(
        benchmark,
        benchmark_schema,
        format_checker=jsonschema.FormatChecker(),
    )
    jsonschema.validate(
        definition,
        definition_schema,
        format_checker=jsonschema.FormatChecker(),
    )


def test_ethz_benchmark_asset_is_exactly_regenerated_from_definition() -> None:
    definition = load_benchmark_definition(DEFINITION)
    committed = load_benchmark(ASSET)
    regenerated = aggregate_benchmark_definition(definition)

    assert regenerated == committed
    assert render_benchmark_markdown(regenerated) == MARKDOWN.read_text(encoding="utf-8")


def test_readme_and_docs_home_embed_the_generated_ethz_table() -> None:
    benchmark = load_benchmark(ASSET)
    start = f"<!-- calibrex-benchmark:{benchmark.benchmark_id}:start -->"
    end = f"<!-- calibrex-benchmark:{benchmark.benchmark_id}:end -->"
    expected = render_benchmark_table_markdown(benchmark)

    for path in [ROOT / "docs" / "index.md"]:
        text = path.read_text(encoding="utf-8")
        embedded = text.split(start, maxsplit=1)[1].split(end, maxsplit=1)[0].strip()
        assert embedded == expected


def test_ethz_benchmark_scoped_winners_match_aggregates() -> None:
    benchmark = load_benchmark(ASSET)
    translation = {
        method_id: summary.metrics["translation_holdout_rmse_m"]
        for method_id, summary in benchmark.method_summaries.items()
    }
    rotation = {
        method_id: summary.metrics["rotation_holdout_rmse_deg"]
        for method_id, summary in benchmark.method_summaries.items()
    }

    assert translation["calibrex_dornaika_horaud_nonlinear"].rank == 1
    assert translation["calibrex_dornaika_horaud_nonlinear"].distribution.mean == (
        pytest.approx(0.010039379464848725)
    )
    assert rotation["shah"].rank == 1
    assert all(summary.failure_rate == 0.0 for summary in benchmark.method_summaries.values())


def test_ethz_benchmark_uses_one_digest_locked_shared_protocol() -> None:
    benchmark = load_benchmark(ASSET)
    split = benchmark.protocol.splits[0]

    assert split.fit_count == 1350
    assert split.holdout_count == 338
    assert split.fit_ids_sha256 == (
        "96ece4f6c83df57b4a087480507e7a0cb0bccdab870d5e5e7544994a8a28637f"
    )
    assert split.holdout_ids_sha256 == (
        "0a5f650eb71f91f4960cf5f99068d1d1c4ce78857f0bb7904d71d098bea9b837"
    )
    assert benchmark.protocol.dataset_source_sha256 == (
        "2454578f731e656a940ddf51017b56e8d535c58d16a50629b85326005dedd6c3"
    )
