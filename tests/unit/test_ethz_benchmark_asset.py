from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[2]
ASSET = ROOT / "docs" / "assets" / "ethz-robot-world-hand-eye-benchmark.json"
SCHEMA = ROOT / "docs" / "assets" / "ethz-robot-world-hand-eye-benchmark.schema.json"


def test_ethz_benchmark_asset_is_schema_valid() -> None:
    benchmark = json.loads(ASSET.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))

    jsonschema.validate(benchmark, schema, format_checker=jsonschema.FormatChecker())


def test_ethz_benchmark_claim_matches_results() -> None:
    benchmark = json.loads(ASSET.read_text(encoding="utf-8"))
    claim = benchmark["claim"]
    results = benchmark["results"]
    winner = min(results, key=lambda result: result[claim["metric"]])

    assert winner["method_id"] == claim["winner"]
    assert winner[claim["metric"]] == pytest.approx(claim["value"])
    assert winner["calibrex_refinement"] is True
    assert len({result["known_bad_detectable_fraction"] for result in results}) == 1


def test_ethz_benchmark_uses_one_shared_protocol() -> None:
    benchmark = json.loads(ASSET.read_text(encoding="utf-8"))
    protocol = benchmark["protocol"]

    assert protocol["fit_pair_count"] == 1350
    assert protocol["holdout_pair_count"] == 338
    assert protocol["known_bad_case_count_per_method"] == 24
    assert protocol["dataset_source_sha256"] == (
        "2454578f731e656a940ddf51017b56e8d535c58d16a50629b85326005dedd6c3"
    )
