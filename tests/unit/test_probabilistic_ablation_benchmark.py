"""Comparison keeps estimator evidence while ignoring declared runtime lineage."""

from copy import deepcopy

import pytest
from tools.benchmark_probabilistic_refinement_ablation import (
    comparable_benchmark,
    comparable_result,
)


@pytest.mark.parametrize(
    "field", ["trace", "acceptance", "options", "holdout", "status", "provider"]
)
def test_result_comparison_keeps_every_quality_field(field):
    value = {
        "provenance": {"created_at": "a", "git_commit": "b", "execution": {}, "input": "c"},
        "trace": [],
        "acceptance": True,
        "options": {},
        "holdout": 1.0,
        "status": "converged",
        "provider": {},
    }
    original = deepcopy(value)
    normalized = comparable_result(value)
    assert value == original
    changed = deepcopy(value)
    changed["provenance"].update(created_at="x", git_commit="y", execution={"backend": "cuda"})
    assert comparable_result(changed) == normalized
    changed[field] = "changed"
    assert comparable_result(changed) != normalized


@pytest.mark.parametrize(
    "field", ["metrics", "status", "failure_reason", "split_id", "config_sha256"]
)
def test_benchmark_comparison_retains_failures_and_trial_identity(field):
    value = {
        "provenance": {"generated_at": "a", "git_commit": "b", "source_artifacts": {"x": "c"}},
        "methods": [{"source_commit": "b", "method_id": "full"}],
        "trials": [
            {
                "runtime_seconds": 1.0,
                "metrics": {},
                "status": "failed",
                "failure_reason": "fixture",
                "split_id": "seed-0000",
                "provenance": {"output_sha256": "d", "notes": ["paired"], "config_sha256": "e"},
            }
        ],
        "method_summaries": {"full": {"runtime_seconds": {}, "success_rate": 0.0}},
    }
    expected = comparable_benchmark(value, cuda=False)
    changed = deepcopy(value)
    if field == "config_sha256":
        changed["trials"][0]["provenance"][field] = "changed"
    else:
        changed["trials"][0][field] = "changed"
    assert comparable_benchmark(changed, cuda=False) != expected
    cuda = deepcopy(value)
    cuda["trials"][0]["provenance"]["notes"].append("pose_evaluation_backend=cuda")
    assert comparable_benchmark(cuda, cuda=True) == expected
    cuda["trials"][0]["provenance"]["notes"].append("undeclared normalization")
    with pytest.raises(AssertionError):
        comparable_benchmark(cuda, cuda=True)
