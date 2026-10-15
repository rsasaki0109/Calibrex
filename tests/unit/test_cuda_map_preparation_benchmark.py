"""CPU-only controls for the full CUDA map comparison protocol."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from tools import benchmark_cuda_map_preparation as benchmark


def _evaluation() -> dict:
    return {"result": {"execution_backend":
            "calibrex.cuda_lidar_map/v0.1;fp64=true;query_fma=false;cupy=14.2.0;"
            "cuda=12090;driver=13000;cc=7.5;query_sha256=" + "0" * 64,
            "transform": [[1., 0.]], "covariance": [[1e-10]], "correspondences": 300},
            "policy_status": "pass", "known_bad": {"detected": True, "score": 12.}}


@pytest.mark.parametrize("mutation", ["identity", "hidden", "count", "numeric", "decision"])
def test_comparator_retains_both_identities_and_full_quality(mutation: str) -> None:
    prior, current = _evaluation(), _evaluation()
    current["result"]["execution_backend"] += ";normal_sha256=" + "1" * 64 + ";normal_cache=lazy"
    assert not benchmark.compare(prior, current)
    if mutation == "identity":
        prior["result"]["execution_backend"] = "untracked"
    elif mutation == "hidden":
        prior["result"]["execution_backend"] += ";hidden=true"
    elif mutation == "count":
        current["result"]["correspondences"] += 1
    elif mutation == "numeric":
        current["result"]["transform"][0][1] = 1e-5
    else:
        current["known_bad"]["detected"] = False
    assert benchmark.compare(prior, current)


def test_summary_keeps_first_compilation_and_rejects_incomplete_pairs() -> None:
    rows = [{"repeat": i, "variant": v, "wall_seconds": t, "api_seconds": t / 2, "errors": []}
            for i, v, t in ((0, "prior", 30.), (0, "current", 20.),
                            (1, "current", 2.), (1, "prior", 4.))]
    summary = benchmark.summarize(rows)
    assert summary["median_seconds"]["wall_seconds"] == {"prior": 17., "current": 11.}
    assert summary["whole_speedup"] == 17. / 11.
    assert summary["workers"] == 4
    assert summary["pairs"] == [{"repeat": 0, "whole_speedup": 1.5},
                                {"repeat": 1, "whole_speedup": 2.}]
    with pytest.raises(ValueError, match="incomplete"):
        benchmark.summarize(rows[:-1])
    changed = deepcopy(rows)
    changed[0]["errors"] = ["changed"]
    assert not benchmark.summarize(changed)["quality_passed"]


def test_invalid_or_existing_outputs_are_preserved(tmp_path: Path) -> None:
    base = ["--sequence", str(tmp_path), "--output-dir", str(tmp_path)]
    with pytest.raises(ValueError, match="half-window"):
        benchmark.main([*base, "--half-window", "2"])
    with pytest.raises(ValueError, match="baseline"):
        benchmark.main(base)
    with pytest.raises(ValueError, match="new"):
        benchmark.main([*base, "--baseline-commit", "HEAD"])
