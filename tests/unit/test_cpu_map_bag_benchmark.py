"""Full-bag summaries must preserve unfavorable workers and strict quality."""

import copy
import math

import pytest
from tools import benchmark_cpu_map_bag as benchmark
from tools.benchmark_cpu_map_bag import compare_workers, main, summarize


def _record(variant, budget):
    commit = "a" * 40 if variant == "prior" else "b" * 40
    command = [
        "python",
        "-m",
        "tools.benchmark_cpu_map_bag",
        "--worker",
        variant,
        "--scenario",
        "map-000",
        "--repeat",
        "0",
    ]
    return {
        "variant": variant,
        "scenario": "map-000",
        "repeat": 0,
        "artifact": {
            "provenance": {"created_at": variant, "command": command, "git_commit": commit},
            "options": {"registration_query_workers": budget},
            "covariance": [[0.1]],
            "decision": "pass",
        },
        "options": {"solver": {"query_workers": budget, "huber_m": 0.05}},
        "fits": [{"transform": [[0.1]], "covariance": [[0.01]], "count": 100}] * 5,
        "prepared_samples": [{"map_points": {"sha256": "c" * 64, "shape": [300, 3]}}],
        "provenance": {
            "query_workers": budget,
            "source_commit": commit,
            "bag_sha256": "d" * 64,
            "odometry_backend": "cuda",
        },
    }


def _triad(repeat, prior, lazy, parallel):
    return {
        "scenario": "map-000",
        "repeat": repeat,
        "seconds": {"prior": prior, "lazy": lazy, "parallel": parallel},
        "errors": {"lazy": [], "parallel": []},
    }


def test_only_declared_run_metadata_and_worker_budgets_are_normalized():
    assert compare_workers(_record("prior", 1), _record("parallel", 8)) == []
    current = _record("lazy", 1)
    current["artifact"]["options"]["other_policy"] = "changed"
    assert "artifact" in compare_workers(_record("prior", 1), current)


@pytest.mark.parametrize("field", ["artifact", "fits", "prepared_samples"])
def test_physical_covariance_and_sample_identity_changes_are_rejected(field):
    current = _record("parallel", 8)
    if field == "artifact":
        current["artifact"]["covariance"][0][0] = math.nextafter(0.1, math.inf)
    elif field == "fits":
        current["fits"][0]["covariance"][0][0] = math.nextafter(0.01, math.inf)
    else:
        current["prepared_samples"][0]["map_points"]["sha256"] = "e" * 64
    assert field in compare_workers(_record("prior", 1), current)


def test_false_thread_and_command_provenance_are_rejected():
    current = _record("parallel", 8)
    current["artifact"]["options"]["registration_query_workers"] = 1
    current["artifact"]["provenance"]["command"] = ["altered"]
    assert {"query_workers", "command"} <= set(compare_workers(_record("prior", 1), current))


def test_slow_worker_and_failed_quality_remain_in_summary():
    rows = [_triad(0, 20.0, 15.0, 10.0), _triad(1, 21.0, 16.0, 40.0), _triad(2, 22.0, 17.0, 12.0)]
    rows[1]["errors"]["parallel"] = ["fits"]
    result = summarize(rows)
    assert result["median_seconds"]["parallel"] == 12.0
    assert result["parallel_faster_pairs"] == 2 and result["workers"] == 9
    assert result["fits"] == 45 and not result["quality_passed"]
    assert result["prior_to_parallel"] == 21.0 / 12.0


@pytest.mark.parametrize("kind", ["duplicate", "missing_repeat", "density", "variant", "time"])
def test_incomplete_or_pooled_cohorts_are_rejected(kind):
    rows = [_triad(0, 20.0, 15.0, 10.0), _triad(1, 21.0, 16.0, 11.0)]
    if kind == "duplicate":
        rows[1]["repeat"] = 0
    elif kind == "missing_repeat":
        rows[1]["repeat"] = 2
    elif kind == "density":
        rows[1]["scenario"] = "map-001"
    elif kind == "variant":
        rows[1]["seconds"].pop("parallel")
    else:
        rows[1]["seconds"]["parallel"] = math.nan
    with pytest.raises(ValueError):
        summarize(rows)


@pytest.mark.parametrize(
    "extra",
    [[], ["--query-workers", "0"], ["--max-seconds", "nan"], ["--map-voxel-sizes-m", "0.2", "0.2"]],
)
def test_invalid_or_missing_benchmark_configuration_fails_before_creating_outputs(
    tmp_path,
    monkeypatch,
    extra,
):
    output = tmp_path / "new-output"

    def unexpected_run(*args, **kwargs):
        pytest.fail("invalid configuration started a benchmark")

    monkeypatch.setattr(benchmark, "_run", unexpected_run)
    common = (
        []
        if not extra
        else [
            "--bag",
            str(tmp_path / "public.bag"),
            "--baseline-commit",
            "a" * 40,
            "--dataset-family",
            "public",
            "--dataset-license",
            "public terms",
        ]
    )
    with pytest.raises(SystemExit) as error:
        main(["--output-dir", str(output), *common, *copy.deepcopy(extra)])
    assert error.value.code == 2 and not output.exists()
