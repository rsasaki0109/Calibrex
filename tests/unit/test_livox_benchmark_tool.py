from copy import deepcopy
from pathlib import Path

import pytest
from tools import benchmark_livox_evaluation as benchmark


def test_result_normalization_preserves_numerical_diagnostics_and_original_payload():
    payload = {
        "run": {"id": "old", "created_at": "old", "git_commit": "measured",
                "provenance": {"generated_at": "old", "input_sha256": "verified"}},
        "metrics": {"score": {"value": 0.123, "grade": "fail", "reason": "unchanged"}},
        "cases": [{"support": 4321, "probe": [1.0, -0.0, 2.0]}],
        "artifacts": {"report": str(Path("output") / "report.html")},
    }
    original = deepcopy(payload)
    normalized = benchmark._normalized_result(payload, Path("output"))
    assert payload == original
    assert normalized["metrics"] == original["metrics"]
    assert normalized["cases"] == original["cases"]
    assert normalized["run"]["git_commit"] == "measured"
    assert normalized["run"]["provenance"]["input_sha256"] == "verified"
    assert normalized["artifacts"]["report"] == "${RUN_OUTPUT}/report.html"
    payload["metrics"]["score"]["value"] += 0.001
    assert benchmark._normalized_result(payload, Path("output")) != normalized


@pytest.mark.parametrize("option", ["--repetitions", "--omp-threads"])
def test_invalid_workload_is_rejected_before_execution(tmp_path, option):
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="must be positive"):
        benchmark.main(["--output-dir", str(output), option, "0"])
    assert not output.exists()
