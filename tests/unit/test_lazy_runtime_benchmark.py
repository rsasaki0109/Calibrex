from __future__ import annotations

from copy import deepcopy

import pytest

from tools.benchmark_lazy_runtime_loading import comparable_result


def _result() -> dict:
    return {
        "provenance": {
            "created_at": "first run",
            "git_commit": "first commit",
            "command": ["calibrex", "refine", "input.json", "--output", "first.json"],
        },
        "trace": [{"objective": 1.25}],
        "acceptance": {"accepted": True},
        "options": {"max_evaluations": 400},
    }


def test_normalization_only_ignores_declared_execution_provenance() -> None:
    prior = _result()
    current = deepcopy(prior)
    current["provenance"].update(created_at="second run", git_commit="second commit")
    current["provenance"]["command"][-1] = "second.json"
    assert comparable_result(prior) == comparable_result(current)
    assert prior == _result()


@pytest.mark.parametrize("field", ["trace", "acceptance", "options", "input"])
def test_normalization_never_hides_changed_results_or_inputs(field: str) -> None:
    prior = _result()
    current = deepcopy(prior)
    if field == "input":
        current["provenance"]["command"][2] = "other-input.json"
    else:
        current[field] = {"changed": True}
    assert comparable_result(prior) != comparable_result(current)
