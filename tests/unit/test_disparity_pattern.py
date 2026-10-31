import numpy as np
from tools.joint_disparity_pattern import search


def test_search_counts_budget_and_reduces_measured_cost():
    calls = []
    target = np.array([0.25, -0.25, 0, 0.1, 0, 0, 0.05])

    def objective(x):
        calls.append(x.copy())
        return float(np.sum((x - target) ** 2))

    fit = search(objective)
    assert fit["success"] and fit["calls"] == len(calls) <= 800
    np.testing.assert_allclose(fit["delta"], target, atol=0.005)
    assert np.all(np.diff(fit["trace"]) <= 0)
    limited = search(objective, budget=10)
    assert limited["calls"] == 10 and not limited["success"]
