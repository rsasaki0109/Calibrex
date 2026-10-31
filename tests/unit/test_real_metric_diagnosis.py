import numpy as np
from tools.diagnose_real_metric_transfer import summarize


def test_summary_excludes_invalid_points_from_signed_distribution():
    result = summarize(
        np.array([-2.0, 0.0, 3.0, 1.0]),
        np.array([-2.0, 0.0, 0.2, 1.0]),
        np.array([True, True, True, False]),
    )
    assert result["supported"] == 3
    assert result["signed_quantiles_m"][2] == 0
    assert result["negative_fraction"] == 1 / 3
    assert result["absolute_over_1m_fraction"] == 2 / 3
    assert result["cost"] > result["supported_cost"]


def test_empty_support_is_explicit_and_finite():
    result = summarize(np.ones(2), np.ones(2), np.zeros(2, dtype=bool))
    assert result["supported"] == 0
    assert result["signed_quantiles_m"] is None
    assert result["supported_cost"] == 0
