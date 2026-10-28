"""Finite-sample ranks, abstention, leakage rejection and paired coverage."""

from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from calibrex.core.empirical_intervals import EmpiricalIntervalModel
from calibrex.evaluation.empirical_intervals import (
    evaluate_empirical_intervals,
    fit_empirical_intervals,
    interval_scope,
)
from calibrex.evaluation.uncertainty_coverage import (
    CoverageAxisObservation,
    CoverageTrial,
    UncertaintyCoverageReport,
    summarize_coverage,
)

SCOPE = "a" * 64
CAL_REPORT = "b" * 64
EVAL_REPORT = "c" * 64
MODEL = "d" * 64
META = {"command": [], "source_sha256": {}, "git_commit": "test"}


def report(errors: list[float], *, start: int = 0) -> UncertaintyCoverageReport:
    trials = [
        CoverageTrial(
            seed=start + i,
            fit_status="stalled",
            policy_status="warn",
            artifact_sha256=hashlib.sha256(str(start + i).encode()).hexdigest(),
            axes=[CoverageAxisObservation(axis="x", error=e, std=1.0)],
        )
        for i, e in enumerate(errors)
    ]
    return UncertaintyCoverageReport(
        nominal_coverage=0.95,
        control_std_scale=0.1,
        producer="synthetic_bag",
        trials=trials,
        axes=summarize_coverage(trials, ["x"]),
        limitations=[],
        **META,
    )


def fit(r: UncertaintyCoverageReport, nominal: float = 0.95) -> EmpiricalIntervalModel:
    return fit_empirical_intervals(
        r, scope_sha256=SCOPE, report_sha256=CAL_REPORT, nominal=nominal, **META
    )


def evaluate(m: EmpiricalIntervalModel, r: UncertaintyCoverageReport, **kwargs: str):
    return evaluate_empirical_intervals(
        m,
        r,
        scope_sha256=kwargs.get("scope", SCOPE),
        model_sha256=MODEL,
        report_sha256=kwargs.get("report_digest", EVAL_REPORT),
        **META,
    )


def test_finite_sample_rank_and_insufficient_data_abstention() -> None:
    for n in (5, 18):
        interval = fit(report([1.0] * n)).intervals[0]
        assert interval.status == "inconclusive" and interval.half_width is None
    for n in (19, 20):
        interval = fit(report(list(range(1, n + 1)))).intervals[0]
        assert interval.order_rank == n and interval.half_width == n


def test_exact_exchangeable_leave_one_out_coverage() -> None:
    # All 20 ranks are equally likely to be the future draw. With 19 calibration
    # observations and p=.95, exactly the held-out largest rank is uncovered.
    covered = 0
    for heldout in range(1, 21):
        training = [x for x in range(1, 21) if x != heldout]
        width = fit(report(training)).intervals[0].half_width
        covered += heldout <= width
    assert covered == 19


def test_paired_coverage_does_not_modify_frozen_model() -> None:
    model = fit(report(list(range(1, 21))))
    before = model.model_dump_json()
    result = evaluate(model, report([1.0, 2.0, 19.0, 21.0], start=100)).axes[0]
    assert result.covered == 3 and result.formal_covered == 1
    assert result.shrunken_covered == 2
    assert result.half_width == 20 and result.available == 4
    assert result.conditional_coverage == 0.75
    assert model.model_dump_json() == before
    # Held-out outliers cannot enlarge the previously fitted radius.
    assert evaluate(model, report([100.0] * 10, start=100)).axes[0].covered == 0
    assert model.model_dump_json() == before


def test_failed_calibration_attempts_are_infinite_scores() -> None:
    r = report(list(range(1, 21)))
    r.trials[-1] = CoverageTrial(
        seed=19, fit_status="exception", policy_status="failed", failure="missing map"
    )
    model = fit(r)
    assert model.intervals[0].order_rank == 20
    assert model.intervals[0].available == 19
    assert model.intervals[0].half_width is None
    score = evaluate(model, report([1.0, 2.0], start=100)).axes[0]
    assert score.available == 0 and score.conditional_coverage is None
    assert score.formal_covered == 1
    assert score.all_attempt_coverage == 0


def test_failed_evaluation_attempts_remain_in_denominator() -> None:
    r = report([1.0, 2.0], start=100)
    r.trials[-1] = CoverageTrial(
        seed=101, fit_status="exception", policy_status="failed", failure="failed fit"
    )
    score = evaluate(fit(report([3.0] * 20)), r).axes[0]
    assert score.attempts == 2 and score.available == 1 and score.covered == 1
    assert score.conditional_coverage == 1 and score.all_attempt_coverage == 0.5


def test_rejects_scope_seed_report_and_renamed_artifact_leakage() -> None:
    training = report([1.0] * 20)
    model = fit(training)
    with pytest.raises(ValueError, match="scope"):
        evaluate(model, report([1.0], start=100), scope="e" * 64)
    with pytest.raises(ValueError, match="seed overlap"):
        evaluate(model, report([1.0]))
    with pytest.raises(ValueError, match="also be evaluation"):
        evaluate(model, report([1.0], start=100), report_digest=CAL_REPORT)
    r = report([1.0], start=100)
    r.trials[0].artifact_sha256 = training.trials[0].artifact_sha256
    with pytest.raises(ValueError, match="artifact overlap"):
        evaluate(model, r)


def test_scope_changes_with_pipeline_settings_or_sources() -> None:
    r = report([1.0] * 20)
    protocol = {
        "producer": r.producer,
        "seeds": [t.seed for t in r.trials],
        "source_sha256": r.source_sha256,
        "duration_s": 8.0,
        "map_rounds": 4,
    }
    scope = interval_scope(r, protocol)
    assert interval_scope(r, dict(protocol, duration_s=12.0)) != scope
    with pytest.raises(ValueError, match="disagree"):
        interval_scope(r, dict(protocol, seeds=[999]))


def test_rejects_forged_rank_or_duplicate_recordings() -> None:
    r = report([1.0] * 20)
    r.trials[1].artifact_sha256 = r.trials[0].artifact_sha256
    with pytest.raises(ValueError, match="duplicate recording"):
        fit(r)
    payload = fit(report([1.0] * 20)).model_dump()
    payload["intervals"][0]["order_rank"] = 1
    with pytest.raises(ValidationError, match="rank/count"):
        EmpiricalIntervalModel.model_validate(payload)


def test_zero_formal_std_does_not_destroy_error_intervals() -> None:
    r = report([2.0] * 20)
    for t in r.trials:
        t.axes[0].std = 0
    assert fit(r).intervals[0].half_width == 2.0


def test_unaligned_bag_gravity_is_excluded_without_score_selection() -> None:
    from calibrex.evaluation.empirical_intervals import eligible_interval_axes

    r = report([1.0] * 20)
    for trial in r.trials:
        trial.axes.append(CoverageAxisObservation(axis="gravity[0]", error=0.0, std=1.0))
    r.axes = summarize_coverage(r.trials, ["x", "gravity[0]"])
    assert eligible_interval_axes(r) == ["x"]
    assert [a.axis for a in fit(r).intervals] == ["x"]
    # Even perfect coverage cannot make unaligned gravity an admissible axis.
    r.producer = "known_planes"
    assert eligible_interval_axes(r) == ["x", "gravity[0]"]
    assert [a.axis for a in fit(r).intervals] == ["x", "gravity[0]"]
