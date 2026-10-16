from __future__ import annotations

import copy

import pytest
from tools.benchmark_trajectory_hand_eye_reuse import replay_work, verify_git


def _event(offset: float, x: float = 0.0) -> dict:
    params = [0.0] * 8
    params[6], params[3] = offset, x
    return {
        "fit": 0,
        "motion_count": 3,
        "residual_count": 18,
        "base": [1.0 if i % 5 == 0 else 0.0 for i in range(16)],
        "params": params,
    }


def test_replay_distinguishes_residual_hits_signed_zeros_and_controls() -> None:
    history = {
        "fits": [{}],
        "normalized": [_event(t) for t in (0.0, 0.0, -0.0, 0.0)],
        "least_squares": [{}],
    }
    result = {"dofs": [{"known_bad_control": {}}, {"known_bad_control": None}]}
    assert replay_work(history, result, False) == {
        "solve_calls": 1,
        "normalized_requests": 4,
        "least_squares_calls": 1,
        "poses_at_calls": 16,
        "motion_sigmas_calls": 6,
    }
    assert replay_work(history, result, True) == {
        "solve_calls": 1,
        "normalized_requests": 4,
        "least_squares_calls": 1,
        "poses_at_calls": 12,
        "motion_sigmas_calls": 3,
    }


def test_replay_bounds_both_caches_and_separates_fits() -> None:
    result = {"dofs": []}
    history = {
        "fits": [{}],
        "normalized": [_event(i * 0.001) for i in range(17)],
        "least_squares": [{}],
    }
    history["normalized"].append(_event(0.0, 1.0))
    assert replay_work(history, result, True)["poses_at_calls"] == 40
    history["normalized"] = [_event(0.0, i * 0.001) for i in range(129)] + [_event(0.0)]
    assert replay_work(history, result, True)["poses_at_calls"] == 6
    history["fits"].append({})
    second = copy.deepcopy(history["normalized"][0])
    second["fit"] = 1
    history["normalized"].append(second)
    assert replay_work(history, result, True)["poses_at_calls"] == 10


@pytest.mark.parametrize("part", ["fit", "dimension", "finite"])
def test_replay_rejects_invalid_evaluation_events(part: str) -> None:
    event = _event(0.0)
    if part == "fit":
        event["fit"] = 1
    elif part == "dimension":
        event["residual_count"] = 17
    else:
        event["params"][2] = float("nan")
    with pytest.raises(ValueError):
        replay_work({"fits": [{}], "normalized": [event], "least_squares": []}, {"dofs": []}, True)


@pytest.mark.parametrize("short", ["abcdef12", "abcdef1234567890"])
def test_actual_abbreviated_git_identity_is_retained(short: str) -> None:
    verify_git("abcdef1234567890" + "0" * 24, short, "abcdef1234567890" + "0" * 24, short)


@pytest.mark.parametrize("part", ["full", "short", "recorded"])
def test_git_identity_rejects_mismatched_source_or_record(part: str) -> None:
    full, short, expected, recorded = "a" * 40, "a" * 8, "a" * 40, "a" * 8
    if part == "full":
        full = "b" * 40
    elif part == "short":
        short = "b" * 8
    else:
        recorded = None
    with pytest.raises(ValueError, match="actual Git identities"):
        verify_git(full, short, expected, recorded)
