from __future__ import annotations

import copy
import json

import numpy as np
import pytest
from jsonschema import validate
from tools.benchmark_gnss_lever_arm_reuse import (
    SCHEMA,
    _same_control_geometry,
    _same_generator_source,
    comparable,
    default_options,
    replay_work,
    verify_git,
)


def event(offset: float, x: float = 0.0) -> dict:
    return {
        "fit": 0,
        "window_ids": ["a", "b"],
        "pair_count": 3,
        "residual_count": 9,
        "lever": [x, 0.0, 0.0],
        "offset": offset,
        "normalize": True,
    }


def history(events: list[dict]) -> dict:
    return {"fits": [{}], "residuals": events, "least_squares": [{}], "excitation": []}


def test_replay_counts_exact_states_batched_svd_and_uncached_evaluation() -> None:
    value = history([event(offset) for offset in [0.0, 0.0, -0.0, 0.0]])
    outside = event(0.0)
    outside["fit"] = None
    outside["normalize"] = False
    value["residuals"].append(outside)
    value["excitation"] = [{"window_ids": ["a", "b"]}]
    assert replay_work(value, False) == {
        "solve_calls": 1,
        "residual_requests": 5,
        "least_squares_calls": 1,
        "lidar_poses_calls": 12,
        "svd_calls": 10,
        "profiled_windows": 10,
    }
    assert replay_work(value, True) == {
        "solve_calls": 1,
        "residual_requests": 5,
        "least_squares_calls": 1,
        "lidar_poses_calls": 8,
        "svd_calls": 4,
        "profiled_windows": 6,
    }


def test_replay_bounds_both_caches_and_keeps_jackknife_contexts_separate() -> None:
    value = history([event(i * 0.001) for i in range(65)] + [event(0.0, 1.0)])
    assert replay_work(value, True)["lidar_poses_calls"] == 132
    value = history([event(0.0, i * 0.001) for i in range(257)] + [event(0.0)])
    assert replay_work(value, True)["svd_calls"] == 258
    assert replay_work(value, True)["lidar_poses_calls"] == 2
    value["fits"].append({})
    second = event(0.0)
    second["fit"] = 1
    value["residuals"].append(second)
    assert replay_work(value, True)["lidar_poses_calls"] == 4


def test_combined_lookup_counts_every_epoch_and_each_actual_interpolation_call() -> None:
    value = history([event(0.0), event(0.0, 0.1), event(0.0)])
    for row in value["residuals"]:
        row["epoch_count"] = 19
    value["excitation"] = [{"window_ids": ["a", "b"], "epoch_count": 19}]
    prior = replay_work(value, False, combined_pose_interpolation=True)
    current = replay_work(value, True, combined_pose_interpolation=True)
    assert prior["interpolated_epochs"] == 76
    assert current["interpolated_epochs"] == 38
    assert prior["lidar_poses_calls"] == 8 and prior["pose_batch_calls"] == 0
    assert current["lidar_poses_calls"] == 2 and current["pose_batch_calls"] == 1


@pytest.mark.parametrize("part", ["fit", "boolean_fit", "dimension", "finite", "normalization"])
def test_replay_rejects_invalid_evaluation(part: str) -> None:
    row = event(0.0)
    if part == "fit":
        row["fit"] = 1
    elif part == "boolean_fit":
        row["fit"] = False
    elif part == "dimension":
        row["residual_count"] = 8
    elif part == "finite":
        row["offset"] = float("nan")
    else:
        row["normalize"] = False
    with pytest.raises(ValueError):
        replay_work(history([row]), True)


@pytest.mark.parametrize("part", ["full", "short", "recorded"])
def test_git_identity_checks_both_actual_forms(part: str) -> None:
    full, short, expected, recorded = "a" * 40, "a" * 8, "a" * 40, "a" * 8
    verify_git(full, short, expected, recorded)
    if part == "full":
        full = "b" * 40
    elif part == "short":
        short = "b" * 8
    else:
        recorded = None
    with pytest.raises(ValueError, match="actual Git identities"):
        verify_git(full, short, expected, recorded)


def test_comparison_preserves_input_and_protocol_after_source_identity_checks() -> None:
    left = {
        "record_type": "history",
        "residuals": [event(0.0)],
        "provenance": {
            "git_commit": "a" * 40,
            "git_commit_short": "a" * 8,
            "problem_sha256": "b" * 64,
            "protocol_sha256": "c" * 64,
        },
    }
    right = copy.deepcopy(left)
    right["provenance"].update(git_commit="d" * 40, git_commit_short="d" * 8)
    assert comparable(left) == comparable(right)
    right["provenance"]["problem_sha256"] = "e" * 64
    assert comparable(left) != comparable(right)
    assert left["provenance"]["git_commit_short"] == "a" * 8


def test_full_default_options_are_json_canonical_and_schema_valid() -> None:
    value = default_options()
    assert value["solver"]["pair_steps"] == [5, 10, 20]
    assert value["jackknife_groups"] == 8
    assert value["solver"]["irls_iterations"] == 4
    validate(value, json.loads(SCHEMA.read_bytes())["$defs"]["protocol"]["properties"]["options"])


@pytest.mark.parametrize(
    "current,pinned,expected",
    [
        (b"x = 1\n", b"x = 1\n", True),
        (b"x = 1\r\n", b"x = 1\n", True),
        (b"x = 1\n", b"x = 1\r\n", True),
        (b"x = 2\n", b"x = 1\r\n", False),
        (b"x = 1\r", b"x = 1\r\n", False),
    ],
)
def test_generator_check_allows_only_lf_crlf_encoding(
    current: bytes, pinned: bytes, expected: bool
) -> None:
    assert _same_generator_source(current, pinned) is expected


@pytest.mark.parametrize("part", ["roundoff", "position", "rotation", "timestamp", "window_id"])
def test_regenerated_geometry_bounds_only_platform_roundoff(part: str) -> None:
    original = {
        "times_s": [0.0],
        "options": {},
        "enu_m": [[0.0, 1.0, 2.0]],
        "windows": [
            {"window_id": "w0", "block": 0, "times_s": [0.0], "poses": [np.eye(4).tolist()]}
        ],
    }
    changed = copy.deepcopy(original)
    if part == "roundoff":
        changed["enu_m"][0][0] += 9.313225746154785e-10
        changed["windows"][0]["poses"][0][0][3] += 9.313225746154785e-10
    elif part == "position":
        changed["enu_m"][0][0] += 1e-6
    elif part == "rotation":
        changed["windows"][0]["poses"][0][0][0] += 1e-9
    elif part == "timestamp":
        changed["times_s"][0] += 1e-9
    else:
        changed["windows"][0]["window_id"] = "different"
    assert _same_control_geometry(original, changed) == (part == "roundoff")
