from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from tools.benchmark_lidar_wheel_preparation import (
    _same_control_geometry,
    _same_generator_source,
    default_options,
    replay_work,
    verify_git,
)


def history(offsets: list[float]) -> dict:
    return {
        "fits": [
            {
                "fit": 0,
                "input_indices": [0, 1, 2],
                "selected_indices": [0, 2],
                "scan_candidates": 301,
            }
        ],
        "least_squares": [{}],
        "residuals": [
            {
                "fit": 0,
                "input_count": 2,
                "selected_count": 2,
                "params": [0.0, 0.0, 0.0, 0.0, 1.0, offset],
            }
            for offset in offsets
        ]
        + [
            {
                "fit": None,
                "input_count": 4,
                "selected_count": 3,
                "params": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            }
        ],
    }


def test_work_replay_counts_every_selection_scan_and_interpolated_sample() -> None:
    trace = history([0.0, 0.0, 0.01, 0.01, 0.0])
    common = {"solve_calls": 1, "residual_requests": 6, "least_squares_calls": 1}
    assert replay_work(trace, False) == {
        **common,
        "moving_calls": 7,
        "motion_selection_visits": 17,
        "wheel_at_calls": 307,
        "wheel_samples_queried": 615,
    }
    assert replay_work(trace, True) == {
        **common,
        "moving_calls": 2,
        "motion_selection_visits": 7,
        "wheel_at_calls": 304,
        "wheel_samples_queried": 609,
    }


def test_clock_replay_uses_signed_zero_exact_steps_and_eight_state_eviction() -> None:
    offsets = [0.0, -0.0, 0.0, -0.0, *[i * 1e-3 for i in range(1, 9)], 0.0]
    assert replay_work(history(offsets), True)["wheel_at_calls"] == 301 + 11 + 1


def test_clock_replay_does_not_share_states_between_fits() -> None:
    trace = history([0.0, 0.0])
    trace["fits"].append({**trace["fits"][0], "fit": 1})
    trace["least_squares"].append({})
    trace["residuals"].insert(2, {**trace["residuals"][0], "fit": 1})
    assert replay_work(trace, True)["wheel_at_calls"] == 2 * 301 + 2 + 1


@pytest.mark.parametrize(
    "current,pinned,expected",
    [
        (b"x=1\n", b"x=1\n", True),
        (b"x=1\r\n", b"x=1\n", True),
        (b"x=1\n", b"x=1\r\n", True),
        (b"x=2\n", b"x=1\r\n", False),
        (b"x=1\r", b"x=1\n", False),
    ],
)
def test_generator_normalization_permits_only_newline_encoding(
    current: bytes, pinned: bytes, expected: bool
) -> None:
    assert _same_generator_source(current, pinned) is expected


@pytest.mark.parametrize(
    "field,delta,expected",
    [
        ("velocity_mps", 1e-8, True),
        ("velocity_mps", 2e-6, False),
        ("angular_rate_rps", 1e-13, True),
        ("angular_rate_rps", 2e-11, False),
        ("start_s", 1e-8, False),
        ("block", 1, False),
    ],
)
def test_only_offline_motion_regeneration_accepts_bounded_platform_roundoff(
    field: str, delta: float, expected: bool
) -> None:
    problem = {
        "times_s": [1.0, 2.0],
        "speed_mps": [3.0, 4.0],
        "yaw_rate_rps": [0.0, 0.0],
        "options": {},
        "motions": [
            {
                "block": 0,
                "start_s": 1.0,
                "end_s": 2.0,
                "velocity_mps": [3.0, 0.0, 0.0],
                "angular_rate_rps": [0.0, 0.0, 0.01],
            }
        ],
    }
    changed = copy.deepcopy(problem)
    value = changed["motions"][0][field]
    if isinstance(value, list):
        value[0] += delta
    else:
        changed["motions"][0][field] += delta
    assert _same_control_geometry(problem, changed) is expected


@pytest.mark.parametrize("field", ["times_s", "speed_mps", "yaw_rate_rps"])
def test_regeneration_requires_exact_original_proxy_samples(field: str) -> None:
    problem = {"motions": [], "times_s": [1.0], "speed_mps": [3.0], "yaw_rate_rps": [0.01]}
    changed = copy.deepcopy(problem)
    changed[field][0] += 1e-12
    assert not _same_control_geometry(problem, changed)


def test_actual_full_and_short_git_identities_are_independently_bound() -> None:
    full = "a" * 40
    verify_git(full, "a" * 8, full, "a" * 8)
    with pytest.raises(ValueError, match="actual Git identities"):
        verify_git(full, "b" * 8, full, "b" * 8)
    with pytest.raises(ValueError, match="actual Git identities"):
        verify_git(full, "a" * 8, full, full)


def test_benchmark_schema_is_valid_and_rejects_boolean_work_counts() -> None:
    import jsonschema

    root = Path(__file__).resolve().parents[2]
    schema = json.loads(
        (root / "tools/schemas/lidar_wheel_preparation_benchmark.schema.json").read_bytes()
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["$defs"]["protocol"]["properties"]["options"] == {"const": default_options()}
    counts = schema["$defs"]["measurement"]["properties"]["counts"]
    value = dict.fromkeys(counts["required"], 0)
    jsonschema.validate(value, counts)
    value["wheel_at_calls"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, counts)


def test_generated_control_removes_common_epoch_and_exercises_clock_differences() -> None:
    import numpy as np
    from tools.benchmark_lidar_wheel_preparation import inputs
    from tools.prepare_lidar_wheel_control import derive_control

    from calibrex.solvers.lidar_wheel_solver import lidar_wheel_residuals

    times = 1e9 + np.arange(12) * 0.1
    relative = times - times[0]
    poses = np.repeat(np.eye(4)[None], len(times), axis=0)
    poses[:, 0, 3] = 8 * relative
    values = np.zeros((len(times), 30))
    values[:, 8] = 8 + 0.1 * relative
    values[:, 22] = 0.01 * relative
    problem = derive_control(times, poses, values)
    assert problem["times_s"] == relative.tolist()
    assert len(problem["motions"]) == len(times) - 2
    motions, wheel = inputs(problem)
    params = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    before = lidar_wheel_residuals(motions, wheel, params, np.eye(3))
    params[5] = np.sqrt(np.finfo(float).eps)
    assert params[5] < np.spacing(times[0])
    after = lidar_wheel_residuals(motions, wheel, params, np.eye(3))
    assert before.tobytes() != after.tobytes()
