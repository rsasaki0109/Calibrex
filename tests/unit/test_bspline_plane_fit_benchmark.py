"""Performance evidence must reject drift in every fitted block and decision."""

from __future__ import annotations

from copy import deepcopy

import pytest
from tools.benchmark_bspline_planes import EXACT_FIELDS, NUMERIC_FIELDS, _compare


def _result() -> dict:
    return {
        "status": "converged",
        "iterations": 22,
        "residual_count": 300,
        "parameter_count": 70,
        "calibration_parameter_names": ["gyro_bias[0]"],
        "initial_cost": 10.0,
        "final_cost": 2.0,
        "residual_variance": 0.01,
        "trajectory_rotations": [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
        "trajectory_positions": [[0.0, 0.0, 0.0]],
        "calibration_covariance": [[0.01]],
        "calibration": {
            key: [0.0] if key == "imu_clock_offset" else [0.0, 0.0, 0.0]
            for key in (
                "imu_rotation",
                "gyro_bias",
                "lever_arm",
                "imu_clock_offset",
                "accel_bias",
                "gravity",
            )
        },
    }


@pytest.mark.parametrize(
    "field",
    [
        *NUMERIC_FIELDS,
        "imu_rotation",
        "gyro_bias",
        "lever_arm",
        "imu_clock_offset",
        "accel_bias",
        "gravity",
    ],
)
def test_every_numeric_output_is_checked(field: str) -> None:
    prior = _result()
    current = deepcopy(prior)
    if field in NUMERIC_FIELDS:
        if isinstance(current[field], list):
            current[field] = []
        else:
            current[field] += 1e-3
    else:
        current["calibration"][field][0] += 1e-3
    assert not _compare(prior, current)["passed"]


@pytest.mark.parametrize("field", EXACT_FIELDS)
def test_every_decision_and_parameter_order_is_checked(field: str) -> None:
    prior = _result()
    current = deepcopy(prior)
    if field == "status":
        current[field] = "max_iterations"
    elif field == "calibration_parameter_names":
        current[field] = ["other[0]"]
    else:
        current[field] += 1
    comparison = _compare(prior, current)
    assert not comparison["passed"]
    assert not comparison["exact_decisions"]


def test_tiny_arithmetic_changes_and_empty_covariance_are_supported() -> None:
    prior = _result()
    current = deepcopy(prior)
    current["calibration"]["gyro_bias"][0] = 1e-12
    prior["calibration_covariance"] = current["calibration_covariance"] = []
    assert _compare(prior, current)["passed"]
