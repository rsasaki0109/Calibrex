from __future__ import annotations

import json
from pathlib import Path

import pytest
from tools.benchmark_continuous_camera_lidar_reuse import _replay_clock_work


def test_public_control_clock_replay_predicts_every_point_and_batch() -> None:
    archive = (
        Path(__file__).resolve().parents[2] / "docs/assets/continuous_camera_lidar_evaluation_reuse"
    )
    problem = json.loads((archive / "problem.json").read_bytes())
    result = json.loads((archive / "00-current/result.json").read_bytes())
    protocol = json.loads((archive / "protocol.json").read_bytes())
    assert _replay_clock_work(problem, result, protocol["options"]) == {
        "deskew_points": 556032,
        "deskew_batch_calls": 543,
        "projection_points": 2462720,
        "projection_batch_calls": 2405,
    }


def test_clock_replay_counts_eviction_signed_zero_and_disjoint_splits() -> None:
    problem = {
        "captures": [
            {
                "capture_id": name,
                "correspondences": [
                    {"reliability": 1.0, "outlier_probability": 0.0},
                    {"reliability": 0.1, "outlier_probability": 0.0},
                ],
            }
            for name in ("a", "b")
        ]
    }
    result = {
        "train_capture_ids": ["a"],
        "holdout_capture_ids": ["b"],
        "initial_time_offset_sec": 0.0,
        "estimated_time_offset_sec": 0.0,
        "trace": [{"time_offset_sec": t} for t in [0.0, 0.0, 0.0, -0.0, 0.0]],
    }
    options = {"minimum_confidence": 0.25, "time_sensitivity_probe_sec": 0.01}
    assert _replay_clock_work(problem, result, options) == {
        "deskew_points": 6,
        "deskew_batch_calls": 6,
        "projection_points": 9,
        "projection_batch_calls": 9,
    }
    result["trace"][3]["time_offset_sec"] = 0.0
    assert _replay_clock_work(problem, result, options)["deskew_points"] == 5
    problem["body_trajectory"] = {"poses": []}
    with pytest.raises(ValueError, match="constant-twist control"):
        _replay_clock_work(problem, result, options)
