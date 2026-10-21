"""Worker partitions preserve complete-rig inference and reject incomplete inputs."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from calibrex.check import estimators
from calibrex.check.browser import run_request_json
from calibrex.check.browser_parallel import (
    _ClosureInput,
    closure_input_from_result,
    merge_check_shards_json,
)
from calibrex.check.estimators import EstimatorRun, PairContext
from calibrex.check.verdict import AxisEstimate
from calibrex.core.calibration_check import (
    CalibrationCheckArtifact,
    CheckPairRecord,
    CheckTransform,
)

SAMPLE = Path(__file__).resolve().parents[2] / "docs/app/samples/check_sample"
GROUPS = [
    ["imu-lidar", "gnss-lidar", "gnss-imu"],
    ["lidar-lidar", "lidar-vehicle", "imu-vehicle"],
]


def _run(ctx: PairContext) -> EstimatorRun:
    # A dependent adapter must see the same per-run memo as it sees serially.
    if ctx.pair.pair == "gnss-imu":
        ran = {key[1] for key in ctx.controls.memo if key[0] == estimators.PAIR_RUN_KEY}
        assert {"imu-lidar", "gnss-lidar"} <= ran
    error = 0.07 * (1 + sum(ctx.pair.pair.encode("ascii")) % 7)
    names = ("roll", "pitch", "yaw", "x", "y", "z")
    return EstimatorRun(
        estimator="deterministic worker test",
        artifacts=(),
        solved=True,
        policy_status="pass",
        policy_reasons=(),
        compared=ctx.candidate,
        estimates=tuple(
            AxisEstimate(
                name=name,
                unit="deg" if index < 3 else "m",  # type: ignore[arg-type]
                candidate_error=error if index < 3 else error / 100,
                std=0.01 if index < 3 else 0.001,
                estimated=True,
            )
            for index, name in enumerate(names)
        ),
    )


@pytest.fixture
def completed(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    monkeypatch.setattr(estimators, "ESTIMATORS", dict.fromkeys(estimators.ESTIMATORS, _run))
    request = {
        "mode": "check",
        "bag_dir": str(SAMPLE),
        "bag_label": "check_sample",
        "vehicle_frame": "base_link",
        "tf_labels": [],
        "pairs": [pair for group in GROUPS for pair in group],
        "max_duration_s": 5,
    }
    shards = [
        json.loads(
            run_request_json(
                json.dumps(
                    {
                        **request,
                        "pairs": pairs,
                        "parallel_shard": True,
                        "scan_memory_mb": 256,
                    }
                )
            )
        )
        for pairs in GROUPS
    ]
    assert all(shard["ok"] for shard in shards), shards
    return request, shards


def _semantic(artifact: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(artifact)
    value["provenance"].pop("created_at")
    value["provenance"]["notes"] = [
        note
        for note in value["provenance"]["notes"]
        if "browser worker(s) from 2 partitions" not in note
    ]
    for pair in value["pairs"]:
        pair.pop("runtime_s", None)
    return value


@pytest.mark.parametrize("reverse", [False, True])
def test_shards_equal_serial_including_closure_and_attribution(
    completed: tuple[dict[str, Any], list[dict[str, Any]]], reverse: bool
) -> None:
    request, shards = completed
    serial = json.loads(run_request_json(json.dumps(request)))
    if reverse:
        shards.reverse()
    reply = json.loads(merge_check_shards_json(json.dumps(request), json.dumps(shards)))
    assert reply["ok"], reply
    artifact = CalibrationCheckArtifact.model_validate(reply["artifact"])
    assert artifact.closures is not None and artifact.closures.loops
    assert artifact.rotation_attribution is not None
    assert _semantic(reply["artifact"]) == _semantic(serial["artifact"])
    assert reply["command"] == serial["command"]
    assert "calibrex check (run)" in reply["text"] and "<html" in reply["html"].lower()


@pytest.mark.parametrize(
    "change,match",
    [
        ("request", "different request"),
        ("source", "different request or source"),
        ("bag", "different inputs"),
        ("options", "different inputs"),
        ("overlap", "overlap"),
        ("missing_pair", "cover all"),
        ("missing_edge", "missing closure"),
        ("duplicate_edge", "duplicate closure"),
        ("failed", "worker failed"),
        ("plan", "completed check"),
        ("duplicate_selection", "duplicate selected"),
        ("one_shard", "two shards"),
    ],
)
def test_merger_rejects_incomplete_or_mixed_runs(
    completed: tuple[dict[str, Any], list[dict[str, Any]]], change: str, match: str
) -> None:
    request, shards = completed
    if change == "request":
        request["camera"] = "changed"
    elif change == "source":
        shards[1]["source_fingerprint"] = "other source"
    elif change == "bag":
        shards[1]["artifact"]["bag"]["input_sha256"] = "0" * 64
    elif change == "options":
        shards[1]["artifact"]["options"]["max_duration_s"] = 6
    elif change == "overlap":
        shards[1]["artifact"]["options"]["pairs"].append("imu-lidar")
    elif change == "missing_pair":
        shards[0]["artifact"]["options"]["pairs"].pop()
    elif change == "missing_edge":
        shards[0]["closure_inputs"].clear()
    elif change == "duplicate_edge":
        shards[0]["closure_inputs"].append(shards[0]["closure_inputs"][0])
    elif change == "failed":
        shards[1]["ok"] = False
    elif change == "plan":
        shards[1]["artifact"]["plan_only"] = True
    elif change == "duplicate_selection":
        request["pairs"].append(request["pairs"][0])
    elif change == "one_shard":
        shards.pop()
    reply = json.loads(merge_check_shards_json(json.dumps(request), json.dumps(shards)))
    assert not reply["ok"] and match in reply["error"], reply


def test_merger_rejects_split_dependencies(
    completed: tuple[dict[str, Any], list[dict[str, Any]]],
) -> None:
    request, shards = completed
    shards[0]["artifact"]["options"]["pairs"].remove("gnss-imu")
    shards[1]["artifact"]["options"]["pairs"].append("gnss-imu")
    reply = json.loads(merge_check_shards_json(json.dumps(request), json.dumps(shards)))
    assert not reply["ok"] and "dependent pairs" in reply["error"]


@pytest.mark.parametrize("workers_used", [1, 2])
def test_provenance_records_actual_workers_after_startup_fallback(
    completed: tuple[dict[str, Any], list[dict[str, Any]]], workers_used: int
) -> None:
    request, shards = completed
    request["workers_used"] = workers_used
    reply = json.loads(merge_check_shards_json(json.dumps(request), json.dumps(shards)))
    assert reply["ok"], reply
    assert any(
        f"{workers_used} browser worker(s) from 2 partitions" in note
        for note in reply["artifact"]["provenance"]["notes"]
    )


def test_transport_preserves_original_matrices_and_infinite_unobserved_std() -> None:
    matrix = np.eye(4)
    matrix[0, 3] = np.nextafter(0.1, 1.0)
    record = CheckPairRecord(
        pair="imu-lidar",
        frames=["imu", "lidar"],
        status="inconclusive",
        compared_transform=CheckTransform(
            parent_frame="lidar",
            child_frame="imu",
            translation_m=[0.1, 0, 0],
            rotation_quat_xyzw=[0, 0, 0, 1],
        ),
    )
    outcome = EstimatorRun(
        estimator="test",
        artifacts=(),
        solved=True,
        policy_status="pass",
        policy_reasons=(),
        compared=matrix,
        estimates=(
            AxisEstimate(
                name="x",
                unit="m",
                candidate_error=0.0,
                std=float("inf"),
                estimated=False,
            ),
        ),
    )
    encoded = closure_input_from_result(record, outcome)
    assert encoded is not None
    json.loads(json.dumps(encoded), parse_constant=lambda name: pytest.fail(name))
    edge = _ClosureInput.model_validate(encoded).edge()
    assert edge.translation is not None and edge.translation[0] == matrix[0, 3]
    assert edge.trans_std_m[0] == float("inf") and not edge.trans_observed[0]
    np.testing.assert_array_equal(edge.rotation, np.eye(3))


def test_real_estimators_keep_sample_results() -> None:
    request = {
        "mode": "check",
        "bag_dir": str(SAMPLE),
        "bag_label": "check_sample",
        "pairs": ["lidar-lidar", "gnss-lidar"],
        "max_duration_s": 5,
    }
    serial = json.loads(run_request_json(json.dumps(request)))
    shards = [
        json.loads(
            run_request_json(
                json.dumps(
                    {
                        **request,
                        "pairs": [pair],
                        "parallel_shard": True,
                        "scan_memory_mb": 256,
                    }
                )
            )
        )
        for pair in request["pairs"]
    ]
    reply = json.loads(merge_check_shards_json(json.dumps(request), json.dumps(shards)))
    assert reply["ok"], reply
    # The native evidence includes its own timestamp/digest. Compare the actual
    # estimates, verdicts, all rig-level inference and every input budget.
    for artifact in (serial["artifact"], reply["artifact"]):
        for pair in artifact["pairs"]:
            for evidence in pair.get("evidence", []):
                evidence.pop("sha256", None)
    assert _semantic(reply["artifact"]) == _semantic(serial["artifact"])
