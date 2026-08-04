from __future__ import annotations

from pathlib import Path

import jsonschema

from calibrex.core.dynamic_window import (
    DynamicWindowConsistencyThresholds,
    dynamic_window_consistency_json_schema,
)
from calibrex.core.result import (
    CalibrationResult,
    FrameGraphSnapshot,
    RunInfo,
    TransformResult,
)
from calibrex.evaluation.dynamic_window import evaluate_dynamic_window_consistency


def _result(
    path: Path,
    label: str,
    translation: list[float],
    *,
    start_ns: int,
) -> CalibrationResult:
    result = CalibrationResult(
        run=RunInfo(
            id=label,
            slac_version="test",
            provenance={
                "rosbag2_capture_window_start_timestamp_ns": start_ns,
                "rosbag2_capture_window_end_timestamp_ns": start_ns + 60_000_000_000,
            },
        ),
        frame_graph=FrameGraphSnapshot(root="livox", frames={"livox": None, "rslidar": "livox"}),
        transforms={
            "T_livox_rslidar": TransformResult(
                parent="livox",
                child="rslidar",
                translation_m=translation,
                rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            )
        },
    )
    result.save(path)
    return result


def test_dynamic_window_consistency_fails_on_transform_drift(tmp_path: Path) -> None:
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"
    first = _result(first_path, "first", [0.0, 0.0, 0.0], start_ns=1_000_000_000)
    second = _result(second_path, "second", [0.2, 0.0, 0.0], start_ns=2_000_000_000)

    artifact = evaluate_dynamic_window_consistency(
        [("first", first), ("second", second)],
        paths={"first": first_path, "second": second_path},
        transform_id="T_livox_rslidar",
        thresholds=DynamicWindowConsistencyThresholds(
            max_translation_delta_m=0.1,
            max_rotation_delta_deg=5.0,
        ),
    )

    assert artifact.grade == "fail"
    assert artifact.pairs[0].grade == "fail"
    assert artifact.pairs[0].translation_delta_m == 0.2
    assert artifact.inputs[0].capture_window_start_timestamp_ns == 1_000_000_000
    assert artifact.provenance.source_sha256["first"]


def test_dynamic_window_consistency_schema_accepts_declared_gate(tmp_path: Path) -> None:
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"
    first = _result(first_path, "first", [0.0, 0.0, 0.0], start_ns=1_000_000_000)
    second = _result(second_path, "second", [0.01, 0.0, 0.0], start_ns=2_000_000_000)
    artifact = evaluate_dynamic_window_consistency(
        [("first", first), ("second", second)],
        paths={"first": first_path, "second": second_path},
        transform_id="T_livox_rslidar",
        thresholds=DynamicWindowConsistencyThresholds(
            max_translation_delta_m=0.1,
            max_rotation_delta_deg=5.0,
        ),
    )

    jsonschema.validate(artifact.model_dump(mode="json"), dynamic_window_consistency_json_schema())
    assert artifact.grade == "warn"
    assert artifact.protocol_compatibility_status == "not_comparable"
