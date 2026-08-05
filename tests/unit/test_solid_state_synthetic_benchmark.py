from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import jsonschema

from calibrex.core.io import read_mapping
from calibrex.core.solid_state_synthetic_benchmark import (
    SolidStateSyntheticBenchmarkArtifact,
)

_TOOL_SPEC = importlib.util.spec_from_file_location(
    "solid_state_synthetic_benchmark_tool",
    Path("tools/run_solid_state_synthetic_benchmark.py").resolve(),
)
assert _TOOL_SPEC is not None and _TOOL_SPEC.loader is not None
_TOOL_MODULE = importlib.util.module_from_spec(_TOOL_SPEC)
sys.modules[_TOOL_SPEC.name] = _TOOL_MODULE
_TOOL_SPEC.loader.exec_module(_TOOL_MODULE)

build_solid_state_synthetic_benchmark = _TOOL_MODULE.build_solid_state_synthetic_benchmark
render_solid_state_synthetic_benchmark_markdown = (
    _TOOL_MODULE.render_solid_state_synthetic_benchmark_markdown
)


def test_synthetic_benchmark_recovers_truth_and_detects_fixed_clock() -> None:
    artifact = build_solid_state_synthetic_benchmark(command=["fixture"])

    assert artifact.aggregate.conclusion == "pass"
    assert artifact.aggregate.all_expected_outcomes_detected is True
    reference = next(
        case for case in artifact.cases if case.case_type == "reference"
    )
    known_bad = next(
        case for case in artifact.cases if case.case_type == "known_bad"
    )
    assert reference.gate_passed is True
    assert reference.final_rotation_error_deg <= 0.5
    assert reference.final_translation_error_m <= 0.02
    assert reference.final_time_offset_error_sec <= 0.005
    assert known_bad.gate_passed is False
    assert known_bad.expected_outcome_detected is True
    assert "final_time_offset_error_exceeds_threshold" in known_bad.failure_reasons


def test_synthetic_benchmark_schema_and_report_are_stable() -> None:
    artifact = build_solid_state_synthetic_benchmark(command=["fixture"])
    schema = json.loads(
        Path("schemas/solid_state_synthetic_benchmark.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.validate(artifact.model_dump(mode="json", exclude_none=True), schema)

    markdown = render_solid_state_synthetic_benchmark_markdown(artifact)
    assert "joint_extrinsic_clock_reference" in markdown
    assert "fixed_clock_known_bad_control" in markdown
    assert "not a public-dataset accuracy claim" in markdown


def test_checked_synthetic_benchmark_asset_is_schema_valid() -> None:
    artifact = SolidStateSyntheticBenchmarkArtifact.model_validate(
        read_mapping(Path("docs/assets/solid-state-synthetic-benchmark-v01.yaml"))
    )
    assert artifact.aggregate.all_expected_outcomes_detected is True
