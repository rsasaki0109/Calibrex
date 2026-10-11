"""Signed-axis sweeps keep every attempt and agree across CPU processes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from tests.unit.test_public_camera_lidar_demo import _fixture

from calibrex.cli.main import main
from calibrex.core.benchmark import BenchmarkDefinition, aggregate_benchmark_definition
from calibrex.core.provenance import sha256_path
from calibrex.core.validation import validate_file
from calibrex.evaluation import public_six_dof_sweep as sweep_module
from calibrex.evaluation.public_six_dof_sweep import (
    public_six_dof_axis_cases,
    run_public_six_dof_sweep,
)
from calibrex.pipelines import public_camera_lidar_demo as demo_module
from calibrex.pipelines.public_camera_lidar_demo import _numerical_payload
from calibrex.solvers.borer_six_dof_solver import BorerSixDofOptions


def test_cpu_processes_preserve_serial_traces_plan_split_and_result_hashes(tmp_path: Path) -> None:
    sequence, provider = _fixture(tmp_path)
    settings = BorerSixDofOptions(max_evaluations=13)
    serial = run_public_six_dof_sweep(
        sequence, provider, output_directory=tmp_path / "serial",
        candidate_backend="numpy", options=settings, command=["fixture"],
    )
    parallel = run_public_six_dof_sweep(
        sequence, provider, output_directory=tmp_path / "parallel",
        candidate_backend="numpy", workers=2, options=settings, command=["fixture"],
    )
    assert not serial.successful and not parallel.successful
    assert len(parallel.benchmark.trials) == 12
    cases = public_six_dof_axis_cases()
    assert list(parallel.demos) == [case.case_id for case in cases]
    assert len({split.fit_ids_sha256 for split in parallel.benchmark.protocol.splits}) == 1
    assert len({split.holdout_ids_sha256 for split in parallel.benchmark.protocol.splits}) == 1
    for case in cases:
        before, after = serial.demos[case.case_id], parallel.demos[case.case_id]
        assert before.numerical_parity and after.numerical_parity
        for key, result in before.trial_results.items():
            assert _numerical_payload(result) == _numerical_payload(after.trial_results[key])
        initial = after.results["numpy"].run.provenance["configuration"]["initial_transform"]
        assert initial["translation_m"] == list(case.translation_m_xyz)
    root = parallel.report_path.parent
    plan = BenchmarkDefinition.model_validate_json(
        (root / "sweep.plan.definition.json").read_bytes()
    )
    assert all(trial.failure_reason == "not executed; frozen plan entry" for trial in plan.trials)
    assert len(plan.trials) == 12 and plan.provenance.data_verified
    assert (
        parallel.benchmark.provenance.source_artifacts["sweep.plan.definition.json"]
        == sha256_path(root / "sweep.plan.definition.json")
    )
    for trial in parallel.benchmark.trials:
        case_id = trial.split_id.split("/")[0]
        assert trial.provenance.output_sha256 == sha256_path(
            root / case_id / f"{trial.method_id}.result.json"
        )
    assert all(validate_file(path).valid for path in root.rglob("*.json"))
    report = parallel.report_path.read_text(encoding="utf-8")
    assert "0 / 12" in report and "No cross-case speed ratio" in report
    assert all(f'{case.case_id}/report.html' in report for case in cases)


def test_plan_is_saved_before_any_optimization_and_drift_is_within_case_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    sequence, provider = _fixture(tmp_path)
    original = sweep_module.run_public_camera_lidar_six_dof_demo
    output = tmp_path / "out"
    seen = []

    def run(*args: Any, **kwargs: Any) -> Any:
        assert validate_file(output / "sweep.plan.definition.json").valid
        seen.append(Path(kwargs["output_directory"]).name)
        return original(*args, **kwargs)

    monkeypatch.setattr(sweep_module, "run_public_camera_lidar_six_dof_demo", run)
    sweep = run_public_six_dof_sweep(
        sequence, provider, output_directory=output, candidate_backend="numpy",
        rotation_deg=0.01, translation_m=0.01, repetitions=2,
        options=BorerSixDofOptions(
            max_evaluations=100, minimum_rotation_step_deg=0.25, minimum_translation_step_m=0.1,
        ),
        command=["fixture"],
    )
    assert seen == [case.case_id for case in public_six_dof_axis_cases()]
    assert sweep.successful and len(sweep.benchmark.trials) == 24
    # Different initializations should not be compared as if they were replays.
    assert (
        _numerical_payload(sweep.demos["x-positive"].results["numpy"])
        != _numerical_payload(sweep.demos["x-negative"].results["numpy"])
    )
    definition = BenchmarkDefinition.model_validate_json(
        (output / "sweep.definition.json").read_bytes()
    )
    assert aggregate_benchmark_definition(definition) == sweep.benchmark
    assert "24 / 24" in sweep.report_path.read_text(encoding="utf-8")
    parallel = run_public_six_dof_sweep(
        sequence, provider, output_directory=tmp_path / "parallel-success",
        candidate_backend="numpy", workers=2, rotation_deg=0.01, translation_m=0.01,
        repetitions=2,
        options=BorerSixDofOptions(
            max_evaluations=100, minimum_rotation_step_deg=0.25, minimum_translation_step_m=0.1,
        ),
        command=["fixture"],
    )
    assert parallel.successful
    for case_id, demo in sweep.demos.items():
        for key, result in demo.trial_results.items():
            assert _numerical_payload(result) == _numerical_payload(
                parallel.demos[case_id].trial_results[key]
            )


def test_runtime_failure_stays_in_suite_and_later_cases_still_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    sequence, provider = _fixture(tmp_path)
    original = demo_module.run_camera_lidar_six_dof_estimate
    calls = 0

    def run(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("candidate failed in first case")
        return original(*args, **kwargs)

    monkeypatch.setattr(demo_module, "run_camera_lidar_six_dof_estimate", run)
    sweep = run_public_six_dof_sweep(
        sequence, provider, output_directory=tmp_path / "out", candidate_backend="numpy",
        options=BorerSixDofOptions(max_evaluations=13), command=["fixture"],
    )
    assert len(sweep.demos) == 6 and calls == 12
    assert not sweep.successful
    failed = sweep.benchmark.trials[1]
    assert failed.failure_reason == "RuntimeError: candidate failed in first case"
    assert failed.provenance.output_sha256 is None
    assert not sweep.demos["x-positive"].numerical_parity
    assert all(demo.numerical_parity for key, demo in sweep.demos.items() if key != "x-positive")
    assert validate_file(sweep.report_path.parent / "sweep.benchmark.json").valid


def test_cli_partial_sweep_retains_all_cases_and_rejects_cuda_processes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    sequence, provider = _fixture(tmp_path)
    assert main([
        "demo", "kitti-six-dof-sweep", str(sequence), str(provider),
        "--candidate-backend", "numpy", "--max-evaluations", "13",
        "--output-dir", str(tmp_path / "out"), "--json",
    ]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "incomplete" and len(payload["cases"]) == 6
    assert all(case["numerical_parity"] for case in payload["cases"].values())
    assert main([
        "demo", "kitti-six-dof-sweep", "missing", "missing", "--workers", "2",
        "--output-dir", str(tmp_path / "missing"), "--json",
    ]) == 2
    assert "CUDA comparisons require" in capsys.readouterr().err
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("kwargs", [
    {"workers": 0}, {"workers": True}, {"workers": 1.5},
    {"repetitions": 0}, {"rotation_deg": float("nan")},
    {"translation_m": -0.1}, {"rotation_deg": 3}, {"translation_m": 2},
])
def test_invalid_sweep_settings_reject_before_inputs(
    tmp_path: Path, kwargs: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        run_public_six_dof_sweep(
            "missing", "missing", output_directory=tmp_path / "out",
            candidate_backend="numpy", command=["fixture"], **kwargs,
        )
    assert not (tmp_path / "out").exists()
