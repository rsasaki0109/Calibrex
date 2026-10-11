import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from test_borer_rotation_benchmark import _write_problem
from test_public_camera_lidar_demo import _fixture

from calibrex.cli.main import main
from calibrex.core.benchmark import BenchmarkDefinition, aggregate_benchmark_definition
from calibrex.core.validation import validate_file
from calibrex.evaluation import borer_rotation_benchmark as loader
from calibrex.pipelines.camera_lidar_input_snapshot import CameraLidarSixDofInputSnapshot
from calibrex.pipelines.camera_lidar_six_dof import run_camera_lidar_six_dof_estimate
from calibrex.pipelines.public_camera_lidar_demo import (
    _numerical_payload,
    run_public_camera_lidar_six_dof_demo,
)
from calibrex.solvers.borer_six_dof_solver import BorerSixDofOptions


def _estimate(path, snapshot=None, **kwargs):
    return run_camera_lidar_six_dof_estimate(
        path, input_snapshot=snapshot, options=BorerSixDofOptions(max_evaluations=13),
        command=["fixture"], **kwargs,
    )


def test_snapshot_preserves_exact_values_and_has_no_writable_array_owner(tmp_path: Path) -> None:
    path = _write_problem(tmp_path)
    original = loader.load_borer_problem(path)
    snapshot = CameraLidarSixDofInputSnapshot(path)
    captured = snapshot._loaded_for_estimate(path)
    for before, after in zip(original.observations, captured.observations, strict=True):
        for field in ("depth_map", "lidar_points", "lidar_range_m"):
            left, right = getattr(before, field), getattr(after, field)
            np.testing.assert_array_equal(left, right)
            left.flat[0] = 99
            assert not np.shares_memory(left, right)
            with pytest.raises(ValueError):
                right.setflags(write=True)
            with pytest.raises(ValueError):
                right.flat[0] = 99
    captured.problem.dataset_id = "mutated"
    captured.depth_provider.observations[0].source_frame = "mutated"
    assert snapshot._loaded_for_estimate(path).problem.dataset_id == original.problem.dataset_id
    assert snapshot._loaded_for_estimate(path).depth_provider == original.depth_provider
    provenance = snapshot.provenance()
    assert provenance["array_bytes"] == sum(
        array.nbytes for item in captured.observations
        for array in (item.depth_map, item.lidar_points, item.lidar_range_m)
    )
    provenance["snapshot_sha256"] = "changed"
    assert snapshot.provenance()["snapshot_sha256"] != "changed"


def test_snapshot_survives_external_changes_but_fresh_verification_rejects_them(
    tmp_path: Path,
) -> None:
    path = _write_problem(tmp_path)
    baseline = _estimate(path)
    snapshot = CameraLidarSixDofInputSnapshot(path)
    image = tmp_path / "000000.png"
    stat = image.stat()
    raw = image.read_bytes()
    image.write_bytes(b"X" + raw[1:])
    os.utime(image, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    # Size/mtime are deliberately identical: a persistent stat-only cache
    # would silently trust these changed bytes. Explicit capture uses no cache.
    assert image.stat().st_size == stat.st_size
    with pytest.raises(ValueError, match="verification failed"):
        _estimate(path)
    reused = _estimate(path, snapshot)
    assert _numerical_payload(baseline) == _numerical_payload(reused)
    assert reused.run.dataset_sha256 == baseline.run.dataset_sha256
    assert reused.run.config_sha256 != baseline.run.config_sha256
    record = reused.run.provenance["input_verification"]
    assert record["external_files_reverified"] is False
    assert record["problem_artifact_reverified"] is True
    assert record["preparation_in_run_timing"] is False
    (tmp_path / "000000.depth.npy").unlink()
    (tmp_path / "000000.lidar.npy").unlink()
    (tmp_path / "depth_provider.yaml").unlink()
    assert _numerical_payload(_estimate(path, snapshot)) == _numerical_payload(baseline)
    saved = tmp_path / "reused.result.json"
    reused.save(saved)
    assert validate_file(saved).valid


def test_snapshot_rejects_wrong_path_and_changed_problem_bytes(tmp_path: Path) -> None:
    path = _write_problem(tmp_path)
    snapshot = CameraLidarSixDofInputSnapshot(path)
    alternate = tmp_path / "alternate.yaml"
    alternate.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="different problem path"):
        _estimate(alternate, snapshot)
    stat = path.stat()
    path.write_bytes(path.read_bytes().replace(b"synthetic-d2d", b"synthetic-new"))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert path.stat().st_size == stat.st_size
    with pytest.raises(ValueError, match="problem digest mismatch"):
        _estimate(path, snapshot)


def test_loader_binds_array_bytes_even_when_data_changes_after_provider_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_problem(tmp_path)
    original = loader.verify_depth_provider_files

    def verify(*args, **kwargs):
        issues = original(*args, **kwargs)
        assert not issues
        depth = tmp_path / "000000.depth.npy"
        data = np.load(depth, allow_pickle=False)
        data.flat[0] = 123
        np.save(depth, data, allow_pickle=False)
        return issues

    monkeypatch.setattr(loader, "verify_depth_provider_files", verify)
    with pytest.raises(ValueError, match="depth digest mismatch"):
        CameraLidarSixDofInputSnapshot(path)


def test_caller_owned_snapshot_supports_concurrent_cpu_estimates(tmp_path: Path) -> None:
    path = _write_problem(tmp_path)
    snapshot = CameraLidarSixDofInputSnapshot(path)
    baseline = _numerical_payload(_estimate(path))
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: _estimate(path, snapshot, frame_workers=2), range(2)))
    assert all(_numerical_payload(result) == baseline for result in results)


def test_public_demo_reuse_verifies_once_and_charges_all_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    sequence, provider = _fixture(tmp_path)
    original = loader.verify_depth_provider_files
    calls = []

    def verify(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(loader, "verify_depth_provider_files", verify)
    demo = run_public_camera_lidar_six_dof_demo(
        sequence, provider, output_directory=tmp_path / "out", candidate_backend="numpy",
        repetitions=2, reuse_verified_inputs=True, options=BorerSixDofOptions(max_evaluations=13),
        command=["fixture"],
    )
    assert len(calls) == 1
    assert demo.numerical_parity
    results = list(demo.trial_results.values())
    assert len(results) == 4
    cost = results[0].run.provenance["input_verification"]["preparation_seconds"]
    shares = [t.metrics["snapshot_preparation_share_seconds"] for t in demo.benchmark.trials]
    assert sum(shares) == cost
    for trial in demo.benchmark.trials:
        result = demo.trial_results[trial.method_id, trial.split_id]
        assert trial.metrics["pipeline_seconds"] == (
            sum(result.run.provenance["timing_seconds"].values()) + cost / 4
        )
    definition_path = demo.report_path.parent / "comparison.definition.json"
    assert validate_file(definition_path).valid
    definition = BenchmarkDefinition.model_validate_json(definition_path.read_bytes())
    assert aggregate_benchmark_definition(definition) == demo.benchmark


@pytest.mark.parametrize("subcommand", ["kitti-six-dof", "kitti-six-dof-sweep"])
def test_public_cli_reuse_and_worker_local_snapshots(
    tmp_path: Path, subcommand: str, capsys,
) -> None:
    sequence, provider = _fixture(tmp_path)
    output = tmp_path / "out"
    args = ["demo", subcommand, str(sequence), str(provider), "--output-dir", str(output),
            "--candidate-backend", "numpy", "--reuse-verified-inputs",
            "--max-evaluations", "13", "--json"]
    if subcommand.endswith("sweep"):
        args += ["--workers", "2"]
    assert main(args) == 2  # Retain the deliberately evaluation-limited attempts.
    assert '"numerical_parity": true' in capsys.readouterr().out or subcommand.endswith("sweep")
    files = list(output.rglob("*.result.json"))
    assert len(files) == (12 if subcommand.endswith("sweep") else 2)
    for file in files:
        record = json.loads(file.read_bytes())["run"]["provenance"]["input_verification"]
        assert record["mode"] == "verified_snapshot"
