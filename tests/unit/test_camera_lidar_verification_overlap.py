from __future__ import annotations

import hashlib
from pathlib import Path
from threading import Event

import pytest
from tests.unit.test_borer_rotation_benchmark import _write_problem

from calibrex.core.camera_lidar_artifacts import load_camera_lidar_problem
from calibrex.core.provenance import sha256_path
from calibrex.data.depth import depth_file_reference, load_depth_provider
from calibrex.evaluation import borer_rotation_benchmark as loader
from calibrex.pipelines import camera_lidar_six_dof as pipeline
from calibrex.pipelines.camera_lidar_input_snapshot import CameraLidarSixDofInputSnapshot
from calibrex.pipelines.public_camera_lidar_demo import _numerical_payload
from calibrex.solvers.borer_six_dof_solver import BorerSixDofOptions, BorerSixDofSolver


def _checkpoint_problem(tmp_path: Path, *, bad: bool = False) -> Path:
    path = _write_problem(tmp_path)
    problem = load_camera_lidar_problem(path)
    provider_path = Path(problem.depth_provider_path)
    provider = load_depth_provider(provider_path)
    checkpoint = tmp_path / "weights.bin"
    checkpoint.write_bytes(b"checkpoint fixture")
    provider.provider.checkpoint = depth_file_reference(checkpoint, encoding="fixture")
    if bad:
        provider.provider.checkpoint.sha256 = "0" * 64
    provider.save(provider_path)
    digest = sha256_path(provider_path)
    assert digest is not None
    problem.depth_provider_sha256 = digest
    problem.save(path)
    return path


def _estimate(path: Path, **kwargs):
    return pipeline.run_camera_lidar_six_dof_estimate(
        path, options=BorerSixDofOptions(max_evaluations=13), command=("fixture",), **kwargs,
    )


def _blocked_checkpoint(monkeypatch: pytest.MonkeyPatch, *, release_in_solver: bool = True):
    started, release, finished = Event(), Event(), Event()
    verify = loader._verify_depth_file_reference
    solve = BorerSixDofSolver.solve

    def blocking_verify(reference, root):
        started.set()
        assert release.wait(10), "checkpoint verification was not overlapped"
        try:
            return verify(reference, root)
        finally:
            finished.set()

    def calculate(self, *args, **kwargs):
        assert started.wait(10)
        if release_in_solver:
            release.set()
        return solve(self, *args, **kwargs)

    monkeypatch.setattr(loader, "_verify_depth_file_reference", blocking_verify)
    monkeypatch.setattr(BorerSixDofSolver, "solve", calculate)
    return release, finished


def test_checkpoint_check_overlaps_solve_and_preserves_full_result(tmp_path, monkeypatch):
    path = _checkpoint_problem(tmp_path)
    reference = _estimate(path, overlap_input_verification=False)
    _, finished = _blocked_checkpoint(monkeypatch)
    result = _estimate(path)
    assert finished.is_set()
    assert _numerical_payload(result) == _numerical_payload(reference)
    verification = result.run.provenance["input_verification"]
    assert verification["strategy"] == "checkpoint_overlap"
    assert verification["external_files_reverified"]
    assert verification["checkpoint_time_overlaps_phases"]
    assert result.run.provenance["data_verified"]
    assert verification["checkpoint_verification_seconds"] >= 0
    assert verification["completion_wait_seconds"] >= 0


@pytest.mark.parametrize("bad_kind", ["digest", "missing"])
def test_failed_checkpoint_never_constructs_a_result(tmp_path, monkeypatch, bad_kind):
    path = _checkpoint_problem(tmp_path, bad=bad_kind == "digest")
    if bad_kind == "missing":
        (tmp_path / "weights.bin").unlink()
    _, finished = _blocked_checkpoint(monkeypatch)
    constructed = []
    monkeypatch.setattr(pipeline, "CalibrationResult", lambda **kw: constructed.append(kw))
    with pytest.raises(ValueError, match="depth provider verification failed"):
        _estimate(path)
    assert finished.is_set()
    assert not constructed


def test_backend_failure_joins_verification_without_masking_error(tmp_path, monkeypatch):
    path = _checkpoint_problem(tmp_path)
    release, finished = _blocked_checkpoint(monkeypatch, release_in_solver=False)

    def fail(*args, **kwargs):
        release.set()
        raise RuntimeError("backend failed")

    monkeypatch.setattr(pipeline, "create_depth_pair_projector", fail)
    with pytest.raises(RuntimeError, match="backend failed"):
        _estimate(path)
    assert finished.is_set()


def test_worker_exception_reaches_caller_before_result(tmp_path, monkeypatch):
    path = _checkpoint_problem(tmp_path)

    def fail(*args):
        raise OSError("checkpoint unreadable")

    monkeypatch.setattr(loader, "_verify_depth_file_reference", fail)
    with pytest.raises(OSError, match="checkpoint unreadable"):
        _estimate(path)


def test_no_checkpoint_is_sequential_and_explicit_snapshot_is_unchanged(tmp_path):
    path = _write_problem(tmp_path)
    sequential = _estimate(path, overlap_input_verification=False)
    ordinary = _estimate(path)
    snapshot = _estimate(path, input_snapshot=CameraLidarSixDofInputSnapshot(path))
    assert _numerical_payload(ordinary) == _numerical_payload(sequential)
    assert _numerical_payload(snapshot) == _numerical_payload(sequential)
    assert ordinary.run.provenance["input_verification"]["strategy"] == "sequential"
    assert snapshot.run.provenance["input_verification"]["mode"] == "verified_snapshot"


def test_problem_digest_binds_the_exact_parsed_bytes(tmp_path, monkeypatch):
    path = _write_problem(tmp_path)
    original_bytes = path.read_bytes()
    read = Path.read_bytes

    def replace_after_read(self):
        raw = read(self)
        if self == path:
            self.write_bytes(raw.replace(b"synthetic-d2d", b"synthetic-new"))
        return raw

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    loaded = loader.load_borer_problem(path)
    assert loaded.problem.problem_id == "synthetic-d2d"
    assert loaded.problem_sha256 == hashlib.sha256(original_bytes).hexdigest()
