"""CPU frame concurrency preserves ordered evidence and executor lifetime."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier, Lock, get_ident
from threading import enumerate as active_threads

import numpy as np
import pytest
from tests.unit.test_borer_depth_to_depth_solver import _synthetic_observation
from tests.unit.test_public_camera_lidar_demo import _fixture

from calibrex.cli.main import main
from calibrex.core.geometry import SE3
from calibrex.core.result import load_result
from calibrex.core.validation import validate_file
from calibrex.data.kitti_camera_lidar_problem import build_kitti_raw_camera_lidar_problem
from calibrex.pipelines import camera_lidar_six_dof as estimate_module
from calibrex.pipelines.public_camera_lidar_demo import (
    _numerical_payload,
    run_public_camera_lidar_six_dof_demo,
)
from calibrex.solvers.borer_depth_to_depth_solver import (
    evaluate_depth_to_depth_mi,
    project_depth_pairs,
)
from calibrex.solvers.borer_six_dof_solver import BorerSixDofOptions
from calibrex.solvers.depth_to_depth_runtime import validate_frame_workers


def test_frames_execute_concurrently_and_reduce_in_input_order() -> None:
    source = _synthetic_observation()
    observations = [replace(source, frame_id=str(i), depth_map=source.depth_map * (i + 1))
                    for i in range(4)]
    observations.insert(1, replace(
        source, frame_id="skipped", depth_map=np.zeros_like(source.depth_map),
    ))
    serial = evaluate_depth_to_depth_mi(observations, SE3.identity())
    barrier = Barrier(2, timeout=10)
    thread_ids = set()
    lock = Lock()

    def project(observation, transform, *, use_z_buffer=True):
        if observation.frame_id != "skipped":
            with lock:
                thread_ids.add(get_ident())
            barrier.wait()
        return project_depth_pairs(observation, transform, use_z_buffer=use_z_buffer)

    with ThreadPoolExecutor(max_workers=2) as executor:
        parallel = evaluate_depth_to_depth_mi(
            observations, SE3.identity(), projector=project, frame_executor=executor,
        )
    assert len(thread_ids) == 2
    assert parallel == serial
    assert parallel.skipped_frame_ids == ("skipped",)
    assert [frame.frame_id for frame in parallel.frames] == ["0", "1", "2", "3"]


def test_paired_demo_preserves_full_trace_holdout_and_worker_provenance(tmp_path: Path) -> None:
    sequence, provider = _fixture(tmp_path)
    demo = run_public_camera_lidar_six_dof_demo(
        sequence, provider, output_directory=tmp_path / "parallel",
        candidate_backend="numpy", candidate_frame_workers=2,
        rotation_offset_deg=(0, 0, 0), translation_offset_m=(0, 0, 0),
        options=BorerSixDofOptions(
            max_evaluations=100, minimum_rotation_step_deg=0.25, minimum_translation_step_m=0.1,
        ), command=["fixture"],
    )
    assert demo.successful
    serial, parallel = demo.results["numpy"], demo.results["numpy_repeat"]
    assert _numerical_payload(serial) == _numerical_payload(parallel)
    assert serial.run.config_sha256 != parallel.run.config_sha256
    assert serial.run.provenance["configuration"]["effective_frame_workers"] == 1
    assert parallel.run.provenance["configuration"]["effective_frame_workers"] == 2
    assert parallel.metrics["final_d2d_mutual_information"].holdout is not None
    assert "2 frame threads" in demo.report_path.read_text(encoding="utf-8")
    assert all(validate_file(path).valid for path in demo.report_path.parent.glob("*.json"))
    assert not any(t.name.startswith("calibrex-d2d") for t in active_threads())


@pytest.mark.parametrize("backend,workers", [
    ("numpy", 0), ("numpy", -1), ("numpy", True), ("numpy", 1.5), ("cuda", 2),
])
def test_invalid_workers_rejected_before_data_loading(backend, workers) -> None:
    with pytest.raises(ValueError, match="frame_workers"):
        estimate_module.run_camera_lidar_six_dof_estimate(
            "missing.problem.json", projection_backend=backend, frame_workers=workers,
            command=["fixture"],
        )
    with pytest.raises(ValueError, match="frame_workers"):
        validate_frame_workers(backend, workers)


def test_worker_failure_propagates_and_pool_threads_are_joined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    sequence, provider = _fixture(tmp_path)
    problem = build_kitti_raw_camera_lidar_problem(
        sequence, provider, rotation_bound_deg=2, translation_bound_m=1, command=("fixture",),
    )
    path = tmp_path / "problem.json"
    problem.save(path)

    def project(*args, **kwargs):
        raise RuntimeError("frame projection failed")

    monkeypatch.setattr(
        estimate_module, "create_depth_pair_projector", lambda *a, **k: (project, "fixture"),
    )
    with pytest.raises(RuntimeError, match="frame projection failed"):
        estimate_module.run_camera_lidar_six_dof_estimate(
            path, frame_workers=2, command=["fixture"],
        )
    assert not any(t.name.startswith("calibrex-d2d") for t in active_threads())


def test_public_cli_cpu_threads_and_cuda_rejection(tmp_path: Path, capsys) -> None:
    sequence, provider = _fixture(tmp_path)
    assert main([
        "demo", "kitti-six-dof", str(sequence), str(provider),
        "--output-dir", str(tmp_path / "parallel"), "--candidate-backend", "numpy",
        "--candidate-frame-workers", "2", "--max-evaluations", "13", "--json",
    ]) == 2
    assert '"numerical_parity": true' in capsys.readouterr().out
    assert main([
        "camera-lidar", "estimate-six-dof", "missing.problem.json",
        "--output", str(tmp_path / "never.json"), "--projection-backend", "cuda",
        "--frame-workers", "2",
    ]) == 2
    assert not (tmp_path / "never.json").exists()


def test_estimate_cli_caps_requested_threads_to_available_frames(tmp_path: Path, capsys) -> None:
    sequence, provider = _fixture(tmp_path, count=1)
    problem = build_kitti_raw_camera_lidar_problem(
        sequence, provider, rotation_bound_deg=2, translation_bound_m=1, command=("fixture",),
    )
    path, output = tmp_path / "problem.json", tmp_path / "result.json"
    problem.save(path)
    assert main([
        "camera-lidar", "estimate-six-dof", str(path), "--output", str(output),
        "--frame-workers", "20", "--minimum-rotation-step-deg", "0.25",
        "--minimum-translation-step-m", "0.1", "--json",
    ]) == 0
    configuration = load_result(output).run.provenance["configuration"]
    assert configuration["frame_workers"] == 20
    assert configuration["effective_frame_workers"] == 1
    assert '"effective_frame_workers": 1' in capsys.readouterr().out
    assert validate_file(output).valid
