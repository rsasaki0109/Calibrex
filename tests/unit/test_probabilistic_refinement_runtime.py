"""Input ownership, binding and optional-runtime contracts run without CUDA."""

from __future__ import annotations

import builtins
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError
from tests.unit.test_probabilistic_camera_lidar_refiner import _artifact, _frames, _problem

from calibrex.core.geometry import SE3
from calibrex.core.probabilistic_correspondence import ProbabilisticRefinementExecution
from calibrex.evaluation import probabilistic_camera_lidar_run as run
from calibrex.evaluation import probabilistic_refinement_ablation as ablation
from calibrex.solvers.probabilistic_camera_lidar_refiner import (
    ProbabilisticCameraLidarRefinementOptions,
    ProbabilisticCameraLidarRefiner,
)


def inputs(tmp_path: Path) -> tuple[Path, Path]:
    frames = _frames(SE3.identity())
    correspondence, problem = tmp_path / "correspondence.json", tmp_path / "problem.json"
    _artifact(frames).save(correspondence)
    _problem(frames).save(problem)
    return correspondence, problem


def test_ablation_validates_each_input_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = inputs(tmp_path)
    counts = {"correspondence": 0, "problem": 0}
    for role, name in (
        ("correspondence", "load_probabilistic_correspondence"),
        ("problem", "load_camera_lidar_problem"),
    ):
        native = getattr(run, name)

        def loader(path, *, role=role, native=native):
            counts[role] += 1
            return native(path)

        monkeypatch.setattr(run, name, loader)
    definition, _ = ablation.run_probabilistic_refinement_ablation(
        *paths,
        result_directory=tmp_path / "results",
        command="pytest owned inputs",
        split_seeds=(0, 1),
        bootstrap_samples=10,
        base_options=ProbabilisticCameraLidarRefinementOptions(max_evaluations=13),
    )
    assert counts == {"correspondence": 1, "problem": 1}
    assert len(definition.trials) == 8
    assert len(list((tmp_path / "results").glob("*.yaml"))) == 8


def test_changed_input_is_rejected_before_next_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = inputs(tmp_path)
    native = ablation._run_refinement_inputs

    def mutate(*args, **kwargs):
        result = native(*args, **kwargs)
        paths[0].write_bytes(paths[0].read_bytes() + b"\n")
        return result

    monkeypatch.setattr(ablation, "_run_refinement_inputs", mutate)
    with pytest.raises(ValueError, match="inputs changed"):
        ablation.run_probabilistic_refinement_ablation(
            *paths,
            result_directory=tmp_path / "results",
            command="pytest changed input",
            split_seeds=(0,),
            bootstrap_samples=10,
            base_options=ProbabilisticCameraLidarRefinementOptions(max_evaluations=13),
        )
    assert len(list((tmp_path / "results").glob("*.yaml"))) == 1


def test_load_time_change_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = inputs(tmp_path)
    native = run.load_probabilistic_correspondence

    def mutate(path):
        result = native(path)
        paths[0].write_bytes(paths[0].read_bytes() + b"\n")
        return result

    monkeypatch.setattr(run, "load_probabilistic_correspondence", mutate)
    with pytest.raises(ValueError, match="inputs changed"):
        run.run_probabilistic_camera_lidar_refinement(*paths)


def test_default_refinement_does_not_import_cupy() -> None:
    code = (
        "import sys; from calibrex.core.geometry import SE3; "
        "from calibrex.solvers.probabilistic_camera_lidar_refiner import "
        "ProbabilisticCameraLidarRefiner as R; R().solve([],SE3.identity()); "
        "assert not any(n=='cupy' or n.startswith('cupy.') for n in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_explicit_missing_cuda_fails_without_silent_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    native = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "calibrex.solvers.cuda_probabilistic_camera_lidar_adapter":
            raise ImportError("deliberately unavailable optional accelerator")
        return native(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(ValueError, match=r"calibrex\[cuda\]"):
        ProbabilisticCameraLidarRefiner(evaluation_backend="cuda").solve([], SE3.identity())
    assert ProbabilisticCameraLidarRefiner().solve([], SE3.identity()).execution is None


def test_execution_lineage_rejects_impossible_counts() -> None:
    with pytest.raises(ValidationError, match="counters"):
        ProbabilisticRefinementExecution(
            backend="cuda",
            runtime="test",
            kernel_sha256="a" * 64,
            calls=1,
            cuda_calls=2,
            numpy_fallback_calls=0,
            upload_bytes=0,
            download_bytes=0,
        )
