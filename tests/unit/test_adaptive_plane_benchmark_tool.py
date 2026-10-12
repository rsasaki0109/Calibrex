"""Protect complete-artifact comparisons and provenance guards."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest
from tools import benchmark_adaptive_plane_search as benchmark

from calibrex.core.geometry import SE3
from calibrex.data.odometry_track import OdometryTrack
from calibrex.solvers.continuous_time_lidar_pair_solver import (
    ContinuousTimeLidarPairOptions,
    ContinuousTimeLidarPairProblem,
)


def test_canonical_preserves_every_numerical_value_status_and_lineage():
    payload = {
        "status": "rejected",
        "estimated_time_offset_sec": -0.0,
        "iterations": [{"accepted": False, "candidate_train_rmse_m": [None, 0.001]}],
        "provenance": {
            "git_commit": "measured",
            "source_sha256": {"data\\recording": "verified"},
        },
        "dataset_path": "data\\recording",
    }
    original = deepcopy(payload)
    canonical = benchmark._canonical(payload)
    normalized = json.loads(canonical)
    assert payload == original
    assert normalized["status"] == "rejected"
    assert normalized["iterations"] == original["iterations"]
    assert normalized["provenance"]["git_commit"] == "measured"
    assert normalized["provenance"]["source_sha256"] == {"data/recording": "verified"}
    assert normalized["dataset_path"] == "data/recording"
    assert '"estimated_time_offset_sec":-0.0' in canonical
    payload["iterations"][0]["accepted"] = True
    assert benchmark._canonical(payload) != canonical


def test_comparison_excludes_only_exact_execution_identity_and_preserves_full_artifact():
    baseline = {"status": "converged", "provenance": {"git_commit": "bound", "notes": [
        "continuous_time_query_execution: counterfeit", "other lineage",
    ]}, "estimated_time_offset_sec": -0.0}
    candidate = deepcopy(baseline)
    identity = (
        "continuous_time_query_execution: calibrex.numba_continuous_time/v0.1;"
        "numba=0.65.1;fastmath=false;cache=true"
    )
    candidate["provenance"]["notes"].append(identity)
    full = benchmark._canonical(candidate)
    assert benchmark._comparison_canonical(candidate) == benchmark._canonical(baseline)
    assert benchmark._canonical(candidate) == full
    assert full != benchmark._canonical(baseline)
    other_version = deepcopy(baseline)
    other_version["provenance"]["notes"].append(identity.replace("v0.1", "v0.2"))
    assert benchmark._comparison_canonical(other_version) == benchmark._canonical(baseline)
    packed_version = deepcopy(baseline)
    packed_version["provenance"]["notes"].append(identity.replace("v0.1", "v0.3"))
    assert benchmark._comparison_canonical(packed_version) == benchmark._canonical(baseline)
    for forged in (
        identity + ";hidden=change", identity.replace("fastmath=false", "fastmath=true"),
        identity.replace("v0.1", "v0.4"),
    ):
        changed = deepcopy(baseline)
        changed["provenance"]["notes"].append(forged)
        assert benchmark._comparison_canonical(changed) != benchmark._comparison_canonical(baseline)
    for key, value in (("status", "rejected"), ("estimated_time_offset_sec", 1e-15)):
        changed = deepcopy(candidate)
        changed[key] = value
        assert benchmark._comparison_canonical(changed) != benchmark._comparison_canonical(baseline)


def test_compiled_default_config_preserves_capture_and_sampling():
    from calibrex.core.config import load_config

    root = "examples/public_datasets/glim_versatile/"
    baseline = load_config(root + "online_continuous_time_default_budget_config.yaml")
    compiled = load_config(root + "online_continuous_time_compiled_queries_config.yaml")
    left, right = [config.model_dump(mode="json") for config in (baseline, compiled)]
    for payload in (left, right):
        payload.pop("project")
    opts = right["pipeline"]["factors"]["lidar_rig_point_to_plane"]["options"]
    assert opts.pop("continuous_time_query_backend") == "numba_cpu"
    assert left == right


@pytest.mark.parametrize("option", ["--repetitions", "--omp-threads"])
def test_invalid_budget_is_rejected_before_reading_inputs_or_creating_outputs(tmp_path, option):
    destination = tmp_path / "output"
    with pytest.raises(ValueError, match="must be positive"):
        benchmark.main(["--output-dir", str(destination), option, "0"])
    assert not destination.exists()


def test_nonempty_outputs_are_preserved(tmp_path):
    destination = tmp_path / "output"
    destination.mkdir()
    evidence = destination / "existing.json"
    evidence.write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="new or empty"):
        benchmark.main(["--output-dir", str(destination)])
    assert evidence.read_text(encoding="utf-8") == "preserve"


def test_compiled_scope_requires_a_declared_optional_backend_before_input_reads(tmp_path):
    destination = tmp_path / "compiled"
    with pytest.raises(ValueError, match="requires continuous_time_query_backend"):
        benchmark.main(["--output-dir", str(destination), "--baseline-scope", "compiled"])
    assert not destination.exists()


def test_digest_streams_and_supports_official_md5(tmp_path):
    path = tmp_path / "input"
    path.write_bytes(b"abc")
    assert benchmark._digest(path, "md5") == "900150983cd24fb0d6963f7d28e17f72"
    assert benchmark._digest(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_split_binding_hashes_the_actual_capped_disjoint_point_identities():
    problem = ContinuousTimeLidarPairProblem(
        source_records=[], target_points=[(1.0, 2.0, 3.0)] * 6,
        target_capture_timestamps_ns=[10, 11, 12, 13, 14, 15],
        odometry_track=OdometryTrack([]), t_base_source=SE3.identity(),
        initial_t_source_target=SE3.identity(), variable="extrinsic", sensor="lidar",
        voxel_size_m=0.5, correspondence_gate_m=1.5,
        train_indices=(0, 1, 2), holdout_indices=(3, 4, 5),
    )
    options = ContinuousTimeLidarPairOptions(max_target_points_per_split=2)
    binding = benchmark._split_binding(problem, options)
    assert binding["train_count"] == binding["holdout_count"] == 2
    assert binding["train_ids_sha256"] != binding["holdout_ids_sha256"]
    assert binding == benchmark._split_binding(problem, options)
    unselected = replace(problem, target_capture_timestamps_ns=[10, 99, 12, 13, 14, 15])
    assert binding == benchmark._split_binding(unselected, options)
    selected = replace(problem, target_capture_timestamps_ns=[10, 11, 99, 13, 14, 15])
    changed = benchmark._split_binding(selected, options)
    assert changed["train_ids_sha256"] != binding["train_ids_sha256"]
    assert changed["holdout_ids_sha256"] == binding["holdout_ids_sha256"]
    moved = replace(problem, target_points=[(9.0, 2.0, 3.0), *problem.target_points[1:]])
    assert benchmark._split_binding(moved, options)["train_ids_sha256"] != (
        binding["train_ids_sha256"]
    )


def test_runtime_scope_contains_all_geometry_interpolation_and_hash_overlap_modules():
    assert set(benchmark.RUNTIME_MODULES) == {
        "calibrex.data.livox", "calibrex.data.odometry_track",
        "calibrex.graph.lidar_point_to_plane", "calibrex.evaluation.lidar",
        "calibrex.solvers.continuous_time_lidar_pair_solver",
        "calibrex.data.adaptive_voxel", "calibrex.pipelines.online",
    }
    assert all(
        "src/" + name.replace(".", "/") + ".py" in benchmark.SOURCES
        for name in benchmark.RUNTIME_MODULES
    )
    assert benchmark._modules_for_scope("compiled") == (*benchmark.RUNTIME_MODULES,
        "calibrex.solvers.fixed_trajectory_se3_solver",
        "calibrex.solvers.numba_continuous_time_adapter",
        "calibrex.solvers.robust_lidar_observations")
    assert benchmark._modules_for_scope("runtime") == benchmark.RUNTIME_MODULES
    assert benchmark._modules_for_scope("adaptive") == (benchmark.MODULE,)


def test_default_budget_config_keeps_the_same_public_capture_and_matches_sampling_defaults():
    from calibrex.core.config import load_config
    from calibrex.solvers.continuous_time_lidar_pair_solver import ContinuousTimeLidarPairOptions

    bounded = load_config(benchmark.CONFIG).model_dump(mode="json")
    default = load_config(
        "examples/public_datasets/glim_versatile/online_continuous_time_default_budget_config.yaml"
    ).model_dump(mode="json")
    defaults = ContinuousTimeLidarPairOptions()
    opts = default["pipeline"]["factors"]["lidar_rig_point_to_plane"]["options"]
    assert opts["continuous_time_max_source_records"] == defaults.max_source_records
    assert opts["continuous_time_max_target_points_per_split"] == (
        defaults.max_target_points_per_split
    )
    for payload in (bounded, default):
        payload.pop("project")
        options = payload["pipeline"]["factors"]["lidar_rig_point_to_plane"]["options"]
        options.pop("continuous_time_max_source_records")
        options.pop("continuous_time_max_target_points_per_split")
    assert bounded == default
