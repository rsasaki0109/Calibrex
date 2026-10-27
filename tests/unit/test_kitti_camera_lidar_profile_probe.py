"""Nuisance-aware controls expose coupling; held-out scores never optimize poses."""
import json
from dataclasses import replace
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from tests.unit.test_kitti_camera_lidar_axis_probe import _Ridge
from tests.unit.test_kitti_camera_lidar_v3_replay import _snapshot
from tests.unit.test_score_hilti_camera_camera import _load

from calibrex.evaluation import camera_lidar_edge as edge

probe = _load("kitti_camera_lidar_profile_probe")


@pytest.mark.parametrize("coupling,detected", [(0.0, True), (1.0, False)])
def test_profiled_controls_separate_a_coupled_ridge_from_an_identifiable_axis(
    coupling: float, detected: bool,
) -> None:
    result = probe.profile_axis(
        [_Ridge(coupling)] * 3, [_Ridge(coupling, w) for w in (0.8, 1, 1.2)],
        0, 2.0, edge.EdgeAlignmentOptions(),
    )
    assert result.direct_control.detected
    assert result.profiled_control.detected is detected
    expected = edge._controls(
        [_Ridge(coupling, w) for w in (0.8, 1, 1.2)],
        np.array([f.base_theta for f in result.folds]), edge.EdgeAlignmentOptions(),
    )[0]
    assert result.direct_control.model_dump() == expected.model_dump()
    for fold in result.folds:
        for direct, profiled in zip(fold.shifted_theta, fold.profiled_theta, strict=True):
            assert direct[0] == profiled[0]  # Tested correction is never optimized away.
            assert direct[3:] == profiled[3:] == [0.0, 0.0, 0.0]
        if coupling:
            assert fold.profiled_drops == pytest.approx([0, 0])
    jsonschema.validate(result.model_dump(mode="json"), probe.ProfileAxis.model_json_schema())


def test_outside_bound_control_still_refits_bounded_nuisance_coordinates() -> None:
    theta = np.zeros(6)
    theta[0] = 14.0
    result = probe.refit_nuisance(_Ridge(1), theta, 0, edge.EdgeAlignmentOptions())
    assert result[0] == 14.0
    assert result[1] == pytest.approx(-12.0)
    assert theta.tolist() == [14.0, 0, 0, 0, 0, 0]


@pytest.mark.parametrize("multi_start", [False, True])
def test_held_out_objective_only_scores_the_five_final_poses_per_fold(multi_start: bool) -> None:
    class HeldOut(_Ridge):
        def __init__(self) -> None:
            super().__init__(1)
            self.calls: list[list[float]] = []

        def score(self, theta: np.ndarray) -> float:
            self.calls.append(theta.tolist())
            return super().score(theta)

    held = [HeldOut() for _ in range(3)]
    result = probe.profile_axis([_Ridge(1)] * 3, held, 0, 2.0, edge.EdgeAlignmentOptions(),
                                multi_start=multi_start)
    for objective, fold in zip(held, result.folds, strict=True):
        assert objective.calls == [fold.base_theta, *fold.shifted_theta, *fold.profiled_theta]


def test_invalid_fold_inputs_are_rejected() -> None:
    for axis, start, count in [(3, 0, 3), (0, float("nan"), 3), (0, 13, 3), (0, 0, 2)]:
        with pytest.raises(ValueError):
            probe.profile_axis([_Ridge(1)] * count, [_Ridge(1)] * count,
                               axis, start, edge.EdgeAlignmentOptions())
    schema = probe.ProfileReport.model_json_schema()
    report = probe.ProfileReport(status="failed", error="kept", provenance={"test": True})
    jsonschema.validate(json.loads(report.model_dump_json()), schema)


def test_training_profile_start_escapes_a_local_nuisance_peak() -> None:
    class TwoPeaks(edge.EdgeObjective):
        def __init__(self) -> None:
            pass

        def score(self, theta: np.ndarray) -> float:
            return float(max(-theta[1] ** 2, 1.0 - 4 * (theta[1] - 4) ** 2)
                         - theta[2] ** 2)

    objective = TwoPeaks()
    start = np.array([14.0, 0, 0, 0, 0, 0])
    options = edge.EdgeAlignmentOptions()
    local = probe.refit_nuisance(objective, start, 0, options)
    wide = probe.refit_nuisance(objective, start, 0, options, multi_start=True)
    assert local[1] == 0 and wide[1] == 4
    assert local[0] == wide[0] == 14  # Tested shifts outside the bound remain fixed.
    assert objective.score(wide) > objective.score(local)
    assert start.tolist() == [14, 0, 0, 0, 0, 0]
    with pytest.raises(ValueError, match="profile range"):
        probe.refit_nuisance(objective, start, 0, replace(options, start_scan_deg=None),
                             multi_start=True)


def test_multistart_retains_original_local_solution_on_ties() -> None:
    class Flat(edge.EdgeObjective):
        def __init__(self) -> None:
            pass

        def score(self, theta: np.ndarray) -> float:
            return 1.0

    start = np.array([2.0, 1.5, -0.5, 0, 0, 0])
    result = probe.refit_nuisance(Flat(), start, 0, edge.EdgeAlignmentOptions(), multi_start=True)
    np.testing.assert_array_equal(result, start)


def test_training_initialization_finds_a_better_basin_without_pooled_input() -> None:
    class TwoPeaks(edge.EdgeObjective):
        def __init__(self) -> None:
            pass

        def score(self, theta: np.ndarray) -> float:
            return float(max(-theta[0] ** 2, 1 - 4 * (theta[0] - 4) ** 2)
                         - theta[1] ** 2 - theta[2] ** 2)

    train = [TwoPeaks() for _ in range(3)]
    options = edge.EdgeAlignmentOptions()
    local = probe.profile_axis(train, train, 0, 0, options)
    profiled = probe.profile_axis(train, train, 0, 0, options, training_only_start=True)
    assert all(f.base_theta[0] == 0 for f in local.folds)
    assert all(f.base_theta[0] == 4 for f in profiled.folds)
    assert all(f.training_start_theta == [4, 0, 0, 0, 0, 0] for f in profiled.folds)


def test_held_out_changes_cannot_change_training_starts_or_fitted_poses() -> None:
    class Peak(edge.EdgeObjective):
        def __init__(self, center: float) -> None:
            self.center = center
            self.calls: list[list[float]] = []

        def score(self, theta: np.ndarray) -> float:
            self.calls.append(theta.tolist())
            return -float((theta[0] - self.center) ** 2 + theta[1] ** 2 + theta[2] ** 2)

    results = []
    for held_center in (-10, 10):
        training = [Peak(center) for center in (2, 4, -2)]
        held = [Peak(held_center) for _ in training]
        result = probe.profile_axis(training, held, 0, 0, edge.EdgeAlignmentOptions(),
                                    multi_start=True, training_only_start=True)
        for objective, fold, center in zip(held, result.folds, (2, 4, -2), strict=True):
            assert fold.training_start_theta == [center, 0, 0, 0, 0, 0]
            assert objective.calls == [fold.base_theta, *fold.shifted_theta, *fold.profiled_theta]
        results.append(result)
    for left, right in zip(results[0].folds, results[1].folds, strict=True):
        assert left.training_start_theta == right.training_start_theta
        assert left.base_theta == right.base_theta
        assert left.shifted_theta == right.shifted_theta
        assert left.profiled_theta == right.profiled_theta
        assert left.direct_drops != right.direct_drops


def test_training_initialization_refuses_pooled_starts_and_requires_profile_range() -> None:
    options = edge.EdgeAlignmentOptions()
    with pytest.raises(ValueError, match="pooled"):
        probe.profile_axis([_Ridge(0)] * 3, [_Ridge(0)] * 3, 0, 2, options,
                           training_only_start=True)
    with pytest.raises(ValueError, match="profile range"):
        probe.training_profile_start(_Ridge(0), 0, replace(options, start_scan_deg=None))
    for conditional, training in ((None, False), (Path("missing.json"), True)):
        with pytest.raises(ValueError, match="choose either"):
            probe.profile_snapshot(Path("missing"), conditional, Path("unused"),
                                   training_only_start=training)


def test_training_only_snapshot_retains_failure_without_conditional_receipt(tmp_path: Path) -> None:
    source = tmp_path / "snapshot"
    _snapshot(source)
    output = tmp_path / "result"
    before = (source / "arrays.npz").read_bytes()
    with pytest.raises(ValueError, match="not enough prepared frames"):
        probe.profile_snapshot(source, None, output, training_only_start=True)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["provenance"]["training_only_start"] is True
    assert report["provenance"]["conditional_report_sha256"] is None
    assert report["provenance"]["new_evaluation_started"] is False
    assert not any(item.startswith("Initialization uses") for item in report["limitations"])
    jsonschema.validate(report, json.loads((output / "report.schema.json").read_text()))
    assert (source / "arrays.npz").read_bytes() == before
