"""Nuisance-aware controls expose coupling; held-out scores never optimize poses."""
import json

import jsonschema
import numpy as np
import pytest
from tests.unit.test_kitti_camera_lidar_axis_probe import _Ridge
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


def test_held_out_objective_only_scores_the_five_final_poses_per_fold() -> None:
    class HeldOut(_Ridge):
        def __init__(self) -> None:
            super().__init__(1)
            self.calls: list[list[float]] = []

        def score(self, theta: np.ndarray) -> float:
            self.calls.append(theta.tolist())
            return super().score(theta)

    held = [HeldOut() for _ in range(3)]
    result = probe.profile_axis([_Ridge(1)] * 3, held, 0, 2.0, edge.EdgeAlignmentOptions())
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
