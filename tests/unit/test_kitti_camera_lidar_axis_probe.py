"""Conditional precision must not be confused with joint identifiability."""
import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from tests.unit.test_kitti_camera_lidar_v3_replay import _snapshot
from tests.unit.test_score_hilti_camera_camera import _load

from calibrex.evaluation import camera_lidar_edge as edge

probe = _load("kitti_camera_lidar_axis_probe")


class _Ridge(edge.EdgeObjective):
    def __init__(self, coupling: float, weight: float = 1.0) -> None:
        self.coupling = coupling
        self.weight = weight

    def score(self, theta: np.ndarray) -> float:
        return -self.weight * float(theta[0] + self.coupling * theta[1] - 2.0) ** 2


@pytest.mark.parametrize("coupling,span", [(0.0, 0.0), (1.0, 4.0)])
def test_nuisance_sweep_exposes_a_ridge_despite_strong_conditional_controls(
    coupling: float, span: float,
) -> None:
    result = probe.probe_axis(
        _Ridge(coupling), [_Ridge(coupling)] * 3,
        [_Ridge(coupling, w) for w in (0.8, 1.0, 1.2)], 0, edge.EdgeAlignmentOptions(),
    )
    assert result.correction_deg == pytest.approx(2.0)
    assert result.conditional_std_deg == pytest.approx(0.025)
    assert result.conditional_control.detected
    assert result.nuisance_correction_span_deg == pytest.approx(span)
    assert {(p.fixed_axis, p.fixed_offset_deg) for p in result.nuisance} == {
        (axis, value) for axis in (1, 2) for value in (-2.0, 2.0)
    }
    assert result.reference_error_deg is None  # Fitting receives no reference.
    assert not result.at_search_bound
    jsonschema.validate(result.model_dump(mode="json"), probe.AxisProbe.model_json_schema())


def test_probe_refuses_invalid_axis_or_unpaired_folds() -> None:
    objective = _Ridge(1)
    for axis, count in ((3, 3), (0, 2)):
        with pytest.raises(ValueError, match="three paired folds"):
            probe.probe_axis(objective, [objective] * count, [objective] * count,
                             axis, edge.EdgeAlignmentOptions())


def test_insufficient_capture_retains_schema_valid_failure_and_input(tmp_path: Path) -> None:
    source = tmp_path / "snapshot"
    _snapshot(source)
    before = (source / "arrays.npz").read_bytes()
    output = tmp_path / "result"
    with pytest.raises(ValueError, match="not enough prepared frames"):
        probe.probe_snapshot(source, output)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed" and report["probes"] == []
    assert report["provenance"]["new_evaluation_started"] is False
    jsonschema.validate(report, json.loads((output / "report.schema.json").read_text()))
    assert (source / "arrays.npz").read_bytes() == before
    with pytest.raises(FileExistsError):
        probe.probe_snapshot(source, output)
    with pytest.raises(ValueError, match="separate"):
        probe.probe_snapshot(source, source / "probe")
