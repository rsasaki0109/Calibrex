"""Batching parity against the original scalar geometry and gate definitions."""

from __future__ import annotations

from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from calibrex.core.geometry import SE3
from calibrex.core.probabilistic_correspondence import (
    ProbabilisticCorrespondenceFrame,
    ProbabilisticImageCorrespondence,
)
from calibrex.data.depth import DepthCameraIntrinsics
from calibrex.solvers import probabilistic_camera_lidar_refiner as refiner


def _frame(
    projection: str,
    distortion: int,
    *,
    count: int = 257,
) -> ProbabilisticCorrespondenceFrame:
    rng = np.random.default_rng(617)
    camera = DepthCameraIntrinsics(
        width=640,
        height=480,
        fx=270.0,
        fy=280.0,
        cx=320.0,
        cy=240.0,
        projection=projection,
        xi=0.7 if projection != "pinhole" else None,
        alpha=0.6 if projection == "double_sphere" else None,
        distortion_model="radial-tangential" if distortion else "none",
        distortion=[0.02, -0.003, 0.0001, -0.0002, 0.0004][:distortion],
    )
    points = rng.uniform(-10.0, 10.0, (count, 3))
    points[:4] = [[0.0, 0.0, 0.0], [0.0, 0.0, 1e-9], [0.0, 0.0, -1.0], [0.0, 0.0, 5.0]]
    return ProbabilisticCorrespondenceFrame(
        frame_id=projection,
        capture_time_ns=0,
        camera_frame="camera",
        lidar_frame="lidar",
        intrinsics=camera,
        correspondences=[
            ProbabilisticImageCorrespondence(
                correspondence_id=str(i),
                point_lidar_m=p.tolist(),
                image_mean_px=rng.uniform(0.0, 640.0, 2).tolist(),
                image_covariance_px2=[1.5, 0.125, 0.125, 2.0],
                reliability=[0.0, 0.25, 0.75, 1.0][i % 4],
                outlier_probability=[0.0, 0.25, 0.5][i % 3],
            )
            for i, p in enumerate(points)
        ],
    )


@pytest.mark.parametrize(
    "projection,distortion",
    [
        ("pinhole", 0),
        ("pinhole", 4),
        ("pinhole", 5),
        ("mei", 4),
        ("double_sphere", 0),
        ("double_sphere", 5),
    ],
)
@pytest.mark.parametrize("flags", list(product([False, True], repeat=3)))
def test_scalar_evaluation_bits_across_models_gates_and_ablations(
    projection: str,
    distortion: int,
    flags: tuple[bool, bool, bool],
) -> None:
    frame = _frame(projection, distortion)
    settings = refiner.ProbabilisticCameraLidarRefinementOptions(
        use_covariance=flags[0],
        use_reliability=flags[1],
        use_outlier_probability=flags[2],
    )
    for transform in (SE3.identity(), SE3((0.1, -0.2, 0.3), (0.15, -0.06, 0.02, 0.95))):
        expected = refiner._evaluate_scalar_pose([frame], transform, settings)
        actual = refiner.evaluate_probabilistic_camera_lidar_pose([frame], transform, settings)
        assert actual == expected
        for key in ("objective", "weighted_reprojection_rmse_px", "mean_mahalanobis_error"):
            if getattr(expected, key) is not None:
                assert np.float64(getattr(actual, key)).view(np.uint64) == (
                    np.float64(getattr(expected, key)).view(np.uint64)
                )


@pytest.mark.parametrize("projection", ["mei", "double_sphere"])
def test_invalid_optical_denominators_and_empty_projections(projection: str) -> None:
    frame = _frame(projection, 4)
    frame.intrinsics.xi = -2.0
    settings = refiner.ProbabilisticCameraLidarRefinementOptions()
    for frames in ([], [frame]):
        actual = refiner.evaluate_probabilistic_camera_lidar_pose(frames, SE3.identity(), settings)
        assert actual == refiner._evaluate_scalar_pose(frames, SE3.identity(), settings)


def test_prepared_distributions_are_local_snapshots_and_budget_fallback_is_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame("pinhole", 5)
    settings = refiner.ProbabilisticCameraLidarRefinementOptions()
    expected = refiner._evaluate_scalar_pose([frame], SE3.identity(), settings)
    prepared = refiner._PreparedPoseEvaluation([frame], settings)
    frame.intrinsics.fx *= 2
    frame.correspondences[3].point_lidar_m[0] = 1000.0
    assert prepared.evaluate(SE3.identity()) == expected
    monkeypatch.setattr(refiner, "_POSE_CAPTURE_BUDGET_BYTES", 1)
    limited = refiner._PreparedPoseEvaluation([frame], settings)
    assert limited.fallback and not limited.frames
    assert limited.evaluate(SE3.identity()) == (
        refiner._evaluate_scalar_pose([frame], SE3.identity(), settings)
    )


def test_complete_solver_trace_selection_and_holdout_match_scalar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = [
        _frame("pinhole", 5, count=97).model_copy(update={"frame_id": str(i)}) for i in range(4)
    ]
    settings = replace(refiner.ProbabilisticCameraLidarRefinementOptions(), max_evaluations=73)
    expected = refiner.ProbabilisticCameraLidarRefiner().solve(frames, SE3.identity(), settings)

    class ScalarEvaluator:
        def __init__(self, rows: object, options: object) -> None:
            self.frames, self.options = rows, options

        def evaluate(self, transform: SE3) -> refiner.ProbabilisticPoseEvaluation:
            return refiner._evaluate_scalar_pose(self.frames, transform, self.options)

    monkeypatch.setattr(refiner, "_PreparedPoseEvaluation", ScalarEvaluator)
    actual = refiner.ProbabilisticCameraLidarRefiner().solve(frames, SE3.identity(), settings)
    assert expected.as_dict() == actual.as_dict()
