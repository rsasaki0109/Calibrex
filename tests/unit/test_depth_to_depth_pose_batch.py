"""Exact arithmetic and pattern-search behavior for optional candidate batching."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_depth_to_depth_solver import (
    DepthPairHistogram,
    DepthToDepthCameraModel,
    DepthToDepthObservation,
    DepthToDepthOptions,
    _histogram_mutual_information,
    _histogram_mutual_information_batch,
    evaluate_depth_to_depth_mi,
    evaluate_depth_to_depth_mi_batch,
    project_depth_pairs,
)
from calibrex.solvers.borer_six_dof_solver import (
    BorerSixDofOptions,
    BorerSixDofSolver,
    apply_local_se3_delta,
)


@pytest.mark.parametrize("bins", [4, 7, 8, 31, 32, 33, 64])
def test_grouped_entropy_retains_native_reduction_bits(bins: int) -> None:
    rng = np.random.default_rng(441)
    histograms = rng.integers(0, 10**8, size=(81, bins, bins)).astype(np.float64)
    histograms[rng.random(histograms.shape) < 0.73] = 0.
    histograms[0] = 0.
    histograms[1] = 0.
    histograms[1, 0, 0] = 1.
    histograms[2, 1, 2] = float(2**54)
    assert _histogram_mutual_information_batch(histograms) == tuple(
        _histogram_mutual_information(row) for row in histograms
    )


def observations() -> tuple[DepthToDepthObservation, ...]:
    rng = np.random.default_rng(904)
    camera = DepthToDepthCameraModel(30, 24, 17., 18., 14.5, 11.5)
    points = rng.uniform((-6., -5., -2.), (6., 5., 20.), (800, 3))
    depth = rng.uniform(1., 24., (24, 30))
    depth[::3, ::5] = np.nan
    return (
        DepthToDepthObservation("first", depth, points, camera),
        DepthToDepthObservation("empty", depth, np.empty((0, 3)), camera),
        DepthToDepthObservation("invalid", np.full((24, 30), np.nan), points, camera),
        DepthToDepthObservation("last", depth[::-1].copy(), points[::-1].copy(), camera),
    )


class BatchProjector:
    def __init__(self) -> None:
        self.preparations = []
        self.pose_counts = []

    def __call__(self, *args, **kwargs):
        return project_depth_pairs(*args, **kwargs)

    def prepare_pose_search(self, *args, **kwargs):
        self.preparations.append((args, kwargs))

    def histogram_depth_pairs_poses(self, frames, transforms, **kwargs):
        self.pose_counts.append(len(transforms))
        for transform in transforms:
            rows = []
            for frame in frames:
                pairs = project_depth_pairs(frame, transform, use_z_buffer=kwargs["use_z_buffer"])
                grid = np.histogram2d(
                    pairs.camera_depth, pairs.lidar_range_m, bins=kwargs["bins"],
                    range=(kwargs["camera_range"], kwargs["lidar_range"]),
                )[0]
                rows.append(DepthPairHistogram(
                    grid, len(pairs.camera_depth), pairs.projected_count_before_visibility,
                ))
            yield tuple(rows)


@pytest.mark.parametrize("z_buffer", [False, True])
def test_pose_batches_match_all_frames_skips_and_empty_inputs(z_buffer: bool) -> None:
    frames = observations()
    poses = tuple(apply_local_se3_delta(SE3.identity(), (i*.08, -.2, .1), (.03, .0, -.02))
                  for i in range(11))
    options = DepthToDepthOptions(
        histogram_bins=9, min_visible_points=4, use_z_buffer=z_buffer,
        camera_depth_range=(2., 21.), lidar_range_m=(1., 18.),
    )
    expected = tuple(evaluate_depth_to_depth_mi(frames, pose, options) for pose in poses)
    assert evaluate_depth_to_depth_mi_batch(frames, poses, options) == expected
    projector = BatchProjector()
    assert evaluate_depth_to_depth_mi_batch(frames, poses, options, projector=projector) == expected
    assert projector.pose_counts == [11]
    assert expected[0].skipped_frame_ids == ("empty", "invalid")
    assert evaluate_depth_to_depth_mi_batch(frames, (), options, projector=projector) == ()
    for empty in ((), frames[1:3]):
        actual = evaluate_depth_to_depth_mi_batch(empty, poses, options, projector=projector)
        assert actual == tuple(evaluate_depth_to_depth_mi(empty, p, options) for p in poses)
    with ThreadPoolExecutor(max_workers=2) as executor:
        assert evaluate_depth_to_depth_mi_batch(
            frames, poses, options, projector=projector, frame_executor=executor,
        ) == expected
    assert projector.pose_counts == [11]  # CPU executors use the ordinary path.


@pytest.mark.parametrize("budget", [13, 14, 20, 25, 41])
@pytest.mark.parametrize("objective", ["mutual_information", "normalized_mutual_information"])
def test_pattern_search_batches_preserve_partial_budget_and_complete_trace(budget, objective):
    frames = observations()
    initial = apply_local_se3_delta(SE3.identity(), (.4, -.3, .2), (.15, -.1, .05))
    options = BorerSixDofOptions(
        max_evaluations=budget, objective=objective,
        rotation_bound_deg=.25, translation_bound_m=.1,
        initial_rotation_step_deg=.25, initial_translation_step_m=.1,
        d2d=DepthToDepthOptions(histogram_bins=8, min_visible_points=4),
    )
    expected = BorerSixDofSolver().solve(frames, initial, options)
    projector = BatchProjector()
    actual = BorerSixDofSolver().solve(frames, initial, replace(options, projector=projector))
    assert actual == expected
    assert actual.evaluation_count == len(actual.trace) == budget
    assert len(projector.preparations) == 1
    assert sum(projector.pose_counts) == budget-1
    assert max(projector.pose_counts) <= 12


@pytest.mark.parametrize("error", ["missing_pose", "extra_pose", "missing_frame"])
def test_invalid_batch_protocol_lengths_raise(error, monkeypatch):
    projector = BatchProjector()
    original = projector.histogram_depth_pairs_poses

    def invalid(frames, poses, **kwargs):
        rows = list(original(frames, poses, **kwargs))
        if error == "missing_pose":
            return rows[:-1]
        if error == "extra_pose":
            return rows+rows[:1]
        return tuple(row[:-1] for row in rows)

    monkeypatch.setattr(projector, "histogram_depth_pairs_poses", invalid)
    with pytest.raises(ValueError, match="histogram count"):
        evaluate_depth_to_depth_mi_batch(
            observations(), (SE3.identity(),)*5, projector=projector,
        )
