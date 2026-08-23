"""Synthetic recovery for sliding-window trajectory marginalization."""

from __future__ import annotations

from typing import Literal

import numpy as np

from calibrex.core.continuous_time_sliding_window import (
    ContinuousTimeSlidingWindowArtifact,
    ContinuousTimeSlidingWindowProvenance,
)
from calibrex.core.continuous_time_sliding_window_fit import (
    fit_sliding_window_trajectory,
)
from calibrex.core.continuous_time_sparse import (
    ContinuousTimeTrajectoryFitOptions,
    ContinuousTimeTrajectoryFitProblem,
    point_to_plane_rmse,
)
from calibrex.core.geometry import SE3
from calibrex.core.provenance import git_commit
from calibrex.core.se3_manifold import se3_exp
from calibrex.evaluation.continuous_time_lidar_point_to_plane import (
    _measurements,
    _truth_knots,
)

_SYNTHETIC_SEED = 20260820
_WINDOW_KNOT_COUNTS = (4, 3)
_OVERLAP_KNOT_COUNT = 1
_KNOWN_BAD_TRANSLATION_M = (0.0, 0.0, 0.15)


def run_synthetic_sliding_window_recovery(
    *,
    seed: int = _SYNTHETIC_SEED,
    command: list[str] | None = None,
    recovery_id: str = "ct-sliding-window-synthetic",
) -> ContinuousTimeSlidingWindowArtifact:
    """Recover a knot path with sliding-window marginalization and controls."""

    rng = np.random.default_rng(seed)
    timestamps = tuple(float(index) for index in range(6))
    truth = _truth_knots(timestamps)
    samples: list[tuple[float, int]] = []
    for sample_index in range(90):
        timestamp = float(rng.uniform(0.05, 4.95))
        plane_index = int(sample_index % 3)
        samples.append((timestamp, plane_index))
    split_start = 0.35 * timestamps[-1]
    split_end = 0.65 * timestamps[-1]
    train_samples = [
        item for item in samples if item[0] <= split_start or item[0] > split_end
    ]
    holdout_samples = [
        item for item in samples if split_start < item[0] <= split_end
    ]
    train = _measurements(truth, timestamps, train_samples, rng, prefix="train")
    holdout = _measurements(truth, timestamps, holdout_samples, rng, prefix="holdout")
    initial = tuple(
        se3_exp(knot, rng.normal(0.0, 0.04, 6)) for knot in truth
    )
    problem = ContinuousTimeTrajectoryFitProblem(
        knot_timestamps=timestamps,
        initial_knot_poses=initial,
        point_to_plane_measurements=train,
    )
    options = ContinuousTimeTrajectoryFitOptions(max_iterations=80)
    sliding = fit_sliding_window_trajectory(
        problem,
        window_knot_counts=_WINDOW_KNOT_COUNTS,
        overlap_knot_count=_OVERLAP_KNOT_COUNT,
        options=options,
        apply_marginalization=True,
    )
    known_bad = fit_sliding_window_trajectory(
        problem,
        window_knot_counts=_WINDOW_KNOT_COUNTS,
        overlap_knot_count=_OVERLAP_KNOT_COUNT,
        options=options,
        apply_marginalization=False,
        bad_overlap_translation_m=_KNOWN_BAD_TRANSLATION_M,
    )
    batch_holdout = point_to_plane_rmse(
        holdout, timestamps, sliding.batch_result.knot_poses
    )
    sliding_holdout = point_to_plane_rmse(
        holdout, timestamps, sliding.sliding_result.knot_poses
    )
    known_bad_holdout = point_to_plane_rmse(
        holdout, timestamps, known_bad.sliding_result.knot_poses
    )
    if batch_holdout is None or sliding_holdout is None or known_bad_holdout is None:
        raise ValueError("sliding-window recovery produced incomplete holdout RMSE")
    known_bad_delta = known_bad_holdout - sliding_holdout
    overlap_start = _WINDOW_KNOT_COUNTS[0] - _OVERLAP_KNOT_COUNT
    known_bad_overlap = _max_overlap_translation_error(
        sliding.batch_result.knot_poses,
        known_bad.sliding_result.knot_poses,
        overlap_start=overlap_start,
    )
    policy_status, policy_reason = _policy(
        batch_status=sliding.batch_result.status,
        max_overlap_translation=sliding.max_overlap_translation_error_m,
        max_overlap_rotation=sliding.max_overlap_rotation_error_rad,
        sliding_holdout=sliding_holdout,
        known_bad_overlap=known_bad_overlap,
        known_bad_delta=known_bad_delta,
    )
    return ContinuousTimeSlidingWindowArtifact(
        recovery_id=recovery_id,
        interpolation="screw_linear",
        seed=seed,
        knot_count=len(truth),
        window_count=sliding.window_count,
        overlap_knot_count=sliding.overlap_knot_count,
        train_measurement_count=len(train),
        holdout_measurement_count=len(holdout),
        batch_fit_status=sliding.batch_result.status,
        sliding_fit_status=sliding.sliding_result.status,
        max_overlap_translation_error_m=sliding.max_overlap_translation_error_m,
        max_overlap_rotation_error_rad=sliding.max_overlap_rotation_error_rad,
        batch_holdout_rmse_m=batch_holdout,
        sliding_holdout_rmse_m=sliding_holdout,
        known_bad_overlap_translation_error_m=known_bad_overlap,
        known_bad_rmse_delta_m=known_bad_delta,
        known_bad_skip_marginalization=True,
        policy_status=policy_status,
        policy_reason=policy_reason,
        provenance=ContinuousTimeSlidingWindowProvenance(
            generator=__name__,
            generator_version="0.1",
            git_commit=git_commit(),
            command=command or [],
            seed=seed,
        ),
    )


def _max_overlap_translation_error(
    reference: tuple[SE3, ...],
    estimate: tuple[SE3, ...],
    *,
    overlap_start: int,
) -> float:
    from calibrex.core.se3_manifold import se3_log

    errors = [
        float(np.linalg.norm(se3_log(reference[index], estimate[index])[:3]))
        for index in range(overlap_start, len(reference))
    ]
    return max(errors) if errors else 0.0


def _policy(
    *,
    batch_status: str,
    max_overlap_translation: float,
    max_overlap_rotation: float,
    sliding_holdout: float,
    known_bad_overlap: float,
    known_bad_delta: float,
) -> tuple[Literal["pass", "fail"], str]:
    if batch_status != "converged":
        return (
            "fail",
            f"batch reference fit status {batch_status!r} is not converged",
        )
    if max_overlap_translation >= 0.05:
        return (
            "fail",
            (
                f"sliding-window overlap translation error "
                f"{max_overlap_translation:.4f} m exceeds the 0.05 budget"
            ),
        )
    if max_overlap_rotation >= 0.05:
        return (
            "fail",
            (
                f"sliding-window overlap rotation error "
                f"{max_overlap_rotation:.4f} rad exceeds the 0.05 budget"
            ),
        )
    if sliding_holdout >= 0.08:
        return (
            "fail",
            f"sliding-window holdout RMSE {sliding_holdout:.4f} m is too large",
        )
    if known_bad_overlap <= max_overlap_translation + 0.02:
        return (
            "fail",
            (
                "skipping marginalization with a bad overlap seed did not "
                f"increase inconsistency enough ({known_bad_overlap:.4f} m vs "
                f"{max_overlap_translation:.4f} m)"
            ),
        )
    if known_bad_delta <= 0.01:
        return (
            "fail",
            (
                "skipping marginalization only increased holdout RMSE by "
                f"{known_bad_delta:.4f} m"
            ),
        )
    return (
        "pass",
        (
            "sliding-window marginalization matches the converged batch overlap "
            "within budget, holdout RMSE stays tight, and skipping "
            "marginalization with a bad overlap seed increases overlap "
            "inconsistency and holdout error"
        ),
    )
