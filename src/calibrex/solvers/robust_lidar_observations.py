"""Deterministic robust filtering for LiDAR point-to-plane observations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3
from calibrex.graph.lidar_point_to_plane import (
    LidarPointToPlaneObservation,
    LidarRigPointToPlaneFactor,
    _PackedObservations,
)


@dataclass(frozen=True)
class RobustObservationFilterResult:
    """Summary of one residual-based observation filter pass."""

    observations: tuple[LidarPointToPlaneObservation, ...]
    total_count: int
    retained_count: int
    rejected_count: int
    median_abs_residual_m: float | None
    mad_m: float | None
    threshold_m: float | None
    fallback_used: bool


def filter_lidar_observations_mad(
    observations: Sequence[LidarPointToPlaneObservation],
    *,
    variable: str,
    t_ego_lidar: SE3,
    sensor: str,
    mad_scale: float = 3.5,
    minimum_threshold_m: float = 0.02,
    minimum_inlier_fraction: float = 0.25,
    minimum_inliers: int = 6,
    _factor: LidarRigPointToPlaneFactor | None = None,
) -> RobustObservationFilterResult:
    """Reject large initial residuals with a MAD gate and a support floor.

    The filter is deterministic and applied before robust LM.  It is intended
    for solid-state captures where transient returns or moving objects can
    contaminate a voxel-plane correspondence set.  If the MAD gate would
    remove too much support, the smallest-residual observations are retained
    to keep the six-DoF problem evaluable; that fallback is reported explicitly.
    """

    source = tuple(observations)
    total_count = len(source)
    if total_count == 0:
        return RobustObservationFilterResult(
            observations=(),
            total_count=0,
            retained_count=0,
            rejected_count=0,
            median_abs_residual_m=None,
            mad_m=None,
            threshold_m=None,
            fallback_used=False,
        )
    if mad_scale < 0.0:
        raise ValueError("mad_scale must be non-negative")
    if minimum_threshold_m < 0.0:
        raise ValueError("minimum_threshold_m must be non-negative")
    if not 0.0 < minimum_inlier_fraction <= 1.0:
        raise ValueError("minimum_inlier_fraction must be in (0, 1]")
    if minimum_inliers < 1:
        raise ValueError("minimum_inliers must be positive")

    factor = _factor if _factor is not None else LidarRigPointToPlaneFactor(
        variable=variable,
        t_ego_lidar=t_ego_lidar,
        observations=source,
        sensor=sensor,
    )
    if _factor is not None and (
        factor.observations is not source or factor.t_ego_lidar is not t_ego_lidar
        or factor.variable != variable or factor.sensor != sensor
    ):
        raise ValueError("prepared factor must match the filter's observations and transform")
    residuals = factor.residuals()
    absolute: NDArray[np.float64] | None = None
    if len(residuals) >= 64 and all(type(value) is float for value in residuals):
        values = np.asarray(residuals, dtype=np.float64)
        if np.isfinite(values).all():
            absolute = np.abs(values)
    if absolute is None:
        absolute_residuals = sorted(abs(value) for value in residuals)
        median_abs = float(median(absolute_residuals))
        deviations = [abs(value - median_abs) for value in absolute_residuals]
        mad = float(median(deviations))
    else:
        median_abs = _partition_median(absolute)
        mad = _partition_median(np.abs(absolute-median_abs))
    robust_scale = 1.4826 * mad
    threshold = max(
        minimum_threshold_m,
        median_abs + mad_scale * robust_scale,
    )
    accepted = np.flatnonzero(absolute <= threshold).tolist() if absolute is not None else [
        index
        for index, observation in enumerate(source)
        if abs(residuals[index]) <= threshold
    ]
    required_count = max(
        minimum_inliers,
        math.ceil(total_count * minimum_inlier_fraction),
    )
    fallback_used = len(accepted) < required_count
    if fallback_used:
        ranked = np.argsort(absolute, kind="stable").tolist() if absolute is not None else sorted(
            range(total_count),
            key=lambda index: (abs(residuals[index]), index),
        )
        accepted = ranked[: min(required_count, total_count)]
    accepted_set = set(accepted)
    filtered = tuple(
        observation
        for index, observation in enumerate(source)
        if index in accepted_set
    )
    if _factor is not None:
        factor._retain_observations(filtered, sorted(accepted_set))
    return RobustObservationFilterResult(
        observations=filtered,
        total_count=total_count,
        retained_count=len(filtered),
        rejected_count=total_count - len(filtered),
        median_abs_residual_m=median_abs,
        mad_m=mad,
        threshold_m=threshold,
        fallback_used=fallback_used,
    )


def _partition_median(values: NDArray[np.float64]) -> float:
    middle = len(values)//2
    positions = (middle-1, middle) if len(values) % 2 == 0 else (middle,)
    partitioned = np.partition(values, positions)
    with np.errstate(over="ignore", invalid="ignore"):
        return (float((partitioned[middle-1]+partitioned[middle])/2.0)
                if len(values) % 2 == 0 else float(partitioned[middle]))


def _filter_packed_factor_mad(
    factor: LidarRigPointToPlaneFactor, *, mad_scale: float, minimum_threshold_m: float,
    minimum_inlier_fraction: float, minimum_inliers: int,
) -> int | None:
    """Retain native packed rows with the public filter's exact gate/support ordering."""
    if type(factor._observations) is not _PackedObservations:
        return None
    total = factor._observation_count()
    if total < 64:
        return None
    if mad_scale < 0.0:
        raise ValueError("mad_scale must be non-negative")
    if minimum_threshold_m < 0.0:
        raise ValueError("minimum_threshold_m must be non-negative")
    if not 0.0 < minimum_inlier_fraction <= 1.0:
        raise ValueError("minimum_inlier_fraction must be in (0, 1]")
    if minimum_inliers < 1:
        raise ValueError("minimum_inliers must be positive")
    initial = factor._compiled_initial_residuals()
    absolute = np.abs(initial if initial is not None
                      else np.asarray(factor.residuals(), dtype=np.float64))
    if not np.isfinite(absolute).all():
        return None
    median_abs = _partition_median(absolute)
    mad = _partition_median(np.abs(absolute-median_abs))
    threshold = max(minimum_threshold_m, median_abs + mad_scale * (1.4826 * mad))
    accepted = np.flatnonzero(absolute <= threshold)
    required = max(minimum_inliers, math.ceil(total * minimum_inlier_fraction))
    if len(accepted) < required:
        accepted = np.sort(np.argsort(absolute, kind="stable")[:min(required, total)])
    factor._retain_packed(accepted)
    return total-len(accepted)
