"""Partition medians, stable support floors and packed subsets keep scalar bits."""

import math
import struct
from statistics import median

import numpy as np
import pytest

from calibrex.core.geometry import SE3
from calibrex.graph.lidar_point_to_plane import (
    LidarPointToPlaneObservation,
    LidarRigPointToPlaneFactor,
)
from calibrex.solvers.robust_lidar_observations import (
    _partition_median,
    filter_lidar_observations_mad,
)


def _bits(value):
    return struct.pack("!d", value)


@pytest.mark.parametrize("count", [1, 2, 63, 64, 65, 4096, 4097])
def test_partition_medians_match_statistics_bits(count):
    rng = np.random.default_rng(27)
    values = rng.normal(size=count)
    for dataset in (values, np.resize(np.array([-0.0, 0.0, 0.02, 0.02, 0.04, 1e308]), count)):
        absolute = np.abs(dataset)
        expected = float(median(sorted(absolute.tolist())))
        actual = _partition_median(absolute)
        assert _bits(actual) == _bits(expected)
        if math.isfinite(expected):
            deviations = np.abs(absolute - expected)
            assert _bits(_partition_median(deviations)) == _bits(float(median(deviations.tolist())))


@pytest.mark.parametrize("force_floor", [False, True])
def test_mad_filter_retains_exact_support_floor_ties_threshold_and_original_order(force_floor):
    z_values = [0.001] * 70 + [0.02] * 10 + [0.2] * 10 + [1.0] * 10
    observations = [
        LidarPointToPlaneObservation((float(i), 0.2, z), (float(i), 0.2, 0.0), (0.0, 0.0, 1.0))
        for i, z in enumerate(z_values)
    ]
    transform = SE3.identity()
    factor = LidarRigPointToPlaneFactor(
        variable="T", sensor="lidar", t_ego_lidar=transform, observations=observations
    )
    residuals = factor.residuals()
    absolute = sorted(abs(r) for r in residuals)
    center = float(median(absolute))
    mad = float(median([abs(r - center) for r in absolute]))
    threshold = max(0.01, center + 3.5 * (1.4826 * mad))
    accepted = [i for i, r in enumerate(residuals) if abs(r) <= threshold]
    required = 95 if force_floor else 6
    fallback = len(accepted) < required
    if fallback:
        accepted = sorted(range(len(residuals)), key=lambda i: (abs(residuals[i]), i))[:required]
    expected = tuple(o for i, o in enumerate(factor.observations) if i in set(accepted))
    result = filter_lidar_observations_mad(
        factor.observations,
        variable="T",
        sensor="lidar",
        t_ego_lidar=transform,
        minimum_threshold_m=0.01,
        minimum_inliers=required,
        _factor=factor,
    )
    assert result.observations == expected and result.fallback_used == fallback
    for actual, reference in (
        (result.median_abs_residual_m, center),
        (result.mad_m, mad),
        (result.threshold_m, threshold),
    ):
        assert _bits(actual) == _bits(reference)
    assert factor.observations is result.observations
    packed = factor._array_observations
    assert packed is not None and packed[0] is factor.observations
    fresh = LidarRigPointToPlaneFactor(
        variable="T", sensor="lidar", t_ego_lidar=transform, observations=result.observations
    )
    assert factor.linearize() == fresh.linearize()


def test_prepared_factor_mismatch_is_rejected():
    observations = tuple(
        LidarPointToPlaneObservation((float(i), 0.1, 0.2), (0.1, 0.2, 0.3), (0.0, 0.0, 1.0))
        for i in range(65)
    )
    transform = SE3.identity()
    factor = LidarRigPointToPlaneFactor(
        variable="T", t_ego_lidar=transform, observations=observations
    )
    with pytest.raises(ValueError, match="must match"):
        filter_lidar_observations_mad(
            observations, variable="other", sensor="lidar0", t_ego_lidar=transform, _factor=factor
        )
