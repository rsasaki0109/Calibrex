from dataclasses import replace

import pytest

from calibrex.core.geometry import SE3
from calibrex.graph.joint_factors import (
    JointPointToPlaneMeasurement,
    JointRadarDopplerMeasurement,
    make_joint_point_to_plane_factor,
    make_joint_prior_factor,
    make_joint_radar_doppler_factor,
    se3_from_tangent,
)


def test_point_to_plane_factor_couples_pose_and_extrinsic_blocks() -> None:
    pose_truth = (0.1, -0.2, 0.05, 0.0, 0.0, 0.0)
    extrinsic_truth = (0.3, 0.1, -0.1, 0.0, 0.0, 0.0)
    point = (1.0, 2.0, 3.0)
    world_point = (
        se3_from_tangent(pose_truth)
        .compose(se3_from_tangent(extrinsic_truth))
        .transform_point(point)
    )
    measurement = JointPointToPlaneMeasurement(
        "plane-1",
        "capture-1",
        point,
        world_point,
        (0.2, -0.3, 0.9),
        SE3.identity(),
        SE3.identity(),
    )
    factor = make_joint_point_to_plane_factor(
        measurement, pose_block="pose", extrinsic_block="extrinsic"
    )

    assert factor.variable_names == ("pose", "extrinsic")
    assert factor.family == "lidar_point_to_plane"
    assert factor.residuals({"pose": pose_truth, "extrinsic": extrinsic_truth}) == pytest.approx(
        (0.0,), abs=1e-12
    )
    assert abs(factor.residuals({"pose": (0.0,) * 6, "extrinsic": (0.0,) * 6})[0]) > 0.01


def test_radar_factor_couples_velocity_lever_arm_rotation_and_time() -> None:
    velocity = (5.0, -1.0, 0.3)
    extrinsic = (0.4, -0.2, 0.1, 0.02, -0.03, 0.08)
    offset = (0.025,)
    base = JointRadarDopplerMeasurement(
        "radar-1",
        "capture-1",
        (0.8, 0.5, 0.1),
        0.0,
        linear_acceleration_body_mps2=(1.2, -0.4, 0.2),
        angular_velocity_body_radps=(0.1, -0.2, 0.7),
    )
    zero_measurement_factor = make_joint_radar_doppler_factor(
        base,
        velocity_block="velocity",
        extrinsic_block="extrinsic",
        time_offset_block="dt",
    )
    predicted = -zero_measurement_factor.residuals(
        {"velocity": velocity, "extrinsic": extrinsic, "dt": offset}
    )[0]
    factor = make_joint_radar_doppler_factor(
        replace(base, measured_radial_velocity_mps=predicted),
        velocity_block="velocity",
        extrinsic_block="extrinsic",
        time_offset_block="dt",
    )

    assert factor.family == "radar_doppler_spatiotemporal"
    assert factor.residuals(
        {"velocity": velocity, "extrinsic": extrinsic, "dt": offset}
    ) == pytest.approx((0.0,), abs=1e-12)
    assert (
        abs(factor.residuals({"velocity": velocity, "extrinsic": extrinsic, "dt": (0.0,)})[0])
        > 1e-3
    )


def test_joint_prior_factor_normalizes_each_dimension() -> None:
    factor = make_joint_prior_factor(
        factor_id="prior",
        observation_group="prior-group",
        block="bias",
        target=(1.0, -2.0),
        sigma=(0.5, 2.0),
    )

    assert factor.residuals({"bias": (2.0, 0.0)}) == pytest.approx((2.0, 1.0))
    assert factor.split_policy == "train_only"


def test_se3_tangent_requires_six_values() -> None:
    with pytest.raises(ValueError, match="six"):
        se3_from_tangent((0.0, 0.0))
