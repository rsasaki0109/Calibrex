import math
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
from calibrex.graph.joint_optimization import (
    BackendNeutralJointOptimizer,
    JointOptimizerOptions,
    JointParameterBlock,
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


def test_multicapture_pose_prior_problem_recovers_shared_extrinsic_truth() -> None:
    truth = (0.02, -0.01, 0.015, 0.005, -0.004, 0.006)
    blocks: list[JointParameterBlock] = []
    factors = []
    for capture in range(4):
        pose_block = f"pose-{capture}"
        pose_initial = se3_from_tangent(
            (0.2 * capture, -0.1 * capture, 0.05 * capture, 0.01 * capture, 0.0, 0.0)
        )
        blocks.append(JointParameterBlock(pose_block, (0.0,) * 6))
        factors.append(
            make_joint_prior_factor(
                factor_id=f"prior-{capture}",
                observation_group=f"prior-{capture}",
                block=pose_block,
                target=(0.0,) * 6,
                sigma=(0.001,) * 3 + (0.0005,) * 3,
            )
        )
        for sample in range(24):
            phase = 0.31 * sample + 0.17 * capture
            point = (
                0.8 * math.sin(phase),
                0.6 * math.cos(0.7 * phase),
                1.2 + 0.2 * math.sin(1.3 * phase),
            )
            normal = (
                math.cos(0.9 * phase),
                math.sin(1.1 * phase),
                0.4 + math.cos(0.5 * phase),
            )
            plane_point = pose_initial.compose(se3_from_tangent(truth)).transform_point(
                point
            )
            measurement = JointPointToPlaneMeasurement(
                f"plane-{capture}-{sample}",
                f"capture-{capture}",
                point,
                plane_point,
                normal,
                pose_initial,
                SE3.identity(),
            )
            factors.append(
                make_joint_point_to_plane_factor(
                    measurement,
                    pose_block=pose_block,
                    extrinsic_block="extrinsic",
                )
            )
    blocks.append(
        JointParameterBlock(
            "extrinsic",
            (0.0,) * 6,
            known_bad_steps=(0.03,) * 3 + (math.radians(0.5),) * 3,
        )
    )

    result = BackendNeutralJointOptimizer().solve(
        blocks,
        factors,
        JointOptimizerOptions(holdout_ratio=0.25, split_seed=7, huber_delta=0.05),
    )

    assert result.status == "converged"
    assert result.optimized_values["extrinsic"] == pytest.approx(truth, abs=2e-5)
    assert result.information_rank == 30
    assert result.holdout_rmse is not None and result.holdout_rmse < 1e-5
    assert len(result.probes) == 12
    assert all(probe.detectable is True for probe in result.probes)
