import math
from dataclasses import replace

import pytest

from calibrex.core.geometry import SE3
from calibrex.graph.joint_factors import (
    JointDepthPointToPlaneMeasurement,
    JointPointToPlaneMeasurement,
    JointRadarDopplerMeasurement,
    JointTrilinearDepthPointToPlaneMeasurement,
    make_joint_centered_trilinear_depth_point_to_plane_factor,
    make_joint_depth_point_to_plane_factor,
    make_joint_lattice_smoothness_factor,
    make_joint_lattice_zero_mean_factor,
    make_joint_point_to_plane_factor,
    make_joint_prior_factor,
    make_joint_radar_doppler_factor,
    make_joint_trilinear_depth_point_to_plane_factor,
    se3_from_tangent,
    trilinear_lattice_weights,
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


def test_depth_factor_couples_pose_extrinsic_scale_and_bias() -> None:
    pose = (0.1, -0.2, 0.05, 0.01, 0.0, -0.01)
    extrinsic = (0.02, -0.01, 0.03, 0.0, 0.01, 0.0)
    calibration = (math.log(1.02), -0.015)
    ray = (0.2, -0.1, 1.0)
    nominal_depth = 2.0
    corrected_depth = math.exp(calibration[0]) * nominal_depth + calibration[1]
    truth_point = (
        se3_from_tangent(pose)
        .compose(se3_from_tangent(extrinsic))
        .transform_point(tuple(value * corrected_depth for value in ray))
    )
    factor = make_joint_depth_point_to_plane_factor(
        JointDepthPointToPlaneMeasurement(
            "depth-plane",
            "capture",
            ray,
            nominal_depth,
            truth_point,
            (0.3, -0.4, 0.8),
            SE3.identity(),
            SE3.identity(),
        ),
        pose_block="pose",
        extrinsic_block="extrinsic",
        depth_calibration_block="depth",
    )

    assert factor.family == "rgbd_depth_point_to_plane"
    assert factor.residuals(
        {"pose": pose, "extrinsic": extrinsic, "depth": calibration}
    ) == pytest.approx((0.0,), abs=1e-12)
    assert (
        abs(factor.residuals({"pose": pose, "extrinsic": extrinsic, "depth": (0.0, 0.0)})[0]) > 1e-3
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


def test_trilinear_lattice_weights_and_depth_factor() -> None:
    weights = trilinear_lattice_weights(
        (0.5, 0.5, 1.5),
        minimum=(0.0, 0.0, 1.0),
        maximum=(1.0, 1.0, 2.0),
        shape=(2, 2, 2),
    )
    truth = tuple(0.01 * index for index in range(8))
    displacement = sum(weight * offset for weight, offset in zip(weights, truth, strict=True))
    factor = make_joint_trilinear_depth_point_to_plane_factor(
        JointTrilinearDepthPointToPlaneMeasurement(
            "lattice-depth",
            "capture",
            (0.0, 0.0, 1.0),
            1.5,
            weights,
            (0.0, 0.0, 1.5 + displacement),
            (0.0, 0.0, 1.0),
            SE3.identity(),
            SE3.identity(),
        ),
        pose_block="pose",
        extrinsic_block="extrinsic",
        depth_lattice_block="lattice",
    )

    assert weights == pytest.approx((0.125,) * 8)
    assert factor.residuals(
        {"pose": (0.0,) * 6, "extrinsic": (0.0,) * 6, "lattice": truth}
    ) == pytest.approx((0.0,), abs=1e-12)
    assert factor.family == "rgbd_trilinear_depth_point_to_plane"


def test_lattice_smoothness_uses_each_axis_neighbor_once() -> None:
    factor = make_joint_lattice_smoothness_factor(
        factor_id="smooth",
        observation_group="regularization",
        block="lattice",
        shape=(2, 2, 2),
        sigma_m=0.1,
    )

    residuals = factor.residuals({"lattice": tuple(float(index) for index in range(8))})
    assert len(residuals) == 12
    assert residuals[:3] == pytest.approx((10.0, 20.0, 40.0))
    assert factor.split_policy == "train_only"


def test_lattice_zero_mean_factor_is_train_only() -> None:
    factor = make_joint_lattice_zero_mean_factor(
        factor_id="center",
        observation_group="regularization",
        block="lattice",
        size=4,
        sigma_m=0.01,
    )

    assert factor.residuals({"lattice": (0.01, -0.02, 0.03, -0.02)}) == pytest.approx((0.0,))
    assert factor.residuals({"lattice": (0.01,) * 4}) == pytest.approx((1.0,))
    assert factor.split_policy == "train_only"


def test_se3_tangent_requires_six_values() -> None:
    with pytest.raises(ValueError, match="six"):
        se3_from_tangent((0.0, 0.0))


def test_multicapture_pose_prior_problem_recovers_shared_extrinsic_truth() -> None:
    truth = (0.02, -0.01, 0.015, 0.005, -0.004, 0.006)
    depth_truth = (math.log(1.03), -0.02)
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
            corrected_depth = math.exp(depth_truth[0]) * point[2] + depth_truth[1]
            corrected_point = (
                point[0] / point[2] * corrected_depth,
                point[1] / point[2] * corrected_depth,
                corrected_depth,
            )
            plane_point = pose_initial.compose(se3_from_tangent(truth)).transform_point(
                corrected_point
            )
            measurement = JointDepthPointToPlaneMeasurement(
                f"plane-{capture}-{sample}",
                f"capture-{capture}",
                (point[0] / point[2], point[1] / point[2], 1.0),
                point[2],
                plane_point,
                normal,
                pose_initial,
                SE3.identity(),
            )
            factors.append(
                make_joint_depth_point_to_plane_factor(
                    measurement,
                    pose_block=pose_block,
                    extrinsic_block="extrinsic",
                    depth_calibration_block="depth",
                )
            )
    blocks.append(
        JointParameterBlock(
            "extrinsic",
            (0.0,) * 6,
            known_bad_steps=(0.03,) * 3 + (math.radians(0.5),) * 3,
        )
    )
    blocks.append(
        JointParameterBlock(
            "depth",
            (0.0, 0.0),
            known_bad_steps=(0.02, 0.02),
        )
    )

    result = BackendNeutralJointOptimizer().solve(
        blocks,
        factors,
        JointOptimizerOptions(holdout_ratio=0.25, split_seed=7, huber_delta=0.05),
    )

    assert result.status == "converged"
    assert result.optimized_values["extrinsic"] == pytest.approx(truth, abs=2e-5)
    assert result.optimized_values["depth"] == pytest.approx(depth_truth, abs=2e-5)
    assert result.information_rank == 32
    assert result.holdout_rmse is not None and result.holdout_rmse < 1e-5
    assert len(result.probes) == 16
    assert all(probe.detectable is True for probe in result.probes)


def test_trilinear_depth_lattice_recovers_synthetic_control_truth() -> None:
    shape = (2, 2, 2)
    minimum = (-0.5, -0.4, 1.0)
    maximum = (0.5, 0.4, 2.0)
    bias_truth = 0.04
    truth = (-0.03, 0.01, 0.02, -0.01, 0.025, -0.015, 0.03, -0.03)
    assert sum(truth) == pytest.approx(0.0)
    blocks = [
        JointParameterBlock("pose", (0.0,) * 6, fixed=True),
        JointParameterBlock("extrinsic", (0.0,) * 6, fixed=True),
        JointParameterBlock("bias", (0.0,), known_bad_steps=(0.02,)),
        JointParameterBlock("lattice", (0.0,) * 8, known_bad_steps=(0.02,) * 8),
    ]
    factors = []
    sample_index = 0
    for z_fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        for y_fraction in (0.0, 0.5, 1.0):
            for x_fraction in (0.0, 0.5, 1.0):
                point = (
                    minimum[0] + x_fraction * (maximum[0] - minimum[0]),
                    minimum[1] + y_fraction * (maximum[1] - minimum[1]),
                    minimum[2] + z_fraction * (maximum[2] - minimum[2]),
                )
                weights = trilinear_lattice_weights(
                    point, minimum=minimum, maximum=maximum, shape=shape
                )
                displacement = bias_truth + sum(
                    weight * offset for weight, offset in zip(weights, truth, strict=True)
                )
                ray = (point[0] / point[2], point[1] / point[2], 1.0)
                corrected = tuple(value * (point[2] + displacement) for value in ray)
                factors.append(
                    make_joint_centered_trilinear_depth_point_to_plane_factor(
                        JointTrilinearDepthPointToPlaneMeasurement(
                            f"lattice-{sample_index:03d}",
                            f"capture-{sample_index % 5}",
                            ray,
                            point[2],
                            weights,
                            corrected,
                            (0.2 + x_fraction, 0.3 + y_fraction, 1.0),
                            SE3.identity(),
                            SE3.identity(),
                        ),
                        pose_block="pose",
                        extrinsic_block="extrinsic",
                        depth_bias_block="bias",
                        centered_lattice_block="lattice",
                    )
                )
                sample_index += 1
    factors.append(
        make_joint_lattice_zero_mean_factor(
            factor_id="lattice-center",
            observation_group="regularization",
            block="lattice",
            size=8,
            sigma_m=1.0e-4,
        )
    )

    result = BackendNeutralJointOptimizer().solve(
        blocks,
        factors,
        JointOptimizerOptions(holdout_ratio=0.2, split_seed=11),
    )

    assert result.status == "converged"
    assert result.optimized_values["bias"] == pytest.approx((bias_truth,), abs=1e-7)
    assert result.optimized_values["lattice"] == pytest.approx(truth, abs=1e-7)
    assert result.information_rank == 9
    assert result.holdout_rmse is not None and result.holdout_rmse < 1e-8
    assert len(result.probes) == 18
    assert all(probe.detectable is True for probe in result.probes)
