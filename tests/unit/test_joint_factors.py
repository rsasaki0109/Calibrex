import math
from dataclasses import replace

import pytest

from calibrex.core.geometry import SE3, rotate_vector_xyzw
from calibrex.graph.joint_factors import (
    JointDepthPointToPlaneMeasurement,
    JointPointToPlaneMeasurement,
    JointRadarDopplerMeasurement,
    JointTrilinearDepthPointToPlaneMeasurement,
    JointTrilinearXYZPairMeasurement,
    JointTrilinearXYZPointToPlaneMeasurement,
    estimate_xyz_lattice_local_rotations,
    make_joint_centered_trilinear_depth_point_to_plane_factor,
    make_joint_depth_point_to_plane_factor,
    make_joint_lattice_smoothness_factor,
    make_joint_lattice_zero_mean_factor,
    make_joint_point_to_plane_factor,
    make_joint_prior_factor,
    make_joint_radar_doppler_factor,
    make_joint_trilinear_depth_point_to_plane_factor,
    make_joint_trilinear_xyz_pair_factor,
    make_joint_trilinear_xyz_point_to_plane_factor,
    make_joint_xyz_lattice_rigid_gauge_factor,
    make_joint_xyz_lattice_shape_factor,
    regular_lattice_control_points,
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
    assert factor.sqrt_information == ((2.0, 0.0), (0.0, 0.5))


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


def test_trilinear_xyz_factor_and_shape_preserving_regularizer() -> None:
    shape = (2, 2, 2)
    controls = regular_lattice_control_points(
        minimum=(0.0, 0.0, 1.0), maximum=(1.0, 1.0, 2.0), shape=shape
    )
    point = (0.5, 0.5, 1.5)
    weights = trilinear_lattice_weights(
        point, minimum=(0.0, 0.0, 1.0), maximum=(1.0, 1.0, 2.0), shape=shape
    )
    translation = (0.03, -0.02, 0.01)
    offsets = translation * len(controls)
    factor = make_joint_trilinear_xyz_point_to_plane_factor(
        JointTrilinearXYZPointToPlaneMeasurement(
            "xyz-alignment",
            "capture",
            point,
            weights,
            tuple(point[axis] + translation[axis] for axis in range(3)),
            (0.2, -0.3, 0.9),
            SE3.identity(),
            SE3.identity(),
        ),
        pose_block="pose",
        extrinsic_block="extrinsic",
        xyz_lattice_block="xyz-lattice",
    )
    regularizer = make_joint_xyz_lattice_shape_factor(
        factor_id="xyz-shape",
        observation_group="regularization",
        block="xyz-lattice",
        control_points_m=controls,
        local_rotations_xyzw=((0.0, 0.0, 0.0, 1.0),) * len(controls),
        shape=shape,
        sigma_m=0.1,
    )

    values = {
        "pose": (0.0,) * 6,
        "extrinsic": (0.0,) * 6,
        "xyz-lattice": offsets,
    }
    assert factor.residuals(values) == pytest.approx((0.0,), abs=1e-12)
    assert factor.family == "rgbd_trilinear_xyz_point_to_plane"
    assert regularizer.residuals(values) == pytest.approx((0.0,) * 72, abs=1e-12)
    assert regularizer.split_policy == "train_only"
    distorted = (*offsets[:-1], offsets[-1] + 0.02)
    assert max(abs(value) for value in regularizer.residuals({"xyz-lattice": distorted})) > 0.1


def test_trilinear_xyz_pair_factor_calibrates_both_correspondence_sides() -> None:
    shape = (2, 2, 2)
    minimum = (0.0, 0.0, 1.0)
    maximum = (1.0, 1.0, 2.0)
    source = (0.0, 0.0, 1.0)
    target = (1.0, 0.0, 1.0)
    source_weights = trilinear_lattice_weights(
        source, minimum=minimum, maximum=maximum, shape=shape
    )
    target_weights = trilinear_lattice_weights(
        target, minimum=minimum, maximum=maximum, shape=shape
    )
    offsets = (0.03, -0.02, 0.01) * 8
    factor = make_joint_trilinear_xyz_pair_factor(
        JointTrilinearXYZPairMeasurement(
            "xyz-pair",
            "capture-pair",
            source,
            target,
            source_weights,
            target_weights,
            (1.0, 0.0, 0.0),
            SE3.identity(),
            SE3((-1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        ),
        source_pose_block="source-pose",
        target_pose_block="target-pose",
        xyz_lattice_block="xyz-lattice",
    )

    values = {
        "source-pose": (0.0,) * 6,
        "target-pose": (0.0,) * 6,
        "xyz-lattice": offsets,
    }
    assert factor.residuals(values) == pytest.approx((0.0,), abs=1e-12)
    assert factor.residuals({**values, "source-pose": (0.01, 0.0, 0.0, 0.0, 0.0, 0.0)}) == (
        pytest.approx(0.01),
    )
    assert factor.family == "rgbd_trilinear_xyz_pair_point_to_plane"


def test_trilinear_xyz_pair_graph_recovers_synthetic_field_truth() -> None:
    shape = (2, 2, 2)
    minimum = (-0.5, -0.4, 1.0)
    maximum = (0.5, 0.4, 2.0)
    controls = regular_lattice_control_points(minimum=minimum, maximum=maximum, shape=shape)
    truth = tuple(
        component
        for index in range(len(controls))
        for component in (
            0.002 * (index - 3.5),
            -0.003 * (index % 3 - 1),
            0.0025 * ((index + 1) % 4 - 1.5),
        )
    )
    blocks = (
        JointParameterBlock("source-pose", (0.0,) * 6, fixed=True),
        JointParameterBlock("target-pose", (0.0,) * 6, fixed=True),
        JointParameterBlock(
            "xyz-lattice",
            (0.0,) * len(truth),
            known_bad_steps=(0.02,) * len(truth),
        ),
    )
    axis_geometry = (
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)),
        ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0)),
    )
    factors = []
    for replicate in range(5):
        for index, point in enumerate(controls):
            weights = trilinear_lattice_weights(
                point, minimum=minimum, maximum=maximum, shape=shape
            )
            calibrated = tuple(point[axis] + truth[3 * index + axis] for axis in range(3))
            for axis, (normal, target_rotation) in enumerate(axis_geometry):
                rotated = rotate_vector_xyzw(target_rotation, calibrated)
                target_translation = tuple(
                    calibrated[component] - rotated[component] for component in range(3)
                )
                factors.append(
                    make_joint_trilinear_xyz_pair_factor(
                        JointTrilinearXYZPairMeasurement(
                            f"xyz-pair-{replicate}-{index}-{axis}",
                            f"capture-set-{replicate}",
                            point,
                            point,
                            weights,
                            weights,
                            normal,
                            SE3.identity(),
                            SE3(target_translation, target_rotation),
                        ),
                        source_pose_block="source-pose",
                        target_pose_block="target-pose",
                        xyz_lattice_block="xyz-lattice",
                    )
                )

    result = BackendNeutralJointOptimizer().solve(
        blocks,
        factors,
        JointOptimizerOptions(holdout_ratio=0.2, split_seed=7),
    )

    assert result.status == "converged"
    assert result.optimized_values["xyz-lattice"] == pytest.approx(truth, abs=1e-8)
    assert result.information_rank == 24
    assert result.holdout_rmse is not None and result.holdout_rmse < 1e-9
    assert len(result.probes) == 48
    assert all(probe.detectable is True for probe in result.probes)


def test_xyz_lattice_rigid_gauge_separates_translation_and_rotation() -> None:
    controls = regular_lattice_control_points(
        minimum=(-1.0, -1.0, -1.0), maximum=(1.0, 1.0, 1.0), shape=(2, 2, 2)
    )
    gauge = make_joint_xyz_lattice_rigid_gauge_factor(
        factor_id="xyz-gauge",
        observation_group="regularization",
        block="xyz-lattice",
        control_points_m=controls,
        translation_sigma_m=0.1,
        rotation_sigma_rad=0.1,
    )
    translation = (0.03, -0.02, 0.01) * len(controls)
    translated = gauge.residuals({"xyz-lattice": translation})
    assert translated[:3] == pytest.approx((0.3, -0.2, 0.1))
    assert translated[3:] == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)

    omega = (0.0, 0.0, 0.01)
    rotational = tuple(
        component
        for point in controls
        for component in (
            -omega[2] * point[1],
            omega[2] * point[0],
            0.0,
        )
    )
    rotated = gauge.residuals({"xyz-lattice": rotational})
    assert rotated[:3] == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    assert rotated[3:] == pytest.approx((0.0, 0.0, 0.1), abs=1e-12)
    assert gauge.split_policy == "train_only"


def test_xyz_lattice_local_rotation_fit_recovers_rigid_field() -> None:
    shape = (2, 2, 2)
    controls = regular_lattice_control_points(
        minimum=(-1.0, -0.5, 1.0), maximum=(1.0, 0.5, 2.0), shape=shape
    )
    angle = 0.12
    cosine = math.cos(angle)
    sine = math.sin(angle)
    translation = (0.03, -0.02, 0.01)
    offsets = tuple(
        component
        for x, y, z in controls
        for component in (
            cosine * x - sine * y + translation[0] - x,
            sine * x + cosine * y + translation[1] - y,
            translation[2],
        )
    )

    rotations = estimate_xyz_lattice_local_rotations(
        control_points_m=controls, offsets_m=offsets, shape=shape
    )

    expected = (cosine, sine, 0.0)
    assert len(rotations) == 8
    assert all(
        rotate_vector_xyzw(rotation, (1.0, 0.0, 0.0)) == pytest.approx(expected, abs=1e-12)
        for rotation in rotations
    )


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


def test_trilinear_xyz_lattice_recovers_synthetic_truth_and_probes() -> None:
    shape = (2, 2, 2)
    minimum = (-0.5, -0.4, 1.0)
    maximum = (0.5, 0.4, 2.0)
    controls = regular_lattice_control_points(minimum=minimum, maximum=maximum, shape=shape)
    truth = tuple(
        component
        for index in range(len(controls))
        for component in (
            0.003 * (index - 3.5),
            -0.002 * (index % 3 - 1),
            0.004 * ((index + 1) % 4 - 1.5),
        )
    )
    blocks = (
        JointParameterBlock("pose", (0.0,) * 6, fixed=True),
        JointParameterBlock("extrinsic", (0.0,) * 6, fixed=True),
        JointParameterBlock(
            "xyz-lattice", (0.0,) * len(truth), known_bad_steps=(0.02,) * len(truth)
        ),
    )
    factors = []
    normals = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    for capture in range(5):
        for index, point in enumerate(controls):
            weights = trilinear_lattice_weights(
                point, minimum=minimum, maximum=maximum, shape=shape
            )
            calibrated = tuple(point[axis] + truth[3 * index + axis] for axis in range(3))
            for axis, normal in enumerate(normals):
                factors.append(
                    make_joint_trilinear_xyz_point_to_plane_factor(
                        JointTrilinearXYZPointToPlaneMeasurement(
                            f"xyz-{capture}-{index}-{axis}",
                            f"capture-{capture}",
                            point,
                            weights,
                            calibrated,
                            normal,
                            SE3.identity(),
                            SE3.identity(),
                        ),
                        pose_block="pose",
                        extrinsic_block="extrinsic",
                        xyz_lattice_block="xyz-lattice",
                    )
                )

    result = BackendNeutralJointOptimizer().solve(
        blocks,
        factors,
        JointOptimizerOptions(holdout_ratio=0.2, split_seed=13),
    )

    assert result.status == "converged"
    assert result.optimized_values["xyz-lattice"] == pytest.approx(truth, abs=1e-8)
    assert result.information_rank == 24
    assert result.holdout_rmse is not None and result.holdout_rmse < 1e-9
    assert len(result.probes) == 48
    assert all(probe.detectable is True for probe in result.probes)
