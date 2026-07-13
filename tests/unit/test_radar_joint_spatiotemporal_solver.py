import math

import pytest

from calibrex.core.geometry import SE3, quaternion_conjugate_xyzw, rotate_vector_xyzw
from calibrex.evaluation.radar_spatiotemporal import (
    NuScenesRadarSpatiotemporalEvidence,
    _evidence_metrics,
)
from calibrex.solvers import RadarJointSpatiotemporalSolver as PublicSolver
from calibrex.solvers.radar_joint_spatiotemporal_solver import (
    RadarJointSpatiotemporalOptions,
    RadarJointSpatiotemporalSolver,
)
from calibrex.solvers.radar_spatiotemporal_lever_arm_solver import (
    RadarSpatiotemporalLeverArmOptions,
    RadarSpatiotemporalLeverArmSolver,
    RadarVelocityMeasurement,
    ReferenceKinematicSample,
)


def _cross(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _reference() -> list[ReferenceKinematicSample]:
    output = []
    for index in range(251):
        time = index * 0.02
        output.append(
            ReferenceKinematicSample(
                time,
                (
                    2.0 + 0.3 * time + 0.1 * time * time,
                    math.sin(0.8 * time) + 0.2 * time,
                    math.cos(0.6 * time) + 0.1 * time * time,
                ),
                (
                    0.4 + 0.2 * math.sin(0.7 * time),
                    -0.3 + 0.15 * math.cos(0.5 * time),
                    0.5 + 0.1 * time,
                ),
            )
        )
    return output


def _measurements(
    reference: list[ReferenceKinematicSample], truth: SE3, offset: float
) -> list[RadarVelocityMeasurement]:
    by_index = {round(item.timestamp_sec / 0.02): item for item in reference}
    output = []
    for index in range(6, 57):
        timestamp = index * 0.08
        sample = by_index[round((timestamp + offset) / 0.02)]
        lever = _cross(sample.angular_velocity_body_radps, truth.translation_m)
        body_velocity = tuple(
            sample.linear_velocity_body_mps[axis] + lever[axis] for axis in range(3)
        )
        radar_velocity = rotate_vector_xyzw(
            quaternion_conjugate_xyzw(truth.rotation_quat_xyzw),
            body_velocity,  # type: ignore[arg-type]
        )
        output.append(RadarVelocityMeasurement(f"scan-{index:03d}", timestamp, radar_velocity))
    return output


def _rotation_error_deg(left: SE3, right: SE3) -> float:
    delta = left.inverse().compose(right)
    return math.degrees(2.0 * math.acos(min(1.0, abs(delta.rotation_quat_xyzw[3]))))


def test_joint_radar_solver_recovers_seven_dof_truth_and_probes() -> None:
    assert PublicSolver is RadarJointSpatiotemporalSolver
    truth = SE3.from_lists(
        [0.32, -0.18, 0.11],
        [0.061917, -0.037150, 0.049534, 0.996153],
    )
    reference = _reference()
    measurements = _measurements(reference, truth, 0.02)
    options = RadarJointSpatiotemporalOptions(
        holdout_ratio=0.25,
        split_seed=9,
        known_bad_margin_mps=1.0e-4,
    )

    result = RadarJointSpatiotemporalSolver().solve(
        measurements, reference, SE3.identity(), options
    )

    assert result.status == "converged"
    assert result.transform_body_radar is not None
    assert math.dist(result.transform_body_radar.translation_m, truth.translation_m) < 1.0e-5
    assert _rotation_error_deg(result.transform_body_radar, truth) < 1.0e-4
    assert result.time_offset_sec == pytest.approx(0.02, abs=1.0e-6)
    assert result.information_rank == 7
    assert result.holdout_rmse_mps is not None and result.holdout_rmse_mps < 1.0e-6
    assert len(result.probes) == 14
    assert all(probe.detectable is True for probe in result.probes)
    assert set(result.train_measurement_ids).isdisjoint(result.holdout_measurement_ids)
    serialized = result.as_dict()
    assert serialized["primary_reference"]["doi"] == "10.1109/TRO.2023.3311680"  # type: ignore[index]
    assert serialized["time_convention"] == "radar_time + dt_radar = reference_time"
    assert serialized["initial_transform_body_radar"] == SE3.identity().as_dict()

    staged = RadarSpatiotemporalLeverArmSolver().solve(
        measurements,
        reference,
        truth.rotation_quat_xyzw,
        RadarSpatiotemporalLeverArmOptions(
            holdout_ratio=0.25,
            split_seed=9,
            min_train_measurements=12,
        ),
    )
    evidence = NuScenesRadarSpatiotemporalEvidence(
        "scored",
        result.reason,
        staged,
        None,
        result,
        "SYNTHETIC_RADAR",
        truth.rotation_quat_xyzw,
        truth.translation_m,
        len(measurements),
        len(reference),
        0,
        (),
    )
    metrics = _evidence_metrics(evidence)
    assert metrics["radar_joint_spatiotemporal_information_rank"].value == 7.0
    assert metrics["radar_joint_spatiotemporal_known_bad_detectable_fraction"].value == 1.0
    assert metrics["radar_joint_spatiotemporal_common_split_consistent"].value == 1.0


def test_joint_radar_solver_reports_constant_motion_degeneracy() -> None:
    reference = [
        ReferenceKinematicSample(index * 0.1, (4.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        for index in range(50)
    ]
    measurements = [
        RadarVelocityMeasurement(f"scan-{index}", 0.3 + index * 0.1, (4.0, 0.0, 0.0))
        for index in range(35)
    ]

    result = RadarJointSpatiotemporalSolver().solve(measurements, reference, SE3.identity())

    assert result.status == "degenerate_motion"
    assert result.information_rank < 7
    assert set(result.weak_parameter_blocks) == {
        "radar_extrinsic_correction",
        "radar_time_latent",
    }


def test_joint_radar_solver_rejects_gross_initial_outliers_before_split() -> None:
    truth = SE3.from_lists(
        [0.32, -0.18, 0.11],
        [0.061917, -0.037150, 0.049534, 0.996153],
    )
    reference = _reference()
    measurements = _measurements(reference, truth, 0.02)
    rejected = {measurements[index].measurement_id for index in (4, 13, 29, 44)}
    for index in (4, 13, 29, 44):
        item = measurements[index]
        velocity = item.velocity_radar_mps
        measurements[index] = RadarVelocityMeasurement(
            item.measurement_id,
            item.timestamp_sec,
            (velocity[0] + 10.0, velocity[1] - 8.0, velocity[2] + 6.0),
        )

    result = RadarJointSpatiotemporalSolver().solve(
        measurements,
        reference,
        SE3.identity(),
        RadarJointSpatiotemporalOptions(holdout_ratio=0.2, split_seed=4),
    )

    assert result.status == "converged"
    assert set(result.rejected_initial_residual_ids) == rejected
    assert result.eligible_measurement_count == len(measurements) - len(rejected)
    assert result.transform_body_radar is not None
    assert math.dist(result.transform_body_radar.translation_m, truth.translation_m) < 1.0e-5
    assert result.time_offset_sec == pytest.approx(0.02, abs=1.0e-6)


def test_joint_radar_solver_rejects_duplicate_lineage_and_invalid_time_bound() -> None:
    measurement = RadarVelocityMeasurement("same", 1.0, (1.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="IDs must be unique"):
        RadarJointSpatiotemporalSolver().solve([measurement, measurement], [], SE3.identity())
    with pytest.raises(ValueError, match="strictly inside"):
        RadarJointSpatiotemporalSolver().solve(
            [],
            [],
            SE3.identity(),
            RadarJointSpatiotemporalOptions(initial_time_offset_sec=0.1),
        )
