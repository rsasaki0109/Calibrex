"""Closure evidence of ``calibrex check``: loops over the estimates of a run."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from calibrex.check.closure import (
    ClosureEdge,
    _loop_steps,
    _spanning_edges,
    build_closure_report,
    closure_floor_verdict,
    edge_from_estimates,
    evaluate_loop,
    shared_inputs_of,
)
from calibrex.check.verdict import AxisEstimate, VerdictOptions
from calibrex.core.calibration_check import CheckPairRecord, CheckTransform

OPTIONS = VerdictOptions()
STD = (0.05, 0.05, 0.05)


def rot(rotvec_deg: tuple[float, float, float]) -> np.ndarray:
    return Rotation.from_rotvec(np.radians(rotvec_deg)).as_matrix()


def edge(
    pair: str,
    parent: str,
    child: str,
    rotation: np.ndarray,
    *,
    translation: tuple[float, float, float] | None = None,
    observed: tuple[bool, bool, bool] = (True, True, True),
    std: tuple[float, float, float] = STD,
    trans_std: float = 0.01,
) -> ClosureEdge:
    return ClosureEdge(
        pair=pair,
        parent=parent,
        child=child,
        rotation=rotation,
        rot_std_deg=std,
        rot_observed=observed,
        translation=None if translation is None else np.array(translation),
        trans_std_m=(trans_std,) * 3,
        trans_observed=(translation is not None,) * 3,
    )


def rig() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A consistent rig: ``R_vehicle_lidar``, ``R_vehicle_imu`` and ``R_imu_lidar``."""

    vehicle_lidar = rot((3.0, -2.0, 40.0))
    vehicle_imu = rot((-1.0, 0.5, 10.0))
    return vehicle_lidar, vehicle_imu, vehicle_imu.T @ vehicle_lidar


def run_loop(edges: list[ClosureEdge]):  # type: ignore[no-untyped-def]
    forest = _spanning_edges(edges)
    assert forest.count(False) == 1
    extra = forest.index(False)
    found = _loop_steps(edges, extra, forest)
    assert found is not None
    steps, frames = found
    return evaluate_loop(steps, frames, OPTIONS)


def test_consistent_parallel_edges_close() -> None:
    vehicle_lidar, _, _ = rig()
    loop = run_loop(
        [
            edge("lidar-vehicle", "base", "lidar", vehicle_lidar),
            edge("lidar-wheel_odometry", "base", "lidar", vehicle_lidar),
        ]
    )

    assert loop.kind == "parallel"
    assert loop.verdict == "pass"
    assert [axis.name for axis in loop.axes] == ["roll", "pitch", "yaw"]
    assert all(abs(axis.candidate_error) < 1e-9 for axis in loop.axes)
    # first-order std of the difference of two independent 0.05 deg axes
    assert (
        loop.axes[0].estimate_std == np.float64(0.05 * np.sqrt(2.0))
        or abs(loop.axes[0].estimate_std - 0.05 * np.sqrt(2.0)) < 1e-6
    )


def test_one_degree_error_on_one_edge_fails_that_axis() -> None:
    vehicle_lidar, _, _ = rig()
    bad = rot((0.0, 0.0, 1.0)) @ vehicle_lidar  # 1 deg about the base z axis
    edges = [
        edge("lidar-vehicle", "base", "lidar", vehicle_lidar),
        edge("lidar-wheel_odometry", "base", "lidar", bad),
    ]
    forest = _spanning_edges(edges)
    steps, frames = _loop_steps(edges, forest.index(False), forest)  # type: ignore[misc]
    strict = VerdictOptions(rotation_floor_deg=0.25)
    loop = evaluate_loop(steps, frames, strict)

    by_axis = {axis.name: axis for axis in loop.axes}
    assert loop.verdict == "fail"
    assert by_axis["yaw"].status == "fail"
    assert abs(abs(by_axis["yaw"].candidate_error) - 1.0) < 1e-6
    assert by_axis["roll"].status == "pass"
    # at the default floor 1 deg is warn: beyond 1x the 0.5 deg tolerance
    assert evaluate_loop(steps, frames, OPTIONS).verdict in {"warn", "fail"}


def test_triangle_cycle_closes_and_catches_one_edge() -> None:
    vehicle_lidar, vehicle_imu, imu_lidar = rig()

    def triangle(imu_lidar_rotation: np.ndarray) -> list[ClosureEdge]:
        return [
            edge("imu-lidar", "lidar", "imu", imu_lidar_rotation.T),  # T_lidar_imu
            edge("lidar-vehicle", "base", "lidar", vehicle_lidar),
            edge("imu-vehicle", "base", "imu", vehicle_imu),
        ]

    good = run_loop(triangle(imu_lidar))
    assert good.kind == "cycle"
    assert good.verdict == "pass"
    assert len(good.members) == 3
    assert all(abs(axis.candidate_error) < 1e-9 for axis in good.axes)

    # rotate the imu-lidar estimate by 1 deg about the lidar frame's z axis
    bad_rotation = (rot((0.0, 0.0, 1.0)) @ imu_lidar.T).T
    edges = triangle(bad_rotation)
    forest = _spanning_edges(edges)
    steps, frames = _loop_steps(edges, forest.index(False), forest)  # type: ignore[misc]
    bad = evaluate_loop(steps, frames, VerdictOptions(rotation_floor_deg=0.25))
    assert bad.verdict == "fail"
    magnitude = np.linalg.norm([axis.candidate_error for axis in bad.axes])
    assert abs(magnitude - 1.0) < 1e-6


def test_closure_std_matches_a_monte_carlo_with_translation() -> None:
    rng = np.random.default_rng(5)
    vehicle_lidar, vehicle_imu, imu_lidar = rig()
    t_vl, t_vi, t_il = np.array([1.0, 0.2, 0.7]), np.array([0.5, -0.1, 0.3]), None
    # T_imu_lidar translation making the loop consistent: T_v_l = T_v_i T_i_l
    t_il = vehicle_imu.T @ (t_vl - t_vi)
    sigma_deg, sigma_m = 0.3, 0.02

    def build(noise: bool) -> list[ClosureEdge]:
        def r(matrix: np.ndarray) -> np.ndarray:
            if not noise:
                return matrix
            return (
                Rotation.from_rotvec(np.radians(rng.normal(0, sigma_deg, 3))).as_matrix() @ matrix
            )

        def t(vector: np.ndarray) -> tuple[float, float, float]:
            value = vector + (rng.normal(0, sigma_m, 3) if noise else 0.0)
            return (float(value[0]), float(value[1]), float(value[2]))

        std = (sigma_deg,) * 3
        return [
            edge(
                "lidar-vehicle",
                "base",
                "lidar",
                r(vehicle_lidar),
                translation=t(t_vl),
                std=std,
                trans_std=sigma_m,
            ),
            edge(
                "imu-vehicle",
                "base",
                "imu",
                r(vehicle_imu),
                translation=t(t_vi),
                std=std,
                trans_std=sigma_m,
            ),
            # reverse member in the loop: T_lidar_imu = (T_imu_lidar)^-1
            edge(
                "imu-lidar",
                "imu",
                "lidar",
                r(imu_lidar),
                translation=t(t_il),
                std=std,
                trans_std=sigma_m,
            ),
        ]

    nominal = run_loop(build(False))
    assert all(abs(axis.candidate_error) < 1e-9 for axis in nominal.axes)
    assert [axis.name for axis in nominal.axes] == ["roll", "pitch", "yaw", "x", "y", "z"]
    predicted = np.array([axis.estimate_std for axis in nominal.axes])

    errors = np.array(
        [[axis.candidate_error for axis in run_loop(build(True)).axes] for _ in range(4000)]
    )
    empirical = errors.std(axis=0)
    np.testing.assert_allclose(empirical, predicted, rtol=0.1)


def test_unobserved_axis_leaves_the_closure_axis_unchecked() -> None:
    vehicle_lidar, _, _ = rig()
    loop = run_loop(
        [
            edge("lidar-vehicle", "base", "lidar", vehicle_lidar, observed=(False, True, True)),
            edge(
                "lidar-wheel_odometry", "base", "lidar", vehicle_lidar, observed=(False, True, True)
            ),
        ]
    )

    assert [axis.name for axis in loop.axes] == ["pitch", "yaw"]
    unchecked = {axis.name: axis for axis in loop.unchecked_axes}
    assert set(unchecked) == {"roll"}
    assert "does not observe its roll" in unchecked["roll"].reason
    assert "translation not closed" in " ".join(loop.notes)


def _record(
    pair: str, frames: list[str], parent: str, child: str, status: str = "pass"
) -> CheckPairRecord:
    return CheckPairRecord.model_validate(
        {
            "pair": pair,
            "frames": frames,
            "status": status,
            "compared_transform": CheckTransform(
                parent_frame=parent,
                child_frame=child,
                translation_m=[0.0, 0.0, 0.0],
                rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            ).model_dump(),
        }
    )


def test_report_marks_derived_and_unusable_estimates_and_floors_the_overall_verdict() -> None:
    vehicle_lidar, _, _ = rig()
    records = [
        _record("lidar-vehicle", ["lidar", "base"], "base", "lidar"),
        _record("gnss-imu", ["gnss", "imu"], "imu", "gnss"),
        _record("lidar-wheel_odometry", ["lidar", "base"], "base", "lidar"),
        _record("imu-vehicle", ["imu", "base"], "base", "imu", status="inconclusive"),
    ]
    bad = rot((0.0, 0.0, 2.0)) @ vehicle_lidar
    edges = {
        ("lidar-vehicle", ("lidar", "base")): edge("lidar-vehicle", "base", "lidar", vehicle_lidar),
        ("lidar-wheel_odometry", ("lidar", "base")): edge(
            "lidar-wheel_odometry", "base", "lidar", bad
        ),
    }

    report = build_closure_report(records, edges, OPTIONS)

    assert report is not None
    roles = {edge_.pair: edge_.role for edge_ in report.edges}
    assert roles == {
        "lidar-vehicle": "independent",
        "gnss-imu": "derived",
        "lidar-wheel_odometry": "independent",
        "imu-vehicle": "unusable",
    }
    assert len(report.loops) == 1
    assert report.verdict == "fail"
    assert closure_floor_verdict(report) == "warn"
    assert closure_floor_verdict(None) is None


def test_no_loop_without_two_estimates_of_a_cycle() -> None:
    vehicle_lidar, _, _ = rig()
    records = [_record("lidar-vehicle", ["lidar", "base"], "base", "lidar")]
    report = build_closure_report(
        records,
        {
            ("lidar-vehicle", ("lidar", "base")): edge(
                "lidar-vehicle", "base", "lidar", vehicle_lidar
            )
        },
        OPTIONS,
    )

    assert report is not None and report.loops == [] and report.verdict is None
    assert closure_floor_verdict(report) is None


def test_edge_from_estimates_inverts_the_candidate_error() -> None:
    estimate = rot((1.0, 2.0, 3.0))
    candidate = rot((0.2, -0.1, 0.4)) @ estimate
    error = Rotation.from_matrix(candidate @ estimate.T).as_rotvec()
    estimates = [
        AxisEstimate(name, "deg", float(np.degrees(error[i])), 0.1, True)  # type: ignore[arg-type]
        for i, name in enumerate(("roll", "pitch", "yaw"))
    ]
    compared = np.eye(4)
    compared[:3, :3] = candidate

    recovered = edge_from_estimates("lidar-vehicle", "base", "lidar", compared, estimates)

    np.testing.assert_allclose(recovered.rotation, estimate, atol=1e-12)
    assert recovered.translation is None


def test_members_sharing_an_input_are_noted() -> None:
    vehicle_lidar, _, _ = rig()
    shared = shared_inputs_of("lidar-vehicle", ["lidar", "base"])
    assert shared == ("lidar odometry of lidar",)
    first = edge("lidar-vehicle", "base", "lidar", vehicle_lidar)
    second = edge("lidar-wheel_odometry", "base", "lidar", vehicle_lidar)

    plain = run_loop([first, second])
    marked = run_loop(
        [
            replace(first, shared_inputs=shared),
            replace(
                second, shared_inputs=shared_inputs_of("lidar-wheel_odometry", ["lidar", "base"])
            ),
        ]
    )

    assert not any("share the" in note for note in plain.notes)
    assert any("share the lidar odometry of lidar" in note for note in marked.notes)
    assert shared_inputs_of("imu-lidar", ["imu", "lidar"]) == ()
