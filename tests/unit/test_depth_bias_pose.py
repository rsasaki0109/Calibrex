import json

import numpy as np
from tools.depth_bias_pose import (
    bias_residual,
    biased_observations,
    identifiability,
    shifted,
    solve,
)
from tools.evaluate_depth_bias_pose import PRIOR, STARTS, attempt, outcome, reproduced, summarize
from tools.joint_metric_pose import depth_residual
from tools.joint_occlusion_pose import occlusion_residual
from tools.known_geometry_joint import CAMERA
from tools.occlusion_geometry import observations

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_depth_to_depth_solver import DepthToDepthObservation

IDENTITY = SE3((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def plane(slope: float) -> list[DepthToDepthObservation]:
    """Toy plane slope * x + z = 8 seen by coincident sensors; not an experiment scene."""
    _, u = np.indices((CAMERA.height, CAMERA.width))
    depth = 8.0 / (1 + slope * (u - CAMERA.cx) / CAMERA.fx)
    lv, lu = np.meshgrid(np.linspace(20.3, 170.3, 12), np.linspace(30.7, 290.7, 20), indexing="ij")
    a, b = (lu - CAMERA.cx) / CAMERA.fx, (lv - CAMERA.cy) / CAMERA.fy
    z = 8.0 / (1 + slope * a)
    points = np.stack([a * z, b * z, z], -1).reshape(-1, 3)
    return [DepthToDepthObservation("plane", depth, points, CAMERA, "metric_z")]


def test_zero_bias_reproduces_frozen_observations_and_bias_shifts_depth_only():
    for condition in ["exact_z_visible", "exact_z_all"]:
        frozen = observations("original", condition)
        for a, b in zip(frozen, biased_observations("original", condition, 0.0), strict=True):
            np.testing.assert_array_equal(a.depth_map, b.depth_map)
            np.testing.assert_array_equal(a.lidar_points, b.lidar_points)
        for a, b in zip(frozen, biased_observations("original", condition, -0.25), strict=True):
            np.testing.assert_array_equal(a.depth_map - 0.25, b.depth_map)
            np.testing.assert_array_equal(a.lidar_points, b.lidar_points)


def test_bias_variable_equals_correcting_the_depth_map():
    batch = shifted(plane(0.2), 0.3)
    frozen = np.concatenate([occlusion_residual(*depth_residual(o, IDENTITY)) for o in batch])
    np.testing.assert_array_equal(bias_residual(batch, IDENTITY, 0.0), frozen)
    corrected = np.concatenate(
        [occlusion_residual(*depth_residual(o, IDENTITY)) for o in shifted(batch, -0.3)]
    )
    np.testing.assert_allclose(bias_residual(batch, IDENTITY, 0.3), corrected, atol=1e-9)
    np.testing.assert_allclose(corrected, 0.0, atol=1e-5)  # bilinear error on a slant


def test_fronto_parallel_plane_cannot_separate_bias_from_depth_translation():
    report = identifiability(plane(0.0), IDENTITY, 0.0)
    assert report["unexplained_bias_fraction"] < 1e-6
    assert not report["distinguishable"]
    np.testing.assert_allclose(report["alias_translation_m_per_m"], [0, 0, 1], atol=1e-6)
    assert report["bias_sd_joint_per_unit_noise_m"] is None
    assert report["unit_column_condition"] is None
    assert report["residual_evaluations"] == 15
    assert report["rows"] == report["supported_rows"] == 240


def test_joint_solver_keeps_full_truth_and_accounts_calls():
    result = solve(shifted(plane(0.2), 0.1), IDENTITY, 0.1)
    assert result["converged"] and result["support_retained"]
    assert abs(result["bias_m"] - 0.1) < 1e-4
    assert 1 <= result["actual_residual_calls"] <= 800
    assert result["initial_support"] == result["final_support"] == [240]


def _records(distinguishable: bool, joint_pass: bool = True):
    key = {"scene": "s", "condition": "p"}
    metrics = {"recovered": True}
    fit = {"result": {"cost": 0.0}, "actual_residual_calls": 10, "metrics": metrics}
    pose_only, sensitivity, joint = [], [], []
    for bias in (0.0, 0.25, -0.25):
        for start in ("+1", "-1", "truth_pose"):
            ratio = 1.0 if bias else None
            pose_only.append(
                key | fit | {"bias_m": bias, "start": start, "absorption_ratio": ratio}
            )
            joint.append(
                key
                | fit
                | {
                    "bias_m": bias,
                    "start": start,
                    "metrics": {"recovered": joint_pass},
                    "bias_recovered": True,
                    "at_estimate": {"distinguishable": distinguishable, "residual_evaluations": 15},
                }
            )
        profile = [{"assumed_bias_m": b} | fit for b in (-0.25, 0.0, 0.25)]
        sensitivity.append(
            key
            | {
                "bias_m": bias,
                "at_reference": {"distinguishable": distinguishable, "residual_evaluations": 15},
                "profile": profile,
            }
        )
    return pose_only, sensitivity, joint


def test_summary_rules_are_strict():
    assert summarize(*_records(True), True)["verdict"] == "supported_synthetic_diagnostic"
    assert summarize(*_records(True), False)["verdict"] == "invalid_reproduction_failed"
    assert summarize(*_records(False), True)["verdict"] == "not_identifiable"
    summary = summarize(*_records(True, joint_pass=False), True)
    assert summary["verdict"] == "not_supported"
    assert len(summary["stage_c_zero_bias_regressions"]) == 3
    assert summary["stage_a_absorption_ratio"]["translation_absorbs_bias"]
    assert summary["actual_residual_calls"]["identifiability"] == 15 * 3 + 15 * 9


def test_reproduction_check_matches_every_frozen_zero_bias_result():
    prior = json.loads(PRIOR.read_text())
    labels = {sign: label for label, sign in STARTS.items()}
    rows = [
        {
            "scene": c["scene"],
            "condition": c["condition"],
            "bias_m": 0.0,
            "start": labels[c["sign"]],
        }
        | {k: c["methods"]["asymmetric"][k] for k in ("result", "metrics", "population")}
        for c in prior["cases"]
    ]
    assert reproduced(rows, prior)
    rows[-1] = rows[-1] | {"metrics": rows[-1]["metrics"] | {"recovered": False}}
    assert not reproduced(rows, prior)
    assert not reproduced(rows[:-1], prior)


def test_failed_fit_is_recorded_with_implied_call_count():
    def exhausted() -> dict:
        raise RuntimeError("hard residual budget exceeded")

    result, error, seconds = attempt(exhausted)
    row = outcome(plane(0.0), result, error, seconds, 0.0)
    assert row["result"] is None and row["actual_residual_calls"] == 801
    assert row["metrics"]["recovered"] is False and row["metrics"]["valid_output"] is False
