from __future__ import annotations

import importlib.util
import math

import pytest

from calibrex.core.geometry import SE3
from calibrex.solvers.opencv_hand_eye_adapters import (
    OpenCvHandEyeAdapter,
    OpenCvRobotWorldHandEyeAdapter,
)
from calibrex.solvers.park_martin_hand_eye_solver import HandEyeMotionPair
from calibrex.solvers.shah_robot_world_hand_eye_solver import (
    RobotWorldHandEyePosePair,
    evaluate_robot_world_hand_eye_poses,
)
from calibrex.solvers.tsai_lenz_hand_eye_solver import evaluate_hand_eye_motions

requires_opencv = pytest.mark.skipif(
    importlib.util.find_spec("cv2") is None,
    reason="opencv-python-headless optional dependency is not installed",
)


def _pose(
    axis: tuple[float, float, float],
    angle_deg: float,
    translation: tuple[float, float, float],
) -> SE3:
    norm = math.sqrt(sum(value * value for value in axis))
    unit = tuple(value / norm for value in axis)
    half = math.radians(angle_deg) / 2.0
    return SE3(
        translation,
        (
            unit[0] * math.sin(half),
            unit[1] * math.sin(half),
            unit[2] * math.sin(half),
            math.cos(half),
        ),
    )


def _problem() -> tuple[
    list[RobotWorldHandEyePosePair],
    list[HandEyeMotionPair],
]:
    transform_x = _pose((1.0, -2.0, 0.5), 23.0, (0.24, -0.16, 0.11))
    transform_y = _pose((-0.5, 1.0, 2.0), -31.0, (-0.3, 0.5, 0.2))
    definitions = (
        ((1.0, 0.0, 0.0), 18.0, (0.2, -0.1, 0.05)),
        ((0.0, 1.0, 0.0), -25.0, (-0.1, 0.15, 0.03)),
        ((0.0, 0.0, 1.0), 32.0, (0.04, 0.08, -0.12)),
        ((1.0, 1.0, 0.0), -40.0, (-0.05, 0.02, 0.1)),
        ((0.0, 1.0, 1.0), 48.0, (0.12, -0.04, 0.07)),
        ((1.0, 0.0, 1.0), -21.0, (-0.09, 0.03, 0.02)),
        ((1.0, -1.0, 0.5), 55.0, (0.03, 0.11, -0.06)),
        ((-0.5, 1.0, 1.0), 37.0, (-0.02, -0.07, 0.09)),
    )
    poses: list[RobotWorldHandEyePosePair] = []
    for index, (axis, angle, translation) in enumerate(definitions):
        pose_a = _pose(axis, angle, translation)
        pose_b = transform_y.inverse().compose(pose_a).compose(transform_x)
        poses.append(RobotWorldHandEyePosePair(f"pose-{index:03d}", pose_a, pose_b))
    first = poses[0]
    motions = [
        HandEyeMotionPair(
            pair_id=pose.pair_id,
            motion_a=first.pose_a.inverse().compose(pose.pose_a),
            motion_b=first.pose_b.inverse().compose(pose.pose_b),
        )
        for pose in poses[1:]
    ]
    return poses, motions


@pytest.mark.parametrize("method", ["tsai", "park", "horaud", "andreff", "daniilidis"])
@requires_opencv
def test_opencv_hand_eye_methods_follow_calibrex_ax_xb_convention(method: str) -> None:
    poses, motions = _problem()

    result = OpenCvHandEyeAdapter().solve(poses, method)  # type: ignore[arg-type]

    assert result.status == "converged"
    assert result.transform_x is not None
    evaluation = evaluate_hand_eye_motions(motions, result.transform_x)
    assert evaluation.rotation_closure_rmse_deg is not None
    assert evaluation.rotation_closure_rmse_deg < 1.0e-5
    assert evaluation.translation_closure_rmse_m is not None
    assert evaluation.translation_closure_rmse_m < 1.0e-9
    assert result.as_dict()["tool"]["license_spdx"] == "Apache-2.0"  # type: ignore[index]


@pytest.mark.parametrize("method", ["shah", "li"])
@requires_opencv
def test_opencv_robot_world_methods_follow_calibrex_ax_yb_convention(
    method: str,
) -> None:
    poses, _motions = _problem()

    result = OpenCvRobotWorldHandEyeAdapter().solve(poses, method)  # type: ignore[arg-type]

    assert result.status == "converged"
    assert result.transform_x is not None
    assert result.transform_y is not None
    evaluation = evaluate_robot_world_hand_eye_poses(
        poses,
        result.transform_x,
        result.transform_y,
    )
    assert evaluation.rotation_closure_rmse_deg is not None
    assert evaluation.rotation_closure_rmse_deg < 1.0e-5
    assert evaluation.translation_closure_rmse_m is not None
    assert evaluation.translation_closure_rmse_m < 1.0e-9
    assert result.as_dict()["paper_doi"] is not None


def test_opencv_adapter_reports_unavailable_without_optional_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    poses, _motions = _problem()

    def missing_opencv(name: str):
        assert name == "cv2"
        raise ImportError

    monkeypatch.setattr(
        "calibrex.solvers.opencv_hand_eye_adapters.importlib.import_module",
        missing_opencv,
    )

    hand_eye = OpenCvHandEyeAdapter().solve(poses, "park")
    robot_world = OpenCvRobotWorldHandEyeAdapter().solve(poses, "shah")

    assert hand_eye.status == "unavailable"
    assert hand_eye.as_dict()["tool"]["external_code_executed"] is False  # type: ignore[index]
    assert robot_world.status == "unavailable"
    assert robot_world.as_dict()["tool"]["external_code_executed"] is False  # type: ignore[index]
