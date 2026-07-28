"""Optional OpenCV hand-eye and robot-world/hand-eye adapter boundaries.

The adapter imports OpenCV only when executed. OpenCV receives the same typed
absolute pose pairs as the native robot-world solvers; evaluation and holdout
splitting remain Calibrex responsibilities.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3, quaternion_xyzw_from_rotation_matrix
from calibrex.solvers.shah_robot_world_hand_eye_solver import (
    RobotWorldHandEyePosePair,
)

OpenCvHandEyeMethod = Literal[
    "tsai",
    "park",
    "horaud",
    "andreff",
    "daniilidis",
]
OpenCvRobotWorldHandEyeMethod = Literal["shah", "li"]
OpenCvAdapterStatus = Literal["converged", "unavailable", "failed", "insufficient_poses"]
FloatArray: TypeAlias = NDArray[np.float64]

OPENCV_LICENSE_SPDX = "Apache-2.0"
OPENCV_SOURCE_URL = "https://github.com/opencv/opencv"
OPENCV_API_REFERENCE = "https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html"

_HAND_EYE_CONSTANTS: dict[OpenCvHandEyeMethod, str] = {
    "tsai": "CALIB_HAND_EYE_TSAI",
    "park": "CALIB_HAND_EYE_PARK",
    "horaud": "CALIB_HAND_EYE_HORAUD",
    "andreff": "CALIB_HAND_EYE_ANDREFF",
    "daniilidis": "CALIB_HAND_EYE_DANIILIDIS",
}
_ROBOT_WORLD_CONSTANTS: dict[OpenCvRobotWorldHandEyeMethod, str] = {
    "shah": "CALIB_ROBOT_WORLD_HAND_EYE_SHAH",
    "li": "CALIB_ROBOT_WORLD_HAND_EYE_LI",
}
_ROBOT_WORLD_DOIS: dict[OpenCvRobotWorldHandEyeMethod, str] = {
    "shah": "10.1115/1.4024473",
    "li": "10.5897/IJPS.9000501",
}


@dataclass(frozen=True)
class OpenCvHandEyeAdapterResult:
    """Result of one optional OpenCV ``AX=XB`` execution."""

    status: OpenCvAdapterStatus
    method: OpenCvHandEyeMethod
    transform_x: SE3 | None
    reason: str
    opencv_version: str | None

    def as_dict(self) -> dict[str, Any]:
        """Return machine-readable method and license provenance."""

        return {
            "status": self.status,
            "method": f"opencv_calibrateHandEye/{self.method}",
            "transform_x": _transform_dict(self.transform_x),
            "reason": self.reason,
            "tool": {
                "name": "opencv",
                "version": self.opencv_version,
                "source_url": OPENCV_SOURCE_URL,
                "api_reference": OPENCV_API_REFERENCE,
                "license_spdx": OPENCV_LICENSE_SPDX,
                "execution_mode": "optional_import",
                "external_code_executed": self.status != "unavailable",
            },
            "input_contract": {
                "gripper2base": "pose_a",
                "target2cam": "inverse(pose_b)",
                "output": "transform_x",
                "equation": "pose_a_relative X = X pose_b_relative",
            },
        }


@dataclass(frozen=True)
class OpenCvRobotWorldHandEyeAdapterResult:
    """Result of one optional OpenCV ``AX=YB`` execution."""

    status: OpenCvAdapterStatus
    method: OpenCvRobotWorldHandEyeMethod
    transform_x: SE3 | None
    transform_y: SE3 | None
    reason: str
    opencv_version: str | None

    def as_dict(self) -> dict[str, Any]:
        """Return machine-readable method, paper, and license provenance."""

        return {
            "status": self.status,
            "method": f"opencv_calibrateRobotWorldHandEye/{self.method}",
            "transform_x": _transform_dict(self.transform_x),
            "transform_y": _transform_dict(self.transform_y),
            "reason": self.reason,
            "paper_doi": _ROBOT_WORLD_DOIS[self.method],
            "tool": {
                "name": "opencv",
                "version": self.opencv_version,
                "source_url": OPENCV_SOURCE_URL,
                "api_reference": OPENCV_API_REFERENCE,
                "license_spdx": OPENCV_LICENSE_SPDX,
                "execution_mode": "optional_import",
                "external_code_executed": self.status != "unavailable",
            },
            "input_contract": {
                "world2cam": "pose_a",
                "base2gripper": "pose_b",
                "output_base2world": "transform_x",
                "output_gripper2cam": "transform_y",
                "equation": "pose_a transform_x = transform_y pose_b",
            },
        }


class OpenCvHandEyeAdapter:
    """Run one OpenCV hand-eye method without importing OpenCV in the core."""

    def solve(
        self,
        poses: Sequence[RobotWorldHandEyePosePair],
        method: OpenCvHandEyeMethod,
    ) -> OpenCvHandEyeAdapterResult:
        """Fit one OpenCV ``calibrateHandEye`` method on absolute poses."""

        if len(poses) < 3:
            return OpenCvHandEyeAdapterResult(
                status="insufficient_poses",
                method=method,
                transform_x=None,
                reason="OpenCV hand-eye calibration requires at least three poses",
                opencv_version=None,
            )
        cv2 = _optional_opencv()
        if cv2 is None:
            return OpenCvHandEyeAdapterResult(
                status="unavailable",
                method=method,
                transform_x=None,
                reason="opencv-python-headless optional dependency is not installed",
                opencv_version=None,
            )
        function = getattr(cv2, "calibrateHandEye", None)
        constant = getattr(cv2, _HAND_EYE_CONSTANTS[method], None)
        if not callable(function) or constant is None:
            return OpenCvHandEyeAdapterResult(
                status="unavailable",
                method=method,
                transform_x=None,
                reason="installed OpenCV does not expose the OpenCV 4 hand-eye API",
                opencv_version=_opencv_version(cv2),
            )
        rotations_a, translations_a = _opencv_pose_arrays([pose.pose_a for pose in poses])
        rotations_b, translations_b = _opencv_pose_arrays([pose.pose_b.inverse() for pose in poses])
        try:
            rotation, translation = function(
                rotations_a,
                translations_a,
                rotations_b,
                translations_b,
                method=constant,
            )
            transform = _se3_from_opencv(rotation, translation)
        except Exception as exc:
            return OpenCvHandEyeAdapterResult(
                status="failed",
                method=method,
                transform_x=None,
                reason=f"OpenCV hand-eye execution failed: {exc}",
                opencv_version=_opencv_version(cv2),
            )
        return OpenCvHandEyeAdapterResult(
            status="converged",
            method=method,
            transform_x=transform,
            reason="OpenCV hand-eye method converged",
            opencv_version=_opencv_version(cv2),
        )


class OpenCvRobotWorldHandEyeAdapter:
    """Run one OpenCV robot-world/hand-eye method behind an optional import."""

    def solve(
        self,
        poses: Sequence[RobotWorldHandEyePosePair],
        method: OpenCvRobotWorldHandEyeMethod,
    ) -> OpenCvRobotWorldHandEyeAdapterResult:
        """Fit one OpenCV ``calibrateRobotWorldHandEye`` method."""

        if len(poses) < 3:
            return OpenCvRobotWorldHandEyeAdapterResult(
                status="insufficient_poses",
                method=method,
                transform_x=None,
                transform_y=None,
                reason="OpenCV robot-world/hand-eye calibration requires at least three poses",
                opencv_version=None,
            )
        cv2 = _optional_opencv()
        if cv2 is None:
            return OpenCvRobotWorldHandEyeAdapterResult(
                status="unavailable",
                method=method,
                transform_x=None,
                transform_y=None,
                reason="opencv-python-headless optional dependency is not installed",
                opencv_version=None,
            )
        function = getattr(cv2, "calibrateRobotWorldHandEye", None)
        constant = getattr(cv2, _ROBOT_WORLD_CONSTANTS[method], None)
        if not callable(function) or constant is None:
            return OpenCvRobotWorldHandEyeAdapterResult(
                status="unavailable",
                method=method,
                transform_x=None,
                transform_y=None,
                reason="installed OpenCV does not expose the OpenCV 4 robot-world API",
                opencv_version=_opencv_version(cv2),
            )
        rotations_a, translations_a = _opencv_pose_arrays([pose.pose_a for pose in poses])
        rotations_b, translations_b = _opencv_pose_arrays([pose.pose_b for pose in poses])
        try:
            rotation_x, translation_x, rotation_y, translation_y = function(
                rotations_a,
                translations_a,
                rotations_b,
                translations_b,
                method=constant,
            )
            transform_x = _se3_from_opencv(rotation_x, translation_x)
            transform_y = _se3_from_opencv(rotation_y, translation_y)
        except Exception as exc:
            return OpenCvRobotWorldHandEyeAdapterResult(
                status="failed",
                method=method,
                transform_x=None,
                transform_y=None,
                reason=f"OpenCV robot-world/hand-eye execution failed: {exc}",
                opencv_version=_opencv_version(cv2),
            )
        return OpenCvRobotWorldHandEyeAdapterResult(
            status="converged",
            method=method,
            transform_x=transform_x,
            transform_y=transform_y,
            reason="OpenCV robot-world/hand-eye method converged",
            opencv_version=_opencv_version(cv2),
        )


def _optional_opencv() -> ModuleType | None:
    try:
        return importlib.import_module("cv2")
    except ImportError:
        return None


def _opencv_version(cv2: ModuleType) -> str | None:
    version = getattr(cv2, "__version__", None)
    return str(version) if version is not None else None


def _opencv_pose_arrays(
    transforms: Sequence[SE3],
) -> tuple[list[FloatArray], list[FloatArray]]:
    rotations: list[FloatArray] = []
    translations: list[FloatArray] = []
    for transform in transforms:
        rotations.append(_rotation_matrix(transform))
        translations.append(np.asarray(transform.translation_m, dtype=np.float64).reshape(3, 1))
    return rotations, translations


def _rotation_matrix(transform: SE3) -> FloatArray:
    x, y, z, w = transform.rotation_quat_xyzw
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def _se3_from_opencv(rotation: object, translation: object) -> SE3:
    rotation_array = np.asarray(rotation, dtype=np.float64)
    translation_array = np.asarray(translation, dtype=np.float64).reshape(-1)
    if rotation_array.shape != (3, 3) or translation_array.shape != (3,):
        msg = (
            "unexpected OpenCV transform shapes: "
            f"rotation={rotation_array.shape}, translation={translation_array.shape}"
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(rotation_array)) or not np.all(np.isfinite(translation_array)):
        msg = "OpenCV returned a non-finite transform"
        raise ValueError(msg)
    return SE3(
        (
            float(translation_array[0]),
            float(translation_array[1]),
            float(translation_array[2]),
        ),
        quaternion_xyzw_from_rotation_matrix(rotation_array.reshape(-1)),
    )


def _transform_dict(transform: SE3 | None) -> dict[str, list[float]] | None:
    if transform is None:
        return None
    return {
        "translation_m": list(transform.translation_m),
        "rotation_quat_xyzw": list(transform.rotation_quat_xyzw),
    }
