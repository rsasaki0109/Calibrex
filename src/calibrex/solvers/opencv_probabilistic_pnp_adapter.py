"""Optional OpenCV PnP adapter for probabilistic 2D--3D correspondences."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3, quaternion_xyzw_from_rotation_matrix
from calibrex.core.probabilistic_correspondence import (
    CorrespondenceProviderIdentity,
    ProbabilisticCorrespondenceFrame,
    ProbabilisticPnpResultArtifact,
    ProbabilisticPnpResultProvenance,
    ProbabilisticPnpToolIdentity,
    load_probabilistic_correspondence,
)
from calibrex.core.provenance import sha256_path
from calibrex.core.result import TransformResult

FloatArray: TypeAlias = NDArray[np.float64]
PnpStatus = Literal[
    "converged",
    "unavailable",
    "failed",
    "insufficient_correspondences",
    "unsupported_camera",
]

OPENCV_LICENSE_SPDX = "Apache-2.0"
OPENCV_SOURCE_URL = "https://github.com/opencv/opencv"
OPENCV_PNP_REFERENCE = "https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html"


@dataclass(frozen=True)
class OpenCvProbabilisticPnpOptions:
    """Deterministic filtering and PnP-RANSAC settings."""

    minimum_confidence: float = 0.25
    minimum_correspondences: int = 6
    ransac_reprojection_threshold_px: float = 4.0
    ransac_confidence: float = 0.999
    ransac_iterations: int = 1000
    mahalanobis_inlier_threshold: float = 3.0
    random_seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be in [0, 1]")
        if self.minimum_correspondences < 4:
            raise ValueError("minimum_correspondences must be at least four")
        if self.ransac_reprojection_threshold_px <= 0.0:
            raise ValueError("RANSAC reprojection threshold must be positive")
        if not 0.0 < self.ransac_confidence < 1.0:
            raise ValueError("RANSAC confidence must be in (0, 1)")
        if self.ransac_iterations < 1:
            raise ValueError("RANSAC iterations must be positive")
        if self.mahalanobis_inlier_threshold <= 0.0:
            raise ValueError("Mahalanobis inlier threshold must be positive")


@dataclass(frozen=True)
class OpenCvProbabilisticPnpResult:
    """PnP result plus uncertainty-aware diagnostics and full lineage."""

    status: PnpStatus
    transform_camera_lidar: SE3 | None
    reason: str
    selected_correspondence_count: int
    ransac_inlier_count: int
    probabilistic_inlier_count: int
    weighted_reprojection_rmse_px: float | None
    mean_mahalanobis_error: float | None
    artifact_id: str
    artifact_sha256: str
    frame_id: str
    camera_frame: str
    lidar_frame: str
    initial_transform_camera_lidar: SE3 | None
    initialization_artifact_sha256: str | None
    provider: dict[str, Any]
    opencv_version: str | None

    def as_dict(self) -> dict[str, Any]:
        """Return a provenance-complete, schema-safe result mapping."""

        return {
            "status": self.status,
            "method": "opencv_probabilistic_pnp_ransac/v0.1",
            "transform_camera_lidar": (
                self.transform_camera_lidar.as_dict()
                if self.transform_camera_lidar is not None
                else None
            ),
            "reason": self.reason,
            "selected_correspondence_count": self.selected_correspondence_count,
            "ransac_inlier_count": self.ransac_inlier_count,
            "probabilistic_inlier_count": self.probabilistic_inlier_count,
            "weighted_reprojection_rmse_px": self.weighted_reprojection_rmse_px,
            "mean_mahalanobis_error": self.mean_mahalanobis_error,
            "input": {
                "artifact_id": self.artifact_id,
                "artifact_sha256": self.artifact_sha256,
                "frame_id": self.frame_id,
                "provider": self.provider,
                "initial_transform_camera_lidar": (
                    self.initial_transform_camera_lidar.as_dict()
                    if self.initial_transform_camera_lidar is not None
                    else None
                ),
                "initialization_artifact_sha256": (
                    self.initialization_artifact_sha256
                ),
            },
            "tool": {
                "name": "opencv",
                "version": self.opencv_version,
                "source_url": OPENCV_SOURCE_URL,
                "api_reference": OPENCV_PNP_REFERENCE,
                "license_spdx": OPENCV_LICENSE_SPDX,
                "execution_mode": "optional_import",
                "external_code_executed": self.status != "unavailable",
            },
        }

    def to_artifact(
        self,
        *,
        result_id: str,
        options: OpenCvProbabilisticPnpOptions,
        command: list[str] | None = None,
    ) -> ProbabilisticPnpResultArtifact:
        """Build a schema-valid, provenance-complete result artifact."""

        transform = (
            TransformResult(
                parent=self.camera_frame,
                child=self.lidar_frame,
                translation_m=list(self.transform_camera_lidar.translation_m),
                rotation_quat_xyzw=list(
                    self.transform_camera_lidar.rotation_quat_xyzw
                ),
            )
            if self.transform_camera_lidar is not None
            else None
        )
        return ProbabilisticPnpResultArtifact(
            result_id=result_id,
            status=self.status,
            method="opencv_probabilistic_pnp_ransac/v0.1",
            frame_id=self.frame_id,
            initial_transform_camera_lidar=(
                TransformResult(
                    parent=self.camera_frame,
                    child=self.lidar_frame,
                    translation_m=list(
                        self.initial_transform_camera_lidar.translation_m
                    ),
                    rotation_quat_xyzw=list(
                        self.initial_transform_camera_lidar.rotation_quat_xyzw
                    ),
                )
                if self.initial_transform_camera_lidar is not None
                else None
            ),
            transform_camera_lidar=transform,
            selected_correspondence_count=self.selected_correspondence_count,
            ransac_inlier_count=self.ransac_inlier_count,
            probabilistic_inlier_count=self.probabilistic_inlier_count,
            weighted_reprojection_rmse_px=self.weighted_reprojection_rmse_px,
            mean_mahalanobis_error=self.mean_mahalanobis_error,
            reason=self.reason,
            provider=CorrespondenceProviderIdentity.model_validate(self.provider),
            tool=ProbabilisticPnpToolIdentity(
                name="opencv",
                version=self.opencv_version,
                source_url=OPENCV_SOURCE_URL,
                api_reference=OPENCV_PNP_REFERENCE,
                license_spdx=OPENCV_LICENSE_SPDX,
                execution_mode="optional_import",
            ),
            options={
                "minimum_confidence": options.minimum_confidence,
                "minimum_correspondences": options.minimum_correspondences,
                "ransac_reprojection_threshold_px": (
                    options.ransac_reprojection_threshold_px
                ),
                "ransac_confidence": options.ransac_confidence,
                "ransac_iterations": options.ransac_iterations,
                "mahalanobis_inlier_threshold": (
                    options.mahalanobis_inlier_threshold
                ),
                "random_seed": options.random_seed,
            },
            provenance=ProbabilisticPnpResultProvenance(
                generator=type(self).__module__,
                generator_version="0.1",
                command=command or [],
                correspondence_artifact_id=self.artifact_id,
                correspondence_artifact_sha256=self.artifact_sha256,
                initialization_artifact_sha256=(
                    self.initialization_artifact_sha256
                ),
            ),
        )


class OpenCvProbabilisticPnpAdapter:
    """Run PnP-RANSAC without making OpenCV a core dependency."""

    def solve_artifact(
        self,
        artifact_path: str | Path,
        frame_id: str,
        options: OpenCvProbabilisticPnpOptions | None = None,
        *,
        initial_transform_camera_lidar: SE3 | None = None,
        initialization_artifact_sha256: str | None = None,
    ) -> OpenCvProbabilisticPnpResult:
        """Validate a correspondence artifact and solve one named frame."""

        path = Path(artifact_path)
        artifact = load_probabilistic_correspondence(path)
        digest = sha256_path(path)
        if digest is None:
            raise ValueError(f"correspondence artifact is not readable: {path}")
        matches = [frame for frame in artifact.frames if frame.frame_id == frame_id]
        if not matches:
            raise ValueError(f"frame_id is absent from correspondence artifact: {frame_id}")
        return self.solve_frame(
            matches[0],
            artifact_id=artifact.artifact_id,
            artifact_sha256=digest,
            provider=artifact.provider.model_dump(mode="json", exclude_none=True),
            options=options,
            initial_transform_camera_lidar=initial_transform_camera_lidar,
            initialization_artifact_sha256=initialization_artifact_sha256,
        )

    def solve_frame(
        self,
        frame: ProbabilisticCorrespondenceFrame,
        *,
        artifact_id: str,
        artifact_sha256: str,
        provider: dict[str, Any],
        options: OpenCvProbabilisticPnpOptions | None = None,
        initial_transform_camera_lidar: SE3 | None = None,
        initialization_artifact_sha256: str | None = None,
    ) -> OpenCvProbabilisticPnpResult:
        """Solve one validated pinhole frame."""

        settings = options or OpenCvProbabilisticPnpOptions()
        if frame.intrinsics.projection != "pinhole":
            return _empty_result(
                "unsupported_camera",
                "OpenCV PnP adapter v0.1 supports pinhole correspondences only",
                artifact_id,
                artifact_sha256,
                frame.frame_id,
                frame.camera_frame,
                frame.lidar_frame,
                initial_transform_camera_lidar,
                initialization_artifact_sha256,
                provider,
            )
        selected = [
            item
            for item in frame.correspondences
            if item.reliability * (1.0 - item.outlier_probability)
            >= settings.minimum_confidence
        ]
        if len(selected) < settings.minimum_correspondences:
            return _empty_result(
                "insufficient_correspondences",
                "too few correspondences passed the prespecified confidence gate",
                artifact_id,
                artifact_sha256,
                frame.frame_id,
                frame.camera_frame,
                frame.lidar_frame,
                initial_transform_camera_lidar,
                initialization_artifact_sha256,
                provider,
                selected_count=len(selected),
            )
        cv2 = _optional_opencv()
        if cv2 is None:
            return _empty_result(
                "unavailable",
                "opencv-python-headless optional dependency is not installed",
                artifact_id,
                artifact_sha256,
                frame.frame_id,
                frame.camera_frame,
                frame.lidar_frame,
                initial_transform_camera_lidar,
                initialization_artifact_sha256,
                provider,
                selected_count=len(selected),
            )
        object_points: FloatArray = np.asarray(
            [item.point_lidar_m for item in selected], dtype=np.float64
        )
        image_points: FloatArray = np.asarray(
            [item.image_mean_px for item in selected], dtype=np.float64
        )
        camera_matrix: FloatArray = np.asarray(
            [
                [frame.intrinsics.fx, 0.0, frame.intrinsics.cx],
                [0.0, frame.intrinsics.fy, frame.intrinsics.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        distortion: FloatArray = np.asarray(
            frame.intrinsics.distortion, dtype=np.float64
        )
        try:
            cv2.setRNGSeed(settings.random_seed)
            rotation_guess: FloatArray | None = None
            translation_guess: FloatArray | None = None
            use_guess = initial_transform_camera_lidar is not None
            pnp_flag = cv2.SOLVEPNP_EPNP
            if initial_transform_camera_lidar is not None:
                rotation_guess, _jacobian = cv2.Rodrigues(
                    _rotation_matrix(initial_transform_camera_lidar)
                )
                translation_guess = np.asarray(
                    initial_transform_camera_lidar.translation_m,
                    dtype=np.float64,
                ).reshape(3, 1)
                pnp_flag = cv2.SOLVEPNP_ITERATIVE
            success, rotation_vector, translation, inliers = cv2.solvePnPRansac(
                object_points,
                image_points,
                camera_matrix,
                distortion,
                rvec=rotation_guess,
                tvec=translation_guess,
                useExtrinsicGuess=use_guess,
                iterationsCount=settings.ransac_iterations,
                reprojectionError=settings.ransac_reprojection_threshold_px,
                confidence=settings.ransac_confidence,
                flags=pnp_flag,
            )
            if not success or inliers is None:
                return _empty_result(
                    "failed",
                    "OpenCV solvePnPRansac did not return a pose",
                    artifact_id,
                    artifact_sha256,
                    frame.frame_id,
                    frame.camera_frame,
                    frame.lidar_frame,
                    initial_transform_camera_lidar,
                    initialization_artifact_sha256,
                    provider,
                    selected_count=len(selected),
                    opencv_version=_opencv_version(cv2),
                )
            inlier_indices = np.asarray(inliers, dtype=np.int64).reshape(-1)
            if inlier_indices.size >= 4 and hasattr(cv2, "solvePnPRefineLM"):
                rotation_vector, translation = cv2.solvePnPRefineLM(
                    object_points[inlier_indices],
                    image_points[inlier_indices],
                    camera_matrix,
                    distortion,
                    rotation_vector,
                    translation,
                )
            projected, _jacobian = cv2.projectPoints(
                object_points,
                rotation_vector,
                translation,
                camera_matrix,
                distortion,
            )
            rotation_matrix, _jacobian = cv2.Rodrigues(rotation_vector)
        except Exception as exc:
            return _empty_result(
                "failed",
                f"OpenCV PnP execution failed: {exc}",
                artifact_id,
                artifact_sha256,
                frame.frame_id,
                frame.camera_frame,
                frame.lidar_frame,
                initial_transform_camera_lidar,
                initialization_artifact_sha256,
                provider,
                selected_count=len(selected),
                opencv_version=_opencv_version(cv2),
            )
        residuals = np.asarray(projected, dtype=float).reshape(-1, 2) - image_points
        covariances: FloatArray = np.asarray(
            [item.image_covariance_px2 for item in selected], dtype=float
        ).reshape(-1, 2, 2)
        inverse_covariances = np.linalg.inv(covariances)
        squared_mahalanobis = np.einsum(
            "ni,nij,nj->n", residuals, inverse_covariances, residuals
        )
        probabilities: FloatArray = np.asarray(
            [
                item.reliability * (1.0 - item.outlier_probability)
                for item in selected
            ],
            dtype=float,
        )
        squared_pixel_error = np.einsum("ni,ni->n", residuals, residuals)
        weighted_rmse = math.sqrt(
            float(np.sum(probabilities * squared_pixel_error))
            / float(np.sum(probabilities))
        )
        translation_values = np.asarray(translation, dtype=float).reshape(3)
        transform = SE3(
            translation_m=(
                float(translation_values[0]),
                float(translation_values[1]),
                float(translation_values[2]),
            ),
            rotation_quat_xyzw=quaternion_xyzw_from_rotation_matrix(
                np.asarray(rotation_matrix, dtype=float).reshape(9)
            ),
        )
        probabilistic_inliers = int(
            np.count_nonzero(
                squared_mahalanobis
                <= settings.mahalanobis_inlier_threshold**2
            )
        )
        return OpenCvProbabilisticPnpResult(
            status="converged",
            transform_camera_lidar=transform,
            reason="PnP-RANSAC converged and uncertainty diagnostics were evaluated",
            selected_correspondence_count=len(selected),
            ransac_inlier_count=int(inlier_indices.size),
            probabilistic_inlier_count=probabilistic_inliers,
            weighted_reprojection_rmse_px=weighted_rmse,
            mean_mahalanobis_error=float(np.mean(np.sqrt(squared_mahalanobis))),
            artifact_id=artifact_id,
            artifact_sha256=artifact_sha256,
            frame_id=frame.frame_id,
            camera_frame=frame.camera_frame,
            lidar_frame=frame.lidar_frame,
            initial_transform_camera_lidar=initial_transform_camera_lidar,
            initialization_artifact_sha256=initialization_artifact_sha256,
            provider=provider,
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


def _empty_result(
    status: PnpStatus,
    reason: str,
    artifact_id: str,
    artifact_sha256: str,
    frame_id: str,
    camera_frame: str,
    lidar_frame: str,
    initial_transform_camera_lidar: SE3 | None,
    initialization_artifact_sha256: str | None,
    provider: dict[str, Any],
    *,
    selected_count: int = 0,
    opencv_version: str | None = None,
) -> OpenCvProbabilisticPnpResult:
    return OpenCvProbabilisticPnpResult(
        status=status,
        transform_camera_lidar=None,
        reason=reason,
        selected_correspondence_count=selected_count,
        ransac_inlier_count=0,
        probabilistic_inlier_count=0,
        weighted_reprojection_rmse_px=None,
        mean_mahalanobis_error=None,
        artifact_id=artifact_id,
        artifact_sha256=artifact_sha256,
        frame_id=frame_id,
        camera_frame=camera_frame,
        lidar_frame=lidar_frame,
        initial_transform_camera_lidar=initial_transform_camera_lidar,
        initialization_artifact_sha256=initialization_artifact_sha256,
        provider=provider,
        opencv_version=opencv_version,
    )


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
