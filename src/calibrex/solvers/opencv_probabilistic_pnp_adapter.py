"""Optional OpenCV PnP adapter for probabilistic 2D--3D correspondences."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3, quaternion_xyzw_from_rotation_matrix
from calibrex.core.probabilistic_correspondence import (
    CorrespondenceProviderIdentity,
    ProbabilisticCorrespondenceArtifact,
    ProbabilisticCorrespondenceFrame,
    ProbabilisticImageCorrespondence,
    ProbabilisticPnpResultArtifact,
    ProbabilisticPnpResultProvenance,
    ProbabilisticPnpToolIdentity,
    load_probabilistic_correspondence,
)
from calibrex.core.provenance import sha256_path
from calibrex.core.result import TransformResult
from calibrex.solvers.probabilistic_camera_lidar_refiner import project_camera_point

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
OPENCV_PNP_ADAPTER_VERSION = "0.3"
OPENCV_PNP_METHOD = "opencv_probabilistic_pnp_ransac/v0.2"
OPENCV_AGGREGATE_PNP_METHOD = "opencv_probabilistic_aggregate_pnp_ransac/v0.3"
AGGREGATE_FRAME_SELECTION_RULE = (
    "minimum_effective_confidence_and_frame_correspondence_support/v0.1"
)
_MINIMUM_VIRTUAL_BEARING_Z = 1.0e-6


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
    minimum_frame_correspondences: int = 4
    minimum_frames: int = 2

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
        if self.minimum_frame_correspondences < 4:
            raise ValueError("minimum frame correspondences must be at least four")
        if self.minimum_frames < 1:
            raise ValueError("minimum frames must be positive")


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
    initializer_calibration_id: str | None = None
    initializer_calibration_sha256: str | None = None
    ransac_inlier_reprojection_rmse_px: float | None = None
    source_frame_ids: tuple[str, ...] = ()
    frame_selection_rule_id: str = "explicit_single_frame/v0.1"
    method: str = OPENCV_PNP_METHOD

    def as_dict(self) -> dict[str, Any]:
        """Return a provenance-complete, schema-safe result mapping."""

        return {
            "status": self.status,
            "method": self.method,
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
            "ransac_inlier_reprojection_rmse_px": (
                self.ransac_inlier_reprojection_rmse_px
            ),
            "mean_mahalanobis_error": self.mean_mahalanobis_error,
            "input": {
                "artifact_id": self.artifact_id,
                "artifact_sha256": self.artifact_sha256,
                "frame_id": self.frame_id,
                "source_frame_ids": list(
                    self.source_frame_ids or (self.frame_id,)
                ),
                "frame_selection_rule_id": self.frame_selection_rule_id,
                "provider": self.provider,
                "initial_transform_camera_lidar": (
                    self.initial_transform_camera_lidar.as_dict()
                    if self.initial_transform_camera_lidar is not None
                    else None
                ),
                "initialization_artifact_sha256": (
                    self.initialization_artifact_sha256
                ),
                "initializer_calibration_id": self.initializer_calibration_id,
                "initializer_calibration_sha256": (
                    self.initializer_calibration_sha256
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
            method=self.method,
            frame_id=self.frame_id,
            source_frame_ids=list(self.source_frame_ids or (self.frame_id,)),
            frame_selection_rule_id=self.frame_selection_rule_id,
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
            ransac_inlier_reprojection_rmse_px=(
                self.ransac_inlier_reprojection_rmse_px
            ),
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
                "minimum_frame_correspondences": (
                    options.minimum_frame_correspondences
                ),
                "minimum_frames": options.minimum_frames,
            },
            provenance=ProbabilisticPnpResultProvenance(
                generator=type(self).__module__,
                generator_version=OPENCV_PNP_ADAPTER_VERSION,
                command=command or [],
                correspondence_artifact_id=self.artifact_id,
                correspondence_artifact_sha256=self.artifact_sha256,
                initialization_artifact_sha256=(
                    self.initialization_artifact_sha256
                ),
                initializer_calibration_id=self.initializer_calibration_id,
                initializer_calibration_sha256=(
                    self.initializer_calibration_sha256
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

    def solve_artifact_aggregate(
        self,
        artifact_path: str | Path,
        options: OpenCvProbabilisticPnpOptions | None = None,
        *,
        initial_transform_camera_lidar: SE3 | None = None,
        initialization_artifact_sha256: str | None = None,
        initializer_calibration_id: str | None = None,
        initializer_calibration_sha256: str | None = None,
    ) -> OpenCvProbabilisticPnpResult:
        """Validate an artifact and solve one shared pose from eligible frames."""

        path = Path(artifact_path)
        artifact = load_probabilistic_correspondence(path)
        digest = sha256_path(path)
        if digest is None:
            raise ValueError(f"correspondence artifact is not readable: {path}")
        return self.solve_aggregate(
            artifact,
            artifact_sha256=digest,
            options=options,
            initial_transform_camera_lidar=initial_transform_camera_lidar,
            initialization_artifact_sha256=initialization_artifact_sha256,
            initializer_calibration_id=initializer_calibration_id,
            initializer_calibration_sha256=initializer_calibration_sha256,
        )

    def solve_aggregate(
        self,
        artifact: ProbabilisticCorrespondenceArtifact,
        *,
        artifact_sha256: str,
        options: OpenCvProbabilisticPnpOptions | None = None,
        initial_transform_camera_lidar: SE3 | None = None,
        initialization_artifact_sha256: str | None = None,
        initializer_calibration_id: str | None = None,
        initializer_calibration_sha256: str | None = None,
    ) -> OpenCvProbabilisticPnpResult:
        """Select supported frames without a reference pose and solve shared SE(3)."""

        if (initializer_calibration_id is None) != (
            initializer_calibration_sha256 is None
        ):
            raise ValueError(
                "initializer calibration ID and SHA-256 must be supplied together"
            )
        settings = options or OpenCvProbabilisticPnpOptions()
        frames = sorted(artifact.frames, key=lambda item: item.frame_id)
        first = frames[0]
        incompatible = [
            frame.frame_id
            for frame in frames
            if frame.camera_frame != first.camera_frame
            or frame.lidar_frame != first.lidar_frame
            or frame.intrinsics != first.intrinsics
        ]
        if incompatible:
            raise ValueError(
                "aggregate PnP requires identical frames and camera intrinsics: "
                + ", ".join(incompatible)
            )
        eligible = [
            frame
            for frame in frames
            if sum(
                _effective_confidence(item) >= settings.minimum_confidence
                for item in frame.correspondences
            )
            >= settings.minimum_frame_correspondences
        ]
        aggregate_id = f"aggregate:{len(eligible)}"
        provider = artifact.provider.model_dump(mode="json", exclude_none=True)
        if len(eligible) < settings.minimum_frames:
            result = _empty_result(
                "insufficient_correspondences",
                "too few frames passed the prespecified aggregate support gate "
                f"({len(eligible)} retained, {settings.minimum_frames} required)",
                artifact.artifact_id,
                artifact_sha256,
                aggregate_id,
                first.camera_frame,
                first.lidar_frame,
                initial_transform_camera_lidar,
                initialization_artifact_sha256,
                provider,
                selected_count=sum(
                    _effective_confidence(item) >= settings.minimum_confidence
                    for frame in eligible
                    for item in frame.correspondences
                ),
            )
            return replace(
                result,
                source_frame_ids=tuple(frame.frame_id for frame in frames),
                frame_selection_rule_id=AGGREGATE_FRAME_SELECTION_RULE,
                method=OPENCV_AGGREGATE_PNP_METHOD,
                initializer_calibration_id=initializer_calibration_id,
                initializer_calibration_sha256=initializer_calibration_sha256,
            )
        combined = [
            item.model_copy(
                update={
                    "correspondence_id": (
                        f"{frame.frame_id}:{item.correspondence_id}"
                    )
                }
            )
            for frame in eligible
            for item in frame.correspondences
        ]
        aggregate = ProbabilisticCorrespondenceFrame(
            frame_id=aggregate_id,
            capture_time_ns=min(frame.capture_time_ns for frame in eligible),
            camera_frame=first.camera_frame,
            lidar_frame=first.lidar_frame,
            intrinsics=first.intrinsics,
            correspondences=combined,
        )
        result = self.solve_frame(
            aggregate,
            artifact_id=artifact.artifact_id,
            artifact_sha256=artifact_sha256,
            provider=provider,
            options=settings,
            initial_transform_camera_lidar=initial_transform_camera_lidar,
            initialization_artifact_sha256=initialization_artifact_sha256,
        )
        return replace(
            result,
            source_frame_ids=tuple(frame.frame_id for frame in eligible),
            frame_selection_rule_id=AGGREGATE_FRAME_SELECTION_RULE,
            method=OPENCV_AGGREGATE_PNP_METHOD,
            initializer_calibration_id=initializer_calibration_id,
            initializer_calibration_sha256=initializer_calibration_sha256,
            reason=(
                result.reason
                + "; aggregate frame support selected "
                f"{len(eligible)}/{len(frames)} frames"
            ),
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
        """Solve one validated pinhole or MEI frame."""

        settings = options or OpenCvProbabilisticPnpOptions()
        if frame.intrinsics.projection not in {"pinhole", "mei"}:
            return _empty_result(
                "unsupported_camera",
                "OpenCV PnP adapter v0.2 supports pinhole and MEI "
                "correspondences only",
                artifact_id,
                artifact_sha256,
                frame.frame_id,
                frame.camera_frame,
                frame.lidar_frame,
                initial_transform_camera_lidar,
                initialization_artifact_sha256,
                provider,
            )
        confidence_selected = [
            item
            for item in frame.correspondences
            if item.reliability * (1.0 - item.outlier_probability)
            >= settings.minimum_confidence
        ]
        if len(confidence_selected) < settings.minimum_correspondences:
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
                selected_count=len(confidence_selected),
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
                selected_count=len(confidence_selected),
            )
        selected = confidence_selected
        observed_image_points: FloatArray = np.asarray(
            [item.image_mean_px for item in selected], dtype=np.float64
        )
        camera_matrix = _camera_matrix(frame)
        distortion: FloatArray = np.asarray(
            frame.intrinsics.distortion, dtype=np.float64
        )
        image_points = observed_image_points
        excluded_bearing_count = 0
        if frame.intrinsics.projection == "mei":
            try:
                selected, image_points, camera_matrix = _mei_virtual_pinhole_inputs(
                    cv2,
                    frame,
                    selected,
                )
            except (ArithmeticError, ValueError) as exc:
                return _empty_result(
                    "failed",
                    f"MEI bearing conversion failed: {exc}",
                    artifact_id,
                    artifact_sha256,
                    frame.frame_id,
                    frame.camera_frame,
                    frame.lidar_frame,
                    initial_transform_camera_lidar,
                    initialization_artifact_sha256,
                    provider,
                    selected_count=0,
                    opencv_version=_opencv_version(cv2),
                )
            excluded_bearing_count = len(confidence_selected) - len(selected)
            observed_image_points = np.asarray(
                [item.image_mean_px for item in selected], dtype=np.float64
            )
            distortion = np.zeros(4, dtype=np.float64)
            if len(selected) < settings.minimum_correspondences:
                return _empty_result(
                    "insufficient_correspondences",
                    "too few confidence-gated MEI rays lie in the positive virtual "
                    f"pinhole hemisphere ({len(selected)} retained, "
                    f"{excluded_bearing_count} excluded)",
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
        object_points: FloatArray = np.asarray(
            [item.point_lidar_m for item in selected], dtype=np.float64
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
        if frame.intrinsics.projection == "mei":
            projected = _project_mei_diagnostics(
                object_points,
                np.asarray(rotation_matrix, dtype=float).reshape(3, 3),
                np.asarray(translation, dtype=float).reshape(3),
                frame,
            )
        else:
            projected, _jacobian = cv2.projectPoints(
                object_points,
                rotation_vector,
                translation,
                camera_matrix,
                distortion,
            )
        if not np.all(np.isfinite(projected)):
            return _empty_result(
                "failed",
                "the solved pose produced an invalid declared-camera projection",
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
        residuals = (
            np.asarray(projected, dtype=float).reshape(-1, 2)
            - observed_image_points
        )
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
        inlier_rmse = math.sqrt(
            float(np.mean(squared_pixel_error[inlier_indices]))
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
            reason=(
                "PnP-RANSAC converged and uncertainty diagnostics were evaluated"
                if frame.intrinsics.projection == "pinhole"
                else "PnP-RANSAC converged through the MEI bearing-to-virtual-"
                "pinhole adapter; diagnostics use the declared MEI model; "
                f"{excluded_bearing_count} confidence-gated rays outside the "
                "positive virtual hemisphere were excluded"
            ),
            selected_correspondence_count=len(selected),
            ransac_inlier_count=int(inlier_indices.size),
            probabilistic_inlier_count=probabilistic_inliers,
            weighted_reprojection_rmse_px=weighted_rmse,
            ransac_inlier_reprojection_rmse_px=inlier_rmse,
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


def _camera_matrix(frame: ProbabilisticCorrespondenceFrame) -> FloatArray:
    return np.asarray(
        [
            [frame.intrinsics.fx, 0.0, frame.intrinsics.cx],
            [0.0, frame.intrinsics.fy, frame.intrinsics.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _mei_virtual_pinhole_inputs(
    cv2: ModuleType,
    frame: ProbabilisticCorrespondenceFrame,
    selected: list[ProbabilisticImageCorrespondence],
) -> tuple[list[ProbabilisticImageCorrespondence], FloatArray, FloatArray]:
    """Map valid MEI pixels to one positive-hemisphere virtual pinhole."""

    xi = frame.intrinsics.xi
    if xi is None:
        raise ValueError("MEI intrinsics require xi")
    raw_points = np.asarray(
        [item.image_mean_px for item in selected], dtype=np.float64
    ).reshape(-1, 1, 2)
    undistorted = np.asarray(
        cv2.undistortPoints(
            raw_points,
            _camera_matrix(frame),
            np.asarray(frame.intrinsics.distortion, dtype=np.float64),
        ),
        dtype=np.float64,
    ).reshape(-1, 2)
    retained: list[ProbabilisticImageCorrespondence] = []
    virtual_normalized: list[tuple[float, float]] = []
    for item, normalized in zip(selected, undistorted, strict=True):
        x_value = float(normalized[0])
        y_value = float(normalized[1])
        radius_squared = x_value * x_value + y_value * y_value
        radicand = 1.0 + (1.0 - xi * xi) * radius_squared
        if not math.isfinite(radicand) or radicand < 0.0:
            continue
        scale = (xi + math.sqrt(radicand)) / (1.0 + radius_squared)
        bearing_z = scale - xi
        if not math.isfinite(bearing_z) or bearing_z <= _MINIMUM_VIRTUAL_BEARING_Z:
            continue
        bearing_x = scale * x_value
        bearing_y = scale * y_value
        virtual_x = bearing_x / bearing_z
        virtual_y = bearing_y / bearing_z
        if not math.isfinite(virtual_x) or not math.isfinite(virtual_y):
            continue
        retained.append(item)
        virtual_normalized.append((virtual_x, virtual_y))
    virtual_focal = math.sqrt(frame.intrinsics.fx * frame.intrinsics.fy)
    virtual_camera_matrix: FloatArray = np.asarray(
        [
            [virtual_focal, 0.0, 0.0],
            [0.0, virtual_focal, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    virtual_image_points: FloatArray = virtual_focal * np.asarray(
        virtual_normalized, dtype=np.float64
    ).reshape(-1, 2)
    return retained, virtual_image_points, virtual_camera_matrix


def _project_mei_diagnostics(
    object_points: FloatArray,
    rotation_matrix: FloatArray,
    translation: FloatArray,
    frame: ProbabilisticCorrespondenceFrame,
) -> FloatArray:
    camera_points = (
        np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
        @ object_points.T
    ).T + np.asarray(translation, dtype=np.float64).reshape(1, 3)
    return np.asarray(
        [
            project_camera_point(
                (float(point[0]), float(point[1]), float(point[2])), frame
            )
            for point in camera_points
        ],
        dtype=np.float64,
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


def _effective_confidence(item: ProbabilisticImageCorrespondence) -> float:
    return item.reliability * (1.0 - item.outlier_probability)


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
