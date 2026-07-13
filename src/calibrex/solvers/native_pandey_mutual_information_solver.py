"""Native pipeline adapter for Pandey mutual-information calibration."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, cast

import numpy as np

from calibrex.core.config import CalibrationConfig, FactorConfig
from calibrex.core.frames import FrameGraph
from calibrex.core.geometry import SE3, quaternion_xyzw_from_rotation_matrix
from calibrex.core.io import read_mapping
from calibrex.core.provenance import sha256_path
from calibrex.core.result import MetricResult, ObservabilityResult
from calibrex.data.a2d2 import read_a2d2_lidar_points_reflectivity
from calibrex.data.inspect import DatasetInspection
from calibrex.data.kitti import read_png_luminance
from calibrex.solvers.base import SolverAdapter, SolverAdapterResult
from calibrex.solvers.pandey_mutual_information_solver import (
    MutualInformationCameraModel,
    MutualInformationObservation,
    PandeyMutualInformationOptions,
    PandeyMutualInformationResult,
    PandeyMutualInformationSolver,
)

NATIVE_PANDEY_MUTUAL_INFORMATION_BACKEND = "native_pandey_mutual_information"
PANDEY_PAPER_DOI = "10.1609/aaai.v26i1.8379"
PANDEY_PRIMARY_PDF = "https://robots.engin.umich.edu/publications/gpandey-2012a.pdf"
A2D2_TUTORIAL_URL = (
    "https://aev-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/tutorial.ipynb"
)
_PINNED_A2D2_SHA256 = {
    "20180810150607_camera_frontleft_000000060.png": (
        "b07c5bc0c985a0fa9604c81b03649033855dde6f7587a4dc33e0d6216e958aca"
    ),
    "20180810150607_camera_frontleft_000000061.png": (
        "a1389f0ed253dc0fe84d175034edfe6159e3f220813776c601a7455e01f00616"
    ),
    "20180810150607_lidar_frontleft_000000060.npz": (
        "1605ad835324e73d31998c94493c646307716404662d3e446a15d3f3f0616c63"
    ),
    "20180810150607_lidar_frontleft_000000061.npz": (
        "fee0b3ef3f347e422ee38c01fe41b2722b1148a076062b9c4a0df9a42dd24f91"
    ),
    "cams_lidars.json": (
        "ffec04167050b9c0397121720b8f0bad2cacee83d03c1b7e864619394629c8d2"
    ),
}
_FRAME_ID = re.compile(r"_(\d{9})\.(?:png|npz)$")


class NativePandeyMutualInformationSolver(SolverAdapter):
    """Run the independent Pandey implementation on paired A2D2 files."""

    backend = NATIVE_PANDEY_MUTUAL_INFORMATION_BACKEND

    def solve(
        self,
        config: CalibrationConfig,
        frame_graph: FrameGraph,
        inspection: DatasetInspection,
    ) -> SolverAdapterResult:
        del frame_graph, inspection
        factor = _factor(config)
        options = factor.options if factor is not None else {}
        dataset_path = Path(config.dataset.path)
        try:
            observations, input_files = _load_a2d2_observations(dataset_path, options)
        except (KeyError, OSError, ValueError) as exc:
            return SolverAdapterResult(
                backend=self.backend,
                available=True,
                status="invalid_input",
                warnings=[str(exc)],
            )
        solver_options = PandeyMutualInformationOptions(
            min_train_observations=_int_option(options, "min_train_observations", 2),
            holdout_ratio=config.evaluation.holdout_ratio,
            split_seed=config.solver.seed or 0,
            histogram_bins=_int_option(options, "histogram_bins", 32),
            min_projected_points=_int_option(options, "min_projected_points", 64),
            max_iterations=config.solver.max_iterations,
            convergence_tolerance=config.solver.convergence_tolerance,
            initial_step_size=_float_option(options, "initial_step_size", 0.02),
            min_step_size=_float_option(options, "min_step_size", 1.0e-5),
            max_step_size=_float_option(options, "max_step_size", 0.1),
            known_bad_translation_m=_float_option(
                options, "known_bad_translation_m", 0.05
            ),
            known_bad_rotation_deg=_float_option(options, "known_bad_rotation_deg", 5.0),
        )
        result = PandeyMutualInformationSolver().solve(
            observations,
            _initial_transform(options),
            solver_options,
        )
        if result.transform_camera_lidar is None:
            return SolverAdapterResult(
                backend=self.backend,
                available=True,
                status=result.status,
                metrics=_metrics(result, preregistered=True),
                provenance=_provenance(result, input_files, preregistered=True),
                warnings=[result.reason],
            )
        curvature = result.curvature
        weak_directions = []
        if curvature is not None and curvature.rank < 6:
            weak_directions.append("mutual_information_objective_rank_deficient")
        warnings = [
            "A2D2 point clouds are pre-registered into each camera view; this run is "
            "execution and perturbation evidence, not an independent extrinsic accuracy test"
        ]
        return SolverAdapterResult(
            backend=self.backend,
            available=True,
            status="inconclusive",
            metrics=_metrics(result, preregistered=True),
            transforms={"T_camera0_lidar0": result.transform_camera_lidar},
            provenance=_provenance(result, input_files, preregistered=True),
            warnings=warnings,
            observability=ObservabilityResult(
                grade="pass" if curvature is not None and curvature.rank == 6 else "warn",
                rank=curvature.rank if curvature is not None else None,
                condition_number=(
                    curvature.positive_condition_number if curvature is not None else None
                ),
                weak_directions=weak_directions,
            ),
        )


def _load_a2d2_observations(
    dataset_path: Path, options: dict[str, Any]
) -> tuple[list[MutualInformationObservation], list[Path]]:
    calibration_path = dataset_path / str(options.get("calibration_file", "cams_lidars.json"))
    calibration = read_mapping(calibration_path)
    camera_name = str(options.get("camera_name", "front_left"))
    cameras = calibration.get("cameras")
    if not isinstance(cameras, dict) or not isinstance(cameras.get(camera_name), dict):
        raise ValueError(f"A2D2 camera calibration is missing for {camera_name}")
    camera_payload = cameras[camera_name]
    assert isinstance(camera_payload, dict)
    matrix = camera_payload.get("CamMatrixOriginal")
    distortion = camera_payload.get("Distortion")
    resolution = camera_payload.get("Resolution")
    if not _matrix3(matrix) or not _resolution(resolution):
        raise ValueError("A2D2 camera intrinsics or resolution are malformed")
    matrix_values = cast(list[list[float]], matrix)
    resolution_values = cast(list[float], resolution)
    distortion_values = _flat_floats(distortion)
    padded_distortion = distortion_values + [0.0] * 4
    distortion4 = (
        padded_distortion[0],
        padded_distortion[1],
        padded_distortion[2],
        padded_distortion[3],
    )
    camera = MutualInformationCameraModel(
        width=int(resolution_values[0]),
        height=int(resolution_values[1]),
        fx=float(matrix_values[0][0]),
        fy=float(matrix_values[1][1]),
        cx=float(matrix_values[0][2]),
        cy=float(matrix_values[1][2]),
        projection="opencv_fisheye",
        distortion=distortion4,
        axes="forward_x_left_y_up_z",
    )
    images = {_file_frame_id(path): path for path in sorted(dataset_path.glob("*.png"))}
    clouds = {_file_frame_id(path): path for path in sorted(dataset_path.glob("*.npz"))}
    frame_ids = sorted(set(images) & set(clouds))
    if not frame_ids:
        raise ValueError("no filename-matched A2D2 camera/LiDAR pairs were found")
    max_points = _int_option(options, "max_points_per_frame", 5_000)
    observations: list[MutualInformationObservation] = []
    input_files: list[Path] = [calibration_path]
    for frame_id in frame_ids:
        decoded = read_png_luminance(images[frame_id])
        if decoded is None:
            raise ValueError(f"unsupported A2D2 PNG: {images[frame_id]}")
        points, reflectivity = read_a2d2_lidar_points_reflectivity(
            clouds[frame_id], max_points=max_points
        )
        observations.append(
            MutualInformationObservation(
                frame_id=frame_id,
                luminance=np.asarray(decoded.rows, dtype=float),
                lidar_points=np.asarray(points, dtype=float),
                reflectivity=np.asarray(reflectivity, dtype=float),
                camera=camera,
            )
        )
        input_files.extend((images[frame_id], clouds[frame_id]))
    return observations, input_files


def _metrics(
    result: PandeyMutualInformationResult, *, preregistered: bool
) -> dict[str, MetricResult]:
    probes = result.probes
    detectable_count = sum(item.detectable for item in probes)
    detectable_fraction = detectable_count / len(probes) if probes else None
    curvature = result.curvature
    transform = result.transform_camera_lidar
    reference_translation_error = (
        math.sqrt(sum(value * value for value in transform.translation_m))
        if transform is not None and preregistered
        else None
    )
    reference_rotation_error = (
        math.degrees(2.0 * math.acos(min(1.0, abs(transform.rotation_quat_xyzw[3]))))
        if transform is not None and preregistered
        else None
    )
    return {
        "lidar_camera_mutual_information_score": MetricResult(
            train=result.train_evaluation.normalized_mutual_information,
            holdout=result.holdout_evaluation.normalized_mutual_information,
            grade="warn",
            reason=(
                "normalized MI recomputed on A2D2 pre-registered camera-view clouds"
                if preregistered
                else "normalized held-out mutual information"
            ),
        ),
        "pandey_mutual_information_raw": MetricResult(
            train=result.train_evaluation.raw_mutual_information,
            holdout=result.holdout_evaluation.raw_mutual_information,
            unit="nats",
            grade="warn",
            reason="objective magnitude has no universal accuracy threshold",
        ),
        "pandey_projected_point_count": MetricResult(
            train=float(result.train_evaluation.projected_point_count),
            holdout=float(result.holdout_evaluation.projected_point_count),
            unit="points",
            grade="pass" if result.train_evaluation.projected_point_count > 0 else "fail",
        ),
        "pandey_known_bad_detectable_fraction": MetricResult(
            value=detectable_fraction,
            unit="fraction",
            grade=(
                "pass"
                if detectable_fraction is not None and detectable_fraction >= 0.8
                else "warn"
            ),
            reason=f"{detectable_count}/{len(probes)} signed held-out controls lowered MI",
        ),
        "pandey_objective_curvature_rank": MetricResult(
            value=float(curvature.rank) if curvature is not None else None,
            unit="rank",
            grade="pass" if curvature is not None and curvature.rank == 6 else "warn",
            reason="negative-MI numerical Hessian rank; not covariance or CRLB",
        ),
        "pandey_public_accuracy_independent": MetricResult(
            value=0.0 if preregistered else 1.0,
            unit="bool",
            grade="warn" if preregistered else "pass",
            reason="A2D2 distributes LiDAR points already mapped into the camera view",
        ),
        "pandey_preregistered_reference_translation_error_m": MetricResult(
            value=reference_translation_error,
            unit="m",
            grade="warn",
            reason=(
                "error to A2D2's distributed camera-view registration; "
                "not independent metrology"
            ),
        ),
        "pandey_preregistered_reference_rotation_error_deg": MetricResult(
            value=reference_rotation_error,
            unit="deg",
            grade="warn",
            reason=(
                "error to A2D2's distributed camera-view registration; "
                "not independent metrology"
            ),
        ),
    }


def _provenance(
    result: PandeyMutualInformationResult,
    input_files: list[Path],
    *,
    preregistered: bool,
) -> dict[str, Any]:
    raw_inputs = [_raw_input_payload(path) for path in input_files]
    data_verified = all(
        item["digest_matches_pinned_sample"] is True for item in raw_inputs
    )
    return {
        "pandey_mutual_information": result.as_dict(),
        "paper_doi": PANDEY_PAPER_DOI,
        "primary_paper_pdf": PANDEY_PRIMARY_PDF,
        "a2d2_official_tutorial": A2D2_TUTORIAL_URL,
        "a2d2_preregistered_camera_view": preregistered,
        "public_evidence_claim": "inconclusive_independent_accuracy" if preregistered else "scored",
        "metrics_origin": "recomputed",
        "data_verified": data_verified,
        "raw_input_files": raw_inputs,
        "implementation_origin": "independent equations-based implementation; no external code",
        "license_boundary": "Calibrex Apache-2.0 core; A2D2 inputs CC BY-ND 4.0",
    }


def _raw_input_payload(path: Path) -> dict[str, object]:
    digest = sha256_path(path)
    expected = _PINNED_A2D2_SHA256.get(path.name)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest,
        "expected_sha256": expected,
        "digest_matches_pinned_sample": expected is not None and digest == expected,
        "source": "A2D2 public sequence 20180810_150607",
        "license": "CC BY-ND 4.0",
    }


def _factor(config: CalibrationConfig) -> FactorConfig | None:
    factor = config.pipeline.factors.get("lidar_camera_mutual_information")
    return factor if factor is not None and factor.enabled else None


def _initial_transform(options: dict[str, Any]) -> SE3:
    translation = _float_sequence(options.get("initial_translation_m"), (0.02, -0.015, 0.01))
    rpy_deg = _float_sequence(options.get("initial_rotation_rpy_deg"), (0.3, -0.2, 0.4))
    roll, pitch, yaw = (math.radians(value) for value in rpy_deg)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        )
    )
    return SE3(
        translation,
        quaternion_xyzw_from_rotation_matrix(rotation.reshape(-1)),
    )


def _file_frame_id(path: Path) -> str:
    match = _FRAME_ID.search(path.name)
    if match is None:
        raise ValueError(f"A2D2 filename has no nine-digit frame id: {path.name}")
    return match.group(1)


def _matrix3(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(row, list) and len(row) == 3 for row in value)
    )


def _resolution(value: object) -> bool:
    return isinstance(value, list) and len(value) == 2


def _flat_floats(value: object) -> list[float]:
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    return [float(item) for item in value] if isinstance(value, list) else []


def _float_sequence(
    value: object, default: tuple[float, float, float]
) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return default
    return (float(value[0]), float(value[1]), float(value[2]))


def _int_option(options: dict[str, Any], name: str, default: int) -> int:
    value = options.get(name, default)
    return int(value) if isinstance(value, (int, float)) else default


def _float_option(options: dict[str, Any], name: str, default: float) -> float:
    value = options.get(name, default)
    return float(value) if isinstance(value, (int, float)) else default
