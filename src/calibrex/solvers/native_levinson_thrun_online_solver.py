"""Native A2D2 adapter for Levinson--Thrun online edge calibration."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from calibrex.core.config import CalibrationConfig, FactorConfig
from calibrex.core.frames import FrameGraph
from calibrex.core.geometry import SE3, quaternion_xyzw_from_rotation_matrix
from calibrex.core.io import read_mapping
from calibrex.core.provenance import sha256_path
from calibrex.core.result import MetricResult, ObservabilityResult
from calibrex.data.a2d2 import read_a2d2_lidar_boundary_points
from calibrex.data.inspect import DatasetInspection
from calibrex.data.kitti import read_png_luminance
from calibrex.solvers.base import SolverAdapter, SolverAdapterResult
from calibrex.solvers.levinson_thrun_online_solver import (
    LevinsonCameraModel,
    LevinsonOnlineFrame,
    LevinsonThrunOnlineOptions,
    LevinsonThrunOnlineResult,
    LevinsonThrunOnlineSolver,
    levinson_edge_response,
)

NATIVE_LEVINSON_THRUN_ONLINE_BACKEND = "native_levinson_thrun_online"
LEVINSON_THRUN_DOI = "10.15607/RSS.2013.IX.029"
LEVINSON_THRUN_PDF = "https://roboticsproceedings.org/rss09/p29.pdf"
_FRAME_ID = re.compile(r"_(\d{9})\.(?:png|npz)$")
_PINNED_SHA256 = {
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


class NativeLevinsonThrunOnlineSolver(SolverAdapter):
    """Run local-optimum monitoring on real A2D2 camera/boundary pairs."""

    backend = NATIVE_LEVINSON_THRUN_ONLINE_BACKEND

    def solve(
        self,
        config: CalibrationConfig,
        frame_graph: FrameGraph,
        inspection: DatasetInspection,
    ) -> SolverAdapterResult:
        del frame_graph, inspection
        factor = _factor(config)
        options = factor.options if factor is not None else {}
        try:
            frames, input_files, extraction = _load_a2d2_frames(
                Path(config.dataset.path), options
            )
        except (KeyError, OSError, ValueError) as exc:
            return SolverAdapterResult(
                backend=self.backend,
                available=True,
                status="invalid_input",
                warnings=[str(exc)],
            )
        window_size = _int_option(options, "window_size", 9)
        solver_options = LevinsonThrunOnlineOptions(
            window_size=window_size,
            min_train_frames=_int_option(options, "min_train_frames", window_size),
            min_holdout_frames=_int_option(options, "min_holdout_frames", 3),
            holdout_ratio=config.evaluation.holdout_ratio,
            split_seed=config.solver.seed or 0,
            tracking_enabled=bool(options.get("tracking_enabled", True)),
            translation_grid_step_m=_float_option(
                options, "translation_grid_step_m", 0.01
            ),
            rotation_grid_step_deg=_float_option(
                options, "rotation_grid_step_deg", 0.05
            ),
            min_projected_points=_int_option(options, "min_projected_points", 16),
            known_bad_translation_m=_float_option(
                options, "known_bad_translation_m", 0.1
            ),
            known_bad_rotation_deg=_float_option(
                options, "known_bad_rotation_deg", 0.25
            ),
        )
        result = LevinsonThrunOnlineSolver().solve(
            frames, _initial_transform(options), solver_options
        )
        public_default_window_complete = window_size >= 9 and len(frames) >= 12
        preregistered = True
        status = (
            "inconclusive"
            if result.status != "insufficient_observations"
            else result.status
        )
        warnings = [
            "A2D2 clouds are pre-registered into the camera view; monitoring is not "
            "independent extrinsic accuracy evidence",
            "A2D2 boundary flags replace Eq.2 near-side beam-neighbor extraction because "
            "the view-filtered NPZ does not preserve organized beam neighborhoods",
        ]
        if not public_default_window_complete:
            warnings.append(
                "the two-frame public sample uses w=1; the paper's robust w=9 window is "
                "not available and no deployment claim is allowed"
            )
        transforms = (
            {"T_camera0_lidar0": result.final_transform_camera_lidar}
            if solver_options.tracking_enabled and result.status == "tracked"
            else {}
        )
        curvature = result.curvature
        weak_directions = []
        if curvature is None or curvature.rank < 6:
            weak_directions.append("online_edge_objective_rank_deficient")
        return SolverAdapterResult(
            backend=self.backend,
            available=True,
            status=status,
            metrics=_metrics(result, public_default_window_complete),
            transforms=transforms,
            provenance=_provenance(
                result,
                input_files,
                extraction,
                public_default_window_complete=public_default_window_complete,
                preregistered=preregistered,
            ),
            warnings=warnings,
            observability=ObservabilityResult(
                rank=curvature.rank if curvature is not None else None,
                condition_number=(
                    curvature.positive_condition_number if curvature is not None else None
                ),
                weak_directions=weak_directions,
                grade="pass" if curvature is not None and curvature.rank == 6 else "warn",
            ),
        )


def _load_a2d2_frames(
    dataset_path: Path, options: dict[str, Any]
) -> tuple[list[LevinsonOnlineFrame], list[Path], dict[str, object]]:
    calibration_path = dataset_path / str(options.get("calibration_file", "cams_lidars.json"))
    calibration = read_mapping(calibration_path)
    camera_name = str(options.get("camera_name", "front_left"))
    cameras = calibration.get("cameras")
    if not isinstance(cameras, dict) or not isinstance(cameras.get(camera_name), dict):
        raise ValueError(f"A2D2 camera calibration is missing for {camera_name}")
    camera_payload = cast(dict[str, Any], cameras[camera_name])
    matrix = camera_payload.get("CamMatrixOriginal")
    distortion = camera_payload.get("Distortion")
    resolution = camera_payload.get("Resolution")
    if not _matrix3(matrix) or not _resolution(resolution):
        raise ValueError("A2D2 camera intrinsics or resolution are malformed")
    matrix_values = cast(list[list[float]], matrix)
    resolution_values = cast(list[float], resolution)
    downsample = _int_option(options, "image_downsample", 4)
    if downsample <= 0:
        raise ValueError("image_downsample must be positive")
    width = int(resolution_values[0]) // downsample
    height = int(resolution_values[1]) // downsample
    distortion_values = _flat_floats(distortion) + [0.0] * 4
    camera = LevinsonCameraModel(
        width=width,
        height=height,
        fx=float(matrix_values[0][0]) / downsample,
        fy=float(matrix_values[1][1]) / downsample,
        cx=float(matrix_values[0][2]) / downsample,
        cy=float(matrix_values[1][2]) / downsample,
        projection="opencv_fisheye",
        distortion=(
            distortion_values[0],
            distortion_values[1],
            distortion_values[2],
            distortion_values[3],
        ),
        axes="forward_x_left_y_up_z",
    )
    images = {_file_frame_id(path): path for path in sorted(dataset_path.glob("*.png"))}
    clouds = {_file_frame_id(path): path for path in sorted(dataset_path.glob("*.npz"))}
    frame_ids = sorted(set(images) & set(clouds))
    if not frame_ids:
        raise ValueError("no filename-matched A2D2 camera/LiDAR pairs were found")
    max_points = _int_option(options, "max_boundary_points_per_frame", 2_000)
    gamma_full_resolution = _float_option(options, "edge_gamma", 0.98)
    gamma_downsampled = gamma_full_resolution**downsample
    max_radius = _int_option(options, "edge_max_radius_downsampled", 32)
    frames: list[LevinsonOnlineFrame] = []
    input_files: list[Path] = [calibration_path]
    boundary_counts: dict[str, int] = {}
    for frame_id in frame_ids:
        decoded = read_png_luminance(images[frame_id])
        if decoded is None:
            raise ValueError(f"unsupported A2D2 PNG: {images[frame_id]}")
        luminance = np.asarray(decoded.rows, dtype=float)
        reduced = _block_mean(luminance, downsample)
        points = read_a2d2_lidar_boundary_points(
            clouds[frame_id], max_points=max_points
        )
        boundary_counts[frame_id] = len(points)
        frames.append(
            LevinsonOnlineFrame(
                frame_id=frame_id,
                edge_response=levinson_edge_response(
                    reduced,
                    alpha=1.0 / 3.0,
                    gamma=gamma_downsampled,
                    max_radius=max_radius,
                ),
                lidar_points=np.asarray(points, dtype=float),
                discontinuity_weights=np.ones(len(points), dtype=float),
                camera=camera,
            )
        )
        input_files.extend((images[frame_id], clouds[frame_id]))
    return (
        frames,
        input_files,
        {
            "camera_name": camera_name,
            "frame_ids": frame_ids,
            "boundary_point_counts": boundary_counts,
            "image_downsample": downsample,
            "edge_alpha": 1.0 / 3.0,
            "edge_gamma_full_resolution": gamma_full_resolution,
            "edge_gamma_downsampled": gamma_downsampled,
            "edge_max_radius_downsampled": max_radius,
            "depth_discontinuity_source": "A2D2 pcloud_attr.boundary with unit weights",
            "equation_2_recomputed": False,
        },
    )


def _metrics(
    result: LevinsonThrunOnlineResult, public_default_window_complete: bool
) -> dict[str, MetricResult]:
    final_step = result.timeline[-1] if result.timeline else None
    detectable = sum(probe.detectable for probe in result.probes)
    probe_fraction = detectable / len(result.probes) if result.probes else None
    curvature = result.curvature
    return {
        "levinson_edge_objective": MetricResult(
            train=result.train_evaluation.normalized_objective,
            holdout=result.holdout_evaluation.normalized_objective,
            grade="warn",
            reason="support-normalized companion to the paper's raw Eq.3 objective",
        ),
        "levinson_projected_boundary_point_count": MetricResult(
            train=float(result.train_evaluation.projected_point_count),
            holdout=float(result.holdout_evaluation.projected_point_count),
            unit="points",
            grade="pass" if result.train_evaluation.projected_point_count else "fail",
        ),
        "levinson_worsening_fraction": MetricResult(
            value=final_step.worsening_fraction if final_step is not None else None,
            unit="fraction",
            grade="warn",
            reason="fraction of 728 non-center grid candidates below the center objective",
        ),
        "levinson_calibrated_probability": MetricResult(
            value=final_step.calibrated_probability if final_step is not None else None,
            unit="probability",
            grade="warn",
            reason="Eq.4 monitor on a pre-registered single-frame public window",
        ),
        "levinson_known_bad_detectable_fraction": MetricResult(
            value=probe_fraction,
            unit="fraction",
            grade="pass" if probe_fraction is not None and probe_fraction >= 0.8 else "warn",
            reason=f"{detectable}/{len(result.probes)} signed shadow-holdout controls detected",
        ),
        "levinson_objective_curvature_rank": MetricResult(
            value=float(curvature.rank) if curvature is not None else None,
            unit="rank",
            grade="pass" if curvature is not None and curvature.rank == 6 else "warn",
            reason="negative held-out objective Hessian; not covariance",
        ),
        "levinson_default_window_complete": MetricResult(
            value=1.0 if public_default_window_complete else 0.0,
            unit="bool",
            grade="pass" if public_default_window_complete else "warn",
            reason="paper reports robust monitoring with w=9; public sample contains two frames",
        ),
        "levinson_equation_2_recomputed": MetricResult(
            value=0.0,
            unit="bool",
            grade="warn",
            reason="A2D2 boundary flags replace unavailable organized-beam Eq.2 extraction",
        ),
        "levinson_public_accuracy_independent": MetricResult(
            value=0.0,
            unit="bool",
            grade="warn",
            reason="A2D2 clouds are distributed already mapped into the camera view",
        ),
    }


def _provenance(
    result: LevinsonThrunOnlineResult,
    input_files: list[Path],
    extraction: dict[str, object],
    *,
    public_default_window_complete: bool,
    preregistered: bool,
) -> dict[str, Any]:
    raw_inputs = [_raw_input_payload(path) for path in input_files]
    return {
        "levinson_thrun_online": result.as_dict(),
        "paper_doi": LEVINSON_THRUN_DOI,
        "primary_paper_pdf": LEVINSON_THRUN_PDF,
        "a2d2_preregistered_camera_view": preregistered,
        "paper_default_window_complete": public_default_window_complete,
        "public_evidence_claim": "inconclusive_window_and_independent_accuracy",
        "extraction": extraction,
        "metrics_origin": "recomputed",
        "data_verified": all(
            item["digest_matches_pinned_sample"] is True for item in raw_inputs
        ),
        "raw_input_files": raw_inputs,
        "implementation_origin": "independent equations-based implementation; no external code",
        "license_boundary": "Calibrex Apache-2.0 core; A2D2 inputs CC BY-ND 4.0",
    }


def _raw_input_payload(path: Path) -> dict[str, object]:
    digest = sha256_path(path)
    expected = _PINNED_SHA256.get(path.name)
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
    factor = config.pipeline.factors.get("lidar_camera_depth_edge_alignment")
    return factor if factor is not None and factor.enabled else None


def _block_mean(image: NDArray[np.float64], factor: int) -> NDArray[np.float64]:
    height = image.shape[0] // factor * factor
    width = image.shape[1] // factor * factor
    cropped = image[:height, :width]
    return cropped.reshape(height // factor, factor, width // factor, factor).mean(axis=(1, 3))


def _initial_transform(options: dict[str, Any]) -> SE3:
    translation = _float_sequence(options.get("initial_translation_m"), (0.0, 0.0, 0.0))
    rpy_deg = _float_sequence(options.get("initial_rotation_rpy_deg"), (0.0, 0.0, 0.0))
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
    return SE3(translation, quaternion_xyzw_from_rotation_matrix(rotation.reshape(-1)))


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
