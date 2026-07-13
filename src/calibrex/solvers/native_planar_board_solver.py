"""Native adapter for extracted multi-plane LiDAR-camera observations."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from calibrex.core.config import CalibrationConfig, FactorConfig
from calibrex.core.frames import FrameGraph
from calibrex.core.geometry import SE3
from calibrex.core.provenance import sha256_path
from calibrex.core.result import MetricResult, ObservabilityResult
from calibrex.data.inspect import DatasetInspection
from calibrex.solvers.base import SolverAdapter, SolverAdapterResult
from calibrex.solvers.horn_point_lidar_camera_solver import (
    HornPointLidarCameraSolver,
    HornPointObservation,
    HornPointSolverOptions,
)
from calibrex.solvers.planar_board_lidar_camera_solver import (
    OrientedPlane,
    PlanarBoardLidarCameraSolver,
    PlanarBoardObservation,
    PlanarBoardSolverOptions,
)
from calibrex.solvers.point_plane_lidar_camera_solver import (
    PointPlaneLidarCameraSolver,
    PointPlaneObservation,
    PointPlaneSolverOptions,
)

NATIVE_PLANAR_BOARD_BACKEND = "native_planar_board"
ACFR_VLP_FORMAT = "acfr_vlp_poses_v1"
ACFR_VLP_SOURCE_COMMIT = "ecda574ed902913fbc2ae50f92ad3356f318431c"
ACFR_VLP_SOURCE_URL = (
    "https://raw.githubusercontent.com/acfr/cam_lidar_calibration/"
    f"{ACFR_VLP_SOURCE_COMMIT}/data/vlp/poses.csv"
)


class NativePlanarBoardSolver(SolverAdapter):
    """Run the ROS-independent plane solver on adapter-extracted observations."""

    backend = NATIVE_PLANAR_BOARD_BACKEND

    def solve(
        self,
        config: CalibrationConfig,
        frame_graph: FrameGraph,
        inspection: DatasetInspection,
    ) -> SolverAdapterResult:
        factor = _factor(config)
        options = factor.options if factor is not None else {}
        poses_path = _poses_path(config, options)
        if not poses_path.exists():
            return SolverAdapterResult(
                backend=self.backend,
                available=True,
                status="missing_observations",
                metrics={
                    "planar_board_observation_file_available": MetricResult(
                        value=0.0,
                        unit="bool",
                        grade="fail",
                        reason=f"plane observation file is missing: {poses_path}",
                    )
                },
                warnings=[f"plane observation file is missing: {poses_path}"],
            )
        try:
            observations = read_acfr_vlp_plane_observations(poses_path)
            point_plane_observations = read_acfr_vlp_point_plane_observations(poses_path)
            horn_observations = read_acfr_vlp_horn_point_observations(poses_path)
        except ValueError as exc:
            return SolverAdapterResult(
                backend=self.backend,
                available=True,
                status="invalid_observations",
                warnings=[str(exc)],
            )

        solver_options = PlanarBoardSolverOptions(
            holdout_ratio=config.evaluation.holdout_ratio,
            split_seed=config.solver.seed or 0,
            max_iterations=config.solver.max_iterations,
            convergence_tolerance=config.solver.convergence_tolerance,
            normal_sign_policy=_normal_sign_policy(options),
        )
        initial = _initial_camera_lidar(frame_graph)
        solved = PlanarBoardLidarCameraSolver().solve(
            observations,
            initial_transform=initial,
            options=solver_options,
        )
        point_plane_options = PointPlaneSolverOptions(
            holdout_ratio=config.evaluation.holdout_ratio,
            split_seed=config.solver.seed or 0,
            normal_scale_m=_float_option(options, "point_plane_normal_scale_m", 1.0),
            huber_delta_m=_float_option(options, "point_plane_huber_delta_m", 0.05),
            max_iterations=config.solver.max_iterations,
            convergence_tolerance=config.solver.convergence_tolerance,
        )
        point_plane = PointPlaneLidarCameraSolver().solve(
            point_plane_observations,
            point_plane_options,
        )
        horn = HornPointLidarCameraSolver().solve(
            horn_observations,
            HornPointSolverOptions(
                holdout_ratio=config.evaluation.holdout_ratio,
                split_seed=config.solver.seed or 0,
                huber_delta_m=_float_option(options, "horn_point_huber_delta_m", 0.05),
                max_iterations=config.solver.max_iterations,
                convergence_tolerance=config.solver.convergence_tolerance,
            ),
        )
        point_plane_detectable = sum(probe.detectable is True for probe in point_plane.probes)
        point_plane_probe_count = len(point_plane.probes)
        point_plane_detectable_fraction = (
            point_plane_detectable / point_plane_probe_count if point_plane_probe_count else None
        )
        point_plane_holdout = point_plane.holdout_evaluation
        point_plane_pass = (
            point_plane.status == "converged"
            and point_plane.joint_rank == 6
            and point_plane_holdout.center_rmse_m is not None
            and point_plane_holdout.center_rmse_m
            <= _float_option(options, "max_point_plane_holdout_center_rmse_m", 0.03)
            and point_plane_holdout.normal_rmse_deg is not None
            and point_plane_holdout.normal_rmse_deg
            <= _float_option(options, "max_point_plane_holdout_normal_rmse_deg", 3.0)
            and point_plane_detectable_fraction is not None
            and point_plane_detectable_fraction
            >= _float_option(options, "min_point_plane_known_bad_detectable_fraction", 0.8)
        )
        horn_detectable = sum(probe.detectable is True for probe in horn.probes)
        horn_probe_count = len(horn.probes)
        horn_detectable_fraction = horn_detectable / horn_probe_count if horn_probe_count else None
        horn_holdout = horn.holdout_evaluation
        horn_pass = (
            horn.status == "converged"
            and horn.transform_camera_lidar is not None
            and horn.centered_geometry_rank >= 2
            and horn.joint_rank == 6
            and horn_holdout.point_rmse_m is not None
            and horn_holdout.point_rmse_m
            <= _float_option(options, "max_horn_point_holdout_rmse_m", 0.03)
            and horn_detectable_fraction is not None
            and horn_detectable_fraction
            >= _float_option(options, "min_horn_point_known_bad_detectable_fraction", 0.8)
            and horn.quaternion_normalized_eigengap is not None
            and horn.quaternion_normalized_eigengap
            >= _float_option(options, "min_horn_point_normalized_eigengap", 1.0e-4)
        )
        holdout = solved.holdout_evaluation
        detectable = sum(probe.detectable is True for probe in solved.probes)
        probe_count = len(solved.probes)
        detectable_fraction = detectable / probe_count if probe_count else None
        converged = solved.status == "converged" and solved.transform_camera_lidar is not None
        evidence_pass = (
            converged
            and holdout.normal_rmse_deg is not None
            and holdout.normal_rmse_deg
            <= _float_option(options, "max_holdout_normal_rmse_deg", 3.0)
            and holdout.offset_rmse_m is not None
            and holdout.offset_rmse_m <= _float_option(options, "max_holdout_offset_rmse_m", 0.05)
            and detectable_fraction is not None
            and detectable_fraction
            >= _float_option(options, "min_known_bad_detectable_fraction", 0.8)
        )
        metrics = {
            "planar_board_observation_count": MetricResult(
                value=float(len(observations)), unit="captures", grade="pass"
            ),
            "planar_board_normal_rmse_deg": MetricResult(
                train=solved.train_evaluation.normal_rmse_deg,
                holdout=holdout.normal_rmse_deg,
                unit="deg",
                grade="pass" if evidence_pass else "warn",
            ),
            "planar_board_offset_rmse_m": MetricResult(
                train=solved.train_evaluation.offset_rmse_m,
                holdout=holdout.offset_rmse_m,
                unit="m",
                grade="pass" if evidence_pass else "warn",
            ),
            "planar_board_known_bad_detectable_fraction": MetricResult(
                value=detectable_fraction,
                unit="fraction",
                grade="pass" if evidence_pass else "warn",
                reason=f"{detectable}/{probe_count} held-out 6-DoF controls detected",
            ),
            "point_plane_center_rmse_m": MetricResult(
                train=point_plane.train_evaluation.center_rmse_m,
                holdout=point_plane_holdout.center_rmse_m,
                unit="m",
                grade="pass" if point_plane_pass else "warn",
            ),
            "point_plane_normal_rmse_deg": MetricResult(
                train=point_plane.train_evaluation.normal_rmse_deg,
                holdout=point_plane_holdout.normal_rmse_deg,
                unit="deg",
                grade="pass" if point_plane_pass else "warn",
            ),
            "point_plane_offset_rmse_m": MetricResult(
                train=point_plane.train_evaluation.plane_offset_rmse_m,
                holdout=point_plane_holdout.plane_offset_rmse_m,
                unit="m",
                grade="warn",
                reason="induced plane-offset closure; no acceptance threshold is declared",
            ),
            "point_plane_known_bad_detectable_fraction": MetricResult(
                value=point_plane_detectable_fraction,
                unit="fraction",
                grade="pass" if point_plane_pass else "warn",
                reason=(
                    f"{point_plane_detectable}/{point_plane_probe_count} held-out "
                    "centre/normal 6-DoF controls detected"
                ),
            ),
            "point_plane_joint_rank": MetricResult(
                value=float(point_plane.joint_rank),
                unit="rank",
                grade="pass" if point_plane.joint_rank == 6 else "warn",
            ),
            "point_plane_joint_condition_number": MetricResult(
                value=point_plane.joint_condition_number,
                unit="ratio",
                grade="pass" if point_plane_pass else "warn",
            ),
            "horn_point_rmse_m": MetricResult(
                train=horn.train_evaluation.point_rmse_m,
                holdout=horn_holdout.point_rmse_m,
                unit="m",
                grade="pass" if horn_pass else "warn",
            ),
            "horn_point_known_bad_detectable_fraction": MetricResult(
                value=horn_detectable_fraction,
                unit="fraction",
                grade="pass" if horn_pass else "warn",
                reason=(
                    f"{horn_detectable}/{horn_probe_count} held-out point-only "
                    "6-DoF controls detected"
                ),
            ),
            "horn_point_joint_rank": MetricResult(
                value=float(horn.joint_rank),
                unit="rank",
                grade="pass" if horn.joint_rank == 6 else "warn",
            ),
            "horn_point_joint_condition_number": MetricResult(
                value=horn.joint_condition_number,
                unit="ratio",
                grade="pass" if horn_pass else "warn",
            ),
            "horn_point_quaternion_normalized_eigengap": MetricResult(
                value=horn.quaternion_normalized_eigengap,
                unit="ratio",
                grade="pass" if horn_pass else "warn",
            ),
            "horn_point_rms_scale_ratio": MetricResult(
                value=horn.rms_scale_ratio_camera_over_lidar,
                unit="ratio",
                grade="warn",
                reason="unit-consistency diagnostic; rigid extrinsic scale remains fixed to one",
            ),
        }
        comparison = _transform_comparison(
            solved.transform_camera_lidar,
            point_plane.transform_camera_lidar,
        )
        metrics["point_plane_vs_plane_translation_delta_m"] = MetricResult(
            value=comparison["translation_delta_m"],
            unit="m",
            grade="warn",
            reason="independent baseline delta; no metrology threshold is declared",
        )
        metrics["point_plane_vs_plane_rotation_delta_deg"] = MetricResult(
            value=comparison["rotation_delta_deg"],
            unit="deg",
            grade="warn",
            reason="independent baseline delta; no metrology threshold is declared",
        )
        horn_vs_plane = _transform_comparison(
            solved.transform_camera_lidar,
            horn.transform_camera_lidar,
        )
        horn_vs_point_plane = _transform_comparison(
            point_plane.transform_camera_lidar,
            horn.transform_camera_lidar,
        )
        metrics["horn_point_vs_plane_translation_delta_m"] = MetricResult(
            value=horn_vs_plane["translation_delta_m"],
            unit="m",
            grade="warn",
            reason="normal-free baseline delta; no metrology threshold is declared",
        )
        metrics["horn_point_vs_plane_rotation_delta_deg"] = MetricResult(
            value=horn_vs_plane["rotation_delta_deg"],
            unit="deg",
            grade="warn",
            reason="normal-free baseline delta; no metrology threshold is declared",
        )
        metrics["horn_point_vs_point_plane_translation_delta_m"] = MetricResult(
            value=horn_vs_point_plane["translation_delta_m"],
            unit="m",
            grade="warn",
            reason="normal-free versus centre+normal delta; no threshold is declared",
        )
        metrics["horn_point_vs_point_plane_rotation_delta_deg"] = MetricResult(
            value=horn_vs_point_plane["rotation_delta_deg"],
            unit="deg",
            grade="warn",
            reason="normal-free versus centre+normal delta; no threshold is declared",
        )
        warnings = []
        if not evidence_pass:
            warnings.append("multi-plane evidence did not pass all default gates")
        if not point_plane_pass:
            warnings.append("point+plane independent baseline did not pass all default gates")
        if not horn_pass:
            warnings.append("Horn point-only independent baseline did not pass all default gates")
        transforms = (
            {"T_camera0_lidar0": solved.transform_camera_lidar}
            if solved.transform_camera_lidar is not None
            else {}
        )
        return SolverAdapterResult(
            backend=self.backend,
            available=True,
            status="pass" if evidence_pass else solved.status,
            metrics=metrics,
            transforms=transforms,
            observability=ObservabilityResult(
                rank=solved.normal_rank,
                condition_number=solved.normal_condition_number,
                weak_directions=[] if solved.normal_rank == 3 else ["plane_normal_span"],
                grade="pass" if solved.normal_rank == 3 else "fail",
            ),
            provenance={
                "metrics_origin": "recomputed",
                "data_verified": True,
                "raw_input_files": [
                    {
                        "path": str(poses_path),
                        "role": "extracted_board_center_normal_observations",
                        "sha256": sha256_path(poses_path),
                        "size_bytes": poses_path.stat().st_size,
                        "source_url": ACFR_VLP_SOURCE_URL,
                    }
                ],
                "native_planar_board": {
                    "result": solved.as_dict(),
                    "input_format": ACFR_VLP_FORMAT,
                    "extractor": "acfr/cam_lidar_calibration upstream feature extractor",
                    "extractor_commit": ACFR_VLP_SOURCE_COMMIT,
                    "extractor_license_spdx": "Apache-2.0",
                    "external_code_executed": False,
                },
                "native_point_plane_baseline": {
                    "result": point_plane.as_dict(),
                    "comparison_to_plane_only": comparison,
                    "evidence_pass": point_plane_pass,
                    "input_format": ACFR_VLP_FORMAT,
                    "source_rows": {
                        "camera_center": 0,
                        "camera_normal": 1,
                        "lidar_center": 6,
                        "lidar_normal": 7,
                        "rows_per_capture": 19,
                    },
                    "extractor": "acfr/cam_lidar_calibration upstream feature extractor",
                    "extractor_commit": ACFR_VLP_SOURCE_COMMIT,
                    "extractor_license_spdx": "Apache-2.0",
                    "external_code_executed": False,
                },
                "native_horn_point_baseline": {
                    "result": horn.as_dict(),
                    "comparison_to_plane_only": horn_vs_plane,
                    "comparison_to_point_plane": horn_vs_point_plane,
                    "evidence_pass": horn_pass,
                    "common_capture_split": {
                        "matches_plane_only": (
                            list(horn.train_frame_ids) == list(solved.train_frame_ids)
                            and list(horn.holdout_frame_ids) == list(solved.holdout_frame_ids)
                        ),
                        "matches_point_plane": (
                            horn.train_frame_ids == point_plane.train_frame_ids
                            and horn.holdout_frame_ids == point_plane.holdout_frame_ids
                        ),
                    },
                    "input_format": ACFR_VLP_FORMAT,
                    "source_rows": {
                        "camera_center": 0,
                        "lidar_center": 6,
                        "rows_per_capture": 19,
                    },
                    "extractor": "acfr/cam_lidar_calibration upstream feature extractor",
                    "extractor_commit": ACFR_VLP_SOURCE_COMMIT,
                    "extractor_license_spdx": "Apache-2.0",
                    "external_code_executed": False,
                },
            },
            warnings=warnings,
        )


def read_acfr_vlp_plane_observations(path: str | Path) -> list[PlanarBoardObservation]:
    """Read ACFR's public VLP quick-start feature rows without importing ROS code."""

    rows = _read_acfr_vlp_rows(path)
    observations: list[PlanarBoardObservation] = []
    for start in range(0, len(rows), 19):
        camera_center = np.asarray(rows[start], dtype=float) / 1000.0
        camera_normal = np.asarray(rows[start + 1], dtype=float)
        lidar_center = np.asarray(rows[start + 6], dtype=float) / 1000.0
        lidar_normal = np.asarray(rows[start + 7], dtype=float)
        observations.append(
            PlanarBoardObservation(
                frame_id=f"acfr-vlp-{start // 19 + 1:03d}",
                camera_plane=OrientedPlane(
                    tuple(camera_normal), -float(camera_normal @ camera_center)
                ),
                lidar_plane=OrientedPlane(tuple(lidar_normal), -float(lidar_normal @ lidar_center)),
            )
        )
    return observations


def read_acfr_vlp_point_plane_observations(
    path: str | Path,
) -> list[PointPlaneObservation]:
    """Read ACFR checkerboard centre/normal pairs for an independent baseline."""

    rows = _read_acfr_vlp_rows(path)
    observations: list[PointPlaneObservation] = []
    for start in range(0, len(rows), 19):
        observations.append(
            PointPlaneObservation(
                frame_id=f"acfr-vlp-{start // 19 + 1:03d}",
                camera_center_m=_millimetres_to_metres(rows[start]),
                camera_normal=rows[start + 1],
                lidar_center_m=_millimetres_to_metres(rows[start + 6]),
                lidar_normal=rows[start + 7],
            )
        )
    return observations


def read_acfr_vlp_horn_point_observations(
    path: str | Path,
) -> list[HornPointObservation]:
    """Read only ACFR checkerboard centre correspondences for Horn alignment."""

    rows = _read_acfr_vlp_rows(path)
    observations: list[HornPointObservation] = []
    for start in range(0, len(rows), 19):
        observations.append(
            HornPointObservation(
                frame_id=f"acfr-vlp-{start // 19 + 1:03d}",
                camera_point_m=_millimetres_to_metres(rows[start]),
                lidar_point_m=_millimetres_to_metres(rows[start + 6]),
            )
        )
    return observations


def _read_acfr_vlp_rows(path: str | Path) -> list[tuple[float, float, float]]:
    input_path = Path(path)
    rows: list[tuple[float, float, float]] = []
    with input_path.open(newline="", encoding="utf-8") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if len(row) != 3:
                raise ValueError(f"{input_path}:{line_number}: expected three CSV values")
            try:
                rows.append((float(row[0]), float(row[1]), float(row[2])))
            except ValueError as exc:
                raise ValueError(f"{input_path}:{line_number}: non-numeric CSV value") from exc
    if not rows or len(rows) % 19 != 0:
        raise ValueError(f"{input_path}: ACFR poses must contain complete 19-row captures")
    return rows


def _millimetres_to_metres(value: tuple[float, float, float]) -> tuple[float, float, float]:
    return (value[0] / 1000.0, value[1] / 1000.0, value[2] / 1000.0)


def _transform_comparison(
    reference: SE3 | None,
    candidate: SE3 | None,
) -> dict[str, float | None]:
    if reference is None or candidate is None:
        return {"translation_delta_m": None, "rotation_delta_deg": None}
    relative = reference.inverse().compose(candidate)
    translation_delta = math.sqrt(sum(value * value for value in relative.translation_m))
    scalar = min(1.0, abs(relative.rotation_quat_xyzw[3]))
    return {
        "translation_delta_m": translation_delta,
        "rotation_delta_deg": math.degrees(2.0 * math.acos(scalar)),
    }


def _factor(config: CalibrationConfig) -> FactorConfig | None:
    return config.pipeline.factors.get("planar_board_plane_correspondence")


def _poses_path(config: CalibrationConfig, options: dict[str, Any]) -> Path:
    configured = options.get("poses_file", "poses.csv")
    path = Path(str(configured))
    return path if path.is_absolute() else Path(config.dataset.path) / path


def _normal_sign_policy(
    options: dict[str, Any],
) -> Literal["initial_alignment", "preserve"]:
    policy = str(options.get("normal_sign_policy", "preserve"))
    if policy not in {"initial_alignment", "preserve"}:
        raise ValueError(f"unsupported normal_sign_policy: {policy}")
    return cast(Literal["initial_alignment", "preserve"], policy)


def _float_option(options: dict[str, Any], name: str, default: float) -> float:
    return float(options.get(name, default))


def _initial_camera_lidar(frame_graph: FrameGraph) -> SE3 | None:
    lidar = frame_graph.nodes.get("lidar0")
    if lidar is not None and lidar.parent == "camera0":
        return lidar.transform_to_parent
    return None
