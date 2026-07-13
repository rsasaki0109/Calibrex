"""Native adapter for extracted multi-plane LiDAR-camera observations."""

from __future__ import annotations

import csv
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
from calibrex.solvers.planar_board_lidar_camera_solver import (
    OrientedPlane,
    PlanarBoardLidarCameraSolver,
    PlanarBoardObservation,
    PlanarBoardSolverOptions,
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
        }
        warnings = [] if evidence_pass else ["multi-plane evidence did not pass all default gates"]
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
                        "role": "extracted_plane_observations",
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
            },
            warnings=warnings,
        )


def read_acfr_vlp_plane_observations(path: str | Path) -> list[PlanarBoardObservation]:
    """Read ACFR's public VLP quick-start feature rows without importing ROS code."""

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
                lidar_plane=OrientedPlane(
                    tuple(lidar_normal), -float(lidar_normal @ lidar_center)
                ),
            )
        )
    return observations


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
