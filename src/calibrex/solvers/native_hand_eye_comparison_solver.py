"""Native comparison adapter for three independent hand-eye solvers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from calibrex.core.config import CalibrationConfig
from calibrex.core.frames import FrameGraph
from calibrex.core.provenance import sha256_path
from calibrex.core.result import MetricResult, ObservabilityResult
from calibrex.data.ethz_hand_eye import (
    ETHZ_HAND_EYE_COMMIT,
    ETHZ_ROBOT_ARM_REAL_SHA256,
    ETHZ_ROBOT_ARM_REAL_URL,
    HandEyeMotionDataset,
    read_ethz_robot_arm_hand_eye_motions,
)
from calibrex.data.inspect import DatasetInspection
from calibrex.solvers.base import SolverAdapter, SolverAdapterResult
from calibrex.solvers.daniilidis_hand_eye_solver import (
    DaniilidisHandEyeOptions,
    DaniilidisHandEyeResult,
    DaniilidisHandEyeSolver,
)
from calibrex.solvers.park_martin_hand_eye_solver import (
    HandEyeMotionPair,
    ParkMartinHandEyeOptions,
    ParkMartinHandEyeResult,
    ParkMartinHandEyeSolver,
)
from calibrex.solvers.tsai_lenz_hand_eye_solver import (
    TsaiLenzHandEyeOptions,
    TsaiLenzHandEyeResult,
    TsaiLenzHandEyeSolver,
    evaluate_hand_eye_known_bad_probes,
)

NATIVE_HAND_EYE_COMPARISON_BACKEND = "native_hand_eye_comparison"


class NativeHandEyeComparisonSolver(SolverAdapter):
    """Run Park-Martin, Tsai-Lenz, and Daniilidis on one motion protocol."""

    backend = NATIVE_HAND_EYE_COMPARISON_BACKEND

    def solve(
        self,
        config: CalibrationConfig,
        frame_graph: FrameGraph,
        inspection: DatasetInspection,
    ) -> SolverAdapterResult:
        del frame_graph, inspection
        options = _factor_options(config)
        archive_path = _archive_path(config, options)
        if not archive_path.exists():
            return SolverAdapterResult(
                backend=self.backend,
                available=True,
                status="missing_dataset",
                metrics={
                    "hand_eye_dataset_available": MetricResult(
                        value=0.0,
                        grade="fail",
                        unit="bool",
                        reason=f"hand-eye archive is missing: {archive_path}",
                    )
                },
                warnings=[f"hand-eye archive is missing: {archive_path}"],
            )
        dataset = read_ethz_robot_arm_hand_eye_motions(
            archive_path,
            maximum_time_delta_sec=float(options.get("maximum_time_delta_sec", 0.011)),
            motion_stride=int(options.get("motion_stride", 60)),
        )
        motions = tuple(
            HandEyeMotionPair(item.pair_id, item.motion_a, item.motion_b)
            for item in dataset.motions
        )
        holdout_ratio = config.evaluation.holdout_ratio
        split_seed = config.solver.seed or 0
        park_options = ParkMartinHandEyeOptions(
            holdout_ratio=holdout_ratio, split_seed=split_seed
        )
        tsai_options = TsaiLenzHandEyeOptions(
            holdout_ratio=holdout_ratio, split_seed=split_seed
        )
        dual_options = DaniilidisHandEyeOptions(
            holdout_ratio=holdout_ratio, split_seed=split_seed
        )
        park = ParkMartinHandEyeSolver().solve(motions, park_options)
        tsai = TsaiLenzHandEyeSolver().solve(motions, tsai_options)
        dual = DaniilidisHandEyeSolver().solve(motions, dual_options)
        metrics = _comparison_metrics(
            dataset, motions, park, tsai, dual, tsai_options, options
        )
        all_converged = all(
            transform is not None
            for transform in (park.transform_x, tsai.transform_x, dual.transform_x)
        )
        selected = dual.transform_x
        warnings = _comparison_warnings(metrics, dataset)
        archive_sha256 = sha256_path(archive_path)
        return SolverAdapterResult(
            backend=self.backend,
            available=True,
            status="pass" if all_converged and not warnings else "inconclusive",
            metrics=metrics,
            transforms={"T_hand_eye": selected} if selected is not None else {},
            observability=ObservabilityResult(
                rank=dual.linear_rank,
                condition_number=dual.observable_condition_number,
                weak_directions=([] if not warnings else ["hand_eye_perturbation_power"]),
                grade="pass" if all_converged else "fail",
            ),
            provenance={
                "metrics_origin": "recomputed",
                "data_verified": archive_sha256 == ETHZ_ROBOT_ARM_REAL_SHA256,
                "raw_input_files": [
                    {
                        "path": str(archive_path),
                        "role": "ethz_hand_eye_robot_arm_real_archive",
                        "sha256": archive_sha256,
                        "expected_sha256": ETHZ_ROBOT_ARM_REAL_SHA256,
                        "size_bytes": archive_path.stat().st_size,
                        "source_url": ETHZ_ROBOT_ARM_REAL_URL,
                    }
                ],
                "native_hand_eye_comparison": {
                    "dataset": dataset.as_dict(),
                    "dataset_doi": "10.3929/ethz-c-000788527",
                    "upstream_commit": ETHZ_HAND_EYE_COMMIT,
                    "upstream_license_spdx": "BSD-3-Clause",
                    "external_code_executed": False,
                    "selected_output": "daniilidis",
                    "results": {
                        "park_martin": park.as_dict(),
                        "tsai_lenz": tsai.as_dict(),
                        "daniilidis": dual.as_dict(),
                    },
                },
            },
            warnings=warnings,
        )


def _comparison_metrics(
    dataset: HandEyeMotionDataset,
    motions: tuple[HandEyeMotionPair, ...],
    park: ParkMartinHandEyeResult,
    tsai: TsaiLenzHandEyeResult,
    dual: DaniilidisHandEyeResult,
    probe_options: TsaiLenzHandEyeOptions,
    factor_options: dict[str, Any],
) -> dict[str, MetricResult]:
    max_rotation = float(factor_options.get("max_holdout_rotation_rmse_deg", 2.0))
    max_translation = float(factor_options.get("max_holdout_translation_rmse_m", 0.03))
    min_detectable = float(factor_options.get("min_known_bad_detectable_fraction", 0.75))
    metrics: dict[str, MetricResult] = {
        "hand_eye_motion_pair_count": MetricResult(
            value=float(len(dataset.motions)), unit="pairs", grade="pass"
        ),
        "hand_eye_absolute_pose_reuse_count": MetricResult(
            value=float(dataset.absolute_pose_reuse_count),
            unit="poses",
            grade="pass" if dataset.absolute_pose_reuse_count == 0 else "fail",
            reason="absolute poses must not leak between relative-motion pairs",
        ),
    }
    park_probes = (
        evaluate_hand_eye_known_bad_probes(
            [
                motion
                for motion in motions
                if motion.pair_id in set(park.holdout_pair_ids)
            ],
            park.transform_x,
            park.holdout_evaluation,
            probe_options,
        )
        if park.transform_x is not None
        else ()
    )
    probes_by_method = {
        "park_martin": park_probes,
        "tsai_lenz": tsai.probes,
        "daniilidis": dual.probes,
    }
    for name, result in (
        ("park_martin", park),
        ("tsai_lenz", tsai),
        ("daniilidis", dual),
    ):
        rotation = result.holdout_evaluation.rotation_closure_rmse_deg
        translation = result.holdout_evaluation.translation_closure_rmse_m
        probes = probes_by_method[name]
        fraction = (
            sum(probe.detectable is True for probe in probes) / len(probes)
            if probes
            else None
        )
        closure_pass = (
            rotation is not None
            and rotation <= max_rotation
            and translation is not None
            and translation <= max_translation
        )
        metrics[f"hand_eye_{name}_holdout_rotation_rmse_deg"] = MetricResult(
            value=rotation, unit="deg", grade="pass" if closure_pass else "fail"
        )
        metrics[f"hand_eye_{name}_holdout_translation_rmse_m"] = MetricResult(
            value=translation, unit="m", grade="pass" if closure_pass else "fail"
        )
        metrics[f"hand_eye_{name}_known_bad_detectable_fraction"] = MetricResult(
            value=fraction,
            unit="fraction",
            grade="pass" if fraction is not None and fraction >= min_detectable else "warn",
            reason="held-out ±5 deg / ±5 cm falsification power",
        )
    return metrics


def _comparison_warnings(
    metrics: dict[str, MetricResult], dataset: HandEyeMotionDataset
) -> list[str]:
    warnings = [
        name
        for name, metric in metrics.items()
        if metric.grade == "warn" and name.endswith("known_bad_detectable_fraction")
    ]
    result = []
    if warnings:
        result.append(
            "public hand-eye run has weak perturbation power: " + ", ".join(warnings)
        )
    if dataset.absolute_pose_reuse_count:
        result.append("absolute pose reuse detected across motion pairs")
    return result


def _factor_options(config: CalibrationConfig) -> dict[str, Any]:
    factor = config.pipeline.factors.get("hand_eye_motion_closure")
    return factor.options if factor is not None else {}


def _archive_path(config: CalibrationConfig, options: dict[str, Any]) -> Path:
    configured = Path(str(options.get("archive_file", "robot_arm_w_color_camera_real.zip")))
    return configured if configured.is_absolute() else Path(config.dataset.path) / configured
