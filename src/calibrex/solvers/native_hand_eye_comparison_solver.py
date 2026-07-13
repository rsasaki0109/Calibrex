"""Native comparison adapter for hand-eye and robot-world/hand-eye solvers."""

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
from calibrex.solvers.andreff_hand_eye_solver import (
    AndreffHandEyeOptions,
    AndreffHandEyeResult,
    AndreffHandEyeSolver,
)
from calibrex.solvers.base import SolverAdapter, SolverAdapterResult
from calibrex.solvers.daniilidis_hand_eye_solver import (
    DaniilidisHandEyeOptions,
    DaniilidisHandEyeResult,
    DaniilidisHandEyeSolver,
)
from calibrex.solvers.horaud_dornaika_hand_eye_solver import (
    HoraudDornaikaHandEyeOptions,
    HoraudDornaikaHandEyeResult,
    HoraudDornaikaHandEyeSolver,
)
from calibrex.solvers.park_martin_hand_eye_solver import (
    HandEyeMotionPair,
    ParkMartinHandEyeOptions,
    ParkMartinHandEyeResult,
    ParkMartinHandEyeSolver,
)
from calibrex.solvers.shah_robot_world_hand_eye_solver import (
    RobotWorldHandEyePosePair,
    ShahRobotWorldHandEyeOptions,
    ShahRobotWorldHandEyeResult,
    ShahRobotWorldHandEyeSolver,
)
from calibrex.solvers.tsai_lenz_hand_eye_solver import (
    TsaiLenzHandEyeOptions,
    TsaiLenzHandEyeResult,
    TsaiLenzHandEyeSolver,
    evaluate_hand_eye_known_bad_probes,
)

NATIVE_HAND_EYE_COMPARISON_BACKEND = "native_hand_eye_comparison"


class NativeHandEyeComparisonSolver(SolverAdapter):
    """Run five AX=XB baselines plus Shah's absolute-pose AX=YB baseline."""

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
        absolute_poses = tuple(
            RobotWorldHandEyePosePair(item.pair_id, item.pose_a, item.pose_b)
            for item in dataset.absolute_pose_pairs
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
        horaud_options = HoraudDornaikaHandEyeOptions(
            holdout_ratio=holdout_ratio, split_seed=split_seed
        )
        andreff_options = AndreffHandEyeOptions(
            holdout_ratio=holdout_ratio,
            split_seed=split_seed,
            min_rotation_rad=tsai_options.min_rotation_rad,
            minimum_rotation_width=float(
                options.get("min_andreff_rotation_width", 1.0e-4)
            ),
            max_rotation_nullspace_ratio=float(
                options.get("max_andreff_rotation_nullspace_ratio", 0.25)
            ),
        )
        shah_options = ShahRobotWorldHandEyeOptions(
            holdout_ratio=holdout_ratio,
            split_seed=split_seed,
            minimum_rotation_normalized_gap=float(
                options.get("min_shah_rotation_normalized_gap", 1.0e-3)
            ),
            max_translation_condition_number=float(
                options.get("max_shah_translation_condition_number", 1.0e8)
            ),
            max_so3_projection_correction_frobenius=float(
                options.get("max_shah_so3_projection_correction_frobenius", 0.05)
            ),
        )
        park = ParkMartinHandEyeSolver().solve(motions, park_options)
        tsai = TsaiLenzHandEyeSolver().solve(motions, tsai_options)
        dual = DaniilidisHandEyeSolver().solve(motions, dual_options)
        horaud = HoraudDornaikaHandEyeSolver().solve(motions, horaud_options)
        andreff = AndreffHandEyeSolver().solve(motions, andreff_options)
        shah = ShahRobotWorldHandEyeSolver().solve(absolute_poses, shah_options)
        metrics = _comparison_metrics(
            dataset,
            motions,
            park,
            tsai,
            dual,
            horaud,
            andreff,
            shah,
            tsai_options,
            options,
        )
        all_converged = all(
            transform is not None
            for transform in (
                park.transform_x,
                tsai.transform_x,
                dual.transform_x,
                horaud.transform_x,
                andreff.transform_x,
                shah.transform_x,
                shah.transform_y,
            )
        )
        all_gates_pass = all(metric.grade != "fail" for metric in metrics.values())
        horaud_observable = (
            metrics[
                "hand_eye_horaud_dornaika_quaternion_normalized_eigengap"
            ].grade
            == "pass"
        )
        andreff_observable = all(
            metrics[name].grade == "pass"
            for name in (
                "hand_eye_andreff_rotation_observable_rank",
                "hand_eye_andreff_rotation_minimum_width",
                "hand_eye_andreff_rotation_nullspace_ratio",
                "hand_eye_andreff_translation_rank",
                "hand_eye_andreff_so3_projection_correction_frobenius",
            )
        )
        shah_observable = all(
            metrics[name].grade == "pass"
            for name in (
                "robot_world_hand_eye_shah_rotation_dominant_multiplicity",
                "robot_world_hand_eye_shah_rotation_normalized_gap",
                "robot_world_hand_eye_shah_translation_rank",
                "robot_world_hand_eye_shah_translation_condition_number",
                "robot_world_hand_eye_shah_so3_projection_correction_frobenius_max",
            )
        )
        selected = dual.transform_x
        warnings = _comparison_warnings(metrics, dataset)
        archive_sha256 = sha256_path(archive_path)
        return SolverAdapterResult(
            backend=self.backend,
            available=True,
            status=(
                "pass"
                if all_converged and all_gates_pass and not warnings
                else "inconclusive"
            ),
            metrics=metrics,
            transforms={
                **({"T_hand_eye": selected} if selected is not None else {}),
                **(
                    {"T_robot_world": shah.transform_y}
                    if shah.transform_y is not None
                    else {}
                ),
            },
            observability=ObservabilityResult(
                rank=dual.linear_rank,
                condition_number=dual.observable_condition_number,
                weak_directions=[
                    *([] if horaud_observable else ["horaud_quaternion_minimum_width"]),
                    *([] if andreff_observable else ["andreff_kronecker_observability"]),
                    *([] if shah_observable else ["shah_robot_world_hand_eye_observability"]),
                    *([] if not warnings else ["hand_eye_perturbation_power"]),
                ],
                grade=(
                    "pass"
                    if all_converged
                    and horaud_observable
                    and andreff_observable
                    and shah_observable
                    else "fail"
                ),
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
                        "horaud_dornaika": horaud.as_dict(),
                        "andreff": andreff.as_dict(),
                        "shah_robot_world_hand_eye": shah.as_dict(),
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
    horaud: HoraudDornaikaHandEyeResult,
    andreff: AndreffHandEyeResult,
    shah: ShahRobotWorldHandEyeResult,
    probe_options: TsaiLenzHandEyeOptions,
    factor_options: dict[str, Any],
) -> dict[str, MetricResult]:
    max_rotation = float(factor_options.get("max_holdout_rotation_rmse_deg", 2.0))
    max_translation = float(factor_options.get("max_holdout_translation_rmse_m", 0.03))
    min_detectable = float(factor_options.get("min_known_bad_detectable_fraction", 0.75))
    min_horaud_gap = float(
        factor_options.get("min_horaud_quaternion_normalized_eigengap", 1.0e-3)
    )
    min_andreff_width = float(factor_options.get("min_andreff_rotation_width", 1.0e-4))
    max_andreff_projection = float(
        factor_options.get("max_andreff_so3_projection_correction_frobenius", 0.05)
    )
    max_andreff_nullspace_ratio = float(
        factor_options.get("max_andreff_rotation_nullspace_ratio", 0.25)
    )
    min_shah_gap = float(factor_options.get("min_shah_rotation_normalized_gap", 1.0e-3))
    max_shah_projection = float(
        factor_options.get("max_shah_so3_projection_correction_frobenius", 0.05)
    )
    max_shah_condition = float(
        factor_options.get("max_shah_translation_condition_number", 1.0e8)
    )
    shah_projection_values = tuple(
        value
        for value in (
            shah.rotation_x_projection_correction_frobenius,
            shah.rotation_y_projection_correction_frobenius,
        )
        if value is not None
    )
    shah_projection_max = max(shah_projection_values) if shah_projection_values else None
    common_split = len(
        {
            (result.train_pair_ids, result.holdout_pair_ids)
            for result in (park, tsai, dual, horaud, andreff)
        }
    ) == 1
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
        "hand_eye_horaud_dornaika_rotation_axis_rank": MetricResult(
            value=float(horaud.rotation_axis_rank),
            unit="rank",
            grade="pass" if horaud.rotation_axis_rank >= 2 else "fail",
            reason="at least two independent rotation-axis directions are required",
        ),
        "hand_eye_horaud_dornaika_quaternion_normalized_eigengap": MetricResult(
            value=horaud.quaternion_normalized_eigengap,
            unit="ratio",
            grade=(
                "pass"
                if horaud.quaternion_normalized_eigengap is not None
                and horaud.quaternion_normalized_eigengap >= min_horaud_gap
                else "fail"
            ),
            reason=f"closed-form minimum-width gate >= {min_horaud_gap:g}",
        ),
        "hand_eye_common_split_consistent": MetricResult(
            value=float(common_split),
            unit="bool",
            grade="pass" if common_split else "fail",
            reason="all hand-eye estimators must use identical train and holdout pair IDs",
        ),
        "hand_eye_andreff_rotation_observable_rank": MetricResult(
            value=float(andreff.rotation_observable_rank),
            unit="rank",
            grade="pass" if andreff.rotation_observable_rank == 8 else "fail",
            reason="equation (13) requires eight observable directions and one kernel",
        ),
        "hand_eye_andreff_rotation_minimum_width": MetricResult(
            value=andreff.rotation_minimum_width,
            unit="ratio",
            grade=(
                "pass"
                if andreff.rotation_minimum_width is not None
                and andreff.rotation_minimum_width >= min_andreff_width
                else "fail"
            ),
            reason=f"eighth-to-first singular-value ratio >= {min_andreff_width:g}",
        ),
        "hand_eye_andreff_rotation_nullspace_ratio": MetricResult(
            value=andreff.rotation_nullspace_ratio,
            unit="ratio",
            grade=(
                "pass"
                if andreff.rotation_nullspace_ratio is not None
                and andreff.rotation_nullspace_ratio <= max_andreff_nullspace_ratio
                else "fail"
            ),
            reason=f"kernel-to-eighth singular-value ratio <= {max_andreff_nullspace_ratio:g}",
        ),
        "hand_eye_andreff_translation_rank": MetricResult(
            value=float(andreff.translation_rank),
            unit="rank",
            grade="pass" if andreff.translation_rank == 3 else "fail",
            reason="conditional translation system must constrain all three directions",
        ),
        "hand_eye_andreff_so3_projection_correction_frobenius": MetricResult(
            value=andreff.so3_projection_correction_frobenius,
            unit="frobenius",
            grade=(
                "pass"
                if andreff.so3_projection_correction_frobenius is not None
                and andreff.so3_projection_correction_frobenius
                <= max_andreff_projection
                else "fail"
            ),
            reason=f"determinant-normalized kernel projection <= {max_andreff_projection:g}",
        ),
        "robot_world_hand_eye_pose_pair_count": MetricResult(
            value=float(len(dataset.absolute_pose_pairs)),
            unit="pairs",
            grade="pass" if len(dataset.absolute_pose_pairs) >= 3 else "fail",
            reason="one-to-one synchronized absolute hand/eye pose pairs",
        ),
        "robot_world_hand_eye_shah_rotation_dominant_multiplicity": MetricResult(
            value=float(shah.rotation_dominant_multiplicity),
            unit="multiplicity",
            grade="pass" if shah.rotation_dominant_multiplicity == 1 else "fail",
            reason="Shah rotation solution requires one dominant singular-vector pair",
        ),
        "robot_world_hand_eye_shah_rotation_normalized_gap": MetricResult(
            value=shah.rotation_normalized_gap,
            unit="ratio",
            grade=(
                "pass"
                if shah.rotation_normalized_gap is not None
                and shah.rotation_normalized_gap >= min_shah_gap
                else "fail"
            ),
            reason=f"dominant-to-second Kronecker singular-value gap >= {min_shah_gap:g}",
        ),
        "robot_world_hand_eye_shah_translation_rank": MetricResult(
            value=float(shah.translation_rank),
            unit="rank",
            grade="pass" if shah.translation_rank == 6 else "fail",
            reason="conditional [t_Y; t_X] system must constrain all six translations",
        ),
        "robot_world_hand_eye_shah_translation_condition_number": MetricResult(
            value=shah.translation_condition_number,
            grade=(
                "pass"
                if shah.translation_condition_number is not None
                and shah.translation_condition_number <= max_shah_condition
                else "fail"
            ),
            reason=f"conditional translation condition number <= {max_shah_condition:g}",
        ),
        "robot_world_hand_eye_shah_so3_projection_correction_frobenius_max": MetricResult(
            value=shah_projection_max,
            unit="frobenius",
            grade=(
                "pass"
                if len(shah_projection_values) == 2
                and shah_projection_max is not None
                and shah_projection_max <= max_shah_projection
                else "fail"
            ),
            reason=(
                "determinant-normalized rotations require correction "
                f"<= {max_shah_projection:g}"
            ),
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
        "horaud_dornaika": horaud.probes,
        "andreff": andreff.probes,
    }
    for name, result in (
        ("park_martin", park),
        ("tsai_lenz", tsai),
        ("daniilidis", dual),
        ("horaud_dornaika", horaud),
        ("andreff", andreff),
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
    shah_rotation = shah.holdout_evaluation.rotation_closure_rmse_deg
    shah_translation = shah.holdout_evaluation.translation_closure_rmse_m
    shah_closure_pass = (
        shah_rotation is not None
        and shah_rotation <= max_rotation
        and shah_translation is not None
        and shah_translation <= max_translation
    )
    shah_detectable = [probe.detectable for probe in shah.probes if probe.detectable is not None]
    shah_fraction = (
        sum(value is True for value in shah_detectable) / len(shah_detectable)
        if shah_detectable
        else None
    )
    metrics["robot_world_hand_eye_shah_holdout_rotation_rmse_deg"] = MetricResult(
        value=shah_rotation,
        unit="deg",
        grade="pass" if shah_closure_pass else "fail",
    )
    metrics["robot_world_hand_eye_shah_holdout_translation_rmse_m"] = MetricResult(
        value=shah_translation,
        unit="m",
        grade="pass" if shah_closure_pass else "fail",
    )
    metrics["robot_world_hand_eye_shah_known_bad_detectable_fraction"] = MetricResult(
        value=shah_fraction,
        unit="fraction",
        grade=(
            "pass"
            if shah_fraction is not None and shah_fraction >= min_detectable
            else "warn"
        ),
        reason="held-out signed six-DoF perturbations of both X and Y",
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
