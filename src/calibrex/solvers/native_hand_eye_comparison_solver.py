"""Native comparison adapter for hand-eye and robot-world/hand-eye solvers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from calibrex.core.config import CalibrationConfig
from calibrex.core.frames import FrameGraph
from calibrex.core.geometry import SE3
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
from calibrex.solvers.dornaika_horaud_nonlinear_robot_world_hand_eye_solver import (
    DornaikaHoraudNonlinearOptions,
    DornaikaHoraudNonlinearResult,
    DornaikaHoraudNonlinearSolver,
)
from calibrex.solvers.dornaika_horaud_robot_world_hand_eye_solver import (
    DornaikaHoraudRobotWorldHandEyeOptions,
    DornaikaHoraudRobotWorldHandEyeResult,
    DornaikaHoraudRobotWorldHandEyeSolver,
)
from calibrex.solvers.horaud_dornaika_hand_eye_solver import (
    HoraudDornaikaHandEyeOptions,
    HoraudDornaikaHandEyeResult,
    HoraudDornaikaHandEyeSolver,
)
from calibrex.solvers.li_robot_world_hand_eye_solver import (
    LiRobotWorldHandEyeOptions,
    LiRobotWorldHandEyeResult,
    LiRobotWorldHandEyeSolver,
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
    """Run five AX=XB baselines plus four absolute-pose AX=YB baselines."""

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
        park_options = ParkMartinHandEyeOptions(holdout_ratio=holdout_ratio, split_seed=split_seed)
        tsai_options = TsaiLenzHandEyeOptions(holdout_ratio=holdout_ratio, split_seed=split_seed)
        dual_options = DaniilidisHandEyeOptions(holdout_ratio=holdout_ratio, split_seed=split_seed)
        horaud_options = HoraudDornaikaHandEyeOptions(
            holdout_ratio=holdout_ratio, split_seed=split_seed
        )
        andreff_options = AndreffHandEyeOptions(
            holdout_ratio=holdout_ratio,
            split_seed=split_seed,
            min_rotation_rad=tsai_options.min_rotation_rad,
            minimum_rotation_width=float(options.get("min_andreff_rotation_width", 1.0e-4)),
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
        li_options = LiRobotWorldHandEyeOptions(
            holdout_ratio=holdout_ratio,
            split_seed=split_seed,
            max_condition_number=float(options.get("max_li_linear_condition_number", 1.0e8)),
            max_so3_projection_correction_frobenius=float(
                options.get("max_li_so3_projection_correction_frobenius", 0.05)
            ),
        )
        dornaika_options = DornaikaHoraudRobotWorldHandEyeOptions(
            holdout_ratio=holdout_ratio,
            split_seed=split_seed,
            minimum_rotation_normalized_gap=float(
                options.get("min_dornaika_horaud_rotation_normalized_gap", 1.0e-3)
            ),
            max_translation_condition_number=float(
                options.get("max_dornaika_horaud_translation_condition_number", 1.0e8)
            ),
        )
        nonlinear_options = DornaikaHoraudNonlinearOptions(
            holdout_ratio=holdout_ratio,
            split_seed=split_seed,
            max_iterations=int(options.get("dornaika_horaud_nonlinear_max_iterations", 50)),
            max_data_jacobian_condition_number=float(
                options.get(
                    "max_dornaika_horaud_nonlinear_data_jacobian_condition_number",
                    1.0e12,
                )
            ),
            max_so3_projection_correction_frobenius=float(
                options.get(
                    "max_dornaika_horaud_nonlinear_so3_projection_correction_frobenius",
                    1.0e-4,
                )
            ),
        )
        park = ParkMartinHandEyeSolver().solve(motions, park_options)
        tsai = TsaiLenzHandEyeSolver().solve(motions, tsai_options)
        dual = DaniilidisHandEyeSolver().solve(motions, dual_options)
        horaud = HoraudDornaikaHandEyeSolver().solve(motions, horaud_options)
        andreff = AndreffHandEyeSolver().solve(motions, andreff_options)
        shah = ShahRobotWorldHandEyeSolver().solve(absolute_poses, shah_options)
        li = LiRobotWorldHandEyeSolver().solve(absolute_poses, li_options)
        dornaika = DornaikaHoraudRobotWorldHandEyeSolver().solve(absolute_poses, dornaika_options)
        nonlinear = DornaikaHoraudNonlinearSolver().solve(absolute_poses, nonlinear_options)
        metrics = _comparison_metrics(
            dataset,
            motions,
            park,
            tsai,
            dual,
            horaud,
            andreff,
            shah,
            li,
            dornaika,
            nonlinear,
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
                li.transform_x,
                li.transform_z,
                dornaika.transform_x,
                dornaika.transform_z,
                nonlinear.transform_x,
                nonlinear.transform_z,
            )
        )
        all_gates_pass = all(metric.grade != "fail" for metric in metrics.values())
        horaud_observable = (
            metrics["hand_eye_horaud_dornaika_quaternion_normalized_eigengap"].grade == "pass"
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
        li_observable = all(
            metrics[name].grade == "pass"
            for name in (
                "robot_world_hand_eye_li_linear_rank",
                "robot_world_hand_eye_li_linear_condition_number",
                "robot_world_hand_eye_li_so3_projection_correction_frobenius_max",
            )
        )
        dornaika_observable = all(
            metrics[name].grade == "pass"
            for name in (
                "robot_world_hand_eye_dornaika_horaud_rotation_dominant_multiplicity",
                "robot_world_hand_eye_dornaika_horaud_rotation_normalized_gap",
                "robot_world_hand_eye_dornaika_horaud_quaternion_unit_error_max",
                "robot_world_hand_eye_dornaika_horaud_sign_synchronization_fraction",
                "robot_world_hand_eye_dornaika_horaud_translation_rank",
                "robot_world_hand_eye_dornaika_horaud_translation_condition_number",
            )
        )
        nonlinear_observable = all(
            metrics[name].grade == "pass"
            for name in (
                "robot_world_hand_eye_dornaika_horaud_nonlinear_objective_nonincrease",
                "robot_world_hand_eye_dornaika_horaud_nonlinear_data_jacobian_rank",
                "robot_world_hand_eye_dornaika_horaud_nonlinear_data_jacobian_condition_number",
                "robot_world_hand_eye_dornaika_horaud_nonlinear_so3_projection_correction_frobenius_max",
            )
        )
        selected = dual.transform_x
        warnings = _comparison_warnings(metrics, dataset)
        archive_sha256 = sha256_path(archive_path)
        return SolverAdapterResult(
            backend=self.backend,
            available=True,
            status=(
                "pass" if all_converged and all_gates_pass and not warnings else "inconclusive"
            ),
            metrics=metrics,
            transforms={
                **({"T_hand_eye": selected} if selected is not None else {}),
                **({"T_robot_world": shah.transform_y} if shah.transform_y is not None else {}),
            },
            observability=ObservabilityResult(
                rank=dual.linear_rank,
                condition_number=dual.observable_condition_number,
                weak_directions=[
                    *([] if horaud_observable else ["horaud_quaternion_minimum_width"]),
                    *([] if andreff_observable else ["andreff_kronecker_observability"]),
                    *([] if shah_observable else ["shah_robot_world_hand_eye_observability"]),
                    *([] if li_observable else ["li_robot_world_hand_eye_observability"]),
                    *(
                        []
                        if dornaika_observable
                        else ["dornaika_horaud_robot_world_hand_eye_observability"]
                    ),
                    *(
                        []
                        if nonlinear_observable
                        else ["dornaika_horaud_nonlinear_robot_world_hand_eye_observability"]
                    ),
                    *([] if not warnings else ["hand_eye_perturbation_power"]),
                ],
                grade=(
                    "pass"
                    if all_converged
                    and horaud_observable
                    and andreff_observable
                    and shah_observable
                    and li_observable
                    and dornaika_observable
                    and nonlinear_observable
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
                        "li_robot_world_hand_eye": li.as_dict(),
                        "dornaika_horaud_robot_world_hand_eye": dornaika.as_dict(),
                        "dornaika_horaud_nonlinear_robot_world_hand_eye": (nonlinear.as_dict()),
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
    li: LiRobotWorldHandEyeResult,
    dornaika: DornaikaHoraudRobotWorldHandEyeResult,
    nonlinear: DornaikaHoraudNonlinearResult,
    probe_options: TsaiLenzHandEyeOptions,
    factor_options: dict[str, Any],
) -> dict[str, MetricResult]:
    max_rotation = float(factor_options.get("max_holdout_rotation_rmse_deg", 2.0))
    max_translation = float(factor_options.get("max_holdout_translation_rmse_m", 0.03))
    min_detectable = float(factor_options.get("min_known_bad_detectable_fraction", 0.75))
    min_horaud_gap = float(factor_options.get("min_horaud_quaternion_normalized_eigengap", 1.0e-3))
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
    max_shah_condition = float(factor_options.get("max_shah_translation_condition_number", 1.0e8))
    max_li_condition = float(factor_options.get("max_li_linear_condition_number", 1.0e8))
    max_li_projection = float(
        factor_options.get("max_li_so3_projection_correction_frobenius", 0.05)
    )
    min_dornaika_gap = float(
        factor_options.get("min_dornaika_horaud_rotation_normalized_gap", 1.0e-3)
    )
    max_dornaika_condition = float(
        factor_options.get("max_dornaika_horaud_translation_condition_number", 1.0e8)
    )
    max_dornaika_unit_error = float(
        factor_options.get("max_dornaika_horaud_quaternion_unit_error", 1.0e-12)
    )
    min_dornaika_sign_fraction = float(
        factor_options.get("min_dornaika_horaud_sign_synchronization_fraction", 0.99)
    )
    max_nonlinear_condition = float(
        factor_options.get(
            "max_dornaika_horaud_nonlinear_data_jacobian_condition_number",
            1.0e12,
        )
    )
    max_nonlinear_projection = float(
        factor_options.get(
            "max_dornaika_horaud_nonlinear_so3_projection_correction_frobenius",
            1.0e-4,
        )
    )
    max_nonlinear_orthogonality = float(
        factor_options.get(
            "max_dornaika_horaud_nonlinear_orthogonality_error_frobenius",
            2.0e-4,
        )
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
    li_projection_values = tuple(
        value
        for value in (
            li.rotation_x_projection_correction_frobenius,
            li.rotation_z_projection_correction_frobenius,
        )
        if value is not None
    )
    li_projection_max = max(li_projection_values) if li_projection_values else None
    nonlinear_orthogonality_max = _maximum_optional(
        nonlinear.rotation_x_orthogonality_error_frobenius,
        nonlinear.rotation_z_orthogonality_error_frobenius,
    )
    nonlinear_projection_max = _maximum_optional(
        nonlinear.rotation_x_projection_correction_frobenius,
        nonlinear.rotation_z_projection_correction_frobenius,
    )
    common_split = (
        len(
            {
                (result.train_pair_ids, result.holdout_pair_ids)
                for result in (park, tsai, dual, horaud, andreff)
            }
        )
        == 1
    )
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
                and andreff.so3_projection_correction_frobenius <= max_andreff_projection
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
                f"determinant-normalized rotations require correction <= {max_shah_projection:g}"
            ),
        ),
        "robot_world_hand_eye_li_linear_rank": MetricResult(
            value=float(li.linear_rank),
            unit="rank",
            grade="pass" if li.linear_rank == 24 else "fail",
            reason="Li equations (17)-(19) must constrain all 24 linear unknowns",
        ),
        "robot_world_hand_eye_li_linear_condition_number": MetricResult(
            value=li.linear_condition_number,
            grade=(
                "pass"
                if li.linear_condition_number is not None
                and li.linear_condition_number <= max_li_condition
                else "fail"
            ),
            reason=f"simultaneous linear condition number <= {max_li_condition:g}",
        ),
        "robot_world_hand_eye_li_raw_linear_residual_rmse": MetricResult(
            value=li.raw_linear_residual_rmse,
            grade="warn",
            reason="diagnostic residual of Li's unconstrained 24-variable linear solve",
        ),
        "robot_world_hand_eye_li_so3_projection_correction_frobenius_max": MetricResult(
            value=li_projection_max,
            unit="frobenius",
            grade=(
                "pass"
                if len(li_projection_values) == 2
                and li_projection_max is not None
                and li_projection_max <= max_li_projection
                else "fail"
            ),
            reason=f"raw Li rotations require SO(3) correction <= {max_li_projection:g}",
        ),
        "robot_world_hand_eye_li_shah_common_split_consistent": MetricResult(
            value=float(
                li.train_pair_ids == shah.train_pair_ids
                and li.holdout_pair_ids == shah.holdout_pair_ids
            ),
            unit="bool",
            grade=(
                "pass"
                if li.train_pair_ids == shah.train_pair_ids
                and li.holdout_pair_ids == shah.holdout_pair_ids
                else "fail"
            ),
            reason="Li and Shah must use identical absolute-pose train/holdout splits",
        ),
        "robot_world_hand_eye_absolute_common_split_consistent": MetricResult(
            value=float(
                len(
                    {
                        (result.train_pair_ids, result.holdout_pair_ids)
                        for result in (shah, li, dornaika, nonlinear)
                    }
                )
                == 1
            ),
            unit="bool",
            grade=(
                "pass"
                if len(
                    {
                        (result.train_pair_ids, result.holdout_pair_ids)
                        for result in (shah, li, dornaika, nonlinear)
                    }
                )
                == 1
                else "fail"
            ),
            reason="all absolute-pose estimators must use one common split",
        ),
        "robot_world_hand_eye_dornaika_horaud_rotation_dominant_multiplicity": (
            MetricResult(
                value=float(dornaika.rotation_dominant_multiplicity),
                unit="multiplicity",
                grade=("pass" if dornaika.rotation_dominant_multiplicity == 1 else "fail"),
                reason="closed-form quaternion minimum requires one dominant singular pair",
            )
        ),
        "robot_world_hand_eye_dornaika_horaud_rotation_normalized_gap": MetricResult(
            value=dornaika.rotation_normalized_gap,
            unit="ratio",
            grade=(
                "pass"
                if dornaika.rotation_normalized_gap is not None
                and dornaika.rotation_normalized_gap >= min_dornaika_gap
                else "fail"
            ),
            reason=f"dominant quaternion singular-value gap >= {min_dornaika_gap:g}",
        ),
        "robot_world_hand_eye_dornaika_horaud_rotation_objective": MetricResult(
            value=dornaika.rotation_objective,
            grade="warn",
            reason="diagnostic mean squared quaternion equation residual",
        ),
        "robot_world_hand_eye_dornaika_horaud_quaternion_unit_error_max": MetricResult(
            value=(
                max(dornaika.quaternion_x_unit_error, dornaika.quaternion_z_unit_error)
                if dornaika.quaternion_x_unit_error is not None
                and dornaika.quaternion_z_unit_error is not None
                else None
            ),
            grade=(
                "pass"
                if dornaika.quaternion_x_unit_error is not None
                and dornaika.quaternion_z_unit_error is not None
                and max(
                    dornaika.quaternion_x_unit_error,
                    dornaika.quaternion_z_unit_error,
                )
                <= max_dornaika_unit_error
                else "fail"
            ),
            reason=f"both closed-form quaternion unit errors <= {max_dornaika_unit_error:g}",
        ),
        "robot_world_hand_eye_dornaika_horaud_sign_flip_count": MetricResult(
            value=float(dornaika.quaternion_sign_flip_count),
            unit="poses",
            grade="warn",
            reason="diagnostic quaternion double-cover signs changed before fitting",
        ),
        "robot_world_hand_eye_dornaika_horaud_sign_synchronization_fraction": (
            MetricResult(
                value=dornaika.quaternion_sign_synchronization_fraction,
                unit="fraction",
                grade=(
                    "pass"
                    if dornaika.quaternion_sign_synchronization_fraction is not None
                    and dornaika.quaternion_sign_synchronization_fraction
                    >= min_dornaika_sign_fraction
                    else "fail"
                ),
                reason=(
                    "weighted pairwise quaternion sign consistency "
                    f">= {min_dornaika_sign_fraction:g}"
                ),
            )
        ),
        "robot_world_hand_eye_dornaika_horaud_translation_rank": MetricResult(
            value=float(dornaika.translation_rank),
            unit="rank",
            grade="pass" if dornaika.translation_rank == 6 else "fail",
            reason="conditional [t_X; t_Z] system must constrain all six translations",
        ),
        "robot_world_hand_eye_dornaika_horaud_translation_condition_number": (
            MetricResult(
                value=dornaika.translation_condition_number,
                grade=(
                    "pass"
                    if dornaika.translation_condition_number is not None
                    and dornaika.translation_condition_number <= max_dornaika_condition
                    else "fail"
                ),
                reason=(f"conditional translation condition number <= {max_dornaika_condition:g}"),
            )
        ),
        "robot_world_hand_eye_dornaika_horaud_nonlinear_objective_nonincrease": (
            MetricResult(
                value=(
                    float(
                        nonlinear.initial_cost is not None
                        and nonlinear.final_cost is not None
                        and nonlinear.final_cost
                        <= nonlinear.initial_cost + 1.0e-12 * max(nonlinear.initial_cost, 1.0)
                    )
                ),
                unit="bool",
                grade=(
                    "pass"
                    if nonlinear.initial_cost is not None
                    and nonlinear.final_cost is not None
                    and nonlinear.final_cost
                    <= nonlinear.initial_cost + 1.0e-12 * max(nonlinear.initial_cost, 1.0)
                    else "fail"
                ),
                reason="accepted LM solution must not increase the paper objective",
            )
        ),
        "robot_world_hand_eye_dornaika_horaud_nonlinear_accepted_step_count": (
            MetricResult(
                value=float(nonlinear.accepted_step_count),
                unit="steps",
                grade="warn",
                reason="diagnostic number of accepted Levenberg-Marquardt updates",
            )
        ),
        "robot_world_hand_eye_dornaika_horaud_nonlinear_final_data_residual_rmse": (
            MetricResult(
                value=nonlinear.final_data_residual_rmse,
                grade="warn",
                reason="diagnostic mixed-unit residual under the paper's mu1=mu2=1",
            )
        ),
        "robot_world_hand_eye_dornaika_horaud_nonlinear_data_jacobian_rank": MetricResult(
            value=float(nonlinear.data_jacobian_rank),
            unit="rank",
            grade="pass" if nonlinear.data_jacobian_rank == 24 else "fail",
            reason="all 24 simultaneous data directions must be locally observable",
        ),
        "robot_world_hand_eye_dornaika_horaud_nonlinear_data_jacobian_condition_number": (
            MetricResult(
                value=nonlinear.data_jacobian_condition_number,
                grade=(
                    "pass"
                    if nonlinear.data_jacobian_condition_number is not None
                    and nonlinear.data_jacobian_condition_number <= max_nonlinear_condition
                    else "fail"
                ),
                reason=f"final data Jacobian condition number <= {max_nonlinear_condition:g}",
            )
        ),
        "robot_world_hand_eye_dornaika_horaud_nonlinear_orthogonality_error_frobenius_max": (
            MetricResult(
                value=nonlinear_orthogonality_max,
                unit="frobenius",
                grade=(
                    "pass"
                    if nonlinear_orthogonality_max is not None
                    and nonlinear_orthogonality_max <= max_nonlinear_orthogonality
                    else "fail"
                ),
                reason=f"paper rotation penalty residual <= {max_nonlinear_orthogonality:g}",
            )
        ),
        "robot_world_hand_eye_dornaika_horaud_nonlinear_so3_projection_correction_frobenius_max": (
            MetricResult(
                value=nonlinear_projection_max,
                unit="frobenius",
                grade=(
                    "pass"
                    if nonlinear_projection_max is not None
                    and nonlinear_projection_max <= max_nonlinear_projection
                    else "fail"
                ),
                reason=(
                    "optimized rotations require SO(3) correction "
                    f"<= {max_nonlinear_projection:g}"
                ),
            )
        ),
    }
    park_probes = (
        evaluate_hand_eye_known_bad_probes(
            [motion for motion in motions if motion.pair_id in set(park.holdout_pair_ids)],
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
            sum(probe.detectable is True for probe in probes) / len(probes) if probes else None
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
        grade=("pass" if shah_fraction is not None and shah_fraction >= min_detectable else "warn"),
        reason="held-out signed six-DoF perturbations of both X and Y",
    )
    li_rotation = li.holdout_evaluation.rotation_closure_rmse_deg
    li_translation = li.holdout_evaluation.translation_closure_rmse_m
    li_closure_pass = (
        li_rotation is not None
        and li_rotation <= max_rotation
        and li_translation is not None
        and li_translation <= max_translation
    )
    li_detectable = [probe.detectable for probe in li.probes if probe.detectable is not None]
    li_fraction = (
        sum(value is True for value in li_detectable) / len(li_detectable)
        if li_detectable
        else None
    )
    metrics["robot_world_hand_eye_li_holdout_rotation_rmse_deg"] = MetricResult(
        value=li_rotation, unit="deg", grade="pass" if li_closure_pass else "fail"
    )
    metrics["robot_world_hand_eye_li_holdout_translation_rmse_m"] = MetricResult(
        value=li_translation, unit="m", grade="pass" if li_closure_pass else "fail"
    )
    metrics["robot_world_hand_eye_li_known_bad_detectable_fraction"] = MetricResult(
        value=li_fraction,
        unit="fraction",
        grade=("pass" if li_fraction is not None and li_fraction >= min_detectable else "warn"),
        reason="held-out signed six-DoF perturbations of both X and Z",
    )
    dornaika_rotation = dornaika.holdout_evaluation.rotation_closure_rmse_deg
    dornaika_translation = dornaika.holdout_evaluation.translation_closure_rmse_m
    dornaika_closure_pass = (
        dornaika_rotation is not None
        and dornaika_rotation <= max_rotation
        and dornaika_translation is not None
        and dornaika_translation <= max_translation
    )
    dornaika_detectable = [
        probe.detectable for probe in dornaika.probes if probe.detectable is not None
    ]
    dornaika_fraction = (
        sum(value is True for value in dornaika_detectable) / len(dornaika_detectable)
        if dornaika_detectable
        else None
    )
    metrics["robot_world_hand_eye_dornaika_horaud_holdout_rotation_rmse_deg"] = MetricResult(
        value=dornaika_rotation,
        unit="deg",
        grade="pass" if dornaika_closure_pass else "fail",
    )
    metrics["robot_world_hand_eye_dornaika_horaud_holdout_translation_rmse_m"] = MetricResult(
        value=dornaika_translation,
        unit="m",
        grade="pass" if dornaika_closure_pass else "fail",
    )
    metrics["robot_world_hand_eye_dornaika_horaud_known_bad_detectable_fraction"] = MetricResult(
        value=dornaika_fraction,
        unit="fraction",
        grade=(
            "pass"
            if dornaika_fraction is not None and dornaika_fraction >= min_detectable
            else "warn"
        ),
        reason="held-out signed six-DoF perturbations of both X and Z",
    )
    nonlinear_rotation = nonlinear.holdout_evaluation.rotation_closure_rmse_deg
    nonlinear_translation = nonlinear.holdout_evaluation.translation_closure_rmse_m
    nonlinear_closure_pass = (
        nonlinear_rotation is not None
        and nonlinear_rotation <= max_rotation
        and nonlinear_translation is not None
        and nonlinear_translation <= max_translation
    )
    nonlinear_detectable = [
        probe.detectable for probe in nonlinear.probes if probe.detectable is not None
    ]
    nonlinear_fraction = (
        sum(value is True for value in nonlinear_detectable) / len(nonlinear_detectable)
        if nonlinear_detectable
        else None
    )
    metrics["robot_world_hand_eye_dornaika_horaud_nonlinear_holdout_rotation_rmse_deg"] = (
        MetricResult(
            value=nonlinear_rotation,
            unit="deg",
            grade="pass" if nonlinear_closure_pass else "fail",
        )
    )
    metrics["robot_world_hand_eye_dornaika_horaud_nonlinear_holdout_translation_rmse_m"] = (
        MetricResult(
            value=nonlinear_translation,
            unit="m",
            grade="pass" if nonlinear_closure_pass else "fail",
        )
    )
    metrics["robot_world_hand_eye_dornaika_horaud_nonlinear_known_bad_detectable_fraction"] = (
        MetricResult(
            value=nonlinear_fraction,
            unit="fraction",
            grade=(
                "pass"
                if nonlinear_fraction is not None and nonlinear_fraction >= min_detectable
                else "warn"
            ),
            reason="held-out signed six-DoF perturbations of both optimized transforms",
        )
    )
    comparisons = (
        ("x", li.transform_x, shah.transform_x),
        ("z", li.transform_z, shah.transform_y),
    )
    for role, li_transform, shah_transform in comparisons:
        rotation_delta, translation_delta = _transform_delta(li_transform, shah_transform)
        metrics[f"robot_world_hand_eye_li_shah_{role}_rotation_delta_deg"] = MetricResult(
            value=rotation_delta,
            unit="deg",
            grade="warn",
            reason="method-comparison diagnostic; no accuracy claim without ground truth",
        )
        metrics[f"robot_world_hand_eye_li_shah_{role}_translation_delta_m"] = MetricResult(
            value=translation_delta,
            unit="m",
            grade="warn",
            reason="method-comparison diagnostic; no accuracy claim without ground truth",
        )
    dornaika_comparisons = (
        ("x", dornaika.transform_x, shah.transform_x),
        ("z", dornaika.transform_z, shah.transform_y),
    )
    for role, dornaika_transform, shah_transform in dornaika_comparisons:
        rotation_delta, translation_delta = _transform_delta(dornaika_transform, shah_transform)
        metrics[f"robot_world_hand_eye_dornaika_horaud_shah_{role}_rotation_delta_deg"] = (
            MetricResult(
                value=rotation_delta,
                unit="deg",
                grade="warn",
                reason="method-comparison diagnostic; no accuracy claim without ground truth",
            )
        )
        metrics[f"robot_world_hand_eye_dornaika_horaud_shah_{role}_translation_delta_m"] = (
            MetricResult(
                value=translation_delta,
                unit="m",
                grade="warn",
                reason="method-comparison diagnostic; no accuracy claim without ground truth",
            )
        )
    nonlinear_comparisons = (
        ("x", nonlinear.transform_x, dornaika.transform_x),
        ("z", nonlinear.transform_z, dornaika.transform_z),
    )
    for role, nonlinear_transform, closed_transform in nonlinear_comparisons:
        rotation_delta, translation_delta = _transform_delta(nonlinear_transform, closed_transform)
        metrics[
            f"robot_world_hand_eye_dornaika_horaud_nonlinear_closed_form_{role}_rotation_delta_deg"
        ] = MetricResult(
            value=rotation_delta,
            unit="deg",
            grade="warn",
            reason="method-comparison diagnostic; no accuracy claim without ground truth",
        )
        metrics[
            f"robot_world_hand_eye_dornaika_horaud_nonlinear_closed_form_{role}_translation_delta_m"
        ] = MetricResult(
            value=translation_delta,
            unit="m",
            grade="warn",
            reason="method-comparison diagnostic; no accuracy claim without ground truth",
        )
    return metrics


def _transform_delta(left: SE3 | None, right: SE3 | None) -> tuple[float | None, float | None]:
    if left is None or right is None:
        return None, None
    delta = left.inverse().compose(right)
    quaternion_w = min(1.0, max(-1.0, abs(delta.rotation_quat_xyzw[3])))
    rotation_deg = math.degrees(2.0 * math.acos(quaternion_w))
    translation_m = math.sqrt(sum(value * value for value in delta.translation_m))
    return rotation_deg, translation_m


def _maximum_optional(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return max(left, right)


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
        result.append("public hand-eye run has weak perturbation power: " + ", ".join(warnings))
    if dataset.absolute_pose_reuse_count:
        result.append("absolute pose reuse detected across motion pairs")
    return result


def _factor_options(config: CalibrationConfig) -> dict[str, Any]:
    factor = config.pipeline.factors.get("hand_eye_motion_closure")
    return factor.options if factor is not None else {}


def _archive_path(config: CalibrationConfig, options: dict[str, Any]) -> Path:
    configured = Path(str(options.get("archive_file", "robot_arm_w_color_camera_real.zip")))
    return configured if configured.is_absolute() else Path(config.dataset.path) / configured
