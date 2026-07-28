#!/usr/bin/env python3
"""Generate shared-split ETHZ native-versus-OpenCV hand-eye benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import time
import tracemalloc
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from calibrex.core.benchmark import (
    BenchmarkDefinition,
    BenchmarkMethodDefinition,
    BenchmarkMetricDefinition,
    BenchmarkProtocol,
    BenchmarkProvenance,
    BenchmarkSplit,
    BenchmarkTrial,
    BenchmarkTrialProvenance,
    aggregate_benchmark_definition,
    render_benchmark_markdown,
)
from calibrex.data.ethz_hand_eye import (
    ETHZ_ROBOT_ARM_REAL_SHA256,
    read_ethz_robot_arm_hand_eye_motions,
)
from calibrex.evaluation.hand_eye_benchmark import (
    HandEyeAbsolutePoseSplit,
    opencv_compatible_hand_eye_motions,
    split_absolute_hand_eye_poses,
)
from calibrex.solvers.andreff_hand_eye_solver import (
    AndreffHandEyeOptions,
    AndreffHandEyeSolver,
)
from calibrex.solvers.chou_kamel_hand_eye_solver import (
    ChouKamelHandEyeOptions,
    ChouKamelHandEyeSolver,
)
from calibrex.solvers.daniilidis_hand_eye_solver import (
    DaniilidisHandEyeOptions,
    DaniilidisHandEyeSolver,
)
from calibrex.solvers.dornaika_horaud_nonlinear_robot_world_hand_eye_solver import (
    DornaikaHoraudNonlinearOptions,
    DornaikaHoraudNonlinearSolver,
)
from calibrex.solvers.dornaika_horaud_robot_world_hand_eye_solver import (
    DornaikaHoraudRobotWorldHandEyeOptions,
    DornaikaHoraudRobotWorldHandEyeSolver,
)
from calibrex.solvers.horaud_dornaika_hand_eye_solver import (
    HoraudDornaikaHandEyeOptions,
    HoraudDornaikaHandEyeSolver,
)
from calibrex.solvers.horaud_dornaika_nonlinear_hand_eye_solver import (
    HoraudDornaikaNonlinearHandEyeOptions,
    HoraudDornaikaNonlinearHandEyeSolver,
)
from calibrex.solvers.li_robot_world_hand_eye_solver import (
    LiRobotWorldHandEyeOptions,
    LiRobotWorldHandEyeSolver,
)
from calibrex.solvers.opencv_hand_eye_adapters import (
    OPENCV_LICENSE_SPDX,
    OpenCvHandEyeAdapter,
    OpenCvRobotWorldHandEyeAdapter,
)
from calibrex.solvers.park_martin_hand_eye_solver import (
    HandEyeMotionPair,
    ParkMartinHandEyeOptions,
    ParkMartinHandEyeSolver,
    evaluate_hand_eye_motions,
)
from calibrex.solvers.shah_robot_world_hand_eye_solver import (
    RobotWorldHandEyePosePair,
    ShahRobotWorldHandEyeOptions,
    ShahRobotWorldHandEyeSolver,
    evaluate_robot_world_hand_eye_poses,
)
from calibrex.solvers.shiu_ahmad_hand_eye_solver import (
    ShiuAhmadHandEyeOptions,
    ShiuAhmadHandEyeSolver,
)
from calibrex.solvers.tsai_lenz_hand_eye_solver import (
    TsaiLenzHandEyeOptions,
    TsaiLenzHandEyeSolver,
)
from calibrex.solvers.zhuang_roth_sudhakar_robot_world_hand_eye_solver import (
    ZhuangRothSudhakarOptions,
    ZhuangRothSudhakarSolver,
)

_DOI = {
    "park_martin": "10.1109/70.326576",
    "tsai_lenz": "10.1109/70.34770",
    "daniilidis": "10.1177/027836499801700107",
    "andreff": "10.1023/A:1008192712159",
    "shiu_ahmad": "10.1109/70.88052",
    "chou_kamel": "10.1109/70.313098",
    "horaud_dornaika": "10.1177/027836499501400301",
    "shah": "10.1115/1.4024473",
    "li_wang_wu": "10.5897/IJPS.9000501",
    "dornaika_horaud": "10.1177/027836499501400301",
    "zhuang_roth_sudhakar": "10.1109/70.313098",
}
_OPENCV_HAND_EYE_DOI_KEY = {
    "tsai": "tsai_lenz",
    "park": "park_martin",
    "horaud": "horaud_dornaika",
    "andreff": "andreff",
    "daniilidis": "daniilidis",
}


@dataclass(frozen=True)
class _Execution:
    status: str
    reason: str
    metrics: dict[str, float]
    output: dict[str, Any]
    runtime_seconds: float
    peak_memory_mb: float


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _opencv_package_version() -> str | None:
    try:
        return importlib.metadata.version("opencv-python-headless")
    except importlib.metadata.PackageNotFoundError:
        return None


def _measure(call: Callable[[], tuple[str, str, dict[str, float], dict[str, Any]]]) -> _Execution:
    tracemalloc.start()
    started = time.perf_counter()
    try:
        status, reason, metrics, output = call()
    except Exception as exc:
        status, reason, metrics, output = "failed", f"{type(exc).__name__}: {exc}", {}, {}
    runtime = time.perf_counter() - started
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return _Execution(
        status=status,
        reason=reason,
        metrics=metrics,
        output=output,
        runtime_seconds=runtime,
        peak_memory_mb=peak / (1024.0 * 1024.0),
    )


def _native_ax_xb_methods() -> list[
    tuple[str, str, str, Callable[[Sequence[HandEyeMotionPair]], Any]]
]:
    return [
        (
            "park_martin",
            "Calibrex Park-Martin",
            _DOI["park_martin"],
            lambda motions: ParkMartinHandEyeSolver().solve(
                motions, ParkMartinHandEyeOptions(holdout_ratio=0.0)
            ),
        ),
        (
            "tsai_lenz",
            "Calibrex Tsai-Lenz",
            _DOI["tsai_lenz"],
            lambda motions: TsaiLenzHandEyeSolver().solve(
                motions, TsaiLenzHandEyeOptions(holdout_ratio=0.0)
            ),
        ),
        (
            "daniilidis",
            "Calibrex Daniilidis",
            _DOI["daniilidis"],
            lambda motions: DaniilidisHandEyeSolver().solve(
                motions, DaniilidisHandEyeOptions(holdout_ratio=0.0)
            ),
        ),
        (
            "andreff",
            "Calibrex Andreff",
            _DOI["andreff"],
            lambda motions: AndreffHandEyeSolver().solve(
                motions, AndreffHandEyeOptions(holdout_ratio=0.0)
            ),
        ),
        (
            "shiu_ahmad",
            "Calibrex Shiu-Ahmad",
            _DOI["shiu_ahmad"],
            lambda motions: ShiuAhmadHandEyeSolver().solve(
                motions, ShiuAhmadHandEyeOptions(holdout_ratio=0.0)
            ),
        ),
        (
            "chou_kamel",
            "Calibrex Chou-Kamel",
            _DOI["chou_kamel"],
            lambda motions: ChouKamelHandEyeSolver().solve(
                motions, ChouKamelHandEyeOptions(holdout_ratio=0.0)
            ),
        ),
        (
            "horaud_dornaika",
            "Calibrex Horaud-Dornaika",
            _DOI["horaud_dornaika"],
            lambda motions: HoraudDornaikaHandEyeSolver().solve(
                motions, HoraudDornaikaHandEyeOptions(holdout_ratio=0.0)
            ),
        ),
        (
            "horaud_dornaika_nonlinear",
            "Calibrex H-D nonlinear",
            _DOI["horaud_dornaika"],
            lambda motions: HoraudDornaikaNonlinearHandEyeSolver().solve(
                motions, HoraudDornaikaNonlinearHandEyeOptions(holdout_ratio=0.0)
            ),
        ),
    ]


def _native_ax_yb_methods() -> list[
    tuple[str, str, str, Callable[[Sequence[RobotWorldHandEyePosePair]], Any]]
]:
    return [
        (
            "shah",
            "Calibrex Shah",
            _DOI["shah"],
            lambda poses: ShahRobotWorldHandEyeSolver().solve(
                poses, ShahRobotWorldHandEyeOptions(holdout_ratio=0.0)
            ),
        ),
        (
            "li_wang_wu",
            "Calibrex Li-Wang-Wu",
            _DOI["li_wang_wu"],
            lambda poses: LiRobotWorldHandEyeSolver().solve(
                poses, LiRobotWorldHandEyeOptions(holdout_ratio=0.0)
            ),
        ),
        (
            "dornaika_horaud",
            "Calibrex Dornaika-Horaud",
            _DOI["dornaika_horaud"],
            lambda poses: DornaikaHoraudRobotWorldHandEyeSolver().solve(
                poses, DornaikaHoraudRobotWorldHandEyeOptions(holdout_ratio=0.0)
            ),
        ),
        (
            "zhuang_roth_sudhakar",
            "Calibrex Zhuang-Roth-Sudhakar",
            _DOI["zhuang_roth_sudhakar"],
            lambda poses: ZhuangRothSudhakarSolver().solve(
                poses, ZhuangRothSudhakarOptions(holdout_ratio=0.0)
            ),
        ),
        (
            "dornaika_horaud_nonlinear",
            "Calibrex D-H nonlinear",
            _DOI["dornaika_horaud"],
            lambda poses: DornaikaHoraudNonlinearSolver().solve(
                poses, DornaikaHoraudNonlinearOptions(holdout_ratio=0.0)
            ),
        ),
    ]


def _native_ax_xb_execution(
    solve: Callable[[Sequence[HandEyeMotionPair]], Any],
    fit: Sequence[HandEyeMotionPair],
    holdout: Sequence[HandEyeMotionPair],
) -> tuple[str, str, dict[str, float], dict[str, Any]]:
    result = solve(fit)
    output = result.as_dict()
    if result.status != "converged" or result.transform_x is None:
        return "failed", result.reason, {}, output
    evaluation = evaluate_hand_eye_motions(holdout, result.transform_x)
    if (
        evaluation.rotation_closure_rmse_deg is None
        or evaluation.translation_closure_rmse_m is None
    ):
        return "failed", "holdout evaluation produced no metrics", {}, output
    return (
        "success",
        result.reason,
        {
            "rotation_closure_rmse_deg": evaluation.rotation_closure_rmse_deg,
            "translation_closure_rmse_m": evaluation.translation_closure_rmse_m,
        },
        output,
    )


def _opencv_ax_xb_execution(
    method: str,
    split: HandEyeAbsolutePoseSplit,
    holdout: Sequence[HandEyeMotionPair],
) -> tuple[str, str, dict[str, float], dict[str, Any]]:
    result = OpenCvHandEyeAdapter().solve(split.fit_poses, method)  # type: ignore[arg-type]
    output = result.as_dict()
    if result.status != "converged" or result.transform_x is None:
        return "failed", result.reason, {}, output
    evaluation = evaluate_hand_eye_motions(holdout, result.transform_x)
    if (
        evaluation.rotation_closure_rmse_deg is None
        or evaluation.translation_closure_rmse_m is None
    ):
        return "failed", "holdout evaluation produced no metrics", {}, output
    return (
        "success",
        result.reason,
        {
            "rotation_closure_rmse_deg": evaluation.rotation_closure_rmse_deg,
            "translation_closure_rmse_m": evaluation.translation_closure_rmse_m,
        },
        output,
    )


def _native_ax_yb_execution(
    solve: Callable[[Sequence[RobotWorldHandEyePosePair]], Any],
    split: HandEyeAbsolutePoseSplit,
) -> tuple[str, str, dict[str, float], dict[str, Any]]:
    result = solve(split.fit_poses)
    output = result.as_dict()
    if result.status != "converged" or result.transform_x is None or result.transform_y is None:
        return "failed", result.reason, {}, output
    evaluation = evaluate_robot_world_hand_eye_poses(
        split.holdout_poses, result.transform_x, result.transform_y
    )
    if (
        evaluation.rotation_closure_rmse_deg is None
        or evaluation.translation_closure_rmse_m is None
    ):
        return "failed", "holdout evaluation produced no metrics", {}, output
    return (
        "success",
        result.reason,
        {
            "rotation_closure_rmse_deg": evaluation.rotation_closure_rmse_deg,
            "translation_closure_rmse_m": evaluation.translation_closure_rmse_m,
        },
        output,
    )


def _opencv_ax_yb_execution(
    method: str,
    split: HandEyeAbsolutePoseSplit,
) -> tuple[str, str, dict[str, float], dict[str, Any]]:
    result = OpenCvRobotWorldHandEyeAdapter().solve(split.fit_poses, method)  # type: ignore[arg-type]
    output = result.as_dict()
    if result.status != "converged" or result.transform_x is None or result.transform_y is None:
        return "failed", result.reason, {}, output
    evaluation = evaluate_robot_world_hand_eye_poses(
        split.holdout_poses, result.transform_x, result.transform_y
    )
    if (
        evaluation.rotation_closure_rmse_deg is None
        or evaluation.translation_closure_rmse_m is None
    ):
        return "failed", "holdout evaluation produced no metrics", {}, output
    return (
        "success",
        result.reason,
        {
            "rotation_closure_rmse_deg": evaluation.rotation_closure_rmse_deg,
            "translation_closure_rmse_m": evaluation.translation_closure_rmse_m,
        },
        output,
    )


def _metric_definitions() -> list[BenchmarkMetricDefinition]:
    return [
        BenchmarkMetricDefinition(
            name="rotation_closure_rmse_deg",
            label="Holdout rotation RMSE",
            unit="deg",
            direction="lower",
            primary=True,
            interpretation="SE(3) equation closure rotation RMSE on untouched holdout poses",
        ),
        BenchmarkMetricDefinition(
            name="translation_closure_rmse_m",
            label="Holdout translation RMSE",
            unit="m",
            display_unit="mm",
            display_scale=1000.0,
            direction="lower",
            primary=True,
            interpretation="SE(3) equation closure translation RMSE on untouched holdout poses",
        ),
    ]


def _method_definition(
    method_id: str,
    label: str,
    doi: str,
    *,
    opencv_version: str | None = None,
) -> BenchmarkMethodDefinition:
    external = method_id.startswith("opencv_")
    return BenchmarkMethodDefinition(
        method_id=method_id,
        label=label,
        implementation="adapter" if external else "calibrex_native",
        tool_name="opencv" if external else "calibrex",
        tool_version=opencv_version,
        source_commit=_git_commit() if not external else None,
        paper_doi=doi,
        license_spdx=OPENCV_LICENSE_SPDX if external else "Apache-2.0",
        adapter_version="opencv-hand-eye/v0.1" if external else None,
    )


def _trial(
    method_id: str,
    split: HandEyeAbsolutePoseSplit,
    execution: _Execution,
    *,
    family: str,
) -> BenchmarkTrial:
    input_digest = _sha256_json(
        {
            "family": family,
            "fit_ids_sha256": split.fit_ids_sha256,
            "holdout_ids_sha256": split.holdout_ids_sha256,
        }
    )
    return BenchmarkTrial(
        method_id=method_id,
        split_id=split.split_id,
        status="success" if execution.status == "success" else "failed",
        metrics=execution.metrics,
        runtime_seconds=execution.runtime_seconds,
        peak_memory_mb=execution.peak_memory_mb,
        failure_reason=None if execution.status == "success" else execution.reason,
        provenance=BenchmarkTrialProvenance(
            command=(
                "python3 tools/generate_ethz_hand_eye_benchmarks.py "
                "data/public/ethz_hand_eye_robot_arm_real/"
                "robot_arm_w_color_camera_real.zip "
                f"--seeds {split.seed} --fit-count {len(split.fit_poses)} "
                f"--holdout-count {len(split.holdout_poses)}"
            ),
            config_sha256=_sha256_json(
                {
                    "family": family,
                    "method_id": method_id,
                    "seed": split.seed,
                    "fit_count": len(split.fit_poses),
                    "holdout_count": len(split.holdout_poses),
                    "protocol_version": "v0.1",
                }
            ),
            input_sha256=input_digest,
            output_sha256=_sha256_json(execution.output),
            notes=[execution.reason],
        ),
    )


def _protocol(
    family: str,
    splits: Sequence[HandEyeAbsolutePoseSplit],
) -> BenchmarkProtocol:
    return BenchmarkProtocol(
        protocol_id=f"ethz-real-hand-eye/{family}/absolute-pose-split/v0.1",
        dataset_id="ethz_hand_eye_robot_arm_real",
        dataset_version="ethz-asl-hand-eye@966cd925",
        dataset_doi="10.3929/ethz-c-000788527",
        dataset_source_sha256=ETHZ_ROBOT_ARM_REAL_SHA256,
        data_license="upstream research dataset; terms documented on dataset page",
        split_policy=(
            "seeded disjoint absolute-pose sampling before fit/holdout relative-pair "
            "construction; bounded pose counts; sorted IDs and SHA-256 digests recorded"
        ),
        splits=[
            BenchmarkSplit(
                split_id=split.split_id,
                seed=split.seed,
                fit_count=len(split.fit_poses),
                holdout_count=len(split.holdout_poses),
                fit_ids_sha256=split.fit_ids_sha256,
                holdout_ids_sha256=split.holdout_ids_sha256,
            )
            for split in splits
        ],
        initial_estimate_policy=(
            "method default; nonlinear methods initialize only from their native closed form"
        ),
        tuning_policy="fixed defaults selected before execution; holdout is evaluation-only",
        failure_policy="retain every method-by-split failure in the failure-rate denominator",
    )


def _write_outputs(
    definition: BenchmarkDefinition,
    output_dir: Path,
    stem: str,
) -> None:
    artifact = aggregate_benchmark_definition(definition)
    output_dir.mkdir(parents=True, exist_ok=True)
    definition.save(output_dir / f"{stem}.definition.json")
    artifact.save(output_dir / f"{stem}.json")
    (output_dir / f"{stem}.md").write_text(render_benchmark_markdown(artifact), encoding="utf-8")


def generate(
    archive: Path,
    output_dir: Path,
    *,
    seeds: Sequence[int],
    fit_count: int,
    holdout_count: int,
) -> None:
    """Execute and write both equation-family benchmark artifacts."""

    archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if archive_digest != ETHZ_ROBOT_ARM_REAL_SHA256:
        raise ValueError(
            "ETHZ archive digest mismatch: "
            f"expected {ETHZ_ROBOT_ARM_REAL_SHA256}, got {archive_digest}"
        )
    dataset = read_ethz_robot_arm_hand_eye_motions(archive)
    poses = tuple(
        RobotWorldHandEyePosePair(pair.pair_id, pair.pose_a, pair.pose_b)
        for pair in dataset.absolute_pose_pairs
    )
    splits = [
        split_absolute_hand_eye_poses(
            poses,
            seed=seed,
            fit_count=fit_count,
            holdout_count=holdout_count,
            split_id=f"seed-{seed:03d}",
        )
        for seed in seeds
    ]
    provenance = BenchmarkProvenance(
        generator="tools/generate_ethz_hand_eye_benchmarks.py",
        generator_version="0.1",
        git_commit=_git_commit(),
        command=(
            "python3 tools/generate_ethz_hand_eye_benchmarks.py "
            f"{archive} --output-dir {output_dir} --seeds {','.join(map(str, seeds))} "
            f"--fit-count {fit_count} --holdout-count {holdout_count}"
        ),
        source_artifacts={"dataset_archive": ETHZ_ROBOT_ARM_REAL_SHA256},
        data_verified=True,
    )

    xb_methods = [
        _method_definition(method_id, label, doi)
        for method_id, label, doi, _solve in _native_ax_xb_methods()
    ] + [
        _method_definition(
            f"opencv_{method}",
            f"OpenCV {method.title()}",
            _DOI[_OPENCV_HAND_EYE_DOI_KEY[method]],
            opencv_version=_opencv_package_version(),
        )
        for method in ("tsai", "park", "horaud", "andreff", "daniilidis")
    ]
    xb_trials: list[BenchmarkTrial] = []
    for split in splits:
        fit_motions = opencv_compatible_hand_eye_motions(split.fit_poses)
        holdout_motions = opencv_compatible_hand_eye_motions(split.holdout_poses)
        for method_id, _label, _doi, solve in _native_ax_xb_methods():
            execution = _measure(
                lambda solve=solve, fit_motions=fit_motions, holdout_motions=holdout_motions: (
                    _native_ax_xb_execution(solve, fit_motions, holdout_motions)
                )
            )
            xb_trials.append(_trial(method_id, split, execution, family="ax-xb"))
        for method in ("tsai", "park", "horaud", "andreff", "daniilidis"):
            execution = _measure(
                lambda method=method, split=split, holdout_motions=holdout_motions: (
                    _opencv_ax_xb_execution(method, split, holdout_motions)
                )
            )
            xb_trials.append(_trial(f"opencv_{method}", split, execution, family="ax-xb"))
    xb_definition = BenchmarkDefinition(
        benchmark_id="ethz-real-hand-eye-ax-xb-multiseed-2026-07-28",
        title="ETHZ real hand-eye AX=XB native and OpenCV shared-split benchmark",
        protocol=_protocol("ax-xb", splits),
        metrics=_metric_definitions(),
        methods=xb_methods,
        trials=xb_trials,
        reference_method_id="horaud_dornaika_nonlinear",
        bootstrap_samples=5000,
        bootstrap_seed=20260728,
        limitations=[
            (
                "The dataset has no accepted ground-truth extrinsic; closure is "
                "consistency, not absolute accuracy."
            ),
            (
                "Pairwise relative motions share absolute poses and therefore "
                "are not independent samples."
            ),
            (
                "Runtime is single-process wall time on the recorded host; "
                "Python peak memory excludes native allocator visibility."
            ),
        ],
        provenance=provenance,
    )
    _write_outputs(xb_definition, output_dir, "ethz-hand-eye-ax-xb-benchmark")

    yb_methods = [
        _method_definition(method_id, label, doi)
        for method_id, label, doi, _solve in _native_ax_yb_methods()
    ] + [
        _method_definition(
            f"opencv_{method}",
            f"OpenCV {method.title()}",
            _DOI["li_wang_wu" if method == "li" else method],
            opencv_version=_opencv_package_version(),
        )
        for method in ("shah", "li")
    ]
    yb_trials: list[BenchmarkTrial] = []
    for split in splits:
        for method_id, _label, _doi, solve in _native_ax_yb_methods():
            execution = _measure(
                lambda solve=solve, split=split: _native_ax_yb_execution(solve, split)
            )
            yb_trials.append(_trial(method_id, split, execution, family="ax-yb"))
        for method in ("shah", "li"):
            execution = _measure(
                lambda method=method, split=split: _opencv_ax_yb_execution(method, split)
            )
            yb_trials.append(_trial(f"opencv_{method}", split, execution, family="ax-yb"))
    yb_definition = BenchmarkDefinition(
        benchmark_id="ethz-real-robot-world-hand-eye-ax-yb-multiseed-2026-07-28",
        title="ETHZ real robot-world hand-eye AX=YB native and OpenCV benchmark",
        protocol=_protocol("ax-yb", splits),
        metrics=_metric_definitions(),
        methods=yb_methods,
        trials=yb_trials,
        reference_method_id="dornaika_horaud_nonlinear",
        bootstrap_samples=5000,
        bootstrap_seed=20260728,
        limitations=[
            (
                "The dataset has no accepted ground-truth extrinsic; closure is "
                "consistency, not absolute accuracy."
            ),
            "AX=YB results are ranked separately from the AX=XB equation family.",
            (
                "Runtime is single-process wall time on the recorded host; "
                "Python peak memory excludes native allocator visibility."
            ),
        ],
        provenance=provenance,
    )
    _write_outputs(yb_definition, output_dir, "ethz-hand-eye-ax-yb-benchmark")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets"))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--fit-count", type=int, default=32)
    parser.add_argument("--holdout-count", type=int, default=16)
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(","))
    generate(
        args.archive,
        args.output_dir,
        seeds=seeds,
        fit_count=args.fit_count,
        holdout_count=args.holdout_count,
    )


if __name__ == "__main__":
    main()
