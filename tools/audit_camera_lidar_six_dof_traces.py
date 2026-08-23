"""Independently audit frozen camera--LiDAR six-DoF trace collections."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from calibrex.core.camera_lidar_artifacts import CalibrationCandidateTrace

EXPECTED_SOLVER = (
    "bounded_se3_pattern_search/v0.1;projection_backend="
    "calibrex.numba_depth_pair_projector/v0.1;numba=0.65.1"
)
EXPECTED_VERSION = "calibrex.borer_six_dof_benchmark/v0.3"


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"expected a mapping: {path}")
    return loaded


def _normalized_quaternion(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0.0 or not math.isfinite(norm):
        raise ValueError("quaternion must have a finite non-zero norm")
    return [value / norm for value in values]


def _quaternion_matrix(values: list[float]) -> list[list[float]]:
    x, y, z, w = _normalized_quaternion(values)
    return [
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
    ]


def _euler_xyz_matrix(values_deg: list[float]) -> list[list[float]]:
    roll, pitch, yaw = [math.radians(value) for value in values_deg]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _matrix_multiply(
    left: list[list[float]], right: list[list[float]]
) -> list[list[float]]:
    return [
        [
            sum(left[row][inner] * right[inner][column] for inner in range(3))
            for column in range(3)
        ]
        for row in range(3)
    ]


def _max_matrix_delta(
    left: list[list[float]], right: list[list[float]]
) -> float:
    return max(
        abs(left[row][column] - right[row][column])
        for row in range(3)
        for column in range(3)
    )


def _transform_errors(
    transform: dict[str, Any], reference: dict[str, Any]
) -> tuple[float, float]:
    estimate_q = _normalized_quaternion(transform["rotation_quat_xyzw"])
    reference_q = _normalized_quaternion(reference["rotation_quat_xyzw"])
    dot = abs(
        sum(
            estimate * expected
            for estimate, expected in zip(estimate_q, reference_q, strict=True)
        )
    )
    rotation = math.degrees(
        2.0 * math.acos(min(1.0, max(-1.0, dot)))
    )
    translation = math.sqrt(
        sum(
            (estimate - expected) ** 2
            for estimate, expected in zip(
                transform["translation_m"],
                reference["translation_m"],
                strict=True,
            )
        )
    )
    return rotation, translation


def _assert_finite(value: Any, context: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"non-finite {context}: {value!r}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, f"{context}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{context}.{key}")


def audit_trace_collection(
    run_directory: Path,
    *,
    label: str,
    hit_rate_gate: float,
) -> dict[str, Any]:
    """Recompute frozen trace identities, transforms, outcomes, and bindings."""

    problem_path = run_directory / "problem-integrated-r5.yaml"
    protocol_path = run_directory / "six-dof-protocol-integrated-r5.yaml"
    trace_directory = (
        run_directory / "six-dof-traces-integrated-r5-numba-v03"
    )
    definition_path = (
        run_directory / "six-dof-definition-integrated-r5-numba-v03.yaml"
    )
    problem = _load_yaml(problem_path)
    protocol = _load_yaml(protocol_path)
    definition = _load_yaml(definition_path)
    problem_sha256 = _digest(problem_path)
    protocol_sha256 = _digest(protocol_path)
    if protocol["problem_sha256"] != problem_sha256:
        raise ValueError("protocol problem digest does not match")
    if protocol["degrees_of_freedom"] != "six_dof":
        raise ValueError("protocol is not six_dof")
    if protocol["isolation"]["test_data_used_for_selection"] is not False:
        raise ValueError("protocol declares test selection")
    perturbations = {
        item["trial_id"]: item for item in protocol["perturbations"]
    }
    if len(perturbations) != protocol["perturbation_count"] or len(
        perturbations
    ) != 200:
        raise ValueError("protocol perturbation identities are incomplete")
    expected_names = {
        f"{trial_id}.trace.yaml" for trial_id in perturbations
    }
    observed_paths = sorted(trace_directory.glob("*.yaml"))
    observed_names = {path.name for path in observed_paths}
    if observed_names != expected_names:
        raise ValueError("trace file set does not exactly match the protocol")

    reference = problem["reference_transform_camera_lidar"]
    reference_matrix = _quaternion_matrix(reference["rotation_quat_xyzw"])
    max_initial_matrix_delta = 0.0
    max_initial_translation_delta = 0.0
    max_rotation_recompute_delta = 0.0
    max_translation_recompute_delta = 0.0
    hit_count = 0
    statuses: dict[str, int] = {}
    trace_hashes: dict[str, str] = {}
    recomputed: dict[str, tuple[float, float, bool, float, float]] = {}

    for path in observed_paths:
        raw = _load_yaml(path)
        trace = CalibrationCandidateTrace.model_validate(raw)
        trial_id = trace.trial_id
        if path.name != f"{trial_id}.trace.yaml" or trial_id not in perturbations:
            raise ValueError(f"trace identity mismatch: {path}")
        perturbation = perturbations[trial_id]
        if raw["trace_id"] != f'{protocol["protocol_id"]}:{trial_id}':
            raise ValueError(f"trace_id mismatch: {trial_id}")
        if (
            trace.problem_sha256 != problem_sha256
            or trace.protocol_sha256 != protocol_sha256
            or trace.solver != EXPECTED_SOLVER
            or trace.solver_version != EXPECTED_VERSION
        ):
            raise ValueError(f"trace source or solver mismatch: {trial_id}")
        if trace.status not in {
            "converged",
            "max_evaluations",
            "insufficient_observations",
        }:
            raise ValueError(f"unexpected trace status: {trial_id}")
        statuses[trace.status] = statuses.get(trace.status, 0) + 1
        if not math.isfinite(trace.runtime_seconds):
            raise ValueError(f"non-finite runtime: {trial_id}")
        for key in (
            "initial_transform_camera_lidar",
            "output_transform_camera_lidar",
        ):
            if (
                raw[key]["parent"] != reference["parent"]
                or raw[key]["child"] != reference["child"]
            ):
                raise ValueError(f"transform frame mismatch: {trial_id}")

        expected_translation = [
            reference_value + perturbation_value
            for reference_value, perturbation_value in zip(
                reference["translation_m"],
                perturbation["translation_m_xyz"],
                strict=True,
            )
        ]
        observed_translation = raw["initial_transform_camera_lidar"][
            "translation_m"
        ]
        translation_delta = max(
            abs(expected - observed)
            for expected, observed in zip(
                expected_translation, observed_translation, strict=True
            )
        )
        max_initial_translation_delta = max(
            max_initial_translation_delta, translation_delta
        )
        if translation_delta > 1.0e-15:
            raise ValueError(f"initial translation mismatch: {trial_id}")

        expected_rotation = _matrix_multiply(
            _euler_xyz_matrix(perturbation["rotation_deg_xyz"]),
            reference_matrix,
        )
        observed_rotation = _quaternion_matrix(
            raw["initial_transform_camera_lidar"]["rotation_quat_xyzw"]
        )
        matrix_delta = _max_matrix_delta(
            expected_rotation, observed_rotation
        )
        max_initial_matrix_delta = max(
            max_initial_matrix_delta, matrix_delta
        )
        if matrix_delta > 2.0e-15:
            raise ValueError(f"initial rotation mismatch: {trial_id}")
        rotation_norm = math.sqrt(
            sum(value * value for value in perturbation["rotation_deg_xyz"])
        )
        translation_norm = math.sqrt(
            sum(value * value for value in perturbation["translation_m_xyz"])
        )
        if (
            abs(rotation_norm - protocol["rotation_magnitude_deg"])
            > 1.0e-12
            or abs(
                translation_norm - protocol["translation_magnitude_m"]
            )
            > 1.0e-12
        ):
            raise ValueError(f"perturbation magnitude mismatch: {trial_id}")

        rotation_error, translation_error = _transform_errors(
            raw["output_transform_camera_lidar"], reference
        )
        rotation_error_delta = abs(
            rotation_error - trace.outcome.rotation_error_deg
        )
        translation_error_delta = abs(
            translation_error - trace.outcome.translation_error_m
        )
        max_rotation_recompute_delta = max(
            max_rotation_recompute_delta, rotation_error_delta
        )
        max_translation_recompute_delta = max(
            max_translation_recompute_delta, translation_error_delta
        )
        if rotation_error_delta > 2.0e-10 or translation_error_delta > 1.0e-15:
            raise ValueError(f"outcome recomputation mismatch: {trial_id}")
        hit = (
            rotation_error < protocol["hit"]["rotation_error_max_deg"]
            and translation_error
            < protocol["hit"]["translation_error_max_m"]
        )
        if hit is not trace.outcome.hit:
            raise ValueError(f"strict hit mismatch: {trial_id}")
        hit_count += int(hit)
        initial_rotation_error, initial_translation_error = _transform_errors(
            raw["initial_transform_camera_lidar"], reference
        )
        recomputed[trial_id] = (
            rotation_error,
            translation_error,
            hit,
            initial_rotation_error,
            initial_translation_error,
        )

        _assert_finite(
            raw["initial_transform_camera_lidar"], f"{trial_id}.initial"
        )
        _assert_finite(
            raw["output_transform_camera_lidar"], f"{trial_id}.output"
        )
        _assert_finite(raw["evaluations"], f"{trial_id}.evaluations")
        if len(trace.evaluations) > int(
            protocol["optimizer_options"]["max_evaluations"]
        ) or any(
            item.evaluated_frame_count < 0
            or item.evaluated_frame_count > len(protocol["frame_ids"])
            for item in trace.evaluations
        ):
            raise ValueError(f"evaluation bounds mismatch: {trial_id}")

        provenance = raw["provenance"]
        if (
            provenance["generator"]
            != "calibrex.evaluation.borer_six_dof_benchmark"
            or provenance["generator_version"] != EXPECTED_VERSION
            or provenance["config_sha256"] != protocol_sha256
            or provenance["source_sha256"] != problem_sha256
        ):
            raise ValueError(f"trace provenance mismatch: {trial_id}")
        command = provenance["command"]
        if "--start" not in command or "--stop" not in command:
            raise ValueError(f"trace command range missing: {trial_id}")
        start = int(command[command.index("--start") + 1])
        stop = int(command[command.index("--stop") + 1])
        trial_index = int(trial_id.rsplit("-", 1)[1])
        if not 0 <= start <= trial_index < stop <= 200:
            raise ValueError(f"trace command range mismatch: {trial_id}")
        if (
            command[command.index("--projection-backend") + 1]
            != "numba_cpu"
        ):
            raise ValueError(f"trace backend command mismatch: {trial_id}")
        lower_command = [str(item).lower() for item in command]
        if (
            str(problem_path).lower() not in lower_command
            or str(protocol_path).lower() not in lower_command
        ):
            raise ValueError(f"trace source command mismatch: {trial_id}")
        trace_hashes[trial_id] = _digest(path)

    if statuses.get("insufficient_observations", 0):
        raise ValueError("trace collection contains computational failures")
    sources = definition["provenance"]["source_artifacts"]
    if (
        sources["problem"] != problem_sha256
        or sources["protocol"] != protocol_sha256
    ):
        raise ValueError("definition source digest mismatch")
    for trial_id, trace_sha256 in trace_hashes.items():
        if sources[f"trace:{trial_id}"] != trace_sha256:
            raise ValueError(f"definition trace digest mismatch: {trial_id}")

    definition_trials = definition["trials"]
    if len(definition_trials) != 400:
        raise ValueError("definition trial count mismatch")
    trials_by_key = {
        (item["method_id"], item["split_id"]): item
        for item in definition_trials
    }
    if len(trials_by_key) != 400:
        raise ValueError("definition trial identities are not unique")
    for trial_id, values in recomputed.items():
        rotation, translation, hit, initial_rotation, initial_translation = values
        native = trials_by_key[("native_borer_d2d_six_dof", trial_id)]
        initial = trials_by_key[("unoptimized_initial", trial_id)]
        if native["status"] != "success":
            raise ValueError(f"native definition status mismatch: {trial_id}")
        if (
            abs(native["metrics"]["rotation_error_deg"] - rotation)
            > 2.0e-10
            or abs(native["metrics"]["translation_error_m"] - translation)
            > 1.0e-15
            or native["metrics"]["hit"] != float(hit)
            or abs(
                initial["metrics"]["rotation_error_deg"]
                - initial_rotation
            )
            > 2.0e-10
            or abs(
                initial["metrics"]["translation_error_m"]
                - initial_translation
            )
            > 1.0e-15
        ):
            raise ValueError(f"definition trial recomputation mismatch: {trial_id}")

    hit_rate = hit_count / len(observed_paths)
    return {
        "dataset": label,
        "trace_count": len(observed_paths),
        "schema_valid_count": len(observed_paths),
        "problem_sha256": problem_sha256,
        "protocol_sha256": protocol_sha256,
        "definition_sha256": _digest(definition_path),
        "solver": EXPECTED_SOLVER,
        "solver_version": EXPECTED_VERSION,
        "statuses": statuses,
        "hit_count": hit_count,
        "hit_rate": hit_rate,
        "prespecified_hit_gate": hit_rate_gate,
        "gate_passed": hit_rate >= hit_rate_gate,
        "max_initial_rotation_matrix_abs_delta": max_initial_matrix_delta,
        "max_initial_translation_abs_delta_m": max_initial_translation_delta,
        "max_outcome_rotation_recompute_delta_deg": (
            max_rotation_recompute_delta
        ),
        "max_outcome_translation_recompute_delta_m": (
            max_translation_recompute_delta
        ),
        "definition_trace_digest_matches": len(trace_hashes),
        "definition_trial_recomputations": len(definition_trials),
        "strict_less_than_recomputed": True,
        "provenance_command_ranges_checked": len(observed_paths),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--hit-rate-gate", required=True, type=float)
    return parser


def main() -> int:
    """Run the independent audit and print its deterministic summary."""

    args = _parser().parse_args()
    report = audit_trace_collection(
        args.run_directory,
        label=args.label,
        hit_rate_gate=args.hit_rate_gate,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
