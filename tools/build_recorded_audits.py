"""Build ``slac.sota_audit_result`` records for two audits recorded only as committed runs.

The KITTI board camera-LiDAR audit (#350) and the NCLT lidar-wheel_odometry audit (#355)
were each pre-registered, run once and recorded on their benchmark pages, but not as
SOTA audit records, so ``calibrex sota leaderboard`` did not list them. This reads their
committed per-case scores and pre-registrations, writes one benchmark per gated metric,
the protocol with every pre-registered gate of both claims, and the audit result.
Nothing is re-run and no gate is changed; each record's provenance names the committed
files it was built from by SHA-256::

    python tools/build_recorded_audits.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calibrex import __version__  # noqa: E402
from calibrex.core.benchmark import (  # noqa: E402
    BenchmarkDefinition,
    BenchmarkMethodDefinition,
    BenchmarkMetricDefinition,
    BenchmarkProtocol,
    BenchmarkProvenance,
    BenchmarkSplit,
    BenchmarkTrial,
    BenchmarkTrialProvenance,
    aggregate_benchmark_definition,
)
from calibrex.core.sota_audit import (  # noqa: E402
    SotaAuditProtocol,
    SotaAuditProvenance,
    SotaClaimRequirement,
    SotaClaimScope,
)
from calibrex.evaluation.sota_audit import audit_sota_claim  # noqa: E402

ANCHOR = BenchmarkMethodDefinition(
    method_id="reference_zero_anchor",
    label="Mathematical zero anchor (not an empirical calibration method)",
    implementation="imported",
    tool_name="dataset_reference_zero",
)
Value = float | str  # a metric value, or the reason the case has none


@dataclass(frozen=True)
class Gate:
    name: str
    metric: str
    statistic: str
    comparison: str
    threshold: float
    description: str


@dataclass(frozen=True)
class Recorded:
    protocol_id: str
    title: str
    method: BenchmarkMethodDefinition
    scope: dict[str, Any]
    claim: str
    family: str
    rig: str
    license: str
    preregistration: Path
    scores: Path
    metrics: dict[str, tuple[str, str]]  # name -> (unit, label)
    rows: Callable[[], dict[str, dict[str, Value]]]  # metric -> case -> value
    gates: Callable[[dict[str, Any]], list[Gate]]
    output: Path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _benchmark(spec: Recorded, metric: str, rows: dict[str, Value]) -> BenchmarkDefinition:
    digest = _digest(spec.scores)
    unit, label = spec.metrics[metric]
    provenance = BenchmarkTrialProvenance(
        command=f"recorded {spec.protocol_id} case", config_sha256=digest, input_sha256=digest
    )
    splits, trials = [], []
    for key, value in rows.items():
        splits.append(
            BenchmarkSplit(
                split_id=key,
                seed=0,
                fit_count=1,
                holdout_count=1,
                fit_ids_sha256=hashlib.sha256((key + "/fit").encode()).hexdigest(),
                holdout_ids_sha256=hashlib.sha256((key + "/holdout").encode()).hexdigest(),
            )
        )
        trials.append(
            BenchmarkTrial(
                method_id=ANCHOR.method_id,
                split_id=key,
                status="success",
                metrics={metric: 0.0},
                provenance=provenance,
            )
        )
        failed = isinstance(value, str)
        trials.append(
            BenchmarkTrial(
                method_id=spec.method.method_id,
                split_id=key,
                status="failed" if failed else "success",
                metrics={} if failed else {metric: float(value)},
                failure_reason=value if failed else None,
                provenance=provenance,
            )
        )
    return BenchmarkDefinition(
        benchmark_id=f"{spec.protocol_id}-{metric}",
        title=f"{spec.title}: {label}",
        protocol=BenchmarkProtocol(
            protocol_id=spec.protocol_id,
            dataset_id=spec.family,
            dataset_source_sha256=digest,
            data_license=spec.license,
            split_policy="One recording per split, as pre-registered; every case retained.",
            splits=splits,
            initial_estimate_policy="As pre-registered; the reference is never an input.",
            tuning_policy="Method and gates frozen before the single run; nothing tuned after.",
            failure_policy="A case without an estimate is a failed trial with its reason.",
        ),
        metrics=[
            BenchmarkMetricDefinition(
                name=metric,
                label=label,
                unit=unit,
                direction="lower",
                primary=True,
                interpretation="Lower is better; the dataset calibration is a reference estimate.",
            )
        ],
        methods=[ANCHOR, spec.method],
        trials=trials,
        reference_method_id=ANCHOR.method_id,
        bootstrap_samples=2000,
        bootstrap_seed=0,
        limitations=[
            "The zero anchor is not an empirical competitor.",
            "Built from the committed run records; nothing was re-run.",
        ],
        provenance=BenchmarkProvenance(
            generator="tools.build_recorded_audits",
            generator_version=__version__,
            command="python tools/build_recorded_audits.py",
            source_artifacts={
                spec.scores.relative_to(ROOT).as_posix(): digest,
                spec.preregistration.relative_to(ROOT).as_posix(): _digest(spec.preregistration),
            },
            data_verified=True,
            generated_at="recorded",
        ),
    )


def build(spec: Recorded) -> dict[str, Any]:
    prereg = yaml.safe_load(spec.preregistration.read_text(encoding="utf-8"))
    spec.output.mkdir(parents=True, exist_ok=True)
    paths = {}
    for metric, rows in spec.rows().items():
        artifact = aggregate_benchmark_definition(_benchmark(spec, metric, rows))
        path = spec.output / f"{spec.protocol_id}_{metric}.yaml"
        artifact.save(path)
        paths[metric] = path
    sources = {
        spec.scores.relative_to(ROOT).as_posix(): _digest(spec.scores),
        spec.preregistration.relative_to(ROOT).as_posix(): _digest(spec.preregistration),
    }
    requirements = [
        SotaClaimRequirement(
            requirement_id=gate.name,
            phase="audit",
            description=gate.description,
            dataset_family=spec.family,
            independent_rig=True,
            rig_id=spec.rig,
            evidence_path=paths[gate.metric].name,
            evidence_sha256=_digest(paths[gate.metric]),
            method_id=spec.method.method_id,
            metric=None if gate.statistic == "failure_rate" else gate.metric,
            statistic=gate.statistic,  # type: ignore[arg-type]
            comparison=gate.comparison,  # type: ignore[arg-type]
            threshold=gate.threshold,
        )
        for gate in spec.gates(prereg)
    ]
    protocol = SotaAuditProtocol(
        protocol_id=spec.protocol_id,
        scope=SotaClaimScope.model_validate(spec.scope),
        claim_text=spec.claim,
        minimum_dataset_families=1,
        minimum_independent_rigs=1,
        requirements=requirements,
        provenance=SotaAuditProvenance(
            generator="tools.build_recorded_audits",
            generator_version=__version__,
            git_commit=str(prereg["frozen_code_commit"]),
            command=["python", "tools/build_recorded_audits.py"],
            source_sha256=sources,
            created_at="recorded",
        ),
    )
    protocol_path = spec.output / f"{spec.protocol_id}_sota_protocol.yaml"
    protocol.save(protocol_path)
    # audited from the repository root, so the result names its evidence by repo path
    inside = spec.output.is_relative_to(ROOT)
    cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        audit = audit_sota_claim(protocol_path.relative_to(ROOT) if inside else protocol_path)
    finally:
        os.chdir(cwd)
    # the result depends only on the inputs pinned in source_sha256, not on the checkout:
    # no build time and no commit, so a rebuild on any later commit is byte-identical
    audit = audit.model_copy(
        update={
            "provenance": audit.provenance.model_copy(
                update={"created_at": "recorded", "git_commit": None}
            )
        }
    )
    audit_path = spec.output / f"{spec.protocol_id}_sota_audit.yaml"
    audit.save(audit_path)
    return {
        "audit": audit_path.relative_to(ROOT).as_posix()
        if audit_path.is_relative_to(ROOT)
        else str(audit_path),
        "verdict": audit.verdict,
        "requirements": {r.requirement_id: r.status for r in audit.requirements},
    }


# ----------------------------------------------------------------------------- NCLT


NCLT_SCORES = ROOT / "docs/assets/nclt_lidar_wheel/evaluation/summary.json"
NCLT_PREREG = ROOT / "docs/benchmarks/nclt_lidar_wheel_preregistration.yaml"


def _nclt_rows() -> dict[str, dict[str, Value]]:
    scores = json.loads(NCLT_SCORES.read_text(encoding="utf-8"))["scores"]
    rows: dict[str, dict[str, Value]] = {
        "rotation_reference_difference_deg": {},
        "lever_reference_difference_m": {},
        "check_failed_reference": {},
        "axes_beyond_check_fail_bound": {},
    }
    for session, item in scores.items():
        if not item.get("converged"):
            reason = item.get("error") or f"solver status {item.get('solver_status')}"
            for metric in rows:
                rows[metric][session] = reason
            continue
        rows["rotation_reference_difference_deg"][session] = item["rotation_error_deg"]
        rows["lever_reference_difference_m"][session] = item["lever_error_m"]
        rows["check_failed_reference"][session] = float(item["check_status"] == "fail")
        rows["axes_beyond_check_fail_bound"][session] = float(len(item["beyond_fail_bound"]))
    return rows


def _nclt_gates(prereg: dict[str, Any]) -> list[Gate]:
    accuracy = prereg["claims"]["nclt-lidar-wheel-accuracy"]["gates"]
    honesty = prereg["claims"]["nclt-lidar-wheel-check-honesty"]["gates"]
    sessions = len(prereg["sessions"]["evaluation"])
    return [
        Gate("accuracy-converged", "rotation_reference_difference_deg", "failure_rate",
             "less_equal", 0.0, "the estimator converges on every session"),
        Gate("accuracy-rotation", "rotation_reference_difference_deg", "metric_maximum",
             "less_than", float(accuracy["max_rotation_error_deg"]),
             "rotation within 0.5 deg of NCLT's calibration on every session"),
        Gate("accuracy-lever", "lever_reference_difference_m", "metric_maximum",
             "less_than", float(accuracy["max_position_error_m"]),
             "lever within 0.20 m of NCLT's on every session"),
        Gate("honesty-no-fail", "check_failed_reference", "metric_maximum", "less_equal", 0.0,
             "calibrex check does not fail NCLT's calibration on any session"),
        Gate("honesty-fail-bound", "axes_beyond_check_fail_bound", "metric_maximum",
             "less_equal", 0.0, "no rotation axis beyond 2 x max(3 std, 0.5 deg)"),
        Gate("honesty-converged-sessions", "axes_beyond_check_fail_bound", "failure_rate",
             "less_equal", (sessions - honesty["min_converged_sessions"]) / sessions,
             "the estimator converges on at least 3 of the 5 sessions"),
    ]  # fmt: skip


NCLT = Recorded(
    protocol_id="nclt-lidar-wheel-odometry-v1",
    title="NCLT lidar-wheel_odometry",
    method=BenchmarkMethodDefinition(
        method_id="calibrex_lidar_wheel_odometry",
        label="calibrex check lidar-wheel_odometry (LiDAR odometry against wheel odometry)",
        implementation="calibrex_native",
        tool_name="calibrex",
    ),
    scope={
        "modalities": ["lidar", "wheel_odometry"],
        "quantities": ["rotation", "lever_arm"],
        "category": "targetless_lidar_wheel_odometry",
    },
    claim=(
        "On one hour of each of NCLT's sessions 2012-02-02, 2012-05-11, 2012-08-04, "
        "2012-11-04 and 2013-02-23, calibrex check's lidar-wheel_odometry estimator converges, "
        "its rotation is within 0.5 deg and its lever within 0.20 m of NCLT's calibration on "
        "every session; and check does not fail NCLT's calibration, with no rotation axis "
        "beyond check's fail bound, on at least three converged sessions."
    ),
    family="nclt",
    rig="nclt-segway",
    license="ODbL 1.0",
    preregistration=NCLT_PREREG,
    scores=NCLT_SCORES,
    metrics={
        "rotation_reference_difference_deg": ("deg", "rotation difference from NCLT"),
        "lever_reference_difference_m": ("m", "lever difference from NCLT"),
        "check_failed_reference": ("ratio", "check failed NCLT's calibration (1 = yes)"),
        "axes_beyond_check_fail_bound": ("count", "rotation axes beyond check's fail bound"),
    },
    rows=_nclt_rows,
    gates=_nclt_gates,
    output=ROOT / "docs/assets/nclt_lidar_wheel/audit",
)


# ----------------------------------------------------------------------------- KITTI board


BOARD_DIR = ROOT / "docs/assets/kitti_board_camera_lidar"
BOARD_SCORES = BOARD_DIR / "summary.json"
BOARD_PREREG = ROOT / "docs/benchmarks/kitti_board_camera_lidar_preregistration.yaml"


def _board_rows() -> dict[str, dict[str, Value]]:
    scores = json.loads(BOARD_SCORES.read_text(encoding="utf-8"))["scores"]
    rows: dict[str, dict[str, Value]] = {
        "rotation_reference_difference_deg": {},
        "position_reference_difference_m": {},
        "observed_axes_beyond_check_fail_bound": {},
    }
    for day, item in scores.items():
        if not item.get("converged"):
            estimate = json.loads((BOARD_DIR / f"{day}_estimate.json").read_text("utf-8"))
            reason = f"{estimate.get('status')}: {estimate.get('reason')}"
            for metric in rows:
                rows[metric][day] = reason
            continue
        rows["rotation_reference_difference_deg"][day] = item["rotation_error_deg"]
        rows["position_reference_difference_m"][day] = item["position_error_m"]
        rows["observed_axes_beyond_check_fail_bound"][day] = float(len(item["check_fail_axes"]))
    return rows


def _board_gates(prereg: dict[str, Any]) -> list[Gate]:
    accuracy = prereg["claims"]["kitti-board-camera-lidar-accuracy"]["gates"]
    honesty = prereg["claims"]["kitti-board-camera-lidar-check-honesty"]["gates"]
    days = len(prereg["recordings"]["evaluation"])
    return [
        Gate("accuracy-converged", "rotation_reference_difference_deg", "failure_rate",
             "less_equal", 0.0, "the calibration converges on every day"),
        Gate("accuracy-rotation", "rotation_reference_difference_deg", "metric_maximum",
             "less_than", float(accuracy["max_rotation_error_deg"]),
             "rotation within 0.5 deg of that day's KITTI calibration on every day"),
        Gate("accuracy-position", "position_reference_difference_m", "metric_maximum",
             "less_than", float(accuracy["max_position_error_m"]),
             "position within 0.20 m of that day's KITTI calibration on every day"),
        Gate("honesty-fail-bound", "observed_axes_beyond_check_fail_bound", "metric_maximum",
             "less_equal", 0.0, "no observed axis beyond 2 x max(3 std, floor)"),
        Gate("honesty-converged-days", "observed_axes_beyond_check_fail_bound", "failure_rate",
             "less_equal", (days - honesty["min_converged_days"]) / days,
             "the calibration converges on at least 3 of the 5 days"),
    ]  # fmt: skip


BOARD = Recorded(
    protocol_id="kitti-board-camera-lidar-v1",
    title="KITTI calibration recordings, board camera-LiDAR",
    method=BenchmarkMethodDefinition(
        method_id="calibrex_board_camera_lidar",
        label="calibrex checkerboard camera-LiDAR (single shot of several boards)",
        implementation="calibrex_native",
        tool_name="calibrex",
    ),
    scope={
        "modalities": ["camera", "lidar"],
        "quantities": ["rotation", "translation"],
        "category": "target_board_camera_lidar",
    },
    claim=(
        "On KITTI's five calibration recordings, the checkerboard camera-LiDAR calibration "
        "converges on every day and its T_camera_lidar is within 0.5 deg of rotation and "
        "0.20 m of position of that day's KITTI calibration; and on every converged day, and "
        "on at least three, no observed axis is beyond twice max(3 std, floor)."
    ),
    family="kitti_raw",
    rig="kitti-platform",
    license="CC BY-NC-SA 3.0",
    preregistration=BOARD_PREREG,
    scores=BOARD_SCORES,
    metrics={
        "rotation_reference_difference_deg": ("deg", "rotation difference from KITTI"),
        "position_reference_difference_m": ("m", "position difference from KITTI"),
        "observed_axes_beyond_check_fail_bound": ("count", "observed axes beyond the fail bound"),
    },
    rows=_board_rows,
    gates=_board_gates,
    output=BOARD_DIR / "audit",
)


# ----------------------------------------------------------------------------- TUM-VI


TUMVI_DIR = ROOT / "docs/assets/tumvi_camera_focal/evaluation"
TUMVI_SCORES = TUMVI_DIR / "summary.json"
TUMVI_PREREG = ROOT / "docs/benchmarks/tumvi_camera_focal_preregistration.yaml"


def _tumvi_rows() -> dict[str, dict[str, Value]]:
    scores = json.loads(TUMVI_SCORES.read_text(encoding="utf-8"))["scores"]
    rows: dict[str, dict[str, Value]] = {
        "kalibr_check_failed": {},
        "kalibr_focal_judged": {},
        "perturbed_check_passed": {},
        "perturbed_check_failed": {},
    }
    for sequence, variants in scores.items():
        for variant in ("kalibr", "plus3", "minus3"):
            for camera in ("cam0", "cam1"):
                case = variants.get(variant, {}).get(camera)
                key = f"{sequence}/{camera}" + ("" if variant == "kalibr" else f"/{variant}")
                metrics = (
                    ("kalibr_check_failed", "kalibr_focal_judged")
                    if variant == "kalibr"
                    else ("perturbed_check_passed", "perturbed_check_failed")
                )
                for metric in metrics:
                    if case is None:
                        rows[metric][key] = variants.get("error") or "no camera-focal record"
                        continue
                    value = {
                        "kalibr_check_failed": case["status"] == "fail",
                        "kalibr_focal_judged": bool(case["judged_axes"]),
                        "perturbed_check_passed": case["status"] == "pass",
                        "perturbed_check_failed": case["status"] == "fail",
                    }[metric]
                    rows[metric][key] = float(value)
    return rows


def _tumvi_gates(prereg: dict[str, Any]) -> list[Gate]:
    honesty = prereg["claims"]["tumvi-camera-focal-check-honesty"]["gates"]
    detection = prereg["claims"]["tumvi-camera-focal-check-detection"]["gates"]
    return [
        Gate("honesty-no-fail", "kalibr_check_failed", "metric_maximum", "less_equal",
             float(honesty["max_kalibr_fails"]), "check never fails Kalibr's focal lengths"),
        Gate("honesty-judged", "kalibr_focal_judged", "metric_mean", "greater_equal",
             float(honesty["min_kalibr_judged_fraction"]),
             "check judges a focal length on at least 25 % of the Kalibr cases"),
        Gate("detection-no-pass", "perturbed_check_passed", "metric_maximum", "less_equal",
             float(detection["max_perturbed_passes"]),
             "check passes none of the 3 %-wrong focal lengths"),
        Gate("detection-fail", "perturbed_check_failed", "metric_mean", "greater_equal",
             float(detection["min_perturbed_fail_fraction"]),
             "check fails at least 25 % of the 3 %-wrong focal lengths"),
        Gate("completion", "kalibr_check_failed", "failure_rate", "less_equal", 0.0,
             "every case produced a camera-focal record"),
    ]  # fmt: skip


TUMVI = Recorded(
    protocol_id="tumvi-camera-focal-check-v1",
    title="TUM-VI rooms, camera-focal check",
    method=BenchmarkMethodDefinition(
        method_id="calibrex_camera_focal_check",
        label="calibrex check camera-focal (camera angular rates against the gyro)",
        implementation="calibrex_native",
        tool_name="calibrex",
    ),
    scope={
        "modalities": ["camera"],
        "quantities": ["intrinsics"],
        "category": "targetless_camera_focal_check",
    },
    claim=(
        "On TUM-VI's room2-room6 (both fisheye cameras), calibrex check never fails "
        "camera-focal with Kalibr's camchain and judges a focal length on at least 25 % of "
        "the cases; with the focal lengths 3 % long or short it passes none and fails at "
        "least 25 %."
    ),
    family="tum_vi",
    rig="tumvi-handheld",
    license="CC BY 4.0",
    preregistration=TUMVI_PREREG,
    scores=TUMVI_SCORES,
    metrics={
        "kalibr_check_failed": ("ratio", "check failed Kalibr's focal lengths (1 = yes)"),
        "kalibr_focal_judged": ("ratio", "check judged a focal length (1 = yes)"),
        "perturbed_check_passed": ("ratio", "check passed a 3 %-wrong focal length (1 = yes)"),
        "perturbed_check_failed": ("ratio", "check failed a 3 %-wrong focal length (1 = yes)"),
    },
    rows=_tumvi_rows,
    gates=_tumvi_gates,
    output=ROOT / "docs/assets/tumvi_camera_focal/audit",
)


# ----------------------------------------------------------------------------- TUM-VI camera-imu


TUMVI_IMU_DIR = ROOT / "docs/assets/tumvi_camera_imu/evaluation"
TUMVI_IMU_SCORES = TUMVI_IMU_DIR / "summary.json"
TUMVI_IMU_PREREG = ROOT / "docs/benchmarks/tumvi_camera_imu_preregistration.yaml"
TUMVI_IMU_TURNS = {"rx2": "roll", "ry2": "pitch", "rz2": "yaw"}


def _tumvi_imu_rows() -> dict[str, dict[str, Value]]:
    scores = json.loads(TUMVI_IMU_SCORES.read_text(encoding="utf-8"))["scores"]
    rows: dict[str, dict[str, Value]] = {
        "kalibr_check_failed": {},
        "kalibr_rotation_judged": {},
        "turned_check_passed": {},
        "turned_axis_failed": {},
    }
    for sequence, variants in scores.items():
        for variant in ("kalibr", *TUMVI_IMU_TURNS):
            for camera in ("cam0", "cam1"):
                case = variants.get(variant, {}).get(camera)
                key = f"{sequence}/{camera}" + ("" if variant == "kalibr" else f"/{variant}")
                metrics = (
                    ("kalibr_check_failed", "kalibr_rotation_judged")
                    if variant == "kalibr"
                    else ("turned_check_passed", "turned_axis_failed")
                )
                for metric in metrics:
                    if case is None:
                        rows[metric][key] = variants.get("error") or "no camera-imu record"
                        continue
                    axis = case["axes"].get(TUMVI_IMU_TURNS.get(variant, ""), {})
                    value = {
                        "kalibr_check_failed": case["status"] == "fail",
                        "kalibr_rotation_judged": bool(case["judged_axes"]),
                        "turned_check_passed": case["status"] == "pass",
                        "turned_axis_failed": case["status"] == "fail"
                        and axis.get("status") == "fail",
                    }[metric]
                    rows[metric][key] = float(value)
    return rows


def _tumvi_imu_gates(prereg: dict[str, Any]) -> list[Gate]:
    honesty = prereg["claims"]["tumvi-camera-imu-check-honesty"]["gates"]
    detection = prereg["claims"]["tumvi-camera-imu-check-detection"]["gates"]
    return [
        Gate("honesty-no-fail", "kalibr_check_failed", "metric_maximum", "less_equal",
             float(honesty["max_kalibr_fails"]), "check never fails Kalibr's T_cam_imu"),
        Gate("honesty-judged", "kalibr_rotation_judged", "metric_mean", "greater_equal",
             float(honesty["min_kalibr_judged_fraction"]),
             "check judges a rotation axis on at least 50 % of the Kalibr cases"),
        Gate("detection-no-pass", "turned_check_passed", "metric_maximum", "less_equal",
             float(detection["max_perturbed_passes"]),
             "check passes none of the 2 deg-turned T_cam_imu"),
        Gate("detection-fail", "turned_axis_failed", "metric_mean", "greater_equal",
             float(detection["min_perturbed_detected_fraction"]),
             "check fails at least 50 % of the turned cases on the turned axis"),
        Gate("completion", "kalibr_check_failed", "failure_rate", "less_equal", 0.0,
             "every case produced a camera-imu record"),
    ]  # fmt: skip


TUMVI_IMU = Recorded(
    protocol_id="tumvi-camera-imu-check-v1",
    title="TUM-VI corridors, camera-imu check",
    method=BenchmarkMethodDefinition(
        method_id="calibrex_camera_imu_check",
        label="calibrex check camera-imu (camera rotations against the gyro)",
        implementation="calibrex_native",
        tool_name="calibrex",
    ),
    scope={
        "modalities": ["camera", "imu"],
        "quantities": ["rotation"],
        "category": "targetless_camera_imu_check",
    },
    claim=(
        "On TUM-VI's corridor1-corridor5 (both fisheye cameras), calibrex check never fails "
        "camera-imu with Kalibr's camchain and judges a rotation axis on at least 50 % of "
        "the cases; with T_cam_imu turned 2 deg about a camera axis it passes none and "
        "fails at least 50 % on the turned axis."
    ),
    family="tum_vi",
    rig="tumvi-handheld",
    license="CC BY 4.0",
    preregistration=TUMVI_IMU_PREREG,
    scores=TUMVI_IMU_SCORES,
    metrics={
        "kalibr_check_failed": ("ratio", "check failed Kalibr's T_cam_imu (1 = yes)"),
        "kalibr_rotation_judged": ("ratio", "check judged a rotation axis (1 = yes)"),
        "turned_check_passed": ("ratio", "check passed a 2 deg-turned T_cam_imu (1 = yes)"),
        "turned_axis_failed": ("ratio", "check failed the turned axis (1 = yes)"),
    },
    rows=_tumvi_imu_rows,
    gates=_tumvi_imu_gates,
    output=ROOT / "docs/assets/tumvi_camera_imu/audit",
)


# ----------------------------------------------------------------------------- TUM-VI camera-imu v2


TUMVI_IMU_V2_DIR = ROOT / "docs/assets/tumvi_camera_imu_v2/evaluation"
TUMVI_IMU_V2_SCORES = TUMVI_IMU_V2_DIR / "summary.json"
TUMVI_IMU_V2_PREREG = ROOT / "docs/benchmarks/tumvi_camera_imu_v2_preregistration.yaml"


def _tumvi_imu_v2_rows() -> dict[str, dict[str, Value]]:
    scores = json.loads(TUMVI_IMU_V2_SCORES.read_text(encoding="utf-8"))["scores"]
    rows: dict[str, dict[str, Value]] = {
        "kalibr_check_failed": {},
        "kalibr_full_pass": {},
        "turned_unflagged": {},
        "turned_axis_failed": {},
    }
    for sequence, variants in scores.items():
        for variant in ("kalibr", *TUMVI_IMU_TURNS):
            for camera in ("cam0", "cam1"):
                case = variants.get(variant, {}).get(camera)
                key = f"{sequence}/{camera}" + ("" if variant == "kalibr" else f"/{variant}")
                metrics = (
                    ("kalibr_check_failed", "kalibr_full_pass")
                    if variant == "kalibr"
                    else ("turned_unflagged", "turned_axis_failed")
                )
                for metric in metrics:
                    if case is None:
                        rows[metric][key] = variants.get("error") or "no camera-imu record"
                        continue
                    axis = case["axes"].get(TUMVI_IMU_TURNS.get(variant, ""), {})
                    value = {
                        "kalibr_check_failed": case["status"] == "fail",
                        "kalibr_full_pass": case["status"] == "pass" and case["coverage"] == "full",
                        "turned_unflagged": case["exit_status"] == 0,
                        "turned_axis_failed": case["status"] == "fail"
                        and axis.get("status") == "fail",
                    }[metric]
                    rows[metric][key] = float(value)
    return rows


def _tumvi_imu_v2_gates(prereg: dict[str, Any]) -> list[Gate]:
    honesty = prereg["claims"]["tumvi-camera-imu-check-v2-honesty"]["gates"]
    detection = prereg["claims"]["tumvi-camera-imu-check-v2-detection"]["gates"]
    return [
        Gate("honesty-no-fail", "kalibr_check_failed", "metric_maximum", "less_equal",
             float(honesty["max_kalibr_fails"]), "check never fails Kalibr's T_cam_imu"),
        Gate("honesty-full-pass", "kalibr_full_pass", "metric_mean", "greater_equal",
             float(honesty["min_kalibr_full_pass_fraction"]),
             "check passes Kalibr's T_cam_imu with full coverage on at least 50 % of cases"),
        Gate("detection-none-unflagged", "turned_unflagged", "metric_maximum", "less_equal",
             float(detection["max_turned_unflagged"]),
             "--fail-on partial flags every 2 deg-turned T_cam_imu"),
        Gate("detection-fail", "turned_axis_failed", "metric_mean", "greater_equal",
             float(detection["min_turned_detected_fraction"]),
             "check fails at least 50 % of the turned cases on the turned axis"),
        Gate("completion", "kalibr_check_failed", "failure_rate", "less_equal", 0.0,
             "every case produced a camera-imu record"),
    ]  # fmt: skip


TUMVI_IMU_V2 = Recorded(
    protocol_id="tumvi-camera-imu-check-v2",
    title="TUM-VI slides and magistrale, camera-imu check with --fail-on partial",
    method=BenchmarkMethodDefinition(
        method_id="calibrex_camera_imu_check_fail_on_partial",
        label="calibrex check camera-imu --fail-on partial",
        implementation="calibrex_native",
        tool_name="calibrex",
    ),
    scope={
        "modalities": ["camera", "imu"],
        "quantities": ["rotation"],
        "category": "targetless_camera_imu_check",
    },
    claim=(
        "On TUM-VI's slides1-slides3, magistrale3 and magistrale5 (both fisheye cameras), "
        "calibrex check never fails camera-imu with Kalibr's camchain and passes it with full "
        "coverage on at least 50 % of the cases; with T_cam_imu turned 2 deg about a camera "
        "axis, --fail-on partial flags every case and check fails at least 50 % on the turned "
        "axis."
    ),
    family="tum_vi",
    rig="tumvi-handheld",
    license="CC BY 4.0",
    preregistration=TUMVI_IMU_V2_PREREG,
    scores=TUMVI_IMU_V2_SCORES,
    metrics={
        "kalibr_check_failed": ("ratio", "check failed Kalibr's T_cam_imu (1 = yes)"),
        "kalibr_full_pass": ("ratio", "check passed Kalibr's T_cam_imu, all axes judged (1 = yes)"),
        "turned_unflagged": ("ratio", "--fail-on partial let a turned T_cam_imu through (1 = yes)"),
        "turned_axis_failed": ("ratio", "check failed the turned axis (1 = yes)"),
    },
    rows=_tumvi_imu_v2_rows,
    gates=_tumvi_imu_v2_gates,
    output=ROOT / "docs/assets/tumvi_camera_imu_v2/audit",
)


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    for spec in (BOARD, NCLT, TUMVI, TUMVI_IMU, TUMVI_IMU_V2):
        print(json.dumps(build(spec)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
