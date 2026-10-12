"""Reproducible full-scale KITTI Camera-LiDAR falsification benchmark."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, cast

from pydantic import Field

from calibrex import __version__
from calibrex.core.assessment import AssessmentArtifact
from calibrex.core.evidence_bundle import (
    EvidenceBundleVerification,
    verify_evidence_bundle,
    write_evidence_bundle_verification,
)
from calibrex.core.geometry import SE3
from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.result import StrictModel
from calibrex.data.kitti import (
    find_camera_lidar_pairs,
    read_velodyne_to_camera_transform,
)
from calibrex.evaluation.kitti_edge_falsification import (
    FrameSelection,
    KITTIEdgeDecision,
    KITTIEdgeTrial,
    run_kitti_edge_decision,
)
from calibrex.pipelines.calibrate import CalibrationRunOptions, run_calibration

KITTI_FALSIFICATION_SCHEMA_VERSION: Literal["slac.kitti_lidar_camera_falsification/v0.1"] = (
    "slac.kitti_lidar_camera_falsification/v0.1"
)
KITTI_SEQUENCE: Literal["2011_09_26_drive_0005_sync"] = "2011_09_26_drive_0005_sync"
KITTI_SOURCE_URL = "https://www.cvlibs.net/datasets/kitti/raw_data.php"
BenchmarkStatus = Literal["pass", "fail", "inconclusive"]
# The metrics a published benchmark reports per trial: the held-out edge evidence the
# policy gates on, and the known-bad controls that make a pass meaningful.
KITTI_HEADLINE_METRICS: tuple[str, ...] = (
    "lidar_camera_edge_alignment_score",
    "lidar_camera_depth_edge_alignment_score",
    "lidar_camera_projection_ratio",
    "lidar_camera_projection_frame_count",
    "lidar_camera_perturbation_case_count",
    "lidar_camera_perturbation_detectable_fraction",
    "lidar_camera_perturbation_mandatory_case_count",
    "lidar_camera_perturbation_mandatory_detectable_count",
)


class KITTIBenchmarkArtifactDigest(StrictModel):
    """Digest-bound file used or produced by one benchmark trial."""

    role: str
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class KITTIBenchmarkMetric(StrictModel):
    """One headline metric of a trial, as its ``metrics.json`` records it."""

    metric_id: str
    value: float | None = None
    train: float | None = None
    holdout: float | None = None
    grade: str | None = None
    unit: str | None = None


class KITTIBenchmarkRule(StrictModel):
    """One gate of the trial's falsification policy and its outcome."""

    rule_id: str
    status: Literal["pass", "fail", "inconclusive"]
    reason: str


# Evidence checks that compare the candidate with the dataset's own calibration rather
# than with the images and scans: a candidate rejected only by these is not falsified by
# the data.
KITTI_REFERENCE_CHECKS: frozenset[str] = frozenset({"Dataset Reference Consistency"})


class KITTIBenchmarkEvidenceCheck(StrictModel):
    """One LiDAR-camera evidence check of a trial (``evidence.json`` summaries)."""

    check: str
    status: str
    evidence: str
    uses_dataset_reference: bool = Field(
        description="the check compares with the dataset's calibration, not with the data"
    )


class KITTIBenchmarkTrial(StrictModel):
    """One frozen candidate evaluated by the KITTI protocol."""

    candidate_id: Literal["dataset_reference", "known_bad"]
    expected_assessment: Literal["pass", "fail"]
    assessment_status: Literal["pass", "fail", "inconclusive"]
    bundle_valid: bool
    artifacts: list[KITTIBenchmarkArtifactDigest]
    assessment_reason: str | None = None
    rules: list[KITTIBenchmarkRule] = Field(default_factory=list)
    metrics: list[KITTIBenchmarkMetric] = Field(
        default_factory=list,
        description="headline metrics of the trial (see KITTI_HEADLINE_METRICS)",
    )
    evidence_checks: list[KITTIBenchmarkEvidenceCheck] = Field(
        default_factory=list,
        description="the LiDAR-camera evidence checks behind the decision-boundary gate",
    )


class KITTIFalsificationProvenance(StrictModel):
    """Benchmark generation lineage."""

    producer: Literal["calibrex"] = "calibrex"
    calibrex_version: str
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_commit: str | None = None
    source_config: str
    source_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class KITTIFalsificationBenchmarkArtifact(StrictModel):
    """Decision artifact for the full-scale KITTI falsification run."""

    schema_version: Literal["slac.kitti_lidar_camera_falsification/v0.1"] = (
        KITTI_FALSIFICATION_SCHEMA_VERSION
    )
    benchmark_id: Literal["kitti_raw_2011_09_26_drive_0005_camera_lidar"] = (
        "kitti_raw_2011_09_26_drive_0005_camera_lidar"
    )
    dataset_sequence: Literal["2011_09_26_drive_0005_sync"] = KITTI_SEQUENCE
    dataset_path: str
    source_url: str
    source_note: str
    selected_frame_ids: list[str] = Field(min_length=2)
    split_seed: int
    holdout_ratio: float = Field(gt=0.0, lt=1.0)
    projection_sample_points: int = Field(gt=0)
    protocol_id: str
    known_bad_declaration: str
    known_bad_transform_camera_lidar: dict[str, list[float]]
    status: BenchmarkStatus
    reason: str
    trials: list[KITTIBenchmarkTrial] = Field(min_length=2, max_length=2)
    input_artifacts: list[KITTIBenchmarkArtifactDigest] = Field(min_length=2)
    provenance: KITTIFalsificationProvenance
    data_only_status: BenchmarkStatus | None = Field(
        default=None,
        description="the decision of the images and scans alone (edge alignment with the "
        "calibrex check rule, frames spread over the drive); status above also counts the "
        "comparison with the dataset's calibration",
    )
    data_only: list[KITTIEdgeDecision] = Field(
        default_factory=list,
        description="data-only decisions: the spread selection first, then sensitivity rows",
    )

    def save(self, path: str | Path) -> None:
        """Write the benchmark artifact as JSON or YAML."""

        write_mapping(Path(path), self.model_dump(mode="json", exclude_none=True))


def kitti_falsification_json_schema() -> dict[str, object]:
    """Return the full-scale KITTI falsification JSON Schema."""

    return KITTIFalsificationBenchmarkArtifact.model_json_schema()


def run_kitti_falsification_benchmark(
    dataset_path: str | Path,
    *,
    config_path: str | Path,
    output_dir: str | Path,
    max_frames: int = 50,
    projection_sample_points: int = 4000,
    seed: int = 20260729,
    source_url: str = KITTI_SOURCE_URL,
    source_note: str = "official KITTI raw download; user-supplied local copy",
    data_only: bool = True,
) -> KITTIFalsificationBenchmarkArtifact:
    """Run reference and declared known-bad candidates under one frozen protocol."""

    commit = git_commit()  # the code that runs, read before a long run can see new commits
    dataset = Path(dataset_path)
    source_config = Path(config_path)
    output = Path(output_dir)
    _require_official_sequence_layout(dataset)
    reference = read_velodyne_to_camera_transform(dataset)
    if reference is None:
        msg = "KITTI calib_velo_to_cam.txt could not be parsed"
        raise ValueError(msg)
    pairs = find_camera_lidar_pairs(dataset, max_pairs=max_frames)
    if len(pairs) < 2:
        msg = "KITTI benchmark requires at least two camera-LiDAR frame pairs"
        raise ValueError(msg)
    selected_frame_ids = [f"camera:{pair.camera_index}:lidar:{pair.lidar_index}" for pair in pairs]

    output.mkdir(parents=True, exist_ok=True)
    frozen_config = output / "frozen-config.yaml"
    config_payload = _frozen_config_payload(
        source_config,
        dataset,
        output,
        max_frames=len(pairs),
        projection_sample_points=projection_sample_points,
        seed=seed,
    )
    write_mapping(frozen_config, config_payload)

    candidates = kitti_benchmark_candidates(reference)
    trials: list[KITTIBenchmarkTrial] = []
    for candidate_id, transform in candidates.items():
        trial_dir = output / candidate_id
        candidate_path = output / f"{candidate_id}-candidate.yaml"
        _write_candidate(candidate_path, transform, candidate_id=candidate_id)
        result = run_calibration(
            frozen_config,
            CalibrationRunOptions(
                output_dir=trial_dir,
                candidate_extrinsics=(candidate_path,),
                seed=seed,
            ),
        )
        if result is None:
            msg = f"KITTI benchmark trial {candidate_id} did not produce a result"
            raise RuntimeError(msg)
        assessment = AssessmentArtifact.model_validate(read_mapping(trial_dir / "assessment.json"))
        verification = _verify_trial(trial_dir)
        reason, rules, metrics, checks = summarize_kitti_trial(trial_dir)
        trials.append(
            KITTIBenchmarkTrial(
                candidate_id=candidate_id,
                expected_assessment="pass" if candidate_id == "dataset_reference" else "fail",
                assessment_status=assessment.status,
                bundle_valid=verification.valid,
                artifacts=_trial_artifacts(trial_dir, output),
                assessment_reason=reason,
                rules=rules,
                metrics=metrics,
                evidence_checks=checks,
            )
        )

    status, reason = _benchmark_decision(trials)
    protocol = read_mapping(output / "dataset_reference" / "protocol.json")
    source_config_digest = _required_digest(source_config)
    benchmark = KITTIFalsificationBenchmarkArtifact(
        dataset_path=str(dataset),
        source_url=source_url,
        source_note=source_note,
        selected_frame_ids=selected_frame_ids,
        split_seed=seed,
        holdout_ratio=float(
            cast(int | float, _mapping(config_payload, "evaluation")["holdout_ratio"])
        ),
        projection_sample_points=projection_sample_points,
        protocol_id=_lidar_camera_protocol_id(protocol),
        known_bad_declaration=(
            "right-composed camera-frame +5 deg yaw and +0.25 m x translation; "
            "declared before scoring"
        ),
        known_bad_transform_camera_lidar={
            "translation_m": list(candidates["known_bad"].translation_m),
            "rotation_quat_xyzw": list(candidates["known_bad"].rotation_quat_xyzw),
        },
        status=status,
        reason=reason,
        trials=trials,
        input_artifacts=[
            _artifact_digest(source_config, "source_config"),
            _artifact_digest(
                _find_calibration(dataset, "calib_velo_to_cam.txt"),
                "kitti_calibration",
            ),
            _artifact_digest(
                _find_calibration(dataset, "calib_cam_to_cam.txt"),
                "kitti_calibration",
            ),
            *[
                artifact
                for pair in pairs
                for artifact in (
                    _artifact_digest(Path(pair.camera_path), "kitti_camera_image"),
                    _artifact_digest(Path(pair.lidar_path), "kitti_velodyne_scan"),
                )
            ],
        ],
        provenance=KITTIFalsificationProvenance(
            calibrex_version=__version__,
            git_commit=commit,
            source_config=str(source_config),
            source_config_sha256=source_config_digest,
        ),
    )
    if data_only:
        benchmark = attach_data_only_decisions(benchmark, dataset, frame_count=len(pairs))
    benchmark.save(output / "benchmark.json")
    return benchmark


DATA_ONLY_SELECTIONS: tuple[FrameSelection, ...] = ("spread", "first")


def kitti_benchmark_candidates(
    reference: SE3,
) -> dict[Literal["dataset_reference", "known_bad"], SE3]:
    """The dataset reference and the declared known-bad candidate (+5 deg yaw, +0.25 m x)."""

    known_bad_delta = SE3(
        (0.25, 0.0, 0.0),
        (0.0, 0.0, math.sin(math.radians(5.0) / 2.0), math.cos(math.radians(5.0) / 2.0)),
    )
    return {"dataset_reference": reference, "known_bad": reference.compose(known_bad_delta)}


def attach_data_only_decisions(
    benchmark: KITTIFalsificationBenchmarkArtifact,
    dataset: str | Path,
    *,
    frame_count: int | None = None,
) -> KITTIFalsificationBenchmarkArtifact:
    """The benchmark with its data-only decisions (spread selection, then the first frames)."""

    reference = read_velodyne_to_camera_transform(dataset)
    if reference is None:
        msg = "KITTI calib_velo_to_cam.txt could not be parsed"
        raise ValueError(msg)
    candidates = kitti_benchmark_candidates(reference)
    declared = benchmark.known_bad_transform_camera_lidar
    if [round(v, 12) for v in candidates["known_bad"].translation_m] != [
        round(v, 12) for v in declared["translation_m"]
    ]:
        msg = "the dataset's reference does not reproduce the benchmark's known-bad candidate"
        raise ValueError(msg)
    count = frame_count or len(benchmark.selected_frame_ids)
    decisions = [
        run_kitti_edge_decision(dataset, candidates, frame_count=count, selection=selection)
        for selection in DATA_ONLY_SELECTIONS
    ]
    return benchmark.model_copy(
        update={"data_only_status": decisions[0].status, "data_only": decisions}
    )


def _frozen_config_payload(
    source_config: Path,
    dataset: Path,
    output: Path,
    *,
    max_frames: int,
    projection_sample_points: int,
    seed: int,
) -> dict[str, object]:
    payload = read_mapping(source_config)
    dataset_payload = _mapping(payload, "dataset")
    dataset_payload["type"] = "kitti_raw"
    dataset_payload["path"] = str(dataset)
    project = _mapping(payload, "project")
    project["output_dir"] = str(output)
    solver = _mapping(payload, "solver")
    solver["seed"] = seed
    evaluation = _mapping(payload, "evaluation")
    kitti = _mapping(evaluation, "kitti")
    kitti["max_projection_pairs"] = max_frames
    kitti["projection_sample_points"] = projection_sample_points
    kitti["use_frame_graph_candidate"] = True
    pipeline = _mapping(payload, "pipeline")
    factors = _mapping(pipeline, "factors")
    comparison = factors.setdefault(
        "lidar_camera_baseline_comparison",
        {"enabled": True, "options": {}},
    )
    if not isinstance(comparison, dict):
        msg = "pipeline.factors.lidar_camera_baseline_comparison must be a mapping"
        raise ValueError(msg)
    comparison["enabled"] = True
    options = comparison.setdefault("options", {})
    if not isinstance(options, dict):
        msg = "lidar_camera_baseline_comparison.options must be a mapping"
        raise ValueError(msg)
    options["max_frames"] = max_frames
    options["max_points"] = projection_sample_points
    return payload


def _write_candidate(path: Path, transform: SE3, *, candidate_id: str) -> None:
    write_mapping(
        path,
        {
            "candidate_extrinsics": {
                "T_base_link_camera0": {
                    "parent": "base_link",
                    "child": "camera0",
                    "translation_m": [0.0, 0.0, 0.0],
                    "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "provenance": {
                        "producer": "dataset_provider",
                        "execution_mode": "imported",
                        "role_in_comparison": "candidate",
                        "evidence_level": "dataset_provided",
                        "source": "KITTI raw calibration",
                    },
                },
                "T_base_link_lidar0": {
                    "parent": "base_link",
                    "child": "lidar0",
                    "translation_m": list(transform.translation_m),
                    "rotation_quat_xyzw": list(transform.rotation_quat_xyzw),
                    "provenance": {
                        "producer": (
                            "dataset_provider" if candidate_id == "dataset_reference" else "human"
                        ),
                        "execution_mode": "imported",
                        "role_in_comparison": (
                            "selected_reference"
                            if candidate_id == "dataset_reference"
                            else "candidate"
                        ),
                        "evidence_level": (
                            "dataset_provided"
                            if candidate_id == "dataset_reference"
                            else "synthetic_truth"
                        ),
                        "source": "KITTI raw calibration plus declared perturbation",
                    },
                },
            }
        },
    )


def _verify_trial(trial_dir: Path) -> EvidenceBundleVerification:
    bundle = trial_dir / "bundle.json"
    verification = verify_evidence_bundle(bundle, require_raw_recomputed=True)
    return write_evidence_bundle_verification(
        trial_dir / "verification.json",
        verification,
        source_bundle_path=bundle,
    )


def summarize_kitti_trial(
    trial_dir: str | Path,
) -> tuple[
    str,
    list[KITTIBenchmarkRule],
    list[KITTIBenchmarkMetric],
    list[KITTIBenchmarkEvidenceCheck],
]:
    """The assessment reason, policy gates, headline metrics and evidence checks of a trial."""

    directory = Path(trial_dir)
    assessment = read_mapping(directory / "assessment.json")
    summaries = read_mapping(directory / "evidence.json").get("summaries", [])
    metrics = read_mapping(directory / "metrics.json").get("metrics", {})
    if not isinstance(metrics, dict):
        msg = f"{directory / 'metrics.json'} has no metrics mapping"
        raise ValueError(msg)
    rules = assessment.get("rules", [])
    return (
        str(assessment.get("reason", "")),
        [
            KITTIBenchmarkRule(
                rule_id=str(rule["rule_id"]),
                status=rule["status"],
                reason=str(rule.get("reason", "")),
            )
            for rule in (rules if isinstance(rules, list) else [])
            if isinstance(rule, dict)
        ],
        [
            KITTIBenchmarkMetric(
                metric_id=metric_id,
                **{
                    key: metrics[metric_id][key]
                    for key in ("value", "train", "holdout", "grade", "unit")
                    if key in metrics[metric_id]
                },
            )
            for metric_id in KITTI_HEADLINE_METRICS
            if isinstance(metrics.get(metric_id), dict)
        ],
        [
            KITTIBenchmarkEvidenceCheck(
                check=str(item["check"]),
                status=str(item["status"]),
                evidence=str(item.get("evidence", "")),
                uses_dataset_reference=str(item["check"]) in KITTI_REFERENCE_CHECKS,
            )
            for item in (summaries if isinstance(summaries, list) else [])
            if isinstance(item, dict) and item.get("family") == "lidar_camera"
        ],
    )


def render_kitti_falsification_markdown(benchmark: KITTIFalsificationBenchmarkArtifact) -> str:
    """A deterministic Markdown summary of a benchmark artifact (for the published page)."""

    def number(value: float | None) -> str:
        if value is None:
            return "-"
        return f"{value:.0f}" if float(value).is_integer() else f"{value:.4f}"

    trials: dict[str, KITTIBenchmarkTrial] = {
        trial.candidate_id: trial for trial in benchmark.trials
    }
    order = ("dataset_reference", "known_bad")
    lines = [
        f"**Status: `{benchmark.status}`**: {benchmark.reason}.",
        "",
        f"- Sequence: `{benchmark.dataset_sequence}`, {len(benchmark.selected_frame_ids)} "
        f"camera-LiDAR frame pairs, seed {benchmark.split_seed}, holdout ratio "
        f"{benchmark.holdout_ratio:g}, {benchmark.projection_sample_points} projected points "
        "per frame.",
        f"- Protocol: `{benchmark.protocol_id}`.",
        f"- Known-bad candidate: {benchmark.known_bad_declaration}.",
        f"- Generated by Calibrex {benchmark.provenance.calibrex_version}"
        + (
            f" at commit `{benchmark.provenance.git_commit[:12]}`"
            if benchmark.provenance.git_commit
            else ""
        )
        + f", {benchmark.provenance.generated_at[:10]}.",
        "",
        "| | " + " | ".join(f"`{name}`" for name in order) + " |",
        "| --- | --- | --- |",
        "| expected | " + " | ".join(trials[name].expected_assessment for name in order) + " |",
        "| assessment | "
        + " | ".join(f"**{trials[name].assessment_status}**" for name in order)
        + " |",
        "| evidence bundle verified | "
        + " | ".join("yes" if trials[name].bundle_valid else "no" for name in order)
        + " |",
    ]
    metric_ids = [metric.metric_id for metric in trials["dataset_reference"].metrics]
    for metric_id in metric_ids:
        cells = []
        for name in order:
            metric = next((m for m in trials[name].metrics if m.metric_id == metric_id), None)
            if metric is None:
                cells.append("-")
            elif metric.holdout is not None:
                cells.append(f"{number(metric.train)} train / {number(metric.holdout)} holdout")
            else:
                cells.append(number(metric.value))
        lines.append(f"| `{metric_id}` | " + " | ".join(cells) + " |")
    lines += ["", "Policy gates:", "", "| Gate | " + " | ".join(f"`{n}`" for n in order) + " |"]
    lines.append("| --- | --- | --- |")
    rule_ids = [rule.rule_id for rule in trials["dataset_reference"].rules]
    for rule_id in rule_ids:
        cells = []
        for name in order:
            rule = next((r for r in trials[name].rules if r.rule_id == rule_id), None)
            cells.append("-" if rule is None else rule.status)
        lines.append(f"| `{rule_id}` | " + " | ".join(cells) + " |")
    check_names = [item.check for item in trials["dataset_reference"].evidence_checks]
    if check_names:
        lines += [
            "",
            "LiDAR-camera evidence checks (the decision-boundary gate combines the others):",
            "",
            "| Check | " + " | ".join(f"`{n}`" for n in order) + " |",
            "| --- | --- | --- |",
        ]
        for check in check_names:
            cells = []
            for name in order:
                item = next((c for c in trials[name].evidence_checks if c.check == check), None)
                cells.append("-" if item is None else item.status)
            marker = (
                " (compares with the dataset's calibration)"
                if check in KITTI_REFERENCE_CHECKS
                else ""
            )
            lines.append(f"| {check}{marker} | " + " | ".join(cells) + " |")
    for name in order:
        reason = trials[name].assessment_reason
        if reason:
            lines += ["", f"`{name}`: {reason}."]
        failed = [
            item
            for item in trials[name].evidence_checks
            if item.status == "fail" and item.check != "Decision Boundary"
        ]
        if failed and all(item.uses_dataset_reference for item in failed):
            lines += [
                "",
                f"The protocol rejects `{name}` only through "
                + ", ".join(f"*{item.check}* ({item.evidence})" for item in failed)
                + ": a comparison with the dataset's own calibration. None of the protocol's "
                "checks of the images and scans rejects it.",
            ]
    if benchmark.data_only:
        lines += [
            "",
            f"**Data-only status: `{benchmark.data_only_status}`**: the images and scans "
            "alone, judged by edge alignment with the `calibrex check` per-axis rule "
            "(rotation only; `error / tolerance` in deg).",
            "",
            "| Frames | `dataset_reference` | `known_bad` | Decision |",
            "| --- | --- | --- | --- |",
        ]
        for decision in benchmark.data_only:
            by_id: dict[str, KITTIEdgeTrial] = {
                trial.candidate_id: trial for trial in decision.trials
            }
            cells = []
            for name in order:
                trial = by_id[name]
                axes = ", ".join(
                    f"{axis.name} {abs(axis.candidate_error_deg):.2f} / {axis.tolerance_deg:.2f}"
                    + ("" if axis.status == "pass" else f" {axis.status}")
                    for axis in trial.axes
                )
                unchecked = (
                    f"; unchecked {', '.join(trial.unchecked_rotation_axes)}"
                    if trial.unchecked_rotation_axes
                    else ""
                )
                cells.append(f"**{trial.verdict}** ({axes}{unchecked})")
            label = (
                f"{len(decision.frame_ids)} spread over the drive"
                if decision.frame_selection == "spread"
                else f"first {len(decision.frame_ids)} (the protocol's)"
            )
            lines.append(f"| {label} | " + " | ".join(cells) + f" | {decision.status} |")
    return "\n".join(lines) + "\n"


def _trial_artifacts(
    trial_dir: Path, output: Path | None = None
) -> list[KITTIBenchmarkArtifactDigest]:
    names = (
        "result.yaml",
        "evidence.json",
        "policy.json",
        "assessment.json",
        "observability.json",
        "protocol.json",
        "transforms.json",
        "bundle.json",
        "verification.json",
    )
    artifacts = [_artifact_digest(trial_dir / name, name.rsplit(".", 1)[0]) for name in names]
    if output is None:
        return artifacts
    # Paths relative to the benchmark's output directory, so the artifact can be moved.
    return [
        artifact.model_copy(update={"path": Path(artifact.path).relative_to(output).as_posix()})
        for artifact in artifacts
    ]


def _benchmark_decision(
    trials: list[KITTIBenchmarkTrial],
) -> tuple[BenchmarkStatus, str]:
    by_id = {trial.candidate_id: trial for trial in trials}
    reference = by_id["dataset_reference"]
    known_bad = by_id["known_bad"]
    if not reference.bundle_valid or not known_bad.bundle_valid:
        return "fail", "one or more digest-bound evidence bundles failed verification"
    if reference.assessment_status == "pass" and known_bad.assessment_status == "fail":
        return "pass", "dataset reference passed and declared known-bad candidate failed"
    if reference.assessment_status == "fail" or known_bad.assessment_status == "pass":
        return (
            "fail",
            "frozen protocol did not accept the reference and reject the known-bad candidate",
        )
    return (
        "inconclusive",
        "frozen protocol lacked enough evidence to both accept reference and reject known-bad",
    )


def _require_official_sequence_layout(dataset: Path) -> None:
    if dataset.name != KITTI_SEQUENCE:
        msg = f"KITTI benchmark requires the pinned sequence directory {KITTI_SEQUENCE}"
        raise ValueError(msg)
    for path in (
        dataset / "image_02" / "data",
        dataset / "velodyne_points" / "data",
        _find_calibration(dataset, "calib_velo_to_cam.txt"),
        _find_calibration(dataset, "calib_cam_to_cam.txt"),
    ):
        if not path.exists():
            msg = f"KITTI benchmark input is missing: {path}"
            raise ValueError(msg)


def _find_calibration(dataset: Path, filename: str) -> Path:
    for directory in (dataset, *dataset.parents[:2]):
        candidate = directory / filename
        if candidate.exists():
            return candidate
    return dataset.parent / filename


def _mapping(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        msg = f"config section {key} must be a mapping"
        raise ValueError(msg)
    return value


def _lidar_camera_protocol_id(payload: dict[str, object]) -> str:
    protocols = payload.get("protocols")
    if isinstance(protocols, list):
        for protocol in protocols:
            if isinstance(protocol, dict) and protocol.get("family") == "lidar_camera":
                protocol_id = protocol.get("protocol_id")
                if isinstance(protocol_id, str):
                    return protocol_id
    msg = "protocol artifact did not contain a LiDAR-camera protocol ID"
    raise ValueError(msg)


def _artifact_digest(path: Path, role: str) -> KITTIBenchmarkArtifactDigest:
    digest = _required_digest(path)
    return KITTIBenchmarkArtifactDigest(
        role=role,
        path=str(path),
        sha256=digest,
        size_bytes=path.stat().st_size,
    )


def _required_digest(path: Path) -> str:
    digest = sha256_path(path)
    if digest is None:
        msg = f"could not digest required benchmark artifact: {path}"
        raise ValueError(msg)
    return digest
