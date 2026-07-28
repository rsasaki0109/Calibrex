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
from calibrex.pipelines.calibrate import CalibrationRunOptions, run_calibration

KITTI_FALSIFICATION_SCHEMA_VERSION: Literal[
    "slac.kitti_lidar_camera_falsification/v0.1"
] = "slac.kitti_lidar_camera_falsification/v0.1"
KITTI_SEQUENCE: Literal["2011_09_26_drive_0005_sync"] = (
    "2011_09_26_drive_0005_sync"
)
KITTI_SOURCE_URL = "https://www.cvlibs.net/datasets/kitti/raw_data.php"
BenchmarkStatus = Literal["pass", "fail", "inconclusive"]


class KITTIBenchmarkArtifactDigest(StrictModel):
    """Digest-bound file used or produced by one benchmark trial."""

    role: str
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class KITTIBenchmarkTrial(StrictModel):
    """One frozen candidate evaluated by the KITTI protocol."""

    candidate_id: Literal["dataset_reference", "known_bad"]
    expected_assessment: Literal["pass", "fail"]
    assessment_status: Literal["pass", "fail", "inconclusive"]
    bundle_valid: bool
    artifacts: list[KITTIBenchmarkArtifactDigest]


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
) -> KITTIFalsificationBenchmarkArtifact:
    """Run reference and declared known-bad candidates under one frozen protocol."""

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
    selected_frame_ids = [
        f"camera:{pair.camera_index}:lidar:{pair.lidar_index}" for pair in pairs
    ]

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

    known_bad_delta = SE3(
        (0.25, 0.0, 0.0),
        (0.0, 0.0, math.sin(math.radians(5.0) / 2.0), math.cos(math.radians(5.0) / 2.0)),
    )
    candidates: dict[Literal["dataset_reference", "known_bad"], SE3] = {
        "dataset_reference": reference,
        "known_bad": reference.compose(known_bad_delta),
    }
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
        assessment = AssessmentArtifact.model_validate(
            read_mapping(trial_dir / "assessment.json")
        )
        verification = _verify_trial(trial_dir)
        trials.append(
            KITTIBenchmarkTrial(
                candidate_id=candidate_id,
                expected_assessment="pass" if candidate_id == "dataset_reference" else "fail",
                assessment_status=assessment.status,
                bundle_valid=verification.valid,
                artifacts=_trial_artifacts(trial_dir),
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
            git_commit=git_commit(),
            source_config=str(source_config),
            source_config_sha256=source_config_digest,
        ),
    )
    benchmark.save(output / "benchmark.json")
    return benchmark


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


def _trial_artifacts(trial_dir: Path) -> list[KITTIBenchmarkArtifactDigest]:
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
    return [_artifact_digest(trial_dir / name, name.rsplit(".", 1)[0]) for name in names]


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
