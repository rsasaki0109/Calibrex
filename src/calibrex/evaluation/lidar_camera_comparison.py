"""Common-frame evaluation for Camera-LiDAR extrinsic baseline candidates."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from calibrex.core.capture_time import LidarCaptureTimePolicy
from calibrex.core.config import CalibrationConfig
from calibrex.core.geometry import SE3
from calibrex.core.provenance import sha256_path
from calibrex.core.result import CalibrationResult, Grade, MetricResult
from calibrex.data.inspect import DatasetInspection
from calibrex.data.kitti import (
    find_camera_lidar_pairs,
    project_velodyne_to_camera,
    read_velodyne_to_camera_transform,
    score_lidar_camera_depth_edge_alignment,
    score_lidar_camera_edge_alignment,
)
from calibrex.evaluation.holdout import split_indices
from calibrex.evaluation.lidar_camera import camera_lidar_from_rig_candidates

DATASET_REFERENCE_MAX_TRANSLATION_M = 0.10
DATASET_REFERENCE_MAX_ROTATION_DEG = 2.0


@dataclass(frozen=True)
class LidarCameraCandidateScore:
    """One transform evaluated on a shared frame-disjoint KITTI split."""

    candidate_id: str
    train_frame_ids: tuple[str, ...]
    holdout_frame_ids: tuple[str, ...]
    train_projection_ratio: float | None
    holdout_projection_ratio: float | None
    train_edge_alignment: float | None
    holdout_edge_alignment: float | None
    train_depth_edge_alignment: float | None
    holdout_depth_edge_alignment: float | None
    scored_frame_count: int
    transform_camera_lidar: SE3
    training_isolation_declared: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "train_frame_ids": list(self.train_frame_ids),
            "holdout_frame_ids": list(self.holdout_frame_ids),
            "train_projection_ratio": self.train_projection_ratio,
            "holdout_projection_ratio": self.holdout_projection_ratio,
            "train_edge_alignment": self.train_edge_alignment,
            "holdout_edge_alignment": self.holdout_edge_alignment,
            "train_depth_edge_alignment": self.train_depth_edge_alignment,
            "holdout_depth_edge_alignment": self.holdout_depth_edge_alignment,
            "scored_frame_count": self.scored_frame_count,
            "transform_camera_lidar": self.transform_camera_lidar.as_dict(),
            "training_isolation_declared": self.training_isolation_declared,
            "evaluation_policy": "shared seeded frame holdout with fresh projection",
        }


@dataclass(frozen=True)
class LidarCameraCandidateDelta:
    """SE(3) and holdout-evidence delta from one declared reference candidate."""

    reference_candidate_id: str
    candidate_id: str
    translation_delta_m: float
    rotation_delta_deg: float
    holdout_projection_ratio_delta: float | None
    holdout_edge_alignment_delta: float | None
    holdout_depth_edge_alignment_delta: float | None
    both_training_isolated: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "reference_candidate_id": self.reference_candidate_id,
            "candidate_id": self.candidate_id,
            "translation_delta_m": self.translation_delta_m,
            "rotation_delta_deg": self.rotation_delta_deg,
            "holdout_projection_ratio_delta": self.holdout_projection_ratio_delta,
            "holdout_edge_alignment_delta": self.holdout_edge_alignment_delta,
            "holdout_depth_edge_alignment_delta": self.holdout_depth_edge_alignment_delta,
            "both_training_isolated": self.both_training_isolated,
            "delta_convention": "candidate minus reference; higher evidence scores are better",
        }


@dataclass(frozen=True)
class LidarCameraCaptureTimeEvidence:
    """Structured declaration of the time model used by KITTI projection comparison."""

    status: Literal["undeclared", "invalid", "declared_unavailable"]
    reason: str
    policy: LidarCaptureTimePolicy | None
    camera_reference: str
    per_point_times_available: bool
    deskew_applied: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "policy": self.policy.as_dict() if self.policy is not None else None,
            "camera_reference": self.camera_reference,
            "per_point_times_available": self.per_point_times_available,
            "deskew_applied": self.deskew_applied,
            "lidar_message_timestamp_source": "velodyne_points/timestamps.txt",
            "camera_timestamp_source": "image_02/timestamps.txt",
            "point_timestamp_source": None,
            "projection_time_model": "rigid scan at LiDAR message timestamp",
        }


@dataclass(frozen=True)
class _FrameScore:
    frame_id: str
    projection_ratio: float
    edge_alignment: float | None
    depth_edge_alignment: float | None


def evaluate_lidar_camera_candidates_on_kitti(
    dataset_path: str | Path,
    candidates: Mapping[str, tuple[SE3, bool]],
    *,
    max_frames: int,
    max_points: int,
    holdout_ratio: float,
    split_seed: int,
) -> dict[str, LidarCameraCandidateScore]:
    """Evaluate all candidates on exactly the same KITTI frame IDs."""

    pairs = find_camera_lidar_pairs(dataset_path, max_pairs=max_frames)
    frame_ids = tuple(f"camera:{pair.camera_index}:lidar:{pair.lidar_index}" for pair in pairs)
    train_indices, holdout_indices = _nonempty_split(len(pairs), holdout_ratio, split_seed)
    train_ids = tuple(frame_ids[index] for index in train_indices)
    holdout_ids = tuple(frame_ids[index] for index in holdout_indices)
    output: dict[str, LidarCameraCandidateScore] = {}
    for candidate_id, (transform, isolated) in sorted(candidates.items()):
        frames: list[_FrameScore] = []
        for frame_id, pair in zip(frame_ids, pairs, strict=True):
            projection = project_velodyne_to_camera(
                dataset_path,
                camera_path=pair.camera_path,
                lidar_path=pair.lidar_path,
                max_points=max_points,
                t_camera_lidar_override=transform,
            )
            if projection.status != "projected":
                continue
            edge = score_lidar_camera_edge_alignment(projection)
            depth = score_lidar_camera_depth_edge_alignment(projection)
            frames.append(
                _FrameScore(
                    frame_id,
                    projection.projected_point_count / projection.sampled_point_count
                    if projection.sampled_point_count
                    else 0.0,
                    edge.edge_fraction if edge.status == "scored" else None,
                    depth.edge_fraction if depth.status == "scored" else None,
                )
            )
        by_id = {frame.frame_id: frame for frame in frames}
        train = [by_id[item] for item in train_ids if item in by_id]
        holdout = [by_id[item] for item in holdout_ids if item in by_id]
        output[candidate_id] = LidarCameraCandidateScore(
            candidate_id,
            train_ids,
            holdout_ids,
            _mean(frame.projection_ratio for frame in train),
            _mean(frame.projection_ratio for frame in holdout),
            _mean_optional(frame.edge_alignment for frame in train),
            _mean_optional(frame.edge_alignment for frame in holdout),
            _mean_optional(frame.depth_edge_alignment for frame in train),
            _mean_optional(frame.depth_edge_alignment for frame in holdout),
            len(frames),
            transform,
            isolated,
        )
    return output


def compare_lidar_camera_candidate_scores(
    scores: Mapping[str, LidarCameraCandidateScore],
    *,
    reference_candidate_id: str,
) -> dict[str, LidarCameraCandidateDelta]:
    """Compare every non-reference candidate in transform and holdout-evidence space."""

    reference = scores.get(reference_candidate_id)
    if reference is None:
        return {}
    output: dict[str, LidarCameraCandidateDelta] = {}
    for candidate_id, candidate in sorted(scores.items()):
        if candidate_id == reference_candidate_id:
            continue
        relative = reference.transform_camera_lidar.inverse().compose(
            candidate.transform_camera_lidar
        )
        output[candidate_id] = LidarCameraCandidateDelta(
            reference_candidate_id=reference_candidate_id,
            candidate_id=candidate_id,
            translation_delta_m=math.sqrt(sum(value * value for value in relative.translation_m)),
            rotation_delta_deg=_rotation_angle_deg(relative),
            holdout_projection_ratio_delta=_optional_delta(
                candidate.holdout_projection_ratio,
                reference.holdout_projection_ratio,
            ),
            holdout_edge_alignment_delta=_optional_delta(
                candidate.holdout_edge_alignment,
                reference.holdout_edge_alignment,
            ),
            holdout_depth_edge_alignment_delta=_optional_delta(
                candidate.holdout_depth_edge_alignment,
                reference.holdout_depth_edge_alignment,
            ),
            both_training_isolated=(
                reference.training_isolation_declared and candidate.training_isolation_declared
            ),
        )
    return output


def lidar_camera_comparison_metrics_from_result(
    config: CalibrationConfig,
    result: CalibrationResult,
    inspection: DatasetInspection,
) -> dict[str, MetricResult]:
    """Collect dataset/applied/external transforms and materialize common metrics."""

    factor = config.pipeline.factors.get("lidar_camera_baseline_comparison")
    if config.dataset.type != "kitti_raw" or factor is None or not factor.enabled:
        return {}
    candidates: dict[str, tuple[SE3, bool]] = {}
    reference = read_velodyne_to_camera_transform(inspection.path)
    if reference is not None:
        candidates["kitti_dataset_reference"] = (reference, True)
    applied = _applied_transform(result, config=config)
    if applied is not None:
        candidates["calibrex_applied"] = (applied, False)
    external, isolation = _external_transform(result.run.provenance)
    if external is not None:
        candidates["koide_external"] = (external, isolation)
    options = factor.options
    scores = evaluate_lidar_camera_candidates_on_kitti(
        inspection.path,
        candidates,
        max_frames=int(options.get("max_frames", config.evaluation.kitti.max_projection_pairs)),
        max_points=int(options.get("max_points", config.evaluation.kitti.projection_sample_points)),
        holdout_ratio=config.evaluation.holdout_ratio,
        split_seed=config.solver.seed or 0,
    )
    reference_candidate_id = "kitti_dataset_reference"
    deltas = compare_lidar_camera_candidate_scores(
        scores,
        reference_candidate_id=reference_candidate_id,
    )
    capture_time = _capture_time_evidence(options)
    result.run.provenance["lidar_camera_baseline_comparison"] = {
        "status": "scored" if scores else "unavailable",
        "candidates": {name: score.as_dict() for name, score in scores.items()},
        "candidate_ids": sorted(scores),
        "reference_candidate_id": reference_candidate_id,
        "pairwise_deltas": {name: delta.as_dict() for name, delta in deltas.items()},
        "capture_time_evidence": capture_time.as_dict(),
        "comparison_warning": (
            "post-hoc holdout is not training-isolated for candidates that do not declare it"
        ),
        "raw_input_files": _comparison_input_manifests(
            inspection.path,
            int(options.get("max_frames", config.evaluation.kitti.max_projection_pairs)),
        ),
    }
    manifests = result.run.provenance["lidar_camera_baseline_comparison"]["raw_input_files"]
    if isinstance(manifests, list) and manifests:
        result.run.provenance.setdefault("raw_input_files", []).extend(manifests)
        result.run.provenance["data_verified"] = True
        result.run.provenance["metrics_origin"] = "recomputed"
    metrics: dict[str, MetricResult] = {
        "lidar_camera_comparison_candidate_count": MetricResult(
            value=float(len(scores)),
            unit="candidates",
            grade="pass" if len(scores) >= 2 else "warn",
        ),
        "lidar_camera_capture_time_policy_declared": MetricResult(
            value=1.0 if capture_time.policy is not None else 0.0,
            unit="bool",
            grade="pass" if capture_time.policy is not None else "warn",
            reason=capture_time.reason,
        ),
        "lidar_camera_capture_time_deskew_applied": MetricResult(
            value=1.0 if capture_time.deskew_applied else 0.0,
            unit="bool",
            grade="pass" if capture_time.deskew_applied else "warn",
            reason=capture_time.reason,
        ),
    }
    for name, score in scores.items():
        prefix = f"lidar_camera_comparison_{name}"
        grade: Grade = "pass" if score.training_isolation_declared else "warn"
        reason = (
            "shared untouched frame holdout"
            if score.training_isolation_declared
            else "shared post-hoc holdout; training isolation was not declared"
        )
        metrics[f"{prefix}_holdout_edge_alignment"] = MetricResult(
            train=score.train_edge_alignment,
            holdout=score.holdout_edge_alignment,
            value=score.holdout_edge_alignment,
            grade=grade,
            reason=reason,
        )
        metrics[f"{prefix}_holdout_depth_edge_alignment"] = MetricResult(
            train=score.train_depth_edge_alignment,
            holdout=score.holdout_depth_edge_alignment,
            value=score.holdout_depth_edge_alignment,
            grade=grade,
            reason=reason,
        )
        metrics[f"{prefix}_holdout_projection_ratio"] = MetricResult(
            train=score.train_projection_ratio,
            holdout=score.holdout_projection_ratio,
            value=score.holdout_projection_ratio,
            grade=grade,
            reason=reason,
        )
    for name, delta in deltas.items():
        prefix = f"lidar_camera_comparison_{name}_vs_{reference_candidate_id}"
        reason = (
            "diagnostic candidate-minus-reference comparison on the shared frame holdout; "
            "no acceptance threshold is declared"
        )
        metrics[f"{prefix}_translation_delta_m"] = MetricResult(
            value=delta.translation_delta_m,
            unit="m",
            grade="warn",
            reason=reason,
        )
        metrics[f"{prefix}_rotation_delta_deg"] = MetricResult(
            value=delta.rotation_delta_deg,
            unit="deg",
            grade="warn",
            reason=reason,
        )
        metrics[f"{prefix}_holdout_edge_alignment_delta"] = MetricResult(
            value=delta.holdout_edge_alignment_delta,
            unit="fraction",
            grade="warn",
            reason=reason,
        )
    return metrics


def _nonempty_split(count: int, ratio: float, seed: int) -> tuple[list[int], list[int]]:
    train, holdout = split_indices(count, ratio, seed)
    if count > 1 and not holdout:
        train, holdout = split_indices(count, 1.0 / count, seed)
    return train, holdout


def _applied_transform(
    result: CalibrationResult,
    *,
    config: CalibrationConfig | None = None,
) -> SE3 | None:
    if config is not None and config.evaluation.kitti.use_frame_graph_candidate:
        derived = camera_lidar_from_rig_candidates(result.candidate_extrinsics)
        if derived is not None:
            return derived
    camera = result.transforms.get("T_base_link_camera0")
    lidar = result.transforms.get("T_base_link_lidar0")
    if camera is None or lidar is None:
        return None
    return camera.as_se3().inverse().compose(lidar.as_se3())


def _external_transform(provenance: dict[str, Any]) -> tuple[SE3 | None, bool]:
    loaded = provenance.get("koide_lidar_camera_loaded_transforms")
    if not isinstance(loaded, dict):
        return None, False
    raw = loaded.get("T_camera0_lidar0")
    if not isinstance(raw, dict):
        return None, False
    translation = raw.get("translation_m")
    rotation = raw.get("rotation_quat_xyzw")
    if not isinstance(translation, list) or not isinstance(rotation, list):
        return None, False
    try:
        transform = SE3.from_lists(translation, rotation)
    except ValueError:
        return None, False
    return transform, bool(provenance.get("koide_lidar_camera_training_isolation_declared"))


def _comparison_input_manifests(
    dataset_path: str | Path, max_frames: int
) -> list[dict[str, object]]:
    paths: list[tuple[Path, str]] = []
    for pair in find_camera_lidar_pairs(dataset_path, max_pairs=max_frames):
        paths.append((Path(pair.camera_path), "kitti_camera_image"))
        paths.append((Path(pair.lidar_path), "kitti_velodyne_scan"))
    root = Path(dataset_path)
    for directory in (root, *root.parents[:2]):
        for filename in ("calib_velo_to_cam.txt", "calib_cam_to_cam.txt"):
            path = directory / filename
            if path.exists():
                paths.append((path, "kitti_calibration"))
    unique: dict[str, tuple[Path, str]] = {
        str(path): (path, role) for path, role in paths if path.exists()
    }
    return [
        {
            "path": str(path),
            "role": role,
            "sha256": sha256_path(path),
            "size_bytes": path.stat().st_size,
            "source_url": "https://www.cvlibs.net/datasets/kitti/raw_data.php",
        }
        for path, role in unique.values()
    ]


def _capture_time_evidence(options: Mapping[str, Any]) -> LidarCameraCaptureTimeEvidence:
    raw = options.get("capture_time_policy")
    camera_reference = str(options.get("camera_reference", "exposure_timestamp"))
    if raw is None:
        return LidarCameraCaptureTimeEvidence(
            "undeclared",
            "capture-time policy was not declared",
            None,
            camera_reference,
            False,
            False,
        )
    if not isinstance(raw, Mapping):
        return LidarCameraCaptureTimeEvidence(
            "invalid",
            "capture-time policy must be a mapping, not a free-form label",
            None,
            camera_reference,
            False,
            False,
        )
    try:
        policy = LidarCaptureTimePolicy.from_mapping(raw)
    except ValueError as exc:
        return LidarCameraCaptureTimeEvidence(
            "invalid",
            str(exc),
            None,
            camera_reference,
            False,
            False,
        )
    return LidarCameraCaptureTimeEvidence(
        "declared_unavailable",
        (
            "KITTI Velodyne .bin payloads do not carry per-point capture offsets; "
            "the declared policy is recorded but per-point deskew is not applied"
        ),
        policy,
        camera_reference,
        False,
        False,
    )


def _rotation_angle_deg(transform: SE3) -> float:
    scalar = min(1.0, max(-1.0, abs(transform.rotation_quat_xyzw[3])))
    return math.degrees(2.0 * math.acos(scalar))


def _optional_delta(value: float | None, reference: float | None) -> float | None:
    return value - reference if value is not None and reference is not None else None


def _mean(values: Any) -> float | None:
    items = list(values)
    return sum(items) / len(items) if items else None


def _mean_optional(values: Any) -> float | None:
    items = [value for value in values if value is not None]
    return sum(items) / len(items) if items else None
