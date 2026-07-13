"""Common-frame evaluation for Camera-LiDAR extrinsic baseline candidates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    applied = _applied_transform(result)
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
    result.run.provenance["lidar_camera_baseline_comparison"] = {
        "status": "scored" if scores else "unavailable",
        "candidates": {name: score.as_dict() for name, score in scores.items()},
        "candidate_ids": sorted(scores),
        "capture_time_policy": options.get("capture_time_policy", "not_declared"),
        "comparison_warning": (
            "post-hoc holdout is not training-isolated for candidates that do not declare it"
        ),
        "raw_input_files": _comparison_input_manifests(
            inspection.path,
            int(options.get("max_frames", config.evaluation.kitti.max_projection_pairs)),
        ),
    }
    manifests = result.run.provenance["lidar_camera_baseline_comparison"][
        "raw_input_files"
    ]
    if isinstance(manifests, list) and manifests:
        result.run.provenance.setdefault("raw_input_files", []).extend(manifests)
        result.run.provenance["data_verified"] = True
        result.run.provenance["metrics_origin"] = "recomputed"
    metrics: dict[str, MetricResult] = {
        "lidar_camera_comparison_candidate_count": MetricResult(
            value=float(len(scores)),
            unit="candidates",
            grade="pass" if len(scores) >= 2 else "warn",
        )
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
    return metrics


def _nonempty_split(count: int, ratio: float, seed: int) -> tuple[list[int], list[int]]:
    train, holdout = split_indices(count, ratio, seed)
    if count > 1 and not holdout:
        train, holdout = split_indices(count, 1.0 / count, seed)
    return train, holdout


def _applied_transform(result: CalibrationResult) -> SE3 | None:
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


def _mean(values: Any) -> float | None:
    items = list(values)
    return sum(items) / len(items) if items else None


def _mean_optional(values: Any) -> float | None:
    items = [value for value in values if value is not None]
    return sum(items) / len(items) if items else None
