"""Adapt a recorded Calibrex trajectory into a continuous-time problem."""

from __future__ import annotations

from pathlib import Path

from calibrex import __version__
from calibrex.core.continuous_time_camera_lidar_artifacts import (
    ContinuousTimeArtifactProvenance,
    ContinuousTimeCameraLidarProblemArtifact,
    ContinuousTimeTrajectoryArtifact,
    ContinuousTimeTrajectoryPoseArtifact,
    load_continuous_time_camera_lidar_problem,
)
from calibrex.core.io import read_mapping
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.result import TransformResult
from calibrex.core.trajectory import TrajectoryArtifact


def attach_recorded_body_trajectory(
    problem_path: str | Path,
    trajectory_path: str | Path,
    *,
    problem_id: str | None = None,
    command: list[str] | None = None,
) -> ContinuousTimeCameraLidarProblemArtifact:
    """Return a new digest-pinned problem using recorded ``T_world_body`` poses."""

    problem_file = Path(problem_path)
    trajectory_file = Path(trajectory_path)
    problem = load_continuous_time_camera_lidar_problem(problem_file)
    trajectory = TrajectoryArtifact.model_validate(read_mapping(trajectory_file))
    problem_digest = _required_digest(problem_file, "problem")
    trajectory_digest = _required_digest(trajectory_file, "trajectory")
    if len(trajectory.poses) < 2:
        raise ValueError("recorded trajectory requires at least two poses")
    body_frames = {item.body_frame for item in problem.captures}
    if body_frames != {trajectory.frame_semantics.child_frame_id}:
        raise ValueError(
            "recorded trajectory child frame does not match the problem body frame"
        )
    adapted_trajectory = ContinuousTimeTrajectoryArtifact(
        world_frame=trajectory.frame_semantics.world_frame_id,
        body_frame=trajectory.frame_semantics.child_frame_id,
        source_sha256=trajectory_digest,
        poses=[
            ContinuousTimeTrajectoryPoseArtifact(
                timestamp_sec=item.timestamp_ns * 1.0e-9,
                transform_world_body=TransformResult(
                    parent=trajectory.frame_semantics.world_frame_id,
                    child=trajectory.frame_semantics.child_frame_id,
                    translation_m=item.translation,
                    rotation_quat_xyzw=item.rotation_quat_xyzw,
                ),
            )
            for item in trajectory.poses
        ],
    )
    payload = problem.model_dump(mode="python")
    payload["problem_id"] = (
        problem_id or f"{problem.problem_id}-piecewise-trajectory"
    )
    payload["body_trajectory"] = adapted_trajectory.model_dump(mode="python")
    payload["provenance"] = ContinuousTimeArtifactProvenance(
        generator=__name__,
        generator_version=__version__,
        git_commit=git_commit(),
        command=command or [],
        source_sha256=problem_digest,
    ).model_dump(mode="python")
    return ContinuousTimeCameraLidarProblemArtifact.model_validate(payload)


def _required_digest(path: Path, label: str) -> str:
    digest = sha256_path(path)
    if digest is None:
        raise ValueError(f"{label} artifact is not readable: {path}")
    return digest
