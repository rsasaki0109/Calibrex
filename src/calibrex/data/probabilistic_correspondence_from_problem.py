"""Build probabilistic correspondences by projecting LiDAR into the camera."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

from calibrex.core.probabilistic_correspondence import (
    CorrespondenceProviderIdentity,
    CorrespondenceTrainingDeclaration,
    ProbabilisticCorrespondenceArtifact,
    ProbabilisticCorrespondenceFrame,
    ProbabilisticCorrespondenceProvenance,
    ProbabilisticImageCorrespondence,
)
from calibrex.core.provenance import git_commit
from calibrex.data.depth import DepthCameraIntrinsics, DepthProviderIdentity
from calibrex.evaluation.borer_rotation_benchmark import load_borer_problem
from calibrex.solvers.borer_depth_to_depth_solver import (
    project_lidar_image_correspondences,
)
from calibrex.solvers.borer_six_dof_solver import apply_local_se3_delta

PROBABILISTIC_CORRESPONDENCE_FROM_PROBLEM_VERSION = (
    "calibrex.probabilistic_correspondence_from_problem/v0.1"
)
_MIN_CORRESPONDENCES_PER_FRAME = 4
ProjectionTransformSource = Literal[
    "initial",
    "reference",
    "reference_perturbed",
]
FEATDEPTH_CORRESPONDENCE_PERTURBATION_TRANSLATION_M = (0.015, -0.008, 0.005)
FEATDEPTH_CORRESPONDENCE_PERTURBATION_ROTATION_DEG = (0.0, 0.0, 0.25)


def build_probabilistic_correspondence_from_problem(
    problem_path: str | Path,
    *,
    transform_source: Literal["initial", "reference"] = "initial",
    projection_transform_source: ProjectionTransformSource | None = None,
    use_depth_provider_identity: bool = False,
    max_points_per_frame: int = 200,
    pixel_noise_std: float = 1.5,
    depth_relative_gate: float = 0.25,
    seed: int = 0,
    artifact_id: str | None = None,
) -> ProbabilisticCorrespondenceArtifact:
    """Project digest-verified LiDAR points into the camera for refinement.

    When ``projection_transform_source`` is ``reference_perturbed``, correspondences
    are built from a fixed offset relative to the vendor reference so image
    observations are not self-consistent with the solver initial transform.
    Set ``use_depth_provider_identity`` to propagate FeatDepth (or other depth
    provider) lineage into the correspondence artifact.
    """

    loaded = load_borer_problem(problem_path)
    problem = loaded.problem
    if transform_source == "reference":
        legacy_transform = loaded.reference_transform_camera_lidar
    else:
        legacy_transform = problem.initial_transform_camera_lidar.as_se3()
    projection_source = projection_transform_source or transform_source
    if projection_source in {"initial", "reference"}:
        if projection_source != transform_source:
            transform = (
                loaded.reference_transform_camera_lidar
                if projection_source == "reference"
                else problem.initial_transform_camera_lidar.as_se3()
            )
        else:
            transform = legacy_transform
    elif projection_source == "reference_perturbed":
        transform = apply_local_se3_delta(
            loaded.reference_transform_camera_lidar,
            FEATDEPTH_CORRESPONDENCE_PERTURBATION_ROTATION_DEG,
            FEATDEPTH_CORRESPONDENCE_PERTURBATION_TRANSLATION_M,
        )
    else:
        raise ValueError(f"unsupported projection transform source: {projection_source}")
    reference = problem.reference_transform_camera_lidar
    camera_frame = reference.parent
    lidar_frame = reference.child
    variance = pixel_noise_std * pixel_noise_std
    covariance = [variance, 0.0, 0.0, variance]
    frames: list[ProbabilisticCorrespondenceFrame] = []
    for frame_index, binding in enumerate(problem.observations):
        frame_rng = random.Random(seed + frame_index)
        observation = next(
            item for item in loaded.observations if item.frame_id == binding.frame_id
        )
        depth_observation = next(
            item
            for item in loaded.depth_provider.observations
            if item.frame_id == binding.depth_observation_frame_id
        )
        projections = project_lidar_image_correspondences(
            observation,
            transform,
            max_points=max_points_per_frame,
            depth_relative_gate=depth_relative_gate,
            seed=seed + int(binding.frame_id),
        )
        if len(projections) < _MIN_CORRESPONDENCES_PER_FRAME:
            msg = (
                f"frame {binding.frame_id} produced {len(projections)} correspondences; "
                f"need at least {_MIN_CORRESPONDENCES_PER_FRAME}"
            )
            raise ValueError(msg)
        intrinsics = DepthCameraIntrinsics(
            width=depth_observation.intrinsics.width,
            height=depth_observation.intrinsics.height,
            fx=depth_observation.intrinsics.fx,
            fy=depth_observation.intrinsics.fy,
            cx=depth_observation.intrinsics.cx,
            cy=depth_observation.intrinsics.cy,
        )
        correspondences = [
            ProbabilisticImageCorrespondence(
                correspondence_id=f"{binding.frame_id}-{index:04d}",
                point_lidar_m=list(projection.point_lidar_m),
                image_mean_px=[
                    projection.image_u_px
                    + frame_rng.gauss(0.0, pixel_noise_std),
                    projection.image_v_px
                    + frame_rng.gauss(0.0, pixel_noise_std),
                ],
                image_covariance_px2=covariance,
                outlier_probability=0.02,
                reliability=0.98,
            )
            for index, projection in enumerate(projections)
        ]
        frames.append(
            ProbabilisticCorrespondenceFrame(
                frame_id=binding.frame_id,
                capture_time_ns=binding.lidar_capture_time_ns,
                camera_frame=camera_frame,
                lidar_frame=lidar_frame,
                intrinsics=intrinsics,
                correspondences=correspondences,
            )
        )
    provider = loaded.depth_provider.provider
    problem_digest = loaded.problem_sha256
    provider_digest = loaded.depth_provider_sha256
    correspondence_provider = (
        _correspondence_provider_identity(provider)
        if use_depth_provider_identity
        else CorrespondenceProviderIdentity(
            provider="calibrex",
            model="depth-lidar-projection",
            version=PROBABILISTIC_CORRESPONDENCE_FROM_PROBLEM_VERSION,
            source_repository=provider.source_repository,
            source_commit=provider.source_commit,
            license_spdx=provider.license_spdx,
            checkpoint_redistribution=provider.checkpoint_redistribution,
        )
    )
    return ProbabilisticCorrespondenceArtifact(
        artifact_id=artifact_id or f"{problem.problem_id}-correspondence",
        dataset_id=problem.dataset_id,
        split_id=problem.observations[0].split_id,
        dataset_license_spdx="CC-BY-NC-SA-3.0",
        provider=correspondence_provider,
        frames=frames,
        provenance=ProbabilisticCorrespondenceProvenance(
            generator=__name__,
            generator_version=PROBABILISTIC_CORRESPONDENCE_FROM_PROBLEM_VERSION,
            git_commit=git_commit(),
            input_sha256={
                str(Path(problem_path).resolve()): problem_digest,
                problem.depth_provider_path: provider_digest,
            },
        ),
    )


def build_featdepth_correspondence_from_problem(
    problem_path: str | Path,
    *,
    max_points_per_frame: int = 200,
    pixel_noise_std: float = 1.5,
    depth_relative_gate: float = 0.35,
    seed: int = 0,
    artifact_id: str | None = None,
) -> ProbabilisticCorrespondenceArtifact:
    """Build FeatDepth-gated correspondences with external provider lineage.

    LiDAR points are projected with the problem initial transform and retained
    only when FeatDepth depth agrees within ``depth_relative_gate``. Pixel
    mean jitter and resample bootstrap (in empirical uncertainty) break exact
    self-consistency without biasing the refit away from the vendor reference.
    """

    return build_probabilistic_correspondence_from_problem(
        problem_path,
        transform_source="initial",
        use_depth_provider_identity=True,
        max_points_per_frame=max_points_per_frame,
        pixel_noise_std=pixel_noise_std,
        depth_relative_gate=depth_relative_gate,
        seed=seed,
        artifact_id=artifact_id,
    )


def save_probabilistic_correspondence_from_problem(
    problem_path: str | Path,
    output_path: str | Path,
    *,
    transform_source: Literal["initial", "reference"] = "initial",
    max_points_per_frame: int = 200,
    pixel_noise_std: float = 1.5,
    depth_relative_gate: float = 0.25,
    seed: int = 0,
    artifact_id: str | None = None,
) -> ProbabilisticCorrespondenceArtifact:
    """Build and persist a correspondence artifact from a problem file."""

    artifact = build_probabilistic_correspondence_from_problem(
        problem_path,
        transform_source=transform_source,
        max_points_per_frame=max_points_per_frame,
        pixel_noise_std=pixel_noise_std,
        depth_relative_gate=depth_relative_gate,
        seed=seed,
        artifact_id=artifact_id,
    )
    artifact.save(output_path)
    return artifact


def _correspondence_provider_identity(
    provider: DepthProviderIdentity,
) -> CorrespondenceProviderIdentity:
    training = None
    if provider.training is not None:
        training = CorrespondenceTrainingDeclaration(
            training_datasets=[provider.training.training_dataset],
            evaluation_overlap=provider.training.evaluation_overlap,
            test_data_used_for_online_refinement=(
                provider.training.test_data_used_for_online_refinement
            ),
            evidence=provider.training.evidence,
        )
    return CorrespondenceProviderIdentity(
        provider=provider.provider,
        model=provider.model,
        version=provider.version,
        source_repository=provider.source_repository,
        source_commit=provider.source_commit,
        license_spdx=provider.license_spdx,
        checkpoint=provider.checkpoint,
        checkpoint_redistribution=provider.checkpoint_redistribution,
        training=training,
    )
