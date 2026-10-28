"""Reference-assisted controls on spent data; never a deployable calibration score."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict
from tools.diagnose_stereo_translation import identified_correspondences
from tools.koide_shared_bundle_pandey import read_koide_ply
from tools.run_stereo_translation_refinement import RefinementRun, digest

from calibrex.solvers.stereo_translation_refiner import (
    StereoSurfaceObservation,
    StereoTranslationOptions,
    StereoTranslationResult,
    _surface,
    refine_stereo_translation,
)


class SurfaceCheck(BaseModel):
    frame_id: str
    selected_count: int
    signed_z_median_m: float
    absolute_z_p90_m: float
    absolute_plane_median_m: float


class Control(BaseModel):
    case_id: str
    mode: str
    initial_T_camera_lidar: list[list[float]]
    reference_T_camera_lidar: list[list[float]]
    candidate_translation_difference_m: float
    candidate_displacement_m: float
    input_sha256: dict[str, str]
    result: StereoTranslationResult


class CaseChecks(BaseModel):
    case_id: str
    frames: list[SurfaceCheck]


class Report(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    scope: str = "post-hoc reference-assisted controls; not a calibration evaluation"
    source_commit: str
    source_sha256: dict[str, str]
    archive_plan_sha256: str
    controls: list[Control]
    surface_checks_at_reference: list[CaseChecks]


def surface_check(
    observation: StereoSurfaceObservation, reference: np.ndarray, options: StereoTranslationOptions
) -> SurfaceCheck:
    """Summarize gated, selected pairs at reference; not an unbiased depth-error estimate."""
    surface = _surface(observation, options)
    _, _, _, ids = identified_correspondences(surface, reference[:3, :3], reference[:3, 3], options)
    h, w = surface.valid.shape
    point_ids, pixels = ids // (h * w), ids % (h * w)
    v, u = pixels // w, pixels % w
    camera = observation.lidar_points[point_ids] @ reference[:3, :3].T + reference[:3, 3]
    delta = camera - surface.vertices[v, u]
    plane = np.sum(surface.normals[v, u] * delta, axis=1)
    return SurfaceCheck(
        frame_id=observation.frame_id,
        selected_count=len(ids),
        signed_z_median_m=float(np.median(delta[:, 2])),
        absolute_z_p90_m=float(np.percentile(np.abs(delta[:, 2]), 90)),
        absolute_plane_median_m=float(np.median(np.abs(plane))),
    )


def run(archive: Path, output: Path) -> Report:
    plan_path = archive / "plan.json"
    plan = json.loads(plan_path.read_text())
    for filename, expected in plan["source_sha256"].items():
        if digest(Path(filename)) != expected:
            raise ValueError(f"changed frozen source: {filename}")
    report = Report(
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256={**plan["source_sha256"], str(Path(__file__)): digest(Path(__file__))},
        archive_plan_sha256=digest(plan_path),
        controls=[],
        surface_checks_at_reference=[],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("do not overwrite a prior diagnostic")
    for case in plan["cases"]:
        bundle = Path(case["bundle"])
        trace_path = archive / case["case_id"] / "refinement.json"
        saved = RefinementRun.model_validate_json(trace_path.read_text())
        for filename, expected in saved.input_sha256.items():
            if digest(Path(filename)) != expected:
                raise ValueError(f"changed input: {filename}")
        config = json.loads((bundle / "calib-frontend.json").read_text())
        mapping = json.loads((bundle / "frontend.json").read_text())
        options = StereoTranslationOptions(**saved.result.options)
        observations = []
        for name in config["meta"]["bag_names"]:
            metadata = Path(mapping[name + ".ply"]["source"]).parent / "frontend.json"
            observations.append(
                StereoSurfaceObservation(
                    name,
                    np.load(bundle / (name + ".depth.npy"), allow_pickle=False),
                    read_koide_ply(bundle / (name + ".ply"))[:, :3],
                    tuple(config["camera"]["intrinsics"]),
                    json.loads(metadata.read_text())["stereo"]["baseline_m"],
                )
            )
        seed_path = bundle / "seed.json"
        reference = np.array(json.loads(seed_path.read_text())["reference_T_camera_lidar"])
        hashes = {
            **saved.input_sha256,
            str(seed_path): digest(seed_path),
            str(trace_path): digest(trace_path),
        }
        report.surface_checks_at_reference.append(
            CaseChecks(
                case_id=case["case_id"],
                frames=[surface_check(o, reference, options) for o in observations],
            )
        )
        for mode in ["reference_rotation", "reference_pose"]:
            initial = (
                np.array(saved.initial_T_camera_lidar)
                if mode == "reference_rotation"
                else reference.copy()
            )
            initial[:3, :3] = reference[:3, :3]
            fit = refine_stereo_translation(observations, initial, options)
            candidate = np.array(fit.candidate_T_camera_lidar)
            np.testing.assert_array_equal(candidate[:3, :3], reference[:3, :3])
            if not fit.accepted:
                np.testing.assert_array_equal(fit.output_T_camera_lidar, initial)
            report.controls.append(
                Control(
                    case_id=case["case_id"],
                    mode=mode,
                    initial_T_camera_lidar=initial.tolist(),
                    reference_T_camera_lidar=reference.tolist(),
                    candidate_translation_difference_m=float(
                        np.linalg.norm(candidate[:3, 3] - reference[:3, 3])
                    ),
                    candidate_displacement_m=float(
                        np.linalg.norm(candidate[:3, 3] - initial[:3, 3])
                    ),
                    input_sha256=hashes,
                    result=fit,
                )
            )
            output.write_text(report.model_dump_json(indent=2) + "\n")
            print(
                case["case_id"],
                mode,
                fit.status,
                report.controls[-1].candidate_translation_difference_m,
                flush=True,
            )
    output.with_suffix(".schema.json").write_text(
        json.dumps(Report.model_json_schema(), indent=2) + "\n"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.archive, args.output)
