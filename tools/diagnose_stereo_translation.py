"""Post-hoc correspondence replay; never optimize or read reference extrinsics."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict
from tools.koide_shared_bundle_pandey import read_koide_ply
from tools.run_stereo_translation_refinement import RefinementRun, digest

from calibrex.solvers.stereo_translation_refiner import (
    StereoSurfaceObservation,
    StereoTranslationOptions,
    _correspondences,
    _Surface,
    _surface,
)


class Transition(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    iteration: int
    step_m: float
    full_pairs_before: int
    full_pairs_retained: int
    selected_pairs_before: int
    selected_pairs_retained: int
    selected_still_eligible_but_unsampled: int
    fixed_cost_before: float
    fixed_cost_after: float
    reassociated_cost_after: float


class CaseDiagnostic(BaseModel):
    case_id: str
    trace_sha256: str
    input_sha256: dict[str, str]
    transitions: list[Transition]


class Diagnostic(BaseModel):
    scope: str = "post-hoc replay on spent data; no optimizer, reference, or verdict changes"
    source_commit: str
    source_sha256: dict[str, str]
    plan_sha256: str
    cases: list[CaseDiagnostic]


def identified_correspondences(
    surface: _Surface,
    rotation: np.ndarray,
    translation: np.ndarray,
    options: StereoTranslationOptions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = surface.observation.lidar_points @ rotation.T
    camera = points + translation
    h, w = surface.valid.shape
    fx, fy, cx, cy = surface.observation.intrinsics
    indices = np.flatnonzero(camera[:, 2] > 0.1)
    xyz = camera[indices]
    u = np.rint(fx * xyz[:, 0] / xyz[:, 2] + cx).astype(int)
    v = np.rint(fy * xyz[:, 1] / xyz[:, 2] + cy).astype(int)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    indices, u, v = indices[inside], u[inside], v[inside]
    # Keep the nearest point at each pixel before matching stereo surfaces.
    by_depth = np.argsort(camera[indices, 2], kind="stable")
    _, first = np.unique((v * w + u)[by_depth], return_index=True)
    selected = by_depth[first]
    indices, u, v = indices[selected], u[selected], v[selected]
    valid = surface.valid[v, u]
    indices, u, v = indices[valid], u[valid], v[valid]
    gate = np.maximum(options.depth_gate_floor_m, 3 * surface.sigma_depth[v, u])
    valid = np.abs(camera[indices, 2] - surface.vertices[v, u, 2]) <= gate
    indices, u, v = indices[valid], u[valid], v[valid]
    all_ids = indices * (h * w) + v * w + u
    if len(indices) > options.max_points_per_frame:
        selected = np.linspace(0, len(indices) - 1, options.max_points_per_frame, dtype=int)
        indices, u, v = indices[selected], u[selected], v[selected]
    normals = surface.normals[v, u]
    sigma = surface.sigma_plane[v, u]
    design = normals / sigma[:, None]
    target = np.sum(normals * (surface.vertices[v, u] - points[indices]), axis=1) / sigma
    return design, target, all_ids, indices * (h * w) + v * w + u


def robust_cost(design: np.ndarray, target: np.ndarray, translation: np.ndarray) -> float:
    """SciPy soft-L1 cost with f_scale=1, on an explicitly fixed set of pairs."""
    residual = design @ translation - target
    return float(np.sum(np.sqrt(1 + residual**2) - 1))


def overlap(before: np.ndarray, after: np.ndarray) -> int:
    return len(np.intersect1d(before, after))


def diagnose_case(case_id: str, bundle: Path, trace_path: Path) -> CaseDiagnostic:
    saved = RefinementRun.model_validate_json(trace_path.read_text())
    for filename, expected in saved.input_sha256.items():
        if digest(Path(filename)) != expected:
            raise ValueError(f"changed original input: {filename}")
    solver = Path("src/calibrex/solvers/stereo_translation_refiner.py")
    if digest(solver) != saved.solver_sha256:
        raise ValueError("original solver changed")
    config = json.loads((bundle / "calib-frontend.json").read_text())
    mapping = json.loads((bundle / "frontend.json").read_text())
    options = StereoTranslationOptions(**saved.result.options)
    surfaces = []
    for name in config["meta"]["bag_names"]:
        metadata = Path(mapping[name + ".ply"]["source"]).parent / "frontend.json"
        obs = StereoSurfaceObservation(
            name,
            np.load(bundle / (name + ".depth.npy"), allow_pickle=False),
            read_koide_ply(bundle / (name + ".ply"))[:, :3],
            tuple(config["camera"]["intrinsics"]),
            json.loads(metadata.read_text())["stereo"]["baseline_m"],
        )
        surfaces.append(_surface(obs, options))
    initial = np.array(saved.initial_T_camera_lidar)
    rotation, translation = initial[:3, :3], initial[:3, 3]

    def pairs(t: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        result = []
        for surface in surfaces:
            identified = identified_correspondences(surface, rotation, t, options)
            original = _correspondences(surface, rotation, t, options)
            # Fail closed if this diagnostic copy diverges from the frozen matcher.
            np.testing.assert_array_equal(identified[0], original[0])
            np.testing.assert_array_equal(identified[1], original[1])
            result.append(identified)
        return result

    before = pairs(translation)
    transitions = []
    for entry in saved.result.trace:
        if "translation_m" not in entry:
            continue
        next_translation = np.array(entry["translation_m"])
        after = pairs(next_translation)
        a = np.concatenate([p[0] for p in before])
        b = np.concatenate([p[1] for p in before])
        fixed_after = robust_cost(a, b, next_translation)
        np.testing.assert_allclose(fixed_after, entry["robust_cost"], rtol=1e-10, atol=1e-8)
        if [len(p[0]) for p in before] != entry["counts"]:
            raise ValueError("recorded support mismatch")
        transitions.append(
            Transition(
                iteration=entry["iteration"],
                step_m=entry["step_m"],
                full_pairs_before=sum(len(p[2]) for p in before),
                full_pairs_retained=sum(
                    overlap(p[2], q[2]) for p, q in zip(before, after, strict=True)
                ),
                selected_pairs_before=sum(len(p[3]) for p in before),
                selected_pairs_retained=sum(
                    overlap(p[3], q[3]) for p, q in zip(before, after, strict=True)
                ),
                selected_still_eligible_but_unsampled=sum(
                    overlap(p[3], q[2]) - overlap(p[3], q[3])
                    for p, q in zip(before, after, strict=True)
                ),
                fixed_cost_before=robust_cost(a, b, translation),
                fixed_cost_after=fixed_after,
                reassociated_cost_after=robust_cost(
                    np.concatenate([p[0] for p in after]),
                    np.concatenate([p[1] for p in after]),
                    next_translation,
                ),
            )
        )
        translation, before = next_translation, after
    return CaseDiagnostic(
        case_id=case_id,
        trace_sha256=digest(trace_path),
        input_sha256=saved.input_sha256,
        transitions=transitions,
    )


def run(archive: Path, output: Path) -> Diagnostic:
    plan_path = archive / "plan.json"
    plan = json.loads(plan_path.read_text())
    for filename, expected in plan["source_sha256"].items():
        if digest(Path(filename)) != expected:
            raise ValueError(f"changed frozen source: {filename}")
    cases = []
    for case in plan["cases"]:
        bundle = Path(case["bundle"])
        if digest(bundle / "d2d.json") != case["d2d_sha256"]:
            raise ValueError("changed D2D input")
        cases.append(
            diagnose_case(case["case_id"], bundle, archive / case["case_id"] / "refinement.json")
        )
        print(case["case_id"], "replayed", flush=True)
    sources = [Path(__file__), *(Path(p) for p in plan["source_sha256"])]
    report = Diagnostic(
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256={str(p): digest(p) for p in sources},
        plan_sha256=digest(plan_path),
        cases=cases,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2) + "\n")
    output.with_suffix(".schema.json").write_text(
        json.dumps(Diagnostic.model_json_schema(), indent=2) + "\n"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.archive, args.output)
