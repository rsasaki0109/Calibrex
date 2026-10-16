"""Compare complete artifact-backed probabilistic refinement in fresh workers.

The fixture builder makes a projected execution control from public KITTI
geometry. Its image correspondences are generated, not independent sensor truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.benchmark_bspline_lidar_imu import _freeze, _verify_sources  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> None:
    if path.name in {"protocol.json", "measurement.json", "summary.json", "attempts.json"}:
        import jsonschema

        schema = ROOT / "tools/schemas/probabilistic_refinement_benchmark.schema.json"
        jsonschema.validate(value, json.loads(schema.read_bytes()))
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def comparable(value: dict) -> dict:
    """Only the two declared source/run provenance fields may differ."""
    value = json.loads(json.dumps(value))
    for key in ("created_at", "git_commit"):
        value["provenance"].pop(key)
    return value


def build_fixture(sequence: Path, output: Path, per_frame: int) -> None:
    import numpy as np

    from calibrex.core.camera_lidar_artifacts import (
        CameraLidarArtifactProvenance,
        CameraLidarCalibrationProblem,
        CameraLidarObservationBinding,
    )
    from calibrex.core.geometry import SE3
    from calibrex.core.probabilistic_correspondence import (
        CorrespondenceProviderIdentity,
        ProbabilisticCorrespondenceArtifact,
        ProbabilisticCorrespondenceFrame,
        ProbabilisticCorrespondenceProvenance,
        ProbabilisticImageCorrespondence,
    )
    from calibrex.core.result import TransformResult
    from calibrex.data.depth import DepthCameraIntrinsics, DepthFileReference
    from calibrex.solvers.borer_six_dof_solver import apply_local_se3_delta
    from calibrex.solvers.probabilistic_camera_lidar_refiner import project_camera_point

    if per_frame < 4:
        raise ValueError("fixture requires at least four points per frame")
    scans = sorted((sequence / "velodyne_points/data").glob("*.bin"))
    if len(scans) < 4:
        raise ValueError("provide at least four original KITTI scans")
    output.mkdir(parents=True, exist_ok=False)
    truth = SE3((0.1, -0.05, 0.02), (0.5, -0.5, 0.5, 0.5))
    initial = apply_local_se3_delta(
        truth, np.array([0.4, -0.3, 0.2]), np.array([0.04, -0.03, 0.02])
    )
    rng, frames, bindings, hashes = np.random.default_rng(20261016), [], [], {}
    descriptor = output / "initializer.txt"
    descriptor.write_text(
        "Generated execution control; supplied perturbed pose; no depth provider or "
        "independent camera/LiDAR truth. Image means are projections of original KITTI "
        "geometry plus fixed noise. This fixture supports performance/parity claims only.\n",
        encoding="utf-8",
    )
    for index, path in enumerate(scans):
        camera = DepthCameraIntrinsics(
            width=1280,
            height=720,
            fx=600.0,
            fy=600.0,
            cx=640.0,
            cy=360.0,
            projection=("pinhole", "mei", "double_sphere")[index % 3],
            xi=0.7 if index % 3 else None,
            alpha=0.6 if index % 3 == 2 else None,
            distortion_model="radial-tangential" if index % 3 else "none",
            distortion=[0.01, -0.001, 0.0001, -0.0002] if index % 3 else [],
        )
        frame = ProbabilisticCorrespondenceFrame(
            frame_id=path.stem,
            capture_time_ns=index * 100_000_000,
            camera_frame="camera",
            lidar_frame="lidar",
            intrinsics=camera,
            correspondences=[
                ProbabilisticImageCorrespondence(
                    correspondence_id=str(i),
                    point_lidar_m=[0.0, 0.0, 1.0],
                    image_mean_px=[0.0, 0.0],
                    image_covariance_px2=[1.0, 0.0, 0.0, 1.0],
                    outlier_probability=0.0,
                    reliability=1.0,
                )
                for i in range(4)
            ],
        )
        points = np.fromfile(path, dtype="<f4").reshape(-1, 4)[:, :3].astype(float)
        front = points[np.isfinite(points).all(axis=1) & (points[:, 0] > 1.0)]
        retained = []
        for point in front:
            uv = project_camera_point(truth.transform_point(point), frame)
            if 8.0 <= uv[0] < camera.width - 8 and 8.0 <= uv[1] < camera.height - 8:
                retained.append((point.tolist(), uv))
        chosen = np.linspace(0, len(retained) - 1, min(per_frame, len(retained)), dtype=int)
        correspondences = []
        for row in chosen:
            point, uv = retained[row]
            noise = rng.normal(0.0, 0.35, 2)
            correspondences.append(
                ProbabilisticImageCorrespondence(
                    correspondence_id=str(row),
                    point_lidar_m=point,
                    image_mean_px=(np.asarray(uv) + noise).tolist(),
                    image_covariance_px2=[1.5, 0.125, 0.125, 2.0],
                    outlier_probability=0.05,
                    reliability=0.98,
                )
            )
        frame.correspondences = correspondences
        # Revalidate after replacing the temporary projection carrier.
        frame = ProbabilisticCorrespondenceFrame.model_validate(frame.model_dump())
        frames.append(frame)
        hashes[path.name] = digest(path)
        selected = output / (path.stem + ".npy")
        np.save(selected, np.asarray([c.point_lidar_m for c in correspondences]))
        bindings.append(
            CameraLidarObservationBinding(
                frame_id=frame.frame_id,
                split_id="execution-control",
                depth_observation_frame_id=frame.frame_id,
                lidar=DepthFileReference(
                    path=str(selected.resolve()),
                    sha256=digest(selected),
                    size_bytes=selected.stat().st_size,
                    encoding="npy_float64_xyz",
                ),
                lidar_capture_time_ns=frame.capture_time_ns,
            )
        )
    artifact = ProbabilisticCorrespondenceArtifact(
        artifact_id="kitti-projected-execution-control",
        dataset_id="kitti-projected-control",
        split_id="execution-control",
        dataset_family="kitti_raw",
        sequence_id=sequence.name,
        dataset_license_spdx="LicenseRef-KITTI",
        provider=CorrespondenceProviderIdentity(
            provider="generated-projection-execution-control",
            model="declared-camera-projections",
            version="1",
            source_repository="https://github.com/rsasaki0109/Calibrex",
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            license_spdx="Apache-2.0",
            checkpoint_redistribution="no checkpoint",
        ),
        frames=frames,
        provenance=ProbabilisticCorrespondenceProvenance(
            generator=__name__,
            generator_version="1",
            input_sha256=hashes,
        ),
        warnings=[descriptor.read_text(encoding="utf-8").strip()],
    )
    artifact.save(output / "correspondence.json")

    def transform(pose: SE3) -> TransformResult:
        return TransformResult(parent="camera", child="lidar", **pose.as_dict())

    problem = CameraLidarCalibrationProblem(
        problem_id="kitti-projected-control",
        dataset_id=artifact.dataset_id,
        dataset_family=artifact.dataset_family,
        sequence_id=sequence.name,
        depth_provider_path=str(descriptor.resolve()),
        depth_provider_sha256=digest(descriptor),
        observations=bindings,
        reference_transform_camera_lidar=transform(truth),
        initial_transform_camera_lidar=transform(initial),
        time_convention="generated synchronized execution control; no temporal accuracy claim",
        rotation_bound_deg=2.0,
        translation_bound_m=0.25,
        provenance=CameraLidarArtifactProvenance(
            generator=__name__,
            generator_version="1",
            source_sha256=digest(output / "correspondence.json"),
        ),
    )
    problem.save(output / "problem.json")
    write(
        output / "fixture-protocol.json",
        {
            "scope": __doc__,
            "input_sha256": hashes,
            "builder_sha256": digest(Path(__file__)),
            "seed": 20261016,
            "points_per_frame": per_frame,
            "retained_rows": [len(f.correspondences) for f in frames],
            "correspondence_sha256": digest(output / "correspondence.json"),
            "problem_sha256": digest(output / "problem.json"),
        },
    )


def worker(args: argparse.Namespace) -> None:
    root = Path(args.source_root).resolve()
    protocol = json.loads((Path(args.output_dir).parent / "protocol.json").read_bytes())
    if digest(Path(__file__)) != protocol["launcher_sha256"]:
        raise ValueError("benchmark launcher changed")
    helper = ROOT / "tools/benchmark_bspline_lidar_imu.py"
    if digest(helper) != protocol["source_freezer_sha256"]:
        raise ValueError("source freezer changed")
    for key in ("correspondence", "problem"):
        if digest(getattr(args, key)) != protocol["inputs"][key]:
            raise ValueError("benchmark input changed")
    expected = _verify_sources(root)
    from calibrex.core.validation import validate_file
    from calibrex.evaluation.probabilistic_camera_lidar_run import (
        run_probabilistic_camera_lidar_refinement,
    )
    from calibrex.solvers.probabilistic_camera_lidar_refiner import (
        ProbabilisticCameraLidarRefinementOptions,
    )

    begin = time.perf_counter()
    result = run_probabilistic_camera_lidar_refinement(
        args.correspondence,
        args.problem,
        options=ProbabilisticCameraLidarRefinementOptions(max_evaluations=args.max_evaluations),
        command=["python", "tools/benchmark_probabilistic_refinement.py", "--worker"],
    )
    result.provenance.git_commit = args.source_commit
    result.save(Path(args.output_dir) / "result.json")
    if not validate_file(Path(args.output_dir) / "result.json", "probabilistic-refinement").valid:
        raise ValueError("refinement result failed schema validation")
    api_seconds = time.perf_counter() - begin
    for name, module in list(sys.modules.items()):
        if name == "calibrex" or name.startswith("calibrex."):
            path = Path(module.__file__).resolve()
            if (
                not path.is_relative_to(root / "src")
                or digest(path) != expected[path.relative_to(root / "src").as_posix()]
            ):
                raise ValueError("executed module is outside the frozen source snapshot")
    _verify_sources(root)
    for key in ("correspondence", "problem"):
        if digest(getattr(args, key)) != protocol["inputs"][key]:
            raise ValueError("benchmark input changed during execution")
    if digest(Path(__file__)) != protocol["launcher_sha256"]:
        raise ValueError("benchmark launcher changed during execution")
    write(
        Path(args.output_dir) / "measurement.json",
        {
            "api_seconds": api_seconds,
            "trace_evaluations": len(result.trace),
            "provenance": {
                "source_commit": args.source_commit,
                "source_manifest_sha256": digest(root / "manifest.json"),
                "launcher_sha256": digest(Path(__file__)),
                "result_sha256": digest(Path(args.output_dir) / "result.json"),
                "python": sys.version,
                "blas_threads": {
                    k: os.environ[k]
                    for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
                },
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-kitti-fixture", type=Path)
    parser.add_argument("--points-per-frame", type=int, default=1024)
    parser.add_argument("--correspondence", type=Path)
    parser.add_argument("--problem", type=Path)
    parser.add_argument("--baseline-commit")
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--max-evaluations", type=int, default=400)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    if args.build_kitti_fixture:
        build_fixture(args.build_kitti_fixture, args.output_dir, args.points_per_frame)
        return
    if args.correspondence is None or args.problem is None:
        parser.error("--correspondence and --problem are required")
    if args.worker:
        worker(args)
        return
    if not args.baseline_commit or args.repetitions < 2:
        parser.error("provide --baseline-commit and at least two repetitions")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    commits = {
        "prior": subprocess.check_output(
            ["git", "rev-parse", args.baseline_commit], text=True
        ).strip(),
        "current": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    protocol = {
        "scope": __doc__,
        "commits": commits,
        "repetitions": args.repetitions,
        "max_evaluations": args.max_evaluations,
        "fresh_workers": True,
        "host_unreserved": True,
        "source_manifests": {},
        "inputs": {},
        "launcher_sha256": digest(Path(__file__)),
        "source_freezer_sha256": digest(ROOT / "tools/benchmark_bspline_lidar_imu.py"),
    }
    for variant, commit in commits.items():
        frozen = output / ("sources-" + variant)
        _freeze(frozen, commit)
        protocol["source_manifests"][variant] = digest(frozen / "manifest.json")
    for name in ("correspondence", "problem"):
        path = getattr(args, name).resolve()
        shutil.copyfile(path, output / (name + ".json"))
        protocol["inputs"][name] = digest(path)
    shutil.copyfile(__file__, output / "benchmark.py.txt")
    shutil.copyfile(ROOT / "tools/benchmark_bspline_lidar_imu.py", output / "source-freezer.py.txt")
    write(output / "protocol.json", protocol)
    records = []
    for repeat in range(args.repetitions):
        for variant in ("prior", "current") if repeat % 2 == 0 else ("current", "prior"):
            folder = output / f"{repeat:02d}-{variant}"
            folder.mkdir()
            env = os.environ.copy()
            env.update(
                dict.fromkeys(("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"), "1")
            )
            env["PYTHONPATH"] = str(output / ("sources-" + variant) / "src")
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--source-root",
                str(output / ("sources-" + variant)),
                "--source-commit",
                commits[variant],
                "--correspondence",
                str(output / "correspondence.json"),
                "--problem",
                str(output / "problem.json"),
                "--max-evaluations",
                str(args.max_evaluations),
                "--output-dir",
                str(folder),
            ]
            start = time.perf_counter()
            with (folder / "worker.log").open("wb") as log:
                process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            records.append(
                {
                    "repeat": repeat,
                    "variant": variant,
                    "whole_seconds": time.perf_counter() - start,
                    "exit_code": process.returncode,
                }
            )
            write(output / "attempts.json", records)
            print(json.dumps(records[-1]), flush=True)
            if process.returncode:
                raise RuntimeError("worker failed; all attempts and logs are retained")
    for repeat in range(args.repetitions):
        results = [
            json.loads((output / f"{repeat:02d}-{v}/result.json").read_bytes()) for v in commits
        ]
        if comparable(results[0]) != comparable(results[1]):
            raise ValueError("complete results, evaluations or candidate traces differ")
    medians = {
        v: statistics.median(r["whole_seconds"] for r in records if r["variant"] == v)
        for v in commits
    }
    api_medians = {
        v: statistics.median(
            json.loads((output / f"{i:02d}-{v}/measurement.json").read_bytes())["api_seconds"]
            for i in range(args.repetitions)
        )
        for v in commits
    }
    summary = {
        "whole_median_seconds": medians,
        "whole_speedup": medians["prior"] / medians["current"],
        "api_median_seconds": api_medians,
        "api_speedup": api_medians["prior"] / api_medians["current"],
        "exact_complete_results": True,
        "workers": len(records),
        "provenance": {
            "protocol_sha256": digest(output / "protocol.json"),
            "attempts_sha256": digest(output / "attempts.json"),
        },
    }
    write(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
