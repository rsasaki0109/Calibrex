"""Complete CPU candidate checks, with empty-cache and reused-input timings."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import inspect
import json
import shutil
import subprocess
import sys
import time
import weakref
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

import numpy as np
from pydantic import Field
from scipy.spatial.transform import Rotation

from calibrex.core.camera_lidar_edge import CameraLidarEdgeArtifact, CameraLidarEdgeProvenance
from calibrex.core.result import StrictModel
from calibrex.data.ros2_camera_lidar import CameraLidarFrame
from calibrex.evaluation import camera_lidar_edge as edge
from calibrex.evaluation.edge_preparation_cache import (
    EdgePreparationCache,
    EdgePreparationCacheDiagnostics,
)

_Digest: TypeAlias = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
_Commit: TypeAlias = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
_Decision: TypeAlias = tuple[str, bool, list[list[float]], float | tuple[int, float]]


class _Call(StrictModel):
    seconds: float = Field(gt=0)
    artifact: CameraLidarEdgeArtifact


class _Sequence(StrictModel):
    calls: list[_Call] = Field(min_length=3, max_length=3)
    init_seconds: float = Field(ge=0)
    clear_seconds: float = Field(ge=0)
    seconds: float = Field(gt=0)
    diagnostics: EdgePreparationCacheDiagnostics | None


class _Attempt(StrictModel):
    repetition: int = Field(ge=0)
    order: list[Literal["baseline", "reuse"]] = Field(min_length=2, max_length=2)
    baseline: _Sequence
    reuse: _Sequence
    complete_artifacts_equal: Literal[True]
    cold_ratio: float = Field(gt=0)
    warm_ratios: list[float] = Field(min_length=2, max_length=2)
    three_request_ratio: float = Field(gt=0)


class _Replay(StrictModel):
    artifacts: list[CameraLidarEdgeArtifact] = Field(min_length=3, max_length=3)
    decisions: list[list[_Decision]] = Field(min_length=3, max_length=3)
    decision_sha256: list[_Digest] = Field(min_length=3, max_length=3)
    pose_counts: list[int] = Field(min_length=3, max_length=3)


class PreparationReuseRun(StrictModel):
    status: Literal["PASS"]
    fixture: str = ""
    fixture_index: int = Field(default=0, ge=0)
    input_sha256: _Digest
    input_provenance: dict[str, Any]
    input_load_seconds: float = Field(ge=0)
    frames: int = Field(ge=1)
    raw_points: int = Field(ge=0)
    estimate_translation: bool
    candidate_offsets_deg: list[tuple[float, float, float]]
    attempts: list[_Attempt] = Field(min_length=3)
    baseline_replay: _Replay
    reuse_replay: _Replay
    every_score_and_decision_equal: Literal[True]
    paired_median_cold_ratio: float = Field(gt=0)
    paired_median_warm_ratio: float = Field(gt=0)
    paired_median_three_request_ratio: float = Field(gt=0)
    baseline_source_head: _Commit
    source_head: _Commit
    scope: str


class _Provenance(StrictModel):
    created_at: str
    command: list[str]
    source_head: _Commit
    source_worktree_dirty: bool
    baseline_source_head: _Commit
    baseline_source_sha256: _Digest
    sources: dict[str, _Digest]
    whole_calibration_claim: Literal[False]
    pyodide_version: str | None = None
    browser_version: str | None = None


class PreparationReuseBenchmark(StrictModel):
    schema_version: Literal["calibrex.browser_edge_preparation_probe/v0.1"]
    status: Literal["RUNNING", "PASS", "FAIL", "TIMEOUT"]
    provenance: _Provenance
    runs: list[PreparationReuseRun]
    logs: list[str]
    errors: list[str]
    checks: dict[str, Any]


class PreparationCachedBaselineRun(PreparationReuseRun):
    baseline_cache_enabled: Literal[True] = True


class _CachedBaselineProvenance(_Provenance):
    baseline_cache_enabled: Literal[True] = True


class PreparationCachedBaselineBenchmark(StrictModel):
    """Compare preparation changes with equal cache capacity in both implementations."""

    schema_version: Literal["calibrex.browser_edge_preparation_probe/v0.2"]
    status: Literal["RUNNING", "PASS", "FAIL", "TIMEOUT"]
    provenance: _CachedBaselineProvenance
    runs: list[PreparationCachedBaselineRun]
    logs: list[str]
    errors: list[str]
    checks: dict[str, Any]


class _Trace:
    def __init__(self) -> None:
        self.keys: weakref.WeakKeyDictionary[Any, str] = weakref.WeakKeyDictionary()
        self.records: list[Any] = []
        self.poses = 0

    def __call__(self, request: Any, reply: Any) -> None:
        if request.objective not in self.keys:
            objective = request.objective
            digest = hashlib.sha256(json.dumps(objective.camera.as_dict(), sort_keys=True).encode())
            for name in (
                "_points",
                "_weights",
                "_flat",
                "_offset",
                "_stride",
                "_shift",
                "_inverse_scale",
                "_u_limit",
                "_v_limit",
                "_start_translation",
            ):
                array = getattr(objective, name)
                digest.update(str(array.dtype).encode())
                digest.update(str(array.shape).encode())
                digest.update(array.tobytes())
            self.keys[objective] = digest.hexdigest()
        self.records.append(
            [
                self.keys[request.objective],
                request.select_maximum,
                request.theta.tolist(),
                list(reply) if isinstance(reply, tuple) else reply,
            ]
        )
        self.poses += len(request.theta)


def _payload(artifact: CameraLidarEdgeArtifact, expected_head: str) -> dict[str, Any]:
    value = artifact.model_dump(mode="json")
    assert value["provenance"].pop("git_commit") == expected_head
    return value


async def measure(
    path: str,
    runtime: Any = None,
    repetitions: int = 3,
    *,
    source_head: str,
    baseline_source_head: str,
    baseline_path: str = "/baseline_edge.py",
    baseline_cache_enabled: bool = False,
) -> dict[str, Any]:
    """Time ordinary complete APIs; replay all objective arrays/decisions separately."""
    del runtime  # This CPU probe never requests a GPU or a second worker.
    if repetitions < 3:
        raise ValueError("at least three alternating paired repeats are required")
    if type(baseline_cache_enabled) is not bool:
        raise ValueError("baseline_cache_enabled must be a boolean")
    for head in (source_head, baseline_source_head):
        if len(head) != 40 or any(c not in "0123456789abcdef" for c in head):
            raise ValueError("full source commits are required")
    spec = importlib.util.spec_from_file_location("_edge_preparation_baseline", baseline_path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot import the frozen baseline")
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    if (
        baseline_cache_enabled
        and "preparation_cache" not in inspect.signature(baseline.estimate_from_frames).parameters
    ):
        raise ValueError("the selected baseline does not support a preparation cache")
    began = time.perf_counter()
    input_bytes = Path(path).read_bytes()
    with np.load(path, allow_pickle=False) as data:
        lineage = json.loads(str(data["provenance_json"]))
        intrinsics = data["intrinsics"].tolist()
        initial = data["candidate"].copy()
        translation = bool(data["estimate_translation"])
        frames = [
            CameraLidarFrame(
                time_s=float(i),
                image_time_s=float(i),
                image=data[f"image_{i}"].copy(),
                points=data[f"points_{i}"].copy(),
                cloud_frame_id="lidar",
                image_frame_id="camera",
            )
            for i in range(int(data["frames"]))
        ]
    load_seconds = time.perf_counter() - began
    digest = hashlib.sha256(input_bytes).hexdigest()
    del input_bytes
    offsets = [(0.0, 0.0, 0.0), (0.5, -0.3, 0.2), (-0.4, 0.2, -0.3)]
    candidates = []
    for offset in offsets:
        matrix = initial.copy()
        matrix[:3, :3] = (
            Rotation.from_euler("xyz", offset, degrees=True).as_matrix() @ initial[:3, :3]
        )
        candidates.append(matrix)
    created = datetime.now(timezone.utc).isoformat()

    def settings(module: Any, head: str) -> tuple[Any, Any, CameraLidarEdgeProvenance]:
        return (
            module.EdgeCamera(*intrinsics),
            module.EdgeAlignmentOptions(estimate_translation=translation),
            CameraLidarEdgeProvenance(
                generator="tools.browser_edge_preparation_probe",
                generator_version="0.1",
                git_commit=head,
                command=["measure", path, "three-declared-candidates"],
                dataset_family="synthetic",
                sequence_ids=["distinct-ray-cast-scenes"],
                image_topic="/image",
                lidar_topic="/points",
                input_sha256=digest,
                input_digest_scope="complete NPZ bytes; candidate offsets declared in probe",
                dataset_license="Apache-2.0 synthetic fixture generator",
                created_at=created,
            ),
        )

    def sequence(reuse: bool) -> _Sequence:
        module, head = (edge, source_head) if reuse else (baseline, baseline_source_head)
        camera, options, provenance = settings(module, head)
        began = time.perf_counter()
        cache = EdgePreparationCache() if reuse or baseline_cache_enabled else None
        init_seconds = time.perf_counter() - began
        calls = []
        for candidate in candidates:
            began = time.perf_counter()
            artifact = module.estimate_from_frames(
                frames,
                camera,
                candidate,
                options=options,
                provenance=provenance,
                **({"preparation_cache": cache} if cache is not None else {}),
            )
            seconds = time.perf_counter() - began
            calls.append(_Call(seconds=seconds, artifact=artifact))
        diagnostics = cache.diagnostics if cache is not None else None
        began = time.perf_counter()
        if cache is not None:
            cache.clear()
            assert cache.diagnostics.retained_bytes == cache.diagnostics.entries == 0
        clear_seconds = time.perf_counter() - began
        return _Sequence(
            calls=calls,
            init_seconds=init_seconds,
            clear_seconds=clear_seconds,
            seconds=init_seconds + clear_seconds + sum(call.seconds for call in calls),
            diagnostics=diagnostics,
        )

    attempts = []
    reference = None
    for repetition in range(repetitions):
        order = ["baseline", "reuse"] if repetition % 2 == 0 else ["reuse", "baseline"]
        values = {name: sequence(name == "reuse") for name in order}
        original, current = values["baseline"], values["reuse"]
        for index, (a, b) in enumerate(zip(original.calls, current.calls, strict=True)):
            assert _payload(a.artifact, baseline_source_head) == _payload(b.artifact, source_head)
            if reference is not None:
                assert _payload(a.artifact, baseline_source_head) == reference[index]
        reference = [_payload(a.artifact, baseline_source_head) for a in original.calls]
        attempts.append(
            _Attempt(
                repetition=repetition,
                order=order,
                baseline=original,
                reuse=current,
                complete_artifacts_equal=True,
                cold_ratio=(original.calls[0].seconds + original.init_seconds)
                / (current.calls[0].seconds + current.init_seconds),
                warm_ratios=[
                    a.seconds / b.seconds
                    for a, b in zip(original.calls[1:], current.calls[1:], strict=True)
                ],
                three_request_ratio=original.seconds / current.seconds,
            )
        )

    def replay(reuse: bool) -> _Replay:
        module, head = (edge, source_head) if reuse else (baseline, baseline_source_head)
        camera, options, provenance = settings(module, head)
        cache = EdgePreparationCache() if reuse or baseline_cache_enabled else None
        artifacts, decisions, counts, digests = [], [], [], []
        try:
            for candidate in candidates:
                trace = _Trace()
                artifact = module._drive_edge_workflow(
                    module._estimate_workflow(
                        frames,
                        camera,
                        candidate,
                        options=options,
                        provenance=provenance,
                        **({"preparation_cache": cache} if cache is not None else {}),
                    ),
                    observer=trace,
                )
                artifacts.append(artifact)
                decisions.append(trace.records)
                counts.append(trace.poses)
                digests.append(
                    hashlib.sha256(
                        json.dumps(trace.records, separators=(",", ":")).encode()
                    ).hexdigest()
                )
        finally:
            if cache is not None:
                cache.clear()
        return _Replay(
            artifacts=artifacts, decisions=decisions, decision_sha256=digests, pose_counts=counts
        )

    original_replay, current_replay = replay(False), replay(True)
    assert original_replay.decisions == current_replay.decisions
    assert original_replay.decision_sha256 == current_replay.decision_sha256
    assert original_replay.pose_counts == current_replay.pose_counts
    for index, (a, b) in enumerate(
        zip(original_replay.artifacts, current_replay.artifacts, strict=True)
    ):
        assert _payload(a, baseline_source_head) == _payload(b, source_head) == reference[index]
    run_model = PreparationCachedBaselineRun if baseline_cache_enabled else PreparationReuseRun
    result = run_model(
        status="PASS",
        input_sha256=digest,
        input_provenance=lineage,
        input_load_seconds=load_seconds,
        frames=len(frames),
        raw_points=sum(len(frame.points) for frame in frames),
        estimate_translation=translation,
        candidate_offsets_deg=offsets,
        attempts=attempts,
        baseline_replay=original_replay,
        reuse_replay=current_replay,
        every_score_and_decision_equal=True,
        paired_median_cold_ratio=float(np.median([a.cold_ratio for a in attempts])),
        paired_median_warm_ratio=float(np.median([v for a in attempts for v in a.warm_ratios])),
        paired_median_three_request_ratio=float(
            np.median([a.three_request_ratio for a in attempts])
        ),
        baseline_source_head=baseline_source_head,
        source_head=source_head,
        scope=(
            "Three distinct candidates of the same raw in-memory frames. Ordinary complete CPU "
            "APIs include preparation, every search/jackknife/control and artifact validation. "
            "Cache construction and final clear included in the three-request sum; inter-call "
            "reporting excluded. Empty-cache first request and two reused requests reported "
            "separately. NPZ loading, imports, runtime startup and bag decoding excluded. "
            "Separate decision replays are untimed. No GPU or cross-origin isolation required; "
            "no real-data accuracy claim."
            + (
                " Both implementations use fresh caches of equal 128 MiB capacity."
                if baseline_cache_enabled
                else " The frozen baseline has no runtime preparation cache."
            )
        ),
    )
    return result.model_dump(mode="json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fixture", type=Path, action="append", default=[])
    parser.add_argument("--baseline-ref", default="d050acca2ded92938ceacf0149a053917e21f54c")
    parser.add_argument(
        "--baseline-cache",
        action="store_true",
        help="give the frozen baseline the same 128 MiB cache; emits the v0.2 report",
    )
    args = parser.parse_args()
    report_model = (
        PreparationCachedBaselineBenchmark if args.baseline_cache else PreparationReuseBenchmark
    )
    if args.schema:
        args.schema.parent.mkdir(parents=True, exist_ok=True)
        args.schema.write_text(
            json.dumps(report_model.model_json_schema(), indent=2) + "\n",
            encoding="utf-8",
        )
        return
    if not args.out or not args.fixture:
        parser.error("--out NEW_DIR and --fixture NPZ are required")
    root = Path(__file__).resolve().parents[1]
    args.out.mkdir(exist_ok=False)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    baseline_head = subprocess.check_output(
        ["git", "rev-parse", args.baseline_ref], cwd=root, text=True
    ).strip()
    captured = subprocess.check_output(
        ["git", "show", f"{baseline_head}:src/calibrex/evaluation/camera_lidar_edge.py"], cwd=root
    )
    baseline_path = args.out / "baseline_edge.py"
    baseline_path.write_bytes(captured)
    sources = [Path(__file__), *sorted((root / "src/calibrex").rglob("*.py")), *args.fixture]
    hashes = {}
    for source in sources:
        name = source.resolve().relative_to(root).as_posix()
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        if source.suffix == ".py":
            target = args.out / "sources" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    provenance_model = _CachedBaselineProvenance if args.baseline_cache else _Provenance
    report = report_model(
        schema_version=(
            "calibrex.browser_edge_preparation_probe/v0.2"
            if args.baseline_cache
            else "calibrex.browser_edge_preparation_probe/v0.1"
        ),
        status="RUNNING",
        provenance=provenance_model(
            created_at=datetime.now(timezone.utc).isoformat(),
            command=sys.argv,
            source_head=head,
            baseline_source_head=baseline_head,
            baseline_source_sha256=hashlib.sha256(captured).hexdigest(),
            sources=hashes,
            source_worktree_dirty=bool(
                subprocess.check_output(["git", "status", "--porcelain"], cwd=root)
            ),
            whole_calibration_claim=False,
        ),
        runs=[],
        logs=[],
        errors=[],
        checks={"runtime": "native CPU"},
    )

    def save() -> None:
        (args.out / "report.json").write_text(
            report.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )

    save()
    try:
        for i, fixture in enumerate(args.fixture):
            run_model = PreparationCachedBaselineRun if args.baseline_cache else PreparationReuseRun
            run = run_model.model_validate(
                asyncio.run(
                    measure(
                        str(fixture),
                        source_head=head,
                        baseline_source_head=baseline_head,
                        baseline_path=str(baseline_path.resolve()),
                        baseline_cache_enabled=args.baseline_cache,
                    )
                )
            )
            run.fixture, run.fixture_index = fixture.name, i
            report.runs.append(run)
            print(
                json.dumps(
                    {
                        "fixture": run.fixture,
                        "cold_ratio": run.paired_median_cold_ratio,
                        "warm_ratio": run.paired_median_warm_ratio,
                        "three_request_ratio": run.paired_median_three_request_ratio,
                    }
                ),
                flush=True,
            )
            save()
        assert all(
            hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
            for name, digest in hashes.items()
        )
        report.checks["sources_unchanged"] = True
        report.status = "PASS"
    except Exception as error:
        report.status = "FAIL"
        report.errors.append(f"{type(error).__name__}: {error}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
