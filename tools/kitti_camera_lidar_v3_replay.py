#!/usr/bin/env python3
"""Capture spent KITTI raw check inputs and compare two rules on identical inputs.

Development only. Capture runs seven real checks, saving each run's synchronized,
motion-corrected frames before estimation. Replay never opens the bag or the network.
It preserves missing/failed drives and publishes descriptive metrics without audit gates.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Literal
from unittest.mock import patch

import numpy as np
from pydantic import Field
from scipy.spatial.transform import Rotation

from calibrex.check.estimators import camera_lidar_axis_estimates
from calibrex.check.verdict import VerdictOptions, judge_pair
from calibrex.core.camera_lidar_edge import (
    CameraLidarEdgeArtifact,
    CameraLidarEdgeMotion,
    CameraLidarEdgeProvenance,
    load_camera_lidar_edge,
)
from calibrex.core.result import StrictModel
from calibrex.data.ros2_camera_lidar import CameraLidarFrame
from calibrex.evaluation import camera_lidar_edge as edge

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "baseline": None,
    **{f"inject_{axis}+{angle}": f"{axis}:{angle}" for angle in (3, 1) for axis in "xyz"},
}


def _tool(name: str) -> ModuleType:
    key = f"_v3_replay_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ROOT / "tools" / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def spent_drives() -> tuple[str, ...]:
    """The fixed spent-data list, from the two committed v2 score files."""

    result = []
    for phase in ("development", "evaluation"):
        path = ROOT / f"docs/assets/kitti_camera_lidar_v2/{phase}/scores.json"
        result.extend(json.loads(path.read_text())["drives"])
    if len(result) != 55 or len(set(result)) != 55:
        raise ValueError("expected the 55 distinct spent KITTI raw drives")
    return tuple(result)


def _require_spent(drive: str) -> None:
    if drive not in spent_drives():
        raise ValueError(f"not a spent KITTI raw development drive: {drive}")


def _reference(conversion: dict[str, Any], drive: str) -> np.ndarray:
    date, number = drive[:10], drive[11:]
    sources = conversion.get("source", {})
    source_drives = [item["drive"] for item in sources.get("drives", [])]
    if (
        sources.get("dataset_family") != "kitti_raw"
        or source_drives != [f"{date}_drive_{number}_sync"]
        or conversion.get("camera", {}).get("directory") != "image_02"
        or conversion.get("velodyne_time") != "azimuth"
    ):
        raise ValueError("bag conversion must match this spent drive, image_02 and azimuth time")
    reference = np.asarray(conversion["camera"]["t_cam_velo"], dtype=np.float64)
    if reference.shape != (4, 4) or not np.isfinite(reference).all():
        raise ValueError("conversion must record a finite 4x4 reference transform")
    return reference


def _check_candidate(candidate: np.ndarray, reference: np.ndarray, run: str) -> None:
    expected = reference.copy()
    injection = RUNS[run]
    if injection is not None:
        axis, amount = injection.split(":")
        turn = Rotation.from_rotvec(np.radians(float(amount)) * np.eye(3)["xyz".index(axis)])
        expected[:3, :] = turn.as_matrix() @ reference[:3, :]
    if not np.allclose(candidate, expected, rtol=0.0, atol=1e-8):
        raise ValueError(f"{run}: candidate differs from the converted calibration/injection")


class FrameRecord(StrictModel):
    """Non-array part of one exact estimator input frame."""

    time_s: float = Field(allow_inf_nan=False)
    image_time_s: float = Field(allow_inf_nan=False)
    cloud_frame_id: str
    image_frame_id: str
    has_point_times: bool


class Snapshot(StrictModel):
    """Versioned JSON manifest for a non-pickle NumPy input archive."""

    schema_version: Literal["calibrex.camera_lidar_development_input/v1"] = (
        "calibrex.camera_lidar_development_input/v1"
    )
    drive: str
    run: str
    arrays_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    conversion_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    camera: dict[str, Any]
    options: dict[str, Any]
    frames: list[FrameRecord]
    input_provenance: CameraLidarEdgeProvenance
    parent_frame: str
    child_frame: str
    reference_label: str | None = None
    start_note: str | None = None
    motion_compensation: CameraLidarEdgeMotion | None = None
    generator_sha256: str
    estimator_sha256: str


def save_snapshot(
    output: Path,
    frames: Sequence[CameraLidarFrame],
    camera: edge.EdgeCamera,
    candidate: np.ndarray,
    *,
    drive: str,
    run: str,
    conversion_sha256: str,
    options: edge.EdgeAlignmentOptions,
    provenance: CameraLidarEdgeProvenance,
    parent_frame: str = "camera",
    child_frame: str = "lidar",
    reference_label: str | None = None,
    start_note: str | None = None,
    motion_compensation: CameraLidarEdgeMotion | None = None,
) -> None:
    """Atomically save exact post-motion frames; refuse existing outputs and fresh IDs."""

    _require_spent(drive)
    if run not in RUNS or options.start_scan_min_control_t is not None:
        raise ValueError("capture a declared run with the current default adoption rule")
    if output.exists():
        raise FileExistsError(output)
    arrays = {"candidate": np.asarray(candidate, dtype=np.float64)}
    records = []
    for i, frame in enumerate(frames):
        arrays[f"image_{i}"] = frame.image
        arrays[f"points_{i}"] = frame.points
        if frame.point_time_offsets_s is not None:
            arrays[f"times_{i}"] = frame.point_time_offsets_s
        records.append(
            FrameRecord(
                time_s=frame.time_s,
                image_time_s=frame.image_time_s,
                cloud_frame_id=frame.cloud_frame_id,
                image_frame_id=frame.image_frame_id,
                has_point_times=frame.point_time_offsets_s is not None,
            )
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".capture-") as name:
        stage = Path(name) / "snapshot"
        stage.mkdir()
        np.savez_compressed(stage / "arrays.npz", **arrays)
        metadata = Snapshot(
            drive=drive,
            run=run,
            arrays_sha256=_sha(stage / "arrays.npz"),
            conversion_sha256=conversion_sha256,
            camera=camera.as_dict(),
            options=options.as_dict(),
            frames=records,
            input_provenance=provenance,
            parent_frame=parent_frame,
            child_frame=child_frame,
            reference_label=reference_label,
            start_note=start_note,
            motion_compensation=motion_compensation,
            generator_sha256=_sha(Path(__file__)),
            estimator_sha256=_sha(Path(edge.__file__)),
        )
        _write(stage / "snapshot.json", metadata.model_dump(mode="json"))
        os.rename(stage, output)


def load_snapshot(
    path: Path,
) -> tuple[
    Snapshot, list[CameraLidarFrame], edge.EdgeCamera, np.ndarray, edge.EdgeAlignmentOptions
]:
    """Verify full payload bytes, schema and array shapes; never unpickle an object."""

    metadata = Snapshot.model_validate_json((path / "snapshot.json").read_bytes())
    _require_spent(metadata.drive)
    if metadata.run not in RUNS:
        raise ValueError("unknown snapshot run")
    if set(metadata.options) != {item.name for item in fields(edge.EdgeAlignmentOptions)}:
        raise ValueError("option fields changed; use an explicit snapshot migration")
    values = dict(metadata.options)
    for key in ("rotation_levels_deg", "translation_levels_m"):
        values[key] = tuple(values[key])
    options = edge.EdgeAlignmentOptions(**values)
    if options.start_scan_min_control_t is not None:
        raise ValueError("snapshot is not from the current default adoption rule")
    camera_values = dict(metadata.camera)
    camera_values["distortion"] = tuple(camera_values["distortion"])
    camera = edge.EdgeCamera(**camera_values)
    if _sha(path / "arrays.npz") != metadata.arrays_sha256:
        raise ValueError("snapshot arrays digest mismatch")
    with np.load(path / "arrays.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    expected = {"candidate"}
    frames_out = []
    for i, record in enumerate(metadata.frames):
        expected.update((f"image_{i}", f"points_{i}"))
        image, points = arrays[f"image_{i}"], arrays[f"points_{i}"]
        if image.dtype != np.uint8 or image.ndim != 2:
            raise ValueError("snapshot image must be uint8 luminance")
        if points.dtype != np.float32 or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("snapshot points must be float32 Nx3")
        times = None
        if record.has_point_times:
            expected.add(f"times_{i}")
            times = arrays[f"times_{i}"]
            if times.dtype != np.float64 or times.shape != (len(points),):
                raise ValueError("snapshot point times must be float64, one per point")
        frames_out.append(
            CameraLidarFrame(
                time_s=record.time_s,
                image_time_s=record.image_time_s,
                image=image,
                points=points,
                cloud_frame_id=record.cloud_frame_id,
                image_frame_id=record.image_frame_id,
                point_time_offsets_s=times,
            )
        )
    candidate = arrays["candidate"]
    if candidate.dtype != np.float64 or candidate.shape != (4, 4):
        raise ValueError("snapshot candidate must be float64 4x4")
    if set(arrays) != expected or any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("snapshot has extra arrays or nonfinite values")
    for array in arrays.values():
        array.setflags(write=False)
    return metadata, frames_out, camera, candidate, options


def capture_drive(bag: Path, drive: str, output: Path) -> int:
    """Run the real check seven times; retain per-run inputs and failed attempts."""

    from calibrex.cli.main import main as check_main

    _require_spent(drive)  # before inspecting any bag contents
    if output.resolve().is_relative_to(bag.resolve()) or bag.resolve().is_relative_to(
        output.resolve()
    ):
        raise ValueError("capture output must be separate from the source bag")
    conversion_path = bag / "calibrex_conversion.json"
    conversion = json.loads(conversion_path.read_text())
    reference = _reference(conversion, drive)
    output.mkdir(parents=True, exist_ok=False)
    (output / "conversion.json").write_bytes(conversion_path.read_bytes())
    conversion_sha = _sha(conversion_path)
    original = edge.estimate_from_frames
    records: dict[str, Any] = {}
    _write(
        output / "capture_manifest.json",
        {
            "analysis": "post hoc development",
            "drive": drive,
            "complete": False,
            "conversion_sha256": conversion_sha,
            "runs": {},
            "files_sha256": {},
            "generator_sha256": _sha(Path(__file__)),
        },
    )
    for run, injection in RUNS.items():
        calls = 0

        def capture(
            frames: Sequence[CameraLidarFrame],
            camera: edge.EdgeCamera,
            candidate: np.ndarray,
            *,
            capture_run: str = run,
            **kwargs: Any,
        ) -> CameraLidarEdgeArtifact:
            nonlocal calls
            calls += 1
            _check_candidate(candidate, reference, capture_run)
            save_snapshot(
                output / capture_run,
                frames,
                camera,
                candidate,
                drive=drive,
                run=capture_run,
                conversion_sha256=conversion_sha,
                **kwargs,
            )
            artifact = original(frames, camera, candidate, **kwargs)
            artifact.save(output / capture_run / "captured_edge.json")
            return artifact

        command = [
            "check",
            str(bag),
            "--pairs",
            "camera-lidar",
            "--no-cache",
            "--fail-on",
            "never",
            "--output",
            str(output / f"{run}.json"),
            "--evidence-dir",
            str(output / f"{run}_evidence"),
        ]
        started = time.monotonic()
        try:
            if injection is not None:
                axis, amount = injection.split(":")
                target = output / f"{run}_frames.yaml"
                _tool("kitti_camera_lidar_sota_run").turned_frames(bag, axis, float(amount), target)
                command += ["--tf", str(target)]
            with (
                (output / f"{run}.log").open("w", encoding="utf-8") as log,
                redirect_stdout(log),
                redirect_stderr(log),
                patch.object(edge, "estimate_from_frames", capture),
            ):
                code = check_main(command)
            if code != 0 or calls != 1:
                raise ValueError(
                    f"check exit {code}; captured {calls} estimator calls (expected 1)"
                )
            if not (output / run / "captured_edge.json").is_file():
                payload = json.loads((output / f"{run}.json").read_text())
                (pair,) = [item for item in payload["pairs"] if item["pair"] == "camera-lidar"]
                raise ValueError(f"check did not produce estimator evidence: {pair.get('reason')}")
            records[run] = {"ok": True, "command": command, "wall_s": time.monotonic() - started}
        except Exception as exc:
            records[run] = {
                "ok": False,
                "command": command,
                "error": f"{type(exc).__name__}: {exc}",
                "wall_s": time.monotonic() - started,
            }
    files = {
        p.relative_to(output).as_posix(): _sha(p)
        for p in output.rglob("*")
        if p.is_file() and p.name != "capture_manifest.json"
    }
    complete = all(record["ok"] for record in records.values())
    _write(
        output / "capture_manifest.json",
        {
            "analysis": "post hoc development",
            "drive": drive,
            "complete": complete,
            "conversion_sha256": conversion_sha,
            "runs": records,
            "files_sha256": files,
            "generator_sha256": _sha(Path(__file__)),
        },
    )
    return 0 if complete else 1


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _same(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
    return left == right


def _run_input(artifact: CameraLidarEdgeArtifact, candidate: np.ndarray) -> Any:
    scorer = _tool("score_kitti_camera_lidar_v2")
    judgement = judge_pair(camera_lidar_axis_estimates(artifact, candidate), VerdictOptions())
    rotation = (
        None
        if artifact.transform is None
        else Rotation.from_quat(artifact.transform.rotation_quat_xyzw).as_matrix()
    )
    return scorer.RunInput(
        verdict=judgement.verdict,
        solver_status=artifact.solver_status,
        candidate=candidate[:3, :3],
        estimate=rotation,
        judged={axis.name: axis.status for axis in judgement.axes},
    )


def compare_drives(
    snapshots: Path,
    output: Path,
    *,
    drives: Sequence[str] | None = None,
    margin: float = 3.5,
) -> dict[str, Any]:
    """Replay both rules, retain failures, require capture parity and never set audit gates."""

    edge.EdgeAlignmentOptions(start_scan_min_control_t=margin)
    order = list(spent_drives() if drives is None else drives)
    if len(set(order)) != len(order) or not order:
        raise ValueError("choose distinct spent drives")
    for drive in order:
        _require_spent(drive)
    if output.resolve().is_relative_to(snapshots.resolve()) or snapshots.resolve().is_relative_to(
        output.resolve()
    ):
        raise ValueError("comparison output must be separate from the input snapshots")
    output.mkdir(parents=True, exist_ok=False)
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    scorer = _tool("score_kitti_camera_lidar_v2")
    scores: dict[str, dict[str, Any]] = {"current": {}, "proposed": {}}
    errors: dict[str, str] = {}
    manifests: dict[str, str] = {}
    for drive in order:
        root = snapshots / drive
        try:
            manifest_path = root / "capture_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            if manifest["drive"] != drive or not manifest["complete"]:
                raise ValueError("capture is incomplete or names a different drive")
            if set(manifest["runs"]) != set(RUNS) or not all(
                entry["ok"] for entry in manifest["runs"].values()
            ):
                raise ValueError("capture does not contain seven successful checks")
            manifests[drive] = _sha(manifest_path)
            for name, digest in manifest["files_sha256"].items():
                path = root / name
                if not path.resolve().is_relative_to(root.resolve()) or _sha(path) != digest:
                    raise ValueError("capture file path or digest mismatch")
            if "conversion.json" not in manifest["files_sha256"] or (
                _sha(root / "conversion.json") != manifest["conversion_sha256"]
            ):
                raise ValueError("capture lacks its digest-pinned conversion reference")
            reference = _reference(json.loads((root / "conversion.json").read_text()), drive)
            inputs: dict[str, dict[str, Any]] = {"current": {}, "proposed": {}}
            for run in RUNS:
                path = root / run
                for filename in ("snapshot.json", "arrays.npz", "captured_edge.json"):
                    if f"{run}/{filename}" not in manifest["files_sha256"]:
                        raise ValueError(f"capture manifest omits {run}/{filename}")
                if f"{run}.json" not in manifest["files_sha256"]:
                    raise ValueError(f"capture manifest omits {run}.json")
                meta, frames_in, camera, candidate, options = load_snapshot(path)
                expected_options = edge.EdgeAlignmentOptions(max_seconds=options.max_seconds)
                if options.as_dict() != expected_options.as_dict():
                    raise ValueError("capture recipe differs from the default check options")
                _check_candidate(candidate, reference, run)
                if (
                    meta.drive != drive
                    or meta.run != run
                    or (meta.conversion_sha256 != manifest["conversion_sha256"])
                ):
                    raise ValueError("mixed snapshot identity or conversion digest")
                captured = load_camera_lidar_edge(path / "captured_edge.json")
                for rule, threshold in (("current", None), ("proposed", margin)):
                    destination = output / drive / rule
                    destination.mkdir(parents=True, exist_ok=True)
                    provenance = meta.input_provenance.model_copy(
                        update={
                            "generator": "tools/kitti_camera_lidar_v3_replay.py",
                            "generator_version": "1",
                            "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "input_sha256": _sha(path / "snapshot.json"),
                            "input_digest_scope": (
                                "snapshot JSON in full, pins full arrays.npz SHA-256"
                            ),
                            "command": [
                                "compare",
                                str(snapshots),
                                "--min-far-control-t",
                                str(margin),
                            ],
                        }
                    )
                    artifact = edge.estimate_from_frames(
                        frames_in,
                        camera,
                        candidate,
                        options=replace(options, start_scan_min_control_t=threshold),
                        provenance=provenance,
                        parent_frame=meta.parent_frame,
                        child_frame=meta.child_frame,
                        reference_label=meta.reference_label,
                        start_note=meta.start_note,
                        motion_compensation=meta.motion_compensation,
                    )
                    artifact.save(destination / f"{run}.json")
                    if rule == "current" and not _same(
                        artifact.model_dump(exclude={"provenance"}),
                        captured.model_dump(exclude={"provenance"}),
                    ):
                        raise ValueError(f"current replay differs from captured evidence: {run}")
                    inputs[rule][run] = _run_input(artifact, candidate)
                    if rule == "current":
                        checks = json.loads((root / f"{run}.json").read_text())["pairs"]
                        (pair,) = [item for item in checks if item["pair"] == "camera-lidar"]
                        judged = {
                            item["name"]: item["status"]
                            for item in pair["axes"]
                            if item["name"] in ("roll", "pitch", "yaw")
                        }
                        if pair["verdict"] != inputs[rule][run].verdict or (
                            judged != inputs[rule][run].judged
                        ):
                            raise ValueError(f"replayed verdict differs from real check: {run}")
            for rule in scores:
                score = scorer.score_drive(
                    scorer.DriveInput(
                        baseline=inputs[rule]["baseline"],
                        turned={spec: inputs[rule][run] for run, spec in RUNS.items() if spec},
                    )
                )
                scores[rule][drive] = asdict(score)
        except Exception as exc:
            errors[drive] = f"{type(exc).__name__}: {exc}"
            for rule in scores:
                scores[rule][drive] = asdict(scorer.DriveScore("failed", errors[drive]))
    summary = _tool("kitti_camera_lidar_v3_development").summarize_scores
    metrics = {rule: summary(values) for rule, values in scores.items()}
    before = metrics["current"]["judged_fraction_scored"]
    after = metrics["proposed"]["judged_fraction_scored"]
    full = (
        set(order) == set(spent_drives())
        and not errors
        and not any(value["failed_drives"] for value in metrics.values())
    )
    report = {
        "analysis": "post hoc development",
        "status": "complete" if full else "incomplete",
        "drives": order,
        "errors": errors,
        "metrics": metrics,
        "scores": scores,
        "proposed_margin": margin,
        "coverage_change_percentage_points": None
        if before is None or after is None
        else 100.0 * (after - before),
        "audit_thresholds": None,
        "provenance": {
            "generator": "tools/kitti_camera_lidar_v3_replay.py",
            "generator_version": "1",
            "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "generator_sha256": _sha(Path(__file__)),
            "estimator_sha256": _sha(Path(edge.__file__)),
            "scoring_helpers_sha256": {
                name: _sha(ROOT / "tools" / f"{name}.py")
                for name in ("score_kitti_camera_lidar_v2", "kitti_camera_lidar_v3_development")
            },
            "capture_manifests_sha256": manifests,
            "reference": "each KITTI day's published calibration, not ground truth",
        },
    }
    _write(output / "comparison.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture", help="spent raw bag -> seven check runs and snapshots")
    capture.add_argument("bag", type=Path)
    capture.add_argument("--drive", required=True)
    capture.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare", help="saved snapshots -> paired development replay")
    compare.add_argument("snapshots", type=Path)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--drive", action="append", default=None)
    compare.add_argument("--min-far-control-t", type=float, default=3.5)
    args = parser.parse_args(argv)
    if args.command == "capture":
        return capture_drive(args.bag, args.drive, args.output)
    report = compare_drives(
        args.snapshots,
        args.output,
        drives=args.drive,
        margin=args.min_far_control_t,
    )
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
