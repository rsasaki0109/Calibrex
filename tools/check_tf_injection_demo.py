"""Reproducible demo: inject a yaw error into the deployed tf and let ``calibrex check`` catch it.

The headline of ``calibrex check`` is that a deployed extrinsic that is a degree
off is flagged without any target or reference. This script reproduces it on the
KITTI raw dev drives of the lidar-vehicle protocol (0005, 0009, 0014, 0015, 0022):

1. ``calibrex convert kitti-raw`` writes one pooled rosbag2 (skipped when ``--bag`` exists);
2. ``calibrex check`` runs on the vendor ``/tf_static`` (the KITTI calibration);
3. the same check runs with ``velo_link`` turned by +1 and +3 degrees about its parent's
   z axis, through ``--tf`` (a ``slac.check_frames`` file written by
   ``tools/make_check_frames_perturbation.py``).

The estimators run once on the first variant; the estimator cache (``--cache-dir``)
makes the others a few seconds, because the estimates do not depend on the candidate.
Outputs in ``--out-dir``: ``summary.md`` (the table), ``terminal_summary.txt`` (the
verdict section of the CLI output for the +1 degree run), ``summary.json`` (provenance
and the numbers) and ``check_yaw1.html`` (the HTML report of the +1 degree run). Absolute
paths are shortened to their last component, so the files can be committed::

    python tools/check_tf_injection_demo.py --work-dir /tmp/demo \\
        --drive ~/data/kitti_raw/2011_09_26/2011_09_26_drive_0005_sync ... \\
        --out-dir docs/assets/calibrex_check_demo
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from calibrex.check.runner import CheckRunOptions, build_calibration_check, format_check_table
from calibrex.check.verdict import VerdictOptions
from calibrex.core.calibration_check import CalibrationCheckArtifact
from calibrex.core.io import write_mapping
from calibrex.visualization.check_report import write_check_html

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PAIRS = ("lidar-vehicle", "imu-vehicle", "ins-lidar", "lidar-wheel_odometry", "imu-lidar")
DEFAULT_DRIVES = ("0005", "0009", "0014", "0015", "0022")
PATH = re.compile(r"(?<![\w.])/(?:[^\s'\"<>/]+/)*[^\s'\"<>/]+")


def redact(text: str) -> str:
    """Shorten absolute paths to ``.../name`` so the outputs carry no local paths."""

    return PATH.sub(lambda match: ".../" + match.group(0).rsplit("/", 1)[-1], text)


def convert(drives: list[Path], bag: Path) -> None:
    """Pool the drives into one rosbag2 with ``calibrex convert kitti-raw``."""

    command = [
        sys.executable,
        "-m",
        "calibrex",
        "convert",
        "kitti-raw",
        *map(str, drives),
        "--output",
        str(bag),
        "--overwrite",
    ]
    subprocess.run(command, check=True)


def perturbed_frames(bag: Path, frame: str, yaw_deg: float, output: Path) -> None:
    """Write the ``--tf`` file that turns ``frame`` by ``yaw_deg``."""

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "make_check_frames_perturbation.py"),
            str(bag),
            frame,
            f"{yaw_deg:g}",
            "--output",
            str(output),
        ],
        check=True,
    )


def _pair(artifact: CalibrationCheckArtifact, name: str) -> str:
    record = next((p for p in artifact.pairs if p.pair == name), None)
    if record is None or record.status == "skipped":
        return "skipped"
    judged = ", ".join(
        f"{axis.name} {abs(axis.candidate_error):.2f}/{axis.tolerance:.2f}" for axis in record.axes
    )
    cover = " (partial)" if record.coverage == "partial" and record.axes else ""
    return f"**{record.status}**{cover}" + (f"<br>{judged} deg" if judged else "")


def _closure(artifact: CalibrationCheckArtifact) -> str:
    report = artifact.closures
    if report is None or not report.loops:
        return "no loop"
    loop = report.loops[0]
    judged = ", ".join(f"{a.name} {a.candidate_error:+.2f}" for a in loop.axes)
    return f"{loop.verdict}<br>{judged} deg"


def terminal_summary(artifact: CalibrationCheckArtifact) -> str:
    """The verdict section of the CLI table (pairs, closure, overall verdict)."""

    lines = format_check_table(artifact).splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("pairs ("))
    stop = next(i for i, line in enumerate(lines) if line.startswith("summary:"))
    return redact("\n".join(lines[start:stop]).rstrip() + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--work-dir", type=Path, required=True, help="bag, cache and raw artifacts")
    parser.add_argument("--bag", type=Path, help="converted pooled bag (default: WORK_DIR/bag)")
    parser.add_argument("--drive", type=Path, action="append", default=[], help="raw drive dir")
    parser.add_argument("--out-dir", type=Path, default=REPO / "docs/assets/calibrex_check_demo")
    parser.add_argument("--cache-dir", type=Path, help="estimator cache (default: WORK_DIR/cache)")
    parser.add_argument("--yaw-deg", type=float, nargs="+", default=[1.0, 3.0])
    parser.add_argument("--frame", default="velo_link", help="frame whose rotation is perturbed")
    parser.add_argument("--vehicle-frame", default="base_link")
    parser.add_argument("--pairs", default=",".join(DEFAULT_PAIRS))
    args = parser.parse_args(argv)

    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    bag = args.bag or work / "bag"
    if not bag.exists():
        if not args.drive:
            parser.error(f"{bag} does not exist: pass --drive once per raw drive to convert")
        convert(args.drive, bag)
    cache = args.cache_dir or work / "cache"
    variants: list[tuple[str, float | None]] = [("vendor", None)]
    variants += [(f"yaw{value:g}", value) for value in args.yaw_deg]

    results: dict[str, CalibrationCheckArtifact] = {}
    runtimes: dict[str, float] = {}
    for name, yaw in variants:
        tf_files: list[Path] = []
        if yaw is not None:
            frames = work / f"frames_{name}.yaml"
            perturbed_frames(bag, args.frame, yaw, frames)
            tf_files = [frames]
        started = time.monotonic()
        artifact = build_calibration_check(
            bag,
            tf_files=tf_files,
            vehicle_frame=args.vehicle_frame,
            topic_kinds={"/oxts/twist": "wheel"},
            command=[
                "calibrex",
                "check",
                str(bag),
                "--vehicle-frame",
                args.vehicle_frame,
                "--topic-kind",
                "/oxts/twist=wheel",
                "--pairs",
                args.pairs,
                *(["--tf", str(tf_files[0])] if tf_files else []),
            ],
            run=CheckRunOptions(
                verdict=VerdictOptions(),
                pairs=tuple(args.pairs.split(",")),
                evidence_dir=work / f"check_{name}_evidence",
                base_dir=work,
                cache_dir=cache,
            ),
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
        runtimes[name] = time.monotonic() - started
        write_mapping(
            work / f"check_{name}.yaml", artifact.model_dump(mode="json", exclude_none=True)
        )
        results[name] = artifact
        print(f"{name}: {artifact.overall_verdict} ({runtimes[name]:.0f} s)", file=sys.stderr)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    first = next(iter(results.values()))
    table = [
        "| deployed tf | overall | lidar-vehicle | lidar-wheel_odometry | ins-lidar "
        "| imu-vehicle | closure |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, yaw in variants:
        artifact = results[name]
        label = "vendor (KITTI calibration)" if yaw is None else f"{args.frame} yaw {yaw:+g} deg"
        table.append(
            f"| {label} | **{artifact.overall_verdict}** | {_pair(artifact, 'lidar-vehicle')} "
            f"| {_pair(artifact, 'lidar-wheel_odometry')} | {_pair(artifact, 'ins-lidar')} "
            f"| {_pair(artifact, 'imu-vehicle')} | {_closure(artifact)} |"
        )
    (out / "summary.md").write_text(
        "# calibrex check: yaw injected into the deployed tf (KITTI raw dev drives)\n\n"
        "Cells are the pair verdict and `|candidate error| / tolerance` per judged axis. "
        "The estimates (and so the closure) do not depend on the candidate.\n\n"
        + "\n".join(table)
        + "\n",
        encoding="utf-8",
    )
    reference = results.get("yaw1") or next(iter(results.values()))
    (out / "terminal_summary.txt").write_text(terminal_summary(reference), encoding="utf-8")
    summary = {
        "schema": "calibrex.check_tf_injection_demo/v0",
        "bag": {
            "name": bag.name,
            "input_sha256": first.bag.input_sha256,
            "input_digest_scope": first.bag.input_digest_scope,
            "topic_count": first.bag.topic_count,
        },
        "drives": [path.name for path in args.drive] or list(DEFAULT_DRIVES),
        "generator": first.provenance.generator,
        "generator_version": first.provenance.generator_version,
        "git_commit": first.provenance.git_commit,
        "frame": args.frame,
        "variants": {
            name: {
                "yaw_injected_deg": yaw,
                "overall_verdict": results[name].overall_verdict,
                "command": [redact(item) for item in results[name].provenance.command],
                "created_at": results[name].provenance.created_at,
                "pairs": {
                    pair.pair: {
                        "status": pair.status,
                        "coverage": pair.coverage,
                        "axes": {
                            axis.name: {
                                "candidate_error_deg": round(axis.candidate_error, 4),
                                "tolerance_deg": round(axis.tolerance, 4),
                                "status": axis.status,
                            }
                            for axis in pair.axes
                        },
                        "unchecked": [axis.name for axis in pair.unchecked_axes],
                    }
                    for pair in results[name].pairs
                    if pair.status != "skipped"
                },
                "closure": results[name].closures.model_dump(mode="json", exclude_none=True)
                if results[name].closures
                else None,
            }
            for name, yaw in variants
        },
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_name = "yaw1" if "yaw1" in results else next(iter(results))
    write_check_html(results[report_name], out / f"check_{report_name}.html", redact_paths=True)
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
