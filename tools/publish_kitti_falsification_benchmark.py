#!/usr/bin/env python3
"""Publish the full-scale KITTI camera-LiDAR falsification benchmark to ``docs/assets``.

A maintenance tool, not part of the Calibrex package. It runs the benchmark of
:mod:`calibrex.evaluation.kitti_falsification_benchmark` on the official KITTI raw drive
``2011_09_26_drive_0005_sync`` (50 frame pairs, the dataset's calibration and a declared
known-bad candidate), or reuses the output directory of such a run, and writes:

* ``docs/assets/kitti-camera-lidar-falsification.json``: the schema-valid
  ``slac.kitti_lidar_camera_falsification/v0.1`` artifact, with each trial's policy
  gates and headline metrics, every trial file bound by SHA-256 (paths relative to
  the run's output directory), and the data-only decisions of
  :mod:`calibrex.evaluation.kitti_edge_falsification`;
* ``docs/assets/kitti-camera-lidar-falsification.md``: its rendered summary, which
  the benchmark page includes and a test regenerates.

::

    python tools/publish_kitti_falsification_benchmark.py \\
        --dataset data/public/kitti_raw/2011_09_26/2011_09_26_drive_0005_sync \\
        --output-dir outputs/kitti-falsification
    # or, after a run: --from-output outputs/kitti-falsification

The KITTI data stay outside the repository (CC BY-NC-SA 3.0); a full run takes about
40 minutes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from calibrex.core.io import read_mapping
from calibrex.core.provenance import sha256_path
from calibrex.evaluation.kitti_falsification_benchmark import (
    KITTIFalsificationBenchmarkArtifact,
    attach_data_only_decisions,
    render_kitti_falsification_markdown,
    run_kitti_falsification_benchmark,
    summarize_kitti_trial,
)

ASSET_NAME = "kitti-camera-lidar-falsification"
DEFAULT_CONFIG = Path("examples/public_datasets/kitti_raw_2011_09_26_drive_0005/config.yaml")


def publishable(
    benchmark: KITTIFalsificationBenchmarkArtifact, output_dir: Path
) -> KITTIFalsificationBenchmarkArtifact:
    """The benchmark with trial summaries filled in and trial paths relative to the run.

    A run made before trials carried their summaries gets them from the trial files.
    Every trial file is re-hashed: a file that changed since the run is refused.
    """

    trials = []
    for trial in benchmark.trials:
        artifacts = []
        for artifact in trial.artifacts:
            path = Path(artifact.path)
            if path.is_absolute():
                path = path.relative_to(output_dir.resolve())
            if sha256_path(output_dir / path) != artifact.sha256:
                raise ValueError(f"{output_dir / path} changed since the benchmark ran")
            artifacts.append(artifact.model_copy(update={"path": path.as_posix()}))
        reason, rules, metrics, checks = summarize_kitti_trial(output_dir / trial.candidate_id)
        trials.append(
            trial.model_copy(
                update={
                    "artifacts": artifacts,
                    "assessment_reason": reason,
                    "rules": rules,
                    "metrics": metrics,
                    "evidence_checks": checks,
                }
            )
        )
    return benchmark.model_copy(update={"trials": trials})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path, help="official 2011_09_26_drive_0005_sync")
    source.add_argument("--from-output", type=Path, help="output directory of a finished run")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, help="where a new run writes its trials")
    parser.add_argument("--assets", type=Path, default=Path("docs/assets"))
    args = parser.parse_args(argv)

    if args.from_output is not None:
        output_dir = args.from_output
        benchmark = KITTIFalsificationBenchmarkArtifact.model_validate(
            read_mapping(output_dir / "benchmark.json")
        )
    else:
        if args.output_dir is None:
            parser.error("--dataset needs --output-dir")
        output_dir = args.output_dir
        benchmark = run_kitti_falsification_benchmark(
            args.dataset, config_path=args.config, output_dir=output_dir
        )
    published = publishable(benchmark, output_dir)
    if not published.data_only:
        # A run made before the data-only decision existed: add it (its own git commit
        # is recorded in each decision).
        published = attach_data_only_decisions(published, published.dataset_path)
    args.assets.mkdir(parents=True, exist_ok=True)
    published.save(args.assets / f"{ASSET_NAME}.json")
    (args.assets / f"{ASSET_NAME}.md").write_text(
        render_kitti_falsification_markdown(published), encoding="utf-8"
    )
    print(f"{published.status}: {published.reason}; wrote {args.assets / ASSET_NAME}.json and .md")
    return 0 if published.status == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
