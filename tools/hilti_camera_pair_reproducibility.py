#!/usr/bin/env python3
"""Split-half reproducibility of the camera-camera check on Hilti 2022's five cameras.

An evaluation harness, not part of the Calibrex package. On ``exp01`` and ``exp21`` the
side and down camera pairs read differently by up to 0.3 deg, several times their
jackknife std (``docs/tutorials/calibrex_check.md``). This tool asks whether that std is
honest within one recording: it reads frames where all five alphasense cameras fired
together, ``--spacing-s`` apart, and estimates every camera pair on

* ``all`` frames, the check's own setting when there are 48;
* the ``first`` and ``second`` half in time: different parts of the scene;
* the ``even`` and ``odd`` frames: the same parts of the scene, interleaved.

For each split pair it reports the difference per axis over its combined std
(``z``). A std that describes the estimate's spread gives |z| below 2 or so on both
splits; scene-dependent bias shows as large ``first``/``second`` z with small
``even``/``odd`` z.

::

    python tools/hilti_camera_pair_reproducibility.py exp21_head.bag \\
        --camchain calib_3_cam0-1-camchain-imucam.yaml \\
        --camchain calib_3_cam2-camchain-imucam.yaml \\
        --camchain calib_3_cam3-camchain-imucam.yaml \\
        --camchain calib_3_cam4-camchain-imucam.yaml \\
        --output exp21_reproducibility.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from calibrex.check.estimators import transform_matrix
from calibrex.check.tf_sources import camchain_entries, load_tf_file, merge_sources
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.stereo_rotation import StereoRotationArtifact, StereoRotationProvenance
from calibrex.data import ros_cdr
from calibrex.data.rosbag2 import iter_messages
from calibrex.evaluation.stereo_rotation import (
    StereoRotationOptions,
    _gray,
    estimate_stereo_rotation,
    shared_view_fraction,
    stereo_camera,
)
from calibrex.evaluation.visual_rotation import CameraModel

TOPICS = tuple(f"/alphasense/cam{index}/image_raw" for index in range(5))
SPLITS = ("all", "first", "second", "even", "odd")
GROUP_TOLERANCE_NS = 2_000_000


def read_synchronised(
    bag: Path, topics: tuple[str, ...], spacing_s: float, count: int
) -> tuple[list[dict[str, np.ndarray]], list[float]]:
    """Up to ``count`` sets of images, one per topic, stamped within 2 ms of each other."""

    groups: list[dict[str, np.ndarray]] = []
    stamps: list[float] = []
    current: dict[str, np.ndarray] = {}
    anchor: int | None = None
    due = -1
    for connection, stamp, payload in iter_messages(bag, topics=set(topics)):
        header = ros_cdr.decode_ros2_image(connection.topic, stamp, payload, include_data=False)
        when = header.timestamp_ns
        if anchor is not None and when - anchor > 10 * GROUP_TOLERANCE_NS:
            current, anchor = {}, None  # a camera missed this set: start again
        if anchor is None:
            if when < due:
                continue
            anchor = when
        if abs(when - anchor) <= GROUP_TOLERANCE_NS and connection.topic not in current:
            current[connection.topic] = _gray(connection.topic, stamp, payload).astype(np.float32)
        if len(current) == len(topics):
            groups.append(current)
            stamps.append(anchor * 1.0e-9)
            due = anchor + round(spacing_s * 1.0e9)
            current, anchor = {}, None
            if len(groups) == count:
                break
    return groups, stamps


def split_indices(count: int) -> dict[str, list[int]]:
    half = count // 2
    return {
        "all": list(range(count)),
        "first": list(range(half)),
        "second": list(range(half, 2 * half)),
        "even": list(range(0, count, 2)),
        "odd": list(range(1, count, 2)),
    }


def summary(artifact: StereoRotationArtifact) -> dict[str, object]:
    return {
        "policy_status": artifact.policy_status,
        "frames": artifact.frames,
        "patches": artifact.patches,
        "patches_constraining_rows": artifact.patches_constraining_rows,
        "axes": {
            axis.name: {
                "error_deg": round(axis.candidate_error_deg, 4),
                "std_deg": round(axis.std_deg, 4),
                "control_response": round(axis.control_response, 3),
                "observable": axis.observable,
            }
            for axis in artifact.axes
        },
    }


def half_differences(splits: dict[str, dict[str, object]]) -> dict[str, dict[str, float]]:
    """Per axis and split pair: ``z`` = difference over the combined std."""

    out: dict[str, dict[str, float]] = {}
    for a, b in (("first", "second"), ("even", "odd")):
        axes_a = splits[a]["axes"]
        axes_b = splits[b]["axes"]
        assert isinstance(axes_a, dict) and isinstance(axes_b, dict)
        for name in ("roll", "pitch", "yaw"):
            if name not in axes_a or name not in axes_b:
                continue
            ea, eb = axes_a[name], axes_b[name]
            sigma = math.hypot(ea["std_deg"], eb["std_deg"])
            out.setdefault(name, {})[f"{a}-{b}_deg"] = round(ea["error_deg"] - eb["error_deg"], 4)
            out[name][f"{a}-{b}_z"] = round((ea["error_deg"] - eb["error_deg"]) / sigma, 2)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("bag", type=Path)
    parser.add_argument("--camchain", type=Path, action="append", required=True)
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--spacing-s", type=float, default=1.0)
    parser.add_argument("--pairs", default=None, help="e.g. 0-1,1-3 (default: every pair)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    commit = git_commit()
    tree, _ = merge_sources(None, [load_tf_file(path) for path in args.camchain])
    models = {
        frame: CameraModel.from_kalibr(entry)
        for path in args.camchain
        for _, frame, entry in camchain_entries(path)
    }
    started = time.monotonic()
    groups, stamps = read_synchronised(args.bag, TOPICS, args.spacing_s, args.frames)
    print(f"read {len(groups)} synchronised sets in {time.monotonic() - started:.0f} s", flush=True)
    if len(groups) < 16:
        parser.error(f"only {len(groups)} synchronised sets")
    height, width = groups[0][TOPICS[0]].shape
    wanted = (
        [tuple(int(v) for v in item.split("-")) for item in args.pairs.split(",")]
        if args.pairs
        else list(itertools.combinations(range(5), 2))
    )
    options = StereoRotationOptions(max_pairs=len(groups))
    results: dict[str, object] = {}
    for i, j in wanted:
        left_frame, right_frame = TOPICS[i].lstrip("/"), TOPICS[j].lstrip("/")
        relation = tree.lookup(left_frame, right_frame)
        if relation is None:
            continue
        candidate = transform_matrix(relation.translation_m, relation.rotation_quat_xyzw)
        left = stereo_camera(models[left_frame], width, height)
        right = stereo_camera(models[right_frame], width, height)
        shared = shared_view_fraction(left, right, candidate)
        name = f"cam{i}-cam{j}"
        if shared < options.min_shared_view:
            results[name] = {"shared_view": round(shared, 3), "skipped": "no shared view"}
            print(name, "skipped", flush=True)
            continue
        provenance = StereoRotationProvenance(
            generator="tools/hilti_camera_pair_reproducibility.py",
            generator_version="0.1",
            git_commit=commit,
            left_topic=TOPICS[i],
            right_topic=TOPICS[j],
            left_frame=left_frame,
            right_frame=right_frame,
            intrinsics_source=", ".join(path.name for path in args.camchain),
        )
        splits: dict[str, dict[str, object]] = {}
        for split, indices in split_indices(len(groups)).items():
            pairs = [
                (groups[k][TOPICS[i]].astype(np.float64), groups[k][TOPICS[j]].astype(np.float64))
                for k in indices
            ]
            artifact = estimate_stereo_rotation(
                pairs, left, right, candidate, provenance=provenance, options=options
            )
            splits[split] = summary(artifact)
        results[name] = {
            "shared_view": round(shared, 3),
            "splits": splits,
            "differences": half_differences(splits),
        }
        print(name, json.dumps(results[name]["differences"]), flush=True)
    output = {
        "frames": len(groups),
        "spacing_s": args.spacing_s,
        "span_s": round(stamps[-1] - stamps[0], 1),
        "pairs": results,
        "provenance": {
            "git_commit": commit,
            "bag": {"name": args.bag.name, "size_bytes": args.bag.stat().st_size},
            "camchain_sha256": {path.name: sha256_path(path) for path in args.camchain},
            "elapsed_s": round(time.monotonic() - started, 1),
        },
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
