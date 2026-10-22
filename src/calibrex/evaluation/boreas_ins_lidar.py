"""Boreas adapter for the native INS--LiDAR hand-eye evaluation."""

from __future__ import annotations

import hashlib
from pathlib import Path

from calibrex import __version__
from calibrex.core.ins_lidar_hand_eye import InsLidarHandEyeArtifact, InsLidarHandEyeProvenance
from calibrex.core.progress import emit_stage
from calibrex.core.provenance import git_commit
from calibrex.data.boreas_ins_lidar import (
    boreas_scan_time_s,
    read_boreas_ins_trajectory,
    read_boreas_lidar_xyz,
    read_boreas_reference_extrinsic,
)
from calibrex.evaluation.ins_lidar_hand_eye import (
    InsLidarRunOptions,
    _odometry_summary,
    build_ins_lidar_artifact,
    build_sensor_motions,
    evaluate_ins_lidar_hand_eye,
)
from calibrex.solvers.scan_to_scan_odometry import odometry_from_loader


def run_boreas_ins_lidar_hand_eye(
    sequence: str | Path,
    options: InsLidarRunOptions | None = None,
    *,
    max_frames: int | None = None,
    command: list[str] | None = None,
) -> InsLidarHandEyeArtifact:
    """Register raw LiDAR scans and evaluate against the independent Applanix track.

    All numerical settings and the artifact schema are shared with the KITTI
    adapter. Sensor pose CSVs, INS-seeded scan registration, and reference
    extrinsics as priors are not used. Local input files must be verified by
    the acquisition receipt before this function is used for a frozen audit.
    """
    root = Path(sequence)
    opts = options or InsLidarRunOptions()
    if max_frames is not None and (type(max_frames) is not int or max_frames < 2):
        raise ValueError("max_frames must be at least two")
    paths = sorted((root / "lidar").glob("*.bin"))
    if max_frames is not None:
        paths = paths[:max_frames]
    if len(paths) < 2:
        raise ValueError("at least two Boreas LiDAR scans required")
    times = [boreas_scan_time_s(path) for path in paths]
    reference_path = root / "applanix/gps_post_process.csv"
    calibration_path = root / "calib/T_applanix_lidar.txt"
    reference = read_boreas_ins_trajectory(
        reference_path, start_s=times[0] - 1.0, end_s=times[-1] + 1.0
    )
    if not all(reference.covers(t) for t in (times[0], times[-1])):
        raise ValueError("Applanix reference does not cover the selected scan times")
    emit_stage(f"INS-LiDAR: registering {len(paths)} Boreas scans without an INS prior")
    odometry = odometry_from_loader(
        lambda index: read_boreas_lidar_xyz(paths[index]), len(paths), opts.odometry, times_s=times
    )
    motions = build_sensor_motions(
        times, odometry.poses, opts, id_prefix=root.name + "/",
        registration_valid=(
            [item.converged for item in odometry.registrations]
            if opts.exclude_failed_registrations else None
        ),
    )
    # evaluate_ins_lidar_hand_eye uses this reference transform only when
    # constructing the post-fit per-axis comparison; the solver never sees it.
    vendor = read_boreas_reference_extrinsic(calibration_path)
    evaluation = evaluate_ins_lidar_hand_eye(reference, motions, opts, reference_transform=vendor)
    digest = hashlib.sha256()
    for path in [reference_path, calibration_path, *paths]:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    return build_ins_lidar_artifact(
        evaluation,
        opts,
        odometry=_odometry_summary(odometry.registrations, len(paths), []),
        provenance=InsLidarHandEyeProvenance(
            generator=__name__,
            generator_version=__version__,
            git_commit=git_commit(),
            command=command or [],
            dataset_family="boreas",
            drive_ids=[root.name],
            input_sha256=digest.hexdigest(),
            dataset_license="CC BY 4.0",
        ),
        reference_calibration="Boreas T_applanix_lidar.txt (dataset-provided reference estimate)",
        limitations=(
            "Rigid-scan registration: published per-point time offsets are not used for deskewing.",
            "The Applanix post-processed trajectory is a reference estimate, "
            "not independent metrology.",
            "The reference extrinsic is comparison-only; no translation or full-extrinsic "
            "claim follows from a partial result.",
        ),
    )
