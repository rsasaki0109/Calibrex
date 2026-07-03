#!/usr/bin/env python3
"""Generate README fixed 3D LiDAR-to-LiDAR calibration evidence GIFs.

The default visual uses real public Livox Horizon-Horizon PCD frames from the
official Livox automatic calibration example. It does not redistribute the
upstream archives and does not claim benchmark accuracy. The animation is an
evidence-viewer proxy for how Calibrex compares a fixed 3D LiDAR-to-LiDAR
candidate against a selected reference. Alternate sources also use public raw
data; README-facing assets must not use deterministic fallback geometry.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import shutil
import struct
import subprocess
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.error import URLError
from urllib.request import Request, urlopen

WIDTH = 960
HEIGHT = 540
FPS = 12
FRAME_COUNT = 36
README_GIF_MANIFEST = Path("docs/assets/readme-gif-gallery.json")
README_GIF_MANIFEST_SCHEMA_VERSION = "calibrex.readme_gif_gallery/v0.3"
ONLINE_PIPELINE_SOURCE = "calibrex calibrate --online"
ONLINE_PIPELINE_MODE = "real_online"
ONLINE_TIMELINE_SCHEMA_VERSION = "calibrex.online_timeline/v0.2"
ONLINE_BATCH_SIZE = 400
ONLINE_HOLDOUT_RATIO = 0.2
ONLINE_ROLLING_WINDOW = 2000
ONLINE_RESIDUAL_CHART_MAX_M = 0.45
ONLINE_RESIDUAL_CHART_MIN_M = 0.020
# A2D2 VLP-16 sparse clouds converge near ~0.29 m holdout RMSE on the front pair
# (see tests/integration/test_native_point_to_plane_pipeline.py). Library defaults
# stay at 0.05 m; the README online GIF uses an explicit, manifest-recorded gate.
ONLINE_GIF_MAX_HOLDOUT_RMSE_M = 0.40
ONLINE_GIF_MAX_ROLLING_REGRESSION_M = 0.15

A2D2_SENSOR_CONFIG_URL = (
    "https://aev-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/"
    "cams_lidars.json"
)
A2D2_LIDAR_SAMPLE_URL = (
    "https://aev-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/"
    "camera_lidar-20180810150607_lidar_frontleft.tar"
)
A2D2_LIDAR_SAMPLE_NAME = "20180810150607_lidar_front_left_000000060.npz"
A2D2_LIDAR_SAMPLE_START = 1536
A2D2_LIDAR_SAMPLE_SIZE = 2_977_425
A2D2_LIDAR_ID_TO_NAME = {
    0: "front_center",
    1: "front_left",
    2: "front_right",
    3: "rear_left",
    4: "rear_right",
}
LIVOX_BASE_PCD_URL = (
    "https://terra-1-g.djicdn.com/65c028cd298f4669a7f0e40e50ba1131/"
    "Showcase/Base_LiDAR_Frames.tar.gz"
)
LIVOX_TARGET_PCD_URL = (
    "https://terra-1-g.djicdn.com/65c028cd298f4669a7f0e40e50ba1131/"
    "Showcase/Target-LiDAR-Frames.tar.gz"
)
LIVOX_BASE_SAMPLE_NAME = "base_horizon_100432.pcd"
LIVOX_TARGET_SAMPLE_NAME = "target_horizon_100538.pcd"

SCENE_PANEL = (28, 86, 594, 426)
RIGHT_PANEL = (650, 92, 278, 414)
CHART = (676, 294, 216, 48)

Color = tuple[int, int, int]
Point3 = tuple[float, float, float]
Point2 = tuple[int, int]
VisualMode = Literal["evidence", "online"]

BG = (10, 15, 27)
PANEL = (18, 27, 43)
PANEL_ALT = (13, 20, 34)
GRID = (55, 68, 90)
TEXT_DIM = (148, 163, 184)
REFERENCE = (52, 211, 153)
CANDIDATE = (251, 113, 133)
OPTIMIZED = (34, 211, 238)
WARNING = (245, 158, 11)
GOOD = (34, 197, 94)
VEHICLE = (42, 52, 68)
ROAD = (27, 38, 55)


@dataclass(frozen=True)
class LidarPose:
    name: str
    origin: Point3


@dataclass(frozen=True)
class LidarCloudPair:
    source_points: list[Point3]
    target_points: list[Point3]
    source_label: str
    target_label: str
    bar_labels: tuple[str, str, str, str]
    bar_values: tuple[float, float, float, float]
    source_pose_name: str
    target_pose_name: str
    source_total: int
    target_total: int
    source_path: Path
    subtitle: str
    scene_caption: str
    legend: str
    provenance: str
    shared_voxel_count: int
    source_recall: float
    shared_centroid_rmse_m: float
    holdout_plane_match_count: int
    holdout_point_to_plane_p90_m: float
    holdout_unmatched_fraction: float
    known_bad_detectable_fraction: float
    known_bad_max_rmse_delta_m: float
    known_bad_max_point_to_plane_p90_delta_m: float
    support_summary: str
    holdout_summary: str
    known_bad_summary: str
    protocol_summary: str
    case_summary: str


@dataclass(frozen=True)
class OnlineGifFrameState:
    """One rendered frame derived from a real online calibration timeline."""

    batch_index: int
    progress: float
    cycle: float
    residual_history: list[float]
    gate_statuses: tuple[str, ...]
    batch_point_counts: tuple[int, ...]
    batch_accepted: tuple[bool, ...]
    current_holdout_rmse_m: float | None
    current_rolling_rmse_m: float | None
    visual_offset: Point3
    convergence_ratio: float
    final_gate_status: str
    accepted_batch_count: int


@dataclass(frozen=True)
class OnlineGifGateThresholds:
    """Explicit online gate settings used for the README online GIF run."""

    min_rank: int
    max_holdout_rmse_m: float
    max_rolling_regression_m: float


@dataclass(frozen=True)
class OnlineGifRun:
    """Real online pipeline output used to render the README online GIF."""

    timeline_path: Path
    frame_states: tuple[OnlineGifFrameState, ...]
    batch_count: int
    accepted_batch_count: int
    rejected_batch_count: int
    inconclusive_batch_count: int
    final_gate_status: str
    gate_thresholds: OnlineGifGateThresholds


@dataclass(frozen=True)
class ReadmeGifJob:
    source: str
    output: Path
    visual: VisualMode = "evidence"
    a2d2_source_id: int = 0
    a2d2_target_id: int = 1


README_GIF_JOBS = (
    ReadmeGifJob(
        source="a2d2",
        output=Path("docs/assets/online-calibration-loop.gif"),
        visual="online",
        a2d2_source_id=0,
        a2d2_target_id=1,
    ),
    ReadmeGifJob(
        source="livox-horizon-horizon",
        output=Path("docs/assets/calibration-evidence-demo.gif"),
    ),
    ReadmeGifJob(
        source="a2d2",
        output=Path("docs/assets/a2d2-multilidar-evidence-demo.gif"),
        a2d2_source_id=0,
        a2d2_target_id=1,
    ),
    ReadmeGifJob(
        source="a2d2",
        output=Path("docs/assets/a2d2-front-rear-evidence-demo.gif"),
        a2d2_source_id=1,
        a2d2_target_id=3,
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["livox-horizon-horizon", "a2d2"],
        default="livox-horizon-horizon",
        help="Public data source used to generate the evidence animation.",
    )
    parser.add_argument(
        "--sensor-config",
        type=Path,
        help="Optional local A2D2 cams_lidars.json. If omitted, the public URL is tried.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Directory for the extracted public data sample.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/calibration-evidence-demo.gif"),
    )
    parser.add_argument(
        "--visual",
        choices=["evidence", "online"],
        default="evidence",
        help="Animation layout to render for a single-source GIF.",
    )
    parser.add_argument(
        "--readme-gallery",
        action="store_true",
        help="Generate every public-data GIF referenced by the README.",
    )
    parser.add_argument("--frames", type=int, default=FRAME_COUNT)
    parser.add_argument(
        "--a2d2-source-id",
        type=int,
        default=0,
        choices=sorted(A2D2_LIDAR_ID_TO_NAME),
        help="A2D2 lidar_id used as the source point set.",
    )
    parser.add_argument(
        "--a2d2-target-id",
        type=int,
        default=1,
        choices=sorted(A2D2_LIDAR_ID_TO_NAME),
        help="A2D2 lidar_id used as the target point set.",
    )
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="Require already downloaded public inputs instead of fetching them.",
    )
    parser.add_argument(
        "--allow-metadata-fallback",
        action="store_true",
        help="Allow built-in fixed-rig metadata if public A2D2 metadata is unavailable.",
    )
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required to generate the GIF")

    if args.readme_gallery:
        if args.data_dir is not None:
            raise SystemExit("--data-dir is only supported for single-source GIF generation")
        generate_readme_gallery(
            frames=args.frames,
            sensor_config=args.sensor_config,
            allow_network=not args.no_network,
            allow_fallback=args.allow_metadata_fallback,
        )
        return 0

    cloud_pair, lidars, metadata_source = load_gif_inputs(
        source=args.source,
        data_dir=args.data_dir,
        sensor_config=args.sensor_config,
        allow_network=not args.no_network,
        allow_fallback=args.allow_metadata_fallback,
        a2d2_source_id=args.a2d2_source_id,
        a2d2_target_id=args.a2d2_target_id,
    )
    generate_gif(
        output=args.output,
        frames=args.frames,
        cloud_pair=cloud_pair,
        lidars=lidars,
        metadata_source=metadata_source,
        visual=args.visual,
        a2d2_source_id=args.a2d2_source_id,
        a2d2_target_id=args.a2d2_target_id,
        data_dir=args.data_dir,
        allow_network=not args.no_network,
    )
    matching_job = next((job for job in README_GIF_JOBS if job.output == args.output), None)
    if matching_job is not None:
        patch_readme_gallery_manifest(
            job=matching_job,
            cloud_pair=cloud_pair,
            metadata_source=metadata_source,
            frames=args.frames,
            allow_fallback=args.allow_metadata_fallback,
        )
    return 0


def generate_readme_gallery(
    *,
    frames: int,
    sensor_config: Path | None,
    allow_network: bool,
    allow_fallback: bool,
) -> None:
    """Generate every README GIF from its public data source."""

    manifest_assets: list[dict[str, object]] = []
    for job in README_GIF_JOBS:
        cloud_pair, lidars, metadata_source = load_gif_inputs(
            source=job.source,
            data_dir=None,
            sensor_config=sensor_config,
            allow_network=allow_network,
            allow_fallback=allow_fallback,
            a2d2_source_id=job.a2d2_source_id,
            a2d2_target_id=job.a2d2_target_id,
        )
        generate_gif(
            output=job.output,
            frames=frames,
            cloud_pair=cloud_pair,
            lidars=lidars,
            metadata_source=metadata_source,
            visual=job.visual,
            a2d2_source_id=job.a2d2_source_id,
            a2d2_target_id=job.a2d2_target_id,
            data_dir=None,
            allow_network=allow_network,
        )
        manifest_assets.append(
            readme_gallery_manifest_asset(
                job=job,
                cloud_pair=cloud_pair,
                metadata_source=metadata_source,
                online_run=load_online_run_for_manifest(job),
            )
        )
    write_readme_gallery_manifest(
        README_GIF_MANIFEST,
        frames=frames,
        allow_fallback=allow_fallback,
        assets=manifest_assets,
    )


def readme_gallery_manifest_asset(
    *,
    job: ReadmeGifJob,
    cloud_pair: LidarCloudPair,
    metadata_source: str,
    online_run: OnlineGifRun | None = None,
) -> dict[str, object]:
    """Return a stable provenance manifest entry for one README GIF."""

    if job.source == "livox-horizon-horizon":
        public_inputs: list[dict[str, object]] = [
            {
                "kind": "pcd_tar_gz",
                "url": LIVOX_BASE_PCD_URL,
                "selected_member": LIVOX_BASE_SAMPLE_NAME,
            },
            {
                "kind": "pcd_tar_gz",
                "url": LIVOX_TARGET_PCD_URL,
                "selected_member": LIVOX_TARGET_SAMPLE_NAME,
            },
        ]
        sensor_pair = {
            "source": "base_horizon",
            "target": "target_horizon",
        }
    elif job.source == "a2d2":
        public_inputs = [
            {
                "kind": "npz_range_from_tar",
                "url": A2D2_LIDAR_SAMPLE_URL,
                "selected_member": A2D2_LIDAR_SAMPLE_NAME,
                "byte_range": [
                    A2D2_LIDAR_SAMPLE_START,
                    A2D2_LIDAR_SAMPLE_START + A2D2_LIDAR_SAMPLE_SIZE - 1,
                ],
            },
            {
                "kind": "sensor_metadata_json",
                "url": A2D2_SENSOR_CONFIG_URL,
            },
        ]
        sensor_pair = {
            "source_lidar_id": job.a2d2_source_id,
            "source": A2D2_LIDAR_ID_TO_NAME[job.a2d2_source_id],
            "target_lidar_id": job.a2d2_target_id,
            "target": A2D2_LIDAR_ID_TO_NAME[job.a2d2_target_id],
        }
    else:
        raise SystemExit(f"unsupported README GIF source: {job.source}")

    asset: dict[str, object] = {
        "output": str(job.output),
        "sha256": sha256_file(job.output),
        "size_bytes": job.output.stat().st_size,
        "source": job.source,
        "visual": job.visual,
        "sensor_pair": sensor_pair,
        "source_label": cloud_pair.source_label,
        "target_label": cloud_pair.target_label,
        "public_inputs": public_inputs,
        "metadata_source": metadata_source,
        "uses_builtin_metadata_fallback": metadata_source == "fixed multi-LiDAR fallback",
    }
    if job.visual == "online":
        if online_run is None:
            raise SystemExit("online README GIF manifest requires a real online pipeline run")
        asset["pipeline"] = {
            "mode": ONLINE_PIPELINE_MODE,
            "source": ONLINE_PIPELINE_SOURCE,
            "timeline_schema_version": ONLINE_TIMELINE_SCHEMA_VERSION,
        }
        asset["online_run"] = {
            "batch_count": online_run.batch_count,
            "accepted_batch_count": online_run.accepted_batch_count,
            "rejected_batch_count": online_run.rejected_batch_count,
            "inconclusive_batch_count": online_run.inconclusive_batch_count,
            "final_gate_status": online_run.final_gate_status,
            "gate_thresholds": {
                "min_rank": online_run.gate_thresholds.min_rank,
                "max_holdout_rmse_m": online_run.gate_thresholds.max_holdout_rmse_m,
                "max_rolling_regression_m": online_run.gate_thresholds.max_rolling_regression_m,
            },
        }
    return asset


def patch_readme_gallery_manifest(
    *,
    job: ReadmeGifJob,
    cloud_pair: LidarCloudPair,
    metadata_source: str,
    frames: int,
    allow_fallback: bool,
) -> None:
    """Update one README GIF entry inside the gallery manifest."""

    manifest_path = README_GIF_MANIFEST
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "schema_version": README_GIF_MANIFEST_SCHEMA_VERSION,
            "generator": "tools/generate_calibration_evidence_gif.py",
            "dimensions": {
                "width": WIDTH,
                "height": HEIGHT,
                "fps": FPS,
                "frames": frames,
            },
            "fallback_metadata_allowed": allow_fallback,
            "assets": [],
        }

    online_run = load_online_run_for_manifest(job) if job.visual == "online" else None
    updated_asset = readme_gallery_manifest_asset(
        job=job,
        cloud_pair=cloud_pair,
        metadata_source=metadata_source,
        online_run=online_run,
    )
    assets = [asset for asset in manifest.get("assets", []) if asset["output"] != str(job.output)]
    assets.append(updated_asset)
    assets.sort(key=lambda asset: str(asset["output"]))
    manifest["schema_version"] = README_GIF_MANIFEST_SCHEMA_VERSION
    manifest["generator"] = "tools/generate_calibration_evidence_gif.py"
    manifest["dimensions"] = {
        "width": WIDTH,
        "height": HEIGHT,
        "fps": FPS,
        "frames": frames,
    }
    manifest["fallback_metadata_allowed"] = allow_fallback
    manifest["assets"] = assets
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_online_run_for_manifest(job: ReadmeGifJob) -> OnlineGifRun | None:
    """Return the cached online run metadata written during GIF generation."""

    if job.visual != "online":
        return None
    cache_path = job.output.with_suffix(".online-run.json")
    if not cache_path.exists():
        return None
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    gate_payload = payload["gate_thresholds"]
    return OnlineGifRun(
        timeline_path=Path(payload["timeline_path"]),
        frame_states=(),
        batch_count=int(payload["batch_count"]),
        accepted_batch_count=int(payload["accepted_batch_count"]),
        rejected_batch_count=int(payload["rejected_batch_count"]),
        inconclusive_batch_count=int(payload["inconclusive_batch_count"]),
        final_gate_status=str(payload["final_gate_status"]),
        gate_thresholds=OnlineGifGateThresholds(
            min_rank=int(gate_payload["min_rank"]),
            max_holdout_rmse_m=float(gate_payload["max_holdout_rmse_m"]),
            max_rolling_regression_m=float(gate_payload["max_rolling_regression_m"]),
        ),
    )


def write_readme_gallery_manifest(
    path: Path,
    *,
    frames: int,
    allow_fallback: bool,
    assets: list[dict[str, object]],
) -> None:
    """Write a machine-readable manifest for README GIF assets."""

    payload = {
        "schema_version": README_GIF_MANIFEST_SCHEMA_VERSION,
        "generator": "tools/generate_calibration_evidence_gif.py",
        "dimensions": {
            "width": WIDTH,
            "height": HEIGHT,
            "fps": FPS,
            "frames": frames,
        },
        "fallback_metadata_allowed": allow_fallback,
        "assets": assets,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a local asset."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_gif_inputs(
    *,
    source: str,
    data_dir: Path | None,
    sensor_config: Path | None,
    allow_network: bool,
    allow_fallback: bool,
    a2d2_source_id: int,
    a2d2_target_id: int,
) -> tuple[LidarCloudPair, list[LidarPose], str]:
    """Load the public data and rig metadata for one GIF."""

    if source == "livox-horizon-horizon":
        resolved_data_dir = data_dir or Path("data/public/livox_horizon_horizon_pair")
        ensure_livox_horizon_pair(resolved_data_dir, allow_network=allow_network)
        return (
            load_livox_horizon_cloud_pair(resolved_data_dir),
            livox_horizon_setup(),
            "Livox official PCD sample",
        )

    if source == "a2d2":
        resolved_data_dir = data_dir or Path("data/public/a2d2_lidar_pair")
        ensure_a2d2_lidar_sample(resolved_data_dir, allow_network=allow_network)
        local_sensor_config = resolved_data_dir / "cams_lidars.json"
        resolved_sensor_config = sensor_config
        if resolved_sensor_config is None and local_sensor_config.exists():
            resolved_sensor_config = local_sensor_config
        cloud_pair = load_a2d2_lidar_cloud_pair(
            resolved_data_dir / A2D2_LIDAR_SAMPLE_NAME,
            source_lidar_id=a2d2_source_id,
            target_lidar_id=a2d2_target_id,
        )
        lidars, metadata_source = load_a2d2_lidar_setup(
            resolved_sensor_config,
            allow_network=allow_network,
            allow_fallback=allow_fallback,
        )
        if resolved_sensor_config == local_sensor_config:
            metadata_source = "A2D2 public cams_lidars.json"
        return cloud_pair, lidars, metadata_source

    raise SystemExit(f"unsupported GIF source: {source}")


def generate_gif(
    *,
    output: Path,
    frames: int,
    cloud_pair: LidarCloudPair,
    lidars: list[LidarPose],
    metadata_source: str,
    visual: VisualMode,
    a2d2_source_id: int = 0,
    a2d2_target_id: int = 1,
    data_dir: Path | None = None,
    allow_network: bool = True,
) -> None:
    """Render one evidence animation to a GIF."""

    output.parent.mkdir(parents=True, exist_ok=True)
    online_run: OnlineGifRun | None = None
    if visual == "online":
        sample_path = cloud_pair.source_path
        online_run = run_online_gif_pipeline(
            sample_path=sample_path,
            source_lidar_id=a2d2_source_id,
            target_lidar_id=a2d2_target_id,
            frames=frames,
            data_dir=data_dir,
            allow_network=allow_network,
        )
        write_online_run_cache(output, online_run)

    with tempfile.TemporaryDirectory(prefix="calibrex_lidar_lidar_gif_") as tmp_name:
        tmp = Path(tmp_name)
        residual_history: list[float] = []
        for index in range(frames):
            image = bytearray(bytes(BG) * (WIDTH * HEIGHT))
            if visual == "online":
                assert online_run is not None
                frame_state = online_run.frame_states[index]
                draw_online_calibration_frame(
                    image=image,
                    lidars=lidars,
                    cloud_pair=cloud_pair,
                    frame_state=frame_state,
                    metadata_source=metadata_source,
                )
            else:
                progress = smoothstep(index / max(1, frames - 1))
                residual_history.append(residual_proxy(progress))
                draw_frame(
                    image=image,
                    lidars=lidars,
                    cloud_pair=cloud_pair,
                    progress=progress,
                    residual_history=residual_history,
                    metadata_source=metadata_source,
                )
            write_ppm(tmp / f"frame_{index:03d}.ppm", image)
        encode_gif(
            tmp,
            output,
            cloud_pair,
            visual=visual,
            online_run=online_run,
        )


def write_online_run_cache(output: Path, online_run: OnlineGifRun) -> None:
    """Persist online-run summary for manifest provenance."""

    cache_path = output.with_suffix(".online-run.json")
    payload = {
        "timeline_path": str(online_run.timeline_path),
        "batch_count": online_run.batch_count,
        "accepted_batch_count": online_run.accepted_batch_count,
        "rejected_batch_count": online_run.rejected_batch_count,
        "inconclusive_batch_count": online_run.inconclusive_batch_count,
        "final_gate_status": online_run.final_gate_status,
        "gate_thresholds": {
            "min_rank": online_run.gate_thresholds.min_rank,
            "max_holdout_rmse_m": online_run.gate_thresholds.max_holdout_rmse_m,
            "max_rolling_regression_m": online_run.gate_thresholds.max_rolling_regression_m,
        },
    }
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_online_gif_pipeline(
    *,
    sample_path: Path,
    source_lidar_id: int,
    target_lidar_id: int,
    frames: int,
    data_dir: Path | None,
    allow_network: bool,
) -> OnlineGifRun:
    """Run `calibrex calibrate --online` and build per-frame GIF states."""

    from calibrex.core.io import read_mapping
    from calibrex.core.online_timeline import OnlineCalibrationTimelineArtifact
    from calibrex.pipelines.online import (
        OnlineCalibrationRunOptions,
        OnlineGateThresholds,
        run_online_calibration,
    )

    gate_thresholds = OnlineGateThresholds(
        max_holdout_rmse_m=ONLINE_GIF_MAX_HOLDOUT_RMSE_M,
        max_rolling_regression_m=ONLINE_GIF_MAX_ROLLING_REGRESSION_M,
    )
    gif_gate_thresholds = OnlineGifGateThresholds(
        min_rank=gate_thresholds.min_rank,
        max_holdout_rmse_m=gate_thresholds.max_holdout_rmse_m,
        max_rolling_regression_m=gate_thresholds.max_rolling_regression_m,
    )

    source_name = A2D2_LIDAR_ID_TO_NAME[source_lidar_id]
    target_name = A2D2_LIDAR_ID_TO_NAME[target_lidar_id]
    with tempfile.TemporaryDirectory(prefix="calibrex_online_gif_") as tmp_name:
        tmp = Path(tmp_name)
        dataset_dir = tmp / "a2d2_online_pair"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        source_npz = dataset_dir / f"source_{source_name}.npz"
        target_npz = dataset_dir / f"target_{target_name}.npz"
        write_a2d2_lidar_id_filtered_npz(sample_path, source_npz, lidar_id=source_lidar_id)
        write_a2d2_lidar_id_filtered_npz(sample_path, target_npz, lidar_id=target_lidar_id)

        output_dir = tmp / "outputs"
        config_path = write_a2d2_online_gif_config(
            config_path=tmp / "online_config.yaml",
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            source_name=source_name,
            target_name=target_name,
        )
        result = run_online_calibration(
            config_path,
            OnlineCalibrationRunOptions(
                output_dir=output_dir,
                batch_size=ONLINE_BATCH_SIZE,
                rolling_window=ONLINE_ROLLING_WINDOW,
                holdout_ratio=ONLINE_HOLDOUT_RATIO,
                gate_thresholds=gate_thresholds,
            ),
        )
        if result is None:
            raise SystemExit("online GIF generation requires a real online calibration run")

        timeline_path = Path(result.run.provenance["online_timeline_path"])
        timeline = OnlineCalibrationTimelineArtifact.model_validate(read_mapping(timeline_path))
        if not timeline.batches:
            raise SystemExit("online calibration produced no timeline batches for the GIF")

        initial_transform = _transform_result_to_se3(timeline.batches[0].estimate)
        final_transform = _accepted_transform_at_batch(timeline.batches, len(timeline.batches) - 1)
        frame_states = build_online_gif_frame_states(
            timeline=timeline,
            frames=frames,
            initial_transform=initial_transform,
            final_transform=final_transform,
        )
        return OnlineGifRun(
            timeline_path=timeline_path,
            frame_states=tuple(frame_states),
            batch_count=len(timeline.batches),
            accepted_batch_count=timeline.accepted_batch_count,
            rejected_batch_count=timeline.rejected_batch_count,
            inconclusive_batch_count=timeline.inconclusive_batch_count,
            final_gate_status=timeline.final_gate_status,
            gate_thresholds=gif_gate_thresholds,
        )


def write_a2d2_online_gif_config(
    *,
    config_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    source_name: str,
    target_name: str,
) -> Path:
    """Write a minimal online calibration config for one A2D2 sensor pair."""

    source_sensor = f"lidar_{source_name}"
    target_sensor = f"lidar_{target_name}"
    config_path.write_text(
        f"""
schema_version: calibrex.config/v0.1
project:
  name: readme_online_gif
  output_dir: {output_dir}
dataset:
  type: a2d2_lidar
  path: {dataset_dir}
sensors:
  {source_sensor}:
    type: lidar
  {target_sensor}:
    type: lidar
frames:
  base_link:
    root: true
  {source_sensor}:
    parent: base_link
    transform:
      estimate: false
  {target_sensor}:
    parent: base_link
    transform:
      estimate: true
      prior_sigma:
        translation_m: 0.2
        rotation_deg: 5.0
pipeline:
  type: multi_sensor_slac
  factors:
    lidar_rig_point_to_plane:
      enabled: true
      options:
        voxel_size_m: 0.75
        correspondence_gate_m: 1.0
        max_target_points: 3000
solver:
  backend: native_lidar_point_to_plane
  max_iterations: 40
""",
        encoding="utf-8",
    )
    return config_path


def write_a2d2_lidar_id_filtered_npz(
    source: Path,
    output: Path,
    *,
    lidar_id: int,
) -> None:
    """Write one A2D2 NPZ containing only points from a physical LiDAR id."""

    with zipfile.ZipFile(source) as archive:
        points_header, points_raw = read_npy_array(archive, "pcloud_points.npy")
        _ids_header, lidar_ids_raw = read_npy_array(archive, "pcloud_attr.lidar_id.npy")
        _valid_header, valid_raw = read_npy_array(archive, "pcloud_attr.valid.npy")
        attr_arrays: dict[str, tuple[dict[str, object], tuple[object, ...]]] = {
            "pcloud_attr.lidar_id.npy": (_ids_header, lidar_ids_raw),
            "pcloud_attr.valid.npy": (_valid_header, valid_raw),
        }
        for member in archive.namelist():
            if (
                member.startswith("pcloud_attr.")
                and member.endswith(".npy")
                and member not in attr_arrays
            ):
                attr_arrays[member] = read_npy_array(archive, member)

    shape = points_header["shape"]
    if not isinstance(shape, tuple) or len(shape) != 2 or shape[1] != 3:
        raise SystemExit(f"{source} pcloud_points must be Nx3")
    selected_indices = [
        index
        for index, (raw_lidar_id, valid) in enumerate(zip(lidar_ids_raw, valid_raw, strict=True))
        if int(raw_lidar_id) == lidar_id and bool(valid)
    ]
    if not selected_indices:
        raise SystemExit(f"{source} does not contain usable lidar_id {lidar_id} points")

    filtered_points: list[float] = []
    for index in selected_indices:
        filtered_points.extend(
            (
                float(points_raw[index * 3]),
                float(points_raw[index * 3 + 1]),
                float(points_raw[index * 3 + 2]),
            )
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        _write_npy_array(
            archive,
            "pcloud_points.npy",
            "<f8",
            (len(selected_indices), 3),
            filtered_points,
        )
        for name, (header, values) in attr_arrays.items():
            if name == "pcloud_attr.lidar_id.npy":
                filtered = [lidar_id] * len(selected_indices)
                _write_npy_array(archive, name, "<i8", (len(selected_indices),), filtered)
                continue
            if name == "pcloud_attr.valid.npy":
                filtered = [True] * len(selected_indices)
                _write_npy_array(archive, name, "|b1", (len(selected_indices),), filtered)
                continue
            filtered = [values[index] for index in selected_indices]
            dtype = str(header.get("descr"))
            header_shape = header.get("shape", ())
            if len(header_shape) == 1:
                filtered_shape = (len(selected_indices),)
            else:
                filtered_shape = (len(selected_indices), 1)
            _write_npy_array(archive, name, dtype, filtered_shape, filtered)


def _write_npy_array(
    archive: zipfile.ZipFile,
    name: str,
    descr: str,
    shape: tuple[int, ...],
    values: list[object],
) -> None:
    header = {
        "descr": descr,
        "fortran_order": False,
        "shape": shape,
    }
    header_bytes = (repr(header) + " " * 64).encode("latin1")
    padding = 16 - ((10 + len(header_bytes) + 1) % 16)
    header_bytes = header_bytes + b" " * padding + b"\n"
    if descr == "<f8":
        body = struct.pack("<" + "d" * len(values), *(float(value) for value in values))
    elif descr == "<i8":
        body = struct.pack("<" + "q" * len(values), *(int(value) for value in values))
    elif descr == "|b1":
        body = bytes(1 if bool(value) else 0 for value in values)
    else:
        raise SystemExit(f"unsupported dtype {descr} for {name}")
    archive.writestr(
        name,
        b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header_bytes)) + header_bytes + body,
    )


def build_online_gif_frame_states(
    *,
    timeline: object,
    frames: int,
    initial_transform: object,
    final_transform: object,
) -> list[OnlineGifFrameState]:
    """Map timeline batches onto the fixed README GIF frame count."""

    batches = timeline.batches
    final_gate_status = timeline.final_gate_status
    accepted_batch_count = timeline.accepted_batch_count
    frame_states: list[OnlineGifFrameState] = []
    residual_history: list[float] = []
    for frame_index in range(frames):
        progress = smoothstep(frame_index / max(1, frames - 1))
        cycle = frame_index / max(1, frames)
        batch_index = min(
            len(batches) - 1,
            round(progress * max(1, len(batches) - 1)),
        )
        batch = batches[batch_index]
        residual_value = _timeline_residual_value(batch)
        if residual_value is not None:
            residual_history.append(residual_value)
        elif residual_history:
            residual_history.append(residual_history[-1])
        else:
            residual_history.append(ONLINE_RESIDUAL_CHART_MIN_M)

        visible_batches = batches[: batch_index + 1]
        current_transform = _accepted_transform_at_batch(batches, batch_index)
        convergence_ratio = _transform_convergence_ratio(
            initial=initial_transform,
            current=current_transform,
            final=final_transform,
        )
        frame_states.append(
            OnlineGifFrameState(
                batch_index=batch_index,
                progress=progress,
                cycle=cycle,
                residual_history=list(residual_history),
                gate_statuses=tuple(batch.gate_status for batch in visible_batches),
                batch_point_counts=tuple(batch.point_count for batch in visible_batches),
                batch_accepted=tuple(batch.estimate_accepted for batch in visible_batches),
                current_holdout_rmse_m=batch.batch_holdout_rmse_m,
                current_rolling_rmse_m=batch.rolling_rmse_m,
                visual_offset=_estimate_visual_offset(
                    initial=initial_transform,
                    current=_transform_result_to_se3(batch.estimate),
                    remaining_ratio=1.0 - convergence_ratio,
                ),
                convergence_ratio=convergence_ratio,
                final_gate_status=final_gate_status,
                accepted_batch_count=accepted_batch_count,
            )
        )
    return frame_states


def _timeline_residual_value(batch: object) -> float | None:
    if batch.rolling_rmse_m is not None:
        return float(batch.rolling_rmse_m)
    if batch.batch_holdout_rmse_m is not None:
        return float(batch.batch_holdout_rmse_m)
    return None


def _transform_result_to_se3(transform_result: object) -> object:
    from calibrex.core.geometry import SE3

    return SE3.from_lists(
        transform_result.translation_m,
        transform_result.rotation_quat_xyzw,
    )


def _accepted_transform_at_batch(batches: list[object], batch_index: int) -> object:
    from calibrex.core.geometry import SE3

    accepted = SE3.identity()
    found = False
    for index in range(batch_index + 1):
        batch = batches[index]
        if batch.estimate_accepted:
            accepted = _transform_result_to_se3(batch.estimate)
            found = True
    if not found:
        return _transform_result_to_se3(batches[0].estimate)
    return accepted


def _transform_convergence_ratio(*, initial: object, current: object, final: object) -> float:
    initial_error = _translation_error_m(initial, final)
    current_error = _translation_error_m(current, final)
    if initial_error <= 1.0e-9:
        return 1.0
    return max(0.0, min(1.0, 1.0 - current_error / initial_error))


def _translation_error_m(left: object, right: object) -> float:
    delta = (
        left.translation_m[0] - right.translation_m[0],
        left.translation_m[1] - right.translation_m[1],
        left.translation_m[2] - right.translation_m[2],
    )
    return math.sqrt(delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2)


def _estimate_visual_offset(*, initial: object, current: object, remaining_ratio: float) -> Point3:
    delta = (
        current.translation_m[0] - initial.translation_m[0],
        current.translation_m[1] - initial.translation_m[1],
        current.translation_m[2] - initial.translation_m[2],
    )
    scale = 8.0
    return (
        delta[0] * scale * remaining_ratio,
        delta[1] * scale * remaining_ratio,
        delta[2] * scale * remaining_ratio,
    )


def ensure_livox_horizon_pair(data_dir: Path, *, allow_network: bool) -> None:
    """Ensure two real Livox Horizon PCD frames are present locally."""

    base_path = data_dir / LIVOX_BASE_SAMPLE_NAME
    target_path = data_dir / LIVOX_TARGET_SAMPLE_NAME
    if base_path.exists() and target_path.exists():
        return
    if not allow_network:
        raise SystemExit(
            f"{base_path} and {target_path} are required. Run without --no-network "
            "or fetch the Livox Horizon-Horizon sample first."
        )
    data_dir.mkdir(parents=True, exist_ok=True)
    extract_first_pcd_from_tar_gz(LIVOX_BASE_PCD_URL, base_path)
    extract_first_pcd_from_tar_gz(LIVOX_TARGET_PCD_URL, target_path)


def extract_first_pcd_from_tar_gz(url: str, output: Path) -> None:
    """Stream a public tar.gz and write its first PCD member."""

    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=90) as response, tarfile.open(
        fileobj=response,
        mode="r|gz",
    ) as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".pcd"):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            output.write_bytes(handle.read())
            return
    raise SystemExit(f"{url} did not contain a PCD file")


def ensure_a2d2_lidar_sample(data_dir: Path, *, allow_network: bool) -> None:
    """Ensure a small real A2D2 LiDAR NPZ sample is present locally."""

    sample_path = data_dir / A2D2_LIDAR_SAMPLE_NAME
    if sample_path.exists() and sample_path.stat().st_size == A2D2_LIDAR_SAMPLE_SIZE:
        return
    if not allow_network:
        raise SystemExit(
            f"{sample_path} is required. Run without --no-network or fetch the A2D2 "
            "sample with tools/download_public_dataset.py a2d2_lidar_pair_sample."
        )
    data_dir.mkdir(parents=True, exist_ok=True)
    start = A2D2_LIDAR_SAMPLE_START
    end = start + A2D2_LIDAR_SAMPLE_SIZE - 1
    request = Request(
        A2D2_LIDAR_SAMPLE_URL,
        headers={"Range": f"bytes={start}-{end}", "User-Agent": "Mozilla/5.0"},
    )
    with urlopen(request, timeout=90) as response:
        data = response.read()
    if len(data) != A2D2_LIDAR_SAMPLE_SIZE:
        raise SystemExit(
            f"downloaded {len(data)} bytes from A2D2, expected {A2D2_LIDAR_SAMPLE_SIZE}"
        )
    sample_path.write_bytes(data)


def load_livox_horizon_cloud_pair(data_dir: Path) -> LidarCloudPair:
    """Load two solid-state Livox Horizon PCD frames from public example data."""

    base_path = data_dir / LIVOX_BASE_SAMPLE_NAME
    target_path = data_dir / LIVOX_TARGET_SAMPLE_NAME
    source_points, source_total = read_binary_pcd_xyz(base_path, stride=8)
    target_points, target_total = read_binary_pcd_xyz(target_path, stride=7)
    if not source_points or not target_points:
        raise SystemExit(f"{data_dir} does not contain usable Livox Horizon PCD points")
    return LidarCloudPair(
        source_points=source_points[:2400],
        target_points=target_points[:2400],
        source_label="base Horizon points",
        target_label="target Horizon points",
        bar_labels=(
            "source voxel recall",
            "P90 point-to-plane",
            "known-bad controls",
            "evidence protocol",
        ),
        bar_values=(
            0.24594992636229748 / 0.40,
            1.0 - min(1.0, 0.8064419329166412 / 2.0),
            1.0,
            1.0,
        ),
        source_pose_name="base_horizon",
        target_pose_name="target_horizon",
        source_total=source_total,
        target_total=target_total,
        source_path=base_path,
        subtitle="Livox official Horizon-Horizon public PCD sample",
        scene_caption="Real solid-state Livox points: candidate transform vs known-bad controls",
        legend="green/cyan: candidate-aligned returns   pink: known-bad perturbation",
        provenance="provenance: Livox public PCD",
        shared_voxel_count=167,
        source_recall=0.24594992636229748,
        shared_centroid_rmse_m=0.5357028293135087,
        holdout_plane_match_count=16_884,
        holdout_point_to_plane_p90_m=0.8064419329166412,
        holdout_unmatched_fraction=0.2865714527169779,
        known_bad_detectable_fraction=1.0,
        known_bad_max_rmse_delta_m=0.019921626765770584,
        known_bad_max_point_to_plane_p90_delta_m=0.03373608924315874,
        support_summary="167 shared 1 m voxels / 0.246 source recall",
        holdout_summary="16,884 plane matches / P90 |p2plane| 0.806 m",
        known_bad_summary="24 known-bad controls / 1.00 detected",
        protocol_summary="protocol: single-pair holdout",
        case_summary="pitch +1deg -> P90 point-to-plane +0.034 m",
    )


def read_binary_pcd_xyz(path: Path, *, stride: int) -> tuple[list[Point3], int]:
    """Read x/y/z from a binary float32 PCD file with no third-party deps."""

    data = path.read_bytes()
    header_end = data.index(b"\n", data.index(b"DATA binary")) + 1
    header = data[:header_end].decode("ascii", errors="strict")
    fields: list[str] = []
    sizes: list[int] = []
    types: list[str] = []
    counts: list[int] = []
    point_count = 0
    for line_text in header.splitlines():
        parts = line_text.split()
        if not parts:
            continue
        key = parts[0]
        if key == "FIELDS":
            fields = parts[1:]
        elif key == "SIZE":
            sizes = [int(value) for value in parts[1:]]
        elif key == "TYPE":
            types = parts[1:]
        elif key == "COUNT":
            counts = [int(value) for value in parts[1:]]
        elif key == "POINTS":
            point_count = int(parts[1])
    if not fields or not sizes or not types:
        raise SystemExit(f"{path} has incomplete PCD metadata")
    if not counts:
        counts = [1] * len(fields)
    if fields[:3] != ["x", "y", "z"]:
        raise SystemExit(f"{path} must store x/y/z as the first fields")
    if sizes[:3] != [4, 4, 4] or types[:3] != ["F", "F", "F"]:
        raise SystemExit(f"{path} must store x/y/z as float32 fields")
    point_step = sum(size * count for size, count in zip(sizes, counts, strict=True))
    if point_step <= 0:
        raise SystemExit(f"{path} has invalid PCD point step")
    available = (len(data) - header_end) // point_step
    point_count = min(point_count or available, available)
    points: list[Point3] = []
    total = 0
    for index in range(point_count):
        x, y, z = struct.unpack_from("<fff", data, header_end + index * point_step)
        if not (1.0 <= x <= 62.0 and -22.0 <= y <= 22.0 and -10.0 <= z <= 6.0):
            continue
        total += 1
        if total % stride == 0:
            points.append((float(x), float(y), float(z)))
    return points, total


def load_a2d2_lidar_cloud_pair(
    path: Path,
    *,
    source_lidar_id: int,
    target_lidar_id: int,
) -> LidarCloudPair:
    """Load two physical LiDAR point sets from a real A2D2 NPZ file."""

    if source_lidar_id == target_lidar_id:
        raise SystemExit("A2D2 source and target lidar_id must differ")

    with zipfile.ZipFile(path) as archive:
        points_header, points_raw = read_npy_array(archive, "pcloud_points.npy")
        _ids_header, lidar_ids_raw = read_npy_array(archive, "pcloud_attr.lidar_id.npy")
        _valid_header, valid_raw = read_npy_array(archive, "pcloud_attr.valid.npy")

    shape = points_header["shape"]
    if shape[1] != 3:
        raise SystemExit(f"{path} pcloud_points must be Nx3")

    source_points: list[Point3] = []
    target_points: list[Point3] = []
    source_total = 0
    target_total = 0
    for index, (lidar_id, valid) in enumerate(zip(lidar_ids_raw, valid_raw, strict=True)):
        if not valid:
            continue
        x = float(points_raw[index * 3])
        y = float(points_raw[index * 3 + 1])
        z = float(points_raw[index * 3 + 2])
        if not (2.0 <= x <= 55.0 and -22.0 <= y <= 22.0 and -3.2 <= z <= 7.5):
            continue
        point = (x, y, z)
        if lidar_id == source_lidar_id:
            source_total += 1
            if source_total % 5 == 0:
                source_points.append(point)
        elif lidar_id == target_lidar_id:
            target_total += 1
            if target_total % 4 == 0:
                target_points.append(point)

    if not source_points or not target_points:
        raise SystemExit(
            f"{path} does not contain usable lidar_id "
            f"{source_lidar_id}/{target_lidar_id} point sets"
        )

    shared_voxels, source_recall, centroid_rmse = voxel_support_metrics(
        source_points,
        target_points,
        voxel_size_m=2.0,
    )

    source_name = A2D2_LIDAR_ID_TO_NAME[source_lidar_id]
    target_name = A2D2_LIDAR_ID_TO_NAME[target_lidar_id]

    return LidarCloudPair(
        source_points=source_points[:1800],
        target_points=target_points[:1800],
        source_label=f"lidar_id {source_lidar_id} ({source_name}) points",
        target_label=f"lidar_id {target_lidar_id} ({target_name}) points",
        bar_labels=(
            "2 m voxel recall",
            "real point sample",
            "public metadata",
            "evidence protocol",
        ),
        bar_values=(
            min(1.0, source_recall / 0.20),
            1.0,
            1.0,
            1.0,
        ),
        source_pose_name=source_name,
        target_pose_name=target_name,
        source_total=source_total,
        target_total=target_total,
        source_path=path,
        subtitle=(
            f"A2D2 public NPZ point cloud: lidar_id "
            f"{source_lidar_id} -> {target_lidar_id}"
        ),
        scene_caption=f"Real A2D2 returns: {source_name} to {target_name}",
        legend="green: selected real returns   cyan/pink: candidate preview",
        provenance="provenance: A2D2 real NPZ range sample",
        shared_voxel_count=shared_voxels,
        source_recall=source_recall,
        shared_centroid_rmse_m=centroid_rmse,
        holdout_plane_match_count=0,
        holdout_point_to_plane_p90_m=0.0,
        holdout_unmatched_fraction=0.0,
        known_bad_detectable_fraction=0.0,
        known_bad_max_rmse_delta_m=0.0,
        known_bad_max_point_to_plane_p90_delta_m=0.0,
        support_summary=f"{shared_voxels} shared 2 m voxels / {source_recall:.3f} recall",
        holdout_summary=f"{source_total + target_total:,} valid public returns",
        known_bad_summary="fixed-rig preview / no toy geometry",
        protocol_summary="public metadata + protocol",
        case_summary=f"shared centroid RMSE {centroid_rmse:.2f} m",
    )


def voxel_support_metrics(
    source_points: list[Point3],
    target_points: list[Point3],
    *,
    voxel_size_m: float,
) -> tuple[int, float, float]:
    """Return shared voxel count, source recall, and shared centroid RMSE."""

    source_voxels = voxel_centroids(source_points, voxel_size_m=voxel_size_m)
    target_voxels = voxel_centroids(target_points, voxel_size_m=voxel_size_m)
    shared_keys = sorted(set(source_voxels).intersection(target_voxels))
    source_recall = len(shared_keys) / max(1, len(source_voxels))
    if not shared_keys:
        return 0, source_recall, 0.0
    squared_errors = []
    for key in shared_keys:
        source = source_voxels[key]
        target = target_voxels[key]
        squared_errors.append(
            (source[0] - target[0]) ** 2
            + (source[1] - target[1]) ** 2
            + (source[2] - target[2]) ** 2
        )
    rmse = math.sqrt(sum(squared_errors) / len(squared_errors))
    return len(shared_keys), source_recall, rmse


def voxel_centroids(
    points: list[Point3],
    *,
    voxel_size_m: float,
) -> dict[tuple[int, int, int], Point3]:
    """Group points into voxels and return one centroid per occupied cell."""

    accumulators: dict[tuple[int, int, int], list[float]] = {}
    for x, y, z in points:
        key = (
            math.floor(x / voxel_size_m),
            math.floor(y / voxel_size_m),
            math.floor(z / voxel_size_m),
        )
        bucket = accumulators.setdefault(key, [0.0, 0.0, 0.0, 0.0])
        bucket[0] += x
        bucket[1] += y
        bucket[2] += z
        bucket[3] += 1.0
    return {
        key: (value[0] / value[3], value[1] / value[3], value[2] / value[3])
        for key, value in accumulators.items()
        if value[3]
    }


def read_npy_array(
    archive: zipfile.ZipFile,
    name: str,
) -> tuple[dict[str, object], tuple[object, ...]]:
    """Read simple C-order numeric NPY arrays from an NPZ archive."""

    data = archive.read(name)
    if data[:6] != b"\x93NUMPY":
        raise SystemExit(f"{name} is not an NPY array")
    offset = 6
    version = data[offset : offset + 2]
    offset += 2
    if version == b"\x01\x00":
        header_length = struct.unpack("<H", data[offset : offset + 2])[0]
        offset += 2
    else:
        header_length = struct.unpack("<I", data[offset : offset + 4])[0]
        offset += 4
    header = ast.literal_eval(data[offset : offset + header_length].decode("latin1").strip())
    offset += header_length
    if header.get("fortran_order") is not False:
        raise SystemExit(f"{name} uses unsupported Fortran order")
    shape = header.get("shape")
    if not isinstance(shape, tuple):
        raise SystemExit(f"{name} has invalid shape metadata")
    count = 1
    for dimension in shape:
        count *= int(dimension)
    dtype = header.get("descr")
    type_map = {"<f8": "d", "<i8": "q", "|b1": "?"}
    if dtype not in type_map:
        raise SystemExit(f"{name} uses unsupported dtype {dtype}")
    values = struct.unpack_from("<" + type_map[dtype] * count, data, offset)
    return header, values


def load_a2d2_lidar_setup(
    sensor_config: Path | None,
    *,
    allow_network: bool,
    allow_fallback: bool,
) -> tuple[list[LidarPose], str]:
    """Load A2D2 LiDAR origins from public metadata."""

    if sensor_config is not None:
        payload = json.loads(sensor_config.read_text(encoding="utf-8"))
        lidars = parse_lidars(payload)
        if lidars:
            return lidars, f"A2D2 metadata: {sensor_config}"
        if not allow_fallback:
            raise SystemExit(f"{sensor_config} does not contain usable A2D2 LiDAR origins")

    if allow_network:
        try:
            with urlopen(A2D2_SENSOR_CONFIG_URL, timeout=12) as response:
                payload = json.loads(response.read().decode("utf-8"))
            lidars = parse_lidars(payload)
            if lidars:
                return lidars, "A2D2 public cams_lidars.json"
        except (OSError, URLError, json.JSONDecodeError):
            pass

    if allow_fallback:
        return fallback_lidars(), "fixed multi-LiDAR fallback"

    raise SystemExit(
        "A2D2 public LiDAR metadata is required. Run with network access, pass "
        "--sensor-config, or explicitly opt into --allow-metadata-fallback for "
        "non-README local previews."
    )


def livox_horizon_setup() -> list[LidarPose]:
    """Return a compact fixed rig for the Livox Horizon-Horizon example."""

    return [
        LidarPose("base_horizon", (1.58, -0.18, 1.18)),
        LidarPose("target_horizon", (1.66, 0.32, 1.12)),
    ]


def parse_lidars(payload: dict[str, object]) -> list[LidarPose]:
    raw_lidars = payload.get("lidars")
    if not isinstance(raw_lidars, dict):
        return []

    lidars: list[LidarPose] = []
    for name, raw_lidar in raw_lidars.items():
        if not isinstance(raw_lidar, dict):
            continue
        raw_view = raw_lidar.get("view")
        if not isinstance(raw_view, dict):
            continue
        raw_origin = raw_view.get("origin")
        if not isinstance(raw_origin, list) or len(raw_origin) != 3:
            continue
        try:
            origin = tuple(float(value) for value in raw_origin)
        except (TypeError, ValueError):
            continue
        lidars.append(LidarPose(str(name), origin))  # type: ignore[arg-type]
    return sorted(lidars, key=lambda lidar: lidar.name)


def fallback_lidars() -> list[LidarPose]:
    """Return fixed vehicle 3D LiDAR poses matching the A2D2 setup shape."""

    return [
        LidarPose("front_center", (1.72, 0.00, 1.12)),
        LidarPose("front_left", (1.71, 0.63, 1.12)),
        LidarPose("front_right", (1.71, -0.65, 1.12)),
        LidarPose("rear_left", (-0.43, 0.59, 1.15)),
        LidarPose("rear_right", (-0.43, -0.61, 1.12)),
    ]


def draw_frame(
    *,
    image: bytearray,
    lidars: list[LidarPose],
    cloud_pair: LidarCloudPair,
    progress: float,
    residual_history: list[float],
    metadata_source: str,
) -> None:
    fill_rect(image, 0, 0, WIDTH, HEIGHT, BG)
    fill_rect(image, SCENE_PANEL[0], SCENE_PANEL[1], SCENE_PANEL[2], SCENE_PANEL[3], PANEL)
    fill_rect(image, RIGHT_PANEL[0], RIGHT_PANEL[1], RIGHT_PANEL[2], RIGHT_PANEL[3], PANEL)
    rect(image, SCENE_PANEL[0], SCENE_PANEL[1], SCENE_PANEL[2], SCENE_PANEL[3], GRID, alpha=0.85)
    rect(
        image,
        RIGHT_PANEL[0],
        RIGHT_PANEL[1],
        RIGHT_PANEL[2],
        RIGHT_PANEL[3],
        GRID,
        alpha=0.85,
    )

    draw_calibration_scene(image, lidars, cloud_pair, progress)
    draw_evidence_panel(image, cloud_pair, progress, residual_history, metadata_source)
    draw_timeline(image, progress)


def draw_online_calibration_frame(
    *,
    image: bytearray,
    lidars: list[LidarPose],
    cloud_pair: LidarCloudPair,
    frame_state: OnlineGifFrameState,
    metadata_source: str,
) -> None:
    fill_rect(image, 0, 0, WIDTH, HEIGHT, BG)
    fill_rect(image, SCENE_PANEL[0], SCENE_PANEL[1], SCENE_PANEL[2], SCENE_PANEL[3], PANEL)
    fill_rect(image, RIGHT_PANEL[0], RIGHT_PANEL[1], RIGHT_PANEL[2], RIGHT_PANEL[3], PANEL)
    rect(image, SCENE_PANEL[0], SCENE_PANEL[1], SCENE_PANEL[2], SCENE_PANEL[3], GRID, alpha=0.85)
    rect(
        image,
        RIGHT_PANEL[0],
        RIGHT_PANEL[1],
        RIGHT_PANEL[2],
        RIGHT_PANEL[3],
        GRID,
        alpha=0.85,
    )

    draw_online_calibration_scene(
        image,
        lidars,
        cloud_pair,
        frame_state,
    )
    draw_online_evidence_panel(
        image,
        cloud_pair,
        frame_state,
        metadata_source,
    )
    draw_online_timeline(image, frame_state)


def draw_online_calibration_scene(
    image: bytearray,
    lidars: list[LidarPose],
    cloud_pair: LidarCloudPair,
    frame_state: OnlineGifFrameState,
) -> None:
    draw_road_grid(image)
    draw_live_lidar_clouds(image, cloud_pair, frame_state)
    draw_vehicle_box(image)

    source = lidar_by_name(lidars, cloud_pair.source_pose_name)
    target = lidar_by_name(lidars, cloud_pair.target_pose_name)
    current_target = candidate_pose_from_offset(target, frame_state.visual_offset)

    draw_scanline(image, frame_state.cycle)
    draw_packet_flow(image, frame_state.cycle)
    draw_candidate_trail(image, target, frame_state)
    for lidar in lidars:
        strong = lidar.name in {source.name, target.name}
        color = REFERENCE if lidar.name != target.name else mix(
            CANDIDATE, OPTIMIZED, frame_state.convergence_ratio
        )
        draw_lidar_sensor(image, lidar.origin, color, strong=strong)
        if strong:
            draw_lidar_sweep(image, lidar.origin, frame_state.cycle, color)

    draw_lidar_sensor(
        image,
        current_target.origin,
        mix(CANDIDATE, OPTIMIZED, frame_state.convergence_ratio),
        strong=True,
    )
    draw_transform_arrow(image, source.origin, target.origin, REFERENCE, alpha=0.42)
    draw_transform_arrow(
        image,
        source.origin,
        current_target.origin,
        mix(CANDIDATE, OPTIMIZED, frame_state.convergence_ratio),
        alpha=0.94,
    )
    draw_delta_vector(image, target.origin, current_target.origin, frame_state.convergence_ratio)
    draw_lock_envelope(image, target.origin, frame_state.convergence_ratio)


def draw_live_lidar_clouds(
    image: bytearray,
    cloud_pair: LidarCloudPair,
    frame_state: OnlineGifFrameState,
) -> None:
    remaining = 1.0 - frame_state.convergence_ratio
    offset = frame_state.visual_offset
    candidate_color = mix(CANDIDATE, OPTIMIZED, frame_state.convergence_ratio)
    for index, point in enumerate(cloud_pair.source_points):
        if index % 2:
            continue
        px, py = project_lidar_point(point)
        pulse = 1.0 - min(
            1.0, abs(((index * 0.019 + frame_state.cycle) % 1.0) - 0.5) * 3.5
        )
        circle(image, px, py, 1, REFERENCE, alpha=0.18 + 0.48 * pulse)
        if pulse > 0.78 and index % 7 == 0:
            circle(image, px, py, 2, (134, 239, 172), alpha=0.32)
    for index, point in enumerate(cloud_pair.target_points):
        if index % 2:
            continue
        shifted = (
            point[0] + offset[0] * remaining,
            point[1] + offset[1] * remaining,
            point[2] + offset[2] * remaining,
        )
        px, py = project_lidar_point(shifted)
        pulse = 1.0 - min(
            1.0, abs(((index * 0.023 + frame_state.cycle + 0.28) % 1.0) - 0.5) * 3.2
        )
        circle(image, px, py, 1, candidate_color, alpha=0.20 + 0.56 * pulse)
        if pulse > 0.80 and index % 6 == 0:
            circle(image, px, py, 2, candidate_color, alpha=0.34)


def draw_scanline(image: bytearray, cycle: float) -> None:
    x = SCENE_PANEL[0] + round(SCENE_PANEL[2] * cycle)
    fill_rect(image, x - 5, SCENE_PANEL[1] + 2, 2, SCENE_PANEL[3] - 4, OPTIMIZED, alpha=0.12)
    fill_rect(image, x - 2, SCENE_PANEL[1] + 2, 4, SCENE_PANEL[3] - 4, OPTIMIZED, alpha=0.44)
    fill_rect(image, x + 4, SCENE_PANEL[1] + 2, 2, SCENE_PANEL[3] - 4, OPTIMIZED, alpha=0.16)
    for offset in (54, 132, 218):
        yy = SCENE_PANEL[1] + offset
        line(
            image,
            SCENE_PANEL[0] + 16,
            yy,
            SCENE_PANEL[0] + SCENE_PANEL[2] - 16,
            yy,
            GRID,
            alpha=0.30,
        )


def draw_packet_flow(image: bytearray, cycle: float) -> None:
    x0 = SCENE_PANEL[0] + 24
    y = SCENE_PANEL[1] + 46
    width = SCENE_PANEL[2] - 48
    fill_rect(image, x0, y, width, 5, (31, 41, 55), alpha=0.72)
    for index in range(18):
        phase = (cycle + index / 18) % 1.0
        x = x0 + round(width * phase)
        color = OPTIMIZED if index % 3 else GOOD
        circle(image, x, y + 2, 4, color, alpha=0.54)


def draw_candidate_trail(
    image: bytearray,
    target: LidarPose,
    frame_state: OnlineGifFrameState,
) -> None:
    trail_points: list[Point2] = []
    for index in range(8):
        trail_progress = frame_state.convergence_ratio * index / 7
        offset = (
            frame_state.visual_offset[0] * (1.0 - trail_progress),
            frame_state.visual_offset[1] * (1.0 - trail_progress),
            frame_state.visual_offset[2] * (1.0 - trail_progress),
        )
        pose = candidate_pose_from_offset(target, offset)
        x, y = project((pose.origin[0], pose.origin[1], pose.origin[2] + 0.42))
        trail_points.append((x, y))
        color = mix(CANDIDATE, OPTIMIZED, trail_progress)
        circle(image, x, y, 3 + index // 3, color, alpha=0.32 + 0.06 * index)
    for index in range(1, len(trail_points)):
        trail_progress = frame_state.convergence_ratio * index / 7
        line_between(
            image,
            trail_points[index - 1],
            trail_points[index],
            mix(CANDIDATE, OPTIMIZED, trail_progress),
            0.46,
            thickness=2,
        )


def candidate_pose_from_offset(reference: LidarPose, offset: Point3) -> LidarPose:
    return LidarPose(
        f"{reference.name}_candidate",
        (
            reference.origin[0] + offset[0],
            reference.origin[1] + offset[1],
            reference.origin[2] + offset[2],
        ),
    )


def draw_lock_envelope(image: bytearray, origin: Point3, progress: float) -> None:
    x, y = project((origin[0], origin[1], origin[2] + 0.42))
    radius = max(9, round(34 * (1.0 - progress) + 9))
    color = mix(WARNING, GOOD, progress)
    for index in range(3):
        circle(image, x, y, radius + index * 7, color, alpha=0.06 + progress * 0.04)
    circle(image, x, y, 8, color, alpha=0.32 + progress * 0.34)


def draw_lidar_sweep(image: bytearray, origin: Point3, cycle: float, color: Color) -> None:
    ox, oy = project(origin)
    for index in range(9):
        phase = cycle * math.tau + index * 0.14
        endpoint = (
            origin[0] + 2.35 * math.cos(phase),
            origin[1] + 2.35 * math.sin(phase),
            origin[2] - 0.08,
        )
        ex, ey = project(endpoint)
        line(image, ox, oy, ex, ey, color, alpha=0.15 + index * 0.030)


def draw_online_evidence_panel(
    image: bytearray,
    cloud_pair: LidarCloudPair,
    frame_state: OnlineGifFrameState,
    metadata_source: str,
) -> None:
    current_residual = frame_state.residual_history[-1] if frame_state.residual_history else None
    residual_score = 0.0
    if current_residual is not None:
        residual_score = 1.0 - min(
            1.0,
            (current_residual - ONLINE_RESIDUAL_CHART_MIN_M)
            / (ONLINE_RESIDUAL_CHART_MAX_M - ONLINE_RESIDUAL_CHART_MIN_M),
        )
    visible_batches = max(1, len(frame_state.gate_statuses))
    stability_score = frame_state.accepted_batch_count / visible_batches
    holdout_score = 0.0
    if frame_state.current_holdout_rmse_m is not None:
        holdout_score = 1.0 - min(
            1.0, frame_state.current_holdout_rmse_m / ONLINE_GIF_MAX_HOLDOUT_RMSE_M
        )
    provenance_score = 1.0

    fill_rect(image, 674, 132, 220, 132, PANEL_ALT)
    rect(image, 674, 132, 220, 132, GRID, alpha=0.92)
    metric_bar(image, 674, 156, residual_score, mix(WARNING, GOOD, frame_state.convergence_ratio))
    metric_bar(image, 674, 180, stability_score, OPTIMIZED)
    metric_bar(image, 674, 204, holdout_score, REFERENCE)
    metric_bar(image, 674, 228, provenance_score, GOOD)

    chart_x, chart_y, chart_width, chart_height = CHART
    fill_rect(image, chart_x, chart_y, chart_width, chart_height, PANEL_ALT)
    rect(image, chart_x, chart_y, chart_width, chart_height, GRID, alpha=0.95)
    for line_index in range(1, 4):
        y = chart_y + line_index * chart_height // 4
        line(image, chart_x, y, chart_x + chart_width, y, GRID, alpha=0.45)
    threshold_ratio = 1.0 - (
        (ONLINE_GIF_MAX_HOLDOUT_RMSE_M - ONLINE_RESIDUAL_CHART_MIN_M)
        / (ONLINE_RESIDUAL_CHART_MAX_M - ONLINE_RESIDUAL_CHART_MIN_M)
    )
    threshold_y = chart_y + round(chart_height * max(0.0, min(1.0, threshold_ratio)))
    line(image, chart_x, threshold_y, chart_x + chart_width, threshold_y, GOOD, alpha=0.42)
    draw_curve(
        image,
        frame_state.residual_history,
        CHART,
        OPTIMIZED,
        min_value=ONLINE_RESIDUAL_CHART_MIN_M,
        max_value=ONLINE_RESIDUAL_CHART_MAX_M,
    )

    draw_stream_packets(image, 674, 354, frame_state)
    draw_gate_cells(image, 674, 424, frame_state)
    draw_source_badge(image, metadata_source)


def draw_stream_packets(
    image: bytearray,
    x: int,
    y: int,
    frame_state: OnlineGifFrameState,
) -> None:
    fill_rect(image, x, y, 220, 42, PANEL_ALT)
    rect(image, x, y, 220, 42, GRID, alpha=0.80)
    counts = frame_state.batch_point_counts[-12:]
    accepted = frame_state.batch_accepted[-12:]
    if not counts:
        return
    max_count = max(counts)
    for index, point_count in enumerate(counts):
        packet_x = x + 10 + index * 17
        phase = (frame_state.cycle + index * 0.083) % 1.0
        normalized = point_count / max(1, max_count)
        height = 8 + round(20 * normalized * (0.35 + 0.65 * math.sin(phase * math.pi) ** 2))
        color = OPTIMIZED if accepted[index] else WARNING
        fill_rect(image, packet_x, y + 32 - height, 10, height, color, alpha=0.86)


def draw_gate_cells(image: bytearray, x: int, y: int, frame_state: OnlineGifFrameState) -> None:
    labels = 6
    statuses = list(frame_state.gate_statuses[-labels:])
    for index in range(labels):
        cell_x = x + index * 35
        if index < len(statuses):
            status = statuses[index]
            if status == "pass":
                color = GOOD
            elif status == "fail":
                color = CANDIDATE
            else:
                color = WARNING
        else:
            color = TEXT_DIM
        fill_rect(image, cell_x, y, 24, 22, color, alpha=0.90 if index < len(statuses) else 0.35)
        rect(image, cell_x, y, 24, 22, (229, 231, 235), alpha=0.24)
        if index < len(statuses) and statuses[index] == "pass":
            rect(
                image,
                cell_x - 2,
                y - 2,
                28,
                26,
                color,
                alpha=0.40 + 0.25 * frame_state.convergence_ratio,
            )


def draw_online_timeline(image: bytearray, frame_state: OnlineGifFrameState) -> None:
    x0, y, width = 82, 516, 786
    fill_rect(image, x0, y, width, 8, (31, 41, 55), alpha=1.0)
    fill_rect(image, x0, y, round(width * frame_state.progress), 8, OPTIMIZED, alpha=0.95)
    cursor_x = x0 + round(width * frame_state.cycle)
    fill_rect(image, cursor_x - 2, y - 12, 4, 32, WARNING, alpha=0.82)
    for marker in [0.0, 0.33, 0.66, 1.0]:
        x = x0 + round(width * marker)
        circle(
            image,
            x,
            y + 4,
            8,
            OPTIMIZED if frame_state.progress >= marker else TEXT_DIM,
            alpha=1.0,
        )


def draw_calibration_scene(
    image: bytearray,
    lidars: list[LidarPose],
    cloud_pair: LidarCloudPair,
    progress: float,
) -> None:
    draw_road_grid(image)
    draw_real_lidar_clouds(image, cloud_pair, progress)
    draw_vehicle_box(image)

    source = lidar_by_name(lidars, cloud_pair.source_pose_name)
    target = lidar_by_name(lidars, cloud_pair.target_pose_name)
    current_target = animated_candidate_pose(target, progress)

    for lidar in lidars:
        color = REFERENCE if lidar.name != target.name else mix(CANDIDATE, OPTIMIZED, progress)
        draw_lidar_sensor(
            image,
            lidar.origin,
            color,
            strong=lidar.name in {source.name, target.name},
        )
    draw_lidar_sensor(
        image,
        current_target.origin,
        mix(CANDIDATE, OPTIMIZED, progress),
        strong=True,
    )

    draw_transform_arrow(image, source.origin, target.origin, REFERENCE, alpha=0.60)
    draw_transform_arrow(
        image,
        source.origin,
        current_target.origin,
        mix(CANDIDATE, OPTIMIZED, progress),
    )
    draw_delta_vector(image, target.origin, current_target.origin, progress)


def lidar_by_name(lidars: list[LidarPose], name: str) -> LidarPose:
    for lidar in lidars:
        if lidar.name == name:
            return lidar
    return lidars[0]


def animated_candidate_pose(reference: LidarPose, progress: float) -> LidarPose:
    remaining = 1.0 - progress
    offset = (
        0.22 * remaining,
        -0.30 * remaining,
        0.13 * remaining,
    )
    return LidarPose(
        f"{reference.name}_candidate",
        (
            reference.origin[0] + offset[0],
            reference.origin[1] + offset[1],
            reference.origin[2] + offset[2],
        ),
    )


def draw_road_grid(image: bytearray) -> None:
    fill_rect(image, 48, 372, 540, 116, ROAD, alpha=0.84)
    for y in [-4.0, -2.0, 0.0, 2.0, 4.0]:
        left = project((-3.0, y, -0.05))
        right = project((7.0, y, -0.05))
        line(image, left[0], left[1], right[0], right[1], GRID, alpha=0.38)
    for x in [-2.0, 0.0, 2.0, 4.0, 6.0]:
        bottom = project((x, -4.5, -0.05))
        top = project((x, 4.5, -0.05))
        line(image, bottom[0], bottom[1], top[0], top[1], GRID, alpha=0.38)


def draw_real_lidar_clouds(
    image: bytearray,
    cloud_pair: LidarCloudPair,
    progress: float,
) -> None:
    remaining = 1.0 - progress
    color = mix(CANDIDATE, OPTIMIZED, progress)
    for index, point in enumerate(cloud_pair.source_points):
        px, py = project_lidar_point(point)
        circle(image, px, py, 1, REFERENCE if index % 3 else (110, 231, 183), alpha=0.58)
    for index, point in enumerate(cloud_pair.target_points):
        px, py = project_lidar_point(point)
        circle(image, px, py, 1, REFERENCE, alpha=0.23)
        if index % 2:
            continue
        shifted = (
            point[0] + 1.8 * remaining,
            point[1] - 1.2 * remaining,
            point[2] + 0.45 * remaining,
        )
        x, y = project_lidar_point(shifted)
        circle(image, x, y, 1, color, alpha=0.72)


def scene_points() -> list[Point3]:
    points: list[Point3] = []
    for x_index in range(34):
        x = -2.2 + x_index * 0.26
        for lane_y in (-1.8, 1.8):
            wave = math.sin(x * 1.7) * 0.08
            points.append((x, lane_y + wave, 0.0))
    for side_y in (-3.2, 3.2):
        for x_index in range(24):
            x = -1.4 + x_index * 0.32
            for z_index in range(4):
                z = 0.35 + z_index * 0.44
                points.append((x, side_y, z))
    for angle_index in range(60):
        angle = angle_index * math.tau / 60
        radius = 1.35 + 0.16 * math.sin(angle * 3.0)
        points.append((3.8 + radius * math.cos(angle), radius * math.sin(angle), 0.18))
    return points


def draw_vehicle_box(image: bytearray) -> None:
    bottom = [(-1.0, -1.25, 0.0), (4.0, -1.25, 0.0), (4.0, 1.25, 0.0), (-1.0, 1.25, 0.0)]
    roof = [(-0.6, -1.0, 1.35), (2.6, -1.0, 1.35), (2.6, 1.0, 1.35), (-0.6, 1.0, 1.35)]
    bottom_points = [project(point) for point in bottom]
    roof_points = [project(point) for point in roof]
    fill_polygon(image, bottom_points, (30, 41, 58), alpha=0.86)
    fill_polygon(image, roof_points, VEHICLE, alpha=0.92)
    for index in range(4):
        next_index = (index + 1) % 4
        line_between(image, bottom_points[index], bottom_points[next_index], (100, 116, 139), 0.70)
        line_between(image, roof_points[index], roof_points[next_index], (185, 196, 214), 0.82)
        line_between(image, bottom_points[index], roof_points[index], (100, 116, 139), 0.48)


def draw_lidar_sensor(image: bytearray, origin: Point3, color: Color, *, strong: bool) -> None:
    x, y = project(origin)
    radius = 6 if strong else 4
    circle(image, x, y, radius + 3, (5, 9, 16), alpha=0.56)
    circle(image, x, y, radius, color, alpha=1.0)
    endpoints = (
        (origin[0] + 0.55, origin[1], origin[2]),
        (origin[0], origin[1] + 0.42, origin[2]),
    )
    for endpoint in endpoints:
        ex, ey = project(endpoint)
        line(image, x, y, ex, ey, color, alpha=0.68)


def draw_transform_arrow(
    image: bytearray,
    source: Point3,
    target: Point3,
    color: Color,
    *,
    alpha: float = 0.95,
) -> None:
    sx, sy = project((source[0], source[1], source[2] + 0.10))
    tx, ty = project((target[0], target[1], target[2] + 0.10))
    thick_line(image, sx, sy, tx, ty, color, thickness=2, alpha=alpha)
    circle(image, tx, ty, 5, color, alpha=alpha)


def draw_delta_vector(
    image: bytearray,
    reference: Point3,
    candidate: Point3,
    progress: float,
) -> None:
    if progress > 0.97:
        return
    rx, ry = project((reference[0], reference[1], reference[2] + 0.28))
    cx, cy = project((candidate[0], candidate[1], candidate[2] + 0.28))
    line(image, rx, ry, cx, cy, WARNING, alpha=0.88)
    circle(image, rx, ry, 3, REFERENCE, alpha=0.95)
    circle(image, cx, cy, 3, WARNING, alpha=0.95)


def draw_evidence_panel(
    image: bytearray,
    cloud_pair: LidarCloudPair,
    progress: float,
    residual_history: list[float],
    metadata_source: str,
) -> None:
    fill_rect(image, 674, 132, 220, 132, PANEL_ALT)
    rect(image, 674, 132, 220, 132, GRID, alpha=0.92)
    metric_bar(image, 674, 156, cloud_pair.bar_values[0], mix(WARNING, GOOD, progress))
    metric_bar(image, 674, 180, cloud_pair.bar_values[1], OPTIMIZED)
    metric_bar(image, 674, 204, cloud_pair.bar_values[2], GOOD)
    metric_bar(image, 674, 228, cloud_pair.bar_values[3], REFERENCE)

    chart_x, chart_y, chart_width, chart_height = CHART
    fill_rect(image, chart_x, chart_y, chart_width, chart_height, PANEL_ALT)
    rect(image, chart_x, chart_y, chart_width, chart_height, GRID, alpha=0.95)
    for line_index in range(1, 4):
        y = chart_y + line_index * chart_height // 4
        line(image, chart_x, y, chart_x + chart_width, y, GRID, alpha=0.45)
    draw_curve(image, residual_history, CHART, OPTIMIZED)

    draw_case_detail(image, 674, 354, progress)
    draw_dof_cells(image, 674, 424, progress)
    draw_source_badge(image, metadata_source)


def draw_source_badge(image: bytearray, metadata_source: str) -> None:
    color = GOOD if metadata_source.startswith(("A2D2", "Livox")) else WARNING
    fill_rect(image, 674, 460, 220, 42, PANEL_ALT)
    rect(image, 674, 460, 220, 42, color, alpha=0.72)


def draw_curve(
    image: bytearray,
    values: list[float],
    chart: tuple[int, int, int, int],
    color: Color,
    *,
    min_value: float = 0.020,
    max_value: float = 0.110,
) -> None:
    if len(values) < 2:
        return
    x, y, width, height = chart
    points: list[Point2] = []
    for index, value in enumerate(values):
        px = x + round(index * width / max(1, len(values) - 1))
        ratio = (value - min_value) / (max_value - min_value)
        py = y + height - round(max(0.0, min(1.0, ratio)) * height)
        points.append((px, py))
    for index in range(1, len(points)):
        line_between(image, points[index - 1], points[index], color, 0.95, thickness=2)
    circle(image, points[-1][0], points[-1][1], 4, color, alpha=1.0)


def metric_bar(image: bytearray, x: int, y: int, value: float, color: Color) -> None:
    fill_rect(image, x, y, 220, 8, (31, 41, 55), alpha=1.0)
    fill_rect(image, x, y, round(220 * max(0.0, min(1.0, value))), 8, color, alpha=0.95)


def draw_dof_cells(image: bytearray, x: int, y: int, progress: float) -> None:
    for index in range(6):
        cell_x = x + index * 35
        good = index in {0, 1, 3, 4} or progress > 0.82
        color = GOOD if good else WARNING
        fill_rect(image, cell_x, y, 24, 22, color, alpha=0.90)
        rect(image, cell_x, y, 24, 22, (229, 231, 235), alpha=0.24)


def draw_case_detail(image: bytearray, x: int, y: int, progress: float) -> None:
    fill_rect(image, x, y, 220, 42, PANEL_ALT)
    rect(image, x, y, 220, 42, GRID, alpha=0.80)
    # Three compact rows approximate the HTML Known-Bad Case Details table.
    row_widths = (
        round(174 * min(1.0, progress + 0.20)),
        round(132 * min(1.0, progress + 0.08)),
        round(82 * min(1.0, progress)),
    )
    colors = (GOOD, OPTIMIZED, WARNING)
    for index, row_width in enumerate(row_widths):
        row_y = y + 8 + index * 10
        fill_rect(image, x + 84, row_y, 96, 5, (31, 41, 55), alpha=1.0)
        fill_rect(image, x + 84, row_y, min(96, row_width), 5, colors[index], alpha=0.92)


def draw_timeline(image: bytearray, progress: float) -> None:
    x0, y, width = 82, 516, 786
    fill_rect(image, x0, y, width, 8, (31, 41, 55), alpha=1.0)
    fill_rect(image, x0, y, round(width * progress), 8, OPTIMIZED, alpha=0.95)
    for marker in [0.0, 0.33, 0.66, 1.0]:
        x = x0 + round(width * marker)
        circle(image, x, y + 4, 8, OPTIMIZED if progress >= marker else TEXT_DIM, alpha=1.0)


def residual_proxy(progress: float) -> float:
    remaining = 1.0 - progress
    return 0.024 + 0.080 * remaining * remaining


def project(point: Point3) -> Point2:
    x, y, z = point
    return (
        round(318 + x * 61 - y * 72),
        round(410 + x * 17 + y * 21 - z * 92),
    )


def project_lidar_point(point: Point3) -> Point2:
    x, y, z = point
    return (
        round(292 + x * 2.6 + y * 7.0),
        round(430 - x * 2.0 - z * 15.0),
    )


def smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def mix(left: Color, right: Color, ratio: float) -> Color:
    return tuple(
        round(left[index] + (right[index] - left[index]) * max(0.0, min(1.0, ratio)))
        for index in range(3)
    )


def fill_rect(
    image: bytearray,
    x: int,
    y: int,
    width: int,
    height: int,
    color: Color,
    *,
    alpha: float = 1.0,
) -> None:
    for yy in range(max(0, y), min(HEIGHT, y + height)):
        for xx in range(max(0, x), min(WIDTH, x + width)):
            pixel(image, xx, yy, color, alpha)


def rect(
    image: bytearray,
    x: int,
    y: int,
    width: int,
    height: int,
    color: Color,
    *,
    alpha: float = 1.0,
) -> None:
    line(image, x, y, x + width, y, color, alpha=alpha)
    line(image, x + width, y, x + width, y + height, color, alpha=alpha)
    line(image, x + width, y + height, x, y + height, color, alpha=alpha)
    line(image, x, y + height, x, y, color, alpha=alpha)


def fill_polygon(image: bytearray, points: list[Point2], color: Color, *, alpha: float) -> None:
    if not points:
        return
    min_y = max(0, min(y for _x, y in points))
    max_y = min(HEIGHT - 1, max(y for _x, y in points))
    for y in range(min_y, max_y + 1):
        intersections: list[int] = []
        for index, start in enumerate(points):
            end = points[(index + 1) % len(points)]
            if (start[1] <= y < end[1]) or (end[1] <= y < start[1]):
                ratio = (y - start[1]) / (end[1] - start[1])
                intersections.append(round(start[0] + ratio * (end[0] - start[0])))
        intersections.sort()
        for index in range(0, len(intersections), 2):
            if index + 1 >= len(intersections):
                continue
            for x in range(max(0, intersections[index]), min(WIDTH, intersections[index + 1])):
                pixel(image, x, y, color, alpha)


def thick_line(
    image: bytearray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: Color,
    *,
    thickness: int,
    alpha: float,
) -> None:
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for index in range(steps + 1):
        x = round(x0 + (x1 - x0) * index / steps)
        y = round(y0 + (y1 - y0) * index / steps)
        circle(image, x, y, thickness, color, alpha=alpha)


def line_between(
    image: bytearray,
    left: Point2,
    right: Point2,
    color: Color,
    alpha: float,
    *,
    thickness: int = 1,
) -> None:
    if thickness <= 1:
        line(image, left[0], left[1], right[0], right[1], color, alpha=alpha)
    else:
        thick_line(
            image,
            left[0],
            left[1],
            right[0],
            right[1],
            color,
            thickness=thickness,
            alpha=alpha,
        )


def line(
    image: bytearray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: Color,
    *,
    alpha: float = 1.0,
) -> None:
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for index in range(steps + 1):
        x = round(x0 + (x1 - x0) * index / steps)
        y = round(y0 + (y1 - y0) * index / steps)
        pixel(image, x, y, color, alpha)


def circle(
    image: bytearray,
    cx: int,
    cy: int,
    radius: int,
    color: Color,
    *,
    alpha: float = 1.0,
) -> None:
    r2 = radius * radius
    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r2:
                pixel(image, x, y, color, alpha)


def pixel(image: bytearray, x: int, y: int, color: Color, alpha: float) -> None:
    if x < 0 or y < 0 or x >= WIDTH or y >= HEIGHT:
        return
    index = (y * WIDTH + x) * 3
    if alpha >= 1.0:
        image[index] = color[0]
        image[index + 1] = color[1]
        image[index + 2] = color[2]
        return
    inverse = 1.0 - alpha
    image[index] = round(image[index] * inverse + color[0] * alpha)
    image[index + 1] = round(image[index + 1] * inverse + color[1] * alpha)
    image[index + 2] = round(image[index + 2] * inverse + color[2] * alpha)


def write_ppm(path: Path, image: bytearray) -> None:
    with path.open("wb") as handle:
        handle.write(f"P6\n{WIDTH} {HEIGHT}\n255\n".encode("ascii"))
        handle.write(image)


def encode_gif(
    frame_dir: Path,
    output: Path,
    cloud_pair: LidarCloudPair,
    *,
    visual: VisualMode,
    online_run: OnlineGifRun | None = None,
) -> None:
    palette = frame_dir / "palette.png"
    text_filter = (
        build_online_text_filter(cloud_pair, online_run=online_run)
        if visual == "online"
        else build_text_filter(cloud_pair)
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-framerate",
            str(FPS),
            "-thread_queue_size",
            "64",
            "-i",
            str(frame_dir / "frame_%03d.ppm"),
            "-vf",
            f"{text_filter},palettegen=max_colors=96",
            "-frames:v",
            "1",
            "-update",
            "true",
            str(palette),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-framerate",
            str(FPS),
            "-thread_queue_size",
            "64",
            "-i",
            str(frame_dir / "frame_%03d.ppm"),
            "-i",
            str(palette),
            "-lavfi",
            f"[0:v]{text_filter}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
            "-loop",
            "0",
            str(output),
        ],
        check=True,
    )


def build_text_filter(cloud_pair: LidarCloudPair) -> str:
    font_file = subprocess.check_output(
        ["fc-match", "-f", "%{file}", "Noto Sans"],
        text=True,
    ).strip()
    labels = [
        ("Fixed 3D LiDAR to fixed 3D LiDAR", 34, 22, 26, "E5E7EB"),
        (cloud_pair.subtitle, 34, 55, 16, "CBD5E1"),
        (cloud_pair.scene_caption, 50, 104, 17, "E5E7EB"),
        (
            cloud_pair.legend,
            50,
            486,
            14,
            "CBD5E1",
        ),
        ("Evidence", 674, 104, 17, "E5E7EB"),
        (cloud_pair.bar_labels[0], 674, 138, 13, "CBD5E1"),
        (cloud_pair.bar_labels[1], 674, 162, 13, "CBD5E1"),
        (cloud_pair.bar_labels[2], 674, 186, 13, "CBD5E1"),
        (cloud_pair.bar_labels[3], 674, 210, 13, "CBD5E1"),
        (cloud_pair.support_summary, 674, 238, 12, "CBD5E1"),
        (cloud_pair.holdout_summary, 674, 252, 12, "CBD5E1"),
        (cloud_pair.known_bad_summary, 674, 266, 12, "CBD5E1"),
        ("evidence curve", 676, 274, 16, "E5E7EB"),
        ("Known-Bad Case Details", 674, 330, 16, "E5E7EB"),
        ("pitch +1deg", 684, 360, 12, "CBD5E1"),
        ("P90 +0.034m", 684, 370, 12, "CBD5E1"),
        (cloud_pair.case_summary, 674, 400, 12, "CBD5E1"),
        ("DoF visibility", 674, 410, 14, "E5E7EB"),
        ("x  y  z  r  p  yaw", 674, 450, 12, "CBD5E1"),
        (cloud_pair.provenance, 682, 468, 12, "CBD5E1"),
        (cloud_pair.protocol_summary, 682, 482, 12, "CBD5E1"),
        ("evidence.json sidecar", 682, 494, 12, "CBD5E1"),
        ("public setup", 62, 520, 14, "CBD5E1"),
        ("candidate", 325, 520, 14, "CBD5E1"),
        ("optimize", 588, 520, 14, "CBD5E1"),
        ("report", 842, 520, 14, "CBD5E1"),
    ]
    return ",".join(drawtext(font_file, *label) for label in labels)


def build_online_text_filter(
    cloud_pair: LidarCloudPair,
    *,
    online_run: OnlineGifRun | None = None,
) -> str:
    font_file = subprocess.check_output(
        ["fc-match", "-f", "%{file}", "Noto Sans"],
        text=True,
    ).strip()
    if online_run is not None:
        assessment = f"assessment: {online_run.final_gate_status.upper()} from timeline"
        accepted_summary = (
            f"{online_run.accepted_batch_count}/{online_run.batch_count} batches accepted"
        )
        provenance = "provenance: A2D2 NPZ + calibrate --online timeline"
    else:
        assessment = "assessment: timeline unavailable"
        accepted_summary = "accepted window"
        provenance = cloud_pair.provenance
    labels = [
        ("Online Calibration Evidence Loop", 34, 22, 26, "E5E7EB"),
        (cloud_pair.subtitle, 34, 55, 16, "CBD5E1"),
        ("Live point batches, rolling residuals, and schema-valid gates", 50, 104, 17, "E5E7EB"),
        ("green: reference stream   cyan/pink: online estimate converging", 50, 486, 14, "CBD5E1"),
        ("Streaming checks", 674, 104, 17, "E5E7EB"),
        ("rolling residual", 674, 138, 13, "CBD5E1"),
        ("temporal stability", 674, 162, 13, "CBD5E1"),
        ("holdout gate", 674, 186, 13, "CBD5E1"),
        ("provenance lock", 674, 210, 13, "CBD5E1"),
        (cloud_pair.support_summary, 674, 238, 12, "CBD5E1"),
        (cloud_pair.holdout_summary, 674, 252, 12, "CBD5E1"),
        ("rolling residual", 676, 274, 16, "E5E7EB"),
        ("Stream Batches", 674, 330, 16, "E5E7EB"),
        ("sensor packets", 684, 360, 12, "CBD5E1"),
        (accepted_summary, 684, 380, 12, "CBD5E1"),
        ("Policy gates", 674, 410, 14, "E5E7EB"),
        ("pass  fail  inconclusive (real batches)", 674, 450, 12, "CBD5E1"),
        (provenance, 682, 468, 12, "CBD5E1"),
        ("timeline.json + evidence.json", 682, 482, 12, "CBD5E1"),
        (assessment, 682, 494, 12, "CBD5E1"),
        ("ingest", 62, 520, 14, "CBD5E1"),
        ("estimate", 320, 520, 14, "CBD5E1"),
        ("holdout", 586, 520, 14, "CBD5E1"),
        ("assess", 842, 520, 14, "CBD5E1"),
    ]
    return ",".join(drawtext(font_file, *label) for label in labels)


def drawtext(
    font_file: str,
    text: str,
    x: int,
    y: int,
    size: int,
    color: str,
) -> str:
    escaped_text = (
        text.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace(":", "\\:")
        .replace(",", "\\,")
        .replace("%", "\\%")
    )
    escaped_font = font_file.replace("\\", "\\\\").replace(":", "\\:")
    return (
        "drawtext="
        f"fontfile='{escaped_font}':"
        f"text='{escaped_text}':"
        f"x={x}:y={y}:fontsize={size}:fontcolor=0x{color}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
