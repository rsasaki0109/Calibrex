"""Acquire only frozen spent-drive frames and prepare unchanged joint inputs."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.spatial.transform import Rotation
from tools.kitti_aligned_stereo_bundle import build
from tools.kitti_drive_fetch import central_directory, drive_url, extract, fetch_range, head
from tools.run_camera_lidar_development_matrix import DevelopmentMatrixPlan, digest

from calibrex.data.kitti_raw_to_rosbag2 import convert_kitti_raw_to_rosbag2


class PreparationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scope: str
    plan_sha256: str
    driver_sha256: str
    drives: list[dict[str, Any]]


def prepare(
    plan_path: Path, root: Path, calibration_root: Path, reuse_raw: Path | None = None
) -> None:
    """Keep original frame IDs/timestamps and require complete local pose coverage."""
    plan = DevelopmentMatrixPlan.model_validate_json(plan_path.read_text())
    for filename, expected in plan.source_sha256.items():
        if digest(Path(filename)) != expected:
            raise ValueError(f"changed frozen source: {filename}")
    cached = {}
    if reuse_raw is not None:
        old = PreparationReport.model_validate_json((reuse_raw / "preparation.json").read_text())
        cached = {m["member"]: m["sha256"] for d in old.drives for m in d.get("members", [])}
    root.mkdir(exist_ok=False, parents=True)
    results = []
    report = PreparationReport(
        scope="sparse acquisition of already-spent drives; no fresh evaluation",
        plan_sha256=digest(plan_path),
        driver_sha256=digest(Path(__file__)),
        drives=results,
    )

    def save() -> None:
        report.drives = results
        (root / "preparation.json").write_text(report.model_dump_json(indent=2) + "\n")
        (root / "preparation.schema.json").write_text(
            json.dumps(PreparationReport.model_json_schema(), indent=2) + "\n"
        )

    try:
        for drive in plan.drives:
            record = {"drive": drive.id, "status": "preparing"}
            results.append(record)
            save()
            date, number = drive.id[:10], drive.id[-4:]
            url = drive_url(date, number)
            size, etag = head(url)
            if size != drive.archive_size or etag != drive.archive_etag:
                raise ValueError("archive pin mismatch")
            infos = central_directory(url, size, etag)
            prefix = f"{date}/{date}_drive_{number}_sync/"
            relatives = [
                "image_02/timestamps.txt",
                "image_03/timestamps.txt",
                "oxts/timestamps.txt",
                "oxts/dataformat.txt",
                "velodyne_points/timestamps.txt",
                "velodyne_points/timestamps_start.txt",
                "velodyne_points/timestamps_end.txt",
            ]
            for frame in drive.frames:
                relatives.extend(
                    [
                        f"image_02/data/{frame:010d}.png",
                        f"image_03/data/{frame:010d}.png",
                        f"velodyne_points/data/{frame:010d}.bin",
                    ]
                )
            pose_ids = sorted({i for f in drive.frames for i in range(f - 2, f + 3)})
            if min(pose_ids) < 0 or max(pose_ids) >= drive.frame_count:
                raise ValueError("pose neighborhood falls outside recording")
            relatives.extend(f"oxts/data/{i:010d}.txt" for i in pose_ids)
            selected = {
                i.filename: i for i in infos if i.filename in {prefix + r for r in relatives}
            }
            if len(selected) != len(relatives):
                raise ValueError("required archive member missing")
            total = sum(
                i.compress_size + 30 + len(i.filename.encode()) + len(i.extra) + 4097
                for i in selected.values()
            )
            if total > 64 * 1024 * 1024:
                raise ValueError("sparse acquisition exceeds the frozen 64 MiB budget per drive")

            def fetch(
                name: str,
                selected: dict[str, Any] = selected,
                size: int = size,
                url: str = url,
                etag: str = etag,
            ) -> dict[str, Any]:
                info = selected[name]
                end = min(
                    size - 1,
                    info.header_offset
                    + 30
                    + len(info.filename.encode())
                    + len(info.extra)
                    + info.compress_size
                    + 4096,
                )
                path = root / "raw" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                cache_path = reuse_raw / "raw" / name if reuse_raw else None
                cache_hit = cache_path is not None and name in cached
                if cache_hit:
                    if digest(cache_path) != cached[name]:
                        raise ValueError("cached acquisition bytes changed")
                    payload = cache_path.read_bytes()
                else:
                    payload = extract(
                        fetch_range(url, info.header_offset, end, etag), info.header_offset, info
                    )
                path.write_bytes(payload)
                return {
                    "member": name,
                    "cached": cache_hit,
                    "sha256": digest(path),
                    "bytes": len(payload),
                    "range": [info.header_offset, end],
                }

            with ThreadPoolExecutor(max_workers=8) as pool:
                members = list(pool.map(fetch, sorted(selected)))
            record.update(
                url=url,
                etag=etag,
                archive_size=size,
                frame_ids=drive.frames,
                pose_ids=pose_ids,
                planned_range_bytes=total,
                fetched_bytes=sum(
                    m["range"][1] - m["range"][0] + 1 for m in members if not m["cached"]
                ),
                members=members,
            )
            save()
            raw = root / "raw" / date / f"{date}_drive_{number}_sync"
            bag = root / "bags" / drive.id
            convert_kitti_raw_to_rosbag2(
                raw,
                bag,
                calibration_dir=calibration_root / date,
                camera="image_02",
                velodyne_time="azimuth",
            )
            metadata = json.loads((bag / "calibrex_conversion.json").read_text())
            reference = np.array(metadata["camera"]["t_cam_velo"])
            record["conversion_sha256"] = digest(bag / "calibrex_conversion.json")
            shared = root / drive.id
            shared.mkdir()
            for frame in drive.frames:
                bundle = shared / f"frame-{frame}"
                build(
                    bag / f"{drive.id}_0.db3",
                    raw,
                    calibration_root / date,
                    raw / f"image_03/data/{frame:010d}.png",
                    raw / "image_03/timestamps.txt",
                    frame,
                    bundle,
                )
                for p in plan.perturbations:
                    initial = reference.copy()
                    initial[:3, :3] = (
                        Rotation.from_euler("xyz", p.rotation_deg, degrees=True).as_matrix()
                        @ reference[:3, :3]
                    )
                    initial[:3, 3] += p.translation_m
                    directory = root / f"{drive.id}-{frame:010d}-{p.id}"
                    directory.mkdir()
                    (directory / "seed.json").write_text(
                        json.dumps(
                            {
                                "reference_T_camera_lidar": reference.tolist(),
                                "initial_T_camera_lidar": initial.tolist(),
                            },
                            indent=2,
                        )
                        + "\n"
                    )
            record["status"] = "completed"
            save()
            print(drive.id, "prepared", total, "bytes", flush=True)
    finally:
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["plan", "output", "calibration"]:
        parser.add_argument(name, type=Path)
    parser.add_argument("--reuse-raw", type=Path)
    a = parser.parse_args()
    prepare(a.plan, a.output, a.calibration, a.reuse_raw)
