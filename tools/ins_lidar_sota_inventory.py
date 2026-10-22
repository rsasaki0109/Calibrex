"""Pin the first three eligible KITTI candidates using existing counts and live ZIP metadata.

No sensor member is extracted or decoded. The source pool is the earlier
camera--LiDAR audit's metadata, not its scores or outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import kitti_drive_fetch as fetch

SELECTION_RULE = (
    "From the eighteen other-day KITTI raw archives inventoried for camera-LiDAR v2, "
    "select the first three chronological recording IDs with at least 600 archived "
    "Velodyne frames and 600 Velodyne/OXTS members in their first-600-index download "
    "metadata. Exclude spent INS-LiDAR development recordings. Inspect only live HEAD "
    "and ZIP central-directory metadata; retain all three even if a pin/budget fails."
)
SPENT = {"2011_09_30_0018"}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SelectedArchive(_Strict):
    case_id: str
    date: str = Field(pattern=r"^\d{4}_\d{2}_\d{2}$")
    drive: str = Field(pattern=r"^\d{4}$")
    url: str
    size_bytes: int = Field(gt=0)
    etag: str
    archive_frames: dict[str, int]
    selected_frames: dict[str, int]
    selected_range_bytes: int = Field(ge=0)
    selected_output_bytes_upper_bound: int = Field(ge=0)
    selected_members_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_members: list[dict[str, Any]]
    source_metadata_path: str
    source_metadata_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class KittiInsLidarInventory(_Strict):
    schema_version: Literal["calibrex.ins_lidar_kitti_inventory/v0.1"] = (
        "calibrex.ins_lidar_kitti_inventory/v0.1"
    )
    provenance: dict[str, Any]
    selection_rule: str = SELECTION_RULE
    source_pool_metadata_sha256: dict[str, str]
    eligible_recording_ids: list[str]
    selected: list[SelectedArchive]
    sensor_member_download_bytes: int = Field(default=0, ge=0, le=0)


def collect(pool: Path, *, source_root: Path) -> KittiInsLidarInventory:
    """Choose from metadata counts only, then check the selected archives' live pins."""
    eligible = []
    sources = {}
    for path in sorted(pool.glob("*/download.json")):
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        sources[path.relative_to(source_root).as_posix()] = digest
        record = json.loads(data)
        frames = record.get("frames", {})
        archived = record.get("archive_frames", {})
        if (
            path.parent.name not in SPENT
            and archived.get("velodyne_points", 0) >= 600
            and frames.get("velodyne_points") == frames.get("oxts") == 600
        ):
            eligible.append((path.parent.name, path, record, digest))
    if len(sources) != 18 or len(eligible) < 3:
        raise ValueError("the declared eighteen-archive metadata pool is incomplete")
    selected = []
    for case_id, path, record, digest in eligible[:3]:
        date, _, drive = case_id.rpartition("_")
        url = fetch.drive_url(date, drive)
        size, etag = fetch.head(url)
        if (size, etag) != (record["size_bytes"], record["etag"]):
            raise ValueError(f"{case_id}: live archive pin differs from the source metadata")
        infos = fetch.central_directory(url, size, etag)
        prefix = f"{date}/{date}_drive_{drive}_sync/"
        chosen = fetch.select(infos, prefix, 600, streams=("velodyne_points", "oxts"))
        frames = {
            stream: sum(f"/{stream}/data/" in info.filename for info in chosen)
            for stream in ("velodyne_points", "oxts")
        }
        if any(value != 600 for value in frames.values()):
            raise ValueError(f"{case_id}: selected streams do not contain the declared 600 frames")
        spans = fetch.spans(infos, chosen, size)
        member_list = [
            {"name": info.filename, "bytes": info.file_size, "crc32": info.CRC} for info in chosen
        ]
        selected.append(
            SelectedArchive(
                case_id=case_id,
                date=date,
                drive=drive,
                url=url,
                size_bytes=size,
                etag=etag,
                archive_frames=record["archive_frames"],
                selected_frames=frames,
                selected_range_bytes=sum(hi - lo + 1 for lo, hi in spans),
                selected_output_bytes_upper_bound=sum(info.file_size for info in chosen),
                selected_members_sha256=fetch.members_sha256(member_list),
                selected_members=member_list,
                source_metadata_path=path.relative_to(source_root).as_posix(),
                source_metadata_sha256=digest,
            )
        )
    return KittiInsLidarInventory(
        provenance={
            "generator": "tools.ins_lidar_sota_inventory",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "command": [sys.executable, *sys.argv],
            "accuracy_claim": False,
            "source_worktree_dirty": bool(
                subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
            ),
            "exposure": "These recordings were read for camera-LiDAR v2. "
            "No INS-LiDAR method is run by this metadata inventory.",
            "independent_rig_policy": "All KITTI dates count as one platform, "
            "separate from Boreas.",
        },
        source_pool_metadata_sha256=sources,
        eligible_recording_ids=[item[0] for item in eligible],
        selected=selected,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.with_suffix(".schema.json").exists():
        parser.error("preserve the earlier inventory and choose a fresh output")
    root = Path(__file__).resolve().parents[1]
    result = collect(root / "docs/assets/kitti_camera_lidar_v2/evaluation/runs", source_root=root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".schema.json").write_text(
        json.dumps(result.model_json_schema(), indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selected": [
                    {
                        "id": r.case_id,
                        "range_bytes": r.selected_range_bytes,
                        "output_bytes_upper_bound": r.selected_output_bytes_upper_bound,
                    }
                    for r in result.selected
                ],
                "sensor_member_download_bytes": 0,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
