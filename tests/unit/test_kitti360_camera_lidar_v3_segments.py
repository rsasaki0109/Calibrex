import hashlib
import json
from pathlib import Path

import pytest
from tools import kitti360_camera_lidar_v3_segments as tool

ROOT = Path(__file__).resolve().parents[2]
COUNTS = ROOT / "docs/assets/kitti360_sota_readiness/paired-frame-counts.json"
FIXED = ROOT / "docs/assets/kitti_camera_lidar_v3/kitti360_prep/segments.json"


def _record(sequence: str, first: int, count: int, extra_lidar: int = 0) -> dict:
    return {
        "sequence": sequence,
        "image_first": first,
        "image_last": first + count - 1,
        "image_count": count,
        "lidar_first": first,
        "lidar_last": first + count - 1 + extra_lidar,
        "lidar_count": count + extra_lidar,
        "common_first": first,
        "common_count": count,
    }


def test_rule_cuts_disjoint_60_s_segments_with_60_s_gaps() -> None:
    assert tool.segments(_record("0004", 0, 599)) == []
    assert tool.segments(_record("0004", 0, 600)) == [{"start_frame": 0, "frames": 600}]
    assert tool.segments(_record("0004", 0, 1799)) == [{"start_frame": 0, "frames": 600}]
    assert [s["start_frame"] for s in tool.segments(_record("0002", 4391, 3000, 242))] == [
        4391,
        5591,
        6791,
    ]


def test_rule_rejects_gapped_or_inconsistent_counts() -> None:
    gapped = _record("0004", 0, 1000) | {"image_count": 999}
    with pytest.raises(ValueError, match="image IDs are not contiguous"):
        tool.segments(gapped)
    with pytest.raises(ValueError, match="common range"):
        tool.segments(_record("0004", 0, 1000) | {"common_count": 900})


def test_build_needs_metadata_only_counts_for_every_candidate() -> None:
    counts = json.loads(COUNTS.read_text())
    tainted = counts | {"provenance": counts["provenance"] | {"sensor_payload_bytes": 1}}
    with pytest.raises(ValueError, match="metadata only"):
        tool.build(tainted)
    partial = counts | {"sequences": [r for r in counts["sequences"] if r["sequence"] != "0007"]}
    with pytest.raises(ValueError, match="0007"):
        tool.build(partial)


def test_fixed_segment_list_is_what_the_committed_rule_gives() -> None:
    fixed = json.loads(FIXED.read_text())
    assert (
        fixed["provenance"]["generator_sha256"]
        == hashlib.sha256(
            (ROOT / "tools/kitti360_camera_lidar_v3_segments.py").read_bytes()
        ).hexdigest()
    )
    assert fixed["provenance"]["counts_sha256"] == hashlib.sha256(COUNTS.read_bytes()).hexdigest()
    rebuilt = tool.build(json.loads(COUNTS.read_text()))
    for key in ("evaluation", "development", "not_counted", "spent_without_counts"):
        assert fixed[key] == rebuilt[key]
    assert fixed["sensor_payload_bytes_read"] == 0
    assert len(fixed["evaluation"]) == 42 and len(fixed["development"]) == 22
    assert {s["sequence"] for s in fixed["evaluation"]} == set(tool.CANDIDATES)
    assert not {s["sequence"] for s in fixed["development"]} & set(tool.CANDIDATES)


def test_cli_refuses_to_overwrite_a_written_list(tmp_path: Path) -> None:
    output = tmp_path / "segments.json"
    assert tool.main([str(COUNTS), str(output)]) == 0
    with pytest.raises(ValueError, match="refuse overwrite"):
        tool.main([str(COUNTS), str(output)])
