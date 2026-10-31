import hashlib
import json
from pathlib import Path

from tools import kitti_camera_lidar_v3_rule_screen as screen

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs/assets/kitti_camera_lidar_v3/start_agreement_20261031"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_published_stages_match_their_manifests_without_sensor_arrays():
    for part in ("cached", "recaptured"):
        manifest = json.loads((RESULTS / part / "manifest.json").read_text())
        listed = {item["path"]: item for item in manifest["files"]}
        on_disk = {
            p.relative_to(RESULTS / part).as_posix()
            for p in (RESULTS / part).rglob("*")
            if p.is_file() and p.name != "manifest.json"
        }
        assert set(listed) == on_disk
        for name, item in listed.items():
            assert _sha(RESULTS / part / name) == item["sha256"]
    assert not list(RESULTS.rglob("*.npz"))


def test_frozen_screen_verdict_reproduces_from_the_published_comparisons():
    published = json.loads((RESULTS / "screen.json").read_text())
    inputs = [Path(path) for path in published["provenance"]["inputs_sha256"]]
    local = [
        RESULTS / Path(*path.parts[path.parts.index("start_agreement_20261031") + 1 :])
        for path in inputs
    ]
    for path, original in zip(local, inputs, strict=True):
        assert _sha(path) == published["provenance"]["inputs_sha256"][str(original)]
    result = screen.screen([json.loads(path.read_text()) for path in local])
    assert result["verdict"] == published["verdict"] == "not_carried_forward"
    assert result["checks"] == published["checks"]
    assert [k for k, v in result["checks"].items() if not v] == ["turn_3deg_flagged"]
    assert result["metrics"] == published["metrics"]
    inventory = json.loads(
        (ROOT / "docs/assets/kitti_camera_lidar_v3/development_inventory.json").read_text()
    )
    assert result["metrics"]["current"] == inventory["current_v2_metrics"]
    assert len(result["large_errors"]["proposed"]) == 2
    assert result["large_errors"]["new_under_proposed"] == []
