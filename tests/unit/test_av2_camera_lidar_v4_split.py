import json
from pathlib import Path

import pytest
from tools import av2_camera_lidar_v4_split as split

ROOT = Path(__file__).resolve().parents[2]
FIXED = ROOT / "docs/assets/camera_lidar_v4/av2_split.json"


def _lister(val: int = 40, test: int = 5, drop: str | None = None):
    names = {
        "val": [f"{i:08x}-v" for i in range(val)],
        "test": [f"{i:08x}-t" for i in range(test)],
        "train": ["train-a", "train-b"],
    }

    def lister(prefix: str, delimiter: str):
        rest = prefix[len(split.PREFIX) :]
        parts = rest.strip("/").split("/")
        if len(parts) == 1:
            return [f"{prefix}{name}/" for name in names[parts[0]]]
        base = split.PREFIX + "/".join(parts[:2]) + "/"
        tail = "/".join(parts[2:])
        files = {
            "": ["city_SE3_egovehicle.feather"],
            "calibration": ["egovehicle_SE3_sensor.feather", "intrinsics.feather"],
            "sensors/lidar": ["1.feather", "2.feather"],
            f"sensors/cameras/{split.CAMERA}": ["1.jpg", "2.jpg", "3.jpg"],
        }[tail]
        key_base = base + (tail + "/" if tail else "")
        return [(key_base + f, 10, '"e"') for f in files if f != drop] + (
            [key_base + "x/"] if tail == "" else []
        )

    return lister


def test_rule_takes_the_first_validation_logs_and_every_test_log() -> None:
    result = split.rule([f"{i:02d}" for i in range(40, 0, -1)], ["t2", "t1"])
    assert result["development"] == [f"{i:02d}" for i in range(1, 31)]
    assert result["evaluation"] == ["t1", "t2"]
    assert len(result["reserve_validation"]) == 10
    with pytest.raises(ValueError, match="distinct"):
        split.rule(["a", "b"], ["b"])


def test_build_pins_members_and_keeps_incomplete_logs() -> None:
    built = split.build(_lister())
    assert len(built["development"]) == 30 and len(built["evaluation"]) == 5
    first = built["development"][0]
    assert first["complete"] and first["sweeps"] == 2 and first["images"] == 3
    assert first["bytes"] == 10 * (3 + 2 + 3)
    assert built["train_logs"] == 2 and built["sensor_payload_bytes_read"] == 0
    other = split.build(_lister(drop="2.jpg"))["development"][0]
    assert other["members_sha256"] != first["members_sha256"]
    missing = split.build(_lister(drop="intrinsics.feather"))["evaluation"][0]
    assert not missing["complete"] and missing["missing"][0].endswith("intrinsics.feather")


def test_fixed_split_holds_the_rule() -> None:
    fixed = json.loads(FIXED.read_text())
    development = [item["log"] for item in fixed["development"]]
    evaluation = [item["log"] for item in fixed["evaluation"]]
    assert development == sorted(development) and len(development) == 30
    assert len(evaluation) == 150 and not set(development) & set(evaluation)
    assert "02678d04-cc9f-3148-9f95-1ba66347dff9" in development
    assert not set(fixed["reserve_validation"]) & set(development)
    assert fixed["sensor_payload_bytes_read"] == 0
    assert all(item["complete"] for item in fixed["development"] + fixed["evaluation"])


def test_cli_refuses_to_overwrite(tmp_path: Path) -> None:
    out = tmp_path / "split.json"
    out.write_text("{}")
    with pytest.raises(ValueError, match="refuse overwrite"):
        split.main([str(out)])
