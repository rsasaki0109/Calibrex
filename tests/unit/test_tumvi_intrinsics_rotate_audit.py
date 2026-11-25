"""Offline tests for window selection, scoring, and bag pins."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation
from tools import tumvi_intrinsics_rotate_audit as audit


def test_window_requires_two_axes_and_prefers_lower_translation() -> None:
    times = np.arange(0.0, 25.01, 0.025)
    positions = np.column_stack([np.maximum(12 - times, 0), 0 * times, 0 * times])
    sweep = 15 * np.sin(times)
    rotations = Rotation.from_rotvec(np.column_stack([sweep, sweep, 0 * times]), degrees=True)
    window = audit.select_window(times, positions, rotations)
    assert window is not None
    assert window["start_seconds"] == 12
    assert window["rig_position_diameter_m"] == 0

    single_axis = Rotation.from_rotvec(np.column_stack([sweep, 0 * times, 0 * times]), degrees=True)
    assert audit.select_window(times, positions, single_axis) is None
    sparse = slice(None, None, 20)
    assert audit.select_window(times[sparse], positions[sparse], rotations[sparse]) is None


def test_errors_and_inclusive_thresholds() -> None:
    kalibr = {"intrinsics": [200, 200, 256, 256], "distortion_coeffs": [0.01, 0, 0, 0]}
    estimate = {"intrinsics": [225, 175, 259, 250], "distortion_coeffs": [0, 1, 2, 3]}
    delta = audit.errors(estimate, kalibr)
    assert delta == {
        "fx_pct": 12.5,
        "fy_pct": -12.5,
        "cx_px": 3,
        "cy_px": -6,
        "k1": -0.01,
        "k2": 1,
        "k3": 2,
        "k4": 3,
    }
    gates = {"max_focal_error_pct": 12.5, "max_principal_error_px": 6}
    assert audit.verdict(delta, gates) == "pass"
    assert audit.verdict({**delta, "fx_pct": -12.6}, gates) == "fail"
    assert audit.verdict({**delta, "cy_px": 6.1}, gates) == "fail"
    assert audit.verdict({**delta, "k4": float("nan")}, gates) == "inconclusive"


def test_command_does_not_pass_the_reference(tmp_path: Path) -> None:
    bag, output = tmp_path / "test.bag", tmp_path / "estimate.yaml"
    command = audit.rotate_command(bag, "cam1", output, 12)
    for flag, value in {
        "--bag": str(bag),
        "--topic": "/cam1/image_raw",
        "--distortion": "equidistant",
        "--camera-name": "cam1",
        "--seconds": "10",
        "--start-seconds": "12",
    }.items():
        assert command[command.index(flag) + 1] == value
    assert str(audit.CAMCHAIN) not in command
    assert "--start-seconds" not in audit.rotate_command(bag, "cam1", output, 0)


@pytest.mark.parametrize("changed", [None, "etag", "size", "range"])
def test_download_verifies_ordered_ranges(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, changed: str | None
) -> None:
    data = b"test bag bytes"
    monkeypatch.setattr(audit, "DEVELOPMENT", {"room1": (len(data), "pinned")})
    bag = tmp_path / "test.bag"

    def curl(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command[-1] == audit.bag_url("room1")
        assert 'If-Match: "pinned"' in command
        start, end = map(int, command[command.index("--range") + 1].split("-"))
        body = data[start : end + 1]
        Path(command[command.index("-o") + 1]).write_bytes(body if changed != "size" else b"")
        etag = "changed" if changed == "etag" else "pinned"
        total = len(data) - (changed == "range")
        headers = f'ETag: "{etag}"\nContent-Range: bytes {start}-{end}/{total}\n'
        return subprocess.CompletedProcess(command, 0, stdout=headers)

    monkeypatch.setattr(audit.subprocess, "run", curl)
    if changed:
        with pytest.raises(ValueError, match="does not match pinned bag"):
            audit.download("room1", bag)
    else:
        audit.download("room1", bag)
        assert bag.read_bytes() == data
    assert list(tmp_path.iterdir()) == [bag]


def test_existing_reference_pin() -> None:
    assert audit.sha256(audit.CAMCHAIN) == audit.CAMCHAIN_SHA256
    assert list(yaml.safe_load(audit.CAMCHAIN.read_text())) == list(audit.CAMERAS)
