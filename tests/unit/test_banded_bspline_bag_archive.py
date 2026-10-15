"""Replay the complete bag and strict fitted-array checks without sensor inputs."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import ModuleType

import pytest
from tools.audit_bspline_lidar_imu_benchmark import audit

ARCHIVE = Path(__file__).resolve().parents[2] / "docs/assets/numba_bspline_banded_bag"


def _strict(root: Path) -> dict[str, int]:
    helper = root / "validate-full-fits.py.txt"
    module = ModuleType("_structured_spline_bag_validation")
    module.__file__ = str(helper)
    exec(compile(helper.read_bytes(), str(helper), "exec"), module.__dict__)
    return module.validate(root)


def test_complete_bag_and_every_fitted_array() -> None:
    result = audit(ARCHIVE)
    assert result["workers"] == 2 and result["quality_passed"]
    assert result["whole_speedup"] < 1  # Preserve the negative timing observation.
    assert _strict(ARCHIVE) == {"verified_full_fits": 4}


def test_rebound_digest_cannot_hide_a_changed_fitted_calibration(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    shutil.copytree(ARCHIVE, root)
    path = root / "00-current/measurement.json"
    record = json.loads(path.read_bytes())
    record["rounds"][0]["calibration"]["gyro_bias"][0] += 0.1
    path.write_text(json.dumps(record), encoding="utf-8")
    path = root / "full-fit-validation.json"
    report = json.loads(path.read_bytes())
    report["provenance"]["measurements_sha256"]["current"] = hashlib.sha256(
        (root / "00-current/measurement.json").read_bytes()
    ).hexdigest()
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(AssertionError):
        _strict(root)
