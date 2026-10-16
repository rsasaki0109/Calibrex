from __future__ import annotations

import builtins
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from tools.benchmark_png_luminance import audit

ARCHIVE = Path(__file__).resolve().parents[2] / "docs/assets/png_luminance_decoding"


def _changed(root: Path, relative: str, value: object, *, yaml_file: bool = False) -> None:
    path = root / relative
    text = yaml.safe_dump(value) if yaml_file else json.dumps(value)
    path.write_text(text, encoding="utf-8")
    index = json.loads((root / "index.json").read_bytes())
    index["files"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "index.json").write_text(json.dumps(index), encoding="utf-8")


@pytest.fixture
def archive_copy(tmp_path: Path) -> Path:
    destination = tmp_path / "archive"
    shutil.copytree(ARCHIVE, destination)
    return destination


def test_retained_complete_cli_archive_without_optional_image_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = builtins.__import__

    def without_pillow(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "PIL" or name.startswith("PIL."):
            raise ModuleNotFoundError("optional image package is absent")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_pillow)
    result = audit(ARCHIVE)
    assert result["workers"] == 12
    assert result["pixels_exact"]
    assert set(result["cases"]) == {"levinson", "pandey"}
    assert all(case["complete_results_exact"] for case in result["cases"].values())


def test_audit_rejects_pixel_change_after_inventory_rebinding(archive_copy: Path) -> None:
    relative = "pixels-current/pixels.json"
    value = json.loads((archive_copy / relative).read_bytes())
    value[next(iter(value))]["sha256_float64_le"] = "0" * 64
    _changed(archive_copy, relative, value)
    with pytest.raises(ValueError, match="decoded pixels differ"):
        audit(archive_copy)


def test_audit_rejects_changed_calibration_trace_after_inventory_rebinding(
    archive_copy: Path,
) -> None:
    relative = "levinson-00-current/artifacts/result.yaml"
    value = yaml.safe_load((archive_copy / relative).read_text(encoding="utf-8"))
    value["run"]["provenance"]["levinson_thrun_online"]["timeline"][0]["center_objective"] += 0.0001
    _changed(archive_copy, relative, value, yaml_file=True)
    with pytest.raises(ValueError, match="complete calibration result differs"):
        audit(archive_copy)


@pytest.mark.parametrize("failure", ["missing", "reordered", "failed"])
def test_audit_rejects_incomplete_or_selected_attempts(
    archive_copy: Path,
    failure: str,
) -> None:
    relative = "attempts.json"
    value = json.loads((archive_copy / relative).read_bytes())
    if failure == "missing":
        value.pop()
    elif failure == "reordered":
        value[0], value[1] = value[1], value[0]
    else:
        value[0]["exit_code"] = 1
    _changed(archive_copy, relative, value)
    with pytest.raises(ValueError, match="incomplete, reordered or failed attempts"):
        audit(archive_copy)


def test_audit_rejects_unfrozen_imports(archive_copy: Path) -> None:
    relative = "levinson-00-current/imports.json"
    value = json.loads((archive_copy / relative).read_bytes())
    value["modules"]["calibrex.data.kitti"] = "C:/outside/frozen-source/kitti.py"
    _changed(archive_copy, relative, value)
    with pytest.raises(ValueError):
        audit(archive_copy)
