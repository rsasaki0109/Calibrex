from __future__ import annotations

import builtins
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from tools.benchmark_empirical_refits import audit

ARCHIVE = Path(__file__).resolve().parents[2] / "docs/assets/empirical_refit_workers"


def _changed(root: Path, name: str, value: object, *, yaml_file: bool = False) -> None:
    path = root / name
    text = yaml.safe_dump(value) if yaml_file else json.dumps(value)
    path.write_text(text, encoding="utf-8")
    index = json.loads((root / "index.json").read_bytes())
    index["files"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "index.json").write_text(json.dumps(index), encoding="utf-8")


@pytest.fixture
def archive_copy(tmp_path: Path) -> Path:
    root = tmp_path / "archive"
    shutil.copytree(ARCHIVE, root)
    return root


def test_complete_empirical_archive_without_accelerator_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = builtins.__import__

    def cpu_only(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"cupy", "numba", "PIL", "torch"}:
            raise ModuleNotFoundError("optional accelerator is absent")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", cpu_only)
    result = audit(ARCHIVE)
    assert result["workers"] == 6
    assert result["refits_per_worker"] == 20
    assert result["evaluations_per_worker"] == [6224] * 6
    assert result["complete_results_exact"]


@pytest.mark.parametrize("failure", ["missing", "reordered", "failed"])
def test_audit_rejects_selected_attempts_even_after_receipt_rebinding(
    archive_copy: Path, failure: str,
) -> None:
    attempts = json.loads((archive_copy / "attempts.json").read_bytes())
    if failure == "missing":
        attempts.pop()
    elif failure == "reordered":
        attempts[0], attempts[1] = attempts[1], attempts[0]
    else:
        attempts[0]["exit_code"] = 1
    _changed(archive_copy, "attempts.json", attempts)
    receipt = json.loads((archive_copy / "postprocessing.json").read_bytes())
    receipt["provenance"]["attempts_sha256"] = hashlib.sha256(
        (archive_copy / "attempts.json").read_bytes()
    ).hexdigest()
    _changed(archive_copy, "postprocessing.json", receipt)
    with pytest.raises(ValueError, match="missing, failed or reordered attempts"):
        audit(archive_copy)


def test_audit_checks_child_hash_before_normalizing_it(archive_copy: Path) -> None:
    name = "00-prior/resamples/resample-0000.yaml"
    value = yaml.safe_load((archive_copy / name).read_bytes())
    value["trace"][0]["objective"] += 0.001
    _changed(archive_copy, name, value, yaml_file=True)
    with pytest.raises(ValueError, match="child output hash binding differs"):
        audit(archive_copy)


def test_audit_rejects_changed_interval_after_inventory_rebinding(archive_copy: Path) -> None:
    name = "00-current/uncertainty.yaml"
    value = yaml.safe_load((archive_copy / name).read_bytes())
    interval = value["axis_intervals"][0]
    interval["lower"] -= 0.001
    interval["upper"] += 0.001
    interval["half_width"] = (interval["upper"] - interval["lower"]) / 2.0
    value["interval_halfwidth_translation_m"] = sum(
        item["half_width"] for item in value["axis_intervals"][:3]
    ) / 3.0
    _changed(archive_copy, name, value, yaml_file=True)
    from calibrex.core.validation import validate_file

    assert validate_file(archive_copy / name).valid
    with pytest.raises(ValueError, match="complete refit or uncertainty result differs"):
        audit(archive_copy)


def test_audit_rejects_spawned_import_outside_capsule(archive_copy: Path) -> None:
    path = next((archive_copy / "00-current").glob("process-*-imports.json"))
    value = json.loads(path.read_bytes())
    value["calibrex.solvers.probabilistic_camera_lidar_refiner"] = "C:/unfrozen/refiner.py"
    _changed(archive_copy, path.relative_to(archive_copy).as_posix(), value)
    with pytest.raises(ValueError):
        audit(archive_copy)


def test_audit_rejects_a_rebound_worker_strategy_flag(archive_copy: Path) -> None:
    name = "00-current/uncertainty.yaml"
    value = yaml.safe_load((archive_copy / name).read_bytes())
    value["provenance"]["command"][-1] = "2"
    _changed(archive_copy, name, value, yaml_file=True)
    with pytest.raises(ValueError, match="uncertainty input or worker binding differs"):
        audit(archive_copy)
