"""Interference guards stop owned work and retain a reviewable failure record."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from tools import benchmark_cuda_map_warm as bench


@pytest.mark.parametrize(
    ("stdout", "code", "owned", "expected"),
    [
        ("31\n", 0, 31, {"own": 1, "foreign": 0, "unknown": 0}),
        ("31\n42\n", 0, 31, {"own": 1, "foreign": 1, "unknown": 0}),
        ("N/A\n", 0, None, {"own": 0, "foreign": 0, "unknown": 1}),
        ("", 1, None, {"own": 0, "foreign": 0, "unknown": 1}),
    ],
)
def test_gpu_observations_keep_foreign_and_unknown_work_visible(
    monkeypatch,
    stdout: str,
    code: int,
    owned: int | None,
    expected: dict[str, int],
) -> None:
    monkeypatch.setattr(
        bench.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout=stdout, returncode=code),
    )
    assert bench._observation(owned) == expected


@pytest.mark.parametrize("error", [FileNotFoundError(), subprocess.TimeoutExpired("gpu query", 10)])
def test_missing_or_unresponsive_query_is_unknown(monkeypatch, error: Exception) -> None:
    def failed_query(*args, **kwargs):
        raise error

    monkeypatch.setattr(bench.subprocess, "run", failed_query)
    assert bench._observation(31) == {"own": 0, "foreign": 0, "unknown": 1}


def test_foreign_observation_stops_only_the_launched_worker_and_retains_attempt(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    raw = tmp_path / "sequence/velodyne_points/data/0000000115.bin"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"test input")
    launcher = source / "map-benchmark.py.txt"
    launcher.write_bytes(b"frozen map launcher")
    for variant in ("prior", "current"):
        folder = source / f"sources-{variant}"
        folder.mkdir()
        (folder / "manifest.json").write_text("{}", encoding="utf-8")
        (folder / "tools").mkdir()
        (folder / "tools/benchmark_cuda_lidar_map.py").write_bytes(launcher.read_bytes())
    (source / "protocol.json").write_text(
        json.dumps(
            {
                "commits": {"prior": "1" * 40, "current": "2" * 40},
                "map_launcher_sha256": hashlib.sha256(launcher.read_bytes()).hexdigest(),
                "input_sha256": {str(raw): hashlib.sha256(raw.read_bytes()).hexdigest()},
            }
        ),
        encoding="utf-8",
    )
    observations = iter(
        (
            {"own": 0, "foreign": 0, "unknown": 0},
            {"own": 1, "foreign": 0, "unknown": 0},
            {"own": 1, "foreign": 1, "unknown": 0},
        )
    )
    monkeypatch.setattr(bench, "_observation", lambda pid: next(observations))
    actions = []

    class OwnedWorker:
        pid = 31
        stopped = False

        def poll(self):
            return -15 if self.stopped else None

        def terminate(self):
            actions.append("terminate owned")
            self.stopped = True

        def wait(self, timeout=None):
            if self.stopped:
                return -15
            raise subprocess.TimeoutExpired("owned worker", timeout)

    monkeypatch.setattr(bench.subprocess, "Popen", lambda *a, **k: OwnedWorker())
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark", "--comparison-dir", str(source), "--output", str(output), "--repeats", "1"],
    )
    with pytest.raises(RuntimeError, match="worker failed or was stopped"):
        bench.main()
    assert actions == ["terminate owned"]
    attempt = json.loads((output / "attempts.json").read_bytes())[0]
    assert attempt["interrupted"] and attempt["exit_code"] == -15
    assert json.loads((output / "observations.json").read_bytes())[-1]["foreign"] == 1
    assert (output / "protocol.json").is_file()
