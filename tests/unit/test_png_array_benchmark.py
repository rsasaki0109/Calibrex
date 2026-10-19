from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
from tools.benchmark_png_array import audit, execute_worker, sha, write


@pytest.fixture
def evidence(tmp_path: Path) -> Path:
    source = b"raise RuntimeError('archived source must never execute')\n"
    import hashlib

    manifest = {"calibrex/__init__.py": hashlib.sha256(source).hexdigest()}
    write(tmp_path / "source-manifest.json", manifest)
    with tarfile.open(tmp_path / "source.tar.gz", "w:gz") as capsule:
        member = tarfile.TarInfo("calibrex/__init__.py")
        member.size = len(source)
        capsule.addfile(member, io.BytesIO(source))
    (tmp_path / "launcher.py").write_text("# archived only\n", encoding="utf-8")
    inputs = {f"image_02/data/{i:010d}.png": "0" * 64 for i in range(48)}
    protocol = {
        "source_manifest_sha256": sha(tmp_path / "source-manifest.json"),
        "source_capsule_sha256": sha(tmp_path / "source.tar.gz"),
        "launcher_sha256": sha(tmp_path / "launcher.py"),
        "inputs_sha256": inputs,
        "repetitions": 2,
        "order": ["rows", "array", "array", "rows"],
    }
    write(tmp_path / "protocol.json", protocol)
    observations = []
    for index, variant in enumerate(protocol["order"]):
        folder = tmp_path / f"{index:02d}-{variant}"
        folder.mkdir()
        write(
            folder / "measurement.json",
            {
                "variant": variant,
                "decode_seconds": 2.0 if variant == "rows" else 0.5,
                "images": 48,
                "pixel_count": 48,
                "array_bytes": 48 * 8,
                "pixels": {
                    name: {"shape": [1, 1], "dtype": "float64-le", "sha256": "0" * 64}
                    for name in inputs
                },
                "imports": {"calibrex": "calibrex/__init__.py"},
                "provenance": {
                    "protocol_sha256": sha(tmp_path / "protocol.json"),
                    "source_manifest_sha256": protocol["source_manifest_sha256"],
                    "launcher_sha256": protocol["launcher_sha256"],
                },
            },
        )
        observations.append(
            {
                "variant": variant,
                "folder": folder.name,
                "exit_code": 0,
                "seconds": 4.0 if variant == "rows" else 3.0,
                "measurement_sha256": sha(folder / "measurement.json"),
            }
        )
    write(
        tmp_path / "attempts.json",
        {
            "observations": observations,
            "provenance": {"protocol_sha256": sha(tmp_path / "protocol.json")},
        },
    )
    return tmp_path


def test_cpu_only_replay_never_imports_optional_backends_or_executes_capsule(
    evidence: Path,
) -> None:
    script = """
import sys
from importlib.abc import MetaPathFinder
class Block(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'PIL', 'numba', 'cupy', 'calibrex'}:
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, Block())
from pathlib import Path
from tools.benchmark_png_array import audit
assert audit(Path(sys.argv[1]))['decode_ratio'] == 4.0
"""
    value = subprocess.run(
        [sys.executable, "-c", script, str(evidence)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert value.returncode == 0, value.stdout + value.stderr


@pytest.mark.parametrize("kind", ["pixels", "time", "work", "missing"])
def test_replay_rejects_changed_evidence_even_after_measurement_hash_rebinding(
    evidence: Path,
    kind: str,
) -> None:
    attempts = json.loads((evidence / "attempts.json").read_bytes())
    if kind == "missing":
        attempts["observations"].pop()
    else:
        record = attempts["observations"][1]
        path = evidence / record["folder"] / "measurement.json"
        value = json.loads(path.read_bytes())
        if kind == "pixels":
            next(iter(value["pixels"].values()))["sha256"] = "1" * 64
        elif kind == "time":
            record["seconds"] = 0.001
        elif kind == "work":
            value["pixel_count"] -= 1
        write(path, value)
        record["measurement_sha256"] = sha(path)
    write(evidence / "attempts.json", attempts)
    with pytest.raises(ValueError):
        audit(evidence)


@pytest.mark.parametrize(
    "error,code", [(subprocess.TimeoutExpired("worker", 1), -1), (OSError(), -2)]
)
def test_failed_launches_still_produce_an_observation_and_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    code: int,
) -> None:
    def failed(*_args, **_kwargs):
        raise error

    monkeypatch.setattr("tools.benchmark_png_array.subprocess.run", failed)
    assert execute_worker(["worker"], {}, tmp_path) == code
    assert (tmp_path / "stdout.txt").is_file()
    assert (tmp_path / "stderr.txt").is_file()
