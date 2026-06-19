#!/usr/bin/env python3
"""Run the local release smoke test used before packaging."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

EXPECTED_SCHEMA_COUNT = 15
DEFAULT_VENV = Path("/tmp/calibrex-release-smoke")
DEFAULT_BUILD_ENV = Path("/tmp/calibrex-release-build")
DEFAULT_SCHEMA_DIR = Path("/tmp/calibrex-release-schemas")
DEFAULT_REPORT_DIR = Path("/tmp/calibrex-release-report")
CACHED_EVIDENCE_RESULT = Path(
    "examples/public_datasets/livox_horizon_horizon_pcd_sample/cached_evidence_result.yaml"
)


def main() -> int:
    """Run a clean package smoke test from the repository root."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    parser.add_argument("--build-env", type=Path, default=DEFAULT_BUILD_ENV)
    parser.add_argument("--schema-dir", type=Path, default=DEFAULT_SCHEMA_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()

    _require_repo_root()
    _remove(Path("dist"))
    _remove(Path("build"))
    _remove(Path("src/calibrex.egg-info"))
    _remove(args.build_env)
    _remove(args.venv)
    _remove(args.schema_dir)
    _remove(args.report_dir)

    build_python = _build_python(args.build_env)
    _run([str(build_python), "-m", "build"])
    _run([sys.executable, "-m", "venv", str(args.venv)])

    wheel = _built_wheel()
    smoke_python = _venv_executable(args.venv, "python")
    smoke_calibrex = _venv_executable(args.venv, "calibrex")
    _run([str(smoke_python), "-m", "pip", "install", str(wheel)])
    _run([str(smoke_calibrex), "doctor", "--json"])
    _run([str(smoke_calibrex), "schema", "all", "--output-dir", str(args.schema_dir)])
    _assert_schema_count(args.schema_dir)
    _run(
        [
            str(smoke_calibrex),
            "validate",
            str(CACHED_EVIDENCE_RESULT),
            "--kind",
            "result",
            "--json",
        ]
    )
    _run(
        [
            str(smoke_calibrex),
            "report",
            str(CACHED_EVIDENCE_RESULT),
            "--output-dir",
            str(args.report_dir),
            "--json",
        ]
    )
    _run(
        [
            str(smoke_calibrex),
            "validate",
            str(args.report_dir / "summary.json"),
            "--kind",
            "report-summary",
            "--json",
        ]
    )
    _run(
        [
            str(smoke_calibrex),
            "validate",
            str(args.report_dir / "assessment.json"),
            "--kind",
            "assessment",
            "--json",
        ]
    )
    _run(
        [
            str(smoke_calibrex),
            "validate",
            str(args.report_dir / "policy.json"),
            "--kind",
            "policy",
            "--json",
        ]
    )
    reassessment_path = args.report_dir / "reassessment.json"
    _run(
        [
            str(smoke_calibrex),
            "assess",
            str(args.report_dir / "evidence.json"),
            "--policy",
            str(args.report_dir / "policy.json"),
            "--output",
            str(reassessment_path),
            "--json",
        ],
        allowed_return_codes={0, 1},
    )
    _run(
        [
            str(smoke_calibrex),
            "validate",
            str(reassessment_path),
            "--kind",
            "assessment",
            "--json",
        ]
    )
    _run(
        [
            str(smoke_calibrex),
            "validate",
            str(args.report_dir / "protocol.json"),
            "--kind",
            "protocol",
            "--json",
        ]
    )
    _run(
        [
            str(smoke_calibrex),
            "validate",
            str(args.report_dir / "transforms.json"),
            "--kind",
            "transforms",
            "--json",
        ]
    )
    _run(
        [
            str(smoke_calibrex),
            "validate",
            str(args.report_dir / "bundle.json"),
            "--kind",
            "evidence-bundle",
            "--json",
        ]
    )
    verification_path = args.report_dir / "verification.json"
    _run(
        [
            str(smoke_calibrex),
            "verify",
            str(args.report_dir / "bundle.json"),
            "--output",
            str(verification_path),
            "--json",
        ]
    )
    _run(
        [
            str(smoke_calibrex),
            "validate",
            str(verification_path),
            "--kind",
            "evidence-bundle-verification",
            "--json",
        ]
    )
    _run([str(smoke_calibrex), "verify", str(verification_path), "--json"])
    _run(
        [
            str(smoke_calibrex),
            "validate",
            str(args.report_dir / "evidence.json"),
            "--kind",
            "report-evidence",
            "--json",
        ]
    )
    comparison_path = args.report_dir / "comparison.json"
    _run(
        [
            str(smoke_calibrex),
            "compare",
            str(CACHED_EVIDENCE_RESULT),
            str(CACHED_EVIDENCE_RESULT),
            "--output",
            str(comparison_path),
        ]
    )
    _run(
        [
            str(smoke_calibrex),
            "validate",
            str(comparison_path),
            "--kind",
            "comparison",
            "--json",
        ]
    )
    print("local release smoke passed")
    return 0


def _require_repo_root() -> None:
    if not Path("pyproject.toml").exists() or not Path("src/calibrex").is_dir():
        raise SystemExit("run this script from the Calibrex repository root")


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _run(command: list[str], *, allowed_return_codes: set[int] | None = None) -> None:
    print("+ " + " ".join(command))
    completed = subprocess.run(command, check=False)
    allowed = allowed_return_codes or {0}
    if completed.returncode not in allowed:
        raise subprocess.CalledProcessError(completed.returncode, command)


def _build_python(build_env: Path) -> Path:
    if _module_available(sys.executable, "build"):
        return Path(sys.executable)
    _run([sys.executable, "-m", "venv", str(build_env)])
    python = _venv_executable(build_env, "python")
    _run([str(python), "-m", "pip", "install", "build>=1.2"])
    return python


def _module_available(python: str, module: str) -> bool:
    completed = subprocess.run(
        [python, "-c", f"import {module}"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0


def _built_wheel() -> Path:
    wheels = sorted(Path("dist").glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected exactly one wheel in dist/, found {len(wheels)}")
    return wheels[0]


def _venv_executable(venv: Path, name: str) -> Path:
    if sys.platform == "win32":
        return venv / "Scripts" / f"{name}.exe"
    return venv / "bin" / name


def _assert_schema_count(schema_dir: Path) -> None:
    count = len(list(schema_dir.glob("*.schema.json")))
    if count != EXPECTED_SCHEMA_COUNT:
        raise SystemExit(
            f"expected {EXPECTED_SCHEMA_COUNT} schema files in {schema_dir}, found {count}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
