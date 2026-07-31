import sys
from pathlib import Path

from calibrex.solvers.external_adapter import (
    external_command_argv,
    load_external_transform_mapping,
    run_external_command,
)


def test_external_process_reports_missing_executable() -> None:
    result = run_external_command(
        ("calibrex-definitely-missing-executable",),
        working_directory=None,
        timeout_seconds=1.0,
    )

    assert result.attempted is False
    assert result.error == "command executable is not available"


def test_external_process_reports_nonzero_exit() -> None:
    result = run_external_command(
        (sys.executable, "-c", "raise SystemExit(7)"),
        working_directory=None,
        timeout_seconds=1.0,
    )

    assert result.attempted is True
    assert result.returncode == 7
    assert result.success is False


def test_external_process_reports_timeout() -> None:
    result = run_external_command(
        (sys.executable, "-c", "import time; time.sleep(1)"),
        working_directory=None,
        timeout_seconds=0.1,
    )

    assert result.timed_out is True
    assert result.duration_seconds is not None


def test_external_command_string_is_split_without_shell() -> None:
    assert external_command_argv("solver --output 'a b.yaml'") == (
        "solver",
        "--output",
        "a b.yaml",
    )


def test_external_transform_loader_rejects_malformed_result(tmp_path: Path) -> None:
    path = tmp_path / "malformed.yaml"
    path.write_text("transforms: []\n", encoding="utf-8")

    try:
        load_external_transform_mapping(path)
    except ValueError as exc:
        assert "mapping" in str(exc)
    else:
        raise AssertionError("malformed external result was accepted")
