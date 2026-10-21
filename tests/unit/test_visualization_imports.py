"""Fresh-process import boundaries and existing public visualization exports."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def fresh(script: str) -> None:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("submodule", [False, True])
def test_importing_check_report_does_not_load_unrelated_report_models(submodule):
    fresh(
        "import sys\n"
        "import calibrex.visualization as package\n"
        + (
            "from calibrex.visualization.check_report import render_check_html\n"
            if submodule
            else ""
        )
        + "assert set(package.__all__) <= set(dir(package))\n"
        "assert 'calibrex.visualization.report' not in sys.modules\n"
        "assert 'calibrex.visualization.comparison_table' not in sys.modules\n"
        "assert 'calibrex.visualization.evidence_card' not in sys.modules\n"
        "assert 'calibrex.core.assessment' not in sys.modules\n"
        "try:\n    package.missing_export\n"
        "except AttributeError as error:\n    assert 'missing_export' in str(error)\n"
        "else:\n    raise AssertionError('missing attributes must raise AttributeError')\n"
    )


def test_every_public_export_is_the_original_function_and_stays_cached():
    fresh(
        "import importlib, typing\n"
        "import calibrex.visualization as package\n"
        "expected = {\n"
        "'render_comparison_table':'comparison_table', "
        "'write_comparison_table':'comparison_table',\n"
        "'render_evidence_card':'evidence_card', 'write_evidence_card':'evidence_card',\n"
        "'render_html_report':'report'}\n"
        "assert set(package.__all__) == set(expected)\n"
        "for name, module in expected.items():\n"
        "    source = importlib.import_module('calibrex.visualization.' + module)\n"
        "    original = getattr(source, name)\n"
        "    assert getattr(package, name) is original\n"
        "    assert package.__dict__[name] is original\n"
        "    assert typing.get_type_hints(original)\n"
        "namespace = {}\n"
        "exec('from calibrex.visualization import *', namespace)\n"
        "assert {key for key in namespace if not key.startswith('__')} == set(expected)\n"
        "assert all(namespace[name] is getattr(package, name) for name in expected)\n"
    )
