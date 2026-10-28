"""Prevent optional CPU tests from silently disappearing from extra-enabled CI."""
from __future__ import annotations

import ast
import shlex
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
JOBS = {
    "compressed-bag-formats": {"lz4", "zstandard", "mcap"},
    "opencv-adapter": {"cv2"},
    "numba-adapter": {"numba"},
}


def _optional_imports(nodes: list[ast.AST]) -> set[str]:
    modules = set()
    for node in nodes:
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "pytest"
                and node.func.attr == "importorskip" and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            modules.add(node.args[0].value.split(".")[0])
    return modules


@pytest.mark.parametrize("job,dependencies", JOBS.items())
def test_optional_cpu_test_files_are_selected_by_their_ci_job(job, dependencies):
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    selected: set[Path] = set()
    for step in workflow["jobs"][job]["steps"]:
        words = shlex.split(step.get("run", ""))
        if "pytest" not in words:
            continue
        for word in words[words.index("pytest") + 1:]:
            if not word.startswith("tests/"):
                continue
            # A node selector does not establish coverage of the entire file.
            assert "::" not in word, f"{job}: list whole optional test files, not {word}"
            paths = list(ROOT.glob(word))
            assert paths, f"{job}: test path does not exist: {word}"
            for path in paths:
                selected.update(path.rglob("test_*.py") if path.is_dir() else [path])

    required = set()
    for path in (ROOT / "tests/unit").rglob("test_*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        # Module-level CuPy gating makes the whole file GPU-only. A CuPy skip
        # inside an individual test does not exempt the remaining CPU tests.
        top_level_calls = [node.value for node in tree.body if isinstance(node, ast.Expr)]
        if "cupy" in _optional_imports(top_level_calls):
            continue
        if _optional_imports(list(ast.walk(tree))) & dependencies:
            required.add(path)
    missing = sorted(str(path.relative_to(ROOT)) for path in required - selected)
    assert not missing, f"{job} omits optional CPU test files: {missing}"
