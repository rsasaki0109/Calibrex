from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_readme_uses_versioned_github_release_wheel() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    version_match = re.search(r'^version = "([^"]+)"$', pyproject, flags=re.MULTILINE)

    assert version_match is not None
    version = version_match.group(1)
    wheel_url = (
        "https://github.com/rsasaki0109/Calibrex/releases/download/"
        f"v{version}/calibrex-{version}-py3-none-any.whl"
    )

    # the workflow block, the evidence pipeline quickstart, and the Install section
    assert readme.count(wheel_url) == 3
    assert "currently distributed from source" not in readme
    assert "pypi.org" not in readme.lower()
    assert "python -m pip install calibrex" not in readme.lower()
    # the check demo is newer than the released wheel, so its block installs from the repo
    demo = readme.split("## Try it in one minute")[1].split("```bash")[1].split("```")[0]
    if "calibrex demo check" in demo and wheel_url not in demo:
        assert 'pip install "git+https://github.com/rsasaki0109/Calibrex.git"' in demo
