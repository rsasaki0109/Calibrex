from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_readme_installs_from_pypi_and_pins_the_release_wheel() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    version_match = re.search(r'^version = "([^"]+)"$', pyproject, flags=re.MULTILINE)

    assert version_match is not None
    version = version_match.group(1)
    wheel_url = (
        "https://github.com/rsasaki0109/Calibrex/releases/download/"
        f"v{version}/calibrex-{version}-py3-none-any.whl"
    )

    # the demo and the quickstarts install the PyPI package; the Install section also
    # names the release wheel of the current version for a pinned, no-index-free install
    demo = readme.split("## Try it in one minute")[1].split("```bash")[1].split("```")[0]
    assert "python -m pip install calibrex" in demo
    assert readme.count("python -m pip install calibrex\n") >= 3
    assert readme.count(wheel_url) == 1
    assert "https://pypi.org/project/calibrex/" in readme
    assert "currently distributed from source" not in readme
    assert "not on pypi" not in readme.lower()
