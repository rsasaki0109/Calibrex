"""Build the Calibrex wheel that the browser calibration page installs with micropip.

Writes ``docs/app/wheels/<wheel>`` and ``docs/app/wheels/manifest.json``
(``{"wheel": ..., "version": ...}``); run it before ``mkdocs build``::

    python tools/build_browser_wheel.py
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

from calibrex import __version__

OUTPUT = Path("docs/app/wheels")


def main() -> None:
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True)
    if importlib.util.find_spec("pip") is not None:
        command = [sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "--quiet"]
        command += ["-w", str(OUTPUT)]
    else:  # uv-managed environments have no pip
        command = ["uv", "build", "--wheel", "--quiet", "--out-dir", str(OUTPUT)]
    subprocess.run(command, check=True)
    wheels = sorted(OUTPUT.glob("calibrex-*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected one calibrex wheel, found {wheels}")
    manifest = {"wheel": wheels[0].name, "version": __version__}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    print(manifest)


if __name__ == "__main__":
    main()
