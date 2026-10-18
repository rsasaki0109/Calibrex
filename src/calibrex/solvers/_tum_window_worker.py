"""Private CLI entry point; library callers need no multiprocessing main guard."""

from __future__ import annotations

import sys
from pathlib import Path

from calibrex.solvers._tum_window_processes import _execute_request

if __name__ == "__main__":
    _execute_request(Path(sys.argv[1]), Path(sys.argv[2]))
