"""MCAP dataset adapter boundary.

The core package keeps MCAP optional. This module exposes a stable inspection
surface without making robotics users import ROS message classes.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from slac.data.base import StreamSummary


def inspect_mcap(path: str | Path) -> tuple[list[StreamSummary], list[str]]:
    """Inspect an MCAP file when optional dependencies are available."""

    mcap_path = Path(path)
    warnings: list[str] = []
    if not mcap_path.exists():
        warnings.append("MCAP path does not exist")
        return [], warnings

    if importlib.util.find_spec("mcap") is None:
        warnings.append("MCAP inspection requires optional dependency slac[mcap].")
        return [], warnings

    warnings.append("MCAP channel decoding is not enabled in this alpha adapter.")
    return [], warnings
