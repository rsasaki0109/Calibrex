"""Keep optional compiler caches inside pytest's writable cache directory."""

from __future__ import annotations

import os

import pytest


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    # CuPy can bind its disk cache at import time, before fixtures run. Respect
    # an explicit user setting and let the cacheprovider initialize first.
    if "CUPY_CACHE_DIR" not in os.environ and hasattr(config, "cache"):
        patch = pytest.MonkeyPatch()
        patch.setenv("CUPY_CACHE_DIR", str(config.cache.mkdir("cupy")))
        config.add_cleanup(patch.undo)
