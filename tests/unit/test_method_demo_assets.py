"""Gallery receipts preserve source identity and synthetic stereo behavior."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from build_method_demos import gallery_receipt, stereo_demo  # noqa: E402


def test_gallery_sources_and_synthetic_stereo_matches() -> None:
    samples = ROOT / "docs/app/samples"
    index = json.loads((samples / "fisheye_stereo_demo.json").read_text())
    for path, digest in index["provenance"]["sources"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert json.loads((samples / "method_demo_metrics.json").read_text()) == gallery_receipt()
    generated = stereo_demo()["models"]
    for code, receipt in index["models"].items():
        raw = (samples / receipt["url"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == receipt["sha256"]
        model = json.loads(raw)
        for field, size in [("raw", (640, 240)), ("rectified", (640, 240)), ("depth", (320, 240))]:
            assert (
                Image.open(io.BytesIO(base64.b64decode(model[field].split(",", 1)[1]))).size == size
            )
        native = generated[code]
        assert native["matches"] > 20
        assert native["vertical_median_px"] < 0.2
        assert native["median_relative_depth_error"] < 0.03
        assert model["median_relative_depth_error"] == pytest.approx(
            native["median_relative_depth_error"], abs=0.002
        )
