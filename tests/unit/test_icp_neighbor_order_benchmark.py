"""Historical calibration comparisons require complete, intact source bindings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from tools.benchmark_cuda_point_to_point_icp import SOURCES
from tools.benchmark_icp_neighbor_order import _source_mapping


def _baseline(root: Path) -> None:
    files = {}
    for source in SOURCES:
        raw = source.encode()
        (root/Path(source).name).write_bytes(raw)
        files[source] = hashlib.sha256(raw).hexdigest()
    (root/"manifest.json").write_text(json.dumps({"commit":"fixture", "files":files}))


def test_rejects_modified_historical_runtime_source(tmp_path: Path) -> None:
    _baseline(tmp_path)
    (tmp_path/"_exact_neighbors.py").write_bytes(b"modified")
    with pytest.raises(ValueError, match="baseline source digest mismatch"):
        _source_mapping(tmp_path)


def test_rejects_incomplete_or_extra_historical_manifest(tmp_path: Path) -> None:
    _baseline(tmp_path)
    path = tmp_path/"manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["files"].pop(SOURCES[0])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="every declared ICP source"):
        _source_mapping(tmp_path)
    manifest["files"]["src/unexpected.py"] = "0"*64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="every declared ICP source"):
        _source_mapping(tmp_path)
