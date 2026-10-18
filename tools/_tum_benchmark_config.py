"""Keep historical serial TUM workloads independent of the evolving example.

The template is the checked-in example before 5e5f105 enabled window workers.
Its original bytes are pinned so archive audits cannot silently adopt changed
sample counts, diagnostics or execution options from the current example.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

SERIAL_TEMPLATE = Path(__file__).with_name("fixtures") / "tum_joint_serial_v1.yaml"
SERIAL_TEMPLATE_SHA256 = "ea677809a17141a9017365f77621af156c53a0de38ca9509afc6aa6c1a03c957"


def load_serial_tum_config() -> dict[str, Any]:
    """Read the original, hash-pinned serial benchmark configuration afresh."""
    raw = SERIAL_TEMPLATE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SERIAL_TEMPLATE_SHA256:
        raise ValueError("historical serial TUM template differs")
    config: dict[str, Any] = yaml.safe_load(raw)
    return config
