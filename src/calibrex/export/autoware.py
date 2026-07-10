"""Autoware-oriented export surface.

This keeps Autoware integration as an adapter shape rather than a core runtime
dependency.
"""

from __future__ import annotations

from pathlib import Path

from calibrex.core.io import write_mapping
from calibrex.core.result import CalibrationResult


def export_autoware_yaml(result: CalibrationResult, output: str | Path) -> None:
    """Export a conservative Autoware sensor-kit calibration YAML."""

    sensors = []
    for name, transform in sorted(result.transforms.items()):
        sensors.append(
            {
                "transform": name,
                "parent_frame": transform.parent,
                "child_frame": transform.child,
                "translation": {
                    "x": transform.translation_m[0],
                    "y": transform.translation_m[1],
                    "z": transform.translation_m[2],
                },
                "rotation_xyzw": {
                    "x": transform.rotation_quat_xyzw[0],
                    "y": transform.rotation_quat_xyzw[1],
                    "z": transform.rotation_quat_xyzw[2],
                    "w": transform.rotation_quat_xyzw[3],
                },
            }
        )
    write_mapping(
        Path(output),
        {
            "format": "slac.autoware/v0.1",
            "source_run": result.run.id,
            "quality_grade": result.quality.grade,
            "sensors": sensors,
        },
    )
