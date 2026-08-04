"""Autoware-oriented export surface.

This keeps Autoware integration as an adapter shape rather than a core runtime
dependency.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from calibrex.core.io import write_mapping
from calibrex.core.result import CalibrationResult, Grade, TransformResult


def export_autoware_yaml(result: CalibrationResult, output: str | Path) -> None:
    """Export a conservative Autoware sensor-kit calibration YAML."""

    export_autoware_transforms(
        result.transforms,
        output,
        source_run=result.run.id,
        quality_grade=result.quality.grade,
    )


def export_autoware_transforms(
    transforms: Mapping[str, TransformResult],
    output: str | Path,
    *,
    source_run: str | None = None,
    quality_grade: Grade | None = None,
) -> None:
    """Export a transform mapping in a conservative Autoware YAML shape."""

    sensors = []
    for name, transform in sorted(transforms.items()):
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
    payload: dict[str, object] = {
        "format": "slac.autoware/v0.1",
        "sensors": sensors,
    }
    if source_run is not None:
        payload["source_run"] = source_run
    if quality_grade is not None:
        payload["quality_grade"] = quality_grade
    write_mapping(Path(output), payload)
