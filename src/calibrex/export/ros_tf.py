"""ROS TF YAML export without importing ROS packages."""

from __future__ import annotations

from pathlib import Path

from calibrex.core.io import write_mapping
from calibrex.core.result import CalibrationResult


def export_ros_tf_yaml(result: CalibrationResult, output: str | Path) -> None:
    """Export transforms in a ROS-friendly YAML shape."""

    transforms = []
    for name, transform in sorted(result.transforms.items()):
        transforms.append(
            {
                "name": name,
                "parent": transform.parent,
                "child": transform.child,
                "translation_m": transform.translation_m,
                "rotation_quat_xyzw": transform.rotation_quat_xyzw,
            }
        )
    write_mapping(Path(output), {"transforms": transforms})
