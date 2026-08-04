"""ROS TF YAML export without importing ROS packages."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from calibrex.core.io import write_mapping
from calibrex.core.result import CalibrationResult, TransformResult


def export_ros_tf_yaml(result: CalibrationResult, output: str | Path) -> None:
    """Export transforms in a ROS-friendly YAML shape."""

    export_ros_tf_transforms(result.transforms, output)


def export_ros_tf_transforms(
    transforms: Mapping[str, TransformResult],
    output: str | Path,
) -> None:
    """Export any schema-valid transform mapping in a ROS-friendly shape."""

    transform_list = []
    for name, transform in sorted(transforms.items()):
        transform_list.append(
            {
                "name": name,
                "parent": transform.parent,
                "child": transform.child,
                "translation_m": transform.translation_m,
                "rotation_quat_xyzw": transform.rotation_quat_xyzw,
            }
        )
    write_mapping(Path(output), {"transforms": transform_list})
