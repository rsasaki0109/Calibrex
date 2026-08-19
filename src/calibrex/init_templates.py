"""Sensor-template discovery and export for ``calibrex init --template``."""

from __future__ import annotations

import shutil
from pathlib import Path

from calibrex.core.exceptions import CalibrexError

_TEMPLATE_ALIASES: dict[str, str] = {
    "velodyne-camera": "velodyne_vlp16_pair_rosbag2",
    "velodyne-lidar-pair": "velodyne_vlp16_pair_rosbag2",
    "velodyne-lidar-pair-rosbag1": "velodyne_vlp16_pair_rosbag1",
    "velodyne-lidar-pair-rosbag2": "velodyne_vlp16_pair_rosbag2",
    "ouster-lidar-pair-rosbag2": "ouster_os1_pair_rosbag2",
    "lidar-camera-planar-board": "spinning_lidar_camera_planar_board",
}


def sensor_templates_root() -> Path:
    """Return the directory that contains named sensor template folders."""

    module_dir = Path(__file__).resolve().parent
    candidates = (
        module_dir.parent.parent / "examples" / "sensor_templates",
        module_dir / "resources" / "sensor_templates",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    msg = (
        "sensor templates are unavailable; expected examples/sensor_templates "
        "in the Calibrex repository checkout"
    )
    raise CalibrexError(msg)


def list_sensor_template_names() -> list[str]:
    """List canonical template directory names sorted alphabetically."""

    root = sensor_templates_root()
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and (path / "config.yaml").is_file()
    )


def resolve_sensor_template_name(name: str) -> str:
    """Resolve a template name or alias to a canonical directory name."""

    canonical = _TEMPLATE_ALIASES.get(name, name)
    template_dir = sensor_templates_root() / canonical
    if not template_dir.is_dir() or not (template_dir / "config.yaml").is_file():
        available = ", ".join(list_sensor_template_names())
        msg = f"unknown sensor template {name!r}; available templates: {available}"
        raise CalibrexError(msg)
    return canonical


def write_sensor_template(template_name: str, output_path: Path) -> Path:
    """Copy one template config to ``output_path`` or ``output_path/config.yaml``."""

    canonical = resolve_sensor_template_name(template_name)
    source = sensor_templates_root() / canonical / "config.yaml"
    destination = output_path
    if destination.suffix not in {".yaml", ".yml"}:
        destination.mkdir(parents=True, exist_ok=True)
        destination = destination / "config.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def template_summary() -> list[dict[str, str]]:
    """Return template metadata for CLI listing."""

    summaries: list[dict[str, str]] = []
    for name in list_sensor_template_names():
        summaries.append({"name": name, "path": str(sensor_templates_root() / name)})
    for alias, target in sorted(_TEMPLATE_ALIASES.items()):
        if alias != target:
            summaries.append({"name": alias, "path": target, "alias_for": target})
    return summaries
