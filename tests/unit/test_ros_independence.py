"""The core must stay ROS-independent (ADR 0001, AGENTS.md).

The static check holds everywhere.  The runtime check is decisive in a shell
where ROS is sourced, which is exactly the environment most users run in; CI
exercises it in the ROS-sourced job.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
import sys
from pathlib import Path

import calibrex
from calibrex.core.environment_readiness import python_path_isolation

PACKAGE_ROOT = Path(calibrex.__file__).resolve().parent
ROS_TOP_LEVEL_MODULES = (
    "rclpy",
    "rosbag2_py",
    "rosidl_runtime_py",
    "rosidl_parser",
    "rclpy_message_converter",
    "launch",
    "launch_ros",
    "launch_testing",
    "ament_index_python",
    "sensor_msgs",
    "std_msgs",
    "geometry_msgs",
    "tf2_ros",
    "rospy",
    "rosbag",
    "roslib",
)


def _imported_top_level_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module.split(".", 1)[0])
    return modules


def test_core_source_never_imports_ros_modules() -> None:
    offenders = {
        str(path.relative_to(PACKAGE_ROOT)): sorted(
            _imported_top_level_modules(path) & set(ROS_TOP_LEVEL_MODULES)
        )
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
    }

    assert {name: found for name, found in offenders.items() if found} == {}


def test_importing_every_core_module_loads_no_ros_module() -> None:
    skipped: list[str] = []
    for module in pkgutil.walk_packages(calibrex.__path__, prefix="calibrex."):
        try:
            importlib.import_module(module.name)
        except ImportError as exc:
            # Optional dependencies (open3d, opencv, numba, mcap) may be absent.
            skipped.append(f"{module.name}: {exc}")

    assert python_path_isolation(loaded_modules=list(sys.modules)).loaded_ros_modules == []
    assert len(skipped) < 10, skipped
