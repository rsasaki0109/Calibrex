"""Frame graph validation and transform access."""

from __future__ import annotations

from dataclasses import dataclass

from slac.core.config import CalibrationConfig, FrameConfig
from slac.core.exceptions import FrameGraphError
from slac.core.geometry import SE3
from slac.core.result import FrameGraphSnapshot


@dataclass(frozen=True)
class FrameNode:
    """A single frame graph node."""

    name: str
    parent: str | None
    root: bool
    transform_to_parent: SE3
    estimate: bool


class FrameGraph:
    """Directed frame graph using `T_parent_child` edge transforms."""

    def __init__(self, nodes: dict[str, FrameNode]) -> None:
        self.nodes = nodes
        self.root = self._validate()

    @classmethod
    def from_config(cls, config: CalibrationConfig) -> FrameGraph:
        """Build a frame graph from a validated config."""

        nodes: dict[str, FrameNode] = {}
        for name, frame in config.frames.items():
            nodes[name] = _node_from_config(name, frame)
        return cls(nodes)

    def _validate(self) -> str:
        if not self.nodes:
            msg = "frame graph must not be empty"
            raise FrameGraphError(msg)

        roots = [name for name, node in self.nodes.items() if node.root]
        if len(roots) != 1:
            msg = f"frame graph requires exactly one root, found {len(roots)}"
            raise FrameGraphError(msg)
        root = roots[0]

        for node in self.nodes.values():
            if node.parent is not None and node.parent not in self.nodes:
                msg = f"frame '{node.name}' references missing parent '{node.parent}'"
                raise FrameGraphError(msg)

        for name in self.nodes:
            self._path_to_root(name, root)
        return root

    def _path_to_root(self, frame: str, root: str | None = None) -> list[str]:
        path: list[str] = []
        seen: set[str] = set()
        current: str | None = frame
        expected_root = root if root is not None else self.root
        while current is not None:
            if current in seen:
                msg = f"cycle detected at frame '{current}'"
                raise FrameGraphError(msg)
            seen.add(current)
            path.append(current)
            current = self.nodes[current].parent
        if path[-1] != expected_root:
            msg = f"frame '{frame}' is disconnected from root '{expected_root}'"
            raise FrameGraphError(msg)
        return path

    def transform_to_root(self, frame: str) -> SE3:
        """Return `T_root_frame`."""

        if frame not in self.nodes:
            msg = f"unknown frame '{frame}'"
            raise FrameGraphError(msg)
        transform = SE3.identity()
        current = frame
        while current != self.root:
            node = self.nodes[current]
            transform = node.transform_to_parent.compose(transform)
            if node.parent is None:
                msg = f"frame '{current}' has no parent before reaching root"
                raise FrameGraphError(msg)
            current = node.parent
        return transform

    def snapshot(self) -> FrameGraphSnapshot:
        """Return a serializable frame graph snapshot."""

        return FrameGraphSnapshot(
            root=self.root,
            frames={name: node.parent for name, node in sorted(self.nodes.items())},
        )


def _node_from_config(name: str, frame: FrameConfig) -> FrameNode:
    if frame.transform is None:
        transform = SE3.identity()
        estimate = False
    else:
        transform = SE3.from_lists(
            frame.transform.initial.translation,
            frame.transform.initial.rotation_quat_xyzw,
        )
        estimate = frame.transform.estimate
    return FrameNode(
        name=name,
        parent=frame.parent,
        root=frame.root,
        transform_to_parent=transform,
        estimate=estimate,
    )
